import torch
import torchaudio
import torchaudio.functional as F
from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
from core.audio_visual_encoder.pe import PEAudioFrameOutput
import os
import re
import math
import argparse
from pydub import AudioSegment
from tqdm import tqdm
from typing import Iterable, List, Tuple, Dict, Union, Optional
from collections import defaultdict

Interval = Tuple[float, float, Tuple[float, float]]   # start, end, (prob_mean, prob_max)
# start, end, "prob_max/prob_mean" (debug mode)
# or 
# start, end, "prob_max"           (normal mode)
Interval_str = Tuple[str, str, str]                   
# label -> "No events detected" or list of Interval_strs
Events = Dict[str, Union[str, List[Interval_str]]]

def mmss_to_sec(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)

def sec_to_mmss(x: int) -> str:
    return f"{x//60:02d}:{x%60:02d}"

def merge_intervals(intervals: Iterable[Interval], tolerance: float = 0.0) -> List[Interval]:
    xs = sorted((min(s, e), max(s, e), p) for s, e, p in intervals)  # normalize + sort
    merged: List[Interval] = []
    for s, e, p in xs:
        if not merged:
            merged.append((s, e, p))
            continue

        ps, pe, pp = merged[-1]
        # merge if the gap is small enough
        if s <= pe + tolerance:
            merged[-1] = (ps, max(pe, e), max(pp, p))
        else:
            merged.append((s, e, p))
    return merged

def print_overlap_timeline(events: Events, tolerance_sec: int = 0) -> None:
    """
    Prints segments in time order.
    If events overlap in time, their labels appear together in that segment.
    tolerance_sec expands each interval by +/- tolerance_sec seconds.
    """
    # Whether a second is associated with multiple labels
    painted_with_labels = defaultdict(dict)  # second -> dict(label: prob)

    # 1) paint timeline per second (inclusive endpoints)
    for label, spans in events.items():
        if not spans or isinstance(spans, str):  # e.g. "No events detected"
            continue
        for a, b, p in spans:
            s, e = mmss_to_sec(a), mmss_to_sec(b)
            assert e >= s, f"Invalid interval: {a}-{b}"
            s = max(0, s - tolerance_sec)
            e = e + tolerance_sec
            for t in range(s, e + 1):
                painted_with_labels[t][label] = p

    if not painted_with_labels:
        return

    # 2) compress consecutive seconds with same label-set
    t0 = min(painted_with_labels)
    t1 = max(painted_with_labels)

    seg_start = t0
    curr_labels = painted_with_labels.get(t0, dict())

    def print_seg_labels(seg_start: int, seg_end: int, labels: dict):
        if labels:
            labels_str = ", ".join(f"{label} ({prob})" for label, prob in sorted(labels.items()))
            print(f"{sec_to_mmss(seg_start)}-{sec_to_mmss(seg_end)}: {labels_str}")

    for t in range(t0 + 1, t1 + 2):  # +2 to flush the last segment
        new_labels = painted_with_labels.get(t, dict())
        # If no event is detected at t, new_labels will be empty set.
        # The condition below will hold and the current segment will be printed.
        if new_labels.keys() & curr_labels.keys() == set():
            # If new_labels and curr_labels contain no common elements, 
            # print and start a new segment
            print_seg_labels(seg_start, t - 1, curr_labels)
            seg_start = t
            curr_labels = new_labels
        else:
            # If new_labels and curr_labels share some elements, 
            # merge the label dicts and continue to check
            curr_labels = {**curr_labels, **new_labels}

def split_waveform(wav: torch.Tensor, seg_samples: int):
    """
    wav: (channels, samples)
    yields: (segment_tensor, valid_len)
      - segment_tensor always has seg_samples (last chunk is zero-padded)
      - valid_len is the non-padded length in samples (<= seg_samples)
    """
    c, total = wav.shape
    for start in range(0, total, seg_samples):
        end = min(start + seg_samples, total)
        chunk = wav[:, start:end]
        valid_len = end - start
        if valid_len < seg_samples:
            pad = torch.zeros((c, seg_samples - valid_len), dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=1)
        yield chunk, valid_len

def PEAudioFrame_forward(
    self,
    input_ids: torch.Tensor,  # tokenized text
    input_values: Optional[torch.Tensor] = None,  # audio waveform (may be None if input_features is provided)
    input_features: Optional[torch.Tensor] = None,  # codec_features (if already computed)
    attention_mask: Optional[torch.Tensor] = None,  # text attention mask
    padding_mask: Optional[torch.Tensor] = None,  # audio padding mask
    threshold: float = 0.3,
    return_spans: bool = True,
    return_span_probs: bool = False,
) -> PEAudioFrameOutput:
    audio_output = self.audio_model(input_values, padding_mask, input_features=input_features)
    text_model_output = self._get_text_output(input_ids, attention_mask)

    text_embeds = self.text_head(text_model_output.pooler_output)
    audio_embeds = self.audio_head(audio_output.last_hidden_state)

    spans = None
    if return_spans:
        bsz = input_ids.size(0)
        unscaled_logits = audio_embeds @ text_embeds.unsqueeze(1).transpose(-1, -2)
        logits = unscaled_logits.squeeze(-1) * self.logit_scale + self.logit_bias
        probs = logits.sigmoid()

        preds = probs > threshold
        # Find where predictions changed from False->True and True->False
        changes = torch.diff(torch.nn.functional.pad(preds, (1, 1), value=False), dim=1).nonzero()
        span_tensor = torch.cat([changes[::2], changes[1::2, [1]]], dim=1)
        # Convert audio frame index to time
        dac_config = self.config.audio_model.dac_vae_encoder
        sec_per_frame = dac_config.hop_length / dac_config.sampling_rate

        spans = []
        # Each instance in the batch corresponds to one sublist in spans.
        for i in range(bsz):
            # (nspans, 2) start,end in frames
            # [..., 1:] to drop batch index, since the batch index is implicitly i.
            spans_i = span_tensor[span_tensor[:, 0] == i, 1:]  

            if spans_i.numel() == 0:
                # If no spans detected, return empty list for this instance.
                spans.append([])
                continue

            if return_span_probs:
                out_i = []
                for s_f, e_f in spans_i.tolist():
                    s_f = int(s_f)
                    e_f = int(e_f)
                    span_p = (probs[i, s_f:e_f].mean().item() if e_f > s_f else probs[i, s_f].item(),
                              probs[i, s_f:e_f].max().item()  if e_f > s_f else probs[i, s_f].item())
                    out_i.append([s_f * sec_per_frame, e_f * sec_per_frame, span_p])
                spans.append(out_i)
            else:
                spans.append((spans_i.float() * sec_per_frame).tolist())

    return PEAudioFrameOutput(
        text_embeds=text_embeds,
        audio_embeds=audio_embeds,
        spans=spans,
        text_output=text_model_output,
        audio_output=audio_output,
    )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# -------------------- args --------------------
parser = argparse.ArgumentParser(description="Audio Source Separation using SAM-Audio (chunked + tqdm)")
parser.add_argument("--model_size", type=str, default="large", choices=["small", "medium", "large"], help="Model size to use")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input audio file (.wav/.mp4/...)")
parser.add_argument("--segment_seconds", type=float, default=120.0, help="Chunk length in seconds (e.g. 5, 10, 30)")
parser.add_argument("--max_segments", type=int, default=0, help="Maximum number of segments to process (0 = all)")
parser.add_argument("--sample_rate", type=int, default=48000, help="Target sample rate for event detection")
parser.add_argument("--weak_thres_discount", type=float, default=0.9, 
                    help="Discount factor for weak event detection threshold")
parser.add_argument("--debug", type=str2bool, const=True, default=True, help="Whether to print debug info")
args = parser.parse_args()

device = "cuda"
# Load model and transform
model = PEAudioFrame.from_config(f"pe-a-frame-{args.model_size}", pretrained=True).to(device)
transform = PEAudioFrameTransform.from_config(f"pe-a-frame-{args.model_size}")

# Patch the forward method to include span detection
model.forward = PEAudioFrame_forward.__get__(model, PEAudioFrame)

# Define audio file and event descriptions
input_file = args.input_file
input_trunk = input_file.rsplit(".", 1)[0]

# If MP4, extract to WAV once
if input_file.lower().endswith(".mp4"):
    audio_file = input_trunk + ".wav"
    if not os.path.exists(audio_file):
        print(f"Extracting audio from {input_file} -> {audio_file} ...")
        video = AudioSegment.from_file(input_file, format="mp4")
        video.export(audio_file, format="wav")
else:
    audio_file = input_file

desc2thres = {
    "open and close a closet": 0.5,
    "human voice": 0.6,
    "door slams": 0.5,
    "chopstick clutters": 0.5,
    "dish clutters": 0.5,
}

# -------------------- load + resample once --------------------
wav, sr = torchaudio.load(input_file)  # wav: (channels, samples)
seg_samples = max(1, int(round(args.segment_seconds * sr)))
total_samples = wav.shape[1]
num_segments = max(1, math.ceil(total_samples / seg_samples))

print(f"Loaded: {input_file} | sr={sr} | chunk={args.segment_seconds}s ({seg_samples} samples) | segments={num_segments}")

target_chunks = []
residual_chunks = []

pbar = tqdm(
    split_waveform(wav, seg_samples),
    total=num_segments,
    desc="Matching by Segments",
    unit="seg",
    dynamic_ncols=True,
)

t0 = 0
desc2intervals = {desc: [] for desc in desc2thres.keys()}

for chunk, valid_len in pbar:
    chunk_rs = F.resample(chunk, orig_freq=sr, new_freq=args.sample_rate)
    # Process inputs
    inputs = transform(audio=[chunk_rs], text=list(desc2thres.keys()), sampling_rate=args.sample_rate).to(device)

    # Run inference
    with torch.inference_mode():
        # Discount the minimal threshold a bit to allow "weak (less confident) events" to be detected 
        # and later merged with adjacent "strong (confident) events".
        outputs = model(**inputs, return_spans=True, threshold=min(desc2thres.values()) * args.weak_thres_discount, 
                        return_span_probs=True)

    # outputs: PEAudioFrameOutput
    # audio_embeds, text_embeds, spans, audio_output, text_output
    # Print detected time spans for each event
    for description, spans in zip(desc2thres.keys(), outputs.spans):
        if spans:
            # Discount the threshold to allow weak events to pass.
            desc2intervals[description].extend([(float(s) + t0, float(e) + t0, p) for s, e, p in spans])

    t0 += args.segment_seconds

all_events = {}

for description in desc2thres.keys():
    spans = merge_intervals(desc2intervals[description], tolerance=2)
    if spans:
        span_strs = []
        for start, end, (prob_mean, prob_max) in spans:
            # Filter weak events below threshold.
            if prob_max < desc2thres[description]:
                continue
            start_min = int(start) // 60
            start_sec = int(start) %  60
            end_min   = int(end)   // 60
            end_sec   = int(end)   %  60
            if args.debug:
                span_strs.append((f"{start_min:02d}:{start_sec:02d}", f"{end_min:02d}:{end_sec:02d}", f"{prob_max:.2f}/{prob_mean:.2f}"))
            else:
                span_strs.append((f"{start_min:02d}:{start_sec:02d}", f"{end_min:02d}:{end_sec:02d}", f"{prob_max:.2f}"))

        all_events[description] = span_strs

        #span_str = ", ".join(span_strs)
        #print(f'"{description}": [{span_str}]')
    #else:
    #    print(f'"{description}": No events detected')

print_overlap_timeline(all_events, tolerance_sec=1)
