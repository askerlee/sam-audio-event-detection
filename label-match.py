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
from datetime import datetime, timedelta

Interval = Tuple[float, float, Tuple[float, float], float]   # start, end, (prob_mean, prob_max), dbfs
# start, end, "prob_max/prob_mean/dbfs"
# or 
# start, end, dbfs
Interval_str = Tuple[str, str, str]
# label -> "No events detected" or list of Interval_strs
Events = Dict[str, Union[str, List[Interval_str]]]
# Reolink camera recordings.
PATTERN1 = re.compile(r"(?P<date>\d{8})[/\\](?P<prefix>.+)-(?P<start>\d{6})-(?P<end>\d{6})\.(mp4|wav)$", re.IGNORECASE)
# Manually recorded videos.
PATTERN2 = re.compile(r"(?P<prefix>.+)_(?P<date>\d{8})_(?P<start>\d{6})\.(mp4|wav)$", re.IGNORECASE)

# For phones with stereo recording, convert to mono using mid-side technique.
def stereo_to_mono(wav: torch.Tensor) -> torch.Tensor:
    L, R = wav[0], wav[1]

    # L+R (mid) keeps content common to both channels (often voice).
    mid  = 0.5 * (L + R)
    # L−R (side) keeps content that differs between channels (sometimes wind/ambient/spatial).
    side = 0.5 * (L - R)

    # pick the one with higher RMS energy
    def rms(x): return x.pow(2).mean().sqrt()

    mono = mid if rms(mid) >= rms(side) else side
    mono = mono.unsqueeze(0)  # [1, T]
    return mono

def mmss_to_sec(t: str) -> int:
    # t: "MM:SS" or "MMM:MM:SS"
    m, s = t.split(":")
    return int(m) * 60 + int(s)

def sec_to_mmss(x: int) -> str:
    return f"{x//60:02d}:{x%60:02d}"

def merge_intervals(intervals: Iterable[Interval], tolerance: float = 0.0) -> List[Interval]:
    xs = sorted((s, e, p, db) for s, e, p, db in intervals)
    merged: List[Interval] = []
    for s, e, p, db in xs:
        if not merged:
            # The first interval.
            merged.append((s, e, p, db))
            continue

        s0, e0, p0, db0 = merged[-1]
        if s <= e0 + tolerance:
            # merge if the gap is small enough
            # Take the max prob_mean and prob_max of the two intervals, as the new prob.
            merged[-1] = (s0, max(e0, e), max(p0, p), max(db0, db))
        else:
            # the gap is too large, start a new interval.
            merged.append((s, e, p, db))
    return merged


# Extract event segments from event_mask and compute dBFS for each
def extract_event_segments_from_mask(audio: torch.Tensor, mask: torch.Tensor, sample_rate: int,
                                     db_offset: float) -> List[Tuple[float, float]]:
    """
    Extract contiguous True segments from boolean mask and compute dBFS for each.
    Returns list of (dbfs, duration_seconds) tuples.
    """
    segments = []
    if mask.sum() == 0:
        return segments
        
    # Find boundaries where mask changes from False to True or True to False
    padded_mask = torch.cat([torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)])
    changes = torch.diff(padded_mask.int())
    
    starts = torch.where(changes == 1)[0]  # False -> True transitions
    ends = torch.where(changes == -1)[0]   # True -> False transitions
    
    for start_idx, end_idx in zip(starts, ends):
        start_idx = start_idx.item()
        end_idx = end_idx.item()
        
        # Extract audio segment
        segment_audio = audio[:, start_idx:end_idx]
        if segment_audio.shape[1] > 0:
            # Compute dBFS for this segment
            segment_dbfs = calc_dbfs(segment_audio, db_offset)
            duration_seconds = (end_idx - start_idx) / sample_rate
            segments.append((segment_dbfs, duration_seconds))
    
    return segments

def print_overlap_timeline(events: Events, start_date_obj: Optional[datetime] = None, 
                           tolerance_sec: int = 0, debug=False) -> None:
    """
    Prints segments in time order.
    If events overlap in time, their labels appear together in that segment.
    tolerance_sec expands each interval by +/- tolerance_sec seconds.
    """
    # Whether a second is associated with multiple labels
    painted_with_labels = defaultdict(dict)  # second -> dict(label: prob)
    # painted_with_labels2: painted_with_labels extended with time tolerance at both ends.
    painted_with_labels2 = defaultdict(dict)  # second -> dict(label: prob)
    event_boundaries = []

    # 1) paint timeline per second (inclusive endpoints)
    for label, spans in events.items():
        if not spans or isinstance(spans, str):  # e.g. "No events detected"
            continue
        for a, b, p in spans:
            s, e = mmss_to_sec(a), mmss_to_sec(b)
            assert e >= s, f"Invalid interval: {a}-{b}"
            for t in range(s, e + 1):
                painted_with_labels[t][label] = p

            # Only add tolerance to the start time, not the end time, 
            # to avoid adding extra seconds to the end of each event.
            s = max(0, s - tolerance_sec)
            for t in range(s, e + 1):
                painted_with_labels2[t][label] = p

    if not painted_with_labels:
        return

    # 2) compress consecutive seconds with same label-set
    t0 = min(painted_with_labels.keys())
    t1 = max(painted_with_labels2.keys())

    seg_start = t0
    curr_labels = painted_with_labels.get(t0, dict())

    def print_seg_labels(seg_start: int, seg_end: int, labels: dict, 
                         start_date_obj: Optional[datetime] = None):
        if labels:
            if debug:
                labels_str = ", ".join(f"{label} ({prob})" for label, prob in sorted(labels.items()))
            else:
                labels_str = ", ".join(sorted(labels.keys()))
            if start_date_obj:
                seg_start_dt = start_date_obj + timedelta(seconds=seg_start)
                seg_end_dt   = start_date_obj + timedelta(seconds=seg_end)
                print(f"{seg_start_dt.strftime('%H:%M:%S')}-{seg_end_dt.strftime('%H:%M:%S')}: {labels_str}")
            else:
                print(f"{sec_to_mmss(seg_start)}-{sec_to_mmss(seg_end)}: {labels_str}")

    for t in range(t0 + 1, t1 + 2):  # +2 to flush the last segment
        # Get labels from painted_with_labels2 (with time tolerance).
        new_labels = painted_with_labels2.get(t, dict())
        # If no event is detected at t, new_labels will be empty set.
        # The condition below will hold and the current segment will be printed.
        if new_labels.keys() & curr_labels.keys() == set():
            # If new_labels and curr_labels contain no common elements, 
            # print and start a new segment
            print_seg_labels(seg_start, t - 1, curr_labels, start_date_obj=start_date_obj)
            # Mark the segment as eventful in event_boundaries.
            # Note if curr_labels is empty, it means no event is detected in this segment.
            # So we need to skip inserting event_boundaries in that case.
            if curr_labels:
                # Note to use t-1 instead of t as the end, because if we use t, 
                # after repeating by sr, event_boundaries will delineate one second longer than total_samples.
                event_boundaries.append((seg_start, t - 1))

            seg_start = t
            curr_labels = painted_with_labels.get(t, dict())
        else:
            # If new_labels and curr_labels share some elements, 
            # merge the label dicts and continue to check
            curr_labels = {**curr_labels, **new_labels}

    return event_boundaries

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

def calc_dbfs(wav: torch.Tensor, db_offset: float) -> float:
    wav = wav - wav.mean()
    rms = torch.sqrt(torch.mean(wav**2))
    dbfs = 20.0 * torch.log10(torch.clamp(rms, min=1e-12)) + db_offset
    return dbfs.item()

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
parser.add_argument("--weak_thres_discount", type=float, default=0.8, 
                    help="Discount factor for weak event detection threshold")
# This offset has been calibrated by comparing dBFS values measured by the DB meter and those calculated here.
parser.add_argument("--db_offset", type=float, default=85, help="dBFS offset to add to all detected events' dBFS")
parser.add_argument("--loud_db_offset", type=float, default=8.0, help="dB offset above average event dBFS to consider an event 'loud'")
parser.add_argument("--abs_time", type=str2bool, nargs='?', const=True, default=False, 
                    help="Whether to print absolute time for timestamps")
parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=True, help="Whether to print debug info")

def analyze_audio_labels(model: PEAudioFrame, transform: PEAudioFrameTransform, 
                         input_file: str, segment_seconds: float, sample_rate: int, 
                         weak_thres_discount: float, db_offset: float, loud_db_offset: float, 
                         desc2det_thres: Dict[str, float], desc2db_thres: Dict[str, float],
                         device: str, print_abs_time: bool, debug: bool):
    """
    Analyze audio file for specific event labels and print detected events with timestamps.
    """
    # Define audio file and event descriptions
    input_trunk = input_file.rsplit(".", 1)[0]
    if print_abs_time:
        # Try to extract start time from filename
        m1 = PATTERN1.search(input_file)
        m2 = PATTERN2.search(input_file)
        if m1:
            start_date = m1.group("date")
            start_time = m1.group("start")
            # 10262025
            month, day, year = int(start_date[0:2]), int(start_date[2:4]), int(start_date[4:8])
            hr, minute, second = int(start_time[0:2]), int(start_time[2:4]), int(start_time[4:6])
            start_date_obj = datetime(year, month, day, hr, minute, second)
        elif m2:
            start_date = m2.group("date")
            start_time = m2.group("start")
            # 20251026
            year, month, day = int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8])
            hr, minute, second = int(start_time[0:2]), int(start_time[2:4]), int(start_time[4:6])
            start_date_obj = datetime(year, month, day, hr, minute, second)
        else:
            print(f"Warning: Could not extract start time from filename {os.path.basename(input_file)}. Output relative time instead.")
            print_abs_time = False
            start_date_obj = None
    else:
        start_date_obj = None

    # If MP4, extract to WAV once
    if input_file.lower().endswith(".mp4"):
        audio_file = input_trunk + ".wav"
        if not os.path.exists(audio_file):
            print(f"Extracting audio from {input_file} -> {audio_file} ...")
            video = AudioSegment.from_file(input_file, format="mp4")
            video.export(audio_file, format="wav")
    else:
        audio_file = input_file

    # -------------------- load + resample once --------------------
    wav, sr = torchaudio.load(input_file)  # wav: (channels, samples)
    # Video/Audio recorded by phones.
    if wav.shape[0] == 2:
        wav = stereo_to_mono(wav)
    seg_samples = max(1, int(round(segment_seconds * sr)))
    # total_samples: total points in wav.
    total_samples = wav.shape[1]
    num_segments = max(1, math.ceil(total_samples / seg_samples))

    print(f"Loaded: {input_file} | sr={sr} | chunk={segment_seconds}s ({seg_samples} samples) | segments={num_segments}")
  
    # -------------------- process each description over chunks --------------------
    pbar = tqdm(
        split_waveform(wav, seg_samples),
        total=num_segments,
        desc="Matching by Segments",
        unit="seg",
        dynamic_ncols=True,
    )

    t0 = 0
    desc2intervals = {desc: [] for desc in desc2det_thres.keys()}
    event_db_mask = torch.zeros(total_samples, dtype=float).to(device)

    for chunk, valid_len in pbar:
        chunk_rs = F.resample(chunk, orig_freq=sr, new_freq=sample_rate)
        # Process inputs
        inputs = transform(audio=[chunk_rs], text=list(desc2det_thres.keys()), sampling_rate=sample_rate).to(device)
        # inputs.input_values have been normalized to [-1.02, 1.02] in the transform.
        # Run inference
        with torch.inference_mode():
            # Discount the minimal threshold a bit to allow "weak (less confident) events" to be detected 
            # and later merged with adjacent "strong (confident) events".
            outputs = model(**inputs, return_spans=True, threshold=min(desc2det_thres.values()) * weak_thres_discount, 
                            return_span_probs=True)

        # outputs: PEAudioFrameOutput
        # audio_embeds, text_embeds, spans, audio_output, text_output
        # Print detected time spans for each event
        for description, spans in zip(desc2det_thres.keys(), outputs.spans):
            if spans:
                # Discount the threshold to allow weak events to pass.
                # Filter out events below the discounted threshold here.
                # p: (prob_mean, prob_max). p[0] is prob_mean.
                for s, e, p in spans:
                    if p[1] < desc2det_thres[description] * weak_thres_discount:
                        continue
                    # NOTE: chunk_rs has been resampled to args.sample_rate.
                    # But we compute start/end, dbfs in the original chunk.
                    start_s = float(s)
                    end_s   = float(e)
                    start = int(start_s * sr)
                    end   = min(int(end_s   * sr), valid_len)
                    global_start_s = start_s + t0
                    global_start   = int(global_start_s * sr)
                    global_end_s   = end_s   + t0
                    global_end     = int(global_end_s * sr)
                    # Take the span from the original chunk (not resampled).
                    span_sound = chunk[:, start:end]
                    dbfs = calc_dbfs(span_sound, db_offset)
                    # Ignore events below their dBFS thresholds.
                    # If in debug mode, keep all events for analysis.
                    if (not debug) and (dbfs < desc2db_thres[description]):
                        continue
                    # event_db_mask corresponds to the original wav's timeline.
                    event_db_mask[global_start: global_end] = torch.maximum(event_db_mask[global_start: global_end], torch.tensor(dbfs, device=device))
                    desc2intervals[description].append((global_start_s, global_end_s, p, dbfs))

        t0 += segment_seconds

    all_events = {}

    for description in desc2det_thres.keys():
        spans = merge_intervals(desc2intervals[description], tolerance=2)
        if spans:
            span_strs = []
            for start, end, (prob_mean, prob_max), dbfs in spans:
                # Filter weak events below threshold.
                if prob_max < desc2det_thres[description]:
                    continue
                start = round(start)
                end   = round(end)
                start_min = start // 60
                start_sec = start %  60
                end_min   = end   // 60
                end_sec   = end   %  60

                if debug:
                    span_strs.append((f"{start_min:02d}:{start_sec:02d}", f"{end_min:02d}:{end_sec:02d}", f"{prob_max:.2f}/{prob_mean:.2f}/{dbfs:.1f}db"))
                else:
                    span_strs.append((f"{start_min:02d}:{start_sec:02d}", f"{end_min:02d}:{end_sec:02d}", f"{dbfs:.1f}db"))

            all_events[description] = span_strs

            #span_str = ", ".join(span_strs)
            #print(f'"{description}": [{span_str}]')
        #else:
        #    print(f'"{description}": No events detected')

    print_overlap_timeline(all_events, start_date_obj=start_date_obj, 
                           tolerance_sec=4, debug=debug)
    wav = wav.to(device)
    avg_dbfs = calc_dbfs(wav, db_offset)
    print(f"Average Audio dBFS: {avg_dbfs:.1f}db")
    if event_db_mask.sum() == 0:
        print("No events detected, skipping event dBFS analysis.")
        return
    nonevent_mask = (event_db_mask == 0)
    all_nonevent_dbfs = extract_event_segments_from_mask(wav, nonevent_mask, sr, db_offset)
    avg_nonevent_dbfs = sum(dbfs * dur for dbfs, dur in all_nonevent_dbfs) / sum(dur for _, dur in all_nonevent_dbfs)
    print(f"Average Non-Event dBFS: {avg_nonevent_dbfs:.1f}db")

    event_db_mask_s = torch.nn.functional.interpolate(
        event_db_mask.unsqueeze(0).unsqueeze(0), 
        scale_factor=1.0/sr,
        mode='linear',
    ).squeeze()
    # breakpoint()

    avg_event_dbfs = event_db_mask[event_db_mask > 0].mean().item()
    print(f"Average Event dBFS: {avg_event_dbfs:.1f}db")
    print(f"Duration of All Events:                  {(event_db_mask > 0).sum().item() / sr:.1f} seconds")
    if (event_db_mask > avg_nonevent_dbfs + loud_db_offset).sum() == 0:
        print(f"No loud events (> avg + {loud_db_offset}dB) detected.")
        return
    print(f"Duration of Loud Events (> avg + {loud_db_offset}dB): {(event_db_mask > avg_nonevent_dbfs + loud_db_offset).sum().item() / sr:.1f} seconds")
    avg_loud_dbfs = event_db_mask[event_db_mask > avg_nonevent_dbfs + loud_db_offset].mean().item()
    print(f"Avg dBFS of Loud Events: {avg_loud_dbfs:.1f}db = Average + {avg_loud_dbfs - avg_nonevent_dbfs:.1f}db")

if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda"
    # Load model and transform
    model = PEAudioFrame.from_config(f"pe-a-frame-{args.model_size}", pretrained=True).to(device)
    transform = PEAudioFrameTransform.from_config(f"pe-a-frame-{args.model_size}")

    # Patch the forward method to include span detection
    model.forward = PEAudioFrame_forward.__get__(model, PEAudioFrame)

    desc2det_thres = {
        "open and close closet": 0.55,
        "door slam": 0.65,
        "chopstick clatter": 0.5,
        "ceramic dish clatter": 0.5,
        "kitchen clatter": 0.5,
        "cutlery clinking": 0.5,
        "thud or thump": 0.6,
        "clack and clunk": 0.6,
        "chopping or cutting": 0.4,
        "human talking": 0.65,
        "oil sizzle": 0.5,
        "water splattering": 0.5,
        "plastic bag rustle": 0.5,
    }

    desc2db_thres = {
        "open and close closet": 55,
        "door slam": 55,
        "chopstick clatter": 55,
        "ceramic dish clatter": 55,
        "kitchen clatter": 55,
        "cutlery clinking": 55,
        "thud or thump": 55,
        "clack and clunk": 55,
        "chopping or cutting": 55,
        # Noises below are generally softer, but still disturbing.
        # So we set a lower dBFS threshold to catch more of them.
        "human talking": 50,
        "oil sizzle": 50,
        "water splattering": 50,
        "plastic bag rustle": 50,
    }

    analyze_audio_labels(
        model=model,
        transform=transform,
        input_file=args.input_file,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
        weak_thres_discount=args.weak_thres_discount,
        db_offset=args.db_offset,
        loud_db_offset=args.loud_db_offset,
        desc2det_thres=desc2det_thres,
        desc2db_thres=desc2db_thres,
        device=device,
        print_abs_time=args.abs_time,
        debug=args.debug,
    )
    