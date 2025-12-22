import torch
import torchaudio
import torchaudio.functional as F
from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
import os
import re
import math
import argparse
from pydub import AudioSegment
from tqdm import tqdm
from typing import Iterable, List, Tuple, Dict, Union
from collections import defaultdict

Interval = Tuple[float, float]
Events = Dict[str, Union[str, List[Interval]]]

def mmss_to_sec(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)

def sec_to_mmss(x: int) -> str:
    return f"{x//60:02d}:{x%60:02d}"

def merge_intervals(intervals: Iterable[Interval], tolerance: float = 0.0) -> List[Interval]:
    xs = sorted((min(s, e), max(s, e)) for s, e in intervals)  # normalize + sort
    merged: List[Interval] = []
    for s, e in xs:
        if not merged:
            merged.append((s, e))
            continue

        ps, pe = merged[-1]
        # merge if the gap is small enough
        if s <= pe + tolerance:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def print_overlap_timeline(events: Events, tolerance_sec: int = 0) -> None:
    """
    Prints segments in time order.
    If events overlap in time, their labels appear together in that segment.
    tolerance_sec expands each interval by +/- tolerance_sec seconds.
    """
    painted = defaultdict(set)  # second -> set(labels)

    # 1) paint timeline per second (inclusive endpoints)
    for label, spans in events.items():
        if not spans or isinstance(spans, str):  # e.g. "No events detected"
            continue
        for a, b in spans:
            s, e = mmss_to_sec(a), mmss_to_sec(b)
            if e < s:
                s, e = e, s
            s = max(0, s - tolerance_sec)
            e = e + tolerance_sec
            for t in range(s, e + 1):
                painted[t].add(label)

    if not painted:
        return

    # 2) compress consecutive seconds with same label-set
    t0 = min(painted)
    t1 = max(painted)
    cur_set = painted.get(t0, set())
    seg_start = t0

    def flush(seg_s: int, seg_e: int, labels: set):
        if labels:
            labs = ", ".join(sorted(labels))
            print(f"{sec_to_mmss(seg_s)}-{sec_to_mmss(seg_e)}: {labs}")

    for t in range(t0 + 1, t1 + 2):  # +2 to flush last segment
        s = painted.get(t, set())
        if s != cur_set:
            flush(seg_start, t - 1, cur_set)
            seg_start = t
            cur_set = s

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

# -------------------- args --------------------
parser = argparse.ArgumentParser(description="Audio Source Separation using SAM-Audio (chunked + tqdm)")
parser.add_argument("--model_size", type=str, default="large", choices=["small", "medium", "large"], help="Model size to use")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input audio file (.wav/.mp4/...)")
parser.add_argument("--segment_seconds", type=float, default=120.0, help="Chunk length in seconds (e.g. 5, 10, 30)")
parser.add_argument("--max_segments", type=int, default=0, help="Maximum number of segments to process (0 = all)")
parser.add_argument("--detection_threshold", type=float, default=0.5, help="Threshold for event detection")
parser.add_argument("--sample_rate", type=int, default=48000, help="Target sample rate for event detection")
args = parser.parse_args()

device = "cuda"
# Load model and transform
model = PEAudioFrame.from_config(f"pe-a-frame-{args.model_size}", pretrained=True).to(device)
transform = PEAudioFrameTransform.from_config(f"pe-a-frame-{args.model_size}")

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

descriptions = [
    "open and close a closet",
    "human voice",
    "door slams",
    "chopstick clutters",
    "dishes clutters",
]

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

desc2intervals = {desc: [] for desc in descriptions}

for chunk, valid_len in pbar:
    chunk_rs = F.resample(chunk, orig_freq=sr, new_freq=args.sample_rate)
    # Process inputs
    inputs = transform(audio=[chunk_rs], text=descriptions, sampling_rate=args.sample_rate).to(device)

    # Run inference
    with torch.inference_mode():
        outputs = model(**inputs, return_spans=True, threshold=args.detection_threshold)

    # Print detected time spans for each event
    for description, spans in zip(descriptions, outputs.spans):
        if spans:
            desc2intervals[description].extend([(float(s) + t0, float(e) + t0) for s, e in spans])

    t0 += args.segment_seconds

all_events = {}

for description in descriptions:
    spans = merge_intervals(desc2intervals[description], tolerance=2)
    if spans:
        span_strs = []
        for start, end in spans:
            start_min = int(start) // 60
            start_sec = int(start) % 60
            end_min = int(end) // 60
            end_sec = int(end) % 60
            span_strs.append((f"{start_min:02d}:{start_sec:02d}", f"{end_min:02d}:{end_sec:02d}"))

        all_events[description] = span_strs

        #span_str = ", ".join(span_strs)
        #print(f'"{description}": [{span_str}]')
    #else:
    #    print(f'"{description}": No events detected')

print_overlap_timeline(all_events, tolerance_sec=1)
