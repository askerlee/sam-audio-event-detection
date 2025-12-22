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
parser.add_argument("--segment_seconds", type=float, default=30.0, help="Chunk length in seconds (e.g. 5, 10, 30)")
parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate (resample if needed)")
parser.add_argument("--predict_spans", action="store_true", help="Enable span prediction (slower, may improve quality)")
parser.add_argument("--reranking_candidates", type=int, default=1, help="Number of candidates to rerank (quality vs speed)")
parser.add_argument("--detection_threshold", type=float, default=0.5, help="Threshold for event detection")
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
    "opening and closing a closet",
    "person talking",
    "door slamming",
    "chopstick cluttering",
    "dishes cluttering",
]

# Process inputs
inputs = transform(audio=audio_file, text=descriptions).to(device)

# Run inference
with torch.inference_mode():
    # The default threshold is 0.3. Higher threshold means more strict detection.
    outputs = model(**inputs, return_spans=True, threshold=args.detection_threshold)

# Print detected time spans for each event
for description, spans in zip(descriptions, outputs.spans):
    if spans:
        span_str = ", ".join([f"({start:.2f}s, {end:.2f}s)" for start, end in spans])
        print(f'"{description}": [{span_str}]')
    else:
        print(f'"{description}": No events detected')
