import os
import re
import math
import argparse

import torch
import torchaudio
import torchaudio.functional as F
from pydub import AudioSegment
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor


def slugify(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] or "desc"


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
args = parser.parse_args()

# -------------------- load model --------------------
model_id = f"facebook/sam-audio-{args.model_size}"
model = SAMAudio.from_pretrained(model_id).eval().cuda()
processor = SAMAudioProcessor.from_pretrained(model_id)

input_file = args.input_file
input_trunk = input_file.rsplit(".", 1)[0]

# If MP4, extract to WAV once
if input_file.lower().endswith(".mp4"):
    audio_file = input_trunk + ".wav"
    if not os.path.exists(audio_file):
        print(f"Extracting audio from {input_file} -> {audio_file} ...")
        video = AudioSegment.from_file(input_file, format="mp4")
        video.export(audio_file, format="wav")
    input_file = audio_file

# -------------------- load + resample once --------------------
wav, sr = torchaudio.load(input_file)  # wav: (channels, samples)
target_sr = processor.audio_sampling_rate
if sr != target_sr:
    wav = F.resample(wav, orig_freq=sr, new_freq=target_sr)
    sr = target_sr

seg_samples = max(1, int(round(args.segment_seconds * sr)))
total_samples = wav.shape[1]
num_segments = max(1, math.ceil(total_samples / seg_samples))

print(f"Loaded: {input_file} | sr={sr} | chunk={args.segment_seconds}s ({seg_samples} samples) | segments={num_segments}")

descriptions = [
    "open and close a closet",
    "human voice",
    "door slamming",
    "chopstick clutter",
    "dishes clutter",
]

# -------------------- process each description over chunks --------------------
for raw_desc in descriptions:
    prompt = f"Isolate the sound of {raw_desc}."
    out_tag = slugify(raw_desc)

    target_chunks = []
    residual_chunks = []

    pbar = tqdm(
        split_waveform(wav, seg_samples),
        total=num_segments,
        desc=f"[{out_tag}]",
        unit="seg",
        dynamic_ncols=True,
    )

    for chunk, valid_len in pbar:
        batch = processor(
            audios=[chunk],        # waveform tensor (channels, samples)
            descriptions=[prompt], # one prompt per audio
        ).to("cuda")

        # text_ranker is used only when args.reranking_candidates > 1.
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=args.predict_spans,
                reranking_candidates=args.reranking_candidates,
            )

        tgt = result.target[0].cpu()
        res = result.residual[0].cpu()
        target_chunks.append(tgt)
        residual_chunks.append(res)

    full_target = torch.cat(target_chunks, dim=-1)    # (channels, total_samples)
    full_residual = torch.cat(residual_chunks, dim=-1)

    full_target_rs   = F.resample(full_target,   orig_freq=sr, new_freq=args.sample_rate)
    full_residual_rs = F.resample(full_residual, orig_freq=sr, new_freq=args.sample_rate)
    torchaudio.save(f"{input_trunk}-{out_tag}.wav",          full_target_rs,   args.sample_rate)
    torchaudio.save(f"{input_trunk}-{out_tag}-residual.wav", full_residual_rs, args.sample_rate)

    print(f"Saved: {input_trunk}-{out_tag}.wav and -residual.wav")
