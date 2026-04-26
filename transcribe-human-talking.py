import argparse
import difflib
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


SUPPORTED_MEDIA_EXTENSIONS = {".mp3", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CHINESE_LANGUAGE_CODES = {"zh", "zh-cn", "zh-tw", "chinese", "mandarin", "cantonese", "yue"}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a media file or a folder of media files with openai/whisper-large-v3."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Media file or folder containing media files to transcribe.",
    )
    parser.add_argument(
        "--model-id",
        default="openai/whisper-large-v3",
        help="Hugging Face model ID for automatic speech recognition.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for video files.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint, for example 'en' or 'zh'.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper generation task.",
    )
    parser.add_argument(
        "--chunk-length-s",
        type=float,
        default=30.0,
        help="Chunk length in seconds for long-form transcription.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used by the ASR pipeline.",
    )
    parser.add_argument(
        "--fallback-text",
        default="unrecognizable",
        help="Text to print when the audio is too noisy or the transcription is unreliable.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4],
        help="Whisper decoding temperatures. Multiple values enable fallback decoding.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=positive_float,
        default=1.35,
        help="Fallback threshold for repetitive outputs. Lower values are stricter.",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=-0.5,
        help="Fallback threshold for low-confidence decoding. Higher values are stricter.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.3,
        help="Threshold for treating a clip as no speech / unusable speech.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=non_negative_int,
        default=32,
        help="Maximum number of tokens Whisper may emit for each clip. Lower values are more conservative.",
    )
    parser.add_argument(
        "--force-timestamps",
        action="store_true",
        help="Use Whisper's timestamp-based decoding path even for short clips so rejection heuristics apply.",
    )
    parser.add_argument(
        "--consensus-max-duration",
        type=positive_float,
        default=8.0,
        help="Apply a second decode to clips up to this duration in seconds and reject unstable results.",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.72,
        help="Minimum similarity required between two short-clip decodes to keep a transcript.",
    )
    return parser.parse_args()


def find_video_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS]
    return sorted(files)


def resolve_input_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
            return []
        return [input_path]

    if input_path.is_dir():
        return find_video_files(input_path, recursive)

    return []


def load_audio_for_whisper(video_path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(video_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = F.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
    return waveform.squeeze(0)


def build_asr_pipeline(model_id: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        ignore_warning=True,
    )
    return asr_pipeline, processor.feature_extractor.sampling_rate


def has_repeated_substring(text: str, min_repeats: int = 4) -> bool:
    for unit_length in range(2, min(16, len(text) // min_repeats + 1)):
        for start in range(0, len(text) - unit_length * min_repeats + 1):
            unit = text[start : start + unit_length]
            if len(set(unit)) == 1:
                continue

            repeats = 1
            cursor = start + unit_length
            while cursor + unit_length <= len(text) and text[cursor : cursor + unit_length] == unit:
                repeats += 1
                cursor += unit_length

            if repeats >= min_repeats:
                covered = repeats * unit_length
                if covered >= max(12, len(text) // 3):
                    return True

    return False


def count_cjk_characters(text: str) -> int:
    total = 0
    for ch in text:
        codepoint = ord(ch)
        if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
            total += 1
    return total


def violates_expected_script(text: str, language: str | None) -> bool:
    if not language:
        return False

    normalized_language = language.strip().lower()
    if normalized_language not in CHINESE_LANGUAGE_CODES:
        return False

    cjk_count = count_cjk_characters(text)
    latin_alpha_count = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    if cjk_count == 0 and latin_alpha_count >= 4:
        return True

    if latin_alpha_count >= max(6, cjk_count * 2):
        return True

    return False


def normalize_for_comparison(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def transcripts_agree(first: str, second: str, threshold: float) -> bool:
    first_normalized = normalize_for_comparison(first)
    second_normalized = normalize_for_comparison(second)

    if not first_normalized or not second_normalized:
        return False

    if first_normalized == second_normalized:
        return True

    similarity = difflib.SequenceMatcher(None, first_normalized, second_normalized).ratio()
    return similarity >= threshold


def looks_hallucinated(text: str, duration_seconds: float, language: str | None = None) -> bool:
    normalized = "".join(ch for ch in text if not ch.isspace())
    if not normalized:
        return True

    if violates_expected_script(text, language):
        return True

    max_chars = max(24, int(duration_seconds * 12))
    if len(normalized) > max_chars:
        return True

    if len(normalized) >= 6 and len(set(normalized)) <= 2:
        return True

    if has_repeated_substring(normalized):
        return True

    longest_run = 1
    current_run = 1
    for idx in range(1, len(normalized)):
        if normalized[idx] == normalized[idx - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    return longest_run >= 6


def transcribe_audio(
    asr_pipeline,
    audio: torch.Tensor,
    sample_rate: int,
    chunk_length_s: float,
    batch_size: int,
    generate_kwargs: dict,
    return_timestamps: bool,
) -> str:
    result = asr_pipeline(
        {"array": audio.cpu().numpy(), "sampling_rate": sample_rate},
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        ignore_warning=True,
        generate_kwargs=generate_kwargs,
        return_timestamps=return_timestamps,
    )
    return result["text"].strip()


def main() -> int:
    args = parse_args()

    if not args.input_path.exists():
        print(f"Input path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    video_files = resolve_input_files(args.input_path, args.recursive)
    if not video_files:
        print(f"No supported video files found for {args.input_path}", file=sys.stderr)
        return 1

    asr_pipeline, target_sample_rate = build_asr_pipeline(args.model_id)

    generate_kwargs = {
        "task": args.task,
        "temperature": tuple(args.temperature),
        "compression_ratio_threshold": args.compression_ratio_threshold,
        "logprob_threshold": args.logprob_threshold,
        "no_speech_threshold": args.no_speech_threshold,
        "condition_on_prev_tokens": False,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.language:
        generate_kwargs["language"] = args.language

    for video_path in video_files:
        try:
            audio = load_audio_for_whisper(video_path, target_sample_rate)
            duration_seconds = audio.numel() / target_sample_rate
            text = transcribe_audio(
                asr_pipeline=asr_pipeline,
                audio=audio,
                sample_rate=target_sample_rate,
                chunk_length_s=args.chunk_length_s,
                batch_size=args.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=args.force_timestamps,
            )

            if duration_seconds <= args.consensus_max_duration:
                consensus_kwargs = dict(generate_kwargs)
                consensus_kwargs["temperature"] = (0.0,)
                alternate_text = transcribe_audio(
                    asr_pipeline=asr_pipeline,
                    audio=audio,
                    sample_rate=target_sample_rate,
                    chunk_length_s=args.chunk_length_s,
                    batch_size=1,
                    generate_kwargs=consensus_kwargs,
                    return_timestamps=not args.force_timestamps,
                )
                if not transcripts_agree(text, alternate_text, args.consensus_threshold):
                    text = args.fallback_text
        except Exception as exc:
            print(f"[{video_path}] ERROR: {exc}", file=sys.stderr)
            continue

        if looks_hallucinated(text, duration_seconds, args.language):
            text = args.fallback_text
        print(f"=== {video_path} ===")
        print(text)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())