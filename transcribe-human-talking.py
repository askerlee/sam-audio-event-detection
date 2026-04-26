import argparse
import difflib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as NN_F
import torchaudio
import torchaudio.functional as F
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


SUPPORTED_MEDIA_EXTENSIONS = {".mp3", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
CHINESE_LANGUAGE_CODES = {"zh", "zh-cn", "zh-tw", "chinese", "mandarin", "cantonese", "yue"}
WHISPER_SAMPLE_RATE = 16000


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
        default="zh",
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
    parser.add_argument(
        "--min-token-confidence",
        type=float,
        default=0.55,
        help="Minimum average next-token probability required for short-clip direct decoding.",
    )
    parser.add_argument(
        "--min-voiced-fraction",
        type=float,
        default=0.08,
        help="Minimum fraction of the clip that must survive VAD for the transcript to be trusted.",
    )
    parser.add_argument(
        "--seed-consensus-threshold",
        type=float,
        default=0.6,
        help="Minimum similarity required between two sampled Whisper decodes for short clips.",
    )
    parser.add_argument(
        "--seed-consensus-temperature",
        type=positive_float,
        default=0.8,
        help="Sampling temperature used for the seed-based double transcription check.",
    )
    parser.add_argument(
        "--seed-consensus-seeds",
        type=int,
        nargs="+",
        default=[1234, 5678, 91011],
        help="Seeds used for repeated sampled Whisper decoding on short clips.",
    )
    parser.add_argument(
        "--consensus-support-ratio",
        type=positive_float,
        default=0.75,
        help="Minimum fraction of decode variants that must support a transcript before it is accepted.",
    )
    parser.add_argument(
        "--deterministic-repeat-threshold",
        type=float,
        default=0.9,
        help="Minimum similarity required between repeated deterministic decodes of the same short clip.",
    )
    parser.add_argument(
        "--reruns",
        type=non_negative_int,
        default=3,
        help="Repeat the full short-clip decode stack this many times and require cross-rerun consensus.",
    )
    parser.add_argument(
        "--rerun-consensus-threshold",
        type=float,
        default=0.9,
        help="Minimum similarity required between final rerun transcripts for short clips.",
    )
    parser.add_argument(
        "--process-reruns",
        type=non_negative_int,
        default=3,
        help="Run short-clip transcription in fresh Python processes this many times and require cross-process agreement.",
    )
    parser.add_argument(
        "--min-duration-for-short-text-reject",
        type=positive_float,
        default=2.5,
        help="Reject very short transcripts when the clip is at least this long in seconds.",
    )
    parser.add_argument(
        "--min-cjk-chars-for-longer-clip",
        type=non_negative_int,
        default=3,
        help="Minimum number of CJK characters required once the clip is long enough for short-text rejection.",
    )
    parser.add_argument("--child-output-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-json-output", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-process-reruns", action="store_true", help=argparse.SUPPRESS)
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
    return asr_pipeline, processor, model, processor.feature_extractor.sampling_rate


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


def has_repeated_clauses(text: str, min_clause_length: int = 4, min_repeats: int = 2) -> bool:
    clauses = [clause.strip() for clause in re.split(r"[，。！？；,.;!?]+", text) if clause.strip()]
    if len(clauses) < min_repeats:
        return False

    counts: dict[str, int] = {}
    for clause in clauses:
        if len(clause) < min_clause_length:
            continue
        counts[clause] = counts.get(clause, 0) + 1
        if counts[clause] >= min_repeats:
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


def transcript_similarity(first: str, second: str) -> float:
    first_normalized = normalize_for_comparison(first)
    second_normalized = normalize_for_comparison(second)
    if not first_normalized or not second_normalized:
        return 0.0
    return difflib.SequenceMatcher(None, first_normalized, second_normalized).ratio()


def choose_consensus_transcript(candidates: list[str], threshold: float, support_ratio: float = 0.75) -> str | None:
    normalized_candidates = [candidate for candidate in candidates if normalize_for_comparison(candidate)]
    if not normalized_candidates:
        return None

    min_support = max(3, math.ceil(len(normalized_candidates) * support_ratio))

    best_candidate = None
    best_support = -1
    best_score = -1.0

    for candidate in normalized_candidates:
        similarities = [transcript_similarity(candidate, other) for other in normalized_candidates]
        support = sum(similarity >= threshold for similarity in similarities)
        mean_score = sum(similarities) / len(similarities)
        if support > best_support or (support == best_support and mean_score > best_score):
            best_candidate = candidate
            best_support = support
            best_score = mean_score

    if best_support < min_support:
        return None

    return best_candidate


def sampled_transcripts_are_stable(reference_text: str, sampled_texts: list[str], threshold: float) -> bool:
    if not sampled_texts:
        return False

    if any(transcript_similarity(reference_text, sampled_text) < threshold for sampled_text in sampled_texts):
        return False

    for idx, left_text in enumerate(sampled_texts):
        for right_text in sampled_texts[idx + 1 :]:
            if transcript_similarity(left_text, right_text) < threshold:
                return False

    return True


def transcripts_are_pairwise_stable(texts: list[str], threshold: float) -> bool:
    normalized_texts = [text for text in texts if normalize_for_comparison(text)]
    if len(normalized_texts) < 2:
        return False

    for idx, left_text in enumerate(normalized_texts):
        for right_text in normalized_texts[idx + 1 :]:
            if transcript_similarity(left_text, right_text) < threshold:
                return False

    return True


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

    if has_repeated_clauses(text):
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


def looks_hallucinated_with_args(text: str, duration_seconds: float, args: argparse.Namespace) -> bool:
    if looks_hallucinated(text, duration_seconds, args.language):
        return True

    normalized = "".join(ch for ch in text if not ch.isspace())
    if duration_seconds >= args.min_duration_for_short_text_reject:
        if count_cjk_characters(normalized) < args.min_cjk_chars_for_longer_clip:
            return True

    return False


def compute_voiced_fraction(audio: torch.Tensor, sample_rate: int) -> float:
    if audio.numel() == 0:
        return 0.0

    waveform = audio.unsqueeze(0)
    voiced_waveform = F.vad(waveform, sample_rate=sample_rate)
    if voiced_waveform.numel() == 0:
        return 0.0

    return voiced_waveform.shape[-1] / waveform.shape[-1]


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


def direct_generate_with_confidence(
    processor,
    model,
    audio: torch.Tensor,
    sample_rate: int,
    generate_kwargs: dict,
) -> tuple[str, float]:
    inputs = processor(
        audio.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output = model.generate(
        **model_inputs,
        return_dict_in_generate=True,
        output_scores=True,
        **generate_kwargs,
    )

    text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
    if not output.scores:
        return text, 1.0

    generated_token_ids = output.sequences[0, -len(output.scores) :]
    token_probabilities = []
    for step_scores, token_id in zip(output.scores, generated_token_ids):
        probabilities = NN_F.softmax(step_scores[0], dim=-1)
        token_probabilities.append(probabilities[token_id].item())

    average_probability = sum(token_probabilities) / len(token_probabilities)
    return text, average_probability


def sampled_generate_text(
    processor,
    model,
    audio: torch.Tensor,
    sample_rate: int,
    generate_kwargs: dict,
    seed: int,
) -> str:
    inputs = processor(
        audio.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.random.fork_rng(devices=[model.device] if model.device.type == "cuda" else []):
        torch.manual_seed(seed)
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        output = model.generate(
            **model_inputs,
            do_sample=True,
            **generate_kwargs,
        )
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def remove_language_hint(generate_kwargs: dict) -> dict:
    updated_kwargs = dict(generate_kwargs)
    updated_kwargs.pop("language", None)
    return updated_kwargs


def append_multi_value_option(command: list[str], flag: str, values: list[float] | list[int]) -> None:
    command.append(flag)
    command.extend(str(value) for value in values)


def trim_common_name_prefix(names: list[str]) -> str:
    if not names:
        return ""

    prefix = os.path.commonprefix(names)
    if not prefix:
        return ""

    last_separator = max(prefix.rfind("_"), prefix.rfind("-"), prefix.rfind(" "))
    if last_separator >= 0:
        return prefix[: last_separator + 1]

    return prefix


def format_summary_stem(stem: str) -> str:
    match = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})", stem)
    if match:
        return f"{match.group(1)}:{match.group(2)}-{match.group(3)}:{match.group(4)}"

    return stem


def build_summary_labels(video_files: list[Path]) -> dict[Path, str]:
    if not video_files:
        return {}

    parent_paths = [str(video_path.parent) for video_path in video_files]
    common_parent = Path(os.path.commonpath(parent_paths)) if parent_paths else None
    stem_prefix = trim_common_name_prefix([video_path.stem for video_path in video_files])

    labels: dict[Path, str] = {}
    for video_path in video_files:
        stem = video_path.stem
        shortened_stem = stem[len(stem_prefix) :] if stem.startswith(stem_prefix) else stem
        formatted_stem = format_summary_stem(shortened_stem or stem)
        relative_dir = ""
        if common_parent is not None:
            try:
                relative_dir = str(video_path.parent.relative_to(common_parent))
            except ValueError:
                relative_dir = str(video_path.parent)

        if relative_dir in {"", "."}:
            labels[video_path] = formatted_stem
        else:
            labels[video_path] = f"{relative_dir}/{formatted_stem}"

    return labels


def build_child_process_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.input_path),
        "--model-id",
        args.model_id,
        "--task",
        args.task,
        "--chunk-length-s",
        str(args.chunk_length_s),
        "--batch-size",
        str(args.batch_size),
        "--fallback-text",
        args.fallback_text,
        "--compression-ratio-threshold",
        str(args.compression_ratio_threshold),
        "--logprob-threshold",
        str(args.logprob_threshold),
        "--no-speech-threshold",
        str(args.no_speech_threshold),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--consensus-max-duration",
        str(args.consensus_max_duration),
        "--consensus-threshold",
        str(args.consensus_threshold),
        "--min-token-confidence",
        str(args.min_token_confidence),
        "--min-voiced-fraction",
        str(args.min_voiced_fraction),
        "--seed-consensus-threshold",
        str(args.seed_consensus_threshold),
        "--seed-consensus-temperature",
        str(args.seed_consensus_temperature),
        "--consensus-support-ratio",
        str(args.consensus_support_ratio),
        "--deterministic-repeat-threshold",
        str(args.deterministic_repeat_threshold),
        "--reruns",
        str(args.reruns),
        "--rerun-consensus-threshold",
        str(args.rerun_consensus_threshold),
        "--process-reruns",
        "0",
        "--min-duration-for-short-text-reject",
        str(args.min_duration_for_short_text_reject),
        "--min-cjk-chars-for-longer-clip",
        str(args.min_cjk_chars_for_longer_clip),
        "--child-json-output",
        "--disable-process-reruns",
    ]

    if args.recursive:
        command.append("--recursive")
    if args.language:
        command.extend(["--language", args.language])

    append_multi_value_option(command, "--temperature", args.temperature)
    append_multi_value_option(command, "--seed-consensus-seeds", args.seed_consensus_seeds)

    if args.force_timestamps:
        command.append("--force-timestamps")

    return command


def choose_process_consensus_text(args: argparse.Namespace, texts: list[str]) -> str:
    if not transcripts_are_pairwise_stable(texts, args.rerun_consensus_threshold):
        return args.fallback_text

    consensus_text = choose_consensus_transcript(texts, args.rerun_consensus_threshold, 1.0)
    if consensus_text is None:
        return args.fallback_text

    return consensus_text


def run_fresh_process_reruns(args: argparse.Namespace, video_files: list[Path]) -> dict[Path, str]:
    rerun_count = max(1, args.process_reruns)
    rerun_results: list[dict[str, str]] = []

    for _ in range(rerun_count):
        completed = subprocess.run(
            build_child_process_command(args),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "child process failed"
            raise RuntimeError(stderr)
        try:
            rerun_results.append(json.loads(completed.stdout))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"failed to parse child output: {exc}") from exc

    consensus_by_path: dict[Path, str] = {}
    for video_path in video_files:
        path_key = str(video_path)
        texts = [result.get(path_key, args.fallback_text) for result in rerun_results]
        consensus_by_path[video_path] = choose_process_consensus_text(args, texts)

    return consensus_by_path


def transcribe_short_clip_once(
    args: argparse.Namespace,
    asr_pipeline,
    processor,
    model,
    audio: torch.Tensor,
    sample_rate: int,
    base_generate_kwargs: dict,
) -> str:
    duration_seconds = audio.numel() / sample_rate
    text = transcribe_audio(
        asr_pipeline=asr_pipeline,
        audio=audio,
        sample_rate=sample_rate,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        generate_kwargs=base_generate_kwargs,
        return_timestamps=args.force_timestamps,
    )

    consensus_kwargs = dict(base_generate_kwargs)
    consensus_kwargs["temperature"] = (0.0,)
    candidate_texts = [text]
    repeated_text = transcribe_audio(
        asr_pipeline=asr_pipeline,
        audio=audio,
        sample_rate=sample_rate,
        chunk_length_s=args.chunk_length_s,
        batch_size=1,
        generate_kwargs=consensus_kwargs,
        return_timestamps=args.force_timestamps,
    )
    candidate_texts.append(repeated_text)
    if not transcripts_agree(text, repeated_text, args.deterministic_repeat_threshold):
        return args.fallback_text

    alternate_text = transcribe_audio(
        asr_pipeline=asr_pipeline,
        audio=audio,
        sample_rate=sample_rate,
        chunk_length_s=args.chunk_length_s,
        batch_size=1,
        generate_kwargs=consensus_kwargs,
        return_timestamps=not args.force_timestamps,
    )
    candidate_texts.append(alternate_text)
    if not transcripts_agree(text, alternate_text, args.consensus_threshold):
        return args.fallback_text

    if args.language:
        language_free_text = transcribe_audio(
            asr_pipeline=asr_pipeline,
            audio=audio,
            sample_rate=sample_rate,
            chunk_length_s=args.chunk_length_s,
            batch_size=1,
            generate_kwargs=remove_language_hint(consensus_kwargs),
            return_timestamps=args.force_timestamps,
        )
        candidate_texts.append(language_free_text)
        if not transcripts_agree(text, language_free_text, args.consensus_threshold):
            return args.fallback_text

    direct_text, average_probability = direct_generate_with_confidence(
        processor=processor,
        model=model,
        audio=audio,
        sample_rate=sample_rate,
        generate_kwargs=consensus_kwargs,
    )
    candidate_texts.append(direct_text)
    if average_probability < args.min_token_confidence:
        return args.fallback_text
    if not transcripts_agree(text, direct_text, args.consensus_threshold):
        return args.fallback_text

    sampled_kwargs = dict(consensus_kwargs)
    sampled_kwargs["temperature"] = args.seed_consensus_temperature
    sampled_kwargs.pop("compression_ratio_threshold", None)
    sampled_kwargs.pop("logprob_threshold", None)
    sampled_kwargs.pop("no_speech_threshold", None)
    sampled_texts = [
        sampled_generate_text(
            processor=processor,
            model=model,
            audio=audio,
            sample_rate=sample_rate,
            generate_kwargs=sampled_kwargs,
            seed=seed,
        )
        for seed in args.seed_consensus_seeds
    ]
    candidate_texts.extend(sampled_texts)
    if not sampled_transcripts_are_stable(text, sampled_texts, args.seed_consensus_threshold):
        return args.fallback_text

    consensus_text = choose_consensus_transcript(
        candidate_texts,
        min(args.consensus_threshold, args.seed_consensus_threshold),
        args.consensus_support_ratio,
    )
    if consensus_text is None:
        return args.fallback_text

    if looks_hallucinated_with_args(consensus_text, duration_seconds, args):
        return args.fallback_text

    return consensus_text


def transcribe_video_files(
    args: argparse.Namespace,
    video_files: list[Path],
    generate_kwargs: dict,
) -> list[tuple[Path, str]]:
    results: list[tuple[Path, str]] = []
    asr_pipeline = None
    processor = None
    model = None
    target_sample_rate = WHISPER_SAMPLE_RATE

    def ensure_asr_resources():
        nonlocal asr_pipeline, processor, model, target_sample_rate
        if asr_pipeline is None or processor is None or model is None:
            asr_pipeline, processor, model, target_sample_rate = build_asr_pipeline(args.model_id)

    for video_path in video_files:
        try:
            audio = load_audio_for_whisper(video_path, target_sample_rate)
            duration_seconds = audio.numel() / target_sample_rate
            voiced_fraction = compute_voiced_fraction(audio, target_sample_rate)
            if voiced_fraction < args.min_voiced_fraction:
                text = args.fallback_text
            elif duration_seconds <= args.consensus_max_duration:
                ensure_asr_resources()
                rerun_texts = [
                    transcribe_short_clip_once(
                        args=args,
                        asr_pipeline=asr_pipeline,
                        processor=processor,
                        model=model,
                        audio=audio,
                        sample_rate=target_sample_rate,
                        base_generate_kwargs=generate_kwargs,
                    )
                    for _ in range(max(1, args.reruns))
                ]
                text = choose_process_consensus_text(args, rerun_texts)
            else:
                ensure_asr_resources()
                text = transcribe_audio(
                    asr_pipeline=asr_pipeline,
                    audio=audio,
                    sample_rate=target_sample_rate,
                    chunk_length_s=args.chunk_length_s,
                    batch_size=args.batch_size,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=args.force_timestamps,
                )

        except Exception as exc:
            print(f"[{video_path}] ERROR: {exc}", file=sys.stderr)
            continue

        if looks_hallucinated_with_args(text, duration_seconds, args):
            text = args.fallback_text
        results.append((video_path, text))

    return results


def main() -> int:
    args = parse_args()

    if not args.input_path.exists():
        print(f"Input path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    video_files = resolve_input_files(args.input_path, args.recursive)
    if not video_files:
        print(f"No supported video files found for {args.input_path}", file=sys.stderr)
        return 1

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

    if args.process_reruns > 0 and not args.disable_process_reruns:
        results = [(video_path, text) for video_path, text in run_fresh_process_reruns(args, video_files).items()]
    else:
        results = transcribe_video_files(args, video_files, generate_kwargs)

    if args.child_json_output:
        print(json.dumps({str(video_path): text for video_path, text in results}, ensure_ascii=False))
        return 0

    summary_labels = build_summary_labels([video_path for video_path, _ in results])
    for video_path, text in results:
        if args.child_output_only:
            print(text)
        else:
            print(f"{summary_labels.get(video_path, video_path.stem)}\t{text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())