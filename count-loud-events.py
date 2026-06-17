import argparse
import re
from collections import Counter
from pathlib import Path


EVENT_DB_PATTERN = re.compile(
    r"\((?P<absolute>[+-]?\d+(?:\.\d+)?)\s*db\s*/\s*(?P<delta>[+-]?\d+(?:\.\d+)?)\s*db\)",
    re.IGNORECASE,
)
MAX_SUBTITLE_LINES = 12
folder_mapping = { '<Zeng Zheng>-<DVD1>': 'cdrt-8-9',
                   '<Zeng Zheng>-<DVD2>': 'cdrt-10-12-1'}
char_mapping = { '<': '＜', '>': '＞'}
excluded_files = { '＜19 October 2025＞-＜11-39-03＞-＜kitchen noises＞.mp4'}
included_files = { '13 October 2025', 
                   '＜15 October 2025＞-＜20-13-44＞-＜kitchen noises (sound level meter)＞',
                   '＜01 January 2026＞' }

def matches_any_partial_name(value: str, patterns: set[str]) -> bool:
    return any(pattern and pattern in value for pattern in patterns)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count events in a text file whose dB value meets a threshold. "
            "By default the threshold is applied to the signed delta value after the slash, "
            "for example '+2.8db' in '(55.2db/+2.8db)'."
        )
    )
    parser.add_argument("input_file", type=Path, help="Path to the input text file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=13.0,
        help="Minimum dB value required for an event to be counted.",
    )
    parser.add_argument(
        "--metric",
        choices=["delta", "absolute"],
        default="delta",
        help="Which dB value to compare against the threshold.",
    )
    parser.add_argument(
        "--show-lines",
        action="store_true",
        help="Print the count for each matching line in addition to the total.",
    )
    parser.add_argument(
        "--breakdown-by-file",
        action="store_true",
        help="Print loud-event counts grouped by the first field in each input row.",
    )
    parser.add_argument(
        "--loud-event-freq-thres",
        type=int,
        default=0,
        help="Only print per-file loud-event counts greater than or equal to this value.",
    )
    parser.add_argument(
        "--num-loudest-files",
        type=int,
        default=-1,
        help="Number of top files to print after frequency filtering; -1 prints all files.",
    )
    parser.add_argument(
        "--output-kept-script",
        type=Path,
        help="Write all input lines whose derived file path is kept after filtering to this file.",
    )
    parser.add_argument(
        "--output-srt-files",
        action="store_true",
        help="Write one SRT file per kept file, using the derived output path with '.srt' suffix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write kept-script output and derived SRT files into.",
    )
    parser.add_argument(
        "--kept-file-event-db-thres",
        type=float,
        help="When writing kept-file script lines, only save rows with at least one event meeting this dB threshold.",
    )
    return parser.parse_args()


def split_record_fields(line: str) -> tuple[str, str, str, str, str, str]:
    parts = line.strip().split(",", 3)
    if len(parts) < 4:
        return "", "", "", "", "", ""

    original_folder = parts[0]
    original_name = parts[1]
    formatted_folder = folder_mapping.get(parts[0], parts[0])
    for char, replacement in char_mapping.items():
        parts[1] = parts[1].replace(char, replacement)
    formatted_filename = parts[1] + ".mp4"
    formatted_path = formatted_folder + "/" + formatted_filename
    for excluded_file in excluded_files:
        if excluded_file in formatted_filename:
            return "", "", "", "", "", ""
    return original_folder, original_name, formatted_path, formatted_filename, parts[2], parts[3]


def count_events_in_line(line: str, threshold: float, metric: str) -> int:
    _, _, _, _, _, events_field = split_record_fields(line)
    if not events_field:
        return 0

    count = 0
    for event_text in events_field.split(";"):
        value = event_db_value(event_text, metric)
        if value is None:
            continue
        if value >= threshold:
            count += 1

    return count


def event_db_value(event_text: str, metric: str = "delta") -> float | None:
    match = EVENT_DB_PATTERN.search(event_text)
    if not match:
        return None
    return float(match.group(metric))

def sec_to_srt_timestamp(x: int) -> str:
    hours = x // 3600
    minutes = (x % 3600) // 60
    seconds = x % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},000"


def hhmmss_to_seconds(value: str) -> int:
    hours_text, minutes_text, seconds_text = value.split(":")
    return int(hours_text) * 3600 + int(minutes_text) * 60 + int(seconds_text)


def parse_time_range(time_range: str) -> tuple[int, int]:
    start_text, end_text = (part.strip() for part in time_range.split(" to ", 1))
    return hhmmss_to_seconds(start_text), hhmmss_to_seconds(end_text)


def subtitle_time_bounds(start_seconds: int, end_seconds: int) -> tuple[int, int]:
    return max(0, start_seconds - 1), end_seconds + 1


def playback_segment(start_seconds: int, end_seconds: int) -> tuple[int, int]:
    segment_start = max(0, start_seconds - 1)
    segment_duration = max(1, end_seconds - start_seconds + 2)
    return segment_start, segment_duration


def merge_playback_segments(
    segments: list[tuple[int, int]], max_gap_seconds: int = 1
) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged_segments: list[tuple[int, int]] = []
    current_start, current_duration = segments[0]
    current_end = current_start + current_duration

    for next_start, next_duration in segments[1:]:
        next_end = next_start + next_duration
        if next_start - current_end <= max_gap_seconds:
            current_end = max(current_end, next_end)
            continue

        merged_segments.append((current_start, current_end - current_start))
        current_start = next_start
        current_end = next_end

    merged_segments.append((current_start, current_end - current_start))
    return merged_segments


def merge_subtitle_rows(
    rows: list[tuple[str, str]], max_gap_seconds: int = 0
) -> list[tuple[int, int, list[str]]]:
    if not rows:
        return []

    merged_rows: list[tuple[int, int, list[str]]] = []
    current_start_seconds, current_end_seconds = subtitle_time_bounds(
        *parse_time_range(rows[0][0])
    )
    current_entries = [subtitle_entry(rows[0][0], rows[0][1])]

    for time_range, events_field in rows[1:]:
        start_seconds, end_seconds = parse_time_range(time_range)
        subtitle_start, subtitle_end = subtitle_time_bounds(start_seconds, end_seconds)
        entry = subtitle_entry(time_range, events_field)

        if subtitle_start <= current_end_seconds + max_gap_seconds:
            current_end_seconds = max(current_end_seconds, subtitle_end)
            current_entries.append(entry)
            continue

        merged_rows.append(
            (
                current_start_seconds,
                current_end_seconds,
                flatten_subtitle_entries(
                    trim_subtitle_entries(current_entries, MAX_SUBTITLE_LINES)
                ),
            )
        )
        current_start_seconds = subtitle_start
        current_end_seconds = subtitle_end
        current_entries = [entry]

    merged_rows.append(
        (
            current_start_seconds,
            current_end_seconds,
            flatten_subtitle_entries(
                trim_subtitle_entries(current_entries, MAX_SUBTITLE_LINES)
            ),
        )
    )
    return merged_rows


def subtitle_entry(time_range: str, events_field: str) -> tuple[str, list[str]]:
    return (
        time_range,
        [event_text.strip() for event_text in events_field.split(";") if event_text.strip()],
    )


def trim_subtitle_entries(
    entries: list[tuple[str, list[str]]], max_total_lines: int
) -> list[tuple[str, list[str]]]:
    structured_entries = [
        {
            "time_range": time_range,
            "events": [
                {
                    "text": event_text,
                    "delta": event_db_value(event_text, "delta"),
                }
                for event_text in event_lines
            ],
        }
        for time_range, event_lines in entries
    ]

    total_lines = sum(1 + len(entry["events"]) for entry in structured_entries)
    if total_lines <= max_total_lines:
        return entries

    while total_lines > max_total_lines:
        removable_entry_indexes = [
            entry_index
            for entry_index, entry in enumerate(structured_entries)
            if entry["events"]
        ]
        if not removable_entry_indexes:
            break

        lowest_priority_entry_index = min(
            removable_entry_indexes,
            key=lambda entry_index: (
                max(
                    event["delta"] if event["delta"] is not None else float("inf")
                    for event in structured_entries[entry_index]["events"]
                ),
                len(structured_entries[entry_index]["events"]),
                entry_index,
            ),
        )
        entry_line_count = 1 + len(structured_entries[lowest_priority_entry_index]["events"])
        if total_lines - entry_line_count >= max_total_lines:
            structured_entries[lowest_priority_entry_index]["events"] = []
            total_lines -= entry_line_count
            continue
        break

    removable_events = sorted(
        (
            event["delta"] if event["delta"] is not None else float("inf"),
            entry_index,
            event_index,
        )
        for entry_index, entry in enumerate(structured_entries)
        for event_index, event in enumerate(entry["events"])
    )

    for _, entry_index, event_index in removable_events:
        if total_lines <= max_total_lines:
            break
        entry = structured_entries[entry_index]
        event = entry["events"][event_index]
        if event is None:
            continue
        entry["events"][event_index] = None
        total_lines -= 1
        if not any(remaining is not None for remaining in entry["events"]):
            total_lines -= 1

    trimmed_entries: list[tuple[str, list[str]]] = []
    for entry in structured_entries:
        event_lines = [event["text"] for event in entry["events"] if event is not None]
        if event_lines:
            trimmed_entries.append((entry["time_range"], event_lines))
    return trimmed_entries


def flatten_subtitle_entries(entries: list[tuple[str, list[str]]]) -> list[str]:
    subtitle_lines: list[str] = []
    for time_range, event_lines in entries:
        subtitle_lines.append(time_range)
        subtitle_lines.extend(event_lines)
    return subtitle_lines


def resolve_output_path(path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return path
    return output_dir / path.name


def build_vlc_python(
    formatted_filename: str, segments: list[tuple[int, int]]
) -> str:
    segment_lines = [
        f"    ({segment_start}, {segment_duration}),"
        for segment_start, segment_duration in segments
    ]
    segments_block = "\n".join(segment_lines)

    return (
        "import queue\n"
        "import threading\n"
        "import time\n"
        "import vlc\n"
        "from pathlib import Path\n\n"
        f'video = Path(__file__).with_name("{formatted_filename}")\n\n'
        "segments = [\n"
        f"{segments_block}\n"
        "]\n\n"
        "command_queue = queue.SimpleQueue()\n\n"
        "def read_navigation_keys():\n"
        "    try:\n"
        "        import msvcrt\n"
        "    except ImportError:\n"
            "        import select\n"
            "        import sys\n"
            "        import termios\n"
            "        import tty\n\n"
            "        stdin = sys.stdin.fileno()\n"
            "        original_settings = termios.tcgetattr(stdin)\n"
            "        try:\n"
            "            tty.setcbreak(stdin)\n"
            "            while True:\n"
            "                ready, _, _ = select.select([sys.stdin], [], [], 0.1)\n"
            "                if not ready:\n"
            "                    continue\n"
            "                key = sys.stdin.read(1)\n"
            "                if key != \"\\x1b\":\n"
            "                    continue\n"
            "                if not select.select([sys.stdin], [], [], 0.05)[0]:\n"
            "                    continue\n"
            "                if sys.stdin.read(1) != \"[\":\n"
            "                    continue\n"
            "                if not select.select([sys.stdin], [], [], 0.05)[0]:\n"
            "                    continue\n"
            "                key = sys.stdin.read(1)\n"
            "                if key == \"D\":\n"
            "                    command_queue.put(\"prev\")\n"
            "                elif key == \"C\":\n"
            "                    command_queue.put(\"next\")\n"
            "        finally:\n"
            "            termios.tcsetattr(stdin, termios.TCSADRAIN, original_settings)\n"
            "        return\n\n"
            "    while True:\n"
            "        key = msvcrt.getch()\n"
            "        if key not in (b\"\\x00\", b\"\\xe0\"):\n"
            "            continue\n"
            "        key = msvcrt.getch()\n"
            "        if key == b\"K\":\n"
            "            command_queue.put(\"prev\")\n"
            "        elif key == b\"M\":\n"
            "            command_queue.put(\"next\")\n\n"
        "def wait_for_command_or_timeout(duration):\n"
        "    end_time = time.monotonic() + duration\n"
        "    while time.monotonic() < end_time:\n"
        "        try:\n"
        "            return command_queue.get_nowait()\n"
        "        except queue.Empty:\n"
        "            time.sleep(0.1)\n"
        "    return None\n\n"
        "instance = vlc.Instance()\n"
        "player = instance.media_player_new()\n"
        "media = instance.media_new(str(video))\n"
        "player.set_media(media)\n\n"
        "threading.Thread(target=read_navigation_keys, daemon=True).start()\n"
        "print(\"Controls: Left Arrow = previous event, Right Arrow = next event\")\n\n"
        "player.play()\n"
        "time.sleep(1.5)\n\n"
        "index = 0\n"
        "while 0 <= index < len(segments):\n"
        "    start, duration = segments[index]\n"
        "    start_label = f\"{start // 60:02d}:{start % 60:02d}\"\n"
        "    print(f\"Playing {index + 1}/{len(segments)} {start_label} for {duration}s\")\n"
        "    player.set_time(start * 1000)\n"
        "    time.sleep(0.2)\n"
        "    player.play()\n"
        "    command = wait_for_command_or_timeout(duration)\n"
        "    player.pause()\n"
        "    if command == \"prev\":\n"
        "        index = max(0, index - 1)\n"
        "        continue\n"
        "    if command == \"next\":\n"
        "        index += 1\n"
        "        continue\n"
        "    if index < len(segments) - 1:\n"
        "        time.sleep(1)\n\n"
        "    index += 1\n\n"
        "player.stop()\n"
    )


def build_python_launcher_batch(python_filename: str) -> str:
    return (
        "@echo off\r\n"
        "chcp 65001\r\n"
        "cd /d \"%~dp0\"\r\n"
        f"python \"{python_filename}\"\r\n"
        "pause\r\n"
    )

def main() -> int:
    args = parse_args()

    total_count = 0
    counts_by_file: Counter[str] = Counter()
    included_output_files: set[str] = set()
    script_lines_in_order: list[tuple[str, str]] = []
    srt_rows_in_order: list[tuple[str, str, str, str]] = []
    with args.input_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _, original_name, source_file, formatted_filename, time_range, events_field = split_record_fields(line)
            is_included_file = source_file and (
                matches_any_partial_name(original_name, included_files)
                or matches_any_partial_name(formatted_filename, included_files)
            )
            if is_included_file:
                included_output_files.add(source_file)
            line_count = count_events_in_line(line, args.threshold, args.metric)
            total_count += line_count
            if (args.output_kept_script or args.output_srt_files) and source_file:
                save_line = True
                if args.kept_file_event_db_thres is not None and not is_included_file:
                    save_line = (
                        count_events_in_line(
                            line, args.kept_file_event_db_thres, args.metric
                        )
                        > 0
                    )
                if save_line:
                    if args.output_kept_script:
                        script_lines_in_order.append((source_file, line.rstrip("\n")))
                    if args.output_srt_files and formatted_filename and time_range and events_field:
                        srt_rows_in_order.append(
                            (source_file, formatted_filename, time_range, events_field)
                        )

            if args.breakdown_by_file and line_count:
                if source_file:
                    counts_by_file[source_file] += line_count
            if args.show_lines and line_count:
                print(f"line {line_number}: {line_count}")

    kept_count = 0
    saved_script_line_count = 0
    saved_srt_file_count = 0
    kept_files: list[str] = []
    if args.breakdown_by_file:
        filtered_counts = [
            (source_file, count)
            for source_file, count in sorted(
                counts_by_file.items(), key=lambda item: (-item[1], item[0])
            )
            if count >= args.loud_event_freq_thres
        ]
        if args.num_loudest_files > 0:
            filtered_counts = filtered_counts[: args.num_loudest_files]

        for source_file in sorted(included_output_files):
            if source_file not in {path for path, _ in filtered_counts}:
                filtered_counts.append((source_file, counts_by_file.get(source_file, 0)))

        for source_file, count in filtered_counts:
            print(f"{count}\n{source_file}")
            kept_count += 1
            kept_files.append(source_file)

    if not args.breakdown_by_file:
        kept_files.extend(sorted(included_output_files))
        kept_count = len(kept_files)

    kept_file_set = set(kept_files)
    if args.output_kept_script:
        kept_script_path = resolve_output_path(args.output_kept_script, args.output_dir)
        kept_script_path.parent.mkdir(parents=True, exist_ok=True)
        with kept_script_path.open("w", encoding="utf-8") as handle:
            for source_file, script_line in script_lines_in_order:
                if source_file in kept_file_set:
                    handle.write(f"{script_line}\n")
                    saved_script_line_count += 1

    if args.output_srt_files:
        srt_rows_by_file: dict[str, tuple[str, list[tuple[str, str]]]] = {}
        for source_file, formatted_filename, time_range, events_field in srt_rows_in_order:
            if source_file in kept_file_set:
                if source_file not in srt_rows_by_file:
                    srt_rows_by_file[source_file] = (formatted_filename, [])
                srt_rows_by_file[source_file][1].append((time_range, events_field))

        for source_file, (formatted_filename, rows) in srt_rows_by_file.items():
            srt_path = resolve_output_path(
                Path(formatted_filename).with_suffix(".srt"), args.output_dir
            )
            srt_path.parent.mkdir(parents=True, exist_ok=True)
            playback_segments: list[tuple[int, int]] = []
            merged_subtitle_rows = merge_subtitle_rows(rows)
            with srt_path.open("w", encoding="utf-8") as handle:
                for index, (time_range, events_field) in enumerate(rows, start=1):
                    start_seconds, end_seconds = parse_time_range(time_range)
                    playback_segments.append(
                        playback_segment(start_seconds, end_seconds)
                    )
                for index, (subtitle_start, subtitle_end, subtitle_lines) in enumerate(
                    merged_subtitle_rows, start=1
                ):
                    handle.write(f"{index}\n")
                    handle.write(
                        f"{sec_to_srt_timestamp(subtitle_start)} --> {sec_to_srt_timestamp(subtitle_end)}\n"
                    )
                    handle.write("\n".join(subtitle_lines))
                    handle.write("\n\n")

            python_path = resolve_output_path(
                Path(formatted_filename).with_suffix(".py"), args.output_dir
            )
            python_path.write_text(
                build_vlc_python(
                    formatted_filename,
                    merge_playback_segments(playback_segments),
                ),
                encoding="utf-8",
            )

            batch_path = resolve_output_path(
                Path(formatted_filename).with_suffix(".bat"), args.output_dir
            )
            batch_path.write_text(
                build_python_launcher_batch(python_path.name),
                encoding="utf-8",
                newline="",
            )
            saved_srt_file_count += 1

    print(f"Total loud-event count: {total_count}")
    print(f"Total kept files: {kept_count}")
    if args.output_kept_script:
        print(f"Total saved script lines: {saved_script_line_count}")
    if args.output_srt_files:
        print(f"Total saved SRT files: {saved_srt_file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())