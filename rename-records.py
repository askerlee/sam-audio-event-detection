#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path
import datetime
import shutil

# Only match reolink recordings.
PATTERN1 = re.compile(r"^(?P<prefix>.+)-(?P<start>\d{6})-(?P<end>\d{6})\.(mp4|wav)$", re.IGNORECASE)
PATTERN2 = re.compile(r"(?P<prefix>.+)_(?P<date>\d{8})_(?P<start>\d{6})\.(mp4|wav)$", re.IGNORECASE)

def hhmmss_to_seconds(hhmmss: str) -> int:
    h = int(hhmmss[0:2])
    m = int(hhmmss[2:4])
    s = int(hhmmss[4:6])
    return h * 3600 + m * 60 + s

def seconds_to_hhmmss(sec: int) -> str:
    sec = sec % (24 * 3600)  # handle wrap past midnight
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}{m:02d}{s:02d}"

def ffprobe_duration_seconds(path: Path) -> float:
    # Returns duration in seconds (float)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def unique_target(path: Path) -> Path:
    """If target exists, append .dupN before extension."""
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    for i in range(1, 10000):
        cand = path.with_name(f"{stem}.dup{i}{suf}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique name for {path}")

def main():
    ap = argparse.ArgumentParser(description="Rename ...-START-END.(mp4|wav) using real duration.")
    ap.add_argument("dir", nargs="?", default=".", help="Directory to scan (default: .)")
    ap.add_argument("--apply", action="store_true", help="Actually rename (default: dry-run)")
    ap.add_argument("--copy_to", type=str, default=None, 
                    help="Copy to this directory instead of rename")
    ap.add_argument("--rounding_method", choices=["nearest", "floor", "ceil"], default="ceil",
                    help="How to convert duration to whole seconds (default: ceil)")
    ap.add_argument("--cutoff_date", type=str, default=None,
                    help="Only process files before or on this date (YYYYMMDD)")
    ap.add_argument("--min-change", type=int, default=0,
                    help="Only rename if end changes by at least N seconds (default: 0)")
    ap.add_argument("--cdrt_format", action="store_true",
                    help="Whether to use CDRT format for renaming.")
    args = ap.parse_args()

    import math
    round_fn = {
        "nearest": lambda x: int(round(x)),
        "floor":   lambda x: int(math.floor(x)),
        "ceil":    lambda x: int(math.ceil(x)),
    }[args.rounding_method]

    root = Path(args.dir)
    files = sorted(root.rglob("*.mp4")) + sorted(root.rglob("*.wav"))

    keep_file_count = 0
    rename_file_count = 0

    if args.copy_to:
        copy_to_path = Path(args.copy_to)
        copy_to_path.mkdir(parents=True, exist_ok=True)

    for p in files:
        m1 = PATTERN1.match(p.name)
        m2 = PATTERN2.match(p.name)
        if not m1 and not m2:
            print(f"SKIP  {p.name}: does not match expected patterns")
            continue

        if m1:
            # hubCH01-01-120000-123000.mp4
            # prefix: hubCH01-01
            prefix = m1.group("prefix")
            # .mp4
            suffix = p.suffix
            # start-time
            start = m1.group("start")
            # end-time
            old_end = m1.group("end")
            # Parent subdir is date MMDDYYYY
            rec_date = str(p.parent.name)
            date_obj = datetime.datetime.strptime(rec_date, "%m%d%Y")
        else:
            # IphoneVideo_20251009_210642.mp4
            # or
            # TimeVideo_20251010_185751.mp4
            # prefix: IphoneVideo or TimeVideo
            prefix = m2.group("prefix")
            start = m2.group("start")
            # Infer end time as start + duration later
            # For now, just set end to start (will be updated)
            old_end = start  
            suffix = p.suffix
            # From PATTERN2, we have date in the filename.
            # YYYYMMDD
            rec_date = m2.group("date")
            date_obj = datetime.datetime.strptime(rec_date, "%Y%m%d")

        if args.cutoff_date:
            cutoff_date = datetime.datetime.strptime(args.cutoff_date, "%m%d%Y")
            if date_obj > cutoff_date:
                print(f"SKIP  {p.name}: date {date_obj.strftime('%Y-%m-%d')} > cutoff {cutoff_date.strftime('%Y-%m-%d')}")
                continue
                
        if not args.cdrt_format:
            try:
                dur = ffprobe_duration_seconds(p)
            except Exception as e:
                print(f"SKIP  {p.name}: ffprobe failed ({e})")
                continue
            
            dur_s = round_fn(dur)
            start_s = hhmmss_to_seconds(start)
            new_end_s = start_s + dur_s
            new_end = seconds_to_hhmmss(new_end_s)

            old_end_s = hhmmss_to_seconds(old_end)
            diff = abs((new_end_s % (24*3600)) - old_end_s)
            # diff across midnight is tricky; this is a simple check:
            diff = min(diff, 24*3600 - diff)

            if diff < args.min_change:
                print(f"KEEP  {p.name}  (dur={dur:.3f}s -> {dur_s}s, end stays ~same)")
                continue
            new_name = f"{prefix}-{start}-{new_end}{suffix}"

        else:
            # In CDRT format, we don't care about duration; just rename based on start time.
            dur = 0.0
            dur_s = 0

            # 20160605 -> 05 June 2016
            date_str = date_obj.strftime("%d %B %Y")
            start_time_str = f"{start[0:2]}-{start[2:4]}-{start[4:6]}"
            if m1:
                new_name = f"＜{date_str}＞-＜{start_time_str}＞-＜kitchen noises＞{suffix}"
            else:
                new_name = f"＜{date_str}＞-＜{start_time_str}＞-＜kitchen noises (sound level meter)＞{suffix}"

        target = p.with_name(new_name)
        if target == p:
            print(f"KEEP  {p.name}  (dur={dur:.3f}s -> {dur_s}s, name unchanged)")
            keep_file_count += 1
            continue

        if args.apply:
            if args.copy_to:
                target = Path(args.copy_to) / new_name
                if target.exists() and target.stat().st_size == p.stat().st_size:
                    if args.cdrt_format:
                        print(f"SKIP COPY {p.resolve()} -> {target.resolve()} (already exists with same size)")
                    else:
                        print(f"SKIP COPY {p.resolve()} -> {target.name} (already exists with same size)")

                    keep_file_count += 1
                    continue

                rename_file_count += 1
                shutil.copy2(p, target)
                if args.cdrt_format:
                    print(f"COPIED {p.resolve()} -> {target.resolve()}")
                else:
                    print(f"COPIED {p.resolve()} -> {target.name}  (dur={dur:.3f}s -> {dur_s}s)")
            else:
                rename_file_count += 1
                os.rename(p, target)
                if args.cdrt_format:
                    print(f"RENAMED {p.resolve()} -> {target.resolve()}")
                else:
                    print(f"RENAMED {p.resolve()} -> {target.name}  (dur={dur:.3f}s -> {dur_s}s)")
        else:
            rename_file_count += 1
            if args.cdrt_format:
                target = Path(args.copy_to) / new_name
                shutil.copy2(p, target)
                print(f"DRYRUN {p.resolve()} -> {target.resolve()}")
            else:
                print(f"DRYRUN {p.resolve()} -> {target.name}  (dur={dur:.3f}s -> {dur_s}s)")

    print(f"\nSummary: {keep_file_count} files kept, {rename_file_count} files renamed/copied.")

if __name__ == "__main__":
    main()
