#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path

PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<start>\d{6})-(?P<end>\d{6})( \(.+\))?\.(mp4|wav)$", re.IGNORECASE)

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
    ap.add_argument("--round", choices=["nearest", "floor", "ceil"], default="ceil",
                    help="How to convert duration to whole seconds (default: ceil)")
    ap.add_argument("--min-change", type=int, default=0,
                    help="Only rename if end changes by at least N seconds (default: 0)")
    args = ap.parse_args()

    import math
    round_fn = {
        "nearest": lambda x: int(round(x)),
        "floor":   lambda x: int(math.floor(x)),
        "ceil":    lambda x: int(math.ceil(x)),
    }[args.round]

    root = Path(args.dir)
    files = sorted(root.rglob("*.mp4")) + sorted(root.rglob("*.wav"))

    keep_file_count = 0
    rename_file_count = 0

    for p in files:
        m = PATTERN.match(p.name)
        if not m:
            if p.name.endswith(('.mp4', '.wav')):
                print(f"SKIP  {p.name}: filename does not match pattern")
            continue

        prefix = m.group("prefix")
        suffix = p.suffix
        start = m.group("start")
        old_end = m.group("end")

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
        target = p.with_name(new_name)
        if target == p:
            print(f"KEEP  {p.name}  (dur={dur:.3f}s -> {dur_s}s, name unchanged)")
            keep_file_count += 1
            continue

        rename_file_count += 1

        if args.apply:
            os.rename(p, target)
            print(f"RENAMED {p.name} -> {target.name}  (dur={dur:.3f}s -> {dur_s}s)")
        else:
            print(f"DRYRUN {p.name} -> {target.name}  (dur={dur:.3f}s -> {dur_s}s)")

    print(f"\nSummary: {keep_file_count} files kept, {rename_file_count} files renamed.")

if __name__ == "__main__":
    main()
