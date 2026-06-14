import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path


SUMATRA_PDF_PATH = Path(r"C:\Users\asker\Downloads\SumatraPDF-3.5.2-64\SumatraPDF-3.5.2-64.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count pages in PDF files under a folder, including nested subfolders."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder to scan for PDF files.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="NAME_OR_PATH",
        help=(
            "Folder name or relative path to exclude from the scan. "
            "Repeat this option to exclude multiple folders."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the total number of PDFs and pages.",
    )
    parser.add_argument(
        "--print",
        dest="print_files",
        action="store_true",
        help="Print each discovered PDF using SumatraPDF and the default printer.",
    )
    parser.add_argument(
        "--num-printed-files",
        type=int,
        default=-1,
        help="Maximum number of files to print; -1 prints all files.",
    )
    parser.add_argument(
        "--print-copies",
        type=int,
        default=1,
        help="Number of copies to print for each file.",
    )
    return parser.parse_args()


def normalize_excluded_dirs(excluded_dirs: list[str]) -> tuple[set[str], set[Path]]:
    excluded_names = set()
    excluded_paths = set()

    for value in excluded_dirs:
        cleaned = value.strip().strip("/")
        if not cleaned:
            continue

        excluded_names.add(Path(cleaned).name)
        excluded_paths.add(Path(cleaned))

    return excluded_names, excluded_paths


def is_excluded(path: Path, root: Path, excluded_names: set[str], excluded_paths: set[Path]) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return False

    for candidate in [relative_path, *relative_path.parents]:
        if candidate == Path("."):
            continue
        if candidate in excluded_paths:
            return True

    return any(part in excluded_names for part in relative_path.parts[:-1])


def find_pdf_files(root: Path, excluded_names: set[str], excluded_paths: set[Path]) -> list[Path]:
    pdf_files = []
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not is_excluded(current_path / dirname, root, excluded_names, excluded_paths)
        ]

        for filename in filenames:
            pdf_path = current_path / filename
            if pdf_path.suffix.lower() != ".pdf":
                continue
            if is_excluded(pdf_path, root, excluded_names, excluded_paths):
                continue
            pdf_files.append(pdf_path)

    return sorted(pdf_files)


def count_pages(pdf_path: Path) -> int:
    if shutil.which("pdfinfo"):
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
        raise RuntimeError(f"Could not find page count in pdfinfo output for: {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "This script requires either the 'pdfinfo' command or the 'pypdf' package. "
            "Install poppler-utils or run 'pip install pypdf'."
        ) from error

    with pdf_path.open("rb") as handle:
        return len(PdfReader(handle).pages)


def print_pdf(pdf_path: Path, print_copies: int) -> None:
    if platform.system() != "Windows":
        raise RuntimeError("The --print option is only supported on Windows.")
    if not SUMATRA_PDF_PATH.exists():
        raise RuntimeError(f"SumatraPDF not found at: {SUMATRA_PDF_PATH}")

    print(f"Printing: {pdf_path}")
    subprocess.run(
        [
            str(SUMATRA_PDF_PATH),
            "-print-to-default",
            "-print-settings",
            f"{print_copies}x,paper=A4,fit",
            "-silent",
            str(pdf_path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    root = args.folder.expanduser().resolve()

    if args.num_printed_files == 0 or args.num_printed_files < -1:
        raise ValueError("--num-printed-files must be -1 or a positive integer.")
    if args.print_copies < 1:
        raise ValueError("--print-copies must be a positive integer.")

    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    excluded_names, excluded_paths = normalize_excluded_dirs(args.exclude_dir)
    pdf_files = find_pdf_files(root, excluded_names, excluded_paths)

    total_pages = 0
    printed_files = 0
    for pdf_file in pdf_files:
        page_count = count_pages(pdf_file)
        total_pages += page_count
        if args.print_files and (
            args.num_printed_files == -1 or printed_files < args.num_printed_files
        ):
            print_pdf(pdf_file, args.print_copies)
            printed_files += 1
        if not args.summary_only:
            print(f"{page_count}\t{pdf_file.relative_to(root)}")

    print(f"PDF files: {len(pdf_files)}")
    print(f"Total pages: {total_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())