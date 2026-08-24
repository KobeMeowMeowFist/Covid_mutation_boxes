#!/usr/bin/env python3
"""Step 1: Download and decompress Nextstrain SARS-CoV-2 open data.

Downloads metadata.tsv and aligned.fasta from the Nextstrain open data
endpoint (GenBank-sourced, publicly available). Default mode is 'full'
(all GenBank strains). Use 'subsampled' for a quick ~4000-sequence test.

Usage:
    python 01_download.py --mode full --outdir data/
    python 01_download.py --mode subsampled --outdir data/
"""

from __future__ import annotations

import argparse
import lzma
import os
import sys
import urllib.request
from pathlib import Path

# tqdm for progress bar (optional, falls back to simple print)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Nextstrain SARS-CoV-2 open data"
    )
    parser.add_argument(
        "--mode",
        choices=("full", "subsampled"),
        default="full",
        help="full = all GenBank strains (default); subsampled = ~4000 global strains",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data"),
        help="Output directory for decompressed files",
    )
    return parser.parse_args()


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a progress bar."""
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")

    # Get file size for progress bar
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))

    if total_size > 0:
        size_mb = total_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print("  File size: unknown")

    with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
        if HAS_TQDM and total_size > 0:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="  Downloading",
            ) as pbar:
                while True:
                    chunk = response.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
        else:
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = 100 * downloaded / total_size
                    print(f"\r  Progress: {pct:.1f}% ({downloaded/(1024*1024):.1f} MB)", end="")
            print()

    print(f"  Done: {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)")


def decompress_zst(src: Path, dest: Path) -> None:
    """Decompress a .zst file using the zstandard library."""
    import zstandard as zstd

    print(f"  Decompressing: {src} -> {dest}")
    dctx = zstd.ZstdDecompressor()

    with open(src, "rb") as f_in:
        with dctx.stream_reader(f_in) as reader:
            with open(dest, "wb") as f_out:
                while True:
                    chunk = reader.read(1024 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)

    print(f"  Done: {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)")


def decompress_xz(src: Path, dest: Path) -> None:
    """Decompress an .xz file using lzma."""
    print(f"  Decompressing: {src} -> {dest}")

    with lzma.open(src, "rb") as f_in:
        with open(dest, "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

    print(f"  Done: {dest} ({dest.stat().st_size / (1024*1024):.1f} MB)")


def count_fasta_sequences(path: Path) -> int:
    """Count the number of sequences in a FASTA file."""
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count


def get_date_range_from_metadata(path: Path) -> tuple[str, str]:
    """Get the min and max dates from metadata.tsv."""
    import csv

    min_date = None
    max_date = None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        date_col = None
        for col in reader.fieldnames:
            if col.lower() == "date":
                date_col = col
                break
        if date_col is None:
            return ("unknown", "unknown")

        for row in reader:
            d = row.get(date_col, "")
            if d and len(d) >= 7:  # at least YYYY-MM
                if min_date is None or d < min_date:
                    min_date = d
                if max_date is None or d > max_date:
                    max_date = d

    return (min_date or "unknown", max_date or "unknown")


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Select URLs based on mode
    if args.mode == "full":
        urls = config.URLS_FULL
        compression = "zst"
    else:
        urls = config.URLS_SUBSAMPLED
        compression = "xz"

    print(f"=== Downloading Nextstrain SARS-CoV-2 data (mode: {args.mode}) ===")
    print(f"  Output directory: {outdir}")
    print()

    # Define file paths
    files = {
        "metadata": {
            "url": urls["metadata"],
            "compressed": outdir / f"metadata.tsv.{compression}",
            "decompressed": outdir / "metadata.tsv",
        },
        "aligned": {
            "url": urls["aligned"],
            "compressed": outdir / f"aligned.fasta.{compression}",
            "decompressed": outdir / "aligned.fasta",
        },
    }

    # Download and decompress each file
    for name, info in files.items():
        dest = info["decompressed"]

        # Skip if already decompressed
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{name}] Already exists: {dest} ({dest.stat().st_size/(1024*1024):.1f} MB)")
            continue

        # Download
        print(f"\n[{name}]")
        compressed_path = info["compressed"]

        if not compressed_path.exists():
            download_with_progress(info["url"], compressed_path)
        else:
            print(f"  Compressed file already exists: {compressed_path}")

        # Decompress
        if compression == "zst":
            decompress_zst(compressed_path, dest)
        else:
            decompress_xz(compressed_path, dest)

        # Remove compressed file to save disk space
        compressed_path.unlink()
        print(f"  Removed compressed file: {compressed_path}")

    # Report statistics
    print("\n=== Summary ===")
    for name, info in files.items():
        dest = info["decompressed"]
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  {name}: {dest.name} ({size_mb:.1f} MB)")

    # Count sequences and date range
    aligned_path = files["aligned"]["decompressed"]
    metadata_path = files["metadata"]["decompressed"]

    if aligned_path.exists():
        print(f"\n  Counting sequences in {aligned_path.name}...")
        n_seqs = count_fasta_sequences(aligned_path)
        print(f"  Sequences: {n_seqs:,}")

    if metadata_path.exists():
        print(f"\n  Scanning date range in {metadata_path.name}...")
        min_date, max_date = get_date_range_from_metadata(metadata_path)
        print(f"  Date range: {min_date} to {max_date}")

    print(f"\n  Filter date range: {config.DATE_MIN} to {config.DATE_MAX}")
    print("\n=== Download complete ===")
    print(f"  Next step: python 02_filter_subsample.py --indir {outdir} --outdir {outdir}")


if __name__ == "__main__":
    main()
