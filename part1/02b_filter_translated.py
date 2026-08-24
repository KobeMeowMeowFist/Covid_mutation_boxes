#!/usr/bin/env python3
"""Step 2b: Filter existing Nextclade translated FASTA files to match subsampled strains.

After re-running step 02 (which produces a new filtered_metadata.tsv with ~4000
strains), use this script to filter the EXISTING Nextclade translated_<CDS>.fasta
files to only include those strains. This avoids re-running Nextclade (which is
the most expensive step).

Memory-efficient: uses awk for FASTA filtering (line-by-line in C, ~200 KB for
strain ID hash map). Never loads full FASTA files into Python memory.

Usage:
    python 02b_filter_translated.py --indir data/ --outdir data/
    python 02b_filter_translated.py --indir data/ --outdir data/ --backup
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter Nextclade translated FASTA files to match subsampled strains"
    )
    parser.add_argument("--indir", type=Path, default=Path("data"),
                        help="Directory with filtered_metadata.tsv and translated_<CDS>.fasta")
    parser.add_argument("--outdir", type=Path, default=Path("data"),
                        help="Output directory for filtered files (default: same as indir)")
    parser.add_argument("--backup", action="store_true",
                        help="Backup original files to <CDS>.fasta.bak before overwriting")
    return parser.parse_args()


def load_strain_ids(metadata_path: Path) -> set[str]:
    """Read strain IDs from filtered_metadata.tsv."""
    ids: set[str] = set()
    with open(metadata_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("strain", "")
            if sid:
                ids.add(sid)
    return ids


def filter_fasta_awk(
    input_path: Path, strain_ids: set[str], output_path: Path
) -> tuple[int, int]:
    """Filter a FASTA file using a robust Python streaming implementation.

    Tries several header-normalization strategies to match strain IDs from
    metadata to FASTA headers (handles 'strain|CDS', dots, and last-token
    matches). Returns (n_before, n_after).
    """
    n_before = 0
    n_after = 0

    # Precompute simple normalization variants for faster matching
    base_ids = {s.split(".")[0] for s in strain_ids}
    last_token_ids = {s.split("/")[-1] for s in strain_ids}

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        write_current = False
        current_id = None
        for line in f_in:
            if line.startswith(">"):
                n_before += 1
                header = line[1:].strip()
                token = header.split()[0]
                token = token.split("|")[0]
                token_base = token.split(".")[0]
                token_last = token.split("/")[-1]

                # Try several matching strategies
                matched = (
                    token in strain_ids
                    or token_base in base_ids
                    or token_last in last_token_ids
                )

                write_current = matched
                if write_current:
                    f_out.write(line)
                    n_after += 1
            else:
                if write_current:
                    f_out.write(line)

    return n_before, n_after


def _filter_fasta_python(
    input_path: Path, strain_ids: set[str], output_path: Path
) -> int:
    """Python fallback: line-by-line filtering (never loads full file into memory)."""
    n_written = 0
    in_target = False
    current_id = None

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if line.startswith(">"):
                # Header may be 'strain|CDS' or 'strain CDS' -> take part before '|'
                current_id = line[1:].strip().split()[0].split("|")[0]
                in_target = current_id in strain_ids
                if in_target:
                    f_out.write(line)
                    n_written += 1
            else:
                if in_target:
                    f_out.write(line)

    return n_written


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Step 2b: Filter translated FASTA to subsampled strains ===")
    print("  (Memory-efficient: awk line-by-line, no full-file loading)")

    # --- Load strain IDs from filtered_metadata.tsv ---
    meta_path = indir / "filtered_metadata.tsv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run 02_filter_subsample.py first.")
        sys.exit(1)

    strain_ids = load_strain_ids(meta_path)
    print(f"  Target strains from metadata: {len(strain_ids):,}")

    if not strain_ids:
        print("ERROR: No strain IDs found in metadata.")
        sys.exit(1)

    # --- Filter each CDS file ---
    total_before = 0
    total_after = 0
    cds_found = 0

    for cds_name in config.NEXTCLADE_CDS:
        path = indir / f"translated_{cds_name}.fasta"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping")
            continue

        cds_found += 1

        # Backup if requested
        if args.backup and indir == outdir:
            bak_path = path.with_suffix(".fasta.bak")
            if not bak_path.exists():
                shutil.copy2(path, bak_path)
                print(f"  Backed up: {bak_path.name}")

        # Filter with awk
        out_path = outdir / f"translated_{cds_name}.fasta"
        n_before, n_after = filter_fasta_awk(path, strain_ids, out_path)

        kept_pct = 100.0 * n_after / n_before if n_before > 0 else 0
        print(f"  {cds_name:>8s}: {n_before:>6d} -> {n_after:>6d}  ({kept_pct:.1f}% kept)")

        total_before += n_before
        total_after += n_after

    if cds_found == 0:
        print("\nERROR: No translated_<CDS>.fasta files found.")
        print("Run 03_nextclade_translate.py first, or check --indir path.")
        sys.exit(1)

    print(f"\n  Total: {total_before:,} -> {total_after:,}")
    print(f"  Expected: ~{len(strain_ids):,} per CDS")

    if total_after == 0:
        print("\nERROR: No sequences remaining after filtering.")
        print("Check that strain IDs in filtered_metadata.tsv match the FASTA headers.")
        sys.exit(1)

    avg_per_cds = total_after // cds_found
    missing = len(strain_ids) - avg_per_cds
    if missing > 0:
        print(f"  NOTE: ~{missing} strains from metadata not found in translated files")
        print(f"  (these will be skipped during concatenation in step 04)")

    print(f"\n=== Filter complete ===")
    print(f"  Next step: python 04_concatenate_orfs.py --indir {outdir} --outdir {outdir}")


if __name__ == "__main__":
    main()
