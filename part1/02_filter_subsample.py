#!/usr/bin/env python3
"""Step 2: Filter Nextstrain data by date and subsample equitably per lineage.

Memory-efficient two-pass design:
  Pass 1: Count records per lineage in date range (memory: ~100 KB for counts)
  Pass 2: Reservoir sample per lineage (memory: ~800 KB for 4000 selected records)
  Sequence extraction: awk line-by-line (memory: ~200 KB for strain ID hash map)

Total peak memory: ~1 MB, regardless of input size.
Handles 7M+ metadata rows and 10 GB+ aligned.fasta without loading either
into memory.

Outputs:
  - filtered_sequences.fasta : nucleotide FASTA for Nextclade input
  - filtered_metadata.tsv    : strain, date, pango_lineage, country

Usage:
    python 02_filter_subsample.py --indir data/ --outdir data/
    python 02_filter_subsample.py --indir data/ --outdir data/ --per-lineage 300
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter Nextstrain data by date and subsample per lineage"
    )
    parser.add_argument("--indir", type=Path, default=Path("data"),
                        help="Input directory with metadata.tsv and aligned.fasta")
    parser.add_argument("--outdir", type=Path, default=Path("data"),
                        help="Output directory for filtered files")
    parser.add_argument("--per-lineage", type=int, default=config.PER_LINEAGE_TARGET,
                        help=f"Target sequences per lineage (default: {config.PER_LINEAGE_TARGET})")
    parser.add_argument("--min-lineage", type=int, default=config.MIN_PER_LINEAGE,
                        help=f"Minimum sequences to include a lineage (default: {config.MIN_PER_LINEAGE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible subsampling (default: 42)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_date(date: str) -> str:
    """Pad incomplete dates (YYYY -> YYYY-00-00, YYYY-MM -> YYYY-MM-00) for comparison."""
    if len(date) == 4:
        return date + "-00-00"
    elif len(date) == 7:
        return date + "-00"
    return date


def _find_columns(header: list[str]) -> dict[str, int]:
    """Find column indices for strain, date, lineage, country from header row."""
    cols: dict[str, int] = {}
    for i, name in enumerate(header):
        if name == "strain":
            cols["strain"] = i
        elif name == "date":
            cols["date"] = i
        elif name == "pango_lineage":
            cols["pango_lineage"] = i
        elif name == "Nextclade_pango":
            cols["nextclade_pango"] = i
        elif name == "country":
            cols["country"] = i
    return cols


def _get_lineage(row: list[str], cols: dict[str, int]) -> str:
    """Extract lineage from a row, trying pango_lineage then Nextclade_pango."""
    for key in ("pango_lineage", "nextclade_pango"):
        if key in cols:
            idx = cols[key]
            if idx < len(row):
                val = row[idx]
                if val and val != "None":
                    return val
    return "unknown"


# ---------------------------------------------------------------------------
# Pass 1: Count records per lineage (memory: O(n_lineages) ~ 100 KB)
# ---------------------------------------------------------------------------

def count_by_lineage(
    metadata_path: Path, date_min: str, date_max: str
) -> tuple[dict[str, int], int, int, dict[str, int]]:
    """Pass 1: Count records per lineage in date range.

    Uses csv.reader (list per row) instead of DictReader (dict per row)
    for speed and lower per-row overhead. Only lineage counts are retained.

    Returns: (counts, total_records, filtered_records, column_indices)
    """
    counts: dict[str, int] = defaultdict(int)
    total = 0
    filtered = 0
    cols: dict[str, int] = {}

    with open(metadata_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cols = _find_columns(header)

        if "date" not in cols:
            print("ERROR: 'date' column not found in metadata header.")
            print(f"  Available columns: {header}")
            sys.exit(1)

        date_idx = cols["date"]

        for row in reader:
            total += 1
            if total % 500000 == 0:
                print(f"  Pass 1: read {total:,} records, {filtered:,} in date range")

            if date_idx >= len(row):
                continue
            date = row[date_idx]
            if not date or len(date) < 4:
                continue

            date_padded = _pad_date(date)
            if not (date_min <= date_padded <= date_max):
                continue

            filtered += 1
            lineage = _get_lineage(row, cols)
            counts[lineage] += 1

    return dict(counts), total, filtered, cols


# ---------------------------------------------------------------------------
# Compute sampling targets
# ---------------------------------------------------------------------------

def compute_targets(
    counts: dict[str, int], per_lineage: int, min_lineage: int, total_cap: int
) -> dict[str, int]:
    """Compute how many to sample from each lineage.

    Dynamically reduces per_lineage if n_lineages * per_lineage > total_cap.
    """
    eligible = {lin: c for lin, c in counts.items() if c >= min_lineage}
    n_eligible = len(eligible)
    if n_eligible == 0:
        return {}

    effective_per_lineage = min(
        per_lineage, max(min_lineage, total_cap // n_eligible)
    )
    if effective_per_lineage < per_lineage:
        print(f"  NOTE: {n_eligible} eligible lineages found.")
        print(f"  Reducing per-lineage target from {per_lineage} to "
              f"{effective_per_lineage} to stay within total cap of {total_cap}.")

    targets = {lin: min(effective_per_lineage, c) for lin, c in eligible.items()}

    # Print distribution table
    print(f"\n  Lineage distribution:")
    print(f"  {'Lineage':<20s} {'Count':>8s} {'Target':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")

    for lineage in sorted(counts.keys()):
        n = counts[lineage]
        if n < min_lineage:
            print(f"  {lineage:<20s} {n:>8d} {'DROPPED':>8s}  (< {min_lineage})")
        else:
            t = targets[lineage]
            print(f"  {lineage:<20s} {n:>8d} {t:>8d}")

    total_target = sum(targets.values())
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<20s} {sum(counts.values()):>8d} {total_target:>8d}")

    return targets


# ---------------------------------------------------------------------------
# Pass 2: Reservoir sample (memory: O(sum(targets)) ~ 800 KB for 4000 records)
# ---------------------------------------------------------------------------

def reservoir_sample(
    metadata_path: Path,
    date_min: str,
    date_max: str,
    targets: dict[str, int],
    cols: dict[str, int],
    seed: int,
) -> list[dict]:
    """Pass 2: Reservoir sample per lineage using Algorithm R.

    For each lineage, maintains a reservoir of size k = targets[lineage].
    When the i-th record (1-indexed) for that lineage is encountered:
      - If reservoir not full, append it.
      - Otherwise, generate j = random(0, i-1). If j < k, replace reservoir[j].

    This gives each record an equal k/n probability of being selected,
    matching the behavior of random.sample but in a single streaming pass.

    Memory: only the reservoirs are kept (~4000 records x ~200 bytes = ~800 KB).
    """
    rng = random.Random(seed)

    reservoirs: dict[str, list[dict]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)

    date_idx = cols["date"]
    strain_idx = cols.get("strain", -1)
    country_idx = cols.get("country", -1)

    with open(metadata_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header

        n_processed = 0
        for row in reader:
            n_processed += 1
            if n_processed % 500000 == 0:
                n_selected = sum(len(r) for r in reservoirs.values())
                print(f"  Pass 2: processed {n_processed:,} records, "
                      f"reservoirs hold {n_selected:,}")

            if date_idx >= len(row):
                continue
            date = row[date_idx]
            if not date or len(date) < 4:
                continue

            date_padded = _pad_date(date)
            if not (date_min <= date_padded <= date_max):
                continue

            lineage = _get_lineage(row, cols)
            if lineage not in targets:
                continue

            k = targets[lineage]
            counters[lineage] += 1
            n = counters[lineage]

            record = {
                "strain": row[strain_idx] if strain_idx >= 0 and strain_idx < len(row) else "",
                "date": date,
                "pango_lineage": lineage,
                "country": row[country_idx] if country_idx >= 0 and country_idx < len(row) else "",
            }

            if len(reservoirs[lineage]) < k:
                reservoirs[lineage].append(record)
            else:
                j = rng.randint(0, n - 1)
                if j < k:
                    reservoirs[lineage][j] = record

    # Flatten in sorted lineage order for reproducibility
    selected: list[dict] = []
    for lineage in sorted(reservoirs.keys()):
        selected.extend(reservoirs[lineage])

    return selected


# ---------------------------------------------------------------------------
# Write metadata
# ---------------------------------------------------------------------------

def write_metadata(selected: list[dict], out_path: Path) -> None:
    """Write selected strain metadata to TSV."""
    fieldnames = ["strain", "date", "pango_lineage", "country"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        for record in selected:
            writer.writerow(record)
    print(f"  Written: {len(selected):,} records to {out_path.name}")


# ---------------------------------------------------------------------------
# Sequence extraction (awk — line-by-line in C, ~200 KB memory)
# ---------------------------------------------------------------------------

def extract_sequences_awk(
    aligned_path: Path, strain_ids: set[str], out_path: Path
) -> int:
    """Extract sequences using awk — memory-efficient for large FASTA files.

    awk reads the file line by line in C, keeping only the strain ID hash map
    (~4000 entries x 50 bytes = 200 KB) in memory. This avoids Python's
    memory overhead for multi-GB FASTA files.
    """
    # Write strain IDs to a temp file for awk to read
    tmp_path = out_path.parent / "_strain_ids.txt"
    with open(tmp_path, "w") as f:
        for sid in sorted(strain_ids):
            f.write(sid + "\n")

    print(f"  Extracting {len(strain_ids):,} sequences with awk...")
    print(f"  (Scanning {aligned_path.name} line by line — "
          f"this may take a while for large files)")

    awk_script = (
        f'BEGIN{{while((getline line < "{tmp_path}") > 0) ids[line]=1}} '
        f'/^>/{{id=substr($1,2); if(id in ids) p=1; else p=0}} '
        f'p'
    )

    with open(out_path, "w") as f_out:
        result = subprocess.run(
            ["awk", awk_script, str(aligned_path)],
            stdout=f_out,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        print(f"  WARNING: awk failed ({result.stderr}), falling back to Python")
        tmp_path.unlink(missing_ok=True)
        return _extract_sequences_python(aligned_path, strain_ids, out_path)

    # Count written sequences
    n_written = 0
    with open(out_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                n_written += 1

    tmp_path.unlink(missing_ok=True)
    return n_written


def _extract_sequences_python(
    aligned_path: Path, strain_ids: set[str], out_path: Path
) -> int:
    """Python fallback: line-by-line extraction with progress reporting.

    Reads one line at a time — never loads the full file into memory.
    Only the current sequence (~30 KB for a 29K nt genome) is buffered.
    """
    print(f"  Python fallback: scanning {aligned_path.name} line by line...")
    n_written = 0
    n_scanned = 0
    current_id = None
    current_seq: list[str] = []
    in_target = False

    with open(aligned_path, "r") as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            if line.startswith(">"):
                # Write previous target sequence
                if in_target and current_id is not None:
                    seq = "".join(current_seq)
                    f_out.write(f">{current_id}\n")
                    for i in range(0, len(seq), 80):
                        f_out.write(seq[i:i + 80] + "\n")
                    n_written += 1

                current_id = line[1:].strip().split()[0]
                current_seq = []
                in_target = current_id in strain_ids
                n_scanned += 1
                if n_scanned % 100000 == 0:
                    print(f"  ...scanned {n_scanned:,} sequences, wrote {n_written:,}")
            else:
                if in_target:
                    current_seq.append(line.strip())

        # Write last sequence
        if in_target and current_id is not None:
            seq = "".join(current_seq)
            f_out.write(f">{current_id}\n")
            for i in range(0, len(seq), 80):
                f_out.write(seq[i:i + 80] + "\n")
            n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    metadata_path = indir / "metadata.tsv"
    aligned_path = indir / "aligned.fasta"

    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run 01_download.py first.")
        sys.exit(1)
    if not aligned_path.exists():
        print(f"ERROR: {aligned_path} not found. Run 01_download.py first.")
        sys.exit(1)

    # === Pass 1: Count records per lineage ===
    print(f"Pass 1: Counting records per lineage")
    print(f"  Date filter: {config.DATE_MIN} to {config.DATE_MAX}")
    print(f"  Metadata: {metadata_path.name}")
    counts, total, filtered, cols = count_by_lineage(
        metadata_path, config.DATE_MIN, config.DATE_MAX
    )
    print(f"  Total records: {total:,}")
    print(f"  In date range: {filtered:,}")
    print(f"  Lineages: {len(counts)}")

    if not counts:
        print("ERROR: No strains found in date range.")
        sys.exit(1)

    # === Compute sampling targets ===
    print(f"\nComputing sampling targets "
          f"(per-lineage={args.per_lineage}, min={args.min_lineage}, "
          f"cap={config.TARGET_SAMPLE_SIZE})...")
    targets = compute_targets(
        counts, args.per_lineage, args.min_lineage, config.TARGET_SAMPLE_SIZE
    )

    if not targets:
        print("ERROR: No lineages meet the minimum threshold.")
        sys.exit(1)

    # === Pass 2: Reservoir sample ===
    print(f"\nPass 2: Reservoir sampling (seed={args.seed})...")
    selected = reservoir_sample(
        metadata_path, config.DATE_MIN, config.DATE_MAX,
        targets, cols, args.seed,
    )

    if not selected:
        print("ERROR: No strains selected after reservoir sampling.")
        sys.exit(1)

    # Safety: final random sample if still over cap
    if len(selected) > config.TARGET_SAMPLE_SIZE:
        print(f"  NOTE: Total {len(selected)} exceeds cap "
              f"{config.TARGET_SAMPLE_SIZE}, final random sample.")
        rng = random.Random(args.seed + 1)
        selected = rng.sample(selected, config.TARGET_SAMPLE_SIZE)

    print(f"  Total selected: {len(selected):,} strains")

    # === Write filtered metadata ===
    meta_out_path = outdir / "filtered_metadata.tsv"
    print(f"\nWriting filtered metadata to {meta_out_path.name}...")
    write_metadata(selected, meta_out_path)

    # === Extract sequences (awk for memory efficiency) ===
    strain_ids = {s["strain"] for s in selected if s["strain"]}
    if not strain_ids:
        print("ERROR: No valid strain IDs found in selected records.")
        sys.exit(1)

    seq_out_path = outdir / "filtered_sequences.fasta"
    print(f"\nExtracting sequences from {aligned_path.name}...")
    n_written = extract_sequences_awk(aligned_path, strain_ids, seq_out_path)

    n_missing = len(strain_ids) - n_written
    if n_missing > 0:
        print(f"  WARNING: {n_missing} strain IDs not found in aligned.fasta")

    # === Summary ===
    dates = [s["date"] for s in selected if s["date"]]
    print("\n=== Summary ===")
    print(f"  Date range: {config.DATE_MIN} to {config.DATE_MAX}")
    print(f"  Total metadata records: {total:,}")
    print(f"  In date range: {filtered:,}")
    print(f"  Lineages (in range): {len(counts)}")
    print(f"  Selected: {len(selected):,}")
    print(f"  Sequences written: {n_written:,}")
    if dates:
        print(f"  Date range of selected: {min(dates)} to {max(dates)}")
    print(f"\n=== Filter + subsample complete ===")
    print(f"  Peak memory: ~1 MB (two-pass reservoir sampling + awk)")
    print(f"  Next step: python 03_nextclade_translate.py --indir {outdir} --outdir {outdir}")
    print(f"  Or skip Nextclade: python 02b_filter_translated.py --indir {outdir} --outdir {outdir}")


if __name__ == "__main__":
    main()
