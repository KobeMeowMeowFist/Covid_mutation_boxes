#!/usr/bin/env python3
"""Step 3: Translate nucleotide sequences to amino acid using Nextclade CLI.

Nextclade is the authoritative translation tool maintained by the Nextstrain
team. It correctly handles the -1 ribosomal frameshift, producing separate
ORF1a and ORF1b protein sequences.

This module:
  1. Checks that the nextclade CLI is installed
  2. Downloads the Nextclade dataset (nextstrain/sars-cov-2/wuhan-hu-1/orfs)
  3. Runs nextclade run on the filtered nucleotide sequences
  4. Parses the translated output and splits by CDS into per-CDS FASTA files

Outputs (in --outdir):
  - nextclade_dataset/         : downloaded Nextclade dataset
  - nextclade_results/         : raw Nextclade output
  - translated_<CDS>.fasta     : per-CDS protein FASTA (one per ORF)

Usage:
    python 03_nextclade_translate.py --indir data/ --outdir data/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate nucleotide sequences using Nextclade CLI"
    )
    parser.add_argument("--indir", type=Path, default=Path("data"),
                        help="Input directory with filtered_sequences.fasta")
    parser.add_argument("--outdir", type=Path, default=Path("data"),
                        help="Output directory for translated sequences")
    parser.add_argument("--dataset-name", type=str,
                        default=config.NEXTCLADE_DATASET,
                        help=f"Nextclade dataset name (default: {config.NEXTCLADE_DATASET})")
    return parser.parse_args()


def find_nextclade() -> str:
    """Find the nextclade CLI binary."""
    # Check common locations
    candidates = [
        "nextclade",
        "nextclade2",
        os.path.expanduser("~/.local/bin/nextclade"),
        "/usr/local/bin/nextclade",
        "/workspace/bin/nextclade",
    ]
    for c in candidates:
        path = shutil.which(c)
        if path:
            return path
    return ""


def check_nextclade() -> str:
    """Check that nextclade CLI is installed and return its path."""
    path = find_nextclade()
    if not path:
        print("ERROR: nextclade CLI not found.")
        print("\nTo install nextclade CLI:")
        print("  Option 1 (npm):  npm install -g @nextclade/nextclade")
        print("  Option 2 (binary): Download from https://github.com/nextstrain/nextclade/releases")
        print("  Option 3 (conda): conda install -c bioconda nextclade")
        sys.exit(1)

    # Check version
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True)
        version = result.stdout.strip() or result.stderr.strip()
        print(f"  Nextclade CLI found: {path}")
        print(f"  Version: {version}")
    except Exception as e:
        print(f"  WARNING: Could not get nextclade version: {e}")

    return path


def download_dataset(nextclade_path: str, dataset_name: str,
                     dataset_dir: Path) -> bool:
    """Download the Nextclade dataset."""
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"  Dataset already exists: {dataset_dir}")
        return True

    print(f"  Downloading dataset: {dataset_name}")
    print(f"  Output dir: {dataset_dir}")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        nextclade_path, "dataset", "get",
        "--name", dataset_name,
        "--output-dir", str(dataset_dir),
    ]
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Dataset download failed (exit code {result.returncode})")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

    print(f"  Dataset downloaded successfully")
    return True


def run_nextclade(nextclade_path: str, dataset_dir: Path,
                  input_fasta: Path, output_dir: Path) -> bool:
    """Run nextclade run to translate sequences.

    Nextclade outputs:
      - translated.fasta : all translated CDS sequences in one FASTA file
      - results.json     : analysis results (mutations, clades, etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    translated_path = output_dir / "translated_all.fasta"
    results_path = output_dir / "nextclade_results.json"

    # Use --output-all to let Nextclade produce the full set of outputs
    # into the output directory; later combine per-CDS translation files
    # into a single translated_all.fasta for backward-compatible parsing.
    cmd = [
        nextclade_path, "run",
        "--input-dataset", str(dataset_dir),
        "--output-all", str(output_dir),
        str(input_fasta),
    ]

    print(f"\n  Running Nextclade translation...")
    print(f"  Input: {input_fasta.name}")
    print(f"  Output: {translated_path.name}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Nextclade run failed (exit code {result.returncode})")
        print(f"  stdout: {result.stdout[-2000:]}")
        print(f"  stderr: {result.stderr[-2000:]}")
        return False

    # Print any warnings from stderr
    if result.stderr:
        # Only show last few lines (progress bar noise)
        stderr_lines = result.stderr.strip().split("\n")
        if len(stderr_lines) > 5:
            print(f"  (Nextclade stderr: {len(stderr_lines)} lines, showing last 3)")
            for line in stderr_lines[-3:]:
                print(f"    {line}")
        else:
            for line in stderr_lines:
                print(f"    {line}")

    print(f"  Nextclade run completed successfully")
    return True


def combine_translations(output_dir: Path, combined_path: Path) -> bool:
    """Find per-CDS translation FASTA files produced by Nextclade and
    combine them into a single FASTA where headers are 'strain|CDS'.
    """
    # Search for files that look like translations
    fasta_files = list(output_dir.rglob("*.fasta"))
    translation_files = []
    for p in fasta_files:
        name = p.name.lower()
        if "translation" in name or "cds" in name or "cds_translation" in name:
            translation_files.append(p)

    if not translation_files:
        # No per-CDS files found; maybe Nextclade produced a single file
        single = output_dir / "translated_all.fasta"
        if single.exists():
            shutil.copy(single, combined_path)
            return True
        print("  ERROR: No translation FASTA files found in Nextclade output")
        return False

    # Combine: each file typically contains >strain entries; annotate with CDS
    with open(combined_path, "w") as out:
        for p in sorted(translation_files):
            # Derive CDS name from filename if possible
            parts = p.stem.split('.')
            cds = parts[-1] if parts else p.stem
            # sanitize cds
            cds = cds.replace('cds_translation', '').replace('translation', '').strip('._-')
            if not cds:
                cds = p.stem

            with open(p, "r") as inp:
                for line in inp:
                    if line.startswith(">"):
                        header = line[1:].strip().split()[0]
                        out.write(f">{header}|{cds}\n")
                    else:
                        out.write(line)

    return True


def parse_translated_fasta(translated_path: Path, outdir: Path) -> dict[str, int]:
    """Parse the Nextclade translated FASTA and split by CDS.

    Nextclade translated FASTA format:
      >strain_name|gene_name
      MSEQ...

    Or sometimes:
      >strain_name gene_name
      MSEQ...

    We split by gene/CDS name and write per-CDS FASTA files.
    """
    print(f"\n  Parsing translated sequences from {translated_path.name}...")

    # Read all entries
    entries = []  # list of (strain_id, cds_name, sequence)
    current_header = None
    current_seq = []

    with open(translated_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    entries.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_header is not None:
        entries.append((current_header, "".join(current_seq)))

    print(f"  Total translated entries: {len(entries):,}")

    # Parse headers to extract strain_id and cds_name
    # Format: "strain_id|cds_name" or "strain_id cds_name"
    cds_sequences = defaultdict(dict)  # cds_name -> {strain_id: sequence}
    parse_errors = 0

    for header, seq in entries:
        # Try pipe separator first, then space
        if "|" in header:
            parts = header.split("|")
            strain_id = parts[0]
            cds_name = parts[-1]
        elif " " in header:
            parts = header.split()
            strain_id = parts[0]
            cds_name = parts[-1]
        else:
            # No separator — might be just strain_id with CDS in a different format
            parse_errors += 1
            continue

        cds_sequences[cds_name][strain_id] = seq

    if parse_errors > 0:
        print(f"  WARNING: {parse_errors} entries could not be parsed (no separator found)")

    # Write per-CDS FASTA files
    cds_counts = {}
    for cds_name in config.NEXTCLADE_CDS:
        if cds_name not in cds_sequences:
            print(f"  WARNING: CDS '{cds_name}' not found in translated output")
            cds_counts[cds_name] = 0
            continue

        strains = cds_sequences[cds_name]
        out_path = outdir / f"translated_{cds_name}.fasta"
        with open(out_path, "w") as f:
            for strain_id, seq in strains.items():
                f.write(f">{strain_id}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")

        cds_counts[cds_name] = len(strains)
        print(f"  {cds_name:>8s}: {len(strains):>6d} sequences -> {out_path.name}")

    # Report any unexpected CDS names
    expected = set(config.NEXTCLADE_CDS)
    unexpected = set(cds_sequences.keys()) - expected
    if unexpected:
        print(f"  NOTE: Unexpected CDS names found: {sorted(unexpected)}")

    return cds_counts


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    input_fasta = indir / "filtered_sequences.fasta"
    if not input_fasta.exists():
        print(f"ERROR: {input_fasta} not found. Run 02_filter_subsample.py first.")
        sys.exit(1)

    # --- Check nextclade CLI ---
    print("=== Step 3: Nextclade translation ===")
    print("\nChecking for Nextclade CLI...")
    nextclade_path = check_nextclade()

    # --- Download dataset ---
    dataset_dir = outdir / "nextclade_dataset"
    print(f"\nDownloading Nextclade dataset...")
    if not download_dataset(nextclade_path, args.dataset_name, dataset_dir):
        sys.exit(1)

    # --- Run Nextclade ---
    output_dir = outdir / "nextclade_results"
    if not run_nextclade(nextclade_path, dataset_dir, input_fasta, output_dir):
        sys.exit(1)

    # --- Combine per-CDS translation files into a single file for parsing ---
    translated_path = outdir / "translated_all.fasta"
    print(f"\nCombining Nextclade translations into {translated_path.name}...")
    if not combine_translations(output_dir, translated_path):
        print(f"ERROR: Could not assemble {translated_path}. Nextclade output missing or unexpected.")
        sys.exit(1)

    print(f"\nSplitting translated sequences by CDS...")
    cds_counts = parse_translated_fasta(translated_path, outdir)

    # --- Summary ---
    print("\n=== Summary ===")
    total_cds = sum(1 for c in cds_counts.values() if c > 0)
    print(f"  CDS found: {total_cds} / {len(config.NEXTCLADE_CDS)}")
    for cds_name in config.NEXTCLADE_CDS:
        count = cds_counts.get(cds_name, 0)
        status = "OK" if count > 0 else "MISSING"
        print(f"    {cds_name:>8s}: {count:>6d} sequences  [{status}]")

    print(f"\n=== Nextclade translation complete ===")
    print(f"  Next step: python 04_concatenate_orfs.py --indir {outdir} --outdir {outdir}")


if __name__ == "__main__":
    main()
