#!/usr/bin/env python3
"""Step 4 v2: Concatenate 12 ORFs with reference alignment.

Changes from v1:
  - Translates reference.fasta with Nextclade to get Wuhan-Hu-1 reference
  - Aligns each strain's ORFs to the reference, inserting gaps (-) for deletions
  - All sequences output at reference length (9814 aa), including original Wuhan
  - No length filtering — all strains kept (with gaps where needed)
  - wuhan_aa.fasta contains the ORIGINAL Wuhan-Hu-1 (MN908947.3, no gaps)

Usage:
    python 04_concatenate_orfs_v2.py --indir data/ --outdir data/ \
        --nextclade /path/to/nextclade --nextclade-dataset data/nextclade_dataset
"""

from __future__ import annotations

import argparse
import csv
import difflib
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, Counter

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate 12 ORFs per strain, aligned to Wuhan-Hu-1 reference"
    )
    parser.add_argument("--indir", type=Path, default=Path("data"),
                        help="Input directory with translated_<CDS>.fasta files")
    parser.add_argument("--outdir", type=Path, default=Path("data"),
                        help="Output directory for concatenated sequences")
    parser.add_argument("--nextclade", type=str, default="nextclade",
                        help="Path to nextclade CLI binary")
    parser.add_argument("--nextclade-dataset", type=Path, default=None,
                        help="Path to Nextclade dataset directory (for reference translation)")
    parser.add_argument("--reference-fasta", type=Path, default=None,
                        help="Path to reference.fasta (nucleotide). If not given, "
                             "uses <nextclade-dataset>/reference.fasta")
    parser.add_argument("--skip-nextclade", action="store_true",
                        help="Skip Nextclade run on reference (use existing ref translations)")
    parser.add_argument("--ref-translate-dir", type=Path, default=None,
                        help="Directory with existing reference translations "
                             "(nextclade.cds_translation.<CDS>.fasta). Used with --skip-nextclade.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def load_fasta(path: Path) -> dict[str, str]:
    """Load sequences from a FASTA file into a dict {id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


def write_fasta(path: Path, sequences: dict[str, str], line_width: int = 80) -> None:
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for sid, seq in sequences.items():
            f.write(f">{sid}\n")
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + "\n")


def strip_stop_codon(seq: str) -> str:
    """Remove stop codon characters ('*') from a translated CDS sequence."""
    return seq.replace("*", "")


# ---------------------------------------------------------------------------
# Nextclade reference translation
# ---------------------------------------------------------------------------

def run_nextclade_on_reference(
    nextclade_bin: str,
    dataset_dir: Path,
    reference_fasta: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Run Nextclade on the reference genome to get per-CDS translations.

    Returns: {cds_name: reference_aa_sequence}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        nextclade_bin, "run",
        "--input-dataset", str(dataset_dir),
        "--output-all", str(output_dir),
        "--output-selection", "translations",
        "--include-reference",  # include reference peptides in output
        str(reference_fasta),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Nextclade stderr: {result.stderr[:1000]}")
        # Check if output files exist anyway
    else:
        print(f"  Nextclade completed successfully")

    # List all files in output_dir for debugging
    all_files = sorted(output_dir.glob("**/*"))
    if all_files:
        print(f"  Files in output dir:")
        for f in all_files:
            print(f"    {f.relative_to(output_dir)}")
    else:
        print(f"  WARNING: No files found in {output_dir}")

    # Build a map of CDS name -> file path by searching recursively
    # Nextclade 3.x outputs: nextclade.cds_translation.<CDS>.fasta
    cds_file_map = {}
    for f in output_dir.glob("**/nextclade.cds_translation.*.fasta"):
        # Extract CDS name: nextclade.cds_translation.S.fasta -> S
        cds = f.stem.replace("nextclade.cds_translation.", "")
        cds_file_map[cds] = f
    # Also try alternative naming: translated_<CDS>.fasta
    for f in output_dir.glob("**/translated_*.fasta"):
        cds = f.stem.replace("translated_", "")
        if cds not in cds_file_map:
            cds_file_map[cds] = f

    # Load reference translations
    ref_translations = {}
    for cds_name in config.NEXTCLADE_CDS:
        path = cds_file_map.get(cds_name)
        if path is None:
            print(f"  WARNING: Reference translation for {cds_name} not found")
            continue
        seqs = load_fasta(path)
        if seqs:
            # Take the first (and only) sequence
            ref_translations[cds_name] = strip_stop_codon(list(seqs.values())[0])
            print(f"    {cds_name}: {len(ref_translations[cds_name])} aa (from {path.name})")

    return ref_translations


def load_existing_ref_translations(ref_dir: Path) -> dict[str, str]:
    """Load existing reference translations from a directory."""
    # Build CDS name -> file path map (recursive search)
    cds_file_map = {}
    for f in ref_dir.glob("**/nextclade.cds_translation.*.fasta"):
        cds = f.stem.replace("nextclade.cds_translation.", "")
        cds_file_map[cds] = f
    for f in ref_dir.glob("**/translated_*.fasta"):
        cds = f.stem.replace("translated_", "")
        if cds not in cds_file_map:
            cds_file_map[cds] = f

    ref_translations = {}
    for cds_name in config.NEXTCLADE_CDS:
        path = cds_file_map.get(cds_name)
        if path is None:
            print(f"  WARNING: Reference translation for {cds_name} not found in {ref_dir}")
            continue
        seqs = load_fasta(path)
        if seqs:
            ref_translations[cds_name] = strip_stop_codon(list(seqs.values())[0])
            print(f"    {cds_name}: {len(ref_translations[cds_name])} aa (from {path.name})")
    return ref_translations


# ---------------------------------------------------------------------------
# Alignment: insert gaps into strain sequences to match reference length
# ---------------------------------------------------------------------------

def align_to_reference(ref_seq: str, strain_seq: str) -> str:
    """Align strain_seq to ref_seq, inserting gaps (-) where the strain has deletions.

    Uses difflib.SequenceMatcher for fast alignment of similar sequences.
    Returns a gapped strain sequence with the same length as ref_seq.

    If strain_seq is longer than ref_seq (insertion in strain), the extra
    amino acids are skipped to maintain reference length.
    """
    if len(strain_seq) == len(ref_seq):
        return strain_seq

    sm = difflib.SequenceMatcher(None, ref_seq, strain_seq, autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            # Copy from strain sequence
            result.append(strain_seq[j1:j2])
        elif tag == "delete":
            # Reference has amino acids that strain doesn't → insert gaps
            result.append("-" * (i2 - i1))
        elif tag == "insert":
            # Strain has extra amino acids not in reference → skip them
            pass

    gapped = "".join(result)

    # Verify length
    if len(gapped) != len(ref_seq):
        # Fallback: if alignment didn't produce correct length,
        # pad with gaps at the end or truncate
        if len(gapped) < len(ref_seq):
            gapped = gapped + "-" * (len(ref_seq) - len(gapped))
        else:
            gapped = gapped[:len(ref_seq)]

    return gapped


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_all_cds(indir: Path) -> dict[str, dict[str, str]]:
    """Load all per-CDS FASTA files.
    Returns: {cds_name: {strain_id: sequence}}
    """
    all_cds = {}
    for cds_name in config.NEXTCLADE_CDS:
        path = indir / f"translated_{cds_name}.fasta"
        if not path.exists():
            print(f"  WARNING: {path.name} not found — CDS '{cds_name}' will be missing")
            all_cds[cds_name] = {}
            continue
        seqs = load_fasta(path)
        # Strip stop codons from all sequences
        all_cds[cds_name] = {sid: strip_stop_codon(seq) for sid, seq in seqs.items()}
        print(f"  {cds_name:>8s}: {len(seqs):>6d} sequences loaded")
    return all_cds


def get_all_strain_ids(all_cds: dict[str, dict[str, str]]) -> set[str]:
    """Get the set of strain IDs that appear in ALL CDS files."""
    if not all_cds:
        return set()
    first_cds = next(iter(all_cds.values()))
    common = set(first_cds.keys())
    for seqs in all_cds.values():
        common &= set(seqs.keys())
    return common


def load_filtered_metadata(path: Path) -> dict[str, dict]:
    """Load filtered_metadata.tsv into a dict {strain_id: metadata_dict}."""
    metadata = {}
    if not path.exists():
        return metadata
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("strain", "")
            if sid:
                metadata[sid] = row
    return metadata


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Step 4 v2: Concatenate 12 ORFs (reference-aligned) ===")

    # --- Step 1: Get reference translations ---
    print("\n--- Getting Wuhan-Hu-1 reference translations ---")

    if args.skip_nextclade and args.ref_translate_dir:
        print(f"  Loading existing reference translations from {args.ref_translate_dir}")
        ref_translations = load_existing_ref_translations(args.ref_translate_dir)
    else:
        # Determine reference.fasta path
        ref_fasta = args.reference_fasta
        if ref_fasta is None:
            dataset_dir = args.nextclade_dataset or (indir / "nextclade_dataset")
            ref_fasta = dataset_dir / "reference.fasta"

        if not ref_fasta.exists():
            print(f"ERROR: reference.fasta not found at {ref_fasta}")
            sys.exit(1)

        dataset_dir = args.nextclade_dataset or (indir / "nextclade_dataset")
        ref_output_dir = outdir / "_ref_translations"

        print(f"  Reference genome: {ref_fasta}")
        print(f"  Nextclade dataset: {dataset_dir}")
        ref_translations = run_nextclade_on_reference(
            args.nextclade, dataset_dir, ref_fasta, ref_output_dir
        )

    if not ref_translations:
        print("ERROR: No reference translations obtained.")
        sys.exit(1)

    # Compute reference concatenated length
    ref_concat = "".join(ref_translations.get(name, "") for name in config.CONCATENATION_ORDER)
    ref_total_len = len(ref_concat)
    print(f"\n  Reference concatenated length: {ref_total_len} aa")
    print(f"  Per-ORF reference lengths:")
    ref_orf_lengths = {}
    ref_orf_offsets = {}
    _off = 0
    for name in config.CONCATENATION_ORDER:
        rlen = len(ref_translations.get(name, ""))
        ref_orf_lengths[name] = rlen
        ref_orf_offsets[name] = _off
        _off += rlen
        print(f"    {name:>8s}: {rlen:>5d} aa")

    # Write orf_lengths.json so config.py can auto-load the authoritative
    # Nextclade-derived per-ORF lengths (overrides NCBI-coordinate estimates)
    import json as _json
    orf_lengths_path = outdir / "orf_lengths.json"
    orf_lengths_data = {
        "source": "Nextclade translation of reference.fasta (MN908947.3 / Wuhan-Hu-1)",
        "total_aa_length": ref_total_len,
        "orfs": {
            name: {
                "aa_length": ref_orf_lengths[name],
                "concat_offset": ref_orf_offsets[name],
            }
            for name in config.CONCATENATION_ORDER
        },
    }
    with open(orf_lengths_path, "w") as f:
        _json.dump(orf_lengths_data, f, indent=2)
    print(f"\n  orf_lengths.json written: {orf_lengths_path}")
    print(f"    (config.py will auto-load this to override NCBI estimates)")

    # --- Step 2: Load dataset translations ---
    print("\n--- Loading per-CDS translated sequences ---")
    all_cds = load_all_cds(indir)

    n_cds_found = sum(1 for v in all_cds.values() if v)
    if n_cds_found == 0:
        print("ERROR: No translated CDS files found. Run 03_nextclade_translate.py first.")
        sys.exit(1)
    print(f"\n  CDS files loaded: {n_cds_found} / {len(config.NEXTCLADE_CDS)}")

    # --- Step 3: Find strains present in ALL CDS ---
    print("\n--- Finding strains present in all CDS ---")
    all_strains = get_all_strain_ids(all_cds)
    print(f"  Strains in all CDS: {len(all_strains):,}")

    if not all_strains:
        print("ERROR: No strains found in all CDS files.")
        sys.exit(1)

    # --- Step 4: Load metadata ---
    meta_path = indir / "filtered_metadata.tsv"
    metadata = load_filtered_metadata(meta_path)
    print(f"  Metadata loaded: {len(metadata):,} records")

    # --- Step 5: Align each strain's ORFs to reference and concatenate ---
    print(f"\n--- Aligning and concatenating {len(all_strains):,} strains ---")

    concatenated = {}
    skipped = 0
    gap_cache = {}  # cache: (cds_name, strain_seq) -> gapped_seq

    for strain_id in sorted(all_strains):
        parts = []
        skip_strain = False

        for cds_name in config.CONCATENATION_ORDER:
            ref_seq = ref_translations.get(cds_name, "")
            strain_seq = all_cds.get(cds_name, {}).get(strain_id, "")

            if not strain_seq:
                skip_strain = True
                break

            if not ref_seq:
                # No reference for this CDS — use strain seq as-is
                parts.append(strain_seq)
                continue

            # Check cache
            cache_key = (cds_name, strain_seq)
            if cache_key in gap_cache:
                parts.append(gap_cache[cache_key])
            else:
                gapped = align_to_reference(ref_seq, strain_seq)
                gap_cache[cache_key] = gapped
                parts.append(gapped)

        if skip_strain:
            skipped += 1
            continue

        concat_seq = "".join(parts)
        concatenated[strain_id] = concat_seq

    print(f"  Concatenated: {len(concatenated):,} strains")
    print(f"  Skipped (missing CDS): {skipped}")
    print(f"  Cache entries (unique ORF sequences): {len(gap_cache):,}")

    if not concatenated:
        print("ERROR: No sequences could be concatenated.")
        sys.exit(1)

    # --- Step 6: Add reference strain ---
    ref_id = "MN908947.3|Wuhan-Hu-1"
    concatenated[ref_id] = ref_concat
    # Add metadata for reference
    metadata[ref_id] = {
        "strain": ref_id,
        "date": "2019-12-26",
        "pango_lineage": "B",
        "country": "China",
    }
    print(f"\n  Added reference: {ref_id} ({ref_total_len} aa)")

    # --- Step 7: Check lengths ---
    lengths = [len(seq) for seq in concatenated.values()]
    unique_lengths = set(lengths)
    print(f"\n  Sequence lengths:")
    print(f"    Unique lengths: {unique_lengths}")
    print(f"    All same length: {len(unique_lengths) == 1}")
    if len(unique_lengths) > 1:
        length_counts = Counter(lengths)
        print(f"    Length distribution:")
        for l, c in sorted(length_counts.items()):
            print(f"      {l} aa: {c} strains")
        # Keep only strains with reference length
        before = len(concatenated)
        concatenated = {sid: seq for sid, seq in concatenated.items()
                        if len(seq) == ref_total_len}
        print(f"  Kept {len(concatenated)} / {before} strains (length = {ref_total_len})")

    # --- Step 8: Write outputs ---
    print("\n--- Writing outputs ---")

    # aa_sequences.fasta
    aa_path = outdir / "aa_sequences.fasta"
    write_fasta(aa_path, concatenated)
    print(f"  aa_sequences.fasta: {len(concatenated):,} strains, {ref_total_len} aa each")

    # wuhan_aa.fasta (the original reference, no gaps)
    wuhan_path = outdir / "wuhan_aa.fasta"
    write_fasta(wuhan_path, {ref_id: ref_concat})
    print(f"  wuhan_aa.fasta: {ref_id} ({ref_total_len} aa, no gaps)")

    # strain_info.tsv
    info_path = outdir / "strain_info.tsv"
    with open(info_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["strain_id", "date", "lineage", "country", "aa_length"])
        for strain_id in sorted(concatenated):
            seq = concatenated[strain_id]
            meta = metadata.get(strain_id, {})
            date = meta.get("date", "")
            lineage = meta.get("pango_lineage", meta.get("Nextclade_pango", ""))
            country = meta.get("country", "")
            writer.writerow([strain_id, date, lineage, country, len(seq)])
    print(f"  strain_info.tsv: {len(concatenated):,} records")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Reference: {ref_id} ({ref_total_len} aa)")
    print(f"  Total strains (including reference): {len(concatenated):,}")
    print(f"  Sequence length: {ref_total_len} aa (all aligned to reference)")
    print(f"  Skipped: {skipped}")

    # Count strains with gaps
    n_gapped = sum(1 for seq in concatenated.values() if "-" in seq)
    print(f"  Strains with gaps: {n_gapped:,}")
    print(f"  Strains without gaps (full-length): {len(concatenated) - n_gapped:,}")

    # Lineage distribution
    lineage_counts = defaultdict(int)
    for sid in concatenated:
        meta = metadata.get(sid, {})
        lineage = meta.get("pango_lineage", meta.get("Nextclade_pango", "unknown"))
        lineage_counts[lineage] += 1

    print(f"\n  Lineage distribution ({len(lineage_counts)} lineages):")
    for lineage, count in sorted(lineage_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {lineage:<20s} {count:>6d}")
    if len(lineage_counts) > 15:
        print(f"    ... and {len(lineage_counts) - 15} more lineages")

    print(f"\n=== Concatenation complete ===")
    print(f"  Next step: python 05_mutation_boxes.py --indir {outdir} --outdir {outdir}")


if __name__ == "__main__":
    main()
