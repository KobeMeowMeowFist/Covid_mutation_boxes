#!/usr/bin/env python3
"""Step 3: Identify stable/change blocks and define mutation boxes with Gaussian context extension.

Reads the concatenated AA sequences from step 2, performs a position-wise
variability scan across all strains, identifies stable and change blocks,
and defines mutation boxes using Gaussian (bell-curve) context extension.

A mutation box is a contiguous region where mutations are ALLOWED during
the evolution walk. Positions outside boxes are frozen (no mutations).

Algorithm:
  1. For each position, compute variability = 1 - max_freq(amino_acids)
     (positions with only 'X'/'-' are treated as conserved, variability=0)
  2. A position is "variable" if variability > 0 (any non-consensus AA observed)
  3. Merge adjacent variable positions into "change blocks"
  4. Extend each change block with Gaussian context:
     - For each flanking position at distance d from the nearest change block,
       include it if exp(-d^2 / (2*sigma^2)) > threshold
     - This gives a bell-curve extension ~2*sigma on each side
  5. Merge overlapping extended blocks into final mutation boxes
  6. Output: mutation_boxes.tsv, mutation_boxes.json, visualization PNG

Usage:
    python 03_mutation_boxes.py --indir data/ --outdir data/
    python 03_mutation_boxes.py --indir data/ --outdir data/ --sigma 5 --threshold 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Define mutation boxes from strain variability"
    )
    parser.add_argument("--indir", type=Path, default=Path("data"),
                        help="Input directory with aa_sequences.fasta")
    parser.add_argument("--outdir", type=Path, default=Path("data"),
                        help="Output directory for mutation box files")
    parser.add_argument("--sigma", type=float, default=config.GAUSSIAN_SIGMA,
                        help=f"Gaussian sigma for context extension (default: {config.GAUSSIAN_SIGMA})")
    parser.add_argument("--threshold", type=float, default=config.GAUSSIAN_THRESHOLD,
                        help=f"Gaussian inclusion threshold (default: {config.GAUSSIAN_THRESHOLD})")
    parser.add_argument("--min-block-size", type=int, default=1,
                        help="Minimum change block size to keep (default: 1)")
    parser.add_argument("--variability-threshold", type=float, default=0.01,
                        help="Minimum variability to mark a position as variable (default: 0.01 = 1%%). "
                             "With 4000 strains, a position needs >=40 strains with a different AA to count.")
    parser.add_argument("--reference-id", type=str, default=None,
                        help="Strain ID to use as reference for consensus (default: Wuhan/earliest)")
    return parser.parse_args()


def load_aa_sequences(path: Path) -> dict[str, str]:
    """Load AA sequences from FASTA file."""
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


def compute_variability(sequences: dict[str, str], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-position variability and consensus amino acid.

    variability[i] = 1 - max_freq / total_valid
    consensus[i] = most common amino acid at position i

    Positions with only 'X'/'-' are treated as conserved (variability=0).
    """
    n_strains = len(sequences)
    print(f"  Computing variability across {n_strains:,} strains, {seq_len} positions...")

    # Build position-wise count matrix
    # Use array of Counters for memory efficiency
    position_counts = [Counter() for _ in range(seq_len)]

    for sid, seq in sequences.items():
        # Ensure sequence is long enough
        for i in range(min(len(seq), seq_len)):
            aa = seq[i]
            # Skip gap and unknown characters in variability computation
            if aa not in ("-", "X", "*"):
                position_counts[i][aa] += 1

    # Compute variability and consensus
    variability = np.zeros(seq_len, dtype=np.float32)
    consensus = np.array(["-"] * seq_len, dtype="<U1")

    for i in range(seq_len):
        counts = position_counts[i]
        total = sum(counts.values())
        if total == 0:
            # No valid amino acids at this position -> conserved (gap/unknown only)
            variability[i] = 0.0
            consensus[i] = "-"
        else:
            most_common_aa, most_common_count = counts.most_common(1)[0]
            consensus[i] = most_common_aa
            variability[i] = 1.0 - (most_common_count / total)

    return variability, consensus


def identify_change_blocks(variability: np.ndarray, min_block_size: int = 1,
                           variability_threshold: float = 0.0) -> list[tuple[int, int]]:
    """Identify contiguous blocks of variable positions.

    A position is "variable" if variability > variability_threshold.
    With threshold=0, any non-consensus AA counts (use only for small samples).
    With threshold=0.01, at least 1%% of strains must have a different AA.

    Returns: list of (start, end) 0-based half-open intervals
    """
    seq_len = len(variability)
    is_variable = variability > variability_threshold

    blocks = []
    in_block = False
    block_start = 0

    for i in range(seq_len):
        if is_variable[i] and not in_block:
            block_start = i
            in_block = True
        elif not is_variable[i] and in_block:
            block_end = i
            if block_end - block_start >= min_block_size:
                blocks.append((block_start, block_end))
            in_block = False

    # Close final block
    if in_block:
        block_end = seq_len
        if block_end - block_start >= min_block_size:
            blocks.append((block_start, block_end))

    return blocks


def gaussian_extend_block(block_start: int, block_end: int, seq_len: int,
                          sigma: float, threshold: float) -> tuple[int, int]:
    """Extend a change block using Gaussian context.

    For each flanking position at distance d from the block edge,
    include it if exp(-d^2 / (2*sigma^2)) > threshold.

    This gives a bell-curve extension on each side.
    """
    # Extend left
    new_start = block_start
    for d in range(1, block_start + 1):
        weight = np.exp(-(d ** 2) / (2 * sigma ** 2))
        if weight > threshold:
            new_start = block_start - d
        else:
            break

    # Extend right
    new_end = block_end
    for d in range(1, seq_len - block_end + 1):
        weight = np.exp(-(d ** 2) / (2 * sigma ** 2))
        if weight > threshold:
            new_end = block_end + d
        else:
            break

    # Clamp to sequence bounds
    new_start = max(0, new_start)
    new_end = min(seq_len, new_end)

    return (new_start, new_end)


def merge_overlapping_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent blocks."""
    if not blocks:
        return []

    blocks_sorted = sorted(blocks, key=lambda b: b[0])
    merged = [blocks_sorted[0]]

    for start, end in blocks_sorted[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # overlap or adjacent
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def build_mutation_boxes(variability: np.ndarray, sigma: float, threshold: float,
                         min_block_size: int,
                         variability_threshold: float = 0.0) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build mutation boxes from variability profile.

    Returns:
        change_blocks: raw variable blocks (before extension)
        mutation_boxes: final extended and merged boxes
    """
    seq_len = len(variability)

    # Step 1: Identify change blocks
    change_blocks = identify_change_blocks(variability, min_block_size, variability_threshold)
    print(f"  Variability threshold: {variability_threshold}")
    print(f"  Positions above threshold: {int((variability > variability_threshold).sum())} / {seq_len}")
    print(f"  Change blocks identified: {len(change_blocks)}")

    # Step 2: Gaussian extend each block
    extended_blocks = []
    for start, end in change_blocks:
        ext_start, ext_end = gaussian_extend_block(start, end, seq_len, sigma, threshold)
        extended_blocks.append((ext_start, ext_end))

    # Step 3: Merge overlapping extended blocks
    mutation_boxes = merge_overlapping_blocks(extended_blocks)
    print(f"  Mutation boxes after merge: {len(mutation_boxes)}")

    return change_blocks, mutation_boxes


def build_allowed_positions(mutation_boxes: list[tuple[int, int]],
                            seq_len: int) -> np.ndarray:
    """Build boolean array: True = position is inside a mutation box (allowed)."""
    allowed = np.zeros(seq_len, dtype=bool)
    for start, end in mutation_boxes:
        allowed[start:end] = True
    return allowed


def map_positions_to_orfs(positions: list[int]) -> list[str]:
    """Map concatenated AA positions to ORF names."""
    orf_names = []
    for pos in positions:
        for name, _, _ in config.ORFS:
            offset = config.ORF_OFFSETS[name]
            aa_len = config.ORF_AA_LENGTHS[name]
            if offset <= pos < offset + aa_len:
                orf_names.append(name)
                break
        else:
            orf_names.append("unknown")
    return orf_names


def visualize(variability: np.ndarray, change_blocks: list[tuple[int, int]],
              mutation_boxes: list[tuple[int, int]], outdir: Path):
    """Create visualization of variability profile and mutation boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    matplotlib.rcParams["svg.fonttype"] = "none"

    seq_len = len(variability)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- Panel 1: Variability profile ---
    ax = axes[0]
    positions = np.arange(seq_len)
    ax.fill_between(positions, variability, alpha=0.5, color="#0279EE")
    ax.plot(positions, variability, color="#0279EE", linewidth=0.3)
    ax.set_ylabel("Variability (1 - max freq)")
    ax.set_title("SARS-CoV-2 proteome variability across all strains")
    ax.set_xlim(0, seq_len)

    # Mark ORF boundaries
    for name, _, _ in config.ORFS:
        offset = config.ORF_OFFSETS[name]
        ax.axvline(x=offset, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        # Label at top
        ax.text(offset + config.ORF_AA_LENGTHS[name] / 2, 0.95, name,
                ha="center", va="top", fontsize=6, color="gray", rotation=45)

    # --- Panel 2: Change blocks ---
    ax = axes[1]
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Change blocks")
    ax.set_yticks([])
    for start, end in change_blocks:
        ax.axvspan(start, end, alpha=0.7, color="#FF9400")
    ax.set_xlim(0, seq_len)

    # --- Panel 3: Mutation boxes ---
    ax = axes[2]
    ax.set_xlim(0, seq_len)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mutation boxes")
    ax.set_xlabel("Concatenated AA position (0-based)")
    ax.set_yticks([])
    for start, end in mutation_boxes:
        ax.axvspan(start, end, alpha=0.7, color="#75A025")
    ax.set_xlim(0, seq_len)

    plt.tight_layout()
    fig_path = outdir / "mutation_boxes.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved: {fig_path}")

    # Also save SVG
    fig2, ax2 = plt.subplots(figsize=(16, 4))
    ax2.fill_between(positions, variability, alpha=0.5, color="#0279EE")
    ax2.set_ylabel("Variability")
    ax2.set_xlabel("Position")
    ax2.set_title("Variability profile with mutation boxes")
    for start, end in mutation_boxes:
        ax2.axvspan(start, end, alpha=0.3, color="#75A025")
    ax2.set_xlim(0, seq_len)
    svg_path = outdir / "mutation_boxes.svg"
    fig2.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig2)
    print(f"  SVG saved: {svg_path}")


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    aa_path = indir / "aa_sequences.fasta"
    if not aa_path.exists():
        print(f"ERROR: {aa_path} not found. Run 04_concatenate_orfs.py first.")
        sys.exit(1)

    # --- Load AA sequences ---
    print("Loading AA sequences...")
    sequences = load_aa_sequences(aa_path)
    print(f"  Loaded {len(sequences):,} sequences")

    if not sequences:
        print("ERROR: No sequences found.")
        sys.exit(1)

    # Determine sequence length (use first sequence)
    seq_len = len(next(iter(sequences.values())))
    print(f"  Sequence length: {seq_len} aa")
    print(f"  Expected: {config.TOTAL_AA_LENGTH} aa")

    if seq_len != config.TOTAL_AA_LENGTH:
        print(f"  WARNING: Length mismatch! Using actual length {seq_len}.")

    # --- Compute variability ---
    print("\nComputing position-wise variability...")
    variability, consensus = compute_variability(sequences, seq_len)

    # --- Build mutation boxes ---
    print(f"\nBuilding mutation boxes (sigma={args.sigma}, threshold={args.threshold})...")
    # Print variability distribution to help user choose threshold
    print(f"  Variability distribution:")
    for t in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        n_above = int((variability > t).sum())
        print(f"    > {t:.3f}: {n_above:>6d} positions ({100*n_above/seq_len:.1f}%)")

    change_blocks, mutation_boxes = build_mutation_boxes(
        variability, args.sigma, args.threshold, args.min_block_size,
        variability_threshold=args.variability_threshold
    )

    # --- Build allowed positions array ---
    allowed = build_allowed_positions(mutation_boxes, seq_len)
    n_allowed = int(allowed.sum())
    n_frozen = seq_len - n_allowed
    print(f"\n  Allowed positions (in boxes): {n_allowed:,} ({100*n_allowed/seq_len:.1f}%)")
    print(f"  Frozen positions (outside boxes): {n_frozen:,} ({100*n_frozen/seq_len:.1f}%)")

    # --- Save outputs ---

    # 1. mutation_boxes.tsv
    boxes_path = outdir / "mutation_boxes.tsv"
    print(f"\nWriting mutation boxes to {boxes_path.name}...")
    with open(boxes_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["box_id", "start_0based", "end_0based", "start_1based",
                         "end_1based", "size", "orfs"])
        for i, (start, end) in enumerate(mutation_boxes):
            # Map to ORFs
            orf_names = set(map_positions_to_orfs(list(range(start, end))))
            writer.writerow([
                i + 1, start, end, start + 1, end,
                end - start,
                ",".join(sorted(orf_names)),
            ])

    # 2. mutation_boxes.json (with detailed info)
    json_path = outdir / "mutation_boxes.json"
    print(f"Writing mutation boxes JSON to {json_path.name}...")
    boxes_data = {
        "parameters": {
            "sigma": args.sigma,
            "threshold": args.threshold,
            "min_block_size": args.min_block_size,
            "n_strains": len(sequences),
            "seq_len": seq_len,
        },
        "summary": {
            "n_change_blocks": len(change_blocks),
            "n_mutation_boxes": len(mutation_boxes),
            "n_allowed_positions": n_allowed,
            "n_frozen_positions": n_frozen,
            "percent_allowed": round(100 * n_allowed / seq_len, 2),
        },
        "change_blocks": [
            {"id": i + 1, "start": s, "end": e, "size": e - s}
            for i, (s, e) in enumerate(change_blocks)
        ],
        "mutation_boxes": [
            {
                "id": i + 1,
                "start": s,
                "end": e,
                "start_1based": s + 1,
                "end_1based": e,
                "size": e - s,
                "orfs": sorted(set(map_positions_to_orfs(list(range(s, e))))),
            }
            for i, (s, e) in enumerate(mutation_boxes)
        ],
    }
    with open(json_path, "w") as f:
        json.dump(boxes_data, f, indent=2)

    # 3. allowed_positions.npy (boolean array for the engine)
    npy_path = outdir / "allowed_positions.npy"
    np.save(npy_path, allowed)
    print(f"Allowed positions array saved: {npy_path.name}")

    # 4. consensus.npy (consensus amino acid at each position)
    consensus_path = outdir / "consensus_aa.npy"
    np.save(consensus_path, consensus)
    print(f"Consensus sequence saved: {consensus_path.name}")

    # 5. variability.npy
    var_path = outdir / "variability.npy"
    np.save(var_path, variability)
    print(f"Variability profile saved: {var_path.name}")

    # --- Visualization ---
    print("\nCreating visualization...")
    visualize(variability, change_blocks, mutation_boxes, outdir)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Strains analyzed: {len(sequences):,}")
    print(f"  Sequence length: {seq_len} aa")
    print(f"  Change blocks: {len(change_blocks)}")
    print(f"  Mutation boxes: {len(mutation_boxes)}")
    print(f"  Allowed positions: {n_allowed:,} ({100*n_allowed/seq_len:.1f}%)")
    print(f"  Frozen positions: {n_frozen:,} ({100*n_frozen/seq_len:.1f}%)")

    # Box size distribution
    box_sizes = [e - s for s, e in mutation_boxes]
    if box_sizes:
        print(f"\n  Box size distribution:")
        print(f"    Min: {min(box_sizes)}")
        print(f"    Max: {max(box_sizes)}")
        print(f"    Mean: {np.mean(box_sizes):.1f}")
        print(f"    Median: {np.median(box_sizes):.1f}")

    # Per-ORF box coverage
    print(f"\n  Per-ORF mutation box coverage:")
    for name, _, _ in config.ORFS:
        offset = config.ORF_OFFSETS[name]
        aa_len = config.ORF_AA_LENGTHS[name]
        orf_allowed = allowed[offset:offset + aa_len].sum()
        print(f"    {name:8s}: {orf_allowed:>5d} / {aa_len:>5d} "
              f"({100*orf_allowed/aa_len:.1f}%)")

    print(f"\n=== Mutation box definition complete ===")
    print(f"  Next step: python 06_prepare_target.py --indir {outdir} --outdir {outdir}")


if __name__ == "__main__":
    main()
