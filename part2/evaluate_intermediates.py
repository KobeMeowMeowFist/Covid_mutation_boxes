#!/usr/bin/env python3
"""Post-hoc evaluation: compare walker trajectories against hidden intermediate strains.

The walker evolves from Wuhan-Hu-1 toward BA.2, never seeing the intermediate
strains (Alpha, Delta, BA.1) during the run. This script evaluates whether the
walker's trajectory naturally passes through sequences similar to those
intermediates.

Key output: Hamming distance curves showing how the walker's sequence
diverges from Wuhan and converges toward BA.2, and whether it gets
temporarily closer to Alpha, Delta, and BA.1 along the way.

Usage:
    python evaluate_intermediates.py \
        --evolution-dir evolution_output/ \
        --data-dir data/ \
        --outdir evolution_output/evaluation/ \
        --intermediates "B.1.1.7,B.1.617.2,BA.1" \
        --target-lineage "BA.2"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import ncov_config as config


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id = None
    current_seq: list[str] = []
    with open(path) as f:
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


# ---------------------------------------------------------------------------
# Lineage consensus computation
# ---------------------------------------------------------------------------

def compute_lineage_consensus(
    sequences: list[str],
    seq_len: int,
    reference: str,
) -> str:
    """Compute consensus amino acid at each position for a set of sequences.

    Positions where all strains have gaps/unknowns are filled with the
    Wuhan reference amino acid.
    """
    consensus = list(reference)
    for i in range(seq_len):
        counts: Counter = Counter()
        for seq in sequences:
            if i < len(seq):
                aa = seq[i]
                if aa not in ("-", "X", "*"):
                    counts[aa] += 1
        if counts:
            consensus[i] = counts.most_common(1)[0][0]
    return "".join(consensus)


def select_lineage_strains(
    metadata_path: Path,
    aa_sequences: dict[str, str],
    lineage: str,
) -> list[str]:
    """Find all sequences belonging to a lineage (prefix match)."""
    df = pd.read_csv(metadata_path, sep="\t", dtype=str, low_memory=False)

    lineage_col = None
    for col in ["pango_lineage", "Nextclade_pango"]:
        if col in df.columns:
            lineage_col = col
            break
    if lineage_col is None:
        raise ValueError(f"No pango_lineage column in {metadata_path}")

    strain_col = "strain" if "strain" in df.columns else None
    if strain_col is None:
        raise ValueError("No strain column in metadata")

    mask = df[lineage_col].astype(str).str.startswith(lineage)
    strain_ids = df.loc[mask, strain_col].tolist()

    # Filter to strains that have AA sequences
    available = [sid for sid in strain_ids if sid in aa_sequences]
    return available


# ---------------------------------------------------------------------------
# Hamming distance
# ---------------------------------------------------------------------------

def hamming_distance(seq1: str, seq2: str) -> int:
    """Count positions where two equal-length sequences differ."""
    return sum(a != b for a, b in zip(seq1, seq2))


def hamming_distance_allowed(
    seq1: str,
    seq2: str,
    allowed_positions: np.ndarray,
) -> int:
    """Hamming distance restricted to allowed positions (mutation boxes)."""
    diff = 0
    for i in np.where(allowed_positions)[0]:
        if seq1[int(i)] != seq2[int(i)]:
            diff += 1
    return diff


# ---------------------------------------------------------------------------
# ORF mapping
# ---------------------------------------------------------------------------

def position_to_orf(pos: int) -> str:
    """Map a concatenated AA position to its ORF name."""
    for name in config.CONCATENATION_ORDER:
        offset = config.ORF_OFFSETS[name]
        length = config.ORF_AA_LENGTHS[name]
        if offset <= pos < offset + length:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hamming_curves(
    hamming_df: pd.DataFrame,
    outdir: Path,
    intermediates: list[str],
    target_lineage: str,
):
    """Plot Hamming distance to each strain over walk steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    matplotlib.rcParams["svg.fonttype"] = "none"

    walkers = sorted(hamming_df["walker"].unique())
    n_walkers = len(walkers)

    # Color scheme: Wuhan=blue, Alpha=orange, Delta=red, BA.1=purple, BA.2=green
    strain_colors = {
        "Wuhan": "#0279EE",
        "Alpha": "#FF9400",
        "Delta": "#E9342A",
        "BA.1": "#9B59B6",
        target_lineage: "#75A025",
    }
    strain_labels = ["Wuhan", "Alpha", "Delta", "BA.1", target_lineage]

    fig, axes = plt.subplots(
        1, n_walkers, figsize=(8 * n_walkers, 6), squeeze=False
    )

    for ax_idx, walker_id in enumerate(walkers):
        ax = axes[0, ax_idx]
        walker_data = hamming_df[hamming_df["walker"] == walker_id].sort_values("step")

        for strain_label in strain_labels:
            col = f"hamming_to_{strain_label}"
            if col in walker_data.columns:
                ax.plot(
                    walker_data["step"],
                    walker_data[col],
                    label=strain_label,
                    color=strain_colors.get(strain_label, "gray"),
                    linewidth=1.5,
                )

        ax.set_xlabel("Walk step")
        ax.set_ylabel("Hamming distance")
        ax.set_title(f"Walker {walker_id}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Hamming distance from walker to Wuhan, intermediates, and BA.2 target",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    png_path = outdir / "hamming_distance_curves.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    svg_path = outdir / "hamming_distance_curves.svg"
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {svg_path}")


def plot_prcs_progress(
    hamming_df: pd.DataFrame,
    outdir: Path,
):
    """Plot PRCS to target over walk steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    matplotlib.rcParams["svg.fonttype"] = "none"

    walkers = sorted(hamming_df["walker"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for walker_id in walkers:
        walker_data = hamming_df[
            hamming_df["walker"] == walker_id
        ].sort_values("step")
        ax.plot(
            walker_data["step"],
            walker_data["prcs"],
            label=f"Walker {walker_id}",
            linewidth=1.5,
        )

    ax.set_xlabel("Walk step")
    ax.set_ylabel("PRCS to BA.2 target")
    ax.set_title("PRCS progress toward BA.2 target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = outdir / "prcs_progress.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


def plot_mutation_progress(
    hamming_df: pd.DataFrame,
    outdir: Path,
    wuhan_seq: str,
):
    """Plot cumulative mutations over time, broken down by ORF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    matplotlib.rcParams["svg.fonttype"] = "none"

    walkers = sorted(hamming_df["walker"].unique())
    orf_colors = {
        "ORF1a": "#0279EE", "ORF1b": "#1A77C4", "S": "#E9342A",
        "ORF3a": "#FF9400", "E": "#FFD700", "M": "#75A025",
        "ORF6": "#33AA33", "ORF7a": "#16A085", "ORF7b": "#1ABC9C",
        "ORF8": "#9B59B6", "N": "#E91E63", "ORF9b": "#795548",
    }

    fig, axes = plt.subplots(
        1, len(walkers), figsize=(8 * len(walkers), 5), squeeze=False
    )

    for ax_idx, walker_id in enumerate(walkers):
        ax = axes[0, ax_idx]
        walker_data = hamming_df[
            hamming_df["walker"] == walker_id
        ].sort_values("step")

        # Compute per-ORF mutation counts at each step
        orf_counts = {orf: [] for orf in config.CONCATENATION_ORDER}
        for _, row in walker_data.iterrows():
            seq = row["sequence"]
            mutations_by_orf = {orf: 0 for orf in config.CONCATENATION_ORDER}
            for i in range(len(seq)):
                if seq[i] != wuhan_seq[i]:
                    orf = position_to_orf(i)
                    if orf in mutations_by_orf:
                        mutations_by_orf[orf] += 1
            for orf in config.CONCATENATION_ORDER:
                orf_counts[orf].append(mutations_by_orf[orf])

        for orf in config.CONCATENATION_ORDER:
            ax.plot(
                walker_data["step"],
                orf_counts[orf],
                label=orf,
                color=orf_colors.get(orf, "gray"),
                linewidth=1.2,
            )

        ax.set_xlabel("Walk step")
        ax.set_ylabel("Mutations (vs Wuhan)")
        ax.set_title(f"Walker {walker_id}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Cumulative mutations by ORF over walk steps",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    png_path = outdir / "mutation_progress.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate walker trajectories against hidden intermediate strains"
    )
    parser.add_argument(
        "--evolution-dir",
        type=Path,
        required=True,
        help="Directory with walker_trajectories/ and run_metadata.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with aa_sequences.fasta, wuhan_aa.fasta, filtered_metadata.tsv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--intermediates",
        type=str,
        default="B.1.1.7,B.1.617.2,BA.1",
        help="Comma-separated Pango lineages for intermediate strains",
    )
    parser.add_argument(
        "--target-lineage",
        type=str,
        default="BA.2",
        help="Target lineage (for labeling)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    intermediates = [s.strip() for s in args.intermediates.split(",") if s.strip()]
    target_lineage = args.target_lineage

    # --- Load metadata ---
    meta_path = args.data_dir / "filtered_metadata.tsv"
    if not meta_path.exists():
        meta_path = args.data_dir / "metadata.tsv"
    if not meta_path.exists():
        raise FileNotFoundError("No metadata file found")

    # --- Load AA sequences ---
    aa_path = args.data_dir / "aa_sequences.fasta"
    if not aa_path.exists():
        raise FileNotFoundError(f"Missing {aa_path}")
    print("Loading AA sequences...")
    aa_sequences = read_fasta(aa_path)
    print(f"  {len(aa_sequences)} sequences loaded")

    # --- Load Wuhan reference ---
    wuhan_path = args.data_dir / "wuhan_aa.fasta"
    if not wuhan_path.exists():
        raise FileNotFoundError(f"Missing {wuhan_path}")
    wuhan_seqs = read_fasta(wuhan_path)
    wuhan_name, wuhan_seq = next(iter(wuhan_seqs.items()))
    seq_len = len(wuhan_seq)
    print(f"Wuhan reference: {wuhan_name}, {seq_len} aa")

    # --- Load allowed positions ---
    allowed_path = args.data_dir / "allowed_positions.npy"
    allowed_positions = np.load(allowed_path)

    # --- Compute consensus for each intermediate lineage ---
    print("\nComputing lineage consensus sequences...")
    consensus_seqs: dict[str, str] = {}
    consensus_seqs["Wuhan"] = wuhan_seq

    # Friendly names for intermediates
    friendly_names = {
        "B.1.1.7": "Alpha",
        "B.1.617.2": "Delta",
        "BA.1": "BA.1",
    }

    for lineage in intermediates:
        strain_ids = select_lineage_strains(meta_path, aa_sequences, lineage)
        if not strain_ids:
            print(f"  WARNING: No strains found for lineage {lineage}")
            continue
        seqs = [aa_sequences[sid] for sid in strain_ids if sid in aa_sequences]
        consensus = compute_lineage_consensus(seqs, seq_len, wuhan_seq)
        label = friendly_names.get(lineage, lineage)
        consensus_seqs[label] = consensus
        n_mut = hamming_distance(consensus, wuhan_seq)
        print(f"  {lineage} ({label}): {len(seqs)} strains, "
              f"{n_mut} mutations from Wuhan")

    # --- Compute target consensus ---
    target_strains = select_lineage_strains(meta_path, aa_sequences, target_lineage)
    if target_strains:
        target_seqs = [aa_sequences[sid] for sid in target_strains if sid in aa_sequences]
        target_consensus = compute_lineage_consensus(target_seqs, seq_len, wuhan_seq)
        consensus_seqs[target_lineage] = target_consensus
        n_mut = hamming_distance(target_consensus, wuhan_seq)
        print(f"  {target_lineage}: {len(target_seqs)} strains, "
              f"{n_mut} mutations from Wuhan")
    else:
        print(f"  WARNING: No {target_lineage} strains found for target consensus")

    # --- Load walker trajectories ---
    trajectory_dir = args.evolution_dir / "walker_trajectories"
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Missing trajectory directory: {trajectory_dir}")

    trajectory_files = sorted(trajectory_dir.glob("walker_*.csv"))
    if not trajectory_files:
        raise FileNotFoundError("No walker trajectory CSV files found")

    print(f"\nLoading {len(trajectory_files)} walker trajectories...")

    all_rows = []
    for traj_path in trajectory_files:
        walker_id = int(traj_path.stem.split("_")[1])
        print(f"  Walker {walker_id}: {traj_path.name}")
        traj = pd.read_csv(traj_path)

        for _, row in traj.iterrows():
            step = int(row["step"])
            seq = row["sequence"]
            prcs = float(row["prcs"])

            record = {
                "walker": walker_id,
                "step": step,
                "prcs": prcs,
                "sequence": seq,
            }

            # Hamming distance to each strain
            for label, ref_seq in consensus_seqs.items():
                record[f"hamming_to_{label}"] = hamming_distance(seq, ref_seq)

            # Hamming distance at allowed positions only
            for label, ref_seq in consensus_seqs.items():
                record[f"hamming_allowed_to_{label}"] = hamming_distance_allowed(
                    seq, ref_seq, allowed_positions
                )

            # Total mutations from Wuhan
            record["n_mutations_from_wuhan"] = hamming_distance(seq, wuhan_seq)

            all_rows.append(record)

    hamming_df = pd.DataFrame(all_rows)
    print(f"  Total rows: {len(hamming_df)}")

    # --- Save TSV ---
    tsv_path = args.outdir / "intermediate_comparison.tsv"
    # Drop the full sequence column for the TSV (too large)
    tsv_cols = [c for c in hamming_df.columns if c != "sequence"]
    hamming_df[tsv_cols].to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSaved: {tsv_path}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_hamming_curves(hamming_df, args.outdir, intermediates, target_lineage)
    plot_prcs_progress(hamming_df, args.outdir)
    plot_mutation_progress(hamming_df, args.outdir, wuhan_seq)

    # --- Summary report ---
    print("\nWriting evaluation report...")
    report_path = args.outdir / "evaluation_report.md"

    # Compute key metrics
    report_lines = [
        "# Evaluation Report: Walker Trajectory vs Intermediate Strains",
        "",
        "## Setup",
        f"- Start: Wuhan-Hu-1 ({wuhan_name}, {seq_len} aa)",
        f"- Target: {target_lineage} consensus",
        f"- Intermediates: {', '.join(intermediates)}",
        f"- Walkers: {sorted(hamming_df['walker'].unique().tolist())}",
        f"- Steps: {hamming_df['step'].max() + 1}",
        "",
        "## Consensus Strain Distances from Wuhan",
        "",
        "| Strain | Hamming to Wuhan | Hamming (allowed only) |",
        "|--------|-----------------|----------------------|",
    ]

    for label in ["Wuhan"] + [friendly_names.get(l, l) for l in intermediates] + [target_lineage]:
        if label in consensus_seqs:
            total = hamming_distance(consensus_seqs[label], wuhan_seq)
            allowed = hamming_distance_allowed(
                consensus_seqs[label], wuhan_seq, allowed_positions
            )
            report_lines.append(f"| {label} | {total} | {allowed} |")

    report_lines.extend([
        "",
        "## Walker Results",
        "",
        "| Walker | Start PRCS | Final PRCS | Best PRCS | Final mutations |",
        "|--------|-----------|------------|-----------|-----------------|",
    ])

    for walker_id in sorted(hamming_df["walker"].unique()):
        walker_data = hamming_df[hamming_df["walker"] == walker_id]
        start_prcs = walker_data.iloc[0]["prcs"]
        final_prcs = walker_data.iloc[-1]["prcs"]
        best_prcs = walker_data["prcs"].max()
        final_mut = walker_data.iloc[-1]["n_mutations_from_wuhan"]
        report_lines.append(
            f"| {walker_id} | {start_prcs:.4f} | {final_prcs:.4f} | "
            f"{best_prcs:.4f} | {int(final_mut)} |"
        )

    # Check if walker passes through intermediates
    report_lines.extend([
        "",
        "## Intermediate Passage Analysis",
        "",
        "For each intermediate, we check whether the walker's Hamming distance",
        "to that intermediate reaches a local minimum (gets closer) before",
        "moving toward the target.",
        "",
    ])

    for walker_id in sorted(hamming_df["walker"].unique()):
        walker_data = hamming_df[
            hamming_df["walker"] == walker_id
        ].sort_values("step")

        report_lines.append(f"### Walker {walker_id}")
        report_lines.append("")
        report_lines.append(
            "| Intermediate | Min Hamming | Step at min | Final Hamming |"
        )
        report_lines.append(
            "|-------------|-------------|-------------|---------------|"
        )

        for lineage in intermediates:
            label = friendly_names.get(lineage, lineage)
            col = f"hamming_to_{label}"
            if col not in walker_data.columns:
                continue
            values = walker_data[col].values
            min_idx = int(np.argmin(values))
            min_val = int(values[min_idx])
            min_step = int(walker_data.iloc[min_idx]["step"])
            final_val = int(values[-1])
            report_lines.append(
                f"| {label} | {min_val} | {min_step} | {final_val} |"
            )

        # Target
        target_col = f"hamming_to_{target_lineage}"
        if target_col in walker_data.columns:
            values = walker_data[target_col].values
            min_idx = int(np.argmin(values))
            min_val = int(values[min_idx])
            min_step = int(walker_data.iloc[min_idx]["step"])
            final_val = int(values[-1])
            report_lines.append(
                f"| {target_lineage} (target) | {min_val} | {min_step} | {final_val} |"
            )

        report_lines.append("")

    # Interpretation
    report_lines.extend([
        "## Interpretation",
        "",
        "If the walker naturally follows the evolutionary timeline, we expect:",
        "1. Hamming to Wuhan increases monotonically (walker diverges from start)",
        "2. Hamming to Alpha reaches a minimum early, then increases",
        "3. Hamming to Delta reaches a minimum later, then increases",
        "4. Hamming to BA.1 reaches a minimum even later, then increases",
        "5. Hamming to BA.2 (target) decreases monotonically",
        "",
        "The order of minimum Hamming distance should be:",
        "Alpha (earliest) < Delta < BA.1 < BA.2 (latest)",
        "",
        "## Files",
        f"- `hamming_distance_curves.png` — Hamming distance to each strain over steps",
        f"- `prcs_progress.png` — PRCS to target over steps",
        f"- `mutation_progress.png` — Cumulative mutations by ORF over steps",
        f"- `intermediate_comparison.tsv` — Per-step Hamming distances",
    ])

    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"  Saved: {report_path}")

    print("\n=== Evaluation complete ===")
    print(f"  Results in: {args.outdir}")


if __name__ == "__main__":
    main()
