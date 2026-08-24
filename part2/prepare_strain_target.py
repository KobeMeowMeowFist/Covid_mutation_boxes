#!/usr/bin/env python3
"""Prepare target embedding (BA.2) and start embedding (Wuhan-Hu-1).

Mirrors prepare_targets.py but for the full-proteome windowed walk:
1. Auto-select the latest-date BA.2 strain from filtered_metadata.tsv
2. Load its concatenated AA sequence from aa_sequences.fasta
3. Fill gaps with Wuhan reference amino acids (ESM2 can't process '-')
4. Compute windowed ESM2 embedding of the BA.2 target
5. Compute windowed ESM2 embedding of the Wuhan start
6. Save both as .pt artifacts

Usage:
    python prepare_strain_target.py --data-dir data/ --outdir target_artifacts/ --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import corrected_adaptive_engine as core
import ncov_config as config
import windowed_engine as weng


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA file into {id: sequence}."""
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


def read_single_fasta(path: Path) -> tuple[str, str]:
    """Read a single-sequence FASTA file -> (name, sequence)."""
    seqs = read_fasta(path)
    if not seqs:
        raise ValueError(f"Empty FASTA: {path}")
    name, seq = next(iter(seqs.items()))
    return name, seq


# ---------------------------------------------------------------------------
# Target strain selection
# ---------------------------------------------------------------------------

def select_target_strain(
    data_dir: Path,
    target_lineage: str = "BA.2",
) -> tuple[str, str, str, str]:
    """Auto-select the latest-date strain of the given lineage.

    Searches filtered_metadata.tsv for strains whose pango_lineage starts with
    target_lineage (e.g., "BA.2" matches "BA.2", "BA.2.1", "BA.2.12.1").
    Selects the one with the latest date. Falls back to metadata.tsv if
    filtered_metadata.tsv is not found.

    Returns:
        (strain_id, date, lineage, sequence)
    """
    # Try filtered_metadata.tsv first, then metadata.tsv
    meta_path = data_dir / "filtered_metadata.tsv"
    if not meta_path.exists():
        meta_path = data_dir / "metadata.tsv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata file found in {data_dir} "
            "(expected filtered_metadata.tsv or metadata.tsv)"
        )

    print(f"Loading metadata from {meta_path.name}...")
    df = pd.read_csv(meta_path, sep="\t", dtype=str, low_memory=False)

    # Find lineage column
    lineage_col = None
    for col in ["pango_lineage", "Nextclade_pango"]:
        if col in df.columns:
            lineage_col = col
            break
    if lineage_col is None:
        raise ValueError(
            f"No pango_lineage or Nextclade_pango column in {meta_path}"
        )

    # Find strain and date columns
    strain_col = "strain" if "strain" in df.columns else None
    date_col = "date" if "date" in df.columns else None
    if strain_col is None or date_col is None:
        raise ValueError(
            f"Missing strain or date column in {meta_path}"
        )

    # Filter for target lineage (prefix match: BA.2 matches BA.2, BA.2.1, etc.)
    mask = df[lineage_col].astype(str).str.startswith(target_lineage)
    target_df = df[mask].copy()
    print(f"  Found {len(target_df)} strains with lineage starting '{target_lineage}'")

    if target_df.empty:
        raise ValueError(
            f"No strains with lineage '{target_lineage}*' found in {meta_path}"
        )

    # Sort by date descending, take the latest
    target_df["_date_sort"] = target_df[date_col].fillna("0000-00-00")
    target_df = target_df.sort_values("_date_sort", ascending=False)
    latest = target_df.iloc[0]
    strain_id = str(latest[strain_col])
    date = str(latest[date_col])
    lineage = str(latest[lineage_col])

    print(f"  Selected target: {strain_id} (date={date}, lineage={lineage})")

    # Load sequence from aa_sequences.fasta
    aa_path = data_dir / "aa_sequences.fasta"
    if not aa_path.exists():
        raise FileNotFoundError(f"Missing {aa_path}")

    print(f"Loading AA sequences from {aa_path.name}...")
    all_seqs = read_fasta(aa_path)

    # Find the target strain (try exact match, then partial match)
    if strain_id in all_seqs:
        sequence = all_seqs[strain_id]
    else:
        # Try partial match (strain ID might have different formatting)
        matches = [sid for sid in all_seqs if strain_id in sid or sid in strain_id]
        if matches:
            sequence = all_seqs[matches[0]]
            strain_id = matches[0]
            print(f"  Matched via partial ID: {strain_id}")
        else:
            raise ValueError(
                f"Target strain '{strain_id}' not found in {aa_path}. "
                f"Available strains: {len(all_seqs)}"
            )

    print(f"  Target sequence length: {len(sequence)} aa")
    return strain_id, date, lineage, sequence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare target (BA.2) and start (Wuhan) embeddings for windowed walk"
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
        help="Output directory for target artifacts",
    )
    parser.add_argument(
        "--target-lineage",
        type=str,
        default="BA.2",
        help="Pango lineage prefix for target selection (default: BA.2)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load Wuhan reference (starting sequence) ---
    wuhan_path = args.data_dir / "wuhan_aa.fasta"
    if not wuhan_path.exists():
        raise FileNotFoundError(f"Missing Wuhan reference: {wuhan_path}")

    wuhan_name, wuhan_seq = read_single_fasta(wuhan_path)
    print(f"Wuhan reference: {wuhan_name}, length={len(wuhan_seq)} aa")

    # Validate Wuhan sequence (should have no gaps)
    weng.validate_sequence_for_esm(wuhan_seq)
    seq_len = len(wuhan_seq)

    # --- Load allowed positions ---
    allowed_path = args.data_dir / "allowed_positions.npy"
    if not allowed_path.exists():
        raise FileNotFoundError(f"Missing allowed_positions: {allowed_path}")
    allowed_positions = np.load(allowed_path)
    n_allowed = int(allowed_positions.sum())
    print(f"Allowed positions: {n_allowed} / {seq_len} ({100*n_allowed/seq_len:.1f}%)")

    # --- Select target BA.2 strain ---
    target_id, target_date, target_lineage, target_seq = select_target_strain(
        args.data_dir, args.target_lineage
    )

    # --- Fill gaps in target with Wuhan reference ---
    n_gaps = target_seq.count("-")
    if n_gaps > 0:
        print(f"  Filling {n_gaps} gaps in target with Wuhan reference amino acids")
        target_seq = weng.fill_gaps_with_reference(target_seq, wuhan_seq)

    weng.validate_sequence_for_esm(target_seq)
    assert len(target_seq) == seq_len, (
        f"Target length {len(target_seq)} != Wuhan length {seq_len}"
    )

    # --- Load ESM2 (low-mem mode for CUDA, standard for CPU) ---
    if args.device.startswith("cuda"):
        model, alphabet, batch_converter = weng.load_esm2_lowmem(args.device)
    else:
        model, alphabet, batch_converter = core.load_esm2(args.device)

    # --- Create windowed embedder ---
    embedder = weng.WindowedEmbedder(
        model, alphabet, batch_converter, args.device, seq_len
    )

    # --- Compute target embedding ---
    print("\nComputing BA.2 target embedding (windowed)...")
    target_embedding = embedder.compute_full_embedding(target_seq).detach().cpu()
    print(f"  Target embedding shape: {target_embedding.shape}")

    target_digest = hashlib.sha256(target_seq.encode("ascii")).hexdigest()
    target_artifact = {
        "protein": "sars_cov_2_full",
        "length": seq_len,
        "target_embedding": target_embedding,
        "target_sha256": target_digest,
        "model": core.ESM2_MODEL,
        "repr_layer": core.REPR_LAYER,
        "target_strain_id": target_id,
        "target_date": target_date,
        "target_lineage": target_lineage,
        "target_sequence": target_seq,
        "n_gaps_filled": n_gaps,
        "n_allowed_positions": n_allowed,
        "windowed": True,
        "n_windows": embedder.n_windows,
        "window_size": config.WINDOW_SIZE,
        "window_step": config.WINDOW_STEP,
    }
    target_path = args.outdir / "sars_cov_2_full.target_embedding.pt"
    torch.save(target_artifact, target_path)
    print(f"  Saved: {target_path}")

    # --- Compute Wuhan start embedding ---
    print("\nComputing Wuhan start embedding (windowed)...")
    start_embedding = embedder.compute_full_embedding(wuhan_seq).detach().cpu()
    print(f"  Start embedding shape: {start_embedding.shape}")

    wuhan_digest = hashlib.sha256(wuhan_seq.encode("ascii")).hexdigest()
    start_artifact = {
        "protein": "sars_cov_2_full",
        "length": seq_len,
        "start_embedding": start_embedding,
        "start_sha256": wuhan_digest,
        "model": core.ESM2_MODEL,
        "repr_layer": core.REPR_LAYER,
        "start_strain_id": wuhan_name,
        "start_sequence": wuhan_seq,
    }
    start_path = args.outdir / "wuhan_start_embedding.pt"
    torch.save(start_artifact, start_path)
    print(f"  Saved: {start_path}")

    # --- Save manifest ---
    manifest = pd.DataFrame([
        {
            "artifact": target_path.name,
            "type": "target",
            "strain_id": target_id,
            "date": target_date,
            "lineage": target_lineage,
            "length": seq_len,
            "sha256": target_digest,
            "model": core.ESM2_MODEL,
            "repr_layer": core.REPR_LAYER,
        },
        {
            "artifact": start_path.name,
            "type": "start",
            "strain_id": wuhan_name,
            "date": "2019-12",
            "lineage": "B",
            "length": seq_len,
            "sha256": wuhan_digest,
            "model": core.ESM2_MODEL,
            "repr_layer": core.REPR_LAYER,
        },
    ])
    manifest_path = args.outdir / "artifact_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"\nManifest saved: {manifest_path}")

    # --- Summary ---
    print("\n=== Preparation complete ===")
    print(f"  Start: {wuhan_name} ({seq_len} aa)")
    print(f"  Target: {target_id} (date={target_date}, lineage={target_lineage})")
    print(f"  Allowed positions: {n_allowed}")
    print(f"  Windows: {embedder.n_windows}")
    print(f"  Artifacts: {args.outdir}")


if __name__ == "__main__":
    main()
