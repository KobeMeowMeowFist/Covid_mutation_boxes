#!/usr/bin/env python3
"""Run the windowed adaptive walk: Wuhan-Hu-1 -> BA.2.

Entry point for the full-proteome evolution run. Mirrors run_final_adaptive.py
but uses windowed embeddings and mutation box constraints.

Key differences from the original:
1. Loads allowed_positions from data/allowed_positions.npy
2. Loads Wuhan reference as starting sequence (not random)
3. Creates WalkConfig via make_strain_walk_config() (bypasses 1022 limit)
4. Calls run_windowed_walk() instead of run_walk()
5. Both walkers start from the SAME Wuhan reference (different RNG seeds)
6. Saves rejection log for epistatic pair rule violations

Usage:
    python run_strain_adaptive.py \
        --artifact-dir target_artifacts/ \
        --data-dir data/ \
        --outdir evolution_output/ \
        --seed 42 --walkers 2 --total-steps 1000 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

import corrected_adaptive_engine as core
import windowed_engine as weng
import windowed_walk
import ncov_config as config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_artifact(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_target_artifact(artifact: dict) -> None:
    required = {
        "protein", "length", "target_embedding", "target_sha256",
        "model", "repr_layer",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Target artifact missing fields: {sorted(missing)}")
    if artifact["model"] != core.ESM2_MODEL:
        raise ValueError("Target artifact was built with a different ESM model")
    if artifact["repr_layer"] != core.REPR_LAYER:
        raise ValueError("Target artifact repr_layer mismatch")
    if tuple(artifact["target_embedding"].shape)[:1] != (int(artifact["length"]),):
        raise ValueError("Target artifact embedding shape is inconsistent")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Windowed adaptive walk: Wuhan-Hu-1 -> BA.2"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory with target and start embedding artifacts",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with allowed_positions.npy and wuhan_aa.fasta",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for evolution results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walkers", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.walkers < 1:
        raise ValueError("--walkers must be positive")
    if args.total_steps < 1:
        raise ValueError("--total-steps must be positive")

    # --- Load target artifact ---
    target_path = args.artifact_dir / "sars_cov_2_full.target_embedding.pt"
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target artifact: {target_path}")
    target_artifact = load_artifact(target_path)
    validate_target_artifact(target_artifact)
    length = int(target_artifact["length"])
    target_strain_id = target_artifact.get("target_strain_id", "unknown")
    target_date = target_artifact.get("target_date", "unknown")
    target_lineage = target_artifact.get("target_lineage", "unknown")

    print(f"Target: {target_strain_id} (date={target_date}, lineage={target_lineage})")
    print(f"  Length: {length} aa")

    # --- Load start artifact (Wuhan) ---
    start_path = args.artifact_dir / "wuhan_start_embedding.pt"
    if not start_path.exists():
        raise FileNotFoundError(f"Missing start artifact: {start_path}")
    start_artifact = load_artifact(start_path)
    start_sequence = start_artifact["start_sequence"]
    start_strain_id = start_artifact.get("start_strain_id", "MN908947.3")

    print(f"Start: {start_strain_id} ({len(start_sequence)} aa)")

    if len(start_sequence) != length:
        raise ValueError(
            f"Start sequence length ({len(start_sequence)}) != target length ({length})"
        )

    # --- Load allowed positions ---
    allowed_path = args.data_dir / "allowed_positions.npy"
    if not allowed_path.exists():
        raise FileNotFoundError(f"Missing allowed_positions: {allowed_path}")
    allowed_positions = np.load(allowed_path)
    n_allowed = int(allowed_positions.sum())
    print(f"Allowed positions: {n_allowed} / {length} ({100*n_allowed/length:.1f}%)")

    if len(allowed_positions) != length:
        raise ValueError(
            f"allowed_positions length ({len(allowed_positions)}) != sequence length ({length})"
        )

    # --- Create walk config (bypass 1022 limit) ---
    walk_config = weng.make_strain_walk_config(length, n_allowed, args.total_steps)
    print(f"\nWalkConfig:")
    print(f"  k_single = {walk_config.k_single}")
    print(f"  p_coupled = {walk_config.p_coupled:.2f}")
    print(f"  n_groups = {walk_config.n_groups}")
    print(f"  heating_steps = {walk_config.heating_steps}")
    print(f"  cooling_rate = {walk_config.cooling_rate:.6f}")

    # --- Set up output directories ---
    args.outdir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.outdir / "walker_trajectories"
    sequence_dir = args.outdir / "final_sequences"
    checkpoint_dir = args.outdir / "checkpoints"
    trajectory_dir.mkdir(exist_ok=True)
    sequence_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # --- Record initial walker info ---
    # Both walkers start from the SAME Wuhan reference
    initial_rows = []
    for walker_id in range(args.walkers):
        walk_seed = args.seed + walker_id * 1009 + 17
        initial_rows.append({
            "walker": walker_id,
            "walk_seed": walk_seed,
            "start_strain": start_strain_id,
            "start_sha256": hashlib.sha256(
                start_sequence.encode("ascii")
            ).hexdigest(),
            "start_prcs": None,  # filled after first walker runs
        })
    initial_path = args.outdir / "initial_walkers.csv"
    initial_frame = pd.DataFrame(initial_rows)
    if initial_path.exists():
        existing = pd.read_csv(initial_path)
        if not existing.equals(initial_frame):
            raise RuntimeError(
                "Existing initial walker manifest does not match this run"
            )
    else:
        initial_frame.to_csv(initial_path, index=False)

    # --- Load ESM2 (low-mem mode for CUDA, standard for CPU) ---
    if args.device.startswith("cuda"):
        model, alphabet, batch_converter = weng.load_esm2_lowmem(args.device)
    else:
        print(f"\nLoading ESM2 on {args.device}...")
        model, alphabet, batch_converter = core.load_esm2(args.device)
    target_embedding = target_artifact["target_embedding"].to(args.device)

    # --- Run walkers ---
    summaries = []
    all_rejections = []

    for walker_id in range(args.walkers):
        walk_seed = args.seed + walker_id * 1009 + 17
        print(f"\n{'='*60}")
        print(f"  Walker {walker_id} (seed={walk_seed})")
        print(f"{'='*60}")

        result = windowed_walk.run_windowed_walk(
            protein_name="sars_cov_2_full",
            walker_id=walker_id,
            start_sequence=start_sequence,
            target_embedding=target_embedding,
            model=model,
            alphabet=alphabet,
            batch_converter=batch_converter,
            device=args.device,
            rng=random.Random(walk_seed),
            walk_config=walk_config,
            allowed_positions=allowed_positions,
            checkpoint_path=checkpoint_dir / f"walker_{walker_id:02d}.pkl",
        )

        # Save trajectory
        pd.DataFrame(result["trajectory"]).to_csv(
            trajectory_dir / f"walker_{walker_id:02d}.csv", index=False
        )

        # Save final and best sequences
        core.write_sequence_fasta(
            sequence_dir / f"walker_{walker_id:02d}_final.fasta",
            f"sars_cov_2_seed{args.seed}_walker{walker_id}_final",
            result["final_sequence"],
        )
        core.write_sequence_fasta(
            sequence_dir / f"walker_{walker_id:02d}_best.fasta",
            f"sars_cov_2_seed{args.seed}_walker{walker_id}_best",
            result["best_sequence"],
        )

        # Collect rejections
        if result["rejection_log"]:
            for rej in result["rejection_log"]:
                rej["walker"] = walker_id
                all_rejections.append(rej)

        summaries.append({
            "walker": walker_id,
            "walk_seed": walk_seed,
            "start_prcs": result["start_prcs"],
            "final_prcs": result["final_prcs"],
            "best_prcs": result["best_prcs"],
            "final_combined": result["final_combined"],
            "best_combined": result["best_combined"],
            "acceptance_rate_single": result["acceptance_rate_single"],
            "acceptance_rate_coupled": result["acceptance_rate_coupled"],
            "n_rejections": len(result["rejection_log"]),
        })
        pd.DataFrame(summaries).to_csv(
            args.outdir / "final_walker_summary.csv", index=False
        )

    # --- Save rejection log ---
    if all_rejections:
        pd.DataFrame(all_rejections).to_csv(
            args.outdir / "rejection_log.csv", index=False
        )
        print(f"\nWARNING: {len(all_rejections)} epistatic pair rule violations logged")
    else:
        # Write empty file to confirm no violations
        pd.DataFrame(columns=["walker", "reason", "positions", "violations"]).to_csv(
            args.outdir / "rejection_log.csv", index=False
        )
        print(f"\nNo epistatic pair rule violations (rejection_log.csv is empty)")

    # --- Save run metadata ---
    metadata = {
        "algorithm": "windowed_adaptive_walk",
        "protein": "sars_cov_2_full",
        "seed": args.seed,
        "walkers": args.walkers,
        "total_steps_per_walker": args.total_steps,
        "proposals_per_step": core.PROPOSALS_PER_STEP,
        "total_candidate_evaluations": (
            args.walkers * args.total_steps * core.PROPOSALS_PER_STEP
        ),
        "target_artifact": target_path.name,
        "target_strain_id": target_strain_id,
        "target_date": target_date,
        "target_lineage": target_lineage,
        "target_length": length,
        "target_model": target_artifact["model"],
        "target_sha256": target_artifact["target_sha256"],
        "start_strain_id": start_strain_id,
        "start_sha256": start_artifact["start_sha256"],
        "n_allowed_positions": n_allowed,
        "windowed": True,
        "window_size": config.WINDOW_SIZE,
        "window_step": config.WINDOW_STEP,
        "n_windows": len(config.get_windows(length)),
        "live_target_sequence_access": False,
        "config": walk_config.as_dict(),
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\n{'='*60}")
    print(f"  Evolution complete: {args.outdir}")
    print(f"  Walkers: {args.walkers}, Steps: {args.total_steps}")
    for s in summaries:
        print(f"    Walker {s['walker']}: "
              f"start={s['start_prcs']:.4f} -> "
              f"final={s['final_prcs']:.4f}, "
              f"best={s['best_prcs']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
