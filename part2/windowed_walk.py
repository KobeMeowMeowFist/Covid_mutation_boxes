#!/usr/bin/env python3
"""Windowed adaptive walk execution loop.

Mirrors adaptive_walk.py's run_walk() but uses WindowedEmbedder for all
embedding operations and enforces mutation box constraints on all proposals.

Key differences from the original walk loop:
1. Embeddings via WindowedEmbedder (overlapping windows + triangular stitching)
2. Proposals constrained to allowed_positions (mutation boxes)
3. Coupling map computed within windows only
4. Coupling groups pre-filtered to allowed positions (epistatic pair rule)
5. PLL diagnostic skipped (infeasible for 9803 aa)
6. Candidates scored one at a time to manage GPU memory
7. Rejection log for epistatic pair rule violations
"""

from __future__ import annotations

import math
from pathlib import Path
import random

import numpy as np
import torch

import corrected_adaptive_engine as core
import ncov_config as config
import windowed_engine as weng


def run_windowed_walk(
    *,
    protein_name: str,
    walker_id: int,
    start_sequence: str,
    target_embedding: torch.Tensor,
    model,
    alphabet,
    batch_converter,
    device: str,
    rng: random.Random,
    walk_config: core.WalkConfig,
    allowed_positions: np.ndarray,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run one windowed adaptive walker from start_sequence toward target_embedding.

    All mutations are constrained to allowed_positions (mutation boxes).
    The walk uses windowed ESM2 embeddings with triangular stitching.

    Args:
        protein_name: name for logging
        walker_id: walker index
        start_sequence: Wuhan-Hu-1 reference (starting sequence)
        target_embedding: stitched embedding of BA.2 target (seq_len, 1280)
        model, alphabet, batch_converter: ESM2 model components
        device: torch device string
        rng: random number generator (seeded per walker)
        walk_config: WalkConfig from make_strain_walk_config()
        allowed_positions: boolean array (seq_len,), True = inside mutation box
        checkpoint_path: path for resumable checkpoint

    Returns:
        dict with final_sequence, best_sequence, trajectory, acceptance rates, etc.
    """
    length = len(start_sequence)
    if length != walk_config.length:
        raise ValueError(
            f"Start sequence length ({length}) != config length ({walk_config.length})"
        )
    if tuple(target_embedding.shape)[:1] != (length,):
        raise ValueError("Target embedding length does not match start sequence")

    allowed_positions_set = set(int(p) for p in np.where(allowed_positions)[0])
    n_allowed = len(allowed_positions_set)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create windowed embedder
    embedder = weng.WindowedEmbedder(
        model, alphabet, batch_converter, device, length
    )

    rejection_log: list[dict] = []

    # ------------------------------------------------------------------
    # Resume from checkpoint or initialize fresh
    # ------------------------------------------------------------------
    if checkpoint_path is not None and checkpoint_path.exists():
        state = core.load_checkpoint(checkpoint_path)
        if state.get("protein_name") != protein_name or state.get("walker_id") != walker_id:
            raise RuntimeError("Checkpoint identity does not match this walk")
        if state.get("config") != walk_config.as_dict():
            raise RuntimeError("Checkpoint configuration does not match this walk")

        current_sequence = state["current_sequence"]
        start_sequence = state["start_sequence"]
        start_prcs = float(state["start_prcs"])
        current_prcs = float(state["current_prcs"])
        current_combined = float(state["current_combined"])
        best_sequence = state["best_sequence"]
        best_prcs = float(state["best_prcs"])
        best_combined = float(state["best_combined"])
        temperature = float(state["temperature"])
        lambda_max = float(state["lambda_max"])
        effective_threshold = float(state["effective_threshold"])
        groups = state["groups"]
        coupling_scores = state["coupling_scores"]
        coupling_map_jaccard = float(state.get("coupling_map_jaccard", np.nan))
        trajectory = state["trajectory"]
        accepted_single = int(state["accepted_single"])
        proposed_single = int(state["proposed_single"])
        accepted_coupled = int(state["accepted_coupled"])
        proposed_coupled = int(state["proposed_coupled"])
        rng.setstate(state["rng_state"])
        first_step = int(state["step"]) + 1

        # Recompute embedding from sequence (no window cache in checkpoint)
        current_embedding = embedder.compute_full_embedding(current_sequence)
        _, current_per_position = core.score_prcs(current_embedding, target_embedding)
        supported_pairs = core.coupling_support_set(coupling_scores, effective_threshold)

        print(
            f"  [Walker {walker_id}] Resuming at step {first_step}; "
            f"PRCS={current_prcs:.4f}, best={best_prcs:.4f}"
        )
    else:
        # Fresh start
        current_sequence = start_sequence
        current_embedding = embedder.compute_full_embedding(start_sequence)
        start_prcs, current_per_position = core.score_prcs(
            current_embedding, target_embedding
        )
        current_prcs = start_prcs
        current_combined = start_prcs

        print(
            f"  [Walker {walker_id}] {protein_name}: L={length}, "
            f"allowed={n_allowed}, steps={walk_config.max_steps}, "
            f"start PRCS={start_prcs:.4f}"
        )
        if walk_config.heating_steps:
            print(
                f"  [Walker {walker_id}] Heating for {walk_config.heating_steps} steps; "
                f"k={walk_config.heating_k_min}-{walk_config.heating_k_max}, "
                f"probability={walk_config.heating_probability:.2f}"
            )

        # Initial coupling map (within-window only)
        sample_count = core.get_coupling_sample_pairs(n_allowed)
        print(
            f"  [Walker {walker_id}] Computing initial coupling map "
            f"({sample_count} sampled pairs, within-window)"
        )
        coupling_scores = weng.compute_windowed_coupling_map(
            current_sequence,
            model,
            alphabet,
            batch_converter,
            device,
            allowed_positions,
            embedder.windows,
            sample_count,
            rng,
        )
        groups, effective_threshold = weng.build_constrained_coupling_groups(
            coupling_scores,
            length,
            allowed_positions_set,
            walk_config.n_groups,
            walk_config.group_size_min,
            walk_config.group_size_max,
            core.COUPLING_THRESHOLD,
        )
        lambda_max, mean_coupling = core.compute_adaptive_lambda_max(
            coupling_scores, effective_threshold
        )
        supported_pairs = core.coupling_support_set(
            coupling_scores, effective_threshold
        )
        coupling_map_jaccard = np.nan

        print(
            f"  [Walker {walker_id}] groups={len(groups)}, "
            f"threshold={effective_threshold:.4f}, "
            f"lambda_max={lambda_max:.3f}, mean coupling={mean_coupling:.4f}"
        )

        temperature = core.INITIAL_TEMPERATURE
        best_sequence = current_sequence
        best_prcs = current_prcs
        best_combined = current_combined
        trajectory = []
        accepted_single = proposed_single = 0
        accepted_coupled = proposed_coupled = 0
        first_step = 0

    sample_count = core.get_coupling_sample_pairs(n_allowed)

    # ------------------------------------------------------------------
    # Checkpoint helper
    # ------------------------------------------------------------------
    def checkpoint_state(step: int) -> dict:
        return {
            "schema_version": 2,
            "protein_name": protein_name,
            "walker_id": walker_id,
            "config": walk_config.as_dict(),
            "step": step,
            "start_sequence": start_sequence,
            "start_prcs": start_prcs,
            "current_sequence": current_sequence,
            "current_prcs": current_prcs,
            "current_combined": current_combined,
            "best_sequence": best_sequence,
            "best_prcs": best_prcs,
            "best_combined": best_combined,
            "temperature": temperature,
            "lambda_max": lambda_max,
            "effective_threshold": effective_threshold,
            "groups": groups,
            "coupling_scores": coupling_scores,
            "coupling_map_jaccard": coupling_map_jaccard,
            "trajectory": trajectory,
            "accepted_single": accepted_single,
            "proposed_single": proposed_single,
            "accepted_coupled": accepted_coupled,
            "proposed_coupled": proposed_coupled,
            "rng_state": rng.getstate(),
            "n_allowed": n_allowed,
            "rejection_log": rejection_log,
        }

    # ------------------------------------------------------------------
    # Main walk loop
    # ------------------------------------------------------------------
    for step in range(first_step, walk_config.max_steps):

        # --- Recompute coupling map periodically ---
        if step > 0 and step % core.COUPLING_RECOMPUTE_EVERY == 0:
            print(f"  [Walker {walker_id}] Step {step}: recomputing coupling map")
            previous_supported_pairs = supported_pairs
            coupling_scores = weng.compute_windowed_coupling_map(
                current_sequence,
                model,
                alphabet,
                batch_converter,
                device,
                allowed_positions,
                embedder.windows,
                sample_count,
                rng,
            )
            groups, effective_threshold = weng.build_constrained_coupling_groups(
                coupling_scores,
                length,
                allowed_positions_set,
                walk_config.n_groups,
                walk_config.group_size_min,
                walk_config.group_size_max,
                core.COUPLING_THRESHOLD,
            )
            lambda_max, _ = core.compute_adaptive_lambda_max(
                coupling_scores, effective_threshold
            )
            supported_pairs = core.coupling_support_set(
                coupling_scores, effective_threshold
            )
            coupling_map_jaccard = core.set_jaccard(
                previous_supported_pairs, supported_pairs
            )

        # --- Compute current score ---
        coupling_lambda = core.compute_lambda(current_prcs, start_prcs, lambda_max)
        current_combined, current_prcs, current_epistasis, current_per_position = (
            core.compute_combined_score(
                current_embedding, target_embedding, groups, coupling_lambda
            )
        )

        # --- Adaptive refinement ---
        (
            effective_k,
            effective_p_coupled,
            adaptive_stage,
            adaptive_progress,
            adaptive_plateau,
        ) = core.choose_adaptive_refinement(
            trajectory,
            current_prcs,
            start_prcs,
            walk_config.k_single,
            walk_config.p_coupled,
        )

        # --- Generate candidates ---
        candidate_sequences = []
        candidate_coupled = []
        candidate_positions = []

        for _ in range(walk_config.proposals_per_step):
            heating_active = bool(
                walk_config.heating_steps and step < walk_config.heating_steps
            )
            use_heating = heating_active and rng.random() < walk_config.heating_probability

            if use_heating:
                heating_k = rng.randint(
                    walk_config.heating_k_min, walk_config.heating_k_max
                )
                candidate, positions = weng.propose_single_guided_constrained(
                    current_sequence,
                    heating_k,
                    current_per_position,
                    allowed_positions,
                    rng,
                    guided_probability=core.P_GUIDED_HEATING,
                )
                is_coupled = False
            elif rng.random() < effective_p_coupled and groups:
                candidate, positions = weng.propose_coupled_constrained(
                    current_sequence,
                    groups,
                    allowed_positions_set,
                    allowed_positions,
                    rng,
                    rejection_log=rejection_log,
                )
                is_coupled = True
            else:
                candidate, positions = weng.propose_single_guided_constrained(
                    current_sequence,
                    effective_k,
                    current_per_position,
                    allowed_positions,
                    rng,
                    guided_probability=core.P_GUIDED_ADAPTIVE,
                )
                is_coupled = False

            candidate_sequences.append(candidate)
            candidate_coupled.append(is_coupled)
            candidate_positions.append(positions)

        # --- Compute candidate embeddings (windowed, incremental) ---
        window_new_embs = embedder.compute_candidate_window_embeddings(
            candidate_sequences, candidate_positions
        )

        # --- Score candidates one at a time (memory management) ---
        best_candidate_combined = float("-inf")
        best_candidate_idx = 0
        best_candidate_prcs = 0.0
        best_candidate_epistasis = 0.0
        best_candidate_per_position = None
        best_candidate_coupled = False
        best_candidate_positions: list[int] = []

        for cand_idx in range(len(candidate_sequences)):
            cand_emb = embedder.stitch_candidate(cand_idx, window_new_embs)
            combined, prcs, epistasis, per_position = core.compute_combined_score(
                cand_emb, target_embedding, groups, coupling_lambda
            )

            if combined > best_candidate_combined:
                best_candidate_combined = combined
                best_candidate_idx = cand_idx
                best_candidate_prcs = prcs
                best_candidate_epistasis = epistasis
                best_candidate_per_position = per_position
                best_candidate_coupled = candidate_coupled[cand_idx]
                best_candidate_positions = candidate_positions[cand_idx]

            if candidate_coupled[cand_idx]:
                proposed_coupled += 1
            else:
                proposed_single += 1

            del cand_emb

        # --- Metropolis acceptance ---
        delta = best_candidate_combined - current_combined
        accepted = delta > 0 or (
            temperature > 1e-10 and rng.random() < math.exp(delta / temperature)
        )

        if accepted:
            current_sequence = candidate_sequences[best_candidate_idx]
            # Update window cache for accepted candidate
            embedder.update_cache_for_accepted(
                best_candidate_idx,
                best_candidate_positions,
                window_new_embs,
            )
            # Re-stitch to get the current embedding (using updated cache)
            current_embedding = embedder.stitch_candidate(
                best_candidate_idx, window_new_embs
            )
            current_prcs = best_candidate_prcs
            current_epistasis = best_candidate_epistasis
            current_combined = best_candidate_combined
            current_per_position = best_candidate_per_position

            if best_candidate_coupled:
                accepted_coupled += 1
            else:
                accepted_single += 1

            if current_prcs > best_prcs:
                best_sequence = current_sequence
                best_prcs = current_prcs
                best_combined = current_combined

        temperature *= walk_config.cooling_rate

        # --- Diagnostic score ---
        diagnostic, lower_quartile, weakest_window = core.score_fixed_diagnostic(
            current_per_position
        )

        # --- Record trajectory ---
        trajectory.append(
            {
                "step": step,
                "prcs": current_prcs,
                "epistasis": current_epistasis,
                "candidate_epistasis": best_candidate_epistasis,
                "combined": current_combined,
                "diagnostic_score": diagnostic,
                "prcs_lower_quartile": lower_quartile,
                "prcs_weakest_window": weakest_window,
                "best_prcs": best_prcs,
                "lambda": coupling_lambda,
                "lambda_max": lambda_max,
                "coupling_pairs_tested": len(coupling_scores),
                "coupling_pairs_supported": len(supported_pairs),
                "coupling_group_count": len(groups),
                "coupling_map_jaccard": coupling_map_jaccard,
                "temperature": temperature,
                "accepted": accepted,
                "candidate_coupled": best_candidate_coupled,
                "heating_active": bool(
                    walk_config.heating_steps and step < walk_config.heating_steps
                ),
                "k_single_effective": effective_k,
                "effective_p_coupled": effective_p_coupled,
                "adaptive_stage": adaptive_stage,
                "adaptive_progress": adaptive_progress,
                "adaptive_plateau": adaptive_plateau,
                "delta": delta,
                "mutated_positions": ";".join(
                    str(p) for p in best_candidate_positions
                ),
                "sequence": current_sequence,
            }
        )

        # --- Logging ---
        if step > 0 and step % core.LOG_EVERY == 0:
            print(
                f"  [Walker {walker_id}] Step {step}: PRCS={current_prcs:.4f}, "
                f"epi={current_epistasis:.4f}, combined={current_combined:.4f}, "
                f"k={effective_k}, p_c={effective_p_coupled:.2f}, "
                f"stage={adaptive_stage}, progress={adaptive_progress:.2f}, "
                f"lambda={coupling_lambda:.3f}/{lambda_max:.3f}, T={temperature:.4f}"
            )

        # --- Checkpoint ---
        if (
            checkpoint_path is not None
            and step > 0
            and step % core.CHECKPOINT_EVERY == 0
        ):
            core.save_checkpoint(checkpoint_path, checkpoint_state(step))

    # Final checkpoint
    if checkpoint_path is not None and walk_config.max_steps > 0:
        core.save_checkpoint(
            checkpoint_path, checkpoint_state(walk_config.max_steps - 1)
        )

    # --- Summary ---
    single_rate = (
        100.0 * accepted_single / proposed_single if proposed_single else 0.0
    )
    coupled_rate = (
        100.0 * accepted_coupled / proposed_coupled if proposed_coupled else 0.0
    )
    print(
        f"  [Walker {walker_id}] Finished: final PRCS={current_prcs:.4f}, "
        f"best PRCS={best_prcs:.4f}, single acceptance={single_rate:.1f}%, "
        f"coupled acceptance={coupled_rate:.1f}%, "
        f"rejections={len(rejection_log)}"
    )

    return {
        "protein_name": protein_name,
        "walker_id": walker_id,
        "start_sequence": start_sequence,
        "start_prcs": start_prcs,
        "final_sequence": current_sequence,
        "final_prcs": current_prcs,
        "final_combined": current_combined,
        "best_sequence": best_sequence,
        "best_prcs": best_prcs,
        "best_combined": best_combined,
        "trajectory": trajectory,
        "acceptance_rate_single": single_rate,
        "acceptance_rate_coupled": coupled_rate,
        "n_allowed": n_allowed,
        "rejection_log": rejection_log,
    }
