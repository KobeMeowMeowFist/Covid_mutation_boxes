#!/usr/bin/env python3
"""Windowed ESM2 engine for full-proteome adaptive walk.

Extends corrected_adaptive_engine.py to handle sequences longer than ESM2's
1022-token limit by dividing the sequence into overlapping windows, embedding
each window separately, and stitching the per-residue representations with
triangular (Bartlett) weights.

All mutations are constrained to allowed positions (mutation boxes).
Coupled mutations respect the epistatic pair rule: both positions must be
inside allowed boxes.

This module imports pure functions from corrected_adaptive_engine as `core`
and does NOT modify the original file.
"""

from __future__ import annotations

from itertools import combinations
import math
import random
from typing import Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn.functional as F

import corrected_adaptive_engine as core
import ncov_config as config


# ---------------------------------------------------------------------------
# Low-memory ESM2 loading (bypass CPU RAM bottleneck)
# ---------------------------------------------------------------------------

def load_esm2_lowmem(device: str):
    """Load ESM2 model with minimal CPU memory.

    The standard esm.pretrained.esm2_t33_650M_UR50D() loads model weights to
    CPU first (map_location="cpu"), which requires ~5 GB peak CPU RAM (state
    dict + model object). On systems with tight cgroup memory limits (~4.5 GB),
    this causes OOM.

    This function monkey-patches esm.pretrained.load_hub_workaround to load
    weights directly to the target device (e.g. cuda), reducing peak CPU memory
    to ~2.6 GB (just the model object, not the state dict).

    Args:
        device: "cuda" or "cuda:0" etc. Must be a CUDA device.

    Returns:
        (model, alphabet, batch_converter) — same as core.load_esm2()
    """
    import esm.pretrained as ep
    import gc
    from pathlib import Path

    if not device.startswith("cuda"):
        raise ValueError(
            f"load_esm2_lowmem requires a CUDA device, got '{device}'. "
            f"Use core.load_esm2() for CPU."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check that PyTorch was installed with "
            "CUDA support and that a GPU is visible.\n"
            "Verify with: python -c 'import torch; print(torch.cuda.is_available())'"
        )

    print(f"Loading ESM2 (low-mem mode, weights → {device})...")

    # Save original function
    _original_load = ep.load_hub_workaround

    # Patch: load state dict directly to target device instead of CPU
    def _patched_load(url):
        import urllib.error
        try:
            data = torch.hub.load_state_dict_from_url(
                url, progress=False, map_location=device
            )
        except RuntimeError:
            # PyTorch version fallback (same as original but with map_location)
            fn = Path(url).name
            data = torch.load(
                f"{torch.hub.get_dir()}/checkpoints/{fn}",
                map_location=device,
            )
        except urllib.error.HTTPError as e:
            raise Exception(
                f"Could not load {url}, check if you specified a correct model name?"
            )
        return data

    ep.load_hub_workaround = _patched_load

    try:
        model, alphabet = ep.esm2_t33_650M_UR50D()
    finally:
        # Always restore original, even if loading fails
        ep.load_hub_workaround = _original_load

    # Model was created on CPU by _load_model_and_alphabet_core_v2,
    # state dict was on GPU, load_state_dict copied GPU→CPU.
    # Now move model to GPU and free CPU memory.
    model = model.eval().to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded on {device} (low-mem mode)")
    return model, alphabet, batch_converter


# ---------------------------------------------------------------------------
# WindowedEmbedder
# ---------------------------------------------------------------------------

class WindowedEmbedder:
    """Manages windowed ESM2 embedding with triangular stitching and incremental cache.

    The full sequence is divided into overlapping windows (default: 900 aa,
    step 700, overlap 200). Each window is embedded separately by ESM2, and
    the per-residue representations are stitched together using triangular
    (Bartlett) weights: center positions get weight 1.0, edge positions get
    weight ~0.0. Positions in the overlap region receive a weighted average
    of two windows.

    When a mutation is proposed, only the affected windows are recomputed;
    unaffected windows use cached embeddings. This keeps each step fast.
    """

    def __init__(
        self,
        model,
        alphabet,
        batch_converter,
        device: str,
        seq_len: int,
        window_size: int = config.WINDOW_SIZE,
        step: int = config.WINDOW_STEP,
    ):
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = batch_converter
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.window_size = window_size
        self.step = step

        # Pre-compute windows: list of (start, end) 0-based half-open
        self.windows = config.get_windows(seq_len, window_size, step)
        self.n_windows = len(self.windows)

        # Pre-compute triangular weights for each window
        self.weights = []
        for start, end in self.windows:
            wsize = end - start
            local_indices = np.arange(wsize)
            w = np.array(
                [config.triangular_weight(int(i), wsize) for i in local_indices],
                dtype=np.float32,
            )
            self.weights.append(w)

        # Pre-convert weights to tensors on device
        self.weight_tensors = [
            torch.tensor(w, dtype=torch.float32, device=self.device)
            for w in self.weights
        ]

        # Cache for per-window embeddings (populated by compute_full_embedding)
        self.window_cache: list[torch.Tensor] | None = None

        print(
            f"  [WindowedEmbedder] seq_len={seq_len}, {self.n_windows} windows, "
            f"window_size={window_size}, step={step}, overlap={window_size - step}"
        )

    @torch.no_grad()
    def compute_full_embedding(self, sequence: str) -> torch.Tensor:
        """Compute the stitched per-residue embedding for the full sequence.

        Embeds each window separately via ESM2, then stitches with triangular
        weights. Caches per-window embeddings for incremental updates.

        Returns:
            tensor of shape (seq_len, 1280)
        """
        # Embed each window (batched in groups of 4, same-length only)
        # Windows 0-12 are 900 aa, window 13 is 703 aa — must not mix lengths
        window_seqs = [sequence[start:end] for start, end in self.windows]
        window_embeddings = []
        batch_size = 4
        i = 0
        while i < len(window_seqs):
            # Collect consecutive same-length windows into one batch
            batch = [window_seqs[i]]
            j = i + 1
            while (j < len(window_seqs)
                   and len(window_seqs[j]) == len(window_seqs[i])
                   and len(batch) < batch_size):
                batch.append(window_seqs[j])
                j += 1
            embs = core.get_per_residue_embeddings_batch(
                batch,
                self.model,
                self.alphabet,
                self.batch_converter,
                str(self.device),
                max_batch=len(batch),
            )
            window_embeddings.extend(embs)
            i = j

        # Cache
        self.window_cache = window_embeddings

        # Stitch with triangular weights
        return self._stitch(window_embeddings)

    @torch.no_grad()
    def compute_candidate_window_embeddings(
        self,
        candidates: list[str],
        candidate_positions: list[list[int]],
    ) -> dict[int, dict[int, torch.Tensor]]:
        """Compute per-window embeddings for candidates, batched by window.

        For each candidate, only the windows containing mutated positions are
        recomputed. Unaffected windows will use the cached embeddings during
        stitching.

        Args:
            candidates: list of candidate sequences
            candidate_positions: list of mutated position lists (one per candidate)

        Returns:
            dict: {window_idx: {candidate_idx: embedding_tensor}}
        """
        # For each window, collect candidate subsequences
        window_seqs: dict[int, list[str]] = {}
        window_cand_indices: dict[int, list[int]] = {}

        for cand_idx, (candidate, positions) in enumerate(
            zip(candidates, candidate_positions)
        ):
            affected = config.positions_to_windows(positions, self.windows)
            for wi in affected:
                start, end = self.windows[wi]
                if wi not in window_seqs:
                    window_seqs[wi] = []
                    window_cand_indices[wi] = []
                window_seqs[wi].append(candidate[start:end])
                window_cand_indices[wi].append(cand_idx)

        # Batch through ESM2 for each window
        window_new_embs: dict[int, dict[int, torch.Tensor]] = {}
        batch_size = 4
        for wi, seqs in window_seqs.items():
            embs = core.get_per_residue_embeddings_batch(
                seqs,
                self.model,
                self.alphabet,
                self.batch_converter,
                str(self.device),
                max_batch=batch_size,
            )
            window_new_embs[wi] = {}
            for j, cand_idx in enumerate(window_cand_indices[wi]):
                window_new_embs[wi][cand_idx] = embs[j]

        return window_new_embs

    @torch.no_grad()
    def stitch_candidate(
        self,
        cand_idx: int,
        window_new_embs: dict[int, dict[int, torch.Tensor]],
    ) -> torch.Tensor:
        """Stitch embedding for one candidate using cached + new window embeddings.

        Args:
            cand_idx: index into the candidate list
            window_new_embs: output from compute_candidate_window_embeddings

        Returns:
            tensor of shape (seq_len, 1280)
        """
        if self.window_cache is None:
            raise RuntimeError("window_cache is None; call compute_full_embedding first")

        stitched = torch.zeros(
            self.seq_len, config.ESM2_EMBEDDING_DIM, device=self.device
        )
        total_weight = torch.zeros(self.seq_len, device=self.device)

        for i, (start, end) in enumerate(self.windows):
            weights = self.weight_tensors[i]
            if i in window_new_embs and cand_idx in window_new_embs[i]:
                emb = window_new_embs[i][cand_idx]
            else:
                emb = self.window_cache[i]
            stitched[start:end] += emb * weights.unsqueeze(1)
            total_weight[start:end] += weights

        stitched = stitched / total_weight.unsqueeze(1).clamp(min=1e-8)
        return stitched

    def _stitch(self, window_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Stitch per-window embeddings into a full-length representation."""
        stitched = torch.zeros(
            self.seq_len, config.ESM2_EMBEDDING_DIM, device=self.device
        )
        total_weight = torch.zeros(self.seq_len, device=self.device)

        for i, (start, end) in enumerate(self.windows):
            weights = self.weight_tensors[i]
            emb = window_embeddings[i]
            stitched[start:end] += emb * weights.unsqueeze(1)
            total_weight[start:end] += weights

        stitched = stitched / total_weight.unsqueeze(1).clamp(min=1e-8)
        return stitched

    def update_cache_for_accepted(
        self,
        cand_idx: int,
        candidate_positions: list[int],
        window_new_embs: dict[int, dict[int, torch.Tensor]],
    ) -> None:
        """Update the window cache after a candidate is accepted.

        Replaces cached embeddings for affected windows with the new ones.
        """
        if self.window_cache is None:
            return
        affected = config.positions_to_windows(candidate_positions, self.windows)
        for wi in affected:
            if wi in window_new_embs and cand_idx in window_new_embs[wi]:
                self.window_cache[wi] = window_new_embs[wi][cand_idx]


# ---------------------------------------------------------------------------
# Constrained proposal functions
# ---------------------------------------------------------------------------

def propose_single_guided_constrained(
    sequence: str,
    mutation_count: int,
    per_position_prcs: np.ndarray,
    allowed_positions: np.ndarray,
    rng: random.Random,
    guided_probability: float = core.P_GUIDED_ADAPTIVE,
) -> tuple[str, list[int]]:
    """Propose single mutations only at allowed positions (inside mutation boxes).

    Same logic as core.propose_single_guided but restricted to allowed_positions.
    Positions outside mutation boxes are never selected.
    """
    allowed_list = np.where(allowed_positions)[0]
    n_allowed = len(allowed_list)
    if n_allowed == 0:
        raise ValueError("No allowed positions to mutate")

    mutation_count = min(max(1, mutation_count), n_allowed)

    # Per-position PRCS at allowed positions only
    allowed_prcs = per_position_prcs[allowed_list]

    # Worst positions among allowed
    worst_count = max(mutation_count, int(core.PRCS_WORST_FRACTION * n_allowed))
    worst_indices = np.argsort(allowed_prcs)[:worst_count]
    worst_positions = allowed_list[worst_indices]

    selected: list[int] = []
    used: set[int] = set()
    for _ in range(mutation_count):
        guided_available = [
            int(p) for p in worst_positions if int(p) not in used
        ]
        all_available = [
            int(p) for p in allowed_list if int(p) not in used
        ]
        if rng.random() < guided_probability and guided_available:
            position = rng.choice(guided_available)
        elif all_available:
            position = rng.choice(all_available)
        else:
            break
        selected.append(position)
        used.add(position)

    replacements = [
        rng.choice(
            [aa for aa in core.AMINO_ACIDS if aa != sequence[p]]
        )
        for p in selected
    ]
    return core.mutate_positions(sequence, selected, replacements), selected


def propose_coupled_constrained(
    sequence: str,
    groups: list[list[int]],
    allowed_positions_set: set[int],
    allowed_positions: np.ndarray,
    rng: random.Random,
    rejection_log: list[dict] | None = None,
) -> tuple[str, list[int]]:
    """Propose coupled mutations only at allowed positions.

    Enforces the epistatic pair rule:
    - Both positions inside boxes: propose (groups are pre-filtered)
    - Any position outside boxes: reject + log (defensive check)
    """
    if not groups:
        return propose_single_guided_constrained(
            sequence, 1, np.zeros(len(sequence)), allowed_positions, rng, 0.0
        )

    positions = list(rng.choice(groups))

    # Defensive check: all positions must be in allowed_positions_set
    violations = [p for p in positions if int(p) not in allowed_positions_set]
    if violations:
        if rejection_log is not None:
            rejection_log.append(
                {
                    "reason": "coupled_position_outside_box",
                    "positions": [int(p) for p in positions],
                    "violations": violations,
                }
            )
        # Fall back to single constrained mutation
        return propose_single_guided_constrained(
            sequence, 1, np.zeros(len(sequence)), allowed_positions, rng, 0.0
        )

    replacements = [
        rng.choice([aa for aa in core.AMINO_ACIDS if aa != sequence[p]])
        for p in positions
    ]
    return core.mutate_positions(sequence, positions, replacements), positions


# ---------------------------------------------------------------------------
# Windowed coupling map
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_windowed_coupling_map(
    sequence: str,
    model,
    alphabet,
    batch_converter,
    device: str,
    allowed_positions: np.ndarray,
    windows: list[tuple[int, int]],
    sample_count: int,
    rng: random.Random,
) -> list[tuple[int, int, float]]:
    """Compute epistatic coupling map using within-window pairs only.

    For each window, samples pairs where both positions are in that window AND
    both in allowed_positions. Computes masked marginals within the window
    subsequence (not the full sequence), so ESM2 sees the local context.

    Cross-window pairs (positions that never co-occur in a single window) are
    NOT sampled. This is the deferred item — the user wants to discuss handling
    later.

    Returns:
        list of (left_global, right_global, coupling_score), sorted by score descending
    """
    allowed_set = set(int(p) for p in np.where(allowed_positions)[0])

    # For each window, find allowed positions within that window
    window_allowed: dict[int, list[int]] = {}
    for wi, (start, end) in enumerate(windows):
        positions = [p for p in range(start, end) if p in allowed_set]
        if len(positions) >= 2:
            window_allowed[wi] = positions

    if not window_allowed:
        return []

    # Distribute sample_count across windows proportionally
    total_allowed = sum(len(v) for v in window_allowed.values())
    window_samples: dict[int, int] = {}
    remaining = sample_count
    window_keys = sorted(window_allowed.keys())

    for wi in window_keys:
        n = len(window_allowed[wi])
        n_pairs = n * (n - 1) // 2
        alloc = max(1, int(sample_count * n / total_allowed))
        alloc = min(alloc, n_pairs, remaining)
        window_samples[wi] = alloc
        remaining -= alloc

    # Redistribute any remaining samples
    while remaining > 0:
        allocated = False
        for wi in window_keys:
            n = len(window_allowed[wi])
            n_pairs = n * (n - 1) // 2
            if window_samples[wi] < n_pairs:
                window_samples[wi] += 1
                remaining -= 1
                allocated = True
                if remaining <= 0:
                    break
        if not allocated:
            break

    scores: list[tuple[int, int, float]] = []

    for wi, (start, end) in enumerate(windows):
        n_samples = window_samples.get(wi, 0)
        if n_samples == 0:
            continue

        positions = window_allowed[wi]  # global positions
        n_pos = len(positions)
        if n_pos < 2:
            continue

        # All possible pairs (by index into positions list)
        all_pairs = list(combinations(range(n_pos), 2))
        if len(all_pairs) <= n_samples:
            sampled = all_pairs
        else:
            sampled = rng.sample(all_pairs, n_samples)

        # Window subsequence
        window_seq = sequence[start:end]

        # Local positions (0-based within window)
        local_positions = [p - start for p in positions]

        # Unique positions needed for single marginals
        unique_local_indices = sorted(set(idx for pair in sampled for idx in pair))
        unique_local_positions = [local_positions[idx] for idx in unique_local_indices]

        # Compute single masked marginals within window
        single_marginals = core.compute_single_masked_marginals(
            window_seq,
            model,
            alphabet,
            batch_converter,
            device,
            unique_local_positions,
        )

        # For each pair, compute double marginals and JS divergence
        for left_idx, right_idx in sampled:
            left_local = local_positions[left_idx]
            right_local = local_positions[right_idx]
            left_global = positions[left_idx]
            right_global = positions[right_idx]

            single_left = single_marginals[left_local]
            single_right = single_marginals[right_local]

            double_left, double_right = core.compute_double_masked_marginals(
                window_seq,
                model,
                alphabet,
                batch_converter,
                device,
                left_local,
                right_local,
            )

            coupling = (
                jensenshannon(single_left, double_left)
                + jensenshannon(single_right, double_right)
            ) / 2

            if not np.isnan(coupling):
                scores.append((left_global, right_global, float(coupling)))

    scores.sort(key=lambda item: item[2], reverse=True)
    return scores


def build_constrained_coupling_groups(
    coupling_scores: list[tuple[int, int, float]],
    length: int,
    allowed_positions_set: set[int],
    n_groups: int,
    group_size_min: int,
    group_size_max: int,
    threshold: float,
) -> tuple[list[list[int]], float]:
    """Build coupling groups, filtered to only contain allowed positions.

    Calls core.build_coupling_groups to get raw groups, then filters to keep
    only groups where ALL positions are in allowed_positions_set.

    This enforces the epistatic pair rule by construction:
    - Both positions inside boxes: group is kept
    - Any position outside boxes: group is removed
    """
    raw_groups, effective_threshold = core.build_coupling_groups(
        coupling_scores,
        length,
        n_groups,
        group_size_min,
        group_size_max,
        threshold,
    )

    filtered_groups = [
        group
        for group in raw_groups
        if all(int(p) in allowed_positions_set for p in group)
    ]

    return filtered_groups, effective_threshold


# ---------------------------------------------------------------------------
# WalkConfig creation (bypass 1022 limit)
# ---------------------------------------------------------------------------

def make_strain_walk_config(
    seq_len: int,
    n_allowed: int,
    max_steps: int,
) -> core.WalkConfig:
    """Create WalkConfig directly, bypassing for_sequence() 1022 check.

    The original for_sequence() raises ValueError for length > 1022.
    We create the config directly with the same logic but using n_allowed
    for k_single calculation (exploration rate based on allowed region,
    not total length).
    """
    k_single = max(1, round(n_allowed / 50))
    group_min = min(core.COUPLING_GROUP_MIN, n_allowed)
    group_max = max(group_min, min(core.COUPLING_GROUP_MAX, n_allowed))
    average_group = (group_min + group_max) / 2.0
    n_groups = max(15, int(0.7 * n_allowed / average_group))
    p_coupled = min(0.80, 0.30 + n_allowed / 1000.0)
    cooling_rate = (
        core.FINAL_TEMPERATURE / core.INITIAL_TEMPERATURE
    ) ** (1.0 / max_steps)

    # Big protein heating (n_allowed >= 300, which it is at 2535)
    heating_steps = min(core.BIG_HEATING_STEPS, max_steps // 4)
    heating_k_min = min(core.BIG_HEATING_K_MIN, n_allowed)
    heating_k_max = min(core.BIG_HEATING_K_MAX, n_allowed)

    return core.WalkConfig(
        length=seq_len,
        max_steps=max_steps,
        k_single=k_single,
        p_coupled=p_coupled,
        n_groups=n_groups,
        group_size_min=group_min,
        group_size_max=group_max,
        cooling_rate=cooling_rate,
        heating_steps=heating_steps,
        heating_k_min=heating_k_min,
        heating_k_max=heating_k_max,
        heating_probability=core.BIG_HEATING_PROBABILITY,
    )


# ---------------------------------------------------------------------------
# Gap handling helper
# ---------------------------------------------------------------------------

def fill_gaps_with_reference(sequence: str, reference: str) -> str:
    """Replace non-standard characters in sequence with reference amino acids.

    ESM2 cannot process '-', 'X', '*', 'B', 'Z', 'U', 'O' or other non-standard
    characters. These represent gaps, unknowns, stop codons, or ambiguous AAs.
    Filling with the reference amino acid is reasonable since the walk starts
    from the reference and can only mutate to standard AAs.
    """
    if len(sequence) != len(reference):
        raise ValueError(
            f"Length mismatch: sequence={len(sequence)}, reference={len(reference)}"
        )
    valid_aas = set(core.AMINO_ACIDS)
    filled = list(sequence)
    n_filled = 0
    for i, aa in enumerate(sequence):
        if aa not in valid_aas:
            filled[i] = reference[i]
            n_filled += 1
    if n_filled > 0:
        print(f"  Filled {n_filled} non-standard characters with Wuhan reference")
    return "".join(filled)


def validate_sequence_for_esm(sequence: str) -> None:
    """Check that a sequence contains only standard amino acids (no gaps/X/*)."""
    invalid = sorted(set(sequence) - set(core.AMINO_ACIDS))
    if invalid:
        raise ValueError(
            f"Sequence contains invalid characters for ESM2: {invalid}. "
            f"Use fill_gaps_with_reference() first."
        )
