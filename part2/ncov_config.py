#!/usr/bin/env python3
"""Shared configuration for the SARS-CoV-2 mutation box pipeline.

All constants needed across download, filtering, Nextclade translation,
ORF concatenation, mutation box definition, ESM2 target preparation,
and the evolution engine.

Reference genome: MN908947.3 (Wuhan-Hu-1), GenBank accession NC_045512.2
Length: 29,903 nucleotides

Translation is performed by Nextclade CLI (authoritative Nextstrain tool),
NOT by custom code. Nextclade handles the -1 ribosomal frameshift and
produces separate ORF1a and ORF1b protein sequences.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reference genome
# ---------------------------------------------------------------------------
REFERENCE_ID = "MN908947.3"
REFERENCE_LENGTH = 29903

# ---------------------------------------------------------------------------
# ORF table — 1-based inclusive nucleotide coordinates on MN908947.3
#
# These coordinates are for REFERENCE ONLY (used to compute expected AA
# lengths and concatenation offsets). Actual translation is done by Nextclade,
# which handles the -1 ribosomal frameshift at the slippery site correctly
# and produces separate ORF1a and ORF1b protein sequences.
#
# Coordinates sourced from the NCBI annotation of MN908947.3 / NC_045512.2.
# ---------------------------------------------------------------------------
ORFS = [
    # (name, nt_start_1based, nt_end_1based)
    ("ORF1a",  266,   13483),
    ("ORF1b",  13468, 21555),
    ("S",      21563, 25384),
    ("ORF3a",  25393, 26220),
    ("E",      26245, 26472),
    ("M",      26523, 27191),
    ("ORF6",   27202, 27387),
    ("ORF7a",  27394, 27759),
    ("ORF7b",  27756, 27887),
    ("ORF8",   27894, 28259),
    ("N",      28274, 29533),
    ("ORF9b",  28284, 28577),
]

# Concatenation order (same as ORFS list above)
CONCATENATION_ORDER = [name for name, _, _ in ORFS]

# Pre-compute ORF amino-acid lengths (EXCLUDING the stop codon)
# Each ORF's last codon is a stop in the reference; we drop it so the
# concatenated sequence contains only standard amino acids for ESM2.
#
# NOTE: These NCBI-coordinate-based values are FALLBACK estimates (~9,807 aa).
# The ACTUAL per-ORF lengths come from Nextclade's translation of the reference
# genome, which may differ slightly (e.g. 9,814 aa). When 04_concatenate_orfs_v2.py
# runs, it outputs orf_lengths.json with the authoritative Nextclade-derived
# lengths. ncov_config.py auto-loads that file at import time to override these values.
ORF_AA_LENGTHS_NCBI = {}
ORF_OFFSETS_NCBI = {}
_offset = 0
for _name, _start, _end in ORFS:
    _nt_len = _end - _start + 1
    _total_codons = _nt_len // 3
    _aa_len = _total_codons - 1  # exclude stop codon
    ORF_AA_LENGTHS_NCBI[_name] = _aa_len
    ORF_OFFSETS_NCBI[_name] = _offset
    _offset += _aa_len
TOTAL_AA_LENGTH_NCBI = _offset  # ~9,807

# Default to NCBI values; may be overridden by orf_lengths.json below
ORF_AA_LENGTHS = dict(ORF_AA_LENGTHS_NCBI)
ORF_OFFSETS = dict(ORF_OFFSETS_NCBI)
TOTAL_AA_LENGTH = TOTAL_AA_LENGTH_NCBI

# ---------------------------------------------------------------------------
# Auto-load actual ORF lengths from orf_lengths.json (output by step 04 v2)
# This overrides the NCBI-coordinate estimates with Nextclade-derived values.
# ---------------------------------------------------------------------------
import json as _json
import os as _os
from pathlib import Path as _Path

def _try_load_orf_lengths():
    """Try to load orf_lengths.json from common locations.

    Search order:
      1. $NCOV_DATA_DIR/orf_lengths.json (environment variable)
      2. data/orf_lengths.json (default data dir)
      3. data/data/orf_lengths.json (nested data dir, as on some clusters)
      4. <config_dir>/data/orf_lengths.json
      5. <config_dir>/orf_lengths.json
    """
    global ORF_AA_LENGTHS, ORF_OFFSETS, TOTAL_AA_LENGTH

    candidates = []
    env_dir = _os.environ.get("NCOV_DATA_DIR")
    if env_dir:
        candidates.append(_Path(env_dir) / "orf_lengths.json")
    candidates.extend([
        _Path("data") / "orf_lengths.json",
        _Path("data/data") / "orf_lengths.json",
        _Path(__file__).parent / "data" / "orf_lengths.json",
        _Path(__file__).parent / "data" / "data" / "orf_lengths.json",
        _Path(__file__).parent / "orf_lengths.json",
    ])

    for path in candidates:
        if path.exists():
            try:
                with open(path) as f:
                    data = _json.load(f)
                new_lengths = {}
                new_offsets = {}
                offset = 0
                for name in CONCATENATION_ORDER:
                    orf_info = data.get("orfs", {}).get(name)
                    if orf_info and "aa_length" in orf_info:
                        length = orf_info["aa_length"]
                    else:
                        # Fall back to NCBI estimate for this ORF
                        length = ORF_AA_LENGTHS_NCBI.get(name, 0)
                    new_lengths[name] = length
                    new_offsets[name] = offset
                    offset += length
                ORF_AA_LENGTHS = new_lengths
                ORF_OFFSETS = new_offsets
                TOTAL_AA_LENGTH = offset
                print(f"  [config] Loaded ORF lengths from {path} "
                      f"(total: {TOTAL_AA_LENGTH} aa)")
                return True
            except Exception as e:
                print(f"  [config] WARNING: Failed to load {path}: {e}")
    return False

_try_load_orf_lengths()

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# Nextclade settings (translation via CLI, not custom code)
# ---------------------------------------------------------------------------
NEXTCLADE_DATASET = "nextstrain/sars-cov-2/wuhan-hu-1/orfs"
NEXTCLADE_CDS = [
    "ORF1a", "ORF1b", "S", "ORF3a", "E", "M",
    "ORF6", "ORF7a", "ORF7b", "ORF8", "N", "ORF9b",
]
# Nextclade output directory structure:
#   <output_dir>/translated_<CDS>.fasta  for each CDS

# ---------------------------------------------------------------------------
# ESM2 model settings
# ---------------------------------------------------------------------------
ESM2_MODEL = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
ESM2_EMBEDDING_DIM = 1280
ESM2_MAX_TOKENS = 1022  # max input length (excluding BOS/EOS tokens)

# ---------------------------------------------------------------------------
# Window parameters — EXACTLY per project document example:
#   Window 1: positions 1-900
#   Window 2: positions 701-1600
#   Window 3: positions 1401-2300
# => window_size=900, step=700, overlap=200
# ---------------------------------------------------------------------------
WINDOW_SIZE = 900
WINDOW_STEP = 700
WINDOW_OVERLAP = 200

# ---------------------------------------------------------------------------
# Mutation box parameters
# ---------------------------------------------------------------------------
GAUSSIAN_SIGMA = 5          # sigma for bell-curve context extension
GAUSSIAN_THRESHOLD = 0.05   # include position if exp(-d^2/(2*sigma^2)) > threshold
                             # => extends ~2*sigma = ~10 aa on each side

# ---------------------------------------------------------------------------
# Date range filter — Wuhan (Dec 2019) to Omicron BA.2 dominance (Mar 2022)
# ---------------------------------------------------------------------------
DATE_MIN = "2019-12-01"
DATE_MAX = "2022-03-31"

# ---------------------------------------------------------------------------
# Subsampling parameters
# ---------------------------------------------------------------------------
TARGET_SAMPLE_SIZE = 4000   # total target number of strains
PER_LINEAGE_TARGET = 300    # target sequences per Pango lineage
MIN_PER_LINEAGE = 50        # minimum to include a lineage at all

# ---------------------------------------------------------------------------
# Nextstrain open data URLs (GenBank-sourced, publicly available)
# ---------------------------------------------------------------------------
NEXTSTRAIN_BASE = "https://data.nextstrain.org/files/ncov/open"

URLS_FULL = {
    "metadata": f"{NEXTSTRAIN_BASE}/metadata.tsv.zst",
    "aligned":  f"{NEXTSTRAIN_BASE}/aligned.fasta.zst",
}

URLS_SUBSAMPLED = {
    "metadata": f"{NEXTSTRAIN_BASE}/global/metadata.tsv.xz",
    "aligned":  f"{NEXTSTRAIN_BASE}/global/aligned.fasta.xz",
}

# ---------------------------------------------------------------------------
# Evolution algorithm parameters (from existing corrected_adaptive_engine.py)
# ---------------------------------------------------------------------------
INITIAL_TEMPERATURE = 0.30
FINAL_TEMPERATURE = 0.005
PROPOSALS_PER_STEP = 40
COUPLING_RECOMPUTE_EVERY = 150
COUPLING_THRESHOLD = 0.01
COUPLING_ROBUST_Z = 2.5
COUPLING_GROUP_MIN = 2
COUPLING_GROUP_MAX = 4
LAMBDA_COUPLING_SCALE = 5.0
CHECKPOINT_EVERY = 200
LOG_EVERY = 50

# ---------------------------------------------------------------------------
# Helper: build the list of window (start, end) tuples for a given seq length
# ---------------------------------------------------------------------------
def get_windows(seq_len: int, window_size: int = WINDOW_SIZE,
                step: int = WINDOW_STEP) -> list[tuple[int, int]]:
    """Return list of (start, end) 0-based half-open windows covering seq_len."""
    windows = []
    pos = 0
    while pos < seq_len:
        end = min(pos + window_size, seq_len)
        windows.append((pos, end))
        if end >= seq_len:
            break
        pos += step
    return windows


def get_window_index_for_position(position: int, windows: list[tuple[int, int]]) -> list[int]:
    """Return indices of all windows that contain the given 0-based position."""
    return [i for i, (s, e) in enumerate(windows) if s <= position < e]


def positions_to_windows(positions: list[int], windows: list[tuple[int, int]]) -> set[int]:
    """Return the set of window indices affected by any of the given positions."""
    affected = set()
    for p in positions:
        affected.update(get_window_index_for_position(p, windows))
    return affected


# ---------------------------------------------------------------------------
# Helper: triangular (Bartlett) weight for a position within a window
# ---------------------------------------------------------------------------
def triangular_weight(local_index: int, window_size: int) -> float:
    """Bartlett window weight: 1.0 at center, 0.0 at edges, linear in between."""
    if window_size <= 1:
        return 1.0
    center = (window_size - 1) / 2.0
    return 1.0 - abs(local_index - center) / center


if __name__ == "__main__":
    print(f"Reference: {REFERENCE_ID} ({REFERENCE_LENGTH} nt)")
    print(f"Total concatenated AA length: {TOTAL_AA_LENGTH}")
    print()
    print("ORF table:")
    for name, start, end in ORFS:
        aa_len = ORF_AA_LENGTHS[name]
        offset = ORF_OFFSETS[name]
        print(f"  {name:8s}  nt {start:>6d}-{end:>6d}  "
              f"aa {aa_len:>5d}  concat offset {offset:>6d}")
    print()
    windows = get_windows(TOTAL_AA_LENGTH)
    print(f"Windows for full proteome ({TOTAL_AA_LENGTH} aa):")
    for i, (s, e) in enumerate(windows):
        print(f"  Window {i+1:2d}: positions {s+1:>5d}-{e:>5d}  (size {e-s})")
    print(f"  Total windows: {len(windows)}")
    print()
    print(f"Nextclade dataset: {NEXTCLADE_DATASET}")
    print(f"Nextclade CDS: {', '.join(NEXTCLADE_CDS)}")
    print(f"Date filter: {DATE_MIN} to {DATE_MAX}")
    print(f"Target sample size: {TARGET_SAMPLE_SIZE} ({PER_LINEAGE_TARGET} per lineage)")
