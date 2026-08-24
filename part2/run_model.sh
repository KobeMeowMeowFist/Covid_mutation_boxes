#!/bin/bash
# run_model.sh — Run the windowed adaptive walk: Wuhan-Hu-1 -> BA.2
#
# Three steps:
#   1. prepare_strain_target.py  — auto-select BA.2 target, compute embeddings
#   2. run_strain_adaptive.py    — 2 walkers x 1000 steps from Wuhan
#   3. evaluate_intermediates.py — Hamming distance curves vs Alpha/Delta/BA.1
#
# Usage:
#   ./run_model.sh
#   ./run_model.sh --steps 1000 --walkers 2 --seed 42 --device cuda
#   ./run_model.sh --data-dir /path/to/data --outdir /path/to/output

set -euo pipefail

# ── Paths (adjust for your cluster) ──────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/evolution_output"
TARGET_DIR="${SCRIPT_DIR}/target_artifacts"
EVAL_DIR="${OUTPUT_DIR}/evaluation"

# ── Conda ────────────────────────────────────────────────────────────
CONDA_BASE="/scratch/fanm01/dms_predict/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate esmwalk
fi

PYTHON="${CONDA_BASE}/envs/esmwalk/bin/python"
if [ ! -f "$PYTHON" ]; then
    # Fallback: use local user env if present (some systems keep envs/ at project root)
    if [ -x "/scratch/fanm01/dms_predict/envs/esmwalk/bin/python" ]; then
        PYTHON="/scratch/fanm01/dms_predict/envs/esmwalk/bin/python"
    else
        PYTHON="python"
    fi
fi

# ── Parse args ───────────────────────────────────────────────────────
STEPS=1000
WALKERS=2
SEED=42
DEVICE="cuda"
INTERMEDIATES="B.1.1.7,B.1.617.2,BA.1"
TARGET_LINEAGE="BA.2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)          STEPS="$2"; shift 2 ;;
        --walkers)        WALKERS="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        --outdir)         OUTPUT_DIR="$2"; shift 2 ;;
        --target-dir)     TARGET_DIR="$2"; shift 2 ;;
        --intermediates)  INTERMEDIATES="$2"; shift 2 ;;
        --target-lineage) TARGET_LINEAGE="$2"; shift 2 ;;
        --help)
            echo "Usage: ./run_model.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --steps N          Steps per walker (default: 1000)"
            echo "  --walkers N        Number of walkers (default: 2)"
            echo "  --seed N           Random seed (default: 42)"
            echo "  --device STR       Torch device (default: cuda)"
            echo "  --data-dir PATH    Data directory (default: ./data)"
            echo "  --outdir PATH      Output directory (default: ./evolution_output)"
            echo "  --target-dir PATH  Target artifact directory (default: ./target_artifacts)"
            echo "  --intermediates STR  Comma-separated Pango lineages (default: B.1.1.7,B.1.617.2,BA.1)"
            echo "  --target-lineage STR  Target lineage (default: BA.2)"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

EVAL_DIR="${OUTPUT_DIR}/evaluation"

echo "============================================================"
echo "  Windowed Adaptive Walk: Wuhan-Hu-1 -> ${TARGET_LINEAGE}"
echo "============================================================"
echo "  Data:     ${DATA_DIR}"
echo "  Target:   ${TARGET_DIR}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Steps:    ${STEPS} | Walkers: ${WALKERS} | Seed: ${SEED}"
echo "  Device:   ${DEVICE}"
echo "  Intermediates: ${INTERMEDIATES}"
echo "============================================================"
echo ""

# ── Check data files ─────────────────────────────────────────────────
echo ">>> Checking data files..."
for f in aa_sequences.fasta wuhan_aa.fasta allowed_positions.npy; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "  ERROR: Missing ${DATA_DIR}/${f}"
        echo "  Run the data pipeline first: ./run_data.sh"
        exit 1
    fi
    echo "  OK: ${f}"
done

# Check metadata
META_FOUND=0
for f in filtered_metadata.tsv metadata.tsv; do
    if [ -f "${DATA_DIR}/${f}" ]; then
        echo "  OK: ${f}"
        META_FOUND=1
        break
    fi
done
if [ "$META_FOUND" -eq 0 ]; then
    echo "  WARNING: No metadata file found (filtered_metadata.tsv or metadata.tsv)"
    echo "           Target strain auto-selection will fail."
fi

# ── Check original ESM_EVOLUTION files ───────────────────────────────
echo ""
echo ">>> Checking ESM_EVOLUTION engine files..."
for f in corrected_adaptive_engine.py; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "  ERROR: Missing ${SCRIPT_DIR}/${f}"
        echo "  This file must be in the same directory as the extension files."
        exit 1
    fi
    echo "  OK: ${f}"
done

# ── Check extension files ────────────────────────────────────────────
for f in windowed_engine.py windowed_walk.py prepare_strain_target.py \
         run_strain_adaptive.py evaluate_intermediates.py ncov_config.py; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        echo "  ERROR: Missing ${SCRIPT_DIR}/${f}"
        exit 1
    fi
    echo "  OK: ${f}"
done
echo ""

# ── Step 1: Prepare target + Wuhan start embeddings ──────────────────
echo ">>> [1/3] Prepare target (${TARGET_LINEAGE}) + Wuhan start embeddings"
echo "---"
mkdir -p "$TARGET_DIR"
$PYTHON "${SCRIPT_DIR}/prepare_strain_target.py" \
    --data-dir "$DATA_DIR" \
    --outdir "$TARGET_DIR" \
    --target-lineage "$TARGET_LINEAGE" \
    --device "$DEVICE"
echo ""

# ── Step 2: Run evolution ────────────────────────────────────────────
echo ">>> [2/3] Run evolution (${WALKERS} walkers x ${STEPS} steps)"
echo "---"
mkdir -p "$OUTPUT_DIR"
$PYTHON "${SCRIPT_DIR}/run_strain_adaptive.py" \
    --artifact-dir "$TARGET_DIR" \
    --data-dir "$DATA_DIR" \
    --outdir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --walkers "$WALKERS" \
    --total-steps "$STEPS" \
    --device "$DEVICE"
echo ""

# ── Step 3: Evaluate against intermediates ───────────────────────────
echo ">>> [3/3] Evaluate against intermediates (${INTERMEDIATES})"
echo "---"
mkdir -p "$EVAL_DIR"
$PYTHON "${SCRIPT_DIR}/evaluate_intermediates.py" \
    --evolution-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --outdir "$EVAL_DIR" \
    --intermediates "$INTERMEDIATES" \
    --target-lineage "$TARGET_LINEAGE"
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  Complete!"
echo "============================================================"
echo ""
echo "  Evolution output:  ${OUTPUT_DIR}"
echo "    final_sequences/       - Final and best evolved sequences (FASTA)"
echo "    walker_trajectories/   - Per-step trajectory (CSV with sequence)"
echo "    checkpoints/           - Resumable checkpoints"
echo "    final_walker_summary.csv - PRCS summary per walker"
echo "    run_metadata.json      - Run configuration"
echo "    rejection_log.csv      - Epistatic pair rule violations"
echo ""
echo "  Evaluation output: ${EVAL_DIR}"
echo "    hamming_distance_curves.png - Hamming distance to Wuhan/Alpha/Delta/BA.1/BA.2"
echo "    prcs_progress.png           - PRCS to target over steps"
echo "    mutation_progress.png       - Mutations by ORF over steps"
echo "    intermediate_comparison.tsv - Per-step Hamming distances"
echo "    evaluation_report.md        - Summary report"
echo ""
echo "============================================================"
