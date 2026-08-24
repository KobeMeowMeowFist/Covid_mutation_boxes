#!/bin/bash
#SBATCH --job-name=ncov_evolution

#SBATCH --partition=scavenger

# NOTE: Do NOT hardcode a single node here. Earlier versions forced
# H100 via --nodelist=g016 which prevents choosing A5000 for short-tests.
# By default we leave node selection to the scheduler or use
# find_best_gpu.sh to pick an appropriate node (prefer A5000 on
# scavenger). If you must pin a node, uncomment and set --nodelist.
# SBATCH --nodelist=g016  # disabled: do not hardcode node here; use find_best_gpu.sh or sbatch args

# Request one GPU. When submitting short-tests, pass `--gres=gpu:a5000:1`
# on the sbatch command line to prefer A5000; command-line args override
# the #SBATCH headers.
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --time=24:00:00

#SBATCH --qos=standard

#SBATCH --output=ncov_evolution_%j.out

#SBATCH --error=ncov_evolution_%j.err
## ── Curta (FU Berlin) GPU 规则要点 ──────────────────────────────────
## 1. GPU 作业请使用带 GPU 的分区（例如 `scavenger`）并用 `--gres=gpu:n` 申请卡
## 2. 每张 GPU 卡最多配 8 核 CPU、1/4 节点内存（官方建议），本脚本已按此配置
## 3. 本项目只能用 H100 或 A5000（库存：4x H100, 16x A5000；其余 1080Ti/2080Ti 不可用）
## 4. 选节点推荐用 find_best_gpu.sh 自动完成：
##      bash find_best_gpu.sh --dry     # 先看它选了哪个节点
##      bash find_best_gpu.sh           # 确认后真正提交
##    手动指定节点则取消下面这行注释并填节点名：
## #SBATCH --nodelist=g009
## 5. 脚本启动后自动检查 GPU 型号，不是 H100/A5000 会立刻报错退出（不浪费机时）
## ────────────────────────────────────────────────────────────────────

# ── SLURM submission script for windowed adaptive walk ───────────────
# Wuhan-Hu-1 (MN908947.3) -> BA.2
# 2 walkers x 1000 steps, full 9803 aa proteome, windowed ESM2 embeddings
#
# Submit: sbatch submit_ncov_evolution.sh   （或 bash find_best_gpu.sh 自动选节点）
# Monitor: squeue -u $USER
# Logs: tail -f ncov_evolution_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

# Allow PyTorch to see the GPU allocated by SLURM
unset CUDA_VISIBLE_DEVICES

# Working directory: prefer the repository location so SLURM's spool copy
# does not change behavior. Set REPO_DIR to the known workspace path.
REPO_DIR="/scratch/fanm01/dms_predict/esm-evolution-walk/mutation_boxes_adaptive/refactored_final_code"
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR"
else
    # Fall back to script location (older behavior)
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$REPO_DIR"
fi

echo "=========================================="
echo " Job ID: ${SLURM_JOB_ID}"
echo " Node: ${SLURM_JOB_NODELIST}"
echo " Partition: ${SLURM_JOB_PARTITION}"
echo " Work dir: ${REPO_DIR}"
echo " Start: $(date)"
echo "=========================================="
echo ""
# ── GPU check: 必须是 H100 或 A5000，否则立刻退出 ────────────────────
echo ">>> GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: nvidia-smi 不可用，没有分到 GPU"; exit 1; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if ! echo "$GPU_NAME" | grep -qiE "H100|A5000"; then
    echo ""
    echo "ERROR: 分到的 GPU 是 '${GPU_NAME}'，不是 H100 或 A5000。"
    echo "       本作业已自动退出，没有消耗机时。"
    echo "       请用 find_best_gpu.sh 重新选择节点，或修改本脚本的 --nodelist。"
    exit 1
fi
echo "  GPU 型号检查通过: ${GPU_NAME}"
echo ""

# Data directory — MUST point to results_major_lineages/ (15 lineages, new run).
# data/ root still contains the OLD 1,441-lineage results — do NOT use it.
DATA_DIR="/scratch/fanm01/dms_predict/esm-evolution-walk/mutation_boxes_adaptive/data/results_major_lineages"

echo ">>> Data directory: ${DATA_DIR}"
echo ""

# Run the full pipeline: prepare -> evolve -> evaluate
# run_model.sh handles conda activation and Python path internally
# Allow overriding output locations via environment variables.
# For short-tests set: SHORTTEST=1 OUTDIR=/scratch/.../evolution_test_short_fanm01 \
# TARGET_ARTIFACTS=/scratch/.../target_artifacts_fanm01 sbatch submit_ncov_evolution.sh
TARGET_ARTIFACTS="${TARGET_ARTIFACTS:-/scratch/fanm01/dms_predict/target_artifacts_fanm01}"
OUTDIR="${OUTDIR:-/scratch/fanm01/dms_predict/evolution_test_short_fanm01}"

mkdir -p "$TARGET_ARTIFACTS" "$OUTDIR"

bash "${REPO_DIR}/run_model.sh" \
    --data-dir "$DATA_DIR" \
    --device cuda \
    --outdir "$OUTDIR" \
    --target-dir "$TARGET_ARTIFACTS" \
    --steps ${STEPS:-1000} \
    --walkers ${WALKERS:-2} \
    --seed ${SEED:-42}

echo ""
echo "=========================================="
echo " End: $(date)"
echo "=========================================="