# SARS-CoV-2 Proteome Evolution: Code Usage Guide

## Overview

The project evolves the full SARS-CoV-2 proteome (12 ORFs concatenated, 9,803 aa) from Wuhan-Hu-1 toward BA.2 using ESM2 embeddings and an adaptive independent walker algorithm. The codebase has two parts:

- **Part 1 — Data Pipeline** (7 scripts): Downloads sequences, translates to amino acids, concatenates ORFs, and defines mutation boxes.
- **Part 2 — Evolution Engine** (9 files): Runs the adaptive walk using windowed ESM2 embeddings with Metropolis acceptance and epistatic coupling.

```
Part 1: Data Pipeline                    Part 2: Evolution Engine
─────────────────────                    ────────────────────────
01_download.py                           prepare_strain_target.py
02_filter_subsample.py                   windowed_engine.py
02b_filter_translated.py                 windowed_walk.py
03_nextclade_translate.py                corrected_adaptive_engine.py
04_concatenate_orfs_v2.py                ncov_config.py
05_mutation_boxes.py                     run_strain_adaptive.py
06_prepare_target.py                     evaluate_intermediates.py
                                         run_model.sh
                                         submit_ncov_evolution.sh
```

---

## Prerequisites and Downloads

### 1. System Requirements

- **Python 3.10+**
- **GPU:** NVIDIA H100 or A5000 with CUDA support (required for ESM2 inference on 9,803 aa sequences)
- **SLURM cluster** (optional, only if using `submit_ncov_evolution.sh`)
- **~50 GB disk space** for raw sequences, Nextclade dataset, ESM2 model weights, and output files

### 2. Python Packages

```bash
pip install torch numpy pandas biopython tqdm zstandard
pip install fair-esm          # or: pip install esm
```

| Package | Used by | Purpose |
|---|---|---|
| `torch` (with CUDA) | Part 2 | ESM2 model inference, tensor operations |
| `fair-esm` / `esm` | Part 2 | Load `esm2_t33_650M_UR50D` protein language model |
| `numpy` | Part 1 + 2 | Array operations, allowed positions, variability |
| `pandas` | Part 2 | Trajectory CSV I/O |
| `biopython` | Part 1 | FASTA parsing (optional, some scripts use built-in parsers) |
| `tqdm` | Part 1 | Download progress bar (optional, falls back to print) |
| `zstandard` | Part 1 | Decompress `.zst` files from Nextstrain |

### 3. Nextclade CLI

Required by `03_nextclade_translate.py` and `04_concatenate_orfs_v2.py` for nucleotide-to-amino-acid translation. Handles the -1 ribosomal frameshift correctly.

```bash
# Option A: npm
npm install -g @nextclade/nextclade

# Option B: conda
conda install -c bioconda nextclade

# Option C: standalone binary
# Download from https://github.com/nextstrain/nextclade/releases
# Place on PATH (e.g., /usr/local/bin/nextclade)
```

Verify installation:
```bash
nextclade --version
```

### 4. Nextclade Dataset

Downloaded automatically by `03_nextclade_translate.py` on first run:

- **Dataset name:** `nextstrain/sars-cov-2/wuhan-hu-1/orfs`
- **Contains:** Reference genome (MN908947.3 / Wuhan-Hu-1), gene annotations, translation rules
- **Location:** `data/nextclade_dataset/` (created automatically)

No manual download needed.

### 5. ESM2 Model Weights

Downloaded automatically on first run by the `fair-esm` / `esm` package:

- **Model:** `esm2_t33_650M_UR50D` (650M parameters, 33 transformer layers, 1,280-dim embeddings)
- **Size:** ~2.5 GB
- **Cache location:** `~/.cache/torch/hub/checkpoints/` (default PyTorch hub cache)

No manual download needed. First run will download and cache the weights.

### 6. Nextstrain SARS-CoV-2 Data

Downloaded automatically by `01_download.py`:

- **Source:** Nextstrain open data endpoint (GenBank-sourced, publicly available)
- **Content:** `metadata.tsv` (strain info, dates, lineages) + `aligned.fasta` (aligned nucleotide sequences)
- **Full mode:** ~9.4M records, ~10 GB compressed
- **Subsampled mode:** ~4,000 strains, smaller download

No manual download needed.

### 7. config.py (Data Pipeline Configuration)

A shared `config.py` file is required by all Part 1 scripts. It defines:

- Nextstrain download URLs (`URLS_FULL`, `URLS_SUBSAMPLED`)
- Date filter range (`DATE_MIN`, `DATE_MAX`)
- Subsampling parameters (`PER_LINEAGE_TARGET`, `MIN_PER_LINEAGE`, `TARGET_SAMPLE_SIZE`)
- Nextclade CDS list (`NEXTCLADE_CDS`) and dataset name (`NEXTCLADE_DATASET`)
- ORF table and concatenation order (`ORFS`, `CONCATENATION_ORDER`, `ORF_OFFSETS`, `ORF_AA_LENGTHS`)
- Mutation box parameters (`GAUSSIAN_SIGMA`, `GAUSSIAN_THRESHOLD`)

This file is not included in the 16 code files listed here. It must be present in the same directory as the Part 1 scripts.

---

## Part 1: Data Pipeline

Run scripts in numbered order. Each script reads from `--indir` and writes to `--outdir` (both default to `data/`).

### 01_download.py — Download raw sequences

Downloads Nextstrain SARS-CoV-2 open data (metadata + aligned nucleotide FASTA) and decompresses them.

```bash
python 01_download.py --mode full --outdir data/
# mode=subsampled downloads ~4000 strains for quick tests
```

**Output:** `data/metadata.tsv`, `data/aligned.fasta`

### 02_filter_subsample.py — Filter and subsample

Two-pass memory-efficient pipeline: counts records per Pango lineage within a date range, then reservoir-samples an equal number per lineage (default 266 per lineage, 15 lineages, ~3,990 total). Extracts matching sequences from the aligned FASTA using awk.

```bash
python 02_filter_subsample.py --indir data/ --outdir data/
# --per-lineage 300  (override target per lineage)
```

**Output:** `data/filtered_sequences.fasta`, `data/filtered_metadata.tsv`

### 02b_filter_translated.py — Filter existing translations (shortcut)

If Nextclade has already been run on a larger dataset, this script filters the existing per-CDS translated FASTA files to match the subsampled strain list, avoiding a costly Nextclade re-run.

```bash
python 02b_filter_translated.py --indir data/ --outdir data/
```

**Output:** Overwrites `data/translated_<CDS>.fasta` files with filtered subsets.

### 03_nextclade_translate.py — Translate nucleotides to amino acids

Runs the Nextclade CLI to translate filtered nucleotide sequences into per-CDS protein FASTA files. Handles the -1 ribosomal frameshift correctly, producing separate ORF1a and ORF1b.

```bash
python 03_nextclade_translate.py --indir data/ --outdir data/
```

**Output:** `data/translated_<CDS>.fasta` for each ORF (S, ORF1a, ORF1b, ORF3a, E, M, ORF6, ORF7a, ORF7b, ORF8, N, ORF9b)

### 04_concatenate_orfs_v2.py — Concatenate 12 ORFs per strain

Translates the Wuhan-Hu-1 reference with Nextclade, aligns each strain's ORFs to the reference (inserting gaps for deletions), and concatenates all 12 ORFs in fixed order. All output sequences are the same length (reference length).

```bash
python 04_concatenate_orfs_v2.py --indir data/ --outdir data/ \
    --nextclade nextclade --nextclade-dataset data/nextclade_dataset
# --skip-nextclade --ref-translate-dir <dir>  (reuse existing ref translations)
```

**Output:** `data/aa_sequences.fasta` (all strains, aligned), `data/wuhan_aa.fasta` (reference, no gaps), `data/strain_info.tsv`, `data/orf_lengths.json`

### 05_mutation_boxes.py — Define mutation boxes

Scans all concatenated sequences for per-position variability, identifies change blocks, and extends them with Gaussian context (sigma=5, threshold=0.05) to define mutation boxes — regions where mutations are allowed during the walk. Positions outside boxes are frozen.

```bash
python 05_mutation_boxes.py --indir data/ --outdir data/
# --sigma 5 --threshold 0.05  (Gaussian extension parameters)
```

**Output:** `data/mutation_boxes.tsv`, `data/mutation_boxes.json`, `data/allowed_positions.npy`, `data/consensus_aa.npy`, `data/mutation_boxes.png`

### 06_prepare_target.py — Prepare target FASTAs

Creates per-box masked target FASTAs (positions outside each box are masked with 'X') and a JSON manifest for the evolution engine.

```bash
python 06_prepare_target.py --indir data/ --outdir data/
```

**Output:** `data/targets/target_box_<id>.fasta`, `data/targets_manifest.json`, `data/box_definitions.tsv`

---

## Part 2: Evolution Engine

### corrected_adaptive_engine.py — Core algorithm engine

The original single-protein adaptive walk engine, modified minimally. Contains all algorithm constants (temperature, heating, adaptive stage thresholds, coupling parameters), the `WalkConfig` dataclass, proposal functions, Metropolis acceptance, coupling score (Jensen-Shannon divergence), and PRCS scoring.

**Role:** Imported by `windowed_engine.py` and `windowed_walk.py`. Not run directly.

**Key constants:**
- `INITIAL_TEMPERATURE = 0.30`, `FINAL_TEMPERATURE = 0.005`
- `BIG_HEATING_STEPS = 1000`, `BIG_HEATING_K_MIN = 5`, `BIG_HEATING_K_MAX = 10`, `BIG_HEATING_PROBABILITY = 0.70`
- Adaptive stage thresholds: 0.50 / 0.60 / 0.75 (k caps: 4 / 2 / 1)
- `PROPOSALS_PER_STEP = 40`, `COUPLING_RECOMPUTE_EVERY = 150`

### ncov_config.py — Configuration for full-proteome runs

Centralizes all parameters specific to the full SARS-CoV-2 proteome scenario: ORF table (12 ORFs with nucleotide coordinates and concatenation order), window parameters (WINDOW_SIZE=900, WINDOW_STEP=700, WINDOW_OVERLAP=200), mutation box parameters (Gaussian sigma, threshold, expansion), and temperature constants.

**Role:** Imported by `windowed_engine.py`, `windowed_walk.py`, and `run_strain_adaptive.py`. Not run directly.

### windowed_engine.py — Windowed ESM2 engine

The main algorithmic extension. Provides:
- `load_esm2_lowmem()` — loads ESM2 (`esm2_t33_650M_UR50D`) directly to GPU, bypassing CPU memory bottleneck.
- `WindowedEmbedder` class — splits the 9,803 aa sequence into 14 overlapping windows (900 aa each, 200 aa overlap), embeds each window separately via ESM2, and stitches per-residue representations with triangular (Bartlett) weights. Maintains an incremental cache so only windows affected by a mutation are re-computed.
- `compute_windowed_coupling_map()` — detects epistatic pairs within each window using masked marginal JS divergence.
- `propose_single_guided_constrained()` / `propose_coupled_guided_constrained()` — mutation proposals restricted to allowed positions (inside mutation boxes).
- `make_strain_walk_config()` — creates `WalkConfig` for the full proteome, bypassing the original 1,022 aa length guard. Derives `k_single`, `n_groups`, and `p_coupled` from `n_allowed` (3,315) instead of total length (9,803).
- `fill_gaps_with_reference()` — replaces non-standard amino acids (gaps, X, *) with Wuhan reference.

**Role:** Imported by `windowed_walk.py` and `run_strain_adaptive.py`. Not run directly.

### windowed_walk.py — Walker main loop

The core walk loop, mirroring the original `adaptive_walk.py` but using windowed embeddings. Each step:
1. Generates 40 candidate sequences (single or coupled mutations, constrained to boxes).
2. Scores each candidate: PRCS (per-residue cosine similarity to target) + epistatic coupling term.
3. Selects the best candidate and applies Metropolis acceptance (accept if score improves, or with probability `exp(delta / temperature)`).
4. Cools temperature geometrically (`temperature *= cooling_rate`).
5. Recomputes coupling map every 150 steps.
6. Logs trajectory and checkpoints every 200 steps.

Supports checkpoint resume via v2 schema (includes `rejection_log` and `n_allowed`).

**Role:** Called by `run_strain_adaptive.py`. Not run directly.

### prepare_strain_target.py — Prepare target and start embeddings

Loads the BA.2 target sequence and Wuhan-Hu-1 start sequence, computes their windowed ESM2 embeddings, and saves them as PyTorch checkpoint files for the walker to use.

```bash
python prepare_strain_target.py \
    --data-dir data/ \
    --target-strain <BA.2_strain_id> \
    --artifact-dir target_artifacts/ \
    --device cuda
```

**Output:** `target_artifacts/target_embedding.pt`, `target_artifacts/wuhan_start.pt`, `target_artifacts/allowed_positions.npy`

### run_strain_adaptive.py — Main entry point

Ties everything together: loads the ESM2 model, prepares target/start embeddings (or loads existing ones), creates the walk config via `make_strain_walk_config()`, launches multiple walkers with different RNG seeds, and saves trajectories + final sequences.

```bash
python run_strain_adaptive.py \
    --artifact-dir target_artifacts/ \
    --data-dir data/ \
    --outdir evolution_output/ \
    --seed 42 --walkers 2 --total-steps 1000 \
    --device cuda
```

**Output:** `walker_trajectories/walker_XX.csv`, `final_sequences/walker_XX_final.fasta`, `final_sequences/walker_XX_best.fasta`, `final_walker_summary.csv`, `walker_diagnostics.json`

### evaluate_intermediates.py — Post-hoc evaluation

Compares each walker's trajectory against intermediate SARS-CoV-2 strains (Alpha, Delta, BA.1, BA.2) by computing per-step Hamming distances. The expected result: Hamming distance order should be Alpha < Delta < BA.1 < BA.2, indicating the walker follows the real evolutionary path.

```bash
python evaluate_intermediates.py \
    --data-dir data/ \
    --evolution-dir evolution_output/ \
    --outdir evaluation_output/ \
    --intermediates "Alpha=B.1.1.7,Delta=B.1.617.2,BA.1=BA.1,BA.2=BA.2"
```

**Output:** `intermediate_comparison.tsv`, `hamming_distance_plot.png`, `prcs_progress.png`, `mutation_progress.png`

### run_model.sh — Pipeline orchestrator

Shell script that runs the three Part 2 steps in sequence: (1) prepare targets, (2) run evolution, (3) evaluate intermediates. Handles conda environment activation and Python path setup.

```bash
bash run_model.sh \
    --data-dir data/ \
    --device cuda \
    --outdir evolution_output/ \
    --target-dir target_artifacts/ \
    --steps 1000 --walkers 2 --seed 42
```

### submit_ncov_evolution.sh — SLURM submission script

Submits the full pipeline to a SLURM cluster. Requests 1 GPU (H100 or A5000 only), 8 CPUs, 32 GB RAM, 24h wall time. Checks GPU model at runtime and exits immediately if not H100/A5000.

```bash
sbatch submit_ncov_evolution.sh
# Override defaults:
STEPS=500 WALKERS=2 sbatch submit_ncov_evolution.sh
# Short test on A5000:
SHORTTEST=1 sbatch --gres=gpu:a5000:1 submit_ncov_evolution.sh
```

---

## Complete Pipeline (End to End)

```bash
# === Part 1: Data Pipeline ===
python 01_download.py --mode full --outdir data/
python 02_filter_subsample.py --indir data/ --outdir data/
python 03_nextclade_translate.py --indir data/ --outdir data/
python 04_concatenate_orfs_v2.py --indir data/ --outdir data/ \
    --nextclade nextclade --nextclade-dataset data/nextclade_dataset
python 05_mutation_boxes.py --indir data/ --outdir data/
python 06_prepare_target.py --indir data/ --outdir data/

# === Part 2: Evolution Engine ===
# Option A: Run locally
bash run_model.sh --data-dir data/ --device cuda \
    --outdir evolution_output/ --target-dir target_artifacts/ \
    --steps 1000 --walkers 2 --seed 42

# Option B: Submit to SLURM cluster
sbatch submit_ncov_evolution.sh
```

## File Dependency Graph

```
Part 1 (sequential):
  01_download.py → 02_filter_subsample.py → 03_nextclade_translate.py
                                                       ↓
  02b_filter_translated.py (shortcut, skips 03)
                                                       ↓
  04_concatenate_orfs_v2.py → 05_mutation_boxes.py → 06_prepare_target.py

Part 2 (import structure):
  corrected_adaptive_engine.py (core constants & algorithms)
      ↑ imported by
  ncov_config.py (proteome-specific parameters)
      ↑ imported by
  windowed_engine.py (windowed embedding, constrained proposals, config factory)
      ↑ imported by
  windowed_walk.py (walker main loop)
      ↑ called by
  run_strain_adaptive.py (entry point)
      ↑ called by
  run_model.sh (orchestrator: prepare → evolve → evaluate)
      ↑ called by
  submit_ncov_evolution.sh (SLURM submission)

  prepare_strain_target.py (standalone, called by run_model.sh Step 1)
  evaluate_intermediates.py (standalone, called by run_model.sh Step 3)
```
