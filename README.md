# Towards Adversarially Augmented AI Text Detection via Supervised DistilBERT Fine-Tuning: A Held-Out-Generator Study on RAID

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Authors**: Hammad Masood Mirza, Abdul Rafay Rashid  
**Affiliation**: FAST-NUCES, Islamabad  
**Paper Status**: Submitted to IBCAST 2026; future-work evaluation for top-tier venue

## Overview

This repository implements an ablation study on **DistilBERT** (66M parameters) for AI-generated text detection, trained on the **RAID** benchmark. The study investigates whether supervised fine-tuning on adversarially augmented RAID data generalises to mechanistically distinct held-out generators (gpt3, gpt4, chatgpt, cohere, cohere-chat).

Key contributions:
- **Ablation study** (2×2 design: head depth × freeze strategy) with 4 configurations
- **Config-driven training pipeline** with modular source structure under `src/`
- **Evaluation framework** with config-driven evaluation, checkpoint fallback resolution, GPT-2 XL Perplexity + Binoculars baselines, and planned support for multi-seed (G1), per-attack, calibration/ECE (G5), and low-FPR TPR (G4) analysis
- Designed for **Kaggle serial GPU execution** with serialised GPU phases and three-level defensive timer (8h soft-stop, 8.25h mid-upload stop, 8.5h hard abort)

## Key Results

*To be updated after full multi-seed runs (5 seeds, ~14h on T4). Preliminary single-seed results:*

| Metric | Value |
|--------|-------|
| Relative Robustness Degradation (RRD) | 2.05% – 5.38% |
| Unseen-split F1 (best) | 0.9255 |
| ROC-AUC (unseen, average) | 0.9686 |
| GPT-2 XL Perplexity AUROC | 0.937 |

> **Note**: These are single-seed results from a Kaggle GPU run (seed=42, 2026-05-24) on **dedup'd-but-pre-set-exclusion data** (6,029 human texts overlapped between train and unseen pools — see P-003 in the issue registry). Values above are from run_log.json (ablation_b for lower bound, baseline1 for upper bound). RRD values will change after set-exclusion is applied (paper abstract cites the pre-dedup IBCAST range of 1.1-3.1%). The full revision roadmap — including multi-seed evaluation, deduplication, clean-only baseline, per-attack reporting, and calibration analysis — is documented in `.omo/plans/end-to-end-restructure-plan.md`.

## Ablation Configurations

The study uses a 2×2 design spanning head depth and layer-freezing strategy:

| Config | Classification Head | Frozen Transformer Layers | Description |
|--------|-------------------|--------------------------|-------------|
| `baseline1` | Single (Linear 768→2) | 0 of 6 | Fully fine-tuned; shallow head |
| `ablation_a` | Single (Linear 768→2) | 0–3 (4 of 6) | Partially frozen; shallow head |
| `ablation_b` | Deep (Linear 768→384 → GELU → Dropout → Linear 384→2) | 0–3 (4 of 6) | Partially frozen; deep head (**proposed**) |
| `ablation_c` | Deep (Linear 768→384 → GELU → Dropout → Linear 384→2) | 0 of 6 | Fully fine-tuned; deep head |

## Dataset

### RAID (Primary)
- **Source**: [RAID benchmark](https://github.com/liamdugan/raid) (COLING 2025 Shared Task)
- **Generators**: mpt, mistral, llama-chat, gpt2 (training); gpt3, gpt4, chatgpt, cohere, cohere-chat (held-out evaluation)
- **Domains**: news, reddit, recipes, poetry, abstracts (5 domains after filtering)
- **Splits**: ~120,000 training pool (60K human + 60K AI), held-out generator evaluation set

### DetectRL (Deprecated)
The original codebase was built on DetectRL (~120K samples, 2 attack types). The pipeline has since migrated to RAID for richer generator diversity. DetectRL-specific scripts have been removed during the Phase 0b restructure.

## Reproduction

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 4050 6GB laptop; target: Kaggle P100/T4)
- CUDA 12.4

### Setup

```bash
# Clone the repository
git clone https://github.com/The-Immortal-Wanderer/Adversarially-Robust-AI-Text-Detection-via-Supervised-DistilBERT-Fine-Tuning
cd Adversarially-Robust-AI-Text-Detection-via-Supervised-DistilBERT-Fine-Tuning

# Option A: Using Conda (recommended)
conda env create -f environment.yml
conda activate distilbert-ai-text-detection

# Option B: Using pip (editable install)
pip install -e .
pip install -r requirements.txt
```

### Download Data

```bash
# Download and filter RAID raw data
python src/data/processing/download_raid_raw.py
python src/data/processing/filter_raid_parallel.py   # or filter_raid_sequential.py
```

### Train

```bash
# Train all 4 ablations via the unified config-driven entry point
python scripts/train.py
```

Individual ablation configs are defined in `src/config/default.yaml`. Checkpoints are saved to `artifacts/distilbert_detector/` (or `ann-project-runlog/artifacts/` on Kaggle) as `{dataset}_{ablation}_seed{seed}_best.pt` (e.g., `raid_baseline1_seed42_best.pt`).

### Evaluate

```bash
# Evaluate a specific ablation
python scripts/evaluate.py --checkpoint artifacts/distilbert_detector/raid_baseline1_seed42_best.pt
python scripts/evaluate.py --checkpoint artifacts/distilbert_detector/raid_ablation_b_seed42_best.pt
```

### Run Benchmarks

```bash
# Hardware and throughput benchmarks
python scripts/benchmark.py
```

### Generate Figures

```bash
# Reproduce publication figures
python scripts/generate_figures.py
```

## Project Structure

```
.
├── src/                      # Source package (pip install -e .)
│   ├── config/               # YAML-driven configuration system
│   │   ├── config.py         # Dataclass loader with validation
│   │   └── default.yaml      # Default hyperparameters
│   ├── models/               # DistilBertClassifier definition
│   │   └── distilbert_classifier.py
│   ├── data/                 # Dataset, dataloader, processing scripts
│   │   ├── dataloader.py     # OnTheFlyDataset, CachedTensorDataset
│   │   ├── dataset.py        # RAID dataset loading
│   │   └── processing/       # Download, filter scripts
│   ├── training/             # Trainer module
│   │   └── trainer.py        # train_ablation(), seed_everything()
│   ├── evaluation/           # Metrics
│   │   └── metrics.py        # compute_metrics, ECE, low-FPR TPR
│   └── baselines/            # Perplexity + Binoculars baselines
│       ├── perplexity_baseline.py
│       └── binoculars_baseline.py
├── scripts/                  # Entry points
│   ├── train.py              # Unified training
│   ├── evaluate.py           # Evaluation + checkpoint loading
│   ├── benchmark.py          # Performance benchmarks
│   ├── generate_figures.py   # Publication figures
│   ├── kaggle_run.py         # Kaggle orchestrator + resume
│   ├── g0_decision_gate.py   # Pre-G1 gate: RRD spread check
│   ├── setup_data.py         # RAID download + preprocessing
│   └── sanity_check.py       # Pipeline verification
├── data/                     # Data directory (gitignored content)
│   ├── processed/            # Preprocessed parquet files
│   ├── raw/                  # Raw source data (gitignored)
│   └── contamination_audit.md  # Data pipeline contamination audit
├── notebooks/                # Jupyter notebooks (gitignored; empty after decomposition)
├── results/                  # g0_decision.json (eval JSONs under ann-project-runlog/results/)
├── run_logs/                 # Training run logs (gitignored)
├── ann-project-runlog/       # Kaggle cross-session checkpoints + eval snapshots
├── artifacts/                # Training checkpoints and summaries
│   └── distilbert_detector/
├── figures/                  # Publication figures (PNG + HTML)
├── benchmark_logs/           # Experimental run logs (gitignored)
├── paper/                    # Placeholder directory (Research_Paper.tex at project root)
├── .omo/                     # Issue registry and plans (force-add tracked)
│   ├── issues/               # MASTER_REGISTER.md, AUDIT_TRAIL.md, LESSONS_LEARNED.md
│   ├── plans/                # Revision plans
│   ├── analyses/             # AI review transcripts
│   ├── evidence/             # Verification evidence files
│   ├── drafts/               # Draft documents (currently empty)
│   ├── notepads/            # Release notes etc.
│   ├── run-continuation/     # Cross-session state
│   └── submission-review-prompt.md  # Paper submission review prompt
├── pyproject.toml            # Package metadata + editable install
├── requirements.txt          # pip dependencies
├── environment.yml           # Conda environment
├── .gitattributes            # Line-ending normalisation
├── Claude_Suggestions.md       # 1519-line Claude AI guidance transcript
├── Research_Paper.tex         # IEEEtran LaTeX manuscript
├── .gitignore
└── LICENSE                   # Apache 2.0
```

> Legacy root-level scripts (`train_distilbert_detectrl.py`, `train_distilbert_parallel.py`, `tc3_traindistilbert.py`, `tc4_.py`) were consolidated during Phase 0b restructure. Use `scripts/train.py` and `scripts/benchmark.py` instead.

## Hardware & Optimisations

Experiments were developed and tested on:

| Component | Specification |
|-----------|--------------|
| GPU | **NVIDIA RTX 4050 6GB (laptop)** |
| CPU | AMD Ryzen 5600X (6 cores / 12 threads) |
| RAM | 32 GB |
| CUDA | 12.4 |

**Target execution environment**: Kaggle T4 or P100 (serial single-GPU pipeline; dual T4 task parallelism deferred — see end-to-end plan for status).

**Optimisations implemented:**
- **Parallel preprocessing**: 3.58× speedup (tokenisation pipelined across CPU workers)
- **Optimised DataLoader**: 4–6 workers with pinned memory and CPU-GPU pipeline parallelism
- **Mixed precision (AMP)**: Automatic mixed precision training on T4; Pascal P100 uses pure FP32 (AMP degrades throughput)
- **Defensive timer**: Three-level cascade — 8h soft-stop, 8.25h mid-upload stop, 8.5h hard abort for Kaggle session safety
- **Config-driven pipeline**: YAML configuration for reproducible experiment management

**Kaggle requirements**:
- **Pre-cache DistilBERT** to a Kaggle Dataset (`{owner}/distilbert-base-uncased`) before multi-session runs — unauthenticated HuggingFace Hub has a 100 req/hr rate limit on shared Kaggle IPs, and each session downloads ~268 MB. See [F-004](.omo/issues/MASTER_REGISTER.md#f-004---distilbert-not-pre-cached-in-kaggle-dataset) in the issue registry.
- Data on disk is now set-exclusion cleaned: **zero overlapping human texts** between train and unseen pools (verified after regeneration via `scripts/setup_data.py`). Previous runs used contaminated data with 6,029 overlapping humans (~61% of unseen pool).

## Citation

```bibtex
@inproceedings{masood2026towards,
  title={Towards Adversarially Augmented {AI} Text Detection via Supervised {DistilBERT} Fine-Tuning: A Held-Out-Generator Generalisation Study on {RAID}},
  author={Masood Mirza, Hammad and Rafay Rashid, Abdul},
  booktitle={Proc. IBCAST 2026},
  year={2026},
}
```

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Abdul Rafay Rashid** — Co-author and collaborator
- FAST-NUCES, Islamabad — Academic affiliation and support
- The RAID team for the adversarial detection benchmark
- The DetectRL team for the original benchmark
