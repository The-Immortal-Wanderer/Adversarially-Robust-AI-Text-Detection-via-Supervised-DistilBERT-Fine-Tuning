# Towards Adversarially Robust AI Text Detection via Supervised DistilBERT Fine-Tuning: A Held-Out-Generator Generalisation Study

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Authors**: Hammad Masood Mirza, Abdul Rafay Rashid  
**Affiliation**: FAST-NUCES, Islamabad  
**Paper Status**: Future work — codebase under active development for top-tier venue revision

## Overview

This repository implements an ablation study on **DistilBERT** (66M parameters) for AI-generated text detection, trained on the **RAID** benchmark. The study investigates whether supervised fine-tuning with character-level adversarial perturbations generalises to mechanistically distinct held-out generators (paraphrase, summarisation, prompt-based adversarial).

Key contributions:
- **Ablation study** (2×2 design: head depth × freeze strategy) with 4 configurations
- **Config-driven training pipeline** with modular source structure under `src/`
- **Evaluation framework** with multi-seed support, per-attack breakdowns, low-FPR TPR, calibration (ECE), and Fast-DetectGPT baseline
- Designed for **Kaggle dual-T4 execution** with serialised GPU phases and 8.5h defensive timer

## Key Results

*To be updated after full multi-seed runs (5 seeds, ~14h on T4). Preliminary single-seed results:*

| Metric | Value |
|--------|-------|
| Relative Robustness Degradation (RRD) | 1.11% – 3.13% |
| Unseen-split F1 (best) | 0.9267 |
| ROC-AUC (unseen, average) | 0.9790 |
| Fast-DetectGPT AUROC (reproduction) | 0.937 |

> **Note**: These are single-seed results on an RTX 4050 6GB laptop GPU. The full revision roadmap — including multi-seed evaluation, deduplication, clean-only baseline, per-attack reporting, and calibration analysis — is documented in `.omo/plans/end-to-end-restructure-plan.md`.

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
- **Generators**: llama-70b, gpt-4, chatgpt, cohere, davinci-003 (training); held-out set for evaluation
- **Domains**: news, reddit, recipes, poetry, abstracts (5 domains after filtering)
- **Splits**: ~54,000 training pool, held-out generator evaluation set

### DetectRL (Deprecated)
The original codebase was built on DetectRL (~120K samples, 2 attack types). The pipeline has since migrated to RAID for richer generator diversity. Historical scripts remain in `src/data/processing/` for reference.

## Reproduction

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4050 6GB laptop; target: Kaggle P100/T4)
- CUDA 12.1

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

Individual ablation configs are defined in `src/config/default.yaml`. Checkpoints are saved to `artifacts/distilbert_detector/` as `{dataset}_{ablation}_best.pt`.

### Evaluate

```bash
# Evaluate a specific ablation
python scripts/evaluate.py --ablation baseline1
python scripts/evaluate.py --ablation ablation_b
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
│   └── baselines/            # Fast-DetectGPT / perplexity baseline
│       └── perplexity_baseline.py
├── scripts/                  # Entry points
│   ├── train.py              # Unified training
│   ├── evaluate.py           # Evaluation + checkpoint loading
│   ├── benchmark.py          # Performance benchmarks
│   ├── generate_figures.py   # Publication figures
│   └── sanity_check.py       # Pipeline verification
├── data/                     # Data directory (gitignored content)
│   └── processed/            # Preprocessed parquet files
├── artifacts/                # Training checkpoints and summaries
│   └── distilbert_detector/
├── figures/                  # Publication figures (PNG + HTML)
├── benchmark_logs/           # Experimental run logs (gitignored)
├── paper/                    # LaTeX manuscript
├── .omo/                     # Issue registry and plans (gitignored)
│   ├── issues/               # MASTER_REGISTER.md, AUDIT_TRAIL.md, LESSONS_LEARNED.md
│   └── plans/                # Revision plans
├── config/                   # Legacy config (archived)
├── pyproject.toml            # Package metadata + editable install
├── requirements.txt          # pip dependencies
├── environment.yml           # Conda environment
├── .gitattributes            # Line-ending normalisation
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
| CUDA | 12.1 |

**Target execution environment**: Kaggle dual T4 ×2 (or P100 single-GPU fallback) for full multi-seed experiments.

**Optimisations implemented:**
- **Parallel preprocessing**: 3.58× speedup (tokenisation pipelined across CPU workers)
- **Optimised DataLoader**: 4–6 workers with pinned memory and CPU-GPU pipeline parallelism
- **Mixed precision (AMP)**: Automatic mixed precision training on T4; Pascal P100 uses pure FP32 (AMP degrades throughput)
- **Defensive timer**: 8.5h watchdog for Kaggle session safety
- **Config-driven pipeline**: YAML configuration for reproducible experiment management

## Citation

```bibtex
@inproceedings{masood2026towards,
  title={Towards Adversarially Robust {AI} Text Detection via Supervised {DistilBERT} Fine-Tuning: A Held-Out-Generator Generalisation Study on {RAID}},
  author={Masood Mirza, Hammad and Rafay Rashid, Abdul},
  booktitle={Proceedings of the...},
  year={2026},
  note={In preparation}
}
```

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Abdul Rafay Rashid** — Co-author and collaborator
- FAST-NUCES, Islamabad — Academic affiliation and support
- The RAID team for the adversarial detection benchmark
- The DetectRL team for the original benchmark
