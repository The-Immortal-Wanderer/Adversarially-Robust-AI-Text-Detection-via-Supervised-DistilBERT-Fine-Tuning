# AI-Generated Text Detection via DistilBERT Fine-Tuning: A Cross-Attack Generalization Study

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Authors**: Hammad Masood, Abdul Rafay  
**Affiliation**: FAST-NUCES, Islamabad  
**Paper Status**: In preparation for IEEE submission

## Overview

This repository implements an ablation study on **DistilBERT** (66M parameters) for AI-generated text detection, trained on the **DetectRL** benchmark with adversarial augmentation. The study investigates cross-attack generalization — whether models trained on character-level attacks (homoglyph substitution, word substitution) can generalize to mechanistically distinct semantic-level unseen attacks (paraphrase, prompt-based adversarial).

Key contributions:
- **Ablation study** (2×2 design: head depth × freeze strategy) with 4 configurations
- **Cross-attack evaluation** on unseen semantic-level attack types
- **Fast-DetectGPT reproduction** at AUROC 0.937, extended with NF4 quantization
- **Parallel & distributed computing optimizations** benchmarked on a single RTX 3050 8GB GPU

## Key Results

| Metric | Value |
|--------|-------|
| Relative Robustness Degradation (RRD) | 1.11% – 3.13% |
| Unseen-split F1 (best) | **0.9267** |
| ROC-AUC (unseen, average) | **0.9790** |
| Fast-DetectGPT AUROC (reproduction) | 0.937 |
| Parallel preprocessing speedup | 3.58× |
| AMP training speedup | 3.32× |

All four ablation configurations remain within the 5% generalization threshold, providing a positive empirical answer to whether adversarial augmentation generalizes across the character-level/semantic-level attack divide at 66M parameter scale.

## Ablation Configurations

The study uses a 2×2 design spanning head depth and layer-freezing strategy:

| Config | Classification Head | Frozen Transformer Layers | Description |
|--------|-------------------|--------------------------|-------------|
| `baseline1` | Single (Linear 768→2) | 0 of 6 | Fully fine-tuned; shallow head |
| `ablation_a` | Single (Linear 768→2) | 0–3 (4 of 6) | Partially frozen; shallow head |
| `ablation_b` | Deep (Linear 768→384 → GELU → Dropout → Linear 384→2) | 0–3 (4 of 6) | Partially frozen; deep head (**proposed**) |
| `ablation_c` | Deep (Linear 768→384 → GELU → Dropout → Linear 384→2) | 0 of 6 | Fully fine-tuned; deep head |

**Training times** (RTX 3050 8GB):
- `baseline1`: ~88 minutes (full fine-tune, single head)
- `ablation_a`: ~53 minutes (partially frozen, single head)
- `ablation_b`: ~53 minutes (partially frozen, deep head) — **proposed**
- `ablation_c`: ~88 minutes (full fine-tune, deep head)

## Dataset

### DetectRL Benchmark
- **Source**: [DetectRL](https://github.com/NLPcode/DetectRL) — Adversarial AI text detection benchmark
- **Size**: ~120,000 training samples (60K human + 60K AI-generated)
- **Splits**: Train (80%) / Validation (10%) / Test (10%) from capped pool, plus an independent unseen evaluation set (10,000 samples)
- **Attack types used in training**: Character-level — homoglyph substitution, word substitution (Tasks 1–2)
- **Unseen attack types** (held out from training): Semantic-level — paraphrase rewriting, prompt-based adversarial (Task 4)

### HC3 (Supplementary)
- **Source**: [HC3](https://github.com/Hello-SimpleAI/chatgpt-detector) — Human-ChatGPT Comparison
- Used for: `finance` subset (supplementary analysis)

## Reproduction

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 3050)
- CUDA 12.1

### Setup

```bash
# Clone the repository
git clone https://github.com/The-Immortal-Wanderer/ai-text-detection-ablation.git
cd ai-text-detection-ablation

# Option A: Using Conda (recommended)
conda env create -f environment.yml
conda activate distilbert-ai-text-detection

# Option B: Using pip
pip install -r requirements.txt
```

### Download Data

Data files (.parquet) are excluded from the repository due to size (~875MB). You must download them from the original sources:

1. **DetectRL**: Download from [DetectRL GitHub](https://github.com/NLPcode/DetectRL) and place files in `data/raw/detectrl/`
2. **HC3**: Download from [HC3 GitHub](https://github.com/Hello-SimpleAI/chatgpt-detector) and place files in `data/raw/hc3/`
3. **RAID** (for Fast-DetectGPT evaluation): Download from [RAID benchmark](https://github.com/liamdugan/raid)

Then run the preprocessing scripts:

```bash
python data/download_detectrl_HC3.py
python data/filter.py
```

### Train

```bash
# Train all 4 ablations (recommended — ~5 hours total)
python train_distilbert_detectrl.py

# Or train the proposed config only
python tc3_traindistilbert.py

# Or run the parallel-optimized version
python train_distilbert_parallel.py
```

### Evaluate

Checkpoints are saved to `artifacts/distilbert_detector/` as `.pt` files. Results are summarized in `artifacts/distilbert_detector/summary.csv`.

## Hardware & Optimizations

All experiments were conducted on:

| Component | Specification |
|-----------|--------------|
| GPU | **NVIDIA RTX 3050 8GB** |
| CPU | AMD Ryzen 5600X (6 cores / 12 threads) |
| RAM | 32 GB |
| CUDA | 12.1 |

**Optimizations implemented and benchmarked:**
- **Parallel preprocessing**: 3.58× speedup (tokenization pipelined across CPU workers)
- **Optimized DataLoader**: 4 workers with pinned memory, CPU-GPU pipeline parallelism
- **Mixed Precision (AMP)**: Automatic mixed precision training with enlarged batch size — 3.32× speedup
- **4-bit NF4 Quantization**: Fast-DetectGPT inference with bitsandbytes NF4 — 54.25% latency reduction (380.64 samples/second)
- **Gradient clipping**: Norm clipped at 1.0 for training stability

## Repository Structure

```
.
├── src/
│   ├── __init__.py           # Source package
│   ├── models/               # DistilBERT classifier definition
│   ├── data/                 # Dataset and dataloader
│   └── evaluation/           # Metrics module
├── data/                     # Data download and preprocessing scripts
│   ├── download_detectrl_HC3.py
│   ├── download_raid_raw.py
│   ├── filter.py
│   └── ...
├── artifacts/                # Training checkpoints and result summaries
│   └── distilbert_detector/
│       ├── summary.csv       # Consolidated results
│       ├── best_config.json  # Best config metadata
│       └── training_times.json
├── figures/                  # Publication figures (PNG + interactive HTML)
├── results ss/               # Experimental run logs
├── train_distilbert_detectrl.py   # Main training entry point
├── train_distilbert_parallel.py   # Parallel-optimized training variant
├── tc3_traindistilbert.py         # TC3 demonstration (ablation_b only)
├── requirements.txt          # pip dependencies
├── environment.yml           # Conda environment
├── Research_Paper.tex        # IEEE LaTeX manuscript
└── LICENSE                   # Apache 2.0
```

## Citation

If you use this code or findings in your research, please cite the associated paper:

```bibtex
@inproceedings{masood2026towards,
  title={Towards Adversarially Robust {AI} Text Detection via Supervised {DistilBERT} Fine-Tuning: A Cross-Attack Generalization Study on {DetectRL}},
  author={Masood, Hammad and Rafay, Abdul},
  booktitle={Proceedings of the...},
  year={2026},
  note={In preparation for IEEE submission}
}
```

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Abdul Rafay** — Co-author and collaborator on the research paper and experiments
- FAST-NUCES, Islamabad — Academic affiliation and support
- The DetectRL team for the adversarial detection benchmark
