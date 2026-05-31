# End-to-End Restructure Plan: ANN Project → Top-Tier Venue Revision

**Created**: 2026-05-21
**Last modified**: 2026-05-28
**Status**: Final (3/3 Momus PASS, corrected for G2/G0 sequencing)
**Target**: All compute on Kaggle, code/paper edits on laptop

---

## 0. Key Architecture Decisions (Verified Against Codebase)

| Decision | Rationale | Source |
|----------|-----------|--------|
| Serial execution (single GPU) — ProcessPoolExecutor parallelism DEFERRED | Dual-T4 ProcessPoolExecutor adds complexity (CUDA isolation, result collection, error handling) that cannot be tested without Kaggle access. Current kaggle_run.py runs serial subprocess calls. Revisit after baseline timing results. | Phase 3 verification |
| **DeBERTa-v3-LoRA cross-architecture validation DEFERRED to future work** | Not a reviewer gap item. FlashDeBERTa v0.0.7 is unproven (backward-pass risk). DeBERTa adds ~6.5-12.5h GPU to an already-tight 30h quota. DistilBERT-alone fits in one week. DeBERTa is an enhancement, not a requirement. Revisit via N-021 if quota allows. | 2026-05-26 research-wave analysis |
| AMP FP16 on T4, pure FP32 on P100 | T4 has Tensor Cores (65 TFLOPS FP16), P100 lacks them (9.5 TFLOPS FP32, simulated FP16 is slower) | Gemini + verification |
| Single checkpoint file per run (not dual) | results_{dataset}_{ablation}_seed{N}.pt with everything (state_dict + config + probs + metrics). Periodic overwrite for crash recovery. Avoids dual-write complexity on ephemeral Kaggle storage | Oracle recommendation |
| On-the-fly tokenization on Kaggle (not cached) | /kaggle/working is tmpfs (RAM-backed); writing .pt cache files there competes with model VRAM | Oracle analysis |
| Single-GPU-aware code (no DDP) | DistilBERT 67M params → DDP overhead on PCIe Gen3 > benefit for batch-size-32 workloads | Gemini correction |
| ProcessPoolExecutor for multi-GPU | DEFERRED — will be implemented and tested after single-GPU baseline timing confirms it is needed | Phase 3 verification |

---

## 1. Wave Overview

| Wave | Name | Location | GPU? | Time | Depends On |
|------|------|----------|------|------|------------|
| **P0** | Codebase Restructure | Laptop (editing) | No | ~14-18h | — |
| **P1** | Data Pipeline Fixes | Laptop (editing) | No | ~4-6h | P0 |
| **P2a** | Paper Fixes (No GPU Deps) | Laptop (editing) | No | ~6-8h | P1 |
| **P3** | Kaggle Smoke Test | Kaggle T4 | **Yes** | ~1-2h wall | P0, P1 |
| **P4** | Kaggle GPU Runs | Kaggle T4 | **Yes** | ~25-30h wall serial | P3 |
| **P2b** | Paper Fixes (Post-GPU) | Laptop (editing) | No | ~4-6h | P4 |
| **P5** | Analysis + Final Paper Update | Laptop (editing) | No | ~8-12h | P4 |

> **All GPU work is on Kaggle.** Laptop (RTX 4050 6GB) is used for code editing, LaTeX editing, and analysis only.

---

## 2. Phase 0: Codebase Restructure (~14-18h, laptop only)

**Goal**: Single config-driven training entry point, consolidated model/metrics/dataset sources, unified path system, dead code removal.

### P0.1 — Paths Configuration Singleton (~1h)

**Status**: COMPLETED (2026-05-21)

- **File**: `src/config/config.py` (new)
- **Content**: `PathConfig` dataclass with checkpoint-fallback resolution (centralised path system deferred — paths remain inline per file)
- **Logical path mappings**:
  - `RAW_DIR` → `<root>/data/raw/`
  - `PROCESSED_DIR` → `<root>/data/processed/` (or Kaggle `/kaggle/input/dataset-name/`)
  - `CACHE_DIR` → `<root>/data/processed/tokenized_cache/`
  - `ARTIFACT_DIR` → `<root>/artifacts/distilbert_detector/`
  - `RESULTS_DIR` → `<root>/results/` (or Kaggle `/kaggle/working/results/`)
  - `FIGURES_DIR` → `<root>/figures/`
- **Issues resolved**: Checkpoint fallback resolution (12-entry seed-aware structure in `checkpoint_fallbacks`); path centralisation deferred to future work
- **Verification**: `python -c "from src.config import PathConfig; print(PathConfig())"` prints checkpoint paths

### P0.2 — Consolidate 5× DistilBertClassifier into Single Source (~1h)

**Status**: COMPLETED (2026-05-21)

- **Canonical location**: `src/models/distilbert_classifier.py` (already exists, 60 lines)
- **Actions**:
  - Verify canonical version has all features (head_type="single"|"deep", freeze_layers 0-6)
  - **Fix tc4_ bug**: Add `_freeze_layers()` call to constructor — see P0.7
  - Remove inline copies from: `train_distilbert_detectrl.py`, `train_distilbert_parallel.py`, `tc3_traindistilbert.py`, `tc4_.py`
  - All scripts import from `src.models.distilbert_classifier`
- **Issues resolved**: Model duplication (previously uncataloged — 5 copies)
- **Verification**: `grep -r "class DistilBertClassifier" src/ --include="*.py"` returns exactly 1

### P0.3 — Consolidate 4× compute_metrics into Single Source (~1h)

**Status**: COMPLETED (2026-05-21)

- **Canonical location**: `src/evaluation/metrics.py` (already exists, 137 lines — compute_metrics, compute_rrd, compute_ece, compute_low_fpr_tpr)
- **Unified signature**:
  ```python
  def compute_metrics(
      y_true: list[int] | np.ndarray,
      y_pred: list[int] | np.ndarray,
      y_prob: list[float] | np.ndarray | None = None,
  ) -> dict[str, float]:
  ```
  Returns: `f1_macro`, `f1` (binary, added post-P0.3), `accuracy`, `precision`, `recall`, `roc_auc`, `confusion_matrix`
- **Actions**:
  - Replace canonical implementation with unified signature
  - Update 3 call sites to import from canonical location with new signature
  - Add `compute_rrd(f1_seen, f1_unseen)` export (already exists, used in 3 scripts)
  - Fix tc3 bare `except:` → `except ValueError:` (N-013)
- **Issues resolved**: P-013 (compute_metrics triplicated), N-013 (bare except)
- **Verification**: `grep -r "def compute_metrics" src/ --include="*.py"` returns exactly 1

### P0.4 — Consolidate 3 Dataset Classes into Single Source (~1.5h)

**Status**: COMPLETED (2026-05-21)

- **Canonical location**: `src/data/dataset.py` (already exists, RAIDDataset 83 lines)
- **Classes to include**:
  - `OnTheFlyDataset` — tokenizes in DataLoader workers (from train_distilbert_parallel.py)
  - `CachedTensorDataset` — loads pre-tokenized .pt files (from detectrl + tc3)
   - `DetectRLDataset` — on-the-fly for DetectRL (**NOT IMPLEMENTED** — DetectRL support deprecated alongside DetectRL data pipeline; `src/data/dataset.py` has only `RAIDDataset`, `OnTheFlyDataset`, `CachedTensorDataset`)
- **Strategy selection**: `--dataset-strategy {on_the_fly, cached}` CLI arg
  - Kaggle default: on_the_fly (tmpfs-aware)
  - Local default: cached (persistent, faster)
- **Actions**: Move all 3 classes into src/data/dataset.py, remove inline versions from training scripts
- **Issues resolved**: Missing __init__.py (H-008 will be fixed by proper package structure)
- **Verification**: `from src.data.dataset import OnTheFlyDataset, CachedTensorDataset` succeeds (DetectRLDataset NOT IMPLEMENTED — deprecated alongside DetectRL data pipeline)

### P0.5 — Build Unified `scripts/train.py` (~3h) + YAML config examples

**Status**: PARTIAL — core restructure done (2026-05-21); P0.5b (configs/ YAML files) and F-001/F-002/F-003 still PLANNED

- **Subtask P0.5a**: Create `scripts/train.py` (see below)
- **Subtask P0.5b**: Create minimal `configs/baseline1.yaml`, `configs/ablation_a.yaml`, `configs/ablation_b.yaml`, `configs/ablation_c.yaml` mapping existing hardcoded hyperparameters to the YAML schema
- **Subtask P0.5c**: Create `src/data/dataloader.py` importing `OnTheFlyDataset` + `CachedTensorDataset` from `dataset.py`, with DataLoader factory, worker seeding — decompose from monolithic `scripts/train.py`
- **New file**: `scripts/train.py` (not `src/` — scripts are entry points)
- **Per-ablation config YAMLs**: `configs/` directory with `baseline1.yaml`, `ablation_a.yaml`, `ablation_b.yaml`, `ablation_c.yaml` — **NOT CREATED** (P0.5b pending). Single `default.yaml` used via `--ablation` CLI flag.
- **CLI**:
  ```
  Direct args (argparse):
    --dataset {raid}
    --ablation {baseline1, ablation_a, ablation_b, ablation_c}

  Config overrides (dot notation — overrides YAML defaults):
    Any field in `src/config/default.yaml` can be overridden via
    `--section.key=value`, e.g.:
      --training.batch_size=16  --training.epochs=3
      --training.seed=42       --training.lr=2e-5
      --data.max_length=256    --data.num_workers=2

  See `kaggle_run.py` for the flat-arg wrapper that translates
  `--epochs`/`--batch-size` into dot-notation overrides.
  ```
- **Training loop**: Absorbs structure from `src/training/trainer.py` — class-based API with `train_epoch()` / `evaluate()` / `save_checkpoint()` / `load_checkpoint()` methods
- **Optimizer improvements** (from Claude review evaluation 2026-05-24; **F-001/F-003 PLANNED**):
  - **Weight decay param groups**: Split into `decay` group (all regular params, weight_decay=0.01) and `no_decay` group (bias terms + LayerNorm weights, weight_decay=0.0). **NOT IMPLEMENTED** — F-001 PLANNED.
  - **Per-parameter-group LR**: Pretrained DistilBERT layers at `lr=2e-5`, randomly initialized classifier head at `lr=5e-4`. **NOT IMPLEMENTED** — F-001/F-003 PLANNED.
  - **LR scheduler**: `get_linear_schedule_with_warmup()` from `transformers` with 10% warmup steps and linear decay to 0. **NOT IMPLEMENTED** — F-003 PLANNED.
  - Add `head_lr`, `weight_decay` fields to `TrainingHyperparameters` in config — **NOT IMPLEMENTED**
  - **Files**: `src/training/trainer.py`, `src/config/config.py`, `src/config/default.yaml`
- **DataLoader worker seeding** (from Claude review evaluation 2026-05-24; **F-002 PLANNED**):
  - Add `worker_init_fn(worker_id)` that seeds numpy/torch per worker using `torch.initial_seed()` — **NOT IMPLEMENTED**
  - Pass `worker_init_fn` and `generator` to all DataLoader constructors — **NOT IMPLEMENTED**
  - Without this, multi-worker shuffling is non-reproducible despite `seed_everything()`
  - **Files**: `src/data/dataloader.py`, `src/training/trainer.py`
- **Evaluation**: After training, evaluates on test + unseen split, computes per-attack metrics if labels available
- **Checkpoint**: Single file per run with all keys: state_dict, config, optimizer_state, history, test_metrics, unseen_metrics, all_probs (np.ndarray), all_labels, all_preds, seed, ablation, dataset
- **Issues resolved**: P-014 (probability persistence — probs saved in checkpoint), M-025 (seed parameterization), H-005 (mid-file imports fixed)
- **Verification**: `python scripts/train.py --dataset raid --ablation baseline1 --seed 42 --epochs 1` trains 1 epoch without errors

### P0.6 — Build Kaggle Orchestrator `scripts/kaggle_run.py` (~2h)

**Status**: PARTIAL — kaggle_run.py created and running; F-005 (429 rate-limit batching) still PLANNED

- **New file**: `scripts/kaggle_run.py`
- **Orchestration**:
  - **Serial-only** pipeline runner (no GPU parallelism — single subprocess at a time). ProcessPoolExecutor parallelism DEFERRED — revisit after baseline timing.
  - Iterates configs sequentially: train → evaluate → benchmark for each ablation
  - Uses `subprocess.run()` with `sys.executable` for each script invocation
   - Defensive timer: Three-level cascade (`install_defensive_timer` with thresholds at 8h/8.25h/8.5h):
     - 8.0h (COMPUTE_DEADLINE): Stop launching new ablation runs. Finish current run and save.
     - 8.25h (UPLOAD_DEADLINE): Stop any mid-upload. Save what's uploaded. Flag partial results.
     - 8.5h (HARD_ABORT): sys.exit() as last resort. Prevents Kaggle SIGKILL from corrupting in-progress uploads.
    - Uploads per-ablation: `_upload_run_log()` and `_upload_results_snapshot()` after each ablation (see F-005 — batch end-of-session still PLANNED for 429 rate-limit defense)
   - kagglehub.dataset_upload once at end of session: tar.gz results/ → upload as new dataset version. Single upload avoids Kaggle API rate limiting (~429 after 15-20 versions/session).
- **Environment auto-detection**: Detects Kaggle mode via `KAGGLE_KERNEL_RUN_TYPE` env var; local mode via absence
- **Cross-session resume**: At start, loads `run_log.json`; checks which ablation/epoch/batch_size/seed combinations are complete; skips completed ones
- **Issues resolved**: M-001 (serial GPU constraint — documented), P-017 (dataset accessible — verifies at startup)
- **Verification**: `python scripts/kaggle_run.py --dry-run` prints grid without launching subprocesses

### P0.7 — Fix tc4_ freeze_layers Bug (~30min)

**Status**: COMPLETED (2026-05-21)

- **File**: `scripts/benchmark.py`
- **Bug**: `DistilBertClassifier.__init__` accepts `freeze_layers=0` param but `_freeze_layers()` is NEVER called
- **Fix**: Add `self._freeze_layers()` call after `head_type` setup in constructor
- **Also fix**: Add type annotations to `forward` and `benchmark`; change `weights_only=False` → `True` (N-016); expand `POSSIBLE_PATHS` to 5 entries
- **Issues resolved**: New issue (tc4_ bug), N-016 (weights_only), P-012 (checkpoint paths)
- **Verification**: `python scripts/benchmark.py` runs and reports correct latency; model config matches trained model

### P0.8 — Delete Dead Code (~30min)

**Status**: PARTIAL — deletions done (2026-05-21); H-008 (figures/__init__.py) resolved as SUPERSEDED (not needed — figures/ only has output files per H-008 resolution)

- **Files to delete**:
  - `src/data/filter.py` (329 lines, duplicate of data/filter.py)
- **Files to rewrite**:
  - `src/training/trainer.py` (expanded 217→240 lines: extracted `train_ablation()` / `run_epoch()` from scripts)
- **Files to clean**:
  - Remove unused imports from 4 files (H-001–H-004): `import os` / `import random` in filter scripts
  - Move `import argparse` to top of file in 3 training scripts (H-005)
  - Add `__init__.py` to `figures/` (root `data/` is a script directory, not a package — no `__init__.py` needed there). `src/data/__init__.py` already exists (confirmed).
  - Fix hardcoded user path in generate_figures.py docstring (H-006)
   - Extract hardcoded `batch_size=50_000` to named constant (H-009) — PARTIAL: named `_PARQUET_READ_BATCH_SIZE` in sequential filter; parallel filter already has `PARALLEL_BATCH_SIZE = 50_000` (different name, same purpose)
- **Issues resolved**: N-014, H-001–H-008 (H-009 partial — sequential filter only; see body)
- **Verification**: `git diff --stat` shows deletions; `pytest .` or `python -c "from src.models import *"` succeeds

### P0.9 — Update requirements.txt (~30min)

**Status**: COMPLETED (2026-05-21)

- **Actions**: Scan all `import` statements across codebase, cross-ref against current requirements.txt, add missing deps all with pinned `==` versions (no `>=` — prevents silent metric drift between local 2.6.0 and Kaggle 2.10.0 PyTorch installs)
- **Missing deps expected**: pyarrow, bitsandbytes, kagglehub (scikit-learn==1.7.2 and transformers==5.6.2 are already pinned — verified)
- **Pin all existing deps**: Change all `>=` entries to `==` with specific versions tested on RTX 4050 environment
- **Issues resolved**: P-016/A1e, M-040
- **Verification**: `pip install -r requirements.txt` succeeds in fresh env

### P0.10 — Codebase Cleanup & Reorganization (Phase 0b, ~2-3h, laptop only)

**Status**: COMPLETED (2026-05-21). All items verified: legacy scripts deleted, root parquet->data/processed/, notebooks/ created, plan.md moved to .omo/plans/, `results ss/`->benchmark_logs/, .sisyphus/ deleted. See AUDIT_TRAIL.md for full execution log.

**Goal**: Clean, professional codebase with proper naming, no orphaned files at root, all source code under `src/`, all executable entry points under `scripts/`.

**Target structure**:
```
ANN_Project/
├── src/config/           # (moved from root config/)
├── src/data/processing/  # filter_parallel, filter_sequential (moved from data/)
├── scripts/              # train, evaluate, benchmark (was tc4_), sanity_check, generate_figures
├── notebooks/            # both .ipynb files decomposed into scripts/ (deleted)
├── data/processed/       # seed parquet files (moved from root, NOT git-tracked)
└── benchmark_logs/       # (was "results ss/")
```

**Wave 1 — Parallel file operations (independent, order within wave irrelevant)**:
| # | Operation | Command | Notes |
|---|-----------|---------|-------|
| P0.10.1 | Delete 3 legacy training scripts | `git rm tc3_traindistilbert.py train_distilbert_detectrl.py train_distilbert_parallel.py` | Superseded by scripts/train.py |
| P0.10.2 | Rename tc4_ → scripts/benchmark.py | `git mv tc4_.py scripts/benchmark.py` | Update docstrings, error messages referencing old script names |
| P0.10.3 | Move sanity_check.py | `git mv sanity_check.py scripts/sanity_check.py` | 29-line env diagnostic, pure Python |
| P0.10.4 | Move generate_figures.py | `git mv figures/generate_figures.py scripts/generate_figures.py` | Code lives in scripts/, output stays in figures/ |
| P0.10.5 | Decompose notebooks (files deleted, logic into scripts/train.py + src/training/) | `git rm *.ipynb` (files no longer exist) | 2 notebooks deleted |
| P0.10.6 | Move plan.md | `git mv plan.md .omo/plans/plan.md` | All plans under .omo/plans/ |
| P0.10.7 | Move root parquet → data/processed/ | `Move-Item raid_*.parquet data/processed/` | NOT git-tracked (keep *.parquet in .gitignore) |
| P0.10.8 | Rename `results ss/`→ benchmark_logs/ | `git mv "results ss/" benchmark_logs/` | Contains 5 experiment log .txt files |
| P0.10.9 | Delete .sisyphus/ directory | `Remove-Item -Recurse -Force .sisyphus` | Superseded by .omo/; already in .gitignore |
| P0.10.10 | Update .gitignore | Edit `.gitignore` | Add `.omo/`, `benchmark_logs/`; remove `.sisyphus/` |

**Wave 2 — Dependency-sensitive operations (after Wave 1)**:
| # | Operation | Command | Notes |
|---|-----------|---------|-------|
| P0.10.11 | Resolve duplicate filter.py | `git rm src/data/filter.py` | Canonical filters are `src/data/processing/filter_raid_parallel.py` and `filter_raid_sequential.py`; `data/filter.py` was deleted (was an intermediate duplicate). |
| P0.10.12 | Move config/ → src/config/ | `git mv config/ src/config/` | Updates src/config/__init__.py imports; breaks 2 import sites in scripts/ |

**Wave 3 — Build + import fixes (after Wave 2)**:
| # | Operation | Command | Notes |
|---|-----------|---------|-------|
| P0.10.13 | Add pyproject.toml | Write `pyproject.toml` | setuptools config pointing to `src/`; enables `pip install -e .` |
| P0.10.14 | Fix broken import paths | Edit `scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark.py`, `scripts/g0_decision_gate.py` | config → src.config; remove `sys.path.insert(0, ...)` hack. **COMPLETED** — all sys.path.insert(0, ...) removed; import paths use pyproject.toml + pip install -e . |

**Wave 4 — Verification (after Wave 3)**:
| # | Operation | Command | Notes |
|---|-----------|---------|-------|
| P0.10.15 | Full import verification | `pip install -e .` + test imports | Verify all modules load cleanly |

**Issues resolved**: Root-level cleanup (P0.10), N-015 (duplicate filter.py — originally N-002 in old numbering), P0.10 (legacy TC naming — originally N-003; resolved by script restructuring), notebook relocation (P0.10), H-006 (hardcoded path — if it's in the moved script)

**Verification**: `python -c "from src.config import load_config"` succeeds; `ls *.py` shows only `pyproject.toml`-adjacent files; `ls scripts/*.py` shows 8 entry points (excluding `__init__.py`)

### P0.11 — Pre-Cache DistilBERT to Kaggle Dataset (~30min)

**Status**: NEW — Not yet started

- **Problem**: `DistilBertModel.from_pretrained("distilbert-base-uncased")` at `src/models/distilbert_classifier.py:23` downloads from HF Hub each cold Kaggle session. Unauthenticated rate limit is 100 req/hr per IP — Kaggle T4s share a cluster IP, making this fail silently.
- **Actions**:
  - Create Kaggle Dataset `{owner}/distilbert-base-uncased` containing: config.json, pytorch_model.bin, tokenizer files
  - Update training entry point to check `/kaggle/input/distilbert-base-uncased/` first before falling back to HF Hub
  - Set `TRANSFORMERS_CACHE` to `/kaggle/working/hf_cache/` as second fallback
  - Set `HF_TOKEN` from Kaggle Secrets as third fallback (see also upstream setup for HF_SECRET)
- **Pre-condition for**: P3.1 (Kaggle smoke test), P4 (all GPU runs)
- **Verification**: `python -c "from src.models import DistilBertClassifier; DistilBertClassifier()"` loads without network calls on Kaggle

### P0.12 — Evaluate Script Optimizations (~1h)

**Status**: PARTIAL — inference_mode migration done; eval_batch_size config field pending

- **Problem**: `scripts/evaluate.py` uses `@torch.no_grad()` (not the stricter `@torch.inference_mode()`) and eval batch_size matches training batch_size (32), which is suboptimal for inference.
- **Actions**:
  - Replace `@torch.no_grad()` with `@torch.inference_mode()` in `run_eval()` — disables autograd entirely, ~10% faster than no_grad
  - Add `eval_batch_size` config field (default: 64, 2× training batch) — evaluation is memory-bandwidth-bound, larger batches restore compute-bound behavior
  - Increase eval DataLoader batch sizes in all evaluation entry points
  - **Files**: `scripts/evaluate.py`, `src/config/config.py`, `src/config/default.yaml`
- **Pre-condition for**: P3.3 (benchmark test), P4 (G0/G1 eval phase)
- **Verification**: `python scripts/evaluate.py --eval_batch_size 128` runs and produces same metrics as baseline

---

## 3. Phase 1: Data Pipeline Fixes (~4-6h, laptop only)

### P1.1 — A0: Contamination Audit (~1h)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Read both RAID and DetectRL data pipelines; trace where human-text samples enter the training split; document in `data/contamination_audit.md`
- **Issues resolved**: P-010
- **Verification**: Document at `data/contamination_audit.md` exists with data flow diagram

### P1.2 — Fix Sequential Parquet Naming (~30min)

**Status**: COMPLETED (2026-05-22)

- **File**: `src/data/processing/filter_raid_sequential.py` (was `data/filter_raid_sequential.py`)
- **Fix**: Change output filenames from `train_pool.parquet` → `raid_train_pool.parquet` and `test_unseen.parquet` → `raid_test_unseen.parquet` to match parallel filter
- **Issues resolved**: P-008
- **Verification**: Run filter → output files have `raid_` prefix

### P1.3 — Fix Attack Column Hijack (~1h)

**Status**: COMPLETED (2026-05-22)

- **Files**: `src/data/processing/filter_raid_parallel.py` L217, `src/data/processing/filter_raid_sequential.py` L185 (were `data/` before restructure)
- **Bug**: Both scripts set `attack_type` column to generator value, overwriting the per-attack label
- **Fix**: Rename output column or preserve original attack_type
- **Also: A3 column consumer audit** — trace all scripts that read filtered parquet files and verify column name assumptions (P1.3b, ~30min)
- **Issues resolved**: P-007, M-005
- **Verification**: `grep -r "attack_type" data/processed/ --include="*.parquet"` shows correct values

### P1.4 — Delete Duplicate filter.py (~15min)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Delete `src/data/filter.py`; canonical filters are `src/data/processing/filter_raid_parallel.py` and `filter_raid_sequential.py`
- **Issues resolved**: N-015
- **Verification**: `grep -r "filter_raid" src/data/` returns zero results for filter.py

### P1.5 — Dedup Contamination (A2) (~1h)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Add `df.drop_duplicates(subset=["text"])` to the data pipeline at the point identified by A0 audit. Also added human-text set-exclusion between train and unseen pools in both filter scripts.
- **Issues resolved**: P-003
- **Verification**: Before/after row count diff matches known overlap (6,029 / 5,000)

---

## 4. Phase 2a: Paper Fixes — No GPU Deps (~7.5-9.5h, laptop only)

### B1 — FDG Rename + Binoculars Baseline (~3h)

**Status**: PARTIAL — file rename + internal string audit + Binoculars implementation (545-line `src/baselines/binoculars_baseline.py`) all done. Code strings updated; .tex Fast-DetectGPT paper references retained as literature review (3 occurrences discussing the Bao et al. 2024 paper as prior work).

- **File rename**: `src/baselines/fast_detectgpt.py` → `src/baselines/perplexity_baseline.py`
- **Internal audit**: All imports, docstrings, argparse help, log messages, comments in `.py` files updated. Paper .tex references to Fast-DetectGPT literature retained as prior-work citations (contextually correct).
- **Implement Binoculars baseline**: Add `src/baselines/binoculars_baseline.py` implementing the perplexity/cross-perplexity ratio (Falcon-7b + Falcon-7b-instruct). Binoculars is the current SOTA zero-shot detector (ICML 2024, 90%+ at 0.01% FPR) and outperforms Fast-DetectGPT in multiple settings — required for meaningful comparison at top-tier venues.
- **Files to update**: Any script importing from fast_detectgpt (done); .tex references retained intentionally; evaluation pipeline to include Binoculars scores
- **Issues resolved**: P-001, M-003, + adds missing SOTA zero-shot baseline

### B2 — RRD Naming + Formula Consistency (~1h)

**Status**: N/A — RRD already consistent (20 references, unified formula at \eqref{eq:rrd}); no changes needed

- **Actions**: Standardize RRD formula references in .tex; add `from src.evaluation.metrics import compute_rrd` to any inline formula
- **Issues resolved**: N/A

### B5 — Table IV Confusion Matrix Footnote (~1h)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Add provenance footnote clarifying 0.9161 vs 0.9157 mismatch; no numerical changes needed

### B6 — Threshold Wording (~30min)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Clarify 0.5 argmax threshold as binary sigmoid (not dual softmax) in .tex

### B8 — RRD 5% Threshold Justification (~30min)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Add text acknowledging 5% RRD threshold is arbitrary and uncited; defer to Limitations section or frame as "conventional benchmark in prior work" with a qualifying note
- **Issues resolved**: P-005
- **Verify**: `grep -n "5%" Research_Paper.tex` shows qualified phrasing

### B9 — Table IV Current Stats (~1h)

**Status**: COMPLETED (2026-05-22)

- **Actions**: Verify Table IV values against current `summary.csv`; note any rounding to 4 decimal places is from single-seed evaluation
- **Note**: B5 (footnote) already addresses 0.9161 vs 0.9157 mismatch; B9 adds full-provenance clarity

---

## 5. Phase 3: Kaggle Smoke Test (~1-2h wall, Kaggle T4)

**Goal**: Verify restructured code runs correctly on target Kaggle hardware before committing to 25-30h of GPU runs.

**Status**: LOCAL SMOKE TEST COMPLETED (2026-05-22). Single-seed proof-of-concept Kaggle run completed 2026-05-24 on dual T4s (all 4 ablations trained + evaluated, ~5.6h). Full multi-seed clean evaluation not yet done — requires data regenerated with set-exclusion fix.

### P3.1 — Kaggle Environment Setup (~30min)

**Status**: LOCAL PRECHECK COMPLETED (2026-05-22)

- **Local**: Created `scripts/kaggle_run.py` (~1038 lines, orchestrated pipeline runner)
- **Local**: Relaxed `requirements.txt` pins (`==` → `>=`)
- **Local**: Smoke test ran `python scripts/train.py --dataset raid --ablation ablation_b --training.epochs=1 --training.batch_size=16`
- **Local**: 3 bugs caught and fixed (UnicodeEncodeError → ASCII, samples_per_class clamp, CUBLAS_WORKSPACE_CONFIG)
- **Kaggle remaining**: Upload codebase as Dataset/GitHub-sync; create notebook

### P3.2 — Dual T4 Task Parallelism Test (~30min)

**Status**: DEFERRED — kaggle_run.py uses serial subprocess calls (F4 decision)

- `scripts/kaggle_run.py` dry-run verified locally with 4 ablations
- ProcessPoolExecutor parallelism is deferred — revisit after single-GPU baseline timing
- Full multi-GPU test requires actual Kaggle dual T4 session

### P3.3 — tc4_ Benchmark Test (~15min)

**Status**: COMPLETED (2026-05-22)

- Ran `python scripts/benchmark.py` against checkpoint from local smoke test
- Result: Latency 132.58ms (FP32) → 50.58ms (AMP), **61.85% reduction**
- Throughput: 632.67 samples/sec
- Checkpoint: `artifacts/distilbert_detector/raid_ablation_b_best.pt`

### P3.4 — Cross-Session Resume Test (~30min)

**Status**: BUILT INTO SCRIPTS/KAGGLE_RUN.PY — full test requires Kaggle

- `run_log.json` persistence + `--resume`/`--no-resume` flags implemented
- `is_ablation_completed()` checks epochs + batch_size + seed to detect config changes
- `load_run_log()`/`save_run_log()` with atomic write (`.tmp` + rename) and corrupt-file recovery
- `--run-log-path` CLI flag for persistent storage location

---

## 6. Phase 4: Kaggle GPU Runs — Multi-Seed & Baselines (~25-30h wall, Kaggle T4)

**A single-seed proof-of-concept run (seed=42, all 4 configs) completed 2026-05-24 on dual T4s (~5.6h). Those results used pre-set-exclusion data (6,029 overlapping humans). The following tasks describe the clean multi-seed evaluation pipeline with deduplicated splits.**

**Sequencing**: G0 (Pre-G1 decision gate) runs a single-seed eval on dedup'd data (~2h) to decide whether the full 14h G1 is worth running. G2 (clean baseline training) runs serially after G0 on single GPU (or in parallel with G0 on the second T4 if dual GPU is available). After G0 passes → G1 (multi-seed, 14h) → G3 (FDG dedup, ~4h) → G5 (calibration, ~3h). Net Phase 4 order: (G0 → G2) → DECISION → G1 → G3 → G5.

### G0 — Pre-G1 Decision Gate (~2h on T4)
**Status**: Previously run (2026-05-24) but produced false NO-GO — `g0_decision_gate.py` default checkpoint dir `artifacts/distilbert_detector/` misses checkpoints from Kaggle run (landed in `ann-project-runlog/artifacts/`). Must re-run with correct `--checkpoint-dir`.
**Purpose**: Verify multi-seed evaluation is worth running before committing 14h to G1.
- **Actions**: Single-seed eval on dedup'd split; compute preliminary RRD variance across all 4 ablations
- **Decision criteria** (ALL must pass — multi-criteria, not single-threshold):
  1. **RRD spread** across 4 ablations: use **6-8% threshold** (not 2%). The 2% threshold is smaller than the single-seed noise floor (σ≈2.3%), so it would fire randomly. 6-8% captures genuine pipeline failures (e.g., collapsed F1, DataLoader corruption) without false-alarming on normal single-seed variance.
  2. **F1 floor**: Best config's unseen_F1 ≥ 0.85. If the best-performing config can't reach 0.85 unseen F1, the training pipeline is broken.
  3. **AUROC sanity check**: Best config's unseen AUROC ≥ 0.90. Consistent with published DistilBERT results on RAID.
  4. **Training convergence**: No NaN losses, no GradScaler collapse (scale factor ≥ 256), no extreme training variance across configs.
  5. **Contamination check**: Verify seen/unseen split is disjoint (no shared human texts). Confirm set-exclusion guard is active.
- **If ALL pass** → Proceed to G1.
- **If FAIL** → Pre-written contingency narrative describing the failure mode and its implications for the multi-seed claim. Do NOT proceed to G1 until root cause is identified.
- **Note**: Previous G0 run (2026-05-24) produced false NO-GO because `g0_decision_gate.py` default `--checkpoint-dir` points to `artifacts/distilbert_detector/` but Kaggle checkpoints landed in `ann-project-runlog/artifacts/`. Re-run requires correct checkpoint path.
- **Fallback**: Write contingency narrative if multi-seed falsifies central claim

### G1 — Multi-Seed Evaluation (~14h on T4)
- 5 seeds × 4 configs = 20 runs
- Serial execution (single GPU): 20 sequential runs → 2-3 Kaggle sessions with resume
- ProcessPoolExecutor parallelism deferred — revisit after baseline timing
- **Outputs**: 20 checkpoints with full metrics + all_probs + all_labels + all_preds
- **kagglehub upload**: End of session, tar.gz results/ → upload as new dataset version

### G2 — Clean-Only DistilBERT Baseline (~6h on T4)
**Issue**: N-001 (P1) — All training includes contaminated human texts. Need baseline trained on clean human text only.
- **Actions**: Re-train ablation_b on dedup'd RAID data (human-genuine texts only, no AI-generated training). Same seed=42, same hyperparameters. G2 must complete BEFORE G1 so G1 results can use clean baseline comparison.
- **Pre-requisite**: P1.5 (dedup) must be done before this task
- **Output**: `results/clean_baseline_ablation_b.pt`

### G3 — Zero-Shot Baselines on Dedup'd Data (~4h on T4)
- **Actions**: 
  - Run renamed perplexity_baseline.py on dedup'd data
  - Run binoculars_baseline.py on dedup'd data
- **Caveat**: 
  - The perplexity threshold must be recalibrated on dedup'd data
  - Binoculars threshold is 0.9 (perplexity ratio) — re-verify against dedup'd human texts
- **Comparator standard**: Binoculars achieves 90%+ at 0.01% FPR (ICML 2024); this sets the bar for meaningful detection claims
- **Output**: results/baseline_perplexity_dedup.json + results/baseline_binoculars_dedup.json

### G5 — Calibration/ECE + Brier Score Analysis (~3h on T4)
- **Actions**: Load G1 checkpoints' all_probs arrays; compute Expected Calibration Error per config per seed + Brier Score; plot reliability diagrams
- **Why both**: ECE measures confidence calibration; Brier Score measures overall probabilistic accuracy. NIST GenAI 2026 uses Brier Score alongside AUC-ROC as standard. Tufts et al. (NAACL 2025) demonstrated that high AUROC does not imply good calibration — both metrics needed.
- **Pre-requisite**: P-014 (probability persistence) — resolved by P0.5 unified checkpoint format
- **Output**: Calibration metrics per config (mean ± std across seeds, both ECE and Brier Score)

---

## 7. Phase 2b: Paper Fixes — Post-GPU (~4-6h, laptop)

### B3 — Fix Abstract AUROC Range (~1h)
- **Actions**: Update abstract AUROC values with G1 multi-seed results (mean ± std across 5 seeds)
- **Depends on**: G1 complete

### B7 — Framing Corrections with Real Results (~3-4h)
- **Actions**: Update all numerical claims: RRD values, clean baseline comparison, per-attack breakdowns, primary split designation
  - **Critical**: Report per-generator AUROC (not just F1) for RAID comparability — RAID official benchmark uses macro-averaged AUROC. F1 alone invalidates cross-paper comparison.
- **Add missing citations**: 
  - Tufts et al. (NAACL 2025 Findings) — TPR@FPR evaluation standard, detector limitations
  - Hans et al. (ICML 2024) — Binoculars baseline
  - Fraser et al. (JAIR 2025) — factors influencing detectability survey
  - Wu et al. (Computational Linguistics 2025) — LLM-generated text detection survey
  - NIST GenAI 2026 evaluation plan — calibration/Brier Score as emerging standard
- **Depends on**: G1, G3, G5 complete

---

## 8. Phase 5: Analysis + Final Paper Update (~8.5-12.5h, laptop)

### P5.1 — Per-Attack Metrics (~2h)
- **Actions**: From G1 checkpoints, compute metrics broken down by individual attack type (not just seen/unseen families)
- **Output**: Per-attack tables for paper

### P5.2 — Multi-Seed Variance Analysis (~1h)
- **Actions**: Compute mean ± std across 5 seeds for all metrics; check if ablation_b superiority claim holds
- **Output**: Variance table for paper

### P5.3 — Clean Baseline Comparison (~1h)
- **Actions**: Add clean-only DistilBERT baseline (trained on human text only)
- **Depends on**: G1 checkpoints (or can re-train)

### P5.4 — TPR@FPR + Prevalence Analysis (~1.5h)
- **Actions**: 
  - Report TPR at 1% and 0.5% FPR for all configs — this is now the de facto evaluation standard (Tufts et al. NAACL 2025 Findings, NIST GenAI 2026)
  - Report metrics at low FPR (1%, 5%) to assess practical detection utility
  - Compare against Binoculars baseline at same FPR thresholds
  - Note: Tufts et al. found many detectors have TPR@.01 as low as 0% — this motivates why DistilBERT fine-tuning is needed

### P5.5 — Final .tex Update (~3-4h)
- **Actions**: Update all tables, figures, narrative with final numbers from G1/G3/G5
- **Includes**: G8 (count tables) — generate dataset/attack-sample count tables from dedup'd data for .tex
- **Verify**: LSP diagnostics clean, all \ref{} → valid labels, all \cite{} → \bibitem{}

---

## 9. Dependency Tree

```
P0 (restructure) ──┬── P1 (data pipeline) ──┬── P2a (paper no-GPU)
                   │                        │
                   │                        └── B3/B7 blocked (wait for P4)
                   │
                   └── P3 (Kaggle smoke) ── P4 (GPU runs) ──┬── P2b (post-GPU paper)
                                                             │
                                                             └── P5 (final analysis)
```

**Critical path**: P0 → P3 → P4 (G0 → G2 → G1 → G3 → G5) → P2b → P5
**Parallelizable**: P1 ∥ P2a (both depend on P0 only)
**Phase 4 internal**: (G0 → G2 serial on single GPU, G0≈2h, G2≈6h) → DECISION → G1 (14h) → G3 (4h) → G5 (3h).

---

## 10. Issue Registry → Task Map

| Issue | Category | Priority | Resolved By |
|-------|----------|----------|-------------|
| P-001 (FDG misnamed) | CODE | P0 | B1 (Phase 2a) |
| **P-002 (multi-seed falsification risk)** | **EXPERIMENT** | **P0** | **G0 decision gate (Phase 4)** |
| P-003 (contamination) | DATA | P0 | P1.5 (Phase 1) |
| **P-004 (abstract AUROC range)** | **PAPER** | **P0** | **B3 (Phase 2b)** |
| **P-005 (5% RRD threshold)** | **PAPER** | **P0** | **P2a — add deferred to Limitations** |
| P-006 (CPU autocast) | CODE | P0 | P0.5 (scripts/train.py has AMP guards) |
| P-007 (column hijack) | CODE | P0 | P1.3 (Phase 1) |
| P-008 (parquet naming) | DATA | P0 | P1.2 (Phase 1) |
| P-009 (FDG comparison) | CODE/FRAMING | P0 | B1 rename + P5 analysis |
| P-010 (contamination source) | CODE | P0 | P1.1 (A0 audit) |
| P-011 (tc3 cache crash) | CODE | P0 | Absorbed — tc3 replaced by scripts/train.py (P0.5) |
| P-012 (tc4 paths) | CODE | P0 | P0.7 (fix tc4 paths) |
| P-013 (metrics triplicated) | CODE | P0 | P0.3 (metrics consolidation) |
| P-014 (prob persistence) | CODE | P0 | P0.5 (unified checkpoint includes all_probs) |
| **P-015 (Pre-G1 gate)** | **EXPERIMENT** | **P0** | **G0 (Phase 4)** |
| P-016 (A1e requirements) | HOUSEKEEPING | P0 | P0.9 (requirements.txt update) |
| **P-017 (RAID accessibility)** | **DATA** | **P0** | **P3.1 (Kaggle smoke pre-check) — RESOLVED** |
| P-018 (CUDA env) | INFRASTRUCTURE | P0 | P3.1 (Kaggle smoke test verifies) — RESOLVED |
| P-019 (GPU thermal) | INFRASTRUCTURE | P0 | P3.1 (Kaggle env handles thermal) — PLANNED |
| M-001 (GPU serial) | CODE | P0 | P0.6 (serial orchestrator built; PARTIAL — remaining actions still open: parallel language in plan preamble, explicit wall-clock estimates) |
| **M-002 (priority contradiction)** | **FRAMING** | **P0** | **Section 12 success criteria — note resolved** |
| M-003 (FDG rename + audit) | CODE | P0 | B1 (Phase 2a) |
| M-004 (GPU budget) | INFRASTRUCTURE | P0 | P4 schedule (estimated 25-30h) |
| M-005 (column audit) | CODE | P1 | P1.3b (column consumer audit) |
| M-025 (hardcoded seed) | CODE | P1 | P0.5 (--seed arg in scripts/train.py) |
| M-035 (dependency matrix) | FRAMING | P1 | This plan's dependency tree (corrected) |
| M-040 (missing deps) | HOUSEKEEPING | P1 | P0.9 (requirements.txt) |
| M-041 (Docker) | INFRASTRUCTURE | P1 | Deferred — nice-to-have after restructure |
| **N-001 (clean baseline)** | **EXPERIMENT** | **P1** | **G2 (Phase 4, newly added)** |
| **N-002 (calibration/ECE)** | **EXPERIMENT** | **P1** | **G5 (Phase 4)** |
| **N-003 (low-FPR metrics)** | **EXPERIMENT** | **P2** | **P5.4 (prevalence analysis)** |
| **N-004 (bootstrap signif.)** | **EXPERIMENT** | **P2** | **P5.2 (variance analysis)** |
| **N-005 (per-attack eval)** | **EXPERIMENT** | **P2** | **P5.1 (Phase 5)** |
| **N-006 (count tables)** | **DATA** | **P2** | **P5.5 (final .tex update)** |
| N-013 (bare except) | CODE | RESOLVED | P0.3 (metrics consolidation fixes this) |
| N-014 (trainer.py dead) | CODE | RESOLVED | P0.8 (delete dead code) |
| N-015 (filter.py duplicate) | CODE | RESOLVED | P1.4 (delete duplicate) |
| N-016 (weights_only=False) | CODE | RESOLVED | P0.7 (fix tc4 weights_only) |
| H-001–H-004 (unused imports) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-005 (mid-file imports) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-006 (hardcoded path) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-008 (missing __init__.py) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-009 (inline batch_size) | HOUSEKEEPING | P3 | P0.8 (cleanup pass — PARTIAL: named in sequential filter as `_PARQUET_READ_BATCH_SIZE`; parallel filter already has `PARALLEL_BATCH_SIZE`) |
| M-034 (G1 per-config timing) | CODE | P1 | G1 (timing correction) |
| M-036 (Critical path inclusion) | CODE | P1 | Pre-G1 (path validation) |
| M-037 (Priority/execution diagram) | CODE | P1 | P0.5 (plan diagram update) |
| M-038 (L60 merge conflict) | CODE | P1 | P0.5 (paper merge resolution) |
| M-039 (Timeline underestimation) | CODE | P1 | Plan revision (wave table update) |
| **New**: tc4_ freeze_layers bug | CODE | P0 | P0.7 (fix before any benchmark run) |

**Coverage: ~77 of 114 issues mapped to tasks in MASTER_REGISTER §9 (19 P‑series + 35 M‑series + 15 N‑series + 8 F‑series). This plan maps ~45 core issues directly to tasks; the remainder are tracked in the Issue Registry with plan-ref links. 2 PLANNED P1 items require task slots (M-021→B7 subtask, M-022→new B7 subtask). M-007 already RESOLVED via earlier fix waves. 2 PROPOSED P1 items pending validation (M-026, M-033); the remaining M-026–M-033 range are resolved (M-027, M-030, M-031, M-032) or rejected (M-028, M-029). Binoculars baseline registered as N-020. DeBERTa-v3-LoRA cross-arch (N-021) DEFERRED — 37 unmapped issues tracked separately in registry — not in current plan scope. F‑series (F-001–F-008) tracked in MASTER_REGISTER §9 — not listed in this table for brevity.**

---

## 11. Commit Strategy

| # | Commit Message | When | Files Touched |
|---|---------------|------|---------------|
| 1 | `refactor: paths config + environment detection` | After P0.1 | src/config/config.py (new), + all scripts updated to use it |
| 2 | `refactor: consolidate model/metrics/dataset into canonical sources` | After P0.2-0.4 | src/models/, src/evaluation/, src/data/ |
| 3 | `feat: unified scripts/train.py + scripts/kaggle_run.py` | After P0.5-0.6 | scripts/train.py (new), scripts/kaggle_run.py (new) |
| 4 | `fix: benchmark.py freeze_layers bug + checkpoint paths + security` | After P0.7 | scripts/benchmark.py (was tc4_.py) |
| 5 | `chore: delete dead code, fix imports, add __init__.py` | After P0.8 | Multiple files deleted/cleaned |
| 6 | `fix: parquet naming + column hijack + contamination dedup` | After P1.1-1.5 | src/data/processing/filter_raid_*.py, src/data/filter.py (del) |
| 7 | `feat: FDG rename + Binoculars zero-shot baseline` | After B1 | src/baselines/ |
| 8 | `fix: paper corrections (RRD, Table IV, threshold)` | After P2a | Research_Paper.tex |
| *(Kaggle runs happen uncommitted — results checked in after)* | | | |
| 9 | `fix: abstract AUROC range + framing with final numbers` | After P4 + P2b | Research_Paper.tex |
| 10 | `feat: per-attack metrics, variance, TPR@FPR, calibration (ECE+Brier), Binoculars comparison` | After P5 | Various analysis scripts + .tex |

---

## 12. Success Criteria (End State)

- [ ] Single `scripts/train.py` replaces all 4 old training scripts
- [ ] All 77 mapped issues closed + tc4_ bug fixed
- [ ] Kaggle smoke test passes (1 epoch × 1 config × 1 seed)
- [ ] Single-GPU serial execution confirmed working (deferred: ProcessPoolExecutor parallel, revisit after baseline timing)
- [ ] G2: Clean-only DistilBERT baseline complete (before G1)
- [ ] G1: 5 seeds × 4 configs complete with per-attack metrics
- [ ] G0: Decision gate passed (or contingency narrative written)
- [ ] G3: Both baselines (GPT-2 Perplexity + Binoculars) on dedup'd data complete
- [ ] G5: Calibration metrics (ECE + Brier Score) per config per seed
- [ ] TPR@FPR reported for all configs + baselines
- [ ] Per-generator AUROC reported alongside F1 for all configs
- [ ] Paper .tex has final numbers from all GPU runs
- [ ] LSP diagnostics clean on all changed .tex and .py files
- [ ] Optimizer fixes applied: weight-decay exclusion for bias/LayerNorm, per-param-group LR, LR scheduler with warmup
- [ ] Reproducibility fixes applied: DataLoader worker seeding, pinned library versions, HF_TOKEN from Secrets
- [ ] DistilBERT pre-cached to Kaggle Dataset (no HF Hub download in sessions)
- [ ] DeBERTa-v3-LoRA cross-architecture validation: DEFERRED to future work (N-021) — DistilBERT alone supports all core claims within one Kaggle week
- [ ] evaluate.py uses `inference_mode()` and configurable eval batch size
