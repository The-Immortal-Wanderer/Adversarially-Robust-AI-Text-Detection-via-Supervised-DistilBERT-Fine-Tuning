# End-to-End Restructure Plan: ANN Project → Top-Tier Venue Revision

**Created**: 2026-05-21
**Status**: Final (3/3 Momus PASS, corrected for G2/G0 sequencing)
**Target**: All compute on Kaggle, code/paper edits on laptop

---

## 0. Key Architecture Decisions (Verified Against Codebase)

| Decision | Rationale | Source |
|----------|-----------|--------|
| Kaggle dual T4 task parallelism via ProcessPoolExecutor (NOT DDP) | 2 concurrent single-GPU runs via CUDA_VISIBLE_DEVICES; ThreadPoolExecutor + CUDA = undefined behavior | Oracle review |
| AMP FP16 on T4, pure FP32 on P100 | T4 has Tensor Cores (65 TFLOPS FP16), P100 lacks them (9.5 TFLOPS FP32, simulated FP16 is slower) | Gemini + verification |
| Single checkpoint file per run (not dual) | results_{dataset}_{ablation}_seed{N}.pt with everything (state_dict + config + probs + metrics). Periodic overwrite for crash recovery. Avoids dual-write complexity on ephemeral Kaggle storage | Oracle recommendation |
| On-the-fly tokenization on Kaggle (not cached) | /kaggle/working is tmpfs (RAM-backed); writing .pt cache files there competes with model VRAM | Oracle analysis |
| Single-GPU-aware code (no DDP) | DistilBERT 67M params → DDP overhead on PCIe Gen3 > benefit for batch-size-32 workloads | Gemini correction |
| ProcessPoolExecutor for multi-GPU | Clean CUDA context isolation per subprocess; parent process has no CUDA context | Oracle critical finding |

---

## 1. Wave Overview

| Wave | Name | Location | GPU? | Time | Depends On |
|------|------|----------|------|------|------------|
| **P0** | Codebase Restructure | Laptop (editing) | No | ~10-14h | — |
| **P1** | Data Pipeline Fixes | Laptop (editing) | No | ~4-6h | P0 |
| **P2a** | Paper Fixes (No GPU Deps) | Laptop (editing) | No | ~6-8h | P1 |
| **P3** | Kaggle Smoke Test | Kaggle T4 | **Yes** | ~1-2h wall | P0, P1 |
| **P4** | Kaggle GPU Runs | Kaggle T4 | **Yes** | ~25-30h wall serial | P3 |
| **P2b** | Paper Fixes (Post-GPU) | Laptop (editing) | No | ~4-6h | P4 |
| **P5** | Analysis + Final Paper Update | Laptop (editing) | No | ~8-12h | P4 |

> **All GPU work is on Kaggle.** Laptop (RTX 4050 6GB) is used for code editing, LaTeX editing, and analysis only.

---

## 2. Phase 0: Codebase Restructure (~10-14h, laptop only)

**Goal**: Single config-driven training entry point, consolidated model/metrics/dataset sources, unified path system, dead code removal.

### P0.1 — Paths Configuration Singleton (~1h)
- **File**: `src/config.py` (new)
- **Content**: `Paths` dataclass with environment auto-detection (Kaggle via `KAGGLE_KERNEL_RUN_TYPE`, else local)
- **Logical path mappings**:
  - `RAW_DIR` → `<root>/data/raw/`
  - `PROCESSED_DIR` → `<root>/data/processed/` (or Kaggle `/kaggle/input/dataset-name/`)
  - `CACHE_DIR` → `<root>/data/processed/tokenized_cache/`
  - `ARTIFACT_DIR` → `<root>/artifacts/distilbert_detector/`
  - `RESULTS_DIR` → `<root>/results/` (or Kaggle `/kaggle/working/results/`)
  - `FIGURES_DIR` → `<root>/figures/`
- **Issues resolved**: Fragmentation (91 access points across 16 files with 6 ROOT_DIR strategies)
- **Verification**: `python -c "from src.config import Paths; print(Paths())"` prints correct paths per environment

### P0.2 — Consolidate 5× DistilBertClassifier into Single Source (~1h)
- **Canonical location**: `src/models/distilbert_classifier.py` (already exists, 68 lines)
- **Actions**:
  - Verify canonical version has all features (head_type="single"|"deep", freeze_layers 0-6)
  - **Fix tc4_ bug**: Add `_freeze_layers()` call to constructor — see P0.7
  - Remove inline copies from: `train_distilbert_detectrl.py`, `train_distilbert_parallel.py`, `tc3_traindistilbert.py`, `tc4_.py`
  - All scripts import from `src.models.distilbert_classifier`
- **Issues resolved**: Model duplication (previously uncataloged — 5 copies)
- **Verification**: `grep -r "class DistilBertClassifier" src/ --include="*.py"` returns exactly 1

### P0.3 — Consolidate 4× compute_metrics into Single Source (~1h)
- **Canonical location**: `src/evaluation/metrics.py` (already exists, 37 lines, currently unused)
- **Unified signature**:
  ```python
  def compute_metrics(
      logits: torch.Tensor | None = None,
      labels: torch.Tensor | None = None,
      loss: float | None = None,
      threshold: float = 0.5,
  ) -> dict[str, float]:
  ```
  Returns: `f1` (binary), `f1_macro`, `accuracy`, `precision`, `recall`, `roc_auc`, `loss`, `confusion_matrix`
- **Actions**:
  - Replace canonical implementation with unified signature
  - Update 3 call sites to import from canonical location with new signature
  - Add `compute_rrd(f1_seen, f1_unseen)` export (already exists, currently unused)
  - Fix tc3 bare `except:` → `except ValueError:` (N-013)
- **Issues resolved**: P-013 (compute_metrics triplicated), N-013 (bare except)
- **Verification**: `grep -r "def compute_metrics" src/ --include="*.py"` returns exactly 1

### P0.4 — Consolidate 3 Dataset Classes into Single Source (~1.5h)
- **Canonical location**: `src/data/dataset.py` (already exists, DetectRLDataset 83 lines)
- **Classes to include**:
  - `OnTheFlyDataset` — tokenizes in DataLoader workers (from train_distilbert_parallel.py)
  - `CachedTensorDataset` — loads pre-tokenized .pt files (from detectrl + tc3)
  - `DetectRLDataset` — on-the-fly for DetectRL (already in src/data/dataset.py)
- **Strategy selection**: `--dataset-strategy {on_the_fly, cached}` CLI arg
  - Kaggle default: on_the_fly (tmpfs-aware)
  - Local default: cached (persistent, faster)
- **Actions**: Move all 3 classes into src/data/dataset.py, remove inline versions from training scripts
- **Issues resolved**: Missing __init__.py (H-008 will be fixed by proper package structure)
- **Verification**: `from src.data.dataset import OnTheFlyDataset, CachedTensorDataset, DetectRLDataset` succeeds

### P0.5 — Build Unified `scripts/train.py` (~3h) + YAML config examples

- **Subtask P0.5a**: Create `scripts/train.py` (see below)
- **Subtask P0.5b**: Create minimal `configs/baseline1.yaml`, `configs/ablation_a.yaml`, `configs/ablation_b.yaml`, `configs/ablation_c.yaml` mapping existing hardcoded hyperparameters to the YAML schema
- **New file**: `scripts/train.py` (not `src/` — scripts are entry points)
- **CLI**:
  ```
  --dataset {raid, detectrl}
  --ablation {baseline1, ablation_a, ablation_b, ablation_c}
  --seed N
  --amp (auto-detect: True on T4/Ampere+, False on Pascal)
  --batch-size N (T4=32, P100=16, 4050=8)
  --epochs N (default 3)
  --max-length N (default 256)
  --num-workers N (Kaggle=2, local=4)
  --dataset-strategy {on_the_fly, cached}
  --output-dir PATH
  --config PATH (YAML for grouped configs)
  ```
- **Training loop**: Absorbs structure from `src/training/trainer.py` (the dead code with superior structure) — class-based API with `train_epoch()` / `evaluate()` / `save_checkpoint()` / `load_checkpoint()` methods
- **Evaluation**: After training, evaluates on test + unseen split, computes per-attack metrics if labels available
- **Checkpoint**: Single file per run with all keys: state_dict, config, optimizer_state, history, test_metrics, unseen_metrics, all_probs (np.ndarray), all_labels, all_preds, seed, ablation, dataset
- **Issues resolved**: P-014 (probability persistence — probs saved in checkpoint), M-025 (seed parameterization), H-005 (mid-file imports fixed)
- **Verification**: `python scripts/train.py --dataset raid --ablation baseline1 --seed 42 --epochs 1` trains 1 epoch without errors

### P0.6 — Build Kaggle Orchestrator `scripts/kaggle_run.py` (~2h)
- **New file**: `scripts/kaggle_run.py`
- **Orchestration**:
  - Uses `ProcessPoolExecutor(max_workers=N_GPUS)` — **NOT ThreadPoolExecutor**
  - Sets `CUDA_VISIBLE_DEVICES=0` / `=1` in subprocess environment
  - Parent process has NO CUDA context (`torch.cuda.is_available()` check before spawning)
  - Iterates seed×config grid, dispatches to pool
  - Defensive timer: `signal.signal(signal.SIGALRM, handler)` at 8.5h (11.5h exceeds the ~9h Kaggle session kill time). Note: SIGALRM is Unix-only — Kaggle runs Debian Linux so this is fine locally. Guard with `hasattr(signal, 'SIGALRM')` for local dry-runs on Windows.
  - Collects exit codes per subprocess; on crash, logs and continues
  - Saves `status.json` after each completion for resume
  - Serialized kagglehub.dataset_upload ONLY from parent process, at end or timeout
- **Environment auto-detection**: If `torch.cuda.device_count() == 1`, `max_workers=1` (P100 fallback)
- **Cross-session resume**: At start, downloads previous results via kagglehub, checks which seed×config pairs are complete, skips them
- **Issues resolved**: M-001 (serial GPU constraint — orchestrated parallelism), P-017 (dataset accessible — orchestrator verifies at startup)
- **Verification**: Dry-run on laptop: `python scripts/kaggle_run.py --dry-run --seeds 42 101 --configs baseline1` prints grid without launching subprocesses

### P0.7 — Fix tc4_ freeze_layers Bug (~30min)
- **File**: `tc4_.py`
- **Bug**: `DistilBertClassifier.__init__` accepts `freeze_layers=0` param but `_freeze_layers()` is NEVER called
- **Fix**: Add `self._freeze_layers()` call after `head_type` setup in constructor
- **Also fix**: Add type annotations to `forward` and `benchmark`; change `weights_only=False` → `True` (N-016); expand `POSSIBLE_PATHS` to 5 entries
- **Issues resolved**: New issue (tc4_ bug), N-016 (weights_only), P-012 (checkpoint paths)
- **Verification**: `python tc4_.py` runs and reports correct latency; model config matches trained model

### P0.8 — Delete Dead Code (~30min)
- **Files to delete**:
  - `src/training/trainer.py` (217 lines, zero consumers — structure absorbed into scripts/train.py)
  - `src/data/filter.py` (329 lines, duplicate of data/filter.py)
- **Files to clean**:
  - Remove unused imports from 4 files (H-001–H-004): `import os` / `import random` in filter scripts
  - Move `import argparse` to top of file in 3 training scripts (H-005)
  - Add `__init__.py` to `figures/` (root `data/` is a script directory, not a package — no `__init__.py` needed there). `src/data/__init__.py` already exists (confirmed).
  - Fix hardcoded user path in generate_figures.py docstring (H-006)
  - Extract hardcoded `batch_size=50_000` to named constant (H-009)
- **Issues resolved**: N-014, H-001–H-009
- **Verification**: `git diff --stat` shows deletions; `pytest .` or `python -c "import sys; sys.path.insert(0,'.'); from src.models import *"` succeeds

### P0.9 — Update requirements.txt (~30min)
- **Actions**: Scan all `import` statements across codebase, cross-ref against current requirements.txt, add missing deps with pinned versions
- **Missing deps expected**: pyarrow, bitsandbytes, kagglehub (scikit-learn==1.4.2 and transformers==4.40.0 are already pinned — verified)
- **Issues resolved**: P-016/A1e, M-040
- **Verification**: `pip install -r requirements.txt` succeeds in fresh env

---

## 3. Phase 1: Data Pipeline Fixes (~4-6h, laptop only)

### P1.1 — A0: Contamination Audit (~1h)
- **Actions**: Read both RAID and DetectRL data pipelines; trace where human-text samples enter the training split; document in `data/contamination_audit.md`
- **Issues resolved**: P-010
- **Verification**: Document at `data/contamination_audit.md` exists with data flow diagram

### P1.2 — Fix Sequential Parquet Naming (~30min)
- **File**: `data/filter_raid_sequential.py`
- **Fix**: Change output filenames from `train_pool.parquet` → `raid_train_pool.parquet` and `test_unseen.parquet` → `raid_test_unseen.parquet` to match parallel filter
- **Issues resolved**: P-008
- **Verification**: Run filter → output files have `raid_` prefix

### P1.3 — Fix Attack Column Hijack (~1h)
- **Files**: `data/filter_raid_parallel.py` L217, `data/filter_raid_sequential.py` L185
- **Bug**: Both scripts set `attack_type` column to generator value, overwriting the per-attack label
- **Fix**: Rename output column or preserve original attack_type
- **Also: A3 column consumer audit** — trace all scripts that read filtered parquet files and verify column name assumptions (P1.3b, ~30min)
- **Issues resolved**: P-007, M-005
- **Verification**: `grep -r "attack_type" data/processed/ --include="*.parquet"` shows correct values

### P1.4 — Delete Duplicate filter.py (~15min)
- **Actions**: Delete `src/data/filter.py`; `data/filter.py` is canonical
- **Issues resolved**: N-015
- **Verification**: `grep -r "filter_raid" src/data/` returns zero results for filter.py

### P1.5 — Dedup Contamination (A2) (~1h)
- **Actions**: Add `df.drop_duplicates(subset=["text"])` to the data pipeline at the point identified by A0 audit
- **Issues resolved**: P-003
- **Verification**: Before/after row count diff matches known overlap (6,029 / 5,000)

---

## 4. Phase 2a: Paper Fixes — No GPU Deps (~6-8h, laptop only)

### B1 — FDG Rename (~1.5h)
- **File rename**: `src/baselines/fast_detectgpt.py` → `src/baselines/gpt2_perplexity_baseline.py`
- **Internal audit**: Update all imports, docstrings, argparse help, log messages, comments referencing "Fast-DetectGPT" → "GPT-2 XL Perplexity"
- **Files to update**: Any script importing from fast_detectgpt; .tex references
- **Issues resolved**: P-001, M-003

### B2 — RRD Naming + Formula Consistency (~1h)
- **Actions**: Standardize RRD formula references in .tex; add `from src.evaluation.metrics import compute_rrd` to any inline formula
- **Issues resolved**: Pending verification

### B5 — Table IV Confusion Matrix Footnote (~1h)
- **Actions**: Add provenance footnote clarifying 0.9161 vs 0.9157 mismatch; no numerical changes needed

### B6 — Threshold Wording (~30min)
- **Actions**: Clarify 0.5 argmax threshold as binary sigmoid (not dual softmax) in .tex

### B8 — RRD 5% Threshold Justification (~30min)
- **Actions**: Add text acknowledging 5% RRD threshold is arbitrary and uncited; defer to Limitations section or frame as "conventional benchmark in prior work" with a qualifying note
- **Issues resolved**: P-005
- **Verify**: `grep -n "5%" Research_Paper.tex` shows qualified phrasing

### B9 — Table IV Current Stats (~1h)
- **Actions**: Verify Table IV values against current `summary.csv`; note any rounding to 4 decimal places is from single-seed evaluation
- **Note**: B5 (footnote) already addresses 0.9161 vs 0.9157 mismatch; B9 adds full-provenance clarity

---

## 5. Phase 3: Kaggle Smoke Test (~1-2h wall, Kaggle T4)

**Goal**: Verify restructured code runs correctly on target Kaggle hardware before committing to 25-30h of GPU runs.

### P3.1 — Kaggle Environment Setup (~30min)
- Upload restructured codebase as Kaggle Dataset or GitHub-sync
- Create notebook with:
  ```python
  !pip install -r requirements.txt
  !python scripts/train.py --dataset raid --ablation baseline1 --seed 42 --epochs 1 --batch-size 32 --amp
  ```
- Verify: Training completes, checkpoint saved, metrics reasonable

### P3.2 — Dual T4 Task Parallelism Test (~30min)
- Run orchestrator dry: `python scripts/kaggle_run.py --seeds 42 --configs baseline1 --dry-run`
- Run with 2 concurrent processes: 2 seeds × 1 config (verify ProcessPoolExecutor works)
- Verify: nvidia-smi shows both GPUs utilized, no crashes
- On P100 fallback: auto-detects 1 GPU, runs sequential

### P3.3 — tc4_ Benchmark Test (~15min)
- Run `python tc4_.py` against checkpoint from P3.1
- Verify: Correct latency reported, no freeze_layers mismatch

### P3.4 — Cross-Session Resume Test (~30min)
- Run orchestrator with 2 seeds
- Manually simulate session end (save results to /kaggle/working/)
- Re-run orchestrator — verify it detects completed runs and skips them

---

## 6. Phase 4: Kaggle GPU Runs (~25-30h wall, Kaggle T4)

**All GPU-intensive work. Serialized due to single Kaggle account — only one session at a time.**

**Sequencing**: G0 (Pre-G1 decision gate) runs a single-seed eval on dedup'd data (~2h) to decide whether the full 14h G1 is worth running. G2 (clean baseline training) runs in parallel with G0 on the second T4 when dual GPU is available (both independent, no data dependency). After G0 passes → G1 (multi-seed, 14h) → G3 (FDG dedup, ~4h) → G5 (calibration, ~3h). Net Phase 4 order: (G0 ∥ G2) → DECISION → G1 → G3 → G5.

### G0 — Pre-G1 Decision Gate (~2h on T4)
**Purpose**: Verify multi-seed evaluation is worth running before committing 14h.
- **Actions**: Single-seed eval on dedup'd split; compute preliminary RRD variance
- **Decision**: If single-seed RRD differs from paper claim by <2%, proceed to G1. If >2%, flag for narrative fallback.
- **Fallback**: Write contingency narrative if multi-seed falsifies central claim

### G1 — Multi-Seed Evaluation (~14h on T4)
- 5 seeds × 4 configs = 20 runs
- Dual T4 parallelism: 2 concurrent → ~10 sessions of 2 runs each
- On P100 fallback: sequential 20 runs → ~14-18h (may need 2 Kaggle sessions with resume)
- **Outputs**: 20 checkpoints with full metrics + all_probs + all_labels + all_preds
- **kagglehub upload**: End of session, tar.gz results/ → upload as new dataset version

### G2 — Clean-Only DistilBERT Baseline (~6h on T4)
**Issue**: N-001 (P1) — All training includes contaminated human texts. Need baseline trained on clean human text only.
- **Actions**: Re-train ablation_b on dedup'd RAID data (human-genuine texts only, no AI-generated training). Same seed=42, same hyperparameters. G2 must complete BEFORE G1 so G1 results can use clean baseline comparison.
- **Pre-requisite**: P1.5 (dedup) must be done before this task
- **Output**: `results/clean_baseline_ablation_b.pt`

### G3 — Fast-DetectGPT Baseline on Dedup'd Data (~4h on T4)
- **Actions**: Run renamed gpt2_perplexity_baseline.py on dedup'd data
- **Caveat**: The perplexity classification threshold must be recalibrated on dedup'd data — the original threshold was fit on contaminated data and will produce different FPR/TPR
- **Output**: results/baseline_gpt2_perplexity_dedup.json

### G5 — Calibration/ECE Analysis (~3h on T4)
- **Actions**: Load G1 checkpoints' all_probs arrays; compute Expected Calibration Error per config per seed; plot reliability diagrams
- **Pre-requisite**: P-014 (probability persistence) — resolved by P0.5 unified checkpoint format
- **Output**: Calibration metrics per config (mean ± std across seeds)

---

## 7. Phase 2b: Paper Fixes — Post-GPU (~4-6h, laptop)

### B3 — Fix Abstract AUROC Range (~1h)
- **Actions**: Update abstract AUROC values with G1 multi-seed results (mean ± std across 5 seeds)
- **Depends on**: G1 complete

### B7 — Framing Corrections with Real Results (~3-4h)
- **Actions**: Update all numerical claims: RRD values, clean baseline comparison, per-attack breakdowns, primary split designation
- **Depends on**: G1, G3, G5 complete

---

## 8. Phase 5: Analysis + Final Paper Update (~8-12h, laptop)

### P5.1 — Per-Attack Metrics (~2h)
- **Actions**: From G1 checkpoints, compute metrics broken down by individual attack type (not just seen/unseen families)
- **Output**: Per-attack tables for paper

### P5.2 — Multi-Seed Variance Analysis (~1h)
- **Actions**: Compute mean ± std across 5 seeds for all metrics; check if ablation_b superiority claim holds
- **Output**: Variance table for paper

### P5.3 — Clean Baseline Comparison (~1h)
- **Actions**: Add clean-only DistilBERT baseline (trained on human text only)
- **Depends on**: G1 checkpoints (or can re-train)

### P5.4 — Prevalence Analysis (~1h)
- **Actions**: Report metrics at low FPR (1%, 5%) to assess practical detection utility

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

**Critical path**: P0 → P3 → P4 (G0 → G1 → G3 → G5) → P2b → P5
**Parallelizable**: P1 ∥ P2a (both depend on P0 only)
**Phase 4 internal**: (G0 ∥ G2 in parallel on dual T4, G0≈2h, G2≈6h) → DECISION → G1 (14h) → G3 (4h) → G5 (3h). G0 runs single-seed eval to decide whether full G1 is worthwhile; G2 runs concurrently using the second T4. If G0 fails, G1 is skipped entirely (fallback narrative written).

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
| **P-017 (RAID accessibility)** | **DATA** | **P0** | **P3.1 (Kaggle smoke pre-check)** |
| P-018 (CUDA env) | INFRASTRUCTURE | P0 | P3.1 (Kaggle smoke test verifies) |
| P-019 (GPU thermal) | INFRASTRUCTURE | P0 | P3.1 (Kaggle env handles thermal) |
| M-001 (GPU serial) | CODE | P0 | P0.6 (orchestrator handles parallelism) |
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
| N-013 (bare except) | CODE | P2 | P0.3 (metrics consolidation fixes this) |
| N-014 (trainer.py dead) | CODE | P2 | P0.8 (delete dead code) |
| N-015 (filter.py duplicate) | CODE | P2 | P1.4 (delete duplicate) |
| N-016 (weights_only=False) | CODE | P2 | P0.7 (fix tc4 weights_only) |
| H-001–H-004 (unused imports) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-005 (mid-file imports) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-006 (hardcoded path) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-008 (missing __init__.py) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| H-009 (inline batch_size) | HOUSEKEEPING | P3 | P0.8 (cleanup pass) |
| **New**: tc4_ freeze_layers bug | CODE | P0 | P0.7 (fix before any benchmark run) |

**Coverage: 43 issues — all P0/P1/P2 from registry mapped to tasks. 4 added P0 gaps (P-002, P-004, P-005, P-015, P-017, M-002). 5 added P1 (N-001, N-002). 4 added P2 (N-003–N-006). 1 new (tc4_).**

---

## 11. Commit Strategy

| # | Commit Message | When | Files Touched |
|---|---------------|------|---------------|
| 1 | `refactor: paths config + environment detection` | After P0.1 | src/config.py (new), + all scripts updated to use it |
| 2 | `refactor: consolidate model/metrics/dataset into canonical sources` | After P0.2-0.4 | src/models/, src/evaluation/, src/data/ |
| 3 | `feat: unified scripts/train.py + scripts/kaggle_run.py` | After P0.5-0.6 | scripts/train.py (new), scripts/kaggle_run.py (new) |
| 4 | `fix: tc4_ freeze_layers bug + checkpoint paths + security` | After P0.7 | tc4_.py |
| 5 | `chore: delete dead code, fix imports, add __init__.py` | After P0.8 | Multiple files deleted/cleaned |
| 6 | `fix: parquet naming + column hijack + contamination dedup` | After P1.1-1.5 | data/filter_raid_*.py, src/data/filter.py (del) |
| 7 | `fix: FDG rename to GPT-2 XL Perplexity Baseline` | After B1 | src/baselines/ |
| 8 | `fix: paper corrections (RRD, Table IV, threshold)` | After P2a | Research_Paper.tex |
| *(Kaggle runs happen uncommitted — results checked in after)* | | | |
| 9 | `fix: abstract AUROC range + framing with final numbers` | After P4 + P2b | Research_Paper.tex |
| 10 | `feat: per-attack metrics, variance analysis, calibration, prevalence` | After P5 | Various analysis scripts + .tex |

---

## 12. Success Criteria (End State)

- [ ] Single `scripts/train.py` replaces all 4 old training scripts
- [ ] All 43 mapped issues closed + tc4_ bug fixed
- [ ] Kaggle smoke test passes (1 epoch × 1 config × 1 seed)
- [ ] Dual T4 task parallelism confirmed working (or P100 fallback confirmed)
- [ ] G2: Clean-only DistilBERT baseline complete (before G1)
- [ ] G1: 5 seeds × 4 configs complete with per-attack metrics
- [ ] G0: Decision gate passed (or contingency narrative written)
- [ ] G3: FDG baseline on dedup'd data complete
- [ ] G5: Calibration metrics per config per seed
- [ ] Paper .tex has final numbers from all GPU runs
- [ ] LSP diagnostics clean on all changed .tex and .py files
