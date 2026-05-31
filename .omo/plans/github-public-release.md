> [!CAUTION]
> **SUPERSEDED / ARCHIVED**: This plan covers an earlier GitHub release of the legacy codebase (DetectRL dataset, old script names). The project has since been restructured for RAID and Kaggle. Current reference: [`end-to-end-restructure-plan.md`](./end-to-end-restructure-plan.md).

# GitHub Public Release: AI-Generated Text Detection — DistilBERT Ablation Study

## TL;DR

> **Quick Summary**: Prepare the DistilBERT ablation study repository for public GitHub release. Fix a config-label bug in the module code, clean up large data/model files, write a comprehensive README.md, and initialize git with GitHub push.
>
> **Deliverables**:
> - Bug fix applied to `src/models/distilbert_classifier.py`
> - Surgical `.gitignore` excluding `.pt`, `.parquet`, `__pycache__/`, `.venv/`, internal planning dirs
> - Stale files removed (`CodeFiles.zip`, broken module-pipeline scripts, `__pycache__/`)
> - `src/__init__.py` added (package importability)
> - `LICENSE` file (Apache 2.0)
> - `README.md` with project description, key results, ablation table, reproduction steps, citation
> - Git repo initialized, committed, pushed to `github.com/The-Immortal-Wanderer/ai-text-detection-ablation`
>
> **Estimated Effort**: Medium (~1-2 hours)
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Task 1 (bug fix) → Task 3 (.gitignore) → Task 5 (README) → Task 7 (git init + commit)

---

## Context

### Original Request
Prepare the repository at `C:\Users\madha\source\repos\ANN_Project` for public GitHub release. The project is an ablation study on DistilBERT for AI-generated text detection, accompanying a research paper in preparation for IEEE submission.

### Interview Summary
**Key Discussions**:
- **Bug fix**: `src/models/distilbert_classifier.py:get_model_config()` has ablation_a and ablation_b configs reversed. Single standalone scripts have their OWN correct inline version. Module fix aligns with paper's Table `tab:configs`.
- **License**: Apache 2.0 selected (industry standard for ML research code).
- **Data strategy**: ALL parquet data (~875MB) excluded from repo. Users download via provided scripts (download links in README).
- **Module pipeline**: `src/training/train.py`, `src/training/grid_search.py`, `src/evaluation/evaluator.py` — broken paths, never executed — removed from repo. Paper .tex does NOT reference any file paths.
- **Training time correction**: baseline1/ablation_c = ~88 min, ablation_a/ablation_b = ~53 min per ablation (NOT "~30 min").

**Research Findings**:
- Paper .tex confirms 2×2 ablation design: baseline1 (single, unfrozen), ablation_a (single, freeze 0-3), ablation_b (deep, freeze 0-3) ← proposed, ablation_c (deep, unfrozen).
- Paper line 1002: "\texttt{ablation\_a} (single head, freeze=4)" and "\texttt{ablation\_b} (deep head, freeze=4)" — confirms correct mapping.
- summary.csv confirms trained results match CORRECT mapping (standalone scripts were used for actual training).
- Best config: ablation_b (deep head, freeze 0-3) with best_val_f1=0.9488.
- Paper does NOT reference any file paths — removals safe.

### Metis Review
**Identified Gaps** (addressed):
- **LICENSE**: Apache 2.0 selected by user.
- **Parquet data**: ALL parquet excluded (user opted for download-links approach).
- **Module pipeline**: 3 broken files removed (paper doesn't reference them).
- **src/__init__.py**: Added (self-resolved, trivial).
- **Training time**: Corrected to actual values from training_times.json.
- **Internal files**: .sisyphus/ and .github/plans/ excluded from git.

---

## Work Objectives

### Core Objective
Prepare the repository for public GitHub release — fix a critical config bug, clean up large files, document the project comprehensively, and initialize version control.

### Concrete Deliverables
1. Corrected `src/models/distilbert_classifier.py` (ablation_a/ablation_b configs fixed)
2. `src/__init__.py` (package importability)
3. `.gitignore` (surgical exclusions for large files + internal dirs)
4. Deleted: `CodeFiles.zip`, `src/training/train.py`, `src/training/grid_search.py`, `src/evaluation/evaluator.py`, all `__pycache__/` dirs
5. `LICENSE` (Apache 2.0)
6. `README.md` (comprehensive project documentation)
7. Git repository initialized, committed, pushed to GitHub

### Definition of Done
- [ ] `grep -A 5 "ablation_a" src/models/distilbert_classifier.py` shows correct config: single head, freeze_layers=4
- [ ] `Get-ChildItem -Recurse -Filter "*.pt" | Measure-Object | % {$_.Count}` = 0
- [ ] `Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Measure-Object | % {$_.Count}` = 0
- [ ] `Test-Path CodeFiles.zip` = False
- [ ] `Test-Path src/__init__.py` = True
- [ ] `Test-Path LICENSE` = True
- [ ] `Get-Content .gitignore` contains all required exclusion patterns
- [ ] `README.md` contains: project description, key results, ablation table, dataset info, reproduction steps, hardware notes, license, citation
- [ ] `git rev-parse --is-inside-work-tree` = true
- [ ] `git log --oneline` shows at least 1 commit
- [ ] `git ls-files` — no .pt, .parquet, .venv, __pycache__ files staged
- [ ] All files staged are < 50MB

### Must Have
- [ ] `get_model_config()` labels match paper's Table `tab:configs`
- [ ] No .pt, .parquet files committed
- [ ] README accurately reflects training times from training_times.json
- [ ] `.gitignore` excludes: `*.pt`, `*.parquet`, `__pycache__/`, `.venv/`, `checkpoints/`, `.sisyphus/`, `.github/plans/`, `CodeFiles.zip`

### Must NOT Have (Guardrails)
- [ ] No modification to standalone training scripts (train_distilbert_detectrl.py, train_distilbert_parallel.py, tc3_traindistilbert.py, tc4_.py)
- [ ] No modification to Research_Paper.tex
- [ ] No refactoring/deduplication of model code
- [ ] No test infrastructure added
- [ ] No files > 50MB staged for commit
- [ ] No .py files in `data/` removed (keep download/filter scripts)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None (not in scope)
- **Framework**: N/A
- **Agent-executed QA**: ALWAYS — every task has concrete verification commands

### QA Policy
Every task includes agent-executed QA scenarios. Evidence format described per task.

- **Code verification**: grep/Select-String to read patched lines and assert expected values
- **File system checks**: Test-Path, Get-ChildItem, Measure-Object for existence/count
- **Git verification**: git commands for repo state
- **README structure**: Get-Content with regex assertions on section headers

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Bug Fix + Cleanup Prep — serial, rapid):
├── Task 1: Fix ablation_a/ablation_b config bug [quick]
└── Task 2: Create src/__init__.py + LICENSE + delete CodeFiles.zip + clean __pycache__ [quick]

Wave 2 (MAX PARALLEL — independent config + doc + removal):
├── Task 3: Create surgical .gitignore [quick]
├── Task 4: Remove broken module-pipeline scripts (train.py, grid_search.py, evaluator.py) [quick]
└── Task 5: Write comprehensive README.md [writing]

Wave 3 (Git + GitHub — sequential dependency):
├── Task 6: git init, add, commit (verify no large files staged) [quick]
├── Task 7: Push to GitHub (manual if gh CLI unavailable) [quick]
└── Task 8: Final verification sweep [quick]

Critical Path: Task 1 → Task 3 → Task 5 → Task 6 → Task 7 → Task 8
Max Concurrent: 3 (Wave 2)
```

---

## TODOs

- [x] 1. Fix ablation_a/ablation_b config mapping in `src/models/distilbert_classifier.py`

  **What to do**:
  - Edit `src/models/distilbert_classifier.py` lines 58-59:
    - Change `"ablation_a": {"head_type": "deep", "freeze_layers": 0}` → `"ablation_a": {"head_type": "single", "freeze_layers": 4}`
    - Change `"ablation_b": {"head_type": "single", "freeze_layers": 4}` → `"ablation_b": {"head_type": "deep", "freeze_layers": 4}`
  - The corrected 2×2 design becomes:
    - `baseline1`: single head, no freezing (fully fine-tuned)
    - `ablation_a`: single head, freeze layers 0-3 (partially frozen, shallow head)
    - `ablation_b`: deep head, freeze layers 0-3 (partially frozen, deep head) ← paper's proposed config
    - `ablation_c`: deep head, no freezing (fully fine-tuned, deep head)
  - Do NOT modify any other part of the file (model class, forward, etc.)
  - Verify the fix matches the paper's Table `tab:configs` (Research_Paper.tex lines 408-411)

  **Must NOT do**:
  - Do NOT modify standalone training scripts (they have their own correct inline version)
  - Do NOT refactor the class or deduplicate code
  - Do NOT modify any other function in the file

  **Recommended Agent Profile**:
  - **Category**: `quick` — single file, 2-line change, well-defined
  - **Skills**: `[]` — no specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation task)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 5 (README mentions ablation configs), 6-8 (git)
  - **Blocked By**: None

  **References**:
  - `Research_Paper.tex:408-411` — Table `tab:configs` showing the correct ablation mapping:
    - ablation_a: Single head, frozen 0-3 (4 of 6)
    - ablation_b: Deep head, frozen 0-3 (4 of 6) ← proposed
  - `Research_Paper.tex:1002-1005` — Prose confirming: "\texttt{ablation\_a} (single head, freeze=4)"
  - `src/models/distilbert_classifier.py:55-61` — Current (buggy) function to fix
  - `train_distilbert_detectrl.py:114-118` — Reference: correct mapping used in actual training

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY)**:

  ```
  Scenario: Verify ablation_a config after fix
    Tool: Bash (Select-String)
    Preconditions: File src/models/distilbert_classifier.py has been edited
    Steps:
      1. Select-String -Path "src/models/distilbert_classifier.py" -Pattern "ablation_a"
    Expected Result: Output must contain: "ablation_a": {"head_type": "single", "freeze_layers": 4}
    Failure Indicators: Output shows head_type="deep" or freeze_layers=0
    Evidence: .sisyphus/evidence/task-1-ablation-a-fix.txt

  Scenario: Verify ablation_b config after fix
    Tool: Bash (Select-String)
    Preconditions: File has been edited
    Steps:
      1. Select-String -Path "src/models/distilbert_classifier.py" -Pattern "ablation_b"
    Expected Result: Output must contain: "ablation_b": {"head_type": "deep", "freeze_layers": 4}
    Failure Indicators: Output shows head_type="single" or freeze_layers=0
    Evidence: .sisyphus/evidence/task-1-ablation-b-fix.txt

  Scenario: Verify full get_model_config function
    Tool: Bash (Select-String)
    Preconditions: File has been edited
    Steps:
      1. Select-String -Path "src/models/distilbert_classifier.py" -Pattern "ablation_a|ablation_b|ablation_c|baseline1" -Context 0,0
    Expected Result: All 4 configs present with correct values matching paper's Table `tab:configs`
    Failure Indicators: Any config has wrong head_type or freeze_layers
    Evidence: .sisyphus/evidence/task-1-full-config.txt
  ```

  **Evidence to Capture**:
  - [ ] task-1-ablation-a-fix.txt
  - [ ] task-1-ablation-b-fix.txt
  - [ ] task-1-full-config.txt

  **Commit**: YES
  - Message: `fix: correct ablation_a/ablation_b config mapping in src/models/distilbert_classifier.py`
  - Files: `src/models/distilbert_classifier.py`

---

- [x] 2. Create `src/__init__.py`, `LICENSE` (Apache 2.0), delete `CodeFiles.zip`, clean `__pycache__` directories

  **What to do**:
  1. Create `src/__init__.py` with content: `"""Source package for AI-Generated Text Detection — DistilBERT Ablation Study."""`
  2. Create `LICENSE` file with Apache License 2.0 full text
  3. Delete `CodeFiles.zip` from project root
  4. Find and delete all `__pycache__/` directories recursively:
     - `data/__pycache__/`
     - `src/data/__pycache__/`
     - `src/models/__pycache__/`
     - `src/training/__pycache__/`

  **Must NOT do**:
  - Do NOT modify any existing `.py` files
  - Do NOT create any directories that don't already exist in the structure
  - Do NOT touch `src/evaluation/__pycache__/` (evaluator.py is being removed in Task 4, its __pycache__ will go with it)

  **Recommended Agent Profile**:
  - **Category**: `quick` — file creation + deletions, well-defined
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: None

  **References**:
  - Standard Apache 2.0 license text: https://www.apache.org/licenses/LICENSE-2.0.txt
  - `src/models/__init__.py:1` — Pattern for `__init__.py` docstring style

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Verify src/__init__.py exists and has content
    Tool: Bash (Test-Path, Get-Content)
    Preconditions: File created
    Steps:
      1. Test-Path "src/__init__.py"
      2. Get-Content "src/__init__.py"
    Expected Result: File exists, contains non-empty docstring
    Evidence: .sisyphus/evidence/task-2-init-py.txt

  Scenario: Verify LICENSE exists
    Tool: Bash (Test-Path)
    Preconditions: File created
    Steps:
      1. Test-Path "LICENSE"
      2. Get-Content "LICENSE" -TotalCount 3
    Expected Result: File exists, first line contains "Apache License"
    Evidence: .sisyphus/evidence/task-2-license.txt

  Scenario: Verify CodeFiles.zip deleted
    Tool: Bash (Test-Path)
    Preconditions: File deleted
    Steps:
      1. Test-Path "CodeFiles.zip"
    Expected Result: False (file does not exist)
    Evidence: .sisyphus/evidence/task-2-codefiles-deleted.txt

  Scenario: Verify no __pycache__ directories remain
    Tool: Bash (Get-ChildItem)
    Preconditions: Directories deleted
    Steps:
      1. Get-ChildItem -Recurse -Directory -Filter "__pycache__"
    Expected Result: No output (0 directories found)
    Evidence: .sisyphus/evidence/task-2-no-pycache.txt
  ```

  **Evidence to Capture**:
  - [ ] task-2-init-py.txt
  - [ ] task-2-license.txt
  - [ ] task-2-codefiles-deleted.txt
  - [ ] task-2-no-pycache.txt

  **Commit**: YES (groups with Task 3, 4)
  - Message: `chore: add src/__init__.py, LICENSE (Apache 2.0), clean stale files and __pycache__`
  - Files: `src/__init__.py`, `LICENSE`, `CodeFiles.zip` (delete)

---

- [x] 3. Create surgical `.gitignore` for public release

  **What to do**:
  - Create `.gitignore` at project root with the following patterns:

  ```
  # Python
  __pycache__/
  *.py[cod]
  *.egg-info/
  .eggs/
  dist/
  build/

  # Environments
  .venv/
  .env
  .env.local

  # IDE
  .vscode/
  .idea/
  *.swp
  *.swo

  # Model checkpoints (large weight files)
  *.pt
  *.pth
  *.bin
  *.safetensors
  checkpoints/
  artifacts/**/*.pt

  # Data files (exclude ALL parquet — users download via scripts)
  *.parquet
  *.arrow
  data/raw/
  data/processed/

  # Keep data scripts in data/ (not excluded by the above)
  !data/*.py

  # Keep artifact summaries
  !artifacts/**/*.csv
  !artifacts/**/*.json
  !artifacts/**/*.txt

  # Cache
  .pytest_cache/
  .mypy_cache/
  .ruff_cache/

  # Archives
  CodeFiles.zip

  # Internal planning (not for public)
  .sisyphus/
  .github/plans/

  # OS
  Thumbs.db
  .DS_Store
  ```

  **Must NOT do**:
  - Do NOT accidentally exclude `*.csv`, `*.json`, `*.txt` from artifacts/ (needed for result summaries)
  - Do NOT exclude data download scripts (`data/*.py`)
  - Do NOT exclude `artifacts/pdc/` files (JSON summaries)

  **Recommended Agent Profile**:
  - **Category**: `quick` — single file creation, well-defined
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 1, 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 6 (git add depends on .gitignore)
  - **Blocked By**: None

  **References**:
  - `data/` directory structure (has both `.py` scripts and `.parquet` data)
  - `artifacts/` directory structure (has `.pt` files to exclude, `.csv`/`.json`/`.txt` to keep)

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Verify .gitignore covers critical patterns
    Tool: Bash (Select-String)
    Preconditions: .gitignore created
    Steps:
      1. $patterns = @("\.pt$", "\.parquet$", "__pycache__", "\.venv", "checkpoints", "\.sisyphus", "CodeFiles", "\.github/plans")
      2. $patterns | ForEach-Object { Select-String -Path ".gitignore" -Pattern $_ | Select-Object -First 1 }
    Expected Result: All 8 patterns found in .gitignore
    Failure Indicators: Any pattern missing from .gitignore
    Evidence: .sisyphus/evidence/task-3-gitignore-patterns.txt

  Scenario: Verify artifact summaries NOT excluded
    Tool: Bash (Select-String)
    Preconditions: .gitignore created
    Steps:
      1. Select-String -Path ".gitignore" -Pattern "!artifacts.*\.csv"
    Expected Result: Negation patterns exist for .csv, .json, .txt in artifacts/
    Evidence: .sisyphus/evidence/task-3-gitignore-negations.txt

  Scenario: Verify data scripts NOT excluded
    Tool: Bash (Select-String)
    Preconditions: .gitignore created
    Steps:
      1. Select-String -Path ".gitignore" -Pattern "!data/\*\.py"
    Expected Result: Negation pattern exists for .py in data/
    Evidence: .sisyphus/evidence/task-3-gitignore-data-scripts.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-gitignore-patterns.txt
  - [ ] task-3-gitignore-negations.txt
  - [ ] task-3-gitignore-data-scripts.txt

  **Commit**: YES (groups with Task 2, 4)
  - Files: `.gitignore`

---

- [x] 4. Remove broken module-pipeline scripts

  **What to do**:
  - Delete these files (they have broken paths, never executed, paper doesn't reference them):
    - `src/training/train.py`
    - `src/training/grid_search.py`
    - `src/evaluation/evaluator.py`
  - Keep their companion `__init__.py` files (they are just docstrings or blank)
  - Keep `src/training/__init__.py` and `src/evaluation/__init__.py`

  **Rationale**: These scripts reference `checkpoints/` dir (doesn't exist), `results/` (wrong path), and the evaluator would crash. All actual training used the standalone scripts (`train_distilbert_detectrl.py`, `train_distilbert_parallel.py`, `tc3_traindistilbert.py`). Removing prevents confusion.

  **Must NOT do**:
  - Do NOT remove `src/training/__init__.py` or `src/evaluation/__init__.py`
  - Do NOT remove standalone training scripts at project root

  **Recommended Agent Profile**:
  - **Category**: `quick` — file deletions, well-defined
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 1, 2, 3)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6 (git add)
  - **Blocked By**: None

  **References**:
  - Verification that paper doesn't reference these files (confirmed in §54-§56 of session transcript)
  - `src/training/__init__.py` — keep this
  - `src/evaluation/__init__.py` — keep this

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Verify train.py removed
    Tool: Bash (Test-Path)
    Preconditions: File deleted
    Steps:
      1. Test-Path "src/training/train.py"
    Expected Result: False
    Evidence: .sisyphus/evidence/task-4-train-removed.txt

  Scenario: Verify evaluator.py removed
    Tool: Bash (Test-Path)
    Preconditions: File deleted
    Steps:
      1. Test-Path "src/evaluation/evaluator.py"
    Expected Result: False
    Evidence: .sisyphus/evidence/task-4-evaluator-removed.txt

  Scenario: Verify __init__.py files preserved
    Tool: Bash (Test-Path)
    Preconditions: Deletion done
    Steps:
      1. Test-Path "src/training/__init__.py"
      2. Test-Path "src/evaluation/__init__.py"
    Expected Result: Both True (still exist)
    Evidence: .sisyphus/evidence/task-4-init-preserved.txt
  ```

  **Evidence to Capture**:
  - [ ] task-4-train-removed.txt
  - [ ] task-4-evaluator-removed.txt
  - [ ] task-4-init-preserved.txt

  **Commit**: YES (groups with Task 2, 3)
  - Message: `chore: remove broken module-pipeline scripts (unused; all training used standalone scripts)`
  - Files: `src/training/train.py` (delete), `src/training/grid_search.py` (delete), `src/evaluation/evaluator.py` (delete)

---

- [x] 5. Write comprehensive README.md

  **What to do**:
  - Create `README.md` at project root with the following structure and content:

  ```markdown
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
  │   └── evaluation/           # Metrics (evaluator removed — use standalone scripts)
  ├── data/                     # Data download and preprocessing scripts
  │   ├── download_detectrl_HC3.py
  │   ├── download_raid_raw.py
  │   ├── filter.py
  │   └── ...
  ├── artifacts/                # Training checkpoints and result summaries
  │   └── distilbert_detector/
  │       ├── summary.csv       # ✅ Consolidated results
  │       ├── best_config.json  # ✅ Best config metadata
  │       └── training_times.json
  ├── figures/                  # Publication figures (PNG + interactive HTML)
  ├── results ss/               # Experimental run logs
  ├── train_distilbert_detectrl.py   # ✅ Main training entry point
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
  @inproceedings{masood2025towards,
    title={Towards Adversarially Augmented {AI} Text Detection via Supervised {DistilBERT} Fine-Tuning: A Held-Out-Generator Study on {RAID}},
    author={Masood, Hammad and Rafay, Abdul},
    booktitle={Proceedings of the...},
    year={2025},
    note={In preparation for IEEE submission}
  }
  ```

  ## License

  This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

  ## Acknowledgments

  - **Abdul Rafay** — Co-author and collaborator on the research paper and experiments
  - FAST-NUCES, Islamabad — Academic affiliation and support
  - The DetectRL team for the adversarial detection benchmark
  ```

  **Must NOT do**:
  - Do NOT claim results that aren't backed by summary.csv
  - Do NOT include reproduction commands that would fail (e.g., running removed module scripts)
  - Do NOT include the incorrect "~30 min per ablation" time — use actual times from training_times.json
  - Do NOT add emojis unless spacing/style demands it

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with cleanup tasks 1-4)
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 6 (git add)
  - **Blocked By**: Task 1 (should reflect correct ablation naming), Task 4 (should not reference removed files)

  **References**:
  - `Research_Paper.tex` — Full paper for accurate metric values and ablation descriptions
  - `artifacts/distilbert_detector/summary.csv` — Actual metric values for "Key Results" section
  - `artifacts/distilbert_detector/training_times.json` — Actual training durations
  - `artifacts/distilbert_detector/best_config.json` — Best config identification
  - `environment.yml` — Conda dependency specification
  - `requirements.txt` — pip dependency specification

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Verify README has all required sections
    Tool: Bash (Select-String)
    Preconditions: README.md created
    Steps:
      1. $sections = @("Overview", "Key Results", "Ablation Config", "Dataset", "Reproduction", "Hardware", "License", "Acknowledgments")
      2. $sections | ForEach-Object { Select-String -Path "README.md" -Pattern $_ | Select-Object -First 1 }
    Expected Result: All 8 sections found
    Failure Indicators: Any section missing
    Evidence: .sisyphus/evidence/task-5-readme-sections.txt

  Scenario: Verify training times match actual data
    Tool: Bash (Select-String)
    Preconditions: README.md created
    Steps:
      1. Select-String -Path "README.md" -Pattern "~88 minutes|~53 minutes"
    Expected Result: Correct times present (not "~30 min")
    Failure Indicators: README contains wrong training time estimates
    Evidence: .sisyphus/evidence/task-5-readme-times.txt

  Scenario: Verify project name and paper status in README
    Tool: Bash (Select-String)
    Preconditions: README.md created
    Steps:
      1. Select-String -Path "README.md" -Pattern "DistilBERT"
      2. Select-String -Path "README.md" -Pattern "IEEE"
    Expected Result: Both found
    Evidence: .sisyphus/evidence/task-5-readme-title.txt
  ```

  **Evidence to Capture**:
  - [ ] task-5-readme-sections.txt
  - [ ] task-5-readme-times.txt
  - [ ] task-5-readme-title.txt

  **Commit**: YES
  - Message: `docs: add comprehensive README.md with ablation study overview and reproduction guide`
  - Files: `README.md`

---

- [x] 6. Initialize git, add files, create initial commit

  **What to do**:
  1. `git init` in project root
  2. `git add .` (respects .gitignore — verify no large files staged)
  3. Verify no `.pt`, `.parquet`, `__pycache__/`, `.venv/` files are staged:
     ```powershell
     git ls-files | ForEach-Object { $f = $_; $s = (Get-Item $_).Length; if ($s -gt 50MB) { Write-Output "WARNING: $f is $($s/1MB) MB" } }
     ```
  4. `git commit -m "Initial release: DistilBERT ablation study for AI-generated text detection with cross-attack generalization evaluation"`

  **Must NOT do**:
  - Do NOT stage .pt, .parquet, or __pycache__ files
  - Do NOT run git hooks (pre-commit hooks not set up — not needed)
  - Do NOT push yet (Task 7 covers pushing)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all previous tasks)
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1-5

  **References**:
  - `.gitignore` — Exclusion rules for git add
  - User-specified commit message (provided in original request)

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Verify git repo initialized
    Tool: Bash (git rev-parse)
    Preconditions: git init run
    Steps:
      1. git rev-parse --is-inside-work-tree
    Expected Result: true
    Failure Indicators: "fatal: not a git repository"
    Evidence: .sisyphus/evidence/task-6-git-init.txt

  Scenario: Verify initial commit exists with correct message
    Tool: Bash (git log)
    Preconditions: git commit run
    Steps:
      1. git log --oneline --format="%s"
    Expected Result: First line starts with "Initial release: DistilBERT ablation study"
    Evidence: .sisyphus/evidence/task-6-git-commit.txt

  Scenario: Verify no large files staged
    Tool: Bash (ForEach-Object size check)
    Preconditions: git add run
    Steps:
      1. git ls-files | ForEach-Object { $f = $_; $s = (Get-Item $_).Length; if ($s -gt 50MB) { Write-Output "WARNING: $f" } }
    Expected Result: No output (no files > 50MB)
    Failure Indicators: Any files > 50MB reported
    Evidence: .sisyphus/evidence/task-6-no-large-files.txt

  Scenario: Verify no .pt or .parquet files staged
    Tool: Bash (git ls-files with filter)
    Preconditions: git add run
    Steps:
      1. git ls-files "*.pt" "*.parquet"
    Expected Result: No output (no excluded file types staged)
    Failure Indicators: Any .pt or .parquet files listed
    Evidence: .sisyphus/evidence/task-6-no-excluded-types.txt
  ```

  **Evidence to Capture**:
  - [ ] task-6-git-init.txt
  - [ ] task-6-git-commit.txt
  - [ ] task-6-no-large-files.txt
  - [ ] task-6-no-excluded-types.txt

  **Commit**: YES (the initial commit itself)

---

- [x] 7. Create GitHub repository and push

  **What to do**:
  - **Attempt** to create GitHub repo via `gh` CLI:
    ```powershell
    gh repo create The-Immortal-Wanderer/ai-text-detection-ablation --public --source=. --remote=origin --push
    ```
  - **If `gh` CLI is not available** (confirmed not installed on this machine), provide detailed manual instructions in the execution log:
    1. Go to https://github.com/The-Immortal-Wanderer/ai-text-detection-ablation
    2. Create a new repository named `ai-text-detection-ablation` (public, no README, no .gitignore, no license)
    3. Run:
       ```powershell
       git remote add origin https://github.com/The-Immortal-Wanderer/ai-text-detection-ablation.git
       git branch -M main
       git push -u origin main
       ```
    4. Verify push succeeded: `git log --oneline --all`

  **Must NOT do**:
  - Do NOT create a local .gitignore or README via GitHub UI (already created)
  - Do NOT force push

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 8
  - **Blocked By**: Task 6

  **References**:
  - User-specified repo URL: `github.com/The-Immortal-Wanderer/ai-text-detection-ablation`

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Verify remote configured
    Tool: Bash (git remote -v)
    Preconditions: git remote add run
    Steps:
      1. git remote -v
    Expected Result: Shows origin → github.com/The-Immortal-Wanderer/ai-text-detection-ablation
    Failure Indicators: No remote, or wrong URL
    Evidence: .sisyphus/evidence/task-7-remote.txt

  Scenario: Verify push completed (or documented)
    Tool: Bash (git log --all)
    Preconditions: Push attempted
    Steps:
      1. git log --oneline --all
    Expected Result: Shows at least 1 commit
    Evidence: .sisyphus/evidence/task-7-push-status.txt
  ```

  **Evidence to Capture**:
  - [ ] task-7-remote.txt
  - [ ] task-7-push-status.txt

  **Commit**: NO (git meta operation)

---

- [x] 8. Final verification sweep

  **What to do**:
  - Run ALL verification commands from the "Success Criteria" section at the top of this plan
  - Compile results into a verification report
  - Fix any issues found (unexpected files staged, missing patterns, etc.)

  **Must NOT do**:
  - Do NOT skip any verification command
  - Do NOT mark complete if any verification fails — fix first

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final)
  - **Blocks**: Nothing (final task)
  - **Blocked By**: Task 6

  **References**:
  - "Success Criteria" section of this plan for all commands

  **Acceptance Criteria**:

  **QA Scenarios**:

  ```
  Scenario: Run complete verification suite
    Tool: Bash (all verification commands)
    Preconditions: All previous tasks complete
    Steps:
      1. Run bug fix verification (Select-String on distilbert_classifier.py)
      2. Run cleanup verification (Get-ChildItem for .pt, __pycache__, .parquet)
      3. Run infrastructure verification (Test-Path for __init__.py, LICENSE, .gitignore, README)
      4. Run git verification (git rev-parse, git log, git ls-files)
      5. Run gitignore coverage verification (Select-String on .gitignore)
    Expected Result: All assertions pass
    Failure Indicators: Any assertion fails
    Evidence: .sisyphus/evidence/task-8-verification-sweep.txt
  ```

  **Evidence to Capture**:
  - [ ] task-8-verification-sweep.txt

  **Commit**: NO (verification only)

---

## Final Verification Wave

> 2 review steps run in PARALLEL. ALL must APPROVE. Present consolidated results. Get explicit user "okay" before marking complete.

- [x] V1. **Plan Compliance Audit** — `quick`
  - Verify ALL "Must Have" items are satisfied:
    - [ ] `get_model_config()` labels match paper's Table `tab:configs` (grep the patched file)
    - [ ] No `.pt` files committed (git ls-files "*.pt" = empty)
    - [ ] No `.parquet` files committed (git ls-files "*.parquet" = empty)
    - [ ] README uses correct training times from training_times.json
    - [ ] `.gitignore` contains all required patterns
  - Verify ALL "Must NOT Have" items:
    - [ ] No standalone training scripts modified (check git diff against committed)
    - [ ] Research_Paper.tex unchanged
    - [ ] No files > 50MB in staging
    - [ ] No `.pt`, `.parquet` in staging
    - [ ] No refactoring/deduplication present
  - Read each modified file to verify content quality
  - Run ALL verification commands from Success Criteria section
  - Output: `Must Have [N/N] | Must NOT Have [N/N] | VERDICT: APPROVE/REJECT`

- [x] V2. **Scope Fidelity Check** — `quick`
  - For each task (1-8): read "What to do", verify actual output matches spec
  - Verify 1:1 correspondence — everything in spec was done, nothing beyond spec was done
  - Check for cross-task contamination (e.g., Task 1 modifying other parts of the file)
  - Verify paper `.tex` has zero changes (git diff Research_Paper.tex)
  - Check no unaccounted files were created/modified beyond the plan
  - Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT: APPROVE/REJECT`

---

## Commit Strategy

- **1**: `fix: correct ablation_a/ablation_b config mapping in module code`
  Files: `src/models/distilbert_classifier.py`
- **2**: `chore: add src/__init__.py, LICENSE (Apache 2.0), clean up stale files and __pycache__`
  Files: `src/__init__.py`, `LICENSE`, `CodeFiles.zip` (delete)
- **3**: `chore: add .gitignore for public release`
  Files: `.gitignore`
- **4**: `chore: remove broken module-pipeline scripts (unused, paper uses standalone scripts)`
  Files: `src/training/train.py`, `src/training/grid_search.py`, `src/evaluation/evaluator.py`
- **5**: `docs: add comprehensive README.md with ablation study overview and reproduction guide`
  Files: `README.md`
- **6**: `chore: initial git init and repository setup`
  Files: (git meta)

---

## Success Criteria

### Verification Commands (all must pass)
```powershell
# Bug fix verification
Select-String -Path "src/models/distilbert_classifier.py" -Pattern "ablation_a|ablation_b"
# Expected: ablation_a → head_type="single", freeze_layers=4
# Expected: ablation_b → head_type="deep", freeze_layers=4

# Cleanup verification
Get-ChildItem -Recurse -Filter "*.pt" -Path . | Measure-Object | % { $_.Count }
# Expected: 0
Get-ChildItem -Recurse -Directory -Filter "__pycache__" -Path . | Measure-Object | % { $_.Count }
# Expected: 0
Get-ChildItem -Recurse -Filter "*.parquet" -Path . | Measure-Object | % { $_.Count }
# Expected: 0
Test-Path "CodeFiles.zip"
# Expected: False
Test-Path "src/training/train.py"
# Expected: False
Test-Path "src/training/grid_search.py"
# Expected: False
Test-Path "src/evaluation/evaluator.py"
# Expected: False

# Infrastructure verification
Test-Path "src/__init__.py"
# Expected: True
Test-Path "LICENSE"
# Expected: True
Test-Path ".gitignore"
# Expected: True
Test-Path "README.md"
# Expected: True

# Git verification
git rev-parse --is-inside-work-tree
# Expected: true
git log --oneline
# Expected: at least 1 commit
git ls-files | ForEach-Object { $f = $_; $s = (Get-Item $_).Length; if ($s -gt 50MB) { Write-Output "WARNING: $f is $($s/1MB)N2 MB" } }
# Expected: no output

# Gitignore coverage
Get-Content ".gitignore" | Select-String "\.pt$", "\.parquet$", "__pycache__", "\.venv", "checkpoints", "\.sisyphus", "CodeFiles"
# Expected: all 7 patterns present
```

### Final Checklist
- [ ] All "Must Have" items satisfied
- [ ] All "Must NOT Have" items verified absent
- [ ] Bug fix confirmed matching paper's Table `tab:configs`
- [ ] No large files in git staging
- [ ] No standalone training scripts modified
- [ ] Paper .tex unchanged
- [ ] All verification commands pass
