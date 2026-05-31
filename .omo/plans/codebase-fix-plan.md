> [!CAUTION]
> **SUPERSEDED**: The 18 issues from this plan were absorbed into the [`end-to-end-restructure-plan.md`](./end-to-end-restructure-plan.md) (P0.5, P0.6, P0.8). All active work references the end-to-end plan.

# Codebase Fix Plan — Hyperplan-Derived (18 Issues)

**Source**: 4-critic hyperplan adversarial evaluation of ANN_Project codebase structural health
**Date**: 2026-05-22
**Parent Plan**: `.omo/plans/end-to-end-restructure-plan.md` (now SUPERSEDED by it)
**Total Issues**: 18 (8 Tier 0 Critical + 10 Tier 1 Major)
**Files Touched**: 11 unique source files across 4 work packages

---

## Dependency Graph

```
Wave 1 (parallel — no inter-dependencies):
├── WP1: Model + Training Core    [5 files, ~8 issues]
├── WP3: Config System            [1 file, 2 issues]
└── WP4: Package Hygiene          [2 files, 3 issues]

Wave 2 (after WP1):
└── WP2: Data + Eval Pipeline     [3 files, ~5 issues]
```

## Work Packages

---

### WP1: Model + Training Core

**Files**: `src/models/distilbert_classifier.py`, `src/training/trainer.py`, `scripts/benchmark.py`, `scripts/train.py`, `src/models/__init__.py`
**Issues**: C2, C4, C5, M5, M10, C6, M9, M3
**Dependencies**: None (Wave 1)

#### C2 — Fix embeddings always frozen

**File**: `src/models/distilbert_classifier.py:44-46`
**Change**: Gate embedding freeze on `freeze_layers > 0`:
```python
def _freeze_layers(self) -> None:
    if self.freeze_layers > 0:
        for parameter in self.distilbert.embeddings.parameters():
            parameter.requires_grad = False
    for layer_index, layer in enumerate(self.distilbert.transformer.layer):
        requires_grad = layer_index >= self.freeze_layers
        for parameter in layer.parameters():
            parameter.requires_grad = requires_grad
```
**Verification**: `freeze_layers=0` → embeddings trainable; `freeze_layers=4` → embeddings + L0-3 frozen

#### C4 + M10 — Remove `get_model_config()`, use `AblationsConfig` directly

**File**: `src/models/distilbert_classifier.py:60-69`, `src/models/__init__.py`, `src/training/trainer.py:163-165`
**Changes**:
1. Delete `get_model_config()` function from `distilbert_classifier.py`
2. Remove `get_model_config` from `src/models/__init__.py` exports
3. `trainer.py`: Update `train_ablation()` signature to accept `head_type` + `freeze_layers` directly (remove ablation_name → config lookup). Remove lazy import of `get_model_config`. Move `from src.models import DistilBertClassifier` to top of module.
4. `train.py`: Update call sites (L83-88, L133) to read `head_type` and `freeze_layers` from `cfg.ablations`

#### C5 — Import `DistilBertClassifier` from `src.models` in benchmark.py

**File**: `scripts/benchmark.py:41-68`
**Change**: Delete inline class `DistilBertClassifier`. Add `from src.models import DistilBertClassifier` at module top.

#### C6 — Normalize metric key to `f1_macro`

**File**: `src/training/trainer.py:71-90`
**Change**: `_compute_metrics` → use `average='macro'`, store key as `"f1_macro"`:
```python
"f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
```
**Downstream**:
- `trainer.py:191` — `val_metrics["f1"]` → `val_metrics["f1_macro"]` (best model selection key)
- `train.py:175-176` — remove fallback `.get("f1_macro", .get("f1"))` → just `.get("f1_macro", 0.0)`
- `evaluate.py:308-312, 386` — simplify same fallback to `.get("f1_macro", ...)` (covered in WP2)

#### M9 — Deduplicate RRD computation

**File**: `scripts/train.py:175-178`
**Change**: Replace inline RRD formula with `from src.evaluation.metrics import compute_rrd`

#### M3 — Make STOP_EVENT instance-local

**File**: `src/training/trainer.py:41`
**Change**: Add factory `def make_stop_event() -> threading.Event:`. `install_defensive_timer` takes event as parameter. `train.py` calls factory instead of importing the module-level singleton.

---

### WP2: Data + Eval Pipeline

**Files**: `src/data/dataloader.py`, `scripts/evaluate.py`, `src/baselines/perplexity_baseline.py`
**Issues**: C1, C7, C8, M4, M8
**Dependencies**: WP1 (uses updated metric keys)

#### C1 — Unify data pipelines (evaluate + perplexity_baseline → config-driven)

**File**: `scripts/evaluate.py:275-282`, `src/baselines/perplexity_baseline.py:333-334`
**Change**: Switch both from `get_dataloaders()` (module-level BATCH_SIZE=16, MAX_LENGTH=512) to `_prepare_dataloaders_on_the_fly()` with values from config.
- `evaluate.py`: Load config, pass `cfg.training.batch_size`, `cfg.data.max_length`, `cfg.training.seed`, `cfg.training.num_workers`
- `perplexity_baseline.py`: Same pattern

#### C7 — Fix cached path stratification

**File**: `src/data/dataloader.py:427-429`
**Change**: Replace `torch.randperm` with `train_test_split(stratify=df_capped["label"])` to match on-the-fly path behavior:
```python
from sklearn.model_selection import train_test_split
df_capped = _capped_df(...)  # already available
labels = df_capped["label"].tolist()
indices = list(range(n))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=labels)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, stratify=temp_labels)
```

#### C8 — Fix strict=False checkpoint loading

**File**: `scripts/evaluate.py:262`
**Change**: `model.load_state_dict(state_dict, strict=False)` → `model.load_state_dict(state_dict, strict=True)`

#### M4 — Remove fallback import in dataloader.py

**File**: `src/data/dataloader.py:15-18`
**Change**: Remove `try/except ImportError` fallback. Keep only `from .dataset import CachedTensorDataset, DetectRLDataset, OnTheFlyDataset`.

#### M8 — Fix evaluate.py module-level side effects

**File**: `scripts/evaluate.py:40-65`
**Change**: Wrap config loading, seed, and path construction into a function called from `main()`/`evaluate()`. No module-level code runs on import.

---

### WP3: Config System

**File**: `src/config/config.py`
**Issues**: C3, M6
**Dependencies**: None (Wave 1)

#### C3 — Fix CLI parser "0" as False

**File**: `src/config/config.py:259-276`
**Change**: Check numeric before truthiness:
```python
def _parse_cli_value(value: str) -> Any:
    if value.lower() == "null":
        return None
    try: return int(value)
    except ValueError: pass
    try: return float(value)
    except ValueError: pass
    lower = value.lower()
    if lower in ("true", "yes", "1"): return True
    if lower in ("false", "no"): return False  # no "0"
    return value
```
**Verification**: `_parse_cli_value("0")` → `0` (int), not `False`

#### M6 — Replace runtime `__import__("typing")`

**File**: `src/config/config.py:124`
**Change**: `__import__("typing").Union`/`.Optional` → `from typing import Union as TypingUnion, Optional as TypingOptional` at module top

---

### WP4: Package Hygiene

**Files**: `pyproject.toml`, `src/data/__init__.py`
**Issues**: M1, M2, M7
**Dependencies**: None (Wave 1)

#### M1 — Fix pyproject.toml entry points

**File**: `pyproject.toml:35-37`
**Change**: Remove broken `[project.scripts]` section (scripts outside `src/` make entry points non-functional). README documents direct `python scripts/train.py` usage.

#### M2 — Fix private API exports

**File**: `src/data/__init__.py`
**Change**: Remove `_prepare_dataloaders_cached`, `_prepare_dataloaders_on_the_fly` from `__all__`

#### M7 — Fix license badge mismatch

**File**: `pyproject.toml:11`
**Change**: `license = { text = "MIT" }` → `license = { text = "Apache-2.0" }` (matches LICENSE file)

---

## Execution Order

```
Wave 1: Fire WP1, WP3, WP4 IN PARALLEL
Wave 2: After all Wave 1 → Fire WP2
```

## Commit Strategy

```
A: "refactor(model): fix embedding freeze, remove get_model_config, normalize metrics"
   distilbert_classifier.py, trainer.py, models/__init__.py, benchmark.py, train.py
B: "fix(data+eval): unify data pipelines, stratified cached, strict loading"
   dataloader.py, evaluate.py, perplexity_baseline.py
C: "fix(config): CLI parser '0', runtime typing import"
   config.py
D: "chore: entry points, license, private exports"
   pyproject.toml, src/data/__init__.py
```

## Success Criteria

- [ ] All 18 issues (8 Tier 0 + 10 Tier 1) fixed
- [ ] `python scripts/train.py --ablation baseline1` runs without errors
- [ ] `python scripts/evaluate.py` runs without errors (config-driven pipeline)
- [ ] `python scripts/benchmark.py` imports `DistilBertClassifier` from `src.models` (no inline class)
- [ ] `get_model_config()` no longer exists anywhere (grep returns empty)
- [ ] `_parse_cli_value("0")` returns `int(0)`, not `False`
- [ ] No `__import__("typing")` at runtime
- [ ] No module-level side effects on import in evaluate.py
- [ ] Cached path produces stratified splits (`train_test_split` with `stratify=`)
- [ ] No fallback import in dataloader.py (only `from .dataset import ...`)
- [ ] `evaluate.py` uses `strict=True` in `load_state_dict`
- [ ] `pyproject.toml` has no `[project.scripts]` section
- [ ] `src/data/__init__.py` `__all__` has no `_`-prefixed entries
- [ ] `STOP_EVENT` created via factory (`make_stop_event()`), not module-level singleton
- [ ] `pyproject.toml` license is `"Apache-2.0"` (matches LICENSE file)
- [ ] `train.py` imports `compute_rrd` from `src.evaluation.metrics` (no inline RRD formula)
