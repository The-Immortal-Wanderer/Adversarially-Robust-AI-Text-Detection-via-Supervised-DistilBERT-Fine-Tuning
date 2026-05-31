# Claude Guidance Verification — ANN_Project

**Status**: FINAL
**Generated**: 2026-05-24
**Methodology**: Independent 5-agent cross-reference panel verifying every major technical claim against actual codebase, config files, parquet data, and published model specs.
**Source transcript**: [\Claude_Suggestions.md\](../../Claude_Suggestions.md) — raw 1519-line Claude chat export at project root
**Verification scope**: Codebase audit of all claims in Claude's guidance conversation

## Audit Results

### (1) src/evaluation/metrics.py — compute_metrics() returns f1_macro and f1_positive
**PASS** — Both metrics computed.

### (2) src/training/trainer.py — _compute_metrics() returns f1_macro and f1_positive
**PASS** — Both metrics computed.

### (3) scripts/evaluate.py — run_eval uses @torch.inference_mode()
**PASS** — Line 188: @torch.inference_mode() decorates the function.

### (4) scripts/g0_decision_gate.py — RRD_SPREAD_THRESHOLD named constant
**PASS** — Line 34: RRD_SPREAD_THRESHOLD: float = 6.0 (value 6.0, i.e., 6% RRD spread).

### (5) Root-level .py files
**PASS** — Zero .py files at project root. Legacy scripts consolidated during Phase 0b.

### (6) Stale .ipynb notebook files
**MINOR** — notebooks/kaggle_pipeline.ipynb deprecated per README; superseded by kaggle_run.py.

---

## Additional Code Quality Findings

- **f1_positive is dead code** — computed in both compute_metrics() and _compute_metrics(), but NO consumer ever reads it.
- **Stale detectrl references** — 3 files mention "detectrl" in docstrings.

---

## Key Fix Applied

**7 determinism flags confirmed in seed_everything() (trainer.py:25-41):**
- \CUBLAS_WORKSPACE_CONFIG=\\":4096:8\\"\, \andom.seed()\, \
p.random.seed()\, \	orch.manual_seed()\, \	orch.cuda.manual_seed_all()\, \cudnn.deterministic=True\, \cudnn.benchmark=False\

Note: torch.use_deterministic_algorithms(True) is intentionally omitted because PyTorch 2.x scaled_dot_product_attention lacks a deterministic implementation.

---

## Summary

| Check | Result |
|-------|--------|
| 1. f1_positive removal | ✅ PASS |
| 2. RRD_SPREAD_THRESHOLD docstring | ✅ PASS |
| 3. Claude_Suggestions.md cross-ref | ✅ PASS |
| 4. .pyc / __pycache__ gitignore | ✅ PASS |
| Determinism flag claim | ✅ FIXED (8→7) |
