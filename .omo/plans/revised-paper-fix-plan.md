# Revised Paper Fix Plan (Post-Hyperplan Synthesis)

## Overview & TL;DR

**Source**: Distillation of `.omo/plans/paper-fix-plan.md` × `.omo/plans/hyperplan-synthesis.md` (3-critic adversarial review). Supersedes the original plan.

**Structural change**: 5 parallel waves → **2 execution tracks + GPU rerun phase**. 6 critical gaps (A–F) from the hyperplan are integrated. The FDG Decision Tree is replaced with Option C (rename + mathematical footnote). Timeline corrected from ~45h to ~40h (P0 ≈20h, P1 ≈14h, P2 ≈6h).

| Phase | Effort | GPU Required | Description |
|-------|--------|-------------|-------------|
| Track A | ~8h | No | Code fixes, data pipeline patches, dedup, TF-IDF baseline |
| Track B | ~11h | No | .tex revisions, FDG rename, framing, systems compression |
| GPU Reruns | ~21h | Yes | Multi-seed eval, clean baseline, low-FPR, calibration, significance |

---

## Pre-Requisites

Before starting any track, verify these conditions:

| Pre-requisite | Why | Verification |
|--------------|-----|-------------|
| **Probability output persisted** | W4-T2 (calibration/ECE) requires softmax/sigmoid outputs, not just argmax classes | Check `train_distilbert_detectrl.py` eval loop: does it save `probs` or `logits` to the prediction file? If not, add a 1-line `probs = F.softmax(logits, dim=1)` or `torch.sigmoid(logits)` dump before evaluation. |
| **RAID dataset availability** | GPU reruns (Gaps B/D/E) need `data/processed/raid_*` files | `dir data/processed/` — verify `raid_train_pool.parquet`, `raid_test_unseen.parquet` exist with expected row counts |
| **CUDA environment functional** | ~21h GPU work | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — must return `True` + RTX 3050 |

**Fix the probability output gap first**: If probabilities aren't persisted, add a single line in the eval loop and re-run all 4 configs (~5h GPU). This is a P1 task that blocks W4-T2.

---

## Track A: Code Fixes + Data Pipeline + New Baselines

**Goal**: Make the repo runnable on fresh clone, fix data contamination, add missing baselines.
**Effort**: ~8h (no GPU). **Verification gate**: All 6 crash bugs resolved + schema validated.

### A1. Code Crash Fixes (W2-T1 through W2-T6, ~2h)

| Task | File | Fix | Acceptance |
|------|------|-----|------------|
| A1a. tc3 unseen cache crash | `tc3_traindistilbert.py` | Add fallback for missing unseen cache; verify `DATASET` is `"raid"` | `python tc3_traindistilbert.py --dry-run` (or actual run) exits 0 |
| A1b. Sequential parquet naming | `data/filter_raid_sequential.py` | Prefix outputs: `raid_train_pool.parquet`, `raid_test_unseen.parquet` | Fresh preprocessor run produces expected filenames |
| A1c. tc4 checkpoint paths | `tc4_.py` | Expand `POSSIBLE_PATHS` to 5 entries | `python -m py_compile tc4_.py` |
| A1d. CPU autocast guards | `tc3_traindistilbert.py`, `tc4_.py` | Wrap 7 call sites with `if DEVICE.type == "cuda"` | Run on CPU-only env: no `RuntimeError: CUDA error` |
| A1e. Update requirements.txt | `requirements.txt` | Append `pyarrow>=14.0.0`, `bitsandbytes>=0.43.0` | `pip install -r requirements.txt` succeeds |
| A1f. Fix F1 variant | `src/evaluation/metrics.py` | `average="macro"` → default `binary` | Unit test: binary array produces correct F1 |

**Dependency**: None — all A1 tasks are independent.
**Blockers**: A1b blocks Gap A (dedup uses same filenames). A1d blocks GPU reruns (crashes mid-run).

### A2. Human-Text Deduplication (Gap A, ~2-3h)

**What**: 6,029 of 5,000 held-out human samples appear in training (~60% contamination). Held-out metrics are inflated.

**Implementation**:
1. Add dedup step in `data/filter.py` (or create `data/dedup_splits.py`):
   ```python
   # Before train/held-out split, deduplicate on text content
   df = df.drop_duplicates(subset=["text"])
   # Then split with fixed seed
   ```
2. Add seed param to the split function for reproducibility
3. Rerun preprocessing pipeline (sequential preprocessor)
4. Report metrics on BOTH contaminated and deduplicated splits in paper (appendix table)

**Files**: `data/filter.py`, `data/filter_raid_sequential.py` (may need to rerun), `Research_Paper.tex` (new appendix section)

**Dependency**: Depends on A1b (sequential naming fix) — same preprocessor.
**Blocks**: All GPU reruns that use the held-out split.

### A3. Attack Column Hijack Fix + Per-Attack Eval Prep (W4-T3, ~1h)

**What**: In `data/filter_raid_parallel.py` L217, both `attack_type` and generator name must be preserved in output schema.

**Fix**: Modify column assignment to avoid `attack_type` being overwritten. Add generator name column.

**Dependency**: Independent (separate file from A2).
**Blocks**: GPU rerun T3 (per-attack DistilBERT evaluation).

### A4. TF-IDF + Logistic Regression Baseline (Gap C, ~2h)

**What**: Train TF-IDF + LR on the same deduplicated split to justify the "deep learning for robustness" framing.

**Implementation**: Create `src/baselines/tfidf_baseline.py`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Train on same split, report all metrics
```

**Dependency**: Depends on A2 (deduplicated split must exist).

### Track A Verification Gate

```
□ A1a-f: All 6 code crash bugs confirmed fixed (fresh clone test)
□ A2:   Deduplication produces correct train/held-out split with no overlap
□ A3:   filter_raid_parallel.py preserves attack_type AND generator
□ A4:   TF-IDF + LR metrics generated
□ All:  `python -m py_compile` on all modified .py files passes
```

---

## Track B: .tex Revisions + Framing Corrections

**Goal**: All paper text changes — baseline renaming, framing fixes, systems compression.
**Effort**: ~11h (no GPU). **Verification gate**: Every number in abstract matches body.

### B1. FDG Baseline Rename (Option C) — Former W1-T1 (~2h)

**Decision**: **Option C** (synthesis resolution: rename + mathematical footnote). Do NOT implement actual curvature.

**Action**:
1. Replace all `"Fast-DetectGPT"` in `.tex` with `"GPT-2 XL Perplexity Baseline"`
2. Add footnote at first occurrence:
   ```
   \footnote{The original Fast-DetectGPT algorithm \cite{mitchell2023} 
   computes conditional probability curvature via token perturbation.
   Our implementation computes cumulative log-likelihood (standard perplexity),
   which provides a valid zero-shot baseline but does not implement the 
   full curvature computation.}
   ```
3. Update bibliography: keep Mitchell et al. 2023 citation (it remains valid as the conceptual reference)
4. Update `fast_detectgpt.py` docstring/header comment to reflect the honest name
5. Update Section IV-D, Conclusion, Abstract references

**Dependency**: Independent. Document-only.
**Does NOT block**: W4-T1, W4-T4 (contra original dependency matrix). These were false blockers.

### B2. RRD Naming — 33.40% Is Not RRD (W1-T2, ~1h)

**Action**:
- Abstract L60: `"gpt2-xl RRD: 33.40%"` → `"gpt2-xl adversarial AUROC degradation: 33.40%"`
- Section V.C L1052: `"RRD"` → `"adversarial AUROC degradation"`
- Table caption footnote: define `RRD_adv` as a separate metric from Eq. 2
- Section IV-C: add footnote defining `RRD_adv` metric

### B3. Fix Abstract AUROC Range (W1-T3, ~0.5h)

Abstract L59: `(AUROC 0.669--0.789)` → `(AUROC 0.526--0.789)`.

### B4. Add gpt-j-6B Degradation to Abstract (W1-T4, ~0.5h)

Abstract L60: `(gpt2-xl degradation: 33.40\%)` → `(gpt2-xl degradation: 33.40\%, gpt-j-6B degradation: 14.99\%)`.

### B5. Fix Table IV Confusion Matrix Metrics (W1-T5, ~1h)

Recalculate all ablation_c metrics from Table VI confusion matrix. Corrected values:
- Seen: Acc 0.9456, Prec 0.9161, F1 0.9474
- Held-out: Acc 0.9263, Prec 0.9247, F1 0.9264

Update L802 in `.tex`.

### B6. Fix Decision Threshold Wording (W1-T6, ~0.5h)

Section III-F: Replace imprecise "0.5 argmax" with correct description:
> "All DistilBERT classifications use a 0.5 decision threshold on the positive-class sigmoid output (binary classification)."

### B7. Framing Corrections — W3 Retained Tasks (~3h)

| Task | Description | Line(s) | Status |
|------|-------------|---------|--------|
| B7a. Title softening (W3-T1) | Re-title to reflect held-out-generator evaluation | L1 | ✅ Keep |
| B7b. Qualify "adversarially diverse" (W3-T2) | Replace with precise attack-type language | L40-41, L107-108, L349-350 | ✅ Keep |
| B7c. "demonstrates" → "suggests" (W3-T3) | L1052 | L1052 | ✅ Consensus correct |
| B7d. AMP speedup disentangled (W3-T4) | Add breakdown: batch size vs FP16 | ~AMP section | ✅ Consensus correct |
| B7e. "Under 16 minutes" clarified (W3-T5) | Add "per configuration" or compute total | L1293 | ✅ Keep |
| B7f. Conclusion softened (W3-T6) | "favourable robustness margin" → "competitive held-out performance, acknowledging the comparison is directional" | L1293-1294 | ✅ Keep |
| B7g. Label significance table (W3-T7) | Add `\label{tab:bootstrap}` + text reference | L1047 | ✅ Consensus correct |

**Note**: W3-T6 threshold wording is superseded by B6 (more precise fix).

### B8. Housekeeping — W5 Retained Tasks (~1h)

| Task | Description | Status |
|------|-------------|--------|
| B8a. Expand Limitations (W5-T1) | Energy, RAID noise, unmatched comparison | ✅ Keep |
| B8b. Filter script docstrings (W5-T2) | Match `raid_` prefix output names | ✅ Keep |
| B8c. Peak VRAM note (W5-T3) | Add column/note to wall-clock table | ✅ Keep |
| B8d. AUROC delta rounding (W5-T4) | Fix 0.2637→0.2636, 0.1181→0.1180 | ✅ Keep |

### B9. Systems Benchmarks Compression (Gap F, ~1h)

**Action**: Reduce ~4 figures + ~3 tables on standard PyTorch optimizations to 1 paragraph + supplementary section. Moves detection science narrative to the foreground.

### Track B Verification Gate

```
□ B1:  All "Fast-DetectGPT" references renamed; footnote added
□ B2-B5: Numerical consistency scan — every abstract number matches body
□ B6:   Threshold wording uses "binary sigmoid" not "argmax"
□ B7a-g: All 7 framing corrections applied
□ B8a-d: Housekeeping done
□ B9:   Systems benchmarks compressed
□ All:  `\ref`/`\label` cross-reference audit — no orphaned references
```

---

## GPU Rerun Sequence

**Note**: GPU reruns begin after Track A completes (data pipeline fixed) and can partially overlap with Track B.

**Estimated GPU time**: ~21h on RTX 3050 8GB.

| Sequence | Task | GPU Time | Depends On | Notes |
|----------|------|----------|------------|-------|
| **G1** | Multi-seed eval (Gap B) | ~13h | A1d, A2 | 4 configs × 3 seeds = 12 runs @ ~1.1h/config |
| **G2** | Clean-only training (Gap E) | ~1h | A2 | Train ablation_b on clean RAID only |
| **G3** | FDG on deduplicated split (Gap D) | ~3-4h | A2, B1 | Perplexity baseline on same eval split |
| **G4** | Low-FPR metrics (W4-T1) | ~2h | G1 | Post-hoc on saved predictions |
| **G5** | Calibration/ECE (W4-T2) | ~2h | A2, Probability pre-requisite | Requires saved probs |
| **G6** | Per-attack DistilBERT eval (W4-T3) | ~2h | A3, G1 | Uses fixed data schema |
| **G7** | Bootstrap significance (W4-T4) | ~1h | G1 | Post-hoc analysis |
| **G8** | Count tables (W4-T5) | ~0h | A2 | Data analysis, no GPU |

**Critical path**: G1 → G4/G7. G3 can parallel with G1 (same data pipeline, different model). G5 blocked by probability pre-requisite.

---

## Priority Triage Table

| Priority | Tasks | Effort | Why |
|----------|-------|--------|-----|
| **P0** | A1 (code crashes), A2 (dedup), B1 (FDG rename), B3 (abstract AUROC), B5 (Table IV) | ~20h | Blocking: paper has wrong numbers or crashes on review |
| **P1** | A4 (TF-IDF baseline), B2 (RRD naming), B6-B7 (framing), G1 (multi-seed), G2 (clean baseline), G5 (calibration) | ~14h | Major: reviewer flags, scientific integrity |
| **P2** | B8 (housekeeping), B9 (systems compress), G6 (per-attack), G7 (bootstrap), G8 (count tables) | ~6h | Minor: deferrable, nice-to-have |

---

## Dependency Matrix (Corrected)

| Task | Depends On | Blocks | Reason |
|------|-----------|--------|--------|
| A1a-f (code crashes) | None | All GPU reruns | Repo must run first |
| A2 (dedup) | A1b | G1, G2, G3, G5, G8 | Dedup'd split needed for all eval |
| A3 (attack column) | None | G6 | Schema fix needed for per-attack eval |
| A4 (TF-IDF) | A2 | None | Requires dedup'd split |
| B1 (FDG rename) | None | G3 | Must know baseline name before running on new split |
| B2 (RRD naming) | None | None | Independent text change |
| B3 (abstract AUROC) | None | None | Independent text change |
| B4 (gpt-j-6B) | None | None | Independent text change |
| B5 (Table IV) | None | None | Independent text change |
| B6 (threshold) | None | None | Independent text change |
| B7a-g (framing) | None | None | All independent text changes |
| B8a-d (housekeeping) | None | None | Independent |
| B9 (systems compress) | None | None | Independent |
| G1 (multi-seed) | A1d, A2 | G4, G7 | Requires working code + dedup'd data |
| G2 (clean baseline) | A2 | None | Requires dedup'd data |
| G3 (FDG on dedup) | A2, B1 | None | Requires renamed baseline + dedup'd data |
| G4 (low-FPR) | G1 | None | Post-hoc on saved predictions |
| G5 (calibration) | A2, prob pre-req | None | Requires dedup'd data + persisted probs |
| G6 (per-attack eval) | A3, G1 | None | Requires fixed schema + trained models |
| G7 (bootstrap) | G1 | None | Post-hoc analysis |
| G8 (count tables) | A2 | None | Data analysis |

---

## Parallel Execution Strategy

### Wave 1 (Parallel — No Dependencies)

```
Track A (parallel tasks):
├── A1a (tc3 cache fix)  ─┐
├── A1b (seq naming)     ─┤
├── A1c (tc4 paths)      ─┤ All 6 code fixes are independent
├── A1d (CPU guards)     ─┤
├── A1e (requirements)   ─┤
├── A1f (F1 variant)     ─┘
├── A3 (attack column)     ← independent code fix

Track B (parallel tasks):
├── B1 (FDG rename)      ─┐
├── B2 (RRD naming)      ─┤
├── B3 (abstract AUROC)  ─┤ All 14 text tasks are independent
├── B4 (gpt-j-6B)        ─┤ of each other
├── B5 (Table IV)        ─┤
├── B6 (threshold)       ─┤
├── B7a-g (framing)      ─┤
├── B8a-d (housekeeping) ─┤
├── B9 (systems compress)─┘
```

### Wave 2 (After A1b + A1d Complete)

```
├── A2 (dedup)          ← Depends on A1b (filename fixes)
├── A4 (TF-IDF)         ← Depends on A2 (dedup'd split) — can run during GPU phase
└── All remaining B tasks continue in parallel (no dependencies on A tasks)
```

### Wave 3 (After A2 + A1d Complete — GPU Phase)

```
GPU cluster (where possible in parallel):
├── G1 (multi-seed, 13h) ◄── critical path
├── G2 (clean-only, 1h)  ← parallel with G1
├── G3 (FDG dedup, 3-4h) ← parallel with G1 (different GPU)
└── G5 (calibration pre-run if probs missing)

Post-hoc (after G1):
├── G4 (low-FPR, 1h)
├── G6 (per-attack, 2h)
├── G7 (bootstrap, 1h)
├── G8 (count tables, 0h)
```

---

## Time Estimates (Realistic)

| Phase | Hours | GPU? | Notes |
|-------|-------|------|-------|
| Track A | 8h | No | 2h crashes + 3h dedup + 2h TF-IDF + 1h attack fix |
| Track B | 11h | No | 2h FDG + 5h RRD/AUROC/Table IV/threshold + 3h framing + 1h housekeeping |
| GPU Reruns | 21h | Yes | 13h multi-seed + 1h clean + 3-4h FDG + 2h low-FPR + 2h cal |
| **Total** | **40h** | **~21h GPU** | Realistic for top-tier venue readiness |

---

## Verification Gates

### Gate 1: End of Track A
```
□ python tc3_traindistilbert.py runs without crash (30s smoke test)
□ python tc4_.py runs without crash (30s smoke test)
□ Dedup splits have 0 overlapping human texts
□ TF-IDF + LR metrics file exists at artifacts/baselines/tfidf_metrics.json
□ filter_raid_parallel.py preserves both attack_type and generator columns
```

### Gate 2: End of Track B
```
□ grep -c "Fast-DetectGPT" Research_Paper.tex → 0 (all renamed)
□ Abstract numbers match body: AUROC 0.526–0.789, RRD/gpt2-xl/gpt-j-6B values
□ grep "argmax" Research_Paper.tex → 0 (replaced with sigmoid)
□ grep "RRD" Research_Paper.tex → only qualified uses (RRD_adv, Eq. 2 RRD)
□ Every \ref has matching \label; every \cite has matching \bibitem
□ Systems benchmarks section reduced to 1 paragraph + supplementary reference
```

### Gate 3: End of GPU Reruns
```
□ artifacts/multiseed_results.csv: mean±std for all 4 configs × 3 seeds
□ artifacts/clean_only/ablation_b_metrics.json: RRD with vs without adversarial data
□ artifacts/fdg_dedup/fdg_metrics.json: perplexity baseline on dedup'd held-out
□ artifacts/low_fpr_recall.csv: recall at 1%/5%/10% FPR for all configs
□ artifacts/calibration/ece_scores.csv: ECE for all configs
□ artifacts/per_attack_eval/per_attack_metrics.csv: per-attack DistilBERT results
□ artifacts/bootstrap/bootstrap_significance.csv: pairwise significance test p-values
□ artifacts/count_tables.csv: per-domain/per-attack/per-generator counts
```

### Final Verification
```
□ Track A pass + Track B pass + GPU rerun pass = ALL GATES GREEN
□ git diff --stat: no unintended changes, no stale drift
□ All TF-IDF, dedup, multi-seed results integrated into paper
```

---

## Commit Strategy

### Commit 1: `fix(code): repo reproducibility + data pipeline fixes`
**Scope**: Track A — A1a-f (all 6 code crashes), A2 (dedup), A3 (attack column fix)
**Files**: `tc3_traindistilbert.py`, `tc4_.py`, `data/filter_raid_sequential.py`, `data/filter_raid_parallel.py`, `data/filter.py`, `src/evaluation/metrics.py`, `requirements.txt`

### Commit 2: `fix(paper): baseline integrity, framing, and systems compression`
**Scope**: Track B — B1 (FDG rename), B2-B6 (numerical corrections), B7a-g (framing), B8a-d (housekeeping), B9 (systems compress)
**Files**: `Research_Paper.tex`, `src/baselines/fast_detectgpt.py`

### Commit 3: `feat(baselines): TF-IDF + logistic regression baseline`
**Scope**: A4
**Files**: `src/baselines/tfidf_baseline.py`, `Research_Paper.tex`

### Commit 4: `feat(eval): multi-seed, low-FPR, calibration, per-attack results`
**Scope**: GPU Reruns — G1 through G8 result integration
**Files**: `Research_Paper.tex`, `src/evaluation/metrics.py`, `artifacts/` (new results)

### Commit 5: `chore(docs): final verification and cleanup`
**Scope**: Any remaining polish, verification gate sign-off
**Files**: `.omo/plans/revised-paper-fix-plan.md` (this plan), docstring updates

---

## Success Criteria

### Minimum Viable (Workshop-ready, ~4h of work)
- All code crashes fixed (A1 a-f)
- Abstract AUROC range corrected (B3)
- FDG baseline renamed (B1)
- Paper compiles and renders correctly

### Mid-tier (IBCAST/EACL-ready, ~20h)
- All P0 tasks complete
- Dedup applied (A2)
- TF-IDF baseline (A4)
- Clean-only baseline (G2)
- RRD naming fixed (B2)

### Top-tier (ACL/EMNLP-ready, ~40h)
- ALL tasks complete
- Multi-seed error bars (G1)
- Low-FPR + calibration (G4, G5)
- Per-attack eval (G6)
- Bootstrap significance (G7)
- Systems benchmarks compressed (B9)

---

## Appendices

### Appendix A: FDG Baseline Resolution (Option C)

**Decision**: Rename FDG → "GPT-2 XL Perplexity Baseline" everywhere in the paper. Add mathematical footnote explaining the implementation difference.

**Why Option C (not A or B from original plan)**:
- **Option A** (rename only) was the original "recommended" path — but forgot to include the mathematical footnote. Reviewers familiar with Mitchell et al. 2023 will notice the discrepancy.
- **Option B** (implement curvature) is ~12h GPU work for marginal gain. The baseline's value is as a zero-shot perplexity measure, not a curvature-based detector. Top-tier venues accept both; integrity matters more than algorithm name.
- **Option C** = rename + footnote. Transparent, defensible, minimal effort (~2h, no GPU).

**Mathematical footnote content**:
```
The original Fast-DetectGPT algorithm computes conditional probability curvature:

    curvature(x) = Σ_t log p(x_t | x_{<t}) - (1/k) Σ_{x̃∈top-k(x,t)} log p(x̃_t | x_{<t})

Our implementation computes cumulative log-likelihood (standard perplexity):

    score(x) = Σ_t log p(x_t | x_{<t})

This provides a valid zero-shot baseline but does not implement the perturbation-based
curvature term. We refer to it as a "perplexity baseline" throughout.
```

### Appendix B: Items Explicitly Rejected from Original Plan

| Item | Status | Reason |
|------|--------|--------|
| F1: LSP diagnostics (texlab) | ❌ Reject | Busywork; does not affect scientific integrity |
| Git branch strategy | ❌ Reject | Over-engineering; default branching is fine |
| AI prose check | ❌ Reject | Introduces noise, unnecessary process step |
| W4-T6 (matched FDG if curvature implemented) | ❌ Replaced | Superseded by Gap D (FDG on dedup'd split) |
| W4-T7 (clean gpt2-xl baseline) | ❌ Replaced | Superseded by Gap E (clean-only training baseline) |
| "5 parallel waves" model | ❌ Restructured | Replaced with 2-track model + GPU rerun phase |
| FDG Decision Tree (Option A vs B) | ❌ Replaced | With Option C (rename + footnote) |
