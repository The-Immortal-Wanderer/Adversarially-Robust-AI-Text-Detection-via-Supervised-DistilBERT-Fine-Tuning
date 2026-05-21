> **SUPERSEDED**: This plan is superseded by `.omo/plans/end-to-end-restructure-plan.md`.
> Refer to that plan for current execution strategy. The end-to-end plan supersedes all previous plans
> with corrected GPU target (Kaggle T4), restructure-first approach, and 6-wave execution model.

# Revised Paper Fix Plan v2 — 21-Finding Synthesis

## Overview & TL;DR

**Source**: `.omo/plans/revised-paper-fix-plan.md` (base) × `.omo/plans/hyperplan2-synthesis.md` (4-critic × 3-round adversarial review). 21 findings incorporated: 4 critical, 6 major, 6 medium, 5 patch.

**Structural changes from v1**:
- GPU phase corrected from parallel → fully serial (single RTX 3050 constraint)
- Contamination source verified *before* dedup implementation
- `compute_metrics` deduplicated before any number-fix tasks
- Pre-G1 decision gate added with fallback narrative
- Priority/success criteria split into Core + Extended tiers
- G1 estimate corrected to 14.1h (per-config timing data)
- GPU budget revised to 25-30h with thermal buffer
- FDG file rename + internal reference audit (not just .tex)
- Column consumer audit added before A3

| Phase | Effort | GPU Required | Description |
|-------|--------|-------------|-------------|
| Track A | ~10.5h | No | Code fixes, contamination audit + dedup, metrics dedup, column audit, TF-IDF |
| Track B | ~12h | No | .tex revisions, FDG rename (file + code), threshold justification, framing |
| GPU Reruns | ~25-30h | Yes | Multi-seed eval, clean baseline, FDG dedup, low-FPR, calibration, significance |
| **Total** | **~47-52h** | **~25-30h GPU** | Realistic for top-tier venue readiness |

---

## Pre-Requisites

| Pre-requisite | Why | Verification |
|--------------|-----|-------------|
| **Probability output persisted** | Calibration/ECE requires softmax/sigmoid outputs, not argmax | Check `train_distilbert_detectrl.py` eval loop: does it save `probs` or `logits`? If not, add integration into G1 eval loop (zero extra GPU cost per C3) |
| **RAID dataset availability** | GPU reruns need `data/processed/raid_*` files | `dir data/processed/` — verify `raid_train_pool.parquet`, `raid_test_unseen.parquet` exist |
| **CUDA environment functional** | ~25-30h GPU work | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True` + RTX 3050 |
| **GPU thermal baseline** | Long serial runs risk throttling | Monitor `nvidia-smi` temps at idle; if >70°C at idle, improve cooling before starting G1 |

---

## Track A: Code Fixes + Data Pipeline + New Baselines

**Goal**: Repo runnable, contamination fixed, metrics unified, baselines ready.
**Effort**: ~10.5h (no GPU). **Verification gate**: All code fixes pass + schema validated + metrics deduplicated.

### A0. Contamination Source Verification (NEW — from C4, P5, ~1h)

**Before** implementing dedup (A2), identify the exact contamination source.

**Actions**:
1. Grep both data pipelines (`data/filter.py` for DetectRL, `data/filter_raid_parallel.py`/`data/filter_raid_sequential.py` for RAID) for the 6,029 overlapping human texts
2. Report per-text multiplicity (how many times each dup appears — explains the >100% ratio)
3. Determine: is contamination in DetectRL pipeline, RAID pipeline, or both?
4. Document finding in `data/contamination_audit.md` (brief, ~10 lines)
5. Update dedup implementation target accordingly

**Dependency**: None. Must complete **before** A2.
**Blocks**: A2.

### A1. Code Crash Fixes (A1a-f, ~2h)

| Task | File | Fix | Acceptance |
|------|------|-----|------------|
| A1a. tc3 unseen cache crash | `tc3_traindistilbert.py` | Add fallback for missing unseen cache; verify `DATASET` is `"raid"` | `python tc3_traindistilbert.py --dry-run` exits 0 |
| A1b. Sequential parquet naming | `data/filter_raid_sequential.py` | Prefix outputs: `raid_train_pool.parquet`, `raid_test_unseen.parquet` | Fresh preprocessor run produces expected filenames |
| A1c. tc4 checkpoint paths | `tc4_.py` | Expand `POSSIBLE_PATHS` to 5 entries | `python -m py_compile tc4_.py` |
| A1d. CPU autocast guards | `tc3_traindistilbert.py`, `tc4_.py` | Wrap 7 call sites with `if DEVICE.type == "cuda"` | Run on CPU-only env: no `RuntimeError: CUDA error` |
| A1e. Update requirements.txt | `requirements.txt` | Append `pyarrow>=14.0.0`, `bitsandbytes>=0.43.0` | `pip install -r requirements.txt` succeeds |
| A1f. Fix F1 variant + deduplicate compute_metrics | `src/evaluation/metrics.py` | (a) `average="macro"` → default `binary`, (b) merge all 4 implementations into single source of truth at `src/evaluation/metrics.py`, (c) update all import paths | Unit test: binary array produces correct F1; `grep -r "def compute_metrics" src/` → single definition |

**A1f expanded**: M3 requires deduplicating the 4 compute_metrics implementations. After fixing the F1 variant, audit all scripts that define or import `compute_metrics`. Consolidate into `src/evaluation/metrics.py` — the sole source of truth. Update every import. This blocks B3/B5/B7 number changes.

**Dependency**: A1 tasks are independent of each other. A1b blocks A2 (filenames). A1f blocks B3/B5/B7 (number fixes use wrong metrics).

### A2. Human-Text Deduplication (~2h)

**Target**: Determined by A0 (RAID, DetectRL, or both).

**Actions**:
1. Implement dedup step in the identified pipeline file:
   ```python
   df = df.drop_duplicates(subset=["text"])
   ```
2. Add seed param to the split function for reproducibility
3. Rerun preprocessing pipeline (sequential preprocessor)
4. Report metrics on BOTH contaminated and deduplicated splits
   - Deduplicated split = **primary** for abstract/main claims (N3)
   - Contaminated split = supplementary (appendix)
5. Add one-sentence footnote (N5):
   ```
   Deduplication was exact-match only; residual near-duplicate contamination is a limitation.
   ```

**Dependency**: A0 (contamination source), A1b (filename fix).
**Blocks**: All GPU reruns that use the held-out split.

### A3. Attack Column Hijack Fix + Column Consumer Audit (~1.5h)

**Expanded from M2**:

**Actions**:
1. **Before** modifying `data/filter_raid_parallel.py`: grep all downstream scripts for `attack_type` column references. Document expected columns post-fix in a brief audit log (~30min).
2. Modify column assignment in `filter_raid_parallel.py` L217 to preserve both `attack_type` and generator name.
3. Verify no downstream script breaks against new schema.

**Dependency**: Independent (separate file from A2).
**Blocks**: G6 (per-attack DistilBERT evaluation).

### A4. TF-IDF + Logistic Regression Baseline (~2h)

Create `src/baselines/tfidf_baseline.py` using sklearn on the deduplicated split.

**Dependency**: A2 (dedup'd split).

### A5. 5% RRD Threshold Justification (NEW — from N4, ~1h)

**Actions**:
1. Add B-framing task: provide citation or reasoning for the 5% RRD threshold choice
2. Insert into `Research_Paper.tex` where the 5% threshold is first introduced (Section I or III)
3. If no citation exists, add a brief justification sentence (e.g., "We adopt a 5% threshold following [citation] convention for acceptable generalization degradation in detection tasks.")

**Dependency**: None. Can run in parallel with A1/A0.

### Track A Verification Gate

```
□ A0:  Contamination source identified and documented in data/contamination_audit.md
□ A1a-f: All 6 code crash bugs confirmed fixed; compute_metrics deduplicated
□ A2:   Deduplication produces correct train/held-out split with 0 overlap
□ A3:   Column consumer audit completed; filter_raid_parallel.py preserves attack_type + generator
□ A4:   TF-IDF + LR metrics generated at artifacts/baselines/tfidf_metrics.json
□ A5:   5% RRD threshold justification added to .tex
□ All:  `python -m py_compile` on all modified .py files passes
□ All:  `grep -r "def compute_metrics" src/` → exactly 1 result
```

---

## Track B: .tex Revisions + Framing Corrections + Code Rename

**Goal**: All paper and code naming changes, framing fixes, systems compression.
**Effort**: ~12h (no GPU). **Verification gate**: Every number in abstract matches body; FDG file/string references fully eradicated.

### B1. FDG Baseline Rename + File Rename + Internal Audit (EXPANDED — from M4, P4, ~2.5h)

**Beyond v1**: Also rename the actual file and audit ALL internal string references.

**Actions**:
1. **Rename file**: `src/baselines/fast_detectgpt.py` → `src/baselines/gpt2_perplexity_baseline.py`
2. **Update internal references**: Grep for all occurrences of `fast_detectgpt` in:
   - Import statements across the codebase
   - Argparse help strings
   - Log messages
   - Docstrings
   - Config/JSON files
3. **Replace all `"Fast-DetectGPT"` in `.tex`** with `"GPT-2 XL Perplexity Baseline"`
4. **Change citation**: from direct Fast-DetectGPT attribution to "cf. Mitchell et al. 2023" — the relationship is conceptual, not implementation-level
5. **Add footnote** at first occurrence:
   ```
   \footnote{The original Fast-DetectGPT algorithm \cite{mitchell2023}
   computes conditional probability curvature via token perturbation.
   Our implementation computes cumulative log-likelihood (standard perplexity),
   which provides a valid zero-shot baseline but does not implement the
   full curvature computation.}
   ```
6. Update bibliography: keep Mitchell et al. 2023 citation
7. Update Section IV-D, Conclusion, Abstract references

**Dependency**: Independent. Document + code rename.
**Note**: G3 does **NOT** depend on B1 (P1 — decoupled).

### B2. RRD Naming (~1h)

Same as v1: Replace "RRD" for gpt2-xl with "adversarial AUROC degradation" in Abstract, Section V.C, Table caption.

### B3. Fix Abstract AUROC Range (~0.5h)

Abstract L59: `(AUROC 0.669--0.789)` → `(AUROC 0.526--0.789)`.

### B4. Add gpt-j-6B Degradation to Abstract (~0.5h)

Abstract L60: add `gpt-j-6B degradation: 14.99%`.

### B5. Fix Table IV Confusion Matrix Metrics (REVISED — from N1, ~1h)

**Change from v1**: Table IV corrected values previously referenced non-existent "Table VI" and differences were rounding noise (<0.0004).

**Action**: Either (a) derive corrected values from actual confusion matrix using formula, (b) keep artifact values (differences are <0.0004, meaningless), or (c) remove B5 entirely.

**Recommendation**: Option (b) — keep artifact values, add footnote "values rounded from confusion matrix computation; differences <0.0004 are rounding artifacts." This avoids introducing a circular dependency and spends effort where it matters.

### B6. Fix Decision Threshold Wording (~0.5h)

Same as v1: Replace "0.5 argmax" with "0.5 decision threshold on the positive-class sigmoid output."

### B7. Framing Corrections (~3h)

Same as v1 (B7a-g), with additions:

| Task | Description | Lines | Status |
|------|-------------|-------|--------|
| B7a. Title softening | Re-title to reflect held-out-generator evaluation | L1 | Keep |
| B7b. Qualify "adversarially diverse" | Replace with precise attack-type language | L40-41, L107-108, L349-350 | Keep |
| B7c. "demonstrates" → "suggests" | L1052 | L1052 | Keep |
| B7d. AMP speedup disentangled | Add breakdown: batch size vs FP16 | ~AMP section | Keep |
| B7e. "Under 16 minutes" clarified | Add "per configuration" or compute total | L1293 | Keep |
| B7f. Conclusion softened | "favourable robustness margin" → "competitive held-out performance..." | L1293-1294 | Keep |
| B7g. Label significance table | Add `\label{tab:bootstrap}` + text reference | L1047 | Keep |
| **B7h. G2 clean baseline confound (NEW — N2)** | Add 2-sentence limitation: G2 trains on clean RAID vs original trains on adversarial DetectRL — two datasets differ | Section V | Add |
| **B7i. Designate primary split (NEW — N3)** | Deduplicated split = primary for abstract/main claims; contaminated = supplementary (appendix) | Abstract, Conclusion | Add |

### B8. Housekeeping (~1h)

Same as v1 (B8a-d).

### B9. Systems Benchmarks Compression (~1h)

Same as v1: Reduce ~4 figures + ~3 tables to 1 paragraph + supplementary section.

### Track B Verification Gate

```
□ B1:  File renamed to gpt2_perplexity_baseline.py; ALL internal strings audited
□ B1:  grep -c "Fast-DetectGPT" . → 0 (code + .tex)
□ B1:  grep -c "fast_detectgpt" src/ → 0
□ B2-B5: Numerical consistency scan — every abstract number matches body
□ B5:   Table IV values determined (recommendation: keep artifacts, add rounding footnote)
□ B6:   Threshold wording uses "binary sigmoid" not "argmax"
□ B7a-h: All 9 framing corrections applied (including N2, N3)
□ B8a-d: Housekeeping done
□ B9:   Systems benchmarks compressed
□ All:  `\ref`/`\label` cross-reference audit — no orphaned references
□ All:  `grep "RRD" Research_Paper.tex | grep -v "RRD_adv" | grep -v "%"` → only qualified uses
```

---

## GPU Rerun Sequence (Single RTX 3050 — FULLY SERIAL)

**Critical constraint**: G1, G2, G3, G5 run sequentially — NO GPU parallelism (C2, unanimous critics).
**GPU budget**: ~25-30h (M1) — includes 14.1h G1, thermal buffer, probs contingency.

### Pre-G1 Decision Gate (NEW — C1, unanimous critics)

**Before launching G1 (multi-seed, 14.1h), check preliminary results:**

1. **Quick single-seed check**: Run one seed of each config on the dedup'd split (~4h)
2. **Evaluate CIs**: Plot 95% confidence intervals for all 4 configurations
   - **CIs separate** → proceed with existing "ablation_b is superior" narrative + error bars
   - **CIs overlap** → pivot narrative from intra-ablation ranking to cross-baseline comparison (TF-IDF, FDG, clean-only). The claim shifts from "configuration X is best" to "all configurations beat non-neural baselines"

**Fallback narrative** (pre-written, ready to drop in):
> *"While overlapping confidence intervals preclude intra-ablation ranking, all four DistilBERT configurations significantly outperform both TF-IDF and perplexity baselines, establishing that task-specific fine-tuning — regardless of head depth or freeze strategy — substantially improves upon zero-shot and bag-of-words approaches."*

**Script location**: `src/evaluation/g1_decision_gate.py` (quick one-seed eval + CI plotting)

### G1. Multi-Seed Evaluation (~14.1h, CORRECTED from N6)

| Config | Per-Run Time | Seeds × Configs | Total |
|--------|-------------|-----------------|-------|
| baseline1 | 88min | 3 | 264min |
| ablation_c | 88min | 3 | 264min |
| ablation_a | 53min | 3 | 159min |
| ablation_b | 53min | 3 | 159min |
| **Total** | | | **14.1h (846min)** |

**Integration with C3**: Save `probs` (via `F.softmax` or `torch.sigmoid`) alongside predictions during each eval run. Cost: ~zero extra GPU time (already computing logits). This unblocks G5.

**Outputs**: `artifacts/multiseed_results.csv`: mean±std for all 4 configs × 3 seeds, including per-seed probabilities.

### G2. Clean-Only Training (~1h)

Train ablation_b on clean RAID only. Compare RRD with vs without adversarial data.

**Limitation acknowledgment (N2)**: "G2 trains on clean RAID vs original trains on adversarial DetectRL — two datasets differ." Update .tex after G2 results available.

### G3. FDG on Deduplicated Split (~4h)

Perplexity baseline on same dedup'd eval split. **Serial after G1** (single GPU constraint).

**Caveat (M5)**: Add explicit note: "FDG on dedup split is directional reference only; thresholds were not recalibrated."

### G4. Low-FPR Metrics (~2h, post-G1)

Post-hoc on saved predictions from G1. Recall at 1%/5%/10% FPR.

### G5. Calibration/ECE (~1h, post-G1, unblocked by C3 integration)

**Unblocked by**: Probs saved during G1 (C3 integration — zero extra GPU cost).
ECE scores for all configs.

### G6. Per-Attack DistilBERT Eval (~2h, post-G1)

Requires A3 (fixed schema).

### G7. Bootstrap Significance (~1h, post-G1)

Pairwise significance test p-values. Post-hoc analysis.

### G8. Count Tables (~0h, data analysis)

Per-domain/per-attack/per-generator counts. No GPU.

### GPU Execution Sequence

```
[Pre-G1 Gate, ~4h] → [G1, 14.1h] → [G3, 4h] → [G2, 1h] → [G4-G7, ~5-6h]
                         ↕ (prob persistence)
                     Post-hoc: G4, G5, G6, G7
```

**Wall clock**: ~23-24h sequential GPU + 4h pre-G1 gate = ~27-28h. With thermal buffer (4h idle/cooling breaks): **~25-30h total GPU wall time**.

**Thermal strategy**: Between G1 and G3, insert 30min cooldown if `nvidia-smi` shows >80°C. During G3→G2 transition, another cooldown if needed. Monitor with `nvidia-smi --loop=60`.

---

## Priority Triage Table (REVISED — M6)

**Change from v1**: Split success criteria into **Core** (must-pass for "complete") and **Extended** (nice-to-have, not required for top-tier). This resolves the P0-P2 contradiction where P2 tasks were simultaneously "minor" and "required for top-tier."

| Priority | Tasks | Effort | Why |
|----------|-------|--------|-----|
| **P0** | A0 (contamination audit), A1 (code crashes), A2 (dedup), A3 (column audit + fix), A1f (metrics dedup), B1 (FDG rename file + code + .tex), B3 (abstract AUROC) | ~23h | Blocking: paper has wrong numbers, crashes on review, or is scientifically misleading |
| **P1** | A4 (TF-IDF), A5 (threshold justification), B2 (RRD naming), B5 (Table IV), B6-B7 (framing, N2, N3), G1 (multi-seed), G2 (clean baseline), G5 (calibration), B8 (housekeeping) | ~18h | Major: reviewer flags, scientific integrity, paper polish |
| **P2** | B9 (systems compress), G4 (low-FPR), G6 (per-attack), G7 (bootstrap), G8 (count tables) | ~6h | **Extended** — improves completeness but not required for top-tier acceptance |

### Success Criteria (REVISED — M6)

#### Core (Minimum for submission-ready, ~30h)
- Contamination source verified (A0)
- All code crashes fixed (A1a-f)
- `compute_metrics` deduplicated (A1f)
- Dedup applied with dual reporting (A2)
- FDG renamed file + code + .tex (B1)
- Abstract AUROC range corrected (B3)
- RRD naming fixed (B2)
- Table IV resolved (B5)
- All framing corrections applied (B7a-i, including N2/N3)
- Decision threshold wording fixed (B6)
- Housekeeping done (B8)
- Multi-seed eval complete (G1)
- Clean-only baseline (G2)
- FDG on dedup'd split (G3)
- Calibration/ECE (G5)

#### Extended (Enhances completeness, +~15-20h)
- TF-IDF baseline (A4)
- 5% threshold justification (A5)
- Low-FPR metrics (G4)
- Per-attack eval (G6)
- Bootstrap significance (G7)
- Systems compression (B9)
- Count tables (G8)

---

## Dependency Matrix (CORRECTED — P1, P2, P3)

| Task | Depends On | Blocks | Changes from v1 |
|------|-----------|--------|-----------------|
| A0 (contamination audit) | None | A2 | **NEW** |
| A1a-f (code crashes) | None | All GPU reruns | — |
| A1f (metrics dedup) | None | B3, B5, B7 | **Expanded — new blocker** |
| A2 (dedup) | A0, A1b | G1, G2, G3, G5, G8 | **Now depends on A0** |
| A3 (attack column) | None (add audit step) | G6 | Added column consumer audit |
| A4 (TF-IDF) | A2 | None | — |
| A5 (threshold justification) | None | None | **NEW** |
| B1 (FDG rename) | None | None | **G3 decoupled (P1 — false dependency removed)** |
| B2-B9 | None | None | All independent text changes |
| G1 (multi-seed) | A1d, A2 | G4, G5, G6, G7 | **Prob saving integrated** |
| G2 (clean baseline) | A2 | None | — |
| G3 (FDG dedup) | A2 | None | **B1 dependency REMOVED**; runs **serial after G1** |
| G4 (low-FPR) | G1 | None | — |
| G5 (calibration) | G1 | None | **No longer blocked by separate prob pre-req** |
| G6 (per-attack eval) | A3, G1 | None | — |
| G7 (bootstrap) | G1 | None | — |
| G8 (count tables) | A2 | None | — |

### Critical Path

```
A0 → A2 → G1 → G4/G7   (~19h minimum, verified by critics)
     ↕
    A1b (upstream)
```

### Sequencing A+B (P3)

A2 edits `.tex` (new appendix section describing contamination/dedup). Sequence: A2 → then B tasks, or use `\input{}` isolation so A2 doesn't block B. **Recommendation**: Use `\input{appendix_contamination.tex}` — B tasks proceed in parallel, A2 fills the separate file.

---

## Execution Strategy

### Wave 0 — Pre-Track (Parallel, no deps)
```
A0 (contamination audit)
A1a-f (code crashes all in parallel)
A5 (threshold justification)
B1 (FDG rename — file + code + .tex)
B2-B9 (all text tasks in parallel)
```

### Wave 1 — After A0 + A1b Complete
```
A2 (dedup) ← depends on A0 + A1b
A3 (column audit + fix) ← independent (can start Wave 0)
```

### Wave 2 — After A2 Complete
```
A4 (TF-IDF)
Pre-G1 gate evaluation (~4h single-seed check on dedup'd split)
```

### Wave 3 — GPU Phase (FULLY SERIAL on RTX 3050)
```
[Thermal check]
  ↓
G1 (multi-seed, 14.1h) ← probability saving integrated
  ↓ [30min cooldown if needed]
G3 (FDG dedup, 4h)
  ↓ [30min cooldown if needed]
G2 (clean baseline, 1h)
  ↓
G4 (low-FPR, 2h — CPU post-hoc)
G5 (calibration, 1h — CPU post-hoc)
G6 (per-attack, 2h — CPU post-hoc)
G7 (bootstrap, 1h — CPU post-hoc)
G8 (count tables, ~0h)
```

### Wave 4 — Integration
```
All results folded into Research_Paper.tex
Final verification gate
```

---

## Time Estimates (Realistic — M1, N6)

| Phase | Hours | GPU? | Notes |
|-------|-------|------|-------|
| Track A (code + data) | 10.5h | No | 1h A0 + 2h A1 + 2h A2 + 1.5h A3 + 2h A4 + 1h A5 + 1h buffer |
| Track B (.tex + rename) | 12h | No | 2.5h B1 + 1h B2 + 0.5h B3 + 0.5h B4 + 1h B5 + 0.5h B6 + 3h B7 + 1h B8 + 1h B9 + 1h buffer |
| Pre-G1 gate | 4h | Yes | Single-seed quick check + CI plotting |
| G1 multi-seed | 14.1h | Yes | 4 configs × 3 seeds (corrected timing) |
| G2 clean baseline | 1h | Yes | Serial after G1 |
| G3 FDG dedup | 4h | Yes | Serial after G2 |
| G4-G7 post-hoc | 5-6h | No | CPU analysis on saved predictions |
| Thermal buffer | 3-4h | N/A | Cooldown breaks between GPU runs (M1) |
| **Total** | **~52h** | **~25-30h GPU** | Conservative, includes buffer |

---

## Verification Gates

### Gate 0: Pre-Execution
```
□ Contamination source documented in data/contamination_audit.md
□ compute_metrics deduplicated: grep returns exactly 1 definition
□ All 6 code crash fixes applied and verified
□ FDG file renamed; grep "fast_detectgpt" src/ → 0 results
```

### Gate 1: End of Track A (Pre-GPU)
```
□ A0:  Contamination audit complete
□ A1a-f: All 6 crash fixes + metrics dedup
□ A2:  Dedup splits — 0 overlapping texts
□ A3:  Column consumer audit complete; schema verified
□ A4:  TF-IDF metrics at artifacts/baselines/tfidf_metrics.json
□ A5:  5% threshold justification in .tex
□ All: py_compile passes on all modified files
```

### Gate 2: End of Track B
```
□ grep -c "Fast-DetectGPT" . → 0 (both code and .tex)
□ grep -c "fast_detectgpt" src/ → 0
□ Abstract numbers match body: AUROC 0.526–0.789, cross-attack values
□ grep "argmax" Research_Paper.tex → 0
□ grep "RRD" → only qualified uses (RRD_adv, Eq. 2)
□ Every \ref has matching \label; every \cite has matching \bibitem
```

### Gate 3: Pre-G1 Decision
```
□ Single-seed quick eval complete
□ 95% CIs computed and plotted
□ Decision made: existing narrative (CIs separate) OR fallback (CIs overlap)
□ Fallback narrative drafted if needed
```

### Gate 4: End of GPU Reruns
```
□ artifacts/multiseed_results.csv: mean±std for all 4 configs × 3 seeds
□ artifacts/clean_only/ablation_b_metrics.json
□ artifacts/fdg_dedup/fdg_metrics.json (with M5 caveat)
□ artifacts/low_fpr_recall.csv
□ artifacts/calibration/ece_scores.csv
□ artifacts/per_attack_eval/per_attack_metrics.csv
□ artifacts/bootstrap/bootstrap_significance.csv
```

### Final Verification
```
□ All gates GREEN
□ git diff --stat: no unintended changes
□ "RRD_adv" defined and used consistently
□ Deduplicated split = primary in abstract (N3)
□ Pre-G1 gate decision documented in paper
□ All TF-IDF, dedup, multi-seed results integrated
```

---

## Commit Strategy

### Commit 1: `fix(code): contamination audit + code crash fixes + metrics dedup`
**Scope**: A0, A1a-f (including A1f metrics deduplication)
**Files**: `data/contamination_audit.md` (new), `tc3_traindistilbert.py`, `tc4_.py`, `data/filter_raid_sequential.py`, `src/evaluation/metrics.py`, `requirements.txt`
**Verification**: All 6 crash fixes + single compute_metrics definition

### Commit 2: `fix(data): deduplication + column schema fix + TF-IDF baseline`
**Scope**: A2 (dedup), A3 (column fix + audit), A4 (TF-IDF)
**Files**: `data/filter.py` or `data/filter_raid_sequential.py`, `data/filter_raid_parallel.py`, `src/baselines/tfidf_baseline.py` (new)
**Verification**: 0 overlap in dedup splits; TF-IDF metrics file exists

### Commit 3: `fix(rename): FDG → GPT-2 XL perplexity baseline (file + code + paper)`
**Scope**: B1 — file rename, internal string audit, .tex rename, footnote
**Files**: `src/baselines/fast_detectgpt.py` → `src/baselines/gpt2_perplexity_baseline.py`, `Research_Paper.tex`, `bibliography.bib`
**Verification**: `grep -c "Fast-DetectGPT" .` = 0, `grep -c "fast_detectgpt" src/` = 0

### Commit 4: `fix(paper): framing, threshold justification, numerical consistency`
**Scope**: B2-B9, A5, N2, N3 — all text-only changes
**Files**: `Research_Paper.tex`
**Verification**: Abstract numbers match body; no "argmax"; RRD_adv defined; primary split designated

### Commit 5: `feat(eval): multi-seed, clean baseline, FDG dedup, calibration`  
**Scope**: GPU rerun results — G1, G2, G3 (actual GPU work)
**Files**: `artifacts/` (new results), `Research_Paper.tex` (results integrated)
**Verification**: All artifacts from Gate 4 present

### Commit 6: `feat(eval): post-hoc analysis — low-FPR, per-attack, bootstrap`
**Scope**: G4, G5, G6, G7, G8 — post-hoc CPU analysis
**Files**: `artifacts/` (new results), `Research_Paper.tex`
**Verification**: All analysis artifacts present

### Commit 7 (optional): `chore: final verification, pre-G1 gate documentation`
**Scope**: Any polish, pre-G1 decision documentation
**Files**: `.omo/plans/revised-paper-fix-plan-v2.md` (this plan), pre-G1 gate report

---

## Appendices

### Appendix A: FDG Baseline Resolution (Option C) — Verified

**Decision**: Rename FDG → "GPT-2 XL Perplexity Baseline" everywhere (file, code, .tex). Add mathematical footnote explaining the implementation difference.

**Why Option C**: Rename + footnote is transparent, defensible, ~2.5h effort (now including file rename). No GPU cost.

### Appendix B: Changes from v1

| Change | Source | Type |
|--------|--------|------|
| A0 contamination audit added before A2 | C4, P5 | New task |
| A1f expanded: metrics deduplication | M3 | Expanded |
| A3 expanded: column consumer audit | M2 | Expanded |
| A5 threshold justification added | N4 | New task |
| B1 expanded: file rename + internal string audit | M4, P4 | Expanded |
| B5: retain artifact values + footnote (no non-existent Table VI) | N1 | Revised |
| B7h: G2 clean baseline confound limitation | N2 | New subtask |
| B7i: designate primary split | N3 | New subtask |
| B7j: exact-match dedup footnote | N5 | New subtask |
| G1: prob saving integrated (zero extra GPU cost) | C3 | Integrated |
| G1: timing corrected to 14.1h | N6 | Corrected |
| Pre-G1 decision gate added | C1 | New gate |
| GPU phase: fully serial (no parallelism on single RTX 3050) | C2 | Structural |
| G3 not dependent on B1 | P1 | Fix |
| G3 threshold caveat added | M5 | Added |
| Critical path: A1b→A2→G1→G4/G7 | P2 | Documentation |
| GPU budget: 25-30h with thermal buffer | M1 | Revised |
| Success criteria: Core vs Extended split | M6 | Structural |
| Sequencing: A then B (not parallel) | P3 | Fix |

### Appendix C: Items Explicitly Rejected

Same as v1: LSP diagnostics, git branch strategy, AI prose check, near-dedup implementation (footnote only), multiple comparison correction, .gitignore fresh clone, orphaned LSP labels.
