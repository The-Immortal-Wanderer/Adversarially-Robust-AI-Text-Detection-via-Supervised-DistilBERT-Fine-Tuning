# End-to-End Paper Fix Plan: Remaining Gaps After IBCAST Submission

## TL;DR

> **Quick Summary**: 20 verified issues remain after ~36 fixes applied across 4 audit waves. Fixes split into 4 categories: (A) baseline integrity — FDG actually computes perplexity, RRD violates Eq. 2, abstract range wrong, confusion matrix mismatch; (B) code reproducibility — 5 crash bugs on fresh clone; (C) paper framing — title, diversity claims, production comparisons; (D) scientific depth — low-FPR, calibration, sample counts, significance.

> **Estimated Effort**: ~20 hours immediate fixes (no GPU) + ~25 hours GPU reruns = ~45 hours total
> **Parallel Execution**: YES — 5 parallel waves
> **Critical Path**: Wave 1 (FDG fix) → Wave 2 (crash bugs) → Wave 4 (low-FPR eval) → final verification

---

## Context

### History
The paper "Towards Adversarially Robust AI Text Detection via Supervised DistilBERT Fine-Tuning" was submitted to IBCAST 2026. Post-submission, a comprehensive gap analysis identified ~56 issues across 4+ audit waves (manual, GPT 5.5, Gemini rounds 1-2). ~36 issues were fixed in-session. 20 verified issues remain.

### Verification Methodology
Every issue in this plan was validated by reading the actual file on disk. No claims accepted from agent reports without independent verification.

---

## Work Objectives

### Core Objective
Eliminate all remaining fixable gaps so the paper is submission-ready for top-tier venues (NeurIPS/ACL/EMNLP).

### Concrete Deliverables
- Research_Paper.tex: corrected numbers, titles, phrasing, limitations
- src/baselines/fast_detectgpt.py: either actual FDG curvature or renamed + documented
- tc3_traindistilbert.py, tc4_.py: no crash bugs on fresh clone
- data/filter_raid_sequential.py: output naming aligned
- data/filter_raid_parallel.py: attack column preserved
- requirements.txt: complete dependencies
- Missing tables added (per-domain, per-attack, per-generator counts)

### Must Have
- FDG baseline integrity: either actual curvature or honestly renamed
- Numerical consistency: every metric matches its source table
- Reproducibility: all scripts run without crashes on fresh clone
- Abstract accuracy: every number verifiable from body text

### Must NOT Have
- No "first/novel/groundbreaking" claims without citations
- No incommensurable metric comparisons presented as conclusive
- No ambiguous training time claims

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Baseline Integrity — paper cannot be submitted without these):
├── W1-T1: FDG scoring function — verify vs rename vs implement
├── W1-T2: RRD naming — qualify 33.40% as "Adversarial AUROC Drop"
├── W1-T3: Abstract AUROC range — 0.669→0.789 → 0.526→0.789
├── W1-T4: gpt-j-6B RRD — add to abstract + Section V.C
├── W1-T5: Table IV vs Table VI alignment — recalculate all metrics
└── W1-T6: Decision threshold — declare 0.5 argmax in Section III

Wave 2 (Reproducibility — repo must work on fresh clone):
├── W2-T1: tc3 unseen cache crash — pretokenize + fallback path
├── W2-T2: Sequential parquet naming — add raid_ prefix
├── W2-T3: tc4 checkpoint paths — expand POSSIBLE_PATHS
├── W2-T4: CPU autocast/GradScaler guards — all 7 call sites
├── W2-T5: requirements.txt — add pyarrow + bitsandbytes
└── W2-T6: metrics.py F1 variant — macro → binary

Wave 3 (Paper Framing — honest representation):
├── W3-T1: Title softening
├── W3-T2: "adversarially diverse" qualified
├── W3-T3: "demonstrates" → "suggests" (L1052)
├── W3-T4: AMP speedup disentangled in text
├── W3-T5: "Under 16 minutes" clarified (one config vs full)
├── W3-T6: Conclusion "favourable robustness margin" softened
└── W3-T7: Significance table labeled + referenced

Wave 4 (Scientific Depth — experimental reruns required):
├── W4-T1: Low-FPR metrics (recall at 1%/5%/10% FPR)
├── W4-T2: Calibration (ECE) + prevalence-adjusted precision
├── W4-T3: Attack column hijack fix + per-attack DistilBERT eval
├── W4-T4: Statistical significance tests between ablations
├── W4-T5: Per-domain / per-attack / per-generator count tables
├── W4-T6: Matched FDG comparison (if curvature implemented)
└── W4-T7: Clean gpt2-xl baseline run for true RRD

Wave 5 (Housekeeping):
├── W5-T1: Limitations — energy/power, RAID noise, unmatched comparison
├── W5-T2: Filter script docstrings updated
├── W5-T3: Per-config peak VRAM in wall-clock table
└── W5-T4: AUROC delta rounding fix (0.0001 typo)

Final Verification:
├── F1: LSP diagnostics — 0 errors, 0 warnings
├── F2: Figure/path audit — all \includegraphics files exist
├── F3: Cross-reference audit — every \ref matches \label, every \cite matches \bibitem
├── F4: Abstract-vs-body numerical consistency scan
└── F5: Full git diff review — no unintended changes
```

### Dependency Matrix
- W1-T1 (FDG) blocks W4-T1, W4-T4, W4-T6 (if curvature implemented)
- W1-T2 through W1-T6 are independent — can run in parallel
- W2-T1 through W2-T6 are independent code fixes
- W3-T1 through W3-T7 are independent prose fixes
- W4-T1 through W4-T3 are independent eval script changes
- W4-T4, W4-T6, W4-T7 depend on actually running evaluation (GPU time)
- W5-T1 through W5-T4 are independent trivial fixes

---

## TODOs

### Wave 1 — Baseline Integrity (no GPU, ~4 hours)

- [ ] W1-T1. **FDG Baseline Verification & Fix**

  **What to do**:
  1. Read `src/baselines/fast_detectgpt.py` scoring function (`_score_text`, lines 163-197)
  2. Compare against original Fast-DetectGPT algorithm (Mitchell et al., 2023):
     - Original: computes conditional probability curvature by perturbing input tokens
     - Current: computes cumulative log-likelihood (standard perplexity)
  3. Decision: if baseline IS perplexity (verified), rename the baseline everywhere in the paper
  4. Paper changes needed:
     - Abstract: "Fast-DetectGPT is reproduced..." → "A GPT-2 XL perplexity baseline is reproduced..."
     - Section IV-D: rename all references
     - Conclusion: rename
  5. OR implement actual curvature (requires ~10-12 hours + GPU rerun — see Appendix)

  **Must NOT do**:
  - Do not claim Fast-DetectGPT curvature if only perplexity is implemented
  - Do not delete the perplexity baseline — it is a valid zero-shot baseline, just misnamed

  **QA Scenarios**:
  - Confirm `_score_text` returns cumulative log-likelihood (not curvature)
  - Confirm paper uses "Fast-DetectGPT" name >5 times — all must be renamed
  - Confirm bibliography entry for Mitchell et al. 2023 still valid (it cites the original)

- [ ] W1-T2. **RRD Naming — 33.40% is Not a Valid RRD**

  **What to do**:
  1. Eq. 2 (L508-510) defines RRD = (F1_seen - F1_held-out) / F1_seen × 100  
  2. The 33.40% value for gpt2-xl uses: (a) AUROC not F1, (b) paraphrase as "seen" (paraphrase IS adversarial)
  3. This violates Eq. 2. Rename everywhere:
     - Abstract L60: "gpt2-xl RRD: 33.40%" → "gpt2-xl adversarial degradation: 33.40% (AUROC drop: paraphrase→homoglyph)"
     - Section V.C L1052: "RRD" → "adversarial AUROC degradation"
     - Table caption: clarify this is not Eq. 2 RRD
  4. Add a footnote in Section IV-C defining this as a separate metric: RRD_adv

  **References**:
  - `.tex` L508-510: Eq. 2 definition
  - `.tex` L1052: current RRD claim
  - `.tex` L60: abstract claim

- [ ] W1-T3. **Fix Abstract AUROC Range**

  **What to do**: Change L59 from `(AUROC 0.669--0.789)` to `(AUROC 0.526--0.789)` to reflect gpt2-xl's near-random collapse under homoglyph

- [ ] W1-T4. **Add gpt-j-6B RRD to Abstract**

  **What to do**: Change L60 from `(gpt2-xl RRD: 33.40\%)` to `(gpt2-xl degradation: 33.40\%, gpt-j-6B degradation: 14.99\%)` (matching renamed metric)

- [ ] W1-T5. **Fix Table IV Confusion Matrix Metrics**

  **What to do**: Recompute all ablation_c diagnostic metrics from the confusion matrix in Table VI:
  - Seen: TP=5886, TN=5461, FP=539, FN=114
  - S.Acc = (5461+5886)/12000 = 0.9456 (was 0.9453)
  - S.Prec = 5886/6425 = 0.9161 (was 0.9157)
  - S.F1 = 2×0.9161×0.9810/(0.9161+0.9810) = 0.9474 (was 0.9472)
  - Held-out: TP=4641, TN=4622, FP=378, FN=359
  - H.Acc = (4622+4641)/10000 = 0.9263 (was 0.9261)
  - H.Prec = 4641/5019 = 0.9247 (was 0.9243)
  - H.F1 = 2×0.9247×0.9282/(0.9247+0.9282) = 0.9264 (was 0.9263)
  - Update L802 in Research_Paper.tex with corrected values

- [ ] W1-T6. **Declare Decision Threshold**

  **What to do**: In Section III-F (Model Evaluation), add: "All DistilBERT classifications use the standard 0.5 argmax decision threshold on the positive-class sigmoid output."

---

### Wave 2 — Reproducibility (no GPU, ~2 hours)

- [ ] W2-T1. **Fix tc3 Unseen Cache Crash**

  **What to do**:
  1. In `tc3_traindistilbert.py`, after L277 (pretokenize call for train), add:
  ```
  if unseen_cache.exists():
      print(f"Loading unseen cache from {unseen_cache}")
  else:
      print(f"Unseen cache not found at {unseen_cache}, pretokenizing...")
      df_unseen = pd.read_parquet(f"data/processed/{DATASET}_test_unseen_10k.parquet")
      pretokenize(df_unseen, tokenizer, unseen_cache)
  ```
  2. Ensure the cache file naming matches: `{DATASET}_test_unseen_10k_{MAX_LENGTH}.pt` — verify `DATASET` is "raid" not empty

- [ ] W2-T2. **Fix Sequential Preprocessor Output Names**

  **What to do**: In `data/filter_raid_sequential.py`, L221-222, change:
  - `train_pool.parquet` → `raid_train_pool.parquet`
  - `test_unseen.parquet` → `raid_test_unseen.parquet`
  - Update file header docstring (L11-13)

- [ ] W2-T3. **Expand tc4 Checkpoint Path Fallback**

  **What to do**: In `tc4_.py`, expand `POSSIBLE_PATHS` to include both directories:
  ```python
  POSSIBLE_PATHS = [
      Path("artifacts/distilbert_detector/ablation_b_best.pt"),
      Path("artifacts/distilbert_detector/raid_ablation_b_best.pt"),
      Path("artifacts/distilbert_detector_tc3/ablation_b_tc3_best.pt"),
      Path("artifacts/distilbert_detector_tc3/raid_ablation_b_tc3_best.pt"),
      Path("artifacts/distilbert_detector_tc3/raid_ablation_b_tc3_best_fp32.pt"),
  ]
  ```

- [ ] W2-T4. **Fix CPU Autocast/GradScaler Crashes**

  **What to do**: In both tc3 and tc4, wrap all CUDA-specific calls:
  - `torch.autocast("cuda", ...)` → guard with `if DEVICE.type == "cuda"`
  - `GradScaler(enabled=True)` → `GradScaler(enabled=(DEVICE.type == "cuda"))`
  - `torch.cuda.synchronize()` → guard with `if DEVICE.type == "cuda"`
  - Affected lines: tc3 L188, L230; tc4 L49, L55, L59, L87, L92

- [ ] W2-T5. **Update requirements.txt**

  **What to do**: Append to `requirements.txt`:
  ```
  pyarrow>=14.0.0
  bitsandbytes>=0.43.0
  ```

- [ ] W2-T6. **Fix metrics.py F1 Variant**

  **What to do**: In `src/evaluation/metrics.py` L17, change:
  ```python
  "f1_macro": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
  ```
  to:
  ```python
  "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
  ```
  (default average="binary" for binary classification)

---

### Wave 3 — Paper Framing (no GPU, ~3 hours)

- [ ] W3-T1. **Soften Title**

  **What to do**: Change title from "Towards Adversarially Robust AI Text Detection..." to something that reflects held-out-generator evaluation rather than general adversarial robustness.
  - Suggestion: "Towards Generator-Generalizable AI Text Detection Under Adversarial Conditions: A Held-Out-Generator Study on RAID"
  - If title length is a concern (IEEEtran), shorter: "Held-Out-Generator Generalization in Adversarial AI Text Detection"

- [ ] W3-T2. **Qualify "Adversarially Diverse"**

  **What to do**: Search for "adversarially diverse" (L40-41, L107-108, L349-350) and replace with precise language:
  - "stratified across two attack types (homoglyph substitution and word substitution)"
  - Or "adversarially augmented with character- and word-level perturbations"

- [ ] W3-T3. **Soften L1052 "demonstrates"**

  **What to do**: Change "demonstrates that attack type... serves as the dominant determinant" to "indicates that attack type... serves as a primary determinant"

- [ ] W3-T4. **Disentangle AMP Speedup in Text**

  **What to do**: In the AMP speedup discussion, add explicit breakdown:
  - "The reported 3.32x training speedup reflects a composite benefit: (1) batch size increase from 32 to 128 reduces gradient steps by 4x, and (2) FP16 math accounts for the remainder. A pure FP16-with-same-batch-size control would be needed to isolate these factors."

- [ ] W3-T5. **Clarify "Under 16 Minutes"**

  **What to do**: Change L1293 from "under 16 minutes" to "under 16 minutes per configuration." If the full 4-config sweep was used, update to reflect the actual total.

- [ ] W3-T6. **Soften Conclusion "Favourable Robustness Margin"**

  **What to do**: The conclusion at L1293-1294 already has the caveat at the front. However, "achieves a favourable robustness margin" still overstates the unmatched comparison. Change to: "exhibits competitive held-out performance under adversarial conditions, acknowledging the comparison is directional."

- [ ] W3-T7. **Label + Reference Significance Table**

  **What to do**:
  1. Add `\label{tab:bootstrap}` inside the significance table (currently unlabeled)
  2. Add `Table~\ref{tab:bootstrap}` in the text where the bootstrap test is discussed (L1047)

---

### Wave 4 — Scientific Depth (GPU reruns required, ~25 hours)

- [ ] W4-T1. **Low-FPR Metrics**

  **What to do**:
  1. Add evaluation function to `src/evaluation/metrics.py` that computes recall at 1%, 5%, 10% FPR
  2. Run on all 4 configurations' predictions
  3. Add table to paper: "Recall at Fixed FPR for All Ablation Configurations"
  4. Discuss in Results section

- [ ] W4-T2. **Calibration + Prevalence-Adjusted Precision**

  **What to do**:
  1. Compute Expected Calibration Error (ECE) for all configs
  2. Compute precision (PPV) at 5% and 10% AI prevalence using Bayes' rule from held-out metrics
  3. Add as discussion section or appendix table

- [ ] W4-T3. **Fix Attack Column Hijack + Per-Attack DistilBERT Eval**

  **What to do**:
  1. In `data/filter_raid_parallel.py`, modify L217 to preserve original `attack_type` AND generator name
  2. Rerun data pipeline
  3. Create new table: "Per-Attack DistilBERT Performance on Held-Out Generators"
  4. Add discussion of which attack types are most destructive for supervised models

- [ ] W4-T4. **Statistical Significance Between Ablations**

  **What to do**:
  1. Run bootstrap difference tests on validation F1 between all pairwise configs
  2. Add "No statistically significant difference was detected between ablation_b and ablation_c" or similar
  3. Add to discussion

- [ ] W4-T5. **Add Domain/Attack/Generator Count Tables**

  **What to do**: Add brief subsection in experimental setup:
  - Per-domain counts in training pool
  - Per-attack counts in training pool
  - Per-generator counts in train vs held-out split
  - Can be a single compact table

- [ ] W4-T6. **Matched FDG Baseline (if curvature implemented)**

  **What to do**: Only if W1-T1 chose to implement actual Fast-DetectGPT curvature. Run FDG on the exact same held-out splits as DistilBERT. Report both F1 and AUROC for direct comparison.

- [ ] W4-T7. **Clean gpt2-xl Baseline for True RRD**

  **What to do**: Run gpt2-xl zero-shot evaluation on clean (non-adversarial) text to get a true clean AUROC. Then compute actual RRD = (clean - adversarial) / clean × 100.

---

### Wave 5 — Housekeeping (no GPU, ~1 hour)

- [ ] W5-T1. **Expand Limitations**

  **What to do**: Add to Limitations:
  - "Energy and power consumption were not measured despite the hardware-democratization framing"
  - "RAID dataset quality issues (potential labeling noise, text corruption) are not analyzed"
  - "The DistilBERT vs Fast-DetectGPT comparison remains directional until a matched evaluation is run"

- [ ] W5-T2. **Fix Filter Script Docstrings**

  **What to do**: Update docstrings in both filter scripts to match `raid_` prefix output filenames

- [ ] W5-T3. **Add Per-Config Peak VRAM to Wall-Clock Table**

  **What to do**: Add column or note for peak VRAM per config, or state "per-config peak VRAM was within 8 GB in all cases"

- [ ] W5-T4. **Fix AUROC Delta Rounding**

  **What to do**: Change 0.2637 → 0.2636 and 0.1181 → 0.1180 in the bootstrap table

---

## Final Verification Wave

- [ ] F1. **LSP Diagnostics** — Run `texlab` diagnostics on Research_Paper.tex. 0 errors, 0 warnings.

- [ ] F2. **Figure/Path Audit** — Every `\includegraphics{}` path points to an existing file on disk.

- [ ] F3. **Cross-Reference Audit** — Every `\ref{}` has a matching `\label{}`. Every `\cite{}` has a matching `\bibitem{}`. No orphaned labels.

- [ ] F4. **Abstract-vs-Body Consistency** — Every number in the abstract matches the body text. Read every value and verify.

- [ ] F5. **Git Diff Review** — Review all changes for unintended modifications. No stale drift.

---

## Commit Strategy

- Wave 1-2-3: `fix(all): baseline integrity + reproducibility fixes`
  - Files: Research_Paper.tex, src/baselines/fast_detectgpt.py, tc3_*.py, tc4_.py, filter_raid_*.py, metrics.py, requirements.txt

- Wave 4: `feat(eval): low-FPR, calibration, per-attack results`
  - Files: Research_Paper.tex, src/evaluation/metrics.py, filter_raid_parallel.py

- Wave 5: `fix(docs): limitations, docstrings, minor corrections`
  - Files: Research_Paper.tex, filter_raid_*.py

---

## Success Criteria

### Verification Commands
```bash
# LSP check
texlab --diagnostics Research_Paper.tex  # Expected: 0 errors

# Syntax check Python files
python -m py_compile src/baselines/fast_detectgpt.py  # Expected: exit 0
python -m py_compile tc3_traindistilbert.py  # Expected: exit 0
python -m py_compile tc4_.py  # Expected: exit 0

# Verify figure files exist
ls figures/figure*.png  # Expected: 5 files (not 9)
```

### Final Checklist
- [ ] W1-T1 through W1-T6 complete — baseline integrity restored
- [ ] W2-T1 through W2-T6 complete — repo runs on fresh clone
- [ ] W3-T1 through W3-T7 complete — framing matches evidence
- [ ] W4-T1 through W4-T7 complete — scientific depth added (or acknowledged)
- [ ] W5-T1 through W5-T4 complete — housekeeping done
- [ ] F1 through F5 complete — final verification passed

---

## Appendix: FDG Baseline Decision Tree

```
Does fast_detectgpt.py compute actual curvature?
  YES → Keep name, continue (verify against original paper algorithm)
  NO  → Two options:

  Option A (Recommended, 1 hour):
    → Rename everywhere to "GPT-2 XL Perplexity Baseline"
    → Honest, defensible, no GPU time
    → Weakens the baseline comparison but preserves integrity

  Option B (10-12 hours + GPU):
    → Implement actual conditional probability curvature
    → Requires: perturbation loop, top-k alternative scoring, debugging OOM
    → Strengthens the baseline comparison
    → Choose this ONLY if the paper targets NeurIPS/ACL/EMNLP
```

The prompt is designed to explicitly push the LLM toward Option A by naming it recommended and cheap, while allowing Option B for top-tier targets.
