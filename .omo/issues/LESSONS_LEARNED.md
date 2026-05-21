# Lessons Learned — ANN_Project

**Purpose**: Cumulative knowledge extracted from all audit waves, fix cycles, and review rounds.  
**Status**: Living document — add as new lessons emerge.

---

## 1. Audit Methodology

### 1.1 Multi-AI Independent Auditing Works — But Only With Cross-Reference

The project used Sisyphus, GPT 5.5, and Gemini as independent auditors across 4+ waves. Each found different issues. **Critical rule**: Before accepting ANY external AI audit finding, cross-reference against:
1. What's already been fixed (session history)
2. What's on disk right now (actual source files)
3. Whether the finding is logically coherent (not hallucinated)

**Example**: GPT 5.5 flagged missing ethics/data availability sections. These were valid. But GPT also claimed "p-values incorrectly reported" — those had already been fixed in the previous session.

**Rule**: Always verify, never trust.

### 1.2 The Check-Your-Work Anti-Pattern

Multiple times, a "corrected" value turned out to be wrong because the correction was itself computed from faulty logic. Examples:
- RRD was "corrected" to 33.40% / 14.99% but used the wrong baseline in the formula
- B5 "corrected" Table IV values differed from actual data by <0.0004 with no source trace

**Lesson**: When "correcting" numbers, always trace back to the original data / confusion matrix, not to another derived value.

### 1.3 Always Read the Actual Code Before Believing a Claim

Multiple audit findings were confidently asserted but turned out to be:
- Fixed already (but auditor was looking at old state)
- About a different file than claimed
- Logically flawed (didn't account for code paths)

**Example**: "Probability persistence gap" — one auditor claimed probs were already persisted. Reading the actual code proved otherwise (probs were computed but consumed for ROC-AUC then discarded).

---

## 2. Common Issue Categories

### 2.1 Overclaiming Is the Most Common Paper Issue

Across all audit waves, overclaiming was the single most common paper issue:
- "proves" → "demonstrates" or "indicates"
- "first" → "an"
- "unprecedented" → removed entirely
- "substantially outperforms" → directional with caveats
- "directly contradicts" → "challenges"
- "most capable" → "a compact model"
- "significant increase" (unquantified) → specific range

**Pattern**: Claims that sound impressive but lack evidence, quantification, or comparison.

**Prevention**: After any draft pass, do a dedicated overclaiming grep: `proves|demonstrates|confirms|first|unprecedented|substantially|significantly|directly|establishes|reveals|shows that`

### 2.2 Terminology Inconsistency Is Pervasive

- tokenizer vs tokeniser (US vs UK — paper uses Britsh English "tokeniser")
- quantisation vs quantization (same — unified to "quantisation")
- generalisation vs generalization (unified to "generalisation")
- Fast-DetectGPT vs GPT-2 perplexity (misleading — the real issue)
- RRD (Relative Robustness Degradation) vs plain "degradation" or "drop"

**Lesson**: Define every term ONCE in a terminology table. Run a consistency grep before submission.

### 2.3 Numerical Inaccuracies Cluster Around Extremes

The most common numerical issues:
- RRD formula (two competing definitions in different places)
- Percentages near 0 or 100 (p "≈0.000" → "p < 0.001")
- Speedup factors (3.28× → 3.29×, wrong base)
- Duty cycle ranges (unsupported precision)
- AUROC bounds (abstract upper/lower vs actual values)

**Pattern**: Numbers that look "clean" (round numbers, simple fractions) should be double-checked first. Numbers near boundaries (100%, 0%) are often wrong.

---

## 3. Validation Patterns

### 3.1 The Validation Gate Principle

Every fix must have a verifiable gate BEFORE implementation:
- Wrong: "Fix the AUROC range in the abstract"
- Right: "Change L59 from `0.669--0.789` to `0.526--0.789`. Verify: `grep "0.669--0.789" .tex` returns 0"

**Why**: Without a gate, the fixer doesn't know when they're done.

### 3.2 Three-Point Validation

For any reported issue, validate three things before recording:
1. **Does the file actually exist?** (not a hallucinated path)
2. **Does the line actually contain the claimed text?** (timestamps matter — files change)
3. **Is the inference logically sound?** (did the auditor correctly interpret the code?)

### 3.3 Regression Awareness

Every fix can break something else. Known regression vectors:
- Changing `attack_type` column schema in filter scripts → breaks downstream consumers (src/data/dataloader.py, evaluation scripts)
- Renaming files → breaks imports in other files
- Changing .tex line numbers → breaks cross-references in plan documents

---

## 4. Planning Takeaways

### 4.1 Time Estimates Are Always Too Optimistic

Across all planning rounds:
- Original estimates: ~35h total
- Hyperplan 1 corrected to: 40-80% higher
- Hyperplan 2 corrected to: ~52h including thermal buffer
- Actual GPU: 25-30h (not 13-16h as originally claimed)

**Lesson**: Always add a 30-50% buffer for research code. GPU thermal throttling is real on consumer hardware.

### 4.2 Parallelism Is Overstated

The project has a single RTX 3050 GPU. True parallelism is limited to CPU tasks. Yet every plan draft claimed "parallel execution" for GPU tasks.

**Lesson**: Model the actual hardware constraint explicitly. "5 parallel waves" is misleading when only 2 threads can run on 1 GPU.

### 4.3 Contingency Plans Need Quantified Gates

"Check if CIs overlap" is not actionable. "Plot 95% bootstrap CIs; if the upper bound of one config's CI overlaps with the lower bound of another's, flag as overlapping" is actionable.

**Lesson**: Vague decision criteria produce vague outcomes. Every gate needs a specific, falsifiable test.

---

## 5. Codebase-Specific Knowledge

### 5.1 The Three-Pipeline Problem

The project has three data pipelines:
1. `data/filter.py` — legacy DetectRL pipeline (used by original train script)
2. `data/filter_raid_parallel.py` — RAID pipeline (used by train_distilbert_parallel.py)
3. `data/filter_raid_sequential.py` — RAID pipeline (sequential variant)

They produce different output filenames, have different column schemas, and have different bugs. Fixing one doesn't fix the others.

### 5.2 The Four compute_metrics Problem

`compute_metrics` is defined in 4 places:
1. `src/evaluation/metrics.py` — the canonical location
2. `train_distilbert_detectrl.py` — local definition
3. `tc3_traindistilbert.py` — local definition
4. `train_distilbert_parallel.py` — local definition

They may compute F1 differently (binary vs macro weighted). This means the same eval metric could produce different numbers depending on which script was run.

### 5.3 The Parquet Name Mismatch

- Parallel pipeline outputs: `raid_train_pool.parquet`, `raid_test_unseen.parquet`
- Sequential pipeline outputs: `train_pool.parquet`, `test_unseen.parquet` (missing `raid_` prefix)

This causes silent cache misses when scripts expect the `raid_`-prefixed names.

---

## 6. Process Lessons

### 6.1 Commit Discipline Matters

The previous session had an unauthorized commit (177b859) — the user explicitly said commits should not have been made without permission.

**Lesson**: Never commit without explicit user request. The orchestration agent does NOT own the git tree.

### 6.2 Hyperplan Pipeline Is Effective But Expensive

The 2-round hyperplan (R1 → R2 → R3 → synthesis → plan → Momus) produced high-quality output but at significant token cost. Each round involved 4 critics + lead, each reading and evaluating multiple documents.

**Cost-benefit**: High. The hyperplan caught issues that would have caused 50h of wasted execution. But should only be used for critical path decisions.

### 6.3 File-Existence Verification Is Non-Negotiable

Multiple audit findings referenced files that:
- Didn't exist (bibliography.bib — paper uses embedded thebibliography)
- Had different names (figures renamed across versions)
- Had been deleted (figure6_pdc_speedup.html/png already deleted)

**Rule**: Before diagnosing a file issue, verify the file exists.

---

## 7. Classification Heuristics

Use these heuristics when categorizing new issues:

| If the issue... | Category |
|----------------|----------|
| Changes what the paper SAYS | PAPER |
| Changes what the paper CLAIMS | FRAMING |
| Changes a Python file | CODE |
| Changes how data is processed | DATA |
| Requires running a GPU job | EXPERIMENT |
| Involves git, env, or CI | REPRODUCIBILITY |
| Is about formatting, deps, cleanup | HOUSEKEEPING |
| Is about GPU, RAM, hardware | INFRASTRUCTURE |

---

## 8. Root-Cause Patterns

### 8.1 The Duplication Anti-Pattern

The codebase has multiple forms of problematic duplication:
- **Code duplication**: 4 implementations of `compute_metrics`, 3 filter scripts, 2 copies of `filter.py` (diverged)
- **Data duplication**: 6,029 overlapping human texts across splits
- **Claim duplication**: Same numbers repeated in abstract, body, conclusion without synchronized updates

**Root cause**: No shared library. Each script defines its own infrastructure. Changes to one don't propagate.

**Prevention**: Centralize shared logic (metrics calculation, data loading, evaluation) into `src/` package with single source of truth. Do NOT allow inline function definitions in training scripts.

### 8.2 The Overclaiming Cascade

Overclaiming starts at the claim → infects the abstract → propagates to introduction → hardens in conclusion. Each level adds intensifiers:
- Initial: "DistilBERT showed lower RRD under adversarial conditions"
- Abstract: "substantially outperforms baselines"
- Introduction: "first comprehensive demonstration"
- Conclusion: "proves adversarial robustness"

**The cascade is unidirectional**: Once an intensifier lands in the abstract, the conclusion must echo it. Fixing requires weakening ALL levels simultaneously.

**Prevention**: Start with the weakest defensible claim and only add intensifiers where the data specifically supports them. A single "proves" in the conclusion requires p < 0.01 with 5+ seeds and matched comparison.

### 8.3 The README Divergence Pattern

README.md describes a previous version of the project. Over time, the paper and code evolved (DetectRL→RAID, cross-attack→held-out-generator), but README was never updated. This creates confusion for anyone reading the repo to understand the paper.

**Lesson**: README is a first-class document. Update it whenever the dataset, evaluation protocol, or contribution narrative changes. A stale README undermines reproducibility claims.

### 8.4 The Orphaned-Artifact Pattern

4 figure files exist on disk but are never referenced in the paper. These were generated by the analysis pipeline but not integrated into the manuscript. Likely corresponded to planned figures that were cut during page-limit trimming.

**Lesson**: Maintain a `figures/` manifest or clean unreferenced files before submission. Orphaned artifacts create confusion about what's included.

---

## 9. Codebase Health Takeaways

### 9.1 Unused Import Proliferation

4 files have unused `import os` or `import random`. These are harmless but accumulate. Over 5+ scripts, this suggests the code was refactored from `os.path` to `Path` incrementally without cleaning imports.

### 9.2 The Dead-Code Blind Spot

`src/training/trainer.py` (217 lines) defines a full `Trainer` class with AMP, gradient accumulation, LR scheduling, and early stopping. **Zero references anywhere** — the training scripts all use ad-hoc loops instead. This is dead code that wastes reader attention.

**Lesson**: Dead code is worse than no code — it creates false expectations about what the codebase provides. Delete it or integrate it.

### 9.3 Duplicate Divergence

Two copies of `filter.py` (`data/filter.py` and `src/data/filter.py`) started identical but diverged by 3 lines (different `ROOT_DIR`, different output prefixes). This is a maintenance time-bomb.

**Lesson**: Never copy-paste a module. Import it. If two scripts need different configurations, refactor into a shared function with parameters.

---

## 10. Future-Proofing Notes

### 10.1 Kaggle Migration (Upcoming)

User plans to switch to Kaggle with dual T4 GPUs. This will:
- Allow true GPU parallelism (G1 and G3 could run simultaneously)
- Change thermal profile (T4s are server GPUs with better cooling)
- Require environment portability (Kaggle notebooks vs local Python)
- Need data accessibility (Kaggle datasets vs local parquet files)

### 10.2 Top-Tier Venue Requirements

Beyond current fixes, top-tier venues likely require:
- Multi-benchmark evaluation (not just RAID)
- Cross-architecture comparison (RoBERTa, Llama, etc.)
- Theoretical grounding for RRD
- Human evaluation study
- Ablation on model size (DistilBERT vs BERT vs RoBERTa)

### 10.3 Paper Restructuring

If dedup + multi-seed produces different results, the paper narrative may need restructuring:
- Current: "DistilBERT is robust (all configs <5% RRD)"
- Alternative: "DistilBERT degrades gracefully (2-4% RRD vs 15-33% for perplexity baselines)"
- Or: "Supervised fine-tuning reduces variance across attacks (std 1.2% vs 6.8% for baselines)"

The pre-written fallback narrative in the v2 plan covers the collapse scenario.

---

## 11. Issue Registry Design

### 9.1 Taxonomy First, Then Populate

The registry was designed with an 8-category × 4-priority × 9-status taxonomy BEFORE populating issues. This forced consistent classification and prevented ad-hoc drift. **Lesson**: Define the dimensional axes before collecting data.

### 9.2 The Index↔Detail Pattern Prevents Entry Drop

The two-level structure (index table + detailed entries) served as a checksum: 36 index entries without detail entries were immediately visible during review. **Lesson**: Always have a redundant reference that cross-checks completeness.

### 9.3 Multi-Agent Review of Internal Documentation

Launching 5 parallel critics (taxonomy, plan cross-ref, source verification, hyperplan coverage, internal consistency) found 8+ structural issues that any single reviewer would miss. **Lesson**: Documentation review benefits from the same multi-lens approach as code review.

### 9.4 Dependency Graphs Reveal Hidden Assumptions

The dependency graph in the registry exposed false parallelism assumptions (G1∥G3 on a single GPU), implicit dependencies (A2→Docker), and cycle risks (A3 column fix → consumer audit). **Lesson**: A dependency graph is worth 20 lines of prose for revealing execution ordering problems.

### 9.5 Source Attribution Prevents Blame Drift

Every issue has S-tags (S1-S8) identifying its origin. When investigating which audit wave found what, this allows precise traceability. **Lesson**: Always record provenance — not just the issue, but who identified it and through what method.

---

## 12. Architecture-Level Lessons (This Session)

### 12.1 The Silent `_freeze_layers()` Bug

`tc4_.py` defines `DistilBertClassifier.__init__(freeze_layers=0)` with a parameter that was silently ignored — `self._freeze_layers()` was never called in `__init__`. This means every DistilBertClassifier instance performed full DistilBERT fine-tuning regardless of the `freeze_layers` argument. All reported benchmark results from tc4_ are affected.

**Lesson**: When a class constructor accepts a parameter that is stored but never used, it's not just dead code — it's a silent correctness bug. The parameter's existence misleads readers into thinking it's active. Always trace parameter usage through the full call chain during code review.

### 12.2 ThreadPoolExecutor + CUDA = Crash

The plan initially recommended `ThreadPoolExecutor` for Kaggle task parallelism (running two DistilBERT training jobs concurrently). Oracle caught that `ThreadPoolExecutor` with CUDA causes a crash because CUDA contexts are not thread-safe. The correct alternative is `ProcessPoolExecutor`, which spawns separate processes with independent CUDA contexts.

**Lesson**: CUDA parallelism with threads is a well-known footgun. Always use process-based parallelism (multiprocessing) when running concurrent GPU workloads. This applies to Kaggle's dual T4 where task parallelism across two physical cards requires process isolation.

### 12.3 Multi-Momus Validation Converges on Blind Spots

Three independent Momus reviewers on the same plan found different issues BUT converged on 3 common findings (scikit-learn pinning, SIGALRM platform, data/__init__.py existence). The convergent findings are the highest-confidence issues — if all 3 experts independently flag the same thing, it's almost certainly correct.

**Lesson**: Single-reviewer validation misses issues that multi-reviewer panels catch. Multi-Momus adds cost (~3× the tokens) but produces higher-confidence validation, especially for convergent findings. Use single Momus for routine plans, multi-Momus for critical-path decisions.

Every issue has S-tags (S1-S8) identifying its origin. When investigating which audit wave found what, this allows precise traceability. **Lesson**: Always record provenance — not just the issue, but who identified it and through what method.
