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

### 1.4 Meta-Evaluation of External AI Reviews

The Claude review cross-evaluation (2026-05-24) introduced a new audit pattern: using 5 parallel agents to evaluate another AI's review of the codebase, rather than evaluating the codebase directly.

**Key finding**: Of **~28** claims from Claude's 4-prompt expert review, only **~46% (13)** were confirmed actionable, ~36% **(10)** were rejected as straw-man or directionally wrong, and ~18% **(5)** were directionally correct but wrong on specifics. Ratios vary by reviewer quality — this case yielded an approximately 46/36/18 split.

**Lesson**: Single-reviewer validation (one AI reviewing the codebase, one human accepting/rejecting) is insufficient. Multi-agent meta-evaluation of the reviewer catches:
- Hallucinated claims about non-existent files or already-fixed issues
- Claims that are technically correct but irrelevant to the project's specific architecture
- Directionally correct claims that need adjustment before implementation

**Recommended protocol for future external AI reviews**:
1. Accept no external AI review result as ground truth
2. Run a 3-5 agent parallel evaluation team against the review claims
3. Triage into actionable / rejected / nuanced categories
4. Only implement actionable items after independent code verification

### 1.5 Verifying External AI Recommendations Across Time Horizons

Claude's 13 actionable items targeted three different time horizons, each requiring a different verification strategy:

1. **Existing code** (e.g., "weight decay applied to wrong parameters"): Verify by reading the actual .py file on disk, tracing the code path
2. **Planned code** (e.g., "G0 threshold too tight"): Cross-reference against `.omo/plans/` to check if already addressed
3. **Infrastructure** (e.g., "HF_TOKEN from Kaggle Secrets"): Verify against Kaggle docs, environment config, or prior session knowledge

**Lesson**: A single verification strategy fails when claims target different time horizons. Always classify the claim's target (code / plan / infra) before verifying. Verifying an infrastructure claim against code on disk will produce a false negative (the claim looks wrong because the code for it doesn't exist yet).

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

- tokenizer vs tokeniser (US vs UK — paper uses British English "tokeniser")
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

### 3.4 Third-Party Recommendation Triage

Third-party (external AI) reviews produce a characteristic distribution: ~40% actionable, ~40% rejected/straw-man, ~20% directionally correct but wrong on specifics. Before implementing any third-party recommendation, each claim must be:

1. **Source-checked**: Is this claim about existing code, planned code, or infrastructure?
2. **Local-validated**: Is the reasoning correct for our specific architecture, dataset, and constraints?
3. **Priority-assessed**: Is the benefit worth the implementation cost? (Some technically valid findings may be P3 — nice-to-have but not blocking.)

The "directionally correct but wrong on specifics" category deserves special attention — these are claims whose general insight is valid but whose details (e.g., exact threshold value, recommended package version, specific API) need adjustment for the particular codebase. Do not reject or accept them outright; refine.

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

The project has a single RTX 4050 GPU. True parallelism is limited to CPU tasks. Yet every plan draft claimed "parallel execution" for GPU tasks.

**Lesson**: Model the actual hardware constraint explicitly. "5 parallel waves" is misleading when only 2 threads can run on 1 GPU.

### 4.3 Contingency Plans Need Quantified Gates

"Check if CIs overlap" is not actionable. "Plot 95% bootstrap CIs; if the upper bound of one config's CI overlaps with the lower bound of another's, flag as overlapping" is actionable.

**Lesson**: Vague decision criteria produce vague outcomes. Every gate needs a specific, falsifiable test.

---

## 5. Codebase-Specific Knowledge

### 5.1 The Two-Pipeline Problem

The project has two remaining data pipeline scripts:
1. `src/data/processing/filter_raid_parallel.py`
2. `src/data/processing/filter_raid_sequential.py`

They have different column schemas and different bugs. Fixing one doesn't fix the others. (The output filenames were normalized in P1.2 — see §5.3.)

### 5.2 The Four compute_metrics Problem

`compute_metrics` is now in 1 canonical location: `src/evaluation/metrics.py`. The 3 legacy scripts (`train_distilbert_detectrl.py`, `tc3_traindistilbert.py`, `train_distilbert_parallel.py`) were all deleted during Phase 0b consolidation, resolving the duplication concern.

### 5.3 The Parquet Name Mismatch (Resolved)

**Status**: RESOLVED (P-008 / P1.2 — both filter scripts now use `raid_train_pool.parquet` and `raid_test_unseen.parquet` consistently).

**Historical context**: The sequential pipeline originally output `train_pool.parquet` / `test_unseen.parquet` (missing `raid_` prefix), causing silent cache misses when scripts expected the `raid_`-prefixed names. Fixed during P1.2 by aligning the sequential pipeline's output naming with the parallel pipeline's convention.

### 5.4 Config Validation for Domain-Specific Constraints

The config system accepted any `head_type` string and any `freeze_layers` integer without domain validation. `head_type` values like `"deep"` vs `"deep_double"` would silently use defaults or crash at runtime. `freeze_layers=-1` or `freeze_layers=42` (DistilBERT only has 6 layers) would silently produce incorrect model configurations.  

**Fix**: Added post-load validation in `config.py` after `load_config()`: `head_type in {"single", "deep"}`, `0 <= freeze_layers <= 6`.  

**Lesson**: Config schemas for domain-specific parameters (model architecture, dataset names, generator IDs) need explicit value-range validation — not just type checking. A validated config catches misconfiguration at load time rather than 5 hours into a training run.

### 5.5 UTF-8 BOM Prevention When Using PowerShell for File Operations

PowerShell 5.1's `Set-Content -Encoding UTF8` writes UTF-8 WITH BOM even though the documentation says "UTF8 without BOM" (this was fixed in PowerShell 7+ but not backported). A single `Set-Content` call silently introduced a 3-byte BOM prefix (`EF BB BF`) into `src/data/processing/__init__.py`, which caused the file to fail Python's parser on some platforms.  

**Fix**: Used `python -c "..."` to write UTF-8 without BOM instead of PowerShell cmdlets.  

**Lesson**: Never trust PowerShell 5.1's `-Encoding UTF8` to produce BOM-free output on Windows. Use Python's `encoding='utf-8'` (which never writes BOM) for file operations where BOM matters. If using PowerShell 5.1, pipe through `New-Item -Force | Set-Content -NoNewline` or use `[System.IO.File]::WriteAllText()`.

### 5.6 RAID Official Metric is AUROC, Not F1

The RAID benchmark's official evaluation uses macro-averaged per-generator AUROC, not F1 score. F1 is threshold-dependent (requires choosing a classification cutoff) while AUROC is threshold-free. If the paper compares against RAID-published results using F1 instead of AUROC, the comparison is methodologically invalid.

**Lesson**: Always confirm the metric standard of the benchmark dataset before reporting comparisons. When multiple metrics are possible (F1, AUROC, TPR@FPR), report the benchmark's primary metric first. For RAID, this means: (1) per-generator AUROC for direct comparability, (2) F1 macro as an additional metric, (3) TPR@FPR for practical detection utility.

### 5.7 Weight Decay Must Exclude Bias and LayerNorm Parameters

PyTorch's `AdamW` applies `weight_decay` uniformly to all parameters. Standard BERT/DistilBERT fine-tuning practice (from the original BERT paper, Appendix A) explicitly excludes bias terms and LayerNorm parameters from weight decay via named parameter groups. Applying weight decay to these parameters:
- Penalizes LayerNorm's scale/shift transformations, which need to adjust freely
- Adds unnecessary regularization to bias terms (typically few hundred params each, but cumulatively thousands)
- Makes fine-tuning results slightly worse in a way that doesn't obviously crash but degrades convergence

**Fix**: Use `no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]` param group with `weight_decay=0.0`. This is a 5-line code change that follows standard BERT practice.

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

### 6.4 Line Ending Normalization via .gitattributes

The codebase had 12 files with CRLF line endings (2 mixed CRLF/LF, 10 pure CRLF) on a cross-platform project. Git's `core.autocrlf` is unreliable when the team uses different settings.  

**Fix**: Added `.gitattributes` with `* text=auto` and `*.py text eol=lf`.  

**Lesson**: `.gitattributes` is the canonical way to normalize line endings — it's checked into the repo and applies universally regardless of individual `core.autocrlf` settings. Best practice: define per-extension policies (`*.py text eol=lf`, `*.{yaml,yml} text eol=lf`, `*.md text eol=lf`) at the start of a project. Adding it retroactively can create a massive diff on renormalization.

### 6.5 LSP Server Not Installed in the Session Environment

During code quality sweeps, `lsp_diagnostics` could not provide Python diagnostics because no Python LSP server (pylance, pyright, basedpyright) was installed in the working environment. This eliminated the "type check before marking complete" verification step for Python files. All code verification had to rely on manual file reads, grep patterns, and parallel agent confirmation.  

**Lesson**: When working with a codebase that requires LSP-based verification, ensure the appropriate LSP server is installed and active before starting work. For Python projects, `npm install -g pyright` or `pip install basedpyright` provides `lsp_diagnostics` coverage. Without LSP, type-level bugs and import errors may not be caught by automated verification — extra care is needed in manual review.

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
| Involves git, env, or CI | ~~REPRODUCIBILITY~~ (folded into HOUSEKEEPING/INFRASTRUCTURE) |
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

4 Figure files existed on disk that were not referenced in the paper — resolved by M-027 (PNGs deleted from working tree). Likely corresponded to planned figures that were cut during page-limit trimming.

**Lesson**: Maintain a `figures/` manifest or clean unreferenced files before submission. Orphaned artifacts create confusion about what's included.

---

## 9. Codebase Health Takeaways

### 9.1 Unused Import Proliferation (Resolved)

**Current state**: All unused imports (`import os`, `import random`) were cleaned up across the codebase during the ultrawork verification loop (Waves 1-6). No remaining stale imports are present.

**Lesson**: When refactoring from `os.path` to `Path` across multiple files, always run an unused-import pass afterward. These are harmless individually but accumulate and obscure real code issues.

### 9.2 The Dead-Code Blind Spot (Resolved)

**Current state**: `src/training/trainer.py` (~240 lines) no longer defines a `Trainer` class. It exports functions (`seed_everything`, `train_ablation`, `run_epoch`, `make_stop_event`, `install_defensive_timer`, `move_batch_to_device`, `_compute_metrics`) and IS referenced — imported by both `scripts/train.py` and `scripts/evaluate.py`. This dead-code state was resolved during Phase 0b.

**Lesson**: Don't leave orphaned utility modules. If extracting a module, update the import chain immediately. Dead code is worse than no code — it creates false expectations about what the codebase provides.

### 9.3 Duplicate Divergence

Two copies of `filter.py` (`data/filter.py` and `src/data/filter.py`) started identical but diverged by 3 lines (different `ROOT_DIR`, different output prefixes). This is a maintenance time-bomb.

**Lesson**: Never copy-paste a module. Import it. If two scripts need different configurations, refactor into a shared function with parameters.

---

## 10. Future-Proofing Notes

### 10.1 Kaggle Platform — POC Run Complete

A proof-of-concept Kaggle dual-T4 run completed 2026-05-24 (serial single-seed, all 4 ablations). This validated the full pipeline on Kaggle infrastructure: data loading, training (3 epochs × 4 ablations), evaluation, metric persistence, and cross-session resume via dual-Dataset orchestration. Remaining work for platform-wide adoption:
- True GPU parallelism (G1 and G3 on separate GPUs) — depends on ProcessPoolExecutor with CUDA_VISIBLE_DEVICES
- Thermal/budget profiling (T4 server GPUs vs local laptop)
- Environment portability (Kaggle notebooks vs local Python)
- Data accessibility (Kaggle datasets vs local parquet files)

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

### 11.1 Taxonomy First, Then Populate

The registry was designed with an 8-category × 4-priority × 9-status taxonomy BEFORE populating issues. This forced consistent classification and prevented ad-hoc drift. **Lesson**: Define the dimensional axes before collecting data.

### 11.2 The Index↔Detail Pattern Prevents Entry Drop

The two-level structure (index table + detailed entries) served as a checksum: 36 index entries without detail entries were immediately visible during review. **Lesson**: Always have a redundant reference that cross-checks completeness.

### 11.3 Multi-Agent Review of Internal Documentation

Launching 5 parallel critics (taxonomy, plan cross-ref, source verification, hyperplan coverage, internal consistency) found 8+ structural issues that any single reviewer would miss. **Lesson**: Documentation review benefits from the same multi-lens approach as code review.

### 11.4 Dependency Graphs Reveal Hidden Assumptions

The dependency graph in the registry exposed false parallelism assumptions (G1∥G3 on a single GPU), implicit dependencies (A2→Docker), and cycle risks (A3 column fix → consumer audit). **Lesson**: A dependency graph is worth 20 lines of prose for revealing execution ordering problems.

### 11.5 Source Attribution Prevents Blame Drift

Every issue has S-tags (S1-S8) identifying its origin. When investigating which audit wave found what, this allows precise traceability. **Lesson**: Always record provenance — not just the issue, but who identified it and through what method.

---

## 12. Architecture-Level Lessons (2026-05-21/22 Session)

### 12.1 The Silent `_freeze_layers()` Bug

`tc4_.py` (since renamed to `scripts/benchmark.py` and migrated to `src/models/distilbert_classifier.py`) defined `DistilBertClassifier.__init__(freeze_layers=0)` with a parameter that was silently ignored — `self._freeze_layers()` was never called in `__init__`. This means every DistilBertClassifier instance performed full DistilBERT fine-tuning regardless of the `freeze_layers` argument. All reported benchmark results from tc4_ are affected. The canonical `DistilBertClassifier` now lives in `src/models/distilbert_classifier.py` with the `_freeze_layers()` call correctly wired.

**Lesson**: When a class constructor accepts a parameter that is stored but never used, it's not just dead code — it's a silent correctness bug. The parameter's existence misleads readers into thinking it's active. Always trace parameter usage through the full call chain during code review.

### 12.2 ThreadPoolExecutor + CUDA = Crash

The plan initially recommended `ThreadPoolExecutor` for Kaggle task parallelism (running two DistilBERT training jobs concurrently). Oracle caught that `ThreadPoolExecutor` with CUDA causes a crash because CUDA contexts are not thread-safe. The correct alternative is `ProcessPoolExecutor`, which spawns separate processes with independent CUDA contexts.

**Lesson**: CUDA parallelism with threads is a well-known footgun. Always use process-based parallelism (multiprocessing) when running concurrent GPU workloads. This applies to Kaggle's dual T4 where task parallelism across two physical cards requires process isolation.

### 12.3 Multi-Momus Validation Converges on Blind Spots

Three independent Momus reviewers on the same plan found different issues BUT converged on 3 common findings (scikit-learn pinning, SIGALRM platform, data/__init__.py existence). The convergent findings are the highest-confidence issues — if all 3 experts independently flag the same thing, it's almost certainly correct.

**Lesson**: Single-reviewer validation misses issues that multi-reviewer panels catch. Multi-Momus adds cost (~3× the tokens) but produces higher-confidence validation, especially for convergent findings. Use single Momus for routine plans, multi-Momus for critical-path decisions.

### 12.4 Know When to Defer — Enhancement vs Requirement

The DeBERTa-v3-vs-ModernBERT-vs-DistilBERT research wave (May 2026) produced a clear map of the tradeoffs, but the right decision was to **defer**. Key signals:

1. **Reviewer gap analysis**: DeBERTa cross-architecture validation was never listed in any reviewer gap item. Adding it strengthens the paper but doesn't close any required gap.
2. **Risk/reward asymmetry**: FlashDeBERTa v0.0.7 is unproven on actual workloads. If it fails, the fallback (FP32 LoRA) consumes 2× the GPU time, breaking the 30h quota.
3. **Fit within constraints**: DistilBERT-alone at 22.3h fits cleanly in one Kaggle week. Adding DeBERTa pushes past the quota boundary with no safety margin.

**Lesson**: The best optimisation is often the one you don't make. Every new feature carries an integration risk that compounds with unproven dependencies. Deferring non-essential enhancements preserves budget for the core work. The analysis was worth doing — it confirmed DeBERTa is not needed — but acting on it would have been a mistake.

Also, the research wave itself uncovered a significant error: the "~4.87 GiB" VRAM claim for DeBERTa-v3+LoRA on L4 was actually **Time-to-First-Token** (4.87 ms), not memory. The actual VRAM was 6.657 GiB (Act-LoRA paper Table 4). This 37% understatement would have silently invalidated any T4 memory budget calculations. Always verify numeric claims from external analyses against primary sources — and watch for column misreading in tables.
