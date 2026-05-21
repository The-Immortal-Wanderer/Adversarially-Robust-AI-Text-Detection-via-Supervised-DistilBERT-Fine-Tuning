# Audit Trail — ANN_Project

**Purpose**: Chronological record of every audit wave, finding, decision, and fix applied.  
**Format**: Entries ordered by date, most recent first. Each entry references issue IDs from `MASTER_REGISTER.md`.

---

## 2026-05-21 — Round 4 Structural Fix Cycle — Multi-Source Sync (+3 files, tc4_ bug)

- **Action**: Full structural re-audit across 4 dimensions: registry, plan files, companion files, issue mapping
- **Parallel audit agents**: 4 explore agents (registry structural, plan sync, companion files, Section 10 mapping)
- **Fixes applied**:
  1. Fixed `tc4_` freeze_layers bug — `self._freeze_layers()` was never called despite constructor parameter, all
     DistilBertClassifier instances silently did full fine-tuning instead of frozen-layer mode
  2. Added superseded header to `revised-paper-fix-plan-v2.md`
  3. Fixed LESSONS_LEARNED.md section numbering (8.1→10.1, 8.2→10.2, 8.3→10.3, #9→#11)
  4. Added 3 missing lessons (tc4_ bug, ProcessPoolExecutor, multi-Momus validation)
  5. Updated AUDIT_TRAIL.md stale stats (130→104 issues, 21→25 files, 6→7 plans)
  6. Added this entry + Momus panel entry to AUDIT_TRAIL.md
- **Status**: Files synced. Next step: fix Section 10 issue mapping gaps in end-to-end plan.
- **Registry Statistics**: 121 category assignments (104 unique entries: 94 active, 10 rejected)

---

## 2026-05-21 — End-to-End Restructure Plan Created + Momus-Validated (3/3 PASS)

- **Action**: Created `.omo/plans/end-to-end-restructure-plan.md` — 6-wave, 28-task execution plan
- **Pre-planning**: 4 explore agents → Oracle architecture review → Plan agent (28 tasks) → cross-ref against 31 issues
- **Momus panel**: 3 independent reviewers, all PASS with convergent caveats (3 fixes applied)
- **Key decisions**: ProcessPoolExecutor (not ThreadPool), single checkpoint, on-the-fly tokenization on Kaggle, Phase 2 split, G0 decision gate
- **Status**: Plan ready for execution. All GPU compute on Kaggle. Laptop handles code editing + LaTeX only.

---

## 2026-05-21 — Issue Registry Audit + Expansion (104 Issues, +12 New Entries)

- **Action**: 5-agent parallel review of MASTER_REGISTER.md uncovered 8 structural flaws, 2 priority conflicts, 3 category mismatches, and 36 missing detailed entries
- **Fixes applied**:
  1. Added status transition diagram
  2. Fixed header count (130→92→104)
  3. Generated detailed entries for M-001—M-026 and N-001—N-010 (36 entries)
  4. Added 12 missing entries: P-016—P-019 (pre-requisites/A1e), M-034—M-041 (hyperplan corrections)
  5. Fixed P-001 validation line reference (L47-72→L1-5+L163-197)
  6. Fixed P-006 autocast count (7 sites→2 unguarded sites)
  7. Fixed P-011-P-015 header (removed incorrect A1d reference)
  8. Added missing categories to N-011/N-012 (PAPER,HOUSEKEEPING)
  9. Fixed conflict log N-001→M-008 label
  10. Updated category summary table totals (142 total category tags)
  11. Updated index table with 12 new rows
  12. Added hyperplan source (S4//S5) to N-002

- **Trigger**: User request for systematic, categorized issue tracking system
- **Method**: 5 parallel explore agents (taxonomy review, plan cross-ref, source verification, hyperplan coverage, internal consistency)
- **Registry Statistics** (updated):
  - Total: 121 category assignments (104 unique entries: 94 active, 10 rejected)
  - By priority (category-assignment counts): 31 P0, 45 P1, 20 P2, 15 P3, 10 REJECTED
  - By status: 74 PLANNED, 37 PROPOSED, 10 REJECTED
  - By category: 18 PAPER, 19 CODE, 8 DATA, 11 EXPERIMENT, 28 FRAMING, 23 HOUSEKEEPING, 4 INFRASTRUCTURE, 10 REJECTED

---

## 2026-05-21 — Comprehensive Issue Registry Completed (130 Issues)

- **Action**: Completed population of `.omo/issues/MASTER_REGISTER.md` with all issues
- **Trigger**: User identified memory-based tracking as "catastrophic" — needed systematic classification system
- **Method**: 4 parallel background explore agents mined issues from:
  1. `revised-paper-fix-plan-v2.md` (plan tasks, verification gates, prerequisites)
  2. `hyperplan2-synthesis.md` + `hyperplan-synthesis.md` (21 hyperplan findings + 6 Critical Gaps A-F)
  3. Codebase (all .py files — unused imports, dead code, bare excepts, security concerns)
  4. `Research_Paper.tex` + git (orphaned figures, orphaned labels, overclaiming, LaTeX compliance)

### Registry Statistics (initial)
- **Total**: 92 issues tracked (82 active, 10 rejected)
- **By priority**: 22 P0, 20 P1, 18 P2, 34 P3, 10 REJECTED
- **By status**: 54 PLANNED (in v2 plan), 38 PROPOSED (newly discovered, not in plan), 10 REJECTED
- **By category**: 18 PAPER, 19 CODE, 7 DATA, 9 EXPERIMENT, 18 FRAMING, 19 HOUSEKEEPING, 2 INFRASTRUCTURE, 2 REPRODUCIBILITY
- **Source coverage**: S1 (Sisyphus), S2 (GPT 5.5), S3 (Gemini), S4 (hyperplan 1), S5 (hyperplan 2), S6 (Momus), S7 (deduction), S8 (deep codebase scan)

### Post-Audit Expansion (same day)
- +12 new entries (P-016—P-019, M-034—M-041) from hyperplan cross-ref and missing plan items
- Total: 104 issues (94 active, 10 rejected)

### Key New Findings (Not in v2 Plan)
- 4 unreferenced figure files on disk (M-027)
- Hyperparameters table missing `\label` (M-028)
- Missing `\usepackage{hyperref}` (M-029)
- README/paper title mismatch: RAID vs DetectRL (M-030/31)
- Abstract "adversarially exposed" non-standard (M-032)
- Cross-attack generalisation not actually tested (M-033)
- 9 orphaned `\label` definitions (N-011)
- 10 housekeeping items (unused imports, dead code, docstring errors, etc.)

### Current State Summary
- Registry: Complete with 121 category assignments (104 unique entries — 94 active, 10 rejected), dependency graph, validation log, conflict resolution log
- Git: 4 commits, 25 unstaged modified files
- Plans: 7 plan files in `.omo/plans/`
- Paper: Research_Paper.tex, 14 pages, 22 \bibitem, 28 \ref{}, 35 \label{}

---

## [Entries below this line are from previous sessions — captured from memory + session history]

---

## 2026-05-13 — Momus Panel Validates v2 Plan (Second Hyperplan Cycle)

- **Action**: 3 Momus reviewers evaluated `revised-paper-fix-plan-v2.md`
- **Result**: 2/3 PASS (with minor caveats), 1/3 FAIL (B6 zombie task, B4 provenance)
- **Key findings from Momus**:
  - G2/G3 ordering contradiction (time estimates vs execution sequence)
  - B6 ("0.5 argmax") target text doesn't exist in .tex
  - B4 (14.99%) lacks provenance in plan (only traceable to base plan.md)
  - A4 (TF-IDF) underspecified
  - B7a line reference wrong (L1→L18)
- **Status**: Plan approved with noted rough edges. Awaiting user direction before execution.

## 2026-05-13 — Hyperplan Cycle 2 (Adversarial Team)

- **Team**: ground-critic (unspecified-low), deep-critic (unspecified-high), logic-critic (ultrabrain), lateral-critic (artistry)
- **Round 1**: 42 issues found on the revised plan (16 ground, 10 deep, 8 logic, 8 lateral)
- **Round 2**: Cross-attack — each critic evaluated the other 3's findings
- **Round 3**: Defend/concede/refine — deep conceded 2, logic conceded 1, lateral conceded 1
- **Synthesis**: 21 defensible insights bundled into `hyperplan2-synthesis.md`
- **Key structural changes** from hyperplan:
  1. Pre-G1 decision gate (contingency for claim collapse)
  2. GPU phase fully serial (single RTX 3050, no imaginary second GPU)
  3. A0 contamination audit before A2 dedup
  4. compute_metrics deduplicated
  5. FDG file rename + internal string audit
  6. Core vs Extended success criteria
  7. GPU budget corrected to 25-30h (was 21h)
- **Output**: `.omo/plans/revised-paper-fix-plan-v2.md`

## 2026-05-13 — Hyperplan Cycle 1 (Adversarial Team)

- **Team**: lead, ground-critic, deep-critic, logic-critic, lateral-critic
- **Evaluation of base plan** (`.omo/plans/paper-fix-plan.md`):
  - ground-critic: 12 issues found, time estimates 40-80% low
  - logic-critic: 3 dependency matrix errors (false blocks, missing sequencing constraints, transitive dep)
  - lateral-critic: 8 blind spots
- **Round 2 cross-attack**: logic-critic elevated/demoted 34 findings, deep-critic rated all findings, ground-critic cross-attacked 26 findings, lateral-critic found 4 meta-patterns
- **Consensus**: Plan was structurally sound but underspecified in time, dependencies, and contingencies
- **Output**: `.omo/plans/revised-paper-fix-plan.md` (v1)

## 2026-05-13 — Gemini Deep Audit (Wave 2)

- **Prompt**: Comprehensive deep audit — 46 items across 11 areas (A-K)
- **Findings confirmed as new**:
  - RRD math may violate Eq. 2 definition in paper
  - fast_detectgpt.py is perplexity not curvature (triple-confirmed)
  - attack column overwrite destroys per-attack analysis
  - tc3 cache crash (unseen_cache dependency)
  - sequential parquet naming mismatch (train_pool vs raid_train_pool)
- **Findings already fixed**: AUROC range, title overclaim, conclusions
- **Output**: Cross-referenced into master issue register

## 2026-05-13 — Gemini Deep Audit (Wave 1)

- **Prompt**: Systematic forensic audit of codebase against paper claims
- **Key new findings**:
  - Low-FPR metrics missing (most critical for production AI detection)
  - Calibration (ECE) completely absent from paper
  - AMP/quantization speedup is confounded with other optimizations
  - Seed arg parsing missing from training scripts
  - Sequence length ablation absent
- **Already fixed items**: overclaiming, RRD formula, single-seed limitation note, clean baseline limitation

## 2026-05-13 — IBCAST Submission + Post-Submission Gap Analysis

- **Action**: Paper submitted to IBCAST 2026 as-is (14 pages, IEEEtran format)
- **Post-submission gap map** organized in 5 tiers:
  - Tier 0 (fatal): dedup, multi-seed, clean baseline
  - Tier 1 (methodological): matched FDG, per-attack, low-FPR, calibration, prevalence
  - Tier 2 (baseline gaps): Binoculars, RAID RoBERTa, TF-IDF
  - Tier 3 (systems claims): AMP/quantization confounds
  - Tier 4 (narrative): three papers in one, dense abstract
  - Tier 5 (reproducibility): tagged release, reproduce commands
- **Venue readiness**:
  - Workshop/student track: Ready as-is
  - Mid-tier applied: Needs dedup + multi-seed + clean baseline + per-attack
  - Top-tier venue: Needs ALL of the above + matched baselines + low-FPR + multi-benchmark
  - Journal: Needs theoretical grounding + cross-architecture comparison

## 2026-05-13 — GPT-5.5 Evaluation Integration

- **GPT-5.5 score**: 2.4/5 for top-tier venues
- **Already fixed** before GPT review: p-values, overlap language, single-seed limitation, RRD threshold
- **Newly added from GPT recommendations**:
  - Broader impact / ethics statement
  - Data and code availability section
  - 4 new citations (Tufts 2025, Pedrotti 2025, PADBen 2025, Detecting the Machine 2026)
  - Figure count: 9 → 5 (merged confusion matrices)
  - Table count: 15 → 12
  - Softened "immediate practical significance" → "practically relevant"
- **IEEE lane hardening**: IEEEtran compliance fixes, student IDs → emails, GitHub URL, figure paths

## 2026-05-13 — Final Pre-Submission Verification Sweep

- **Actions**:
  - Added 3 more \ref{} calls (total → 24)
  - "demonstrates" → "indicates" in conclusion (last overclaiming instance)
  - Verified all 26 \ref{}/\eqref{} map to valid labels
  - Verified all 38 \cite{} → 22 \bibitem{} fully mapped
  - Zero overclaiming instances remaining
  - Cross-checked abstract ↔ conclusion ↔ table metrics consistency
  - LSP diagnostics: 0 errors
- **Status**: Paper declared submission-ready for IBCAST

## 2026-05-13 — Deep-Dive Verification Wave (10 items, 7 fixed)

- **Agents**: 4 oracle agents (statistical rigor, codebase verification, LaTeX structural, figure/table data)
- **10 issues found, 7 fixed**:
  1. 38/50 labels orphaned — no cross-references to key tables/figures
  2. Confusion matrix footnotes falsely claimed "Derived algebraically" — 0.9161 vs 0.9157 mismatch
  3. "gradient-step count by approximately half" wrong — 3000→750 = 75% not 50%
  4. Single-seed evaluation absent from Limitations
  5. 6,029 overlap "modestly inflate" understates — all 5K held-out human texts are duplicates
  6. RRD 5% threshold uncited
  7. Conclusion directional caveat buried
- **Deferred** (non-blocking): table* ordering, metrics.py macro F1, 14.99% rounding

## 2026-05-13 — Multi-Agent Paper Re-Audit (11 critical fixes)

- **5 oracle agents**: argumentation, prose/numerical, numerical cross-checks, citations/reproducibility, limitations-to-claims mapping
- **11 fixes applied**:
  - Abstract threshold qualified
  - "proving" → "indicating/suggesting" (2 instances)
  - "directly contradicts" → "challenges"
  - "targets that gap" → "examines the adjacent question"
  - DLO duty cycle range corrected
  - "p≈0.000" → "p<0.001"
  - 3.28× → 3.29× speedup
  - quantisation spelling unified (11 prose instances)
  - "hardware ceiling" → "OOM threshold"
  - "compact supervised detectors" → "a DistilBERT-based detector"
  - IEEEtran compliance (itemize → numbered paragraphs)
  - RQ1 caveat added to Introduction

## 2026-05-13 — Five Remaining Fixes from Gap Analysis

- **14-item gap analysis executed**: 9 clean, 2 partial, 3 open → all resolved
- **5 pending fixes applied**:
  - Data leakage paragraph added to Limitations
  - "Significant increase" → "~1.8–2.6× duty cycle improvement" in efficiency table
  - NUM_WORKERS docstring fix (4→6) in train_distilbert_parallel.py (3 edits)
  - Clean-only baseline limitation paragraph added
  - 6 overclaiming instances softened (proves→demonstrates, first→an, confirms→indicates, etc.)

## 2026-05-13 — Soft Reset + Gap Analysis (14 Items)

- **Action**: Dropped staged commit (177b859) via soft reset, preserved all 22 files' edits
- **14-item gap analysis across codebase + .tex**:
  - 9 items clean: TC renaming, F1 metric naming, domain count alignment, quantization confound text, FDG domain-shift, tokeniser spelling, checkpoint paths, data leakage acknowledgment, "unseen family" reframe
  - 2 partial: RRD formula, overclaiming
  - 3 open: data leakage in Limitations, "Significant increase" vague, NUM_WORKERS docstring, clean-only baseline missing
- **Result**: All 14 gaps closed after fix wave

## 2026-05-13 — Comprehensive Audit (4 Passes, ~40+ Issues Fixed)

- **Audit Pass 1**: 5 major structural issues (quantization confound, broken TC→RAID pipeline, domain shift in FDG, VRAM illusion, docstring mismatch)
- **Audit Pass 2**: 7 more issues (residual TC1-TC4 labels, 5 vs 8 domain mismatch, dueling RRD formulas, thread occupancy, vague metrics, AI model details, tokenizer spelling)
- **Audit Pass 3**: 10 more issues (no clean baseline, F1 macro vs binary, unseen family wording, overclaiming, model selection consistency, coursework names, LaTeX paths, apples-to-oranges comparison, 6029 overlap, DetectRL vs RAID narrative)
- **Applied fixes verified**:
  - Checkpoint path fallbacks in tc4_.py (63.89% latency reduction confirmed)
  - Parquet safety in dataloader.py
  - Domain pruning in filter scripts (5 active domains)
  - US→UK spelling (tokenizer→tokeniser)
  - TC renaming plan (TC1→PP, TC2→DLO, TC3→AMP-T, TC4→AMP-I)
  - RRD recalculation (50.12%→33.40%, 17.65%→14.99%)
  - Hardware confound reframing in Figure 9 caption
  - Baseline VRAM transparency (8,000 OOM for RoBERTa)
  - Domain shift decoupling in discussion
- **Commits**: f25bde2, 43d11c0, 0912a57

---

## 2026-05-13 — Initial Codebase Analysis

- **First read**: README.md, requirements.txt, all source modules
- **Key discoveries**:
  - Project: AI-generated text detection, DistilBERT fine-tuning
  - 4 configs (baseline1 + ablations a/b/c)
  - RAID dataset (not DetectRL — fixed from earlier state)
  - Paper in IEEE format for IEEE venue
  - Cross-attack generalization study
- **Initial issues identified**:
  - DetectRL→RAID history — code defaulted to DetectRL causing 99.8% F1 from shortcut learning
  - Figure paths may not resolve from subdirectory
  - TC coursework nomenclature still present
