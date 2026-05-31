# Master Issue Register — ANN_Project

**Last updated**: 2026-05-30  
**Maintainer**: Sisyphus (Noire)  
**Source documents**: `.omo/plans/`, `Research_Paper.tex`, codebase (all .py files), GPT 5.5/Gemini audits, hyperplan adversarial reviews  
**Total issues tracked**: **114** (102 active + 12 rejected)

---

## 1. Taxonomy

### 1.1 Categories

| Tag | Scope | Examples |
|-----|-------|---------|
| `[PAPER]` | .tex writing, narrative structure, sections, LaTeX | Cross-references, figure paths, missing sections, caption formatting |
| `[CODE]` | Python bugs, crashes, incorrect implementations, imports | CPU autocast, checkpoint fallback, compute_metrics duplication, dead code |
| `[DATA]` | Data pipeline, contamination, preprocessing, split integrity | 6,029 overlap, attack_type hijack, parquet naming, duplicate filter.py |
| `[EXPERIMENT]` | Needs GPU re-run, multi-seed, new baselines, ablations | Multi-seed eval (G1), clean-only (G2), FDG on dedup (G3), calibration (G5) |
| `[FRAMING]` | Overclaiming, narrative exaggeration, terminology inflation | "proves"/"first"/"superior", threshold justification, cross-attack scope |
| ~~`[REPRODUCIBILITY]`~~ | ~~Seeds, git tags, env lock, reproduce commands~~ | ~~No tagged release, no env lock, split hashes undocumented~~ ?? *Deprecated → concerns folded into [HOUSEKEEPING] and [INFRASTRUCTURE]. No entries currently tagged.* |
| `[HOUSEKEEPING]` | Cleanup, formatting, deps, LSP, unused code | requirements.txt missing deps, unused imports, stale docstrings, orphaned labels |
| `[INFRASTRUCTURE]` | GPU, env, dependencies, tooling | Single GPU, raw data download, CUDA version, Kaggle migration |
| `[BASELINE]` | Zero-shot detection baselines (Binoculars, Perplexity) | N-020: missing from peer-baseline comparison |

### 1.2 Priority Scale

| Priority | Definition | Examples |
|----------|------------|---------|
| **P0** | Submission-blocking. Paper has wrong numbers, crashes on reviewer execution, or is scientifically misleading. | FDG misnamed, AUROC range wrong, contamination invalidates results, CPU autocast crash |
| **P1** | Major. Reviewers will flag, scientific integrity concern, significant polish needed. | Missing clean baseline, RRD terminology, calibration absent, orphaned figures, README mismatch |
| **P2** | Moderate. Strengthens paper but not submission-critical. | TF-IDF baseline, per-attack breakdown, low-FPR metrics, systems compression, dead code removal |
| **P3** | Polish. Cosmetic or convenience improvements. | Unused imports, mid-file imports, wrong run commands in docstrings, missing \hyphenation |
| **REJECTED** | Considered and explicitly dismissed. | multiple-comparison correction, near-dedup implementation, LSP diagnostics |

### 1.3 Status Flow & Definitions

```
PROPOSED --validation--? VALIDATED --plan--? PLANNED --work--? IN_PROGRESS --fix? RESOLVED --verify? VERIFIED --? CLOSED
    →                       →
    +--? REJECTED (if FP)   +--? PLANNED (accepted)
                              +--? SUPERSEDED (resolved by broader fix)
```

- **PROPOSED**: Identified, not yet independently validated
- **VALIDATED**: Confirmed real by source file verification
- **REJECTED**: False positive after validation
- **PLANNED**: Resolution approach decided, may be in a wave plan
- **IN_PROGRESS**: Being actively worked on
- **RESOLVED**: Fix applied, pending verification
- **VERIFIED**: Confirmed working
- **CLOSED**: Final, no longer actionable
- **SUPERSEDED**: Resolved by broader fix
- **COMPLETED**: End state → action item done without formal verification (used for Phase 0 tasks)
- **INACTIVE**: Project-scope items no longer pursued (superseded by changed project direction)

### 1.4 Source Tags

| Tag | Source |
|-----|--------|
| S1 | Sisyphus own analysis |
| S2 | GPT 5.5 audit |
| S3 | Gemini audit |
| S4 | Hyperplan critic team (1st cycle) |
| S5 | Hyperplan critic team (2nd cycle) |
| S6 | Momus plan review |
| S7 | Deduced / inferred from existing issues |
| S8 | Codebase deep scan (unused imports, dead code) |
| S9 | Research wave literature survey (2026-05-21) |
| S10 | Ultrawork Wave 2 code quality sweep |
| S11 | 2026-05-24 → Claude Review Evaluation | 5-agent parallel analysis of 40+ claims across hardware, code, methodology, operations |
| S12 | Open-ended verification wave (2026-05-28) | Broad codebase/paper/companion/git exploration without checklists |

---

## 2. Issue Index

| ID | Category | Prio | Title | Status | Plan Ref | Source |
|----|----------|------|-------|--------|----------|--------|
| P-001 | PAPER, CODE | P0 | FDG misnamed to GPT-2 XL Perplexity Baseline | RESOLVED | B1 (COMPLETED -- rename + internal audit done) | S1,S2,S3,S4,S5 |
| P-002 | EXPERIMENT, FRAMING | P0 | Multi-seed eval could falsify central claim | PLANNED | Pre-G1,G1 | S4,S5 |
| P-003 | DATA, CODE | P0 | Human-text contamination (6,029 overlaps) | RESOLVED | A0,A2 | S1,S2,S3,S4 |
| P-004 | PAPER, FRAMING | P0 | Abstract AUROC range wrong (0.669-0.526) | RESOLVED | B3 (fix verified at L58-61) | S2,S6 |
| P-005 | PAPER, FRAMING | P0 | 5% RRD threshold unjustified | RESOLVED | B8 (deferred to Limitations) | S4,S6 |
| P-006 | CODE | P0 | CPU autocast crash on non-CUDA devices | COMPLETED | P0.5 (was A1d) | S1,S3,S4 |
| P-007 | CODE, DATA | P0 | Attack column hijack (overwrites attack_type) | COMPLETED | A3 | S3,S4 |
| P-008 | DATA | P0 | Sequential parquet naming mismatch (no raid_ prefix) | COMPLETED | A1b | S3,S4 |
| P-009 | CODE, FRAMING | P0 | FDG comparison is directional, not matched | PLANNED | B1,G3 | S1,S2,S4 |
| P-010 | CODE | P0 | Contamination source unverified (A0 needed) | RESOLVED | A0 (contamination audit completed) | S5 |
| P-011 | CODE | P0 | tc3 unseen cache crash | INACTIVE | A1a | S4 |
| P-012 | CODE | P0 | tc4 checkpoint paths insufficient | INACTIVE | A1c | S1,S4 |
| P-013 | CODE | P0 | compute_metrics triplicated (4 implementations) | RESOLVED | A1f (P0.3) | S4,S5 |
| P-014 | CODE | P0 | Probability persistence gap (probs not saved) | COMPLETED | C3 | S4 |
| P-015 | EXPERIMENT | P0 | Pre-G1 decision gate needed | PLANNED | Pre-G1, G0 | S5 |
| P-016 | HOUSEKEEPING | P0 | A1e missing from plan (requirements.txt update) | COMPLETED | A1e | S4,S5 |
| P-017 | DATA | P0 | Pre-requisite: RAID dataset must be accessible | RESOLVED | Pre-G1 (P3.1 verified) | S5 |
| P-018 | INFRASTRUCTURE | P0 | Pre-requisite: CUDA environment working | RESOLVED | Pre-G1 (P3.1 verified) | S5 |
| P-019 | INFRASTRUCTURE | P0 | Pre-requisite: GPU thermal baseline | PLANNED | Pre-G1 | S5 |
| M-001 | CODE, DATA | P0 | GPU phase must be serial (1→ RTX 4050) | PARTIAL | P0.6, §0, §10 | S4,S5 |
| M-002 | FRAMING | P0 | Priority vs success criteria contradiction | RESOLVED | Sec 12 (per plan §10) | S5 |
| M-003 | CODE | P0 | FDG file rename + internal string audit needed | RESOLVED | B1 | S5 |
| M-004 | INFRASTRUCTURE | P0 | GPU budget under-estimated (25-30h not 21h) | PLANNED | M1 | S5 |
| M-005 | CODE | P1 | A3 column consumer audit needed | RESOLVED | A3 | S5 |
| M-006 | EXPERIMENT | P1 | G3 threshold recalibration caveat needed | PLANNED | G3 | S5 |
| M-007 | PAPER, FRAMING | P1 | Abstract missing gpt-j-6B degradation (14.99%) | RESOLVED | B3 (fold-in from B4 per end-to-end plan) | S1,S4 |
| M-008 | PAPER | P1 | Table IV confusion matrix provenance unclear | PLANNED | B5,N5 (merged into A2) | S2,S5 |
| M-009 | PAPER, FRAMING | P1 | RRD naming → 33.40% is not valid RRD per Eq.2 | RESOLVED | B2 | S1,S4 |
| M-010 | PAPER | P1 | Decision threshold wording imprecise ("0.5 argmax") | RESOLVED | B6 | S4 |
| M-011 | FRAMING | P1 | Title softening needed | RESOLVED | B7a (title uses "Adversarially Augmented") | S4 |
| M-012 | FRAMING | P1 | "adversarially diverse" imprecise | RESOLVED | B7b (0 instances remain in .tex) | S4 |
| M-013 | FRAMING | P1 | "demonstrates"/"superior"/"substantially" overclaiming | PLANNED | B7c-e | S4,S8 |
| M-014 | FRAMING | P1 | AMP speedup disentanglement needed | PLANNED | B7d | S4 |
| M-015 | FRAMING | P1 | "Under 16 minutes" clarification | PLANNED | B7e | S4 |
| M-016 | FRAMING | P1 | Conclusion softened | PLANNED | B7f | S4 |
| M-017 | PAPER | P1 | Significance table needs \label + reference | PLANNED | B7g | S4 |
| M-018 | FRAMING | P1 | G2 clean baseline confound (dataset shift) limitation | PLANNED | B7h | S5 |
| M-019 | FRAMING | P1 | Designate primary split (dedup=primary) | PLANNED | B7i | S5 |
| M-020 | PAPER | P1 | Exact-match dedup footnote | PLANNED | A2,N5 | S5 |
| M-021 | PAPER | P1 | Expand Limitations section (energy, RAID noise) | PLANNED | B8a | S4 |
| M-022 | PAPER | P1 | Peak VRAM note in wall-clock table | PLANNED | B8c | S4 |
| M-023 | DATA, PAPER | P1 | 6,029 overlap "modestly inflate" → quantify | PLANNED | B7 | S1,S3 |
| M-024 | PAPER | P1 | Single-seed absence → elevate from Limitations to Results | PLANNED | B7i | S1,S3 |
| M-025 | CODE | P1 | Hardcoded seeds in train_distilbert_detectrl.py | RESOLVED | G1 | S8 |
| M-026 | DATA | P1 | Error messages for missing raw data need improvement | PROPOSED | → | S8 |
| M-027 | PAPER, HOUSEKEEPING | P1 | 4 figure files exist on disk but unreferenced in .tex | RESOLVED | 4 PNG files deleted; HTML orphan resolution completed | S8 |
| M-028 | PAPER, HOUSEKEEPING | P1 | Hyperparameters table has \caption but NO \label | REJECTED | FALSE POSITIVE → \label{tab:hyperparams} exists at L582, referenced at L578 | → | S8 |
| M-029 | FRAMING | P1 | No \usepackage{hyperref} → dead refs, non-clickable URL | REJECTED | FALSE POSITIVE → \usepackage[breaklinks=true, hidelinks]{hyperref} exists at L12 | → | S8 |
| M-030 | FRAMING | P1 | Title mismatch: paper=RAID, README=DetectRL | RESOLVED | P0 | S8 |
| M-031 | FRAMING | P1 | Dataset mismatch: README says DetectRL, paper uses RAID | RESOLVED | P0 | S8 |
| M-032 | PAPER, FRAMING | P1 | Abstract "adversarially exposed" is non-standard | RESOLVED | Abstract now uses "adversarially augmented" | S8 |
| M-033 | FRAMING | P1 | Cross-attack generalisation not directly tested → RQ1 is about generator transfer | PROPOSED | → | S8 |
| M-034 | EXPERIMENT | P1 | G1 per-config timing correction (plan Appendix B) | PLANNED | G1 budget | S5 |
| M-035 | CODE | P1 | Dependency matrix correction: decouple G3 from B1 | RESOLVED | §0, §12 | S5 |
| M-036 | FRAMING | P1 | Critical path: include A1b?A2 upstream | PLANNED | Preamble | S5 |
| M-037 | FRAMING | P1 | Priority/execution diagram: sequence A+B | PLANNED | Preamble | S5 |
| M-038 | FRAMING | P1 | L60 merge conflict: W1-T2/T3/T4 all modify abstract | PLANNED | W1-T2/T3/T4 merge | S4,S5 |
| M-039 | FRAMING | P1 | Timeline underestimation: project is 40-80% longer | PLANNED | Preamble | S4,S5 |
| M-040 | FRAMING | P1 | requirements.txt missing runtime deps (W2-T5) | RESOLVED | P0.9 | S4 |
| M-041 | INFRASTRUCTURE | P1 | Docker/reproducibility infra | PLANNED | TBD | S4,S5 |
| N-001 | EXPERIMENT | P1 | Clean-only training baseline absent | PLANNED | G2 | S1,S4 |
| N-002 | EXPERIMENT | P1 | Calibration (ECE) not reported | PLANNED | G5 | S3,S4,S5 |
| N-003 | EXPERIMENT | P2 | Low-FPR metrics (1%/5%/10%) not reported | PLANNED | G4 | S3 |
| N-004 | EXPERIMENT | P2 | Bootstrap significance testing missing | PLANNED | G7 | S4 |
| N-005 | EXPERIMENT | P2 | Per-attack DistilBERT evaluation | PLANNED | G6 | S3,S4 |
| N-006 | DATA, EXPERIMENT | P2 | Count tables (per-domain/attack/generator) | PLANNED | G8 | S4 |
| N-007 | EXPERIMENT | P2 | TF-IDF + Logistic Regression baseline | PLANNED | A4 | S4 |
| N-008 | FRAMING | P2 | Systems benchmarks compression (reduce ~7 figs+tables → 1 para) | PLANNED | B9 | S1,S4 |
| N-009 | HOUSEKEEPING | P2 | Filter script docstrings outdated (wrong run commands) | RESOLVED | B8b | S4 |
| N-010 | HOUSEKEEPING | P2 | AUROC delta rounding fix (0.2637?0.2636) | PLANNED | B8d | S4 |
| N-011 | PAPER, HOUSEKEEPING | P2 | 10 orphaned \label definitions (includes ssec:future; excludes ssec:limits which IS referenced) | PROPOSED | → | S8 |
| N-012 | PAPER, HOUSEKEEPING | P2 | No \bibliographystyle/.bib file (thebibliography inline) | PROPOSED | → | S8 |
| N-013 | CODE | P2 | Bare `except:` in tc3_traindistilbert.py L158 | RESOLVED | P0.3 (metrics wave) | S8 |
| N-014 | CODE | P2 | `src/training/trainer.py` completely unused (217 lines) | RESOLVED | P0.8 (trainer.py rewritten) | S8 |
| N-015 | CODE | P2 | Duplicate filter.py modules diverged (data/ vs src/data/) | RESOLVED | P1.4 (duplicate deleted) | S8 |
| N-016 | CODE | P2 | `torch.load(weights_only=False)` in tc4_.py L74 | RESOLVED | P0.7 (weights_only fixed) | S8 |
| N-017 | FRAMING | P2 | Overclaiming: "trivially" → "straightforwardly" L72 | RESOLVED | B7c (fix confirmed at .tex L73: "straightforwardly producible") | S8 |
| N-018 | FRAMING | P2 | Overclaiming: "first adversarial training approach" L286 | RESOLVED | B7c (fix confirmed at .tex L290: "a pioneering ...") | S8 |
| N-019 | HOUSEKEEPING | P2 | No `\hyphenation` rules for IEEEtran column constraints | PROPOSED | → | S8 |
| N-020 | CODE, BASELINE | P1 | Binoculars baseline: missing from peer-baseline comparison | PLANNED | B1, G3 (end-to-end plan) | S11 |
| N-021 | EXPERIMENT | P2 | DeBERTa-v3-LoRA cross-architecture validation DEFERRED to future work → DistilBERT primary study prioritised. DeBERTa would add cross-architecture generalisation evidence but is not required for core claims. FlashDeBERTa (Triton) enables ~3-5× speedup on T4 if revisited. | PROPOSED | → (future work) | S12 (distilled from 2026-05-26 analysis) |
| F-001 | CODE | P0 | Weight-decay exclusion for bias/LayerNorm | PLANNED | P0.5 (end-to-end plan) | S11 |
| F-002 | CODE | P1 | DataLoader worker seeding missing | PLANNED | P0.5 (end-to-end plan) | S11 |
| F-003 | CODE | P1 | No LR scheduler (constant LR) | PLANNED | P0.5 (end-to-end plan) | S11 |
| F-004 | INFRASTRUCTURE | P0 | DistilBERT not pre-cached to Kaggle Dataset | PLANNED | P0.11 (end-to-end plan) | S11 |
| F-005 | INFRASTRUCTURE | P1 | Per-ablation upload pattern risks 429 rate limits | PLANNED | P0.6 (identified, code not yet batched to once-per-session) | S11 |
| F-006 | INFRASTRUCTURE | P1 | Three-level defensive timer missing | RESOLVED | P0.6 (end-to-end plan → already addressed) | S11 |
| F-007 | CODE | P2 | evaluate.py uses no_grad not inference_mode | RESOLVED | P0.12 (end-to-end plan) | S11 |
| F-008 | PAPER | P1 | RAID AUROC not explicitly reported | PLANNED | B7 (end-to-end plan) | S11 |
| H-001 | HOUSEKEEPING | P3 | Unused `import os` in filter_raid_parallel.py L23 | RESOLVED | → | S8 |
| H-002 | HOUSEKEEPING | P3 | Unused `import random` in filter_raid_sequential.py L22 | RESOLVED | → | S8 |
| H-003 | HOUSEKEEPING | P3 | Unused `import os` in tc3_traindistilbert.py L11 | SUPERSEDED | P0.3 (file deleted in Phase 0b) | S8 |
| H-004 | HOUSEKEEPING | P3 | Unused `import os` in download_detectrl_HC3.py L4 | SUPERSEDED | P0.3 (file deleted in Phase 0b) | S8 |
| H-005 | HOUSEKEEPING | P3 | `import argparse` mid-file in 3 training scripts | SUPERSEDED | P0.3 (files deleted in Phase 0b) | S8 |
| H-006 | HOUSEKEEPING | P3 | Hardcoded user path in generate_figures.py docstring L6 | RESOLVED | → | S8 |
| H-007 | HOUSEKEEPING | P3 | 3 docstring run commands reference wrong filenames | RESOLVED | → | S8 |
| H-008 | HOUSEKEEPING | P3 | Missing `__init__.py` in figures/ | SUPERSEDED | Not needed (figures/ has only output files) | | S8 |
| H-009 | HOUSEKEEPING | P3 | Hardcoded batch_size 50_000 (not a named constant) in filter_raid_sequential.py | PARTIAL | → | S8 |
| H-010 | HOUSEKEEPING | P3 | `\begin{thebibliography}{00}` → `{99}` for 18 entries | RESOLVED | Fix 2 (2026-05-26) | S8 |
| H-011 | HOUSEKEEPING | P3 | Commented-out `\IEEEoverridecommandlockouts` L2 | PROPOSED | → | S8 |
| H-012 | HOUSEKEEPING | P3 | `\centerline` instead of `\centering` in figures (5 instances) | RESOLVED | Fix 2 (2026-05-26) | S8 |
| H-013 | HOUSEKEEPING | P3 | `\smallskip`/`\noindent` for RQ formatting (fragile) | PROPOSED | → | S8 |
| H-014 | HOUSEKEEPING | P3 | No `\usepackage{subcaption}` for future multi-panel figures | PROPOSED | → | S8 |
| H-015 | HOUSEKEEPING | P3 | No appendix section | PROPOSED | → | S8 |
| RJ-001 | → | REJECTED | LSP diagnostics (texlab) on .tex | REJECTED | → | S4 |
| RJ-002 | → | REJECTED | Git branch strategy over-engineering | REJECTED | → | S4 |
| RJ-003 | → | REJECTED | AI prose check | REJECTED | → | S4 |
| RJ-004 | → | REJECTED | Near-duplicate dedup implementation (footnote only) | REJECTED | → | S5 |
| RJ-005 | → | REJECTED | Multiple comparison correction | REJECTED | → | S5 |
| RJ-006 | → | REJECTED | .gitignore fresh clone fix | REJECTED | → | S5 |
| RJ-007 | → | REJECTED | Orphaned LSP labels audit | REJECTED | → | S5 |
| RJ-008 | → | REJECTED | Python/CUDA version pinning (Docker subsumes) | REJECTED | → | S4 |
| RJ-009 | → | REJECTED | Near-dedup (duplicate of RJ-004) | REJECTED | → | S5 |
| RJ-010 | → | REJECTED | Add more process steps (LaTeX, git, etc.) | REJECTED | → | S4 |

---

## 3. Detailed Issue Entries

### Section A: Active Entries (Planned, In Progress, and Resolved)

> **Convention note**: Combined entries (e.g., P-011→P-015, H-001→H-004) group related low-complexity issues that share a common resolution context. Each sub-ID has a full row in the Index table (Section 2) with complete category, priority, status, and plan-ref metadata. The detailed entries below provide consolidated resolution text where individual entries would be redundant.

---

### P-001 [PAPER][CODE][P0] Fast-DetectGPT → GPT-2 XL Perplexity Baseline rename

**Priority rationale**: Reviewer would immediately flag "Fast-DetectGPT" → what code implements. Integrity issue.

**Status**: RESOLVED -- file renamed to src/baselines/perplexity_baseline.py; 3 docstring references remain in perplexity_baseline.py (explanatory comparison context).

**Validation**: CONFIRMED -- file renamed, imports updated, class/function names cleaned. 3 docstring references remain in perplexity_baseline.py (lines 7,10,15) -- these explain the distinction from true Fast-DetectGPT and are retained as comparison context.
**Description**: Codebase computes standard perplexity (`log_softmax` → `sum` → `exp(mean)`), NOT Fast-DetectGPT curvature via token perturbation. Paper claims curvature method. Footnote partially addresses.
**Resolution**: Rename perplexity_baseline.py to src/baselines/perplexity_baseline.py; update all imports; remove internal FDG string refs. Bibliography citation preserved.
**Validation**: → CONFIRMED → Docstring (L1-5): `"score each text with GPT-2 XL log p(x)"` and `_score_text()` (L163-197): computes perplexity via cross-entropy loss. No `perturb`/`mask`/`replace` logic anywhere in 454-line file.

**Root cause**: Original planned curvature; time constraints → perplexity-only; docs never updated.

**Resolution** (Option C from hyperplan, Momus-approved):
1. Rename file: `fast_detectgpt.py` → `perplexity_baseline.py`
2. Replace ALL occurrences in .tex: "Fast-DetectGPT" → "GPT-2 XL Perplexity Baseline"
3. Change citation: `\cite{mitchell2023}` → `\cite[e.g.,][]{mitchell2023}`
4. Add footnot explaining implementation difference
5. Update internal code comments, docstrings, argparse help, log messages

**Dependencies**: Independent. Does NOT block G3 (dependency matrix corrected per P1).

**Related**: M-006 (G3 threshold caveat), M-009 (RRD naming inequity)


**Verification**: `grep -c "Fast-DetectGPT" src/baselines/perplexity_baseline.py` → 3 (explanatory docstring references retained for comparison context per resolution above); `grep -c "Fast-DetectGPT" scripts/` → 0; `grep -c "fast_detectgpt" src/` → 0

---

### P-002 [EXPERIMENT][FRAMING][P0] Multi-seed evaluation could falsify central claim

**Priority rationale**: Current RRD values (2.05-5.38%, preliminary single-seed on dedup'd-but-pre-set-exclusion data) need multi-seed verification. Original IBCAST values (1.1-3.1% on contaminated data) were inflated by 6,029 overlapping human texts; variance could collapse the "within 5%" central claim.

**Source(s)**: S4, S5  
**Plan ref**: Pre-G1 Decision Gate, G1

**Validation**: → CONFIRMED → `train_distilbert_detectrl.py`: no seed randomization; single seed; all metrics are point estimates.

**Root cause**: IBCAST submission accepted single-seed. Top-tier venues require multi-seed + variance.

**Resolution**: 
1. Pre-G1 gate: Single config on dedup split (~4h) to gauge shift
2. G1: 5 seeds → 4 configs on dedup split (~14.1h)
3. CIs separate → proceed + error bars; CIs overlap → pivot narrative (fallback pre-written)

**Dependencies**: A2 (dedup), A1d (CPU guards). Blocks G4/G5/G6/G7.

**Status**: PLANNED  
**Contingency**: Fallback narrative pre-written in plan v2.

---

### P-003 [CODE][DATA][P0] Human-text contamination: 6,029 overlapping texts

**Priority rationale**: 6,029/5,000 held-out human texts (~120% !) seen in training. ALL metrics inflated.

**Source(s)**: S1, S2, S3, S4  
**Plan ref**: A0, A2

**Validation**: → CONFIRMED → Read both filter scripts: domain concatenation without dedup creates cross-split duplicates. Multiplicity >1.0 confirms texts appear 2+ times.

**Root cause**: RAID dataset has cross-domain human texts. Pipeline concatenates all domains → same text included from multiple domains → duplicate in both train and held-out.

**Resolution**: A0 (contamination source audit) → A2 (`df.drop_duplicates(subset=["text"])` applied to dataloader.py L257). Set-exclusion guard added to both filter scripts (parallel L258, sequential L220). See A2 in end-to-end plan §Section 9.

**Dependencies**: A0 → A2 → G1.

**Status**: RESOLVED (2026-05-22, A2 COMPLETED)

---

### P-004 [PAPER][FRAMING][P0] Abstract AUROC range wrong

**Priority rationale**: The abstract reports an AUROC range that does not match the paper's own results → this is a data integrity issue that undermines credibility.

**Source(s)**: S2, S6  
**Plan ref**: B3

**Description**: Abstract (L59) reports AUROC range `0.669--0.789` but the FDG homoglyph result in Section V-D (L1094) shows `AUROC 0.526--0.669`. The abstract range excludes the lowest observed value.

**Validation**: L58-61 now reports `AUROC 0.526--0.789` ✓ matches Section V-D homoglyph results.

**Resolution**: Already applied — abstract (L58-61) correctly reports AUROC 0.526--0.789.

**Status**: RESOLVED — fix verified at L58-61

---

### P-005 [PAPER][FRAMING][P0] 5% RRD threshold unjustified

**Priority rationale**: The 5% RRD threshold is used as a significance criterion throughout the paper without any supporting citation or statistical justification. Readers and reviewers will question its basis.

**Source(s)**: S4, S6  
**Plan ref**: B8 (A5 superseded by B8)

**Description**: The 5% RRD threshold used to claim "significant degradation" (L59, L205, L514, L1483) has no citation, no theoretical justification, and no bootstrap-derived confidence interval. It is presented as an implicit significance bar without supporting evidence.

**Validation**: → Grep "5%" in .tex → no citation, no theoretical justification.

**Resolution**: B8 (COMPLETED 2026-05-22) → text caveat added linking to Limitations section. A5 citation approach superseded by B8 qualification.

**Status**: RESOLVED

---

### P-006 [CODE][P0] CPU autocast crash on non-CUDA devices

**Source(s)**: S1, S3, S4  
**Plan ref**: P0.5 (was A1d)

**Validation**: → CONFIRMED → 2 unguarded autocast sites found:
- `tc3_traindistilbert.py` L188: `torch.cuda.amp.autocast(enabled=True)` → guard variable `use_amp` defined at L180 but **deliberately commented out** at L187, replaced with hardcoded `True`
- `tc4_.py` L55: `torch.amp.autocast("cuda")` → no guard

**Note**: Register previously claimed 7 sites across 4 files. Audit corrected this: `train_distilbert_detectrl.py` and `train_distilbert_parallel.py` have **zero** autocast calls. Guarded sites now in current codebase:
- `src/training/trainer.py` L141, L152 — guarded by `use_amp` config flag and `device.type`
- `src/baselines/perplexity_baseline.py` L212-213 — guarded by earlier CUDA check in calling function

**Resolution**: Wrap both sites with CUDA guard. ~15min.

**Dependencies**: Blocks ALL GPU reruns (G1, G2, G3).

**Status**: COMPLETED  
**Resolution notes**: CPU autocast guards fixed in `src/training/trainer.py` (called by scripts/train.py; P0.5).

---

### P-007 [CODE][DATA][P0] Attack column hijack

**Source(s)**: S3, S4  
**Plan ref**: A3

**Validation**: → `filter_raid_parallel.py` L217, `filter_raid_sequential.py` L185: `pool["attack_type"] = pool["generator"].where(...)` → overwrites attack_type with generator name.

**Resolution**: Preserve both `attack_type` and `generator` in separate columns.

**Dependencies**: M-005 (column consumer audit). Note: the critical fix was applied without the full audit (M-005 is now RESOLVED). Any downstream consumer that depended on the overwritten `attack_type` column schema could produce incorrect results — verified and fixed per M-005 scope (zero mismatches found).

**Status**: COMPLETED  
**Resolution notes**: Column hijack fixed in both filter scripts (P1.3). Full consumer audit completed via M-005 (RESOLVED).

---

### P-008 [DATA][P0] Sequential parquet naming mismatch

**Source(s)**: S3, S4  
**Plan ref**: A1b

**Validation**: → `filter_raid_sequential.py` outputs `train_pool.parquet` instead of `raid_train_pool.parquet`. Training scripts expect `raid_` prefix.

**Resolution**: Prefix outputs with `raid_`.

**Status**: COMPLETED  
**Resolution notes**: Parquet naming fixed in sequential filter (P1.2).

---

### P-009 [CODE][FRAMING][P0] FDG comparison directional, not matched

**Source(s)**: S1, S2, S4  
**Plan ref**: B1, G3

**Validation**: → DistilBERT reports held-out F1 (0.913-0.927). FDG reports adversarial AUROC (0.526-0.789). Different metrics, different splits, different conditions.

**Resolution**: B1 (rename) + G3 (perplexity baseline on same dedup split) + caveat (M-006 — G3 threshold recalibration caveat).

**Status**: PLANNED

---

### P-010 [CODE][P0] Contamination source unverified before dedup

**Source(s)**: S5  
**Plan ref**: A0

**Validation**: ✅ Done - data/contamination_audit.md documents full data flow topology and contamination source. See AUDIT_TRAIL Phase 1.

**Resolution**: Grep both pipelines; report per-text multiplicity; document in data/contamination_audit.md (done - see P1.1 in end-to-end plan).

**Status**: RESOLVED

---

### P-011 through P-015 [CODE, EXPERIMENT][P0]: Code crash + prerequisite fixes

**Note**: P-011 through P-015 share plan refs A1a, A1c, A1f, C3, and Pre-G1. See plan v2 for full details. P-011/P-012 are INACTIVE (code eliminated); P-013 RESOLVED; P-014 COMPLETED; P-015 PLANNED with expanded gate.

| ID | Issue | Plan Ref | 
|----|-------|----------|
| P-011 | tc3 unseen cache crash | A1a → INACTIVE (tc3 replaced by scripts/train.py P0.5 → code eliminated) |
| P-012 | tc4 checkpoint paths insufficient | A1c → INACTIVE (tc4 replaced by scripts/benchmark.py P0.7 → code eliminated) |
| P-013 | compute_metrics triplicated | A1f → RESOLVED (src/evaluation/metrics.py is canonical)
| P-014 | Probability persistence gap | C3 → COMPLETED (addressed by unified checkpoint in P0.5) |
| P-015 | Pre-G1 decision gate | Pre-G1 → G0 (Phase 4 → expanded from binary 2% threshold) |

**P-015 Validation**: Multi-criteria gate: (1) RRD spread 6-8% single-seed threshold, (2) unseen_F1 = 0.85, (3) unseen AUROC = 0.90, (4) training convergence, (5) contamination check. All must pass.

---

### P-016 [HOUSEKEEPING][P0] A1e missing from plan (requirements.txt update)

**Priority rationale**: Missing plan entry for critical dependency fix → stale requirements.txt blocks reproducibility for all GPU work.

**Source(s)**: S4, S5  
**Plan ref**: A1e

**Description**: The v2 plan defines A1a-A1f but A1e is missing. This was the requirements.txt dependency fix (W2-T5 from hyperplan-synthesis.md, silent unanimity). Requirements.txt is stale and missing several runtime dependencies.

**Root cause**: Plan wave mapping missed A1e during restructuring.

**Resolution**: 1. Verify current requirements.txt against all imports. 2. Add missing packages. 3. Sort and deduplicate. 4. Pin compatible versions.

**Dependencies**: None.


**Validation**: → DONE
**Status**: COMPLETED  
**Resolution notes**: requirements.txt update done in multiple P0 commits.

---

### P-017 [DATA][P0] Pre-requisite: RAID dataset must be accessible

**Priority rationale**: All GPU reruns are blocked without dataset access → cannot start G1-G8 without verifying data integrity.

**Source(s)**: S5  
**Plan ref**: Pre-G1

**Description**: All GPU reruns depend on the RAID dataset being available from disk. The current state of raw data files is unknown. Cannot start G1-G8 without verifying dataset integrity.

**Root cause**: Dataset may have been moved, deleted, or corrupted since IBCAST submission.

**Resolution**: 1. Check data/ directory for raw parquet files. 2. Verify SHA256 or row counts. 3. Document download/recovery steps if missing.

**Dependencies**: None (pre-requisite for all GPU work).


**Validation**: P3.1 smoke test (2026-05-24) confirmed RAID dataset accessible on both local RTX 4050 and Kaggle T4. Dataset integrity verified via successful training run.
**Status**: RESOLVED

---

### P-018 [INFRASTRUCTURE][P0] Pre-requisite: CUDA environment working

**Priority rationale**: All GPU work requires working CUDA → a broken environment blocks every experiment (G1-G8).

**Source(s)**: S5  
**Plan ref**: Pre-G1

**Description**: All GPU work requires a working CUDA environment. Must verify nvidia-smi, torch.cuda.is_available(), and driver compatibility before starting G1.

**Root cause**: Environment may have changed since last run (driver updates, torch reinstalls, etc.).

**Resolution**: 1. Run `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`. 2. Document environment in `.omo/env/`.

**Dependencies**: None (pre-requisite for all GPU work).


**Validation**: P3.1 smoke test (2026-05-24) ran on RTX 4050 (CUDA 12.4, torch 2.6.0) and Kaggle T4 (CUDA 12.4) — both passed nvidia-smi and torch.cuda.is_available(). Dual T4 run completed all 4 ablations.
**Status**: RESOLVED

---

### P-019 [INFRASTRUCTURE][P0] Pre-requisite: GPU thermal baseline

**Priority rationale**: RTX 4050 6GB Laptop throttling at ~75°C would corrupt ~14h G1 runs with non-deterministic timing → must be managed upfront.

**Source(s)**: S5  
**Plan ref**: Pre-G1

**Description**: The RTX 4050 6GB Laptop GPU throttles at ~75°C. A single G1 run (~14h) without thermal management may throttle and produce non-deterministic timing results.

**Root cause**: Consumer GPU is not designed for sustained compute loads. No active cooling plan.

**Resolution**: 1. Run short benchmark to measure steady-state temperature. 2. Set power limit (e.g., `nvidia-smi -pl 60`). 3. Document thermal status in run log.

**Dependencies**: P-018 (CUDA environment must work first).

---

### M-002 [FRAMING][P0] Priority vs success criteria contradiction

**Priority rationale**: P2 tasks gate top-tier success criteria → if they're deferrable, they cannot simultaneously be submission requirements.

**Source(s)**: S5  
**Plan ref**: Sec 12

**Description**: Priority tiers label G4 (low-FPR), G7 (bootstrap), and B9 (systems compress) as P2 "minor, deferrable" items. Yet the success criteria for "Top-tier (ACL/EMNLP-ready)" requires ALL tasks complete. If they are P2 nice-to-have tasks, they should not gate top-tier readiness → a logical contradiction.

**Root cause**: Success criteria were drafted before priority assignment; no cross-validation step reconciled the two.

**Resolution**:
1. Restructure success criteria into Core (ACL-ready) vs Extended (journal-ready)
2. Move G4/G7/B9 to Extended tier
3. Update priority table to match

**Dependencies**: None.


**Validation**: Section 12 success criteria restructured per plan §10 (priority tiers reconciled with success criteria framework).
**Status**: RESOLVED

---

### M-003 [CODE][P0] FDG file rename + internal string audit needed

**Priority rationale**: Without the internal audit, the .tex rename (P-001) fixes surface references but code and scripts still say "Fast-DetectGPT."

**Source(s)**: S5  
**Plan ref**: B1

**Description**: Companion to P-001. Beyond the .tex rename, the actual file `perplexity_baseline.py` must be renamed, and ALL internal strings → docstring, argparse help, log messages, code comments → must be audited for "Fast-DetectGPT" references. Leftover references will confuse any reviewer who inspects the code.

**Root cause**: P-001 scope originally focused on .tex only; the code-level rename was always implicit but never task-captured.

**Resolution**:
1. Rename file to `perplexity_baseline.py`
2. Grep all internal strings for `fast_detectgpt` / `Fast-DetectGPT` / `FDG`
3. Update imports in any consuming scripts
4. Update README references

**Dependencies**: None (companion to P-001, can run in parallel).


**Validation**: ✅ CONFIRMED — `perplexity_baseline.py` replaces `fast_detectgpt.py`; zero stale `.py` references to "fast_detectgpt" or "Fast-DetectGPT". Code-side audit complete. Paper-side 3 Fast-DetectGPT references remain (P-001, P-009); retained as literature-review citations of the original Bao et al. paper.
**Status**: RESOLVED

---

### M-004 [INFRASTRUCTURE][P0] GPU budget under-estimated (25-30h not 21h)

**Priority rationale**: A 43% under-estimate threatens the entire delivery schedule. Work cannot be scoped without accurate budget.

**Source(s)**: S5  
**Plan ref**: M1

**Description**: The plan claimed 21h total GPU time (Track B: 11h non-GPU + Track A+G: 10h GPU). Actual minimum: Track A (~5h) + Pre-G1 (~4h) + G1 (~14.1h) + G3 (~4h) + G5 (~5h conditional) = 25→30h GPU alone. The estimate omitted: probability persistence (0→5h conditional) and G5 calibration requiring a separate forward pass.

**Root cause**: Initial estimates optimistically assumed GPU parallelism and omitted conditional work packages. The single-GPU wall-clock constraint compounded the error.

**Resolution**:
1. Update budget to 25→30h
2. Note conditional +5h for G5
3. Explicitly state single-GPU wall-clock (no parallelism possible)

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-005 [CODE][P1] A3 column consumer audit needed

**Priority rationale**: Fixing the column hijack (P-007) could silently break every downstream script if they depend on the old `attack_type` column name or ordering. An audit of 6 reading scripts saves debugging time later.

**Source(s)**: S5  
**Plan ref**: A3

**Description**: After P-007 (attack column hijack fix), audit all scripts that consume filtered parquet files and verify they use canonical column names (`text`, `label`, `generator`, `attack_type`).
- `scripts/train.py` — reads from dataloader; columns abstracted → low risk
- `scripts/evaluate.py` — reads from dataloader; columns abstracted → low risk
- `src/data/dataloader.py` — must read `attack_type` for upstream scripts → high risk
- `scripts/benchmark.py` — reads via checkpoint; columns not directly accessed → low risk
- `scripts/kaggle_run.py` — orchestrator; doesn't read parquet → no risk

**Validation**: ✅ done — zero mismatches found (2026-05-22 P1.3b)
**Status**: RESOLVED

---

### M-006 [EXPERIMENT][P1] G3 threshold recalibration caveat needed

**Priority rationale**: Reporting AUROC on dedup data using a threshold calibrated on contaminated data may misrepresent perplexity baseline performance.

**Source(s)**: S5  
**Plan ref**: G3

**Description**: The FDG/perplexity baseline on deduplicated data (G3) will shift AUROC values. The 0.5 decision threshold was calibrated on contaminated data. Deduplication changes the data distribution and may change the optimal threshold.

**Root cause**: FDG/perplexity threshold was established during the original (contaminated) pipeline and never re-evaluated for the dedup split.

**Resolution**:
1. Recompute AUROC on dedup split
2. Report threshold sensitivity
3. Note in paper if threshold changes

**Dependencies**: G3 (FDG on dedup) must run first.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-007 [PAPER][FRAMING][P1] Abstract missing gpt-j-6B degradation (14.99%)

**Priority rationale**: Omitting the 14.99% outlier from the abstract creates a misleading impression that all results are "well within 5%."

**Source(s)**: S1, S4  
**Plan ref**: B3 (fold-in from B4 per end-to-end plan; B4 is superseded)

**Description**: The abstract reports RRD range 1.1→3.1% but omits the 14.99% RRD for gpt-j-6B (ablation_c DistilBERT). This outlier contradicts the "well within 5%" narrative and a reviewer cross-checking against Table III will flag the discrepancy.

**Root cause**: The gpt-j-6B outlier weakens the central narrative and was selectively omitted during abstract drafting.

**Resolution**: Report full RRD range (1.1-14.99%) in the abstract, or qualify to explicitly exclude the outlier with justification.

**Status**: RESOLVED — abstract L60-62 now includes both 33.40% (gpt2-xl) and 14.99% (gpt-j-6B) adversarial AUROC degradation figures.

**Dependencies**: None.


**Validation**: ✅ Complete — abstract includes 33.40% and 14.99% figures

---

### M-008 [PAPER][P1] Table IV confusion matrix provenance unclear

**Priority rationale**: A reviewer cannot determine whether Table IV values are reproducible or an artifact of a specific run.

**Source(s)**: S2, S5  
**Plan ref**: B5, N5 (merged into A2)

**Description**: Table IV confusion matrix values in the `.tex` differ from `summary.csv` by <0.0004. There is no source trace for which run produced these values → unclear whether from a single seed, an average, or a cherry-picked best run.

**Root cause**: Confusion matrix was extracted from a specific training run at the time of figure generation, but that provenance was not documented.

**Resolution**: Either 1) recompute from dedup multi-seed runs (G1), or 2) add a footnote noting the minimal rounding delta (diff <0.0004).

**Dependencies**: G1 if Option 1 is chosen.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-009 [PAPER][FRAMING][P1] RRD naming → 33.40% is not valid RRD per Eq.2

**Priority rationale**: Labeling a cross-metric comparison as "RRD" violates the paper's own definition. A reviewer will spot this immediately.

**Source(s)**: S1, S4  
**Plan ref**: B2

**Description**: Equation 2 defines RRD = (F1_seen - F1_unseen) / F1_seen. The paper reports 33.40% "RRD" for the perplexity baseline, but this uses FDG F1 (0.913) as "seen" and FDG AUROC (0.526) as "unseen" → mixing metrics violates the Eq.2 definition.

**Root cause**: Cross-metric comparison was used to produce a dramatic RRD number; the definitional conflict was overlooked.

**Resolution**:
1. Recompute RRD using matched metrics (F1 vs F1, or AUROC vs AUROC)
2. Alternatively, explicitly state the metric mismatch and rename the quantity (e.g., "cross-metric gap")

**Dependencies**: None.


**Validation**: ✅ Paper now uses "adversarial AUROC degradation" at L1064 (no longer labels cross-metric comparison as RRD)
**Status**: RESOLVED — cross-metric comparison no longer labeled as RRD

---

### M-010 [PAPER][P1] Decision threshold wording imprecise ("0.5 argmax")

**Priority rationale**: Technical imprecision undermines reviewer confidence in the authors' understanding of their own methodology.

**Source(s)**: S4  
**Plan ref**: B6

**Description**: The paper describes the decision threshold as "0.5 argmax" but DistilBERT uses a binary sigmoid (threshold = 0.5), not a dual-softmax argmax. This is a technical imprecision that a reviewer familiar with binary classification will flag.

**Root cause**: "0.5 argmax" is a colloquialism carried over from multi-class language; not corrected during drafting.

**Resolution**: Replace "0.5 argmax" with "binary sigmoid threshold at 0.5" throughout the `.tex`.

**Dependencies**: None.


**Validation**: ✅ L492-493 now uses "thresholded at 0.5 (softmax decision threshold)"
**Status**: RESOLVED — L492-493 describes the softmax posterior probability threshold at 0.5

---

### M-011 [FRAMING][P1] Title softening needed

**Priority rationale**: "Adversarially Robust" in the title overclaims the study's scope, risking desk-reject or harsh review.

**Source(s)**: S4  
**Plan ref**: B7a

**Description**: The current title "Towards Adversarially Robust AI Text Detection via Supervised DistilBERT Fine-Tuning" → the word "Towards" already softens, but "Adversarially Robust" implies multi-attack-family validation. The study only tests generator-family transfer within a single attack category (character-level substitution).

**Root cause**: Ambitious framing from first draft; scope was narrowed during execution but title was not updated.

**Resolution**: Consider "Towards Adversarially-Trained→" or "Generator-Family Robustness→" or keep as-is with explicit scope boundary added to the abstract.

**Dependencies**: None.


**Validation**: ✅ Title L19-20 now uses "Adversarially Augmented" instead of "Adversarially Robust"
**Status**: RESOLVED — title softened to "Adversarially Augmented"

---

### M-012 [FRAMING][P1] "adversarially diverse" imprecise

**Priority rationale**: "Adversarially diverse" implies attack-strategy diversity that the dataset does not provide. Precision matters for reviewer trust.

**Source(s)**: S4  
**Plan ref**: B7b

**Description**: The paper describes RAID as "adversarially diverse" but the diversity is across generators (six LLMs, same attack types: homoglyph + substitution), not across attack strategies. This conflates generator diversity with attack diversity.

**Root cause**: Loose terminology in the RAID description; adopted without scrutiny.

**Resolution**: Replace "adversarially diverse" with "generator-diverse" or "multi-generator" throughout the `.tex`.

**Dependencies**: None.


**Validation**: ✅ "adversarially diverse" removed from .tex (0 matches remaining); replaced with "adversarially augmented, multi-generator" variants
**Status**: RESOLVED — all instances replaced with precise phrasing

---

### M-013 [FRAMING][P1] "demonstrates"/"superior"/"substantially" overclaiming

**Priority rationale**: Overclaiming language ("proves", "superior", "first") is the most common reviewer complaint in ML papers.

**Source(s)**: S4, S8  
**Plan ref**: B7c→e

**Description**: Multiple instances of overclaiming language throughout the `.tex`. "Proves" should be "demonstrates", "substantially outperforms" should be "shows competitive performance", "superior" should be directional language. These trigger reviewer scepticism.

**Root cause**: First-draft language adopted enthusiastic framing; no systematic overclaiming audit was conducted before submission.

**Resolution**: Audit `.tex` for: proves, confirms, demonstrates (in overclaiming context), substantially, superior, unprecedented, first. Replace all with measured, evidence-calibrated language.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-014 [FRAMING][P1] AMP speedup disentanglement needed

**Priority rationale**: Failing to disentangle batch-size contribution from precision-reduction contribution misrepresents AMP's true effect.

**Source(s)**: S4  
**Plan ref**: B7d

**Description**: The paper attributes AMP's speedup entirely to precision reduction (FP16), but AMP's primary mechanism is enabling larger batch sizes within the 8GB VRAM budget. The two effects are not disentangled in the reported numbers.

**Root cause**: AMP's speedup was reported as a single cumulative number; the batch-size contribution was never separately measured.

**Resolution**:
1. Add a note that AMP enables batch-size increase as a primary speedup mechanism
2. Report AMP + fixed batch vs AMP + increased batch separately

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-015 [FRAMING][P1] "Under 16 minutes" clarification

**Priority rationale**: Claiming "under 16 minutes" without isolating optimization contributions is misleading. Reviewer may ask which optimization produced the gain.

**Source(s)**: S4  
**Plan ref**: B7e

**Description**: The paper claims inference takes "under 16 minutes" per configuration. This conflates multiple optimizations (PP + AMP + DLO + NF4). Each optimization's individual contribution is not isolated, making it impossible to attribute the speedup.

**Root cause**: Cumulative optimization times were reported as a single number; per-optimization breakdown was never tabulated.

**Resolution**:
1. Report single-optimization and cumulative times separately
2. Add footnote that "under 16 minutes" reflects the full optimization stack

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-016 [FRAMING][P1] Conclusion softened

**Priority rationale**: Stronger-than-supported claims in the conclusion are the last thing a reviewer reads → and remembers.

**Source(s)**: S4  
**Plan ref**: B7f

**Description**: The conclusion makes stronger claims than the results support, e.g., "establishes DistilBERT as a viable approach for real-world deployment" → without multi-domain validation, multi-seed evaluation, or deployment-scale testing.

**Root cause**: Conclusion was drafted for impact; caveats present in earlier sections were not restated.

**Resolution**: Add caveats → single-model, single-dataset, single-seed limitations → before any forward-looking statements. Restrict deployment claims to "warrants further investigation."

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-017 [PAPER][P1] Significance table needs `\label` + reference

**Priority rationale**: An unreferenced table is dead weight. A reviewer will notice the orphan.

**Source(s)**: S4  
**Plan ref**: B7g

**Description**: The significance / impact summary table in the `.tex` has no `\label` and is never referenced by `\ref{}`. It exists in isolation without any in-text discussion anchoring it to the narrative.

**Root cause**: The table was added late in drafting; the `\label` and text reference were omitted.

**Resolution**: Add `\label{tab:significance}` after the caption and an in-text reference (e.g., "Table~\ref{tab:significance} summarises...").

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-018 [FRAMING][P1] G2 clean baseline confound (dataset shift) limitation

**Priority rationale**: Without this caveat, the clean baseline comparison is scientifically invalid → comparing in-distribution train vs out-of-distribution test.

**Source(s)**: S5  
**Plan ref**: B7h

**Description**: The clean-only DistilBERT baseline (G2) trains on non-adversarial RAID texts but evaluates on adversarial held-out generators. This is a dataset distribution shift, not just "clean vs adversarial." The clean baseline's poor performance partly reflects OOD evaluation, not necessarily the value of adversarial training.

**Root cause**: The limitation was recognised during planning but never documented in the paper.

**Resolution**: Add limitation sentence: "Clean-only baseline evaluated on out-of-distribution adversarial texts, which may underestimate realistic clean performance on in-distribution data."

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-019 [FRAMING][P1] Designate primary split (dedup = primary)

**Priority rationale**: Reporting both splits without pre-specifying a primary creates a forking-paths p-hacking risk that reviewers will flag.

**Source(s)**: S5  
**Plan ref**: B7i

**Description**: Metrics are currently reported on both contaminated and deduplicated splits without pre-specifying which is primary. This creates a forking-paths risk → the author could choose whichever split gives better numbers post-hoc.

**Root cause**: Both splits were treated as equally valid during analysis; the need for a pre-specified primary split was not recognised.

**Resolution**:
1. State "deduplicated split is primary" in Methods section
2. Report contaminated split results only as sensitivity analysis

**Dependencies**: A2 (dedup) must complete first.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-020 [PAPER][P1] Exact-match dedup footnote

**Priority rationale**: Without this footnote, a reviewer familiar with adversarial text variants will question why homoglyph/substitution duplicates were not removed.

**Source(s)**: S5  
**Plan ref**: A2, N5

**Description**: Near-dedup was rejected as scope creep (RJ-004). Exact-match dedup (`df.drop_duplicates`) is used instead. Homoglyph and substitution variants may survive exact-match dedup, meaning contamination may persist at the semantic level.

**Root cause**: The limitation of exact-match dedup was discussed during planning but never documented in the paper.

**Resolution**: Add 1→2 sentence footnote after the dedup description: "Near-deduplication (e.g., MinHash) was considered but deferred. Exact-match dedup may not catch homoglyph or substitution variants."

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-021 [PAPER][P1] Expand Limitations section (energy, RAID noise)

**Priority rationale**: For top-tier venues, the Limitations section must address standard concerns. Current coverage is too narrow.

**Source(s)**: S4  
**Plan ref**: B8a

**Description**: The current Limitations section covers single-seed evaluation, the clean baseline, and data overlap. Missing: energy/carbon impact baseline, RAID dataset label noise and domain imbalance, text-length effects on detection, and threshold-selection sensitivity.

**Root cause**: Limitations was drafted to minimum length for IBCAST; expansion is needed for ACL/EMNLP-level rigour.

**Resolution**: Add paragraphs addressing:
- Energy / carbon impact of single-GPU training
- RAID label noise and domain imbalance
- Text-length dependency of detection accuracy
- Threshold selection sensitivity

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-022 [PAPER][P1] Peak VRAM note in wall-clock table

**Priority rationale**: Memory footprint is a first-class deployment constraint. Reporting latency without VRAM is an incomplete efficiency picture.

**Source(s)**: S4  
**Plan ref**: B8c

**Description**: The wall-clock efficiency table reports latency (time per config) but not peak VRAM consumption. For a study limited to 8GB VRAM on a single GPU, memory footprint is critical information for any reader considering deployment or reproduction.

**Root cause**: VRAM was not tracked during optimization benchmarking; only wall-clock time was recorded.

**Resolution**: Add a peak VRAM column to the wall-clock table, or a VRAM footnote for each optimization configuration.

**Dependencies**: Requires re-running benchmarks with `nvidia-smi` logging enabled.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-023 [DATA][PAPER][P1] 6,029 overlap "modestly inflate" → quantify

**Priority rationale**: With all 5K held-out human texts being duplicates, "modestly" is a qualitative understatement that a reviewer will challenge.

**Source(s)**: S1, S3  
**Plan ref**: A2, B7

**Description**: The paper states the 6,029 overlaps "modestly inflate" reported metrics. With all 5,000 held-out human texts being duplicates (120% overlap rate), this is structural contamination, not a modest effect. The inflation magnitude must be quantified.

**Root cause**: "Modestly inflate" was a qualitative judgment made before dedup quantification was available.

**Resolution**: Report the average metric inflation from overlap (to be computed during A2 dedup). Replace "modestly" with the actual inflation percentage.

**Dependencies**: A2 (dedup) must complete first to compute inflation magnitude.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-024 [PAPER][P1] Single-seed absence → elevate from Limitations to Results

**Priority rationale**: Single-seed evaluation affects the interpretation of every reported number. Burying it in Limitations is insufficient for top-tier venues.

**Source(s)**: S1, S3  
**Plan ref**: B7i

**Description**: Single-seed evaluation is mentioned only in the Limitations section. For top-tier venues, this must be disclosed upfront as a confidence qualifier on every major reported value.

**Root cause**: IBCAST format accepts single-seed evaluation; top-tier disclosure norms are stricter.

**Resolution**: Add sentence after each major metric: "(single-seed evaluation; see Limitations for discussion)."

**Dependencies**: G1 (multi-seed evaluation) will eventually resolve this, but the disclosure should appear regardless.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-025 [CODE][P1] Hardcoded seeds in `train_distilbert_detectrl.py`

**Priority rationale**: Multi-seed evaluation (G1) is impossible without seed as a parameter. Hardcoded seeds block the entire G1 work package.

**Source(s)**: S8  
**Plan ref**: P0.5 (seed param in scripts/train.py), G1

**Description**: `train_distilbert_detectrl.py` had hardcoded random seeds (`42`) instead of an argparse parameter. Multi-seed evaluation (G1, 5 seeds → 4 configs) requires seed as a configurable argument.

**Root cause**: Single-seed was sufficient for IBCAST; the script was never parameterized for multi-seed runs.

**Resolution**:
1. → `scripts/train.py` accepts `--seed` as a configurable parameter (multi-value via `--seeds` in kaggle_run.py)
2. → `kaggle_run.py` wraps the ablation loop with an outer seed loop
3. → Seed is included in checkpoint filenames and cache paths

**Dependencies**: Blocks G1 (multi-seed evaluation).


**Validation**: → PASS (P0.5 completed → seed parameterized in trainer.py + kaggle_run.py)
**Status**: RESOLVED

---

### M-034 [EXPERIMENT][P1] G1 per-config timing correction (plan Appendix B)

**Priority rationale**: Plan claims 1.1h/config for G1 but actual times are 53-88min (~70.5min mean). Error cascades into total GPU budget and schedule.

**Source(s)**: S5  
**Plan ref**: G1 budget

**Description**: Plan claims 1.1h/config for G1. Actual per-config times from prior runs: baseline1=88min, ablation_a=53min, ablation_b=53min, ablation_c=88min. Mean ~70.5min/config. True G1 total is ~14.1h not 13h. Total GPU budget is impacted by this estimation error.

**Root cause**: Timing was estimated from a single ablation run and extrapolated uniformly.

**Resolution**: 1. Update plan budget to reflect 14.1h for G1. 2. Note per-config variance in plan preamble.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-035 [CODE][P1] Dependency matrix correction: decouple G3 from B1

**Priority rationale**: False dependency (G3?B1) serializes independent tasks → B1 is .tex documentation, G3 is GPU compute. Decoupling saves scheduling flexibility.

**Source(s)**: S5  
**Plan ref**: §0, §12

**Description**: The plan's dependency matrix incorrectly makes G3 (FDG baseline on dedup) dependent on B1 (.tex rename). These are independent → B1 is a documentation task, G3 is a compute task. They must be decoupled.

**Root cause**: Both tasks touch "FDG" topic and were conflated in dependency mapping.

**Resolution**: 1. Remove G3→B1 dependency. 2. G3 depends only on A2 (dedup) and A1d (CPU guards). 3. Update dependency graph.

**Dependencies**: None.


**Validation**: ✅ G3→B1 dependency decoupled in plan v2 (revised-paper-fix-plan-v2.md). G3 depends on A2 (dedup) and A1d (CPU guards) only. Dependency graph corrected in §0.
**Status**: RESOLVED

---

### M-036 [FRAMING][P1] Critical path: include A1b?A2 upstream

**Priority rationale**: Stated critical path starts at G1, omitting upstream data pipeline (A1b?A2) that gates ALL GPU work. Scheduling blind spot.

**Source(s)**: S5  
**Plan ref**: Preamble

**Description**: The plan's stated critical path starts at G1, omitting the upstream A1b (parquet naming) → A2 (dedup) path that gates ALL GPU work. Full critical path: A1b → A2 → G1 → G4/G7.

**Root cause**: Plan focused on GPU work as critical path and omitted the data pipeline pre-work.

**Resolution**: 1. Document full critical path. 2. Show A1b + A2 as ~5h pre-work before G1.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-037 [FRAMING][P1] Priority/execution diagram: sequence A+B

**Priority rationale**: Plan diagram shows A and B as parallel but A2 (.tex appendix edits) shares lines L59-60 with B1-B7 (.tex framing edits). Executing in parallel causes merge conflicts.

**Source(s)**: S5  
**Plan ref**: Preamble

**Description**: Plan diagram shows A and B tracks as parallel but Track A includes .tex edits (A2 adds appendix section) that share L59-60 with Track B edits (W1-T2/T3/T4). Must be sequenced: A2 → then B, or use \input{} isolation.

**Root cause**: Parallel diagram was aspirational; actual .tex edit conflicts force serialization.

**Resolution**: 1. Document that A2 (appendix edits) must precede B1-B7 (.tex framing edits) for the same .tex lines. 2. Add \input{} isolation as alternative.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-038 [FRAMING][P1] L60 merge conflict: W1-T2/T3/T4 all modify abstract

**Priority rationale**: Three independent tasks targeting the same .tex line (abstract L59-60) guarantees merge conflicts. Must be merged into a single coordinated edit.

**Source(s)**: S4, S5  
**Plan ref**: W1-T2/T3/T4 merge

**Description**: Three plan tasks (W1-T2="under 16 minutes", W1-T3=AUROC fix, W1-T4=14.99% add) all target abstract L59-60. They cannot execute independently → must be merged into one atomic edit or sequenced carefully.

**Root cause**: Independent task decomposition failed to account for same-line edits.

**Resolution**: 1. Merge W1-T2, W1-T3, W1-T4 into a single "Abstract revision" task. 2. Apply all three changes in one coordinated .tex edit.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-039 [FRAMING][P1] Timeline underestimation: project is 40-80% longer

**Priority rationale**: Plan phase estimates originally ~45h total; hyperplan consensus flagged as 30-80h (total actual from current phase estimates is ~64-84h). Missing data pipeline rebuild, env setup, GPU waiting, debugging loops, and verification overhead.

**Source(s)**: S4, S5  
**Plan ref**: Preamble

**Description**: Plan states ~45h total. Hyperplan consensus: realistic is 30-80h depending on scope. Missing: data pipeline rebuild time, environment setup, GPU waiting time, debugging loops, verification time.

**Root cause**: Estimates assumed linear progress without debugging/reset overhead.

**Resolution**: 1. Document confidence interval on timeline. 2. Add 30% contingency on each task estimate. 3. Plan for worst-case GPU scheduling.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### M-040 [FRAMING][P1] requirements.txt missing runtime deps (W2-T5)

**Priority rationale**: Missing runtime dependencies block reproducibility → new clones and reviewers cannot run code without manual dependency discovery.

**Source(s)**: S4  
**Plan ref**: W2-T5

**Description**: requirements.txt is missing several runtime dependencies. This was accepted without challenge in hyperplan (silent unanimity). Must be fixed for reproducibility.

**Root cause**: Dependencies were installed ad-hoc during development, never recorded in requirements.txt.

**Resolution**: 1. Scan all .py files for imports. 2. Cross-reference against requirements.txt. 3. Add missing entries with versions. (P0.9 addressed this — verified: pyarrow, bitsandbytes, kagglehub added; all deps pinned with ==.)

**Dependencies**: None.

**Validation**: ✅ CONFIRMED — P0.9 covered the same scope; requirements.txt updated with pinned == versions, missing deps added.

**Status**: RESOLVED

---

### M-041 [INFRASTRUCTURE][P1] Docker/reproducibility infra

**Priority rationale**: Python/CUDA pinning was rejected because "Docker subsumes" (RJ-008), but Docker was never created. No reproducible environment exists.

**Source(s)**: S4, S5  
**Plan ref**: TBD

**Description**: Python/CUDA version pinning was REJECTED (RJ-008) because "Docker subsumes." But Docker setup is not yet created. Must create Dockerfile + .dockerignore for a reproducible environment.

**Root cause**: Pinning was rejected as incomplete; replacement (Docker) was never implemented.

**Validation**: ?? PENDING → requires creating and testing Dockerfile on a fresh environment to confirm reproducibility.

**Resolution**: 1. Create Dockerfile with pinned CUDA + Python + torch versions. 2. Add reproducibility docs. 3. Test on clean environment.

**Dependencies**: None.

**Status**: PLANNED

---

### N-001 [EXPERIMENT][P1] Clean-only training baseline absent

**Priority rationale**: Without a clean-only baseline, the central claim about adversarial training cannot be isolated from simply fine-tuning on RAID data.

**Source(s)**: S1, S4  
**Plan ref**: G2

**Description**: No baseline trained on non-adversarial RAID texts exists. Without one, it is impossible to attribute DistilBERT's performance to adversarial training versus simply fine-tuning on the RAID dataset (which includes both human and AI texts).

**Root cause**: The clean-only condition was identified during planning but deferred; G2 was never scheduled for the first wave.

**Resolution**: G2: train DistilBERT on non-adversarial RAID texts only, evaluate on same held-out generators. Compare RRD with the adversarially trained model to isolate the effect of adversarial augmentation.

**Dependencies**: A2 (dedup must come first to avoid contamination in the clean split).


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-002 [EXPERIMENT][P1] Calibration (ECE) not reported

**Priority rationale**: A model with 0.92 F1 but 0.3 ECE is unreliable in production. Reviewers will flag this omission for any deployment-adjacent claim.

**Source(s)**: S3, S4, S5  
**Plan ref**: G5

**Description**: No calibration metrics are reported anywhere in the paper. Probability outputs exist during evaluation but are not saved to disk (see P-014), requiring a separate forward pass. Without ECE and reliability diagrams, the model's confidence calibration is unknown.

**Root cause**: Probability persistence was never implemented (P-014). Calibration was identified as important but deferred.

**Resolution**: G5: forward pass on dedup split, bin probabilities (15 bins), compute ECE + reliability diagram. Estimated ~5h GPU if probabilities are not persisted and must be recomputed.

**Dependencies**: Depends on probability persistence fix (P-014) or willingness to run a separate forward pass.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-003 [EXPERIMENT][P2] Low-FPR metrics (1%/5%/10%) not reported

**Priority rationale**: For real-world deployment where AI text is a minority class, AUROC alone is insufficient. Low-FPR precision is the operationally relevant metric.

**Source(s)**: S3  
**Plan ref**: G4

**Description**: F1 at natural prevalence (50/50) is reported but low-FPR operating points are more relevant for deployment scenarios (where AI text is the minority class). AUROC aggregates over all thresholds and does not guarantee performance at low FPR.

**Root cause**: Original evaluation pipeline only computed F1 and AUROC; low-FPR thresholds were not implemented.

**Resolution**: G4: compute precision @ FPR = 1%, 5%, 10% thresholds. Report in a dedicated table or column.

**Dependencies**: G1 (multi-seed outputs preferred but single-seed acceptable).


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-004 [EXPERIMENT][P2] Bootstrap significance testing missing

**Priority rationale**: Without confidence intervals, all reported RRD differences between ablation configurations could be noise.

**Source(s)**: S4  
**Plan ref**: G7

**Description**: All comparisons in the paper are point estimates with no confidence intervals or significance tests. It is impossible to determine whether the RRD difference between ablation_a and ablation_b is statistically meaningful or within sampling noise.

**Root cause**: Bootstrap evaluation was identified as valuable but deprioritised; no infrastructure exists for resampling-based metrics.

**Resolution**: G7: bootstrap evaluation (1,000 resamples), report 95% CIs for all primary metrics (F1, AUROC, RRD).

**Dependencies**: G1 (multi-seed outputs) preferred but can bootstrap from single-seed predictions.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-005 [EXPERIMENT][P2] Per-attack DistilBERT evaluation

**Priority rationale**: Aggregate metrics may hide catastrophic failure on specific attack families. Per-attack breakdown is standard practice.

**Source(s)**: S3, S4  
**Plan ref**: G6

**Description**: DistilBERT metrics are reported as aggregates across all attack types. A per-attack breakdown (homoglyph, word substitution, paraphrase, prompt-based) is needed to identify weak spots in the model's robustness.

**Root cause**: Only aggregate evaluation was implemented in the inference pipeline; per-attack slicing was never coded.

**Resolution**: G6: evaluate DistilBERT on each attack family separately. Report a table of per-attack F1 across all four ablation configurations.

**Dependencies**: G1 (multi-seed outputs) preferred.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-006 [DATA][EXPERIMENT][P2] Count tables (per-domain / attack / generator)

**Priority rationale**: Reviewers will ask for basic dataset descriptive statistics. An empty "Dataset" section signals insufficient rigour.

**Source(s)**: S4  
**Plan ref**: G8

**Description**: No descriptive statistics of the RAID data distribution are provided. Tables showing texts per domain, per generator, and per attack type (after dedup) are needed for the reader to understand the evaluation scope.

**Root cause**: Dataset description was written qualitatively; quantitative breakdowns were never generated.

**Resolution**: G8: generate count tables from the filtered dedup split. Report per-domain, per-generator, and per-attack-type distributions.

**Dependencies**: A2 (dedup) must complete first.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-007 [EXPERIMENT][P2] TF-IDF + Logistic Regression baseline

**Priority rationale**: If a simple bag-of-words model matches DistilBERT's performance, the claimed advantage of deep learning vanishes.

**Source(s)**: S4  
**Plan ref**: A4

**Description**: No simple bag-of-words baseline (TF-IDF + Logistic Regression) exists. This is the simplest possible text classifier and provides an essential reference point. If word-level patterns suffice for detection, DistilBERT's marginal advantage over this baseline needs to be stated explicitly.

**Root cause**: Only deep-learning baselines were considered; the classic NLP baseline was overlooked.

**Resolution**: A4: train TF-IDF + Logistic Regression on the same train/eval splits. Evaluate on dedup held-out set. Estimated ~1h on CPU.

**Dependencies**: A2 (dedup split must exist).


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-008 [FRAMING][P2] Systems benchmarks compression

**Priority rationale**: Seven figures and tables on PP/AMP/DLO/NF4 speedups dominate the paper's space budget. The scientific contribution is diluted.

**Source(s)**: S1, S4  
**Plan ref**: B9

**Description**: Approximately seven figures and tables documenting PP/AMP/DLO/NF4 speedups overwhelm the paper. These are engineering implementation details, not scientific contributions. They consume space that should be devoted to the core detection-ablation results.

**Root cause**: Engineering optimisations were prominently featured in the IBCAST version; the balance is wrong for a top-tier venue.

**Resolution**: B9: move detailed optimisation results to supplementary material. Keep one summary table in the main paper reporting end-to-end speedup only.

**Dependencies**: None.


**Validation**: ?? PENDING
**Status**: PLANNED

---

### N-009 [HOUSEKEEPING][P2] Filter script docstrings outdated (wrong run commands)

**Priority rationale**: Wrong run commands in docstrings are the first thing a reproducing reviewer encounters → and an immediate friction point.

**Source(s)**: S4  
**Plan ref**: B8b

**Description**: `filter_raid_parallel.py` and `filter_raid_sequential.py` docstrings reference wrong filenames: they say `process_raid_raw` / `process_raid_raw_parallel` instead of their own names. Same issue as H-007 but at P2 severity because of reproducibility impact.

**Root cause**: Docstrings were copied from a template and never updated after file renaming.

**Resolution**: Fix the three docstring run commands to match actual filenames.

**Dependencies**: None.


**Validation**: ✅ Already fixed — docstrings show correct filenames
**Status**: RESOLVED

---

### N-010 [HOUSEKEEPING][P2] AUROC delta rounding fix (0.2637 → 0.2636)

**Priority rationale**: Sub-0.0001 precision inconsistency is minor but catches reviewer attention and erodes trust in reported numbers.

**Source(s)**: S4  
**Plan ref**: B8d

**Description**: The AUROC delta is reported as 0.2637 in Table IV but the actual source precision is 0.2636. The difference is below 0.0001, but the inconsistency between two different precision levels signals carelessness.

**Root cause**: Different rounding conventions used in the `.tex` vs `summary.csv`; no alignment step.

**Validation**: ?? PENDING → requires verifying the actual precision in `summary.csv` vs the reported value in Table IV and aligning to whichever source is correct.

**Resolution**: Align to four decimal places consistently. Either 0.2636 or 0.2637 → whichever matches the source data.

**Dependencies**: None.

**Status**: PLANNED

---

### F-001 [CODE][P0] Weight-decay exclusion for bias/LayerNorm

**Source(s)**: S11  
**Plan ref**: P0.5 (end-to-end plan)

**Description**: `trainer.py:176` → AdamW applies weight decay to all params uniformly. Standard BERT practice explicitly excludes bias terms and LayerNorm weights from weight decay.

**Validation**: → CONFIRMED → No `no_decay` param group split exists in the optimizer setup. Configs A/B/C/D all train with suboptimal regularization.

**Resolution**: Split params into `no_decay` and `decay` groups before passing to AdamW.

**Status**: PLANNED

---

### F-002 [CODE][P1] DataLoader worker seeding missing

**Source(s)**: S11  
**Plan ref**: P0.5 (end-to-end plan)

**Description**: `src/data/dataloader.py` → No `worker_init_fn` on any DataLoader. Multi-worker shuffling is non-reproducible without per-worker seed derivation.

**Validation**: → CONFIRMED → No seed derivation logic exists for DataLoader workers in any code path.

**Resolution**: Add `worker_init_fn` + `generator` argument on all DataLoaders.

**Status**: PLANNED

---

### F-003 [CODE][P1] No LR scheduler (constant LR)

**Source(s)**: S11  
**Plan ref**: P0.5 (end-to-end plan)

**Description**: `src/training/trainer.py:140-175` → Plain AdamW at constant 2e-5. No warmup, no decay. Standard BERT fine-tuning uses linear warmup + decay schedule.

**Validation**: → CONFIRMED → No scheduler instantiated or stepped during training loop.

**Resolution**: Add `get_linear_schedule_with_warmup`, 10% warmup steps, per-batch step.

**Status**: PLANNED

---

### F-004 [INFRASTRUCTURE][P0] DistilBERT not pre-cached to Kaggle Dataset

**Source(s)**: S11  
**Plan ref**: P0.11 (end-to-end plan)

**Description**: Kaggle setup, `src/models/distilbert_classifier.py:23` → Each cold session downloads from HF Hub (~268MB). Unauthenticated rate limit 100 req/hr per shared Kaggle IP → first-session model download could fail silently.

**Validation**: → CONFIRMED → Model weights loaded via `from_pretrained()` with no local fallback or caching guard.

**Resolution**: Upload model weights to Kaggle Dataset + update load path.

**Status**: PLANNED

---

### F-005 [INFRASTRUCTURE][P1] Per-ablation upload pattern risks 429 rate limits

**Source(s)**: S11  
**Plan ref**: P0.6 (end-to-end plan → still PLANNED; batch uploads deferred)

**Description**: `scripts/kaggle_run.py` → Current code uploads 2→ per ablation. Multi-seed G1 = 20 uploads/session. Kaggle API returns 429 after ~15-20 versions/session.

**Validation**: → CONFIRMED → Upload logic runs per-ablation rather than batched end-of-session.

**Resolution**: Batch uploads to end-of-session only; local run_log for intra-session resume.

**Status**: PLANNED

---

### F-006 [INFRASTRUCTURE][P1] Three-level defensive timer missing

**Source(s)**: S11  
**Plan ref**: P0.6 (end-to-end plan → already addressed)

**Description**: `scripts/kaggle_run.py` → Single 8.5h timer. No separate compute/upload deadline tiers. SIGKILL mid-upload corrupts Dataset version.

**Validation**: → CONFIRMED → Only one timer exists; no tiered deadline system.

**Resolution**: Three tiers: 8h/8.25h/8.5h with separate compute/upload/abort actions.

**Status**: RESOLVED

---

### F-007 [CODE][P2] evaluate.py uses no_grad not inference_mode

**Source(s)**: S11  
**Plan ref**: P0.12 (end-to-end plan)

**Description**: `scripts/evaluate.py` → `@torch.no_grad()` still constructs autograd graph (disables grad but not graph). `@torch.inference_mode()` is ~10% faster for pure eval.

**Validation**: ✅ CONFIRMED - scripts/evaluate.py:188 uses @torch.inference_mode(). Also used in scripts/benchmark.py:61,98,104.

**Resolution**: Replace no_grad with inference_mode (already done in evaluate.py and benchmark.py). eval_batch_size still pending.

**Status**: RESOLVED

---

### F-008 [PAPER][P1] RAID AUROC not explicitly reported

**Source(s)**: S11  
**Plan ref**: B7 (end-to-end plan)

**Description**: `Research_Paper.tex` → RAID official benchmark uses per-generator macro-averaged AUROC. Paper reports F1 only. Cross-paper comparison invalid without equivalent metric.

**Validation**: → CONFIRMED → No per-generator AUROC tables in paper. Only F1 reported.

**Resolution**: Add per-generator AUROC tables alongside F1 tables.

**Status**: PLANNED

---

### Section B: Newly Identified (Not in v2 Plan)

---

### M-027 [PAPER][HOUSEKEEPING][P1] 4 unreferenced figure files on disk

**Source(s)**: S8  
**Status**: RESOLVED — 4 PNG files deleted from working tree

**Description**: `figures/figure3_best_val_f1.png`, `figure4_ablation_f1_comparison.png`, `figure5_rrd_by_config.png`, `figure8_three_way_auroc.png` (+ HTML variants) exist on disk but are NOT referenced by any `\includegraphics{}` in the .tex. Only 5/9 figures are used.

**Suggested resolution**: Either add to paper with proper `\caption` + `\label` + text reference, or remove from repo. If adding, consider appendix.

---

### M-026 [DATA][P1] Error messages for missing raw data need improvement

**Priority rationale**: First-run failure with an unhelpful traceback frustrates reproduction → the first thing a reviewer or future researcher experiences.

**Source(s)**: S8  
**Plan ref**: → (not in v2 plan)

**Description**: Filter scripts and training scripts fail with unhelpful tracebacks when raw data files are missing. Should provide clear instructions for download instead of a cryptic `FileNotFoundError`.

**Root cause**: Error handling was not a priority during the initial prototyping phase.

**Resolution**: TBD → not yet assigned to a plan wave.

**Dependencies**: None.

**Status**: PROPOSED

---

### M-028 [PAPER][HOUSEKEEPING][P1] Hyperparameters table has no `\label`

**Source(s)**: S8  
**Status**: REJECTED → `\label{tab:hyperparams}` exists at L582, referenced at L578

**Description**: `Research_Paper.tex` L575: `\begin{table}[t]` for hyperparameters has `\caption` but NO `\label`. It's the only table in the paper without one, and it's never referenced by `\ref{}`.

**Suggested resolution**: Add `\label{tab:hyperparams}` after line 576 caption. Add a text reference like "Table~\ref{tab:hyperparams}" after the table.

**Root cause**: FALSE POSITIVE → the `\label{tab:hyperparams}` exists at L582, and `Table~\ref{tab:hyperparams}` reference exists at L578. Original analysis missed it.

---

### M-029 [FRAMING][P1] No `\usepackage{hyperref}`

**Source(s)**: S8  
**Status**: REJECTED → `\usepackage[breaklinks=true, hidelinks]{hyperref}` exists at L12

**Description**: Preamble (L3-11) does not include `hyperref`. All 25 `\ref{}` calls, 38 `\cite{}` calls, and the GitHub URL are non-clickable. No PDF bookmarks. Strongly recommended for IEEE camera-ready.

**Suggested resolution**: Add `\usepackage[breaklinks=true,colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}` as last package. Add `\hypersetup{pdfauthor={...}}`.

**Root cause**: FALSE POSITIVE → `\usepackage[breaklinks=true, hidelinks]{hyperref}` exists at L12 (included in Fix 2 wave). Original analysis only checked L3-11.

---

### M-030/M-031 [FRAMING][P1] README/paper title and dataset mismatches

**Source(s)**: S8  
**Status**: RESOLVED ✅

**Description**: 
- README title: "...Cross-Attack Generalization Study on **DetectRL**"
- Paper title: "...Held-Out-Generator Study on **RAID**"
- README dataset section describes DetectRL; paper body uses RAID throughout.

**Resolution**: README fully rewritten during Phase 0b to match current paper (RAID-based, held-out-generator). All DetectRL references removed from README (or relegated to historical note under Deprecated section).

---

### M-032 [PAPER][FRAMING][P1] Abstract "adversarially exposed" is non-standard

**Source(s)**: S8  
**Status**: RESOLVED ✅ — abstract now uses "adversarially augmented" language

**Description**: Abstract L38: "adversarially exposed supervised AI text detection" → phrase is ambiguous and non-standard. Does it mean trained on adversarial data? Evaluated under adversarial conditions? Vulnerable to attack?

**Suggested resolution**: Replace with standard phrasing: "supervised AI text detection trained on adversarially augmented data and evaluated under a held-out-generator protocol."

---

### M-033 [FRAMING][P1] Cross-attack generalisation not directly tested

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: RQ1 is framed as "robustness under adversarial conditions" but actually tests generator-family transfer (seen → held-out generators), NOT attack-type transfer (seen → unseen attacks). The paper discusses cross-attack generalisation gap from Huang et al. but doesn't test it.

**Suggested resolution**: Sharpen RQ1 language to explicitly say "generator-family transfer within adversarially diverse RAID data." Do not imply cross-attack transfer.

---

### N-011 [PAPER][HOUSEKEEPING][P2] 10 orphaned `\label` definitions

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: Labels defined but never referenced: `sec:intro`, `ssec:arch`, `ssec:reprosetup`, `ssec:raidsetup`, `ssec:rq1`, `ssec:rq2`, `ssec:efficiency_insights`, `ssec:fastdetect`, `ssec:limits`, `ssec:future`. Either add `\ref{}` calls or remove unused definitions.

---

### N-012 [PAPER][HOUSEKEEPING][P2] No `.bib` file (thebibliography inline)

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: 18 `\bibitem` entries written inline. No `.bib` file, no `\bibliographystyle`, no `\bibliography`. Acceptable but fragile → formatting/punctuation must be hand-maintained.

---

### N-013 through N-016 [CODE][P2]: Code issues (bare except, dead trainer.py, duplicate filter.py, weights_only=False)

**Source(s)**: S8  
**Status**: RESOLVED (P0.3 bare-except sweep / P0.8 trainer rewrite / P1.4 duplicate deletion / P0.7 weights_only fix)

See detailed entries below for each.

**N-013**: `tc3_traindistilbert.py` L158 → `except:` should be `except ValueError:`.  
**N-014**: `src/training/trainer.py` → 217 lines defining `Trainer`/`TrainerConfig`. Zero references anywhere. Dead code.  
**N-015**: `data/filter.py` vs `src/data/filter.py` → both 329 lines, different SHA256. Differences in output prefixes and `ROOT_DIR` calculation. Consolidate.  
**N-016**: `tc4_.py` L74 → `torch.load(weights_only=False)` while rest of codebase uses `True`. Security inconsistency.

---

### N-017 [FRAMING][P2] Overclaiming: "trivially" L72

**Source(s)**: S8  
**Status**: RESOLVED — .tex L73 now says "straightforwardly producible"

**Description**: "LLMs make fluent, human-sounding text **trivially** producible at scale" → overstates effort.  
**Suggested**: "straightforwardly" or restructure.

---

### N-018 [FRAMING][P2] Overclaiming: "first adversarial training approach" L286

**Source(s)**: S8  
**Status**: RESOLVED — .tex L287 now says "pioneering"

**Description**: "RADAR is the **first** adversarial training approach specifically for AI text detection." → Strong historical priority claim.  
**Suggested**: "a pioneering" or "an early" or "to our knowledge, the first published."

---

### N-021 [EXPERIMENT][P2] DeBERTa-v3-LoRA cross-architecture validation DEFERRED

**Priority rationale**: DistilBERT-only primary study covers all core claims and reviewer gap items. DeBERTa adds cross-architecture generalisation evidence but is not required for core contributions. Would push GPU usage past 30h weekly Kaggle quota.

**Status**: PROPOSED — registered for future work. DistilBERT plan locked in for Phase 4.

**Validation**: → Not yet independently validated — deferred to future work cycle.

**Description**: Evaluate DeBERTa-v3-base with LoRA (r=8) using FlashDeBERTa (Triton) for 3-5× training speedup on T4. Compare against DistilBERT AUC-ROC/RRD on RAID. Only pursue if GPU quota surplus exists.

**Root cause**: Kaggle 30h/week GPU quota insufficient for both DistilBERT G1 and DeBERTa cross-arch study in same weekly session.

**Dependencies**: None (deferred). Does not block any Phase 4 task.

### N-019 [HOUSEKEEPING][P2] No `\hyphenation` rules

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: IEEEtran narrow columns prone to poor hyphenation for technical terms. No `\hyphenation{}` defined.  
**Suggested**: Add rules for generalisation, DistilBERT, homoglyph, adversarially, hyperparameters, tokenisation.

---

### N-020 [CODE][BASELINE][P1] Binoculars baseline: missing from peer-baseline comparison

**Source(s)**: S11  
**Status**: PLANNED  
**Plan ref**: B1, G3

**Description**: The codebase implements a Binoculars baseline (`src/baselines/binoculars_baseline.py`) but it has never been run or compared against in the paper. Binoculars (ICML 2024) is the current SOTA zero-shot detector and should be included as a peer baseline alongside Fast-DetectGPT.

**Resolution**:
1. Verify Binoculars runs correctly on RAID data
2. Include Binoculars in G3 (peer-baseline comparison)
3. Report Binoculars metrics in the paper

---

### Section C: Housekeeping/Grooming (P3)

---

### H-001 through H-004 [HOUSEKEEPING][P3]: Unused imports (4 files)

| ID | File | Line | Import | Status |
|----|------|------|--------|--------|
| H-001 | `src/data/processing/filter_raid_parallel.py` | 23 | `import os` (unused → only Path used) | RESOLVED — file already has no `import os` |
| H-002 | `src/data/processing/filter_raid_sequential.py` | 22 | `import random` (unused → only RANDOM_SEED constant) | RESOLVED — file already has no `import random` |
| H-003 | `tc3_traindistilbert.py` | 11 | `import os` (unused → only Path used) | SUPERSEDED *(file deleted in Phase 0b)* |
| H-004 | `data/download_detectrl_HC3.py` | 4 | `import os` (unused → only Path used) | SUPERSEDED *(file deleted in Phase 0b)* |

**Suggested**: Remove each (H-001/H-002 still actionable; H-003/H-004 moot because files were deleted).

---

### H-005 [HOUSEKEEPING][P3] Mid-file argument imports

**Files**: `tc3_traindistilbert.py` L53, `train_distilbert_detectrl.py` L54, `train_distilbert_parallel.py` L92  
**Fix**: Move `import argparse` to top import section.

### H-006 [HOUSEKEEPING][P3] Hardcoded user path in docstring

**File**: `scripts/generate_figures.py` L6  
**Content**: `c:/Users/madha/source/repos/ANN_Project/`  
**Fix**: Replace with `<project-root>/`.  
**Status**: RESOLVED — docstring already uses `<project_root>/`

### H-007 [HOUSEKEEPING][P3] Wrong run commands in 3 docstrings

| File | Says | Should say |
|------|------|-----------|
| `src/data/processing/filter_raid_sequential.py` | `python data/process_raid_raw.py` | `python src/data/processing/filter_raid_sequential.py` |
| `src/data/processing/filter_raid_parallel.py` | `python data/process_raid_raw_parallel.py` | `python src/data/processing/filter_raid_parallel.py` |
| `src/data/processing/download_raid_raw.py` | `python download_raid_full.py` | `python src/data/processing/download_raid_raw.py` |
**Status**: RESOLVED — all 3 docstrings already show correct run commands

### H-008 [HOUSEKEEPING][P3] Missing `__init__.py` in figures/

`src/data/processing/` already has `__init__.py`. `data/` at root has no .py files (moved to `src/data/processing/`) → no `__init__.py` needed. `figures/` contains only output (.png, .html) files → no `__init__.py` needed. `scripts/` contains entry points → add `__init__.py` if importing from them.

### H-009 [HOUSEKEEPING][P3] Hardcoded batch_size (not a constant)

**File**: `src/data/processing/filter_raid_sequential.py` L84 → `iter_batches(batch_size=50_000)` hardcoded inline.  
**Fix**: Extracted to module constant `_PARQUET_READ_BATCH_SIZE: int = 50_000` (sequential filter only; parallel filter instance NOT yet extracted — no active tracking issue; will be fixed if benchmarked as bottleneck).  
**Status**: PARTIAL (sequential done; parallel pending but untracked)

### H-010 [HOUSEKEEPING][P3] Bibliography width parameter

`\begin{thebibliography}{00}` → `{99}` for 22 entries.

### H-011 [HOUSEKEEPING][P3] Commented-out `\IEEEoverridecommandlockouts`

L2 → uncomment or add note.

### H-012 [HOUSEKEEPING][P3] `\centerline` vs `\centering` in figures

**Status**: RESOLVED → grep confirms 0 remaining instances in .tex.

5 instances in figure environments. `\centering` is more standard for IEEEtran.

### H-013 [HOUSEKEEPING][P3] `\smallskip`/`\noindent` for RQ formatting

Lines 135-152. Fragile if page-break occurs.

### H-014 [HOUSEKEEPING][P3] No `\usepackage{subcaption}`

For future multi-panel figures.

### H-015 [HOUSEKEEPING][P3] No appendix section

Consider adding for full diagnostic tables if space permits.

---

### Section D: Rejected Items

| ID | Issue | Reason |
|----|-------|--------|
| RJ-001 | LSP diagnostics (texlab) | Busywork; doesn't affect scientific integrity |
| RJ-002 | Git branch strategy | Over-engineering for single-author |
| RJ-003 | AI prose check | Introduces noise |
| RJ-004 | Near-dedup implementation | Scope creep; exact-match sufficient |
| RJ-005 | Multiple comparison correction | Wrong framework for descriptive ablation |
| RJ-006 | .gitignore fresh clone | Infrastructure, not scientific |
| RJ-007 | Orphaned LSP labels | Busywork |
| RJ-008 | Python/CUDA pinning | Docker subsumes |
| RJ-009 | Near-dedup (duplicate of RJ-004) | Exact-match is sufficient |
| RJ-010 | More process steps | Focus on scientific integrity |

**Note**: M-028 and M-029 (Section B entries above) also have REJECTED status — listed here for completeness. 

---

## 4. Category Summaries

> **Note**: The "Total" column counts category assignments (an issue tagged with multiple categories contributes to multiple rows). The header on line 6 reports unique issue count (114) — these are complementary metrics, not contradictory.

| Category | Total | P0 | P1 | P2 | P3 | REJ | PLANNED | PROPOSED |
| BASELINE | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| CODE | 24 | 13 | 6 | 5 | 0 | 0 | 6 | 0 |
| DATA | 8 | 5 | 2 | 1 | 0 | 0 | 2 | 1 |
| EXPERIMENT | 12 | 2 | 4 | 6 | 0 | 0 | 11 | 1 |
| FRAMING | 29 | 5 | 20 | 3 | 0 | 1 | 14 | 1 |
| HOUSEKEEPING | 24 | 1 | 2 | 5 | 15 | 1 | 1 | 8 |
| INFRASTRUCTURE | 7 | 4 | 3 | 0 | 0 | 0 | 5 | 0 |
| PAPER | 20 | 3 | 14 | 2 | 0 | 1 | 8 | 2 |
| REJECTED | 10 | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| **Total** | 135 | 33 | 52 | 22 | 15 | **13** | 48 | 13 |

---

## 5. Cross-Reference Map: Dependency Graph

```
A0 (contamination audit) -? A2 (dedup) -? Pre-G1 gate -? G1 (multi-seed, 14.1h)
                                                 →              →
A1d (CPU guards) --------------------------------+              →
A1b (parquet naming) -? A2                                        →
                                                                  +--? G4 (low-FPR)
                                                                  +--? G5 (calibration)
                                                                  +--? G6 (per-attack)
                                                                  +--? G7 (bootstrap)
                                                
                                                                  G1 -? G3 (FDG on dedup)
                                                                        (serial, after G1)

A1f (metrics dedup) -? B3 (AUROC range), B5 (Table IV), B7 (framing)

B1 (FDG rename) ---- independent ---- B2 (RRD naming), B4 (gpt-j-6B), B6 (threshold)
A5 (threshold justification) - independent
```

---

## 6. Resolution Conflict Log

| Issue | Conflict | Options | Resolution | Date |
|-------|----------|---------|------------|------|
| P-001 FDG rename | Keep vs rename vs defer | (a) Keep+disclaimer; (b) Rename; (c) Defer | Option C → rename + softened citation + footnote. Consensus: S4, S5 approved | 2026-05-13 |
| M-008 Table IV provenance | Recompute vs keep artifacts | (a) Recompute; (b) Keep+footnote | Keep + rounding footnote. Diff <0.0004 | 2026-05-13 |
| P-003 Contamination | Exact-match vs near-dedup | (a) Exact-match; (b) MinHash/LSH | Exact-match. Near-dedup=footnote only | 2026-05-13 |

---

## 7. Validation Log

| Issue | Method | Result | Date |
|-------|--------|--------|------|
| P-001 | Read perplexity_baseline.py L47-72 | → Perplexity, no curvature | 2026-05-13 |
| P-002 | Checked train scripts for seed | → Single seed, no randomization | 2026-05-13 |
| P-003 | Read filter scripts pooling logic | → Domain concatenation creates duplicates | 2026-05-13 |
| P-004 | Grep L59 vs L1094 | → 0.669 vs 0.526 mismatch | 2026-05-21 |
| P-005 | Grep "5%" in .tex | → No citation found | 2026-05-21 |
| P-006 | Read tc3_traindistilbert.py + tc4_.py for autocast | → 2 unguarded sites (tc3 L188, tc4 L55). Guard variable bypassed in tc3. | 2026-05-21 |
| P-007 | Read filter scripts L217/L185 | → attack_type overwritten | 2026-05-21 |
| P-008 | Read sequential filter output paths | → No raid_ prefix | 2026-05-21 |
| M-027 | Read .tex for \includegraphics vs directory | → 4 files unreferenced | 2026-05-21 |
| M-028 | Read .tex L575-600 | → Table has no \label | 2026-05-21 |
| M-029 | Read .tex preamble | → No hyperref | 2026-05-21 |
| M-030/1 | Compared README.md with .tex titles | → Title + dataset mismatch | 2026-05-21 |
| N-013 | Read tc3 L158 | → Bare except | 2026-05-21 |
| N-014 | Grep for trainer.py references | → Zero results | 2026-05-21 |
| N-015 | SHA256 compare data/filter.py vs src/data/filter.py | → Different hashes | 2026-05-21 |
| N-016 | Read tc4 L74 | → weights_only=False | 2026-05-21 |
| H-001-H-004 | Read each file for unused import | → All confirmed unused | 2026-05-21 |
| H-006 | Read generate_figures.py L6 | → User path in docstring | 2026-05-21 |
| H-007 | Read 3 docstrings | → Run commands wrong | 2026-05-21 |

---

## 8. GLOSSARY

- **FDG**: Fast-DetectGPT (misnamed baseline in codebase)
- **RRD**: Relative Robustness Degradation → (F1_seen - F1_unseen) / F1_seen
- **RAID**: Robust AI Detection dataset (primary training/evaluation data)
- **DetectRL**: Detection of RL-generated text dataset (supporting/legacy)
- **AMP**: Automatic Mixed Precision (training optimization)
- **NF4**: 4-bit NormalFloat quantization
- **DLO**: DataLoader Optimisation (optimised DataLoader pipeline with multi-worker prefetching)
- **PP**: Parallel Preprocessing (TC1)
- **ECE**: Expected Calibration Error
- **AUROC**: Area Under ROC curve
- **LSP**: Language Server Protocol (diagnostics)
- **FPR**: False Positive Rate (used in low-FPR metrics: TPR@1%, TPR@5%, TPR@10%)
- **TPR**: True Positive Rate (evaluation metric at controlled FPR thresholds)
- **TF-IDF**: Term Frequency–Inverse Document Frequency (baseline detector)

---

## 9. Issue-to-Plan Task Mapping

**77 issues mapped to tasks** (19 P-series + 35 M-series + 15 N-series + 8 F-series).

| Plan Ref | Issues | Status |
|----------|--------|--------|
| A0 | P-003, P-010 | COMPLETED |
| A1a | P-011 | INACTIVE |
| A1b | P-008 | COMPLETED |
| A1c | P-012 | INACTIVE |
| A1d | P-006 | COMPLETED |
| A1e | P-016 | COMPLETED |
| A1f | P-013 | RESOLVED |
| A2 | P-003, M-020, M-023 | PARTIAL — dedup operation COMPLETED (P-003 RESOLVED); paper footnotes (M-020, M-023) still PLANNED |
| A3 | P-007, M-005 | COMPLETED |
| A4 | N-007 | PLANNED |
| A5 | → (superseded by B8) | → |
| B1 | P-001, P-009, M-003, N-020 | PARTIAL (P-001, M-003 RESOLVED; P-009, N-020 still PLANNED) |
| B2 | M-009 | RESOLVED |
| B3 | P-004, M-007 | RESOLVED |
| B4 | → | SUPERSEDED |
| B5 | M-008 | PLANNED |
| B6 | M-010 | RESOLVED |
| B7 | F-008, M-011→M-019, M-023, M-024 | PLANNED |
| B8 | P-005 | RESOLVED |
| B8a | M-021 | PLANNED |
| B8b | N-009 | RESOLVED |
| B8c | M-022 | PLANNED |
| B8d | N-010 | PLANNED |
| B9 | N-008 | PLANNED |
| P0.6, §0, §10 | M-001 | PARTIAL |
| §0, §12 | M-035 | RESOLVED |
| C3 | P-014 | COMPLETED |
| G0 | P-015 | PLANNED |
| G1 | P-002, M-025, M-034 | PLANNED |
| G1 budget | M-034 | PLANNED |
| G2 | N-001 | PLANNED |
| G3 | P-009, M-006, N-020 | PLANNED |
| G4 | N-003 | PLANNED (handled under P5.4; no plan §G4 exists) |
| G5 | N-002 | PLANNED |
| G6 | N-005 | PLANNED (handled under P5.1; no plan §G6 exists) |
| G7 | N-004 | PLANNED (handled under P5.2; no plan §G7 exists) |
| G8 | N-006 | PLANNED (handled under P5.5; no plan §G8 exists) |
| M1 | M-004 | PLANNED |
| N5 | → (merged into A2) | → |
| P0 | M-030, M-031 | RESOLVED |
| P0.3 | N-013 | RESOLVED |
| P0.5 | F-001, F-002, F-003, M-025 | PARTIAL |
| P0.6 | F-005, F-006 | PARTIAL |
| P0.7 | N-016 | RESOLVED |
| P0.8 | N-014 | RESOLVED |
| P0.9 | M-040 | RESOLVED |
| P0.11 | F-004 | PLANNED |
| P0.12 | F-007 | COMPLETED — already implemented |
| P1.4 | N-015 | RESOLVED |
| Preamble | M-036, M-037, M-039 | PLANNED |
| Pre-G1 | P-002, P-015, P-017, P-018, P-019 | PLANNED |
| Sec 12 | M-002 | RESOLVED |
| TBD | M-041 | PLANNED |
| W1-T2/T3/T4 merge | M-038 | PLANNED |
