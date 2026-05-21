# Master Issue Register — ANN_Project

**Last updated**: 2026-05-21  
**Maintainer**: Sisyphus (Noire)  
**Source documents**: `.omo/plans/`, `Research_Paper.tex`, codebase (all .py files), GPT 5.5/Gemini audits, hyperplan adversarial reviews  
**Total issues tracked**: **121** (111 active + 10 rejected)

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
| ~~`[REPRODUCIBILITY]`~~ | ~~Seeds, git tags, env lock, reproduce commands~~ | ~~No tagged release, no env lock, split hashes undocumented~~ ⚠️ *Deprecated — concerns folded into [HOUSEKEEPING] and [INFRASTRUCTURE]. No entries currently tagged.* |
| `[HOUSEKEEPING]` | Cleanup, formatting, deps, LSP, unused code | requirements.txt missing deps, unused imports, stale docstrings, orphaned labels |
| `[INFRASTRUCTURE]` | GPU, env, dependencies, tooling | Single GPU, raw data download, CUDA version, Kaggle migration |

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
PROPOSED ──validation──→ VALIDATED ──plan──→ PLANNED ──work──→ IN_PROGRESS ──fix→ RESOLVED ──verify→ VERIFIED ──→ CLOSED
    │                       │
    └──→ REJECTED (if FP)   └──→ PLANNED (accepted)
                              └──→ SUPERSEDED (resolved by broader fix)
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

---

## 2. Issue Index

| ID | Category | Prio | Title | Status | Plan Ref | Source |
|----|----------|------|-------|--------|----------|--------|
| P-001 | PAPER, CODE | P0 | FDG misnamed → GPT-2 XL Perplexity Baseline | PLANNED | B1 | S1,S2,S3,S4,S5 |
| P-002 | EXPERIMENT, FRAMING | P0 | Multi-seed eval could falsify central claim | PLANNED | Pre-G1,G1 | S4,S5 |
| P-003 | DATA, CODE | P0 | Human-text contamination (6,029 overlaps) | PLANNED | A0,A2 | S1,S2,S3,S4 |
| P-004 | PAPER, FRAMING | P0 | Abstract AUROC range wrong (0.669→0.526) | PLANNED | B3 | S2,S6 |
| P-005 | PAPER, FRAMING | P0 | 5% RRD threshold is unjustified (no citation) | PLANNED | B8 | S4,S6 |
| P-006 | CODE | P0 | CPU autocast crash on non-CUDA devices | PLANNED | A1d | S1,S3,S4 |
| P-007 | CODE, DATA | P0 | Attack column hijack (overwrites attack_type) | PLANNED | A3 | S3,S4 |
| P-008 | DATA | P0 | Sequential parquet naming mismatch (no raid_ prefix) | PLANNED | A1b | S3,S4 |
| P-009 | CODE, FRAMING | P0 | FDG comparison is directional, not matched | PLANNED | B1,G3 | S1,S2,S4 |
| P-010 | CODE | P0 | Contamination source unverified (A0 needed) | PLANNED | A0 | S5 |
| P-011 | CODE | P0 | tc3 unseen cache crash | PLANNED | A1a | S4 |
| P-012 | CODE | P0 | tc4 checkpoint paths insufficient | PLANNED | A1c | S1,S4 |
| P-013 | CODE | P0 | compute_metrics triplicated (4 implementations) | PLANNED | A1f | S4,S5 |
| P-014 | CODE | P0 | Probability persistence gap (probs not saved) | PLANNED | C3,G1 | S4 |
| P-015 | EXPERIMENT | P0 | Pre-G1 decision gate needed | PLANNED | Pre-G1 | S5 |
| P-016 | HOUSEKEEPING | P0 | A1e missing from plan (requirements.txt update) | PLANNED | A1e | S4,S5 |
| P-017 | DATA | P0 | Pre-requisite: RAID dataset must be accessible | PLANNED | Pre-G1 | S5 |
| P-018 | INFRASTRUCTURE | P0 | Pre-requisite: CUDA environment working | PLANNED | Pre-G1 | S5 |
| P-019 | INFRASTRUCTURE | P0 | Pre-requisite: GPU thermal baseline | PLANNED | Pre-G1 | S5 |
| M-001 | CODE, DATA | P0 | GPU phase must be serial (1× RTX 3050) | PLANNED | C2 | S5 |
| M-002 | FRAMING | P0 | Priority vs success criteria contradiction | PLANNED | Sec 12 | S5 |
| M-003 | CODE | P0 | FDG file rename + internal string audit needed | PLANNED | B1,P4 | S5 |
| M-004 | INFRASTRUCTURE | P0 | GPU budget under-estimated (25-30h not 21h) | PLANNED | P4 | S5 |
| M-005 | CODE | P1 | A3 column consumer audit needed | PLANNED | A3 | S5 |
| M-006 | EXPERIMENT | P1 | G3 threshold recalibration caveat needed | PLANNED | G3 | S5 |
| M-007 | PAPER, FRAMING | P1 | Abstract missing gpt-j-6B degradation (14.99%) | PLANNED | B4 | S1,S4 |
| M-008 | PAPER | P1 | Table IV confusion matrix provenance unclear | PLANNED | B5,N1 | S2,S5 |
| M-009 | PAPER, FRAMING | P1 | RRD naming — 33.40% is not valid RRD per Eq.2 | PLANNED | B2 | S1,S4 |
| M-010 | PAPER | P1 | Decision threshold wording imprecise ("0.5 argmax") | PLANNED | B6 | S4 |
| M-011 | FRAMING | P1 | Title softening needed | PLANNED | B7a | S4 |
| M-012 | FRAMING | P1 | "adversarially diverse" imprecise | PLANNED | B7b | S4 |
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
| M-023 | DATA, PAPER | P1 | 6,029 overlap "modestly inflate" → quantify | PLANNED | A2,B7 | S1,S3 |
| M-024 | PAPER | P1 | Single-seed absence → elevate from Limitations to Results | PLANNED | B7i | S1,S3 |
| M-025 | CODE | P1 | Hardcoded seeds in train_distilbert_detectrl.py | PLANNED | G1 | S8 |
| M-026 | DATA | P1 | Error messages for missing raw data need improvement | PROPOSED | — | S8 |
| M-027 | PAPER, HOUSEKEEPING | P1 | 4 figure files exist on disk but unreferenced in .tex | PROPOSED | — | S8 |
| M-028 | PAPER, HOUSEKEEPING | P1 | Hyperparameters table has \caption but NO \label | PROPOSED | — | S8 |
| M-029 | FRAMING | P1 | No \usepackage{hyperref} — dead refs, non-clickable URL | PROPOSED | — | S8 |
| M-030 | FRAMING | P1 | Title mismatch: paper=RAID, README=DetectRL | PROPOSED | — | S8 |
| M-031 | FRAMING | P1 | Dataset mismatch: README says DetectRL, paper uses RAID | PROPOSED | — | S8 |
| M-032 | PAPER, FRAMING | P1 | Abstract "adversarially exposed" is non-standard | PROPOSED | — | S8 |
| M-033 | FRAMING | P1 | Cross-attack generalisation not directly tested — RQ1 is about generator transfer | PROPOSED | — | S8 |
| M-034 | EXPERIMENT | P1 | G1 per-config timing correction (plan Appendix B) | PLANNED | G1 budget | S5 |
| M-035 | CODE | P1 | Dependency matrix correction: decouple G3 from B1 | PLANNED | C2 | S5 |
| M-036 | FRAMING | P1 | Critical path: include A1b→A2 upstream | PLANNED | Preamble | S5 |
| M-037 | FRAMING | P1 | Priority/execution diagram: sequence A+B | PLANNED | Preamble | S5 |
| M-038 | FRAMING | P1 | L60 merge conflict: W1-T2/T3/T4 all modify abstract | PLANNED | W1-T2/T3/T4 merge | S4,S5 |
| M-039 | FRAMING | P1 | Timeline underestimation: project is 40-80% longer | PLANNED | Preamble | S4,S5 |
| M-040 | FRAMING | P1 | requirements.txt missing runtime deps (W2-T5) | PLANNED | P0.9 | S4 |
| M-041 | INFRASTRUCTURE | P1 | Docker/reproducibility infra | PLANNED | TBD | S4,S5 |
| N-001 | EXPERIMENT | P1 | Clean-only training baseline absent | PLANNED | G2 | S1,S4 |
| N-002 | EXPERIMENT | P1 | Calibration (ECE) not reported | PLANNED | G5 | S3,S4,S5 |
| N-003 | EXPERIMENT | P2 | Low-FPR metrics (1%/5%/10%) not reported | PLANNED | G4 | S3 |
| N-004 | EXPERIMENT | P2 | Bootstrap significance testing missing | PLANNED | G7 | S4 |
| N-005 | EXPERIMENT | P2 | Per-attack DistilBERT evaluation | PLANNED | G6 | S3,S4 |
| N-006 | DATA, EXPERIMENT | P2 | Count tables (per-domain/attack/generator) | PLANNED | G8 | S4 |
| N-007 | EXPERIMENT | P2 | TF-IDF + Logistic Regression baseline | PLANNED | A4 | S4 |
| N-008 | FRAMING | P2 | Systems benchmarks compression (reduce ~7 figs+tables → 1 para) | PLANNED | B9 | S1,S4 |
| N-009 | HOUSEKEEPING | P2 | Filter script docstrings outdated (wrong run commands) | PLANNED | B8b | S4 |
| N-010 | HOUSEKEEPING | P2 | AUROC delta rounding fix (0.2637→0.2636) | PLANNED | B8d | S4 |
| N-011 | PAPER, HOUSEKEEPING | P2 | 9 orphaned \label definitions (ssec:arch, ssec:rq1, etc.) | PROPOSED | — | S8 |
| N-012 | PAPER, HOUSEKEEPING | P2 | No \bibliographystyle/.bib file (thebibliography inline) | PROPOSED | — | S8 |
| N-013 | CODE | P2 | Bare `except:` in tc3_traindistilbert.py L158 | PROPOSED | — | S8 |
| N-014 | CODE | P2 | `src/training/trainer.py` completely unused (217 lines) | PROPOSED | — | S8 |
| N-015 | CODE | P2 | Duplicate filter.py modules diverged (data/ vs src/data/) | PROPOSED | — | S8 |
| N-016 | CODE | P2 | `torch.load(weights_only=False)` in tc4_.py L74 | PROPOSED | — | S8 |
| N-017 | FRAMING | P2 | Overclaiming: "trivially" → "straightforwardly" L72 | PROPOSED | — | S8 |
| N-018 | FRAMING | P2 | Overclaiming: "first adversarial training approach" L286 | PROPOSED | — | S8 |
| N-019 | HOUSEKEEPING | P2 | No `\hyphenation` rules for IEEEtran column constraints | PROPOSED | — | S8 |
| H-001 | HOUSEKEEPING | P3 | Unused `import os` in filter_raid_parallel.py L23 | PROPOSED | — | S8 |
| H-002 | HOUSEKEEPING | P3 | Unused `import random` in filter_raid_sequential.py L22 | PROPOSED | — | S8 |
| H-003 | HOUSEKEEPING | P3 | Unused `import os` in tc3_traindistilbert.py L11 | PROPOSED | — | S8 |
| H-004 | HOUSEKEEPING | P3 | Unused `import os` in download_detectrl_HC3.py L4 | PROPOSED | — | S8 |
| H-005 | HOUSEKEEPING | P3 | `import argparse` mid-file in 3 training scripts | PROPOSED | — | S8 |
| H-006 | HOUSEKEEPING | P3 | Hardcoded user path in generate_figures.py docstring L6 | PROPOSED | — | S8 |
| H-007 | HOUSEKEEPING | P3 | 3 docstring run commands reference wrong filenames | PROPOSED | — | S8 |
| H-008 | HOUSEKEEPING | P3 | Missing `__init__.py` in data/ and figures/ | PROPOSED | — | S8 |
| H-009 | HOUSEKEEPING | P3 | Hardcoded batch_size 50_000 (not a named constant) in filter_raid_sequential.py | PROPOSED | — | S8 |
| H-010 | HOUSEKEEPING | P3 | `\begin{thebibliography}{00}` → `{99}` for 18 entries | PROPOSED | — | S8 |
| H-011 | HOUSEKEEPING | P3 | Commented-out `\IEEEoverridecommandlockouts` L2 | PROPOSED | — | S8 |
| H-012 | HOUSEKEEPING | P3 | `\centerline` instead of `\centering` in figures (5 instances) | PROPOSED | — | S8 |
| H-013 | HOUSEKEEPING | P3 | `\smallskip`/`\noindent` for RQ formatting (fragile) | PROPOSED | — | S8 |
| H-014 | HOUSEKEEPING | P3 | No `\usepackage{subcaption}` for future multi-panel figures | PROPOSED | — | S8 |
| H-015 | HOUSEKEEPING | P3 | No appendix section | PROPOSED | — | S8 |
| RJ-001 | — | REJECTED | LSP diagnostics (texlab) on .tex | REJECTED | — | S4 |
| RJ-002 | — | REJECTED | Git branch strategy over-engineering | REJECTED | — | S4 |
| RJ-003 | — | REJECTED | AI prose check | REJECTED | — | S4 |
| RJ-004 | — | REJECTED | Near-duplicate dedup implementation (footnote only) | REJECTED | — | S5 |
| RJ-005 | — | REJECTED | Multiple comparison correction | REJECTED | — | S5 |
| RJ-006 | — | REJECTED | .gitignore fresh clone fix | REJECTED | — | S5 |
| RJ-007 | — | REJECTED | Orphaned LSP labels audit | REJECTED | — | S5 |
| RJ-008 | — | REJECTED | Python/CUDA version pinning (Docker subsumes) | REJECTED | — | S4 |
| RJ-009 | — | REJECTED | Near-dedup (duplicate of RJ-004) | REJECTED | — | S5 |
| RJ-010 | — | REJECTED | Add more process steps (LaTeX, git, etc.) | REJECTED | — | S4 |

---

## 3. Detailed Issue Entries

### Section A: Currently PLANNED (in v2 paper-fix-plan or Momus-approved)

> **Convention note**: Combined entries (e.g., P-011–P-015, H-001–H-004) group related low-complexity issues that share a common resolution context. Each sub-ID has a full row in the Index table (Section 2) with complete category, priority, status, and plan-ref metadata. The detailed entries below provide consolidated resolution text where individual entries would be redundant.

---

### P-001 [PAPER][CODE][P0] Fast-DetectGPT → GPT-2 XL Perplexity Baseline rename

**Priority rationale**: Reviewer would immediately flag "Fast-DetectGPT" ≠ what code implements. Integrity issue.

**Source(s)**: S1, S2, S3, S4, S5  
**Plan ref**: B1 in `revised-paper-fix-plan-v2.md`

**Description**: Codebase computes standard perplexity (`log_softmax` → `sum` → `exp(mean)`), NOT Fast-DetectGPT curvature via token perturbation. Paper claims curvature method. Footnote partially addresses.

**Validation**: ✅ CONFIRMED — Docstring (L1-5): `"score each text with GPT-2 XL log p(x)"` and `_score_text()` (L163-197): computes perplexity via cross-entropy loss. No `perturb`/`mask`/`replace` logic anywhere in 416-line file.

**Root cause**: Original planned curvature; time constraints → perplexity-only; docs never updated.

**Resolution** (Option C from hyperplan, Momus-approved):
1. Rename file: `fast_detectgpt.py` → `gpt2_perplexity_baseline.py`
2. Replace ALL occurrences in .tex: "Fast-DetectGPT" → "GPT-2 XL Perplexity Baseline"
3. Change citation: `\cite{mitchell2023}` → `\cite[e.g.,][]{mitchell2023}`
4. Add footnot explaining implementation difference
5. Update internal code comments, docstrings, argparse help, log messages

**Dependencies**: Independent. Does NOT block G3 (dependency matrix corrected per P1).

**Related**: M-005 (G3 threshold caveat), M-009 (RRD naming inequity)

**Status**: PLANNED  
**Resolution decision**: Option C from hyperplan (rename + footnote + softened citation)

**Verification**: `grep -c "Fast-DetectGPT" .` → 0; `grep -c "fast_detectgpt" src/` → 0

---

### P-002 [EXPERIMENT][FRAMING][P0] Multi-seed evaluation could falsify central claim

**Priority rationale**: All reported RRD values (1.1-3.1%) are single-seed on contaminated data. Variance could collapse "within 5%" central claim.

**Source(s)**: S4, S5  
**Plan ref**: Pre-G1 Decision Gate, G1

**Validation**: ✅ CONFIRMED — `train_distilbert_detectrl.py`: no seed randomization; single seed; all metrics are point estimates.

**Root cause**: IBCAST submission accepted single-seed. Top-tier venues require multi-seed + variance.

**Resolution**: 
1. Pre-G1 gate: Single config on dedup split (~4h) to gauge shift
2. G1: 5 seeds × 4 configs on dedup split (~14.1h)
3. CIs separate → proceed + error bars; CIs overlap → pivot narrative (fallback pre-written)

**Dependencies**: A2 (dedup), A1d (CPU guards). Blocks G4/G5/G6/G7.

**Status**: PLANNED  
**Contingency**: Fallback narrative pre-written in plan v2.

---

### P-003 [CODE][DATA][P0] Human-text contamination: 6,029 overlapping texts

**Priority rationale**: 6,029/5,000 held-out human texts (~120% !) seen in training. ALL metrics inflated.

**Source(s)**: S1, S2, S3, S4  
**Plan ref**: A0, A2

**Validation**: ✅ CONFIRMED — Read both filter scripts: domain concatenation without dedup creates cross-split duplicates. Multiplicity >1.0 confirms texts appear 2+ times.

**Root cause**: RAID dataset has cross-domain human texts. Pipeline concatenates all domains → same text included from multiple domains → duplicate in both train and held-out.

**Resolution**: A0 (contamination source audit) → A2 (`df.drop_duplicates(subset=["text"])` on identified pipeline). Multiplicity report. Dedup=primary split.

**Dependencies**: A0 → A2 → G1. Blocks all GPU reruns.

**Status**: PLANNED

---

### P-004 [PAPER][FRAMING][P0] Abstract AUROC range wrong

**Priority rationale**: The abstract reports an AUROC range that does not match the paper's own results — this is a data integrity issue that undermines credibility.

**Source(s)**: S2, S6  
**Plan ref**: B3

**Description**: Abstract (L59) reports AUROC range `0.669--0.789` but the FDG homoglyph result in Section V-D (L1094) shows `AUROC 0.526--0.669`. The abstract range excludes the lowest observed value.

**Validation**: ✅ L59: `(AUROC 0.669--0.789)` but L1094 reports `AUROC 0.526--0.669` for FDG under homoglyph.

**Resolution**: L59 → `(AUROC 0.526--0.789)` (already verified as correct in previous session)

**Status**: PLANNED (simple L59 edit)

---

### P-005 [PAPER][FRAMING][P0] 5% RRD threshold unjustified

**Priority rationale**: The 5% RRD threshold is used as a significance criterion throughout the paper without any supporting citation or statistical justification. Readers and reviewers will question its basis.

**Source(s)**: S4, S6  
**Plan ref**: A5

**Description**: The 5% RRD threshold used to claim "significant degradation" (L59, L205, L514, L1483) has no citation, no theoretical justification, and no bootstrap-derived confidence interval. It is presented as an implicit significance bar without supporting evidence.

**Validation**: ✅ Grep "5%" in .tex — no citation, no theoretical justification.

**Resolution**: A5 — add citation OR reframe as descriptive OR add bootstrap-based justification.

**Status**: PLANNED

---

### P-006 [CODE][P0] CPU autocast crash on non-CUDA devices

**Source(s)**: S1, S3, S4  
**Plan ref**: A1d

**Validation**: ✅ CONFIRMED — 2 unguarded autocast sites found:
- `tc3_traindistilbert.py` L188: `torch.cuda.amp.autocast(enabled=True)` — guard variable `use_amp` defined at L180 but **deliberately commented out** at L187, replaced with hardcoded `True`
- `tc4_.py` L55: `torch.amp.autocast("cuda")` — no guard

**Note**: Register previously claimed 7 sites across 4 files. Audit corrected this: `train_distilbert_detectrl.py` and `train_distilbert_parallel.py` have **zero** autocast calls. 3 guarded sites exist elsewhere (`trainer.py` L86/L135, `fast_detectgpt.py` L188).

**Resolution**: Wrap both sites with CUDA guard. ~15min.

**Dependencies**: Blocks ALL GPU reruns (G1, G2, G3).

**Status**: PLANNED

---

### P-007 [CODE][DATA][P0] Attack column hijack

**Source(s)**: S3, S4  
**Plan ref**: A3

**Validation**: ✅ `filter_raid_parallel.py` L217, `filter_raid_sequential.py` L185: `pool["attack_type"] = pool["generator"].where(...)` — overwrites attack_type with generator name.

**Resolution**: Preserve both `attack_type` and `generator` in separate columns.

**Dependencies**: M-005 (column consumer audit must come first).

**Status**: PLANNED

---

### P-008 [DATA][P0] Sequential parquet naming mismatch

**Source(s)**: S3, S4  
**Plan ref**: A1b

**Validation**: ✅ `filter_raid_sequential.py` outputs `train_pool.parquet` instead of `raid_train_pool.parquet`. Training scripts expect `raid_` prefix.

**Resolution**: Prefix outputs with `raid_`.

**Status**: PLANNED

---

### P-009 [CODE][FRAMING][P0] FDG comparison directional, not matched

**Source(s)**: S1, S2, S4  
**Plan ref**: B1, G3

**Validation**: ✅ DistilBERT reports held-out F1 (0.913-0.927). FDG reports adversarial AUROC (0.526-0.789). Different metrics, different splits, different conditions.

**Resolution**: B1 (rename) + G3 (perplexity baseline on same dedup split) + caveat (M-005).

**Status**: PLANNED

---

### P-010 [CODE][P0] Contamination source unverified before dedup

**Source(s)**: S5  
**Plan ref**: A0

**Validation**: ✅ Raid vs DetectRL — which pipeline produces the 6,029 overlaps? Both must be checked.

**Resolution**: Grep both pipelines; report per-text multiplicity; document in `data/contamination_audit.md`.

**Status**: PLANNED

---

### P-011 through P-015 [CODE, EXPERIMENT][P0]: Code crash + prerequisite fixes

**Note**: P-011 through P-015 share plan refs A1a, A1c, A1f, C3, and Pre-G1. See plan v2 for full details. All PLANNED, all P0. P-011–P-014 are CODE; P-015 is EXPERIMENT.

| ID | Issue | Plan Ref | 
|----|-------|----------|
| P-011 | tc3 unseen cache crash | A1a |
| P-012 | tc4 checkpoint paths insufficient | A1c |
| P-013 | compute_metrics triplicated | A1f |
| P-014 | Probability persistence gap | C3, G1 |
| P-015 | Pre-G1 decision gate | Pre-G1 |

---

### P-016 [HOUSEKEEPING][P0] A1e missing from plan (requirements.txt update)

**Priority rationale**: Missing plan entry for critical dependency fix — stale requirements.txt blocks reproducibility for all GPU work.

**Source(s)**: S4, S5  
**Plan ref**: A1e

**Description**: The v2 plan defines A1a-A1f but A1e is missing. This was the requirements.txt dependency fix (W2-T5 from hyperplan-synthesis.md, silent unanimity). Requirements.txt is stale and missing several runtime dependencies.

**Root cause**: Plan wave mapping missed A1e during restructuring.

**Resolution**: 1. Verify current requirements.txt against all imports. 2. Add missing packages. 3. Sort and deduplicate. 4. Pin compatible versions.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### P-017 [DATA][P0] Pre-requisite: RAID dataset must be accessible

**Priority rationale**: All GPU reruns are blocked without dataset access — cannot start G1-G8 without verifying data integrity.

**Source(s)**: S5  
**Plan ref**: Pre-G1

**Description**: All GPU reruns depend on the RAID dataset being available from disk. The current state of raw data files is unknown. Cannot start G1-G8 without verifying dataset integrity.

**Root cause**: Dataset may have been moved, deleted, or corrupted since IBCAST submission.

**Resolution**: 1. Check data/ directory for raw parquet files. 2. Verify SHA256 or row counts. 3. Document download/recovery steps if missing.

**Dependencies**: None (pre-requisite for all GPU work).


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### P-018 [INFRASTRUCTURE][P0] Pre-requisite: CUDA environment working

**Priority rationale**: All GPU work requires working CUDA — a broken environment blocks every experiment (G1-G8).

**Source(s)**: S5  
**Plan ref**: Pre-G1

**Description**: All GPU work requires a working CUDA environment. Must verify nvidia-smi, torch.cuda.is_available(), and driver compatibility before starting G1.

**Root cause**: Environment may have changed since last run (driver updates, torch reinstalls, etc.).

**Resolution**: 1. Run `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`. 2. Document environment in `.omo/env/`.

**Dependencies**: None (pre-requisite for all GPU work).


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### P-019 [INFRASTRUCTURE][P0] Pre-requisite: GPU thermal baseline

**Priority rationale**: RTX 3050 throttling at ~75°C would corrupt ~14h G1 runs with non-deterministic timing — must be managed upfront.

**Source(s)**: S5  
**Plan ref**: Pre-G1

**Description**: The RTX 3050 throttles at ~75°C. A single G1 run (~14h) without thermal management may throttle and produce non-deterministic timing results.

**Root cause**: Consumer GPU is not designed for sustained compute loads. No active cooling plan.

**Resolution**: 1. Run short benchmark to measure steady-state temperature. 2. Set power limit (e.g., `nvidia-smi -pl 60`). 3. Document thermal status in run log.

**Dependencies**: P-018 (CUDA environment must work first).


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-001 [CODE][DATA][P0] GPU phase must be serial (1× RTX 3050)

**Priority rationale**: Single-GPU constraint adds ~4h unforeseen wall-clock time directly impacting delivery schedule.

**Source(s)**: S5  
**Plan ref**: C2

**Description**: Only one RTX 3050 8GB GPU is available. The plan previously implied G1 (~14.1h) and G3 (~4h) could run in parallel on a "different GPU" — no second GPU exists. Wall-clock minimum: G1 + G3 must run sequentially = ~18h, a hidden +4h (31%) overrun.

**Root cause**: Plan language assumed multi-GPU parallelism. The single-GPU constraint was documented in hardware specs but not factored into the scheduling dependency graph.

**Resolution**:
1. State single GPU as a hard constraint in plan preamble
2. Sequence G1 then G3 explicitly in the dependency graph
3. Remove "on different GPU" / parallel language
4. Add wall-clock estimate to plan header

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-002 [FRAMING][P0] Priority vs success criteria contradiction

**Priority rationale**: P2 tasks gate top-tier success criteria — if they're deferrable, they cannot simultaneously be submission requirements.

**Source(s)**: S5  
**Plan ref**: M6 (hyperplan)

**Description**: Priority tiers label G4 (low-FPR), G7 (bootstrap), and B9 (systems compress) as P2 "minor, deferrable" items. Yet the success criteria for "Top-tier (ACL/EMNLP-ready)" requires ALL tasks complete. If they are P2 nice-to-have tasks, they should not gate top-tier readiness — a logical contradiction.

**Root cause**: Success criteria were drafted before priority assignment; no cross-validation step reconciled the two.

**Resolution**:
1. Restructure success criteria into Core (ACL-ready) vs Extended (journal-ready)
2. Move G4/G7/B9 to Extended tier
3. Update priority table to match

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-003 [CODE][P0] FDG file rename + internal string audit needed

**Priority rationale**: Without the internal audit, the .tex rename (P-001) fixes surface references but code and scripts still say "Fast-DetectGPT."

**Source(s)**: S5  
**Plan ref**: B1, P4

**Description**: Companion to P-001. Beyond the .tex rename, the actual file `fast_detectgpt.py` must be renamed, and ALL internal strings — docstring, argparse help, log messages, code comments — must be audited for "Fast-DetectGPT" references. Leftover references will confuse any reviewer who inspects the code.

**Root cause**: P-001 scope originally focused on .tex only; the code-level rename was always implicit but never task-captured.

**Resolution**:
1. Rename file to `gpt2_perplexity_baseline.py`
2. Grep all internal strings for `fast_detectgpt` / `Fast-DetectGPT` / `FDG`
3. Update imports in any consuming scripts
4. Update README references

**Dependencies**: None (companion to P-001, can run in parallel).


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-004 [INFRASTRUCTURE][P0] GPU budget under-estimated (25–30h not 21h)

**Priority rationale**: A 43% under-estimate threatens the entire delivery schedule. Work cannot be scoped without accurate budget.

**Source(s)**: S5  
**Plan ref**: M1

**Description**: The plan claimed 21h total GPU time (Track B: 11h non-GPU + Track A+G: 10h GPU). Actual minimum: Track A (~5h) + Pre-G1 (~4h) + G1 (~14.1h) + G3 (~4h) + G5 (~5h conditional) = 25–30h GPU alone. The estimate omitted: probability persistence (0–5h conditional) and G5 calibration requiring a separate forward pass.

**Root cause**: Initial estimates optimistically assumed GPU parallelism and omitted conditional work packages. The single-GPU wall-clock constraint compounded the error.

**Resolution**:
1. Update budget to 25–30h
2. Note conditional +5h for G5
3. Explicitly state single-GPU wall-clock (no parallelism possible)

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-005 [CODE][P1] A3 column consumer audit needed

**Priority rationale**: Fixing the column hijack (P-007) could silently break every downstream script if column-name assumptions are not surfaced first.

**Source(s)**: S5  
**Plan ref**: A3

**Description**: Fixing the `attack_type` column hijack (P-007) may break consumers that depend on the current (broken) column schema. Every script that reads filtered parquet files must be audited for column-name assumptions before the fix is applied.

**Root cause**: The column consumer audit was deferred from P-007 scope to keep that fix focused on the filter scripts themselves.

**Resolution**:
1. Identify all consumers (train scripts, eval scripts, `generate_figures.py`)
2. Check column name usage in each
3. Update any that reference the overwritten column
4. Add integration test for column schema

**Dependencies**: Prerequisite for P-007 (attack column fix).


**Validation**: 🔲 PENDING
**Status**: PLANNED

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


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-007 [PAPER][FRAMING][P1] Abstract missing gpt-j-6B degradation (14.99%)

**Priority rationale**: Omitting the 14.99% outlier from the abstract creates a misleading impression that all results are "well within 5%."

**Source(s)**: S1, S4  
**Plan ref**: B4

**Description**: The abstract reports RRD range 1.1–3.1% but omits the 14.99% RRD for gpt-j-6B (ablation_c DistilBERT). This outlier contradicts the "well within 5%" narrative and a reviewer cross-checking against Table III will flag the discrepancy.

**Root cause**: The gpt-j-6B outlier weakens the central narrative and was selectively omitted during abstract drafting.

**Resolution**: Report full RRD range (1.1–14.99%) in the abstract, or qualify to explicitly exclude the outlier with justification.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-008 [PAPER][P1] Table IV confusion matrix provenance unclear

**Priority rationale**: A reviewer cannot determine whether Table IV values are reproducible or an artifact of a specific run.

**Source(s)**: S2, S5  
**Plan ref**: B5, N1

**Description**: Table IV confusion matrix values in the `.tex` differ from `summary.csv` by <0.0004. There is no source trace for which run produced these values — unclear whether from a single seed, an average, or a cherry-picked best run.

**Root cause**: Confusion matrix was extracted from a specific training run at the time of figure generation, but that provenance was not documented.

**Resolution**: Either 1) recompute from dedup multi-seed runs (G1), or 2) add a footnote noting the minimal rounding delta (diff <0.0004).

**Dependencies**: G1 if Option 1 is chosen.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-009 [PAPER][FRAMING][P1] RRD naming — 33.40% is not valid RRD per Eq.2

**Priority rationale**: Labeling a cross-metric comparison as "RRD" violates the paper's own definition. A reviewer will spot this immediately.

**Source(s)**: S1, S4  
**Plan ref**: B2

**Description**: Equation 2 defines RRD = (F1_seen − F1_unseen) / F1_seen. The paper reports 33.40% "RRD" for the perplexity baseline, but this uses FDG F1 (0.913) as "seen" and FDG AUROC (0.526) as "unseen" — mixing metrics violates the Eq.2 definition.

**Root cause**: Cross-metric comparison was used to produce a dramatic RRD number; the definitional conflict was overlooked.

**Resolution**:
1. Recompute RRD using matched metrics (F1 vs F1, or AUROC vs AUROC)
2. Alternatively, explicitly state the metric mismatch and rename the quantity (e.g., "cross-metric gap")

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-010 [PAPER][P1] Decision threshold wording imprecise ("0.5 argmax")

**Priority rationale**: Technical imprecision undermines reviewer confidence in the authors' understanding of their own methodology.

**Source(s)**: S4  
**Plan ref**: B6

**Description**: The paper describes the decision threshold as "0.5 argmax" but DistilBERT uses a binary sigmoid (threshold = 0.5), not a dual-softmax argmax. This is a technical imprecision that a reviewer familiar with binary classification will flag.

**Root cause**: "0.5 argmax" is a colloquialism carried over from multi-class language; not corrected during drafting.

**Resolution**: Replace "0.5 argmax" with "binary sigmoid threshold at 0.5" throughout the `.tex`.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-011 [FRAMING][P1] Title softening needed

**Priority rationale**: "Adversarially Robust" in the title overclaims the study's scope, risking desk-reject or harsh review.

**Source(s)**: S4  
**Plan ref**: B7a

**Description**: The current title "Towards Adversarially Robust AI Text Detection via Supervised DistilBERT Fine-Tuning" — the word "Towards" already softens, but "Adversarially Robust" implies multi-attack-family validation. The study only tests generator-family transfer within a single attack category (character-level substitution).

**Root cause**: Ambitious framing from first draft; scope was narrowed during execution but title was not updated.

**Resolution**: Consider "Towards Adversarially-Trained…" or "Generator-Family Robustness…" or keep as-is with explicit scope boundary added to the abstract.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-012 [FRAMING][P1] "adversarially diverse" imprecise

**Priority rationale**: "Adversarially diverse" implies attack-strategy diversity that the dataset does not provide. Precision matters for reviewer trust.

**Source(s)**: S4  
**Plan ref**: B7b

**Description**: The paper describes RAID as "adversarially diverse" but the diversity is across generators (six LLMs, same attack types: homoglyph + substitution), not across attack strategies. This conflates generator diversity with attack diversity.

**Root cause**: Loose terminology in the RAID description; adopted without scrutiny.

**Resolution**: Replace "adversarially diverse" with "generator-diverse" or "multi-generator" throughout the `.tex`.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-013 [FRAMING][P1] "demonstrates"/"superior"/"substantially" overclaiming

**Priority rationale**: Overclaiming language ("proves", "superior", "first") is the most common reviewer complaint in ML papers.

**Source(s)**: S4, S8  
**Plan ref**: B7c–e

**Description**: Multiple instances of overclaiming language throughout the `.tex`. "Proves" should be "demonstrates", "substantially outperforms" should be "shows competitive performance", "superior" should be directional language. These trigger reviewer scepticism.

**Root cause**: First-draft language adopted enthusiastic framing; no systematic overclaiming audit was conducted before submission.

**Resolution**: Audit `.tex` for: proves, confirms, demonstrates (in overclaiming context), substantially, superior, unprecedented, first. Replace all with measured, evidence-calibrated language.

**Dependencies**: None.


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-016 [FRAMING][P1] Conclusion softened

**Priority rationale**: Stronger-than-supported claims in the conclusion are the last thing a reviewer reads — and remembers.

**Source(s)**: S4  
**Plan ref**: B7f

**Description**: The conclusion makes stronger claims than the results support, e.g., "establishes DistilBERT as a viable approach for real-world deployment" — without multi-domain validation, multi-seed evaluation, or deployment-scale testing.

**Root cause**: Conclusion was drafted for impact; caveats present in earlier sections were not restated.

**Resolution**: Add caveats — single-model, single-dataset, single-seed limitations — before any forward-looking statements. Restrict deployment claims to "warrants further investigation."

**Dependencies**: None.


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-018 [FRAMING][P1] G2 clean baseline confound (dataset shift) limitation

**Priority rationale**: Without this caveat, the clean baseline comparison is scientifically invalid — comparing in-distribution train vs out-of-distribution test.

**Source(s)**: S5  
**Plan ref**: B7h

**Description**: The clean-only DistilBERT baseline (G2) trains on non-adversarial RAID texts but evaluates on adversarial held-out generators. This is a dataset distribution shift, not just "clean vs adversarial." The clean baseline's poor performance partly reflects OOD evaluation, not necessarily the value of adversarial training.

**Root cause**: The limitation was recognised during planning but never documented in the paper.

**Resolution**: Add limitation sentence: "Clean-only baseline evaluated on out-of-distribution adversarial texts, which may underestimate realistic clean performance on in-distribution data."

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-019 [FRAMING][P1] Designate primary split (dedup = primary)

**Priority rationale**: Reporting both splits without pre-specifying a primary creates a forking-paths p-hacking risk that reviewers will flag.

**Source(s)**: S5  
**Plan ref**: B7i

**Description**: Metrics are currently reported on both contaminated and deduplicated splits without pre-specifying which is primary. This creates a forking-paths risk — the author could choose whichever split gives better numbers post-hoc.

**Root cause**: Both splits were treated as equally valid during analysis; the need for a pre-specified primary split was not recognised.

**Resolution**:
1. State "deduplicated split is primary" in Methods section
2. Report contaminated split results only as sensitivity analysis

**Dependencies**: A2 (dedup) must complete first.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-020 [PAPER][P1] Exact-match dedup footnote

**Priority rationale**: Without this footnote, a reviewer familiar with adversarial text variants will question why homoglyph/substitution duplicates were not removed.

**Source(s)**: S5  
**Plan ref**: A2, N5

**Description**: Near-dedup was rejected as scope creep (RJ-004). Exact-match dedup (`df.drop_duplicates`) is used instead. Homoglyph and substitution variants may survive exact-match dedup, meaning contamination may persist at the semantic level.

**Root cause**: The limitation of exact-match dedup was discussed during planning but never documented in the paper.

**Resolution**: Add 1–2 sentence footnote after the dedup description: "Near-deduplication (e.g., MinHash) was considered but deferred. Exact-match dedup may not catch homoglyph or substitution variants."

**Dependencies**: None.


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-025 [CODE][P1] Hardcoded seeds in `train_distilbert_detectrl.py`

**Priority rationale**: Multi-seed evaluation (G1) is impossible without seed as a parameter. Hardcoded seeds block the entire G1 work package.

**Source(s)**: S8  
**Plan ref**: G1

**Description**: `train_distilbert_detectrl.py` has hardcoded random seeds (e.g., `42`) instead of an argparse parameter. Multi-seed evaluation (G1, 5 seeds × 4 configs) requires seed as a configurable argument.

**Root cause**: Single-seed was sufficient for IBCAST; the script was never parameterized for multi-seed runs.

**Resolution**:
1. Add `--seed` argument to argparse
2. Remove hardcoded seed value
3. Update config to accept a seed list for batch runs

**Dependencies**: Blocks G1 (multi-seed evaluation).


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-034 [EXPERIMENT][P1] G1 per-config timing correction (plan Appendix B)

**Priority rationale**: Plan claims 1.1h/config for G1 but actual times are 53-88min (~70.5min mean). Error cascades into total GPU budget and schedule.

**Source(s)**: S5  
**Plan ref**: G1 budget

**Description**: Plan claims 1.1h/config for G1. Actual per-config times from prior runs: baseline1=88min, ablation_a=53min, ablation_b=53min, ablation_c=88min. Mean ~70.5min/config. True G1 total is ~14.1h not 13h. Total GPU budget is impacted by this estimation error.

**Root cause**: Timing was estimated from a single ablation run and extrapolated uniformly.

**Resolution**: 1. Update plan budget to reflect 14.1h for G1. 2. Note per-config variance in plan preamble.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-035 [CODE][P1] Dependency matrix correction: decouple G3 from B1

**Priority rationale**: False dependency (G3→B1) serializes independent tasks — B1 is .tex documentation, G3 is GPU compute. Decoupling saves scheduling flexibility.

**Source(s)**: S5  
**Plan ref**: C2

**Description**: The plan's dependency matrix incorrectly makes G3 (FDG baseline on dedup) dependent on B1 (.tex rename). These are independent — B1 is a documentation task, G3 is a compute task. They must be decoupled.

**Root cause**: Both tasks touch "FDG" topic and were conflated in dependency mapping.

**Resolution**: 1. Remove G3→B1 dependency. 2. G3 depends only on A2 (dedup) and A1d (CPU guards). 3. Update dependency graph.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-036 [FRAMING][P1] Critical path: include A1b→A2 upstream

**Priority rationale**: Stated critical path starts at G1, omitting upstream data pipeline (A1b→A2) that gates ALL GPU work. Scheduling blind spot.

**Source(s)**: S5  
**Plan ref**: Preamble

**Description**: The plan's stated critical path starts at G1, omitting the upstream A1b (parquet naming) → A2 (dedup) path that gates ALL GPU work. Full critical path: A1b → A2 → G1 → G4/G7.

**Root cause**: Plan focused on GPU work as critical path and omitted the data pipeline pre-work.

**Resolution**: 1. Document full critical path. 2. Show A1b + A2 as ~5h pre-work before G1.

**Dependencies**: None.


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-038 [FRAMING][P1] L60 merge conflict: W1-T2/T3/T4 all modify abstract

**Priority rationale**: Three independent tasks targeting the same .tex line (abstract L59-60) guarantees merge conflicts. Must be merged into a single coordinated edit.

**Source(s)**: S4, S5  
**Plan ref**: W1-T2/T3/T4 merge

**Description**: Three plan tasks (W1-T2="under 16 minutes", W1-T3=AUROC fix, W1-T4=14.99% add) all target abstract L59-60. They cannot execute independently — must be merged into one atomic edit or sequenced carefully.

**Root cause**: Independent task decomposition failed to account for same-line edits.

**Resolution**: 1. Merge W1-T2, W1-T3, W1-T4 into a single "Abstract revision" task. 2. Apply all three changes in one coordinated .tex edit.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-039 [FRAMING][P1] Timeline underestimation: project is 40-80% longer

**Priority rationale**: Plan states ~45h total but hyperplan consensus estimates 30-80h. Missing data pipeline rebuild, env setup, GPU waiting, debugging loops, and verification overhead.

**Source(s)**: S4, S5  
**Plan ref**: Preamble

**Description**: Plan states ~45h total. Hyperplan consensus: realistic is 30-80h depending on scope. Missing: data pipeline rebuild time, environment setup, GPU waiting time, debugging loops, verification time.

**Root cause**: Estimates assumed linear progress without debugging/reset overhead.

**Resolution**: 1. Document confidence interval on timeline. 2. Add 30% contingency on each task estimate. 3. Plan for worst-case GPU scheduling.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-040 [FRAMING][P1] requirements.txt missing runtime deps (W2-T5)

**Priority rationale**: Missing runtime dependencies block reproducibility — new clones and reviewers cannot run code without manual dependency discovery.

**Source(s)**: S4  
**Plan ref**: W2-T5

**Description**: requirements.txt is missing several runtime dependencies. This was accepted without challenge in hyperplan (silent unanimity). Must be fixed for reproducibility.

**Root cause**: Dependencies were installed ad-hoc during development, never recorded in requirements.txt.

**Resolution**: 1. Scan all .py files for imports. 2. Cross-reference against requirements.txt. 3. Add missing entries with versions.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### M-041 [INFRASTRUCTURE][P1] Docker/reproducibility infra

**Priority rationale**: Python/CUDA pinning was rejected because "Docker subsumes" (RJ-008), but Docker was never created. No reproducible environment exists.

**Source(s)**: S4, S5  
**Plan ref**: TBD

**Description**: Python/CUDA version pinning was REJECTED (RJ-008) because "Docker subsumes." But Docker setup is not yet created. Must create Dockerfile + .dockerignore for a reproducible environment.

**Root cause**: Pinning was rejected as incomplete; replacement (Docker) was never implemented.

**Validation**: 🔲 PENDING — requires creating and testing Dockerfile on a fresh environment to confirm reproducibility.

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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
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


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### N-009 [HOUSEKEEPING][P2] Filter script docstrings outdated (wrong run commands)

**Priority rationale**: Wrong run commands in docstrings are the first thing a reproducing reviewer encounters — and an immediate friction point.

**Source(s)**: S4  
**Plan ref**: B8b

**Description**: `filter_raid_parallel.py` and `filter_raid_sequential.py` docstrings reference wrong filenames: they say `process_raid_raw` / `process_raid_raw_parallel` instead of their own names. Same issue as H-007 but at P2 severity because of reproducibility impact.

**Root cause**: Docstrings were copied from a template and never updated after file renaming.

**Resolution**: Fix the three docstring run commands to match actual filenames.

**Dependencies**: None.


**Validation**: 🔲 PENDING
**Status**: PLANNED

---

### N-010 [HOUSEKEEPING][P2] AUROC delta rounding fix (0.2637 → 0.2636)

**Priority rationale**: Sub-0.0001 precision inconsistency is minor but catches reviewer attention and erodes trust in reported numbers.

**Source(s)**: S4  
**Plan ref**: B8d

**Description**: The AUROC delta is reported as 0.2637 in Table IV but the actual source precision is 0.2636. The difference is below 0.0001, but the inconsistency between two different precision levels signals carelessness.

**Root cause**: Different rounding conventions used in the `.tex` vs `summary.csv`; no alignment step.

**Validation**: 🔲 PENDING — requires verifying the actual precision in `summary.csv` vs the reported value in Table IV and aligning to whichever source is correct.

**Resolution**: Align to four decimal places consistently. Either 0.2636 or 0.2637 — whichever matches the source data.

**Dependencies**: None.

**Status**: PLANNED

### Section B: Newly Identified (Not in v2 Plan)

---

### M-027 [PAPER][HOUSEKEEPING][P1] 4 unreferenced figure files on disk

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: `figures/figure3_best_val_f1.png`, `figure4_ablation_f1_comparison.png`, `figure5_rrd_by_config.png`, `figure8_three_way_auroc.png` (+ HTML variants) exist on disk but are NOT referenced by any `\includegraphics{}` in the .tex. Only 5/9 figures are used.

**Suggested resolution**: Either add to paper with proper `\caption` + `\label` + text reference, or remove from repo. If adding, consider appendix.

---

### M-026 [DATA][P1] Error messages for missing raw data need improvement

**Priority rationale**: First-run failure with an unhelpful traceback frustrates reproduction — the first thing a reviewer or future researcher experiences.

**Source(s)**: S8  
**Plan ref**: — (not in v2 plan)

**Description**: Filter scripts and training scripts fail with unhelpful tracebacks when raw data files are missing. Should provide clear instructions for download instead of a cryptic `FileNotFoundError`.

**Root cause**: Error handling was not a priority during the initial prototyping phase.

**Resolution**: TBD — not yet assigned to a plan wave.

**Dependencies**: None.

**Status**: PROPOSED

---

### M-028 [PAPER][HOUSEKEEPING][P1] Hyperparameters table has no `\label`

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: `Research_Paper.tex` L575: `\begin{table}[t]` for hyperparameters has `\caption` but NO `\label`. It's the only table in the paper without one, and it's never referenced by `\ref{}`.

**Suggested resolution**: Add `\label{tab:hyperparams}` after line 576 caption. Add a text reference like "Table~\ref{tab:hyperparams}" after the table.

---

### M-029 [FRAMING][P1] No `\usepackage{hyperref}`

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: Preamble (L3-11) does not include `hyperref`. All 25 `\ref{}` calls, 38 `\cite{}` calls, and the GitHub URL are non-clickable. No PDF bookmarks. Strongly recommended for IEEE camera-ready.

**Suggested resolution**: Add `\usepackage[breaklinks=true,colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}` as last package. Add `\hypersetup{pdfauthor={...}}`.

---

### M-030/M-031 [FRAMING][P1] README/paper title and dataset mismatches

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: 
- README title: "...Cross-Attack Generalization Study on **DetectRL**"
- Paper title: "...Held-Out-Generator Study on **RAID**"
- README dataset section describes DetectRL; paper body uses RAID throughout.

**Suggested resolution**: Overwrite README to match current paper (RAID-based, held-out-generator). Remove DetectRL references from README (or relegate to historical note).

---

### M-032 [PAPER][FRAMING][P1] Abstract "adversarially exposed" is non-standard

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: Abstract L38: "adversarially exposed supervised AI text detection" — phrase is ambiguous and non-standard. Does it mean trained on adversarial data? Evaluated under adversarial conditions? Vulnerable to attack?

**Suggested resolution**: Replace with standard phrasing: "supervised AI text detection trained on adversarially augmented data and evaluated under a held-out-generator protocol."

---

### M-033 [FRAMING][P1] Cross-attack generalisation not directly tested

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: RQ1 is framed as "robustness under adversarial conditions" but actually tests generator-family transfer (seen → held-out generators), NOT attack-type transfer (seen → unseen attacks). The paper discusses cross-attack generalisation gap from Huang et al. but doesn't test it.

**Suggested resolution**: Sharpen RQ1 language to explicitly say "generator-family transfer within adversarially diverse RAID data." Do not imply cross-attack transfer.

---

### N-011 [PAPER][HOUSEKEEPING][P2] 9 orphaned `\label` definitions

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: Labels defined but never referenced: `sec:intro`, `ssec:arch`, `ssec:reprosetup`, `ssec:raidsetup`, `ssec:rq1`, `ssec:rq2`, `ssec:efficiency_insights`, `ssec:fastdetect`, `ssec:limits`, `ssec:future`. Either add `\ref{}` calls or remove unused definitions.

---

### N-012 [PAPER][HOUSEKEEPING][P2] No `.bib` file (thebibliography inline)

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: 18 `\bibitem` entries written inline. No `.bib` file, no `\bibliographystyle`, no `\bibliography`. Acceptable but fragile — formatting/punctuation must be hand-maintained.

---

### N-013 through N-016 [CODE][P2]: Code issues (bare except, dead trainer.py, duplicate filter.py, weights_only=False)

**Source(s)**: S8  
**Status**: PROPOSED

See detailed entries below for each.

**N-013**: `tc3_traindistilbert.py` L158 — `except:` should be `except ValueError:`.  
**N-014**: `src/training/trainer.py` — 217 lines defining `Trainer`/`TrainerConfig`. Zero references anywhere. Dead code.  
**N-015**: `data/filter.py` vs `src/data/filter.py` — both 329 lines, different SHA256. Differences in output prefixes and `ROOT_DIR` calculation. Consolidate.  
**N-016**: `tc4_.py` L74 — `torch.load(weights_only=False)` while rest of codebase uses `True`. Security inconsistency.

---

### N-017 [FRAMING][P2] Overclaiming: "trivially" L72

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: "LLMs make fluent, human-sounding text **trivially** producible at scale" — overstates effort.  
**Suggested**: "straightforwardly" or restructure.

---

### N-018 [FRAMING][P2] Overclaiming: "first adversarial training approach" L286

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: "RADAR is the **first** adversarial training approach specifically for AI text detection." — Strong historical priority claim.  
**Suggested**: "a pioneering" or "an early" or "to our knowledge, the first published."

---

### N-019 [HOUSEKEEPING][P2] No `\hyphenation` rules

**Source(s)**: S8  
**Status**: PROPOSED

**Description**: IEEEtran narrow columns prone to poor hyphenation for technical terms. No `\hyphenation{}` defined.  
**Suggested**: Add rules for generalisation, DistilBERT, homoglyph, adversarially, hyperparameters, tokenisation.

---

### Section C: Housekeeping/Grooming (P3)

---

### H-001 through H-004 [HOUSEKEEPING][P3]: Unused imports (4 files)

| ID | File | Line | Import | 
|----|------|------|--------|
| H-001 | `data/filter_raid_parallel.py` | 23 | `import os` (unused — only Path used) |
| H-002 | `data/filter_raid_sequential.py` | 22 | `import random` (unused — only RANDOM_SEED constant) |
| H-003 | `tc3_traindistilbert.py` | 11 | `import os` (unused — only Path used) |
| H-004 | `data/download_detectrl_HC3.py` | 4 | `import os` (unused — only Path used) |

**Suggested**: Remove each.

---

### H-005 [HOUSEKEEPING][P3] Mid-file argument imports

**Files**: `tc3_traindistilbert.py` L53, `train_distilbert_detectrl.py` L54, `train_distilbert_parallel.py` L92  
**Fix**: Move `import argparse` to top import section.

### H-006 [HOUSEKEEPING][P3] Hardcoded user path in docstring

**File**: `figures/generate_figures.py` L6  
**Content**: `c:/Users/madha/source/repos/ANN_Project/`  
**Fix**: Replace with `<project-root>/`.

### H-007 [HOUSEKEEPING][P3] Wrong run commands in 3 docstrings

| File | Says | Should say |
|------|------|-----------|
| `data/filter_raid_sequential.py` | `python data/process_raid_raw.py` | `python data/filter_raid_sequential.py` |
| `data/filter_raid_parallel.py` | `python data/process_raid_raw_parallel.py` | `python data/filter_raid_parallel.py` |
| `data/download_raid_raw.py` | `python download_raid_full.py` | `python data/download_raid_raw.py` |

### H-008 [HOUSEKEEPING][P3] Missing `__init__.py` in data/ and figures/

These directories contain .py files but no `__init__.py`. If anyone imports from them, it'll fail.

### H-009 [HOUSEKEEPING][P3] Hardcoded batch_size (not a constant)

**File**: `data/filter_raid_sequential.py` L84 — `iter_batches(batch_size=50_000)` hardcoded inline.  
**Fix**: Extract to module constant `SEQUENTIAL_BATCH_SIZE = 50_000`.

### H-010 [HOUSEKEEPING][P3] Bibliography width parameter

`\begin{thebibliography}{00}` → `{99}` for 18 entries.

### H-011 [HOUSEKEEPING][P3] Commented-out `\IEEEoverridecommandlockouts`

L2 — uncomment or add note.

### H-012 [HOUSEKEEPING][P3] `\centerline` vs `\centering` in figures

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

---

## 4. Category Summaries

| Category | Total | P0 | P1 | P2 | P3 | REJ | PLANNED | PROPOSED |
|----------|-------|----|----|----|----|-----|---------|----------|
| PAPER | 18 | 3 | 13 | 2 | 0 | 0 | 13 | 5 |
| CODE | 19 | 12 | 3 | 4 | 0 | 0 | 15 | 4 |
| DATA | 8 | 5 | 2 | 1 | 0 | 0 | 7 | 1 |
| EXPERIMENT | 11 | 2 | 4 | 5 | 0 | 0 | 11 | 0 |
| FRAMING | 28 | 5 | 20 | 3 | 0 | 0 | 21 | 7 |
| HOUSEKEEPING | 23 | 1 | 2 | 5 | 15 | 0 | 3 | 20 |
| INFRASTRUCTURE | 4 | 3 | 1 | 0 | 0 | 0 | 4 | 0 |
| REJECTED | 10 | 0 | 0 | 0 | 0 | 10 | 0 | 0 |
| **Total** | **121** | **31** | **45** | **20** | **15** | **10** | **74** | **37** |

---

## 5. Cross-Reference Map: Dependency Graph

```
A0 (contamination audit) ─→ A2 (dedup) ─→ Pre-G1 gate ─→ G1 (multi-seed, 14.1h)
                                                 │              │
A1d (CPU guards) ────────────────────────────────┘              │
A1b (parquet naming) ─→ A2                                        │
                                                                  ├──→ G4 (low-FPR)
                                                                  ├──→ G5 (calibration)
                                                                  ├──→ G6 (per-attack)
                                                                  └──→ G7 (bootstrap)
                                                
                                                                  G1 ─→ G3 (FDG on dedup)
                                                                        (serial, after G1)

A1f (metrics dedup) ─→ B3 (AUROC range), B5 (Table IV), B7 (framing)

B1 (FDG rename) ──── independent ──── B2 (RRD naming), B4 (gpt-j-6B), B6 (threshold)
A5 (threshold justification) ─ independent
```

---

## 6. Resolution Conflict Log

| Issue | Conflict | Options | Resolution | Date |
|-------|----------|---------|------------|------|
| P-001 FDG rename | Keep vs rename vs defer | (a) Keep+disclaimer; (b) Rename; (c) Defer | Option C — rename + softened citation + footnote. Consensus: S4, S5 approved | 2026-05-13 |
| M-008 Table IV provenance | Recompute vs keep artifacts | (a) Recompute; (b) Keep+footnote | Keep + rounding footnote. Diff <0.0004 | 2026-05-13 |
| P-003 Contamination | Exact-match vs near-dedup | (a) Exact-match; (b) MinHash/LSH | Exact-match. Near-dedup=footnote only | 2026-05-13 |

---

## 7. Validation Log

| Issue | Method | Result | Date |
|-------|--------|--------|------|
| P-001 | Read fast_detectgpt.py L47-72 | ✅ Perplexity, no curvature | 2026-05-13 |
| P-002 | Checked train scripts for seed | ✅ Single seed, no randomization | 2026-05-13 |
| P-003 | Read filter scripts pooling logic | ✅ Domain concatenation creates duplicates | 2026-05-13 |
| P-004 | Grep L59 vs L1094 | ✅ 0.669 vs 0.526 mismatch | 2026-05-21 |
| P-005 | Grep "5%" in .tex | ✅ No citation found | 2026-05-21 |
| P-006 | Read tc3_traindistilbert.py + tc4_.py for autocast | ✅ 2 unguarded sites (tc3 L188, tc4 L55). Guard variable bypassed in tc3. | 2026-05-21 |
| P-007 | Read filter scripts L217/L185 | ✅ attack_type overwritten | 2026-05-21 |
| P-008 | Read sequential filter output paths | ✅ No raid_ prefix | 2026-05-21 |
| M-027 | Read .tex for \includegraphics vs directory | ✅ 4 files unreferenced | 2026-05-21 |
| M-028 | Read .tex L575-600 | ✅ Table has no \label | 2026-05-21 |
| M-029 | Read .tex preamble | ✅ No hyperref | 2026-05-21 |
| M-030/1 | Compared README.md with .tex titles | ✅ Title + dataset mismatch | 2026-05-21 |
| N-013 | Read tc3 L158 | ✅ Bare except | 2026-05-21 |
| N-014 | Grep for trainer.py references | ✅ Zero results | 2026-05-21 |
| N-015 | SHA256 compare data/filter.py vs src/data/filter.py | ✅ Different hashes | 2026-05-21 |
| N-016 | Read tc4 L74 | ✅ weights_only=False | 2026-05-21 |
| H-001-H-004 | Read each file for unused import | ✅ All confirmed unused | 2026-05-21 |
| H-006 | Read generate_figures.py L6 | ✅ User path in docstring | 2026-05-21 |
| H-007 | Read 3 docstrings | ✅ Run commands wrong | 2026-05-21 |

---

## 8. GLOSSARY

- **FDG**: Fast-DetectGPT (misnamed baseline in codebase)
- **RRD**: Relative Robustness Degradation — (F1_seen − F1_unseen) / F1_seen
- **RAID**: Robust AI Detection dataset (primary training/evaluation data)
- **DetectRL**: Detection of RL-generated text dataset (supporting/legacy)
- **AMP**: Automatic Mixed Precision (training optimization)
- **NF4**: 4-bit NormalFloat quantization
- **DLO**: Dynamic Learning-rate Optimization
- **PP**: Parallel Preprocessing (TC1)
- **ECE**: Expected Calibration Error
- **AUROC**: Area Under ROC curve
- **LSP**: Language Server Protocol (diagnostics)
