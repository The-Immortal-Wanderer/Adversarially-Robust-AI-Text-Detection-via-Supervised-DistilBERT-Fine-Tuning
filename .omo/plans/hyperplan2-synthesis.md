> [!CAUTION]
> **SUPERSEDED**: This synthesis was absorbed into the [`end-to-end-restructure-plan.md`](./end-to-end-restructure-plan.md). All active work references the end-to-end plan.

# Hyperplan Round 2 — Defensible Insights Bundle

*Generated from 4 adversarial critics × 3 rounds (independent analysis → cross-attack → defend/refine/concede). 42 initial issues distilled to 21 actionable items below.*

---

## TIER 0 — CRITICAL PATH (MUST FIX BEFORE EXECUTION)

### C1. Multi-Seed Could Falsify Central Claim
**Source**: Lateral #7, Deep #9, Logic (R3 refined), Ground. **UNANIMOUS**.
**Risk**: G1 (3-seed eval) may produce overlapping 95% CIs, invalidating "ablation_b is superior" narrative.
**Action**: Add pre-G1 decision gate with fallback narrative:
- CIs separate → proceed with existing narrative + error bars
- CIs overlap → pivot from intra-ablation ranking to cross-baseline comparison (TF-IDF, FDG, clean-only)
**Severity**: Fatal if ignored.

### C2. GPU Phase Is Serial, Not Parallel
**Source**: Logic #5, Lateral #8, Ground #8. **UNANIMOUS**.
**Error**: Plan says G1 ∥ G3 on "different GPU". Only one RTX 3050 exists.
**Action**: G1 (13h) → G3 (4h) sequentially. Add 3-4h to wall clock. Fix execution diagram.
**Severity**: Structural error, 31% GPU-phase overrun.

### C3. Probability Persistence Gap
**Source**: Ground #1, Logic #2, Deep #7 (conceded R2/R3). **UNANIMOUS**.
**Error**: Probs computed via softmax during eval but NOT saved to disk. G5 (calibration) blocked.
**Action**: Either (a) integrate prob saving into G1 (zero extra GPU cost) or (b) separate ~5h forward pass. Plan must specify which.
**Severity**: G5 blocked until resolved.

### C4. Contamination Source Unverified
**Source**: Ground #15, Deep #1 (refined), Lateral #1.
**Error**: A2 targets `data/filter.py` but 6,029 overlaps may be in RAID pipeline not DetectRL. Also 6,029/5,000 ratio >100% implies multi-duplicate contamination — simple dedup removes only 1 instance per text.
**Action**: Before A2, grep both pipelines to identify exact contamination source. Report per-text multiplicity.
**Severity**: A2 may fix wrong pipeline if not verified first.

---

## TIER 1 — MAJOR (MUST INCORPORATE)

### M1. GPU Budget Underestimation
**Sources**: Ground #8,9,14; Logic #2,4,5; Lateral #2.
**Real budget**: ~25-30h GPU (not 21h). Adding probs contingency and serial G3 pushes to ~30h.
**Actions**: (a) Real GPU total = 25h (no probs re-run) or 30h (with probs), (b) add 4h thermal/throttling buffer, (c) state conditional GPU cost explicitly (±5h).
**Severity**: Major budget miss.

### M2. A3 Downstream Fractures Unmapped
**Sources**: Lateral #4, Deep #6, Ground #7.
**Risk**: Changing `filter_raid_parallel.py` column schema silently corrupts downstream scripts.
**Action**: Add ~30min task: grep all column consumers before implementing A3. Document expected columns post-fix.
**Severity**: Silent data corruption risk.

### M3. compute_metrics Triplication
**Sources**: Deep #6 (refined), Lateral #5, Ground (R3).
**Finding**: 4 implementations of compute_metrics across the codebase. A1f fixes only 1. Wrapper divergence can silently produce different numbers from same predictions.
**Action**: Add ~1h task: deduplicate into single source of truth at `src/evaluation/metrics.py` before applying B3/B5 number fixes.
**Severity**: Code consistency time-bomb.

### M4. B1: FDG Rename Misses File/Internal References
**Sources**: Ground #10, Deep #2 (refined).
**Error**: Plan renames paper references but leaves filename as `fast_detectgpt.py` and internal strings (argparse help, log messages) unchanged.
**Action**: Rename file + audit all internal string references. Change citation from direct attribution to "cf. Mitchell et al. 2023."
**Severity**: Reviewer would spot inconsistency.

### M5. G3 Threshold Recalibration Gap
**Sources**: Lateral #6 (DEFEND), Ground (R3).
**Error**: FDG on dedup'd split needs recalibrated perplexity thresholds. G3 has no recalibration step.
**Action**: Add explicit caveat: "FDG on dedup split is directional reference only; thresholds were not recalibrated."
**Severity**: Comparability concern for top-tier.

### M6. Priority vs Success Criteria Contradiction
**Sources**: Logic #8 (new contradiction found in R3).
**Error**: P2 tasks (G4 low-FPR, G7 bootstrap, B9 compress) labeled "minor/nice-to-have" are also listed as required for "Top-tier (ACL/EMNLP-ready)." Same tasks can't be both deferrable and mandatory.
**Action**: Either (a) promote P2 tasks to P0 for top-tier success tier, or (b) split success criteria into "core" (P0-P1) and "extended" (P2).
**Severity**: Inconsistent planning signals.

---

## TIER 2 — MEDIUM (SHOULD INCORPORATE)

### N1. B5: "Corrected Values" Have No Source
**Sources**: Ground #5,13 (ELEVATED).
**Error**: Table IV corrected values (L149-150) reference non-existent "Table VI." Differences from artifacts are rounding noise (<0.0004).
**Action**: Either (a) derive from actual confusion matrix with formula, (b) keep artifact values, or (c) remove B5 (differences are meaningless).
**Severity**: Circular dependency in task.

### N2. G2 Clean Baseline Confound
**Sources**: Deep #5 (refined), Ground (R3).
**Issue**: G2 trains on clean RAID vs original trains on adversarial DetectRL — two datasets differ.
**Action**: Add 2-sentence limitation acknowledgment in paper. No experimental redesign needed.
**Severity**: Standard ablation limitation.

### N3. Dual Reporting: Designate Primary Split
**Sources**: Deep #8 (conceded/refined), Ground (R3), Lateral (R3).
**Issue**: Plan says report BOTH contaminated and deduplicated splits. This is transparent, but abstract/conclusion must designate one PRIMARY.
**Action**: Deduplicated split = primary for abstract/main claims. Contaminated split = supplementary (appendix).
**Severity**: Prevents appearance of cherry-picking.

### N4. 5% RRD Threshold Justification
**Sources**: Deep #10 (refined), Ground (R3).
**Issue**: No citation or theoretical grounding for 5% threshold.
**Action**: Add B-framing task: provide citation or reasoning for threshold choice.
**Severity**: Paper framing vulnerability.

### N5. Exact-Match Dedup Footnote
**Sources**: Lateral #1 (REFINED), Deep #1 (REFINED).
**Consensus**: Exact-match dedup is sufficient for fix cycle. Near-dedup is over-engineering.
**Action**: Add one-sentence footnote: "Deduplication was exact-match only; residual near-duplicate contamination is a limitation."
**Severity**: Low effort, high transparency value.

### N6. G1 Per-Config Time Correction
**Sources**: Ground #9 (ELEVATED).
**Error**: Plan says 1.1h/config. Actual: 88min for baseline1/ablation_c, 53min for ablation_a/b. True G1 = 14.1h not 13h.
**Action**: Update G1 estimate to 14.1h minimum.
**Severity**: 8% undercount on the single longest task.

---

## TIER 3 — PATCH (FIX IN DOCUMENTATION)

### P1. Dependency Matrix: Decouple G3 from B1
**Consensus**: B1 is .tex rename, G3 is code rerun. Zero coupling. Fix dependency matrix. **Trivial.**

### P2. Critical Path: Include A1b→A2 Upstream
**Consensus**: Add full critical path: A1b → A2 → G1 → G4/G7 (not starting at G1). **Documentation.**

### P3. Priority/Execution Diagram: Sequence A+B
**Consensus**: Parallel A+B diagram is misleading. A2 edits .tex (new appendix section). Sequence: A2 → then B, or use \input{} isolation. **Documentation.**

### P4. Naming: fast_detectgpt.py Rename
**Consensus**: Rename file + update all internal references. **Trivial code fix.**

### P5. 6029/5000: Investigate Multi-Duplicate Ratio
**Consensus**: >100% ratio needs investigation before A2. May indicate multi-duplicate contamination. **Pre-A2 gate.**

---

## CONCESSIONS (Demoted/Dropped During Debate)

- Near-dedup implementation → Footnote only. Exact-match sufficient.
- Multiple comparison correction → Wrong framework (descriptive ≠ inferential). Conceded.
- .gitignore fresh clone → Infrastructure, not scientific. Demoted.
- Orphaned LSP labels → Busywork (rejected in Appendix B). Demoted.
- 16-minute claims → Already handled by B7e. Duplicate.
- Merge conflicts → Manageable with sequencing. Overblown.

---

## Handoff to Plan Agent

The above 21 items (4 critical, 6 major, 6 medium, 5 patch) represent the FULL set of defensible insights from a 4-critic × 3-round adversarial review of `.omo/plans/revised-paper-fix-plan.md`.

**Your task**: Incorporate these changes into a new revision of the paper-fix plan. You own: sequencing, parallelization, verification gates, priority triage, and time estimates. Use the existing `revised-paper-fix-plan.md` as a base and produce `revised-paper-fix-plan-v2.md`.

Key structural constraints the critics unanimously validated:
- A1b → A2 → G1 → G4/G7 is the true critical path (~19h minimum)
- Single RTX 3050 — no GPU parallelism
- ~25-30h GPU budget (inclusive of probs contingency and thermal buffer)
- Pre-G1 decision gate with fallback narrative
- Dedup target: verify RAID pipeline contamination source before implementing
