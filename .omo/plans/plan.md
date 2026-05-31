> **SUPERSEDED**: This plan is from an earlier audit phase and references legacy files (`tc4_.py`, `data/filter_raid_parallel.py`). All tasks have been absorbed into `.omo/plans/end-to-end-restructure-plan.md`.

# Plan: Rigorous Manuscript Audit & Codebase Alignment for Top-Tier Submission

## Objective
Address and resolve all 6 major methodological discrepancies, execution bottlenecks, spelling leaks, and hardware/domain shifts discovered during the deep-dive audit of the DistilBERT fine-tuning manuscript and Python codebase. Ensure the entire pipeline is clean, robust, mathematically accurate, and ready for ARR / top-tier ML venue submission.

## Deliverables
- Fully audited and updated `Research_Paper.tex` with corrected spelling, unified systems abbreviations, reframed quantization speedups/OOM limits, and mathematical RRD alignment.
- Robust checkpoint loader fallback logic in `tc4_.py`.
- Graceful dataset loaders in `src/data/dataloader.py` resolving potential FileNotFoundError.
- Aligned filtering scripts (`data/filter_raid_parallel.py` and `data/filter_raid_sequential.py`) reflecting the exact 5-domain split.

## Acceptance Criteria
1. **Spelling Integrity:** "tokenizer" and "tokenizers" replaced with British English "tokeniser" and "tokenisers" in LaTeX prose (except API calls).
2. **Nomenclature Professionalisation:** All references to "TC1--TC4" and "Test Cases" in LaTeX text and tables replaced with professional systems terms: `PP` (Parallel Preprocessing), `DLO` (DataLoader Optimisation), `AMP-T` (AMP Training), `AMP-I` (AMP Inference).
3. **Execution Robustness:** Running `tc4_.py` loads checkpoints safely from either new runs (`raid_ablation_b_tc3_best_fp32.pt`) or existing checkpoints (`ablation_b_tc3_best.pt`) without raising `FileNotFoundError`.
4. **Data loading Gracefulness:** `src/data/dataloader.py` loads parquet files successfully by trying `raid_` prefixes first before unprefixed names.
5. **Domain Count Consistency:** preprocessing scripts updated to reflect only the 5 populated domains (`news`, `reddit`, `recipes`, `poetry`, `abstracts`), matching the paper and empirical data.
6. **Mathematical Precision:** Table XVIII and body text Fast-DetectGPT RRD values recalculated using paraphrase AUROC as the correct denominator baseline ($50.12\% \rightarrow 33.40\%$, $17.65\% \rightarrow 14.99\%$).
7. **Hardware & Domain Shift Reframing:** Figures and discussion captions reframed to clearly outline the Tesla T4 to RTX 3050 hardware transition and the XSum to RAID domain shifts.

## Implementation Steps

### Phase 1: Codebase Updates
- [ ] Step 1.1: Edit `data/filter_raid_parallel.py` and `data/filter_raid_sequential.py` to limit `KEEP_DOMAINS` to the 5 active domains.
- [ ] Step 1.2: Edit `src/data/dataloader.py` to check for `raid_` parquet files before falling back to unprefixed files.
- [ ] Step 1.3: Edit `tc4_.py` to support fallback paths for loaded checkpoints.

### Phase 2: LaTeX Manuscript Updates
- [ ] Step 2.1: Fix tokenizer spelling leaks (3 occurrences) in `Research_Paper.tex`.
- [ ] Step 2.2: Perform global systems nomenclature update (`TC1--TC4` / `Test Case` to `PP`, `DLO`, `AMP-T`, `AMP-I`, `Stage`) in `Research_Paper.tex`.
- [ ] Step 2.3: Update Table XVIII and Abstract/Body text RRD values for Fast-DetectGPT to mathematically accurate percentages (33.40% and 14.99%).
- [ ] Step 2.4: Reframe the dynamic quantization speedup discussion in captions and text to honestly account for the Tesla T4 / RTX 3050 hardware shift.
- [ ] Step 2.5: Reframe the clean baseline comparison in the discussion to control for the XSum to RAID domain shift.
- [ ] Step 2.6: Change Table VIII's baseline VRAM capacity entry to clearly indicate `8,000 (OOM)` instead of `\sim8,000 (full)`.

### Phase 3: Verification & Compilation
- [ ] Step 3.1: Run `tc4_.py` to verify that checkpoint fallback works.
- [ ] Step 3.2: Review compile output of the LaTeX paper (if applicable) to ensure typesetting is perfect.

## Files & APIs Touched
- `Research_Paper.tex`
- `tc4_.py`
- `src/data/dataloader.py`
- `data/filter_raid_parallel.py`
- `data/filter_raid_sequential.py`

## Manual QA / Verification Steps
- **Smoke Check:** Run `tc4_.py` from terminal and confirm it executes without path-related crashes.
- **Visual Validation:** Double check all tables in `Research_Paper.tex` for alignment and mathematical correctness.
- **Observability Check:** Confirm that no COURSEWORK indicator terms remain.
