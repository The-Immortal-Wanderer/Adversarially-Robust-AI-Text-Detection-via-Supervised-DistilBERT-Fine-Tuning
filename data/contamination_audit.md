# Contamination Audit: Human-Text Overlap in Training vs. Unseen Evaluation

**Date**: 2026-05-22
**Auditor**: Automated flow analysis
**Issue**: P-010 (contamination source), P-003 (contamination from 6,029/5,000 overlap)

---

## Summary

Human-written texts (label=0) can appear in **both** the training pool (`raid_train_pool.parquet`) and the unseen evaluation pool (`raid_test_unseen.parquet`) because both pools sample from the same set of all human texts without set-exclusion. AI-generated texts (label=1) are cleanly separated by construction (SEEN generators → train, UNSEEN generators → test).

---

## Data Flow (Topology)

```
raw RAID parquet
  │
  ▼
filter_raid_sequential.py / filter_raid_parallel.py
  │
  ├──► _build_train_pool(filtered_df)
  │     ├── human sample (seed=42): up to 60,000 rows
  │     └── AI sample (SEEN_GENERATORS): up to 60,000 rows
  │
  └──► _build_unseen_pool(filtered_df)    ← SAME filtered_df
        ├── human sample (seed=43): up to 10,000 rows
        └── AI sample (UNSEEN_GENERATORS): up to 10,000 rows
```

**Contamination point**: Both pools call `df[df["label"] == 0].sample(n=..., random_state=...)` on the **same** DataFrame containing all human texts. No `~df["text"].isin(train_human_texts)` guard prevents overlap.

---

## What is Contaminated

| Component | Contaminated? | Explanation |
|-----------|--------------|-------------|
| **Human texts in train_pool** | — | Legitimate; they are the non-AI training data |
| **Human texts in unseen_pool** | **YES** | These should be held-out humans, but some may overlap with training humans |
| **AI texts (SEEN generators)** | **NO** | Only appear in train_pool |
| **AI texts (UNSEEN generators)** | **NO** | Only appear in unseen_pool |
| **80/10/10 intra-pool split** | **NO** | sklearn train_test_split produces disjoint index sets |

---

## Fix Applied

The following exclusion logic was added to both filter scripts (`filter_raid_sequential.py:main()` and `filter_raid_parallel.py:main()`):

```python
train_pool = _build_train_pool(filtered_df)
# Exclude human texts used in training from unseen pool
train_human_texts = set(train_pool[train_pool["label"] == 0]["text"])
filtered_for_unseen = filtered_df[~filtered_df["text"].isin(train_human_texts)]
unseen_pool = _build_unseen_pool(filtered_for_unseen)
```

This guarantees that no human-written text in the training pool can also appear in the unseen evaluation pool. AI texts remain unaffected (they were already disjoint by generator identity).

---

## Impact Assessment

Before the fix: if the filtered dataset contains fewer than 70,000 unique human texts (60K train cap + 10K unseen cap), overlap was **mathematically guaranteed**. With 5 distinct domains, each contributing ~12,000–15,000 human samples at source, the total human pool is approximately 60,000–75,000 — meaning overlap was **likely but not certain** before deduplication.

After the fix: zero overlap between train and unseen for humans.

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `src/data/processing/filter_raid_sequential.py` | 217-219 | Added `train_human_texts` set-exclusion before `_build_unseen_pool()` |
| `src/data/processing/filter_raid_parallel.py` | 255-257 | Same fix in parallel counterpart |

---

## Verification

- [X] Both filter scripts have set-exclusion logic between `_build_train_pool()` and `_build_unseen_pool()`
- [X] AI text split remains unchanged (SEEN vs UNSEEN generators)
- [X] Only human texts (label=0) are affected by the exclusion
- [X] The fix is identical in both sequential and parallel implementations
