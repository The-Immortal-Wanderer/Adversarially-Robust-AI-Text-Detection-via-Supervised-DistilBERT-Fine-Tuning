# Claude Review Evaluation Synthesis
Generated: 2026-05-24

## Summary
5-agent parallel evaluation of Claude's 40+ claims across hardware, code, methodology, operations.
- **12 confirmed actionable** (implement)
- **10 rejected/straw-man** (ignore)
- **5 nuanced** (directionally right, wrong details)

## Actionable Changes to Sync into Plans
1. Fix weight decay param groups (no_decay for bias/LayerNorm) — trainer.py
2. Add per-parameter-group LR (pretrained 2e-5, head 5e-4)
3. Add LR scheduler (linear warmup + decay, per-batch)
4. Add DataLoader worker seeding (worker_init_fn on all DataLoaders)
5. Batch uploads to once-per-session (avoid 429 rate limits)
6. Pre-download DistilBERT to Kaggle Dataset
7. Remove data mirror (read /kaggle/input/ directly)
8. Three-level timer (8h/8.25h/8.5h)
9. HF_TOKEN from Kaggle Secrets
10. Pin library versions in requirements.txt
11. evaluate.py: inference_mode + larger eval batches
12. G0 gate: fix 2% threshold + multi-criteria logic
13. Paper: report AUROC (not just F1) for RAID comparability
