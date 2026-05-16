"""
train_distilbert_parallel.py — RAID edition

Full training pipeline for the cross-generator generalisation study.
Designed to run as a plain Python script (not a notebook) so that
NUM_WORKERS=4 works correctly on Windows via multiprocessing spawn.

TC2 PARALLEL TOKENIZATION:
    Instead of pre-tokenizing everything upfront, this script tokenizes
    ON-THE-FLY inside DataLoader workers. With NUM_WORKERS=4, four CPU
    processes tokenize batches in parallel while the GPU trains simultaneously.
    
    Timeline:
        Worker 1 → tokenize batch 2 ─┐
        Worker 2 → tokenize batch 3  ├─ happening WHILE GPU trains batch 1
        Worker 3 → tokenize batch 4  │
        Worker 4 → tokenize batch 5 ─┘
    
    This is true CPU-GPU pipeline parallelism — CPU and GPU never idle
    waiting for each other.

Usage
-----
    python train_distilbert_parallel.py

Outputs (all saved to artifacts/distilbert_detector/)
------------------------------------------------------
    baseline1_best.pt
    ablation_a_best.pt
    ablation_b_best.pt
    ablation_c_best.pt
    training_times.json
    best_config.json
    tokenizer files
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DistilBertModel


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent
# ✅ FIXED: was "data/data/processed" (double data folder) — now correct
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR  = ROOT_DIR / "artifacts" / "distilbert_detector"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEED            = 42
EPOCHS          = 3
BATCH_SIZE      = 32
MAX_LENGTH      = 256
LR              = 2e-5
WEIGHT_DECAY    = 0.01
GRAD_CLIP_NORM  = 1.0

# ── TC2: PARALLEL TOKENIZATION SETTINGS ──────────────────────────────────────
# NUM_WORKERS=6 means 6 CPU processes tokenize batches simultaneously
# while the GPU trains on the current batch.
# Ryzen 5600X has 6 physical cores / 12 logical cores — 6 workers uses
# all physical cores, leaving logical threads for GPU/OS overhead.
# MUST run as a .py script (not Jupyter) for Windows multiprocessing to work.
NUM_WORKERS     = 6
PIN_MEMORY      = torch.cuda.is_available()   # pins RAM for faster CPU→GPU transfer
PREFETCH_FACTOR = 2                           # each worker stays 2 batches ahead

TOKENIZER_NAME  = "distilbert-base-uncased"
SAMPLES_PER_CLASS = 60_000   # 60k human + 60k AI = 120k total
UNSEEN_CAP        = 10_000   # keep evaluation fast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

_dataset_parser = argparse.ArgumentParser()
_dataset_parser.add_argument("--dataset", choices=["raid", "detectrl"], default="raid",
                    help="Dataset: raid or detectrl (default: raid)")
_dataset_args, _remaining = _dataset_parser.parse_known_args()
DATASET = _dataset_args.dataset


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── TC2: ON-THE-FLY TOKENIZATION DATASET ─────────────────────────────────────
# Defined at module level so multiprocessing workers can pickle it.
#
# Each DataLoader worker process gets its own copy of this dataset and
# tokenizes samples independently. With NUM_WORKERS=4:
#   - Worker 0 handles batches 0, 4, 8, 12 ...
#   - Worker 1 handles batches 1, 5, 9, 13 ...
#   - Worker 2 handles batches 2, 6, 10, 14 ...
#   - Worker 3 handles batches 3, 7, 11, 15 ...
# All 4 workers run simultaneously on separate CPU cores while GPU trains.
class OnTheFlyDataset(Dataset):
    """
    Tokenizes text on-the-fly inside DataLoader worker processes.
    Enables true CPU-GPU parallelism — tokenization and training overlap.
    """
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer_name: str = TOKENIZER_NAME,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.texts          = texts
        self.labels         = labels
        self.tokenizer_name = tokenizer_name
        self.max_length     = max_length
        self._tokenizer     = None   # loaded lazily per worker process

    def _get_tokenizer(self):
        # Each worker process loads its own tokenizer instance.
        # Lazy loading avoids pickling the tokenizer across processes.
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, local_files_only=False
            )
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # This runs inside a worker process — parallel to GPU training
        tokenizer = self._get_tokenizer()
        enc = tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ──────────────────────────────────────────────────────────────────────
class DistilBertClassifier(nn.Module):
    def __init__(self, head_type: str = "single", freeze_layers: int = 0):
        super().__init__()
        if head_type not in {"single", "deep"}:
            raise ValueError("head_type must be 'single' or 'deep'")
        if not 0 <= freeze_layers <= 6:
            raise ValueError("freeze_layers must be between 0 and 6")

        self.head_type     = head_type
        self.freeze_layers = freeze_layers
        self.distilbert    = DistilBertModel.from_pretrained(TOKENIZER_NAME)
        hidden             = self.distilbert.config.hidden_size

        if head_type == "single":
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden, 384),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(384, 2),
            )
        self._freeze_layers()

    def _freeze_layers(self) -> None:
        for p in self.distilbert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.distilbert.transformer.layer):
            req = i >= self.freeze_layers
            for p in layer.parameters():
                p.requires_grad = req

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def get_model_config(ablation: str) -> dict[str, Any]:
    configs = {
        "baseline1":  {"head_type": "single", "freeze_layers": 0},
        "ablation_a": {"head_type": "single", "freeze_layers": 4},
        "ablation_b": {"head_type": "deep",   "freeze_layers": 4},
        "ablation_c": {"head_type": "deep",   "freeze_layers": 0},
    }
    if ablation not in configs:
        raise ValueError(f"Unknown ablation: {ablation}")
    return configs[ablation]


def count_trainable(model: nn.Module) -> tuple[int, int]:
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters()),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    labels: list[int],
    preds: list[int],
    probs: list[float],
    loss: float,
) -> dict[str, float]:
    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "loss":      loss,
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


# ── Epoch runner ──────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    criterion    = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_labels:  list[int]   = []
    all_preds:   list[int]   = []
    all_probs:   list[float] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
            all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return compute_metrics(all_labels, all_preds, all_probs, running_loss / len(loader.dataset))


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(
    ablation_name: str,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    test_loader:   DataLoader,
    unseen_loader: DataLoader,
) -> dict[str, Any]:
    config    = get_model_config(ablation_name)
    model     = DistilBertClassifier(**config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state: dict | None = None
    history: list[dict]    = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[{ablation_name}] epoch={epoch}/{EPOCHS} starting train…", flush=True)
        t0 = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer)
        train_time    = round(time.perf_counter() - t0, 2)

        print(f"[{ablation_name}] epoch={epoch}/{EPOCHS} train done in {train_time}s — running val…", flush=True)
        t1 = time.perf_counter()
        val_metrics = run_epoch(model, val_loader)
        val_time    = round(time.perf_counter() - t1, 2)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })

        print(f"[{ablation_name}] epoch={epoch}/{EPOCHS} train={train_metrics}", flush=True)
        print(f"[{ablation_name}] epoch={epoch}/{EPOCHS} val={val_metrics}",     flush=True)
        print(
            f"[{ablation_name}] epoch={epoch}/{EPOCHS} "
            f"train_time={train_time}s  val_time={val_time}s  "
            f"total_epoch_time={round(train_time + val_time, 2)}s  "
            f"samples/sec={round(len(train_loader.dataset) / train_time, 1)}  "
            f"peak_vram_mb={round(torch.cuda.max_memory_allocated(DEVICE) / 1024**2, 1) if DEVICE.type == 'cuda' else 'N/A'}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {
                "model_state_dict": model.state_dict(),
                "config":           config,
                "ablation_name":    ablation_name,
                "epoch":            epoch,
                "val_metrics":      val_metrics,
            }

    assert best_state is not None
    model.load_state_dict(best_state["model_state_dict"])

    print(f"\n[{ablation_name}] Running test evaluation…",   flush=True)
    test_metrics   = run_epoch(model, test_loader)
    print(f"[{ablation_name}] Running unseen evaluation…",  flush=True)
    unseen_metrics = run_epoch(model, unseen_loader)

    return {
        "ablation_name":    ablation_name,
        "config":           config,
        "history":          history,
        "best_val_f1":      best_val_f1,
        "test_metrics":     test_metrics,
        "unseen_metrics":   unseen_metrics,
        "model_state_dict": model.state_dict(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    seed_everything(SEED)

    print(f"Project root   : {ROOT_DIR}",   flush=True)
    print(f"Processed data : {PROCESSED_DIR}", flush=True)
    print(f"Artifacts      : {ARTIFACT_DIR}", flush=True)
    print(f"Device         : {DEVICE}",      flush=True)
    print(f"NUM_WORKERS    : {NUM_WORKERS}  ← 6 CPU cores tokenize in parallel during training", flush=True)
    print(f"PIN_MEMORY     : {PIN_MEMORY}   ← faster CPU→GPU tensor transfer", flush=True)
    print(f"PREFETCH_FACTOR: {PREFETCH_FACTOR}  ← workers stay {PREFETCH_FACTOR} batches ahead of GPU", flush=True)

    for name in ["baseline1", "ablation_a", "ablation_b", "ablation_c"]:
        cfg = get_model_config(name)
        m   = DistilBertClassifier(**cfg)
        tr, tot = count_trainable(m)
        print(f"{name:12s} -> {cfg} | trainable={tr:,} / total={tot:,}", flush=True)
        del m

    # ── Prepare capped training parquet ──────────────────────────────────────
    train_parquet  = PROCESSED_DIR / f"{DATASET}_train_pool.parquet"
    capped_parquet = PROCESSED_DIR / f"{DATASET}_train_pool_capped.parquet"

    if not capped_parquet.exists():
        print("\nPreparing capped dataset…", flush=True)
        df       = pd.read_parquet(train_parquet)
        df_human = df[df["label"] == 0].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
        df_ai    = df[df["label"] == 1].sample(n=SAMPLES_PER_CLASS, random_state=SEED)
        df_capped = (
            pd.concat([df_human, df_ai])
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True)
        )
        df_capped.to_parquet(capped_parquet, index=False)
        print(f"Capped dataset saved: {len(df_capped):,} rows", flush=True)
        print(df_capped["label"].value_counts().to_string(), flush=True)
    else:
        df_capped = pd.read_parquet(capped_parquet)
        print(f"Capped dataset found: {len(df_capped):,} rows", flush=True)

    # ── Prepare unseen parquet ────────────────────────────────────────────────
    unseen_parquet = PROCESSED_DIR / f"{DATASET}_test_unseen.parquet"
    df_unseen      = pd.read_parquet(unseen_parquet)
    n_unseen = min(
        UNSEEN_CAP // 2,
        (df_unseen["label"] == 0).sum(),
        (df_unseen["label"] == 1).sum(),
    )
    df_unseen_cap = pd.concat([
        df_unseen[df_unseen["label"] == 0].sample(n=n_unseen, random_state=SEED),
        df_unseen[df_unseen["label"] == 1].sample(n=n_unseen, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Unseen pool: {len(df_unseen_cap):,} rows (capped from {len(df_unseen):,})", flush=True)

    # ── Build datasets ────────────────────────────────────────────────────────
    # TC2: OnTheFlyDataset tokenizes each sample inside worker processes
    # — GPU trains batch N while workers tokenize batch N+1, N+2, N+3, N+4
    texts_all  = df_capped["text"].tolist()
    labels_all = df_capped["label"].tolist()
    indices    = list(range(len(texts_all)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=labels_all
    )
    temp_labels = [labels_all[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    def make_ds(idx_list):
        texts  = [texts_all[i] for i in idx_list]
        labels = [labels_all[i] for i in idx_list]
        return OnTheFlyDataset(texts, labels)

    train_ds   = make_ds(train_idx)
    val_ds     = make_ds(val_idx)
    test_ds    = make_ds(test_idx)
    unseen_ds  = OnTheFlyDataset(
        df_unseen_cap["text"].tolist(),
        df_unseen_cap["label"].tolist(),
    )

    # ── DataLoaders with parallel tokenization ────────────────────────────────
    kw = {
        "num_workers":    NUM_WORKERS,
        "pin_memory":     PIN_MEMORY,
        "prefetch_factor": PREFETCH_FACTOR,
    }
    train_loader  = DataLoader(train_ds,  batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader    = DataLoader(val_ds,    batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader   = DataLoader(test_ds,   batch_size=BATCH_SIZE, shuffle=False, **kw)
    unseen_loader = DataLoader(unseen_ds, batch_size=BATCH_SIZE, shuffle=False, **kw)

    print(f"\nTrain:  {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}", flush=True)
    print(f"train_loader : {len(train_loader)} batches", flush=True)
    print(f"val_loader   : {len(val_loader)} batches",   flush=True)
    print(f"test_loader  : {len(test_loader)} batches",  flush=True)
    print(f"unseen_loader: {len(unseen_loader)} batches (capped to {len(unseen_ds):,})", flush=True)

    # ── Run all 4 ablations ───────────────────────────────────────────────────
    results:        dict[str, Any]   = {}
    training_times: dict[str, float] = {}

    for ablation_name in ["baseline1", "ablation_a", "ablation_b", "ablation_c"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Starting ablation : {ablation_name}", flush=True)
        print(f"Config            : {get_model_config(ablation_name)}", flush=True)
        print(f"{'='*60}", flush=True)

        t_start = time.perf_counter()
        results[ablation_name] = train_model(
            ablation_name, train_loader, val_loader, test_loader, unseen_loader,
        )
        elapsed = round(time.perf_counter() - t_start, 2)
        training_times[ablation_name] = elapsed
        print(
            f"\n[{ablation_name}] total training time: {elapsed}s ({elapsed/3600:.2f} hrs)",
            flush=True,
        )

    # ── Summary + RRD ─────────────────────────────────────────────────────────
    summary_rows = []
    for name, result in results.items():
        row = {
            "ablation_name": name,
            "best_val_f1":   result["best_val_f1"],
            **{f"test_{k}":   v for k, v in result["test_metrics"].items()},
            **{f"unseen_{k}": v for k, v in result["unseen_metrics"].items()},
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    print("\nRelative Robustness Degradation (RRD):", flush=True)
    for _, row in summary_df.iterrows():
        seen_f1   = row["test_f1"]
        unseen_f1 = row["unseen_f1"]
        rrd = (seen_f1 - unseen_f1) / seen_f1 * 100 if seen_f1 > 0 else float("nan")
        print(
            f"  {row['ablation_name']:12s}: seen_F1={seen_f1:.4f}  "
            f"unseen_F1={unseen_f1:.4f}  RRD={rrd:.2f}%",
            flush=True,
        )

    # ── Save artifacts ────────────────────────────────────────────────────────
    best_name = summary_df.sort_values("best_val_f1", ascending=False).iloc[0]["ablation_name"]

    for name, result in results.items():
        ckpt_path = ARTIFACT_DIR / f"{DATASET}_{name}_best.pt"
        torch.save(result, ckpt_path)
        print(f"Saved: {ckpt_path}", flush=True)

    with open(ARTIFACT_DIR / f"{DATASET}_training_times.json", "w", encoding="utf-8") as f:
        json.dump(training_times, f, indent=2)

    with open(ARTIFACT_DIR / f"{DATASET}_best_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_ablation_name": best_name,
            "config":             results[best_name]["config"],
            "best_val_f1":        float(results[best_name]["best_val_f1"]),
        }, f, indent=2)

    tokenizer_save = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer_save.save_pretrained(ARTIFACT_DIR / DATASET)
    summary_df.to_csv(ARTIFACT_DIR / f"{DATASET}_summary.csv", index=False)

    print(f"\nAll artifacts saved to {ARTIFACT_DIR}", flush=True)
    print(f"Best ablation: {best_name}", flush=True)