"""
DetectRL DistilBERT Training Script
Converted from train_distilbert.ipynb for terminal execution on Windows.
NUM_WORKERS=4 is safe when run as a standalone script (not inside Jupyter).
Optimised for Ryzen 5600X (6 cores / 12 threads) + RTX 3050 8GB.
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
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertModel


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if not (ROOT_DIR / "data").exists():
    ROOT_DIR = ROOT_DIR.parent

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR  = ROOT_DIR / "artifacts" / "distilbert_detector"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SEED           = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS         = 3
BATCH_SIZE     = 32
LR             = 2e-5
WEIGHT_DECAY   = 0.01
GRAD_CLIP_NORM = 1.0
# 4 workers = safe on Windows when run as script (not in Jupyter)
# Ryzen 5600X has 12 logical cores — 4 workers leaves headroom for GPU/OS
NUM_WORKERS    = 4
PIN_MEMORY     = torch.cuda.is_available()
TOKENIZER_NAME = "distilbert-base-uncased"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
        self.distilbert    = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size        = self.distilbert.config.hidden_size

        if head_type == "single":
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 384),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(384, 2),
            )

        self._freeze_layers()

    def _freeze_layers(self) -> None:
        for parameter in self.distilbert.embeddings.parameters():
            parameter.requires_grad = False
        for layer_index, layer in enumerate(self.distilbert.transformer.layer):
            requires_grad = layer_index >= self.freeze_layers
            for parameter in layer.parameters():
                parameter.requires_grad = requires_grad

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_hidden)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Ablation configs ───────────────────────────────────────────────────────────
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


# ── Cached tensor dataset (must be at module level for multiprocessing) ────────
class CachedTensorDataset(torch.utils.data.Dataset):
    """Loads pre-tokenized tensors from disk. __getitem__ is O(1)."""
    def __init__(self, cache_path: Path):
        data                = torch.load(cache_path, weights_only=True)
        self.input_ids      = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels         = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ── Metrics helpers ────────────────────────────────────────────────────────────
def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs       = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    targets     = labels.detach().cpu().numpy()
    metrics = {
        "accuracy":  accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall":    recall_score(targets, predictions, zero_division=0),
        "f1":        f1_score(targets, predictions, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(targets, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def move_batch_to_device(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.to(DEVICE) for k, v in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train  = optimizer is not None
    model.train() if is_train else model.eval()
    criterion    = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        batch  = move_batch_to_device(batch)
        labels = batch["labels"]

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss   = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits          = torch.cat(all_logits, dim=0)
    labels          = torch.cat(all_labels, dim=0)
    metrics         = compute_metrics(logits, labels)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


# ── Train one ablation ─────────────────────────────────────────────────────────
def train_model(
    ablation_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    unseen_loader: DataLoader,
) -> dict[str, Any]:
    config    = get_model_config(ablation_name)
    model     = DistilBertClassifier(**config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_state  = None
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[{ablation_name}] epoch={epoch}/{EPOCHS} starting train...", flush=True)
        epoch_start   = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer)
        train_time    = round(time.perf_counter() - epoch_start, 2)

        print(f"[{ablation_name}] epoch={epoch}/{EPOCHS} train done in {train_time}s — running val...", flush=True)
        val_start   = time.perf_counter()
        val_metrics = run_epoch(model, val_loader)
        val_time    = round(time.perf_counter() - val_start, 2)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })

        print(f"[{ablation_name}] epoch={epoch}/{EPOCHS} train={train_metrics}", flush=True)
        print(f"[{ablation_name}] epoch={epoch}/{EPOCHS} val={val_metrics}",    flush=True)
        print(
            f"[{ablation_name}] epoch={epoch}/{EPOCHS} "
            f"train_time={train_time}s  val_time={val_time}s  "
            f"total_epoch_time={round(train_time + val_time, 2)}s",
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
    print(f"\n[{ablation_name}] Running test evaluation...", flush=True)
    test_metrics   = run_epoch(model, test_loader)
    print(f"[{ablation_name}] Running unseen evaluation...", flush=True)
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


# ── Entry point ────────────────────────────────────────────────────────────────
# The if __name__ == "__main__" guard is REQUIRED for NUM_WORKERS > 0 on Windows.
# Without it, each worker process re-imports this module and spawns more workers
# causing an infinite fork bomb crash.
if __name__ == "__main__":
    print(f"Project root   : {ROOT_DIR}", flush=True)
    print(f"Processed data : {PROCESSED_DIR}", flush=True)
    print(f"Artifacts      : {ARTIFACT_DIR}", flush=True)
    seed_everything(SEED)
    print(f"Device  : {DEVICE}", flush=True)
    print(f"Workers : {NUM_WORKERS}", flush=True)

    # Print trainable params per ablation
    for ablation_name in ["baseline1", "ablation_a", "ablation_b", "ablation_c"]:
        cfg   = get_model_config(ablation_name)
        model = DistilBertClassifier(**cfg)
        trainable = count_trainable_parameters(model)
        total     = sum(p.numel() for p in model.parameters())
        print(f"{ablation_name:12s} -> {cfg} | trainable={trainable:,} / total={total:,}", flush=True)
        del model

    # Cap dataset to 60k (30k human + 30k AI)
    print("\nPreparing capped dataset...", flush=True)
    df        = pd.read_parquet("data/processed/train_pool.parquet")
    df_human  = df[df["label"] == 0].sample(n=60000, random_state=42)
    df_ai     = df[df["label"] == 1].sample(n=60000, random_state=42)
    df_capped = (
        pd.concat([df_human, df_ai])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    df_capped.to_parquet("data/processed/train_pool_capped.parquet", index=False)
    print(f"Capped dataset saved: {len(df_capped)} rows", flush=True)
    print(df_capped["label"].value_counts(), flush=True)

    # ── Pre-tokenization ───────────────────────────────────────────────────────
    # Tokenize entire dataset once and cache to disk as tensors.
    # This means DataLoader workers just load pre-computed tensors — near-instant.
    # GPU utilization jumps from 30-50% to 80-90%+ as a result.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def pretokenize_parquet(
        parquet_path: Path,
        cache_path: Path,
        max_samples: int | None = None,
    ) -> None:
        """Tokenize a parquet file and save input_ids, attention_mask, labels as tensors."""
        if cache_path.exists():
            print(f"Cache found, skipping tokenization: {cache_path.name}", flush=True)
            return
        print(f"Pre-tokenizing {parquet_path.name} → {cache_path.name} ...", flush=True)
        df = pd.read_parquet(parquet_path)
        if max_samples is not None:
            df = df.iloc[:max_samples].copy()
        texts  = df["text"].tolist()
        labels = df["label"].tolist()
        encoded = tokenizer(
            texts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        torch.save(
            {
                "input_ids":      encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels":         torch.tensor(labels, dtype=torch.long),
            },
            cache_path,
        )
        print(f"Saved {len(labels):,} samples to {cache_path.name}", flush=True)

    CACHE_DIR = PROCESSED_DIR / "tokenized_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pretokenize_parquet(
        PROCESSED_DIR / "train_pool_capped.parquet",
        CACHE_DIR / "train_pool_capped_256.pt",
    )
    pretokenize_parquet(
        PROCESSED_DIR / "test_unseen.parquet",
        CACHE_DIR / "test_unseen_10k_256.pt",
        max_samples=10000,
    )

    # ── Split train into train/val/test 80/10/10 ──────────────────────────────
    full_ds    = CachedTensorDataset(CACHE_DIR / "train_pool_capped_256.pt")
    n          = len(full_ds)
    n_train    = int(0.8 * n)
    n_val      = int(0.1 * n)
    n_test     = n - n_train - n_val

    # Stratified split — reuse same indices as before (seed=42)
    rng        = torch.Generator().manual_seed(SEED)
    indices    = torch.randperm(n, generator=rng).tolist()
    train_idx  = indices[:n_train]
    val_idx    = indices[n_train:n_train + n_val]
    test_idx   = indices[n_train + n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}", flush=True)

    def make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    unseen_ds     = CachedTensorDataset(CACHE_DIR / "test_unseen_10k_256.pt")
    unseen_loader = make_loader(unseen_ds, shuffle=False)

    print(f"train_loader : {len(train_loader)} batches", flush=True)
    print(f"val_loader   : {len(val_loader)} batches",   flush=True)
    print(f"test_loader  : {len(test_loader)} batches",  flush=True)
    print(f"unseen_loader: {len(unseen_loader)} batches (capped to 10k)", flush=True)

    sample_batch = next(iter(train_loader))
    print({k: v.shape for k, v in sample_batch.items()}, flush=True)

    # Train all 4 ablations
    results: dict[str, Any]       = {}
    training_times: dict[str, float] = {}

    for ablation_name in ["baseline1", "ablation_a", "ablation_b", "ablation_c"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Starting ablation : {ablation_name}", flush=True)
        print(f"Config            : {get_model_config(ablation_name)}", flush=True)
        print(f"{'='*60}", flush=True)

        t_start = time.perf_counter()
        results[ablation_name] = train_model(
            ablation_name, train_loader, val_loader, test_loader, unseen_loader
        )
        training_times[ablation_name] = round(time.perf_counter() - t_start, 2)
        print(
            f"\n[{ablation_name}] total training time: {training_times[ablation_name]}s "
            f"({training_times[ablation_name]/3600:.2f} hrs)",
            flush=True,
        )

    # Summary table
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
    print(f"\n{'='*60}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(summary_df.sort_values("best_val_f1", ascending=False).reset_index(drop=True).to_string(), flush=True)

    # Save checkpoints
    best_ablation_name = summary_df.sort_values("best_val_f1", ascending=False).iloc[0]["ablation_name"]
    best_result        = results[best_ablation_name]

    for name, result in results.items():
        checkpoint_path = ARTIFACT_DIR / f"{name}_best.pt"
        torch.save(result, checkpoint_path)
        print(f"Saved: {checkpoint_path}", flush=True)

    with open(ARTIFACT_DIR / "training_times.json", "w", encoding="utf-8") as f:
        json.dump(training_times, f, indent=2)

    with open(ARTIFACT_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_ablation_name": best_ablation_name,
                "config":             best_result["config"],
                "best_val_f1":        float(best_result["best_val_f1"]),
            },
            f,
            indent=2,
        )

    tokenizer.save_pretrained(ARTIFACT_DIR)
    print(f"\nBest ablation : {best_ablation_name}", flush=True)
    print(f"All artifacts saved to : {ARTIFACT_DIR}", flush=True)