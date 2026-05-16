"""
train_distilbert_tc3.py — RAID edition with AMP (TC3)

This script implements Test Case 3: Automatic Mixed Precision (AMP).
It enables GradScaler and autocast to reduce VRAM usage and speed up training.
"""

from __future__ import annotations

import json
import os
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
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, DistilBertModel

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR  = ROOT_DIR / "artifacts" / "distilbert_detector_tc3"
CACHE_DIR     = PROCESSED_DIR / "tokenized_cache"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEED            = 42
EPOCHS          = 3
BATCH_SIZE      = 128
MAX_LENGTH      = 256
LR              = 2e-5
WEIGHT_DECAY    = 0.01
GRAD_CLIP_NORM  = 1.0
NUM_WORKERS     = 4
PIN_MEMORY      = torch.cuda.is_available()
TOKENIZER_NAME  = "distilbert-base-uncased"

SAMPLES_PER_CLASS = 60_000
UNSEEN_CAP = 10_000

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

# ── Cached tensor dataset ─────────────────────────────────────────────────────
class CachedTensorDataset(Dataset):
    def __init__(self, pt_path: Path) -> None:
        data = torch.load(pt_path, weights_only=True)
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

# ── Pre-tokenisation helper ───────────────────────────────────────────────────
def pretokenize(df: pd.DataFrame, tokenizer, cache_path: Path, max_length: int = MAX_LENGTH) -> Path:
    if cache_path.exists():
        print(f"Cache found, skipping: {cache_path.name}", flush=True)
        return cache_path

    print(f"Pre-tokenising → {cache_path.name} …", flush=True)
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    enc = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    torch.save({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }, cache_path)
    print(f"Saved {len(labels):,} samples to {cache_path.name}", flush=True)
    return cache_path

# ── Model ──────────────────────────────────────────────────────────────────────
class DistilBertClassifier(nn.Module):
    def __init__(self, head_type: str = "single", freeze_layers: int = 0):
        super().__init__()
        self.head_type    = head_type
        self.freeze_layers = freeze_layers
        self.distilbert   = DistilBertModel.from_pretrained(TOKENIZER_NAME)
        hidden            = self.distilbert.config.hidden_size

        if head_type == "single":
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(hidden, 384), nn.GELU(), nn.Dropout(0.2), nn.Linear(384, 2)
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
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

def get_model_config(ablation: str) -> dict[str, Any]:
    configs = {
        "baseline1":  {"head_type": "single", "freeze_layers": 0},
        "ablation_a": {"head_type": "single", "freeze_layers": 4},
        "ablation_b": {"head_type": "deep",   "freeze_layers": 4},
        "ablation_c": {"head_type": "deep",   "freeze_layers": 0},
    }
    return configs[ablation]

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(labels: list[int], preds: list[int], probs: list[float], loss: float) -> dict[str, float]:
    metrics = {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "loss":      loss,
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except:
        metrics["roc_auc"] = float("nan")
    return metrics

# ── Epoch runner with TC3 (AMP) ────────────────────────────────────────────────
def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None, # Added Scaler for TC3
) -> dict[str, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    criterion   = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    # Use autocast for FP16 operations if on CUDA
    use_amp = (DEVICE.type == "cuda")

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, DEVICE)
            
            # --- START AMP (TC3) ---
            #with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss   = criterion(logits, batch["labels"])

            if training:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
            # --- END AMP (TC3) ---

            running_loss += loss.item() * batch["labels"].size(0)
            probs  = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            preds  = logits.argmax(dim=1).detach().cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    return compute_metrics(all_labels, all_preds, all_probs, avg_loss)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(
    ablation_name: str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    unseen_loader: DataLoader,
) -> dict[str, Any]:
    config = get_model_config(ablation_name)
    model  = DistilBertClassifier(**config).to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Initialize Scaler for TC3
    #scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val_f1 = -1.0
    best_state, history = None, []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        # Passing scaler to run_epoch for TC3
        train_metrics = run_epoch(model, train_loader, optimizer, scaler=scaler)
        train_time    = round(time.perf_counter() - t0, 2)

        val_metrics = run_epoch(model, val_loader)
        
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        
        print(f"[{ablation_name}] Epoch {epoch} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f} | VRAM: {torch.cuda.max_memory_allocated(DEVICE)/1024**2:.1f}MB", flush=True)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state  = {"model_state_dict": model.state_dict(), "config": config, "ablation_name": ablation_name}

    model.load_state_dict(best_state["model_state_dict"])
    return {
        "ablation_name": ablation_name,
        "config": config,
        "history": history,
        "best_val_f1": best_val_f1,
        "test_metrics": run_epoch(model, test_loader),
        "unseen_metrics": run_epoch(model, unseen_loader),
        "model_state_dict": model.state_dict(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    seed_everything(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ── Dataset Loading ──────────────────────────────────────
    train_parquet = PROCESSED_DIR / f"{DATASET}_train_pool_capped.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"Run original script first to create {train_parquet}")

    df_capped = pd.read_parquet(train_parquet)
    train_cache = CACHE_DIR / f"{DATASET}_train_pool_capped_{MAX_LENGTH}.pt"
    unseen_cache = CACHE_DIR / f"{DATASET}_test_unseen_10k_{MAX_LENGTH}.pt"

    pretokenize(df_capped, tokenizer, train_cache)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    full_ds = CachedTensorDataset(train_cache)
    indices = list(range(len(full_ds)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=full_ds.labels.tolist())
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED, stratify=[full_ds.labels[i] for i in temp_idx])

    kw = {"num_workers": NUM_WORKERS, "pin_memory": PIN_MEMORY}
    train_loader  = DataLoader(Subset(full_ds, train_idx),  batch_size=BATCH_SIZE, shuffle=True,  **kw)
    val_loader    = DataLoader(Subset(full_ds, val_idx),    batch_size=BATCH_SIZE, shuffle=False, **kw)
    test_loader   = DataLoader(Subset(full_ds, test_idx),   batch_size=BATCH_SIZE, shuffle=False, **kw)
    unseen_loader = DataLoader(CachedTensorDataset(unseen_cache), batch_size=BATCH_SIZE, shuffle=False, **kw)

    # ── Run TC3 Ablation ─────────────────────────────────────────────────────────
    # For TC3 demonstration, we run the specific 'ablation_b' configuration
    ablation_name = "ablation_b"
    print(f"\nRunning TC3 (AMP) Experiment on {ablation_name}...", flush=True)
    
    torch.cuda.reset_peak_memory_stats(DEVICE)
    t_start = time.perf_counter()
    result = train_model(ablation_name, train_loader, val_loader, test_loader, unseen_loader)
    elapsed = round(time.perf_counter() - t_start, 2)
    
    print(f"\n[TC3 RESULT] Total Time: {elapsed}s")
    print(f"[TC3 RESULT] Peak VRAM: {torch.cuda.max_memory_allocated(DEVICE)/1024**2:.2f} MB")
    
    # Save Artifact
    ckpt_path = ARTIFACT_DIR / f"{DATASET}_ablation_b_tc3_best_fp32.pt"
    torch.save(result, ckpt_path)
    print(f"TC3 artifact saved to {ckpt_path}")

if __name__ == "__main__":
    main()