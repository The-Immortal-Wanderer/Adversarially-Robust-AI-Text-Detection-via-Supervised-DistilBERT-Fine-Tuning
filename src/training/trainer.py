"""
Training loop utilities for DetectRL experiments.

Provides seed_everything, run_epoch, and train_ablation for
DistilBERT ablation training and evaluation.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models import DistilBertClassifier


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    # Required for deterministic CuBLAS on CUDA >= 10.2
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Cover non-CuDNN ops (scatter_add, index_add, bincount, …)
    torch.use_deterministic_algorithms(True)

    torch.Generator().manual_seed(seed)


# ── Defensive timer (Kaggle session watchdog) ──────────────────────


def make_stop_event() -> threading.Event:
    """Create a new stop event instance for use with `install_defensive_timer`."""
    return threading.Event()


def _timeout_handler(stop_event: threading.Event) -> None:
    """Set the stop event to signal training loops to halt."""
    stop_event.set()
    print("[TIMER] Kaggle session limit approaching — stopping training.", flush=True)


def install_defensive_timer(limit_hours: float, stop_event: threading.Event) -> threading.Timer:
    """Install a background timer that sets *stop_event* after *limit_hours*.

    Call ``timer.cancel()`` if training finishes before the deadline.
    Returns the ``threading.Timer`` handle for optional cancellation.
    """
    limit_seconds = max(1.0, limit_hours * 3600 - 60)  # 60s safety margin
    timer = threading.Timer(interval=limit_seconds, function=_timeout_handler, args=[stop_event])
    timer.daemon = True
    timer.start()
    return timer


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a batch of tensors to the specified device."""
    return {key: value.to(device) for key, value in batch.items()}


def _compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute classification metrics from model logits and ground-truth labels.

    Returns dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    grad_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training or evaluation epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader yielding dicts with 'input_ids', 'attention_mask', 'labels'.
        optimizer: Optimizer for training; ``None`` for evaluation-only.
        device: Target device.  Defaults to CUDA if available else CPU.
        grad_clip_norm: Max gradient norm for clipping (training only).

    Returns:
        dict with keys: accuracy, precision, recall, f1, roc_auc, loss.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = _compute_metrics(logits, labels)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def train_ablation(
    ablation_name: str,
    head_type: str,
    freeze_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    unseen_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
) -> dict[str, Any]:
    """Train a single ablation configuration: creates model, trains, evaluates."""
    config = {"head_type": head_type, "freeze_layers": freeze_layers}
    model = DistilBertClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_f1 = -1.0
    best_state: dict | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, device, grad_clip_norm)
        val_metrics = run_epoch(model, val_loader, device=device, grad_clip_norm=grad_clip_norm)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        epoch_time = round(time.perf_counter() - epoch_start, 2)
        print(f"[{ablation_name}] epoch={epoch} train={train_metrics} val={val_metrics}", flush=True)
        print(f"[{ablation_name}] epoch={epoch} time={epoch_time}s", flush=True)

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "ablation_name": ablation_name,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

    assert best_state is not None
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, device=device, grad_clip_norm=grad_clip_norm)
    unseen_metrics = run_epoch(model, unseen_loader, device=device, grad_clip_norm=grad_clip_norm)

    return {
        "ablation_name": ablation_name,
        "config": config,
        "history": history,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "unseen_metrics": unseen_metrics,
        "model_state_dict": model.state_dict(),
    }
