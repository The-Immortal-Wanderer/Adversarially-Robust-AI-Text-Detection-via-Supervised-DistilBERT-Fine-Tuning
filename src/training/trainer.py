from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


@dataclass
class TrainerConfig:
    lr: float = 2e-5
    epochs: int = 10
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 2
    ablation_name: str = "baseline1"
    patience: int = 3
    checkpoint_dir: str = "checkpoints"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict[str, Any],
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = TrainerConfig(**config)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.config.lr,
        )

        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.config.gradient_accumulation_steps))
        total_training_steps = max(1, steps_per_epoch * self.config.epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_training_steps,
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_f1 = -1.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _step_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        predictions = torch.argmax(logits, dim=-1)
        return {
            "accuracy": accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy()),
            "f1_macro": f1_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), average="macro"),
        }

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_samples = 0
        all_logits = []
        all_labels = []

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader, start=1):
            batch = self._move_batch(batch)
            labels = batch["labels"]

            with autocast(enabled=self.device.type == "cuda"):
                logits = self.model(batch["input_ids"], batch["attention_mask"])
                loss = self.criterion(logits, labels)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * self.config.gradient_accumulation_steps * batch_size
            running_samples += batch_size
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())

        if len(self.train_loader) % self.config.gradient_accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        if running_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = self._step_metrics(logits, labels)
        metrics["loss"] = running_loss / running_samples
        return metrics

    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        running_samples = 0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch(batch)
                labels = batch["labels"]
                with autocast(enabled=self.device.type == "cuda"):
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_samples += batch_size
                all_logits.append(logits.detach().float().cpu())
                all_labels.append(labels.detach().cpu())

        if running_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = self._step_metrics(logits, labels)
        metrics["loss"] = running_loss / running_samples
        return metrics

    def _save_checkpoint(self, val_f1: float) -> Path:
        checkpoint_path = self.checkpoint_dir / f"{self.config.ablation_name}_best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "val_f1": val_f1,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def train(self) -> dict[str, Any]:
        history = []
        best_checkpoint = None

        for epoch in range(1, self.config.epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            train_metrics = self._train_one_epoch()
            val_metrics = self._evaluate()
            peak_vram_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
                if self.device.type == "cuda"
                else 0.0
            )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "val_f1": float(val_metrics["f1_macro"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "peak_vram_mb": float(peak_vram_mb),
            }
            history.append(epoch_metrics)
            print(
                f"[{self.config.ablation_name}] epoch={epoch} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"val_loss={epoch_metrics['val_loss']:.4f} "
                f"val_f1={epoch_metrics['val_f1']:.4f} "
                f"val_accuracy={epoch_metrics['val_accuracy']:.4f} "
                f"peak_vram_mb={epoch_metrics['peak_vram_mb']:.1f}"
            )

            if epoch_metrics["val_f1"] > self.best_val_f1:
                self.best_val_f1 = epoch_metrics["val_f1"]
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                best_checkpoint = self._save_checkpoint(self.best_val_f1)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.config.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        return {
            "ablation_name": self.config.ablation_name,
            "best_epoch": self.best_epoch,
            "best_val_f1": float(self.best_val_f1),
            "checkpoint_path": str(best_checkpoint) if best_checkpoint is not None else None,
            "history": history,
        }
