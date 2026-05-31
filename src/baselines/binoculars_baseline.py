"""Binoculars zero-shot AI-text detection baseline.

Binoculars (Bhagoji et al., 2024) detects AI-generated text by computing the
ratio of perplexity under an "observer" model to cross-entropy between the
observer's and a "target" model's logits.  A low ratio indicates the text is
more typical of the observer-target distributional relationship (human-written),
while a high ratio indicates a mismatch (AI-generated).

Reference
---------
Bhagoji et al., "Binoculars: Detecting AI-Generated Text Without a Reference
Corpus", 2024.

Implementation
--------------
Score each text as perplexity_obs / cross_entropy, where:

  - perplexity_obs = exp(average loss under observer model)
  - cross_entropy  = average cross-entropy between target and observer logits

A threshold is optimised on the validation split, then classification metrics
are reported on the seen test and unseen test splits.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataloader import _prepare_dataloaders_on_the_fly


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "baseline_binoculars.json"

# Observer and target model names (Falcon-7b family)
DEFAULT_OBSERVER_MODEL = "tiiuae/falcon-7b"
DEFAULT_TARGET_MODEL = "tiiuae/falcon-7b-instruct"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 256
DEFAULT_TORCH_DTYPE = torch.float16
DEFAULT_DEVICE_MAP = "auto"


@dataclass(frozen=True)
class BinocularsConfig:
    """Configuration for the Binoculars baseline."""

    observer_model_name: str = DEFAULT_OBSERVER_MODEL
    target_model_name: str = DEFAULT_TARGET_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    stride: int = DEFAULT_STRIDE
    device_map: str = DEFAULT_DEVICE_MAP
    torch_dtype: str = "float16"
    output_path: str = str(DEFAULT_OUTPUT_PATH)


class RawTextDataset(Dataset):
    """Dataset wrapper that yields raw text and labels."""

    def __init__(self, dataframe: pd.DataFrame):
        if "text" not in dataframe.columns or "label" not in dataframe.columns:
            raise ValueError("DataFrame must contain 'text' and 'label' columns")
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]
        return {
            "text": str(row["text"]),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def _memory_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "free_mb": free_bytes / (1024**2),
        "total_mb": total_bytes / (1024**2),
    }


def _print_memory(prefix: str) -> dict[str, float]:
    snapshot = _memory_snapshot()
    print(
        f"{prefix} GPU memory: allocated={snapshot['allocated_mb']:.1f} MB, "
        f"reserved={snapshot['reserved_mb']:.1f} MB, max_allocated={snapshot['max_allocated_mb']:.1f} MB, "
        f"free={snapshot['free_mb']:.1f} MB / total={snapshot['total_mb']:.1f} MB"
    )
    return snapshot


def _clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()


def _extract_dataframe(loader: DataLoader) -> pd.DataFrame:
    """Recover a DataFrame from a DataLoader's backing dataset."""
    dataset = loader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    # OnTheFlyDataset stores raw texts/labels as lists; access them directly
    if hasattr(dataset, "texts") and hasattr(dataset, "labels"):
        df = pd.DataFrame({"text": dataset.texts, "label": dataset.labels})
        if "text" in df.columns and "label" in df.columns:
            return df

    if isinstance(dataset, pd.DataFrame):
        df = dataset.reset_index(drop=True)
        if "text" in df.columns and "label" in df.columns:
            return df
    if hasattr(dataset, "to_pandas"):
        df = dataset.to_pandas().reset_index(drop=True)
        if "text" in df.columns and "label" in df.columns:
            return df
    if hasattr(dataset, "column_names"):
        df = pd.DataFrame({column: dataset[column] for column in dataset.column_names})
        if "text" in df.columns and "label" in df.columns:
            return df
    df = pd.DataFrame(dataset)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"DataFrame missing required columns. Found: {list(df.columns)}")
    return df


def _build_text_loader(dataframe: pd.DataFrame, batch_size: int) -> DataLoader:
    dataset = RawTextDataset(dataframe)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def _get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Binoculars baseline requires CUDA because Falcon-7b models require GPU inference."
        )
    return torch.device("cuda")


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(model_name: str, label: str, torch_dtype: str = "float16") -> tuple[AutoModelForCausalLM, dict[str, float]]:
    _clear_gpu_memory()
    before = _print_memory(f"Before loading {label}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, torch_dtype),
            device_map=DEFAULT_DEVICE_MAP,
            low_cpu_mem_usage=True,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            raise RuntimeError(
                f"{label} could not be loaded in the available CUDA memory. "
                "Falcon-7b requires ~16 GB VRAM per model in float16."
            ) from exc
        raise
    after = _print_memory(f"After loading {label}")
    print(
        f"GPU memory delta after loading {label}: "
        f"allocated={after['allocated_mb'] - before['allocated_mb']:.1f} MB, "
        f"reserved={after['reserved_mb'] - before['reserved_mb']:.1f} MB"
    )
    model.eval()
    return model, after


def _compute_binoculars_cross_entropy(
    observer_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token cross-entropy H(target || observer).

    For each token position:
        H(p_target, p_observer) = -sum_i p_target(i) * log(p_observer(i))

    Args:
        observer_logits: (..., vocab_size) logits from the observer model.
        target_logits:   (..., vocab_size) logits from the target model.

    Returns:
        Per-token cross-entropy values (...).
    """
    log_observer_probs = torch.log_softmax(observer_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    return -(target_probs * log_observer_probs).sum(dim=-1)


def _score_text(
    observer_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    *,
    max_length: int,
    stride: int,
) -> float:
    """Compute Binoculars score = perplexity_obs / cross_entropy."""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    seq_len = int(input_ids.size(1))
    if seq_len < 2:
        return 0.0

    device = next(observer_model.parameters()).device
    total_logprob_obs = 0.0
    total_ce = 0.0
    total_tokens = 0
    step = min(stride, max_length)

    for start in range(0, seq_len, step):
        begin_loc = max(start + step - max_length, 0)
        end_loc = min(start + step, seq_len)
        target_length = end_loc - start
        if target_length <= 0:
            break

        window = input_ids[:, begin_loc:end_loc].to(device)
        attention_mask = torch.ones_like(window)

        labels = window.clone()
        labels[:, :-target_length] = -100

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=DEFAULT_TORCH_DTYPE):
                    obs_outputs = observer_model(
                        input_ids=window, attention_mask=attention_mask, labels=labels
                    )
                    target_outputs = target_model(
                        input_ids=window, attention_mask=attention_mask
                    )
            else:
                obs_outputs = observer_model(
                    input_ids=window, attention_mask=attention_mask, labels=labels
                )
                target_outputs = target_model(
                    input_ids=window, attention_mask=attention_mask
                )

        # Accumulate observer log-probability (for perplexity)
        total_logprob_obs += float((-obs_outputs.loss * target_length).item())

        # Compute cross-entropy for the non-prefix tokens
        obs_logits = obs_outputs.logits[0, -(target_length):, :]       # (T, V)
        tgt_logits = target_outputs.logits[0, -(target_length):, :]    # (T, V)

        ce_per_token = _compute_binoculars_cross_entropy(obs_logits, tgt_logits)
        total_ce += float(ce_per_token.sum().item())
        total_tokens += target_length

        if end_loc >= seq_len:
            break

    if total_tokens == 0:
        return 0.0

    avg_ce = total_ce / total_tokens
    avg_loss_obs = total_logprob_obs / total_tokens
    perplexity_obs = np.exp(avg_loss_obs)

    # score = perplexity_obs / cross_entropy
    return perplexity_obs / avg_ce


def _score_batch(
    observer_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    *,
    max_length: int,
    stride: int,
) -> list[float]:
    return [
        _score_text(observer_model, target_model, tokenizer, text, max_length=max_length, stride=stride)
        for text in texts
    ]


def _optimize_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        threshold = float(unique_scores[0])
        preds = (scores >= threshold).astype(int)
        return {
            "threshold": threshold,
            "direction": "ge",
            "val_f1": float(f1_score(labels, preds, zero_division=0)),
        }

    candidates = np.concatenate(
        [
            np.array([unique_scores[0] - 1.0], dtype=np.float64),
            (unique_scores[:-1] + unique_scores[1:]) / 2.0,
            np.array([unique_scores[-1] + 1.0], dtype=np.float64),
        ]
    )

    best = {"threshold": float(candidates[0]), "direction": "ge", "val_f1": -1.0}
    for threshold in candidates:
        for direction in ("ge", "le"):
            if direction == "ge":
                predictions = (scores >= threshold).astype(int)
            else:
                predictions = (scores <= threshold).astype(int)
            val_f1 = float(f1_score(labels, predictions, zero_division=0))
            if val_f1 > best["val_f1"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "val_f1": val_f1,
                }
    return best


def _predict_from_scores(scores: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "ge":
        return (scores >= threshold).astype(int)
    if direction == "le":
        return (scores <= threshold).astype(int)
    raise ValueError("direction must be 'ge' or 'le'")


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, direction: str) -> dict[str, float]:
    predictions = _predict_from_scores(scores, threshold, direction)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


@torch.inference_mode()
def _score_loader(
    observer_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    loader: DataLoader,
    *,
    max_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_scores: list[float] = []
    all_labels: list[int] = []

    for batch in loader:
        texts = batch["text"]
        labels = batch["labels"]
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(labels, torch.Tensor):
            label_values = labels.detach().cpu().tolist()
        else:
            label_values = [int(label) for label in labels]

        batch_scores = _score_batch(
            observer_model, target_model, tokenizer, texts,
            max_length=max_length, stride=stride,
        )
        all_scores.extend(batch_scores)
        all_labels.extend(int(label) for label in label_values)

    return np.asarray(all_scores, dtype=np.float64), np.asarray(all_labels, dtype=np.int64)


def run_binoculars_baseline(
    config: BinocularsConfig | None = None,
) -> dict[str, Any]:
    """Run the Binoculars baseline and save results to disk."""
    if config is None:
        config = BinocularsConfig()

    observer_model_name = config.observer_model_name
    target_model_name = config.target_model_name
    batch_size = config.batch_size
    max_length = config.max_length
    stride = config.stride
    output_path = config.output_path

    device = _get_device()
    print(f"Using device: {device}")

    # ── Prepare data loaders ──────────────────────────────────────────────
    _, val_loader, seen_test_loader, unseen_loader = _prepare_dataloaders_on_the_fly(
        "raid",
        batch_size=config.batch_size,
        max_length=512,
        num_workers=0,
        seed=42,
    )

    val_text_loader = _build_text_loader(_extract_dataframe(val_loader), batch_size=batch_size)
    seen_test_text_loader = _build_text_loader(_extract_dataframe(seen_test_loader), batch_size=batch_size)
    unseen_text_loader = _build_text_loader(_extract_dataframe(unseen_loader), batch_size=batch_size)

    # ── Load tokenizer (shared between both models) ───────────────────────
    tokenizer = _load_tokenizer(observer_model_name)

    # ── Load both models ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading observer model")
    print("=" * 60)
    observer_model, after_observer = _load_model(observer_model_name, "observer", config.torch_dtype)

    print("\n" + "=" * 60)
    print("Loading target model")
    print("=" * 60)
    target_model, after_target = _load_model(target_model_name, "target", config.torch_dtype)

    # ── Score validation set → optimise threshold ─────────────────────────
    val_scores, val_labels = _score_loader(
        observer_model, target_model, tokenizer, val_text_loader,
        max_length=max_length, stride=stride,
    )
    threshold_info = _optimize_threshold(val_scores, val_labels)
    threshold = float(threshold_info["threshold"])
    direction = str(threshold_info["direction"])

    # ── Score seen and unseen test sets ────────────────────────────────────
    seen_scores, seen_labels = _score_loader(
        observer_model, target_model, tokenizer, seen_test_text_loader,
        max_length=max_length, stride=stride,
    )
    unseen_scores, unseen_labels = _score_loader(
        observer_model, target_model, tokenizer, unseen_text_loader,
        max_length=max_length, stride=stride,
    )

    seen_metrics = _metrics(seen_labels, seen_scores, threshold, direction)
    unseen_metrics = _metrics(unseen_labels, unseen_scores, threshold, direction)

    # ── Assemble results dictionary ───────────────────────────────────────
    results: dict[str, Any] = {
        "baseline": "binoculars",
        "observer_model_name": observer_model_name,
        "target_model_name": target_model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "stride": stride,
        "threshold": threshold,
        "direction": direction,
        "val": {
            "f1": float(threshold_info["val_f1"]),
            "score_mean": float(np.mean(val_scores)),
            "score_std": float(np.std(val_scores)),
        },
        "seen_test": {
            **seen_metrics,
            "score_mean": float(np.mean(seen_scores)),
            "score_std": float(np.std(seen_scores)),
            "num_samples": int(seen_scores.size),
        },
        "unseen_test": {
            **unseen_metrics,
            "score_mean": float(np.mean(unseen_scores)),
            "score_std": float(np.std(unseen_scores)),
            "num_samples": int(unseen_scores.size),
        },
        "gpu_memory": {
            "after_observer": after_observer,
            "after_target": after_target,
        },
    }

    # ── Save to disk ──────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved Binoculars baseline results to {output_path}")
    print(f"Validation threshold: {threshold:.6f} ({direction}) | val F1: {threshold_info['val_f1']:.4f}")
    print(f"Seen test: {seen_metrics}")
    print(f"Unseen test: {unseen_metrics}")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Binoculars baseline on RAID.")
    parser.add_argument("--observer-model", default=DEFAULT_OBSERVER_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = BinocularsConfig(
        observer_model_name=args.observer_model,
        target_model_name=args.target_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        output_path=args.output_path,
    )
    return run_binoculars_baseline(config)


if __name__ == "__main__":
    main()