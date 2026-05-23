"""
scripts/evaluate.py — Unified evaluation entry point for DistilBERT checkpoint evaluation.

Loads a trained checkpoint, runs evaluation on both seen (test) and unseen held-out
data, and computes all metrics: accuracy, precision, recall, F1, AUROC, RRD,
Expected Calibration Error (ECE), and low-FPR TPR at 1%, 5%, 10% thresholds.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint artifacts/distilbert_detector/ablation_b_best.pt
    python scripts/evaluate.py --dataset raid --checkpoint ablation_b_best.pt --output results.json
    python scripts/evaluate.py --dataset detectrl --config override.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path (editable install from src/ isn't
# always resolved when running the script directly).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration — lazily loaded from YAML with hardcoded fallback defaults
# ---------------------------------------------------------------------------

from src.training.trainer import seed_everything


# Module-level defaults (overridden by _init_config on first call)
ARTIFACT_DIR: str = "artifacts/distilbert_detector"
CHECKPOINT_FALLBACKS: list[str] = [
    "artifacts/distilbert_detector/baseline1_best.pt",
    "artifacts/distilbert_detector/ablation_b_best.pt",
    "artifacts/distilbert_detector/ablation_c_best.pt",
]
BATCH_SIZE: int = 32
MAX_LENGTH: int = 256
NUM_WORKERS: int = 4
SEED: int = 42
DEVICE: str = "cpu"
_CFG_INITIALIZED: bool = False


def _init_config() -> None:
    """Lazy-load configuration from default YAML."""
    global ARTIFACT_DIR, CHECKPOINT_FALLBACKS, BATCH_SIZE, MAX_LENGTH, NUM_WORKERS, SEED, DEVICE, _CFG_INITIALIZED
    if _CFG_INITIALIZED:
        return
    try:
        from src.config import load_config, TrainingConfig

        _cfg: TrainingConfig = load_config()
        ARTIFACT_DIR = str(_cfg.paths.artifact_dir)
        CHECKPOINT_FALLBACKS = list(_cfg.paths.checkpoint_fallbacks)
        BATCH_SIZE = _cfg.training.batch_size
        MAX_LENGTH = _cfg.data.max_length
        NUM_WORKERS = _cfg.training.num_workers
        SEED = _cfg.training.seed
    except Exception as exc:
        warnings.warn(f"Config load failed ({exc}); using hardcoded fallback defaults")
        # Module-level defaults already set above
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(SEED)
    _CFG_INITIALIZED = True

# ---------------------------------------------------------------------------
# Imports from src package
# ---------------------------------------------------------------------------

from src.data.dataloader import _prepare_dataloaders_on_the_fly
from src.evaluation import compute_ece, compute_low_fpr_tpr, compute_metrics, compute_rrd
from src.models import DistilBertClassifier, count_trainable_parameters


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(path: str | None = None) -> dict[str, Any]:
    """
    Load a trained checkpoint from disk with unified fallback path logic.

    Path resolution order:
        1. Explicit ``path`` argument (if provided and file exists)
        2. ``ARTIFACT_DIR / path`` (if path is a bare filename)
        3. Entries in :data:`CHECKPOINT_FALLBACKS` resolved against :data:`ARTIFACT_DIR`

    Args:
        path: Explicit checkpoint path or filename (optional).

    Returns:
        Dictionary containing at minimum ``model_state_dict`` and ``config``.
        The config dict must have ``head_type`` (``"single"`` or ``"deep"``)
        and ``freeze_layers`` (int 0–6).

    Raises:
        FileNotFoundError: If no checkpoint can be found at any resolved path.
    """
    artifact_dir = Path(ARTIFACT_DIR)
    candidates: list[Path] = []

    # 1. Explicit path
    if path is not None:
        p = Path(path)
        if p.is_absolute():
            candidates.append(p)
        else:
            # Try relative to CWD first, then artifact dir
            candidates.append(Path.cwd() / p)
            candidates.append(artifact_dir / p)

    # 2. Fallbacks from config — these are full project-relative paths
    #    (e.g. "artifacts/distilbert_detector/ablation_b_best.pt")
    for fallback in CHECKPOINT_FALLBACKS:
        fb = Path(fallback)
        if fb.is_absolute():
            candidates.append(fb)
        else:
            # Try as-is from CWD first, then relative to artifact dir
            candidates.append(Path.cwd() / fb)
            candidates.append(artifact_dir / fb.name)

    # Try each candidate
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            print(f"Loading checkpoint: {resolved}")
            return torch.load(str(resolved), map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        "No checkpoint found. Tried:\n" + "\n".join(f"  - {p}" for p in seen)
    )


# ---------------------------------------------------------------------------
# Evaluation runner (mirrors run_epoch from training scripts)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """
    Run evaluation on a single DataLoader.

    Mirrors the ``run_epoch`` pattern from training scripts but only in eval
    mode (no optimizer, no gradient computation).

    Args:
        model: The trained model in eval mode.
        loader: DataLoader for evaluation data.
        device: Device to run inference on.

    Returns:
        Dict with keys: ``labels``, ``preds``, ``probs``, ``loss``.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.detach().cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0

    return {
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "loss": avg_loss,
    }


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint_path: str | None = None,
    dataset: str = "raid",
    config_override: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Load a trained checkpoint and evaluate on held-out data.

    Loads both seen (test split) and unseen data, computes all classification
    metrics, calibration error, and low-FPR sensitivity.

    Args:
        checkpoint_path: Path or filename of checkpoint (optional — uses fallback).
        dataset: Dataset name (``"raid"`` or ``"detectrl"``). Defaults to ``"raid"``.
        config_override: Optional dict overriding model config from checkpoint.

    Returns:
        Comprehensive metrics dict with keys:
            - ``ablation_name``, ``config``: Model identification.
            - ``trainable_params``, ``total_params``: Parameter counts.
            - ``test_metrics``: Metrics on seen test split.
            - ``unseen_metrics``: Metrics on unseen held-out split.
            - ``rrd``: Relative Robustness Degradation (percentage).
            - ``ece``: Expected Calibration Error on seen split.
            - ``unseen_ece``: Expected Calibration Error on unseen split.
            - ``low_fpr_tpr_test``: TPR at 1%/5%/10% FPR on seen split.
            - ``low_fpr_tpr_unseen``: TPR at 1%/5%/10% FPR on unseen split.
    """
    # ── Init config (sets DEVICE, BATCH_SIZE, etc.) ──────────────────────
    _init_config()

    # Allow caller to override seed (e.g. from g0 decision gate or --seed CLI)
    if seed is not None:
        global SEED
        SEED = seed
        seed_everything(SEED)

    device = torch.device(DEVICE)

    # ── Load checkpoint ──────────────────────────────────────────────────
    checkpoint = load_checkpoint(checkpoint_path)
    ckpt_config: dict[str, Any] = dict(checkpoint.get("config", {}))
    ablation_name: str = str(checkpoint.get("ablation_name", "unknown"))

    if config_override:
        ckpt_config.update(config_override)

    # ── Build model ──────────────────────────────────────────────────────
    model = DistilBertClassifier(
        head_type=str(ckpt_config.get("head_type", "single")),
        freeze_layers=int(ckpt_config.get("freeze_layers", 0)),
    )

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        # Attempt to treat the entire checkpoint as a state dict
        state_dict = {
            k: v
            for k, v in checkpoint.items()
            if k.startswith("distilbert.") or k.startswith("classifier.")
        }
    if not state_dict:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict', 'state_dict', "
            "or recognisable model parameter keys."
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    trainable = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Ablation      : {ablation_name}")
    print(f"  Config        : {ckpt_config}")
    print(f"  Trainable     : {trainable:,} / {total_params:,} total")
    print(f"  Device        : {device}")

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\nLoading data (dataset={dataset})...")
    _, _, test_loader, unseen_loader = _prepare_dataloaders_on_the_fly(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_length=MAX_LENGTH,
        seed=SEED,
    )

    # ── Run evaluation ───────────────────────────────────────────────────
    print("Evaluating on seen (test) split...")
    test_results = run_eval(model, test_loader, device)

    print("Evaluating on unseen split...")
    unseen_results = run_eval(model, unseen_loader, device)

    # ── Compute standard metrics ─────────────────────────────────────────
    test_metrics = compute_metrics(
        test_results["labels"],
        test_results["preds"],
        test_results["probs"],
    )
    test_metrics["loss"] = test_results["loss"]

    unseen_metrics = compute_metrics(
        unseen_results["labels"],
        unseen_results["preds"],
        unseen_results["probs"],
    )
    unseen_metrics["loss"] = unseen_results["loss"]

    # ── RRD ──────────────────────────────────────────────────────────────
    seen_f1 = float(test_metrics.get("f1_macro", float("nan")))
    unseen_f1 = float(unseen_metrics.get("f1_macro", float("nan")))
    rrd = compute_rrd(seen_f1, unseen_f1)

    # ── ECE ──────────────────────────────────────────────────────────────
    ece = compute_ece(
        test_results["labels"],
        test_results["probs"],
    )
    unseen_ece = compute_ece(
        unseen_results["labels"],
        unseen_results["probs"],
    )

    # ── Low-FPR TPR ──────────────────────────────────────────────────────
    low_fpr_tpr_test = compute_low_fpr_tpr(
        test_results["labels"],
        test_results["probs"],
    )
    low_fpr_tpr_unseen = compute_low_fpr_tpr(
        unseen_results["labels"],
        unseen_results["probs"],
    )

    # ── Assemble result dict ─────────────────────────────────────────────
    result: dict[str, Any] = {
        "ablation_name": ablation_name,
        "config": ckpt_config,
        "dataset": dataset,
        "trainable_params": trainable,
        "total_params": total_params,
        "test_metrics": test_metrics,
        "unseen_metrics": unseen_metrics,
        "rrd": rrd,
        "ece": ece,
        "unseen_ece": unseen_ece,
        "low_fpr_tpr_test": low_fpr_tpr_test,
        "low_fpr_tpr_unseen": low_fpr_tpr_unseen,
    }

    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(result: dict[str, Any]) -> None:
    """Print a formatted evaluation summary table."""

    def _fmt(val: object, precision: int = 4) -> str:
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)

    print("\n" + "=" * 72)
    print("                           EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Ablation          : {result['ablation_name']}")
    print(f"  Config            : {result['config']}")
    print(f"  Dataset           : {result['dataset']}")
    print(f"  Trainable params  : {result['trainable_params']:,}")
    print(f"  Total params      : {result['total_params']:,}")
    print("-" * 72)

    tm = result["test_metrics"]
    um = result["unseen_metrics"]

    rows: list[tuple[str, Any, Any]] = [
        ("Accuracy", tm.get("accuracy", "N/A"), um.get("accuracy", "N/A")),
        ("Precision", tm.get("precision", "N/A"), um.get("precision", "N/A")),
        ("Recall", tm.get("recall", "N/A"), um.get("recall", "N/A")),
        ("F1 (macro)", tm.get("f1_macro", "N/A"), um.get("f1_macro", "N/A")),
        ("ROC-AUC", tm.get("roc_auc", "N/A"), um.get("roc_auc", "N/A")),
        ("Loss", tm.get("loss", "N/A"), um.get("loss", "N/A")),
    ]

    print("  {:<20s} {:>12s} {:>12s}".format("Metric", "Seen", "Unseen"))
    print("  " + "-" * 20 + " " + "-" * 12 + " " + "-" * 12)
    for label, seen_val, unseen_val in rows:
        print(f"  {label:<20s} {_fmt(seen_val):>12s} {_fmt(unseen_val):>12s}")

    print("-" * 72)

    rrd = result["rrd"]
    print(f"  RRD               : {_fmt(rrd, 2)}")
    print(f"  ECE (seen)        : {_fmt(result['ece'])}")
    print(f"  ECE (unseen)      : {_fmt(result.get('unseen_ece', float('nan')))}")

    print("-" * 72)
    print("  Low-FPR TPR (seen split):")
    for key, val in result.get("low_fpr_tpr_test", {}).items():
        label = key.replace("_", " ").replace("pct", "%")
        print(f"    {label:<20s} : {_fmt(val)}")

    print("  Low-FPR TPR (unseen split):")
    for key, val in result.get("low_fpr_tpr_unseen", {}).items():
        label = key.replace("_", " ").replace("pct", "%")
        print(f"    {label:<20s} : {_fmt(val)}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for command-line evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DistilBERT checkpoint on held-out data.",
    )
    parser.add_argument(
        "--dataset",
        choices=["raid", "detectrl"],
        default="raid",
        help="Dataset to use (default: raid).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path or filename of checkpoint .pt file. If omitted, uses fallback search.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON file overriding model config (head_type, freeze_layers).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config (default: use config value).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save metrics as JSON (e.g. results.json).",
    )
    args = parser.parse_args()

    config_override: dict[str, Any] | None = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            config_override = json.load(f)

    result = evaluate(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        config_override=config_override,
        seed=args.seed,
    )

    print_summary(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
