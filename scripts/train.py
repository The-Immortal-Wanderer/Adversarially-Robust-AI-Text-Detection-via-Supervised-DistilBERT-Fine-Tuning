"""
scripts/train.py — Unified config-driven training entry point.

Usage
-----
    python -m scripts.train --dataset raid
    python -m scripts.train --dataset detectrl --ablation ablation_b

Replaces the 3 legacy training scripts with a single config-driven entry
point.  All model / data / training logic lives in ``src/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.config import load_config
from src.data.dataloader import (
    _prepare_dataloaders_cached,
    _prepare_dataloaders_on_the_fly,
)
from src.evaluation.metrics import compute_rrd
from src.models import (
    DistilBertClassifier,
    count_trainable_parameters,
)
from src.training.trainer import (
    install_defensive_timer,
    make_stop_event,
    seed_everything,
    train_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified DistilBERT training entry point.")
    parser.add_argument(
        "--dataset",
        choices=["raid", "detectrl"],
        default="raid",
        help="Dataset to train on (default: raid)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help="Specific ablation to train (default: all four)",
    )
    args, _ = parser.parse_known_args()

    cfg = load_config()
    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(cfg.paths.artifact_dir)
    processed_dir = ROOT_DIR / "data" / "processed"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.training.seed)
    STOP_EVENT = make_stop_event()
    timer = install_defensive_timer(8.5, STOP_EVENT)

    print(f"Project root   : {ROOT_DIR}", flush=True)
    print(f"Processed data : {processed_dir}", flush=True)
    print(f"Artifacts      : {artifact_dir}", flush=True)
    print(f"Device         : {device}", flush=True)
    print(f"Dataset        : {dataset_name}", flush=True)

    ablations = [args.ablation] if args.ablation else ["baseline1", "ablation_a", "ablation_b", "ablation_c"]

    for name in ablations:
        ablation_cfg = getattr(cfg.ablations, name)
        cfg_m = {"head_type": ablation_cfg.head_type, "freeze_layers": ablation_cfg.freeze_layers}
        m = DistilBertClassifier(**cfg_m)
        trainable = count_trainable_parameters(m)
        total = sum(p.numel() for p in m.parameters())
        print(f"{name:12s} -> {cfg_m} | trainable={trainable:,} / total={total:,}", flush=True)
        del m

    tokenization_mode = cfg.data.tokenization_mode if hasattr(cfg.data, "tokenization_mode") else "on_the_fly"
    if tokenization_mode == "cached":
        train_loader, val_loader, test_loader, unseen_loader = _prepare_dataloaders_cached(
            dataset_name,
            processed_dir=processed_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            prefetch_factor=cfg.training.prefetch_factor,
            max_length=cfg.data.max_length,
            seed=cfg.training.seed,
            samples_per_class=cfg.data.samples_per_class,
            unseen_cap=cfg.data.unseen_cap,
        )
    else:
        train_loader, val_loader, test_loader, unseen_loader = _prepare_dataloaders_on_the_fly(
            dataset_name,
            processed_dir=processed_dir,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            prefetch_factor=cfg.training.prefetch_factor,
            max_length=cfg.data.max_length,
            seed=cfg.training.seed,
            samples_per_class=cfg.data.samples_per_class,
            unseen_cap=cfg.data.unseen_cap,
        )

    print(f"\ntrain_loader : {len(train_loader)} batches", flush=True)
    print(f"val_loader   : {len(val_loader)} batches", flush=True)
    print(f"test_loader  : {len(test_loader)} batches", flush=True)
    print(f"unseen_loader: {len(unseen_loader)} batches", flush=True)

    results: dict[str, Any] = {}
    training_times: dict[str, float] = {}

    for ablation_name in ablations:
        if STOP_EVENT.is_set():
            print("Defensive timer triggered. Stopping before next ablation.", flush=True)
            break

        print(f"\n{'=' * 60}", flush=True)
        print(f"Starting ablation : {ablation_name}", flush=True)
        ablation_cfg = getattr(cfg.ablations, ablation_name)
        print(f"Config            : head_type={ablation_cfg.head_type}, freeze_layers={ablation_cfg.freeze_layers}", flush=True)
        print(f"{'=' * 60}", flush=True)

        t_start = time.perf_counter()
        results[ablation_name] = train_ablation(
            ablation_name,
            ablation_cfg.head_type,
            ablation_cfg.freeze_layers,
            train_loader,
            val_loader,
            test_loader,
            unseen_loader,
            device,
            epochs=cfg.training.epochs,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            grad_clip_norm=cfg.training.grad_clip_norm,
        )
        elapsed = round(time.perf_counter() - t_start, 2)
        training_times[ablation_name] = elapsed
        print(f"\n[{ablation_name}] total training time: {elapsed}s ({elapsed / 3600:.2f} hrs)", flush=True)

    timer.cancel()

    if not results:
        print("No results to save.", flush=True)
        return

    summary_rows = []
    for name, result in results.items():
        row = {
            "ablation_name": name,
            "best_val_f1": result["best_val_f1"],
            **{f"test_{k}": v for k, v in result["test_metrics"].items()},
            **{f"unseen_{k}": v for k, v in result["unseen_metrics"].items()},
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60, flush=True)
    print("FINAL SUMMARY", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    for _, row in summary_df.iterrows():
        seen_f1 = row.get("test_f1_macro", 0.0)
        unseen_f1 = row.get("unseen_f1_macro", 0.0)
        rrd = compute_rrd(seen_f1, unseen_f1)
        print(f"  {row['ablation_name']:12s}: seen_F1={seen_f1:.4f}  unseen_F1={unseen_f1:.4f}  RRD={rrd:.2f}%", flush=True)

    best_name = summary_df.sort_values("best_val_f1", ascending=False).iloc[0]["ablation_name"]

    for name, result in results.items():
        ckpt_path = artifact_dir / f"{dataset_name}_{name}_best.pt"
        torch.save(result, ckpt_path)
        print(f"Saved: {ckpt_path}", flush=True)

    times_path = artifact_dir / f"{dataset_name}_training_times.json"
    with open(times_path, "w", encoding="utf-8") as f:
        json.dump(training_times, f, indent=2)

    best_config_path = artifact_dir / f"{dataset_name}_best_config.json"
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_ablation_name": best_name, "config": results[best_name]["config"], "best_val_f1": float(results[best_name]["best_val_f1"])},
            f,
            indent=2,
        )

    tokenizer_save = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer_save.save_pretrained(artifact_dir / dataset_name)

    summary_csv = artifact_dir / f"{dataset_name}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nAll artifacts saved to {artifact_dir}", flush=True)
    print(f"Best ablation: {best_name}", flush=True)


if __name__ == "__main__":
    main()
