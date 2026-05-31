"""
scripts/cache_pretrained.py — Pre-cache pretrained model weights to Kaggle Dataset

One-time setup: run this from a Kaggle notebook to download DistilBERT from
HuggingFace Hub and upload it as a Kaggle Dataset. Subsequent Kaggle sessions
load from the Dataset instead of hitting HF Hub (avoids 100 req/hr rate limits).

Usage (Kaggle notebook cell):
    import sys; sys.path.insert(0, "/kaggle/working/ann_project")
    from scripts.cache_pretrained import cache_and_upload
    cache_and_upload()

Local usage (dry-run, no upload):
    python scripts/cache_pretrained.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _is_kaggle() -> bool:
    """Detect Kaggle kernel environment."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


# ── Model registry ─────────────────────────────────────────────────────────
# (model_id, dataset_owner, dataset_name)
PRETRAINED_REGISTRY: list[dict[str, str]] = [
    {
        "hf_id": "distilbert-base-uncased",
        "dataset_owner": "tetsujin007",
        "dataset_name": "distilbert-base-uncased",
    },
]


def download_model(hf_id: str, cache_dir: Path) -> Path:
    """Download a model from HuggingFace Hub to a local cache directory."""
    print(f"  Downloading {hf_id} from HuggingFace Hub ...", flush=True)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Download config, tokenizer, and model weights
    tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=str(cache_dir))
    tokenizer.save_pretrained(str(cache_dir / hf_id.replace("/", "_") / "tokenizer"))

    # For classifier tasks we download the base model (no head)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        cache_dir=str(cache_dir),
        num_labels=2,
    )
    model.save_pretrained(str(cache_dir / hf_id.replace("/", "_") / "model"))

    print(f"  Done. Model cached at {cache_dir}", flush=True)
    return cache_dir


def upload_to_kaggle(local_path: Path, owner: str, dataset_name: str) -> None:
    """Upload a local directory to a Kaggle Dataset using kagglehub."""
    print(f"  Uploading {local_path} to kaggle datasets {owner}/{dataset_name} ...", flush=True)

    try:
        import kagglehub

        handle = kagglehub.dataset_upload(
            f"{owner}/{dataset_name}",
            str(local_path),
        )
        print(f"  Uploaded. Dataset handle: {handle}", flush=True)
    except Exception as e:
        print(f"  Upload failed: {e}", flush=True)
        print("  You can manually upload the cached directory:", local_path, flush=True)
        print("  Using Kaggle UI: Create Dataset -> Upload -> select this directory", flush=True)


def cache_and_upload(dry_run: bool = False) -> None:
    """Main entry point: download and optionally upload all cached models."""
    print("=" * 60, flush=True)
    print("  Pretrained Model Cache Script", flush=True)
    print("=" * 60, flush=True)

    kaggle = _is_kaggle()

    # Use a temp directory for downloads
    with tempfile.TemporaryDirectory(prefix="model_cache_") as tmp:
        cache_dir = Path(tmp)

        for entry in PRETRAINED_REGISTRY:
            hf_id = entry["hf_id"]
            owner = entry["dataset_owner"]
            dataset = entry["dataset_name"]

            print(f"\n  Model: {hf_id}", flush=True)
            print(f"  Target Kaggle Dataset: {owner}/{dataset}", flush=True)

            if dry_run:
                print("  [DRY-RUN] Skipping download and upload.", flush=True)
                continue

            download_model(hf_id, cache_dir)

            if kaggle:
                upload_to_kaggle(cache_dir, owner, dataset)
            else:
                print(f"\n  Local environment detected. Model cached at: {cache_dir}", flush=True)
                print(f"  To upload, copy this directory to Kaggle or re-run from a Kaggle notebook.", flush=True)

    print("\n  Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-cache pretrained model weights to Kaggle Dataset.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing.")
    args = parser.parse_args()

    cache_and_upload(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
