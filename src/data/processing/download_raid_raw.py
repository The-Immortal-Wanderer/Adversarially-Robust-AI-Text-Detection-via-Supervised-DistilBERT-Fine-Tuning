"""
Download the full RAID dataset from Hugging Face and save it as a parquet file.

This script does not stream or filter the dataset. It downloads the full split,
then writes it to a single parquet file on disk.

Run:
    pip install datasets pyarrow zstandard
    python src/data/processing/download_raid_raw.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_full_raid(hf_repo: str = "liamdugan/raid", output_path: Path | None = None) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Run: pip install datasets pyarrow zstandard") from exc

    root_dir = Path(__file__).resolve().parents[3]  # from src/data/processing/ → project root
    if output_path is None:
        output_path = root_dir / "data" / "raw" / "raid" / "raid_full.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading full RAID dataset from Hugging Face: {hf_repo}", flush=True)
    dataset = load_dataset(hf_repo, split="train", trust_remote_code=True)
    print(f"Loaded {len(dataset):,} rows", flush=True)
    print(f"Writing parquet to {output_path}", flush=True)

    dataset.to_parquet(str(output_path))

    print(f"Saved parquet: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the full RAID dataset and save it as parquet")
    parser.add_argument("--hf-repo", default="liamdugan/raid", help="Hugging Face repo for RAID")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet file path (default: raw/raid/raid_full.parquet)",
    )
    args = parser.parse_args()

    download_full_raid(hf_repo=args.hf_repo, output_path=args.output)


if __name__ == "__main__":
    main()