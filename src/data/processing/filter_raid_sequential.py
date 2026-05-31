"""
Process a downloaded RAID parquet file and produce the filtered training/eval pools.

This script mirrors the filtering logic from the streaming RAID downloader, but it
starts from a locally downloaded parquet file and reads it in batches so it does
not try to load the full raw dataset into memory at once.

Default input:
    data/raw/raid_full.parquet

Outputs:
    data/processed/raid_train_pool.parquet
    data/processed/raid_test_unseen.parquet

Run:
    python src/data/processing/filter_raid_sequential.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError("Run: pip install pyarrow pandas") from exc

ROOT_DIR = Path(__file__).resolve().parents[3]  # from src/data/processing/ → project root
RAW_DIR = ROOT_DIR / "data" / "raw" / "raid"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

_PARQUET_READ_BATCH_SIZE: int = 50_000  # H-009: named constant for sequential parquet batching
RANDOM_SEED: int = 42  # Seed for reproducible sampling in build pools

SEEN_GENERATORS: set[str] = {
    "mpt",            # mpt and mpt-chat (open source)
    "mistral",        # mistral and mistral-chat (open source)
    "llama-chat",     # llama-chat (open source)
    "gpt2",           # gpt2 (oldest, most detectable)
}

UNSEEN_GENERATORS: set[str] = {
    "gpt3",           # gpt3 — stronger, held out
    "gpt4",           # gpt4 — strongest, held out
    "chatgpt",        # chatgpt — held out
    "cohere",         # cohere and cohere-chat — held out
    "cohere-chat",
}

KEEP_DOMAINS: set[str] = {
    "news",
    "reddit",
    "recipes",
    "poetry",
    "abstracts",
}

TARGET_TRAIN_PER_CLASS = 60_000
TARGET_UNSEEN_PER_CLASS = 10_000


def _label_from_model(model_value: object) -> int:
    """
    RAID has no label column.
    Label is derived from the model column:
      model == 'human' → label 0 (human)
      anything else    → label 1 (AI)
    """
    return 0 if str(model_value).strip().lower() == "human" else 1


def _load_raw_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw RAID parquet not found: {path}")
    print(f"Streaming raw RAID parquet from {path}", flush=True)
    parquet_file = pq.ParquetFile(path)

    rows: list[dict[str, object]] = []
    total_rows = 0

    for batch in parquet_file.iter_batches(batch_size=_PARQUET_READ_BATCH_SIZE):
        batch_df = batch.to_pandas()
        total_rows += len(batch_df)

        for _, row in batch_df.iterrows():
            model_name = str(row.get("model", "")).strip().lower()
            label      = _label_from_model(model_name)
            domain     = str(row.get("domain", "")).strip().lower()
            attack     = str(row.get("attack", "none")).strip().lower()
            text       = str(row.get("generation", "")).strip()

            if not text or domain not in KEEP_DOMAINS:
                continue

            if label == 0:
                rows.append({
                    "text":        text,
                    "label":       0,
                    "domain":      domain,
                    "generator":   "human",
                    "attack_type": "human",
                })
                continue

            # AI row — match against known generators
            generator_key = None
            for generator_name in sorted(SEEN_GENERATORS | UNSEEN_GENERATORS, key=len, reverse=True):
                if generator_name in model_name:
                    generator_key = generator_name
                    break
            if generator_key is None:
                continue

            rows.append({
                "text":        text,
                "label":       1,
                "domain":      domain,
                "generator":   generator_key,
                "attack_type": attack if attack != "none" else "direct_prompt",
            })

    filtered_df = pd.DataFrame(rows)
    print(f"Loaded and filtered {total_rows:,} raw rows", flush=True)
    print(f"Filtered rows kept: {len(filtered_df):,}", flush=True)
    return filtered_df


def _build_train_pool(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60, flush=True)
    print("Building TRAIN pool (seen generators)", flush=True)
    print("=" * 60, flush=True)

    human_df = df[df["label"] == 0].copy()
    seen_ai_df = df[(df["label"] == 1) & (df["generator"].isin(SEEN_GENERATORS))].copy()

    print(f"Human samples available : {len(human_df):,}", flush=True)
    print(f"Seen-AI samples available: {len(seen_ai_df):,}", flush=True)
    print("\nGenerator breakdown (seen AI):", flush=True)
    print(seen_ai_df["generator"].value_counts().to_string(), flush=True)
    print("\nDomain breakdown (seen AI):", flush=True)
    print(seen_ai_df["domain"].value_counts().to_string(), flush=True)

    n_human = min(TARGET_TRAIN_PER_CLASS, len(human_df))
    n_ai = min(TARGET_TRAIN_PER_CLASS, len(seen_ai_df))
    n = min(n_human, n_ai)
    if n < 1:
        raise ValueError(
            f"_build_train_pool: capped to 0 samples "
            f"(human={n_human}, ai={n_ai})"
        )

    human_sample = human_df.sample(n=n, random_state=RANDOM_SEED)
    ai_sample = seen_ai_df.sample(n=n, random_state=RANDOM_SEED)

    pool = pd.concat([human_sample, ai_sample], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    )

    print(f"\nFinal TRAIN pool: {len(pool):,} rows", flush=True)
    print(pool["label"].value_counts().to_string(), flush=True)
    return pool


def _build_unseen_pool(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60, flush=True)
    print("Building UNSEEN pool (unseen generators)", flush=True)
    print("=" * 60, flush=True)

    all_human = df[df["label"] == 0].copy()
    unseen_ai_df = df[(df["label"] == 1) & (df["generator"].isin(UNSEEN_GENERATORS))].copy()

    print(f"Human samples available   : {len(all_human):,}", flush=True)
    print(f"Unseen-AI samples available: {len(unseen_ai_df):,}", flush=True)
    print("\nGenerator breakdown (unseen AI):", flush=True)
    print(unseen_ai_df["generator"].value_counts().to_string(), flush=True)

    n_ai = min(TARGET_UNSEEN_PER_CLASS, len(unseen_ai_df))
    n_human = min(TARGET_UNSEEN_PER_CLASS, len(all_human))
    n = min(n_human, n_ai)
    if n < 1:
        raise ValueError(
            f"_build_unseen_pool: capped to 0 samples "
            f"(human={n_human}, ai={n_ai})"
        )

    ai_sample = unseen_ai_df.sample(n=n, random_state=RANDOM_SEED)
    human_sample = all_human.sample(n=n, random_state=RANDOM_SEED + 1)

    pool = pd.concat([human_sample, ai_sample], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED
    )
    # attack_type preserved from _filter_row — do NOT overwrite with generator
    # pool["attack_type"] = pool["generator"].where(pool["label"] == 1, other="human")

    print(f"\nFinal UNSEEN pool: {len(pool):,} rows", flush=True)
    print(pool["label"].value_counts().to_string(), flush=True)
    print(pool["attack_type"].value_counts().to_string(), flush=True)
    return pool


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Filter a downloaded RAID parquet into processed pools")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "raid_full.parquet",
        help="Input raw RAID parquet (default: data/raw/raid_full.parquet)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for processed parquet files",
    )
    args = parser.parse_args()

    filtered_df = _load_raw_parquet(args.input)

    print("\nOverall label distribution:", flush=True)
    print(filtered_df["label"].value_counts().to_string(), flush=True)
    print("\nGenerator distribution:", flush=True)
    print(filtered_df["generator"].value_counts().to_string(), flush=True)

    train_pool = _build_train_pool(filtered_df)
    # Exclude human texts used in training from unseen pool to prevent contamination
    train_human_texts = set(train_pool[train_pool["label"] == 0]["text"])
    filtered_for_unseen = filtered_df[~filtered_df["text"].isin(train_human_texts)]
    unseen_pool = _build_unseen_pool(filtered_for_unseen)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.processed_dir / "raid_train_pool.parquet"
    unseen_path = args.processed_dir / "raid_test_unseen.parquet"

    train_pool.to_parquet(train_path, index=False)
    unseen_pool.to_parquet(unseen_path, index=False)

    print(f"\n{'=' * 60}", flush=True)
    print("Saved:", flush=True)
    print(f"  {train_path}  ({len(train_pool):,} rows)", flush=True)
    print(f"  {unseen_path} ({len(unseen_pool):,} rows)", flush=True)
    end_time = time.perf_counter()
    print(f"Total time: {round(end_time - start_time, 2)}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()