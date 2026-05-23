"""
One-shot data setup: download RAID (if needed) and preprocess into parquet pools.

Checks for existing processed data before downloading. Idempotent — safe to
re-run after interruption.

Usage:
    python scripts/setup_data.py                     # default paths
    python scripts/setup_data.py --processed-dir custom/path
    python scripts/setup_data.py --skip-download      # assume raw is already present
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # scripts/ → project root
RAW_PARQUET = ROOT_DIR / "data" / "raw" / "raid" / "raid_full.parquet"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

TRAIN_POOL = PROCESSED_DIR / "raid_train_pool.parquet"
UNSEEN_POOL = PROCESSED_DIR / "raid_test_unseen.parquet"

DOWNLOAD_SCRIPT = ROOT_DIR / "src" / "data" / "processing" / "download_raid_raw.py"
FILTER_SCRIPT = ROOT_DIR / "src" / "data" / "processing" / "filter_raid_sequential.py"


def _check_processed() -> int:
    """Return count of already-existing processed files (0, 1, or 2)."""
    count = 0
    for path, label in [(TRAIN_POOL, "train pool"), (UNSEEN_POOL, "unseen pool")]:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            print(f"  [OK]  {label}: {path.name} ({size_mb:.1f} MB)")
            count += 1
        else:
            print(f"  [MISS] {label}: {path.name}")
    return count


def _download_raw() -> None:
    """Download the full RAID dataset from Hugging Face."""
    print(f"\nDownloading RAID from Hugging Face -> {RAW_PARQUET} ...")
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(DOWNLOAD_SCRIPT), "--output", str(RAW_PARQUET)],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"download_raid_raw.py failed (rc={result.returncode})")

    for line in result.stdout.strip().splitlines():
        print(f"  {line}")


def _run_filter(processed_dir: Path, raw_path: Path) -> None:
    """Run the sequential filter to produce train + unseen parquet pools."""
    print(f"\nFiltering raw data -> {processed_dir} ...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, str(FILTER_SCRIPT),
            "--input", str(raw_path),
            "--processed-dir", str(processed_dir),
        ],
        capture_output=True, text=True, timeout=7200,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"filter_raid_sequential.py failed (rc={result.returncode})")

    for line in result.stdout.strip().splitlines():
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RAID dataset and preprocess into training/eval pools.",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
        help="Output directory for processed parquet files (default: data/processed)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip raw download; assume data/raw/raid/raid_full.parquet exists",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-process even if processed files exist",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir
    raw_path = args.skip_download and RAW_PARQUET or (
        args.processed_dir.parent.parent / "raw" / "raid" / "raid_full.parquet"
        if args.processed_dir != PROCESSED_DIR
        else RAW_PARQUET
    )
    # ^ keep default raw path OR derive from custom processed-dir

    t0 = time.perf_counter()

    # ── Check current state ────────────────────────────────────────────────
    print("Checking processed data pool:")
    existing = _check_processed()

    if args.force:
        print("\n  --force: will re-download and re-process.")
    elif existing == 2:
        elapsed = time.perf_counter() - t0
        print(f"\nAll pools present. Nothing to do ({elapsed:.1f}s).")
        print("Run with --force to re-download and re-process.")
        return

    # ── Step 1: Download raw ───────────────────────────────────────────────
    if args.skip_download:
        if not raw_path.exists():
            print(f"\n[ERROR] --skip-download but {raw_path} not found.", file=sys.stderr)
            sys.exit(1)
        print(f"\n  --skip-download: using existing {raw_path}")
    elif args.force or not raw_path.exists():
        _download_raw()
    else:
        print(f"\n  Raw data exists at {raw_path}, skipping download.")

    # ── Step 2: Filter ─────────────────────────────────────────────────────
    if args.force or existing < 2:
        _run_filter(processed_dir, raw_path)
    else:
        print("\n  Processed pools already complete, skipping filter.")

    elapsed = time.perf_counter() - t0
    print(f"\nDone ({elapsed:.1f}s).")

    # ── Final check ────────────────────────────────────────────────────────
    final = _check_processed()
    if final < 2:
        print(f"\n[WARNING] Only {final}/2 pools were created.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
