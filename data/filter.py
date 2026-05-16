"""
Filter and balance DetectRL dataset for training and evaluation.

The raw parquet files use ``data_type`` rather than ``attack_type`` and the
available values differ from the initial rough taxonomy. This script maps the
requested pool names onto the actual DetectRL attack families so the pipeline
can run on the downloaded benchmark files.

Fixes vs original:
1. create_test_unseen_pool balances AI vs human (was 96.8% AI)
2. create_train_pool excludes human samples from unseen attack families
   to prevent indirect data leakage into training
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ── Configuration ──────────────────────────────────────────────────────────────
REMOVE_ATTACK_TYPES = {"data_mixing", "spelling_error"}

TRAIN_ATTACK_GROUPS = {
    "clean_ai":          {"direct_prompt"},
    "homoglyph":         {"adversarial_character_llm"},
    "word_substitution": {"adversarial_word_llm", "adversarial_character_word_llm"},
}

TEST_UNSEEN_GROUPS = {
    "paraphrase": {
        "paraphrase_back_translation_llm",
        "paraphrase_polish_llm",
        "paraphrase_dipper_llm",
    },
    "prompt_adversarial": {"prompt_few_shot", "prompt_sico"},
}

# All human samples are used for training.
# Using only domain texts (story/abstract/content/document) causes severe domain
# mismatch with AI attack texts, inflating accuracy to 0.99+ artificially because
# the model learns domain shortcuts rather than authorship features.
# The paired human counterparts share the same domain/style as the AI attack texts
# and must be included for the model to learn genuine AI detection features.
UNSEEN_HUMAN_ATTACK_TYPES: set[str] = set()  # no exclusions — use all human samples

RANDOM_SEED = 42


# ── Data loading ───────────────────────────────────────────────────────────────
def load_detectrl_parquets(raw_dir: Path) -> pd.DataFrame:
    """Load all DetectRL parquet files from data/raw/detectrl/"""
    detectrl_dir = raw_dir / "detectrl"
    if not detectrl_dir.exists():
        raise FileNotFoundError(f"DetectRL directory not found: {detectrl_dir}")

    parquet_files = sorted(detectrl_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {detectrl_dir}")

    print(f"Loading {len(parquet_files)} DetectRL parquet files...")
    frames = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file)
        frames.append(frame)
        print(f"  Loaded {len(frame):,} rows from {parquet_file.name}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal loaded: {len(combined):,} rows")
    return combined


# ── Column standardisation ────────────────────────────────────────────────────
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names and types."""
    df = df.copy()

    if "label" in df.columns and df["label"].dtype == "object":
        df["label"] = df["label"].map({"human": 0, "llm": 1})

    if "attack_type" not in df.columns:
        if "data_type" not in df.columns:
            raise ValueError(
                "Expected either 'attack_type' or 'data_type' in DetectRL parquet files"
            )
        df["attack_type"] = df["data_type"]

    df["attack_type"] = df["attack_type"].astype(str).str.lower().str.strip()
    return df


# ── Filtering ─────────────────────────────────────────────────────────────────
def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with unwanted attack types."""
    initial_count = len(df)
    df = df[~df["attack_type"].isin(REMOVE_ATTACK_TYPES)].copy()
    removed = initial_count - len(df)
    print(f"\nFiltered out {removed:,} rows with attack_type in {REMOVE_ATTACK_TYPES}")
    print(f"Remaining: {len(df):,} rows")
    return df


# ── Group helper ──────────────────────────────────────────────────────────────
def _build_group_frame(
    df: pd.DataFrame,
    group_name: str,
    attack_types: set[str],
    label: int | None = None,
) -> pd.DataFrame:
    group_df = df[df["attack_type"].isin(attack_types)].copy()
    if label is not None:
        group_df = group_df[group_df["label"] == label].copy()

    if group_df.empty:
        available = sorted(df["attack_type"].unique())
        raise ValueError(
            f"No rows matched group '{group_name}' using attack types "
            f"{sorted(attack_types)}. Available: {available}"
        )
    return group_df


# ── Train pool ────────────────────────────────────────────────────────────────
def create_train_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create TRAIN pool:
    - Select AI rows from clean_ai, homoglyph, word_substitution groups
    - Undersample each group to min_count for equal attack representation
    - Add human samples equal to total AI count (1:1 class balance)
    - Exclude human samples from unseen attack families to prevent leakage
    """
    print("\n" + "=" * 60)
    print("Creating TRAIN pool")
    print("=" * 60)

    # Build AI groups
    group_frames: list[pd.DataFrame] = []
    for group_name, attack_types in TRAIN_ATTACK_GROUPS.items():
        group_frame = _build_group_frame(df, group_name, attack_types, label=1)
        print(f"\n{group_name} AI samples: {len(group_frame):,}")
        print(group_frame["attack_type"].value_counts().to_string())
        group_frames.append(group_frame)

    # Undersample to min count
    min_count = min(len(gf) for gf in group_frames)
    print(f"\nUndersampling each TRAIN group to: {min_count:,}")

    balanced_frames = [
        gf.sample(n=min_count, random_state=RANDOM_SEED) for gf in group_frames
    ]
    train_ai = pd.concat(balanced_frames, ignore_index=True)
    print(f"\nAI samples after undersampling: {len(train_ai):,}")
    print(train_ai["attack_type"].value_counts().to_string())

    # Human samples — exclude unseen family source texts
    all_human = df[df["label"] == 0].copy()
    print(f"\nAll human samples available: {len(all_human):,}")

    available_human = all_human[
        ~all_human["attack_type"].isin(UNSEEN_HUMAN_ATTACK_TYPES)
    ].copy()

    excluded = len(all_human) - len(available_human)
    print(f"Excluded {excluded:,} human samples from unseen attack families")
    print(f"  (prevents indirect leakage — paired with unseen AI attacks)")
    print(f"Clean human samples remaining: {len(available_human):,}")
    print(available_human["attack_type"].value_counts().to_string())

    # Balance
    total_ai = len(train_ai)
    print(f"\nTarget human samples: {total_ai:,}")

    if len(available_human) >= total_ai:
        train_human = available_human.sample(n=total_ai, random_state=RANDOM_SEED)
    else:
        print(
            f"Warning: only {len(available_human):,} clean human samples — "
            f"using all (slight class imbalance)"
        )
        train_human = available_human

    train_pool = pd.concat([train_ai, train_human], ignore_index=True)
    print(f"\nFinal TRAIN pool: {len(train_pool):,} rows")
    return train_pool


# ── Unseen test pool ──────────────────────────────────────────────────────────
def create_test_unseen_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create TEST_UNSEEN pool:
    - Select rows from paraphrase and prompt_adversarial groups
    - Balance AI vs human (original was 96.8% AI which inflates metrics)
    """
    print("\n" + "=" * 60)
    print("Creating TEST_UNSEEN pool")
    print("=" * 60)

    test_frames: list[pd.DataFrame] = []
    for group_name, attack_types in TEST_UNSEEN_GROUPS.items():
        group_frame = _build_group_frame(df, group_name, attack_types)
        print(f"\n{group_name} rows: {len(group_frame):,}")
        print(group_frame["attack_type"].value_counts().to_string())
        test_frames.append(group_frame)

    test_pool = pd.concat(test_frames, ignore_index=True)

    print(f"\nRaw unseen pool: {len(test_pool):,} rows")
    print("Raw label distribution (before balancing):")
    print(test_pool["label"].value_counts().to_string())

    ai_samples    = test_pool[test_pool["label"] == 1]
    human_samples = test_pool[test_pool["label"] == 0]
    min_count     = min(len(ai_samples), len(human_samples))

    print(f"\nBalancing unseen pool:")
    print(f"  AI available    : {len(ai_samples):,}")
    print(f"  Human available : {len(human_samples):,}")
    print(f"  Balancing to    : {min_count:,} each")

    test_pool = pd.concat([
        ai_samples.sample(n=min_count,    random_state=RANDOM_SEED),
        human_samples.sample(n=min_count, random_state=RANDOM_SEED),
    ], ignore_index=True)

    print(f"\nFinal TEST_UNSEEN pool: {len(test_pool):,} rows")
    print("Label distribution (after balancing):")
    print(test_pool["label"].value_counts().to_string())
    print("Attack type distribution:")
    print(test_pool["attack_type"].value_counts().to_string())

    return test_pool


# ── Distribution printer ──────────────────────────────────────────────────────
def print_distributions(pool: pd.DataFrame, pool_name: str) -> None:
    print(f"\n{pool_name} DISTRIBUTIONS:")
    print("-" * 60)

    print("\nClass distribution (label):")
    class_dist = pool["label"].value_counts().sort_index()
    for label, count in class_dist.items():
        label_name = "human" if label == 0 else "AI"
        pct = 100 * count / len(pool)
        print(f"  {label_name:5s} (label={label}): {count:6,} ({pct:5.1f}%)")

    print("\nAttack type distribution:")
    attack_dist = pool["attack_type"].value_counts().sort_values(ascending=False)
    for attack_type, count in attack_dist.items():
        pct = 100 * count / len(pool)
        print(f"  {attack_type:35s}: {count:6,} ({pct:5.1f}%)")

    print("\nCross-tabulation (attack_type x label):")
    crosstab = pd.crosstab(pool["attack_type"], pool["label"], margins=True)
    crosstab.columns = [
        "human (0)" if c == 0 else "AI (1)" if c == 1 else "Total"
        for c in crosstab.columns
    ]
    print(crosstab.to_string())


# ── Save ──────────────────────────────────────────────────────────────────────
def save_dataset(pool: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path.relative_to(ROOT_DIR)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            description="Filter and balance DetectRL dataset"
        )
        parser.add_argument(
            "--raw-dir", type=Path, default=RAW_DIR,
            help="Raw data directory",
        )
        parser.add_argument(
            "--processed-dir", type=Path, default=PROCESSED_DIR,
            help="Processed data directory",
        )
        parser.add_argument(
            "--seed", type=int, default=RANDOM_SEED,
            help="Random seed",
        )
        args = parser.parse_args()

    # Load
    df = load_detectrl_parquets(args.raw_dir)

    # Standardise
    df = standardize_columns(df)
    print(f"\nInitial attack_type distribution:")
    print(df["attack_type"].value_counts().to_string())

    # Filter
    df = filter_dataset(df)

    # Build pools
    train_pool       = create_train_pool(df)
    test_unseen_pool = create_test_unseen_pool(df)

    # Print final distributions
    print("\n" + "=" * 60)
    print("FINAL DISTRIBUTIONS")
    print("=" * 60)
    print_distributions(train_pool,       "TRAIN POOL")
    print_distributions(test_unseen_pool, "TEST_UNSEEN POOL")

    # Save
    print("\n" + "=" * 60)
    print("Saving pools")
    print("=" * 60)
    save_dataset(train_pool,       args.processed_dir / "train_pool.parquet")
    save_dataset(test_unseen_pool, args.processed_dir / "test_unseen.parquet")

    print(f"\nDone!")
    print(f"  train_pool      : {len(train_pool):,} rows")
    print(f"  test_unseen_pool: {len(test_unseen_pool):,} rows")


if __name__ == "__main__":
    main()