"""
DataLoader creation for RAID train/val/test splits and unseen test set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .dataset import CachedTensorDataset, RAIDDataset, OnTheFlyDataset


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Configuration
BATCH_SIZE = 16
NUM_WORKERS = 6
PIN_MEMORY = True
PREFETCH_FACTOR = 2
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
RANDOM_SEED = 42


def _load_tokenizer() -> object:
    """Load DistilBert tokenizer."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _data_loader_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["pin_memory"] = pin_memory
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def get_dataloaders(
    parquet_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    prefetch_factor: int = PREFETCH_FACTOR,
    max_length: int = MAX_LENGTH,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load train_pool and create 80/10/10 stratified train/val/test DataLoaders.

    Args:
        parquet_path: Path to train_pool.parquet (default: data/processed/train_pool.parquet)
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory for DataLoader
        prefetch_factor: Prefetch factor for DataLoader
        max_length: Max tokenization length

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Use default path if not provided
    if parquet_path is None:
        parquet_path = PROCESSED_DIR / "raid_train_pool.parquet"
        if not parquet_path.exists():
            parquet_path = PROCESSED_DIR / "train_pool.parquet"
    else:
        parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print("="*60)
    print("Loading train_pool DataLoaders")
    print("="*60)

    # Load dataset
    df = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df):,} samples from {parquet_path.name}")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    # Load tokenizer
    tokenizer = _load_tokenizer()
    print(f"\nUsing tokenizer: {TOKENIZER_NAME}")

    # Stratified 80/10/10 split
    print("\nCreating 80/10/10 train/val/test split (stratified by label)...")

    # First split: 80% train, 20% temp (will be split into val/test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    # Second split: 50% val, 50% test from temp (10% each of original)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    print(f"  Train: {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val:   {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test:  {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")

    # Create datasets
    print("\nCreating RAIDDataset objects...")
    train_dataset = RAIDDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = RAIDDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = RAIDDataset(test_df, tokenizer, max_length=max_length)

    # Create DataLoaders
    print("\nCreating DataLoaders...")
    loader_kwargs = _data_loader_kwargs(num_workers, pin_memory, prefetch_factor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Print sample batch
    print("\nFetching sample batch from train_loader...")
    sample_batch = next(iter(train_loader))
    print(f"  input_ids shape:    {sample_batch['input_ids'].shape}")
    print(f"  attention_mask shape: {sample_batch['attention_mask'].shape}")
    print(f"  labels shape:       {sample_batch['labels'].shape}")
    print(f"  Batch device:       {sample_batch['input_ids'].device}")
    print("\nDataLoaders ready!")
    print(f"  - train_loader: {len(train_loader)} batches of {batch_size}")
    print(f"  - val_loader:   {len(val_loader)} batches of {batch_size}")
    print(f"  - test_loader:  {len(test_loader)} batches of {batch_size}")

    return train_loader, val_loader, test_loader


def get_unseen_loader(
    parquet_path: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    prefetch_factor: int = PREFETCH_FACTOR,
    max_length: int = MAX_LENGTH,
) -> DataLoader:
    """
    Load test_unseen and create DataLoader.

    Args:
        parquet_path: Path to test_unseen.parquet (default: data/processed/test_unseen.parquet)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory for DataLoader
        prefetch_factor: Prefetch factor for DataLoader
        max_length: Max tokenization length

    Returns:
        DataLoader for unseen test set
    """
    if parquet_path is None:
        parquet_path = PROCESSED_DIR / "raid_test_unseen.parquet"
        if not parquet_path.exists():
            parquet_path = PROCESSED_DIR / "test_unseen.parquet"
    else:
        parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    print("="*60)
    print("Loading test_unseen DataLoader")
    print("="*60)

    # Load dataset
    df = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df):,} samples from {parquet_path.name}")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}")
    print(f"Attack type distribution:\n{df['attack_type'].value_counts().to_string()}")

    # Load tokenizer
    tokenizer = _load_tokenizer()
    print(f"\nUsing tokenizer: {TOKENIZER_NAME}")

    # Create dataset
    print("\nCreating RAIDDataset object...")
    unseen_dataset = RAIDDataset(df, tokenizer, max_length=max_length)

    # Create DataLoader
    print("Creating DataLoader...")
    loader_kwargs = _data_loader_kwargs(num_workers, pin_memory, prefetch_factor)
    unseen_loader = DataLoader(
        unseen_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    # Print sample batch
    print("\nFetching sample batch from unseen_loader...")
    sample_batch = next(iter(unseen_loader))
    print(f"  input_ids shape:    {sample_batch['input_ids'].shape}")
    print(f"  attention_mask shape: {sample_batch['attention_mask'].shape}")
    print(f"  labels shape:       {sample_batch['labels'].shape}")
    print(f"  Batch device:       {sample_batch['input_ids'].device}")
    print("\nUnseen DataLoader ready!")
    print(f"  - unseen_loader: {len(unseen_loader)} batches of {batch_size}")

    return unseen_loader


def _capped_df(
    dataset_name: str,
    *,
    processed_dir: Path = PROCESSED_DIR,
    seed: int = 42,
    samples_per_class: int = 60_000,
) -> pd.DataFrame:
    """Load train pool, deduplicate, and cap to ``samples_per_class`` per label.

    Saves the capped result as ``{dataset_name}_train_pool_capped_{seed}_{samples_per_class}.parquet``
    for fast re-use across multiple ablation runs. The seed and cap value are embedded in the
    filename so that different seed or cap values produce independent cache entries.
    """
    train_parquet = processed_dir / f"{dataset_name}_train_pool.parquet"
    capped_parquet = processed_dir / f"{dataset_name}_train_pool_capped_{seed}_{samples_per_class}.parquet"

    if capped_parquet.exists():
        df = pd.read_parquet(capped_parquet)
        print(f"Capped dataset found: {len(df):,} rows", flush=True)
        return df

    print("Preparing capped dataset...", flush=True)
    df = pd.read_parquet(train_parquet)
    before = len(df)
    df = df.loc[~df["text"].duplicated(keep="first")].reset_index(drop=True)
    after = len(df)
    if before > after:
        print(
            f"Deduplicated training pool: {before:,} -> {after:,} rows "
            f"({before - after:,} exact duplicates removed)",
            flush=True,
        )
    n_human = min(samples_per_class, len(df[df["label"] == 0]))
    n_ai = min(samples_per_class, len(df[df["label"] == 1]))
    n = min(n_human, n_ai)  # balance classes after dedup
    if n < samples_per_class:
        print(
            f"Note: samples_per_class={samples_per_class} exceeds available after dedup. "
            f"Using n={n} (human={n_human}, ai={n_ai})",
            flush=True,
        )
    if n < 1:
        raise ValueError(
            f"_capped_df: capped to 0 samples for {dataset_name} "
            f"(human={n_human}, ai={n_ai})"
        )
    df_human = df[df["label"] == 0].sample(n=n, random_state=seed)
    df_ai = df[df["label"] == 1].sample(n=n, random_state=seed)
    df_capped = (
        pd.concat([df_human, df_ai])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    df_capped.to_parquet(capped_parquet, index=False)
    print(f"Capped dataset saved: {len(df_capped):,} rows", flush=True)
    print(df_capped["label"].value_counts().to_string(), flush=True)
    return df_capped


def _unseen_df(
    dataset_name: str,
    *,
    processed_dir: Path = PROCESSED_DIR,
    seed: int = 42,
    unseen_cap: int = 10_000,
) -> pd.DataFrame:
    """Load the unseen test set and cap to ``unseen_cap`` total rows."""
    unseen_parquet = processed_dir / f"{dataset_name}_test_unseen.parquet"
    df_unseen = pd.read_parquet(unseen_parquet)
    # Deduplicate to match _capped_df behavior
    df_unseen = df_unseen.loc[~df_unseen["text"].duplicated(keep="first")].reset_index(drop=True)
    n_unseen = min(
        unseen_cap // 2,
        int((df_unseen["label"] == 0).sum()),
        int((df_unseen["label"] == 1).sum()),
    )
    if n_unseen < 1:
        raise ValueError(
            f"_unseen_df: capped to 0 samples for {dataset_name} "
            f"(min of cap/2={unseen_cap//2}, "
            f"human_count={int((df_unseen['label']==0).sum())}, "
            f"ai_count={int((df_unseen['label']==1).sum())})"
        )
    df_capped = (
        pd.concat(
            [
                df_unseen[df_unseen["label"] == 0].sample(n=n_unseen, random_state=seed),
                df_unseen[df_unseen["label"] == 1].sample(n=n_unseen, random_state=seed),
            ]
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    print(
        f"Unseen pool: {len(df_capped):,} rows (capped from {len(df_unseen):,})",
        flush=True,
    )
    return df_capped


def _prepare_dataloaders_on_the_fly(
    dataset_name: str,
    *,
    processed_dir: Path = PROCESSED_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    prefetch_factor: int = PREFETCH_FACTOR,
    max_length: int = MAX_LENGTH,
    tokenizer_name: str = TOKENIZER_NAME,
    seed: int = 42,
    samples_per_class: int = 60_000,
    unseen_cap: int = 10_000,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Prepare train / val / test / unseen DataLoaders (on-the-fly tokenization).

    Each worker process lazily loads its own tokenizer (via
    ``OnTheFlyDataset``) to avoid pickling across process boundaries.
    """
    df_capped = _capped_df(dataset_name, processed_dir=processed_dir, seed=seed, samples_per_class=samples_per_class)
    df_unseen = _unseen_df(dataset_name, processed_dir=processed_dir, seed=seed, unseen_cap=unseen_cap)

    texts_all = df_capped["text"].tolist()
    labels_all = df_capped["label"].tolist()
    indices = list(range(len(texts_all)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels_all,
    )
    temp_labels = [labels_all[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=temp_labels,
    )

    def _make_ds(idx_list):
        return OnTheFlyDataset(
            [texts_all[i] for i in idx_list],
            [labels_all[i] for i in idx_list],
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)
    test_ds = _make_ds(test_idx)
    unseen_ds = OnTheFlyDataset(
        df_unseen["text"].tolist(),
        df_unseen["label"].tolist(),
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )

    kw = _data_loader_kwargs(num_workers, pin_memory, prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    unseen_loader = DataLoader(unseen_ds, batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, unseen_loader


def _prepare_dataloaders_cached(
    dataset_name: str,
    *,
    processed_dir: Path = PROCESSED_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    prefetch_factor: int = PREFETCH_FACTOR,
    max_length: int = MAX_LENGTH,
    tokenizer_name: str = TOKENIZER_NAME,
    seed: int = 42,
    samples_per_class: int = 60_000,
    unseen_cap: int = 10_000,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Prepare DataLoaders using pre-tokenized (cached) tensors.

    The entire dataset is tokenized once and saved to a ``.pt`` file,
    then loaded via ``CachedTensorDataset`` for O(1) ``__getitem__``.
    A stratified 80/10/10 split is used (labels are available from the
    full DataFrame before tokenization, so class balance is preserved).
    """
    df_capped = _capped_df(dataset_name, processed_dir=processed_dir, seed=seed, samples_per_class=samples_per_class)
    df_unseen = _unseen_df(dataset_name, processed_dir=processed_dir, seed=seed, unseen_cap=unseen_cap)

    cache_dir = processed_dir / "tokenized_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _pretokenize(df: pd.DataFrame, cache_path: Path) -> Path:
        if cache_path.exists():
            print(f"Cache found, skipping: {cache_path.name}", flush=True)
            return cache_path
        print(f"Pre-tokenising -> {cache_path.name} ...", flush=True)
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        encoded = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        torch.save(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": torch.tensor(labels, dtype=torch.long)},
            cache_path,
        )
        print(f"Saved {len(labels):,} samples to {cache_path.name}", flush=True)
        return cache_path

    train_cache = _pretokenize(
        df_capped, cache_dir / f"{dataset_name}_train_pool_capped_{seed}_{max_length}_{samples_per_class}.pt")
    unseen_cache = _pretokenize(
        df_unseen, cache_dir / f"{dataset_name}_test_unseen_{unseen_cap}_{seed}_{max_length}.pt")

    full_ds = CachedTensorDataset(train_cache)
    n = len(full_ds)

    # Use stratified split (same strategy as on-the-fly path)
    labels = df_capped["label"].tolist()
    indices = list(range(n))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=seed, stratify=temp_labels,
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    unseen_ds = CachedTensorDataset(unseen_cache)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}", flush=True)

    kw = _data_loader_kwargs(num_workers, pin_memory, prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    unseen_loader = DataLoader(unseen_ds, batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, unseen_loader