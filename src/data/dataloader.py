"""
DataLoader creation for DetectRL train/val/test splits and unseen test set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from .dataset import DetectRLDataset
except ImportError:
    from dataset import DetectRLDataset


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Configuration
BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
RANDOM_SEED = 42


def _load_tokenizer() -> object:
    """Load DistilBert tokenizer."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _data_loader_kwargs(num_workers: int, pin_memory: bool, prefetch_factor: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
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
    print(f"\nCreating 80/10/10 train/val/test split (stratified by label)...")
    
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
    print(f"\nCreating DetectRLDataset objects...")
    train_dataset = DetectRLDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = DetectRLDataset(val_df, tokenizer, max_length=max_length)
    test_dataset = DetectRLDataset(test_df, tokenizer, max_length=max_length)
    
    # Create DataLoaders
    print(f"\nCreating DataLoaders...")
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
    print(f"\nFetching sample batch from train_loader...")
    sample_batch = next(iter(train_loader))
    print(f"  input_ids shape:    {sample_batch['input_ids'].shape}")
    print(f"  attention_mask shape: {sample_batch['attention_mask'].shape}")
    print(f"  labels shape:       {sample_batch['labels'].shape}")
    print(f"  Batch device:       {sample_batch['input_ids'].device}")
    
    print(f"\n✓ DataLoaders ready!")
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
    # Use default path if not provided
    if parquet_path is None:
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
    print(f"\nCreating DetectRLDataset object...")
    unseen_dataset = DetectRLDataset(df, tokenizer, max_length=max_length)
    
    # Create DataLoader
    print(f"Creating DataLoader...")
    loader_kwargs = _data_loader_kwargs(num_workers, pin_memory, prefetch_factor)
    unseen_loader = DataLoader(
        unseen_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    # Print sample batch
    print(f"\nFetching sample batch from unseen_loader...")
    sample_batch = next(iter(unseen_loader))
    print(f"  input_ids shape:    {sample_batch['input_ids'].shape}")
    print(f"  attention_mask shape: {sample_batch['attention_mask'].shape}")
    print(f"  labels shape:       {sample_batch['labels'].shape}")
    print(f"  Batch device:       {sample_batch['input_ids'].device}")
    
    print(f"\n✓ Unseen DataLoader ready!")
    print(f"  - unseen_loader: {len(unseen_loader)} batches of {batch_size}")
    
    return unseen_loader


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = get_dataloaders()
    unseen_loader = get_unseen_loader()
