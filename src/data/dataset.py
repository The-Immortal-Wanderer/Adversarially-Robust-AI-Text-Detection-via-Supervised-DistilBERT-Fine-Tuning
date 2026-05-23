"""
DetectRL Dataset class for tokenization and on-the-fly processing.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


class DetectRLDataset(Dataset):
    """
    PyTorch Dataset for DetectRL that tokenizes examples on-the-fly.
    
    Args:
        dataset: HuggingFace Dataset object with 'text' and 'label' columns
        tokenizer: DistilBertTokenizer for tokenization
        max_length: Maximum sequence length (default: 512)
    """
    
    def __init__(
        self,
        dataset: object,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """Initialize dataset."""
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure dataset has required columns
        if hasattr(dataset, "column_names"):
            columns = dataset.column_names
        else:
            # Pandas DataFrame
            columns = dataset.columns.tolist()
        
        if "text" not in columns:
            raise ValueError(f"Dataset must have 'text' column. Found: {columns}")
        if "label" not in columns:
            raise ValueError(f"Dataset must have 'label' column. Found: {columns}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get tokenized example.
        
        Returns:
            dict with keys: input_ids, attention_mask, labels (all as tensors)
        """
        # Hugging Face datasets support integer indexing directly; pandas needs iloc.
        if hasattr(self.dataset, "iloc"):
            example = self.dataset.iloc[idx].to_dict()
        else:
            example = self.dataset[idx]
        
        text = example["text"]
        label = example["label"]
        
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Extract and squeeze (remove batch dimension added by return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class OnTheFlyDataset(Dataset):
    """Tokenizes text on-the-fly inside DataLoader worker processes.

    Each worker process lazily loads its own tokenizer instance to avoid
    pickling the tokenizer across process boundaries.  With NUM_WORKERS>1
    this creates true CPU-GPU pipeline parallelism.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 256,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None

    def _get_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, local_files_only=False
            )
        return self._tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokenizer = self._get_tokenizer()
        enc = tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class CachedTensorDataset(Dataset):
    """Loads pre-tokenized tensors from a .pt file.  __getitem__ is O(1).

    Used in ``cached`` tokenization mode: the entire dataset is tokenized
    once and saved to disk, then DataLoader workers serve pre-computed
    tensors with minimal per-sample overhead.
    """

    def __init__(self, pt_path: Path) -> None:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
