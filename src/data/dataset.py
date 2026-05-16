"""
DetectRL Dataset class for tokenization and on-the-fly processing.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


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
