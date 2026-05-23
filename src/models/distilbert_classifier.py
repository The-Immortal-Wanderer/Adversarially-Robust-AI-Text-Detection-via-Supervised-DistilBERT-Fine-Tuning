"""DistilBERT-based classifier for AI-generated text detection.

Provides the primary detection model: DistilBertClassifier with configurable
head type (single / deep) and layer freezing for ablation experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    def __init__(self, head_type: str = "single", freeze_layers: int = 0):
        super().__init__()
        if head_type not in {"single", "deep"}:
            raise ValueError("head_type must be 'single' or 'deep'")
        if not 0 <= freeze_layers <= 6:
            raise ValueError("freeze_layers must be between 0 and 6")

        self.head_type = head_type
        self.freeze_layers = freeze_layers
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.distilbert.config.hidden_size

        if head_type == "single":
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(hidden_size, 384),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(384, 2),
            )

        self._freeze_layers()

    def _freeze_layers(self) -> None:
        if self.freeze_layers > 0:
            for parameter in self.distilbert.embeddings.parameters():
                parameter.requires_grad = False

        for layer_index, layer in enumerate(self.distilbert.transformer.layer):
            requires_grad = layer_index >= self.freeze_layers
            for parameter in layer.parameters():
                parameter.requires_grad = requires_grad

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden_state)
        return logits

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
