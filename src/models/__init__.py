"""Model definitions for DetectRL experiments."""

from .distilbert_classifier import DistilBertClassifier, count_trainable_parameters

__all__ = ["DistilBertClassifier", "count_trainable_parameters"]