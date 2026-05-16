"""Evaluation utilities for DetectRL experiments."""

from .evaluator import run_evaluation
from .metrics import compute_metrics, compute_rrd

__all__ = ["compute_metrics", "compute_rrd", "run_evaluation"]
