"""Inference latency benchmark for DistilBertClassifier on RAID dataset.

Evaluates forward-pass latency under torch.amp.autocast for a single
ablation configuration. Designed for RTX 4050 / Kaggle GPUs (RTX 3050 legacy).

Usage:
    python scripts/benchmark.py

Output:
    Prints average latency (ms) over 100 iterations.
    If no checkpoint is found, train first with:
      python scripts/train.py --dataset raid --ablation ablation_b
"""

import torch
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # Project root for reference only; src/ importable via pip install -e .


from src.models import DistilBertClassifier

import argparse

# -- Config --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = Path("artifacts") / "distilbert_detector"
DATASET = "raid"

args: argparse.Namespace | None = None

def _resolve_checkpoint(ablation: str) -> Path | None:
    for name in (
        f"{DATASET}_{ablation}_seed42_best.pt",
        f"{DATASET}_{ablation}_best.pt",
        f"{ablation}_seed42_best.pt",
        f"{ablation}_best.pt",
    ):
        ckpt = ARTIFACT_DIR / name
        if ckpt.exists():
            return ckpt
    return None

TOKENIZER_NAME  = "distilbert-base-uncased"
BATCH_SIZE = 32 # Increased to 32 to stress the RTX 4050 for measurable latency
SEQ_LENGTH = 256
NUM_TRIALS = 100

def benchmark(model, input_ids, mask, description="Model"):
    # Warm-up
    for _ in range(10):
        _ = model(input_ids, mask)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # inference_mode is faster than no_grad for deployment
    with torch.inference_mode():
        # AMP + SDPA (Native Fast Path)
        with torch.amp.autocast(DEVICE.type):
            for _ in range(NUM_TRIALS):
                _ = model(input_ids, mask)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency = (end_time - start_time) / NUM_TRIALS * 1000
    print(f"{description} Average Latency: {avg_latency:.2f} ms")
    return avg_latency

def main():
    global args
    parser = argparse.ArgumentParser(description="Benchmark inference latency for a single ablation configuration.")
    parser.add_argument("--ablation", default="ablation_b", help="Ablation configuration name (default: ablation_b)")
    args = parser.parse_args()

    CHECKPOINT_PATH = _resolve_checkpoint(args.ablation)
    if CHECKPOINT_PATH is None:
        raise FileNotFoundError(
            f"Checkpoint '{ARTIFACT_DIR / f'{DATASET}_{args.ablation}_best.pt'}' not found.\n\n"
            "Train a model first:\n"
            "  python scripts/train.py --dataset raid --ablation ablation_b"
        )

    print(f"Loading model ({args.ablation}) on {DEVICE}...")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    model = DistilBertClassifier(**checkpoint.get('config', {})).to(DEVICE)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = {
            k: v
            for k, v in checkpoint.items()
            if k.startswith("distilbert.") or k.startswith("classifier.")
        }
    if not state_dict:
        raise KeyError(
            f"Checkpoint at '{CHECKPOINT_PATH}' contains no model state dict. "
            f"Keys found: {list(checkpoint.keys())}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Inputs for benchmark
    vocab_size = getattr(model.distilbert.config, "vocab_size", 30522)
    dummy_input = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LENGTH)).to(DEVICE)
    dummy_mask = torch.ones((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long).to(DEVICE)

    print(f"\n--- Running Inference Benchmark (Batch Size: {BATCH_SIZE}) ---")

    # Baseline: Standard FP32 (No Autocast)
    print("Testing Baseline (FP32)...")
    # Warm-up iterations for CUDA kernel compilation
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input, dummy_mask)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start_f32 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(NUM_TRIALS):
            _ = model(dummy_input, dummy_mask)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - start_f32) / NUM_TRIALS * 1000
    print(f"Baseline FP32 Latency: {baseline_ms:.2f} ms")

    # Optimized: AMP + Inference Mode
    print("\nTesting Optimized (AMP + Inference Mode)...")
    optimized_ms = benchmark(model, dummy_input, dummy_mask, "AMP + Inference Mode")

    speedup = (baseline_ms - optimized_ms) / baseline_ms * 100
    print(f"\n[RESULT] Latency Reduction: {speedup:.2f}%")
    print(f"Throughput: {1000/optimized_ms * BATCH_SIZE:.2f} samples/sec")

if __name__ == "__main__":
    main()