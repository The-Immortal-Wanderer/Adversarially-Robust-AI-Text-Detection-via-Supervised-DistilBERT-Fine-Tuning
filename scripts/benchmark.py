"""Inference latency benchmark for DistilBertClassifier on RAID dataset.

Evaluates forward-pass latency under torch.amp.autocast for a single
ablation configuration. Designed for RTX 3050 / 4050 / Kaggle GPUs.

Usage:
    python scripts/benchmark.py

Output:
    Prints average latency (ms) over 100 iterations.
    If no checkpoint is found, train first with:
      python scripts/train.py --dataset raid --ablation ablation_b
"""

import sys
import torch
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import DistilBertClassifier

# -- Config --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Unified checkpoint fallback: try standard artifact paths from all training scripts
# Unified checkpoint resolution: try all known artifact paths.
# Checkpoints saved by scripts/train.py use:
# {artifact_dir}/{dataset}_{ablation_name}_best.pt
ARTIFACT_DIR = Path("artifacts") / "distilbert_detector"
DATASET = "raid"
ABLATION = "ablation_b"
CHECKPOINT_PATH = ARTIFACT_DIR / f"{DATASET}_{ABLATION}_best.pt"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = ARTIFACT_DIR / f"{ABLATION}_best.pt"
    if not CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH = None

TOKENIZER_NAME  = "distilbert-base-uncased"
BATCH_SIZE = 32 # Increased to 32 to actually stress the RTX 3050
SEQ_LENGTH = 256
NUM_TRIALS = 100

def benchmark(model, input_ids, mask, description="Model"):
    # Warm-up
    for _ in range(10):
        _ = model(input_ids, mask)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # inference_mode is faster than no_grad for deployment
    with torch.inference_mode():
        # AMP (from TC3) + SDPA (Native Fast Path)
        with torch.amp.autocast("cuda"): 
            for _ in range(NUM_TRIALS):
                _ = model(input_ids, mask)
            
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_latency = (end_time - start_time) / NUM_TRIALS * 1000 
    print(f"{description} Average Latency: {avg_latency:.2f} ms")
    return avg_latency

def main():
    if CHECKPOINT_PATH is None:
        raise FileNotFoundError(
            f"Checkpoint not found at '{ARTIFACT_DIR / f'{DATASET}_{ABLATION}_best.pt'}'.\n\n"
            "Train a model first:\n"
            "  python scripts/train.py --dataset raid --ablation ablation_b"
        )

    print(f"Loading model on {DEVICE}...")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    model = DistilBertClassifier(**checkpoint['config']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Inputs for benchmark
    dummy_input = torch.randint(0, 30522, (BATCH_SIZE, SEQ_LENGTH)).to(DEVICE)
    dummy_mask = torch.ones((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long).to(DEVICE)

    print(f"\n--- Running Inference Benchmark (Batch Size: {BATCH_SIZE}) ---")
    
    # Baseline: Standard FP32 (No Autocast)
    print("Testing Baseline (FP32)...")
    torch.cuda.synchronize()
    start_f32 = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_TRIALS):
            _ = model(dummy_input, dummy_mask)
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