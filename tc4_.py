import torch
import time
import torch.nn as nn
from pathlib import Path
from transformers import DistilBertModel

# -- Config --
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("artifacts/distilbert_detector_tc3/ablation_b_tc3_best.pt")
TOKENIZER_NAME  = "distilbert-base-uncased"
BATCH_SIZE = 32 # Increased to 32 to actually stress the RTX 3050
SEQ_LENGTH = 256
NUM_TRIALS = 100

class DistilBertClassifier(nn.Module):
    def __init__(self, head_type: str = "single", freeze_layers: int = 0):
        super().__init__()
        self.head_type = head_type
        self.distilbert = DistilBertModel.from_pretrained(TOKENIZER_NAME)
        hidden = self.distilbert.config.hidden_size

        if head_type == "single":
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 2))
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(hidden, 384), nn.GELU(), nn.Dropout(0.2), nn.Linear(384, 2)
            )

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

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
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Could not find checkpoint at {CHECKPOINT_PATH}")
        return

    print(f"Loading model on {DEVICE}...")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = DistilBertClassifier(**checkpoint['config']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Inputs for benchmark
    dummy_input = torch.randint(0, 30522, (BATCH_SIZE, SEQ_LENGTH)).to(DEVICE)
    dummy_mask = torch.ones((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long).to(DEVICE)

    print(f"\n--- Running TC4 Benchmark (Batch Size: {BATCH_SIZE}) ---")
    
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
    print("\nTesting Optimized (TC4: AMP + Inference Mode)...")
    optimized_ms = benchmark(model, dummy_input, dummy_mask, "TC4 Optimized")

    speedup = (baseline_ms - optimized_ms) / baseline_ms * 100
    print(f"\n[TC4 RESULT] Latency Reduction: {speedup:.2f}%")
    print(f"Throughput: {1000/optimized_ms * BATCH_SIZE:.2f} samples/sec")

if __name__ == "__main__":
    main()