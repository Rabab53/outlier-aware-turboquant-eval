import torch
import numpy as np
import sys
import os
import torch._dynamo

# 1. ENABLE TENSOR CORES (TF32)
torch.set_float32_matmul_precision('high')

# 2. INCREASE THE COMPILATION LIMIT
torch._dynamo.config.recompile_limit = 200

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from unified_fast_wrappers import UnifiedOptimizedQuantizer

def benchmark_compiled(quantizer_instance, x):
    compiled_fn = torch.compile(quantizer_instance.quantize_and_dequantize)
    
    # Warmup (Trigger JIT)
    for _ in range(30): _ = compiled_fn(x)
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    
    latencies = []
    for _ in range(200):
        start.record(); _ = compiled_fn(x); end.record(); torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000) 
        
    return np.mean(latencies)

def main():
    device = "cuda"
    n_tokens = 128000 
    dimensions = [200, 1536, 3072]
    
    configs = [
        {"name": "Baseline 4-bit", "mode": "baseline", "bits": 4},
        {"name": "Baseline 2-bit", "mode": "baseline", "bits": 2},
        {"name": "Outlier-Aware 4-bit + 10% (FP16)", "mode": "outlier", "bits": 4, "out_f": 0.10},
        {"name": "Outlier-Aware 2-bit + 10% (FP16)", "mode": "outlier", "bits": 2, "out_f": 0.10},
        {"name": "Two-Level (4-bit / 10% 6-bit)", "mode": "two_level", "bits": 4, "out_f": 0.10, "out_bits": 6},
        {"name": "Two-Level (4-bit / 10% 8-bit)", "mode": "two_level", "bits": 4, "out_f": 0.10, "out_bits": 8},
        {"name": "Two-Level (2-bit / 10% 4-bit)", "mode": "two_level", "bits": 2, "out_f": 0.10, "out_bits": 4},
        {"name": "Two-Level (2-bit / 10% 6-bit)", "mode": "two_level", "bits": 2, "out_f": 0.10, "out_bits": 6},
        {"name": "Two-Level (2-bit / 10% 8-bit)", "mode": "two_level", "bits": 2, "out_f": 0.10, "out_bits": 8},
    ]

    print(f"ULTIMATE Fair Dimension Scaling Sweep (Tokens: {n_tokens} | TF32: Enabled | Limit: 200)")
    print("-" * 120)
    header = f"{'Method Configuration':<40} | " + " | ".join([f"D={d:<6}" for d in dimensions])
    print(header)
    print("-" * 120)

    for c in configs:
        row = f"{c['name']:<40} | "
        for d in dimensions:
            x = torch.randn(n_tokens, d, device=device).bfloat16()
            q = UnifiedOptimizedQuantizer(
                d=d, 
                mode=c["mode"], 
                bits=c.get("bits", 2), 
                outlier_bits=c.get("out_bits", 4), 
                outlier_fraction=c.get("out_f", 0.0), 
                device=device
            )
            lat = benchmark_compiled(q, x)
            row += f"{lat:>8.1f} us | "
        print(row)
        sys.stdout.flush()

if __name__ == "__main__":
    main()
