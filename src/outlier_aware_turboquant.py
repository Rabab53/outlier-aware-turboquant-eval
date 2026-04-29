import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
# Ensure turboquant is in path for the base quantizer dependency
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'turboquant_lib')))
from turboquant.quantizer import TurboQuantMSE

HF_TOKEN = os.getenv("HF_TOKEN")

class OutlierAwareTurboQuantMSE:
    """
    Wrapper class that intercepts KV cache tensors before quantization,
    extracts the high-magnitude 'outlier' channels to preserve them in FP16/BF16,
    and passes the remaining 'inlier' channels to the original TurboQuant algorithm.
    """
    def __init__(self, d: int, bits: int = 2, outlier_fraction: float = 0.03, device="cuda"):
        self.d = d
        self.bits = bits
        self.outlier_fraction = outlier_fraction
        # Calculate exactly how many channels represent the top X%
        self.num_outliers = max(1, int(d * outlier_fraction))
        self.inlier_d = d - self.num_outliers
        
        # Initialize the base TurboQuant engine to handle only the inlier dimensions
        self.base_quantizer = TurboQuantMSE(dim=self.inlier_d, bits=bits, device=device)
        self.device = device
        
    def quantize_and_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x_float = x.float() 
        shape = x_float.shape
        
        # Flatten the tensor to easily compute channel-wise maximums
        x_flat = x_float.view(-1, self.d)
        
        # Find the absolute maximum value for each channel across all tokens
        channel_max = x_flat.abs().max(dim=0).values
        # Get the indices of the highest magnitude channels (the outliers)
        _, outlier_indices = torch.topk(channel_max, self.num_outliers)
        
        # Create a boolean mask to separate outliers from inliers
        outlier_mask = torch.zeros(self.d, dtype=torch.bool, device=device)
        outlier_mask[outlier_indices] = True
        
        # Physically slice the tensor into two groups
        outliers = x_flat[:, outlier_mask]   # Kept in pure uncompressed format
        inliers = x_flat[:, ~outlier_mask]   # Sent to compression
        
        # Compress and immediately decompress the inliers using TurboQuant
        q_inliers = self.base_quantizer.quantize(inliers)
        dequantized_inliers = self.base_quantizer.dequantize(q_inliers)
        
        # Reconstruct the original tensor by stitching the pristine outliers 
        # and the lossy inliers back into their original channel indices
        reconstructed = torch.zeros_like(x_flat)
        reconstructed[:, outlier_mask] = outliers
        reconstructed[:, ~outlier_mask] = dequantized_inliers
        
        # Reshape back to the original Attention mechanism shape
        return reconstructed.view(shape).to(dtype)

def evaluate_ppl(model, tokenizer, dataset, max_length=2048, stride=1024, max_chunks=100):
    """
    Calculates Perplexity (PPL) using a sliding window approach.
    A lower score indicates a more coherent, less "confused" model.
    """
    # Join the dataset text into a single continuous string and tokenize
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    chunks = 0
    
    # Safely handle multi-GPU or single-GPU model mapping
    device = model.device if hasattr(model, 'device') else "cuda:0"
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        # Extract the sliding window chunk
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # Ignore loss calculation for overlapping (previously processed) context
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss)
        
        prev_end_loc = end_loc
        chunks += 1
        if end_loc == seq_len or chunks >= max_chunks: break
        
    return torch.exp(torch.stack(nlls).mean()).item()

def run_needle_test(model, tokenizer, context_length, depth_percentage):
    """
    Runs the Needle-In-A-Haystack factual retrieval benchmark.
    Hides a specific string at a specific depth percentage inside generic filler text.
    """
    filler_phrase = "The city of Rome is known for its incredible history. "
    tokens_per_phrase = 12 
    total_phrases = context_length // tokens_per_phrase
    
    # Calculate exact index to insert the needle based on depth%
    insert_idx = int(total_phrases * (depth_percentage / 100.0))
    
    needle = "The secret password is 'MUHARB_2026'. "
    parts = [filler_phrase] * total_phrases
    parts.insert(insert_idx, needle)
    haystack = "".join(parts)
    
    prompt = haystack + "\n\nQuestion: What is the secret password?\nAnswer:"
    device = model.device if hasattr(model, 'device') else "cuda:0"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response autoregressively with greedy decoding (no randomness)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    
    # Check if the model hallucinated or successfully retrieved the fact
    return "muharb_2026" in response

def apply_hooks_standard(model, head_dim, mode, bits, outlier_fraction):
    """
    Injects custom PyTorch hooks into every standard Self-Attention layer.
    This allows us to intercept and compress the KV cache dynamically during inference.
    """
    # Clear any existing hooks from previous runs
    for layer in model.model.layers:
        if hasattr(layer.self_attn.k_proj, '_forward_hooks'):
            layer.self_attn.k_proj._forward_hooks.clear()

    # Skip hook injection entirely if running the FP16 baseline
    if mode == "fp16":
        return

    quantizers = {}
    for i, layer in enumerate(model.model.layers):
        # Dynamically grab the exact physical GPU where this layer's weights are stored
        # Crucial for multi-GPU setups (like 32B/70B models) to avoid cross-device memory crashes
        layer_device = layer.self_attn.k_proj.weight.device
        
        # Initialize the appropriate quantizer for this layer
        if mode == "baseline":
            quantizers[i] = TurboQuantMSE(dim=head_dim, bits=bits, device=layer_device)
        elif mode == "outlier":
            quantizers[i] = OutlierAwareTurboQuantMSE(d=head_dim, bits=bits, outlier_fraction=outlier_fraction, device=layer_device)
        
        # Closure to capture the correct layer index (i)
        def make_hook(q_idx):
            def k_proj_hook(module, input, output):
                orig_shape = output.shape
                # Identify which GPU this specific quantizer is instantiated on
                q_device = quantizers[q_idx].device if mode == "baseline" else quantizers[q_idx].base_quantizer.device
                
                # Move the raw activation tensor to the quantizer's GPU (handles device_map="auto" shuffling)
                x = output.view(-1, head_dim).to(q_device)
                
                if mode == "baseline":
                    q = quantizers[q_idx].quantize(x)
                    x_rec = quantizers[q_idx].dequantize(q)
                else:
                    x_rec = quantizers[q_idx].quantize_and_dequantize(x)
                    
                # Return the compressed tensor back to its original shape, device, and datatype
                return x_rec.view(orig_shape).to(output.device).to(output.dtype)
            return k_proj_hook
            
        # Register the hook on the Key Projection output
        layer.self_attn.k_proj.register_forward_hook(make_hook(i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["fp16", "baseline", "outlier"], required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--outlier_fraction", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    print(f"Loading Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN)
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Apply Standard Dense/MoE Hooks
    apply_hooks_standard(model, head_dim, args.mode, args.bits, args.outlier_fraction)

    # 1. PPL Evaluation
    print("Running PPL Evaluation...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = evaluate_ppl(model, tokenizer, dataset, max_chunks=20) # Faster verification chunk size
    print(f"Result -> Mode: {args.mode} | Bits: {args.bits} | Outlier: {args.outlier_fraction} | PPL: {ppl:.4f}")

    # 2. Needle Heatmap
    print("Running Needle Heatmap...")
    context_lengths = [4096, 16384, 32768, 65536, 104000] 
    depths = [10, 30, 50, 70, 90]
    
    matrix = []
    for d in depths:
        row = []
        for c in context_lengths:
            try:
                success = run_needle_test(model, tokenizer, c, d)
            except Exception as e:
                print(f"Failed at Context {c} (Likely RoPE/OOM bounds): {e}")
                success = False
            row.append(1 if success else 0)
        matrix.append(row)

    print("\nNeedle Matrix (Columns: Context Length, Rows: Depth):")
    for row in matrix:
        print(",".join(map(str, row)))
        
    model_short = args.model.split("/")[-1]
    pct = int(args.outlier_fraction * 100)
    
    # Save isolated log file for the orchestrator to aggregate later
    log_name = os.path.join(args.output_dir, f"{model_short}_{args.mode}_{args.bits}b_{pct}out.log")
    with open(log_name, "w") as f:
        f.write(f"Model: {args.model}\nMode: {args.mode}\nBits: {args.bits}\nOutliers: {pct}%\nPPL: {ppl:.4f}\n\nNeedle Matrix:\n")
        for row in matrix:
            f.write(",".join(map(str, row)) + "\n")
            
    print(f"Log saved to {log_name}")

if __name__ == "__main__":
    main()
