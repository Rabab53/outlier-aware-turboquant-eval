import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from outlier_aware_turboquant import OutlierAwareTurboQuantMSE

try:
    from turboquant.quantizer import TurboQuantMSE
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--mode", type=str, choices=["fp16", "baseline", "outlier"], default="fp16")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out_frac", type=float, default=0.10)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"Hooking model for mode={args.mode}, bits={args.bits}, out_frac={args.out_frac}")
    for layer in model.model.layers:
        if hasattr(layer.self_attn.k_proj, '_forward_hooks'):
            layer.self_attn.k_proj._forward_hooks.clear()

    if args.mode != "fp16":
        quantizers = {}
        for i, layer in enumerate(model.model.layers):
            layer_device = layer.self_attn.k_proj.weight.device
            if args.mode == "baseline":
                quantizers[i] = TurboQuantMSE(dim=head_dim, bits=args.bits, device=layer_device)
            elif args.mode == "outlier":
                quantizers[i] = OutlierAwareTurboQuantMSE(d=head_dim, bits=args.bits, outlier_fraction=args.out_frac, device=layer_device)
            
            def make_hook(q_idx):
                def k_proj_hook(module, input, output):
                    orig_shape = output.shape
                    q_device = quantizers[q_idx].device if args.mode == "baseline" else quantizers[q_idx].base_quantizer.device
                    x = output.view(-1, head_dim).to(q_device)
                    if args.mode == "baseline":
                        q = quantizers[q_idx].quantize(x)
                        x_rec = quantizers[q_idx].dequantize(q)
                    else:
                        x_rec = quantizers[q_idx].quantize_and_dequantize(x)
                    return x_rec.view(orig_shape).to(output.device).to(output.dtype)
                return k_proj_hook
            layer.self_attn.k_proj.register_forward_hook(make_hook(i))

    print(f"Loading dataset {args.dataset}...")
    test_data = load_dataset(args.dataset, args.dataset_config, split="test")
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

    nlls = []
    seq_len = args.seq_len
    total_len = encodings.input_ids.size(1)
    
    print(f"Calculating perplexity over {total_len} tokens in chunks of {seq_len}...")
    for i in tqdm(range(0, total_len, seq_len)):
        begin_loc = i
        end_loc = min(i + seq_len, total_len)
        trg_len = end_loc - i
        if trg_len < 2:
            continue
            
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f"\nFinal Perplexity: {ppl:.4f}")
    
    # Save result
    pct = int(args.out_frac * 100)
    name = f"ppl_{args.mode}"
    if args.mode != "fp16":
        name += f"_{args.bits}b"
        if args.mode == "outlier":
            name += f"_{pct}out"
            
    res_path = os.path.join(args.out_dir, f"{name}.txt")
    with open(res_path, "w") as f:
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Bits: {args.bits}\n")
        f.write(f"Outlier Fraction: {args.out_frac}\n")
        f.write(f"Perplexity: {ppl:.4f}\n")

if __name__ == "__main__":
    main()
