import sys
import os
import glob
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure we can import from the sibling src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from outlier_aware_turboquant import OutlierAwareTurboQuantMSE

from transformers import AutoModelForCausalLM, AutoTokenizer
# NOTE: Requires the turboquant base package to be installed in your environment!
try:
    from turboquant.quantizer import TurboQuantMSE
except ImportError:
    print("WARNING: Could not import TurboQuantMSE from turboquant.quantizer.")
    print("Ensure the base turboquant repository is installed in your python environment.")
    sys.exit(1)

NEEDLES = [
    {
        "text": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        "question": "What is the best thing to do in San Francisco?",
        "answers": ["dolores park", "sandwich"]
    },
    {
        "text": "\nThe secret password to access the hidden underground vault is 'MUHARB_2026'.\n",
        "question": "What is the secret password to access the hidden underground vault?",
        "answers": ["muharb_2026", "muharb"]
    },
    {
        "text": "\nThe most important ingredient for the perfect pizza is a specialized San Marzano tomato sauce.\n",
        "question": "What is the most important ingredient for the perfect pizza?",
        "answers": ["san marzano", "tomato"]
    },
    {
        "text": "\nThe CEO of the newly formed AI startup QuantumLeap is Dr. Sarah Jenkins.\n",
        "question": "Who is the CEO of the newly formed AI startup QuantumLeap?",
        "answers": ["sarah jenkins", "jenkins"]
    },
    {
        "text": "\nThe lost ancient artifact was eventually discovered buried under the Eiffel Tower in Paris.\n",
        "question": "Where was the lost ancient artifact eventually discovered?",
        "answers": ["eiffel tower", "paris"]
    }
]

def load_haystack(tokenizer, max_tokens, essays_path):
    essay_files = glob.glob(os.path.join(essays_path, "*.txt"))
    if not essay_files:
        raise FileNotFoundError(f"No essay .txt files found in {essays_path}")
        
    essay_text = ""
    for f in essay_files:
        with open(f, 'r') as file:
            essay_text += file.read() + "\n\n"
            
    while len(tokenizer.encode(essay_text)) < max_tokens + 5000:
        essay_text += essay_text
    
    tokens = tokenizer.encode(essay_text)
    return tokens

def run_single_needle(model, tokenizer, context_length, depth, haystack_tokens, needle_info):
    needle_tokens = tokenizer.encode(needle_info["text"], add_special_tokens=False)
    prompt_suffix = f"\n\nQuestion: {needle_info['question']}\nAnswer:"
    suffix_tokens = tokenizer.encode(prompt_suffix, add_special_tokens=False)
    
    target_haystack_length = context_length - len(needle_tokens)
    context_tokens = haystack_tokens[:target_haystack_length]
    insert_idx = int(target_haystack_length * (depth / 100.0))
    
    final_tokens = context_tokens[:insert_idx] + needle_tokens + context_tokens[insert_idx:] + suffix_tokens
    input_ids = torch.tensor([final_tokens]).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=30, do_sample=False, temperature=None, top_p=None)
        
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    success = any(ans in response for ans in needle_info["answers"])
    return 1.0 if success else 0.0

def run_kamradt_multi_needle(model, tokenizer, head_dim, mode="fp16", bits=4, outlier_fraction=0.10, context_length=100000, depth=50, haystack_tokens=[]):
    # Clear old hooks
    for layer in model.model.layers:
        if hasattr(layer.self_attn.k_proj, '_forward_hooks'):
            layer.self_attn.k_proj._forward_hooks.clear()

    if mode != "fp16":
        quantizers = {}
        for i, layer in enumerate(model.model.layers):
            layer_device = layer.self_attn.k_proj.weight.device
            if mode == "baseline":
                quantizers[i] = TurboQuantMSE(dim=head_dim, bits=bits, device=layer_device)
            elif mode == "outlier":
                quantizers[i] = OutlierAwareTurboQuantMSE(d=head_dim, bits=bits, outlier_fraction=outlier_fraction, device=layer_device)
            
            def make_hook(q_idx):
                def k_proj_hook(module, input, output):
                    orig_shape = output.shape
                    q_device = quantizers[q_idx].device if mode == "baseline" else quantizers[q_idx].base_quantizer.device
                    x = output.view(-1, head_dim).to(q_device)
                    if mode == "baseline":
                        q = quantizers[q_idx].quantize(x)
                        x_rec = quantizers[q_idx].dequantize(q)
                    else:
                        x_rec = quantizers[q_idx].quantize_and_dequantize(x)
                    return x_rec.view(orig_shape).to(output.device).to(output.dtype)
                return k_proj_hook
            layer.self_attn.k_proj.register_forward_hook(make_hook(i))

    scores = []
    for needle in NEEDLES:
        score = run_single_needle(model, tokenizer, context_length, depth, haystack_tokens, needle)
        scores.append(score)
        
    return sum(scores) / len(scores)

def plot_heatmap(data, context_lengths, depths, filename, title_base="Llama-3.1-8B Paul Graham"):
    data_np = np.array(data)
    overall_score = np.mean(data_np) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_np, annot=True, fmt=".1f", cmap="RdYlGn", cbar=False, vmin=0, vmax=1,
                xticklabels=[f"{int(c/1000)}k" for c in context_lengths],
                yticklabels=[f"{d}%" for d in depths])
    plt.title(f"{title_base} (Overall Score: {overall_score:.1f}%)")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("Document Depth (%)")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--essays_path", type=str, required=True, help="Path to Kamradt PaulGrahamEssays folder")
    parser.add_argument("--out_dir", type=str, default="./results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    context_lengths = np.linspace(1000, 100000, num=10, dtype=int).tolist()
    depths = np.linspace(10, 100, num=10, dtype=int).tolist()

    print("Loading Paul Graham Essays...")
    haystack_tokens = load_haystack(tokenizer, max(context_lengths) + 10000, args.essays_path)

    configs = [
        {"mode": "fp16", "bits": 16, "out": 0.0},
        {"mode": "baseline", "bits": 4, "out": 0.0},
        {"mode": "outlier", "bits": 2, "out": 0.10},
    ]

    for config in configs:
        mode = config["mode"]
        bits = config["bits"]
        out_frac = config["out"]
        pct = int(out_frac * 100)
        
        if mode == "fp16":
            name = f"kamradt_fp16"
        elif mode == "baseline":
            name = f"kamradt_baseline_{bits}b"
        else:
            name = f"kamradt_outlier_{bits}b_{pct}out"
            
        print(f"\n--- Running Sweep for {name} ---")
        matrix = []
        for d in depths:
            row = []
            for c in context_lengths:
                score = run_kamradt_multi_needle(model, tokenizer, head_dim, mode=mode, bits=bits, outlier_fraction=out_frac, context_length=c, depth=d, haystack_tokens=haystack_tokens)
                row.append(score)
                print(f"Depth {d}%, Context {c//1000}k: {score:.1f}")
                sys.stdout.flush()
            matrix.append(row)
            
        with open(os.path.join(args.out_dir, f"{name}.txt"), "w") as f:
            for row in matrix:
                f.write(",".join(map(str, row)) + "\n")
        plot_heatmap(matrix, context_lengths, depths, os.path.join(args.out_dir, f"{name}.png"))
