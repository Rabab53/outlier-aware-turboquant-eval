import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from outlier_aware_turboquant import OutlierAwareTurboQuantMSE

try:
    from turboquant.quantizer import TurboQuantMSE
except ImportError:
    pass # Assume it is in PYTHONPATH from the cluster

def build_chat(tokenizer, prompt, max_length):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def get_pred(model, tokenizer, data, max_length, max_gen, dataset_name, dataset2prompt):
    preds = []
    prompt_template = dataset2prompt.get(dataset_name, "{context}\n{input}")
    
    for d in tqdm(data, desc=f"Evaluating {dataset_name}"):
        # LongBench prompt template injection
        prompt = prompt_template.format(**d)
        
        input_text = build_chat(tokenizer, prompt, max_length)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=None,
                top_p=None
            )
        
        out_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        preds.append({
            "id": d.get("_id", d.get("id", str(np.random.randint(10000)))), 
            "pred": out_text, 
            "answers": d["answers"], 
            "all_classes": d.get("all_classes", None), 
            "length": d.get("length", 0)
        })
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["fp16", "baseline", "outlier"])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out_frac", type=float, default=0.10)
    args = parser.parse_args()

    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Grouped by Table 1 categories in paper
    ENGLISH_DATASETS = [
        "narrativeqa", "qasper", "multifieldqa_en", # SingleQA
        "hotpotqa", "2wikimqa", "musique",          # MultiQA
        "gov_report", "qmsum", "multi_news",        # Summarization
        "trec", "triviaqa", "samsum",               # Few-shot
        "passage_count", "passage_retrieval_en",    # Synthetic
        "lcc", "repobench-p"                        # Code
    ]

    # Load prompt formats from LongBench
    try:
        with open("/home/ralomairy_tahakom_com/LongBench/config/dataset2prompt.json", "r") as f:
            dataset2prompt = json.load(f)
        with open("/home/ralomairy_tahakom_com/LongBench/config/dataset2maxlen.json", "r") as f:
            dataset2maxlen = json.load(f)
    except:
        dataset2prompt = {}
        dataset2maxlen = {}

    out_dir = f"/home/ralomairy_tahakom_com/outlier-aware-turboquant-eval/results/longbench_{args.mode}_{args.bits}b_{int(args.out_frac*100)}out"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Hooking model for {args.mode} {args.bits}b {args.out_frac} ---")
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

    for dataset_name in ENGLISH_DATASETS:
        save_path = os.path.join(out_dir, f"{dataset_name}.jsonl")
        if os.path.exists(save_path):
            print(f"Skipping {dataset_name}, already exists.")
            continue

        print(f"Testing {dataset_name}...")
        data = []
        file_path = f"/home/ralomairy_tahakom_com/LongBench/dataset/data/{dataset_name}.jsonl"
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        # We process ALL samples to exactly replicate the paper's results
        # Max lengths: Llama-3.1-8B supports 128k, but we use LongBench's dataset specific max lengths or 100k
        max_length = 120000 
        dataset_maxlen = dataset2maxlen.get(dataset_name, 31500)
        # We let it use the maximum available up to 120k to fully replicate the long context scenario
        
        # Gen limits based on task
        max_gen = 64
        if dataset_name in ["gov_report", "qmsum", "multi_news"]:
            max_gen = 512 # Summarization needs longer outputs
        
        preds = get_pred(model, tokenizer, data, max_length, max_gen, dataset_name, dataset2prompt)
        
        with open(save_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
