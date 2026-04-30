import os
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from two_level_turboquant import TwoLevelTurboQuantMSE

def build_chat(tokenizer, prompt, max_length, dataset_name):
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
    if dataset_name in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
        return prompt
        
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def get_pred(model, tokenizer, data, max_length, max_gen, dataset_name, dataset2prompt):
    preds = []
    prompt_template = dataset2prompt.get(dataset_name, "{context}\n{input}")
    
    for d in tqdm(data, desc=f"Evaluating {dataset_name}"):
        prompt = prompt_template.format(**d)
        
        input_text = build_chat(tokenizer, prompt, max_length, dataset_name)
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
    parser.add_argument("--inlier_bits", type=int, required=True)
    parser.add_argument("--outlier_bits", type=int, required=True)
    parser.add_argument("--out_frac", type=float, default=0.10)
    args = parser.parse_args()

    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("Could not locate the transformer layers list.")

        if hasattr(model.config, "head_dim"):
            head_dim = model.config.head_dim
        elif hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
            head_dim = model.config.hidden_size // model.config.num_attention_heads
        else:
            k_proj_shape = layers[0].self_attn.k_proj.weight.shape
            head_dim = k_proj_shape[0] // model.config.num_key_value_heads
    except Exception as e:
        print(f"Warning: Could not automatically determine head_dim from config. Defaulting to 128. Error: {e}")
        head_dim = 128

    ENGLISH_DATASETS = [
        "narrativeqa", "qasper", "multifieldqa_en",
        "hotpotqa", "2wikimqa", "musique",
        "gov_report", "qmsum", "multi_news",
        "trec", "triviaqa", "samsum",
        "passage_count", "passage_retrieval_en",
        "lcc", "repobench-p"
    ]

    try:
        with open("/home/ralomairy_tahakom_com/LongBench/LongBench/config/dataset2prompt.json", "r") as f:
            dataset2prompt = json.load(f)
        with open("/home/ralomairy_tahakom_com/LongBench/LongBench/config/dataset2maxlen.json", "r") as f:
            dataset2maxlen = json.load(f)
    except:
        dataset2prompt = {}
        dataset2maxlen = {}

    out_dir = f"/home/ralomairy_tahakom_com/outlier-aware-turboquant-eval/results/longbench_twolevel_in{args.inlier_bits}b_out{args.outlier_bits}b_{int(args.out_frac*100)}out"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Hooking model for Two-Level: In={args.inlier_bits}b, Out={args.outlier_bits}b, {args.out_frac} ---")
    for layer in layers:
        if hasattr(layer.self_attn.k_proj, '_forward_hooks'):
            layer.self_attn.k_proj._forward_hooks.clear()

    quantizers = {}
    for i, layer in enumerate(layers):
        layer_device = layer.self_attn.k_proj.weight.device
        quantizers[i] = TwoLevelTurboQuantMSE(d=head_dim, inlier_bits=args.inlier_bits, outlier_bits=args.outlier_bits, outlier_fraction=args.out_frac, device=layer_device)
        
        def make_hook(q_idx):
            def k_proj_hook(module, input, output):
                orig_shape = output.shape
                q_device = quantizers[q_idx].inlier_quantizer.device
                x = output.view(-1, head_dim).to(q_device)
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
        
        max_length = 120000 
        max_gen = dataset2maxlen.get(dataset_name, 128)
        
        preds = get_pred(model, tokenizer, data, max_length, max_gen, dataset_name, dataset2prompt)
        
        with open(save_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
