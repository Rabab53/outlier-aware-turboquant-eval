import os
import json
import argparse
import sys

# Assume the user cloned THUDM/LongBench to their home directory or parallel to this project
default_longbench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "LongBench", "LongBench"))

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="./results", help="Directory containing the longbench_* result folders")
parser.add_argument("--longbench_repo_path", type=str, default=default_longbench_path, help="Path to the official THUDM/LongBench/LongBench directory for metrics")
args = parser.parse_args()

sys.path.append(args.longbench_repo_path)
try:
    from metrics import (
        qa_f1_score,
        rouge_score,
        classification_score,
        retrieval_score,
        count_score,
        code_sim_score,
    )
except ImportError:
    print(f"Error: Could not import metrics from {args.longbench_repo_path}")
    print("Please clone https://github.com/THUDM/LongBench and provide its path via --longbench_repo_path")
    sys.exit(1)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

ENGLISH_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p"
]

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

configs = [
    "longbench_fp16_16b_0out",
    "longbench_baseline_4b_0out",
    "longbench_baseline_3b_0out",
    "longbench_baseline_2b_0out",
    "longbench_outlier_4b_5out",
    "longbench_outlier_3b_5out",
    "longbench_outlier_2b_5out",
    "longbench_outlier_4b_10out",
    "longbench_outlier_3b_10out",
    "longbench_outlier_2b_10out",
    "longbench_twolevel_in4b_out4b_10out",
    "longbench_twolevel_in3b_out4b_10out",
    "longbench_twolevel_in2b_out4b_10out"
]

results = {}

for config in configs:
    config_path = os.path.join(args.results_dir, config)
    results[config] = {}
    
    if not os.path.exists(config_path):
        continue
        
    for dataset in ENGLISH_DATASETS:
        file_path = os.path.join(config_path, f"{dataset}.jsonl")
        if not os.path.exists(file_path):
            continue
            
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        if not data:
            continue
            
        predictions = [d["pred"] for d in data]
        answers = [d["answers"] for d in data]
        all_classes = data[0].get("all_classes", None)
        
        score = scorer(dataset, predictions, answers, all_classes)
        results[config][dataset] = score

categories = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"]
}

print("\n| Model Configuration | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot | Synthetic | Code | **Average** |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for config in configs:
    if config not in results or not results[config]:
        continue
    
    row_str = f"| **{config.replace('longbench_', '')}** | "
    cat_scores = []
    
    for cat_name, datasets in categories.items():
        cat_total = 0
        cat_count = 0
        for d in datasets:
            if d in results[config]:
                cat_total += results[config][d]
                cat_count += 1
        if cat_count > 0:
            avg = cat_total / cat_count
            row_str += f"{avg:.1f} | "
            cat_scores.append(avg)
        else:
            row_str += "- | "
            
    if cat_scores:
        overall_avg = sum(cat_scores) / len(cat_scores)
        row_str += f"**{overall_avg:.1f}** |"
    else:
        row_str += "- |"
        
    print(row_str)
