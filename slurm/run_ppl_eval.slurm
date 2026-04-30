#!/bin/bash
#SBATCH --job-name=ppl-eval
#SBATCH --partition=nodes13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/ppl_eval_%j.out

cd ~/outlier-aware-turboquant-eval
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"

# IMPORTANT: Uncomment the line below and add your token, or export it in your terminal before running sbatch
# export HF_TOKEN="your_huggingface_token_here"

echo "Starting Perplexity (PPL) Evaluation on WikiText-2..."
python eval/run_ppl_eval.py \
    --model_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --mode "outlier" \
    --bits 2 \
    --out_frac 0.10 \
    --out_dir ./results
