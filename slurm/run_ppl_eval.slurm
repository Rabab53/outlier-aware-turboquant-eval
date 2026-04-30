#!/bin/bash
#SBATCH --partition=nodes13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Usage: sbatch --job-name=ppl-eval --output=slurm/logs/ppl_eval_%j.out slurm/run_ppl_eval.slurm <model_id> <mode> <bits> <out_frac>
# Example: sbatch slurm/run_ppl_eval.slurm "unsloth/Meta-Llama-3.1-8B-Instruct" "outlier" 2 0.10

MODEL_ID=${1:-"unsloth/Meta-Llama-3.1-8B-Instruct"}
MODE=${2:-"outlier"}
BITS=${3:-2}
OUT_FRAC=${4:-0.10}

cd ~/outlier-aware-turboquant-eval
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"

# IMPORTANT: Uncomment the line below and add your token, or export it in your terminal before running sbatch
# export HF_TOKEN="your_huggingface_token_here"

echo "Starting Perplexity (PPL) Evaluation on WikiText-2..."
echo "Model: $MODEL_ID | Mode: $MODE | Bits: $BITS | OutFrac: $OUT_FRAC"

python eval/run_ppl_eval.py \
    --model_id "$MODEL_ID" \
    --mode "$MODE" \
    --bits $BITS \
    --out_frac $OUT_FRAC \
    --out_dir ./results
