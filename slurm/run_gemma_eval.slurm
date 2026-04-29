#!/bin/bash
#SBATCH --job-name=gemma-eval
#SBATCH --partition=nodes13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=gemma_eval_%j.out

# Activate shared environment
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"

# IMPORTANT: Uncomment the line below and add your token, or export it in your terminal before running sbatch
# export HF_TOKEN="your_huggingface_token_here"

cd ~/outlier-aware-turboquant-eval
echo "Starting Gemma-4-26B-A4B-it Needle-in-a-Haystack Evaluation..."
python eval/run_kamradt_eval.py \
    --essays_path ./LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays \
    --out_dir ./results \
    --model_id "unsloth/gemma-4-26B-A4B-it" \
    --max_context 32000
