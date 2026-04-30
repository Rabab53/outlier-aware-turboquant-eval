#!/bin/bash
#SBATCH --job-name=two-level
#SBATCH --partition=nodes13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/two_level_llama_%j.out

cd ~/outlier-aware-turboquant-eval
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"

# export HF_TOKEN="your_huggingface_token_here"

echo "Starting Two-Level TurboQuant Llama-3.1-8B Needle-in-a-Haystack Evaluation..."
python eval/run_kamradt_two_level.py \
    --essays_path ./LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays \
    --out_dir ./results \
    --model_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --max_context 100000
