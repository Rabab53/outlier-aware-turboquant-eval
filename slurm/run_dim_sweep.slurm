#!/bin/bash
#SBATCH --job-name=dim-sweep
#SBATCH --partition=nodes13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/dim_sweep_%j.out

# Activate shared environment
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate

# Add turboquant_lib to PYTHONPATH
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"

cd ~/outlier-aware-turboquant-eval

echo "Starting Dimension Scaling Sweep..."
python eval/dimension_sweep_benchmark.py
