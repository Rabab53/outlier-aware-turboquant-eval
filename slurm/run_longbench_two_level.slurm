#!/bin/bash
#SBATCH --partition=nodes13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

cd ~/outlier-aware-turboquant-eval
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"

echo "Running LongBench Two-Level Config: InlierBits=$1 OutlierBits=$2 OutFrac=$3"
python eval/run_longbench_two_level.py --inlier_bits $1 --outlier_bits $2 --out_frac $3
