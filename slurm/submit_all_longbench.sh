#!/bin/bash
cd ~/outlier-aware-turboquant-eval

CONFIGS=(
    "fp16 16 0.0"
    "baseline 4 0.0"
    "baseline 3 0.0"
    "baseline 2 0.0"
    "outlier 4 0.05"
    "outlier 3 0.05"
    "outlier 2 0.05"
    "outlier 4 0.10"
    "outlier 3 0.10"
    "outlier 2 0.10"
)

for conf in "${CONFIGS[@]}"; do
    read -r mode bits out <<< "$conf"
    pct=$(echo "$out * 100" | bc | cut -d'.' -f1)
    job_name="LB_${mode}_${bits}b_${pct}out"
    sbatch --job-name="$job_name" --output="slurm/logs/${job_name}_%j.out" slurm/run_longbench_single.slurm "$mode" "$bits" "$out"
done
echo "Submitted 10 LongBench jobs!"
