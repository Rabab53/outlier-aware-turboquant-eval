#!/bin/bash
cd ~/outlier-aware-turboquant-eval

CONFIGS=(
    "4 4 0.10"
    "3 4 0.10"
    "2 4 0.10"
)

for conf in "${CONFIGS[@]}"; do
    read -r in_bits out_bits frac <<< "$conf"
    pct=$(echo "$frac * 100" | bc | cut -d'.' -f1)
    job_name="LB_TL_in${in_bits}b_out${out_bits}b_${pct}out"
    sbatch --job-name="$job_name" --output="slurm/logs/${job_name}_%j.out" slurm/run_longbench_two_level.slurm "$in_bits" "$out_bits" "$frac"
done
echo "Submitted 3 Two-Level LongBench jobs!"
