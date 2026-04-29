# Outlier-Aware KV Cache Compression (Needle-in-a-Haystack Evaluator)

This repository contains the core implementation of the **Outlier-Aware Mixed Precision Architecture** and the scripts necessary to evaluate its factual recall using Greg Kamradt's 5-Fact "Needle-In-A-Haystack" testing suite on `Meta-Llama-3.1-8B-Instruct`.

## Architecture Overview
The core algorithm isolates the heavy-tailed activation spikes (outliers) critical to LLM routing geometry and preserves them in uncompressed `bfloat16`, while the mathematically "flat" inliers are heavily compressed to 2-bits or 3-bits using TurboQuant. This achieves extreme memory reduction (up to 30% smaller than 4-bit baselines) without sacrificing 100k context factual recall.

*   **Implementation:** `src/outlier_aware_turboquant.py`

## Getting Started (HPC Cluster Instructions for Team Members)

Since you are testing on the shared Slurm HPC cluster with NVIDIA H200 GPUs using your own individual logins, you can leverage the globally readable shared workspace to avoid rebuilding the environment from scratch.

### 1. Request an Interactive GPU Node & Activate Environment
Do not run this on the login node! Request a compute node and activate the shared virtual environment (hosted under the `ralomairy_tahakom_com` workspace) which already has vLLM, FlashInfer, and PyTorch installed.

```bash
# Request an interactive node
srun --partition=nodes13 --nodes=1 --gres=gpu:1 --time=02:00:00 --pty bash

# Activate the shared bare-metal environment (Read-Only access is sufficient)
source /mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/vllm_venv/bin/activate
```

### 2. Clone the Repositories to Your Home Directory
Clone this evaluator repository and Greg Kamradt's official essay dataset into your own home directory (`~`).

```bash
cd ~

# Clone this repository
git clone https://github.com/Rabab53/outlier-aware-turboquant-eval.git
cd outlier-aware-turboquant-eval

# Clone the Paul Graham essay dataset inside the folder
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git
```

### 3. Set Environment Variables
Point the Python path to the base `turboquant_lib` on the shared Lustre drive, and export your HuggingFace token to download the Llama-3.1 model.

```bash
export PYTHONPATH="/mnt/lustre/lustre-in-us-central1-b/team_workspace/ralomairy_tahakom_com/turboquant_lib:$PYTHONPATH"
export HF_TOKEN="your_huggingface_token_here"
```

### 4. Run the Evaluation
Execute the Kamradt evaluation script. It will automatically intercept the Llama-3.1 KV cache, compress it using our Outlier-Aware architecture, and output the fractional heatmaps.

```bash
python eval/run_kamradt_eval.py \
    --essays_path ./LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays \
    --out_dir ./results
```

The script will automatically print the scores to your console and drop the raw `.txt` arrays and the colorized `.png` heatmaps (complete with the overall average scores) into your local `./results` folder.
