# Outlier-Aware KV Cache Compression (Needle-in-a-Haystack Evaluator)

This repository contains the core implementation of the **Outlier-Aware Mixed Precision Architecture** and the scripts necessary to evaluate its factual recall using Greg Kamradt's 5-Fact "Needle-In-A-Haystack" testing suite on `Meta-Llama-3.1-8B-Instruct`.

## Architecture Overview
The core algorithm isolates the heavy-tailed activation spikes (outliers) critical to LLM routing geometry and preserves them in uncompressed `bfloat16`, while the mathematically "flat" inliers are heavily compressed to 2-bits or 3-bits using TurboQuant. This achieves extreme memory reduction (up to 30% smaller than 4-bit baselines) without sacrificing 100k context factual recall.

*   **Implementation:** `src/outlier_aware_turboquant.py`

## Getting Started

### 1. Prerequisites
You must have the baseline `turboquant` module installed in your environment, as our architecture extends `TurboQuantMSE`.

```bash
pip install torch transformers matplotlib seaborn
```

### 2. Download the Dataset
You need to clone Greg Kamradt's repository to access the Paul Graham essays used as the testing haystack.

```bash
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git
```

### 3. Run the Evaluation
The evaluation script takes the path to the Paul Graham essays and generates full 10x10 heatmaps for FP16, Original TurboQuant, and Outlier-Aware TurboQuant.

```bash
python eval/run_kamradt_eval.py \
    --essays_path ./LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays \
    --out_dir ./results
```

The script will automatically print the scores to the console and drop the raw `.txt` arrays and the colorized `.png` heatmaps (complete with the overall average scores) into the `./results` folder.
