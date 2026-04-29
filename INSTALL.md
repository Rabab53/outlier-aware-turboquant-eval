# Installation Guide (External Machines)

If you are setting up this evaluation pipeline on an external machine, local workstation, or a completely different cloud provider (outside of our shared HPC cluster), follow these steps to build the environment from scratch.

## Prerequisites
* **OS:** Linux (Ubuntu 20.04/22.04 recommended)
* **GPU:** NVIDIA GPU with at least 24GB VRAM (for 7B models) or 80GB+ (for larger MoE / 100k context tests)
* **Drivers:** CUDA Toolkit 12.1 or newer
* **Python:** Python 3.10+

---

### Step 1: Clone the Repository
Clone this evaluation repository to your local workspace:
```bash
git clone https://github.com/Rabab53/outlier-aware-turboquant-eval.git
cd outlier-aware-turboquant-eval
```

### Step 2: Install Python Dependencies
We provide both `pip` and `conda` configurations. We highly recommend using Conda to cleanly manage the PyTorch CUDA bindings.

**Option A: Using Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate turboquant-eval
```

**Option B: Using Pip**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Link the Base `turboquant` Library
Because our Outlier-Aware architecture extends the baseline `TurboQuantMSE` logic, you must have the base turboquant library available on your machine.

1. Securely copy the `turboquant_lib` folder from the internal team storage to your external machine.
2. Export the directory to your Python path so the evaluator can import it:
```bash
# Replace this path with wherever you placed the base library folder
export PYTHONPATH="/path/to/your/local/turboquant_lib:$PYTHONPATH"
```

### Step 4: Download the Evaluation Dataset
We use Greg Kamradt's official Paul Graham essays as the testing haystack. Clone it inside or next to this repository:
```bash
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git
```

### Step 5: Authenticate with HuggingFace
To automatically download the gated LLMs (like Llama-3.1 or Mistral), you must provide your HuggingFace access token.
```bash
export HF_TOKEN="your_huggingface_token_here"
```

---

### Step 6: Verify the Installation
Run a quick test to ensure everything is hooked up correctly and your GPU has sufficient memory. 

**Test Mistral (Requires ~24GB VRAM):**
```bash
python eval/run_kamradt_eval.py     --essays_path ./LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays     --out_dir ./results     --model_id "mistralai/Mistral-7B-Instruct-v0.3"     --max_context 32000
```

**Test Llama 3.1 (Requires ~80GB+ VRAM for 100k context):**
```bash
python eval/run_kamradt_eval.py     --essays_path ./LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays     --out_dir ./results     --model_id "unsloth/Meta-Llama-3.1-8B-Instruct"     --max_context 100000
```
If it successfully loads the model and begins printing out fractional scores (e.g., `Depth 10%, Context 3k: 1.0`), your external machine is fully configured!
