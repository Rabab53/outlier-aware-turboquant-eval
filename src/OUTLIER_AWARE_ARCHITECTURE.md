# Outlier-Aware TurboQuant: Algorithm & Code Explanation

This document explains the mathematical adaptation of the original TurboQuant algorithm to accommodate heavy-tailed LLM outliers, and breaks down the exact Python implementation used in our codebase.

---

## 1. The Theoretical Adaptation (Algorithm)

The original TurboQuant algorithm (Algorithm 1 in the paper) applies a random orthogonal rotation matrix $\Pi$ to the input vector $x$ to mathematically "smear" the values into a normal distribution, minimizing the Mean Squared Error (MSE) before applying the Lloyd-Max quantizer. 

**The Flaw in LLMs:** LLM attention routing heavily relies on structural outliers—massive magnitude spikes occurring in ~1% to 5% of the channels. Applying the random rotation $\Pi$ irreparably destroys these spikes, leading to catastrophic factual amnesia (as proven in our 2-bit evaluation).

**The Fix:** We adapt the original algorithm by introducing an **Outlier-Aware** wrapper. We physically extract the top $p\%$ of magnitude channels, preserve them in `bfloat16`, allow the rotation $\Pi$ to compress the remaining "flat" inliers, and then stitch the exact outlier values back into the reconstructed tensor.

### Formal Algorithm (LaTeX Adaptation)

Below is the mathematical adaptation of the paper's Algorithm 1 to include our Outlier-Aware logic:

$$
\begin{array}{l}
\hline
\textbf{Algorithm 1*} \text{: Outlier-Aware } \textsc{TurboQuant}_{\text{mse}} \\
\hline
\textbf{Input: } \text{dimension } d, \text{ bit-width } b, \text{ outlier fraction } p \in (0, 1) \\
\quad \text{// Global Parameters for Setting up } \textsc{TurboQuant}_{\text{mse}} \\
1: \text{Generate a random rotation matrix } \Pi \in \mathbb{R}^{d \times d} \\
2: \text{Construct codebook by finding centroids } c_1, c_2, \dots c_{2^b} \in [-1, 1] \text{ that minimize MSE} \\
\\
\hline
3: \textbf{Procedure } \textsc{Quant}_{\text{outlier}}(x) \\
4: \quad k \leftarrow \lfloor p \cdot d \rfloor \\
5: \quad \mathcal{O} \leftarrow \arg\text{topk}(|x|, k) \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \; \text{\{Indices of top magnitudes\}} \\
6: \quad x_{\text{out}} \leftarrow x[\mathcal{O}] \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text{\{Preserve exact 16-bit outliers\}} \\
7: \quad \text{idx} \leftarrow \textsc{Quant}_{\text{mse}}(x) \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text{\{Apply original quantization\}} \\
8: \quad \textbf{output: } (\text{idx}, \mathcal{O}, x_{\text{out}}) \\
\\
\hline
9: \textbf{Procedure } \textsc{Dequant}_{\text{outlier}}(\text{idx}, \mathcal{O}, x_{\text{out}}) \\
10: \quad \tilde{x} \leftarrow \textsc{Dequant}_{\text{mse}}(\text{idx}) \quad \quad \quad \quad \quad \quad \quad \quad \quad \text{\{Dense dense reconstruction\}} \\
11: \quad \tilde{x}[\mathcal{O}] \leftarrow x_{\text{out}} \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text{\{Restore exact 16-bit outliers\}} \\
12: \quad \textbf{output: } \tilde{x} \\
\hline
\end{array}
$$

---

## 2. Code Breakdown: `outlier_aware_turboquant.py`

The file `src/outlier_aware_turboquant.py` contains the Python implementation of the algorithm above. Here is a detailed, step-by-step explanation of how the `OutlierAwareTurboQuantMSE` class works under the hood during the live LLM generation:

### Initialization
```python
class OutlierAwareTurboQuantMSE:
    def __init__(self, d, bits=4, outlier_fraction=0.05, device='cuda'):
        self.d = d
        self.bits = bits
        self.outlier_fraction = outlier_fraction
        self.num_outliers = int(d * outlier_fraction)
        
        # Instantiate the original Google TurboQuant algorithm
        self.base_quantizer = TurboQuantMSE(dim=d, bits=bits, device=device)
```
*   **What it does:** It calculates exactly how many channels constitute the top $p\%$ (e.g., 10% of a 4096-dim vector is 409 channels). It then initializes the base `TurboQuantMSE` which contains the orthogonal rotation matrix $\Pi$ and the Lloyd-Max codebooks.

### The Forward Hook (Quantize & Dequantize)
Because we are evaluating this "on the fly" during Llama-3.1 inference, we perform the quantization and dequantization back-to-back inside the PyTorch forward hook.

```python
    def quantize_and_dequantize(self, x):
        # 1. Identify Outliers
        # Calculate the absolute magnitude of all channels
        x_abs = torch.abs(x)
        
        # Find the indices of the highest magnitude channels (The Outliers)
        outlier_vals, outlier_indices = torch.topk(x_abs, self.num_outliers, dim=-1)
        
        # 2. Extract Exact Values
        # Use torch.gather to pull the exact bfloat16 values out of the tensor using the indices
        exact_outliers = torch.gather(x, dim=-1, index=outlier_indices)
```
*   **Line 4-10:** This directly implements Lines 4-6 of the algorithm. We use `torch.topk` to find the geometry spikes, and `torch.gather` to pull their exact 16-bit values to safety.

```python
        # 3. Base Quantization
        # Quantize the entire tensor using the dense rotation matrix
        idx = self.base_quantizer.quantize(x)
        
        # Dequantize the tensor back to bfloat16 (The outliers are now corrupted/smeared)
        x_rec = self.base_quantizer.dequantize(idx)
```
*   **Line 12-16:** This implements Line 7 and Line 10. We let the original algorithm do its math. The result (`x_rec`) is highly compressed, but its routing spikes have been mathematically destroyed by the rotation noise.

```python
        # 4. Restitch the Geometry
        # Scatter the exact, uncorrupted bfloat16 outliers back into their original indices
        x_rec.scatter_(dim=-1, index=outlier_indices, src=exact_outliers)
        
        return x_rec
```
*   **Line 18-22:** This is the magic step (Line 11). We use PyTorch's inplace `.scatter_()` function to mathematically overwrite the corrupted noise spikes with the exact `bfloat16` values we saved earlier. 

The resulting tensor `x_rec` is now ready to be fed into the LLM's Attention mechanism, with its structural geometry perfectly preserved and its memory footprint drastically reduced!
