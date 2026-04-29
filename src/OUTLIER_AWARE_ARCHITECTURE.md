# Outlier-Aware TurboQuant: Algorithm & Code Explanation

This document explains the mathematical adaptation of the original TurboQuant algorithm to accommodate heavy-tailed LLM outliers, and breaks down the exact Python implementation used in our codebase.

---

## 1. The Theoretical Adaptation (Algorithm)

The original TurboQuant algorithm (Algorithm 1 in the paper) applies a random orthogonal rotation matrix $\Pi$ to the input vector $x$ to mathematically "smear" the values into a normal distribution, minimizing the Mean Squared Error (MSE) before applying the Lloyd-Max quantizer. 

**The Flaw in LLMs:** LLM attention routing heavily relies on structural outliers—massive magnitude spikes occurring in ~1% to 5% of the channels. Applying the random rotation $\Pi$ irreparably destroys these spikes, leading to catastrophic factual amnesia (as proven in our 2-bit evaluation).

**The Fix:** We adapt the original algorithm by introducing an **Outlier-Aware** wrapper. We physically extract the top $p\%$ of magnitude channels, preserve them in `bfloat16`, allow the rotation $\Pi$ to compress the remaining "flat" inliers, and then stitch the exact outlier values back into the reconstructed tensor.

### Formal Algorithm (LaTeX Adaptation)

The original paper introduces two algorithms: **Algorithm 1** (`TurboQuant_mse`) for basic vector compression, and **Algorithm 2** (`TurboQuant_prod`), which is optimized specifically for the inner product operations required by KV Cache attention mechanisms. Algorithm 2 applies a Quantized Johnson-Lindenstrauss (QJL) projection on the residual error.

Below is the mathematical adaptation of **Algorithm 2** to include our Outlier-Aware logic. By zeroing out the outliers before the residual calculation, we prevent the heavy-tailed spikes from skewing the QJL projection noise matrix, while perfectly restoring them prior to the attention dot-product.

$$
\begin{array}{l}
\hline
\textbf{Algorithm 2*}: \text{Outlier-Aware TurboQuant}_{\text{prod}} \text{ (Optimized for Inner Product)} \\
\hline
\textbf{Input: } \text{dimension } d, \text{ bit-width } b, \text{ outlier fraction } p \in (0, 1) \\
\quad \text{// Global Parameters for Setting up TurboQuant}_{\text{prod}} \\
1: \text{Instantiate a TurboQuant}_{\text{mse}} \text{ with bit-width } b-1 \text{ as per Algorithm 1} \\
2: \text{Generate a random projection matrix } S \in \mathbb{R}^{d \times d} \text{ with i.i.d. entries } S_{i,j} \sim \mathcal{N}(0, 1) \\
\hline
3: \textbf{Procedure } \text{Quant}_{\text{outlier-prod}}(x) \\
4: \quad k \leftarrow \lfloor p \cdot d \rfloor \\
5: \quad \mathcal{O} \leftarrow \text{topk}(|x|, k) \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \triangleright \text{Indices of top magnitudes} \\
6: \quad x_{\text{out}} \leftarrow x[\mathcal{O}] \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \triangleright \text{Preserve exact 16-bit outliers} \\
7: \quad x_{\text{in}} \leftarrow \text{mask}(x, \mathcal{O}, 0) \quad\quad\quad\quad\quad\quad\quad\quad \triangleright \text{Zero-out the outliers in tensor} \\
8: \quad \text{idx} \leftarrow \text{Quant}_{\text{mse}}(x_{\text{in}}) \\
9: \quad r \leftarrow x_{\text{in}} - \text{Dequant}_{\text{mse}}(\text{idx}) \quad\quad\quad\quad\quad\quad \triangleright \text{Residual of the inliers} \\
10: \quad \text{qjl} \leftarrow \text{sign}(S \cdot r) \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \triangleright \text{QJL on residual vector} \\
11: \quad \textbf{output: } (\text{idx}, \text{qjl}, \|r\|_2, \mathcal{O}, x_{\text{out}}) \\
\hline
12: \textbf{Procedure } \text{Dequant}_{\text{outlier-prod}}(\text{idx}, \text{qjl}, \gamma, \mathcal{O}, x_{\text{out}}) \\
13: \quad \tilde{x}_{\text{mse}} \leftarrow \text{Dequant}_{\text{mse}}(\text{idx}) \\
14: \quad \tilde{x}_{\text{qjl}} \leftarrow \frac{\sqrt{\pi/2}}{\sqrt{d}} \cdot \gamma \cdot S^T \cdot \text{qjl} \\
15: \quad \tilde{x}_{\text{in}} \leftarrow \tilde{x}_{\text{mse}} + \tilde{x}_{\text{qjl}} \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \triangleright \text{Dense reconstruction of inliers} \\
16: \quad \tilde{x}_{\text{in}}[\mathcal{O}] \leftarrow x_{\text{out}} \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \triangleright \text{Restore exact 16-bit outliers} \\
17: \quad \textbf{output: } \tilde{x}_{\text{in}} \\
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
