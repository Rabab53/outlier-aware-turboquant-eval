import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'turboquant_lib')))
from turboquant.quantizer import TurboQuantMSE

HF_TOKEN = os.getenv("HF_TOKEN")

class TwoLevelTurboQuantMSE:
    """
    Two-Level TurboQuant Architecture:
    Instead of preserving outliers in 16-bit, we apply a secondary TurboQuant quantization 
    at a higher bit-width (e.g., 4-bit or 6-bit) specifically to the outliers.
    The inliers are aggressively quantized to 2-bit.
    """
    def __init__(self, d, inlier_bits=2, outlier_bits=4, outlier_fraction=0.10, device='cuda'):
        self.d = d
        self.inlier_bits = inlier_bits
        self.outlier_bits = outlier_bits
        self.outlier_fraction = outlier_fraction
        self.num_outliers = int(d * outlier_fraction)
        
        # Instantiate TWO separate TurboQuant algorithm spaces
        self.inlier_quantizer = TurboQuantMSE(dim=d, bits=inlier_bits, device=device)
        self.outlier_quantizer = TurboQuantMSE(dim=d, bits=outlier_bits, device=device)

    def quantize_and_dequantize(self, x):
        if self.num_outliers == 0:
            idx = self.inlier_quantizer.quantize(x)
            return self.inlier_quantizer.dequantize(idx)

        # 1. Identify Outliers
        x_abs = torch.abs(x)
        _, outlier_indices = torch.topk(x_abs, self.num_outliers, dim=-1)
        
        # 2. Create Boolean Masks for pure separation
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_indices, value=True)
        
        # 3. Isolate the tensors
        x_inliers = x.clone()
        x_inliers[mask] = 0.0  # Zero out the outliers
        
        x_outliers = x.clone()
        x_outliers[~mask] = 0.0 # Zero out the inliers
        
        # 4. Level 1: Quantize & Dequantize Inliers (e.g., 2-bit)
        idx_in = self.inlier_quantizer.quantize(x_inliers)
        rec_in = self.inlier_quantizer.dequantize(idx_in)
        
        # 5. Level 2: Quantize & Dequantize Outliers (e.g., 4-bit)
        idx_out = self.outlier_quantizer.quantize(x_outliers)
        rec_out = self.outlier_quantizer.dequantize(idx_out)
        
        # 6. Filter Rotation Noise
        # Because TurboQuant applies orthogonal rotation to dense spaces, quantizing sparse vectors
        # introduces noise into the zeroed spaces during de-rotation. We mathematically filter it out.
        rec_in[mask] = 0.0
        rec_out[~mask] = 0.0
        
        # 7. Reconstruct
        return rec_in + rec_out
