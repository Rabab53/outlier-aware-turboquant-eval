import torch
import sys
import os

# Ensure turboquant is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'turboquant_lib')))
from turboquant.quantizer import TurboQuantMSE

class UnifiedOptimizedQuantizer:
    def __init__(self, d, mode="baseline", bits=2, outlier_bits=4, outlier_fraction=0.10, device='cuda'):
        self.d = d
        self.mode = mode
        self.bits = bits
        self.outlier_bits = outlier_bits
        self.num_outliers = int(d * outlier_fraction)
        self.num_inliers = d - self.num_outliers
        
        # Pre-allocate needed quantizers
        if mode == "baseline":
            self.q1 = TurboQuantMSE(dim=d, bits=bits, device=device)
        elif mode == "outlier":
            self.q1 = TurboQuantMSE(dim=self.num_inliers, bits=bits, device=device)
        elif mode == "two_level":
            self.q1 = TurboQuantMSE(dim=self.num_inliers, bits=bits, device=device)
            self.q2 = TurboQuantMSE(dim=self.num_outliers, bits=outlier_bits, device=device)
        elif mode == "fp16":
            pass # No quantizer needed
            
        self.perm = None
        self.inv_perm = None

    def _prepare_permutation(self, x_flat):
        if self.num_outliers == 0:
            self.perm = torch.arange(self.d, device=x_flat.device)
            self.inv_perm = self.perm
            return
            
        channel_max = x_flat.abs().max(dim=0).values
        _, outlier_indices = torch.topk(channel_max, self.num_outliers)
        is_outlier = torch.zeros(self.d, dtype=torch.bool, device=x_flat.device)
        is_outlier[outlier_indices] = True
        inlier_indices = torch.where(~is_outlier)[0]
        self.perm = torch.cat([inlier_indices, outlier_indices])
        self.inv_perm = torch.argsort(self.perm)

    def quantize_and_dequantize(self, x):
        shape = x.shape
        x_flat = x.view(-1, self.d).float()
        
        # 1. FP16 Baseline (Optimized No-Op / Identity)
        if self.mode == "fp16":
            return x_flat.view(shape).to(x.dtype)
            
        # 2. Pure Baseline (No outliers)
        if self.mode == "baseline":
            return self.q1.dequantize(self.q1.quantize(x_flat)).view(shape).to(x.dtype)

        # Outlier handling logic starts here
        if self.perm is None: self._prepare_permutation(x_flat)
        
        # Fair Permutation logic for all outlier-aware modes
        x_perm = x_flat[:, self.perm]
        x_in = x_perm[:, :self.num_inliers]
        x_out = x_perm[:, self.num_inliers:]
        
        if self.mode == "outlier":
            # Outlier-Aware (Inlier bits + FP16 outliers)
            rec_in = self.q1.dequantize(self.q1.quantize(x_in))
            rec_out = x_out 
            rec_perm = torch.cat([rec_in, rec_out], dim=-1)
            
        elif self.mode == "two_level":
            # Two-Level (Inlier bits + Outlier bits)
            rec_in = self.q1.dequantize(self.q1.quantize(x_in))
            rec_out = self.q2.dequantize(self.q2.quantize(x_out))
            rec_perm = torch.cat([rec_in, rec_out], dim=-1)
            
        return rec_perm[:, self.inv_perm].view(shape).to(x.dtype)
