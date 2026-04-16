"""
ts_quant.utils.feature_selection — Built-in Feature Selection
==============================================================

Mengurangi fitur redundan agar model CatBoost tidak overfitting.

Metode:
    1. Variance Threshold: buang fitur dengan variance ≈ 0
    2. Correlation Filter: buang salah satu dari pasangan fitur
       yang berkorelasi > threshold (keep yang lebih berkorelasi
       dengan target jika target tersedia)
    3. NaN/Inf Filter: buang fitur yang terlalu banyak NaN
"""

from typing import List, Optional, Tuple

import torch
import numpy as np
import pyarrow.parquet as pq

def auto_select_features_from_parquet(
    parquet_path: str,
    variance_threshold: float = 1e-8,
    correlation_threshold: float = 0.95,
    max_nan_ratio: float = 0.5,
    verbose: bool = True,
    device: str = 'cpu',
) -> List[str]:
    """
    Out-of-Core Pipeline seleksi fitur otomatis (ZERO-CORE RAM SPIKE).
    
    Membaca data dari file Parquet secara Streaming.
    1. Hapus NaN features
    2. Hapus constant features
    3. Hapus correlated features (Chunked Covariance)
    """
    pf = pq.ParquetFile(parquet_path)
    B = pf.metadata.num_rows
    N = pf.metadata.num_columns
    names = pf.schema.names
    
    if verbose:
        print(f"    [FeatureSelection] Starting Out-of-Core selection for {N} features...")
    
    # ── PASS 1: Stream for NaN and Variance ──
    nan_count = torch.zeros(N, dtype=torch.int64, device=device)
    sum_x = torch.zeros(N, dtype=torch.float64, device=device)
    sum_x2 = torch.zeros(N, dtype=torch.float64, device=device)
    valid_count = torch.zeros(N, dtype=torch.float64, device=device)
    
    # Read Parquet in chunks of 50k rows to keep memory usage under 200MB
    for batch in pf.iter_batches(batch_size=50000):
        df_batch = batch.to_pandas()
        features = torch.tensor(df_batch.values, device=device, dtype=torch.float32)
        
        batch_nan = torch.isnan(features)
        nan_count += batch_nan.sum(dim=0)
        
        batch_clean = torch.where(batch_nan, torch.zeros_like(features), features).to(torch.float64)
        valid_batch_count = (~batch_nan).sum(dim=0).to(torch.float64)
        valid_count += valid_batch_count
        
        sum_x += batch_clean.sum(dim=0)
        sum_x2 += (batch_clean ** 2).sum(dim=0)
        
        del df_batch, features, batch_nan, batch_clean
        
    # 1. NaN mask
    nan_ratio = nan_count.float() / float(max(B, 1))
    mask_nan = nan_ratio <= max_nan_ratio
    n_after_nan = mask_nan.sum().item()
    
    # 2. Constant mask
    var = torch.zeros_like(sum_x)
    valid_mask_var = valid_count > 1
    var[valid_mask_var] = (sum_x2[valid_mask_var] - (sum_x[valid_mask_var] ** 2) / valid_count[valid_mask_var]) / (valid_count[valid_mask_var] - 1)
    
    mask_const = var > variance_threshold
    valid_mask = mask_nan & mask_const
    n_after_const = valid_mask.sum().item()
    
    # ── PASS 2: Correlation Filter (Stream only VALID features) ──
    valid_idx = torch.where(valid_mask)[0]
    kept_names = []
    
    if len(valid_idx) > 1:
        M = len(valid_idx)
        sum_x_cov = torch.zeros(M, dtype=torch.float64, device=device)
        sum_x2_cov = torch.zeros((M, M), dtype=torch.float64, device=device)
        
        valid_names = [names[i.item()] for i in valid_idx]
        
        for batch in pf.iter_batches(batch_size=50000, columns=valid_names):
            df_batch = batch.to_pandas()
            features = torch.tensor(df_batch.values, device=device, dtype=torch.float64)
            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
            
            sum_x_cov += features.sum(dim=0)
            sum_x2_cov += features.T @ features
            del df_batch, features
            
        # Covariance & Correlation Matrix
        cov = (sum_x2_cov - torch.outer(sum_x_cov, sum_x_cov) / B) / (B - 1)
        std = torch.sqrt(torch.diag(cov).clamp(min=1e-10))
        corr = cov / torch.outer(std, std)
        
        keep_indices = []
        for i in range(M):
            correlated = False
            for j in keep_indices:
                if abs(corr[i, j].item()) > correlation_threshold:
                    correlated = True
                    break
            if not correlated:
                keep_indices.append(i)
                
        kept_names = [valid_names[i] for i in keep_indices]
    else:
        kept_names = [names[i.item()] for i in valid_idx]
        
    n_after_corr = len(kept_names)

    if verbose:
        print(f"    [FeatureSelection] {N} -> {n_after_nan} (NaN) "
              f"-> {n_after_const} (constant) -> {n_after_corr} (correlation)")

    return kept_names
