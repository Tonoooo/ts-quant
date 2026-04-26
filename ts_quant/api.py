"""
ts_quant.api — Orchestrator: generate_features()
==================================================

Fungsi utama library ts-quant. Menerima DataFrame OHLCV,
menjalankan semua engine, dan mengembalikan DataFrame
dengan fitur baru.

Alur:
    1. Validasi input
    2. DataFrame → Tensor 3D
    3. Rolling Windows
    4. Jalankan 5 Engine (Rocket, Catch22, Signatures, Wavelets, Tsfresh)
    5. Feature Selection
    6. Tensor → DataFrame
    7. Gabung dengan DataFrame asli
"""

import gc
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from ts_quant.core.memory_manager import VRAMManager
from ts_quant.core.tensor_utils import df_to_tensor_3d, get_per_stock_tensors
from ts_quant.core.windowing import (
    create_rolling_windows,
    create_rolling_windows_batch,
    unbatch_windows,
)
from ts_quant.utils.validation import validate_dataframe
from ts_quant.utils.feature_selection import auto_select_features_from_parquet

from ts_quant.engines.rocket import RocketEngine
from ts_quant.engines.catch22 import Catch22Engine
from ts_quant.engines.signatures import SignaturesEngine
from ts_quant.engines.wavelets import WaveletsEngine
from ts_quant.engines.tsfresh_core import TsfreshEngine


def generate_features(
    df: pd.DataFrame,
    # ── Device ──
    device: str = 'cuda',
    max_vram_gb: float = None,
    # ── Engine switches ──
    enable_rocket: bool = True,
    enable_catch22: bool = True,
    enable_signatures: bool = True,
    enable_wavelets: bool = True,
    enable_tsfresh: bool = True,
    # ── Engine configs ──
    rocket_n_kernels: int = 250,
    tsfresh_mode: str = 'comprehensive',
    signature_depth: int = 3,
    wavelet_types: list = None,
    catch22_window_sizes: list = None,
    # ── Rolling ──
    window_sizes: list = None,
    # ── Channels ──
    compute_channels: list = None,
    signature_channels: list = None,
    # ── Feature Selection ──
    feature_selection: str = None,
    correlation_threshold: float = 0.95,
    # ── Output ──
    output_dtype: str = 'float32',
    # ── Kolom DataFrame ──
    symbol_col: str = 'symbol',
    date_col: str = 'date',
    close_col: str = 'close',
    volume_col: str = 'volume',
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    # ── Progress ──
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Mengekstrak fitur kuantitatif tingkat lanjut dari data OHLCV saham.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame berisi kolom OHLCV + symbol + date.
        Format long: setiap baris = 1 saham di 1 tanggal.
    device : str
        'cuda' untuk GPU, 'cpu' untuk CPU fallback.
    max_vram_gb : float, optional
        Batas VRAM manual (GB). None = auto-detect.
    enable_rocket, enable_catch22, enable_signatures,
    enable_wavelets, enable_tsfresh : bool
        Toggle setiap engine on/off.
    rocket_n_kernels : int
        Jumlah random kernels untuk MultiRocket. Default: 250.
    tsfresh_mode : str
        'comprehensive' (~361 fitur/ch) atau 'efficient' (~214 fitur/ch).
    signature_depth : int
        Kedalaman Path Signature (1-3). Default: 3.
    wavelet_types : list
        Jenis wavelet. Default: ['haar', 'db4'].
    catch22_window_sizes : list
        Window sizes untuk Catch22. Default: same as window_sizes.
    window_sizes : list
        Ukuran rolling window. Default: [20].
    compute_channels : list
        Kolom untuk engine 1D (Catch22, Tsfresh, Wavelets, Rocket).
        Default: [close_col].
    signature_channels : list
        Kolom untuk Signatures (multivariat). Default: [close_col, volume_col].
    feature_selection : str
        None (default) untuk mengembalikan SEMUA fitur tanpa filter.
        'auto' untuk seleksi otomatis (variance + correlation filter).
        ⚠️ PERINGATAN: 'auto' menyebabkan fitur berbeda antar dataset
        yang berbeda, sehingga TIDAK cocok untuk training+backtesting.
    correlation_threshold : float
        Threshold korelasi untuk feature selection (hanya jika 'auto'). Default: 0.95.
    output_dtype : str
        Presisi output. Default: 'float32'.
    verbose : bool
        Tampilkan progress.

    Returns
    -------
    pd.DataFrame
        DataFrame asli + kolom fitur baru.

    Example
    -------
    >>> from ts_quant import generate_features
    >>> df_features = generate_features(
    ...     df,
    ...     device='cuda',
    ...     tsfresh_mode='comprehensive',
    ...     window_sizes=[20],
    ... )
    """
    t_start = time.time()

    # ═══════════════════════════════════════════════════════
    # DEFAULTS
    # ═══════════════════════════════════════════════════════
    if window_sizes is None:
        window_sizes = [20]
    if wavelet_types is None:
        wavelet_types = ['haar', 'db4']
    if compute_channels is None:
        compute_channels = [close_col]
    if signature_channels is None:
        signature_channels = [close_col, volume_col]

    window_size = window_sizes[0]  # Primary window

    # ═══════════════════════════════════════════════════════
    # STEP 1: VALIDATE
    # ═══════════════════════════════════════════════════════
    if verbose:
        print("━" * 60)
        print("  TS-QUANT: GPU-Accelerated Feature Extraction")
        print("━" * 60)

    required = [symbol_col, date_col] + list(set(
        compute_channels + signature_channels
    ))
    df = validate_dataframe(df, required, symbol_col, date_col)

    n_stocks = df[symbol_col].nunique()
    n_rows = len(df)
    if verbose:
        print(f"  Input: {n_rows:,} baris, {n_stocks} saham")

    # ═══════════════════════════════════════════════════════
    # STEP 2: SETUP DEVICE
    # ═══════════════════════════════════════════════════════
    manager = VRAMManager(
        device=device,
        max_vram_gb=max_vram_gb,
        verbose=verbose,
    )

    # ═══════════════════════════════════════════════════════
    # STEP 3: PROCESS PER STOCK
    # ═══════════════════════════════════════════════════════
    # Group data per saham → window → extract features
    if verbose:
        print(f"\n  Window size: {window_size}")
        engines_active = []
        if enable_rocket:     engines_active.append("Rocket")
        if enable_catch22:    engines_active.append("Catch22")
        if enable_signatures: engines_active.append("Signatures")
        if enable_wavelets:   engines_active.append("Wavelets")
        if enable_tsfresh:    engines_active.append("Tsfresh")
        print(f"  Active engines: {', '.join(engines_active)}")
        print()

    # Initialize engines ONCE
    rocket_eng = RocketEngine(
        n_kernels=rocket_n_kernels, variance_threshold=-1.0, verbose=False
    ) if enable_rocket else None

    catch22_eng = Catch22Engine(verbose=False) if enable_catch22 else None

    sig_eng = SignaturesEngine(
        depth=signature_depth,
        channels=signature_channels,
        augment_time=True,
        verbose=False,
    ) if enable_signatures else None

    wavelet_eng = WaveletsEngine(
        wavelet_types=wavelet_types, verbose=False
    ) if enable_wavelets else None

    tsfresh_eng = TsfreshEngine(
        mode=tsfresh_mode, verbose=False
    ) if enable_tsfresh else None

    # ── Process all windows ──
    import math
    valid_symbols = []
    valid_dates = []
    all_windows = []
    
    stock_groups = df.groupby(symbol_col)
    all_channels = list(set(compute_channels + signature_channels))

    for symbol, group in stock_groups:
        group = group.sort_values(date_col).reset_index(drop=True)
        T = len(group)

        if T < window_size + 5:
            continue

        values = group[all_channels].values.astype(np.float32)
        stock_tensor = torch.tensor(values)  # Keep on CPU to save memory

        windows = create_rolling_windows(stock_tensor, window_size)
        W = windows.shape[0]

        v_dates = group[date_col].values[window_size - 1:]
        if len(v_dates) > W:
            v_dates = v_dates[:W]

        valid_dates.append(v_dates)
        valid_symbols.append(np.full(W, symbol, dtype=object))
        all_windows.append(windows)

    if not all_windows:
        warnings.warn("Tidak ada saham yang memiliki data cukup panjang.")
        return df

    flat_windows = torch.cat(all_windows, dim=0)  # [Total_W, WS, C]
    flat_symbols = np.concatenate(valid_symbols)
    flat_dates = np.concatenate(valid_dates)
    
    Total_W = flat_windows.shape[0]
    chunk_size = 250000  # Massively scale up to saturate GPU VRAM!
    n_chunks = math.ceil(Total_W / chunk_size)
    
    if verbose:
        print(f"  Pre-processing: Built {Total_W:,} windows from {len(all_windows)} stocks")
        print(f"  Processing in {n_chunks} large GPU chunks (Streaming Out-of-Core)...")

    # ── TEMPORARY STORAGE DISK ──
    import os, uuid
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    tmp_file = f"ts_quant_temp_{uuid.uuid4().hex}.parquet"
    writer = None
    chunk_names = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, Total_W)
        batch = flat_windows[start:end].to(manager.device)  # [B, WS, C]

        batch_features = []
        batch_names = []

        # ── Univariate engines ──
        for ch_name in compute_channels:
            ch_all_idx = all_channels.index(ch_name)
            x_1d = batch[:, :, ch_all_idx]  # [B, WS]
            prefix = ch_name

            if rocket_eng is not None:
                feat, names = rocket_eng.extract(x_1d)
                batch_features.append(feat)
                if i == 0: batch_names.extend([f"{prefix}_{n}" for n in names])

            if catch22_eng is not None:
                feat, names = catch22_eng.extract(x_1d)
                batch_features.append(feat)
                if i == 0: batch_names.extend([f"{prefix}_{n}" for n in names])

            if wavelet_eng is not None:
                feat, names = wavelet_eng.extract(x_1d)
                batch_features.append(feat)
                if i == 0: batch_names.extend([f"{prefix}_{n}" for n in names])

            if tsfresh_eng is not None:
                feat, names = tsfresh_eng.extract(x_1d)
                batch_features.append(feat)
                if i == 0: batch_names.extend([f"{prefix}_{n}" for n in names])

        # ── Multivariate engine ──
        if sig_eng is not None:
            sig_ch_indices = [all_channels.index(c) for c in signature_channels if c in all_channels]
            if sig_ch_indices:
                mv_windows = batch[:, :, sig_ch_indices]
                feat, names = sig_eng.extract(mv_windows)
                batch_features.append(feat)
                if i == 0: batch_names.extend(names)

        all_feat = torch.cat(batch_features, dim=1)
        if all_feat.is_cuda:
            all_feat = all_feat.cpu()
            
        all_feat_np = all_feat.numpy()
        
        if i == 0:
            chunk_names = batch_names
            
        # ── DISK WRITING PER CHUNK ──
        table = pa.Table.from_arrays(
            [pa.array(all_feat_np[:, j]) for j in range(all_feat_np.shape[1])],
            names=chunk_names
        )
        if writer is None:
            writer = pq.ParquetWriter(tmp_file, table.schema)
        
        writer.write_table(table)

        # Free up variables memory explicitly
        del batch_features
        del all_feat
        del all_feat_np
        del table
        import gc
        gc.collect()

        if manager.is_gpu:
            manager.clear_cache()

        if verbose:
            print(f"    Processed GPU chunk {i+1}/{n_chunks}...")

    if writer is not None:
        writer.close()

    if verbose:
        print(f"\n  Features extracted: ({Total_W}, {len(chunk_names)})")

    # ═══════════════════════════════════════════════════════
    # STEP 5: FEATURE SELECTION (OUT-OF-CORE)
    # ═══════════════════════════════════════════════════════
    selected_names = chunk_names
    if feature_selection == 'auto':
        from ts_quant.utils.feature_selection import auto_select_features_from_parquet
        selected_names = auto_select_features_from_parquet(
            tmp_file,
            variance_threshold=1e-8,
            correlation_threshold=correlation_threshold,
            max_nan_ratio=0.5,
            verbose=verbose,
            device=('cuda' if torch.cuda.is_available() else 'cpu')
        )

    # ═══════════════════════════════════════════════════════
    # STEP 6: DISK-OFFLOADED MERGE (ZERO-SPIKE MEMORY)
    # ═══════════════════════════════════════════════════════
    if verbose:
        print("  Merging features incrementally from Disk...")

    # Set indices for exact row alignment
    original_index_names = df.index.names
    df.set_index([symbol_col, date_col], inplace=True)
    
    # We must construct target_index aligning the features
    # But wait, we didn't save symbol/date inside Parquet file!
    # Luckily, `flat_symbols` and `flat_dates` match the Parquet rows exactly:
    target_idx = pd.MultiIndex.from_arrays([flat_symbols, flat_dates])
    
    parquet_file = pq.ParquetFile(tmp_file)
    
    # Batch process 100 columns at a time to keep RAM extremely low
    batch_size = 100
    for i in range(0, len(selected_names), batch_size):
        cols_batch = selected_names[i:i+batch_size]
        
        table = parquet_file.read(columns=cols_batch)
        part_df = table.to_pandas()
        part_df.index = target_idx
        
        # Merge column by column!
        for col in cols_batch:
            # Drop NaN rows implicitly during MultiIndex alignment
            # Assigning part_df[col] will match rows!
            df[col] = part_df[col]
            # Replace NaNs with 0 inside Pandas
            df[col].fillna(0.0, inplace=True)
            if output_dtype == 'float32' and df[col].dtype != np.float32:
                df[col] = df[col].astype(np.float32)
                
        del part_df, table
        gc.collect()
        
    # Restore original index if any
    df.reset_index(inplace=True)
    if original_index_names and original_index_names != [None]:
        df.set_index(original_index_names, inplace=True)
        
    # Cleanup disk
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        
    result = df
    
    # ═══════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════
    dt = time.time() - t_start
    if verbose:
        print(f"\n  ✅ Done! {len(selected_names)} new features added in {dt:.1f}s")
        print(f"  Output shape: {result.shape}")
        print("━" * 60)

    # Cleanup
    if manager.is_gpu:
        manager.clear_cache()
    gc.collect()

    return result
