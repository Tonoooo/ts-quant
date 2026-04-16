"""
ts_quant.core.windowing — GPU Sliding Window (Zero-Copy Stride)
================================================================

Membuat rolling/sliding windows dari tensor time-series menggunakan
torch.Tensor.unfold() — operasi zero-copy yang sangat efisien.

Mengubah tensor 2D [T × C] menjadi tensor 3D [W × window_size × C]
dimana W = T - window_size + 1 (jumlah windows).

Semua engine (Catch22, Tsfresh, Wavelets, Rocket) bekerja di atas
output dari modul ini.
"""

from typing import Dict, List, Tuple, Union

import torch
import numpy as np


def create_rolling_windows(
    tensor: torch.Tensor,
    window_size: int,
    step: int = 1,
) -> torch.Tensor:
    """
    Membuat sliding windows dari tensor 2D menggunakan unfold (zero-copy).

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor 2D [T × C] dimana:
            T = jumlah timesteps
            C = jumlah channels (close, volume, ...)
    window_size : int
        Ukuran jendela (misal: 20 untuk 20 hari).
    step : int
        Langkah antar window. Default 1 (setiap hari).

    Returns
    -------
    torch.Tensor
        Tensor 3D [W × window_size × C] dimana:
            W = (T - window_size) // step + 1

    Example
    -------
    >>> # Data 100 hari, 3 channel (close, volume, ret)
    >>> x = torch.randn(100, 3)
    >>> windows = create_rolling_windows(x, window_size=20)
    >>> print(windows.shape)  # [81, 20, 3]
    """
    T, C = tensor.shape

    if window_size > T:
        raise ValueError(
            f"window_size ({window_size}) > T ({T}). "
            f"Tidak cukup data untuk membuat window."
        )

    if window_size < 2:
        raise ValueError(f"window_size harus >= 2, got {window_size}")

    # unfold(dimension, size, step) — zero-copy stride trick
    # Input shape:  [T, C]
    # Setelah unfold di dim 0: [W, C, window_size]
    windows = tensor.unfold(0, window_size, step)  # [W, C, window_size]

    # Transpose ke [W, window_size, C] agar sesuai konvensi
    windows = windows.permute(0, 2, 1)  # [W, window_size, C]

    return windows.contiguous()


def create_rolling_windows_multi(
    tensor: torch.Tensor,
    window_sizes: List[int],
    step: int = 1,
) -> Dict[int, torch.Tensor]:
    """
    Membuat sliding windows untuk beberapa ukuran sekaligus.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor 2D [T × C].
    window_sizes : list of int
        Daftar ukuran window. Contoh: [10, 20, 60].
    step : int
        Langkah antar window.

    Returns
    -------
    dict
        Mapping {window_size: tensor_3d} untuk setiap ukuran.

    Example
    -------
    >>> x = torch.randn(200, 3)
    >>> all_windows = create_rolling_windows_multi(x, [10, 20, 60])
    >>> all_windows[20].shape  # [181, 20, 3]
    """
    result = {}
    for ws in window_sizes:
        if ws > tensor.shape[0]:
            continue  # Skip jika data terlalu pendek
        result[ws] = create_rolling_windows(tensor, ws, step)
    return result


def create_rolling_windows_batch(
    tensors: List[torch.Tensor],
    window_size: int,
    step: int = 1,
    min_length: int = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Membuat rolling windows untuk batch saham sekaligus.

    Setiap saham bisa memiliki panjang time-series berbeda.
    Output di-concatenate menjadi satu tensor besar.

    Parameters
    ----------
    tensors : list of torch.Tensor
        List tensor 2D [Ti × C], satu per saham.
        Setiap tensor bisa punya Ti berbeda, tapi C harus sama.
    window_size : int
        Ukuran jendela.
    step : int
        Langkah antar window.
    min_length : int, optional
        Minimum panjang saham agar diproses. Default = window_size.

    Returns
    -------
    all_windows : torch.Tensor
        Tensor 3D [total_W × window_size × C] — gabungan semua saham.
    stock_boundaries : list of int
        List berisi jumlah windows per saham.
        Berguna untuk memecah kembali hasil per saham.

    Example
    -------
    >>> stocks = [torch.randn(200, 3), torch.randn(150, 3)]
    >>> windows, boundaries = create_rolling_windows_batch(
    ...     stocks, window_size=20
    ... )
    >>> # boundaries = [181, 131] → stock 0 punya 181 windows
    """
    if min_length is None:
        min_length = window_size

    window_list = []
    stock_boundaries = []

    for stock_tensor in tensors:
        T = stock_tensor.shape[0]
        if T < min_length:
            stock_boundaries.append(0)
            continue

        windows = create_rolling_windows(stock_tensor, window_size, step)
        window_list.append(windows)
        stock_boundaries.append(windows.shape[0])

    if not window_list:
        C = tensors[0].shape[1] if tensors else 1
        empty = torch.empty(0, window_size, C, device=tensors[0].device)
        return empty, stock_boundaries

    all_windows = torch.cat(window_list, dim=0)
    return all_windows, stock_boundaries


def unbatch_windows(
    features: torch.Tensor,
    stock_boundaries: List[int],
) -> List[torch.Tensor]:
    """
    Memecah kembali tensor fitur gabungan menjadi per saham.

    Parameters
    ----------
    features : torch.Tensor
        Tensor 2D [total_W × N_features] — output dari engine.
    stock_boundaries : list of int
        Jumlah windows per saham (output dari create_rolling_windows_batch).

    Returns
    -------
    list of torch.Tensor
        List tensor 2D [Wi × N_features] per saham.
    """
    result = []
    offset = 0
    for n_windows in stock_boundaries:
        if n_windows == 0:
            result.append(None)
        else:
            result.append(features[offset:offset + n_windows])
            offset += n_windows
    return result


def compute_window_stats(
    windows: torch.Tensor,
) -> torch.Tensor:
    """
    Menghitung statistik dasar dari setiap window secara vektorisasi.

    Berguna sebagai baseline fitur dan untuk validasi.

    Parameters
    ----------
    windows : torch.Tensor
        Tensor 3D [W × window_size × C].

    Returns
    -------
    torch.Tensor
        Tensor 2D [W × (C * 5)] berisi mean, std, min, max, median
        untuk setiap channel di setiap window.
    """
    # [W, window_size, C]
    w_mean = windows.mean(dim=1)    # [W, C]
    w_std = windows.std(dim=1)      # [W, C]
    w_min = windows.min(dim=1)[0]   # [W, C]
    w_max = windows.max(dim=1)[0]   # [W, C]
    w_median = windows.median(dim=1)[0]  # [W, C]

    # Concatenate: [W, C*5]
    stats = torch.cat([w_mean, w_std, w_min, w_max, w_median], dim=1)
    return stats
