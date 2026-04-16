"""
ts_quant.core.tensor_utils — DataFrame ↔ GPU Tensor Conversion
================================================================

Mengubah DataFrame format panjang (long-format) berisi data OHLCV
multi-saham menjadi Tensor 3D yang siap diproses GPU, dan sebaliknya.

Tensor 3D shape: [N_stocks × N_timesteps × N_channels]
    - N_stocks:    jumlah saham unik
    - N_timesteps: panjang time-series terpanjang (saham pendek di-pad NaN)
    - N_channels:  jumlah kolom numerik (close, volume, ret, ...)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


def df_to_tensor_3d(
    df: pd.DataFrame,
    value_cols: List[str],
    symbol_col: str = 'symbol',
    date_col: str = 'date',
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    pad_value: float = float('nan'),
) -> Tuple[torch.Tensor, List[str], pd.DatetimeIndex, Dict[str, np.ndarray]]:
    """
    Mengubah DataFrame long-format menjadi Tensor 3D.

    DataFrame input memiliki format:
        symbol | date       | close | volume | ...
        ASII   | 2024-01-02 | 5200  | 12000  | ...
        ASII   | 2024-01-03 | 5250  | 15000  | ...
        BBCA   | 2024-01-02 | 9800  | 8000   | ...
        ...

    Tensor output memiliki shape [N_stocks × N_timesteps × N_channels].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame berisi kolom symbol, date, dan value_cols.
    value_cols : list of str
        Kolom numerik yang akan dimasukkan ke tensor.
        Contoh: ['close', 'volume', 'ret_1d']
    symbol_col : str
        Nama kolom berisi kode saham.
    date_col : str
        Nama kolom berisi tanggal.
    device : str
        'cuda' atau 'cpu'.
    dtype : torch.dtype
        Tipe data tensor. Default float32.
    pad_value : float
        Nilai padding untuk saham yang datanya lebih pendek.
        Default: NaN.

    Returns
    -------
    tensor : torch.Tensor
        Tensor 3D [N_stocks × N_timesteps × N_channels]
    stock_names : list of str
        Urutan nama saham sesuai dimensi 0.
    date_index : numpy array
        Urutan tanggal unik sesuai dimensi 1.
    stock_masks : dict
        Mapping {symbol: boolean_mask} — True di posisi yang valid
        (bukan padding).

    Example
    -------
    >>> tensor, stocks, dates, masks = df_to_tensor_3d(
    ...     df, value_cols=['close', 'volume'], device='cuda'
    ... )
    >>> print(tensor.shape)  # [558, 2000, 2]
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ── Identifikasi saham dan tanggal unik ──
    stock_names = sorted(df[symbol_col].unique().tolist())
    all_dates = sorted(df[date_col].unique())
    date_index = np.array(all_dates)

    n_stocks = len(stock_names)
    n_times = len(all_dates)
    n_channels = len(value_cols)

    # ── Buat mapping tanggal → index untuk cepat ──
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # ── Alokasi tensor (diisi pad_value dulu) ──
    tensor = torch.full(
        (n_stocks, n_times, n_channels),
        fill_value=pad_value,
        dtype=dtype,
        device='cpu',  # Bangun di CPU dulu, pindah nanti
    )

    # ── Isi tensor per saham ──
    stock_masks = {}
    stock_to_idx = {s: i for i, s in enumerate(stock_names)}

    for symbol, group in df.groupby(symbol_col):
        s_idx = stock_to_idx[symbol]
        dates = group[date_col].values
        values = group[value_cols].values.astype(np.float32)

        # Map setiap baris ke posisi yang benar di dimensi waktu
        time_indices = np.array([date_to_idx[d] for d in dates])

        tensor[s_idx, time_indices, :] = torch.from_numpy(values)

        # Buat mask: True di posisi yang memiliki data asli
        mask = np.zeros(n_times, dtype=bool)
        mask[time_indices] = True
        stock_masks[symbol] = mask

    # ── Pindahkan ke device target ──
    if device != 'cpu':
        tensor = tensor.to(device)

    return tensor, stock_names, date_index, stock_masks


def tensor_to_df(
    tensor: torch.Tensor,
    stock_names: List[str],
    date_index: np.ndarray,
    feature_names: List[str],
    stock_masks: Optional[Dict[str, np.ndarray]] = None,
    symbol_col: str = 'symbol',
    date_col: str = 'date',
) -> pd.DataFrame:
    """
    Mengubah Tensor 3D kembali menjadi DataFrame long-format.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor 3D [N_stocks × N_timesteps × N_features].
        Bisa di GPU maupun CPU.
    stock_names : list of str
        Nama saham sesuai dimensi 0.
    date_index : numpy array
        Tanggal sesuai dimensi 1.
    feature_names : list of str
        Nama fitur sesuai dimensi 2.
    stock_masks : dict, optional
        Jika diberikan, hanya baris yang valid (True) yang dimasukkan.
        Ini mengecualikan baris padding.
    symbol_col : str
        Nama kolom output untuk symbol.
    date_col : str
        Nama kolom output untuk tanggal.

    Returns
    -------
    pd.DataFrame
        DataFrame [N_valid_rows × (symbol, date, feat_1, feat_2, ...)]
    """
    # Pastikan tensor di CPU untuk konversi ke numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()

    data_np = tensor.numpy()  # [N_stocks × N_timesteps × N_features]

    rows = []
    for s_idx, symbol in enumerate(stock_names):
        if stock_masks is not None and symbol in stock_masks:
            mask = stock_masks[symbol]
            valid_times = np.where(mask)[0]
        else:
            # Tanpa mask, ambil semua baris yang bukan full-NaN
            valid_times = np.where(
                ~np.all(np.isnan(data_np[s_idx]), axis=1)
            )[0]

        if len(valid_times) == 0:
            continue

        stock_data = data_np[s_idx, valid_times, :]  # [n_valid × n_features]
        n_valid = len(valid_times)

        # Buat mini-DataFrame untuk saham ini
        stock_df = pd.DataFrame(stock_data, columns=feature_names)
        stock_df[symbol_col] = symbol
        stock_df[date_col] = date_index[valid_times]

        rows.append(stock_df)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)

    # Reorder kolom: symbol, date dulu
    cols = [symbol_col, date_col] + feature_names
    result = result[cols]

    return result


def get_per_stock_tensors(
    tensor_3d: torch.Tensor,
    stock_masks: Dict[str, np.ndarray],
    stock_names: List[str],
) -> List[Tuple[str, torch.Tensor]]:
    """
    Memecah Tensor 3D menjadi list tensor 2D per saham,
    dengan padding dihapus.

    Berguna untuk engine yang memproses per saham (misal: Signatures).

    Parameters
    ----------
    tensor_3d : torch.Tensor
        [N_stocks × N_timesteps × N_channels]
    stock_masks : dict
        Mapping {symbol: boolean_mask}
    stock_names : list of str
        Nama saham sesuai dimensi 0.

    Returns
    -------
    list of (str, torch.Tensor)
        List of (symbol, tensor_2d) dimana tensor_2d shape
        [valid_timesteps × N_channels] tanpa padding.
    """
    result = []
    for s_idx, symbol in enumerate(stock_names):
        mask = stock_masks.get(symbol)
        if mask is None:
            # Gunakan semua
            stock_tensor = tensor_3d[s_idx]
        else:
            valid_idx = torch.from_numpy(
                np.where(mask)[0]
            ).to(tensor_3d.device)
            stock_tensor = tensor_3d[s_idx].index_select(0, valid_idx)

        result.append((symbol, stock_tensor))

    return result


def estimate_tensor_bytes(
    n_stocks: int,
    n_timesteps: int,
    n_channels: int,
    dtype: torch.dtype = torch.float32,
) -> int:
    """
    Estimasi ukuran memori tensor 3D dalam bytes.

    Parameters
    ----------
    n_stocks : int
    n_timesteps : int
    n_channels : int
    dtype : torch.dtype

    Returns
    -------
    int
        Estimasi bytes.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    return n_stocks * n_timesteps * n_channels * element_size
