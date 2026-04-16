"""
ts_quant.utils.validation — Input Validation & Sanity Checks
==============================================================

Validasi input DataFrame sebelum diproses oleh engine.
"""

from typing import List, Optional

import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Error yang dilempar saat validasi input gagal."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    symbol_col: str = 'symbol',
    date_col: str = 'date',
) -> pd.DataFrame:
    """
    Validasi DataFrame input dan perbaiki masalah umum.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame yang akan divalidasi.
    required_cols : list of str
        Kolom yang wajib ada.
    symbol_col : str
        Nama kolom symbol.
    date_col : str
        Nama kolom tanggal.

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah divalidasi dan dibersihkan.

    Raises
    ------
    ValidationError
        Jika ada kolom yang hilang atau data tidak valid.
    """
    # ── Cek kolom wajib ──
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValidationError(
            f"Kolom berikut tidak ditemukan di DataFrame: {missing}. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    # ── Cek bukan kosong ──
    if len(df) == 0:
        raise ValidationError("DataFrame kosong (0 baris).")

    # ── Cek kolom symbol ──
    if symbol_col in df.columns:
        n_symbols = df[symbol_col].nunique()
        if n_symbols == 0:
            raise ValidationError(
                f"Kolom '{symbol_col}' tidak memiliki nilai unik."
            )

    # ── Cek kolom date ──
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        if df[date_col].isna().all():
            raise ValidationError(
                f"Semua nilai di kolom '{date_col}' adalah NaT/NaN."
            )

    # ── Cek kolom numerik ──
    numeric_cols = [c for c in required_cols if c not in [symbol_col, date_col]]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValidationError(
                f"Kolom '{col}' bukan numerik (dtype: {df[col].dtype})."
            )
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 50:
            raise ValidationError(
                f"Kolom '{col}' memiliki {nan_pct:.1f}% NaN (> 50%)."
            )

    return df


def validate_tensor_output(
    tensor,
    name: str = "output",
) -> None:
    """
    Validasi tensor output dari engine: tidak boleh ada Inf.
    NaN diperbolehkan (untuk padding / data tidak cukup).

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor yang akan divalidasi.
    name : str
        Nama tensor untuk pesan error.

    Raises
    ------
    ValidationError
        Jika tensor mengandung Inf.
    """
    import torch

    if torch.isinf(tensor).any():
        n_inf = torch.isinf(tensor).sum().item()
        raise ValidationError(
            f"Tensor '{name}' mengandung {n_inf} nilai Inf. "
            f"Ini menunjukkan bug dalam perhitungan."
        )

    n_nan = torch.isnan(tensor).sum().item()
    total = tensor.numel()
    nan_pct = n_nan / max(total, 1) * 100

    if nan_pct > 90:
        raise ValidationError(
            f"Tensor '{name}' mengandung {nan_pct:.1f}% NaN (> 90%). "
            f"Kemungkinan data input tidak cukup panjang."
        )
