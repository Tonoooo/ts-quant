"""
ts-quant — GPU-Accelerated Quantitative Feature Extraction
===========================================================

Library untuk mengekstrak ribuan fitur kuantitatif dari data
time-series keuangan menggunakan GPU (PyTorch/CUDA).

5 Engine:
    A. MultiRocket  — Random Convolutional Kernels
    B. Catch22      — 22 Canonical Time-Series Features
    C. Signatures   — Path Signatures (Ordered Interactions)
    D. Wavelets     — Discrete Wavelet Transform
    E. Tsfresh Core — 63 Statistical Functions (300+ features)

Quickstart:
    >>> import pandas as pd
    >>> from ts_quant import generate_features
    >>>
    >>> df = pd.read_csv('stocks.csv')
    >>> df_features = generate_features(df, device='cuda')
"""

__version__ = "0.2.0"
__author__ = "TS-Quant Team"

from ts_quant.api import generate_features

# Engine classes
from ts_quant.engines.rocket import RocketEngine
from ts_quant.engines.catch22 import Catch22Engine
from ts_quant.engines.signatures import SignaturesEngine
from ts_quant.engines.wavelets import WaveletsEngine
from ts_quant.engines.tsfresh_core import TsfreshEngine

# Core utilities
from ts_quant.core.memory_manager import VRAMManager
from ts_quant.core.windowing import create_rolling_windows
from ts_quant.utils.feature_selection import auto_select_features_from_parquet

__all__ = [
    "generate_features",
    "RocketEngine",
    "Catch22Engine",
    "SignaturesEngine",
    "WaveletsEngine",
    "TsfreshEngine",
    "VRAMManager",
    "create_rolling_windows",
    "auto_select_features_from_parquet",
]
