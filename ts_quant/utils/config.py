"""
ts_quant.utils.config — Default Parameters & Presets
=====================================================

Konfigurasi default untuk setiap engine dan parameter global.
Semua nilai dapat di-override melalui API generate_features().
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
# Global Defaults
# ═══════════════════════════════════════════════════════════════

DEFAULT_DEVICE = 'cuda'
DEFAULT_DTYPE = 'float32'
DEFAULT_SAFETY_FACTOR = 0.80
DEFAULT_OVERHEAD_FACTOR = 2.5

# Kolom DataFrame
DEFAULT_SYMBOL_COL = 'symbol'
DEFAULT_DATE_COL = 'date'
DEFAULT_OPEN_COL = 'open'
DEFAULT_HIGH_COL = 'high'
DEFAULT_LOW_COL = 'low'
DEFAULT_CLOSE_COL = 'close'
DEFAULT_VOLUME_COL = 'volume'

# Rolling windows
DEFAULT_WINDOW_SIZES = [20]


# ═══════════════════════════════════════════════════════════════
# Engine Configs
# ═══════════════════════════════════════════════════════════════

@dataclass
class RocketConfig:
    """Konfigurasi Engine A: MultiRocket."""
    n_kernels: int = 10_000
    kernel_lengths: List[int] = field(default_factory=lambda: [7, 9, 11])
    features_per_kernel: int = 4  # PPV, MAX, MEAN, LSPV
    max_dilations: int = 32
    # Feature selection setelah ekstraksi
    variance_threshold: float = 0.0   # Hapus fitur dengan variance 0
    correlation_threshold: float = 0.95  # Hapus fitur yang berkorelasi >0.95


@dataclass
class Catch22Config:
    """Konfigurasi Engine B: Catch22."""
    window_sizes: List[int] = field(default_factory=lambda: [10, 20, 60])
    # 22 fitur × len(window_sizes) = total fitur


@dataclass
class SignaturesConfig:
    """Konfigurasi Engine C: Path Signatures."""
    depth: int = 3
    channels: List[str] = field(
        default_factory=lambda: ['close', 'volume']
    )
    window_size: int = 20
    normalize: bool = True  # Normalize path sebelum signature


@dataclass
class WaveletsConfig:
    """Konfigurasi Engine D: Wavelet Transform."""
    wavelet_types: List[str] = field(
        default_factory=lambda: ['haar', 'db4']
    )
    decomposition_levels: int = 4
    stats_per_level: List[str] = field(
        default_factory=lambda: ['energy', 'mean', 'std', 'max', 'entropy']
    )
    apply_to: List[str] = field(
        default_factory=lambda: ['close', 'volume']
    )


@dataclass
class TsfreshConfig:
    """Konfigurasi Engine E: Tsfresh Complete."""
    mode: str = 'comprehensive'  # 'comprehensive' atau 'efficient'
    window_size: int = 20

    # FFT
    fft_max_coeff: int = 50  # comprehensive: 50, efficient: 20

    # Autocorrelation
    acf_lags: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 20]
    )

    # Change quantiles (comprehensive mode)
    change_quantile_ql: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8]
    )
    change_quantile_qh: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0]
    )

    # AR coefficients
    ar_max_order: int = 10

    # Peak detection
    peak_n_values: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10]
    )

    # Ratio beyond sigma
    sigma_values: List[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0]
    )


# ═══════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════

EFFICIENT_TSFRESH = TsfreshConfig(
    mode='efficient',
    fft_max_coeff=20,
    acf_lags=[1, 2, 3, 5, 10],
    change_quantile_ql=[0.0, 0.4],
    change_quantile_qh=[0.6, 1.0],
    ar_max_order=5,
    peak_n_values=[1, 5],
    sigma_values=[1.0, 2.0, 3.0],
)
