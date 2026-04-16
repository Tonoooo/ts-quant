"""
ts_quant.engines.wavelets — Engine D: Wavelet Transform (GPU)
==============================================================

Implementasi Discrete Wavelet Transform (DWT) menggunakan Conv1D
di PyTorch. Menggantikan PyWavelets dengan operasi GPU native.

Mendukung:
    - Haar wavelet
    - Daubechies-4 (db4) wavelet
    - Multi-level decomposition (hingga 4 level)
    - 5 statistik per level: energy, mean, std, max, entropy

Referensi: Mallat, 2009 — "A Wavelet Tour of Signal Processing"
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# WAVELET FILTER COEFFICIENTS
# ═══════════════════════════════════════════════════════════════

# Koefisien filter wavelet (scaling function = low-pass, wavelet = high-pass)
WAVELET_FILTERS = {
    'haar': {
        'dec_lo': [1 / math.sqrt(2), 1 / math.sqrt(2)],
        'dec_hi': [1 / math.sqrt(2), -1 / math.sqrt(2)],
    },
    'db4': {
        # Daubechies-4 coefficients (8 taps)
        'dec_lo': [
            -0.010597401784997278,
             0.032883011666982945,
             0.030841381835986965,
            -0.18703481171888114,
            -0.027983769416983849,
             0.63088076792959036,
             0.71484657055254153,
             0.23037781330885523,
        ],
        'dec_hi': [
            -0.23037781330885523,
             0.71484657055254153,
            -0.63088076792959036,
            -0.027983769416983849,
             0.18703481171888114,
             0.030841381835986965,
            -0.032883011666982945,
            -0.010597401784997278,
        ],
    },
    'db2': {
        # Daubechies-2 coefficients (4 taps)
        'dec_lo': [
            -0.12940952255092145,
             0.22414386804185735,
             0.83651630373746899,
             0.48296291314469025,
        ],
        'dec_hi': [
            -0.48296291314469025,
             0.83651630373746899,
            -0.22414386804185735,
            -0.12940952255092145,
        ],
    },
    'sym4': {
        # Symlet-4 coefficients (8 taps)
        'dec_lo': [
            -0.07576571478927333,
            -0.02963552764599851,
             0.49761866763201545,
             0.80373875180591614,
             0.29785779560527736,
            -0.09921954357684722,
            -0.01260396726203783,
             0.03222310060404270,
        ],
        'dec_hi': [
            -0.03222310060404270,
            -0.01260396726203783,
             0.09921954357684722,
             0.29785779560527736,
            -0.80373875180591614,
             0.49761866763201545,
             0.02963552764599851,
            -0.07576571478927333,
        ],
    },
}


def _get_filter_tensors(
    wavelet_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mengambil koefisien filter wavelet sebagai tensor Conv1D.

    Returns
    -------
    lo : torch.Tensor [1, 1, filter_len]
    hi : torch.Tensor [1, 1, filter_len]
    """
    if wavelet_name not in WAVELET_FILTERS:
        raise ValueError(
            f"Wavelet '{wavelet_name}' tidak tersedia. "
            f"Pilihan: {list(WAVELET_FILTERS.keys())}"
        )
    coeffs = WAVELET_FILTERS[wavelet_name]
    lo = torch.tensor(coeffs['dec_lo'], device=device, dtype=dtype)
    hi = torch.tensor(coeffs['dec_hi'], device=device, dtype=dtype)
    # Reshape ke format Conv1D kernel: [out_channels, in_channels, kernel_size]
    lo = lo.flip(0).reshape(1, 1, -1)  # flip untuk konvolusi (bukan korelasi)
    hi = hi.flip(0).reshape(1, 1, -1)
    return lo, hi


# ═══════════════════════════════════════════════════════════════
# DWT DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

def dwt_1level(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Satu level Discrete Wavelet Transform.

    Parameters
    ----------
    x : torch.Tensor [B, 1, T]
        Input signal (batched, single channel).
    lo : torch.Tensor [1, 1, K]
        Low-pass decomposition filter.
    hi : torch.Tensor [1, 1, K]
        High-pass decomposition filter.

    Returns
    -------
    approx : torch.Tensor [B, 1, T//2]
        Approximation coefficients (low-frequency).
    detail : torch.Tensor [B, 1, T//2]
        Detail coefficients (high-frequency).
    """
    K = lo.shape[2]
    # Symmetric padding (mirror) untuk menghindari edge effects
    pad_size = K - 1
    x_padded = F.pad(x, (pad_size, pad_size), mode='reflect')

    # Convolution + downsample by 2
    approx = F.conv1d(x_padded, lo)[:, :, ::2]
    detail = F.conv1d(x_padded, hi)[:, :, ::2]

    return approx, detail


def wavedec(
    x: torch.Tensor,
    wavelet_name: str,
    level: int = 4,
) -> List[torch.Tensor]:
    """
    Multi-level wavelet decomposition.

    Parameters
    ----------
    x : torch.Tensor [B, T]
        Batched 1D signals.
    wavelet_name : str
        Nama wavelet ('haar', 'db4', 'db2', 'sym4').
    level : int
        Jumlah level dekomposisi.

    Returns
    -------
    coeffs : list of torch.Tensor
        [approx_N, detail_N, detail_N-1, ..., detail_1]
        Setiap elemen shape [B, T_level]
    """
    device = x.device
    dtype = x.dtype
    lo, hi = _get_filter_tensors(wavelet_name, device, dtype)

    # Reshape: [B, T] → [B, 1, T]
    current = x.unsqueeze(1)

    details = []
    for lv in range(level):
        min_len = lo.shape[2] * 2
        if current.shape[2] < min_len:
            break  # Tidak cukup data untuk dekomposisi lebih lanjut
        approx, detail = dwt_1level(current, lo, hi)
        details.append(detail.squeeze(1))  # [B, T_lv]
        current = approx

    # Output: [approx_final, detail_deepest, ..., detail_shallowest]
    result = [current.squeeze(1)] + details[::-1]
    return result


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FROM COEFFICIENTS
# ═══════════════════════════════════════════════════════════════

def _coeff_energy(coeffs: torch.Tensor) -> torch.Tensor:
    """Energi koefisien: sum(c^2). [B, T] -> [B]"""
    return (coeffs ** 2).sum(dim=1)


def _coeff_entropy(coeffs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy dari distribusi energi. [B, T] -> [B]"""
    c2 = coeffs ** 2
    total = c2.sum(dim=1, keepdim=True).clamp(min=1e-10)
    p = c2 / total
    p = p.clamp(min=1e-10)  # Hindari log(0)
    entropy = -(p * torch.log(p)).sum(dim=1)
    return entropy


def extract_coeff_stats(
    coeffs: torch.Tensor,
    prefix: str,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Ekstrak 5 statistik dari koefisien wavelet.

    Parameters
    ----------
    coeffs : torch.Tensor [B, T]
        Koefisien wavelet satu level.
    prefix : str
        Prefix untuk nama fitur (misal: "haar_d1").

    Returns
    -------
    features : torch.Tensor [B, 5]
    names : list of str
    """
    energy = _coeff_energy(coeffs)
    mean = coeffs.mean(dim=1)
    std = coeffs.std(dim=1).clamp(min=0)
    max_val = coeffs.abs().max(dim=1)[0]
    entropy = _coeff_entropy(coeffs)

    features = torch.stack([energy, mean, std, max_val, entropy], dim=1)
    names = [
        f"{prefix}_energy",
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_max",
        f"{prefix}_entropy",
    ]
    return features, names


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL WAVELET FEATURES
# ═══════════════════════════════════════════════════════════════

def energy_ratio_per_level(coeffs_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Rasio energi di setiap level terhadap total energi.
    Menunjukkan di frekuensi mana kekuatan sinyal terkonsentrasi.

    Parameters
    ----------
    coeffs_list : list of torch.Tensor
        Output dari wavedec(): [approx, detail_N, ..., detail_1]

    Returns
    -------
    ratios : torch.Tensor [B, n_levels]
    """
    energies = []
    for c in coeffs_list:
        e = (c ** 2).sum(dim=1)  # [B]
        energies.append(e)

    energies = torch.stack(energies, dim=1)  # [B, n_levels]
    total = energies.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return energies / total


def wavelet_denoised(
    x: torch.Tensor,
    wavelet_name: str = 'db4',
    level: int = 3,
    threshold_factor: float = 1.0,
) -> torch.Tensor:
    """
    Denoised signal menggunakan wavelet soft-thresholding.

    Menghilangkan noise frekuensi tinggi dari sinyal harga.
    Sinyal bersih ini bisa digunakan sebagai fitur tambahan.

    Parameters
    ----------
    x : torch.Tensor [B, T]
    wavelet_name : str
    level : int
    threshold_factor : float
        Pengali universal threshold. 1.0 = standard.

    Returns
    -------
    denoised : torch.Tensor [B, T]
        Sinyal x yang sudah dibersihkan dari noise.
    """
    # Simplified denoising: gunakan approximation coefficients level terakhir
    coeffs = wavedec(x, wavelet_name, level)
    # Approximation = trend terbersih (tanpa detail = tanpa noise)
    approx = coeffs[0]  # [B, T_approx]

    # Interpolate kembali ke panjang asli
    B, T = x.shape
    denoised = F.interpolate(
        approx.unsqueeze(1),  # [B, 1, T_approx]
        size=T,
        mode='linear',
        align_corners=False,
    ).squeeze(1)  # [B, T]

    return denoised


# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE CLASS
# ═══════════════════════════════════════════════════════════════

class WaveletsEngine:
    """
    Engine D: Discrete Wavelet Transform.

    Mendekomposisi sinyal harga/volume ke frekuensi berbeda
    dan mengekstrak statistik per level.

    Parameters
    ----------
    wavelet_types : list of str
        Jenis wavelet. Default: ['haar', 'db4'].
    decomposition_levels : int
        Jumlah level dekomposisi. Default: 4.
    include_denoised : bool
        Apakah menyertakan sinyal denoised sebagai fitur.
    verbose : bool

    Example
    -------
    >>> engine = WaveletsEngine(wavelet_types=['haar', 'db4'], decomposition_levels=4)
    >>> features, names = engine.extract(x)
    """

    def __init__(
        self,
        wavelet_types: List[str] = None,
        decomposition_levels: int = 4,
        include_denoised: bool = True,
        verbose: bool = True,
    ):
        self.wavelet_types = wavelet_types or ['haar', 'db4']
        self.levels = decomposition_levels
        self.include_denoised = include_denoised
        self.verbose = verbose

        # Validasi
        for wt in self.wavelet_types:
            if wt not in WAVELET_FILTERS:
                raise ValueError(
                    f"Wavelet '{wt}' tidak tersedia. "
                    f"Pilihan: {list(WAVELET_FILTERS.keys())}"
                )

    def extract(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Ekstrak fitur wavelet dari batched signals.

        Parameters
        ----------
        x : torch.Tensor [B, T]
            Batched 1D signals (satu channel, misal close price).

        Returns
        -------
        features : torch.Tensor [B, N_features]
        names : list of str
        """
        B, T = x.shape
        all_features = []
        all_names = []

        for wt in self.wavelet_types:
            # Decomposition
            coeffs = wavedec(x, wt, self.levels)

            # Statistik per level
            n_actual_levels = len(coeffs)
            # coeffs[0] = approx, coeffs[1:] = details (deepest to shallowest)

            # Approximation coefficients
            feat, names = extract_coeff_stats(coeffs[0], f"wl_{wt}_a{n_actual_levels-1}")
            all_features.append(feat)
            all_names.extend(names)

            # Detail coefficients per level
            for d_idx, detail in enumerate(coeffs[1:]):
                level_num = n_actual_levels - 1 - d_idx
                feat, names = extract_coeff_stats(detail, f"wl_{wt}_d{level_num}")
                all_features.append(feat)
                all_names.extend(names)

            # Energy ratio antar level
            ratios = energy_ratio_per_level(coeffs)
            for lv in range(ratios.shape[1]):
                all_features.append(ratios[:, lv:lv+1])
                if lv == 0:
                    all_names.append(f"wl_{wt}_eratio_approx")
                else:
                    all_names.append(f"wl_{wt}_eratio_d{n_actual_levels - lv}")

        # Denoised signal statistics (optional)
        if self.include_denoised:
            denoised = wavelet_denoised(x, self.wavelet_types[0], min(3, self.levels))
            # Korelasi antara original dan denoised
            x_c = x - x.mean(dim=1, keepdim=True)
            d_c = denoised - denoised.mean(dim=1, keepdim=True)
            corr = (x_c * d_c).sum(dim=1) / (
                x_c.norm(dim=1) * d_c.norm(dim=1) + 1e-10
            )
            all_features.append(corr.unsqueeze(1))
            all_names.append("wl_denoised_corr")

            # Noise level estimate: std(original - denoised)
            noise = x - denoised
            noise_std = noise.std(dim=1).unsqueeze(1)
            all_features.append(noise_std)
            all_names.append("wl_noise_level")

        result = torch.cat(all_features, dim=1)

        # Handle NaN/Inf
        result = torch.where(torch.isfinite(result), result,
                             torch.zeros_like(result))

        return result, all_names

    @property
    def n_features(self) -> int:
        """Estimasi jumlah fitur."""
        n = 0
        for wt in self.wavelet_types:
            # Stats per level: (levels+1) × 5
            n += (self.levels + 1) * 5
            # Energy ratios: levels+1
            n += self.levels + 1
        if self.include_denoised:
            n += 2  # corr + noise_level
        return n

    def get_feature_names(self) -> List[str]:
        """Preview nama fitur (tanpa menjalankan extract)."""
        names = []
        for wt in self.wavelet_types:
            # Approx
            for stat in ['energy', 'mean', 'std', 'max', 'entropy']:
                names.append(f"wl_{wt}_a{self.levels}_{stat}")
            # Details
            for lv in range(self.levels, 0, -1):
                for stat in ['energy', 'mean', 'std', 'max', 'entropy']:
                    names.append(f"wl_{wt}_d{lv}_{stat}")
            # Energy ratios
            names.append(f"wl_{wt}_eratio_approx")
            for lv in range(self.levels, 0, -1):
                names.append(f"wl_{wt}_eratio_d{lv}")
        if self.include_denoised:
            names.extend(["wl_denoised_corr", "wl_noise_level"])
        return names
