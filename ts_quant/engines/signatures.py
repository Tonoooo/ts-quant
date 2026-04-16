"""
ts_quant.engines.signatures — Engine C: Path Signatures (GPU)
==============================================================

Implementasi Path Signatures untuk menangkap interaksi urutan
(ordered interactions) antar channel dalam time-series multivariat.

Referensi: Kidger & Lyons, 2021 — "Signatory: differentiable
computations of the signature and logsignature transforms, on both
CPU and GPU"

Path signature dari path berdimensi d pada depth k menghasilkan:
    Level 0: 1 (trivial, diabaikan)
    Level 1: d fitur       (increments per channel)
    Level 2: d² fitur      (pairwise interactions)
    Level 3: d³ fitur      (third-order interactions)

Contoh: d=5 (OHLCV), depth=3 → 5 + 25 + 125 = 155 fitur

Jika `signatory` terinstall, digunakan langsung (sudah GPU-native).
Jika tidak, menggunakan implementasi PyTorch murni via iterated integrals.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════
# PURE-PYTORCH SIGNATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _compute_signature_level1(increments: torch.Tensor) -> torch.Tensor:
    """
    Level 1 signature: sum of increments per channel.

    Parameters
    ----------
    increments : [B, T-1, d]

    Returns
    -------
    sig1 : [B, d]
    """
    return increments.sum(dim=1)


def _compute_signature_level2(increments: torch.Tensor) -> torch.Tensor:
    """
    Level 2 signature: iterated integrals of order 2.

    S_{ij} = Σ_t ( Σ_{s<t} ΔX_i[s] ) × ΔX_j[t]

    Parameters
    ----------
    increments : [B, T-1, d]

    Returns
    -------
    sig2 : [B, d²]
    """
    B, L, d = increments.shape

    # Cumulative sum of increments (prefix sums)
    # cumsum[t] = Σ_{s=0..t} ΔX[s]  →  shifted by 1 for "s < t"
    cumsum = torch.cumsum(increments, dim=1)  # [B, L, d]
    # Shift: at time t, we want Σ_{s<t} ΔX[s] = cumsum[t-1]
    cumsum_shifted = torch.zeros_like(cumsum)
    cumsum_shifted[:, 1:, :] = cumsum[:, :-1, :]  # [B, L, d]

    # S_{ij} = Σ_t cumsum_shifted_i[t] × Δ_j[t]
    # Expand and multiply: [B, L, d, 1] × [B, L, 1, d] → [B, L, d, d]
    outer = cumsum_shifted.unsqueeze(3) * increments.unsqueeze(2)  # [B, L, d, d]
    sig2 = outer.sum(dim=1)  # [B, d, d]

    return sig2.reshape(B, d * d)


def _compute_signature_level3(increments: torch.Tensor) -> torch.Tensor:
    """
    Level 3 signature: iterated integrals of order 3.

    S_{ijk} = Σ_t ( S_{ij}[<t] ) × ΔX_k[t]

    Parameters
    ----------
    increments : [B, T-1, d]

    Returns
    -------
    sig3 : [B, d³]
    """
    B, L, d = increments.shape

    # Level 2 running sum: for each t, compute S_{ij}[0:t]
    cumsum_shifted = torch.zeros(B, L, d, device=increments.device,
                                 dtype=increments.dtype)
    cumsum_shifted[:, 1:, :] = torch.cumsum(increments[:, :-1, :], dim=1)

    # Running level-2 signature: cumsum of outer products
    outer_t = cumsum_shifted.unsqueeze(3) * increments.unsqueeze(2)  # [B, L, d, d]
    sig2_running = torch.cumsum(outer_t, dim=1)  # [B, L, d, d]
    # Shift for "< t"
    sig2_shifted = torch.zeros_like(sig2_running)
    sig2_shifted[:, 1:, :, :] = sig2_running[:, :-1, :, :]

    # S_{ijk} = Σ_t sig2_shifted_{ij}[t] × ΔX_k[t]
    # [B, L, d, d, 1] × [B, L, 1, 1, d] → [B, L, d, d, d]
    triple = sig2_shifted.unsqueeze(4) * increments.unsqueeze(2).unsqueeze(3)
    sig3 = triple.sum(dim=1)  # [B, d, d, d]

    return sig3.reshape(B, d * d * d)


def compute_path_signature(
    path: torch.Tensor,
    depth: int = 3,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Compute truncated path signature.

    Parameters
    ----------
    path : torch.Tensor [B, T, d]
        Batched d-dimensional paths.
    depth : int
        Truncation depth (1, 2, or 3).
    normalize : bool
        Jika True, normalize path sehingga semua channel memiliki
        skala yang sama (penting untuk data keuangan).

    Returns
    -------
    signature : torch.Tensor [B, n_features]
    n_features : int
    """
    B, T, d = path.shape

    if normalize:
        # Z-normalize per channel
        mean = path.mean(dim=1, keepdim=True)
        std = path.std(dim=1, keepdim=True).clamp(min=1e-10)
        path = (path - mean) / std

    increments = path[:, 1:, :] - path[:, :-1, :]  # [B, T-1, d]

    parts = []

    # Level 1
    sig1 = _compute_signature_level1(increments)  # [B, d]
    parts.append(sig1)

    # Level 2
    if depth >= 2:
        sig2 = _compute_signature_level2(increments)  # [B, d²]
        parts.append(sig2)

    # Level 3
    if depth >= 3:
        sig3 = _compute_signature_level3(increments)  # [B, d³]
        parts.append(sig3)

    signature = torch.cat(parts, dim=1)
    return signature, signature.shape[1]


# ═══════════════════════════════════════════════════════════════
# AUGMENTATION TRANSFORMS
# ═══════════════════════════════════════════════════════════════

def _augment_path_time(path: torch.Tensor) -> torch.Tensor:
    """
    Tambahkan dimensi waktu sebagai channel tambahan.
    Ini membuat signature memperhitungkan kecepatan perubahan.

    [B, T, d] → [B, T, d+1]
    """
    B, T, d = path.shape
    t_channel = torch.linspace(0, 1, T, device=path.device, dtype=path.dtype)
    t_channel = t_channel.unsqueeze(0).unsqueeze(2).expand(B, T, 1)
    return torch.cat([t_channel, path], dim=2)


def _augment_path_leadlag(path: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transform: menggandakan setiap channel menjadi
    versi 'lead' dan 'lag'.

    Membuat signature menangkap korelasi intra-channel.
    [B, T, d] → [B, 2T-1, 2d]
    """
    B, T, d = path.shape
    # Lead component (original, repeated)
    lead = path.repeat_interleave(2, dim=1)[:, :-1, :]  # [B, 2T-1, d]
    # Lag component (original, shifted)
    lag = path.repeat_interleave(2, dim=1)[:, 1:, :]    # [B, 2T-1, d]
    return torch.cat([lead, lag], dim=2)  # [B, 2T-1, 2d]


# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE CLASS
# ═══════════════════════════════════════════════════════════════

class SignaturesEngine:
    """
    Engine C: Path Signatures — Ordered Interaction Features.

    Parameters
    ----------
    depth : int
        Kedalaman signature truncation. Default: 3.
        depth=2 → d + d² fitur
        depth=3 → d + d² + d³ fitur
    channels : list of str
        Kolom channel yang digunakan. Default: ['close', 'volume'].
    augment_time : bool
        Tambahkan channel waktu (membuat d+1 channels).
    augment_leadlag : bool
        Terapkan lead-lag augmentation (doubles d, triples T).
        PERHATIAN: sangat meningkatkan jumlah fitur.
    normalize : bool
        Z-normalize path sebelum signature.
    window_size : int
        Ukuran window untuk menghitung signature.
    verbose : bool

    Example
    -------
    >>> engine = SignaturesEngine(depth=3, channels=['close', 'volume'])
    >>> features, names = engine.extract(windows_3d)
    >>> # windows_3d shape: [B, window_size, n_channels]
    """

    def __init__(
        self,
        depth: int = 3,
        channels: List[str] = None,
        augment_time: bool = True,
        augment_leadlag: bool = False,
        normalize: bool = True,
        window_size: int = 20,
        verbose: bool = True,
    ):
        self.depth = depth
        self.channels = channels or ['close', 'volume']
        self.augment_time = augment_time
        self.augment_leadlag = augment_leadlag
        self.normalize = normalize
        self.window_size = window_size
        self.verbose = verbose

        # Coba import signatory (opsional)
        self._use_signatory = False
        try:
            import signatory
            self._signatory = signatory
            self._use_signatory = True
            if self.verbose:
                print("    [Signatures] Using signatory (GPU-native)")
        except ImportError:
            if self.verbose:
                print("    [Signatures] Using pure-PyTorch implementation")

    def extract(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Ekstrak path signature dari batched multivariate windows.

        Parameters
        ----------
        x : torch.Tensor [B, T, d]
            Batched d-dimensional time series windows.
            d = jumlah channels.

        Returns
        -------
        features : torch.Tensor [B, N_features]
        names : list of str
        """
        if x.dim() == 2:
            x = x.unsqueeze(2)  # [B, T] → [B, T, 1]

        B, T, d = x.shape
        d_orig = d

        # ── Augmentations ──
        path = x
        if self.augment_time:
            path = _augment_path_time(path)
            d_aug = d + 1
        else:
            d_aug = d

        # Lead-lag doubles channels
        if self.augment_leadlag:
            path = _augment_path_leadlag(path)
            d_aug *= 2

        # ── Compute signature ──
        if self._use_signatory and not self.augment_leadlag:
            # Use signatory library
            try:
                sig = self._signatory.signature(path, self.depth)
            except Exception:
                sig, _ = compute_path_signature(path, self.depth, self.normalize)
        else:
            sig, _ = compute_path_signature(path, self.depth, self.normalize)

        # ── Sanitize ──
        sig = torch.where(torch.isfinite(sig), sig, torch.zeros_like(sig))

        # ── Generate names ──
        names = self._generate_names(d_aug)

        # Adjust if sizes don't match (edge case)
        if len(names) != sig.shape[1]:
            names = [f"sig_f{i}" for i in range(sig.shape[1])]

        return sig, names

    def _generate_names(self, d: int) -> List[str]:
        """Generate descriptive feature names."""
        names = []

        # Level 1
        for i in range(d):
            names.append(f"sig_L1_ch{i}")

        # Level 2
        if self.depth >= 2:
            for i in range(d):
                for j in range(d):
                    names.append(f"sig_L2_ch{i}_{j}")

        # Level 3
        if self.depth >= 3:
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        names.append(f"sig_L3_ch{i}_{j}_{k}")

        return names

    @property
    def n_features_estimate(self) -> int:
        """Estimasi jumlah fitur."""
        d = len(self.channels)
        if self.augment_time:
            d += 1
        if self.augment_leadlag:
            d *= 2

        total = 0
        for level in range(1, self.depth + 1):
            total += d ** level
        return total

    def get_info(self) -> Dict:
        """Informasi tentang konfigurasi engine."""
        d = len(self.channels)
        if self.augment_time:
            d += 1
        if self.augment_leadlag:
            d *= 2

        return {
            "depth": self.depth,
            "input_channels": len(self.channels),
            "effective_channels": d,
            "n_features": self.n_features_estimate,
            "augments": {
                "time": self.augment_time,
                "leadlag": self.augment_leadlag,
            },
            "backend": "signatory" if self._use_signatory else "pure-pytorch",
        }
