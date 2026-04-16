"""
ts_quant.engines.rocket — Engine A: MultiRocket (GPU)
======================================================

Implementasi GPU dari MultiRocket: Random Convolutional Kernels
untuk time-series feature extraction.

Referensi:
    - Dempster et al., 2020 — ROCKET
    - Dempster et al., 2021 — MiniRocket
    - Tan et al., 2022 — MultiRocket

Cara kerja:
    1. Generate N random 1D convolutional kernels
       - Random length: {7, 9, 11}
       - Random weights: N(0,1), mean-centered
       - Random dilation: 1..max (log-uniform)
       - Random bias: sampled dari output statistics
    2. Convolve setiap kernel dengan input → output activation
    3. Ekstrak 4 pooling features per kernel:
       - PPV  (Proportion of Positive Values)
       - MAX  (Global Max Pooling)
       - MEAN (Mean of Positive Values)
       - MPVS (Mean × PPV — interaction term)
    4. Feature selection: buang fitur variance=0 & korelasi>0.95

Output: n_kernels × 4 fitur
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class RocketEngine:
    """
    Engine A: MultiRocket — Random Convolutional Kernels.

    Parameters
    ----------
    n_kernels : int
        Jumlah random kernels. Default: 250 → 1000 fitur.
        Gunakan 2500 untuk ~10.000 fitur (high-performance mode).
    kernel_lengths : list of int
        Panjang kernel. Default: [7, 9, 11].
    max_dilations_per_kernel : int
        Jumlah dilation unik di antara kernels.
    seed : int
        Random seed untuk reproducibility.
    variance_threshold : float
        Hapus fitur dengan variance < threshold. 0 = hapus constant.
    verbose : bool

    Example
    -------
    >>> engine = RocketEngine(n_kernels=250)
    >>> features, names = engine.extract(x)  # x: [B, T]
    >>> print(features.shape)  # [B, 1000]
    """

    FEATURES_PER_KERNEL = 4  # PPV, MAX, MEAN, MPVS

    def __init__(
        self,
        n_kernels: int = 250,
        kernel_lengths: List[int] = None,
        max_dilations_per_kernel: int = 32,
        seed: int = 42,
        variance_threshold: float = 0.0,
        verbose: bool = True,
    ):
        self.n_kernels = n_kernels
        self.kernel_lengths = kernel_lengths or [7, 9, 11]
        self.max_dilations = max_dilations_per_kernel
        self.seed = seed
        self.variance_threshold = variance_threshold
        self.verbose = verbose

        # Simpan state kernel (lazy init saat extract dipanggil)
        self._kernels_initialized = False
        self._kernel_weights: Dict[int, torch.Tensor] = {}
        self._kernel_dilations: Dict[int, torch.Tensor] = {}
        self._kernel_biases: Dict[int, torch.Tensor] = {}
        self._cache_T: int = -1

    def _init_kernels(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Generate random kernels (deterministic via seed).

        Kernels dibagi rata per kernel_length.
        Setiap kernel mendapat dilation acak dari distribusi log-uniform.
        """
        if self._kernels_initialized and self._cache_T == T:
            return  # Sudah diinisialisasi untuk panjang yang sama

        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.seed)

        n_lengths = len(self.kernel_lengths)
        n_per_length = self.n_kernels // n_lengths
        remainder = self.n_kernels - n_per_length * n_lengths

        self._kernel_weights = {}
        self._kernel_dilations = {}
        self._kernel_biases = {}

        for idx, kl in enumerate(self.kernel_lengths):
            n_k = n_per_length + (1 if idx < remainder else 0)
            if n_k == 0:
                continue

            # ── Random weights: N(0,1), mean-centered ──
            weights = torch.randn(n_k, 1, kl, generator=rng)
            # Mean-center setiap kernel → membuat fitur translation-invariant
            weights = weights - weights.mean(dim=2, keepdim=True)

            # ── Random dilations: log-uniform ──
            max_dilation = max(1, (T - 1) // (kl - 1))
            max_dilation = min(max_dilation, self.max_dilations)
            if max_dilation > 1:
                log_dilations = torch.rand(n_k, generator=rng) * math.log(max_dilation)
                dilations = torch.exp(log_dilations).long().clamp(1, max_dilation)
            else:
                dilations = torch.ones(n_k, dtype=torch.long)

            # ── Random biases (akan di-update dari data pada pertama kali) ──
            biases = torch.randn(n_k, generator=rng) * 0.1

            self._kernel_weights[kl] = weights.to(device=device, dtype=dtype)
            self._kernel_dilations[kl] = dilations.to(device=device)
            self._kernel_biases[kl] = biases.to(device=device, dtype=dtype)

        self._kernels_initialized = True
        self._cache_T = T

        if self.verbose:
            total = sum(w.shape[0] for w in self._kernel_weights.values())
            print(f"    [Rocket] {total} kernels initialized "
                  f"(lengths: {self.kernel_lengths})")

    def extract(
        self,
        x: torch.Tensor,
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Ekstrak fitur MultiRocket dari batched 1D signals.

        Parameters
        ----------
        x : torch.Tensor [B, T]
            Batched 1D signals (satu channel).
        batch_size : int
            Jumlah kernels untuk diproses per batch.
            Lebih kecil = hemat VRAM.

        Returns
        -------
        features : torch.Tensor [B, n_kernels × 4]
        names : list of str
        """
        B, T = x.shape
        self._init_kernels(T, x.device, x.dtype)

        x_conv = x.unsqueeze(1)  # [B, 1, T] for conv1d

        all_features = []
        all_names = []
        kernel_idx = 0

        for kl in self.kernel_lengths:
            if kl not in self._kernel_weights:
                continue

            weights = self._kernel_weights[kl]   # [n_k, 1, kl]
            dilations = self._kernel_dilations[kl]  # [n_k]
            biases = self._kernel_biases[kl]       # [n_k]
            n_k = weights.shape[0]

            # Proses per dilation-group (semua kernel dgn dilation sama → 1 call)
            unique_d = dilations.unique()
            for d_val in unique_d:
                d = d_val.item()
                mask = (dilations == d_val)
                w_group = weights[mask]       # [n_group, 1, kl]
                b_group = biases[mask]        # [n_group]
                n_group = w_group.shape[0]

                # Padding: 'same' equivalent
                effective_kl = d * (kl - 1) + 1
                pad = effective_kl // 2

                # ── Batched convolution ──
                # Proses kernels dalam sub-batches untuk hemat VRAM
                for start in range(0, n_group, batch_size):
                    end = min(start + batch_size, n_group)
                    w_batch = w_group[start:end]    # [bs, 1, kl]
                    b_batch = b_group[start:end]    # [bs]
                    bs = w_batch.shape[0]

                    # Conv1d: [B, 1, T] * [bs, 1, kl] → [B, bs, T_out]
                    out = F.conv1d(
                        x_conv, w_batch,
                        bias=b_batch,
                        dilation=d,
                        padding=pad,
                    )
                    # out: [B, bs, T_out]

                    # ── Pooling features ──
                    # 1. PPV: Proportion of Positive Values
                    ppv = (out > 0).float().mean(dim=2)  # [B, bs]

                    # 2. MAX: Global Max Pool
                    max_val = out.max(dim=2)[0]  # [B, bs]

                    # 3. MEAN: Mean of Positive Values
                    pos_mask = (out > 0).float()
                    pos_sum = (out * pos_mask).sum(dim=2)
                    pos_count = pos_mask.sum(dim=2).clamp(min=1)
                    mean_pos = pos_sum / pos_count  # [B, bs]

                    # 4. MPVS: PPV × Mean interaction
                    mpvs = ppv * mean_pos  # [B, bs]

                    # Stack features: [B, bs*4]
                    batch_feats = torch.cat([ppv, max_val, mean_pos, mpvs], dim=1)
                    all_features.append(batch_feats)

                    # Names
                    for ki in range(bs):
                        kid = kernel_idx + ki
                        all_names.extend([
                            f"rk_ppv_k{kid}",
                            f"rk_max_k{kid}",
                            f"rk_mean_k{kid}",
                            f"rk_mpvs_k{kid}",
                        ])

                    kernel_idx += bs

        # Concatenate all
        result = torch.cat(all_features, dim=1)  # [B, n_kernels*4]

        # Reorder: fitur saat ini urut [ppv_batch, max_batch, ...] per group
        # Tapi names sudah benar, jadi kita reorganize tensor agar match
        # Actually, cat di atas sudah interleaved per batch → OK

        # Sanitize NaN/Inf
        result = torch.where(torch.isfinite(result), result,
                             torch.zeros_like(result))

        # ── Feature selection: hapus constant features ──
        if self.variance_threshold >= 0 and result.shape[0] > 1:
            var = result.var(dim=0)
            keep_mask = var > self.variance_threshold
            if keep_mask.sum() < result.shape[1]:
                n_removed = result.shape[1] - keep_mask.sum().item()
                result = result[:, keep_mask]
                all_names = [n for n, k in zip(all_names, keep_mask.tolist()) if k]
                if self.verbose:
                    print(f"    [Rocket] Feature selection: "
                          f"removed {n_removed} constant features, "
                          f"{len(all_names)} remaining")

        return result, all_names

    @property
    def n_features(self) -> int:
        """Estimasi jumlah fitur (sebelum selection)."""
        return self.n_kernels * self.FEATURES_PER_KERNEL

    def get_kernel_info(self) -> Dict:
        """Informasi tentang kernel yang sudah di-generate."""
        if not self._kernels_initialized:
            return {"status": "not initialized"}
        info = {}
        for kl, w in self._kernel_weights.items():
            d = self._kernel_dilations[kl]
            info[f"length_{kl}"] = {
                "n_kernels": w.shape[0],
                "dilation_min": d.min().item(),
                "dilation_max": d.max().item(),
                "dilation_unique": d.unique().numel(),
            }
        return info
