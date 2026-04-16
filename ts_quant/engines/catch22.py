"""
ts_quant.engines.catch22 — Engine B: Catch22 (22 Canonical Features)
=====================================================================

Implementasi GPU (PyTorch) dari 22 fitur time-series kanonik.
Referensi: Lubba et al., 2019 — "catch22: CAnonical Time-series
CHaracteristics"

Semua fungsi bekerja pada input batched: [B, T]
    B = jumlah windows (dari semua saham)
    T = panjang window

Output: [B, 22] — satu nilai per fitur per window.
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _safe_std(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard deviation dengan proteksi div-by-zero."""
    std = x.std(dim=dim)
    return std.clamp(min=1e-10)


def _z_normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-normalize per baris: (x - mean) / std. [B, T] -> [B, T]"""
    mean = x.mean(dim=1, keepdim=True)
    std = _safe_std(x, dim=1).unsqueeze(1)
    return (x - mean) / std


def _autocorrelation_fft(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Hitung ACF menggunakan FFT (sangat cepat, fully vectorized).
    Input:  [B, T]
    Output: [B, max_lag]
    """
    B, T = x.shape
    x_c = x - x.mean(dim=1, keepdim=True)
    n_fft = 2 * T
    X = torch.fft.rfft(x_c, n=n_fft, dim=1)
    S = X * X.conj()
    acf = torch.fft.irfft(S, n=n_fft, dim=1)[:, :T]
    acf0 = acf[:, 0:1].clamp(min=1e-10)
    acf = acf / acf0
    return acf[:, :min(max_lag, T)]


def _histogram_mode(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Modus histogram (bin center dengan frekuensi tertinggi).
    Input:  [B, T]
    Output: [B]
    """
    x_min, x_max = x.min(dim=1, keepdim=True)[0], x.max(dim=1, keepdim=True)[0]
    rng = x_max - x_min
    
    normalized = (x - x_min) / (rng + 1e-10)
    bin_indices = (normalized * n_bins).long().clamp(0, n_bins - 1)
    
    counts = F.one_hot(bin_indices, num_classes=n_bins).sum(dim=1)  # [B, n_bins]
    mode_bin = counts.argmax(dim=1)  # [B]
    
    mode_vals = x_min.squeeze(1) + (mode_bin.float() + 0.5) * (rng.squeeze(1) / n_bins)
    
    is_const = rng.squeeze(1) < 1e-10
    mode_vals = torch.where(is_const, x_min.squeeze(1), mode_vals)
    return mode_vals


def _longest_stretch(binary: torch.Tensor, target: int) -> torch.Tensor:
    """
    Panjang streak terpanjang dari nilai target di binary sequence.
    Input:  [B, T] (0/1)
    Output: [B]
    """
    B, T = binary.shape
    if target == 0:
        binary = 1 - binary

    result = torch.zeros(B, device=binary.device, dtype=binary.dtype)
    current = torch.zeros(B, device=binary.device, dtype=binary.dtype)

    for t in range(T):
        is_target = binary[:, t]
        current = (current + 1) * is_target  # reset jika bukan target
        result = torch.max(result, current)

    return result


def _count_transitions(symbols: torch.Tensor, n_symbols: int,
                       from_s: int, to_s: int) -> torch.Tensor:
    """
    Hitung transisi from_s → to_s di sequence simbolik.
    Input:  [B, T] (integer 0..n_symbols-1)
    Output: [B] (count)
    """
    is_from = (symbols[:, :-1] == from_s)
    is_to = (symbols[:, 1:] == to_s)
    return (is_from & is_to).sum(dim=1).float()


# ═══════════════════════════════════════════════════════════════
# 22 CATCH22 FEATURES
# ═══════════════════════════════════════════════════════════════

def f01_dn_histogram_mode_5(x: torch.Tensor) -> torch.Tensor:
    """DN_HistogramMode_5: Mode of 5-bin histogram."""
    return _histogram_mode(x, 5)


def f02_dn_histogram_mode_10(x: torch.Tensor) -> torch.Tensor:
    """DN_HistogramMode_10: Mode of 10-bin histogram."""
    return _histogram_mode(x, 10)


def f03_sb_binarystats_diff_longstretch0(x: torch.Tensor) -> torch.Tensor:
    """SB_BinaryStats_diff_longstretch0:
    Longest stretch of 0s in binarized first differences."""
    diff = x[:, 1:] - x[:, :-1]
    mean_diff = diff.mean(dim=1, keepdim=True)
    binary = (diff > mean_diff).long()
    return _longest_stretch(binary, target=0)


def f04_sb_binarystats_mean_longstretch1(x: torch.Tensor) -> torch.Tensor:
    """SB_BinaryStats_mean_longstretch1:
    Longest stretch above mean."""
    mean = x.mean(dim=1, keepdim=True)
    binary = (x > mean).long()
    return _longest_stretch(binary, target=1)


def f05_sb_motifthree_quantile_hh(x: torch.Tensor) -> torch.Tensor:
    """SB_MotifThree_quantile_hh:
    Proportion of 'high-high' 2-letter words in 3-letter quantile alphabet."""
    B, T = x.shape
    # Discretize ke 3 bins via quantiles
    q33 = torch.quantile(x, 0.333, dim=1, keepdim=True)
    q67 = torch.quantile(x, 0.667, dim=1, keepdim=True)
    symbols = torch.zeros_like(x, dtype=torch.long)
    symbols[x > q67] = 2  # high
    symbols[(x > q33) & (x <= q67)] = 1  # medium

    # Count high-high (2→2) transitions
    hh = _count_transitions(symbols, 3, from_s=2, to_s=2)
    total_trans = (T - 1)
    return hh / max(total_trans, 1)


def f06_fc_localsimple_mean1_tauresrat(x: torch.Tensor) -> torch.Tensor:
    """FC_LocalSimple_mean1_tauresrat:
    Ratio of first zero crossing of ACF of residuals from 1-step mean forecast."""
    # Residuals: e[t] = x[t] - x[t-1]
    residuals = x[:, 1:] - x[:, :-1]
    B, T_r = residuals.shape
    max_lag = min(T_r, 40)
    acf_res = _autocorrelation_fft(residuals, max_lag)

    # First zero crossing of ACF of residuals
    tau_res = torch.ones(B, device=x.device, dtype=x.dtype)
    for lag in range(1, max_lag):
        crossed = (acf_res[:, lag] < 0) & (tau_res == 1)
        tau_res = torch.where(crossed, torch.tensor(float(lag), device=x.device),
                              tau_res)

    # ACF of original
    acf_orig = _autocorrelation_fft(x, max_lag)
    tau_orig = torch.ones(B, device=x.device, dtype=x.dtype)
    for lag in range(1, max_lag):
        crossed = (acf_orig[:, lag] < 0) & (tau_orig == 1)
        tau_orig = torch.where(crossed, torch.tensor(float(lag), device=x.device),
                               tau_orig)

    return tau_res / tau_orig.clamp(min=1)


def f07_co_embed2_dist_tau_expfit_meandiff(x: torch.Tensor) -> torch.Tensor:
    """CO_Embed2_Dist_tau_d_expfit_meandiff:
    Mean distance in 2D time-delay embedding space."""
    B, T = x.shape
    xn = _z_normalize(x)

    # Use tau=1 (simplified; original uses first ACF zero crossing)
    tau = 1
    # 2D embedding: (x[t], x[t+tau])
    x1 = xn[:, :-tau]   # [B, T-tau]
    x2 = xn[:, tau:]     # [B, T-tau]

    # Pairwise distances (approximation: use successive pairs)
    dx = x1[:, 1:] - x1[:, :-1]
    dy = x2[:, 1:] - x2[:, :-1]
    dists = torch.sqrt(dx**2 + dy**2 + 1e-10)

    return dists.mean(dim=1)


def f08_co_f1ecac(x: torch.Tensor) -> torch.Tensor:
    """CO_f1ecac: First 1/e crossing of ACF."""
    B, T = x.shape
    max_lag = min(T, 40)
    acf = _autocorrelation_fft(x, max_lag)
    threshold = 1.0 / math.e

    # Cari lag pertama dimana ACF < 1/e
    result = torch.full((B,), float(max_lag), device=x.device, dtype=x.dtype)
    for lag in range(1, max_lag):
        crossed = (acf[:, lag] < threshold) & (result == max_lag)
        result = torch.where(crossed,
                             torch.tensor(float(lag), device=x.device), result)
    return result


def f09_co_firstmin_ac(x: torch.Tensor) -> torch.Tensor:
    """CO_FirstMin_ac: First minimum of ACF."""
    B, T = x.shape
    max_lag = min(T, 40)
    acf = _autocorrelation_fft(x, max_lag)

    result = torch.full((B,), float(max_lag), device=x.device, dtype=x.dtype)
    for lag in range(2, max_lag):
        is_min = (acf[:, lag] > acf[:, lag - 1]) & (result == max_lag)
        result = torch.where(is_min,
                             torch.tensor(float(lag - 1), device=x.device),
                             result)
    return result


def f10_co_histogram_ami_even_2_5(x: torch.Tensor) -> torch.Tensor:
    """CO_HistogramAMI_even_2_5:
    Auto mutual information using 5-bin histogram at lag 2."""
    B, T = x.shape
    n_bins = 5
    lag = 2

    x1 = x[:, :-lag]  # [B, T-lag]
    x2 = x[:, lag:]   # [B, T-lag]
    
    if x1.shape[1] < 5:
        return torch.zeros(B, device=x.device, dtype=x.dtype)
        
    x1_min, x1_max = x1.min(dim=1, keepdim=True)[0], x1.max(dim=1, keepdim=True)[0]
    x2_min, x2_max = x2.min(dim=1, keepdim=True)[0], x2.max(dim=1, keepdim=True)[0]
    
    x1_norm = (x1 - x1_min) / ((x1_max - x1_min) + 1e-10)
    x2_norm = (x2 - x2_min) / ((x2_max - x2_min) + 1e-10)
    
    b1 = (x1_norm * n_bins).long().clamp(0, n_bins - 1)  # [B, N]
    b2 = (x2_norm * n_bins).long().clamp(0, n_bins - 1)  # [B, N]
    
    joint_encoded = b1 * n_bins + b2
    
    joint_counts = F.one_hot(joint_encoded, num_classes=n_bins*n_bins).sum(dim=1).float()  # [B, 25]
    joint_probs = joint_counts / x1.shape[1]
    
    p1 = joint_probs.view(B, n_bins, n_bins).sum(dim=2)  # [B, 5]
    p2 = joint_probs.view(B, n_bins, n_bins).sum(dim=1)  # [B, 5]
    
    p1_p2_flat = (p1.unsqueeze(2) * p2.unsqueeze(1)).view(B, 25)
    
    valid = (joint_probs > 0) & (p1_p2_flat > 0)
    mi = torch.zeros(B, device=x.device, dtype=x.dtype)
    mi = torch.where(valid, joint_probs * torch.log(joint_probs / p1_p2_flat), mi)
    
    return mi.sum(dim=1)


def f11_co_trev_1_num(x: torch.Tensor) -> torch.Tensor:
    """CO_trev_1_num: Time-reversibility statistic.
    mean((x[t+1] - x[t])^3)"""
    diff = x[:, 1:] - x[:, :-1]
    return (diff ** 3).mean(dim=1)


def f12_dn_outlierinclude_p_001_mdrmd(x: torch.Tensor) -> torch.Tensor:
    """DN_OutlierInclude_p_001_mdrmd:
    Effect on median of removing positive outliers."""
    xn = _z_normalize(x)
    B, T = xn.shape
    result = torch.zeros(B, device=x.device, dtype=x.dtype)
    
    # fast cpu operations over lists to avoid slow cuda loops
    xn_cpu = xn.cpu().numpy()
    std_cpu = xn_cpu.std(axis=1)
    
    import numpy as np

    for i in range(B):
        xi = xn_cpu[i]
        meds = []
        s = std_cpu[i]
        for thresh_step in range(1, 20):
            thresh = thresh_step * 0.01 * s
            included = xi[xi <= thresh]
            if len(included) > 0:
                meds.append(np.median(included))
        if len(meds) > 1:
            meds_arr = np.array(meds)
            result[i] = np.median(np.abs(meds_arr[1:] - meds_arr[:-1]))
    return result


def f13_dn_outlierinclude_n_001_mdrmd(x: torch.Tensor) -> torch.Tensor:
    """DN_OutlierInclude_n_001_mdrmd:
    Effect on median of removing negative outliers."""
    xn = _z_normalize(x)
    B, T = xn.shape
    result = torch.zeros(B, device=x.device, dtype=x.dtype)
    
    xn_cpu = xn.cpu().numpy()
    std_cpu = xn_cpu.std(axis=1)
    
    import numpy as np

    for i in range(B):
        xi = xn_cpu[i]
        meds = []
        s = std_cpu[i]
        for thresh_step in range(1, 20):
            thresh = -thresh_step * 0.01 * s
            included = xi[xi >= thresh]
            if len(included) > 0:
                meds.append(np.median(included))
        if len(meds) > 1:
            meds_arr = np.array(meds)
            result[i] = np.median(np.abs(meds_arr[1:] - meds_arr[:-1]))
    return result


def f14_in_automutualinfo_40_gaussian_fmmi(x: torch.Tensor) -> torch.Tensor:
    """IN_AutoMutualInfoStats_40_gaussian_fmmi:
    First minimum of Gaussian auto mutual information."""
    B, T = x.shape
    max_lag = min(40, T - 1)
    acf = _autocorrelation_fft(x, max_lag + 1)

    # Gaussian MI = -0.5 * log(1 - acf^2)
    acf_sq = acf[:, 1:].clamp(-0.999, 0.999) ** 2
    gmi = -0.5 * torch.log(1 - acf_sq + 1e-10)

    # First minimum
    result = torch.full((B,), float(max_lag), device=x.device, dtype=x.dtype)
    for lag in range(1, max_lag - 1):
        is_min = (gmi[:, lag] > gmi[:, lag - 1]) & (result == max_lag)
        result = torch.where(is_min,
                             torch.tensor(float(lag), device=x.device), result)
    return result


def f15_md_hrv_classic_pnn40(x: torch.Tensor) -> torch.Tensor:
    """MD_hrv_classic_pnn40:
    Proportion of successive differences > 0.4 * std(differences)."""
    diff = (x[:, 1:] - x[:, :-1]).abs()
    std_diff = _safe_std(diff, dim=1)
    threshold = 0.4 * std_diff
    pnn40 = (diff > threshold.unsqueeze(1)).float().mean(dim=1)
    return pnn40


def f16_sc_fluctanal_dfa(x: torch.Tensor) -> torch.Tensor:
    """SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1:
    DFA (Detrended Fluctuation Analysis) scaling exponent."""
    B, T = x.shape
    xn = _z_normalize(x)

    # Cumulative sum (profile)
    profile = torch.cumsum(xn, dim=1)  # [B, T]

    # Segment lengths (log-spaced)
    min_seg = 4
    max_seg = T // 4
    if max_seg < min_seg:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    n_segs = min(15, max_seg - min_seg + 1)
    seg_lengths = torch.logspace(
        math.log10(min_seg), math.log10(max_seg), n_segs,
        device=x.device
    ).long().unique()

    log_n = []
    log_f = []

    for seg_len in seg_lengths:
        sl = seg_len.item()
        n_full_segs = T // sl
        if n_full_segs < 1:
            continue

        usable = n_full_segs * sl
        segments = profile[:, :usable].reshape(B, n_full_segs, sl)  # [B, K, sl]

        # Linear detrend per segment
        t_axis = torch.arange(sl, device=x.device, dtype=x.dtype)
        t_mean = t_axis.mean()
        t_centered = t_axis - t_mean

        # Batch linear regression: y = a + b*t
        # b = sum(t_c * y_c) / sum(t_c^2)
        y_mean = segments.mean(dim=2, keepdim=True)
        y_centered = segments - y_mean
        tt = (t_centered ** 2).sum()
        b = (y_centered * t_centered.unsqueeze(0).unsqueeze(0)).sum(dim=2, keepdim=True)
        b = b / tt.clamp(min=1e-10)
        trend = y_mean + b * t_centered.unsqueeze(0).unsqueeze(0)
        residual = segments - trend

        # RMS fluctuation
        f_n = torch.sqrt((residual ** 2).mean(dim=2).mean(dim=1) + 1e-10)

        log_n.append(math.log(sl))
        log_f.append(torch.log(f_n + 1e-10))

    if len(log_n) < 2:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    # Linear regression in log-log space → DFA exponent
    log_n_t = torch.tensor(log_n, device=x.device, dtype=x.dtype)
    log_f_t = torch.stack(log_f, dim=1)  # [B, n_segs]

    n_mean = log_n_t.mean()
    n_c = log_n_t - n_mean
    f_mean = log_f_t.mean(dim=1, keepdim=True)
    f_c = log_f_t - f_mean

    slope = (f_c * n_c.unsqueeze(0)).sum(dim=1) / (n_c ** 2).sum().clamp(min=1e-10)
    return slope


def f17_sc_fluctanal_rsrangefit(x: torch.Tensor) -> torch.Tensor:
    """SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1:
    RS (Rescaled Range) analysis scaling."""
    B, T = x.shape
    xn = _z_normalize(x)

    min_seg = 4
    max_seg = T // 4
    if max_seg < min_seg:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    n_segs = min(15, max_seg - min_seg + 1)
    seg_lengths = torch.logspace(
        math.log10(min_seg), math.log10(max_seg), n_segs,
        device=x.device
    ).long().unique()

    log_n = []
    log_rs = []

    for seg_len in seg_lengths:
        sl = seg_len.item()
        n_full = T // sl
        if n_full < 1:
            continue
        usable = n_full * sl
        segments = xn[:, :usable].reshape(B, n_full, sl)

        # Mean-centered cumulative sum per segment
        seg_mean = segments.mean(dim=2, keepdim=True)
        cum = torch.cumsum(segments - seg_mean, dim=2)

        # R = max - min of cumulative
        R = cum.max(dim=2)[0] - cum.min(dim=2)[0]  # [B, n_full]
        # S = std per segment
        S = segments.std(dim=2).clamp(min=1e-10)
        # RS ratio
        rs = (R / S).mean(dim=1)

        log_n.append(math.log(sl))
        log_rs.append(torch.log(rs + 1e-10))

    if len(log_n) < 2:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    log_n_t = torch.tensor(log_n, device=x.device, dtype=x.dtype)
    log_rs_t = torch.stack(log_rs, dim=1)

    n_mean = log_n_t.mean()
    n_c = log_n_t - n_mean
    f_mean = log_rs_t.mean(dim=1, keepdim=True)
    f_c = log_rs_t - f_mean
    slope = (f_c * n_c.unsqueeze(0)).sum(dim=1) / (n_c ** 2).sum().clamp(min=1e-10)
    return slope


def f18_sp_summaries_welch_rect_area_5_1(x: torch.Tensor) -> torch.Tensor:
    """SP_Summaries_welch_rect_area_5_1:
    Proportion of power in lowest 1/5 of frequencies (Welch PSD)."""
    B, T = x.shape
    xn = _z_normalize(x)

    # Compute PSD via FFT
    X = torch.fft.rfft(xn, dim=1)
    psd = (X * X.conj()).real  # [B, T//2+1]

    n_freqs = psd.shape[1]
    cutoff = max(1, n_freqs // 5)

    low_power = psd[:, :cutoff].sum(dim=1)
    total_power = psd.sum(dim=1).clamp(min=1e-10)

    return low_power / total_power


def f19_sp_summaries_welch_rect_centroid(x: torch.Tensor) -> torch.Tensor:
    """SP_Summaries_welch_rect_centroid:
    Centroid (center of mass) of power spectral density."""
    B, T = x.shape
    xn = _z_normalize(x)

    X = torch.fft.rfft(xn, dim=1)
    psd = (X * X.conj()).real  # [B, n_freqs]

    n_freqs = psd.shape[1]
    freqs = torch.arange(n_freqs, device=x.device, dtype=x.dtype)

    total_power = psd.sum(dim=1).clamp(min=1e-10)
    centroid = (psd * freqs.unsqueeze(0)).sum(dim=1) / total_power
    # Normalize by n_freqs
    return centroid / max(n_freqs, 1)


def f20_sb_transitionmatrix_3ac_sumdiagcov(x: torch.Tensor) -> torch.Tensor:
    """SB_TransitionMatrix_3ac_sumdiagcov:
    Sum of diagonal of covariance of 3-state transition matrix."""
    B, T = x.shape

    # Discretize into 3 states via quantiles
    q33 = torch.quantile(x, 0.333, dim=1, keepdim=True)
    q67 = torch.quantile(x, 0.667, dim=1, keepdim=True)
    symbols = torch.zeros_like(x, dtype=torch.long)
    symbols[x > q67] = 2
    symbols[(x > q33) & (x <= q67)] = 1

    s_from = symbols[:, :-1]
    s_to = symbols[:, 1:]
    encoded = s_from * 3 + s_to  # 0..8
    counts = F.one_hot(encoded, num_classes=9).sum(dim=1).float()  # [B, 9]
    tm = counts.view(B, 3, 3)
    
    # Row-normalize
    row_sums = tm.sum(dim=2, keepdim=True).clamp(min=1)
    tm = tm / row_sums  # [B, 3, 3]
    
    # Covariance of rows of tm
    tm_mean = tm.mean(dim=1, keepdim=True)
    tm_c = tm - tm_mean
    # cov_diag = Var(each col over 3 rows) = sum(xi - x_mean)^2 / (3-1)
    cov_diag = (tm_c ** 2).sum(dim=1) / 2.0  # [B, 3]
    
    return cov_diag.sum(dim=1)


def f21_pd_periodicitywang_th001(x: torch.Tensor) -> torch.Tensor:
    """PD_PeriodicityWang_th0_01:
    Wang periodicity detection using ACF peaks."""
    B, T = x.shape
    max_lag = min(T - 1, 40)
    acf = _autocorrelation_fft(x, max_lag)

    # Find first peak of ACF after lag 0
    result = torch.zeros(B, device=x.device, dtype=x.dtype)
    for lag in range(2, max_lag - 1):
        is_peak = ((acf[:, lag] > acf[:, lag - 1]) &
                   (acf[:, lag] > acf[:, lag + 1]) &
                   (acf[:, lag] > 0.01) &
                   (result == 0))
        result = torch.where(is_peak,
                             torch.tensor(float(lag), device=x.device), result)
    return result


def f22_fc_localsimple_mean3_stderr(x: torch.Tensor) -> torch.Tensor:
    """FC_LocalSimple_mean3_stderr:
    Standard error of 3-step mean prediction residuals."""
    B, T = x.shape
    if T < 4:
        return torch.zeros(B, device=x.device, dtype=x.dtype)

    # Predict x[t] as mean of x[t-3], x[t-2], x[t-1]
    # Using conv1d with kernel [1/3, 1/3, 1/3]
    kernel = torch.ones(1, 1, 3, device=x.device, dtype=x.dtype) / 3.0
    x_pad = x.unsqueeze(1)  # [B, 1, T]
    prediction = F.conv1d(x_pad, kernel)[:, 0, :]  # [B, T-2]

    # Residuals: actual[3:] - prediction[:-1]
    residuals = x[:, 3:] - prediction[:, :-1]
    stderr = _safe_std(residuals, dim=1)
    return stderr


# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE CLASS
# ═══════════════════════════════════════════════════════════════

# Daftar semua 22 fungsi
CATCH22_FUNCTIONS = [
    ("DN_HistogramMode_5", f01_dn_histogram_mode_5),
    ("DN_HistogramMode_10", f02_dn_histogram_mode_10),
    ("SB_BinaryStats_diff_longstretch0", f03_sb_binarystats_diff_longstretch0),
    ("SB_BinaryStats_mean_longstretch1", f04_sb_binarystats_mean_longstretch1),
    ("SB_MotifThree_quantile_hh", f05_sb_motifthree_quantile_hh),
    ("FC_LocalSimple_mean1_tauresrat", f06_fc_localsimple_mean1_tauresrat),
    ("CO_Embed2_Dist_tau_expfit_meandiff", f07_co_embed2_dist_tau_expfit_meandiff),
    ("CO_f1ecac", f08_co_f1ecac),
    ("CO_FirstMin_ac", f09_co_firstmin_ac),
    ("CO_HistogramAMI_even_2_5", f10_co_histogram_ami_even_2_5),
    ("CO_trev_1_num", f11_co_trev_1_num),
    ("DN_OutlierInclude_p_001_mdrmd", f12_dn_outlierinclude_p_001_mdrmd),
    ("DN_OutlierInclude_n_001_mdrmd", f13_dn_outlierinclude_n_001_mdrmd),
    ("IN_AutoMutualInfoStats_40_gaussian_fmmi", f14_in_automutualinfo_40_gaussian_fmmi),
    ("MD_hrv_classic_pnn40", f15_md_hrv_classic_pnn40),
    ("SC_FluctAnal_2_dfa", f16_sc_fluctanal_dfa),
    ("SC_FluctAnal_2_rsrangefit", f17_sc_fluctanal_rsrangefit),
    ("SP_Summaries_welch_rect_area_5_1", f18_sp_summaries_welch_rect_area_5_1),
    ("SP_Summaries_welch_rect_centroid", f19_sp_summaries_welch_rect_centroid),
    ("SB_TransitionMatrix_3ac_sumdiagcov", f20_sb_transitionmatrix_3ac_sumdiagcov),
    ("PD_PeriodicityWang_th0_01", f21_pd_periodicitywang_th001),
    ("FC_LocalSimple_mean3_stderr", f22_fc_localsimple_mean3_stderr),
]


class Catch22Engine:
    """
    Engine B: Catch22 — 22 Canonical Time-Series Characteristics.

    Mengekstrak 22 fitur kanonik dari setiap rolling window.
    Input diproses secara batched untuk efisiensi GPU.

    Parameters
    ----------
    window_sizes : list of int
        Ukuran rolling window. Default: [10, 20, 60].
        22 fitur × len(window_sizes) = total fitur output.
    verbose : bool
        Tampilkan progress.

    Example
    -------
    >>> engine = Catch22Engine(window_sizes=[20])
    >>> features, names = engine.extract(windows_tensor)
    >>> print(features.shape)  # [B, 22]
    """

    def __init__(self, window_sizes: list = None, verbose: bool = True):
        self.window_sizes = window_sizes or [10, 20, 60]
        self.verbose = verbose

    def extract(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Ekstrak semua 22 fitur dari batched windows.

        Parameters
        ----------
        x : torch.Tensor
            Tensor 2D [B, T] — satu channel dari rolling windows.
            B = total windows, T = window size.

        Returns
        -------
        features : torch.Tensor
            [B, 22] — 22 fitur per window.
        names : list of str
            Nama-nama fitur.
        """
        B, T = x.shape
        features = []
        names = []

        for name, func in CATCH22_FUNCTIONS:
            try:
                feat = func(x)  # [B]
                # Handle NaN/Inf
                feat = torch.where(torch.isfinite(feat), feat,
                                   torch.zeros_like(feat))
                features.append(feat)
                names.append(name)
            except Exception as e:
                if self.verbose:
                    print(f"  [Catch22] Warning: {name} failed: {e}")
                features.append(torch.zeros(B, device=x.device, dtype=x.dtype))
                names.append(name)

        result = torch.stack(features, dim=1)  # [B, 22]
        return result, names

    def get_feature_names(self, prefix: str = "c22") -> List[str]:
        """Mengembalikan nama fitur dengan prefix."""
        return [f"{prefix}_{name}" for name, _ in CATCH22_FUNCTIONS]

    @property
    def n_features(self) -> int:
        return 22
