"""
Smoke Test — Fase 4: MultiRocket + Path Signatures + Feature Selection
========================================================================

Test:
1. RocketEngine — Random Conv1D kernels, pooling features
2. SignaturesEngine — Path signature computation (depth 1-3)
3. Feature Selection — variance, correlation, NaN filters
4. Integrasi semua engine
"""

import sys
import time
import torch

sys.path.insert(0, '.')

from ts_quant.engines.rocket import RocketEngine
from ts_quant.engines.signatures import (
    SignaturesEngine,
    compute_path_signature,
    _augment_path_time,
)
from ts_quant.utils.feature_selection import (
    auto_select_features,
    remove_constant_features,
    remove_correlated_features,
)
from ts_quant.engines.catch22 import Catch22Engine
from ts_quant.engines.wavelets import WaveletsEngine
from ts_quant.engines.tsfresh_core import TsfreshEngine
from ts_quant.core.windowing import create_rolling_windows


def create_signals(B=30, T=60):
    t = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    trend = 0.05 * t * torch.randn(B, 1)
    cycle = torch.sin(2 * 3.14159 * torch.rand(B, 1) * 0.2 * t)
    noise = torch.randn(B, T) * 0.3
    return trend + cycle + noise + 100


def test_rocket_basic():
    print("\n" + "="*60)
    print(" TEST 1: MultiRocket — Basic Extraction")
    print("="*60)

    engine = RocketEngine(n_kernels=60, seed=42, verbose=True)
    x = create_signals(B=20, T=40)
    print(f"  Input: {x.shape}")

    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"  Output: {features.shape}")
    print(f"  Fitur: {len(names)}")
    print(f"  Waktu: {dt:.3f}s")

    assert features.shape[0] == 20
    assert features.shape[1] == len(names)
    assert torch.isfinite(features).all(), "Ada NaN/Inf"

    # Check kernel info
    info = engine.get_kernel_info()
    print(f"  Kernel info: {info}")

    print("\n  ✅ Rocket basic OK")


def test_rocket_large():
    print("\n" + "="*60)
    print(" TEST 2: MultiRocket — Large Config (250 kernels)")
    print("="*60)

    engine = RocketEngine(n_kernels=250, seed=42, verbose=True)
    x = create_signals(B=50, T=60)

    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"  Output: {features.shape}")
    print(f"  Waktu: {dt:.3f}s")
    assert features.shape[0] == 50
    assert torch.isfinite(features).all()

    print("\n  ✅ Rocket large OK")


def test_rocket_edge_cases():
    print("\n" + "="*60)
    print(" TEST 3: MultiRocket — Edge Cases")
    print("="*60)

    engine = RocketEngine(n_kernels=30, seed=42, verbose=False)

    # Constant signal
    const = torch.ones(5, 30) * 42.0
    f, n = engine.extract(const)
    assert torch.isfinite(f).all(), "Constant: NaN/Inf"
    print(f"  Constant (5×30): {f.shape} ✓")

    # Short signal
    engine2 = RocketEngine(n_kernels=30, seed=42, verbose=False)
    short = create_signals(5, 15)
    f, n = engine2.extract(short)
    assert torch.isfinite(f).all()
    print(f"  Short (5×15): {f.shape} ✓")

    # Reproducibility (same seed = same output)
    e1 = RocketEngine(n_kernels=30, seed=42, verbose=False)
    e2 = RocketEngine(n_kernels=30, seed=42, verbose=False)
    x = create_signals(5, 30)
    f1, _ = e1.extract(x)
    f2, _ = e2.extract(x)
    assert torch.allclose(f1, f2, atol=1e-6), "Reproducibility failed"
    print("  Reproducibility (same seed): ✓")

    print("\n  ✅ Rocket edge cases OK")


def test_signatures_basic():
    print("\n" + "="*60)
    print(" TEST 4: Path Signatures — Basic")
    print("="*60)

    # Multivariate path: [B, T, d]
    B, T, d = 20, 30, 3
    path = torch.randn(B, T, d)
    print(f"  Input: [{B}, {T}, {d}]")

    engine = SignaturesEngine(
        depth=3,
        channels=['ch0', 'ch1', 'ch2'],
        augment_time=True,
        verbose=True,
    )
    info = engine.get_info()
    print(f"  Config: {info}")

    t0 = time.time()
    features, names = engine.extract(path)
    dt = time.time() - t0

    print(f"  Output: {features.shape}")
    print(f"  Fitur: {len(names)}")
    print(f"  Waktu: {dt:.3f}s")
    print(f"  Estimasi: {engine.n_features_estimate}")

    assert features.shape[0] == B
    assert features.shape[1] == len(names)
    assert torch.isfinite(features).all(), "Ada NaN/Inf"

    print("\n  ✅ Signatures basic OK")


def test_signatures_depths():
    print("\n" + "="*60)
    print(" TEST 5: Path Signatures — Different Depths")
    print("="*60)

    path = torch.randn(10, 20, 2)  # 2D path

    for depth in [1, 2, 3]:
        sig, n = compute_path_signature(path, depth, normalize=True)
        expected = sum(2 ** k for k in range(1, depth + 1))
        print(f"  Depth {depth}: {sig.shape[1]} fitur (expected ~{expected})")
        assert sig.shape[1] == expected, f"Depth {depth}: expected {expected}, got {sig.shape[1]}"
        assert torch.isfinite(sig).all()

    # Time augmented (d=2 → d=3)
    eng = SignaturesEngine(depth=2, channels=['a', 'b'], augment_time=True, verbose=False)
    f, n = eng.extract(path)
    expected_aug = 3 + 3**2  # d=3 with time
    assert f.shape[1] == expected_aug
    print(f"  With time augment (d=3): {f.shape[1]} fitur ✓")

    print("\n  ✅ Signatures depths OK")


def test_signatures_2d_input():
    print("\n" + "="*60)
    print(" TEST 6: Path Signatures — 2D Input (auto-expand)")
    print("="*60)

    # Input tanpa dim channel → harus auto-expand ke [B, T, 1]
    x = torch.randn(10, 30)
    eng = SignaturesEngine(depth=3, channels=['close'], augment_time=True, verbose=False)
    f, n = eng.extract(x)  # d=1+1(time)=2
    expected = 2 + 4 + 8  # d=2, depth=3
    print(f"  2D input [{10}, {30}] → features: {f.shape}")
    assert f.shape[1] == expected
    assert torch.isfinite(f).all()

    print("\n  ✅ Signatures 2D input OK")


def test_feature_selection():
    print("\n" + "="*60)
    print(" TEST 7: Feature Selection")
    print("="*60)

    B = 50
    # Create features: some constant, some correlated, some with NaN
    feats = torch.randn(B, 20)
    names = [f"f{i}" for i in range(20)]

    # Add constant features
    feats[:, 0] = 42.0
    feats[:, 1] = 0.0

    # Add correlated features (f3 ≈ f2)
    feats[:, 3] = feats[:, 2] + torch.randn(B) * 0.01

    # Add NaN features
    feats[:5, 15] = float('nan')

    print(f"  Input: {feats.shape}, {len(names)} fitur")

    # Test removeConstants
    f1, n1 = remove_constant_features(feats, names, threshold=1e-6)
    print(f"  After constant removal: {f1.shape} ({len(n1)} fitur)")
    assert f1.shape[1] < 20

    # Test remove correlated
    f2, n2 = remove_correlated_features(feats[:, 2:5], names[2:5], threshold=0.95)
    print(f"  Correlated [f2,f3,f4]: {names[2:5]} → {n2}")
    assert len(n2) <= 3

    # Test auto pipeline
    f_auto, n_auto = auto_select_features(
        feats, names,
        variance_threshold=1e-6,
        correlation_threshold=0.95,
        verbose=True,
    )
    print(f"  Auto selection: {feats.shape[1]} → {f_auto.shape[1]} fitur")

    print("\n  ✅ Feature selection OK")


def test_full_integration():
    print("\n" + "="*60)
    print(" TEST 8: Full Integration — All 5 Engines")
    print("="*60)

    # Simulasi data OHLCV, 1 saham, 200 hari
    T = 200
    t = torch.arange(T, dtype=torch.float32)
    close = 1000 + t * 0.5 + torch.randn(T) * 10
    volume = 1e6 + torch.randn(T) * 1e5

    signal = torch.stack([close, volume], dim=1)  # [200, 2]
    windows = create_rolling_windows(signal, window_size=20)
    print(f"  Windows: {windows.shape}")  # [181, 20, 2]

    close_w = windows[:, :, 0]  # [181, 20]
    vol_w = windows[:, :, 1]

    total_feats = []
    total_names = []

    # Engine A: Rocket
    t0 = time.time()
    rk_engine = RocketEngine(n_kernels=60, verbose=False)
    rk_f, rk_n = rk_engine.extract(close_w)
    total_feats.append(rk_f)
    total_names.extend(rk_n)
    dt_rk = time.time() - t0
    print(f"  Rocket:     {rk_f.shape[1]:>5} fitur ({dt_rk:.3f}s)")

    # Engine B: Catch22
    t0 = time.time()
    c22_engine = Catch22Engine(verbose=False)
    c22_f, c22_n = c22_engine.extract(close_w)
    total_feats.append(c22_f)
    total_names.extend(c22_n)
    dt_c22 = time.time() - t0
    print(f"  Catch22:    {c22_f.shape[1]:>5} fitur ({dt_c22:.3f}s)")

    # Engine C: Signatures (multivariate)
    t0 = time.time()
    sig_engine = SignaturesEngine(depth=3, channels=['close', 'vol'],
                                  augment_time=True, verbose=False)
    sig_f, sig_n = sig_engine.extract(windows)  # [181, 20, 2]
    total_feats.append(sig_f)
    total_names.extend(sig_n)
    dt_sig = time.time() - t0
    print(f"  Signatures: {sig_f.shape[1]:>5} fitur ({dt_sig:.3f}s)")

    # Engine D: Wavelets
    t0 = time.time()
    wl_engine = WaveletsEngine(wavelet_types=['haar', 'db4'], verbose=False)
    wl_f, wl_n = wl_engine.extract(close_w)
    total_feats.append(wl_f)
    total_names.extend(wl_n)
    dt_wl = time.time() - t0
    print(f"  Wavelets:   {wl_f.shape[1]:>5} fitur ({dt_wl:.3f}s)")

    # Engine E: Tsfresh (efficient untuk speed)
    t0 = time.time()
    ts_engine = TsfreshEngine(mode='efficient', verbose=False)
    ts_f, ts_n = ts_engine.extract(close_w)
    total_feats.append(ts_f)
    total_names.extend(ts_n)
    dt_ts = time.time() - t0
    print(f"  Tsfresh:    {ts_f.shape[1]:>5} fitur ({dt_ts:.3f}s)")

    # Gabung semua
    combined = torch.cat(total_feats, dim=1)
    print(f"\n  ─── Combined (before selection): {combined.shape}")
    print(f"  Total fitur: {len(total_names)}")

    # Feature selection
    selected, sel_names = auto_select_features(
        combined, total_names, verbose=True,
    )
    print(f"  After selection: {selected.shape}")
    print(f"  Final fitur: {len(sel_names)}")

    assert torch.isfinite(selected).all()
    dt_total = dt_rk + dt_c22 + dt_sig + dt_wl + dt_ts
    print(f"\n  Total waktu (semua engine): {dt_total:.2f}s")

    print("\n  ✅ Full integration OK")


if __name__ == '__main__':
    print("="*60)
    print("  TS-QUANT — Fase 4 Smoke Test")
    print("  (MultiRocket + Signatures + Feature Selection)")
    print("="*60)

    test_rocket_basic()
    test_rocket_large()
    test_rocket_edge_cases()
    test_signatures_basic()
    test_signatures_depths()
    test_signatures_2d_input()
    test_feature_selection()
    test_full_integration()

    print("\n" + "="*60)
    print("  ✅✅✅ SEMUA TEST FASE 4 LULUS ✅✅✅")
    print("="*60)
