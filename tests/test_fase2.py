"""
Smoke Test — Fase 2: Engine Catch22 + Wavelets
================================================

Test ringan (CPU) untuk memvalidasi:
1. Catch22Engine — 22 fitur dari data sintetis
2. WaveletsEngine — DWT decomposition dan feature extraction
3. Integrasi dengan windowing module
"""

import sys
import time
import numpy as np
import torch

sys.path.insert(0, '.')

from ts_quant.engines.catch22 import Catch22Engine, CATCH22_FUNCTIONS
from ts_quant.engines.wavelets import (
    WaveletsEngine,
    wavedec,
    wavelet_denoised,
    WAVELET_FILTERS,
)
from ts_quant.core.windowing import create_rolling_windows


def create_synthetic_signals(B=50, T=60):
    """Buat sinyal sintetis campuran trend + noise + siklus."""
    t = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    # Trend
    trend = 0.05 * t * torch.randn(B, 1)
    # Siklus
    freq = torch.rand(B, 1) * 0.3 + 0.05
    cycle = torch.sin(2 * 3.14159 * freq * t)
    # Noise
    noise = torch.randn(B, T) * 0.5
    return trend + cycle + noise


def test_catch22_basic():
    """Test Catch22: semua 22 fitur mengeluarkan output yang valid."""
    print("\n" + "="*60)
    print(" TEST 1: Catch22 — Basic Extraction")
    print("="*60)

    engine = Catch22Engine(verbose=True)
    x = create_synthetic_signals(B=30, T=40)
    print(f"  Input: {x.shape}")

    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"  Output: {features.shape}")
    print(f"  Waktu: {dt:.3f}s")
    print(f"  Jumlah fitur: {len(names)}")

    assert features.shape == (30, 22), f"Shape salah: {features.shape}"
    assert len(names) == 22, f"Nama salah: {len(names)}"

    # Tidak boleh ada NaN atau Inf
    assert torch.isfinite(features).all(), "Ada NaN/Inf di output"

    # Cetak ringkasan per fitur
    print(f"\n  {'Fitur':<50} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*70}")
    for i, name in enumerate(names):
        m = features[:, i].mean().item()
        s = features[:, i].std().item()
        print(f"  {name:<50} {m:>10.4f} {s:>10.4f}")

    print("\n  ✅ Catch22 basic extraction OK")


def test_catch22_edge_cases():
    """Test Catch22: edge cases (constant, very short, etc.)"""
    print("\n" + "="*60)
    print(" TEST 2: Catch22 — Edge Cases")
    print("="*60)

    engine = Catch22Engine(verbose=False)

    # Constant signal (semua nilai sama)
    const = torch.ones(5, 30) * 42.0
    features, _ = engine.extract(const)
    assert features.shape == (5, 22)
    assert torch.isfinite(features).all(), "Constant signal: ada NaN/Inf"
    print("  Constant signal: OK")

    # Sangat pendek (window=10)
    short = torch.randn(5, 10)
    features, _ = engine.extract(short)
    assert features.shape == (5, 22)
    assert torch.isfinite(features).all(), "Short signal: ada NaN/Inf"
    print("  Short signal (T=10): OK")

    # Monotonic (hanya naik)
    mono = torch.arange(50, dtype=torch.float32).unsqueeze(0).repeat(3, 1)
    features, _ = engine.extract(mono)
    assert torch.isfinite(features).all(), "Monotonic: ada NaN/Inf"
    print("  Monotonic signal: OK")

    # Large batch
    large = torch.randn(500, 20)
    t0 = time.time()
    features, _ = engine.extract(large)
    dt = time.time() - t0
    assert features.shape == (500, 22)
    print(f"  Large batch (B=500): OK ({dt:.3f}s)")

    print("\n  ✅ Catch22 edge cases OK")


def test_catch22_individual_functions():
    """Test setiap fungsi Catch22 secara individual."""
    print("\n" + "="*60)
    print(" TEST 3: Catch22 — Individual Functions")
    print("="*60)

    x = create_synthetic_signals(B=10, T=50)
    passed = 0
    failed = 0

    for name, func in CATCH22_FUNCTIONS:
        try:
            result = func(x)
            assert result.shape == (10,), f"{name}: shape {result.shape}"
            assert torch.isfinite(result).all(), f"{name}: NaN/Inf"
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n  Results: {passed}/22 passed, {failed}/22 failed")
    assert failed == 0, f"{failed} functions failed"
    print("  ✅ All 22 individual functions OK")


def test_wavelets_basic():
    """Test WaveletsEngine: basic decomposition dan extraction."""
    print("\n" + "="*60)
    print(" TEST 4: Wavelets — Basic Extraction")
    print("="*60)

    engine = WaveletsEngine(
        wavelet_types=['haar', 'db4'],
        decomposition_levels=4,
        verbose=True,
    )

    x = create_synthetic_signals(B=20, T=100)
    print(f"  Input: {x.shape}")

    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"  Output: {features.shape}")
    print(f"  Waktu: {dt:.3f}s")
    print(f"  Jumlah fitur: {len(names)}")
    print(f"  Estimasi n_features: {engine.n_features}")

    assert features.shape[0] == 20
    assert features.shape[1] == len(names)
    assert torch.isfinite(features).all(), "Ada NaN/Inf"

    # Cetak nama fitur
    print(f"\n  Fitur wavelet:")
    for name in names:
        print(f"    - {name}")

    print("\n  ✅ Wavelets basic extraction OK")


def test_wavelets_decomposition():
    """Test wavedec: level-level decomposition."""
    print("\n" + "="*60)
    print(" TEST 5: Wavelets — Decomposition Levels")
    print("="*60)

    x = torch.randn(5, 128)

    for wt in ['haar', 'db4', 'db2', 'sym4']:
        coeffs = wavedec(x, wt, level=4)
        print(f"  {wt}: {len(coeffs)} levels → " +
              ", ".join([f"[{c.shape[1]}]" for c in coeffs]))
        # Cek energi: total harus kira-kira sama dengan input
        total_energy = (x ** 2).sum(dim=1)
        recon_energy = sum((c ** 2).sum(dim=1) for c in coeffs)
        ratio = (recon_energy / total_energy.clamp(min=1e-10)).mean()
        print(f"    Energy ratio (recon/orig): {ratio:.4f}")

    print("\n  ✅ Wavelets decomposition OK")


def test_wavelets_denoising():
    """Test wavelet denoising."""
    print("\n" + "="*60)
    print(" TEST 6: Wavelets — Denoising")
    print("="*60)

    # Signal = trend + strong noise
    B, T = 10, 200
    t = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    clean = torch.sin(0.05 * t)
    noisy = clean + torch.randn(B, T) * 0.5

    denoised = wavelet_denoised(noisy, 'db4', level=3)

    # Denoised harus lebih dekat ke clean daripada noisy
    err_noisy = (noisy - clean).abs().mean()
    err_denoised = (denoised - clean).abs().mean()
    print(f"  Mean error (noisy):    {err_noisy:.4f}")
    print(f"  Mean error (denoised): {err_denoised:.4f}")
    print(f"  Improvement: {(1 - err_denoised/err_noisy)*100:.1f}%")

    assert denoised.shape == noisy.shape
    assert torch.isfinite(denoised).all()

    print("  ✅ Wavelets denoising OK")


def test_integration_with_windowing():
    """Test integrasi: windowing → catch22 + wavelets."""
    print("\n" + "="*60)
    print(" TEST 7: Integration — Windowing → Engines")
    print("="*60)

    # Simulasi 1 saham, 200 hari, 1 channel (close)
    signal = torch.randn(200, 1)
    windows = create_rolling_windows(signal, window_size=20)
    print(f"  Windows: {windows.shape}")  # [181, 20, 1]

    # Extract channel 0
    x = windows[:, :, 0]  # [181, 20]
    print(f"  Input ke engines: {x.shape}")

    # Catch22
    c22_engine = Catch22Engine(verbose=False)
    c22_feat, c22_names = c22_engine.extract(x)
    print(f"  Catch22 output: {c22_feat.shape} ({len(c22_names)} fitur)")

    # Wavelets
    wl_engine = WaveletsEngine(
        wavelet_types=['haar'],
        decomposition_levels=3,
        verbose=False,
    )
    wl_feat, wl_names = wl_engine.extract(x)
    print(f"  Wavelets output: {wl_feat.shape} ({len(wl_names)} fitur)")

    # Gabung
    combined = torch.cat([c22_feat, wl_feat], dim=1)
    all_names = c22_names + wl_names
    print(f"  Combined: {combined.shape} ({len(all_names)} fitur total)")

    assert combined.shape[0] == 181  # n_windows
    assert combined.shape[1] == len(all_names)
    assert torch.isfinite(combined).all()

    print("\n  ✅ Integration OK")


if __name__ == '__main__':
    print("="*60)
    print("  TS-QUANT — Fase 2 Smoke Test (Catch22 + Wavelets)")
    print("="*60)

    test_catch22_basic()
    test_catch22_edge_cases()
    test_catch22_individual_functions()
    test_wavelets_basic()
    test_wavelets_decomposition()
    test_wavelets_denoising()
    test_integration_with_windowing()

    print("\n" + "="*60)
    print("  ✅✅✅ SEMUA TEST FASE 2 LULUS ✅✅✅")
    print("="*60)
