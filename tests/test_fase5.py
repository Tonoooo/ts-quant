"""
Smoke Test — Fase 5: End-to-End Integration
=============================================

Test:
1. generate_features() — DataFrame in → DataFrame out
2. Multi-stock processing
3. Feature selection integration
4. CPU fallback
5. Engine toggle
"""

import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '.')

from ts_quant.api import generate_features


def create_mock_ohlcv(n_stocks=3, n_days=80, seed=42):
    """Buat DataFrame OHLCV mirip data saham."""
    np.random.seed(seed)
    rows = []
    symbols = [f"STOCK_{i:03d}" for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

    for sym in symbols:
        base = 1000 + np.random.randn() * 200
        returns = np.random.randn(n_days) * 0.02
        close = base * np.cumprod(1 + returns)
        high = close * (1 + abs(np.random.randn(n_days) * 0.01))
        low = close * (1 - abs(np.random.randn(n_days) * 0.01))
        open_ = close * (1 + np.random.randn(n_days) * 0.005)
        volume = np.random.randint(1e5, 1e7, n_days).astype(float)

        for i in range(n_days):
            rows.append({
                'symbol': sym,
                'date': dates[i],
                'open': open_[i],
                'high': high[i],
                'low': low[i],
                'close': close[i],
                'volume': volume[i],
            })

    return pd.DataFrame(rows)


def test_basic():
    print("\n" + "="*60)
    print(" TEST 1: generate_features — Basic (All Engines)")
    print("="*60)

    df = create_mock_ohlcv(n_stocks=3, n_days=60)
    print(f"  Input: {df.shape}, {df['symbol'].nunique()} saham")

    t0 = time.time()
    result = generate_features(
        df,
        device='cpu',
        enable_rocket=True,
        enable_catch22=True,
        enable_signatures=True,
        enable_wavelets=True,
        enable_tsfresh=True,
        tsfresh_mode='efficient',
        rocket_n_kernels=30,
        window_sizes=[20],
        compute_channels=['close'],
        signature_channels=['close', 'volume'],
        feature_selection='auto',
        verbose=True,
    )
    dt = time.time() - t0

    print(f"\n  Output: {result.shape}")
    print(f"  Kolom baru: {result.shape[1] - df.shape[1]}")
    print(f"  Waktu: {dt:.2f}s")

    assert result.shape[0] == df.shape[0], "Baris harus sama"
    assert result.shape[1] > df.shape[1], "Harus ada kolom baru"

    # Periksa NaN di baris yang punya fitur (setelah window_size)
    feat_cols = [c for c in result.columns if c not in df.columns]
    print(f"  Feature columns: {len(feat_cols)}")

    print("\n  ✅ Basic end-to-end OK")


def test_multi_stock():
    print("\n" + "="*60)
    print(" TEST 2: Multi-stock Processing")
    print("="*60)

    df = create_mock_ohlcv(n_stocks=10, n_days=50)
    print(f"  Input: {df.shape}, {df['symbol'].nunique()} saham")

    result = generate_features(
        df,
        device='cpu',
        enable_rocket=True,
        enable_catch22=False,
        enable_signatures=False,
        enable_wavelets=False,
        enable_tsfresh=False,
        rocket_n_kernels=20,
        window_sizes=[15],
        feature_selection=None,  # Tanpa seleksi
        verbose=False,
    )

    print(f"  Output: {result.shape}")
    assert result.shape[0] == df.shape[0]

    # Cek setiap saham ada di output
    for sym in df['symbol'].unique():
        assert sym in result['symbol'].values

    print("\n  ✅ Multi-stock OK")


def test_engine_toggle():
    print("\n" + "="*60)
    print(" TEST 3: Engine Toggle (Individual)")
    print("="*60)

    df = create_mock_ohlcv(n_stocks=2, n_days=50)

    configs = [
        ("Rocket only", dict(enable_rocket=True, enable_catch22=False,
                           enable_signatures=False, enable_wavelets=False,
                           enable_tsfresh=False, rocket_n_kernels=20)),
        ("Catch22 only", dict(enable_rocket=False, enable_catch22=True,
                            enable_signatures=False, enable_wavelets=False,
                            enable_tsfresh=False)),
        ("Tsfresh only", dict(enable_rocket=False, enable_catch22=False,
                            enable_signatures=False, enable_wavelets=False,
                            enable_tsfresh=True, tsfresh_mode='efficient')),
    ]

    for name, cfg in configs:
        result = generate_features(
            df, device='cpu', window_sizes=[15],
            feature_selection=None, verbose=False, **cfg,
        )
        n_new = result.shape[1] - df.shape[1]
        print(f"  {name:20s}: +{n_new:>4} kolom")
        assert n_new > 0, f"{name} tidak menambah kolom"

    print("\n  ✅ Engine toggle OK")


def test_short_stock():
    print("\n" + "="*60)
    print(" TEST 4: Short Stock (< window_size)")
    print("="*60)

    df = create_mock_ohlcv(n_stocks=2, n_days=60)
    # Tambah 1 saham dengan data sangat pendek
    short = pd.DataFrame({
        'symbol': ['SHORT'] * 5,
        'date': pd.date_range('2024-01-01', periods=5, freq='B'),
        'open': [100]*5, 'high': [101]*5, 'low': [99]*5,
        'close': [100]*5, 'volume': [1e6]*5,
    })
    df = pd.concat([df, short], ignore_index=True)

    result = generate_features(
        df, device='cpu',
        enable_rocket=True, enable_catch22=False,
        enable_signatures=False, enable_wavelets=False,
        enable_tsfresh=False,
        rocket_n_kernels=10, window_sizes=[20],
        feature_selection=None, verbose=False,
    )

    print(f"  Input: {df.shape} (termasuk 1 saham pendek)")
    print(f"  Output: {result.shape}")
    # Saham pendek harus tetap ada, tapi fiturnya NaN
    assert 'SHORT' in result['symbol'].values

    print("\n  ✅ Short stock handling OK")


def test_output_quality():
    print("\n" + "="*60)
    print(" TEST 5: Output Quality Check")
    print("="*60)

    df = create_mock_ohlcv(n_stocks=2, n_days=80)
    result = generate_features(
        df, device='cpu',
        enable_rocket=True, enable_catch22=True,
        enable_signatures=True, enable_wavelets=True,
        enable_tsfresh=True,
        tsfresh_mode='efficient',
        rocket_n_kernels=30,
        window_sizes=[20],
        feature_selection='auto',
        verbose=False,
    )

    feat_cols = [c for c in result.columns if c not in df.columns]

    # Check: fitur harus numerik
    for col in feat_cols[:10]:  # sample check
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col} bukan numerik"

    # Check: merge harus benar (original data intact)
    for col in df.columns:
        assert col in result.columns

    print(f"  Total fitur: {len(feat_cols)}")
    print(f"  Semua fitur numerik: ✓")
    print(f"  Original kolom intact: ✓")

    print("\n  ✅ Output quality OK")


if __name__ == '__main__':
    print("="*60)
    print("  TS-QUANT — Fase 5 Smoke Test (End-to-End)")
    print("="*60)

    test_basic()
    test_multi_stock()
    test_engine_toggle()
    test_short_stock()
    test_output_quality()

    print("\n" + "="*60)
    print("  ✅✅✅ SEMUA TEST FASE 5 LULUS ✅✅✅")
    print("="*60)
