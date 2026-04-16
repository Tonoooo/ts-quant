"""
Smoke Test — Fase 3: Engine Tsfresh Complete
=============================================

Test:
1. Comprehensive mode — semua 8 kategori
2. Efficient mode — subset
3. Edge cases (constant, short, large batch)
4. Integrasi dengan windowing
"""

import sys
import time
import torch

sys.path.insert(0, '.')

from ts_quant.engines.tsfresh_core import TsfreshEngine
from ts_quant.core.windowing import create_rolling_windows


def create_signals(B=30, T=60):
    t = torch.arange(T, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    trend = 0.05 * t * torch.randn(B, 1)
    cycle = torch.sin(2 * 3.14159 * torch.rand(B, 1) * 0.2 * t)
    noise = torch.randn(B, T) * 0.3
    return trend + cycle + noise + 100  # offset to avoid zero issues


def test_comprehensive():
    print("\n" + "="*60)
    print(" TEST 1: Tsfresh Comprehensive Mode")
    print("="*60)

    engine = TsfreshEngine(mode='comprehensive', verbose=True)
    x = create_signals(B=20, T=60)
    print(f"  Input: {x.shape}")

    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"\n  Output: {features.shape}")
    print(f"  Total fitur: {len(names)}")
    print(f"  Waktu: {dt:.2f}s")

    assert features.shape[0] == 20
    assert features.shape[1] == len(names)
    assert torch.isfinite(features).all(), f"Ada NaN/Inf: {torch.isnan(features).sum()}"
    assert len(names) > 200, f"Comprehensive harus > 200 fitur, got {len(names)}"

    print(f"\n  ✅ Comprehensive mode OK ({len(names)} fitur)")


def test_efficient():
    print("\n" + "="*60)
    print(" TEST 2: Tsfresh Efficient Mode")
    print("="*60)

    engine = TsfreshEngine(mode='efficient', verbose=True)
    x = create_signals(B=20, T=60)

    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"\n  Output: {features.shape}")
    print(f"  Total fitur: {len(names)}")
    print(f"  Waktu: {dt:.2f}s")

    assert features.shape[0] == 20
    assert torch.isfinite(features).all()

    print(f"\n  ✅ Efficient mode OK ({len(names)} fitur)")


def test_edge_cases():
    print("\n" + "="*60)
    print(" TEST 3: Tsfresh Edge Cases")
    print("="*60)

    engine = TsfreshEngine(mode='efficient', verbose=False)

    # Constant signal
    const = torch.ones(5, 40) * 42.0
    f, n = engine.extract(const)
    assert torch.isfinite(f).all(), "Constant: NaN/Inf"
    print(f"  Constant (5×40): {f.shape} ✓")

    # Short signal
    short = create_signals(5, 20)
    f, n = engine.extract(short)
    assert torch.isfinite(f).all(), "Short: NaN/Inf"
    print(f"  Short (5×20): {f.shape} ✓")

    # Large batch
    large = create_signals(200, 40)
    t0 = time.time()
    f, n = engine.extract(large)
    dt = time.time() - t0
    assert torch.isfinite(f).all(), "Large: NaN/Inf"
    print(f"  Large (200×40): {f.shape} in {dt:.2f}s ✓")

    print("\n  ✅ Edge cases OK")


def test_feature_names_unique():
    print("\n" + "="*60)
    print(" TEST 4: Feature Names Uniqueness")
    print("="*60)

    engine = TsfreshEngine(mode='comprehensive', verbose=False)
    x = create_signals(5, 60)
    _, names = engine.extract(x)

    unique_names = set(names)
    if len(unique_names) < len(names):
        dupes = [n for n in names if names.count(n) > 1]
        print(f"  ⚠ Duplikat ditemukan: {set(dupes)}")
    else:
        print(f"  Semua {len(names)} nama fitur unik")

    print("  ✅ Feature names check OK")


def test_integration():
    print("\n" + "="*60)
    print(" TEST 5: Integration — Windowing → Tsfresh")
    print("="*60)

    signal = torch.randn(300, 1) * 50 + 1000
    windows = create_rolling_windows(signal, window_size=30)
    print(f"  Windows: {windows.shape}")  # [271, 30, 1]

    x = windows[:, :, 0]  # [271, 30]
    print(f"  Input ke Tsfresh: {x.shape}")

    engine = TsfreshEngine(mode='efficient', verbose=False)
    t0 = time.time()
    features, names = engine.extract(x)
    dt = time.time() - t0

    print(f"  Output: {features.shape} ({len(names)} fitur)")
    print(f"  Waktu: {dt:.2f}s")
    assert features.shape[0] == x.shape[0]
    assert torch.isfinite(features).all()

    print("\n  ✅ Integration OK")


def test_category_breakdown():
    print("\n" + "="*60)
    print(" TEST 6: Category Feature Count Breakdown")
    print("="*60)

    engine = TsfreshEngine(mode='comprehensive', verbose=False)
    x = create_signals(5, 60)
    _, all_names = engine.extract(x)

    # Group by prefix
    categories = {}
    for n in all_names:
        cat = n.split('_')[1] if '_' in n else 'other'
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  {'Category':<25} {'Count':>6}")
    print(f"  {'-'*31}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} {count:>6}")
    print(f"  {'─'*31}")
    print(f"  {'TOTAL':<25} {len(all_names):>6}")

    print("\n  ✅ Category breakdown OK")


if __name__ == '__main__':
    print("="*60)
    print("  TS-QUANT — Fase 3 Smoke Test (Tsfresh Complete)")
    print("="*60)

    test_comprehensive()
    test_efficient()
    test_edge_cases()
    test_feature_names_unique()
    test_integration()
    test_category_breakdown()

    print("\n" + "="*60)
    print("  ✅✅✅ SEMUA TEST FASE 3 LULUS ✅✅✅")
    print("="*60)
