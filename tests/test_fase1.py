"""
Smoke Test — Fase 1: Core Infrastructure
=========================================

Test ringan tanpa GPU untuk memvalidasi:
1. VRAMManager (CPU fallback)
2. df_to_tensor_3d / tensor_to_df
3. create_rolling_windows
4. Validasi input
"""

import sys
import numpy as np
import pandas as pd
import torch

# Path ke library
sys.path.insert(0, '.')

from ts_quant.core.memory_manager import VRAMManager
from ts_quant.core.tensor_utils import (
    df_to_tensor_3d,
    tensor_to_df,
    get_per_stock_tensors,
    estimate_tensor_bytes,
)
from ts_quant.core.windowing import (
    create_rolling_windows,
    create_rolling_windows_multi,
    create_rolling_windows_batch,
    unbatch_windows,
    compute_window_stats,
)
from ts_quant.utils.validation import validate_dataframe, ValidationError


def create_sample_data(n_stocks=5, n_days=100):
    """Buat data OHLCV sintetis untuk testing."""
    rows = []
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    for i in range(n_stocks):
        symbol = f'STOCK{i:03d}'
        base_price = 1000 + i * 500
        for j, date in enumerate(dates):
            noise = np.random.randn() * 20
            close = base_price + j * 2 + noise
            rows.append({
                'symbol': symbol,
                'date': date,
                'open': close - abs(noise) * 0.5,
                'high': close + abs(noise),
                'low': close - abs(noise),
                'close': close,
                'volume': int(1e6 + np.random.randint(0, 5e5)),
            })
    return pd.DataFrame(rows)


def test_vram_manager():
    """Test VRAMManager di CPU mode."""
    print("\n" + "="*60)
    print(" TEST 1: VRAMManager (CPU fallback)")
    print("="*60)

    manager = VRAMManager(device='cpu', verbose=True)
    assert not manager.is_gpu, "Seharusnya CPU mode"
    assert manager.get_available_bytes() == 0
    print(f"  repr: {manager}")

    # Test estimate batch size (CPU = semua sekaligus)
    bs = manager.estimate_batch_size(100, 1024)
    assert bs == 100, f"CPU batch size harus = n_items, got {bs}"

    # Test execute_chunked
    data = list(range(10))
    results = manager.execute_chunked(
        func=lambda batch: [x * 2 for x in batch],
        data_list=data,
        bytes_per_item=100,
        desc="Test chunked",
    )
    flat = [item for sublist in results for item in sublist]
    assert flat == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    print("  ✅ VRAMManager CPU mode OK")


def test_tensor_conversion():
    """Test DataFrame ↔ Tensor conversion."""
    print("\n" + "="*60)
    print(" TEST 2: DataFrame ↔ Tensor 3D Conversion")
    print("="*60)

    df = create_sample_data(n_stocks=3, n_days=50)
    print(f"  Input DataFrame: {df.shape}")

    # ── df_to_tensor_3d ──
    tensor, stocks, dates, masks = df_to_tensor_3d(
        df,
        value_cols=['close', 'volume'],
        device='cpu',
    )
    print(f"  Tensor shape: {tensor.shape}")  # [3, 50, 2]
    assert tensor.shape == (3, 50, 2), f"Shape salah: {tensor.shape}"
    assert len(stocks) == 3
    assert len(dates) == 50
    assert len(masks) == 3
    assert tensor.dtype == torch.float32

    # Cek tidak ada NaN (semua saham punya data lengkap)
    assert not torch.isnan(tensor).any(), "Tidak boleh ada NaN"

    # ── tensor_to_df ──
    df_back = tensor_to_df(
        tensor, stocks, dates,
        feature_names=['close', 'volume'],
        stock_masks=masks,
    )
    print(f"  Recovered DataFrame: {df_back.shape}")
    assert df_back.shape[0] == df.shape[0], "Jumlah baris harus sama"
    assert 'symbol' in df_back.columns
    assert 'date' in df_back.columns
    assert 'close' in df_back.columns

    # ── get_per_stock_tensors ──
    per_stock = get_per_stock_tensors(tensor, masks, stocks)
    assert len(per_stock) == 3
    for sym, t in per_stock:
        assert t.shape == (50, 2), f"{sym}: shape {t.shape}"

    # ── estimate_tensor_bytes ──
    est = estimate_tensor_bytes(558, 2000, 50)
    print(f"  Estimasi 558 saham × 2000 hari × 50 fitur: {est / 1024**2:.1f} MB")
    assert est > 0

    print("  ✅ Tensor conversion OK")


def test_tensor_variable_length():
    """Test conversion dengan saham yang panjangnya berbeda."""
    print("\n" + "="*60)
    print(" TEST 3: Variable Length Stocks")
    print("="*60)

    # Saham A: 100 hari, Saham B: 50 hari
    dates_a = pd.bdate_range('2024-01-01', periods=100)
    dates_b = pd.bdate_range('2024-03-01', periods=50)

    df = pd.DataFrame({
        'symbol': ['A'] * 100 + ['B'] * 50,
        'date': list(dates_a) + list(dates_b),
        'close': np.random.randn(150) * 100 + 1000,
        'volume': np.random.randint(1000, 10000, 150),
    })

    tensor, stocks, dates, masks = df_to_tensor_3d(
        df, value_cols=['close', 'volume'], device='cpu'
    )

    n_total_dates = len(set(list(dates_a) + list(dates_b)))
    print(f"  Total unique dates: {n_total_dates}")
    print(f"  Tensor shape: {tensor.shape}")  # [2, n_total_dates, 2]
    assert tensor.shape[0] == 2
    assert tensor.shape[2] == 2

    # Saham B harus punya NaN di tanggal yang tidak ada datanya
    b_idx = stocks.index('B')
    b_mask = masks['B']
    n_valid_b = b_mask.sum()
    n_nan_b = (~b_mask).sum()
    print(f"  Stock B: {n_valid_b} valid, {n_nan_b} padded")
    assert n_valid_b == 50

    print("  ✅ Variable length OK")


def test_rolling_windows():
    """Test sliding window creation."""
    print("\n" + "="*60)
    print(" TEST 4: Rolling Windows (GPU Sliding Window)")
    print("="*60)

    T, C = 200, 3
    x = torch.randn(T, C)

    # ── Single window ──
    w20 = create_rolling_windows(x, window_size=20)
    expected_w = T - 20 + 1  # 181
    print(f"  Input: [{T}, {C}], window=20 → output: {w20.shape}")
    assert w20.shape == (expected_w, 20, C), f"Shape salah: {w20.shape}"

    # Verifikasi isi: window pertama = x[0:20]
    assert torch.allclose(w20[0], x[0:20]), "Window pertama salah"
    # Window terakhir = x[180:200]
    assert torch.allclose(w20[-1], x[180:200]), "Window terakhir salah"

    # ── Multi window ──
    multi = create_rolling_windows_multi(x, [10, 20, 60])
    assert len(multi) == 3
    assert multi[10].shape == (191, 10, C)
    assert multi[20].shape == (181, 20, C)
    assert multi[60].shape == (141, 60, C)
    print(f"  Multi windows: {list(multi.keys())} → shapes OK")

    # ── Batch windows ──
    stocks = [torch.randn(200, C), torch.randn(150, C), torch.randn(30, C)]
    batch_w, boundaries = create_rolling_windows_batch(stocks, 20)
    print(f"  Batch: 3 stocks → {batch_w.shape}, boundaries={boundaries}")
    assert boundaries[0] == 181  # 200 - 20 + 1
    assert boundaries[1] == 131  # 150 - 20 + 1
    assert boundaries[2] == 11   # 30 - 20 + 1
    assert batch_w.shape[0] == sum(boundaries)

    # ── Unbatch ──
    fake_features = torch.randn(sum(boundaries), 10)
    per_stock = unbatch_windows(fake_features, boundaries)
    assert len(per_stock) == 3
    assert per_stock[0].shape == (181, 10)
    assert per_stock[1].shape == (131, 10)

    # ── Window stats ──
    stats = compute_window_stats(w20)
    expected_stats = C * 5  # mean, std, min, max, median per channel
    print(f"  Window stats: {stats.shape} (expected [{expected_w}, {expected_stats}])")
    assert stats.shape == (expected_w, expected_stats)

    print("  ✅ Rolling windows OK")


def test_validation():
    """Test input validation."""
    print("\n" + "="*60)
    print(" TEST 5: Input Validation")
    print("="*60)

    df_good = create_sample_data(3, 50)

    # Valid case
    result = validate_dataframe(df_good, ['symbol', 'date', 'close', 'volume'])
    assert result is not None
    print("  Valid DataFrame: OK")

    # Missing column
    try:
        validate_dataframe(df_good, ['symbol', 'date', 'nonexistent'])
        assert False, "Harusnya raise"
    except ValidationError as e:
        print(f"  Missing column: Caught ✓ ({e})")

    # Empty DataFrame
    try:
        validate_dataframe(pd.DataFrame(), ['symbol'])
        assert False, "Harusnya raise"
    except ValidationError:
        print("  Empty DataFrame: Caught ✓")

    print("  ✅ Validation OK")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print(" TEST 6: Edge Cases")
    print("="*60)

    # Window size > data length
    x = torch.randn(10, 2)
    try:
        create_rolling_windows(x, window_size=20)
        assert False, "Harusnya raise"
    except ValueError as e:
        print(f"  Window > data: Caught ✓")

    # Single stock
    df = create_sample_data(1, 30)
    tensor, stocks, dates, masks = df_to_tensor_3d(
        df, value_cols=['close'], device='cpu'
    )
    assert tensor.shape[0] == 1
    print(f"  Single stock tensor: {tensor.shape} ✓")

    # Very small window
    x = torch.randn(50, 1)
    w = create_rolling_windows(x, window_size=2)
    assert w.shape == (49, 2, 1)
    print(f"  Window size=2: {w.shape} ✓")

    print("  ✅ Edge cases OK")


if __name__ == '__main__':
    print("="*60)
    print("  TS-QUANT — Fase 1 Smoke Test (CPU Only)")
    print("="*60)

    test_vram_manager()
    test_tensor_conversion()
    test_tensor_variable_length()
    test_rolling_windows()
    test_validation()
    test_edge_cases()

    print("\n" + "="*60)
    print("  ✅✅✅ SEMUA TEST FASE 1 LULUS ✅✅✅")
    print("="*60)
