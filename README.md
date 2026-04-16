# ts-quant 🚀

**GPU-Accelerated Quantitative Feature Extraction for Financial Time Series**

[![PyPI](https://img.shields.io/pypi/v/ts-quant)](https://pypi.org/project/ts-quant/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Extract **2000+ quantitative features** from OHLCV stock data using 5 powerful engines — all running on GPU via PyTorch.

## 🎯 Why ts-quant?

| Feature | tsfresh (CPU) | ts-quant (GPU) |
|---|---|---|
| Speed | ~60s per stock | ~0.3s per stock |
| Memory | OOM on large data | VRAM-managed, crash-proof |
| Features | 794 (statistical only) | 2000+ (statistical + wavelets + rocket + signatures) |
| GPU | ❌ | ✅ PyTorch/CUDA |

## 📦 Installation

```bash
pip install ts-quant
```

**Requirements:** Python 3.9+, PyTorch 2.0+ (with CUDA for GPU acceleration)

## 🚀 Quick Start

```python
import pandas as pd
from ts_quant import generate_features

# Load your OHLCV data
df = pd.read_csv('stocks.csv')
# Expected columns: symbol, date, close, volume, open, high, low

# Extract features (GPU)
df_features = generate_features(
    df,
    device='cuda',              # 'cpu' for CPU-only
    tsfresh_mode='comprehensive',
    window_sizes=[20],
)

print(f"New features: {df_features.shape[1] - df.shape[1]}")
```

## 🔧 5 Feature Engines

### Engine A: MultiRocket 🎯
Random convolutional kernels with multi-scale dilations.
```python
from ts_quant import RocketEngine

engine = RocketEngine(n_kernels=250)
features, names = engine.extract(x)  # x: [B, T]
# → 1000 features (250 kernels × 4 pooling ops)
```

### Engine B: Catch22 📊
22 canonical time-series features (ACF, DFA, entropy, etc.).
```python
from ts_quant import Catch22Engine

engine = Catch22Engine()
features, names = engine.extract(x)
# → 22 features per window
```

### Engine C: Path Signatures ✍️
Ordered interaction features via iterated integrals.
```python
from ts_quant import SignaturesEngine

engine = SignaturesEngine(depth=3, channels=['close', 'volume'])
features, names = engine.extract(x_multivariate)  # [B, T, d]
# → d + d² + d³ features
```

### Engine D: Wavelets 🌊
Discrete wavelet transform (Haar, db4, db2, sym4).
```python
from ts_quant import WaveletsEngine

engine = WaveletsEngine(wavelet_types=['haar', 'db4'])
features, names = engine.extract(x)
# → 62 features (energy, entropy, mean, std, max per level)
```

### Engine E: Tsfresh Complete 📈
63 mathematical functions, 361 features in comprehensive mode.
```python
from ts_quant import TsfreshEngine

engine = TsfreshEngine(mode='comprehensive')
features, names = engine.extract(x)
# → 361 features (statistics, ACF, FFT, entropy, trend, ...)
```

## ⚡ VRAM Management

ts-quant automatically manages GPU memory to prevent OOM crashes:

```python
# Works on any GPU size (8GB - 80GB)
df_features = generate_features(
    df,
    device='cuda',
    max_vram_gb=8,  # Optional: manual VRAM limit
)
```

## 🔬 Feature Selection

Built-in redundancy removal:

```python
from ts_quant import auto_select_features

selected, names = auto_select_features(
    features_tensor,
    feature_names,
    correlation_threshold=0.95,  # Remove highly correlated
)
```

## 📁 Input Format

Your DataFrame should have this structure:

| symbol | date | open | high | low | close | volume |
|--------|------|------|------|-----|-------|--------|
| BBCA | 2024-01-02 | 9500 | 9600 | 9400 | 9550 | 1000000 |
| BBRI | 2024-01-02 | 5200 | 5300 | 5100 | 5250 | 2000000 |

## 🏗️ Architecture

```
ts-quant/
├── ts_quant/
│   ├── api.py              # Orchestrator — generate_features()
│   ├── core/
│   │   ├── memory_manager.py   # Dynamic VRAM management
│   │   ├── tensor_utils.py     # DataFrame ↔ Tensor conversion
│   │   └── windowing.py        # GPU sliding windows
│   ├── engines/
│   │   ├── rocket.py           # Engine A: MultiRocket
│   │   ├── catch22.py          # Engine B: Catch22
│   │   ├── signatures.py       # Engine C: Path Signatures
│   │   ├── wavelets.py         # Engine D: Wavelets
│   │   └── tsfresh_core.py     # Engine E: Tsfresh Complete
│   └── utils/
│       ├── config.py           # Default configurations
│       ├── validation.py       # Input validation
│       └── feature_selection.py # Redundancy removal
└── tests/
```

## 📄 License

MIT License
