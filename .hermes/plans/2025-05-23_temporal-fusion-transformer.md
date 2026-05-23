# Temporal Fusion Transformer — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill. Fresh subagent per bite-sized
> task, two-stage review (spec compliance → code quality), TDD enforcement.

**Goal:** Build a production-ready Pure-PyTorch TFT that ingests arbitrary tabular time-series
features (static + known/observed dynamic) and produces multi-horizon probabilistic forecasts
with full interpretability.

**Architecture:** Follows Lim et al. 2021 "Temporal Fusion Transformers for Interpretable
Multi-Horizon Time Series Forecasting". Eight core modules implemented across bite-sized
tasks with strict TDD at every step.

**Tech Stack:** Python ≥ 3.10, PyTorch 2.10 + CUDA, pytest, numpy/scipy/pandas — no
pytorch-lightning dependency. Pure-PyTorch, tested against the paper equations.

---

## Workflow for Every Task

```
1. implementer subagent   → write code (with TDD)
2. spec-compliance review → original spec PASS / list gaps
3. code-quality review    → APPROVED / REQUEST_CHANGES
4. Only proceed when BOTH reviews PASS → mark task complete → git commit
```

---

## Phase 0 — Bootstrap

### Task 0: Install missing dependencies & verify CUDA

**Objective:** Install pytorch-lightning (used by training harness) and verify toolkit

**Workdir:** `/home/mctouch/code/tft-timeseries`

**Step 1:** Run `pip install pytorch-lightning metrics`

Run: `pip install pytorch-lightning torchmetrics --quiet`
Expected: exits 0

**Step 2:** Run verification script

```python
import torch; print("CUDA:", torch.cuda.is_available())
import pytorch_lightning; print("PL:", pytorch_lightning.__version__)
import torchmetrics; print("metrics:", torchmetrics.__version__)
```

Expected: all present

**Step 3:** Commit

```bash
git add -A && git commit -m "chore: install deps, verify CUDA"
```

---

## Phase 1 — Core Infrastructure (file: `tft_timeseries/data.py`)

### Task 1a: `TimeSeriesDataset` — dataset wrapper

**Objective:** `TimeSeriesDataset` that holds static, past, known_future, target arrays and
returns one windowed sample per index.

**Files:**
- Create: `tft_timeseries/data.py`
- Test: `tests/test_data.py`

**Step 1 — Write failing test:**

```python
import numpy as np
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from tft_timeseries.data import TimeSeriesDataset

def test_dataset_returns_correct_window_shapes():
    N, T, H = 4, 6, 2
    static = np.zeros((N, 2))
    past   = np.zeros((N, T, 3))
    known  = np.zeros((N, T+H, 1))
    target = np.arange(N*T).reshape(N, T, 1).astype(float)
    ds = TimeSeriesDataset(static, past, known, target, past_len=T, future_len=H)
    s, p, k, t = ds[1]
    assert s.shape == (2,)
    assert p.shape == (T, 3)
    assert k.shape == (T+H, 1)
    assert t.shape == (H, 1)

def test_dataset_length():
    ds = TimeSeriesDataset(np.zeros((3,1)), np.zeros((3,4,2)), np.zeros((3,6,1)), np.zeros((3,4,1)), past_len=4, future_len=2)
    assert len(ds) == 3
```

Run: `pytest tests/test_data.py::test_dataset_returns_correct_window_shapes tests/test_data.py::test_dataset_length -v`
Expected: **FAIL** — `ModuleNotFoundError: No module named 'tft_timeseries'`

**Step 2 — Minimal implementation:**

```python
import numpy as np

class TimeSeriesDataset:
    def __init__(self, static: np.ndarray, past: np.ndarray,
                 known: np.ndarray, target: np.ndarray,
                 past_len: int, future_len: int):
        assert static.shape[0] == past.shape[0] == known.shape[0] == target.shape[0]
        self.static  = static
        self.past    = past
        self.known   = known
        self.target  = target
        self.past_len    = past_len
        self.future_len  = future_len

    def __len__(self):
        return len(self.static)

    def __getitem__(self, i):
        return (self.static[i],
                self.past[i],
                self.known[i],
                self.target[i])
```

**Step 3:** `pytest tests/test_data.py::test_dataset_returns_correct_window_shapes tests/test_data.py::test_dataset_length -v`
Expected: **2 passed**

**Step 4:** Refactor — add input validation & docstrings

**Step 5:** `pytest tests/test_data.py -v` then `pytest tests/ -q`

**Step 6:** `git add -A && git commit -m "feat: TimeSeriesDataset with window indexing"`

---

### Task 1b: `scale_data` — normalization helpers

**Objective:** Per-feature standardisation and inverse-transform utilities.

**Files:** Extend `tft_timeseries/data.py`

**Step 1 — Test (`test_data.py`):**

```python
def test_scale_standardizes_features():
    from tft_timeseries.data import scale_data, inverse_scale
    arr = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    s, m = scale_data(arr)
    assert np.allclose(s.mean(axis=0), [0.0, 0.0], atol=1e-10)
    assert np.allclose(s.std(axis=0),  [1.0, 1.0], atol=1e-10)
    inv = inverse_scale(s, m)
    assert np.allclose(inv, arr, atol=1e-10)
```

Run: `pytest tests/test_data.py::test_scale_standardizes_features -v`
Expected: FAIL

Continue with GREEN → REFACTOR pattern as in Task 1a.

---

### Task 1c: `create_windowed_samples` — train/test split helper

**Objective:** Convert flat feature matrices → windowed arrays for TFT (split at timestamp `t`).

**Files:** Extend `tft_timeseries/data.py`

**Step 1 — Test:**

```python
def test_create_windowed_samples_shapes():
    from tft_timeseries.data import create_windowed_samples
    # 10 time steps × 3 series
    data = np.arange(30, dtype=float).reshape(10, 3)
    past_len, future_len = 4, 2
    X_past, X_future, y = create_windowed_samples(data, past_len, future_len)
    # 10 - 4 - 2 + 1 = 5 windows
    assert len(X_past) == 5
    assert X_past.shape == (5, past_len, 3)
    assert X_future.shape == (5, future_len, 3)
    assert y.shape == (5, future_len, 3)
```

Green → Refactor → Commit.

---

## Phase 2 — TFT Model Architecture (file: `tft_timeseries/model.py`)

> All model components are pure `torch.nn.Module`, pure-PyTorch, CUDA-ready.
> Every module referenced by page/figure of the TFT paper.

### Task 2a: `VariableSelectionNetwork`

**Objective:** Per-variable GRN gate + per-variable GRN feature extractor; output weighted
feature vector.

**Paper §3.1, Eq. 1–3**

```python
class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, dropout=0.1):
        ...
    def forward(self, embedded_xi, context=None):
        # xi: (B, S, D_in)
        # context: (B, D_ctx)  used to weight per-variable importance
        # Returns: (B, D_out)
```

**Test:** forward pass shape matches expected; gradients flow; no NaNs.

---

### Task 2b: `GatedResidualNetwork` (GRN)

**Objective:** Gated residual with optional context infusion — the building block for VSN
and static covariate encoders.

**Paper §3, Eq. 4**

```python
class GatedResidualNetwork(nn.Module):
    def __init__(self, d, d_ctx=None, d_hidden=None, dropout=0.1):
        ...
    def forward(self, xi, context=None):
        # xi: (B, D)
        # context: (B, D_ctx) optional context projection
        # Returns: (B, D)
```

**Test:** pass gate goes to 1 (linear layer ones), skip gate to 0 → output ≈ xi. Gradients ≠ 0.

---

### Task 2c: `StaticCovariateEncoder`

**Objective:** Encodes all static features: v → {Vs, Ve, Vc} triple (selection, enrichment, context).

**Paper §3.1, Fig. 2 – static path**

```python
class StaticCovariateEncoder(nn.Module):
    def __init__(self, num_static, hidden_size, d_hidden=None, dropout=0.1):
        ...
    def forward(self, xs):
        # xs: (B, num_static) → return dict with keys:
        #     'vs', 've', 'vc'  each (B, hidden_size)
```

**Test:** forward returns 3-tuple of correct shapes; run 10K steps, assert no NaN/Inf.

---

### Task 2d: `TemporalFusionDecoder` (LSTM + Attention fusion)

**Objective:** LSTM encoder + LSTM decoder + quantile attention layer; combines
known_future + decoder output; produces per-quantile p(t).

**Paper §3.3–3.4**

```python
class TemporalFusionDecoder(nn.Module):
    def __init__(self, d_model, num_quantiles, dropout=0.1):
        ...
    def forward(self, past_encoded, future_known, static_ctx):
        # past_encoded:      (B, T_enc, D_model)   from LSTM encoder + VSN
        # future_known:      (B, T_dec, D_known)
        # static_ctx['vc']:  (B, D_model)
        # Returns:            (B, T_dec, num_quantiles)
```

**Test:** forward on random data; output shape correct; quantile 0.5 median near mean.

---

### Task 2e: `TFTModel` — assemble everything

**Objective:** Full end-to-end TFT: input projection → VSN on past + future →
static encoder → LSTM enc/dec → attention fusion → quantile head.

**Paper Fig. 2**

```python
class TFTModel(nn.Module):
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        self.static_encoder   = StaticCovariateEncoder(...)
        self.past_vsn         = VariableSelectionNetwork(...)
        self.future_vsn       = VariableSelectionNetwork(...)
        self.lstm_encoder     = nn.LSTM(...)
        self.lstm_decoder     = nn.LSTM(...)
        self.temporal_fusion  = TemporalFusionDecoder(...)

    @staticmethod
    def from_config(path: str | dict) -> 'TFTModel':  ...
    def forward(self, static_features, past_features, known_future):
        """(B, S_stat) → dict with 'quantiles' (B, H, Q), 'attn_weights' (B, H, T_enc, T_enc)"""
        ...
    def predict(self, ...) -> dict: ...
```

**Step 1 — Test (4 tests):**

```python
@pytest.fixture
def tft_config():
    return TFTConfig(
        num_static=2, num_past=3, num_future=1,
        d_model=32, d_hidden=32, past_len=12, future_len=6,
        num_quantiles=3, quantiles=[0.1, 0.5, 0.9],
        dropout=0.0,
    )

def test_tft_forward_pass_shape(tft_config):
    model = TFTModel(tft_config)
    B = 4
    xs   = torch.randn(B, tft_config.num_static)
    xp   = torch.randn(B, tft_config.past_len, tft_config.num_past)
    xf   = torch.randn(B, tft_config.future_len, tft_config.num_future)
    out  = model(xs, xp, xf)
    assert out['quantiles'].shape == (B, tft_config.future_len, tft_config.num_quantiles)

def test_tft_gradients_nonzero(tft_config):
    model = TFTModel(tft_config)
    ...
    loss.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.allclose(p.grad, torch.zeros_like(p.grad))

def test_tft_no_nan_output(tft_config):
    ...

def test_tft_config_from_yaml(tmp_path):
    cfg_path = str(tmp_path / "cfg.yaml")
    # write config yaml
    model = TFTModel.from_config(cfg_path)
    assert model.config.d_model == 32
```

Continue with full TDD cycle.

---

### Task 2f: `to_onnx` or `QuantizedTFTModel` for inference

**Objective:** Export path: ONNX or dynamic-quantized `torch.jit` for CPU/GPU inference.

**Files:** `tft_timeseries/export.py`

**Test:** export → reload → same output within 1 % relative tolerance.

---

## Phase 3 — Training Harness

### Task 3a: `QuantileLoss`

**Files:** `tft_timeseries/losses.py`

```python
def quantile_loss(y_hat: Tensor, y: Tensor, quantiles: list[float]) -> Tensor:
    """
    y_hat:  (B, H, Q)   predicted quantile levels
    y:      (B, H)       actual values
    Returns: scalar mean pinball loss over all quantile/horizon dims.
    """
```

**Test:** symmetric quantiles [0.5, 0.9, 0.1] → median pinball = 0 exactly for perfect prediction
(y_hat = y med).

---

### Task 3b: `TFTDataModule`

**Files:** `tft_timeseries/data.py` (add to existing module)

```python
class TFTDataModule:
    """Wraps TimeSeriesDataset for training: scaling, batching, DataLoader."""
    def __init__(self, train, val, test, batch_size=64):
        ...
```

---

### Task 3c: `TTrainer`

**Files:** `tft_timeseries/train.py`

```python
class TTrainer:
    def __init__(self, config, model, train_dm, val_dm, ...):
        ...
    def train(self) -> dict: ...
    def evaluate(self) -> dict: ...
```

---

## Phase 4 — Inference & Examples

### Task 4a: `TFTInferenceEngine`

**Files:** `tft_timeseries/inference.py`

```python
class TFTInferenceEngine:
    def __init__(self, checkpoint_path: str, device='auto'): ...
    def predict_batch(self, xs, xp, xf) -> pd.DataFrame: ...
    def predict_series(self, x_past_only, x_known_future) -> dict: ...
```

---

### Task 4b: Example scripts

- `scripts/train_example.py` — train on synthetic data
- `scripts/evaluate_example.py` — load + predict + plot

---

## File Map

```
tft_timeseries/
    __init__.py
    config.py      — TFTConfig dataclass  (num_static, num_past, ...)
    data.py        — TimeSeriesDataset, TFTDataModule, scale helpers
    model.py       — VSN, GRN, StaticCovariateEncoder, LSTMEncoder,
                     TemporalFusionDecoder, TFTModel
    losses.py      — quantile_loss
    inference.py   — TFTInferenceEngine
    export.py      — JIT/ONNX export
tests/
    test_config.py
    test_data.py
    test_model.py
    test_losses.py
    conftest.py
configs/
    tft_default.yaml
scripts/
    train_example.py
    evaluate_example.py
```

---

## Acceptance Criteria

After all tasks complete:
```bash
cd /home/mctouch/code/tft-timeseries
pytest tests/ -q --cov=tft_timeseries --cov-report=term-missing
```

- Coverage ≥ 85 %
- All tests green, no warnings
- `python scripts/train_example.py` finishes 10 steps with `loss < 0.1`
- `python scripts/evaluate_example.py` produces CSV/plot with quantile bands
- `python -m tft_timeseries` → prints version / help

---

## Parallel Task Matrix (subagent-swarm)

| Phase | Tasks               | Can run in parallel? |
|-------|---------------------|----------------------|
| 0     | Bootstrap           | —                    |
| 1     | 1a, 1b, 1c          | Yes                  |
| 2     | 2a, 2b, 2c          | Yes (all in model.py)|
| 2     | 2d                  | after 2a-2c          |
| 2     | 2e                  | after 2a-2d          |
| 2     | 2f                  | after 2e             |
| 3     | 3a, 3b, 3c          | Yes                  |
| 4     | 4a, 4b              | Yes                  |
