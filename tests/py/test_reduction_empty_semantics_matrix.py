# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

import vibetensor.torch as vbt


def _to_device(t, device: str):
    if device == "cuda":
        if not vbt.cuda.is_available():
            pytest.skip("CUDA not available")
        return t.cuda()
    assert device == "cpu"
    return t


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sum_empty_full_reduction_returns_zero(device):
    t = vbt.empty((0,), dtype=vbt.float32)
    t = _to_device(t, device)
    out = t.sum()
    if device == "cuda":
        out = out.cpu()
    assert float(out.item()) == 0.0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mean_empty_full_reduction_returns_nan(device):
    t = vbt.empty((0,), dtype=vbt.float32)
    t = _to_device(t, device)
    out = t.mean()
    if device == "cuda":
        out = out.cpu()
    assert math.isnan(float(out.item()))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_amin_empty_full_reduction_raises_exact_message(device):
    t = vbt.empty((0,), dtype=vbt.float32)
    t = _to_device(t, device)
    with pytest.raises(RuntimeError, match=r"^amin: empty$"):
        _ = t.amin()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_min_empty_full_reduction_raises_exact_message(device):
    t = vbt.empty((0,), dtype=vbt.float32)
    t = _to_device(t, device)
    with pytest.raises(RuntimeError, match=r"^amin: empty$"):
        _ = t.min()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_amax_empty_slice_dim_raises_exact_message(device):
    t = vbt.empty((2, 0, 3), dtype=vbt.float32)
    t = _to_device(t, device)
    with pytest.raises(RuntimeError, match=r"^amax: empty$"):
        _ = t.amax(dim=1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_amax_out_numel_zero_returns_empty(device):
    t = vbt.empty((0, 5), dtype=vbt.float32)
    t = _to_device(t, device)
    out = t.amax(dim=1)
    assert out.sizes == (0,)
