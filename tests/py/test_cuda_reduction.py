# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

import vibetensor.torch as vbt
import math

if not vbt.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


_TORCH_CUDA_REF_OK = None


def _torch_ref(fn, *args, **kwargs):
    """Best-effort torch CUDA reference; fall back to CPU when NVRTC fails."""

    global _TORCH_CUDA_REF_OK

    if not torch.cuda.is_available():
        return fn(*args, **kwargs)

    if _TORCH_CUDA_REF_OK is False:
        return fn(*args, **kwargs)

    try:
        cuda_args = [a.cuda() if isinstance(a, torch.Tensor) else a for a in args]
        out = fn(*cuda_args, **kwargs)
        _TORCH_CUDA_REF_OK = True
        return out
    except RuntimeError as e:
        if "nvrtc: error: invalid value for --gpu-architecture" in str(e):
            _TORCH_CUDA_REF_OK = False
            return fn(*args, **kwargs)
        raise


def _assert_vbt_cuda(t) -> None:
    assert t.device[0] == 2  # kDLCUDA


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("dtype", [vbt.float32, vbt.int64])
def test_sum(shape, dtype):
    t_ref = torch.randint(-10, 10, shape).to(dtype=getattr(torch, dtype.__name__))
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    # Full reduction
    out_ref = _torch_ref(lambda x: x.sum(), t_ref)
    out_vbt = t_vbt.sum()
    _assert_vbt_cuda(out_vbt)
    assert math.isclose(out_ref.item(), out_vbt.item(), rel_tol=1e-5, abs_tol=1e-5)

    # Dim reduction
    for dim in range(len(shape)):
        out_ref_d = _torch_ref(lambda x: x.sum(dim=dim), t_ref)
        out_vbt_d = t_vbt.sum(dim=dim)
        _assert_vbt_cuda(out_vbt_d)
        assert torch.allclose(
            out_ref_d.cpu(), torch.tensor(out_vbt_d.cpu().numpy()), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("shape", [(10, 10), (5, 5, 5)])
def test_mean(shape):
    t_ref = torch.randn(shape)
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(lambda x: x.mean(), t_ref)
    out_vbt = t_vbt.mean()
    _assert_vbt_cuda(out_vbt)
    assert math.isclose(out_ref.item(), out_vbt.item(), rel_tol=1e-5, abs_tol=1e-5)

    for dim in range(len(shape)):
        out_ref_d = _torch_ref(lambda x: x.mean(dim=dim), t_ref)
        out_vbt_d = t_vbt.mean(dim=dim)
        _assert_vbt_cuda(out_vbt_d)
        assert torch.allclose(
            out_ref_d.cpu(), torch.tensor(out_vbt_d.cpu().numpy()), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("shape", [(4, 3), (0, 3)])
def test_mean_int64_raises_value_error_with_exact_message(shape):
    t_ref = torch.randint(-10, 10, shape, dtype=torch.int64)
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()
    _assert_vbt_cuda(t_vbt)

    with pytest.raises(ValueError, match=r"^mean: expected dtype=float32$"):
        _ = t_vbt.mean()


@pytest.mark.parametrize("op", ["min", "max", "prod"])
def test_other_reductions(op):
    shape = (10, 10)
    t_ref = torch.randint(1, 10, shape).float()
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(lambda x: getattr(x, op)(), t_ref)
    out_vbt = getattr(t_vbt, op)()
    _assert_vbt_cuda(out_vbt)

    # prod can overflow easily, so use small numbers and higher tolerance or log diff?
    # for min/max it should be exact.
    assert math.isclose(out_ref.item(), out_vbt.item(), rel_tol=1e-4, abs_tol=1e-4)

    if op in ["min", "max"]:
        out_ref_d = _torch_ref(lambda x: getattr(x, op)(dim=0).values, t_ref)
        out_vbt_d = getattr(t_vbt, op)(dim=0).values
    else:
        out_ref_d = _torch_ref(lambda x: getattr(x, op)(dim=0), t_ref)
        out_vbt_d = getattr(t_vbt, op)(dim=0)

    _assert_vbt_cuda(out_vbt_d)

    assert torch.allclose(
        out_ref_d.cpu(), torch.tensor(out_vbt_d.cpu().numpy()), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    ("vals", "kind", "expected"),
    [
        ([math.inf, math.inf], "min", math.inf),
        ([-math.inf, -math.inf], "max", -math.inf),
        ([1.0, math.inf], "min", 1.0),
        ([-math.inf, 2.0], "max", 2.0),
    ],
)
def test_float32_min_max_identities_handle_inf(vals, kind, expected):
    if not hasattr(torch.Tensor, "amin") or not hasattr(torch.Tensor, "amax"):
        pytest.skip("torch.Tensor.amin/amax not available")

    t_ref = torch.tensor(vals, dtype=torch.float32)
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()
    _assert_vbt_cuda(t_vbt)

    if kind == "min":
        ref_ops = [lambda x: x.amin(), lambda x: x.min()]
        vbt_ops = [lambda x: x.amin(), lambda x: x.min()]
    else:
        ref_ops = [lambda x: x.amax(), lambda x: x.max()]
        vbt_ops = [lambda x: x.amax(), lambda x: x.max()]

    for ref_fn, vbt_fn in zip(ref_ops, vbt_ops):
        out_ref = _torch_ref(ref_fn, t_ref)
        out_vbt = vbt_fn(t_vbt)
        _assert_vbt_cuda(out_vbt)
        assert float(out_ref.item()) == float(out_vbt.item())
        assert float(out_vbt.item()) == float(expected)


@pytest.mark.parametrize("dtype", [vbt.float32, vbt.int64])
def test_sum_scalar(dtype):
    t_ref = torch.tensor(7, dtype=getattr(torch, dtype.__name__))
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(lambda x: x.sum(), t_ref)
    out_vbt = t_vbt.sum()
    _assert_vbt_cuda(out_vbt)

    assert out_ref.item() == out_vbt.item()
