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
    """Best-effort torch CUDA reference; fall back to CPU when NVRTC fails.

    Some environments have a torch build that can't NVRTC-compile for the local
    GPU arch (e.g. nvrtc "invalid value for --gpu-architecture"). In that case,
    use CPU reference values so we still validate VibeTensor outputs.
    """

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


def test_relu_transpose_matches_torch():
    if not vbt._has_vt_op("relu"):
        pytest.skip("vt.relu not available")

    t_ref = torch.randn(32, 16)
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    t_ref_t = t_ref.transpose(0, 1)
    t_vbt_t = t_vbt.transpose(0, 1)

    out_ref = _torch_ref(torch.relu, t_ref_t)

    out_vbt = vbt.ops.vt.relu(t_vbt_t)
    _assert_vbt_cuda(out_vbt)

    assert torch.allclose(
        out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=0.0, atol=0.0
    )


def test_angle_signed_zero_matches_torch():
    if not vbt._has_vt_op("angle"):
        pytest.skip("vt.angle not available")

    x_ref = torch.tensor([-0.0, 0.0, -1.0, 1.0], dtype=torch.float32)

    out_ref = _torch_ref(torch.angle, x_ref)

    x_vbt = vbt.tensor(x_ref.numpy()).cuda()
    out_vbt = vbt.ops.vt.angle(x_vbt)
    _assert_vbt_cuda(out_vbt)

    out_vbt_t = torch.tensor(out_vbt.cpu().numpy())
    out_ref_t = out_ref.cpu()

    assert torch.allclose(out_ref_t, out_vbt_t, rtol=0.0, atol=0.0)
    assert torch.equal(torch.signbit(out_ref_t), torch.signbit(out_vbt_t))


def test_sgn_nan_is_zero_matches_torch():
    if not vbt._has_vt_op("sgn"):
        pytest.skip("vt.sgn not available")

    x_ref = torch.tensor([float("nan"), -3.0, 0.0, 2.0], dtype=torch.float32)

    out_ref = _torch_ref(torch.sgn, x_ref)

    x_vbt = vbt.tensor(x_ref.numpy()).cuda()
    out_vbt = vbt.ops.vt.sgn(x_vbt)
    _assert_vbt_cuda(out_vbt)

    assert torch.allclose(
        out_ref.cpu(),
        torch.tensor(out_vbt.cpu().numpy()),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_bitwise_int32_matches_torch():
    required = ["bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not"]
    for name in required:
        if not vbt._has_vt_op(name):
            pytest.skip(f"vt.{name} not available")

    a_ref = torch.randint(-50, 50, (4, 1), dtype=torch.int32)
    b_ref = torch.randint(-50, 50, (1, 4), dtype=torch.int32)

    a_vbt = vbt.tensor(a_ref.numpy()).cuda()
    b_vbt = vbt.tensor(b_ref.numpy()).cuda()

    out_ref = _torch_ref(torch.bitwise_and, a_ref, b_ref)
    out_vbt = vbt.ops.vt.bitwise_and(a_vbt, b_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))

    out_ref = _torch_ref(torch.bitwise_or, a_ref, b_ref)
    out_vbt = vbt.ops.vt.bitwise_or(a_vbt, b_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))

    out_ref = _torch_ref(torch.bitwise_xor, a_ref, b_ref)
    out_vbt = vbt.ops.vt.bitwise_xor(a_vbt, b_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))

    out_ref = _torch_ref(torch.bitwise_not, a_ref)
    out_vbt = vbt.ops.vt.bitwise_not(a_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))


def test_bitwise_shift_int32_matches_torch():
    required = ["lshift", "rshift", "bitwise_left_shift", "bitwise_right_shift"]
    for name in required:
        if not vbt._has_vt_op(name):
            pytest.skip(f"vt.{name} not available")

    x_ref = torch.randint(0, 64, (4, 1), dtype=torch.int32)
    s_ref = torch.randint(0, 5, (1, 4), dtype=torch.int32)

    x_vbt = vbt.tensor(x_ref.numpy()).cuda()
    s_vbt = vbt.tensor(s_ref.numpy()).cuda()

    out_ref = _torch_ref(torch.bitwise_left_shift, x_ref, s_ref)

    out_vbt = vbt.ops.vt.bitwise_left_shift(x_vbt, s_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))

    out_vbt = vbt.ops.vt.lshift(x_vbt, s_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))

    out_ref = _torch_ref(torch.bitwise_right_shift, x_ref, s_ref)

    out_vbt = vbt.ops.vt.bitwise_right_shift(x_vbt, s_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))

    out_vbt = vbt.ops.vt.rshift(x_vbt, s_vbt)
    _assert_vbt_cuda(out_vbt)
    assert torch.equal(out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_transcendental_ops_require_float_dtype(dtype):
    """Test that transcendental/special math ops require floating-point inputs.

    KNOWN DIVERGENCE FROM PYTORCH:
    ------------------------------
    PyTorch automatically promotes integer inputs to float for transcendental ops
    (exp, log, sin, cos, etc.). VibeTensor currently requires explicit float dtype.

    Rationale:
    - Explicit typing avoids silent precision loss and unexpected output dtypes.
    - Users should cast explicitly: vbt.ops.vt.exp(x.float()) instead of vbt.ops.vt.exp(x).
    - This may be revisited in the future if dtype promotion rules are implemented.

    See also: test_bool_binary_ops_reject for similar intentional divergence on bools.
    """
    x_ref = torch.tensor([1, 2, -3], dtype=dtype)
    y_ref = torch.tensor([3, 4, 5], dtype=dtype)

    x_vbt = vbt.tensor(x_ref.numpy()).cuda()
    y_vbt = vbt.tensor(y_ref.numpy()).cuda()

    unary_ops = [
        "exp",
        "log",
        "log10",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "tanh",
        "sigmoid",
        "expm1",
        "log1p",
        "reciprocal",
        "frac",
        "exp2",
        "log2",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "tan",
        "deg2rad",
        "rad2deg",
        "asin",
        "acos",
        "atan",
        "erf",
        "erfc",
        "lgamma",
        "sinc",
        "logit",
        "hardsigmoid",
        "silu",
        "gelu",
        "mish",
        "selu",
        "softplus",
        "hardshrink",
        "softshrink",
        "celu",
        "elu",
    ]

    for op in unary_ops:
        if not vbt._has_vt_op(op):
            continue
        with pytest.raises(Exception, match="floating"):
            getattr(vbt.ops.vt, op)(x_vbt)

    binary_ops = [
        "atan2",
        "copysign",
        "hypot",
        "logaddexp",
        "logaddexp2",
        "ldexp",
        "float_power",
        "xlogy",
        "xlog1py",
        "special_xlog1py",
        "nextafter",
    ]

    for op in binary_ops:
        if not vbt._has_vt_op(op):
            continue
        with pytest.raises(Exception, match="floating"):
            getattr(vbt.ops.vt, op)(x_vbt, y_vbt)


@pytest.mark.parametrize("op", ["erf", "erfc", "lgamma"])
def test_unary_special(op):
    t_ref = torch.randn(10, 10)
    # lgamma input must be positive for stability (though defined for neg)
    if op == "lgamma":
        t_ref = t_ref.abs() + 0.1

    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(lambda x: getattr(x, op)(), t_ref)
    out_vbt = getattr(t_vbt, op)()
    _assert_vbt_cuda(out_vbt)

    assert torch.allclose(
        out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
    )


def test_pow():
    t_a = torch.rand(10) + 0.1
    t_b = torch.rand(10)

    out_ref = _torch_ref(torch.pow, t_a, t_b)

    if vbt._has_vt_op("pow"):
        out_vbt = vbt.ops.vt.pow(
            vbt.tensor(t_a.numpy()).cuda(), vbt.tensor(t_b.numpy()).cuda()
        )
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


def test_clamp():
    t = torch.randn(10)
    mn = torch.tensor(-0.5)
    mx = torch.tensor(0.5)

    out_ref = _torch_ref(torch.clamp, t, -0.5, 0.5)

    if vbt._has_vt_op("clamp"):
        out_vbt = vbt.ops.vt.clamp(
            vbt.tensor(t.numpy()).cuda(),
            vbt.tensor(mn.numpy()).cuda(),
            vbt.tensor(mx.numpy()).cuda(),
        )
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


def test_lerp():
    start = torch.randn(10)
    end = torch.randn(10)
    weight = torch.rand(10)

    out_ref = _torch_ref(torch.lerp, start, end, weight)

    if vbt._has_vt_op("lerp"):
        out_vbt = vbt.ops.vt.lerp(
            vbt.tensor(start.numpy()).cuda(),
            vbt.tensor(end.numpy()).cuda(),
            vbt.tensor(weight.numpy()).cuda(),
        )
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("op", ["exp2", "expm1", "sinc"])
def test_unary_extended(op):
    t_ref = torch.randn(10, 10)
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(getattr(torch, op), t_ref)

    if vbt._has_vt_op(op):
        out_vbt = getattr(vbt.ops.vt, op)(t_vbt)
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


def test_logit():
    # Logit requires inputs in [0, 1], technically (0, 1) for finite output
    t_ref = torch.rand(10, 10) * 0.9 + 0.05
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(torch.logit, t_ref)

    if vbt._has_vt_op("logit"):
        out_vbt = vbt.ops.vt.logit(t_vbt)
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


def test_polygamma():
    n = 2
    t_ref = torch.randn(10, 10).abs() + 0.1
    t_vbt = vbt.tensor(t_ref.numpy()).cuda()

    out_ref = _torch_ref(torch.polygamma, n, t_ref)

    if vbt._has_vt_op("polygamma"):
        # Expecting polygamma(n, input)
        out_vbt = vbt.ops.vt.polygamma(n, t_vbt)
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("op", ["fmax", "fmin"])
def test_binary_compare(op):
    t1 = torch.randn(10)
    t2 = torch.randn(10)

    out_ref = _torch_ref(getattr(torch, op), t1, t2)

    if vbt._has_vt_op(op):
        out_vbt = getattr(vbt.ops.vt, op)(
            vbt.tensor(t1.numpy()).cuda(), vbt.tensor(t2.numpy()).cuda()
        )
        _assert_vbt_cuda(out_vbt)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("loss_name", ["huber_loss", "mse_loss", "smooth_l1_loss"])
def test_losses(loss_name):
    input = torch.randn(10)
    target = torch.randn(10)

    # Reference calculation with reduction='none'
    if loss_name == "huber_loss":
        out_ref = _torch_ref(
            lambda a, b: torch.nn.functional.huber_loss(
                a, b, reduction="none", delta=1.0
            ),
            input,
            target,
        )
    elif loss_name == "mse_loss":
        out_ref = _torch_ref(
            lambda a, b: torch.nn.functional.mse_loss(a, b, reduction="none"),
            input,
            target,
        )
    elif loss_name == "smooth_l1_loss":
        out_ref = _torch_ref(
            lambda a, b: torch.nn.functional.smooth_l1_loss(
                a, b, reduction="none", beta=1.0
            ),
            input,
            target,
        )

    if vbt._has_vt_op(loss_name):
        v_in = vbt.tensor(input.numpy()).cuda()
        v_tg = vbt.tensor(target.numpy()).cuda()

        # Speculative call signatures
        try:
            if loss_name == "huber_loss":
                # Assuming (input, target, delta) or (input, target)
                try:
                    out_vbt = vbt.ops.vt.huber_loss(v_in, v_tg, 1.0)
                except TypeError:
                    out_vbt = vbt.ops.vt.huber_loss(v_in, v_tg)
            elif loss_name == "smooth_l1_loss":
                try:
                    out_vbt = vbt.ops.vt.smooth_l1_loss(v_in, v_tg, 1.0)
                except TypeError:
                    out_vbt = vbt.ops.vt.smooth_l1_loss(v_in, v_tg)
            else:  # mse_loss
                out_vbt = vbt.ops.vt.mse_loss(v_in, v_tg)

            _assert_vbt_cuda(out_vbt)

            assert torch.allclose(
                out_ref.cpu(),
                torch.tensor(out_vbt.cpu().numpy()),
                rtol=1e-4,
                atol=1e-4,
            )
        except Exception as e:
            pytest.fail(f"Failed to call {loss_name}: {e}")


@pytest.mark.parametrize("op", ["addcmul", "addcdiv"])
def test_addc_ops(op):
    t = torch.randn(10)
    t1 = torch.randn(10)
    t2 = torch.randn(10)
    # Avoid div by zero for addcdiv
    if op == "addcdiv":
        t2 = t2.abs() + 0.1

    value = 0.5

    out_ref = _torch_ref(getattr(torch, op), t, t1, t2, value=value)

    if vbt._has_vt_op(op):
        v_t = vbt.tensor(t.numpy()).cuda()
        v_t1 = vbt.tensor(t1.numpy()).cuda()
        v_t2 = vbt.tensor(t2.numpy()).cuda()

        out_vbt = getattr(vbt.ops.vt, op)(v_t, v_t1, v_t2, value)
        _assert_vbt_cuda(out_vbt)

        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sqrt_negative_domain_matches_torch(dtype):
    if not vbt._has_vt_op("sqrt"):
        pytest.skip("vt.sqrt not available")

    x_ref = torch.tensor([-1.0, -0.0, 0.0, 1.0], dtype=dtype)
    out_ref = torch.sqrt(x_ref)

    x_vbt = vbt.cuda.to_device(x_ref.cpu().numpy())
    out_vbt = vbt.ops.vt.sqrt(x_vbt)
    _assert_vbt_cuda(out_vbt)

    out_vbt_t = torch.tensor(vbt.cuda.from_device(out_vbt))
    out_ref_t = out_ref.cpu()

    assert torch.allclose(out_ref_t, out_vbt_t, rtol=0.0, atol=0.0, equal_nan=True)

    # NaN sign bits are not stable across implementations; only enforce signbit on non-NaNs.
    mask = ~torch.isnan(out_ref_t)
    assert torch.equal(torch.signbit(out_ref_t[mask]), torch.signbit(out_vbt_t[mask]))


def test_remainder_matches_torch_float_signed_zero():
    if not vbt._has_vt_op("remainder"):
        pytest.skip("vt.remainder not available")

    a_ref = torch.tensor([[-1.0], [-0.0], [0.0], [1.0]], dtype=torch.float32)
    b_ref = torch.tensor([[2.0, -2.0]], dtype=torch.float32)

    out_ref = torch.remainder(a_ref, b_ref)

    a_vbt = vbt.cuda.to_device(a_ref.numpy())
    b_vbt = vbt.cuda.to_device(b_ref.numpy())

    out_vbt = vbt.ops.vt.remainder(a_vbt, b_vbt)
    _assert_vbt_cuda(out_vbt)

    out_vbt_t = torch.tensor(out_vbt.cpu().numpy())

    assert torch.allclose(out_ref, out_vbt_t, rtol=0.0, atol=0.0, equal_nan=True)
    assert torch.equal(torch.signbit(out_ref), torch.signbit(out_vbt_t))


@pytest.mark.parametrize("dtype", [torch.int64])
def test_remainder_matches_torch_int(dtype):
    if not vbt._has_vt_op("remainder"):
        pytest.skip("vt.remainder not available")

    a_ref = torch.tensor([[-5], [-1], [0], [1], [5]], dtype=dtype)
    b_ref = torch.tensor([[2, -2, 3, -3]], dtype=dtype)

    out_ref = torch.remainder(a_ref, b_ref)

    a_vbt = vbt.cuda.to_device(a_ref.numpy())
    b_vbt = vbt.cuda.to_device(b_ref.numpy())

    out_vbt = vbt.ops.vt.remainder(a_vbt, b_vbt)
    _assert_vbt_cuda(out_vbt)

    assert torch.equal(out_ref, torch.tensor(out_vbt.cpu().numpy()))


@pytest.mark.parametrize("op", ["sub", "div", "rsub", "fmod", "remainder"])
def test_bool_binary_ops_reject(op):
    """Test that certain binary ops reject boolean inputs.

    KNOWN DIVERGENCE FROM PYTORCH:
    ------------------------------
    PyTorch allows bool inputs for div/sub/fmod/remainder via implicit promotion.
    VibeTensor rejects these to avoid semantically surprising behavior (e.g., True/True=1.0).

    Rationale:
    - Boolean arithmetic (sub, div, fmod, remainder) is rarely intentional.
    - Explicit cast to int/float makes intent clear: x.int() - y.int().
    - Bitwise ops (and, or, xor) are supported for bool tensors.

    See also: test_transcendental_ops_require_float_dtype for similar divergence.
    """
    if not vbt._has_vt_op(op):
        pytest.skip(f"vt.{op} not available")

    a_ref = torch.tensor([True, False], dtype=torch.bool)
    b_ref = torch.tensor([True, True], dtype=torch.bool)

    a_vbt = vbt.tensor(a_ref.numpy()).cuda()
    b_vbt = vbt.tensor(b_ref.numpy()).cuda()

    with pytest.raises(Exception, match="boolean"):
        getattr(vbt.ops.vt, op)(a_vbt, b_vbt)


@pytest.mark.parametrize("op", ["floor", "ceil", "trunc", "round"])
def test_int64_rounding_ops_identity_large(op):
    if not vbt._has_vt_op(op):
        pytest.skip(f"vt.{op} not available")

    x = torch.tensor(
        [
            2**53 + 1,
            -(2**53 + 1),
            2**60 + 123,
            -(2**60 + 123),
        ],
        dtype=torch.int64,
    )

    out_ref = getattr(torch, op)(x)

    x_vbt = vbt.tensor(x.numpy()).cuda()
    out_vbt = getattr(vbt.ops.vt, op)(x_vbt)
    _assert_vbt_cuda(out_vbt)

    assert torch.equal(out_ref, torch.tensor(out_vbt.cpu().numpy()))


def test_int64_fmod_remainder_large_values():
    if not (vbt._has_vt_op("fmod") and vbt._has_vt_op("remainder")):
        pytest.skip("vt.fmod/vt.remainder not available")

    a = torch.tensor([2**60 + 3, -(2**60 + 3)], dtype=torch.int64).reshape(2, 1)
    b = torch.tensor([7, -7], dtype=torch.int64).reshape(1, 2)

    out_ref_fmod = torch.fmod(a, b)
    out_ref_rem = torch.remainder(a, b)

    a_vbt = vbt.tensor(a.numpy()).cuda()
    b_vbt = vbt.tensor(b.numpy()).cuda()

    out_vbt_fmod = vbt.ops.vt.fmod(a_vbt, b_vbt)
    out_vbt_rem = vbt.ops.vt.remainder(a_vbt, b_vbt)

    assert torch.equal(out_ref_fmod, torch.tensor(out_vbt_fmod.cpu().numpy()))
    assert torch.equal(out_ref_rem, torch.tensor(out_vbt_rem.cpu().numpy()))


@pytest.mark.parametrize(
    "dtype, shifts",
    [
        (torch.int32, [-1, 0, 1, 31, 32, 33, 63, 64, 65]),
        (torch.int64, [-1, 0, 1, 31, 32, 33, 63, 64, 65]),
    ],
)
def test_bitwise_shift_edge_cases_matches_torch(dtype, shifts):
    required = ["lshift", "rshift", "bitwise_left_shift", "bitwise_right_shift"]
    for name in required:
        if not vbt._has_vt_op(name):
            pytest.skip(f"vt.{name} not available")

    x_ref = torch.tensor([1, -2, 3], dtype=dtype).reshape(3, 1)
    s_ref = torch.tensor(shifts, dtype=dtype).reshape(1, len(shifts))

    left_ref = torch.bitwise_left_shift(x_ref, s_ref)
    right_ref = torch.bitwise_right_shift(x_ref, s_ref)

    x_vbt = vbt.tensor(x_ref.numpy()).cuda()
    s_vbt = vbt.tensor(s_ref.numpy()).cuda()

    left_vbt = vbt.ops.vt.bitwise_left_shift(x_vbt, s_vbt)
    right_vbt = vbt.ops.vt.bitwise_right_shift(x_vbt, s_vbt)

    assert torch.equal(left_ref, torch.tensor(left_vbt.cpu().numpy()))
    assert torch.equal(right_ref, torch.tensor(right_vbt.cpu().numpy()))

    # Alias ops
    left_vbt2 = vbt.ops.vt.lshift(x_vbt, s_vbt)
    right_vbt2 = vbt.ops.vt.rshift(x_vbt, s_vbt)

    assert torch.equal(left_ref, torch.tensor(left_vbt2.cpu().numpy()))
    assert torch.equal(right_ref, torch.tensor(right_vbt2.cpu().numpy()))


def test_ternary_dispatcher_device_guard_multi_gpu():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices")
    if not vbt._has_vt_op("clamp"):
        pytest.skip("vt.clamp not available")

    orig_dev = torch.cuda.current_device()
    try:
        # Create tensors explicitly on cuda:1
        t = torch.randn(16)
        t_vbt = vbt.cuda.to_device(t.numpy(), device=1)
        mn_vbt = vbt.cuda.to_device(torch.tensor(-0.5, dtype=torch.float32).numpy(), device=1)
        mx_vbt = vbt.cuda.to_device(torch.tensor(0.5, dtype=torch.float32).numpy(), device=1)

        assert t_vbt.device[0] == 2 and t_vbt.device[1] == 1

        # Flip runtime current device away from the tensor device.
        torch.cuda.set_device(0)

        out_vbt = vbt.ops.vt.clamp(t_vbt, mn_vbt, mx_vbt)
        _assert_vbt_cuda(out_vbt)
        assert out_vbt.device[1] == 1

        out_ref = torch.clamp(t, -0.5, 0.5)
        assert torch.allclose(
            out_ref.cpu(), torch.tensor(out_vbt.cpu().numpy()), rtol=1e-4, atol=1e-4
        )
    finally:
        torch.cuda.set_device(orig_dev)
