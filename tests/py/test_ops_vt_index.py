# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C
from vibetensor.torch import ops


def _to_numpy_cpu(t):
    cap = vt.to_dlpack(t)
    # NumPy's from_dlpack historically accepted raw DLPack capsules
    # directly, but newer versions expect an object with a __dlpack__
    # method. Support both by wrapping the capsule when needed.
    try:
        arr = np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover - tiny adapter
                return self._inner

        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr.reshape(tuple(int(s) for s in t.sizes))


def _make_prefix_slice_meta_v0(prefix_len: int = 1):
    """Build vt meta v0 for a single full slice (:) prefix."""

    if prefix_len != 1:
        raise ValueError("only prefix_len==1 supported in tests")

    sent = np.iinfo(np.int64).min
    meta_np = np.array(
        [
            0,  # version
            1,  # adv_kind = Tensor
            0,  # adv_param (reserved)
            prefix_len,
            2,  # kind_tag = Slice
            sent,  # start
            sent,  # stop
            sent,  # step
        ],
        dtype=np.int64,
    )
    return vt.tensor(meta_np, dtype="int64")


def test_ops_vt_index_matches_tensor_getitem_cpu():
    x = vt.arange(6, dtype="float32")
    idx = vt.tensor([0, 3, 5], dtype="int64")

    idx2, meta = C._encode_index_spec(x, idx)

    y_ops = ops.vt.index(x, idx2, meta)
    y_py = x[idx]

    np.testing.assert_allclose(_to_numpy_cpu(y_ops), _to_numpy_cpu(y_py))


def test_ops_vt_index_put_matches_setitem_cpu():
    x_ref = vt.zeros((4,), dtype="float32")
    idx = vt.tensor([1, 3], dtype="int64")
    v = vt.tensor([5.0, 7.0], dtype="float32")

    # Reference using Tensor.__setitem__.
    x_ref[idx] = v

    x_ops = vt.zeros((4,), dtype="float32")
    idx2, meta = C._encode_index_spec(x_ops, idx)
    acc = C._cpu_full([], "bool", False)  # 0-d Bool accumulate=False

    out = ops.vt.index_put(x_ops, idx2, v, meta, acc)

    np.testing.assert_allclose(_to_numpy_cpu(out), _to_numpy_cpu(x_ref))
    np.testing.assert_allclose(_to_numpy_cpu(x_ops), _to_numpy_cpu(x_ref))


def test_ops_vt_index_prefix_slice_matches_python_getitem_cpu():
    x = vt.arange(6, dtype="float32").reshape((2, 3))
    idx = vt.tensor([0, 2], dtype="int64")

    meta = _make_prefix_slice_meta_v0()

    y_ops = ops.vt.index(x, idx, meta)
    y_ref = x[:, idx]

    np.testing.assert_allclose(_to_numpy_cpu(y_ops), _to_numpy_cpu(y_ref))


def test_ops_vt_index_put_prefix_slice_matches_python_setitem_cpu():
    x_ref = vt.zeros((2, 3), dtype="float32")
    idx = vt.tensor([0, 2], dtype="int64")
    v = vt.tensor([[5.0, 7.0], [9.0, 11.0]], dtype="float32")

    x_ref[:, idx] = v

    x_ops = vt.zeros((2, 3), dtype="float32")
    meta = _make_prefix_slice_meta_v0()
    acc = C._cpu_full([], "bool", False)  # 0-d Bool accumulate=False

    out = ops.vt.index_put(x_ops, idx, v, meta, acc)

    np.testing.assert_allclose(_to_numpy_cpu(out), _to_numpy_cpu(x_ref))
    np.testing.assert_allclose(_to_numpy_cpu(x_ops), _to_numpy_cpu(x_ref))


def test_ops_vt_index_invalid_meta_raises_value_error():
    x = vt.arange(4, dtype="float32")
    idx = vt.tensor([0, 1], dtype="int64")
    idx2, meta = C._encode_index_spec(x, idx)

    # Build a non-contiguous 1-D view of meta to trigger vt meta validation.
    meta_view = meta[::2]

    with pytest.raises(
        ValueError,
        match="meta must be 1-D CPU int64 with at least 4 elements",
    ):
        _ = ops.vt.index(x, idx2, meta_view)


def test_ops_vt_index_invalid_meta_version_raises_value_error():
    x = vt.arange(4, dtype="float32")
    idx = vt.tensor([0, 1], dtype="int64")
    idx2, meta = C._encode_index_spec(x, idx)

    meta_np = _to_numpy_cpu(meta).astype(np.int64)
    assert meta_np.shape == (4,)
    meta_np[0] = 1  # unsupported meta version
    meta_bad = vt.tensor(meta_np, dtype="int64")

    with pytest.raises(
        ValueError,
        match="unsupported meta version",
    ):
        _ = ops.vt.index(x, idx2, meta_bad)


def test_ops_vt_index_put_invalid_meta_version_mentions_vt_index_put():
    x = vt.zeros((4,), dtype="float32")
    idx = vt.tensor([0, 1], dtype="int64")
    v = vt.tensor([1.0, 2.0], dtype="float32")
    idx2, meta = C._encode_index_spec(x, idx)

    meta_np = _to_numpy_cpu(meta).astype(np.int64)
    meta_np[0] = 1
    meta_bad = vt.tensor(meta_np, dtype="int64")
    acc = C._cpu_full([], "bool", False)

    with pytest.raises(
        ValueError,
        match=r"vt::index_put: unsupported meta version",
    ):
        _ = ops.vt.index_put(x, idx2, v, meta_bad, acc)


def test_ops_vt_index_prefix_meta_autograd_requires_v2():
    x = vt.arange(6, dtype="float32").reshape((2, 3)).detach()
    x.requires_grad = True
    idx = vt.tensor([0, 2], dtype="int64")
    meta = _make_prefix_slice_meta_v0()

    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(False)  # type: ignore[attr-defined]
    try:
        with pytest.raises(
            ValueError,
            match="prefix meta requires VBT_AUTOGRAD_INDEXING_V2",
        ):
            _ = ops.vt.index(x, idx, meta)
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_ops_vt_index_non_cpu_self_allows_cpu_index_tensor():
    if not getattr(C, "_has_cuda", False) or int(getattr(C, "_cuda_device_count", lambda: 0)()) == 0:
        pytest.skip("CUDA not available")

    x0_np = np.arange(4, dtype=np.float32)
    x_cuda = vt.cuda.to_device(x0_np)

    idx = vt.tensor([0, 1], dtype="int64")

    # Build (index, meta) for a CPU tensor but reuse them with CUDA self.
    tmp_cpu = vt.zeros((4,), dtype="float32")
    idx2, meta = C._encode_index_spec(tmp_cpu, idx)

    y = ops.vt.index(x_cuda, idx2, meta)
    np.testing.assert_allclose(vt.cuda.from_device(y), x0_np[[0, 1]])


def test_ops_vt_index_put_matches_setitem_cuda_no_grad():
    if not getattr(C, "_has_cuda", False) or int(C._cuda_device_count()) == 0:
        pytest.skip("CUDA not available")

    idx = vt.tensor([1, 3], dtype="int64")
    v = vt.tensor([5.0, 7.0], dtype="float32")

    # CPU reference using Tensor.__setitem__.
    x_ref = vt.zeros((4,), dtype="float32")
    x_ref[idx] = v
    x_ref_np = _to_numpy_cpu(x_ref)

    x0_np = np.zeros(4, dtype=np.float32)
    idx_np = np.array([1, 3], dtype=np.int64)
    v_np = np.array([5.0, 7.0], dtype=np.float32)

    x_cuda = vt.cuda.to_device(x0_np)
    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    idx2, meta = C._encode_index_spec(x_cuda, idx_cuda)
    acc = C._cpu_full([], "bool", False)  # 0-d Bool accumulate=False

    ag = C.autograd
    prev = ag.is_grad_enabled()
    try:
        with ag.no_grad():  # type: ignore[attr-defined]
            out = ops.vt.index_put(x_cuda, idx2, v_cuda, meta, acc)
    finally:
        ag.set_grad_enabled(prev)

    x_cuda_np = vt.cuda.from_device(x_cuda)
    np.testing.assert_allclose(x_cuda_np, x_ref_np)
    np.testing.assert_allclose(vt.cuda.from_device(out), x_ref_np)


def test_ops_vt_index_put_cuda_autograd_rejects():
    if not getattr(C, "_has_cuda", False) or int(C._cuda_device_count()) == 0:
        pytest.skip("CUDA not available")

    x0_np = np.zeros(4, dtype=np.float32)
    x_cuda = vt.cuda.to_device(x0_np)
    x_cuda.set_requires_grad(True)

    idx_np = np.array([1, 3], dtype=np.int64)
    v_np = np.array([5.0, 7.0], dtype=np.float32)
    idx_cuda = vt.cuda.to_device(idx_np)
    v_cuda = vt.cuda.to_device(v_np)

    idx2, meta = C._encode_index_spec(x_cuda, idx_cuda)
    acc = C._cpu_full([], "bool", False)

    ag = C.autograd
    prev = ag.is_grad_enabled()
    try:
        ag.set_grad_enabled(True)
        with pytest.raises(
            RuntimeError,
            match="autograd for in-place advanced indexing is not supported",
        ):
            _ = ops.vt.index_put(x_cuda, idx2, v_cuda, meta, acc)
    finally:
        ag.set_grad_enabled(prev)

    # Data should remain unchanged on failure.
    x_after = vt.cuda.from_device(x_cuda)
    np.testing.assert_allclose(x_after, x0_np)
