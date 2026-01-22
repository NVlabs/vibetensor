# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _to_numpy_cpu(t):
    """Convert a vibetensor tensor (CPU) to a NumPy array via DLPack."""
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


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_advanced_index_read_1d_tensor_index_cpu():
    x = vt.arange(6, dtype="float32")
    idx = vt.tensor([0, 3, 5], dtype="int64")

    y = x[idx]

    x_np = _to_numpy_cpu(x)
    idx_np = _to_numpy_cpu(idx).astype(np.int64)
    y_np = _to_numpy_cpu(y)

    assert y_np.shape == (3,)
    np.testing.assert_allclose(y_np, x_np[idx_np])


def test_advanced_index_read_2d_prefix_tensor_index_cpu():
    x = vt.arange(6, dtype="float32").reshape((2, 3))
    idx = vt.tensor([0, 2], dtype="int64")

    y = x[:, idx]

    x_np = _to_numpy_cpu(x)
    idx_np = _to_numpy_cpu(idx).astype(np.int64)
    y_np = _to_numpy_cpu(y)

    assert y_np.shape == (2, 2)
    np.testing.assert_allclose(y_np, x_np[:, idx_np])


def test_advanced_index_read_2d_tensor_index_suffix_full_slice_cpu():
    x = vt.arange(12, dtype="float32").reshape((4, 3))
    idx = vt.tensor([2, 0], dtype="int64")

    # x[idx] for rank>1 (implicit full slice).
    y0 = x[idx]
    y1 = x[idx, :]

    x_np = _to_numpy_cpu(x)
    idx_np = _to_numpy_cpu(idx).astype(np.int64)

    np.testing.assert_allclose(_to_numpy_cpu(y0), x_np[idx_np])
    np.testing.assert_allclose(_to_numpy_cpu(y1), x_np[idx_np, :])


def test_advanced_index_read_2d_prefix_tensor_index_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x_cpu = vt.arange(6, dtype="float32").reshape((2, 3))
    idx_cpu = vt.tensor([0, 2], dtype="int64")

    y_cpu = x_cpu[:, idx_cpu]

    x_np = _to_numpy_cpu(x_cpu)
    idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)

    x_cuda = vt.cuda.to_device(x_np)
    idx_cuda = vt.cuda.to_device(idx_np)

    y_cuda = x_cuda[:, idx_cuda]
    y_cuda_np = vt.cuda.from_device(y_cuda)

    np.testing.assert_allclose(y_cuda_np, _to_numpy_cpu(y_cpu))


def test_advanced_index_read_2d_bool_mask_cpu():
    x = vt.arange(6, dtype="float32").reshape((2, 3))
    mask = vt.tensor([True, False, True], dtype="bool")

    y = x[:, mask]

    x_np = _to_numpy_cpu(x)
    mask_np = _to_numpy_cpu(mask).astype(bool)
    y_np = _to_numpy_cpu(y)

    np.testing.assert_allclose(y_np, x_np[:, mask_np])


def test_advanced_index_read_1d_tensor_index_cuda_matches_cpu():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # CPU baseline
    x_cpu = vt.arange(6, dtype="float32")
    idx_cpu = vt.tensor([0, 3, 5], dtype="int64")
    y_cpu = x_cpu[idx_cpu]

    x_np = _to_numpy_cpu(x_cpu)
    idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)

    # Move base and indices to CUDA via vt.cuda.to_device
    x_cuda = vt.cuda.to_device(x_np)
    idx_cuda = vt.cuda.to_device(idx_np)

    y_cuda = x_cuda[idx_cuda]
    y_cuda_np = vt.cuda.from_device(y_cuda)

    np.testing.assert_allclose(y_cuda_np, _to_numpy_cpu(y_cpu))


def test_advanced_index_read_feature_flag_disabled():
    x = vt.arange(4, dtype="float32")
    idx = vt.tensor([0, 2], dtype="int64")

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(False)
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x[idx]
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_advanced_index_read_invalid_index_dtype():
    x = vt.arange(4, dtype="float32")
    # Float32 index tensor is not on the allowlist and should be rejected at
    # the Python layer before reaching vt kernels.
    idx_bad = C._cpu_full([2], "float32", 0.0)

    with pytest.raises(
        ValueError,
        match="advanced index tensor must be int32, int64, or bool",
    ):
        _ = x[idx_bad]


def test_advanced_index_read_multi_tensor_indices_not_supported():
    x = vt.arange(9, dtype="float32").reshape((3, 3))
    idx0 = vt.tensor([0, 2], dtype="int64")
    idx1 = vt.tensor([0, 1], dtype="int64")

    with pytest.raises(
        NotImplementedError,
        match="multiple tensor/bool indices",
    ):
        _ = x[idx0, idx1]


def test_advanced_index_read_suffix_basic_indices_not_supported():
    x = vt.arange(6, dtype="float32").reshape((2, 3))
    idx = vt.tensor([0, 2], dtype="int64")

    with pytest.raises(
        NotImplementedError,
        match="suffix basic indices",
    ):
        _ = x[idx, 0]


def test_advanced_index_read_sequence_of_scalars_not_supported():
    x = vt.arange(4, dtype="float32")

    with pytest.raises(
        NotImplementedError,
        match="advanced indexing pattern is not supported",
    ):
        _ = x[[0, 1]]


def test_advanced_index_read_zero_dim_advanced_uses_core_error():
    # 0-d base tensor from vt unit helper.
    x0 = C.vt.unit()
    idx = vt.tensor([0], dtype="int64")

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)
        with pytest.raises(
            RuntimeError,
            match="advanced indexing is not supported for 0-d tensors",
        ):
            _ = x0[idx]
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_advanced_index_read_scalar_bool_under_implemented_vt():
    x = vt.arange(4, dtype="float32")

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)
        with pytest.raises(
            NotImplementedError,
            match="advanced indexing pattern is not supported",
        ):
            _ = x[True]
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_advanced_index_read_oob_uses_canonical_substring_cpu():
    x = vt.arange(4, dtype="float32")
    idx = vt.tensor([0, 4], dtype="int64")  # 4 is out of range for size 4

    with pytest.raises(
        IndexError,
        match="advanced indexing: index out of range for dimension with size",
    ):
        _ = x[idx]


def test_advanced_index_read_oob_uses_canonical_substring_cuda():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x_np = np.arange(4, dtype=np.float32)
    idx_np = np.array([0, 4], dtype=np.int64)

    x_cuda = vt.cuda.to_device(x_np)
    idx_cuda = vt.cuda.to_device(idx_np)

    with pytest.raises(
        IndexError,
        match="advanced indexing: index out of range for dimension with size",
    ):
        _ = x_cuda[idx_cuda]


def test_advanced_index_read_bool_mask_cuda_rejected_with_value_error():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    x_np = np.arange(6, dtype=np.float32).reshape((2, 3))
    mask_np = np.array([True, False, True], dtype=bool)

    x_cuda = vt.cuda.to_device(x_np)
    mask_cuda = vt.cuda.to_device(mask_np)

    with pytest.raises(
        ValueError,
        match="CUDA advanced indexing does not support boolean mask indices",
    ):
        _ = x_cuda[:, mask_cuda]
