# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C
from vibetensor.torch import ops


if not hasattr(C, "autograd"):
    pytest.skip("autograd disabled in this build", allow_module_level=True)


def _to_numpy_cpu(t):
    return np.from_dlpack(t).reshape(tuple(int(s) for s in t.sizes))


def _cuda_only() -> bool:
    return bool(getattr(C, "_has_cuda", False)) and int(getattr(C, "_cuda_device_count", lambda: 0)()) > 0


def test_advanced_index_read_backward_cpu_duplicates_accumulate():
    x = vt.arange(6, dtype="float32")
    x.requires_grad = True

    idx = vt.tensor([1, 1, 3, 2, 1], dtype="int64")

    y = x[idx]
    z = C.vt.mul(y, vt.ones_like(y))
    z.backward(vt.ones_like(z))

    assert x.grad is not None
    np.testing.assert_allclose(
        _to_numpy_cpu(x.grad),
        np.array([0.0, 3.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_advanced_index_read_backward_suffix_full_slice_cpu_scatter_rows():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
        x.requires_grad = True

        idx = vt.tensor([1, 1, 3], dtype="int64")

        y = x[idx]
        z = C.vt.mul(y, vt.ones_like(y))
        z.backward(vt.ones_like(z))

        assert x.grad is not None
        np.testing.assert_allclose(
            _to_numpy_cpu(x.grad),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_advanced_index_read_backward_prefix_grad_scatter_cpu():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(6, dtype="float32").reshape((2, 3)).detach()
        x.requires_grad = True

        idx = vt.tensor([0, 2], dtype="int64")

        y = x[:, idx]
        z = C.vt.mul(y, vt.ones_like(y))
        z.backward(vt.ones_like(z))

        assert x.grad is not None
        assert tuple(x.grad.sizes) == tuple(x.sizes)
        np.testing.assert_allclose(
            _to_numpy_cpu(x.grad),
            np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        )
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_advanced_index_read_prefix_errors_when_v2_disabled_cpu():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(False)  # type: ignore[attr-defined]
    try:
        ag = C.autograd
        prev_grad = bool(ag.is_grad_enabled())
        try:
            ag.set_grad_enabled(True)

            x = vt.arange(6, dtype="float32").reshape((2, 3)).detach()
            x.requires_grad = True
            idx = vt.tensor([0, 2], dtype="int64")

            with pytest.raises(
                ValueError,
                match="vt::index: prefix meta requires VBT_AUTOGRAD_INDEXING_V2",
            ):
                _ = x[:, idx]
        finally:
            ag.set_grad_enabled(prev_grad)
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_index_backward_base_mutation_errors_when_v2_disabled_cpu():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(False)  # type: ignore[attr-defined]
    try:
        x = vt.arange(6, dtype="float32")
        x.requires_grad = True

        idx = vt.tensor([1, 3], dtype="int64")
        y = x[idx]

        # Mutate base tensor after forward; legacy v1 behavior saves base and errors.
        ag = C.autograd
        with ag.no_grad():  # type: ignore[attr-defined]
            x[0] = 123.0

        with pytest.raises(RuntimeError, match="version mismatch"):
            y.backward(vt.ones_like(y))
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_index_backward_base_mutation_allowed_when_v2_enabled_cpu():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(6, dtype="float32")
        x.requires_grad = True

        idx = vt.tensor([1, 3], dtype="int64")
        y = x[idx]

        # Mutate base tensor after forward; v2 should not save base.
        ag = C.autograd
        with ag.no_grad():  # type: ignore[attr-defined]
            x[0] = 123.0

        y.backward(vt.ones_like(y))

        assert x.grad is not None
        np.testing.assert_allclose(
            _to_numpy_cpu(x.grad),
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


def test_vt_index_prefix_meta_backward_scatter_adds_cpu():
    prev = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]
    try:
        x = vt.arange(6, dtype="float32").reshape((2, 3)).detach()
        x.requires_grad = True

        idx = vt.tensor([0, 2], dtype="int64")
        sent = np.iinfo(np.int64).min
        meta = vt.tensor(
            np.array([0, 1, 0, 1, 2, sent, sent, sent], dtype=np.int64),
            dtype="int64",
        )

        y = ops.vt.index(x, idx, meta)
        z = C.vt.mul(y, vt.ones_like(y))
        z.backward(vt.ones_like(z))

        assert x.grad is not None
        np.testing.assert_allclose(
            _to_numpy_cpu(x.grad),
            np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        )
    finally:
        C._set_autograd_indexing_v2_enabled_for_tests(prev)  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_advanced_index_read_backward_cuda_scatter_add_shape_and_values():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(6, dtype=np.float32)
        idx_np = np.array([1, 1, 3, 2, 1], dtype=np.int64)

        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)
        idx = vt.cuda.to_device(idx_np)

        y = x[idx]
        ones = vt.cuda.to_device(np.ones(idx_np.shape[0], dtype=np.float32))

        z = C.vt.mul(y, ones)
        z.backward(ones)

        assert x.grad is not None
        assert tuple(x.grad.sizes) == (6,)

        g = vt.cuda.from_device(x.grad())
        np.testing.assert_allclose(
            g,
            np.array([0.0, 3.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)


@pytest.mark.cuda
def test_advanced_index_read_backward_cuda_allows_cpu_index_tensor():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    prev_v2 = C._autograd_indexing_v2_enabled()  # type: ignore[attr-defined]
    C._set_autograd_indexing_v2_enabled_for_tests(True)  # type: ignore[attr-defined]

    ag = C.autograd
    prev_cuda = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    prev_grad = bool(ag.is_grad_enabled())
    try:
        ag.set_grad_enabled(True)
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        x0_np = np.arange(6, dtype=np.float32)
        idx_np = np.array([1, 1, 3, 2, 1], dtype=np.int64)

        x = vt.cuda.to_device(x0_np)
        x.set_requires_grad(True)

        idx_cpu = vt.tensor(idx_np.tolist(), dtype="int64")

        y = x[idx_cpu]
        ones = vt.cuda.to_device(np.ones(idx_np.shape[0], dtype=np.float32))

        z = C.vt.mul(y, ones)
        z.backward(ones)

        assert x.grad is not None
        g = vt.cuda.from_device(x.grad())
        np.testing.assert_allclose(
            g,
            np.array([0.0, 3.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
    finally:
        ag.set_cuda_autograd_enabled(prev_cuda)  # type: ignore[attr-defined]
        ag.set_grad_enabled(prev_grad)
        C._set_autograd_indexing_v2_enabled_for_tests(prev_v2)  # type: ignore[attr-defined]
