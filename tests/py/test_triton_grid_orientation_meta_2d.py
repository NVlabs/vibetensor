# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:
    pytest.skip("triton not available at runtime", allow_module_level=True)

from vibetensor import _C as C
import vibetensor.triton as vt_triton
import vibetensor.torch as vt  # noqa: F401


def _skip_if_triton_arch_unsupported(exc: Exception, what: str) -> None:
    msg = str(exc)
    if "gpu-name" in msg and "is not defined for option 'gpu-name'" in msg:
        pytest.skip(
            "Triton/ptxas does not support the current GPU architecture; "
            f"skipping {what}",
            allow_module_level=False,
        )
    if "PTX .version" in msg and "does not support .target" in msg:
        pytest.skip(
            "Triton/ptxas PTX version does not support the current GPU architecture; "
            f"skipping {what}",
            allow_module_level=False,
        )


@pytest.mark.cuda
def test_triton_grid_orientation_meta_2d_legacy(monkeypatch):
    # Skip if CUDA not present in VBT
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def grid_kernel(out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):  # noqa: D401
        """Minimal 2D kernel; body is irrelevant for this test."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # Touch program ids so the kernel is not completely empty.
        _ = pid_m + pid_n  # pragma: no cover

    sig = "*fp32"
    meta = {"BLOCK_M": 32, "BLOCK_N": 64}

    captured = {}

    def fake_launch(func_handle, grid, block, shared_mem, stream, argv):  # noqa: D401
        """Capture grid and avoid touching the real CUDA driver."""
        captured["grid"] = tuple(grid)
        captured["block"] = tuple(block)
        return None

    monkeypatch.setattr(C, "_cuda_launch", fake_launch)
    monkeypatch.setenv("VBT_TRITON_GRID_ORIENTATION", "legacy")

    schema = "vt::grid2d_legacy(Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]

    M, N = 128, 256

    def out_shape_fn(state, inputs, meta_dict):  # noqa: D401
        """Return a fixed 2D output shape (M, N)."""
        return (M, N)

    vt_triton.register(
        "vt::grid2d_legacy",
        grid_kernel,
        signature=sig,
        meta=meta,
        num_warps=4,
        out_shape_fn=out_shape_fn,
    )

    # Input tensor only supplies device/dtype metadata; kernel ignores it.
    a = C._make_cuda_tensor([M, N], "float32", 0.0)  # type: ignore[attr-defined]

    try:
        out = C._call_op("vt::grid2d_legacy", a)  # type: ignore[attr-defined]
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton grid-orientation legacy test")
        raise
    assert tuple(out.sizes) == (M, N)

    bm = int(meta["BLOCK_M"])
    bn = int(meta["BLOCK_N"])
    expected_x = (N + bn - 1) // bn
    expected_y = (M + bm - 1) // bm
    assert captured["grid"] == (expected_x, expected_y, 1)


@pytest.mark.cuda
def test_triton_grid_orientation_meta_2d_pytorch(monkeypatch):
    # Skip if CUDA not present in VBT
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def grid_kernel(out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):  # noqa: D401
        """Minimal 2D kernel; body is irrelevant for this test."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        _ = pid_m + pid_n  # pragma: no cover

    sig = "*fp32"
    meta = {"BLOCK_M": 32, "BLOCK_N": 64}

    captured = {}

    def fake_launch(func_handle, grid, block, shared_mem, stream, argv):  # noqa: D401
        """Capture grid and avoid touching the real CUDA driver."""
        captured["grid"] = tuple(grid)
        captured["block"] = tuple(block)
        return None

    monkeypatch.setattr(C, "_cuda_launch", fake_launch)
    monkeypatch.setenv("VBT_TRITON_GRID_ORIENTATION", "pytorch")

    schema = "vt::grid2d_pytorch(Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]

    M, N = 128, 256

    def out_shape_fn(state, inputs, meta_dict):  # noqa: D401
        """Return a fixed 2D output shape (M, N)."""
        return (M, N)

    vt_triton.register(
        "vt::grid2d_pytorch",
        grid_kernel,
        signature=sig,
        meta=meta,
        num_warps=4,
        out_shape_fn=out_shape_fn,
    )

    a = C._make_cuda_tensor([M, N], "float32", 0.0)  # type: ignore[attr-defined]

    try:
        out = C._call_op("vt::grid2d_pytorch", a)  # type: ignore[attr-defined]
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton grid-orientation pytorch test")
        raise
    assert tuple(out.sizes) == (M, N)

    bm = int(meta["BLOCK_M"])
    bn = int(meta["BLOCK_N"])
    expected_x = (M + bm - 1) // bm
    expected_y = (N + bn - 1) // bn
    assert captured["grid"] == (expected_x, expected_y, 1)


@pytest.mark.cuda
def test_triton_grid_orientation_meta_2d_infer_failure(monkeypatch):
    """When neither out_shape_fn nor inputs expose 2D shape, raise ValueError."""
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def scalar_kernel(out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):  # noqa: D401
        """Scalar-style kernel; never actually launched in this test."""
        _ = tl.program_id(0)  # pragma: no cover

    sig = "*fp32"
    meta = {"BLOCK_M": 16, "BLOCK_N": 16}

    # Stub launch so no real kernel executes if we regress.
    def fake_launch(func_handle, grid, block, shared_mem, stream, argv):
        raise AssertionError("_cuda_launch should not be reached when grid inference fails")

    monkeypatch.setattr(C, "_cuda_launch", fake_launch)
    monkeypatch.setenv("VBT_TRITON_GRID_ORIENTATION", "pytorch")

    schema = "vt::grid2d_infer_failure(Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]

    # Register without out_shape_fn so 2D meta inference must consult inputs.
    vt_triton.register(
        "vt::grid2d_infer_failure",
        scalar_kernel,
        signature=sig,
        meta=meta,
        num_warps=4,
    )

    # Create a rank-0 (scalar) tensor; no 2D shape is available.
    scalar = C._cuda_empty([], "float32", 0)  # type: ignore[attr-defined]

    try:
        with pytest.raises(ValueError) as excinfo:
            C._call_op("vt::grid2d_infer_failure", scalar)  # type: ignore[attr-defined]
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton grid-orientation infer-failure test")
        raise

    msg = str(excinfo.value)
    assert "cannot infer 2D grid from inputs" in msg
