# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    from triton.runtime.errors import PTXASError  # type: ignore
except Exception:
    pytest.skip("triton not available at runtime", allow_module_level=True)

from vibetensor import _C as C
import vibetensor.triton as vt_triton
import vibetensor.torch as vt  # noqa: F401


def _skip_if_triton_ptxas_arch_mismatch(exc: Exception) -> None:
    msg = str(exc)
    if "PTX .version" in msg and "does not support .target" in msg:
        pytest.skip(
            "Triton/PTXAS does not support the current GPU architecture for the "
            "PTX version used in these tests; skipping Triton grid precedence tests.",
        )

@pytest.mark.cuda
def test_triton_grid_tuple_precedence_over_meta_and_env(monkeypatch):
    # Skip if CUDA not present in VBT
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def simple_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):  # noqa: D401
        """1D kernel; body is irrelevant for this test."""
        _ = tl.program_id(0)  # pragma: no cover

    sig = "*fp32"
    meta = {"BLOCK_SIZE": 128}

    captured = {}

    def fake_launch(func_handle, grid, block, shared_mem, stream, argv):  # noqa: D401
        """Capture grid chosen by register()."""
        captured["grid"] = tuple(grid)
        captured["block"] = tuple(block)
        return None

    monkeypatch.setattr(C, "_cuda_launch", fake_launch)
    # Even in pytorch orientation mode, explicit grid must not be rewritten.
    monkeypatch.setenv("VBT_TRITON_GRID_ORIENTATION", "pytorch")

    schema = "vt::grid_tuple_precedence(Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]

    explicit_grid = (3, 5)

    vt_triton.register(
        "vt::grid_tuple_precedence",
        simple_kernel,
        signature=sig,
        meta=meta,
        num_warps=4,
        grid=explicit_grid,
    )

    a = C._make_cuda_tensor([1024], "float32", 0.0)  # type: ignore[attr-defined]
    try:
        out = C._call_op("vt::grid_tuple_precedence", a)  # type: ignore[attr-defined]
    except PTXASError as exc:  # type: ignore[misc]
        _skip_if_triton_ptxas_arch_mismatch(exc)
        raise
    assert tuple(out.sizes) == (1024,)

    assert captured["grid"] == (explicit_grid[0], explicit_grid[1], 1)


@pytest.mark.cuda
def test_triton_grid_callable_meta_arity(monkeypatch):
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def simple_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):  # noqa: D401
        _ = tl.program_id(0)  # pragma: no cover

    sig = "*fp32"
    meta = {"BLOCK_SIZE": 128}

    captured = {}

    def fake_launch(func_handle, grid, block, shared_mem, stream, argv):
        captured["grid"] = tuple(grid)
        return None

    monkeypatch.setattr(C, "_cuda_launch", fake_launch)

    schema = "vt::grid_callable_meta(Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]

    def grid_from_meta(meta_dict):  # type: ignore[override]
        # meta-only callable; register() should detect the 1-arg arity.
        blocks = (meta_dict["BLOCK_SIZE"] + 255) // 256
        return (blocks,)

    vt_triton.register(
        "vt::grid_callable_meta",
        simple_kernel,
        signature=sig,
        meta=meta,
        num_warps=4,
        grid=grid_from_meta,
    )

    a = C._make_cuda_tensor([1024], "float32", 0.0)  # type: ignore[attr-defined]
    try:
        C._call_op("vt::grid_callable_meta", a)  # type: ignore[attr-defined]
    except PTXASError as exc:  # type: ignore[misc]
        _skip_if_triton_ptxas_arch_mismatch(exc)
        raise

    # grid callable returns rank-1; register() should pad to 3D.
    assert len(captured["grid"]) == 3
    assert captured["grid"][1] == 1 and captured["grid"][2] == 1


@pytest.mark.cuda
def test_triton_grid_param_beats_grid_fn(monkeypatch):
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def simple_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):  # noqa: D401
        _ = tl.program_id(0)  # pragma: no cover

    sig = "*fp32"
    meta = {"BLOCK_SIZE": 128}

    captured = {}

    def fake_launch(func_handle, grid, block, shared_mem, stream, argv):
        captured["grid"] = tuple(grid)
        return None

    monkeypatch.setattr(C, "_cuda_launch", fake_launch)

    schema = "vt::grid_param_overrides_grid_fn(Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]

    def grid_fn(state, inputs, meta_dict):  # pragma: no cover
        # Deliberately different from explicit grid to prove precedence.
        return (1, 1, 1)

    explicit_grid = (7,)

    vt_triton.register(
        "vt::grid_param_overrides_grid_fn",
        simple_kernel,
        signature=sig,
        grid_fn=grid_fn,
        meta=meta,
        num_warps=4,
        grid=explicit_grid,
    )

    a = C._make_cuda_tensor([1024], "float32", 0.0)  # type: ignore[attr-defined]
    try:
        C._call_op("vt::grid_param_overrides_grid_fn", a)  # type: ignore[attr-defined]
    except PTXASError as exc:  # type: ignore[misc]
        _skip_if_triton_ptxas_arch_mismatch(exc)
        raise

    # grid parameter must win over grid_fn.
    assert captured["grid"] == (explicit_grid[0], 1, 1)
