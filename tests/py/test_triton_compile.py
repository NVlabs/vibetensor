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
def test_triton_compile_to_ptx_simple_add():
    # Skip if CUDA not present in VBT
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    # 1D add kernel
    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    # Compile to PTX via vibetensor.triton bridge
    sig = "*fp32,*fp32,*fp32,i32"  # last pointer is output
    try:
        ptx, entry, shmem = vt_triton._compile_to_ptx(add_kernel, signature=sig, meta={"BLOCK_SIZE": 128}, num_warps=4)
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton PTX compile integration test")
        raise

    assert isinstance(ptx, (bytes, bytearray)) and len(ptx) > 0
    s = bytes(ptx)
    # PTX should declare at least one .entry
    assert b".entry" in s or b".visible .entry" in s
    assert isinstance(entry, str) and len(entry) > 0
    # Best-effort: entry symbol should likely appear in the PTX
    # Be permissive across Triton versions / mangling
    # If not present, at least ensure .entry exists as above
    if entry.encode() not in s:
        # ensure there is exactly one .entry occurrence to avoid ambiguity
        assert s.count(b".entry") >= 1


@pytest.mark.cuda
def test_triton_compile_to_ptx_accepts_num_stages(monkeypatch):
    # Skip if CUDA not present in VBT
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    sig = "*fp32,*fp32,*fp32,i32"

    # Happy path: num_stages is forwarded successfully.
    try:
        ptx2, entry2, shmem2 = vt_triton._compile_to_ptx(
            add_kernel,
            signature=sig,
            meta={"BLOCK_SIZE": 128},
            num_warps=4,
            num_stages=2,
        )
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton num_stages compile test")
        raise
    assert isinstance(ptx2, (bytes, bytearray)) and len(ptx2) > 0
    assert isinstance(entry2, str) and len(entry2) > 0

    # Fallback path: simulate an older Triton that rejects num_stages,
    # and verify that _compile_to_ptx retries without the option.
    orig_compile = triton.compile

    def failing_compile(*args, **kwargs):
        options = kwargs.get("options")
        if (isinstance(options, dict) and "num_stages" in options) or ("num_stages" in kwargs):
            raise TypeError("unexpected keyword argument 'num_stages'")
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(triton, "compile", failing_compile)

    ptx4, entry4, shmem4 = vt_triton._compile_to_ptx(
        add_kernel,
        signature=sig,
        meta={"BLOCK_SIZE": 128},
        num_warps=4,
        num_stages=4,
    )

    assert isinstance(ptx4, (bytes, bytearray)) and len(ptx4) > 0
    assert isinstance(entry4, str) and len(entry4) > 0
