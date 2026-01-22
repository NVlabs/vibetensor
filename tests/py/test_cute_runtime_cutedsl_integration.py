# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pytest

# Import vibetensor before numpy on some environments (libstdc++ ABI).
import vibetensor._C as C
import vibetensor.torch.cuda as vc

import numpy as np

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.base_dsl.jit_executor import get_escaped_cubin_bytes

from vibetensor.cute.runtime import CuteKernel, CuteKernelArtifact, CuteParamSpec, clear_cache


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


def _extract_kernels_binary(ir_module) -> bytes:
    # CuTeDSL's default pipeline (enable-cuda-dialect=true) lowers the GPU binary
    # into an LLVM global string called @kernels_binary.
    txt = str(ir_module)
    m = re.search(r'@kernels_binary\("([^"]*)"\)', txt)
    if not m:
        raise AssertionError("expected @kernels_binary(...) global in CuTeDSL IR module")
    return get_escaped_cubin_bytes(m.group(1).encode("utf-8"))


@cute.kernel
def _device_add_one(a: cute.Tensor, b: cute.Tensor):
    # Keep it trivial: with a static 1D contiguous tensor, CuTeDSL encodes
    # shape/stride in the kernel name, and the runtime ABI reduces to raw
    # device pointers.
    b[0] = a[0] + 1


@cute.jit
def _add_one(a: cute.Tensor, b: cute.Tensor):
    _device_add_one(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))


@pytest.mark.cuda
def test_cute_cutedsl_compile_and_vbt_launch_integration():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    clear_cache()

    a = vc.to_device(np.arange(4, dtype=np.float32), device=0)
    b = vc.to_device(np.zeros(4, dtype=np.float32), device=0)

    a_c = from_dlpack(a)
    b_c = from_dlpack(b)

    compiled = cute.compile(_add_one, a_c, b_c)
    assert compiled.kernel_info, "CuTeDSL compile produced no kernel_info"

    # CuTeDSL names kernels with fully-qualified types embedded.
    kernel = next(iter(compiled.kernel_info.keys()))
    fatbin = _extract_kernels_binary(compiled.ir_module)

    art = CuteKernelArtifact(
        cubin=fatbin,
        kernel=kernel,
        params=(
            CuteParamSpec("tensor_ptr"),
            CuteParamSpec("tensor_ptr"),
        ),
    )

    k = CuteKernel(art)
    k.launch(a, b, grid=(1, 1, 1), block=(1, 1, 1), record_stream=False)

    out = vc.from_device(b)
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert float(out[0]) == 1.0
    assert float(out[1]) == 0.0
    assert float(out[2]) == 0.0
    assert float(out[3]) == 0.0

    clear_cache()
