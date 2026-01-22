# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.torch as vbt


def _cuda_only() -> bool:
    return (
        getattr(C, "_has_cuda", False)
        and vbt.cuda.is_available()
        and hasattr(C, "_cuda_device_count")
        and C._cuda_device_count() > 0
    )


def test_tensor_item_raises_on_non_scalar_cpu_tensor():
    t = vbt.ones((2,), dtype=vbt.float32)
    with pytest.raises(
        RuntimeError,
        match=r"Tensor\.item\(\) is only supported for tensors with a single element in VibeTensor",
    ):
        _ = t.item()


@pytest.mark.skipif(not getattr(C, "_has_dlpack_bf16", False), reason="BF16 not supported by DLPack headers")
def test_tensor_item_bfloat16_cpu_falls_back_to_torch():
    torch = pytest.importorskip("torch")

    t_ref = torch.tensor([3.0], dtype=torch.bfloat16)
    cap = torch.utils.dlpack.to_dlpack(t_ref)
    t = vbt.from_dlpack(cap)

    assert t.dtype == "bfloat16"
    assert float(t.item()) == pytest.approx(3.0)


@pytest.mark.cuda
@pytest.mark.skipif(not _cuda_only(), reason="CUDA not available")
@pytest.mark.skipif(not getattr(C, "_has_dlpack_bf16", False), reason="BF16 not supported by DLPack headers")
def test_tensor_item_bfloat16_cuda_falls_back_to_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")

    t_ref = torch.tensor([3.0], device="cuda", dtype=torch.bfloat16)
    cap = torch.utils.dlpack.to_dlpack(t_ref)
    t = vbt.from_dlpack(cap)

    assert t.device[0] == 2
    assert t.dtype == "bfloat16"
    assert float(t.item()) == pytest.approx(3.0)
