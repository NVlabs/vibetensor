# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from vibetensor import _C as C
import vibetensor.fabric as vf
from vibetensor.torch import cuda as vcuda


def _cuda_device_count() -> int:
    try:
        if not bool(getattr(C, "_has_cuda", False)):
            return 0
        return int(C._cuda_device_count())  # type: ignore[attr-defined]
    except Exception:
        return 0


_CUDA_DEVICE_COUNT = _cuda_device_count()
_HAS_TWO_CUDA_DEVICES = _CUDA_DEVICE_COUNT >= 2
_requires_two_cuda_devices = pytest.mark.skipif(
    not _HAS_TWO_CUDA_DEVICES,
    reason="FabricTensor multi-device tests require >=2 CUDA devices",
)

if _CUDA_DEVICE_COUNT < 1:
    pytest.skip("FabricTensor tests require >=1 CUDA device", allow_module_level=True)


@_requires_two_cuda_devices
def test_fabric_tensor_replicated_happy_path_and_reorders_shards() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a1 = vcuda.to_device(np.array([3.0, 4.0], dtype=np.float32), device=1)

    ft = vf.FabricTensor.replicated((0, 1), [a1, a0])

    assert ft.mesh.devices == (0, 1)
    assert ft.shards[0] is a0
    assert ft.shards[1] is a1

    assert ft.placement.kind == "replicated"
    assert ft.placement.global_shape == tuple(int(s) for s in a0.sizes)
    assert ft.placement.shard_offsets == (0, 0)


def test_fabric_tensor_replicated_errors_wrong_shard_count() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.replicated((0, 1), [a0])

    assert "[Fabric]" in str(excinfo.value)


@_requires_two_cuda_devices
def test_fabric_tensor_replicated_errors_dtype_mismatch() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a1 = vcuda.to_device(np.array([3, 4], dtype=np.int64), device=1)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.replicated((0, 1), [a0, a1])

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "dtype" in msg


@_requires_two_cuda_devices
def test_fabric_tensor_replicated_errors_size_mismatch() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a1 = vcuda.to_device(np.array([3.0, 4.0, 5.0], dtype=np.float32), device=1)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.replicated((0, 1), [a0, a1])

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "size" in msg


def test_fabric_tensor_replicated_errors_duplicate_device() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a0b = vcuda.to_device(np.array([3.0, 4.0], dtype=np.float32), device=0)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.replicated((0, 1), [a0, a0b])

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "device" in msg


@_requires_two_cuda_devices
def test_fabric_tensor_sharded_1d_row_happy_path_offsets_and_global_shape() -> None:
    s0 = vcuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3), device=0)
    s1 = vcuda.to_device(np.arange(3, dtype=np.float32).reshape(1, 3), device=1)

    ft = vf.FabricTensor.sharded_1d_row((0, 1), [s1, s0])

    assert ft.shards[0] is s0
    assert ft.shards[1] is s1

    assert ft.placement.kind == "sharded_1d_row"
    assert ft.placement.shard_offsets == (0, 2)
    assert ft.placement.global_shape == (3, 3)


@_requires_two_cuda_devices
def test_fabric_tensor_sharded_1d_row_errors_rank0_shard() -> None:
    s0 = vcuda.to_device(np.array(1.0, dtype=np.float32), device=0)
    s1 = vcuda.to_device(np.arange(3, dtype=np.float32).reshape(1, 3), device=1)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.sharded_1d_row((0, 1), [s0, s1])

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "rank" in msg


@_requires_two_cuda_devices
def test_fabric_tensor_sharded_1d_row_errors_tail_dim_mismatch() -> None:
    s0 = vcuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3), device=0)
    s1 = vcuda.to_device(np.arange(4, dtype=np.float32).reshape(1, 4), device=1)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.sharded_1d_row((0, 1), [s0, s1])

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "tail" in msg


@_requires_two_cuda_devices
def test_fabric_tensor_direct_constructor_requires_mesh_order() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a1 = vcuda.to_device(np.array([3.0, 4.0], dtype=np.float32), device=1)

    mesh = vf.FabricMesh((0, 1))
    placement = vf.FabricPlacement(
        kind="replicated",
        mesh=mesh,
        global_shape=tuple(int(s) for s in a0.sizes),
        shard_offsets=(0, 0),
    )

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor(placement=placement, shards=(a1, a0))

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "devices" in msg


def test_fabric_tensor_dlpack_rejected() -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)

    ft = vf.FabricTensor.replicated((0,), [a0])

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = np.from_dlpack(ft)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "DLPack is not supported for FabricTensor" in msg
    assert "to_local_shards" in msg


@_requires_two_cuda_devices
@pytest.mark.parametrize(
    "op,np_op",
    [
        (vf.add, lambda a, b: a + b),
        (vf.mul, lambda a, b: a * b),
    ],
)
def test_fabric_tensor_add_mul_replicated_delegates_per_shard(op, np_op) -> None:
    a0_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a1_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    b0_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    b1_np = np.array([40.0, 50.0, 60.0], dtype=np.float32)

    a0 = vcuda.to_device(a0_np, device=0)
    a1 = vcuda.to_device(a1_np, device=1)
    b0 = vcuda.to_device(b0_np, device=0)
    b1 = vcuda.to_device(b1_np, device=1)

    ft_a = vf.FabricTensor.replicated((0, 1), [a0, a1])
    ft_b = vf.FabricTensor.replicated((0, 1), [b0, b1])

    out = op(ft_a, ft_b)

    assert isinstance(out, vf.FabricTensor)
    assert out.mesh.devices == (0, 1)
    assert out.placement == ft_a.placement

    np.testing.assert_allclose(vcuda.from_device(out.shards[0]), np_op(a0_np, b0_np))
    np.testing.assert_allclose(vcuda.from_device(out.shards[1]), np_op(a1_np, b1_np))


@_requires_two_cuda_devices
@pytest.mark.parametrize("op", [vf.add, vf.mul])
def test_fabric_tensor_add_mul_rejects_primary_kwarg(op) -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a1 = vcuda.to_device(np.array([3.0, 4.0], dtype=np.float32), device=1)

    ft = vf.FabricTensor.replicated((0, 1), [a0, a1])

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = op(ft, ft, primary=0)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "primary" in msg
    assert "FabricTensor" in msg


@_requires_two_cuda_devices
@pytest.mark.parametrize("op", [vf.add, vf.mul])
def test_fabric_tensor_add_mul_rejects_mesh_mismatch(op) -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    a1 = vcuda.to_device(np.array([3.0, 4.0], dtype=np.float32), device=1)
    b0 = vcuda.to_device(np.array([10.0, 20.0], dtype=np.float32), device=0)
    b1 = vcuda.to_device(np.array([30.0, 40.0], dtype=np.float32), device=1)

    ft_a = vf.FabricTensor.replicated((0, 1), [a0, a1])
    ft_b = vf.FabricTensor.replicated((1, 0), [b0, b1])

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = op(ft_a, ft_b)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "mesh" in msg


@_requires_two_cuda_devices
@pytest.mark.parametrize("op", [vf.add, vf.mul])
def test_fabric_tensor_add_mul_rejects_placement_mismatch(op) -> None:
    r0 = vcuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3), device=0)
    r1 = vcuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3) + 10, device=1)
    s0 = vcuda.to_device(np.arange(6, dtype=np.float32).reshape(2, 3), device=0)
    s1 = vcuda.to_device(np.arange(3, dtype=np.float32).reshape(1, 3), device=1)

    ft_repl = vf.FabricTensor.replicated((0, 1), [r0, r1])
    ft_shard = vf.FabricTensor.sharded_1d_row((0, 1), [s0, s1])

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = op(ft_repl, ft_shard)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "placement" in msg


@pytest.mark.parametrize("op", [vf.add, vf.mul])
def test_fabric_tensor_add_mul_rejects_mixed_operands(op) -> None:
    a0 = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)

    ft = vf.FabricTensor.replicated((0,), [a0])

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = op(ft, a0)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "requires FabricTensor" in msg


@_requires_two_cuda_devices
@pytest.mark.parametrize(
    "op,np_op",
    [
        (vf.add, lambda a, b: a + b),
        (vf.mul, lambda a, b: a * b),
    ],
)
def test_fabric_tensor_add_mul_sharded_1d_row_delegates_per_shard(op, np_op) -> None:
    a0_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    a1_np = (np.arange(3, dtype=np.float32).reshape(1, 3) + 10)
    b0_np = np.ones((2, 3), dtype=np.float32)
    b1_np = np.ones((1, 3), dtype=np.float32) * 2

    a0 = vcuda.to_device(a0_np, device=0)
    a1 = vcuda.to_device(a1_np, device=1)
    b0 = vcuda.to_device(b0_np, device=0)
    b1 = vcuda.to_device(b1_np, device=1)

    ft_a = vf.FabricTensor.sharded_1d_row((0, 1), [a0, a1])
    ft_b = vf.FabricTensor.sharded_1d_row((0, 1), [b0, b1])

    out = op(ft_a, ft_b)

    assert isinstance(out, vf.FabricTensor)
    assert out.placement == ft_a.placement
    assert out.placement.kind == "sharded_1d_row"
    assert out.placement.shard_offsets == (0, 2)
    assert out.placement.global_shape == (3, 3)

    np.testing.assert_allclose(vcuda.from_device(out.shards[0]), np_op(a0_np, b0_np))
    np.testing.assert_allclose(vcuda.from_device(out.shards[1]), np_op(a1_np, b1_np))
