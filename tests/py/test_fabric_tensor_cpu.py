# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from vibetensor import _C as C
import vibetensor.fabric as vf


def test_fabric_tensor_exports_present() -> None:
    exported = set(vf.__all__)
    assert {"FabricMesh", "FabricPlacement", "FabricDevice", "FabricTensor"}.issubset(
        exported
    )


def test_fabric_tensor_rejects_cpu_shard_inputs() -> None:
    t = C._cpu_full([2], "float32", 0.0)

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricTensor.replicated((0,), [t])

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "CUDA" in msg


def test_fabric_mesh_validation() -> None:
    with pytest.raises(vf._FabricError) as excinfo0:  # type: ignore[attr-defined]
        _ = vf.FabricMesh(())
    assert "[Fabric]" in str(excinfo0.value)

    with pytest.raises(vf._FabricError) as excinfo1:  # type: ignore[attr-defined]
        _ = vf.FabricMesh((True,))
    assert "[Fabric]" in str(excinfo1.value)

    with pytest.raises(vf._FabricError) as excinfo2:  # type: ignore[attr-defined]
        _ = vf.FabricMesh((-1,))
    assert "[Fabric]" in str(excinfo2.value)

    with pytest.raises(vf._FabricError) as excinfo3:  # type: ignore[attr-defined]
        _ = vf.FabricMesh((0, 0))
    assert "[Fabric]" in str(excinfo3.value)

    m = vf.FabricMesh([0, 1])
    assert m.devices == (0, 1)


def test_fabric_placement_validation() -> None:
    mesh = vf.FabricMesh((0, 1))

    p = vf.FabricPlacement(
        kind="replicated",
        mesh=mesh,
        global_shape=(2, 3),
        shard_offsets=(0, 0),
    )
    assert p.kind == "replicated"

    with pytest.raises(vf._FabricError) as excinfo0:  # type: ignore[attr-defined]
        _ = vf.FabricPlacement(
            kind="replicated",
            mesh=mesh,
            global_shape=(2, 3),
            shard_offsets=(0, 1),
        )
    assert "[Fabric]" in str(excinfo0.value)

    p2 = vf.FabricPlacement(
        kind="sharded_1d_row",
        mesh=mesh,
        global_shape=(3, 3),
        shard_offsets=(0, 2),
    )
    assert p2.kind == "sharded_1d_row"

    with pytest.raises(vf._FabricError) as excinfo1:  # type: ignore[attr-defined]
        _ = vf.FabricPlacement(
            kind="sharded_1d_row",
            mesh=mesh,
            global_shape=(),
            shard_offsets=(0, 2),
        )
    assert "[Fabric]" in str(excinfo1.value)

    with pytest.raises(vf._FabricError) as excinfo2:  # type: ignore[attr-defined]
        _ = vf.FabricPlacement(
            kind="sharded_1d_row",
            mesh=mesh,
            global_shape=(3, 3),
            shard_offsets=(1, 2),
        )
    assert "[Fabric]" in str(excinfo2.value)

    with pytest.raises(vf._FabricError) as excinfo3:  # type: ignore[attr-defined]
        _ = vf.FabricPlacement(
            kind="sharded_1d_row",
            mesh=mesh,
            global_shape=(3, 3),
            shard_offsets=(0, 2, 3),
        )
    assert "[Fabric]" in str(excinfo3.value)

    with pytest.raises(vf._FabricError) as excinfo4:  # type: ignore[attr-defined]
        _ = vf.FabricPlacement(
            kind="sharded_1d_row",
            mesh=mesh,
            global_shape=(3, 3),
            shard_offsets=(0, -1),
        )
    assert "[Fabric]" in str(excinfo4.value)


def test_fabric_placement_validation_rejects_invalid_kind() -> None:
    mesh = vf.FabricMesh((0, 1))
    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vf.FabricPlacement(
            kind="invalid_kind",  # type: ignore[arg-type]
            mesh=mesh,
            global_shape=(2,),
            shard_offsets=(0, 0),
        )
    assert "[Fabric]" in str(excinfo.value)


def test_torch_to_dlpack_rejects_fabric_tensor_marker() -> None:
    import vibetensor.torch as vt

    class FakeFabric:
        __vbt_fabric_tensor__ = True

        def __getattribute__(self, name: str):
            raise AssertionError("instance attribute access should not be required")

    fake = FakeFabric()

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vt.to_dlpack(fake)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "DLPack is not supported" in msg
    assert "FabricTensor" in msg
    assert "to_local_shards" in msg


def test_torch_ops_rejects_fabric_tensor_marker_even_under_compat(monkeypatch) -> None:
    monkeypatch.setenv("VBT_OPS_COMPAT", "1")

    import vibetensor.torch as vt

    class FakeFabric:
        __vbt_fabric_tensor__ = True

        def __getattribute__(self, name: str):
            raise AssertionError("instance attribute access should not be required")

    fake = FakeFabric()

    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        _ = vt.ops.vt.add(fake, fake)

    msg = str(excinfo.value)
    assert "[Fabric]" in msg
    assert "vibetensor.torch.ops" in msg
    assert "vt::add" in msg
    assert "to_local_shards" in msg
