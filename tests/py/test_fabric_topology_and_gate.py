# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence, Tuple

import pytest

from vibetensor import _C as C
import vibetensor.fabric as vf


def _set_fake_topology(device_count: int, cliques: Sequence[Sequence[int]]) -> None:
    """Install a synthetic topology via the internal test hook.

    Parameters
    ----------
    device_count:
        Total number of CUDA devices in the synthetic view.
    cliques:
        Sequence of sequences of device indices; each inner sequence defines
        one clique. Devices not present in any clique receive clique_id -1.
    """

    def _builder(topo: "C._FabricTopology") -> None:  # type: ignore[name-defined]
        topo.device_count = int(device_count)

        topo.can_access_peer = [
            [False for _ in range(device_count)] for _ in range(device_count)
        ]
        topo.p2p_enabled = [
            [False for _ in range(device_count)] for _ in range(device_count)
        ]

        clique_id: List[int] = [-1 for _ in range(device_count)]
        clique_size: List[int] = []
        for cid, devs in enumerate(cliques):
            clique_size.append(len(devs))
            for d in devs:
                clique_id[int(d)] = cid

        topo.clique_id = clique_id
        topo.clique_size = clique_size

    C._fabric_set_fake_topology_builder_for_tests(_builder)
    C._fabric_reset_state_for_tests()


def _set_uva_forced(value: bool | None) -> None:
    C._fabric_set_forced_uva_ok_for_tests(value)
    C._fabric_reset_state_for_tests()


def test_fabric_import_and_surface() -> None:
    # Basic import should succeed on both CPU-only and CUDA builds.
    assert vf.FabricMode.DISABLED.value == "disabled"
    assert vf.FabricMode.BEST_EFFORT.value == "best_effort"
    assert vf.FabricMode.DRY_RUN.value == "dry_run"

    exported = set(vf.__all__)
    expected = {
        "FabricMode",
        "FabricClique",
        "TopologySnapshot",
        "inspect_topology",
        "enable",
        "disable",
        "is_enabled",
        "cliques",
    }
    assert expected.issubset(exported)

    snap = vf.inspect_topology()
    assert dataclasses.is_dataclass(snap)
    assert isinstance(snap.device_count, int)
    assert isinstance(snap.uva_ok, bool)


def test_fabric_topology_synthetic_0_and_1_gpu() -> None:
    # 0-GPU synthetic topology: NoCuda status and disabled gate.
    _set_fake_topology(0, [])
    snap0 = vf.inspect_topology()

    assert snap0.device_count == 0
    assert snap0.uva_ok is False
    assert snap0.init_status in {"no_cuda", "cuda_error"}
    assert isinstance(snap0.disable_reason, str)
    assert not vf.is_enabled()

    # Enabling or disabling Fabric in a 0-GPU synthetic topology must raise a
    # canonical NoCuda Fabric error.
    with pytest.raises(vf._FabricError) as excinfo0:  # type: ignore[attr-defined]
        vf.enable("best_effort")
    assert "Built without CUDA or no CUDA devices are available" in str(excinfo0.value)

    with pytest.raises(vf._FabricError) as excinfo1:  # type: ignore[attr-defined]
        vf.disable()
    assert "Built without CUDA or no CUDA devices are available" in str(excinfo1.value)

    # 1-GPU synthetic topology: UVA vacuously OK, but gate remains closed.
    _set_fake_topology(1, [[0]])
    snap1 = vf.inspect_topology()

    assert snap1.device_count == 1
    assert snap1.uva_ok is True
    assert snap1.init_status == "ok"

    # Enabling in best_effort or dry_run mode must not raise, but
    # is_enabled() stays False because there is no clique of size >= 2.
    vf.enable("best_effort")
    assert vf.is_enabled() is False

    vf.enable("dry_run")
    assert vf.is_enabled() is False

    vf.disable()
    assert vf.is_enabled() is False


def test_fabric_uva_failure_path() -> None:
    # Two-GPU clique with forced UVA failure.
    _set_fake_topology(2, [[0, 1]])
    C._fabric_set_forced_uva_ok_for_tests(False)
    C._fabric_reset_state_for_tests()

    snap = vf.inspect_topology()
    assert snap.device_count == 2
    assert snap.uva_ok is False
    assert snap.init_status == "uva_failed"
    assert "[Fabric] UVA invariant violated on this platform; Fabric is disabled" in snap.disable_reason

    # Any attempt to enable or disable Fabric now raises _FabricError with
    # the canonical UVA-disabled substring.
    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        vf.enable("best_effort")
    assert "UVA invariant violated" in str(excinfo.value)

    with pytest.raises(vf._FabricError) as excinfo2:  # type: ignore[attr-defined]
        vf.disable()
    assert "UVA invariant violated" in str(excinfo2.value)

    assert vf.is_enabled() is False
    assert C._fabric_is_enabled_for_ops() is False
    assert C._fabric_get_mode() == C._FabricMode.disabled


def test_fabric_enable_disable_mode_semantics() -> None:
    # Two-GPU topology with UVA OK and clique of size >= 2.
    _set_fake_topology(2, [[0, 1]])
    C._fabric_set_forced_uva_ok_for_tests(True)
    C._fabric_reset_state_for_tests()

    snap = vf.inspect_topology()
    assert snap.device_count == 2
    assert snap.uva_ok is True
    assert snap.init_status == "ok"

    # Enabling best_effort is idempotent.
    vf.enable("best_effort")
    vf.enable("best_effort")
    assert vf.is_enabled() is True

    # Switching modes and disabling is also idempotent.
    vf.enable("dry_run")
    assert vf.is_enabled() is True

    vf.disable()
    vf.disable()
    assert vf.is_enabled() is False

    # Invalid mode string raises a clear error.
    with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
        vf.enable("bogus")  # type: ignore[arg-type]
    assert "Invalid Fabric mode" in str(excinfo.value)


def test_fabric_gate_consistency_with_c_binding() -> None:
    # Simple smoke test to ensure Python is_enabled mirrors the C++ gate.
    _set_fake_topology(2, [[0, 1]])
    C._fabric_set_forced_uva_ok_for_tests(True)
    C._fabric_reset_state_for_tests()

    vf.enable("best_effort")
    assert vf.is_enabled() == bool(C._fabric_is_enabled_for_ops())

    vf.disable()
    assert vf.is_enabled() == bool(C._fabric_is_enabled_for_ops())


def test_fabric_inspect_topology_is_pure() -> None:
    # Multiple calls to inspect_topology() must be side-effect free and return
    # equal snapshots.
    _set_fake_topology(2, [[0, 1]])
    C._fabric_set_forced_uva_ok_for_tests(True)
    C._fabric_reset_state_for_tests()

    snap1 = vf.inspect_topology()
    enabled1 = vf.is_enabled()
    gate1 = bool(C._fabric_is_enabled_for_ops())

    snap2 = vf.inspect_topology()
    enabled2 = vf.is_enabled()
    gate2 = bool(C._fabric_is_enabled_for_ops())

    assert snap1 == snap2
    assert enabled1 == enabled2
    assert gate1 == gate2


def teardown_module(module: object) -> None:  # pragma: no cover
    """Clear Fabric test hooks after this module's tests.

    The Python bindings for `_fabric_set_fake_topology_builder_for_tests`
    capture the synthetic-topology builder as a `nanobind::object` inside a
    `std::function` stored in the C++ `FabricTestHooks` singleton. If that
    lambda remains installed when the Python interpreter shuts down, the
    `nanobind::object` destructor can run after `Py_Finalize`, leading to a
    use-after-finalize crash during process teardown.

    Resetting the hooks to `None` at module teardown ensures that the
    `std::function` no longer holds any Python objects by the time global
    C++ static destructors run.
    """

    del module  # unused
    C._fabric_clear_test_hooks_for_tests()
    C._fabric_reset_state_for_tests()
