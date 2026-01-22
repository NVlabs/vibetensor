# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pytest

from vibetensor import _C as C
import vibetensor.fabric as vf
from vibetensor.torch import cuda as vcuda
import vibetensor.autograd as ag


def _has_two_cuda_devices() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) >= 2  # type: ignore[attr-defined]
    except Exception:
        return False


if not _has_two_cuda_devices():
    pytest.skip("Fabric add/mul tests require >=2 CUDA devices", allow_module_level=True)


def _set_fake_topology(
    device_count: int,
    cliques: Sequence[Sequence[int]],
    *,
    can_access_pairs: Iterable[Tuple[int, int]] = (),
) -> None:
    """Install a synthetic topology via internal test hooks."""

    pairs = set((int(a), int(b)) for a, b in can_access_pairs)

    def _builder(topo: "C._FabricTopology") -> None:  # type: ignore[name-defined]
        topo.device_count = int(device_count)

        topo.can_access_peer = [
            [False for _ in range(device_count)] for _ in range(device_count)
        ]
        for i, j in pairs:
            topo.can_access_peer[i][j] = True

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
    C._fabric_set_forced_uva_ok_for_tests(True)
    C._fabric_reset_state_for_tests()


def _reset_stats() -> None:
    C._fabric_reset_stats_for_tests()  # type: ignore[attr-defined]


def _sum_per_device(s: vf.FabricStatsSnapshot, attr: str) -> int:
    return sum(int(getattr(d, attr)) for d in s.per_device)


def _assert_aggregation_invariants(s: vf.FabricStatsSnapshot) -> None:
    assert s.fabric_ops_attempted == _sum_per_device(s, "ops_as_primary")
    assert s.fabric_ops_attempted == _sum_per_device(s, "ops_as_remote")
    assert s.remote_bytes_read == _sum_per_device(s, "remote_bytes_read")
    assert s.remote_bytes_written == _sum_per_device(s, "remote_bytes_written")
    for d in s.per_device:
        assert d.remote_bytes_read == d.remote_bytes_written


def test_fabric_ops_exported_and_stats_smoke() -> None:
    exported = set(vf.__all__)
    assert {"add", "mul", "stats"}.issubset(exported)

    topo = vf.inspect_topology()

    _reset_stats()
    s = vf.stats()
    assert s.fabric_ops_attempted == 0
    assert s.fabric_ops_hit == 0
    assert s.fabric_ops_fallback == 0

    assert s.reasons.no_p2p == 0
    assert s.reasons.requires_grad == 0
    assert s.reasons.in_backward == 0
    assert s.reasons.small_tensor == 0

    assert s.inflight_ops_current == 0
    assert s.inflight_ops_peak == 0

    assert s.event_queue_len_peak == 0
    assert s.event_dropped_total == 0
    assert s.event_failures_total == 0

    assert s.mode_enable_calls == 0
    assert s.mode_disable_calls == 0
    assert s.mode_set_failures == 0

    expected_len = min(int(topo.device_count), 64)
    assert len(s.per_device) == expected_len
    for i, ds in enumerate(s.per_device):
        assert ds.device_index == i
        assert ds.ops_as_primary == 0
        assert ds.ops_as_remote == 0
        assert ds.remote_bytes_read == 0
        assert ds.remote_bytes_written == 0


def test_single_device_does_not_count_attempts() -> None:
    # In single-device mode, Fabric add/mul should behave like vt.add/vt.mul
    # and must not count as a Fabric attempt.
    _set_fake_topology(2, [[0, 1]])
    vf.disable()

    _reset_stats()

    a0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b0 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    a = vcuda.to_device(a0, device=0)
    b = vcuda.to_device(b0, device=0)

    out = vf.add(a, b, primary=0)
    out_ref = C.vt.add(a, b)

    np.testing.assert_allclose(vcuda.from_device(out), vcuda.from_device(out_ref))

    s = vf.stats()
    assert s.fabric_ops_attempted == 0
    assert s.fabric_ops_hit == 0
    assert s.fabric_ops_fallback == 0

    assert s.inflight_ops_current == 0
    assert s.inflight_ops_peak == 0


def test_single_device_mul_matches_vt_and_no_stats() -> None:
    _set_fake_topology(2, [[0, 1]])
    vf.disable()

    _reset_stats()

    a0 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b0 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    a = vcuda.to_device(a0, device=0)
    b = vcuda.to_device(b0, device=0)

    out = vf.mul(a, b, primary=0)
    out_ref = C.vt.mul(a, b)

    np.testing.assert_allclose(vcuda.from_device(out), vcuda.from_device(out_ref))

    s = vf.stats()
    assert s.fabric_ops_attempted == 0
    assert s.fabric_ops_hit == 0
    assert s.fabric_ops_fallback == 0

    assert s.inflight_ops_current == 0
    assert s.inflight_ops_peak == 0


def test_mixed_device_no_p2p_errors_and_counts() -> None:
    # Enable the global gate but block P2P in the topology helper.
    _set_fake_topology(3, [[1, 2]], can_access_pairs=())
    vf.enable("best_effort")

    _reset_stats()

    a = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    b = vcuda.to_device(np.array([10.0, 20.0], dtype=np.float32), device=1)

    with ag.no_grad():
        with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
            vf.add(a, b, primary=0, require_fabric=True, use_copy_fallback=True)

    msg = str(excinfo.value)
    assert "clique" in msg or "P2P" in msg

    s = vf.stats()
    assert s.fabric_ops_attempted == 1
    assert s.fabric_ops_hit == 0
    assert s.fabric_ops_fallback == 0
    assert s.reasons.no_p2p == 1

    assert s.inflight_ops_current == 0
    assert s.inflight_ops_peak == 1

    assert s.remote_bytes_read == 0
    assert s.remote_bytes_written == 0
    _assert_aggregation_invariants(s)


def test_mixed_device_copy_fallback_counts() -> None:
    _set_fake_topology(3, [[1, 2]], can_access_pairs=())
    vf.enable("best_effort")

    _reset_stats()

    a0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b0 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    a = vcuda.to_device(a0, device=0)
    b = vcuda.to_device(b0, device=1)

    with ag.no_grad():
        out = vf.add(a, b, primary=0, require_fabric=False, use_copy_fallback=True)
    np.testing.assert_allclose(vcuda.from_device(out), a0 + b0)

    s = vf.stats()
    assert s.fabric_ops_attempted == 1
    assert s.fabric_ops_hit == 0
    assert s.fabric_ops_fallback == 1
    assert s.reasons.no_p2p == 1

    assert s.inflight_ops_current == 0
    assert s.inflight_ops_peak == 1

    expected_bytes = int(a0.nbytes)
    assert s.remote_bytes_read == expected_bytes
    assert s.remote_bytes_written == expected_bytes
    _assert_aggregation_invariants(s)

    # Role-based per-device attribution.
    assert s.per_device[0].ops_as_primary == 1
    assert s.per_device[1].ops_as_remote == 1
    assert s.per_device[0].remote_bytes_read == expected_bytes


def test_requires_grad_is_rejected_and_counted() -> None:
    # Force a topology where P2P is allowed so that requires_grad is the
    # gating reason.
    _set_fake_topology(2, [[0, 1]], can_access_pairs=((0, 1), (1, 0)))
    vf.enable("best_effort")

    _reset_stats()

    a = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
    b = vcuda.to_device(np.array([10.0, 20.0], dtype=np.float32), device=1)

    a.requires_grad = True

    with ag.no_grad():
        with pytest.raises(vf._FabricError) as excinfo:  # type: ignore[attr-defined]
            vf.add(a, b, primary=0, require_fabric=True, use_copy_fallback=True)

    assert "autograd" in str(excinfo.value)

    s = vf.stats()
    assert s.fabric_ops_attempted == 1
    assert s.fabric_ops_hit == 0
    assert s.fabric_ops_fallback == 0
    assert s.reasons.requires_grad == 1

    assert s.inflight_ops_current == 0
    assert s.inflight_ops_peak == 1

    assert s.remote_bytes_read == 0
    assert s.remote_bytes_written == 0
    _assert_aggregation_invariants(s)


def teardown_module(module: object) -> None:  # pragma: no cover
    del module
    C._fabric_clear_test_hooks_for_tests()  # type: ignore[attr-defined]
    C._fabric_reset_state_for_tests()  # type: ignore[attr-defined]
