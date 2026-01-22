# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
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


def _reset_events() -> None:
    if not hasattr(C, "_fabric_reset_events_for_tests"):
        pytest.skip("Fabric event reset helper not available")
    C._fabric_reset_events_for_tests()  # type: ignore[attr-defined]


def _reset_stats() -> None:
    if not hasattr(C, "_fabric_reset_stats_for_tests"):
        pytest.skip("Fabric stats reset helper not available")
    C._fabric_reset_stats_for_tests()  # type: ignore[attr-defined]


def _set_fake_topology(
    device_count: int,
    cliques: Sequence[Sequence[int]],
    *,
    can_access_pairs: Iterable[Tuple[int, int]] = (),
) -> None:
    """Install a synthetic topology via internal test hooks."""

    if not hasattr(C, "_fabric_set_fake_topology_builder_for_tests"):
        pytest.skip("Fabric topology test hooks not available")

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


def test_fabric_events_exports() -> None:
    exported = set(vf.__all__)
    assert {
        "FabricEventsMode",
        "FabricEvent",
        "FabricEventSnapshot",
        "get_events_mode",
        "set_events_mode",
        "events",
        "wait_for_event_seq",
    }.issubset(exported)


def test_fabric_events_mode_get_set_and_validation() -> None:
    prev = vf.get_events_mode()
    try:
        vf.set_events_mode("off")
        assert vf.get_events_mode() == vf.FabricEventsMode.OFF

        vf.set_events_mode("basic")
        assert vf.get_events_mode() == vf.FabricEventsMode.BASIC

        with pytest.raises(ValueError):
            vf.set_events_mode("nope")
    finally:
        vf.set_events_mode(prev)


def test_fabric_events_snapshot_empty_after_reset() -> None:
    prev = vf.get_events_mode()
    try:
        vf.set_events_mode("basic")
        _reset_events()

        snap = vf.events()
        assert snap.base_seq == 0
        assert snap.next_seq == 0
        assert snap.dropped_total == 0
        assert snap.capacity > 0
        assert snap.events == ()
    finally:
        vf.set_events_mode(prev)


def test_wait_for_event_seq_off_and_timeout() -> None:
    prev = vf.get_events_mode()
    try:
        vf.set_events_mode("off")
        _reset_events()
        assert vf.wait_for_event_seq(1, 1) is False

        vf.set_events_mode("basic")
        _reset_events()
        assert vf.wait_for_event_seq(1, 5) is False
    finally:
        vf.set_events_mode(prev)


def test_fabric_events_copy_fallback_sequence() -> None:
    if not _has_two_cuda_devices():
        pytest.skip("Need >=2 CUDA devices")

    prev = vf.get_events_mode()
    try:
        # Ensure enabling Fabric does not emit events.
        vf.set_events_mode("off")

        _set_fake_topology(3, [[1, 2]], can_access_pairs=())
        vf.enable("best_effort")

        vf.set_events_mode("basic")
        _reset_events()
        _reset_stats()

        a0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b0 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a = vcuda.to_device(a0, device=0)
        b = vcuda.to_device(b0, device=1)

        with ag.no_grad():
            out = vf.add(a, b, primary=0, require_fabric=False, use_copy_fallback=True)

        np.testing.assert_allclose(vcuda.from_device(out), a0 + b0)

        snap = vf.events()
        assert len(snap.events) == 2

        e0, e1 = snap.events
        assert e0.kind == "op_enqueue"
        assert e1.kind == "op_fallback"
        assert e0.level == "info"
        assert e1.level == "info"

        assert e0.op_id == e1.op_id
        assert e0.op_id > 0

        assert e0.primary_device == 0
        assert e0.other_device == 1
        assert e1.primary_device == 0
        assert e1.other_device == 1

        expected_bytes = int(a0.nbytes)
        assert e0.bytes == expected_bytes
        assert e1.bytes == expected_bytes
        assert e0.t_ns > 0
        assert e1.t_ns > 0
    finally:
        vf.set_events_mode(prev)


def test_fabric_events_error_sequence() -> None:
    if not _has_two_cuda_devices():
        pytest.skip("Need >=2 CUDA devices")

    prev = vf.get_events_mode()
    try:
        vf.set_events_mode("off")

        _set_fake_topology(3, [[1, 2]], can_access_pairs=())
        vf.enable("best_effort")

        vf.set_events_mode("basic")
        _reset_events()
        _reset_stats()

        a = vcuda.to_device(np.array([1.0, 2.0], dtype=np.float32), device=0)
        b = vcuda.to_device(np.array([10.0, 20.0], dtype=np.float32), device=1)

        with ag.no_grad():
            with pytest.raises(vf._FabricError):  # type: ignore[attr-defined]
                vf.add(a, b, primary=0, require_fabric=True, use_copy_fallback=True)

        snap = vf.events()
        assert len(snap.events) == 2
        assert snap.events[0].kind == "op_enqueue"
        assert snap.events[1].kind == "op_error"
        assert snap.events[1].level == "error"
    finally:
        vf.set_events_mode(prev)


def test_fabric_events_complete_sequence_if_available() -> None:
    if not _has_two_cuda_devices():
        pytest.skip("Need >=2 CUDA devices")

    prev = vf.get_events_mode()
    try:
        vf.set_events_mode("off")

        # Try a topology that permits Fabric; skip if the runtime can't actually run it.
        _set_fake_topology(2, [[0, 1]], can_access_pairs=((0, 1), (1, 0)))
        vf.enable("best_effort")

        vf.set_events_mode("basic")
        _reset_events()
        _reset_stats()

        a0 = np.array([1.0, 2.0], dtype=np.float32)
        b0 = np.array([10.0, 20.0], dtype=np.float32)
        a = vcuda.to_device(a0, device=0)
        b = vcuda.to_device(b0, device=1)

        with ag.no_grad():
            try:
                out = vf.add(a, b, primary=0, require_fabric=True, use_copy_fallback=False)
            except vf._FabricError:
                pytest.skip("Fabric hit not available on this platform")

        np.testing.assert_allclose(vcuda.from_device(out), a0 + b0)

        snap = vf.events()
        assert len(snap.events) == 2
        assert snap.events[0].kind == "op_enqueue"
        assert snap.events[1].kind == "op_complete"
        assert snap.events[1].level == "info"
    finally:
        vf.set_events_mode(prev)


def test_wait_for_event_seq_success_on_op() -> None:
    if not _has_two_cuda_devices():
        pytest.skip("Need >=2 CUDA devices")

    prev = vf.get_events_mode()
    try:
        vf.set_events_mode("off")

        _set_fake_topology(3, [[1, 2]], can_access_pairs=())
        vf.enable("best_effort")

        vf.set_events_mode("basic")
        _reset_events()

        a0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b0 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        a = vcuda.to_device(a0, device=0)
        b = vcuda.to_device(b0, device=1)

        before = vf.events()
        target = int(before.next_seq) + 2

        exc: List[BaseException] = []

        def _worker() -> None:
            try:
                with ag.no_grad():
                    vf.add(a, b, primary=0, require_fabric=False, use_copy_fallback=True)
            except BaseException as e:
                exc.append(e)

        t = threading.Thread(target=_worker)
        t.start()

        assert vf.wait_for_event_seq(target, 2000) is True

        t.join()
        if exc:
            raise exc[0]
    finally:
        vf.set_events_mode(prev)


def teardown_module(module: object) -> None:  # pragma: no cover
    del module
    # Avoid holding Python callables alive in C++ globals at interpreter shutdown.
    clear_hooks = getattr(C, "_fabric_clear_test_hooks_for_tests", None)
    if clear_hooks is not None:
        clear_hooks()  # type: ignore[misc]
    reset_state = getattr(C, "_fabric_reset_state_for_tests", None)
    if reset_state is not None:
        reset_state()  # type: ignore[misc]
