# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor  # noqa: F401 - import-time patches
import vibetensor._C as C
import vibetensor.autograd as A
import vibetensor.torch as vt


def _make_inputs():
    a = vt.tensor([1.0, 2.0])
    a.requires_grad = True
    b = vt.tensor([3.0, 4.0])
    b.requires_grad = True
    return a, b


def test_engine_toggles_roundtrip_and_overlay_helpers():
    ag = C.autograd

    # Defaults are False
    assert ag.is_multithreading_enabled() is False  # type: ignore[attr-defined]
    assert ag.is_view_replay_enabled() is False  # type: ignore[attr-defined]

    # Toggle via C bindings
    ag.set_multithreading_enabled(True)  # type: ignore[attr-defined]
    ag.set_view_replay_enabled(True)  # type: ignore[attr-defined]

    assert ag.is_multithreading_enabled() is True  # type: ignore[attr-defined]
    assert ag.is_view_replay_enabled() is True  # type: ignore[attr-defined]

    # Python overlay wrappers see the same state
    assert A.is_multithreading_enabled() is True
    assert A.is_view_replay_enabled() is True

    # Toggle back via Python overlay
    A.set_multithreading_enabled(False)
    A.set_view_replay_enabled(False)

    assert ag.is_multithreading_enabled() is False  # type: ignore[attr-defined]
    assert ag.is_view_replay_enabled() is False  # type: ignore[attr-defined]
    assert A.is_multithreading_enabled() is False
    assert A.is_view_replay_enabled() is False


def test_engine_toggles_do_not_affect_autograd_wrapper_stats():
    ag = C.autograd
    ag.reset_stats()

    a, b = _make_inputs()
    C._call_op("vt::add", a, b)
    s1 = ag.stats()["wrapper_invocations"]

    # Flip toggles and call again; stats should still bump normally.
    ag.set_multithreading_enabled(True)  # type: ignore[attr-defined]
    ag.set_view_replay_enabled(True)  # type: ignore[attr-defined]

    a2, b2 = _make_inputs()
    C._call_op("vt::add", a2, b2)
    s2 = ag.stats()["wrapper_invocations"]

    assert s2 > s1
