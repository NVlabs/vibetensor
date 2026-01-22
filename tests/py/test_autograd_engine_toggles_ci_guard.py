# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_engine_toggles_not_used_in_core_autograd_files():
    """Guardrail: wrapper must not consult engine toggle stubs.

    Originally, multithreading/view-replay toggles were intentionally unused.
    Starting in Version 3, the core engine consults `is_multithreading_enabled()`
    to select the multithreaded execution path.

    We keep the guardrail for other toggles (and for wrapper.cc) to avoid
    accidentally growing surface area in unrelated components.
    """

    root = Path(__file__).resolve().parents[2]
    engine_src = (root / "src" / "vbt" / "autograd" / "engine.cc").read_text(encoding="utf-8")
    wrapper_src = (root / "src" / "vbt" / "autograd" / "wrapper.cc").read_text(encoding="utf-8")

    assert "is_multithreading_enabled(" not in wrapper_src

    # Still forbidden: view-replay toggle is unused.
    for src in (engine_src, wrapper_src):
        assert "is_view_replay_enabled(" not in src
