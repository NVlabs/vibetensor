# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor._C as C
import vibetensor.autograd as A


def test_device_mode_toggle_roundtrip_overlay_and_bindings():
    ag = C.autograd

    prev = ag.get_device_mode()  # type: ignore[attr-defined]
    try:
        ag.set_device_mode("single_device")  # type: ignore[attr-defined]
        assert ag.get_device_mode() == "single_device"  # type: ignore[attr-defined]
        assert A.get_device_mode() == "single_device"

        ag.set_device_mode("multi_device_experimental")  # type: ignore[attr-defined]
        assert (
            ag.get_device_mode() == "multi_device_experimental"  # type: ignore[attr-defined]
        )
        assert A.get_device_mode() == "multi_device_experimental"

        # Toggle back via Python overlay.
        A.set_device_mode("single_device")
        assert ag.get_device_mode() == "single_device"  # type: ignore[attr-defined]
        assert A.get_device_mode() == "single_device"
    finally:
        ag.set_device_mode(prev)  # type: ignore[attr-defined]
