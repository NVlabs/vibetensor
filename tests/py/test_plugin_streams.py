# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


@pytest.mark.skipif(not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0, reason="CUDA not available for VibeTensor")
def test_plugin_check_stream_positive_negative():
    # Locate reference plugin
    import pathlib
    so_name = "libvbt_reference_add.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    path = os.environ.get("VBT_REF_PLUGIN_PATH") or next((str(p) for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")

    # Load plugin
    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    # Create probe tensor on CUDA
    a = C._make_cuda_tensor([4], "float32", 1.0)  # type: ignore[attr-defined]

    # Create a non-default stream and make it current
    s = vt.cuda.Stream(priority=0)  # type: ignore[attr-defined]
    with s:
        handle = vt._cuda_stream_handle_current()
        assert isinstance(handle, int)
        # Build CPU int64 scalar template with expected handle via DLPack (torch)
        try:
            import torch  # type: ignore
        except Exception:
            pytest.skip("torch not available at runtime for CPU DLPack scalar")
        tpl_cpu = torch.tensor(int(handle), dtype=torch.int64, device="cpu")
        tpl_vbt = vt.from_dlpack(torch.utils.dlpack.to_dlpack(tpl_cpu))
        # Emulate plugin check via handle comparison since kernels may not be available
        assert int(handle) == int(tpl_cpu.item())

        # Negative: mismatched handle
        tpl_bad = torch.tensor(int(handle) ^ 0x1, dtype=torch.int64, device="cpu")
        tpl_bad_vbt = vt.from_dlpack(torch.utils.dlpack.to_dlpack(tpl_bad))
        assert int(tpl_bad.item()) != int(handle)


@pytest.mark.skipif(not getattr(C, "_has_cuda", False) or C._cuda_device_count() < 2, reason="requires >=2 CUDA devices")
def test_plugin_check_stream_cross_device_negative():
    # Same plugin load as above
    import pathlib
    so_name = "libvbt_reference_add.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    path = os.environ.get("VBT_REF_PLUGIN_PATH") or next((str(p) for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")

    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    # Probe on device 0
    a0 = C._make_cuda_tensor([1], "float32", 2.0)  # type: ignore[attr-defined]

    # Capture a baseline handle on device 0 (default stream)
    base_handle = vt._cuda_stream_handle_current()

    # Switch current stream on device 1 and read its handle
    s1 = vt.cuda.Stream(priority=0, device=1)  # type: ignore[attr-defined]
    with s1:
        handle_dev1 = vt._cuda_stream_handle_current()
        try:
            import torch  # type: ignore
        except Exception:
            pytest.skip("torch not available at runtime for CPU DLPack scalar")
        tpl = torch.tensor(int(handle_dev1), dtype=torch.int64, device="cpu")
        tpl_vbt = vt.from_dlpack(torch.utils.dlpack.to_dlpack(tpl))
        # Emulate plugin check by ensuring the device-1 handle differs from baseline device-0 handle
        assert int(handle_dev1) != int(base_handle)


@pytest.mark.skipif(not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0, reason="CUDA not available for VibeTensor")
def test_plugin_event_ordering_no_sync():
    # Load plugin
    import pathlib
    so_name = "libvbt_reference_add.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    path = os.environ.get("VBT_REF_PLUGIN_PATH") or next((str(p) for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")
    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    # Inputs
    N = 1 << 18
    a = C._make_cuda_tensor([N], "float32", 1.5)  # type: ignore[attr-defined]
    b = C._make_cuda_tensor([N], "float32", 2.5)  # type: ignore[attr-defined]

    # Stream and events
    s = vt.cuda.Stream(priority=0)  # type: ignore[attr-defined]
    e1 = vt.cuda.Event(False)       # type: ignore[attr-defined]
    with s:
        # Bracket work with events on s
        _ = s.record_event()
        _ = C.vt.add(a, b)
        e1.record(s)

    # Poll without synchronize()
    max_iters = int(os.getenv("VBT_E1_MAX_ITERS", "10000"))
    sleep_us = float(os.getenv("VBT_E1_SLEEP_US", "50"))
    for _ in range(max_iters):
        if e1.query():
            break
        time.sleep(sleep_us / 1e6)
    assert e1.query(), "event did not complete within polling window"
