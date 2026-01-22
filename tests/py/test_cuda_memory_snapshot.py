# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor.torch.cuda as cuda


def test_memory_snapshot_cpu_returns_empty_list():
    snap = cuda.memory_snapshot()
    assert isinstance(snap, list)

    # On CPU-only builds or when CUDA bindings are unavailable, the snapshot
    # must be an empty list. On CUDA builds, non-empty snapshots are covered
    # by a separate shape test.
    try:
        import vibetensor._C as _C_mod  # type: ignore[import]
        has_cuda = bool(getattr(_C_mod, "_has_cuda", False)) and int(getattr(_C_mod, "_cuda_device_count", lambda: 0)()) > 0  # type: ignore[attr-defined]
    except Exception:
        has_cuda = False

    if not has_cuda:
        assert snap == []


def test_memory_snapshot_segment_dict_shape_if_present():
    snap = cuda.memory_snapshot()
    assert isinstance(snap, list)
    for seg in snap:
        assert isinstance(seg, dict)
        for key in ("device", "pool_id", "bytes_reserved", "bytes_active", "blocks"):
            assert key in seg
