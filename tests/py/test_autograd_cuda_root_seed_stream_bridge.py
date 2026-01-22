# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C

import vibetensor.torch.cuda as vc


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0  # type: ignore[attr-defined]


def _stream_handle(s: vc.Stream) -> int:
    try:
        _ord, handle = s.__cuda_stream__()
        return int(handle)
    except Exception:
        base = getattr(s, "_base", None)
        return int(getattr(base, "cuda_stream", 0)) if base is not None else 0


@pytest.mark.cuda
def test_cuda_backward_bridges_call_stream_to_root_stream() -> None:
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    if not hasattr(C, "_cuda_empty"):
        pytest.skip("_cuda_empty not available", allow_module_level=False)

    if not hasattr(vc, "from_device_async"):
        pytest.skip("from_device_async not available", allow_module_level=False)

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        s_root = vc.Stream(priority=0)
        s_call = vc.Stream(priority=0)
        assert _stream_handle(s_root) != _stream_handle(s_call)

        # Run forward on s_root so the root node captures that as its canonical stream.
        with s_root:
            x = C._make_cuda_tensor([], "float32", 3.0)
            x.requires_grad = True
            y = C._call_op("vt::add", x, x)

        # Sanity: y's grad_fn should be CUDA allowlisted and point at s_root.
        handle, _output_nr = ag._graph_get_gradient_edge(y)  # type: ignore[attr-defined]
        info = ag._grad_fn_stream_info(handle)  # type: ignore[attr-defined]
        assert info["stream_kind"] == "cuda_allowlisted"
        assert bool(info["has_canonical_stream"]) is True
        assert int(info["stream_id"]) == _stream_handle(s_root)

        # Start a long async op on s_call, then invoke backward() on s_call.
        # With correct bridging, backward() should not complete until the s_call
        # work is visible on the root canonical stream.
        with s_call:
            grad = C._make_cuda_tensor([], "float32", 1.0)
            dev_idx = int(x.device[1])

            # Create a pending async D2H copy on the call stream to keep it busy.
            # This is intentionally heuristic: pick the smallest size that still
            # yields an in-flight event to reduce pinned-memory pressure.
            big = None
            _arr = None
            ev_copy = None
            for n in (1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000):
                try:
                    big = C._cuda_empty([n], "float32", dev_idx)  # type: ignore[attr-defined]
                    _arr, ev_copy = vc.from_device_async(big)
                except MemoryError as e:
                    pytest.skip(f"unable to allocate async D2H copy buffer: {e}")
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "pinned" in msg or ("alloc" in msg and "memory" in msg):
                        pytest.skip(f"unable to allocate async D2H copy buffer: {e}")
                    raise
                if not ev_copy.query():
                    break
            else:
                pytest.skip(
                    "unable to create a pending async D2H copy event; stream-bridge test is inconclusive"
                )

            y.backward(grad)

        # Record an event after backward's work on the root stream.
        ev_back = vc.Event()
        ev_back.record(s_root)
        ev_back.synchronize()

        # If backward bridged streams correctly, the copy should be complete.
        assert ev_copy.query() is True

        gx = x.grad
        assert gx is not None
        assert float(gx.cpu().item()) == pytest.approx(2.0)
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]
