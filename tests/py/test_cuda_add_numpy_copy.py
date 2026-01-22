# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


def _cuda_available() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


@pytest.mark.filterwarnings("ignore:.*")
def test_cuda_add_with_numpy_copies_end2end():
    if not _cuda_available():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    rng = np.random.default_rng(0)
    shapes = [
        (0,),
        (3,),
        (2, 3),
        (1, 4, 5),
        (257, 513),
    ]

    dev = 0
    for shape in shapes:
        a = rng.standard_normal(size=shape).astype(np.float32, copy=False)
        b = rng.standard_normal(size=shape).astype(np.float32, copy=False)
        ref = (a + b).astype(np.float32, copy=False)

        # H2D copies (async enqueue)
        ad = vt.cuda.to_device(a, device=dev, non_blocking=True)
        bd = vt.cuda.to_device(b, device=dev, non_blocking=True)

        # CUDA add via dispatcher-backed op
        cd = C.vt.add(ad, bd)

        # D2H (non-blocking path) and then synchronize default stream

        out = vt.cuda.from_device(cd, non_blocking=True)
        vt.cuda.current_stream().synchronize()  # type: ignore[attr-defined]

        if out.size > 0:
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        else:
            assert out.size == 0


@pytest.mark.filterwarnings("ignore:.*")
def test_cuda_d2h_async_event_and_nonblocking_array():
    if not _cuda_available():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    rng = np.random.default_rng(1)
    shape = (128, 129)
    a = rng.standard_normal(size=shape).astype(np.float32, copy=False)
    b = rng.standard_normal(size=shape).astype(np.float32, copy=False)
    ref = (a + b).astype(np.float32, copy=False)

    dev = 0
    ad = vt.cuda.to_device(a, device=dev, non_blocking=True)
    bd = vt.cuda.to_device(b, device=dev, non_blocking=True)
    cd = C.vt.add(ad, bd)

    # Async D2H returning (array, event)
    arr_async, ev = vt.cuda.from_device_async(cd)
    assert hasattr(ev, "is_created") and ev.is_created()
    # Cannot rely on ev.query() immediately; just synchronize then compare
    ev.synchronize()
    np.testing.assert_allclose(arr_async, ref, rtol=1e-5, atol=1e-6)

    # Non-blocking D2H returning an array backed by a capsule owner
    arr_nb = vt.cuda.from_device(cd, non_blocking=True)
    # Ensure default stream completion before reading
    vt.cuda.current_stream().synchronize()  # type: ignore[attr-defined]
    np.testing.assert_allclose(arr_nb, ref, rtol=1e-5, atol=1e-6)
    # Ownership checks (only meaningful for non-empty arrays)
    assert arr_nb.base is not None
    assert not arr_nb.flags.owndata


@pytest.mark.filterwarnings("ignore:.*")
def test_cuda_to_device_validation_errors():
    if not _cuda_available():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # float64 is supported (alloc + copy)
    x64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    t64 = vt.cuda.to_device(x64, device=0)
    assert t64.dtype == "float64"
    y64 = vt.cuda.from_device(t64)
    np.testing.assert_allclose(y64, x64)

    y64_nb = vt.cuda.from_device(t64, non_blocking=True)
    np.testing.assert_allclose(y64_nb, x64)
    assert y64_nb.base is not None
    assert not y64_nb.flags.owndata

    # Non C-contiguous (Fortran order)
    xF = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert not xF.flags.c_contiguous and xF.flags.f_contiguous
    with pytest.raises(ValueError):
        _ = vt.cuda.to_device(xF, device=0)
