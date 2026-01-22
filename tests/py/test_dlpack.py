# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest


def test_protocol_roundtrip_overlay():
    import vibetensor.torch as vt
    from vibetensor import _C as C

    a = C.vt.unit()
    cap = vt.to_dlpack(a)
    b = vt.from_dlpack(cap)

    assert b.sizes == a.sizes
    assert b.dtype == a.dtype
    assert b.device == a.device


def test_capsule_one_shot_enforced():
    from vibetensor import _C as C

    t = C.vt.unit()
    cap = C._to_dlpack(t)

    # First time OK
    out = C._from_dlpack(cap)

    # Drop the imported tensor to force provider deleter execution and ensure
    # that re-importing does not dereference a potentially freed DLManagedTensor.
    del out
    gc.collect()

    with pytest.raises(RuntimeError, match=r"capsule already consumed"):
        _ = C._from_dlpack(cap)


def test_overlay_kwargs_policy():
    import vibetensor.torch as vt
    from vibetensor import _C as C

    t = C.vt.unit()
    # device accepted for protocol path (CPU only)
    cap = vt.to_dlpack(t)
    _ = vt.from_dlpack(t, device="cpu")

    with pytest.raises(TypeError):
        _ = vt.from_dlpack(t, copy=True)

    # Capsule path forbids kwargs
    with pytest.raises(TypeError):
        _ = vt.from_dlpack(cap, device="cpu")


def test_deleter_runs_once_when_core_importer_throws_after_consumption():
    from vibetensor import _C as C

    if not hasattr(C, "_dlpack_test_make_legacy_capsule_invalid_dtype"):
        pytest.skip("internal dlpack test helpers not available")

    C._dlpack_test_reset_deleter_call_count()
    cap = C._dlpack_test_make_legacy_capsule_invalid_dtype()

    with pytest.raises(RuntimeError):
        _ = C._from_dlpack(cap)

    # The core importer should have invoked the provider deleter exactly once.
    assert C._dlpack_test_get_deleter_call_count() == 1

    # The consumed capsule should not run the deleter again when GC'd.
    del cap
    gc.collect()
    assert C._dlpack_test_get_deleter_call_count() == 1


def test_versioned_capsule_import_calls_deleter_once_on_tensor_free():
    from vibetensor import _C as C

    if not hasattr(C, "_dlpack_test_make_versioned_capsule_float32"):
        pytest.skip("internal dlpack test helpers not available")

    C._dlpack_test_reset_deleter_call_count()
    C._dlpack_test_set_wrap_versioned_alloc_fail_count(0)

    cap = C._dlpack_test_make_versioned_capsule_float32()
    t = C._from_dlpack(cap)

    assert t.sizes == (1,)
    assert t.dtype == "float32"
    assert t.device == (1, 0)

    # Capsule is consumed; dropping it should not invoke provider deleter.
    del cap
    gc.collect()
    assert C._dlpack_test_get_deleter_call_count() == 0

    del t
    gc.collect()
    assert C._dlpack_test_get_deleter_call_count() == 1


def test_versioned_wrapper_allocation_failure_does_not_consume_capsule():
    from vibetensor import _C as C

    if not hasattr(C, "_dlpack_test_make_versioned_capsule_float32"):
        pytest.skip("internal dlpack test helpers not available")

    C._dlpack_test_reset_deleter_call_count()

    cap = C._dlpack_test_make_versioned_capsule_float32()
    C._dlpack_test_set_wrap_versioned_alloc_fail_count(1)

    with pytest.raises(Exception):
        _ = C._from_dlpack(cap)

    # Allocation failure occurs before consumption; the capsule should still be
    # reusable.
    assert C._dlpack_test_get_deleter_call_count() == 0

    # Second attempt should succeed.
    t = C._from_dlpack(cap)
    del cap
    gc.collect()
    assert C._dlpack_test_get_deleter_call_count() == 0

    del t
    gc.collect()
    assert C._dlpack_test_get_deleter_call_count() == 1
