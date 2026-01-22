# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor.triton as vt_triton


def test_triton_code_key_includes_num_stages_in_cache_key():
    # Start from a clean cache state so code_size deltas are predictable.
    vt_triton.clear_cache()
    base_stats = vt_triton.cache_stats()
    base_code_size = base_stats["code_size"]

    def dummy_kernel(a, b):  # pragma: no cover
        return a

    meta = {"BLOCK_SIZE": 128}
    sig = "*fp32,*fp32,*fp32,i32"

    key_default = vt_triton._code_key(
        "vt::add",
        (7, 0),
        "test_version",
        dummy_kernel,
        sig,
        meta,
        128,
        -1,
    )
    key_ns2 = vt_triton._code_key(
        "vt::add",
        (7, 0),
        "test_version",
        dummy_kernel,
        sig,
        meta,
        128,
        2,
    )
    key_ns4 = vt_triton._code_key(
        "vt::add",
        (7, 0),
        "test_version",
        dummy_kernel,
        sig,
        meta,
        128,
        4,
    )

    # num_stages must participate in the code key so different
    # stage settings never alias in the code cache.
    assert key_default != key_ns2
    assert key_ns2 != key_ns4
    assert key_default != key_ns4

    # Insert entries for each key and ensure the code cache size
    # reflects three distinct entries.
    vt_triton._lru_put_code(key_default, (b"ptx0", "entry0", 0))
    vt_triton._lru_put_code(key_ns2, (b"ptx2", "entry2", 0))
    vt_triton._lru_put_code(key_ns4, (b"ptx4", "entry4", 0))

    stats = vt_triton.cache_stats()
    assert stats["code_size"] == base_code_size + 3
