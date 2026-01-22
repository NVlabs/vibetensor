# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from vibetensor.torch import ops


def test_ops_namespace_introspection_and_wrapper_metadata():
    ns = ops.ext
    # Access some ops to populate the cache (cursors)
    _ = ns.square
    _ = ns.add

    # dir should list cached op names (cache-only semantics)
    names = dir(ns)
    assert "square" in names and "add" in names

    # __iter__ should yield wrappers for cached names
    wrappers = list(iter(ns))
    assert len(wrappers) >= 2
    # wrappers carry module and doc metadata
    for w in wrappers:
        assert getattr(w, "__module__", "") == "vibetensor.torch.ops"
        assert isinstance(getattr(w, "__doc__", None), (str, type(None)))
    assert any("VibeTensor operator wrapper for ext::square" in (w.__doc__ or "") for w in wrappers)

    # Top-level ops should reflect accessed namespace
    top_names = dir(ops)
    assert "ext" in top_names

    # __repr__ includes the namespace name
    assert "vibetensor.torch.ops namespace 'ext'" in repr(ns)
