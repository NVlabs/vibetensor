# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

def test_import_and_version():
    import vibetensor
    assert isinstance(vibetensor.__version__, str)


def test_extension_sentinel_and_placement():
    from vibetensor import _C as C
    assert callable(C._vbt_hello)
    assert C._vbt_hello()

    p = Path(C.__file__)
    assert p.name.startswith("_C.")
    assert (p.parent.name == "vibetensor") or ("vibetensor" in p.parts)


def test_overlay_import_only():
    import vibetensor.torch as vt
    assert getattr(vt, "__all__", ()) in ((), [], None)


def test_logging_smoke():
    from vibetensor import _C as C
    C._init_logging(None)
    C._init_logging(2)
