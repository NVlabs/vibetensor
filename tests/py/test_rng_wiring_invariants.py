# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import pkgutil
import importlib

import vibetensor.torch as vt


def _iter_torch_modules():
    """Yield (name, module) for vibetensor.torch and its submodules.

    This deliberately imports all submodules under ``vibetensor.torch`` so we
    can perform simple source-based wiring checks without relying on the
    filesystem layout at test time.
    """

    modules = {}
    root = vt
    modules[root.__name__] = root

    if getattr(root, "__path__", None) is not None:
        for info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
            try:
                mod = importlib.import_module(info.name)
            except Exception:
                # Best-effort: skip modules that fail to import (should not
                # happen in normal test runs).
                continue
            modules[info.name] = mod

    return list(modules.items())


def test_cuda_rng_mutations_use_guarded_bindings() -> None:
    """Ensure CUDA RNG mutation bindings are only used in rng/cuda modules.

    The guarded bindings ``_cuda_rng_manual_seed`` and ``_cuda_rng_set_state``
    must not be referenced directly from arbitrary Python modules; only
    ``vibetensor.torch.rng`` and ``vibetensor.torch.cuda`` are allowed to use
    them so that all CUDA RNG mutations go through the C++ guard.
    """

    offenders: list[str] = []
    for name, mod in _iter_torch_modules():
        try:
            src = inspect.getsource(mod)
        except OSError:
            # Some modules (e.g., C extensions) may not have retrievable
            # source; they cannot contain the Python-level strings we care
            # about.
            continue
        if "_cuda_rng_manual_seed" in src or "_cuda_rng_set_state" in src:
            if name not in ("vibetensor.torch.rng", "vibetensor.torch.cuda"):
                offenders.append(name)

    assert not offenders, f"_cuda_rng_* bindings must only be used in vibetensor.torch.rng and vibetensor.torch.cuda, found in: {offenders!r}"


def test_no_default_cuda_symbol_in_rng_and_cuda_helpers() -> None:
    """Ensure Python RNG helpers do not call default_cuda directly.

    ``default_cuda`` is a C++-side concept; Python RNG helpers should only
    talk to the guarded bindings. This keeps the RNG–Graphs bridge surface
    area small and predictable.
    """

    from vibetensor.torch import rng as vrng  # local import for clarity
    import vibetensor.torch.cuda as vcuda

    src_rng = inspect.getsource(vrng)
    src_cuda = inspect.getsource(vcuda)

    assert "default_cuda(" not in src_rng
    assert "default_cuda(" not in src_cuda
