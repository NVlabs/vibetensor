# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Shared helpers for Allocator Python CUDA Graphs tests.

This module provides a small set of utilities used across the Allocator
Python graphs×allocator regression tests:

* CUDA availability helpers (``cuda_available``, ``cuda_device_count``,
  ``require_cuda_or_skip``, ``require_multi_gpu_or_skip``).
* Deterministic RNG reset for tests (``reset_rng_state_for_test``).
* Backend classification and backend-gated skips for native vs async
  allocator tests (``require_native_allocator_or_skip``,
  ``require_async_backend_or_skip``).
* A light-weight snapshot helper
  (``snapshot_allocator_and_graphs``).
* Simple run helpers (``run_eager`` / ``run_graphed``) that standardise
  seeding and fraction configuration for small workloads.

The helpers are intentionally conservative and degrade to skips when
CUDA or the CUDA Graphs overlay is unavailable.
"""

from typing import Any, Callable, Dict, Optional

import os
import random

import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


# ---------------------------------------------------------------------------
# CUDA availability helpers
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """Return True iff CUDA is built and at least one device is present.

    This mirrors the guards used in existing CUDA/graphs tests but
    centralises them so that Allocator tests do not need to duplicate the
    logic.
    """

    has_cuda = getattr(C, "_has_cuda", False)
    get_count = getattr(C, "_cuda_device_count", None)
    if not has_cuda or get_count is None:
        return False
    try:
        return int(get_count()) > 0  # type: ignore[misc]
    except Exception:
        return False


def cuda_device_count() -> int:
    """Best-effort CUDA device count for tests.

    Returns 0 when CUDA is unavailable or when the binding probes fail.
    """

    get_count = getattr(C, "_cuda_device_count", None)
    if get_count is None:
        return 0
    try:
        return max(0, int(get_count()))  # type: ignore[misc]
    except Exception:
        return 0


def require_cuda_or_skip(reason: str = "CUDA required for Allocator graphs tests") -> None:
    """Skip the current test if CUDA is not available.

    Tests should call this before relying on any CUDA behaviour. It is
    intentionally *not* a module-level skip so that individual tests can
    still run on CPU-only builds when they do not require CUDA.
    """

    if not cuda_available():
        pytest.skip(reason, allow_module_level=False)


def require_multi_gpu_or_skip() -> None:
    """Skip the current test unless there are at least two CUDA devices."""

    require_cuda_or_skip("CUDA with >=2 devices required for this test")
    if cuda_device_count() < 2:
        pytest.skip("at least 2 CUDA devices required for this test", allow_module_level=False)


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------


def set_rng_seed_all_cuda(seed: int) -> None:
    """Best-effort helper to seed all CUDA generators.

    On CPU-only builds or when the CUDA RNG helpers are unavailable this
    function degrades to a no-op. Tests should guard with
    :func:`require_cuda_or_skip` when they rely on CUDA.
    """

    if not cuda_available():
        return
    cuda_mod = getattr(vt, "cuda", None)
    if cuda_mod is None:  # pragma: no cover - defensive
        return
    try:
        cuda_mod.manual_seed_all(int(seed))  # type: ignore[attr-defined]
    except Exception:
        # Best-effort: treat unexpected failures as a no-op in tests.
        return


def reset_rng_state_for_test(seed: Optional[int]) -> None:
    """Reset RNG state for Allocator graphs tests.

    When ``seed`` is not ``None``, this function:

    * Calls :func:`vibetensor.torch.manual_seed` to seed the VibeTensor CPU
      and CUDA RNGs to ``seed``.
    * Best-effort seeds PyTorch's global RNG (when torch is importable)
      via :func:`torch.manual_seed`.
    * Seeds Python's :mod:`random` module.

    When ``seed`` is ``None`` this function is a no-op.

    This helper must be called only when no CUDA Graph capture or replay
    is active; tests are responsible for respecting that invariant.
    """

    if seed is None:
        return

    value = int(seed)

    # 1) VibeTensor CPU + CUDA RNGs
    try:
        vt.manual_seed(value)
    except Exception:
        # If seeding fails (e.g. due to unexpected guard errors) we treat
        # it as best-effort; specific RNG-under-graphs tests cover those
        # error paths more precisely.
        pass

    # 2) PyTorch RNG (optional dependency)
    try:
        import torch  # type: ignore[import]
    except Exception:  # pragma: no cover - torch is optional
        torch = None  # type: ignore[assignment]
    if torch is not None:
        try:
            torch.manual_seed(value)
        except Exception:
            pass

    # 3) Python's stdlib RNG
    random.seed(value)


# ---------------------------------------------------------------------------
# Backend classification helpers
# ---------------------------------------------------------------------------


def _parse_backend_from_env() -> Optional[str]:
    """Return backend token from ``VBT_CUDA_ALLOC_CONF`` if present.

    The env var is a comma- or space-separated list of ``key=value``
    pairs. We look for a ``backend=...`` entry and return its value
    verbatim, or ``None`` if no such entry is present.
    """

    conf = os.environ.get("VBT_CUDA_ALLOC_CONF", "")
    if not conf:
        return None
    for tok in conf.replace(",", " ").split():
        if tok.startswith("backend="):
            return tok.split("=", 1)[1].strip() or None
    return None


def _classify_backend_kind() -> str:
    """Classify allocator backend as ``native``, ``async``, ``none`` or ``unknown``.

    The classification is intentionally conservative:

    * When CUDA is unavailable, returns ``"none"``.
    * When ``backend=cudaMallocAsync`` is set in the env, returns
      ``"async"``.
    * When ``backend`` is missing or set to a token that clearly
      indicates the native backend (``native`` or ``cudaMalloc``),
      returns ``"native"``.
    * For any other value, returns ``"unknown"``.

    Allocator tests treat ``"unknown"`` as a signal to skip backend‑specific
    assertions rather than guessing.
    """

    if not cuda_available():
        return "none"

    backend_token = _parse_backend_from_env()
    if backend_token is None:
        # Default configuration uses the native backend.
        return "native"

    low = backend_token.lower()
    if low in ("native", "cudamalloc"):
        return "native"
    if low == "cudamallocasync":
        return "async"
    return "unknown"


def require_native_allocator_or_skip() -> None:
    """Skip the test unless the native allocator backend is active.

    On CPU-only builds this behaves like :func:`require_cuda_or_skip` and
    skips the test. When CUDA is available but the backend is clearly
    configured as async (``backend=cudaMallocAsync``), the test is
    skipped with a descriptive message.
    """

    if not cuda_available():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    kind = _classify_backend_kind()
    if kind == "async":
        pytest.skip(
            "async allocator backend configured via VBT_CUDA_ALLOC_CONF; "
            "test requires native backend",
            allow_module_level=False,
        )
    if kind == "unknown":
        pytest.skip(
            "unable to classify allocator backend from VBT_CUDA_ALLOC_CONF; "
            "skipping backend-specific test",
            allow_module_level=False,
        )


def require_async_backend_or_skip() -> None:
    """Skip the test unless the async allocator backend is active.

    This helper is used by tests that specifically target the async
    allocator (cudaMallocAsync backend). When the backend is not clearly
    configured as async, the test is skipped rather than attempting to
    force a reconfiguration inside the test process.
    """

    if not cuda_available():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    kind = _classify_backend_kind()
    if kind != "async":
        pytest.skip(
            "async allocator backend not configured; set VBT_CUDA_ALLOC_CONF "
            "to use backend=cudaMallocAsync to run this test",
            allow_module_level=False,
        )


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


def snapshot_allocator_and_graphs(device: Optional[int] = None) -> Dict[str, Any]:
    """Return a combined snapshot of allocator and CUDA Graphs observability.

    The returned dict has the following shape::

        {
            "memory_stats": <cuda.memory_stats_as_nested_dict(device)>,
            "memory_snapshot": <cuda.memory_snapshot(device)>,
            "cuda_graphs_stats": <graphs.cuda_graphs_stats(device)>,
            "graph_pool_stats": <graphs.graph_pool_stats()>,
        }

    On CPU-only builds this function returns a dict with the same keys
    but populated with neutral values (empty dicts/lists).
    """

    out: Dict[str, Any] = {}

    cuda_mod = getattr(vt, "cuda", None)
    if cuda_mod is None or not cuda_available():
        # Neutral values for CPU-only builds or when the CUDA overlay
        # failed to initialise.
        out["memory_stats"] = {}
        out["memory_snapshot"] = []
        out["cuda_graphs_stats"] = {"graphs": {}, "pools": {}}
        out["graph_pool_stats"] = []
        return out

    dev = device

    # 1) Allocator stats and snapshot
    try:
        mem_stats = cuda_mod.memory_stats_as_nested_dict(dev)
    except Exception:
        mem_stats = {}
    try:
        mem_snapshot = cuda_mod.memory_snapshot(dev)
    except Exception:
        mem_snapshot = []

    out["memory_stats"] = mem_stats
    out["memory_snapshot"] = mem_snapshot

    # 2) CUDA Graphs stats and pool stats
    try:
        from vibetensor.torch.cuda import graphs as vgraphs  # type: ignore[import]

        graphs_stats = vgraphs.cuda_graphs_stats(dev)
        pool_stats = vgraphs.graph_pool_stats()
    except Exception:
        graphs_stats = {"graphs": {}, "pools": {}}
        pool_stats = []

    out["cuda_graphs_stats"] = graphs_stats
    out["graph_pool_stats"] = pool_stats

    return out


# ---------------------------------------------------------------------------
# Simple run helpers
# ---------------------------------------------------------------------------


def run_eager(
    fn: Callable[[], Any],
    *,
    device: int = 0,
    seed: Optional[int] = None,
    fraction: Optional[float] = None,
    backend: Optional[str] = None,  # reserved for future use
) -> Any:
    """Run ``fn`` eagerly under a configured fraction and RNG seed.

    The callable ``fn`` is expected to close over any inputs it needs
    (including a canonical device string such as ``"cuda:0"``). This
    helper simply normalises RNG state and, when possible, configures the
    per-process memory fraction via :func:`cuda.set_per_process_memory_fraction`.

    The ``backend`` argument is accepted for readability in tests but is
    currently advisory only; backend selection is controlled by the
    process environment rather than this helper.
    """

    require_cuda_or_skip()
    reset_rng_state_for_test(seed)

    cuda_mod = getattr(vt, "cuda", None)
    prev_fraction: Optional[float] = None
    if cuda_mod is not None and fraction is not None:
        try:
            prev_fraction = float(cuda_mod.get_per_process_memory_fraction(device))  # type: ignore[attr-defined]
            cuda_mod.set_per_process_memory_fraction(float(fraction), device)
        except Exception:
            # Treat unexpected errors as best-effort; dedicated tests
            # cover the detailed fraction/GC behaviour.
            prev_fraction = None

    try:
        return fn()
    finally:
        if cuda_mod is not None and prev_fraction is not None:
            try:
                cuda_mod.set_per_process_memory_fraction(prev_fraction, device)
            except Exception:
                pass


def run_graphed(
    fn: Callable[[], Any],
    *,
    device: int = 0,
    seed: Optional[int] = None,
    pool_handle: Optional[object] = None,
    fraction: Optional[float] = None,
    backend: Optional[str] = None,  # reserved for future use
    num_replays: int = 1,
) -> Any:
    """Run ``fn`` inside a CUDA Graph capture and replay it.

    ``fn`` is expected to close over any necessary inputs and return the
    outputs whose parity should be compared against an eager run. This
    helper:

    * Normalises RNG state via :func:`reset_rng_state_for_test`.
    * Optionally configures the per-process memory fraction.
    * Creates a :class:`vibetensor.torch.cuda.graphs.CUDAGraph` and
      captures ``fn`` inside :class:`vibetensor.torch.cuda.graphs.graph`,
      optionally using an explicit ``pool_handle``.
    * Replays the captured graph ``num_replays`` times (default 1).

    The return value is the object returned by ``fn`` during capture.
    """

    require_cuda_or_skip()
    reset_rng_state_for_test(seed)

    cuda_mod = getattr(vt, "cuda", None)
    if cuda_mod is None:
        pytest.skip("CUDA overlay not available for VibeTensor", allow_module_level=False)

    prev_fraction: Optional[float] = None
    if fraction is not None:
        try:
            prev_fraction = float(cuda_mod.get_per_process_memory_fraction(device))  # type: ignore[attr-defined]
            cuda_mod.set_per_process_memory_fraction(float(fraction), device)
        except Exception:
            prev_fraction = None

    try:
        try:
            from vibetensor.torch.cuda import graphs as vgraphs  # type: ignore[import]
        except Exception:
            pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=False)

        g = vgraphs.CUDAGraph(keep_graph=True)

        ctx_kwargs: Dict[str, Any] = {"cuda_graph": g}
        if pool_handle is not None:
            ctx_kwargs["pool"] = pool_handle

        # Capture the workload once.
        with vgraphs.graph(**ctx_kwargs):  # type: ignore[arg-type]
            out = fn()

        # Replay the captured graph a small, configurable number of times to
        # exercise replay counters and steady-state behaviour.
        for _ in range(max(0, int(num_replays))):
            g.replay()

        return out
    finally:
        if prev_fraction is not None:
            try:
                cuda_mod.set_per_process_memory_fraction(prev_fraction, device)
            except Exception:
                pass
