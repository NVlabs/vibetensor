# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

from vibetensor import _C as _C

# Public API mirrors a tiny subset of torch.cuda, backed by _C bindings when available.

# ----- Streams and events wrappers -----

class Stream:
    def __init__(self, priority: int = 0, device: Optional[int] = None) -> None:
        base_cls = getattr(_C, "_CudaStreamBase", None)
        if base_cls is None:
            # Create a lightweight stub with minimal fields
            self._base = None
            self._priority = int(priority)
            self._device = int(device or 0)
        else:
            if device is None:
                self._base = base_cls(int(priority))
            else:
                self._base = base_cls(int(priority), int(device))

    @staticmethod
    def priority_range() -> Tuple[int, int]:
        base_cls = getattr(_C, "_CudaStreamBase", None)
        if base_cls is None:
            return (0, 0)
        least, greatest = base_cls.priority_range()
        return int(least), int(greatest)

    @staticmethod
    def _wrap_base(base) -> "Stream":
        # Allocation-free wrapper for a native _CudaStreamBase.
        out = Stream.__new__(Stream)
        out._base = base
        return out

    @staticmethod
    def current(device: Optional[int] = None) -> "Stream":
        base_cls = getattr(_C, "_CudaStreamBase", None)
        if base_cls is None:
            return Stream(0, device=device)
        base = base_cls.current() if device is None else base_cls.current(int(device))
        return Stream._wrap_base(base)

    @staticmethod
    def set_current(stream: "Stream") -> None:
        base_cls = getattr(_C, "_CudaStreamBase", None)
        if base_cls is None:
            return
        if stream._base is not None:
            base_cls.set_current(stream._base)

    def query(self) -> bool:
        if self._base is None:
            return True
        return bool(self._base.query())

    def synchronize(self) -> None:
        if self._base is None:
            return
        self._base.synchronize()

    def wait_stream(self, other: "Stream") -> None:
        # Implement via an event recorded on other, waited on self
        ev = Event()
        ev.record(other)
        ev.wait(self)

    def wait_event(self, ev: "Event") -> None:
        base_cls = getattr(_C, "_CudaEventBase", None)
        if base_cls is None or self._base is None:
            return
        if isinstance(ev, Event) and getattr(ev, "_base", None) is not None:
            ev._base.wait(self._base)

    def record_event(self) -> "Event":
        ev = Event()
        ev.record(self)
        return ev

    def __enter__(self) -> "Stream":
        base_cls = getattr(_C, "_CudaStreamBase", None)

        # Snapshot previous stream so we can restore it on exit.
        self._prev_stream = None
        if base_cls is not None:
            try:
                self._prev_stream = base_cls.current()
            except Exception:
                self._prev_stream = None

        if base_cls is not None and self._base is not None:
            # Best-effort: also set runtime current device to stream.device
            try:
                base_cls.set_current_with_device(self._base)
            except Exception:
                base_cls.set_current(self._base)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        base_cls = getattr(_C, "_CudaStreamBase", None)
        if base_cls is None:
            return
        try:
            prev = getattr(self, "_prev_stream", None)
            if prev is None:
                prev = base_cls.current()
            try:
                base_cls.set_current_with_device(prev)
            except Exception:
                base_cls.set_current(prev)
        except Exception:
            pass

    def __repr__(self) -> str:
        dev = 0
        handle = 0
        if self._base is not None:
            try:
                dev = int(self._base.device_index)
            except Exception:
                dev = 0
            try:
                handle = int(self._base.cuda_stream)
            except Exception:
                handle = 0
        return f"<vibetensor.cuda.Stream device=cuda:{dev} cuda_stream=0x{handle:x}>"

    def __cuda_stream__(self) -> Tuple[int, int]:
        # PyTorch ExternalStream expects (ordinal, handle)
        if self._base is None:
            return (0, 0)
        try:
            return (0, int(self._base.cuda_stream))
        except Exception:
            return (0, 0)


class Event:
    def __init__(self, enable_timing: bool = False) -> None:
        base_cls = getattr(_C, "_CudaEventBase", None)
        if base_cls is None:
            self._base = None
        else:
            self._base = base_cls(bool(enable_timing))

    def record(self, stream: Stream) -> None:
        if self._base is None:
            return
        base_stream = getattr(stream, "_base", None)
        if base_stream is None:
            return
        self._base.record(base_stream)

    def wait(self, stream: Stream) -> None:
        if self._base is None:
            return
        base_stream = getattr(stream, "_base", None)
        if base_stream is None:
            return
        self._base.wait(base_stream)

    def query(self) -> bool:
        if self._base is None:
            return True
        return bool(self._base.query())

    def synchronize(self) -> None:
        if self._base is None:
            return
        self._base.synchronize()

    def is_created(self) -> bool:
        if self._base is None:
            return True
        return bool(self._base.is_created())


# Module-level priority_range, matching tests that call vc.priority_range()

def priority_range() -> Tuple[int, int]:
    return Stream.priority_range()


def current_stream() -> Stream:
    return Stream.current()

def to_device(array: Any, *, device: Optional[int] = None, non_blocking: bool = False):
    if _np is None:
        raise TypeError("NumPy is required for cuda.to_device")
    arr = _np.asarray(array)
    tok = _np.dtype(arr.dtype).name
    # Require C-contiguous input; match error policy expected by tests
    if not arr.flags.c_contiguous:
        raise ValueError("to_device: expected a C-contiguous NumPy array")

    # Resolve target device index (mirror RNG helpers)
    dev = int(getattr(_C, "_cuda_current_device", lambda: 0)()) if device is None else int(device)

    # Async H2D path: do NOT introduce an internal temporary copy, since the
    # GPU may still be reading from the host buffer after this function
    # returns. Require a writeable C-contiguous NumPy array so the
    # caller controls the lifetime of the underlying storage.

    # Sync H2D path: always ensure C-contiguous copy for nanobind compatibility
    src = _np.ascontiguousarray(arr)
    tok = _np.dtype(src.dtype).name
    out = _C._cuda_h2d_alloc_copy(src, tok, dev, non_blocking)  # type: ignore[attr-defined]
    
    if non_blocking:
        # Attach reference to source array to keep it alive during async copy
        setattr(out, "_h2d_source_ref", src)
        
    return out


def from_device(tensor: Any, *, non_blocking: bool = False) -> Any:
    if non_blocking:
        arr, ev = _C._cuda_d2h_copy_numpy_async(tensor)  # type: ignore[attr-defined]
        # Synchronize before returning to ensure data is ready (safe non_blocking)
        ev.synchronize()
        return arr
    return _C._cuda_d2h_copy_numpy_sync(tensor)  # type: ignore[attr-defined]


def from_device_async(tensor: Any) -> Tuple[Any, Event]:
    arr, ev_base = _C._cuda_d2h_copy_numpy_async(tensor)  # type: ignore[attr-defined]
    # Wrap event base in our Event class
    e = Event()
    try:
        e._base = ev_base  # type: ignore[attr-defined]
    except Exception:
        pass
    return arr, e


# ----- RNG helpers -----

def _ensure_cuda_available() -> None:
    if not getattr(_C, "_has_cuda", False) or int(_C._cuda_device_count()) <= 0:  # type: ignore[attr-defined]
        raise ValueError("CUDA is not available")


def manual_seed(seed: int) -> None:
    _ensure_cuda_available()
    cur = int(getattr(_C, "_cuda_current_device", lambda: 0)())
    _C._cuda_rng_manual_seed(cur, int(seed))  # type: ignore[attr-defined]


def manual_seed_all(seed: int) -> None:
    _ensure_cuda_available()
    n = int(_C._cuda_device_count())  # type: ignore[attr-defined]
    for k in range(n):
        _C._cuda_rng_manual_seed(k, int(seed))  # type: ignore[attr-defined]


def initial_seed(device: Optional[int] = None) -> int:
    _ensure_cuda_available()
    dev = int(getattr(_C, "_cuda_current_device", lambda: 0)()) if device is None else int(device)
    if dev < 0 or dev >= int(_C._cuda_device_count()):  # type: ignore[attr-defined]
        raise ValueError("device index out of range")
    return int(_C._cuda_rng_initial_seed(dev))  # type: ignore[attr-defined]


def get_rng_state(device: Optional[int] = None) -> bytes:
    _ensure_cuda_available()
    dev = int(getattr(_C, "_cuda_current_device", lambda: 0)()) if device is None else int(device)
    if dev < 0 or dev >= int(_C._cuda_device_count()):  # type: ignore[attr-defined]
        raise ValueError("device index out of range")
    return _C._cuda_rng_get_state(dev)  # type: ignore[attr-defined]


def set_rng_state(state: bytes, device: Optional[int] = None) -> None:
    _ensure_cuda_available()
    if not isinstance(state, (bytes, bytearray)):
        raise TypeError("state must be a bytes object")
    dev = int(getattr(_C, "_cuda_current_device", lambda: 0)()) if device is None else int(device)
    if dev < 0 or dev >= int(_C._cuda_device_count()):  # type: ignore[attr-defined]
        raise ValueError("device index out of range")
    _C._cuda_rng_set_state(dev, bytes(state))  # type: ignore[attr-defined]


# Internal helpers for CUDA memory stats; kept private to this module.
_DEVICE_ERROR_SUFFIX = "device must be >= 0 or None for current device"

_CUDA_MEMORY_STATS_FN = getattr(_C, "_cuda_memoryStats", None)
_CUDA_GET_DEVICE_STATS_FN = getattr(_C, "_cuda_getDeviceStats", None)
_CUDA_DEVICE_COUNT_FN = getattr(_C, "_cuda_device_count", None)

_KNOWN_BYTE_FAMILIES = {"allocated_bytes", "reserved_bytes", "requested_bytes"}


def _zero_family() -> dict[str, dict[str, int]]:
    return {"all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0}}


def _memory_stats_raw(device: Optional[int] = None) -> dict[str, Any]:
    """Low-level CUDA memory stats helper.

    Returns a mapping shaped like _cuda_memoryStats (three byte families plus
    scalar counters) when available, a synthetic three-family mapping based on
    _cuda_getDeviceStats when only that legacy binding exists, or {} when no
    stats bindings are present.

    Negative device indices and obviously invalid device types are rejected
    consistently across CPU-only and CUDA builds.

    On CUDA builds, device argument validation is delegated to C++ bindings
    when stats symbols exist; Python-side validation only applies in the
    "no stats bindings" branch.
    """
    # 1. No stats bindings at all (CPU-only or extremely minimal build)
    if _CUDA_MEMORY_STATS_FN is None and _CUDA_GET_DEVICE_STATS_FN is None:
        if device is not None:
            # Enforce int-like semantics; non-int values raise TypeError/ValueError
            dev_int = int(device)  # may raise
            if dev_int < 0:
                raise RuntimeError(_DEVICE_ERROR_SUFFIX)
        # No stats bindings: treat as "no stats"; higher layer synthesizes zeros.
        return {}

    # 2. Preferred path: _cuda_memoryStats
    if _CUDA_MEMORY_STATS_FN is not None:
        raw = _CUDA_MEMORY_STATS_FN(device)  # may raise RuntimeError on invalid device
        if isinstance(raw, dict) and raw:
            return raw

        # raw is {} – distinguish zero-device CUDA from packaging errors.
        ndev = 0
        if _CUDA_DEVICE_COUNT_FN is not None:
            try:
                ndev = int(_CUDA_DEVICE_COUNT_FN())
            except Exception:
                ndev = 0

        if ndev > 0:
            # Devices exist but stats dict is empty: packaging/logic error.
            # Do NOT fall back; returning {} allows tests to detect missing
            # device_stats.aggregated.* counters while still surfacing three
            # zeroed byte families at the public layer.
            return {}
        # ndev == 0 or unknown: treat as no-device and fall through to legacy.

    # 3. Legacy fallback: _cuda_getDeviceStats
    if _CUDA_GET_DEVICE_STATS_FN is not None:
        a, r, ma, mr = _CUDA_GET_DEVICE_STATS_FN(device)  # may raise for invalid device
        return {
            "allocated_bytes": {
                "all": {"current": int(a), "peak": int(ma), "allocated": 0, "freed": 0}
            },
            "reserved_bytes": {
                "all": {"current": int(r), "peak": int(mr), "allocated": 0, "freed": 0}
            },
            "requested_bytes": {
                "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0}
            },
        }

    # 4. Final fallback: treat as no stats
    return {}


def memory_stats_as_nested_dict(device: Optional[int] = None):
    """Return a PyTorch-like nested dict of CUDA allocator stats.

    The mapping always contains three byte families
    (``allocated_bytes``, ``reserved_bytes``, ``requested_bytes``) with an
    ``"all"`` pool and ``{current, peak, allocated, freed}`` integer gauges.
    When ``_cuda_memoryStats`` is available and returns scalar counters,
    ``DeviceStats`` fields such as ``fraction_cap_breaches``,
    ``fraction_cap_misfires``, ``inactive_split_blocks_all``,
    ``inactive_split_bytes_all``, ``gc_passes``, and ``gc_reclaimed_bytes``
    are surfaced under ``device_stats.aggregated``. Additional dict-valued
    families from ``_cuda_memoryStats`` (if any) are preserved verbatim to keep
    the schema open ended.

    CPU-only builds or zero-device configurations still return the full
    three-family structure, but every gauge and counter is zero. Async-backend
    devices are treated like native ones for stats: byte families track async
    reserved/allocated bytes, while native-only counters such as the
    fraction-cap and GC fields remain at 0.
    """
    raw = _memory_stats_raw(device)

    out: dict[str, dict] = {}

    # 1) Always provide the three byte families.
    for fam in ("allocated_bytes", "reserved_bytes", "requested_bytes"):
        fam_val = raw.get(fam)
        if isinstance(fam_val, dict):
            out[fam] = fam_val
        else:
            out[fam] = _zero_family()

    # 2) Preserve any additional dict-valued families from raw verbatim.
    for key, value in raw.items():
        if key in _KNOWN_BYTE_FAMILIES:
            continue
        if isinstance(value, dict):
            # Future dict families are surfaced as-is and flattened generically.
            out.setdefault(key, value)

    # 3) Collect scalar counters into device_stats.aggregated.
    counters: dict[str, int] = {}
    for key, value in raw.items():
        if key in _KNOWN_BYTE_FAMILIES:
            continue
        if isinstance(value, dict):
            continue  # dict-valued families handled above
        try:
            counters[key] = int(value)
        except Exception:
            continue

    if counters:
        out["device_stats"] = {"aggregated": counters}

    return out


# NOTE: Key ordering groups families in insertion order:
#   1) allocated_bytes.all.*
#   2) reserved_bytes.all.*
#   3) requested_bytes.all.*
#   4) any extra dict families
#   5) device_stats.aggregated.*
# Callers must not rely on lexicographic ordering.

def memory_stats(device: Optional[int] = None):
    """Return a flattened ``OrderedDict`` view of CUDA allocator stats.

    This is a convenience wrapper around
    :func:`memory_stats_as_nested_dict` that produces keys like
    ``"allocated_bytes.all.current"`` and
    ``"device_stats.aggregated.fraction_cap_breaches"`` suitable for logging
    and assertions in tests.

    On CPU-only and zero-device builds the mapping has the same keys but all
    values are 0. Async-backend devices expose their byte-family gauges while
    native-only counters such as the fraction-cap and GC fields remain 0.
    """
    from collections import OrderedDict  # lazy import

    nested = memory_stats_as_nested_dict(device)
    flat: "OrderedDict[str, int]" = OrderedDict()

    for fam, sub in nested.items():
        if not isinstance(sub, dict):
            continue
        for pool, metrics in sub.items():
            if not isinstance(metrics, dict):
                continue
            for metric, val in metrics.items():
                try:
                    flat[f"{fam}.{pool}.{metric}"] = int(val)
                except Exception:
                    continue

    return flat


def empty_cache() -> None:
    try:
        _C._cuda_emptyCache()  # type: ignore[attr-defined]
    except Exception:
        # no-op when CUDA unavailable
        return


def set_per_process_memory_fraction(fraction: float, device: Optional[int] = None) -> None:
    """Set the per-process CUDA memory fraction for a device.

    This is a thin wrapper around the unified C++ allocator setter
    ``Allocator::setMemoryFraction`` exposed as ``_C._cuda_setMemoryFraction``.
    The ``fraction`` argument must be in ``[0.0, 1.0]`` and controls a
    per-process cap on **prospective reserved bytes** for the native backend:

    - ``0.0`` – disallow any further native reserved-bytes growth on the
      chosen device. Subsequent growth allocations will fail with a dedicated
      fraction-cap OOM whose message contains the substring
      ``"per-process memory fraction cap"``.
    - ``0.0 < fraction < 1.0`` – enforce a strict cap derived from
      ``fraction * total_device_bytes``; large growth allocations near the cap
      can fail even if CUDA reports some free memory.
    - ``fraction == 1.0`` – effectively disable the native fraction cap and
      behave like an uncapped caching allocator (subject to normal OOM).

    The cap is applied only to **growth** allocations that would call
    ``cudaMalloc``; per-stream and cross-stream free-list reuse remain
    fraction-neutral and can still succeed even when the allocator’s
    ``reserved_bytes_all_current`` counter is above the logical cap. During
    CUDA Graph capture, the fraction gate and GC ladder are bypassed entirely:
    captures either reuse pre-existing blocks or are denied with a
    capture-denial error, and fraction-cap counters do not change while
    capture is in progress.

    On CPU-only builds or when the binding is missing this function validates
    its inputs and becomes a no-op.
    """
    if not isinstance(fraction, (int, float)):
        raise TypeError("fraction must be a float")
    value = float(fraction)
    if not (0.0 <= value <= 1.0):
        raise ValueError("fraction must be in [0,1]")

    impl = getattr(_C, "_cuda_setMemoryFraction", None)
    if impl is None:
        # CPU-only or missing binding: validated no-op
        return

    # Use _cuda_device_count if present; treat absence as CPU-only.
    dev_count_fn = getattr(_C, "_cuda_device_count", lambda: 0)
    ndev = int(dev_count_fn())  # may raise; propagate packaging errors
    if ndev == 0:
        return

    impl(value, device)


def get_per_process_memory_fraction(device: Optional[int] = None) -> float:
    """Return the configured per-process CUDA memory fraction.

    The returned value is the same per-device fraction used by the native and
    async backends to compute their per-process caps:

    - ``0.0`` – no further native reserved-bytes growth; new growth
      allocations will fail with fraction-cap OOM.
    - ``0.0 < fraction < 1.0`` – cap enforced on prospective reserved bytes
      for new ``cudaMalloc`` segments.
    - ``fraction >= 1.0`` – native fraction cap is effectively disabled.

    On CPU-only builds or when the underlying binding is unavailable, this
    function returns ``1.0`` as a neutral default.
    """
    impl = getattr(_C, "_cuda_getMemoryFraction", None)
    if impl is None:
        return 1.0

    dev_count_fn = getattr(_C, "_cuda_device_count", lambda: 0)
    ndev = int(dev_count_fn())
    if ndev == 0:
        return 1.0

    return float(impl(device))


def memory_snapshot(device: Optional[int] = None):
    """Return a segment-level snapshot of the CUDA caching allocator state.

    For the native backend this returns a list of dicts with keys
    ``"device"``, ``"pool_id"``, ``"bytes_reserved"``, ``"bytes_active"``,
    and ``"blocks"`` for each tracked memory segment. Global segments have
    ``pool_id == 0`` and graph-private segments have ``pool_id > 0``.

    Async-backend devices are omitted from the snapshot, and CPU-only or
    zero-device builds return an empty list instead of raising. This matches
    the async-first observability contract used elsewhere in the CUDA graphs
    helpers.
    """

    impl = getattr(_C, "_cuda_memorySnapshot", None)
    if impl is None:
        return []

    return list(impl(device))  # type: ignore[attr-defined]


# ----- CUDA Graphs overlay re-exports -----
from . import graphs as _graphs

CUDAGraph = _graphs.CUDAGraph
graph = _graphs.graph
graph_pool_handle = _graphs.graph_pool_handle
is_current_stream_capturing = _graphs.is_current_stream_capturing
cuda_graphs_stats = _graphs.cuda_graphs_stats
graph_pool_stats = _graphs.graph_pool_stats
make_graphed_callables = _graphs.make_graphed_callables



def is_available() -> bool:
    """Returns a bool indicating if CUDA is currently available."""
    if not getattr(_C, "_has_cuda", False):
        return False
    try:
        return int(_C._cuda_device_count()) > 0
    except Exception:
        return False


def device_count() -> int:
    """Returns the number of GPUs available."""
    if not is_available():
        return 0
    return int(_C._cuda_device_count())


__all__ = [
    "is_available",
    "device_count",
    "Stream",
    "Event",
    "priority_range",
    "current_stream",
    "to_device",
    "from_device",
    "from_device_async",
    "memory_stats_as_nested_dict",
    "memory_stats",
    "empty_cache",
    "set_per_process_memory_fraction",
    "get_per_process_memory_fraction",
    "memory_snapshot",
    "CUDAGraph",
    "graph",
    "graph_pool_handle",
    "is_current_stream_capturing",
    "cuda_graphs_stats",
    "graph_pool_stats",
    "make_graphed_callables",
]