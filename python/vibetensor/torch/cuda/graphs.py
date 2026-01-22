# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union, overload, TypeAlias
from collections.abc import Iterable

from vibetensor import _C as _C
from . import Stream, empty_cache


def _cuda_available() -> bool:
    """Return True iff CUDA is built and at least one device is present."""
    try:
        if not getattr(_C, "_has_cuda", False):
            return False
        count_fn = getattr(_C, "_cuda_device_count", None)
        if count_fn is None:
            return False
        return int(count_fn()) > 0
    except Exception:
        return False


# Low-level helpers from the extension (may be missing on CPU-only builds).
_cuda_isCurrentStreamCapturing = getattr(
    _C, "_cuda_isCurrentStreamCapturing", lambda: False
)
_CUDAGraphImpl = getattr(_C, "_CUDAGraph", None)
_graph_pool_handle_impl = getattr(_C, "_graph_pool_handle", None)


def _normalize_device_arg(device: Optional[int]) -> Optional[int]:
    if device is None:
        return None
    if not isinstance(device, int):
        raise TypeError("device must be an int or None")
    if device < 0:
        raise ValueError("device must be >= 0 or None")
    return device


class GraphPoolHandle:
    """Opaque handle representing a CUDA graph memory pool.

    A :class:`GraphPoolHandle` identifies a single **graph-private** memory
    pool on a given CUDA device. Pools are created and owned by the C++
    CUDA Graphs runtime; Python code treats them as lightweight ``(device, id)``
    tokens.

    Graph pools have the following high-level lifecycle:

    - Created on demand when graphs are captured with allocator routing
      enabled.
    - Shared across graphs by passing the same handle to
      :class:`CUDAGraph` or :class:`graph`.
    - Considered **busy** while any capture, replay, or allocator prewarm
      is in flight; operations that would GC or repurpose a busy pool raise
      a :class:`RuntimeError` whose message contains the substring
      ``"pool is busy with active capture"``.
    - When all users are quiescent, pool GC may demote segments back to the
      global allocator or free them outright without touching global GC
      counters.
    """

    __slots__ = ("_device", "_id")

    def __init__(self, handle: Tuple[int, int]) -> None:
        dev, ident = handle
        if not isinstance(dev, int) or not isinstance(ident, int):
            raise TypeError("GraphPoolHandle requires a (device, id) integer tuple")
        self._device = int(dev)
        self._id = int(ident)

    @property
    def device(self) -> int:
        return self._device

    @property
    def id(self) -> int:
        return self._id

    def to_tuple(self) -> Tuple[int, int]:
        return (self._device, self._id)

    def stats(self) -> list[dict[str, int]]:
        """Return summary stats for this graph pool.

        Equivalent to :func:`graph_pool_stats` called with ``self``.
        """
        from .graphs import graph_pool_stats  # local import to avoid cycles

        return graph_pool_stats(self)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"GraphPoolHandle(device=cuda:{self._device}, id={self._id})"


def graph_pool_handle(device: Optional[int] = None) -> GraphPoolHandle:
    """Return a :class:`GraphPoolHandle` for the given CUDA device.

    ``device=None`` uses the current CUDA device. On CPU-only builds or when
    no CUDA devices exist, this raises
    ``RuntimeError("CUDA unavailable: no devices")``.

    The returned handle can be reused across captures and passed to
    :class:`CUDAGraph` or the :class:`graph` context manager to route
    allocations through a shared graph-private pool, enabling capture-time
    reuse without new ``cudaMalloc`` calls.
    """

    if not _cuda_available():
        raise RuntimeError("CUDA unavailable: no devices")

    if _graph_pool_handle_impl is None:
        raise RuntimeError("CUDA Graphs are not available in this build")

    handle = _graph_pool_handle_impl(device)
    return GraphPoolHandle(handle)


def is_current_stream_capturing(stream: Optional[Stream] = None) -> bool:
    """Return True if CUDA graph capture is underway on the given stream.

    If ``stream`` is None (the common case), this checks the current CUDA
    stream on the current device, mirroring ``torch.cuda.graphs.is_current_stream_capturing``.

    If ``stream`` is provided, this temporarily sets it as the current stream
    on its device, calls the low-level helper, and restores the previous
    current stream. This does not perform any global device synchronization.

    If CUDA is unavailable, always returns False without raising.
    """

    if not _cuda_available():
        return False

    if stream is None:
        return bool(_cuda_isCurrentStreamCapturing())

    if not isinstance(stream, Stream):
        raise TypeError("stream must be a vibetensor.torch.cuda.Stream")

    base_cls = getattr(_C, "_CudaStreamBase", None)
    base = getattr(stream, "_base", None)
    if base_cls is None or base is None:
        # Treat non-backed or CPU-only streams as not capturing.
        return False

    prev = base_cls.current()
    try:
        try:
            base_cls.set_current_with_device(base)
        except Exception:
            base_cls.set_current(base)
        return bool(_cuda_isCurrentStreamCapturing())
    finally:
        # Best-effort restore; errors suppressed to avoid masking user errors.
        try:
            base_cls.set_current(prev)
        except Exception:
            pass


def cuda_graphs_stats(device: Optional[int] = None) -> dict[str, Any]:
    """Return a snapshot of CUDA Graphs counters and graph-pool gauges.

    The returned dict has the shape::

        {
            "graphs": {<GraphCounters field name>: int, ...},
            "pools": {
                "device": int,
                "graphs_pools_created": int,
                "graphs_pools_active": int,
                "graphs_pools_released": int,
            },
        }

    The ``graphs`` section aggregates lifecycle and legality counters for the
    CUDA Graphs runtime. Notable fields include:

    - ``captures_started`` / ``captures_ended`` – number of capture sessions.
    - ``allocator_capture_denied`` – number of allocations rejected by the
      caching allocator during capture (for example due to missing routing or
      async-backend restrictions). Allocator tests rely on this counter to stay
      unchanged when fraction-cap OOMs occur **outside** of capture.

    The ``pools`` section exposes high-level graph-pool gauges for the
    queried device. On CPU-only builds or when no CUDA devices are present,
    this function returns the same structure with all values set to 0.
    """

    dev = _normalize_device_arg(device)

    impl = getattr(_C, "_cuda_graphs_stats", None)
    if not _cuda_available() or impl is None:
        default_dev = 0 if dev is None else dev
        zeros = {
            "captures_started": 0,
            "captures_ended": 0,
            "denied_default_stream": 0,
            "nested_capture_denied": 0,
            "end_in_dtor": 0,
            "end_in_dtor_errors": 0,
            "graphs_instantiated": 0,
            "graphs_replayed": 0,
            "replay_nesting_errors": 0,
            "unsupported_capture_mode": 0,
            "capture_begin_invalid_state": 0,
            "capture_end_invalid_state": 0,
            "instantiate_invalid_state": 0,
            "instantiate_errors": 0,
            "replay_invalid_state": 0,
            "replay_device_mismatch": 0,
            "replay_errors": 0,
            "graphs_reset": 0,
            "reset_invalid_state": 0,
            "reset_inflight_denied": 0,
            "reset_errors": 0,
            "allocator_capture_denied": 0,
        }
        pools = {
            "device": default_dev,
            "graphs_pools_created": 0,
            "graphs_pools_active": 0,
            "graphs_pools_released": 0,
        }
        return {"graphs": zeros, "pools": pools}

    raw = impl(dev)
    graphs_raw = dict(raw.get("graphs", {}))
    pools_raw = dict(raw.get("pools", {})) if "pools" in raw else {}

    graphs = {str(k): int(v) for k, v in graphs_raw.items()}

    if pools_raw:
        pools = {str(k): int(v) for k, v in pools_raw.items()}
    else:
        default_dev = 0 if dev is None else dev
        pools = {
            "device": default_dev,
            "graphs_pools_created": 0,
            "graphs_pools_active": 0,
            "graphs_pools_released": 0,
        }

    return {"graphs": graphs, "pools": pools}


def _to_pool_filter(pool: object | None) -> object | None:
    """Internal: normalize pool filters for _cuda_graph_pools_snapshot.

    Accepted forms:
      - None: all devices, all pools.
      - (device, id): raw tuple.
      - GraphPoolHandle: uses its underlying (device, id) tuple.
    """
    if pool is None:
        return None

    if isinstance(pool, GraphPoolHandle):
        return pool.to_tuple()

    if isinstance(pool, Iterable) and not isinstance(pool, (str, bytes)):
        dev, pid = pool
        return (int(dev), int(pid))

    raise TypeError("pool must be None, a GraphPoolHandle, or a (device, id) tuple")


def graph_pool_stats(pool: object | None = None) -> list[dict[str, int]]:
    """Return per-pool summaries for CUDA Graph private pools.

    Each dict in the returned list has the keys::

        device, id, segments, blocks, bytes_reserved, bytes_active.

    and the following meanings:

    - ``device`` – CUDA device index for the pool.
    - ``id`` – allocator pool identifier (stable across snapshots).
    - ``segments`` – number of segments owned by the pool.
    - ``blocks`` – total number of blocks across those segments.
    - ``bytes_reserved`` – total reserved bytes in the pool’s segments.
    - ``bytes_active`` – subset of ``bytes_reserved`` currently allocated.

    In Allocator tests, these values are related to the native
    :func:`vibetensor.torch.cuda.memory_snapshot` view via inequalities over
    ``pool_id`` and byte counts. On CPU-only builds or when no CUDA devices
    are available, this function returns an empty list.
    """

    if not _cuda_available():
        return []

    impl = getattr(_C, "_cuda_graph_pools_snapshot", None)
    if impl is None:
        return []

    # Snapshot all pools at the C++ level; apply any user-provided
    # filtering in Python to keep the binding simple and robust across
    # nanobind versions.
    raw = impl(None)  # type: ignore[misc]

    out: list[dict[str, int]] = []
    for entry in raw:  # type: ignore[misc]
        d = dict(entry)
        out.append(
            {
                "device": int(d.get("device", 0)),
                "id": int(d.get("id", 0)),
                "segments": int(d.get("segments", 0)),
                "blocks": int(d.get("blocks", 0)),
                "bytes_reserved": int(d.get("bytes_reserved", 0)),
                "bytes_active": int(d.get("bytes_active", 0)),
            }
        )

    if pool is not None:
        filt = _to_pool_filter(pool)
        if filt is not None:
            dev_f, pid_f = filt
            out = [
                s
                for s in out
                if s.get("device") == int(dev_f) and s.get("id") == int(pid_f)
            ]

    return out


class CUDAGraph:
    """Python wrapper around a single-device CUDA graph.

    This class is a thin front-end to a ``_C._CUDAGraph`` instance and mirrors
    the high-level behavior of :class:`torch.cuda.CUDAGraph` for basic capture
    and replay, with these key differences:

    * Only ``capture_error_mode="thread_local"`` is supported.
    * Advanced debug APIs like ``raw_cuda_graph`` are not exposed.
    * Allocations performed during capture **without** allocator routing are
      forbidden by the caching allocator and raise a :class:`RuntimeError`
      whose message contains the pinned substring
      ``"cuda allocator: allocations are forbidden during CUDA graph capture"``
      and increments the ``allocator_capture_denied`` counter surfaced by
      :func:`cuda_graphs_stats`.
    * When capture is routed through a graph memory pool, the allocator
      reuses pre-warmed blocks from that pool only; no new ``cudaMalloc``
      calls are issued while capture is active.
    """

    def __init__(self, keep_graph: bool = False) -> None:
        if not _cuda_available():
            raise RuntimeError("CUDA Graphs are only available when CUDA is enabled")
        if _CUDAGraphImpl is None:
            raise RuntimeError("CUDA Graphs are not available in this build")
        self._keep_graph = bool(keep_graph)
        self._impl = _CUDAGraphImpl()  # type: ignore[misc]
        self._instantiated = False

    def capture_begin(
        self,
        pool: Optional[GraphPoolHandle] = None,
        capture_error_mode: str = "thread_local",
    ) -> None:
        """Begin capturing CUDA work on the current stream."""

        handle: Optional[Tuple[int, int]] = None
        if pool is not None:
            if not isinstance(pool, GraphPoolHandle):
                raise TypeError("pool must be a GraphPoolHandle or None")
            handle = pool.to_tuple()
        # Delegates to C++ binding, which enforces legality and error substrings.
        self._impl.capture_begin(handle, capture_error_mode)
        self._instantiated = False

    def capture_end(self) -> None:
        """End CUDA graph capture on the current stream.

        Note:
            Unlike :meth:`torch.cuda.CUDAGraph.capture_end`, which only ends
            capture, this method will also instantiate the graph eagerly when
            ``keep_graph`` is False. This mirrors the common PyTorch usage
            pattern of calling :meth:`instantiate` immediately after the first
            capture to make the first replay cheap.
        """

        self._impl.capture_end()
        if not self._keep_graph and not self._instantiated:
            self._impl.instantiate()
            self._instantiated = True

    def instantiate(self) -> None:
        """Instantiate the captured graph explicitly."""

        self._impl.instantiate()
        self._instantiated = True

    def replay(self) -> None:
        """Replay the CUDA work captured by this graph on the capture stream."""

        # If user forgot to instantiate and keep_graph=True, lazily instantiate.
        if not self._instantiated:
            self._impl.instantiate()
            self._instantiated = True
        self._impl.replay()

    def pool(self) -> GraphPoolHandle:
        """Return the GraphPoolHandle for this graph's memory pool."""

        return GraphPoolHandle(self._impl.pool())

    def reset(self) -> None:
        """Reset the CUDA graph held by this instance.

        This releases the underlying CUDA graph, exec instance, and any
        graph-private memory pool retained for this graph. After ``reset()``,
        the object can be used for a fresh capture.
        """

        self._impl.reset()
        self._instantiated = False


import gc
import threading

# Thread-local tracker of the currently active CUDA graph context.
# This is used by RNG factory helpers to mark captures that encountered
# RNG-under-graphs misuse, so that graph.__exit__ can suppress the secondary
# cudaStreamEndCapture failure after the primary error has already been
# surfaced to the caller.
class _ActiveGraphContext(threading.local):
    """Per-thread active CUDA graph context."""
    # Attribute is created lazily on first assignment per thread.
    current: "graph | None"

_ACTIVE_GRAPH_CONTEXT = _ActiveGraphContext()


def _get_active_graph_context() -> "graph | None":
    """Internal: return the active graph context for this thread, if any."""
    return getattr(_ACTIVE_GRAPH_CONTEXT, "current", None)


class graph:
    """Context manager that captures CUDA work into a :class:`CUDAGraph`.

    Arguments:
        cuda_graph (CUDAGraph): Graph object used for capture.
        pool (GraphPoolHandle, optional): Graph pool handle hinting that this
            graph may share memory with the specified pool; when provided,
            allocations during capture are routed to that pool and must reuse
            pre-existing segments (no new ``cudaMalloc`` calls are issued while
            capture is active).
        stream (Stream, optional): Capture stream. If omitted, a
            process-global non-default side stream is used.
        capture_error_mode (str): CUDA stream capture mode; must be
            ``"thread_local"``.

    During capture, allocations issued on streams **without** routing are
    forbidden by the allocator and raise a :class:`RuntimeError` whose message
    contains the canonical capture-denial substring
    ``"cuda allocator: allocations are forbidden during CUDA graph capture"``.
    Such denials increment ``allocator_capture_denied`` in
    :func:`cuda_graphs_stats` and leave fraction-cap and GC stats unchanged.
    """

    default_capture_stream: Optional[Stream] = None

    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: Optional[GraphPoolHandle] = None,
        stream: Optional[Stream] = None,
        capture_error_mode: str = "thread_local",
    ) -> None:
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = Stream()

        self.cuda_graph = cuda_graph
        self.pool = pool
        self.capture_stream = stream or self.__class__.default_capture_stream
        self.capture_error_mode = capture_error_mode
        # Use existing Stream context manager to set current stream.
        self._stream_ctx = self.capture_stream
        # Flag used to suppress capture_end failures when a prior RNG
        # misuse error has already been reported from inside the capture.
        self._skip_capture_end: bool = False
        # Previous active graph context for this thread (for nested contexts).
        self._prev_active_graph: "graph | None" = None

    def __enter__(self) -> None:
        # Free as much memory as we can for the graph, mirroring PyTorch in
        # spirit while keeping semantics async-first at the primitive level.
        self.capture_stream.synchronize()
        gc.collect()
        empty_cache()

        # Enter the stream context so that subsequent CUDA work (including
        # allocator calls) is issued on the capture stream.
        self._stream_ctx.__enter__()

        # Allocator pre-warm for graph-private pools is now handled in C++
        # from CUDAGraph.capture_begin via Allocator::prewarm_graph_pool_for_stream_.
        # Python remains allocator-agnostic here.

        # Mark this context as active on the current thread so RNG helpers can flag misuses.
        self._prev_active_graph = _get_active_graph_context()
        _ACTIVE_GRAPH_CONTEXT.current = self

        self.cuda_graph.capture_begin(
            pool=self.pool,
            capture_error_mode=self.capture_error_mode,
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # End capture and always unwind the stream context; exceptions from
        # capture_end propagate but never prevent restoration of the previous
        # stream/device.
        try:
            try:
                self.cuda_graph.capture_end()
            except RuntimeError as e:
                # When RNG-under-graphs helpers have already surfaced a
                # cross-stream misuse error, cudaStreamEndCapture is expected
                # to report that the capture was invalidated. In that case we
                # suppress this secondary failure and rely on the C++ cleanup
                # path (which has already torn down allocator routing and RNG
                # capture) rather than raising a second exception.
                if not self._skip_capture_end or "cudaStreamEndCapture failed" not in str(e):
                    raise
        finally:
            try:
                self._stream_ctx.__exit__(exc_type, exc, tb)
            finally:
                if _get_active_graph_context() is self:
                    _ACTIVE_GRAPH_CONTEXT.current = self._prev_active_graph
                self._skip_capture_end = False
                self._prev_active_graph = None


# Type aliases for graphed-callables API.
_ModuleOrCallable: TypeAlias = Callable[..., Any]
_GraphedCallableReturn: TypeAlias = Union[
    _ModuleOrCallable,
    tuple[_ModuleOrCallable, ...],
]


@overload
def make_graphed_callables(
    callables: _ModuleOrCallable,
    sample_args: Tuple[Any, ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[GraphPoolHandle] = None,
) -> _ModuleOrCallable:
    ...


@overload
def make_graphed_callables(
    callables: Tuple[_ModuleOrCallable, ...],
    sample_args: Tuple[Tuple[Any, ...], ...],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[GraphPoolHandle] = None,
) -> tuple[_ModuleOrCallable, ...]:
    ...


# NOTE: This is a stub and raises NotImplementedError.
def make_graphed_callables(
    callables: Union[_ModuleOrCallable, Tuple[_ModuleOrCallable, ...]],
    sample_args: Union[Tuple[Any, ...], Tuple[Tuple[Any, ...], ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[GraphPoolHandle] = None,
) -> _GraphedCallableReturn:
    r"""VibeTensor stub for :func:`make_graphed_callables`.

    This function is **not implemented**. The API surface and documentation are
    reserved to match :func:`torch.cuda.graphs.make_graphed_callables` as closely
    as possible for a future implementation. Real semantics will be defined in
    the design docs under ``design/`` (search for ``make_graphed_callables``).

    The stub does not perform any CUDA work. It immediately raises
    :class:`NotImplementedError` on every call.

    The intended future behavior mirrors
    :func:`torch.cuda.graphs.make_graphed_callables` in PyTorch:

    Accept callables (functions or :class:`nn.Module <torch.nn.Module>`\ s)
    and return graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    The graphed callable's forward pass also appends a backward node to the
    autograd graph. During backward, this node runs the callable's backward
    work as a CUDA graph.

    Therefore, each graphed callable should be a drop-in replacement for its
    source callable in an autograd-enabled training loop.

    If you pass a tuple of several callables, their captures will use the same
    memory pool.

    The arguments, constraints, and warnings for the eventual implementation
    are expected to follow the upstream docstring in
    ``torch.cuda.graphs.make_graphed_callables`` (warmup semantics, Tensor-only
    ``sample_args``, no higher order differentiation, module/buffer restrictions,
    and AMP autocast requirements).
    """

    # Avoid unused-variable warnings; there is no runtime behavior beyond
    # raising NotImplementedError in this build.
    _ = (callables, sample_args, num_warmup_iters, allow_unused_input, pool)

    raise NotImplementedError(
        "vibetensor.torch.cuda.graphs.make_graphed_callables is not "
        "implemented yet; this API is reserved for a future implementation."
    )

__all__ = [
    "GraphPoolHandle",
    "graph_pool_handle",
    "is_current_stream_capturing",
    "cuda_graphs_stats",
    "graph_pool_stats",
    "CUDAGraph",
    "graph",
    "make_graphed_callables",
]
