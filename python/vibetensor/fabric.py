# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric: single-process multi-GPU distribution utilities.

This module exposes explicit cross-device ops (``fabric.add``, ``fabric.mul``)
plus best-effort observability:

- ``stats()`` returns per-process counters (attempts/hits/fallbacks, bytes, drops).
- ``events()`` snapshots a bounded per-process ring of diagnostic events when
  events mode is enabled.
- ``wait_for_event_seq()`` is a host-side wait on the event ring sequence number
  (it does **not** imply GPU completion).

Events mode is per-process and can be configured via ``set_events_mode()`` or the
``VBT_FABRIC_EVENTS_MODE`` environment variable (``off`` | ``basic``). BASIC mode
adds overhead and is intended for short debug sessions.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Tuple, cast, overload
import dataclasses
import enum

from . import _C


# DLPack device type codes (reuse DLPack definitions)
_KDLCUDA = 2
_KDLCUDAMANAGED = 13


def _is_cuda_tensor_like(obj: Any) -> Tuple[bool, Optional[int]]:
    dev = getattr(obj, "device", None)
    if not isinstance(dev, (tuple, list)) or len(dev) < 2:
        return (False, None)
    try:
        dev_type = int(dev[0])
        dev_index = int(dev[1])
    except Exception:
        return (False, None)
    if dev_type in (_KDLCUDA, _KDLCUDAMANAGED):
        return (True, dev_index)
    return (False, None)



FabricModeStr = Literal["best_effort", "dry_run"]


class FabricMode(enum.Enum):
    DISABLED = "disabled"
    BEST_EFFORT = "best_effort"
    DRY_RUN = "dry_run"


class FabricEventsMode(enum.Enum):
    OFF = "off"
    BASIC = "basic"


@dataclasses.dataclass(frozen=True)
class FabricEvent:
    """Single Fabric diagnostic event.

    Notes
    -----
    - ``seq`` is a monotonically increasing per-process sequence number.
    - ``t_ns`` is a host steady-clock timestamp captured at record time.
    """

    seq: int
    t_ns: int
    kind: str
    level: str
    primary_device: int
    other_device: int
    op_id: int
    numel: int
    bytes: int
    reason: int
    message: Optional[str]


@dataclasses.dataclass(frozen=True)
class FabricEventSnapshot:
    base_seq: int
    next_seq: int
    dropped_total: int
    capacity: int
    events: Tuple[FabricEvent, ...]


@dataclasses.dataclass(frozen=True)
class FabricClique:
    id: int
    devices: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TopologySnapshot:
    device_count: int
    uva_ok: bool
    mode: FabricMode
    init_status: str
    disable_reason: str
    cliques: Tuple[FabricClique, ...]


@dataclasses.dataclass(frozen=True)
class FabricStatsReasons:
    no_p2p: int
    requires_grad: int
    in_backward: int
    small_tensor: int


@dataclasses.dataclass(frozen=True)
class FabricPerDeviceStats:
    """Per-device Fabric statistics.

    remote_bytes_* fields aggregate bytes for which this device acted as the
    primary (compute) device on a cross-device Fabric candidate. Summing over
    devices matches the global remote_bytes_* counters.
    """

    device_index: int
    ops_as_primary: int
    ops_as_remote: int
    remote_bytes_read: int
    remote_bytes_written: int


@dataclasses.dataclass(frozen=True)
class FabricStatsSnapshot:
    mesh_builds: int
    p2p_pairs_enabled: int
    p2p_pairs_failed: int

    fabric_ops_attempted: int
    fabric_ops_hit: int
    fabric_ops_fallback: int

    remote_bytes_read: int
    remote_bytes_written: int

    inflight_ops_current: int
    inflight_ops_peak: int

    event_queue_len_peak: int
    event_dropped_total: int
    event_failures_total: int

    mode_enable_calls: int
    mode_disable_calls: int
    mode_set_failures: int

    reasons: FabricStatsReasons
    per_device: Tuple[FabricPerDeviceStats, ...]


class _FabricError(RuntimeError):
    """Internal exception type for vibetensor.fabric.

    All user-visible Fabric errors should be raised as _FabricError with a
    message starting with the "[Fabric]" prefix.
    """


def _raise_fabric_error(msg: str) -> None:
    if "[Fabric]" not in msg:
        msg = f"[Fabric] {msg}"
    raise _FabricError(msg)


PlacementKind = Literal["replicated", "sharded_1d_row"]


def _coerce_nonneg_int_tuple(
    values: Any,
    *,
    name: str,
    allow_empty: bool,
) -> Tuple[int, ...]:
    try:
        seq = tuple(values)
    except Exception:
        _raise_fabric_error(f"{name} must be an iterable of ints")

    out: list[int] = []
    for x in seq:
        if isinstance(x, bool) or not isinstance(x, int):
            _raise_fabric_error(f"{name} must contain ints")
        xi = int(x)
        if xi < 0:
            _raise_fabric_error(f"{name} must contain non-negative ints")
        out.append(xi)

    if not allow_empty and not out:
        _raise_fabric_error(f"{name} must be non-empty")

    return tuple(out)


@dataclasses.dataclass(frozen=True)
class FabricMesh:
    """Immutable list of participating CUDA devices."""

    devices: Tuple[int, ...]

    def __post_init__(self) -> None:
        devs = _coerce_nonneg_int_tuple(
            self.devices,
            name="FabricMesh.devices",
            allow_empty=False,
        )

        if len(set(devs)) != len(devs):
            _raise_fabric_error("FabricMesh.devices must be distinct")

        object.__setattr__(self, "devices", devs)


@dataclasses.dataclass(frozen=True)
class FabricPlacement:
    kind: PlacementKind
    mesh: FabricMesh
    global_shape: Tuple[int, ...]
    shard_offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("replicated", "sharded_1d_row"):
            _raise_fabric_error(f"Invalid FabricPlacement.kind: {self.kind!r}")

        if not isinstance(self.mesh, FabricMesh):
            _raise_fabric_error("FabricPlacement.mesh must be a FabricMesh")

        global_shape = _coerce_nonneg_int_tuple(
            self.global_shape,
            name="FabricPlacement.global_shape",
            allow_empty=True,
        )
        shard_offsets = _coerce_nonneg_int_tuple(
            self.shard_offsets,
            name="FabricPlacement.shard_offsets",
            allow_empty=False,
        )

        if len(shard_offsets) != len(self.mesh.devices):
            _raise_fabric_error(
                "FabricPlacement.shard_offsets must have the same length as mesh.devices"
            )

        if self.kind == "replicated":
            if any(int(o) != 0 for o in shard_offsets):
                _raise_fabric_error(
                    "FabricPlacement(replicated) requires shard_offsets to be all zeros"
                )
        else:
            if len(global_shape) < 1:
                _raise_fabric_error(
                    "FabricPlacement(sharded_1d_row) requires global_shape to have rank >= 1"
                )
            if int(shard_offsets[0]) != 0:
                _raise_fabric_error(
                    "FabricPlacement(sharded_1d_row) requires shard_offsets[0] == 0"
                )
            for prev, cur in zip(shard_offsets, shard_offsets[1:]):
                if int(cur) < int(prev):
                    _raise_fabric_error(
                        "FabricPlacement(sharded_1d_row) requires shard_offsets to be non-decreasing"
                    )

        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "shard_offsets", shard_offsets)


@dataclasses.dataclass(frozen=True)
class FabricDevice:
    kind: Literal["fabric"] = "fabric"
    mesh_devices: Tuple[int, ...] = ()
    placement_kind: str = ""


@dataclasses.dataclass(frozen=True, eq=False)
class FabricTensor:
    __vbt_fabric_tensor__ = True

    placement: FabricPlacement
    shards: Tuple[_C.Tensor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.placement, FabricPlacement):
            _raise_fabric_error("FabricTensor.placement must be a FabricPlacement")

        try:
            shards = tuple(self.shards)
        except Exception:
            _raise_fabric_error("FabricTensor.shards must be an iterable of VibeTensor tensors")

        mesh_devices = self.placement.mesh.devices
        if len(shards) != len(mesh_devices):
            _raise_fabric_error(
                "FabricTensor.shards must have the same length as placement.mesh.devices"
            )

        dtypes: list[str] = []
        sizes: list[Tuple[int, ...]] = []
        for i, t in enumerate(shards):
            if not isinstance(t, _C.Tensor):
                _raise_fabric_error("FabricTensor shards must be VibeTensor tensors")

            is_cuda, dev_idx = _is_cuda_tensor_like(t)
            if not is_cuda or dev_idx is None:
                _raise_fabric_error("FabricTensor shards must be CUDA tensors")

            if int(dev_idx) != int(mesh_devices[i]):
                _raise_fabric_error(
                    "FabricTensor shard devices must match placement.mesh.devices"
                )

            dt = getattr(t, "dtype", None)
            if not isinstance(dt, str):
                _raise_fabric_error("FabricTensor shards must have a dtype")

            try:
                sz = tuple(int(s) for s in t.sizes)
            except Exception:
                _raise_fabric_error("FabricTensor shards must have sizes")

            dtypes.append(dt)
            sizes.append(sz)

        if len(set(dtypes)) != 1:
            _raise_fabric_error("FabricTensor shards must have the same dtype")

        if self.placement.kind == "replicated":
            ref = sizes[0]
            if any(sz != ref for sz in sizes[1:]):
                _raise_fabric_error("FabricTensor(replicated) requires equal shard sizes")
            if tuple(self.placement.global_shape) != ref:
                _raise_fabric_error(
                    "FabricTensor(replicated) placement.global_shape must match shard sizes"
                )
            if tuple(self.placement.shard_offsets) != (0,) * len(mesh_devices):
                _raise_fabric_error(
                    "FabricTensor(replicated) placement.shard_offsets must be all zeros"
                )

        elif self.placement.kind == "sharded_1d_row":
            if any(len(sz) < 1 for sz in sizes):
                _raise_fabric_error(
                    "FabricTensor(sharded_1d_row) requires each shard to have rank >= 1"
                )

            tail = sizes[0][1:]
            if any(sz[1:] != tail for sz in sizes[1:]):
                _raise_fabric_error(
                    "FabricTensor(sharded_1d_row) requires all shards to have matching tail dimensions"
                )

            rows = [int(sz[0]) for sz in sizes]
            offsets: list[int] = []
            acc = 0
            for r in rows:
                offsets.append(acc)
                acc += r
            offsets_tuple = tuple(offsets)
            global_shape = (acc,) + tail

            if tuple(self.placement.shard_offsets) != offsets_tuple:
                _raise_fabric_error(
                    "FabricTensor(sharded_1d_row) placement.shard_offsets must match shard sizes"
                )
            if tuple(self.placement.global_shape) != global_shape:
                _raise_fabric_error(
                    "FabricTensor(sharded_1d_row) placement.global_shape must match shard sizes"
                )
        else:  # pragma: no cover - FabricPlacement already validated
            _raise_fabric_error(f"Unsupported FabricPlacement kind: {self.placement.kind!r}")

        object.__setattr__(self, "shards", shards)

    @property
    def mesh(self) -> FabricMesh:
        return self.placement.mesh

    @property
    def global_shape(self) -> Tuple[int, ...]:
        return self.placement.global_shape

    @property
    def dtype(self) -> str:
        return str(getattr(self.shards[0], "dtype", ""))

    @property
    def device(self) -> FabricDevice:
        return FabricDevice(
            kind="fabric",
            mesh_devices=self.mesh.devices,
            placement_kind=str(self.placement.kind),
        )

    def to_local_shards(self) -> Tuple[_C.Tensor, ...]:
        return self.shards

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        _raise_fabric_error(
            "DLPack is not supported for FabricTensor; export a local shard via ft.to_local_shards()"
        )

    def __dlpack_device__(self, *args: Any, **kwargs: Any) -> Any:
        _raise_fabric_error(
            "DLPack is not supported for FabricTensor; export a local shard via ft.to_local_shards()"
        )

    @staticmethod
    def _canonicalize_shards(mesh: FabricMesh, shards: Sequence[_C.Tensor]) -> Tuple[_C.Tensor, ...]:
        try:
            raw = tuple(shards)
        except Exception:
            _raise_fabric_error("FabricTensor shards must be an iterable of VibeTensor tensors")

        if len(raw) != len(mesh.devices):
            _raise_fabric_error("FabricTensor: wrong number of shards for mesh")

        by_dev: dict[int, _C.Tensor] = {}
        for t in raw:
            if not isinstance(t, _C.Tensor):
                _raise_fabric_error("FabricTensor shards must be VibeTensor tensors")
            is_cuda, dev_idx = _is_cuda_tensor_like(t)
            if not is_cuda or dev_idx is None:
                _raise_fabric_error("FabricTensor shards must be CUDA tensors")
            d = int(dev_idx)
            if d in by_dev:
                _raise_fabric_error("FabricTensor shards must not contain duplicate devices")
            by_dev[d] = t

        ordered: list[_C.Tensor] = []
        for d in mesh.devices:
            if int(d) not in by_dev:
                _raise_fabric_error("FabricTensor shards missing a required mesh device")
            ordered.append(by_dev[int(d)])

        return tuple(ordered)

    @classmethod
    def replicated(
        cls,
        mesh: FabricMesh | Sequence[int],
        shards: Sequence[_C.Tensor],
    ) -> "FabricTensor":
        m = mesh if isinstance(mesh, FabricMesh) else FabricMesh(tuple(mesh))
        ordered = cls._canonicalize_shards(m, shards)

        dtype0 = getattr(ordered[0], "dtype", None)
        if not isinstance(dtype0, str):
            _raise_fabric_error("FabricTensor.replicated: shards must have a dtype")
        try:
            shape0 = tuple(int(s) for s in ordered[0].sizes)
        except Exception:
            _raise_fabric_error("FabricTensor.replicated: shards must have sizes")

        for t in ordered[1:]:
            if getattr(t, "dtype", None) != dtype0:
                _raise_fabric_error("FabricTensor.replicated: dtype mismatch")
            try:
                shape = tuple(int(s) for s in t.sizes)
            except Exception:
                _raise_fabric_error("FabricTensor.replicated: shards must have sizes")
            if shape != shape0:
                _raise_fabric_error("FabricTensor.replicated: size mismatch")

        placement = FabricPlacement(
            kind="replicated",
            mesh=m,
            global_shape=shape0,
            shard_offsets=(0,) * len(m.devices),
        )
        return cls(placement=placement, shards=ordered)

    @classmethod
    def sharded_1d_row(
        cls,
        mesh: FabricMesh | Sequence[int],
        shards: Sequence[_C.Tensor],
    ) -> "FabricTensor":
        m = mesh if isinstance(mesh, FabricMesh) else FabricMesh(tuple(mesh))
        ordered = cls._canonicalize_shards(m, shards)

        dtype0 = getattr(ordered[0], "dtype", None)
        if not isinstance(dtype0, str):
            _raise_fabric_error("FabricTensor.sharded_1d_row: shards must have a dtype")

        shapes: list[Tuple[int, ...]] = []
        for t in ordered:
            try:
                shapes.append(tuple(int(s) for s in t.sizes))
            except Exception:
                _raise_fabric_error("FabricTensor.sharded_1d_row: shards must have sizes")

        if any(len(sz) < 1 for sz in shapes):
            _raise_fabric_error("FabricTensor.sharded_1d_row: shards must have rank >= 1")

        rank0 = len(shapes[0])
        if any(len(sz) != rank0 for sz in shapes[1:]):
            _raise_fabric_error("FabricTensor.sharded_1d_row: rank mismatch")

        tail = shapes[0][1:]
        for t, sz in zip(ordered[1:], shapes[1:]):
            if getattr(t, "dtype", None) != dtype0:
                _raise_fabric_error("FabricTensor.sharded_1d_row: dtype mismatch")
            if sz[1:] != tail:
                _raise_fabric_error("FabricTensor.sharded_1d_row: tail-dim mismatch")

        rows = [int(sz[0]) for sz in shapes]
        offsets: list[int] = []
        acc = 0
        for r in rows:
            offsets.append(acc)
            acc += r
        offsets_tuple = tuple(offsets)
        global_shape = (acc,) + tail

        placement = FabricPlacement(
            kind="sharded_1d_row",
            mesh=m,
            global_shape=global_shape,
            shard_offsets=offsets_tuple,
        )
        return cls(placement=placement, shards=ordered)



def _snapshot_from_c() -> TopologySnapshot:
    s = _C._fabric_topology_snapshot()
    return TopologySnapshot(
        device_count=int(s.device_count),
        uva_ok=bool(s.uva_ok),
        mode=FabricMode(s.mode.name),
        init_status=s.init_status.name,
        disable_reason=str(s.disable_reason),
        cliques=tuple(
            FabricClique(id=int(c.id), devices=tuple(int(d) for d in c.devices))
            for c in s.cliques
        ),
    )


def _stats_from_c() -> FabricStatsSnapshot:
    s = _C._fabric_stats_snapshot()
    r = s.reasons
    return FabricStatsSnapshot(
        mesh_builds=int(s.mesh_builds),
        p2p_pairs_enabled=int(s.p2p_pairs_enabled),
        p2p_pairs_failed=int(s.p2p_pairs_failed),
        fabric_ops_attempted=int(s.fabric_ops_attempted),
        fabric_ops_hit=int(s.fabric_ops_hit),
        fabric_ops_fallback=int(s.fabric_ops_fallback),
        remote_bytes_read=int(s.remote_bytes_read),
        remote_bytes_written=int(s.remote_bytes_written),
        inflight_ops_current=int(s.inflight_ops_current),
        inflight_ops_peak=int(s.inflight_ops_peak),
        event_queue_len_peak=int(s.event_queue_len_peak),
        event_dropped_total=int(s.event_dropped_total),
        event_failures_total=int(s.event_failures_total),
        mode_enable_calls=int(s.mode_enable_calls),
        mode_disable_calls=int(s.mode_disable_calls),
        mode_set_failures=int(s.mode_set_failures),
        reasons=FabricStatsReasons(
            no_p2p=int(r.no_p2p),
            requires_grad=int(r.requires_grad),
            in_backward=int(r.in_backward),
            small_tensor=int(r.small_tensor),
        ),
        per_device=tuple(
            FabricPerDeviceStats(
                device_index=int(d.device_index),
                ops_as_primary=int(d.ops_as_primary),
                ops_as_remote=int(d.ops_as_remote),
                remote_bytes_read=int(d.remote_bytes_read),
                remote_bytes_written=int(d.remote_bytes_written),
            )
            for d in s.per_device
        ),
    )


def inspect_topology() -> TopologySnapshot:
    """Return a cached snapshot of Fabric topology and UVA state.

    This call never raises due to UVA failure or lack of CUDA support. Instead,
    it encodes those conditions via init_status and disable_reason.
    """

    return _snapshot_from_c()


def cliques() -> Tuple[Tuple[int, ...], ...]:
    snap = inspect_topology()
    return tuple(c.devices for c in snap.cliques)


def enable(mode: FabricModeStr = "best_effort") -> None:
    """Enable Fabric in the given mode when safe.

    Parameters
    ----------
    mode:
        Either "best_effort" or "dry_run". Any other value raises
        _FabricError.
    """

    if mode not in ("best_effort", "dry_run"):
        _raise_fabric_error(f"Invalid Fabric mode: {mode!r}")

    try:
        if mode == "best_effort":
            _C._fabric_set_mode(_C._FabricMode.best_effort)
        else:
            _C._fabric_set_mode(_C._FabricMode.dry_run)
    except RuntimeError as exc:  # canonical UVA / topology errors bubble up
        _raise_fabric_error(str(exc))


def disable() -> None:
    """Disable Fabric explicitly.

    When UVA is already disabled, this behaves like :func:`enable` and
    raises a canonical Fabric error without changing mode.
    """

    try:
        _C._fabric_set_mode(_C._FabricMode.disabled)
    except RuntimeError as exc:
        _raise_fabric_error(str(exc))


def is_enabled() -> bool:
    """Return True when Fabric is enabled for ops under the current gate.

    This mirrors the C++ fabric_enabled_for_ops(fabric_state()) predicate.
    """

    return bool(_C._fabric_is_enabled_for_ops())


def stats() -> FabricStatsSnapshot:
    """Return a snapshot of Fabric observability counters."""

    return _stats_from_c()


def get_events_mode() -> FabricEventsMode:
    """Return the current Fabric diagnostic events mode for this process."""

    return FabricEventsMode(_C._fabric_get_events_mode().name)


def set_events_mode(mode: FabricEventsMode | str) -> None:
    """Set the per-process Fabric diagnostic events mode.

    Parameters
    ----------
    mode:
        Either :class:`FabricEventsMode` or one of "off" / "basic".

    Notes
    -----
    BASIC mode records events into a bounded ring buffer and adds overhead.
    """

    if isinstance(mode, FabricEventsMode):
        raw = mode.value
    else:
        raw = str(mode)

    raw_lower = raw.lower()
    if raw_lower == "off":
        _C._fabric_set_events_mode(_C._FabricEventsMode.off)
        return
    if raw_lower == "basic":
        _C._fabric_set_events_mode(_C._FabricEventsMode.basic)
        return

    raise ValueError(f"Invalid Fabric events mode: {mode!r}")


def events(min_seq: int = 0, max_events: int = 1024) -> FabricEventSnapshot:
    """Snapshot the Fabric event ring.

    Events are produced only in BASIC mode. The ring is bounded; when it
    overflows, older events are dropped and ``dropped_total`` increases.
    """

    if min_seq < 0:
        raise ValueError("min_seq must be >= 0")
    if max_events < 0:
        raise ValueError("max_events must be >= 0")

    s = _C._fabric_events_snapshot(int(min_seq), int(max_events))

    return FabricEventSnapshot(
        base_seq=int(s.base_seq),
        next_seq=int(s.next_seq),
        dropped_total=int(s.dropped_total),
        capacity=int(s.capacity),
        events=tuple(
            FabricEvent(
                seq=int(e.seq),
                t_ns=int(e.t_ns),
                kind=str(e.kind.name),
                level=str(e.level.name),
                primary_device=int(e.primary_device),
                other_device=int(e.other_device),
                op_id=int(e.op_id),
                numel=int(e.numel),
                bytes=int(e.bytes),
                reason=int(e.reason_raw),
                message=None if e.message is None else str(e.message),
            )
            for e in s.events
        ),
    )


def wait_for_event_seq(target_seq: int, timeout_ms: int) -> bool:
    """Wait (host-side) until the event ring reaches ``target_seq``.

    Returns ``False`` immediately when events mode is OFF.
    """

    if target_seq < 0:
        raise ValueError("target_seq must be >= 0")
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be >= 0")

    return bool(_C._fabric_events_wait_for_seq(int(target_seq), int(timeout_ms)))


def _cpu_scalar_int64(value: int) -> _C.Tensor:
    # Dispatcher args are always tensors; represent scalars as 0-d CPU int64.
    return _C._cpu_full([], "int64", int(value))


def _is_fabric_tensor_marker(obj: Any) -> bool:
    # Type-level marker avoids triggering instance __getattr__.
    return getattr(type(obj), "__vbt_fabric_tensor__", False) is True


def _add_tensor_impl(
    a: _C.Tensor,
    b: _C.Tensor,
    *,
    primary: Optional[int],
    require_fabric: bool,
    use_copy_fallback: bool,
) -> _C.Tensor:
    if not isinstance(a, _C.Tensor) or not isinstance(b, _C.Tensor):
        _raise_fabric_error("vibetensor.fabric.add expects VibeTensor tensors")

    is_cuda_a, dev_a = _is_cuda_tensor_like(a)
    is_cuda_b, dev_b = _is_cuda_tensor_like(b)
    if not is_cuda_a or not is_cuda_b:
        _raise_fabric_error("vibetensor.fabric.add expects CUDA tensors")

    if primary is None:
        if dev_a is None:
            _raise_fabric_error("vibetensor.fabric.add: invalid CUDA device index")
        primary = int(dev_a)

    try:
        return _C._call_op(
            "vt::fabric_add",
            a,
            b,
            _cpu_scalar_int64(int(primary)),
            _cpu_scalar_int64(1 if require_fabric else 0),
            _cpu_scalar_int64(1 if use_copy_fallback else 0),
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "[Fabric]" in msg:
            _raise_fabric_error(msg)
        raise


@overload
def add(
    a: _C.Tensor,
    b: _C.Tensor,
    *,
    primary: Optional[int] = None,
    require_fabric: bool = True,
    use_copy_fallback: bool = True,
) -> _C.Tensor: ...


@overload
def add(
    a: FabricTensor,
    b: FabricTensor,
    *,
    primary: None = None,
    require_fabric: bool = True,
    use_copy_fallback: bool = True,
) -> FabricTensor: ...


def add(
    a: _C.Tensor | FabricTensor,
    b: _C.Tensor | FabricTensor,
    *,
    primary: Optional[int] = None,
    require_fabric: bool = True,
    use_copy_fallback: bool = True,
) -> _C.Tensor | FabricTensor:
    """Elementwise add across up to two CUDA devices.

    This function is an explicit Fabric entrypoint. It accepts two CUDA tensors
    which may reside on different CUDA devices.

    It also supports FabricTensor + FabricTensor when both operands share the
    same mesh and placement.
    """

    is_ft_a = _is_fabric_tensor_marker(a)
    is_ft_b = _is_fabric_tensor_marker(b)
    if is_ft_a or is_ft_b:
        if not (is_ft_a and is_ft_b):
            _raise_fabric_error(
                "vibetensor.fabric.add requires FabricTensor operands when using FabricTensor"
            )
        if primary is not None:
            _raise_fabric_error(
                "vibetensor.fabric.add: primary is not supported for FabricTensor operands"
            )

        if not isinstance(a, FabricTensor) or not isinstance(b, FabricTensor):
            _raise_fabric_error("vibetensor.fabric.add expects FabricTensor operands")

        ft_a = cast(FabricTensor, a)
        ft_b = cast(FabricTensor, b)

        if ft_a.mesh.devices != ft_b.mesh.devices:
            _raise_fabric_error("vibetensor.fabric.add: FabricTensor mesh mismatch")

        if (
            ft_a.placement.kind != ft_b.placement.kind
            or ft_a.placement.global_shape != ft_b.placement.global_shape
            or ft_a.placement.shard_offsets != ft_b.placement.shard_offsets
        ):
            _raise_fabric_error("vibetensor.fabric.add: FabricTensor placement mismatch")

        out_shards: list[_C.Tensor] = []
        for i, dev in enumerate(ft_a.mesh.devices):
            out_shards.append(
                _add_tensor_impl(
                    ft_a.shards[i],
                    ft_b.shards[i],
                    primary=int(dev),
                    require_fabric=require_fabric,
                    use_copy_fallback=use_copy_fallback,
                )
            )

        return FabricTensor(placement=ft_a.placement, shards=tuple(out_shards))

    return _add_tensor_impl(
        cast(_C.Tensor, a),
        cast(_C.Tensor, b),
        primary=primary,
        require_fabric=require_fabric,
        use_copy_fallback=use_copy_fallback,
    )


def _mul_tensor_impl(
    a: _C.Tensor,
    b: _C.Tensor,
    *,
    primary: Optional[int],
    require_fabric: bool,
    use_copy_fallback: bool,
) -> _C.Tensor:
    if not isinstance(a, _C.Tensor) or not isinstance(b, _C.Tensor):
        _raise_fabric_error("vibetensor.fabric.mul expects VibeTensor tensors")

    is_cuda_a, dev_a = _is_cuda_tensor_like(a)
    is_cuda_b, dev_b = _is_cuda_tensor_like(b)
    if not is_cuda_a or not is_cuda_b:
        _raise_fabric_error("vibetensor.fabric.mul expects CUDA tensors")

    if primary is None:
        if dev_a is None:
            _raise_fabric_error("vibetensor.fabric.mul: invalid CUDA device index")
        primary = int(dev_a)

    try:
        return _C._call_op(
            "vt::fabric_mul",
            a,
            b,
            _cpu_scalar_int64(int(primary)),
            _cpu_scalar_int64(1 if require_fabric else 0),
            _cpu_scalar_int64(1 if use_copy_fallback else 0),
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "[Fabric]" in msg:
            _raise_fabric_error(msg)
        raise


@overload
def mul(
    a: _C.Tensor,
    b: _C.Tensor,
    *,
    primary: Optional[int] = None,
    require_fabric: bool = True,
    use_copy_fallback: bool = True,
) -> _C.Tensor: ...


@overload
def mul(
    a: FabricTensor,
    b: FabricTensor,
    *,
    primary: None = None,
    require_fabric: bool = True,
    use_copy_fallback: bool = True,
) -> FabricTensor: ...


def mul(
    a: _C.Tensor | FabricTensor,
    b: _C.Tensor | FabricTensor,
    *,
    primary: Optional[int] = None,
    require_fabric: bool = True,
    use_copy_fallback: bool = True,
) -> _C.Tensor | FabricTensor:
    """Elementwise multiply across up to two CUDA devices.

    Supports FabricTensor + FabricTensor when both operands share the same mesh
    and placement.
    """

    is_ft_a = _is_fabric_tensor_marker(a)
    is_ft_b = _is_fabric_tensor_marker(b)
    if is_ft_a or is_ft_b:
        if not (is_ft_a and is_ft_b):
            _raise_fabric_error(
                "vibetensor.fabric.mul requires FabricTensor operands when using FabricTensor"
            )
        if primary is not None:
            _raise_fabric_error(
                "vibetensor.fabric.mul: primary is not supported for FabricTensor operands"
            )

        if not isinstance(a, FabricTensor) or not isinstance(b, FabricTensor):
            _raise_fabric_error("vibetensor.fabric.mul expects FabricTensor operands")

        ft_a = cast(FabricTensor, a)
        ft_b = cast(FabricTensor, b)

        if ft_a.mesh.devices != ft_b.mesh.devices:
            _raise_fabric_error("vibetensor.fabric.mul: FabricTensor mesh mismatch")

        if (
            ft_a.placement.kind != ft_b.placement.kind
            or ft_a.placement.global_shape != ft_b.placement.global_shape
            or ft_a.placement.shard_offsets != ft_b.placement.shard_offsets
        ):
            _raise_fabric_error("vibetensor.fabric.mul: FabricTensor placement mismatch")

        out_shards: list[_C.Tensor] = []
        for i, dev in enumerate(ft_a.mesh.devices):
            out_shards.append(
                _mul_tensor_impl(
                    ft_a.shards[i],
                    ft_b.shards[i],
                    primary=int(dev),
                    require_fabric=require_fabric,
                    use_copy_fallback=use_copy_fallback,
                )
            )

        return FabricTensor(placement=ft_a.placement, shards=tuple(out_shards))

    return _mul_tensor_impl(
        cast(_C.Tensor, a),
        cast(_C.Tensor, b),
        primary=primary,
        require_fabric=require_fabric,
        use_copy_fallback=use_copy_fallback,
    )


__all__ = [
    "FabricMode",
    "FabricEventsMode",
    "FabricEvent",
    "FabricEventSnapshot",
    "FabricClique",
    "TopologySnapshot",
    "FabricStatsSnapshot",
    "FabricStatsReasons",
    "FabricPerDeviceStats",
    "FabricMesh",
    "FabricPlacement",
    "FabricDevice",
    "FabricTensor",
    "inspect_topology",
    "enable",
    "disable",
    "is_enabled",
    "stats",
    "get_events_mode",
    "set_events_mode",
    "events",
    "wait_for_event_seq",
    "cliques",
    "add",
    "mul",
]
