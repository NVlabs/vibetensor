# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numbers
import queue
import threading
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional

from ._queue import ClosableQueue, QueueClosed
from .collate import default_collate, default_convert
from .pin_memory import pin_memory_batch
from .device_prefetch import prefetch_to_device_batch
from .dataset import IterableDataset
from .sampler import BatchSampler, RandomSampler, SequentialSampler, _next_seed_from_generator
from .worker import WorkerInfo, _set_worker_info

if TYPE_CHECKING:
    from vibetensor.torch.cuda import Event as _CudaEvent


_U64_MASK = (1 << 64) - 1


def _is_iterable_dataset(dataset: Any) -> bool:
    # Prefer explicit marker base class, but accept objects that only provide
    # __iter__ (and not __getitem__) as iterable-style.
    if isinstance(dataset, IterableDataset):
        return True
    if not hasattr(dataset, "__getitem__") and hasattr(dataset, "__iter__"):
        return True
    return False


def _map_fetch(dataset: Any, indices: list[int]) -> list[Any]:
    getitems = getattr(dataset, "__getitems__", None)
    if callable(getitems):
        try:
            out = getitems(indices)
        except NotImplementedError:
            out = None
        if out is not None:
            return list(out)
    return [dataset[i] for i in indices]


def _derive_base_seed(generator: Any | None, *, salt: int) -> int:
    """Derive a base seed and (if provided) bump the generator offset.

    Generator policy:
    - generator is None: nondeterministic base_seed from time and salt.
    - generator is CPU vibetensor.torch.rng.Generator: read state bytes
      [seed:u64][offset:u64] and compute base_seed = seed + offset (u64 wrap),
      then bump offset by +1 (u64 wrap).

    This does not mutate any global Python RNG state.
    """

    if generator is None:
        return int(time.time_ns() ^ int(salt)) & _U64_MASK

    return _next_seed_from_generator(generator, who="DataLoader")


class DataLoader:
    def __init__(
        self,
        dataset: Any,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Optional[Iterable[int]] = None,
        batch_sampler: Optional[Iterable[list[int]]] = None,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0.0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        device: str | int | None = None,
        prefetch_to_device: bool = False,
        non_blocking: bool = False,
    ) -> None:
        # ----- normalize / validate scalar args -----
        if batch_size is not None:
            if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral):
                raise TypeError("DataLoader: batch_size must be an int or None")
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError("DataLoader: batch_size must be a positive integer")

        if isinstance(num_workers, bool) or not isinstance(num_workers, numbers.Integral):
            raise TypeError("DataLoader: num_workers must be an int")
        num_workers = int(num_workers)

        if prefetch_factor is not None:
            if isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, numbers.Integral):
                raise TypeError("DataLoader: prefetch_factor must be an int or None")
            prefetch_factor = int(prefetch_factor)
            if prefetch_factor <= 0:
                raise ValueError("DataLoader: prefetch_factor must be a positive integer")

        if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real):
            raise TypeError("DataLoader: timeout must be a number")
        timeout = float(timeout)
        if not math.isfinite(timeout):
            raise ValueError("DataLoader: timeout must be finite")
        if timeout < 0.0:
            raise ValueError("DataLoader: timeout must be non-negative")

        if generator is not None:
            from vibetensor.torch.factory import _validate_generator

            _validate_generator(generator, "cpu")
            if not callable(getattr(generator, "get_state", None)) or not callable(
                getattr(generator, "set_state", None)
            ):
                raise TypeError("DataLoader: generator must implement get_state() and set_state()")

            state = generator.get_state()
            if not isinstance(state, (bytes, bytearray)) or len(state) != 16:
                raise RuntimeError("DataLoader: generator.get_state() must return 16 bytes")

        # ----- store args -----
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.timeout = float(timeout)
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.generator = generator
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = bool(persistent_workers)
        self.pin_memory_device = str(pin_memory_device)
        self.in_order = bool(in_order)
        self.device = device
        self.prefetch_to_device = bool(prefetch_to_device)
        self.non_blocking = bool(non_blocking)

        self._dataset_is_iterable = _is_iterable_dataset(dataset)

        # ---- v1 gates / invariants ----
        if self.num_workers < 0:
            raise ValueError("DataLoader: num_workers must be >= 0")

        if self.num_workers == 0:
            if self.timeout != 0.0:
                raise ValueError("DataLoader: timeout must be 0 when num_workers==0")
            if self.prefetch_factor is not None:
                raise ValueError("DataLoader: prefetch_factor must be None when num_workers==0")
        else:
            if self._dataset_is_iterable:
                raise NotImplementedError("DataLoader: IterableDataset with num_workers>0 is not supported")

            if self.prefetch_factor is None:
                self.prefetch_factor = 2
            if self.worker_init_fn is not None and not callable(self.worker_init_fn):
                raise TypeError("DataLoader: worker_init_fn must be callable")

        if self.multiprocessing_context is not None:
            raise NotImplementedError("DataLoader: multiprocessing_context is not supported")
        if self.persistent_workers:
            raise NotImplementedError("DataLoader: persistent_workers is not supported")
        if self.pin_memory_device != "":
            raise NotImplementedError("DataLoader: pin_memory_device is not supported")

        self._prefetch_device_index: int | None = None
        if not self.prefetch_to_device:
            if self.device is not None:
                raise ValueError("DataLoader: device requires prefetch_to_device=True")
            if self.non_blocking:
                raise ValueError("DataLoader: non_blocking requires prefetch_to_device=True")
        else:
            # Deterministic validation order:
            # 1) Explicit CPU device is always invalid for device prefetch.
            if self.device is not None and str(self.device) == "cpu":
                raise ValueError(
                    "DataLoader: device must be a CUDA device when prefetch_to_device=True"
                )

            # 2) Fail fast if CUDA is not available.
            from vibetensor.torch import cuda as _vcuda

            _vcuda._ensure_cuda_available()

            # 3) Parse explicit CUDA device (defer device=None to iterator creation time).
            if self.device is not None:
                from vibetensor.torch.factory import _parse_device

                dev_type, dev_idx = _parse_device(self.device)
                if dev_type != "cuda" or dev_idx is None:
                    raise ValueError(
                        "DataLoader: device must be a CUDA device when prefetch_to_device=True"
                    )
                self._prefetch_device_index = int(dev_idx)

        if batch_size is None and self.drop_last:
            raise ValueError("DataLoader: drop_last is invalid when batch_size=None")

        if self._dataset_is_iterable:
            # v1: IterableDataset supports only num_workers==0 and forbids samplers.
            if sampler is not None or batch_sampler is not None:
                raise ValueError("DataLoader: sampler and batch_sampler are not supported for IterableDataset")
            if shuffle not in (None, False):
                raise ValueError("DataLoader: shuffle is not supported for IterableDataset")
        else:
            # Map-style invariants (PyTorch parity).
            if sampler is not None and shuffle not in (None, False):
                raise ValueError("DataLoader: sampler is mutually exclusive with shuffle")

            if batch_sampler is not None:
                # PyTorch-shaped invariant: batch_sampler is mutually exclusive with
                # batch_size, shuffle, sampler, and drop_last.
                # Note: like PyTorch, we cannot distinguish an explicit batch_size=1
                # from the default; so we enforce batch_size==1.
                if batch_size != 1 or sampler is not None or shuffle not in (None, False) or self.drop_last:
                    raise ValueError(
                        "DataLoader: batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last"
                    )
                # Match PyTorch convention: batch_sampler owns batching.
                self.batch_size = None

        # Auto-collation is enabled unless batch_size is None and batch_sampler is None.
        self._auto_collation = self.batch_sampler is not None or self.batch_size is not None

        if collate_fn is None:
            self.collate_fn = default_collate if self._auto_collation else default_convert
        else:
            self.collate_fn = collate_fn

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)

        if self._auto_collation:
            raise NotImplementedError(
                "DataLoader: num_workers>0 requires batch_size=None and batch_sampler=None"
            )
        return _MultiThreadDataLoaderIter(self)

    def __len__(self) -> int:
        # Map-style only.
        if self._dataset_is_iterable:
            raise TypeError("len(DataLoader) is not defined for IterableDataset")

        # If the user provided a batch_sampler, defer to it.
        if self.batch_sampler is not None:
            try:
                return len(self.batch_sampler)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("len(DataLoader): batch sampler has no length")

        # Determine the number of samples.
        if self.sampler is not None:
            try:
                n = len(self.sampler)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("len(DataLoader): sampler has no length")
        else:
            n = len(self.dataset)

        if self.batch_size is None:
            return int(n)

        bs = int(self.batch_size)
        if self.drop_last:
            return int(n) // bs
        return (int(n) + bs - 1) // bs


class _SingleProcessDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataset = loader.dataset
        self._collate_fn = loader.collate_fn
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._batch_size = loader.batch_size
        self._dataset_is_iterable = loader._dataset_is_iterable

        # Base seed is derived once per iterator (epoch).
        self._base_seed: int = _derive_base_seed(loader.generator, salt=id(self))

        self._prefetch_device_index: int | None = None
        self._prefetch_stream = None
        if loader.prefetch_to_device:
            from vibetensor.torch import cuda as _vcuda
            from vibetensor.torch.factory import _parse_device

            if loader._prefetch_device_index is not None:
                device_index = int(loader._prefetch_device_index)
            else:
                dev_type, dev_idx = _parse_device("cuda")
                if dev_type != "cuda" or dev_idx is None:
                    raise RuntimeError("DataLoader internal error: expected a CUDA device index")
                device_index = int(dev_idx)

            self._prefetch_device_index = int(device_index)
            self._prefetch_stream = _vcuda.Stream(device=int(device_index))

        if self._dataset_is_iterable:
            self._data_iter = iter(self._dataset)
            self._index_iter = None
            self._batch_iter = None
        else:
            self._data_iter = None

            if loader.batch_sampler is not None:
                self._batch_iter = iter(loader.batch_sampler)
                self._index_iter = None
            else:
                # Determine index sampler for this epoch.
                if loader.sampler is not None:
                    index_sampler = loader.sampler
                else:
                    if bool(loader.shuffle):
                        index_sampler = RandomSampler(loader.dataset, seed=self._base_seed)
                    else:
                        index_sampler = SequentialSampler(loader.dataset)

                if loader.batch_size is None:
                    self._batch_iter = None
                    self._index_iter = iter(index_sampler)
                else:
                    batch_sampler = BatchSampler(index_sampler, int(loader.batch_size), loader.drop_last)
                    self._batch_iter = iter(batch_sampler)
                    self._index_iter = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._dataset_is_iterable:
            out = self._next_iterable()
        else:
            out = self._next_map()

        if self._loader.pin_memory:
            out = pin_memory_batch(out)

        if not self._loader.prefetch_to_device:
            return out

        from vibetensor.torch import cuda as _vcuda

        device_index = self._prefetch_device_index
        prefetch_stream = self._prefetch_stream
        if device_index is None or prefetch_stream is None:
            raise RuntimeError("DataLoader internal error: missing device-prefetch state")

        # Save/restore the per-device current stream so this iterator does not
        # leak stream state into user code.
        prev = _vcuda.Stream.current(device=int(device_index))
        _vcuda.Stream.set_current(prefetch_stream)

        hold_cuda_refs: list[object] = []
        try:
            cuda_out = prefetch_to_device_batch(
                out,
                device_index=int(device_index),
                non_blocking=bool(self._loader.non_blocking),
                hold_cuda_refs=hold_cuda_refs,
            )
            ready = _vcuda.Event()
            ready.record(prefetch_stream)
        except BaseException:
            # If we started async work, ensure it completes before dropping refs.
            if self._loader.non_blocking and hold_cuda_refs:
                try:
                    prefetch_stream.synchronize()
                except BaseException:
                    pass
            raise
        finally:
            try:
                _vcuda.Stream.set_current(prev)
            except BaseException:
                pass

        cur = _vcuda.Stream.current(device=int(device_index))
        ready.wait(cur)
        return cuda_out

    def _next_map(self):
        # Auto-collation uses a batch sampler; otherwise use index sampler.
        if self._auto_collation:
            if self._batch_iter is None:
                raise RuntimeError("DataLoader internal error: missing batch sampler")
            indices = next(self._batch_iter)  # may raise StopIteration
            samples = _map_fetch(self._dataset, list(indices))
            return self._collate_fn(samples)

        # No auto-batching.
        if self._index_iter is None:
            raise RuntimeError("DataLoader internal error: missing sampler")
        idx = next(self._index_iter)  # may raise StopIteration
        sample = self._dataset[int(idx)]
        return self._collate_fn(sample)

    def _next_iterable(self):
        if self._data_iter is None:
            raise RuntimeError("DataLoader internal error: missing iterable iterator")

        if not self._auto_collation:
            sample = next(self._data_iter)  # may raise StopIteration
            return self._collate_fn(sample)

        # Auto-batching for iterable datasets.
        if self._batch_size is None:
            raise RuntimeError("DataLoader internal error: batch_size must be set when auto_collation")

        bs = int(self._batch_size)
        batch: list[Any] = []
        while len(batch) < bs:
            try:
                batch.append(next(self._data_iter))
            except StopIteration:
                if batch and not self._drop_last:
                    return self._collate_fn(batch)
                raise
        return self._collate_fn(batch)




@dataclass(frozen=True)
class WorkTask:
    send_idx: int
    indices: list[object]


@dataclass(frozen=True)
class DataMsg:
    send_idx: int
    data: object


@dataclass(frozen=True)
class _PrefetchedToDevice:
    data: object
    ready_event: _CudaEvent


_MAX_WORKER_EXC_REPR_CHARS = 512
_MAX_WORKER_TB_STR_CHARS = 4096


def _guarded_truncated_repr(e: BaseException) -> str:
    try:
        s = repr(e)
    except BaseException:
        s = "<repr failed>"

    if len(s) > _MAX_WORKER_EXC_REPR_CHARS:
        suffix = "..."
        keep = max(0, _MAX_WORKER_EXC_REPR_CHARS - len(suffix))
        s = s[:keep] + suffix

    return s


class ErrorState:
    """Shared first-error record for threaded DataLoader workers.

    Stores string-only fields to avoid retaining frames/tracebacks/locals.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_set: bool = False

        # String-only payload (no exception objects / traceback objects).
        self.worker_id: int = -1
        self.exc_type: str = ""
        self.exc_repr: str = ""
        self.tb_str: str = ""

    def is_set(self) -> bool:
        with self._lock:
            return self._is_set

    def try_set(self, worker_id: int, exc: BaseException) -> None:
        exc_type = type(exc).__name__
        exc_repr = _guarded_truncated_repr(exc)

        tb_str = ""
        try:
            import traceback

            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except BaseException:
            tb_str = ""

        if len(tb_str) > _MAX_WORKER_TB_STR_CHARS:
            suffix = "..."
            keep = max(0, _MAX_WORKER_TB_STR_CHARS - len(suffix))
            tb_str = tb_str[:keep] + suffix

        with self._lock:
            if self._is_set:
                return
            self._is_set = True
            self.worker_id = int(worker_id)
            self.exc_type = str(exc_type)
            self.exc_repr = str(exc_repr)
            self.tb_str = str(tb_str)

    def as_runtime_error(self) -> RuntimeError:
        with self._lock:
            if not self._is_set:
                return RuntimeError("DataLoader: unknown worker failure")

            msg = f"DataLoader worker {self.worker_id} failed: {self.exc_type}: {self.exc_repr}"
            if self.tb_str:
                msg = msg + "\n" + self.tb_str
            return RuntimeError(msg)


class PinErrorState:
    """Shared first-error record for the pin_memory stage.

    Stores string-only fields to avoid retaining frames/tracebacks/locals.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_set: bool = False

        # String-only payload (no exception objects / traceback objects).
        self.exc_type: str = ""
        self.exc_repr: str = ""
        self.tb_str: str = ""

    def is_set(self) -> bool:
        with self._lock:
            return self._is_set

    def try_set(self, exc: BaseException) -> None:
        exc_type = type(exc).__name__
        exc_repr = _guarded_truncated_repr(exc)

        tb_str = ""
        try:
            import traceback

            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except BaseException:
            tb_str = ""

        if len(tb_str) > _MAX_WORKER_TB_STR_CHARS:
            suffix = "..."
            keep = max(0, _MAX_WORKER_TB_STR_CHARS - len(suffix))
            tb_str = tb_str[:keep] + suffix

        with self._lock:
            if self._is_set:
                return
            self._is_set = True
            self.exc_type = str(exc_type)
            self.exc_repr = str(exc_repr)
            self.tb_str = str(tb_str)

    def as_exception(self) -> BaseException:
        with self._lock:
            if not self._is_set:
                return RuntimeError("DataLoader: unknown pin_memory failure")

            msg = f"DataLoader pin_memory stage failed: {self.exc_type}: {self.exc_repr}"
            if self.tb_str:
                msg = msg + "\n" + self.tb_str

            if self.exc_type == "TypeError":
                return TypeError(msg)
            return RuntimeError(msg)


class DeviceErrorState:
    """Shared first-error record for the device prefetch stage.

    Stores string-only fields to avoid retaining frames/tracebacks/locals.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_set: bool = False

        # String-only payload (no exception objects / traceback objects).
        self.exc_type: str = ""
        self.exc_repr: str = ""
        self.tb_str: str = ""

    def is_set(self) -> bool:
        with self._lock:
            return self._is_set

    def try_set(self, exc: BaseException) -> None:
        exc_type = type(exc).__name__
        exc_repr = _guarded_truncated_repr(exc)

        tb_str = ""
        try:
            import traceback

            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except BaseException:
            tb_str = ""

        if len(tb_str) > _MAX_WORKER_TB_STR_CHARS:
            suffix = "..."
            keep = max(0, _MAX_WORKER_TB_STR_CHARS - len(suffix))
            tb_str = tb_str[:keep] + suffix

        with self._lock:
            if self._is_set:
                return
            self._is_set = True
            self.exc_type = str(exc_type)
            self.exc_repr = str(exc_repr)
            self.tb_str = str(tb_str)

    def as_exception(self) -> BaseException:
        with self._lock:
            if not self._is_set:
                return RuntimeError("DataLoader: unknown device prefetch failure")

            msg = f"DataLoader device prefetch stage failed: {self.exc_type}: {self.exc_repr}"
            if self.tb_str:
                msg = msg + "\n" + self.tb_str

            if self.exc_type == "TypeError":
                return TypeError(msg)
            if self.exc_type == "ValueError":
                return ValueError(msg)
            return RuntimeError(msg)


def _worker_loop(
    worker_id: int,
    index_q: ClosableQueue[WorkTask],
    data_q: ClosableQueue[DataMsg],
    dataset: Any,
    collate_fn,
    error_state: ErrorState,
    *,
    base_seed: int,
    num_workers: int,
    worker_init_fn,
) -> None:
    # NOTE: keep this function module-level to avoid capturing the iterator object.

    worker_seed = (int(base_seed) + int(worker_id)) & _U64_MASK
    _set_worker_info(
        WorkerInfo(
            id=int(worker_id),
            num_workers=int(num_workers),
            seed=int(worker_seed),
            dataset=dataset,
        )
    )

    task: object | None = None

    try:
        if worker_init_fn is not None:
            worker_init_fn(int(worker_id))

        while True:
            task = index_q.get()
            if task is QueueClosed:
                return

            if not isinstance(task, WorkTask):
                raise RuntimeError("DataLoader internal error: expected WorkTask")
            if len(task.indices) != 1:
                raise RuntimeError("DataLoader internal error: expected single-index task")

            idx = int(task.indices[0])
            sample = dataset[idx]
            out = collate_fn(sample)

            ok = data_q.put(DataMsg(task.send_idx, out))
            if not ok:
                return

    except BaseException as e:
        # Wake consumer and stop other workers.
        error_state.try_set(int(worker_id), e)
        data_q.close()
        index_q.close()
        return

    finally:
        _set_worker_info(None)


def _pin_memory_loop(
    stage0_q: ClosableQueue[DataMsg],
    data_q: ClosableQueue[DataMsg],
    index_q: ClosableQueue[WorkTask],
    pin_error_state: PinErrorState,
) -> None:
    # NOTE: keep this function module-level to avoid capturing the iterator object.
    try:
        while True:
            msg = stage0_q.get()
            if msg is QueueClosed:
                data_q.close()
                return

            if not isinstance(msg, DataMsg):
                raise RuntimeError("DataLoader internal error: expected DataMsg")

            pinned = pin_memory_batch(msg.data)
            ok = data_q.put(DataMsg(msg.send_idx, pinned))
            if not ok:
                # Downstream closed; propagate upstream closure.
                stage0_q.close()
                index_q.close()
                return

    except BaseException as e:
        # Wake consumer and stop workers.
        pin_error_state.try_set(e)
        # Downstream -> upstream close.
        data_q.close()
        stage0_q.close()
        index_q.close()
        return


def _device_prefetch_loop(
    in_q: ClosableQueue[DataMsg],
    out_q: ClosableQueue[DataMsg],
    stage0_q: ClosableQueue[DataMsg],
    index_q: ClosableQueue[WorkTask],
    device_error_state: DeviceErrorState,
    *,
    device_index: int,
    non_blocking: bool,
    ack_q: ClosableQueue[int] | None,
) -> None:
    # NOTE: keep this function module-level to avoid capturing the iterator object.

    prefetch_stream = None
    inflight_by_send_idx: dict[int, _PrefetchedToDevice] = {}
    try:
        from vibetensor.torch import cuda as _vcuda

        prefetch_stream = _vcuda.Stream(device=int(device_index))
        _vcuda.Stream.set_current(prefetch_stream)

        while True:
            # Drain yield acknowledgements so we can release device-side buffers
            # once the consumer has waited on the fence.
            if ack_q is not None:
                while True:
                    try:
                        ack = ack_q.get(timeout=0.0)
                    except queue.Empty:
                        break
                    if ack is QueueClosed:
                        break
                    inflight_by_send_idx.pop(int(ack), None)

            msg = in_q.get()
            if msg is QueueClosed:
                out_q.close()
                return

            if not isinstance(msg, DataMsg):
                raise RuntimeError("DataLoader internal error: expected DataMsg")

            hold_cuda_refs: list[object] = []
            cuda_batch = prefetch_to_device_batch(
                msg.data,
                device_index=int(device_index),
                non_blocking=bool(non_blocking),
                hold_cuda_refs=hold_cuda_refs,
            )
            ready = _vcuda.Event()
            ready.record(prefetch_stream)
            payload = _PrefetchedToDevice(data=cuda_batch, ready_event=ready)

            if non_blocking:
                inflight_by_send_idx[int(msg.send_idx)] = payload

            ok = out_q.put(DataMsg(msg.send_idx, payload))
            if not ok:
                # Downstream closed; propagate upstream closure.
                in_q.close()
                stage0_q.close()
                index_q.close()
                return

    except BaseException as e:
        # Wake consumer and stop workers.
        device_error_state.try_set(e)
        # Downstream -> upstream close.
        out_q.close()
        in_q.close()
        stage0_q.close()
        index_q.close()
        return

    finally:
        if non_blocking and prefetch_stream is not None:
            # non_blocking=True requires lifetime safety: ensure async work is
            # complete before this thread drops its last references to destination
            # tensors.
            try:
                prefetch_stream.synchronize()
            except BaseException:
                pass
        if non_blocking:
            inflight_by_send_idx.clear()


_MT_JOIN_BUDGET_S = 0.5  # best-effort join budget (daemon workers)
_REORDER_MISSING = object()


def _finalize_mt_iter(
    index_q: ClosableQueue[WorkTask],
    stage0_q: ClosableQueue[DataMsg],
    stage1_q: ClosableQueue[DataMsg],
    data_q: ClosableQueue[DataMsg],
    ack_q: ClosableQueue[int] | None,
) -> None:
    # NOTE: keep this function module-level to avoid capturing the iterator object
    # and preventing collection. Finalizers must not block indefinitely.
    try:
        # Downstream -> upstream close.
        data_q.close()
    except BaseException:
        pass
    if ack_q is not None:
        try:
            ack_q.close()
        except BaseException:
            pass
    try:
        stage1_q.close()
    except BaseException:
        pass
    try:
        stage0_q.close()
    except BaseException:
        pass
    try:
        index_q.close()
    except BaseException:
        pass


class _MultiThreadDataLoaderIter:
    """Threaded map-style DataLoader iterator .

    Notes:
    - Map-style only and no auto-collation (batch_size=None, batch_sampler=None).
    - in_order=True yields in submission order via a small reorder buffer keyed by send_idx;
      in_order=False yields first-ready.
    - Prefetch window is currently W=num_workers (prefetch_factor is ignored for now).
    - Breaking early (not exhausting the iterator) triggers best-effort cleanup via a
      `weakref.finalize` GC finalizer that closes queues (non-blocking). Worker threads
      are daemon to avoid process-exit hangs; joins are best-effort.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._dataset = loader.dataset
        self._collate_fn = loader.collate_fn
        self._num_workers = int(loader.num_workers)
        self._in_order = bool(loader.in_order)

        self._timeout_s: float = float(loader.timeout)
        # timeout==0.0 means no timeout; do not call get(timeout=0.0).
        self._data_q_timeout_arg: float | None = None if self._timeout_s == 0.0 else self._timeout_s

        self._reorder: dict[int, object] = {}

        # Should have been rejected by DataLoader.__iter__.
        if loader._dataset_is_iterable:
            raise RuntimeError("DataLoader internal error: threaded iterator requires map-style dataset")
        if loader._auto_collation:
            raise RuntimeError("DataLoader internal error: threaded iterator requires no auto-collation")

        # will implement full backpressure with max_tasks=num_workers*prefetch_factor.
        self._prefetch_factor = loader.prefetch_factor

        # Base seed is derived once per iterator (epoch).
        self._base_seed: int = _derive_base_seed(loader.generator, salt=id(self))

        # Determine index sampler for this epoch.
        if loader.sampler is not None:
            index_sampler = loader.sampler
        else:
            if bool(loader.shuffle):
                index_sampler = RandomSampler(loader.dataset, seed=self._base_seed)
            else:
                index_sampler = SequentialSampler(loader.dataset)

        self._index_iter = iter(index_sampler)

        self._prefetch_to_device = bool(loader.prefetch_to_device)
        self._non_blocking = bool(loader.non_blocking)
        self._prefetch_device_index: int | None = None
        self._device_ack_q: ClosableQueue[int] | None = None
        if self._prefetch_to_device:
            from vibetensor.torch.factory import _parse_device

            if loader._prefetch_device_index is not None:
                device_index = int(loader._prefetch_device_index)
            else:
                dev_type, dev_idx = _parse_device("cuda")
                if dev_type != "cuda" or dev_idx is None:
                    raise RuntimeError("DataLoader internal error: expected a CUDA device index")
                device_index = int(dev_idx)

            self._prefetch_device_index = int(device_index)

            if self._non_blocking:
                self._device_ack_q = ClosableQueue()

        self._index_q: ClosableQueue[WorkTask] = ClosableQueue()
        self._data_q: ClosableQueue[DataMsg] = ClosableQueue()

        # Consumer always reads from _data_q.
        self._stage0_q: ClosableQueue[DataMsg]
        need_stage0_q = bool(loader.pin_memory) or bool(self._prefetch_to_device)
        if need_stage0_q:
            self._stage0_q = ClosableQueue()
        else:
            # No intermediate stages; workers write directly to the consumer queue.
            self._stage0_q = self._data_q

        self._stage1_q: ClosableQueue[DataMsg]

        self._pin_thread: threading.Thread | None = None
        self._pin_error_state = PinErrorState()
        if loader.pin_memory:
            if self._prefetch_to_device:
                self._stage1_q = ClosableQueue()
                pin_out_q = self._stage1_q
            else:
                self._stage1_q = self._data_q
                pin_out_q = self._data_q

            self._pin_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(self._stage0_q, pin_out_q, self._index_q, self._pin_error_state),
                daemon=True,
                name="vbt_dataloader_pin_memory",
            )
            self._pin_thread.start()
        else:
            self._stage1_q = self._stage0_q

        self._device_thread: threading.Thread | None = None
        self._device_error_state = DeviceErrorState()
        if self._prefetch_to_device:
            device_index = self._prefetch_device_index
            if device_index is None:
                raise RuntimeError("DataLoader internal error: missing device-prefetch index")

            self._device_thread = threading.Thread(
                target=_device_prefetch_loop,
                args=(
                    self._stage1_q,
                    self._data_q,
                    self._stage0_q,
                    self._index_q,
                    self._device_error_state,
                ),
                kwargs={
                    "device_index": int(device_index),
                    "non_blocking": bool(self._non_blocking),
                    "ack_q": self._device_ack_q,
                },
                daemon=True,
                name="vbt_dataloader_prefetch_to_device",
            )
            self._device_thread.start()

        self._error_state = ErrorState()

        self._workers: list[threading.Thread] = []

        self._send_idx: int = 0
        self._sent: int = 0
        self._yielded: int = 0
        self._done_sending: bool = False

        self._terminal_exc: BaseException | None = None
        self._terminal_stop: bool = False

        self._shutdown_lock = threading.Lock()
        self._shutdown_started: bool = False
        self._shutdown_reason: str | None = None  # debug-only

        # GC finalizer: if the iterator is dropped without close()/exhaustion, ensure
        # workers are signaled to stop via queue closure (non-blocking).
        self._finalizer: weakref.finalize | None = weakref.finalize(
            self,
            _finalize_mt_iter,
            self._index_q,
            self._stage0_q,
            self._stage1_q,
            self._data_q,
            self._device_ack_q,
        )

        for wid in range(self._num_workers):
            t = threading.Thread(
                target=_worker_loop,
                args=(wid, self._index_q, self._stage0_q, self._dataset, self._collate_fn, self._error_state),
                kwargs={
                    "base_seed": self._base_seed,
                    "num_workers": self._num_workers,
                    "worker_init_fn": loader.worker_init_fn,
                },
                daemon=True,
                name=f"vbt_dataloader_worker_{wid}",
            )
            t.start()
            self._workers.append(t)

    def __iter__(self):
        return self

    def _raise_if_terminal(self) -> None:
        if self._terminal_exc is not None:
            raise self._terminal_exc
        if self._terminal_stop:
            raise StopIteration

    def _raise_after_shutdown(self) -> None:
        self._raise_if_terminal()
        raise RuntimeError("DataLoader internal error: shutdown without terminal state")

    def _wait_and_unwrap(self, payload: object, *, send_idx: int) -> object:
        sid = int(send_idx)
        if not isinstance(payload, _PrefetchedToDevice):
            return payload

        from vibetensor.torch import cuda as _vcuda

        device_index = self._prefetch_device_index
        if device_index is None:
            raise RuntimeError("DataLoader internal error: missing device-prefetch index")

        cur = _vcuda.Stream.current(device=int(device_index))
        payload.ready_event.wait(cur)

        ack_q = self._device_ack_q
        if ack_q is not None:
            # Best-effort: if shutdown already closed the ack queue, ignore.
            try:
                ack_q.put(int(sid))
            except BaseException:
                pass

        return payload.data

    def _shutdown(
        self,
        reason: str,
        *,
        terminal_exc: BaseException | None = None,
        terminal_stop: bool = False,
    ) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._shutdown_reason = str(reason)

            # Compute terminal state (error wins if already known).
            if self._terminal_exc is None:
                if self._device_error_state.is_set():
                    self._terminal_exc = self._device_error_state.as_exception()
                elif self._pin_error_state.is_set():
                    self._terminal_exc = self._pin_error_state.as_exception()
                elif self._error_state.is_set():
                    self._terminal_exc = self._error_state.as_runtime_error()
                elif terminal_exc is not None:
                    self._terminal_exc = terminal_exc
                elif terminal_stop:
                    self._terminal_stop = True

            # Detach finalizer on explicit shutdown (close/exhaustion/timeout/error).
            f = self._finalizer
            self._finalizer = None
            if f is not None and getattr(f, "alive", False):
                try:
                    f.detach()
                except BaseException:
                    pass

            # NOTE: Terminal state must be set BEFORE closing _data_q, so a blocked
            # next() that wakes on QueueClosed can consult it.

            # Downstream -> upstream close.
            self._data_q.close()
            self._stage1_q.close()
            self._stage0_q.close()
            self._index_q.close()
            if self._device_ack_q is not None:
                self._device_ack_q.close()
            self._reorder.clear()

            threads: list[threading.Thread] = []
            if self._device_thread is not None:
                threads.append(self._device_thread)
            if self._pin_thread is not None:
                threads.append(self._pin_thread)
            threads.extend(list(self._workers))

        # Best-effort join; never block indefinitely (threads are daemon).
        deadline = time.monotonic() + _MT_JOIN_BUDGET_S
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                t.join(timeout=remaining)
            except BaseException:
                break

    def close(self) -> None:
        self._shutdown("user_close", terminal_stop=True)

    def _fill_window(self) -> None:
        W = self._num_workers

        while (not self._done_sending) and ((self._sent - self._yielded) < W):
            try:
                raw = next(self._index_iter)
            except StopIteration:
                self._done_sending = True
                self._index_q.close()
                return

            task = WorkTask(send_idx=self._send_idx, indices=[raw])
            ok = self._index_q.put(task)
            if not ok:
                # Treat as a shutdown signal (e.g. worker error closing queues).
                # Avoid raising here to not mask a worker error; __next__ will observe the failure.
                return

            self._send_idx += 1
            self._sent += 1

    def __next__(self):
        # Terminal replay.
        self._raise_if_terminal()

        # Fill prefetch window.
        self._fill_window()

        while True:
            # Terminal replay inside the loop to handle races like user close().
            self._raise_if_terminal()

            # Error wins: if any worker or stage thread recorded an error, raise promptly
            # (unless a prior close() already made the iterator terminal-stop).
            if self._device_error_state.is_set():
                self._shutdown("device_error")
                self._raise_after_shutdown()

            if self._pin_error_state.is_set():
                self._shutdown("pin_error")
                self._raise_after_shutdown()

            if self._error_state.is_set():
                self._shutdown("worker_error")
                self._raise_after_shutdown()

            # in_order fast path: yield buffered expected if present.
            if self._in_order:
                expected = self._yielded
                data = self._reorder.pop(expected, _REORDER_MISSING)
                if data is not _REORDER_MISSING:
                    self._yielded += 1
                    self._fill_window()
                    self._raise_if_terminal()
                    return self._wait_and_unwrap(data, send_idx=expected)

            # Done predicate MUST be checked before any blocking get.
            inflight = self._sent - self._yielded
            if self._done_sending and inflight == 0:
                self._shutdown("exhausted", terminal_stop=True)
                self._raise_after_shutdown()

            try:
                if self._data_q_timeout_arg is None:
                    msg = self._data_q.get()
                else:
                    msg = self._data_q.get(timeout=self._data_q_timeout_arg)
            except queue.Empty:
                exc = RuntimeError(f"DataLoader timed out after {self._timeout_s} seconds")
                self._shutdown("timeout", terminal_exc=exc)
                self._raise_after_shutdown()

            # A close() may have happened while we were blocked in get(); even after
            # close, get() can still return buffered items.
            self._raise_if_terminal()

            # Stage errors can also race with get().
            if self._device_error_state.is_set():
                self._shutdown("device_error")
                self._raise_after_shutdown()

            if self._pin_error_state.is_set():
                self._shutdown("pin_error")
                self._raise_after_shutdown()

            if self._error_state.is_set():
                self._shutdown("worker_error")
                self._raise_after_shutdown()

            if msg is QueueClosed:
                # QueueClosed can come from: user close, worker error, exhaustion, or
                # unexpected exit. Terminal flags must be consulted first to avoid
                # misclassifying user close as an unexpected worker exit.
                self._raise_if_terminal()

                if self._device_error_state.is_set():
                    self._shutdown("device_error")
                    self._raise_after_shutdown()

                if self._pin_error_state.is_set():
                    self._shutdown("pin_error")
                    self._raise_after_shutdown()

                if self._error_state.is_set():
                    self._shutdown("worker_error")
                    self._raise_after_shutdown()

                inflight = self._sent - self._yielded
                if self._done_sending and inflight == 0:
                    self._shutdown("exhausted", terminal_stop=True)
                    self._raise_after_shutdown()

                exc = RuntimeError("DataLoader: worker exited unexpectedly")
                self._shutdown("unexpected_exit", terminal_exc=exc)
                self._raise_after_shutdown()

            if not isinstance(msg, DataMsg):
                exc = RuntimeError("DataLoader internal error: expected DataMsg")
                self._shutdown("internal_error", terminal_exc=exc)
                self._raise_after_shutdown()

            if not self._in_order:
                self._yielded += 1
                self._fill_window()
                self._raise_if_terminal()
                return self._wait_and_unwrap(msg.data, send_idx=int(msg.send_idx))

            sid = int(msg.send_idx)
            if sid < self._yielded:
                exc = RuntimeError("DataLoader internal error: stale send_idx")
                self._shutdown("internal_error", terminal_exc=exc)
                self._raise_after_shutdown()
            if sid in self._reorder:
                exc = RuntimeError("DataLoader internal error: duplicate send_idx")
                self._shutdown("internal_error", terminal_exc=exc)
                self._raise_after_shutdown()

            if sid == self._yielded:
                self._yielded += 1
                self._fill_window()
                self._raise_if_terminal()
                return self._wait_and_unwrap(msg.data, send_idx=sid)

            self._reorder[sid] = msg.data
