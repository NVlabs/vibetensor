# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import queue
import time
import weakref


class _QuickDS:
    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        return int(idx)


def _wait_until(pred, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if pred():
            return
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met before timeout")
        time.sleep(0.01)


def test_dataloader_gc_finalizer_closes_queues_and_stops_workers():
    import vibetensor.torch.utils.data as vtd
    from vibetensor.torch.utils.data._queue import QueueClosed

    dl = vtd.DataLoader(
        _QuickDS(),
        num_workers=2,
        batch_size=None,
        timeout=0.0,
        collate_fn=lambda x: x,
    )

    def _make_and_drop(dl):
        it = iter(dl)
        wr = weakref.ref(it)
        workers = list(it._workers)
        index_q = it._index_q
        data_q = it._data_q
        return wr, workers, index_q, data_q

    wr, workers, index_q, data_q = _make_and_drop(dl)

    # Force collection and finalizer execution.
    deadline = time.monotonic() + 5.0
    while wr() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)

    assert wr() is None

    def _queues_closed() -> bool:
        try:
            return index_q.get(timeout=0.01) is QueueClosed and data_q.get(timeout=0.01) is QueueClosed
        except queue.Empty:
            return False

    _wait_until(_queues_closed, timeout=5.0)

    _wait_until(lambda: all(not t.is_alive() for t in workers), timeout=5.0)
