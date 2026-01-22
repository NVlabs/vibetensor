# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import queue
import threading

import pytest

from vibetensor.torch.utils.data._queue import ClosableQueue, QueueClosed


def test_closable_queue_put_returns_false_when_closed():
    q: ClosableQueue[int] = ClosableQueue()

    assert q.put(1) is True
    q.close()
    assert q.put(2) is False


def test_closable_queue_get_blocks_until_put():
    q: ClosableQueue[int] = ClosableQueue()

    ready = threading.Event()
    done = threading.Event()
    out: dict[str, object] = {}

    def _worker() -> None:
        ready.set()
        out["v"] = q.get()
        done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    assert ready.wait(timeout=1.0)
    assert not done.wait(timeout=0.2)

    assert q.put(123) is True

    assert done.wait(timeout=1.0)
    assert out["v"] == 123


def test_closable_queue_get_unblocks_on_close():
    q: ClosableQueue[int] = ClosableQueue()

    ready = threading.Event()
    done = threading.Event()
    out: dict[str, object] = {}

    def _worker() -> None:
        ready.set()
        out["v"] = q.get()
        done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    assert ready.wait(timeout=1.0)
    assert not done.wait(timeout=0.2)

    q.close()

    assert done.wait(timeout=1.0)
    assert out["v"] is QueueClosed


def test_closable_queue_get_drains_items_after_close_then_returns_sentinel():
    q: ClosableQueue[int] = ClosableQueue()

    assert q.put(1) is True
    assert q.put(2) is True

    q.close()

    assert q.get() == 1
    assert q.get() == 2
    assert q.get() is QueueClosed


def test_closable_queue_close_is_idempotent_and_wakes_all_blocked_getters():
    q: ClosableQueue[int] = ClosableQueue()

    ready1 = threading.Event()
    ready2 = threading.Event()
    done1 = threading.Event()
    done2 = threading.Event()
    out: dict[str, object] = {}

    def _w1() -> None:
        ready1.set()
        out["a"] = q.get()
        done1.set()

    def _w2() -> None:
        ready2.set()
        out["b"] = q.get()
        done2.set()

    t1 = threading.Thread(target=_w1, daemon=True)
    t2 = threading.Thread(target=_w2, daemon=True)
    t1.start()
    t2.start()

    assert ready1.wait(timeout=1.0)
    assert ready2.wait(timeout=1.0)
    assert not done1.wait(timeout=0.2)
    assert not done2.wait(timeout=0.2)

    q.close()
    q.close()  # idempotent

    assert done1.wait(timeout=1.0)
    assert done2.wait(timeout=1.0)
    assert out["a"] is QueueClosed
    assert out["b"] is QueueClosed


def test_closable_queue_get_timeout_raises_empty_when_open():
    q: ClosableQueue[int] = ClosableQueue()

    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)
