# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import math
import numbers
import queue
import threading
import time
from typing import Deque, Final, Generic, TypeVar


class _QueueClosed:
    __slots__ = ()

    def __repr__(self) -> str:
        return "QueueClosed"


QueueClosedType = _QueueClosed
QueueClosed: Final[QueueClosedType] = _QueueClosed()

T = TypeVar("T")


class ClosableQueue(Generic[T]):
    """A closeable, unbounded, blocking queue.

    Semantics (mirrors C++ vbt::autograd::ConcurrentTaskQueue):

    - put(item) -> bool: returns False if the queue is closed.
    - get(timeout=None) -> T | QueueClosed:
        - blocks while open and empty
        - returns queued items even after close
        - returns QueueClosed iff closed and empty
        - if timeout is provided and expires, raises queue.Empty
    - close(): idempotent and wakes all blocked getters.

    Notes:
    - This queue is intentionally minimal; it exists to make shutdown/error
      handling deadlock-free for the threaded DataLoader path.
    """

    def __init__(self) -> None:
        self._items: Deque[T] = collections.deque()
        self._closed: bool = False
        self._cv = threading.Condition()

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()

    def put(self, item: T) -> bool:
        with self._cv:
            if self._closed:
                return False
            self._items.append(item)
            self._cv.notify()
            return True

    def get(self, timeout: float | None = None) -> T | QueueClosedType:
        with self._cv:
            if timeout is None:
                while not self._items:
                    if self._closed:
                        return QueueClosed
                    self._cv.wait()
                return self._items.popleft()

            if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real):
                raise TypeError("timeout must be a number or None")
            timeout = float(timeout)
            if not math.isfinite(timeout):
                raise ValueError("timeout must be finite")
            if timeout < 0.0:
                raise ValueError("timeout must be non-negative")

            deadline = time.monotonic() + timeout
            while not self._items:
                if self._closed:
                    return QueueClosed

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise queue.Empty

                self._cv.wait(timeout=remaining)

            return self._items.popleft()
