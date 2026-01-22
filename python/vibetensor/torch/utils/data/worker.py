# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import threading


_tls = threading.local()


@dataclass(frozen=True)
class WorkerInfo:
    """Worker metadata visible inside worker threads.

    In single-process mode, :func:`get_worker_info` always returns
    ``None``.
    """

    id: int
    num_workers: int
    seed: Optional[int] = None
    dataset: Any = None


def get_worker_info() -> WorkerInfo | None:
    return getattr(_tls, "worker_info", None)


def _set_worker_info(info: WorkerInfo | None) -> None:
    # Internal helper for threaded loader.
    if info is None:
        try:
            delattr(_tls, "worker_info")
        except Exception:
            pass
        return
    _tls.worker_info = info
