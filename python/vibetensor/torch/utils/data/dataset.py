# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generic, Iterator, TypeVar


T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    """Minimal map-style dataset base class.

    Users should implement ``__getitem__`` and (optionally) ``__len__``.

    Optionally implement ``__getitems__(indices)`` on your dataset to provide
    a batched fetch fast-path used by :class:`~vibetensor.torch.utils.data.DataLoader`.
    """

    def __getitem__(self, index: int) -> T_co:  # pragma: no cover - abstract
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError


class IterableDataset(Generic[T_co]):
    """Minimal iterable-style dataset base class."""

    def __iter__(self) -> Iterator[T_co]:  # pragma: no cover - abstract
        raise NotImplementedError
