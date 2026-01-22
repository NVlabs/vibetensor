# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random as _random
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar


T_co = TypeVar("T_co", covariant=True)


_U64_MASK = (1 << 64) - 1


def _next_seed_from_generator(gen: Any, *, who: str) -> int:
    state = gen.get_state()
    if not isinstance(state, (bytes, bytearray)) or len(state) != 16:
        raise RuntimeError(f"{who}: generator.get_state() must return 16 bytes")

    g_seed = int.from_bytes(state[0:8], "little", signed=False)
    offset = int.from_bytes(state[8:16], "little", signed=False)

    out_seed = int((g_seed + offset) & _U64_MASK)
    offset2 = int((offset + 1) & _U64_MASK)
    gen.set_state(g_seed.to_bytes(8, "little") + offset2.to_bytes(8, "little"))
    return out_seed


class Sampler(Generic[T_co]):
    """Minimal sampler base class."""

    def __iter__(self) -> Iterator[T_co]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - optional
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    def __init__(self, data_source: Sequence[object]):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Random permutation sampler.

    If ``seed`` is provided, each iteration yields a deterministic permutation.

    If ``generator`` is provided and ``seed`` is not, we derive a per-iteration
    seed from the generator's 16-byte state ``[seed:u64][offset:u64]`` and bump
    the offset by +1 (u64 wrap). This mirrors the DataLoader generator policy.
    """

    def __init__(
        self,
        data_source: Sequence[object],
        *,
        generator: Any | None = None,
        seed: int | None = None,
    ):
        self.data_source = data_source
        self.generator = generator
        self.seed = seed

        if self.seed is None and self.generator is not None:
            # Validate only when we'll use the generator.
            from vibetensor.torch.factory import _validate_generator

            _validate_generator(self.generator, "cpu")
            if not callable(getattr(self.generator, "get_state", None)) or not callable(
                getattr(self.generator, "set_state", None)
            ):
                raise TypeError("RandomSampler: generator must implement get_state() and set_state()")

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        idx = list(range(n))

        seed = self.seed
        if seed is None and self.generator is not None:
            seed = _next_seed_from_generator(self.generator, who="RandomSampler")

        rng = _random.Random(seed) if seed is not None else _random.Random()
        rng.shuffle(idx)
        return iter(idx)

    def __len__(self) -> int:
        return len(self.data_source)


class BatchSampler(Sampler[list[int]]):
    def __init__(self, sampler: Iterable[int], batch_size: int, drop_last: bool):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        # Best-effort length; may fail if sampler lacks __len__.
        n = len(self.sampler)  # type: ignore[arg-type]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
