# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


def _decode_state(state: bytes) -> tuple[int, int]:
    assert isinstance(state, (bytes, bytearray))
    assert len(state) == 16
    seed = int.from_bytes(state[0:8], "little", signed=False)
    offset = int.from_bytes(state[8:16], "little", signed=False)
    return seed, offset


def test_shuffle_determinism_across_epochs_and_reseed():
    import vibetensor.torch.rng as rng
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 25

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    gen = rng.Generator("cpu")
    gen.manual_seed(123)

    _, off0 = _decode_state(gen.get_state())

    loader = vtd.DataLoader(ds, batch_size=None, shuffle=True, generator=gen)
    order1 = [int(x.numpy()) for x in loader]
    _, off1 = _decode_state(gen.get_state())
    assert off1 == off0 + 1

    order2 = [int(x.numpy()) for x in loader]
    _, off2 = _decode_state(gen.get_state())
    assert off2 == off0 + 2

    assert sorted(order1) == list(range(len(ds)))
    assert sorted(order2) == list(range(len(ds)))
    assert order1 != order2

    # Reseeding reproduces the same epoch-to-epoch orders.
    gen.manual_seed(123)
    loader2 = vtd.DataLoader(ds, batch_size=None, shuffle=True, generator=gen)
    order1b = [int(x.numpy()) for x in loader2]
    order2b = [int(x.numpy()) for x in loader2]

    assert order1b == order1
    assert order2b == order2
