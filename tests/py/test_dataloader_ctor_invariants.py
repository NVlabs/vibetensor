# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_dataloader_ctor_invariants_map_style():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    good_batch_sampler = vtd.BatchSampler(vtd.SequentialSampler(ds), batch_size=2, drop_last=False)

    cases = [
        (
            dict(batch_size=True),
            TypeError,
            "DataLoader: batch_size must be an int or None",
        ),
        (
            dict(num_workers=True),
            TypeError,
            "DataLoader: num_workers must be an int",
        ),
        (
            dict(num_workers=-1),
            ValueError,
            "DataLoader: num_workers must be >= 0",
        ),
        (
            dict(timeout=1.0),
            ValueError,
            "DataLoader: timeout must be 0 when num_workers==0",
        ),
        (
            dict(prefetch_factor=2),
            ValueError,
            "DataLoader: prefetch_factor must be None when num_workers==0",
        ),
        (
            dict(batch_size=None, drop_last=True),
            ValueError,
            "DataLoader: drop_last is invalid when batch_size=None",
        ),
        (
            dict(sampler=vtd.SequentialSampler(ds), shuffle=True),
            ValueError,
            "DataLoader: sampler is mutually exclusive with shuffle",
        ),
        (
            dict(batch_sampler=good_batch_sampler, batch_size=2),
            ValueError,
            "DataLoader: batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last",
        ),
        (
            dict(batch_sampler=good_batch_sampler, shuffle=True),
            ValueError,
            "DataLoader: batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last",
        ),
        (
            dict(batch_sampler=good_batch_sampler, sampler=vtd.SequentialSampler(ds)),
            ValueError,
            "DataLoader: batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last",
        ),
        (
            dict(batch_sampler=good_batch_sampler, drop_last=True),
            ValueError,
            "DataLoader: batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last",
        ),
        (
            dict(multiprocessing_context=object()),
            NotImplementedError,
            "DataLoader: multiprocessing_context is not supported",
        ),
        (
            dict(pin_memory_device="cuda"),
            NotImplementedError,
            "DataLoader: pin_memory_device is not supported",
        ),
    ]

    for kwargs, exc_t, msg in cases:
        with pytest.raises(exc_t) as exc:
            vtd.DataLoader(ds, **kwargs)
        assert msg in str(exc.value)


def test_dataloader_ctor_invariants_iterable_style():
    import vibetensor.torch.utils.data as vtd

    class _IterDS(vtd.IterableDataset[int]):
        def __iter__(self):
            return iter(range(3))

    ids = _IterDS()

    with pytest.raises(ValueError) as exc_sampler:
        vtd.DataLoader(ids, sampler=[0, 1, 2])
    assert "sampler and batch_sampler are not supported for IterableDataset" in str(exc_sampler.value)

    with pytest.raises(ValueError) as exc_batch_sampler:
        vtd.DataLoader(ids, batch_sampler=[[0], [1]])
    assert "sampler and batch_sampler are not supported for IterableDataset" in str(exc_batch_sampler.value)

    with pytest.raises(ValueError) as exc_shuffle:
        vtd.DataLoader(ids, shuffle=True)
    assert "shuffle is not supported for IterableDataset" in str(exc_shuffle.value)
