# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_len_map_style_batching_semantics():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    assert len(vtd.DataLoader(ds, batch_size=4, drop_last=False)) == 3
    assert len(vtd.DataLoader(ds, batch_size=4, drop_last=True)) == 2
    assert len(vtd.DataLoader(ds, batch_size=None, drop_last=False)) == 10


def test_len_uses_batch_sampler_len_when_provided():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    bs = vtd.BatchSampler(vtd.SequentialSampler(ds), batch_size=3, drop_last=False)
    assert len(vtd.DataLoader(ds, batch_sampler=bs)) == len(bs)


def test_len_errors_for_iterable_dataset():
    import vibetensor.torch.utils.data as vtd

    class _Iter(vtd.IterableDataset[int]):
        def __iter__(self):
            return iter(range(5))

    it = _Iter()
    with pytest.raises(TypeError):
        len(vtd.DataLoader(it, batch_size=2))


def test_len_errors_when_sampler_has_no_len():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    class _NoLenSampler:
        def __iter__(self):
            return iter(range(10))

    ds = _DS()
    loader = vtd.DataLoader(ds, batch_size=None, sampler=_NoLenSampler())

    with pytest.raises(TypeError) as exc:
        len(loader)

    assert "len(DataLoader): sampler has no length" in str(exc.value)


def test_len_errors_when_batch_sampler_has_no_len():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    class _NoLenBatchSampler:
        def __iter__(self):
            yield [0]

    ds = _DS()
    loader = vtd.DataLoader(ds, batch_sampler=_NoLenBatchSampler())

    with pytest.raises(TypeError) as exc:
        len(loader)

    assert "len(DataLoader): batch sampler has no length" in str(exc.value)
