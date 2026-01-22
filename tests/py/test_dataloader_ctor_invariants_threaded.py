# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_dataloader_ctor_invariants_threaded_iterable_dataset_rejected():
    import vibetensor.torch.utils.data as vtd

    class _IterDS(vtd.IterableDataset[int]):
        def __iter__(self):
            return iter(range(3))

    ids = _IterDS()

    with pytest.raises(NotImplementedError) as exc:
        vtd.DataLoader(ids, num_workers=1)
    assert "DataLoader: IterableDataset with num_workers>0 is not supported" in str(exc.value)


def test_dataloader_ctor_allows_threaded_timeout():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    dl = vtd.DataLoader(ds, num_workers=1, timeout=0.1)
    assert dl.timeout == 0.1


@pytest.mark.parametrize("bad_timeout", [float("nan"), float("inf"), float("-inf")])
def test_dataloader_ctor_rejects_nonfinite_timeout(bad_timeout: float):
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    with pytest.raises(ValueError) as exc:
        vtd.DataLoader(ds, num_workers=1, timeout=bad_timeout)
    assert "DataLoader: timeout must be finite" in str(exc.value)


def test_dataloader_ctor_invariants_threaded_prefetch_factor_defaults_to_2():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    dl = vtd.DataLoader(ds, num_workers=1)
    assert dl.prefetch_factor == 2


def test_dataloader_ctor_invariants_threaded_worker_init_fn_must_be_callable():
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 10

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    with pytest.raises(TypeError) as exc:
        vtd.DataLoader(ds, num_workers=1, worker_init_fn=123)
    assert "DataLoader: worker_init_fn must be callable" in str(exc.value)
