# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest


def run_with_timeout(fn, timeout: float = 1.0):
    done = threading.Event()
    out: dict[str, object] = {}

    def _t() -> None:
        try:
            out["value"] = fn()
        except BaseException as e:
            out["exc"] = e
        finally:
            done.set()

    t = threading.Thread(target=_t, daemon=True)
    t.start()

    assert done.wait(timeout=timeout)

    if "exc" in out:
        raise out["exc"]  # type: ignore[misc]

    return out.get("value")


class _DS:
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self):
        return int(self._n)

    def __getitem__(self, idx: int):
        return int(idx)


def test_dataloader_basic_threaded_map_smoke_yields_all_samples_and_terminates():
    import vibetensor.torch.utils.data as vtd

    N = 25
    ds = _DS(N)

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
    )

    def _run():
        it = iter(dl)
        assert hasattr(it, "_workers")
        workers = getattr(it, "_workers")
        assert len(workers) == dl.num_workers
        assert all(t.daemon for t in workers)
        assert all(t.name.startswith("vbt_dataloader_worker_") for t in workers)
        return list(it)

    out = run_with_timeout(_run, timeout=2.0)
    assert sorted(out) == list(range(N))


def test_dataloader_basic_threaded_map_empty_dataset_does_not_hang():
    import vibetensor.torch.utils.data as vtd

    ds = _DS(0)

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
    )

    out = run_with_timeout(lambda: list(iter(dl)), timeout=2.0)
    assert out == []


def test_dataloader_basic_threaded_map_in_order_true_is_allowed():
    import vibetensor.torch.utils.data as vtd

    ds = _DS(3)

    dl = vtd.DataLoader(
        ds,
        num_workers=1,
        batch_size=None,
        in_order=True,
        collate_fn=lambda x: x,
    )

    out = run_with_timeout(lambda: list(iter(dl)), timeout=2.0)
    assert out == [0, 1, 2]


def test_dataloader_basic_threaded_map_guardrail_auto_collation_rejected():
    import vibetensor.torch.utils.data as vtd

    ds = _DS(3)

    dl = vtd.DataLoader(ds, num_workers=1, batch_size=1, in_order=False)

    with pytest.raises(NotImplementedError) as exc:
        iter(dl)
    assert "requires batch_size=None and batch_sampler=None" in str(exc.value)


def test_dataloader_basic_threaded_map_worker_init_fn_is_called_once():
    import vibetensor.torch.utils.data as vtd

    ds = _DS(3)

    called: list[int] = []
    lock = threading.Lock()
    ready = threading.Event()

    def _init_fn(worker_id: int) -> None:
        with lock:
            called.append(int(worker_id))
        ready.set()

    dl = vtd.DataLoader(
        ds,
        num_workers=1,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
        worker_init_fn=_init_fn,
    )

    def _run():
        it = iter(dl)
        assert ready.wait(timeout=2.0)
        return list(it)

    out = run_with_timeout(_run, timeout=4.0)
    assert sorted(out) == [0, 1, 2]
    assert called == [0]
