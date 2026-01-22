# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading


_U64_MASK = (1 << 64) - 1


def _decode_state(state: bytes) -> tuple[int, int]:
    assert isinstance(state, (bytes, bytearray))
    assert len(state) == 16
    seed = int.from_bytes(state[0:8], "little", signed=False)
    offset = int.from_bytes(state[8:16], "little", signed=False)
    return seed, offset


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


def test_dataloader_worker_info_tls_present_in_workers_and_none_in_main_thread():
    import vibetensor.torch.rng as rng
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __init__(self, n: int):
            self._n = int(n)

        def __len__(self):
            return int(self._n)

        def __getitem__(self, idx: int):
            info = vtd.get_worker_info()
            assert info is not None
            assert info.dataset is self
            return (int(idx), int(info.id), int(info.num_workers), int(info.seed))

    N = 8
    ds = _DS(N)

    gen = rng.Generator("cpu")
    gen.manual_seed(123)
    seed, offset = _decode_state(gen.get_state())
    expected_base_seed = (seed + offset) & _U64_MASK

    assert vtd.get_worker_info() is None

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
        generator=gen,
    )

    out = run_with_timeout(lambda: list(iter(dl)), timeout=4.0)
    assert vtd.get_worker_info() is None

    assert isinstance(out, list)
    assert len(out) == N

    out_sorted = sorted(out, key=lambda t: t[0])
    assert [t[0] for t in out_sorted] == list(range(N))

    for _, wid, nworkers, wseed in out_sorted:
        assert nworkers == 2
        assert wid in (0, 1)
        assert wseed == ((expected_base_seed + wid) & _U64_MASK)


def test_dataloader_worker_init_fn_called_once_per_worker_and_sees_worker_info():
    import vibetensor.torch.rng as rng
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    gen = rng.Generator("cpu")
    gen.manual_seed(999)
    seed, offset = _decode_state(gen.get_state())
    expected_base_seed = (seed + offset) & _U64_MASK

    calls: list[int] = []
    seen: dict[int, int] = {}
    lock = threading.Lock()
    ready = threading.Event()

    def _init_fn(worker_id: int) -> None:
        info = vtd.get_worker_info()
        assert info is not None
        assert info.id == int(worker_id)
        assert info.dataset is ds

        with lock:
            calls.append(int(worker_id))
            seen[int(worker_id)] = int(info.seed)
            if len(seen) == 2:
                ready.set()

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
        generator=gen,
        worker_init_fn=_init_fn,
    )

    assert vtd.get_worker_info() is None

    it = iter(dl)
    try:
        assert ready.wait(timeout=5.0)
    finally:
        it.close()

    assert sorted(calls) == [0, 1]
    assert len(calls) == 2
    assert seen[0] == ((expected_base_seed + 0) & _U64_MASK)
    assert seen[1] == ((expected_base_seed + 1) & _U64_MASK)
    assert vtd.get_worker_info() is None
