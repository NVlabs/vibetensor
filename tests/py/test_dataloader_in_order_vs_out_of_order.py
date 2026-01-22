# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading


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


class _GateDS:
    def __init__(self):
        self.started0 = threading.Event()
        self.release0 = threading.Event()
        self.done1 = threading.Event()

    def __len__(self):
        return 2

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx == 0:
            self.started0.set()
            self.release0.wait()
            return 0
        if idx == 1:
            # Ensure idx=0 has started so idx=1 deterministically completes first.
            self.started0.wait()
            self.done1.set()
            return 1
        raise IndexError(idx)


def test_dataloader_in_order_false_yields_completion_order():
    import vibetensor.torch.utils.data as vtd

    ds = _GateDS()
    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        first = run_with_timeout(lambda: next(it), timeout=2.0)
        assert first == 1

        ds.release0.set()

        second = run_with_timeout(lambda: next(it), timeout=2.0)
        assert second == 0
    finally:
        ds.release0.set()
        it.close()


def test_dataloader_in_order_true_yields_submission_order():
    import vibetensor.torch.utils.data as vtd

    ds = _GateDS()
    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=True,
        collate_fn=lambda x: x,
    )

    it = iter(dl)

    out: dict[str, object] = {}
    done = threading.Event()

    def _t() -> None:
        try:
            out["value"] = next(it)
        except BaseException as e:
            out["exc"] = e
        finally:
            done.set()

    t = threading.Thread(target=_t, daemon=True)
    t.start()

    try:
        assert ds.done1.wait(timeout=2.0)
        ds.release0.set()

        assert done.wait(timeout=2.0)
        if "exc" in out:
            raise out["exc"]  # type: ignore[misc]
        assert out.get("value") == 0

        second = run_with_timeout(lambda: next(it), timeout=2.0)
        assert second == 1
    finally:
        ds.release0.set()
        it.close()
