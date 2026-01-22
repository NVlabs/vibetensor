# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import numpy as np
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


def _find_ring_allreduce_plugin() -> str | None:
    so_name = "libvbt_ring_allreduce.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    env = os.environ.get("VBT_RING_ALLREDUCE_PLUGIN_PATH")
    if env:
        return env
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _eligible_sm103_devices() -> list[int]:
    n = int(C._cuda_device_count())
    out: list[int] = []
    for i in range(n):
        major, minor = C._cuda_device_cc(i)
        if int(major) == 10 and int(minor) == 3:
            out.append(i)
    return out


def _call_op(world_size: int, outs: list, ins: list, tpl) -> object:
    if world_size == 2:
        return vt.ops.vbt_dist.ring_allreduce_ws2(outs[0], outs[1], ins[0], ins[1], tpl)
    if world_size == 4:
        return vt.ops.vbt_dist.ring_allreduce_ws4(
            outs[0],
            outs[1],
            outs[2],
            outs[3],
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            tpl,
        )
    if world_size == 8:
        return vt.ops.vbt_dist.ring_allreduce_ws8(
            outs[0],
            outs[1],
            outs[2],
            outs[3],
            outs[4],
            outs[5],
            outs[6],
            outs[7],
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            ins[4],
            ins[5],
            ins[6],
            ins[7],
            tpl,
        )
    raise ValueError(f"unsupported world_size: {world_size}")


@pytest.mark.skipif(
    not getattr(C, "_has_cuda", False) or int(C._cuda_device_count()) == 0,
    reason="CUDA not available for VibeTensor",
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ring_allreduce_plugin_numerical(world_size: int) -> None:
    path = _find_ring_allreduce_plugin()
    if path is None:
        pytest.skip(
            "ring_allreduce plugin not found; ensure build produced libvbt_ring_allreduce.so"
        )

    # Load plugin (idempotent)
    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    eligible = _eligible_sm103_devices()
    if len(eligible) < world_size:
        pytest.skip(f"need >= {world_size} SM103 GPUs for this test")

    devs = eligible[:world_size]

    tpl = C._cpu_full([], "int64", 0)

    # Use a small but non-trivial size.
    N = 4096

    outs = []
    ins = []
    for r, dev in enumerate(devs):
        in_arr = np.full((N,), float(r + 1), dtype=np.float32)
        out_arr = np.zeros((N,), dtype=np.float32)
        ins.append(vt.cuda.to_device(in_arr, device=dev))  # type: ignore[attr-defined]
        outs.append(vt.cuda.to_device(out_arr, device=dev))  # type: ignore[attr-defined]

    ret = _call_op(world_size, outs, ins, tpl)
    assert int(ret.numpy().item()) == 0

    expected_val = float(sum(range(1, world_size + 1)))

    for r in range(world_size):
        out = vt.cuda.from_device(outs[r])  # type: ignore[attr-defined]
        inp = vt.cuda.from_device(ins[r])  # type: ignore[attr-defined]

        np.testing.assert_allclose(out, expected_val, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(inp, float(r + 1), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not getattr(C, "_has_cuda", False) or int(C._cuda_device_count()) < 2,
    reason="CUDA not available for VibeTensor",
)
def test_ring_allreduce_plugin_empty_noop() -> None:
    path = _find_ring_allreduce_plugin()
    if path is None:
        pytest.skip(
            "ring_allreduce plugin not found; ensure build produced libvbt_ring_allreduce.so"
        )

    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    eligible = _eligible_sm103_devices()
    if len(eligible) < 2:
        pytest.skip("need >=2 SM103 GPUs")

    dev0, dev1 = eligible[:2]

    tpl = C._cpu_full([], "int64", 0)

    in0 = vt.cuda.to_device(np.zeros((0,), dtype=np.float32), device=dev0)  # type: ignore[attr-defined]
    in1 = vt.cuda.to_device(np.zeros((0,), dtype=np.float32), device=dev1)  # type: ignore[attr-defined]
    out0 = vt.cuda.to_device(np.zeros((0,), dtype=np.float32), device=dev0)  # type: ignore[attr-defined]
    out1 = vt.cuda.to_device(np.zeros((0,), dtype=np.float32), device=dev1)  # type: ignore[attr-defined]

    ret = vt.ops.vbt_dist.ring_allreduce_ws2(out0, out1, in0, in1, tpl)
    assert int(ret.numpy().item()) == 0
