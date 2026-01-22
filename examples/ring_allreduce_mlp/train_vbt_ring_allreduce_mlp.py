#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import vibetensor
from vibetensor import _C as C
import vibetensor.torch as vt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import vibetensor.fabric as fabric
except Exception:
    fabric = None  # type: ignore[assignment]



try:
    from vibe_kernels.gemm import vbt_native as gemm
    from vibe_kernels.activation import vbt_native as act
    from vibe_kernels.loss import vbt_native as loss_ops
except Exception as e:
    raise SystemExit(
        "Failed to import vibe_kernels (requires triton). "
        "Please ensure triton is installed and vibetensor is built with CUDA. "
        f"Original error: {e}"
    )


def _find_ring_allreduce_plugin() -> str | None:
    so_name = "libvbt_ring_allreduce.so"
    candidates = [
        Path.cwd() / so_name,
        _REPO_ROOT / so_name,
        _REPO_ROOT / "build-py" / so_name,
    ]

    env = os.environ.get("VBT_RING_ALLREDUCE_PLUGIN_PATH")
    if env:
        return env

    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _device_cc(dev: int) -> tuple[int, int]:
    major, minor = C._cuda_device_cc(int(dev))
    return int(major), int(minor)



def _call_ring_allreduce(world_size: int, outs: list, ins: list, tpl):
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
    raise ValueError(f"unsupported world_size: {world_size} (expected 2/4/8)")


def _sync_devices(devices: list[int]) -> None:
    for dev in devices:
        vt.cuda.Stream.current(device=int(dev)).synchronize()


def _fabric_ring_allreduce_sum(tensors: list, devices: list[int]):
    """Ring allreduce (sum) implemented in Python using Fabric cross-device add.

    This is a correctness-oriented fallback for non-SM103 hardware where the
    CUTLASS ring_allreduce plugin is unavailable.

    Algorithm: circulate full buffers around the ring and accumulate locally.
    Complexity: O(world_size^2) bytes (fine for small shapes).
    """

    world_size = len(tensors)
    if world_size != len(devices):
        raise ValueError("tensors/devices length mismatch")
    if world_size == 0:
        raise ValueError("empty tensors")
    if world_size == 1:
        return list(tensors)

    if fabric is None:
        raise RuntimeError("vibetensor.fabric is unavailable; cannot run fabric ring allreduce")

    # Ensure producer kernels finished before other devices read.
    _sync_devices(devices)

    # Zeros on each device (used to materialize a remote tensor on the local device).
    shape = [int(s) for s in tensors[0].sizes]
    dtype = str(tensors[0].dtype)
    zeros = [C._cuda_zeros(shape, dtype, int(dev)) for dev in devices]

    acc = list(tensors)
    sbuf = list(tensors)

    for _step in range(world_size - 1):
        new_sbuf = [None] * world_size
        for r, dev in enumerate(devices):
            prev = (r - 1) % world_size
            recv = fabric.add(
                sbuf[prev],
                zeros[r],
                primary=int(dev),
                require_fabric=False,
                use_copy_fallback=True,
            )
            acc[r] = vt.ops.vt.add(acc[r], recv)
            new_sbuf[r] = recv
        sbuf = new_sbuf
        _sync_devices(devices)

    return acc


def _parse_devices_arg(devices_str: str | None, world_size: int) -> list[int]:
    ndev = int(C._cuda_device_count())
    if world_size > ndev:
        raise SystemExit(f"Requested world_size={world_size} but only {ndev} CUDA devices are visible")

    if devices_str is None:
        return list(range(world_size))

    devs = [int(x) for x in devices_str.split(",") if x.strip()]
    if len(devs) != world_size:
        raise SystemExit("--devices must have exactly --world-size comma-separated entries")
    if len(set(devs)) != len(devs):
        raise SystemExit("--devices must be unique")

    for d in devs:
        if d < 0 or d >= ndev:
            raise SystemExit(f"device index out of range: {d} (visible devices: 0..{ndev-1})")

    return devs


def _all_devices_are_sm103(devices: list[int]) -> bool:
    for d in devices:
        major, minor = _device_cc(d)
        if not (major == 10 and minor == 3):
            return False
    return True


def _make_batch(rng: np.random.Generator, *, batch_size: int, in_dim: int, w_true: np.ndarray):
    x = rng.normal(0.0, 1.0, size=(int(batch_size), int(in_dim))).astype(np.float32)
    logits = x @ w_true
    y = logits.argmax(axis=1).astype(np.int64)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Single-process multi-GPU data-parallel training on synthetic data, "
            "synchronizing gradients via ring reduction (plugin on SM103, else Fabric ring fallback)."
        ),
    )
    ap.add_argument("--world-size", type=int, default=2, help="Number of GPUs (ranks) to use")
    ap.add_argument("--devices", type=str, default=None, help="CUDA device indices, e.g. '0,1'")
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "plugin", "fabric_ring"],
        help="Gradient allreduce backend. 'plugin' requires SM103 GPUs.",
    )
    ap.add_argument(
        "--plugin-path",
        type=str,
        default=None,
        help="Override path to libvbt_ring_allreduce.so (else uses VBT_RING_ALLREDUCE_PLUGIN_PATH / build-py).",
    )
    ap.add_argument("--num-step", type=int, default=None, help="Number of training steps (iterations) to run")
    ap.add_argument("--steps", type=int, default=200, help="Deprecated alias for --num-step")
    ap.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--in-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not vibetensor._C._has_cuda or int(vibetensor._C._cuda_device_count()) == 0:
        raise SystemExit("CUDA is required for this example")

    world_size = int(args.world_size)
    if world_size <= 0:
        raise SystemExit(f"--world-size must be >= 1; got {world_size}")
    devices = _parse_devices_arg(args.devices, world_size)

    backend = str(args.backend)
    plugin_path = None if args.plugin_path is None else str(args.plugin_path)
    if plugin_path is None:
        plugin_path = _find_ring_allreduce_plugin()

    can_use_plugin = (
        plugin_path is not None
        and world_size in (2, 4, 8)
        and _all_devices_are_sm103(devices)
    )

    if backend == "auto":
        backend = "plugin" if can_use_plugin else "fabric_ring"

    if backend == "plugin":
        if world_size not in (2, 4, 8):
            raise SystemExit(f"plugin backend only supports world_size in {{2,4,8}}; got {world_size}")
        if plugin_path is None:
            raise SystemExit(
                "ring_allreduce plugin not found. Build it and set VBT_RING_ALLREDUCE_PLUGIN_PATH, "
                "or pass --plugin-path /path/to/libvbt_ring_allreduce.so"
            )
        if not can_use_plugin:
            ccs = {int(d): _device_cc(int(d)) for d in devices}
            raise SystemExit(f"plugin backend requires SM103 GPUs (cc=10.3); got device ccs={ccs}")

        # Load plugin (idempotent)
        try:
            vt.ops.load_library(plugin_path)
        except ValueError as e:
            if "plugin already loaded:" not in str(e):
                raise

        tpl = C._cpu_full([], "int64", 0)
    else:
        tpl = None

    print(f"allreduce backend: {backend}", flush=True)

    seed = int(args.seed)
    in_dim = int(args.in_dim)
    hidden_dim = int(args.hidden_dim)
    num_classes = int(args.num_classes)

    # Fixed teacher weights for synthetic labels.
    rng_true = np.random.default_rng(seed + 123)
    w_true = rng_true.normal(0.0, 1.0, size=(in_dim, num_classes)).astype(np.float32)

    # Model init (identical across ranks).
    rng_init = np.random.default_rng(seed)
    w1_init = (rng_init.normal(0.0, 1.0, size=(in_dim, hidden_dim)).astype(np.float32)) * 0.02
    w2_init = (rng_init.normal(0.0, 1.0, size=(hidden_dim, num_classes)).astype(np.float32)) * 0.02

    w1 = [vt.cuda.to_device(w1_init, device=dev) for dev in devices]  # type: ignore[attr-defined]
    w2 = [vt.cuda.to_device(w2_init, device=dev) for dev in devices]  # type: ignore[attr-defined]

    # Allreduce output buffers (reused each step).
    ar_w1 = [C._cuda_empty([in_dim, hidden_dim], "float32", int(dev)) for dev in devices]
    ar_w2 = [C._cuda_empty([hidden_dim, num_classes], "float32", int(dev)) for dev in devices]

    rngs = [np.random.default_rng(seed + 1000 + r) for r in range(world_size)]

    lr = float(args.lr)
    scale = lr / float(world_size)

    t0 = time.time()

    steps = int(args.steps if args.num_step is None else args.num_step)
    for step in range(1, steps + 1):
        step_t0 = time.perf_counter()

        losses: list[float] = []
        grad_w1: list = []
        grad_w2: list = []

        # Per-rank forward/backward.
        for r, dev in enumerate(devices):
            # IMPORTANT: vibe_kernels (Triton PTX load/launch) relies on the CUDA
            # *current device* for driver API calls (cuModuleLoadData/cuLaunchKernel).
            # Always set the current device before compiling/loading/launching.
            with vt.cuda.Stream.current(device=int(dev)):
                x_np, y_np = _make_batch(
                    rngs[r],
                    batch_size=int(args.batch_size),
                    in_dim=in_dim,
                    w_true=w_true,
                )
                x = vt.cuda.to_device(x_np, device=dev)  # type: ignore[attr-defined]
                y = vt.cuda.to_device(y_np, device=dev)  # type: ignore[attr-defined]

                h_pre = gemm.matmul(x, w1[r])
                h = act.gelu(h_pre)
                logits = gemm.matmul(h, w2[r])

                loss_val, ce_cache = loss_ops.cross_entropy_with_cache(logits, y)
                losses.append(float(loss_val))

                dlogits = loss_ops.cross_entropy_backward(ce_cache)
                dh, dw2 = gemm.matmul_backward(dlogits, h, w2[r], compute_grad_a=True, compute_grad_b=True)
                dh_pre = act.gelu_backward(dh, h_pre)
                _, dw1 = gemm.matmul_backward(dh_pre, x, w1[r], compute_grad_a=False, compute_grad_b=True)

                grad_w1.append(dw1)
                grad_w2.append(dw2)

        # Ring allreduce sum of gradients across ranks.
        if backend == "plugin":
            assert tpl is not None
            st1 = _call_ring_allreduce(world_size, ar_w1, grad_w1, tpl)
            st2 = _call_ring_allreduce(world_size, ar_w2, grad_w2, tpl)
            if int(st1.numpy().item()) != 0 or int(st2.numpy().item()) != 0:
                raise RuntimeError("ring_allreduce plugin returned non-zero status")
            sum_w1 = ar_w1
            sum_w2 = ar_w2
        else:
            sum_w1 = _fabric_ring_allreduce_sum(grad_w1, devices)
            sum_w2 = _fabric_ring_allreduce_sum(grad_w2, devices)

        # SGD update on each rank using the averaged gradient.
        for r in range(world_size):
            w1[r] = vt.ops.vt.sub(w1[r], vt.ops.vt.mul(sum_w1[r], scale))
            w2[r] = vt.ops.vt.sub(w2[r], vt.ops.vt.mul(sum_w2[r], scale))

        if step % int(args.log_every) == 0 or step == 1 or step == steps:
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            avg_loss = float(sum(losses) / float(world_size))

            msg = f"step {step:4d}/{steps} | loss={avg_loss:.4f} | iter_ms={iter_ms:.2f}"

            print(msg, flush=True)

    dt = time.time() - t0
    print(f"done | wall_sec={dt:.2f}")


if __name__ == "__main__":
    main()
