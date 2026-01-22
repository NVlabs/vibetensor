#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN = _REPO_ROOT / "examples" / "ring_allreduce_mlp" / "train_vbt_ring_allreduce_mlp.py"


_STEP_RE = re.compile(r"^step\s+(\d+)/(\d+)\s+\|.*iter_ms=([0-9.]+)")


@dataclass(frozen=True)
class RunResult:
    world_size: int
    batch_size_per_gpu: int
    avg_iter_ms: float
    stdev_iter_ms: float
    throughput_samples_per_s: float


def _run_one(
    *,
    world_size: int,
    batch_size: int,
    steps: int,
    warmup_steps: int,
    backend: str,
    devices: str | None,
    in_dim: int | None,
    hidden_dim: int | None,
    num_classes: int | None,
) -> RunResult:
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be in [0, steps-1]")

    cmd = [
        sys.executable,
        str(_TRAIN),
        "--world-size",
        str(world_size),
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--backend",
        backend,
        "--log-every",
        "1",
    ]
    if devices is not None:
        cmd += ["--devices", devices]
    if in_dim is not None:
        cmd += ["--in-dim", str(in_dim)]
    if hidden_dim is not None:
        cmd += ["--hidden-dim", str(hidden_dim)]
    if num_classes is not None:
        cmd += ["--num-classes", str(num_classes)]

    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Benchmark run failed:\n"
            f"cmd: {' '.join(cmd)}\n"
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    iter_ms: list[float] = []
    for line in proc.stdout.splitlines():
        m = _STEP_RE.match(line.strip())
        if not m:
            continue
        step = int(m.group(1))
        ms = float(m.group(3))
        if step <= warmup_steps:
            continue
        iter_ms.append(ms)

    if not iter_ms:
        raise RuntimeError(f"No iter_ms parsed from output (world_size={world_size}).")

    avg_ms = statistics.fmean(iter_ms)
    stdev_ms = statistics.pstdev(iter_ms) if len(iter_ms) > 1 else 0.0
    global_batch = batch_size * world_size
    throughput = global_batch * 1000.0 / avg_ms

    return RunResult(
        world_size=world_size,
        batch_size_per_gpu=batch_size,
        avg_iter_ms=avg_ms,
        stdev_iter_ms=stdev_ms,
        throughput_samples_per_s=throughput,
    )


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=65536, help="Per-GPU batch size (ignored when --global-batch is set)")
    ap.add_argument(
        "--global-batch",
        type=int,
        default=None,
        help="If set, keep global batch size fixed and derive per-GPU batch as global_batch/world_size (must be divisible).",
    )
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=50)
    ap.add_argument(
        "--backend",
        type=str,
        default="fabric_ring",
        choices=["fabric_ring", "auto", "plugin"],
        help="Backend passed through to the training script.",
    )
    ap.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional CUDA device list to pass through, e.g. '0,1,2,3'. If unset, uses 0..world_size-1.",
    )
    ap.add_argument("--in-dim", type=int, default=None)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--num-classes", type=int, default=None)
    args = ap.parse_args()

    steps = int(args.warmup_steps) + int(args.measure_steps)
    world_sizes = [1, 2, 3, 4]

    global_batch = None if args.global_batch is None else int(args.global_batch)

    results: list[RunResult] = []
    for ws in world_sizes:
        if global_batch is None:
            bs = int(args.batch_size)
        else:
            if global_batch % ws != 0:
                raise SystemExit(
                    f"--global-batch={global_batch} must be divisible by world_size={ws} (try e.g. 196608 for 1/2/3/4)"
                )
            bs = global_batch // ws

        r = _run_one(
            world_size=ws,
            batch_size=bs,
            steps=steps,
            warmup_steps=int(args.warmup_steps),
            backend=str(args.backend),
            devices=str(args.devices) if args.devices is not None else None,
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
        )
        results.append(r)

    base = results[0]

    print("\nBenchmark: ring_allreduce_mlp scaling")
    if global_batch is None:
        batch_desc = f"batch_size(per_gpu)={args.batch_size}"
    else:
        batch_desc = f"global_batch={global_batch} (per_gpu=batch/world_size)"

    print(
        f"config: backend={args.backend} | {batch_desc} | "
        f"warmup_steps={args.warmup_steps} | measure_steps={args.measure_steps}"
    )

    print("\nworld_size | per_gpu_batch | avg_iter_ms | stdev_ms | throughput (samples/s) | speedup(vs1)")
    for r in results:
        speedup = r.throughput_samples_per_s / base.throughput_samples_per_s
        print(
            f"{r.world_size:10d} | {r.batch_size_per_gpu:12d} | {r.avg_iter_ms:10.2f} | {r.stdev_iter_ms:8.2f} | "
            f"{r.throughput_samples_per_s:20.0f} | {speedup:10.3f}"
        )


if __name__ == "__main__":
    main()
