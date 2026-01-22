# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Standalone PyTorch NCCL all-reduce bandwidth benchmark.

This script measures torch.distributed (NCCL) all-reduce performance and reports
metrics similar to the CUTLASS example:
  examples/94_blackwell_ring_allreduce/94_blackwell_ring_allreduce_benchmark.cu

Key features:
- No torchrun required: uses torch.multiprocessing.spawn.
- Sweeps tensor sizes (bytes per rank) and world sizes.
- Times dist.all_reduce() using CUDA events.
- Defines per-iteration latency as the *max across ranks* (via MAX all-reduce of
  a scalar GPU tensor).
- Optional correctness spot-check via dist.all_gather() on sampled elements.

Example:
  python examples/94_blackwell_ring_allreduce/pytorch_allreduce_benchmark.py \
    --world_sizes=2,4,8 \
    --sizes=4KiB,16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB \
    --warmup_iters=10 --measure_iters=50 \
    --nccl_algo=Ring
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import socket
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires PyTorch with torch.distributed support installed."
    ) from exc


_ELEM_SIZE_BYTES = 4  # float32


@dataclasses.dataclass(frozen=True)
class SizeSpec:
    bytes_per_rank: int
    count_elems: int
    token: str


def _split_csv(s: str) -> List[str]:
    return [tok.strip() for tok in s.split(",")]


def _parse_csv_ints(s: str, *, what: str) -> List[int]:
    tokens = _split_csv(s)
    out: List[int] = []
    for tok in tokens:
        if not tok:
            raise ValueError(f"{what} contains an empty token")
        try:
            out.append(int(tok, 10))
        except ValueError as exc:
            raise ValueError(f"invalid int token in {what}: '{tok}'") from exc
    return out


def _format_bytes_binary(num_bytes: int) -> str:
    units: List[Tuple[str, int]] = [
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
        ("B", 1),
    ]

    for suffix, denom in units:
        if num_bytes >= denom and (num_bytes % denom == 0):
            return f"{num_bytes // denom}{suffix}"

    # Fallback: non-integral unit.
    if num_bytes >= 1024**3:
        return f"{num_bytes / (1024**3):.2f}GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / (1024**2):.2f}MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f}KiB"
    return f"{num_bytes}B"


def _parse_sizes_float32(sizes_raw: str) -> List[SizeSpec]:
    out: List[SizeSpec] = []
    for raw in _split_csv(sizes_raw):
        if not raw:
            raise ValueError("sizes contains an empty token")

        tok = raw.strip()
        lower = tok.lower()

        elem_suffixes = ["elements", "element", "elems", "elem", "e"]
        elem_suffix = next((suf for suf in elem_suffixes if lower.endswith(suf)), "")

        if elem_suffix:
            num = lower[: -len(elem_suffix)].strip()
            if not num:
                raise ValueError(f"invalid element-count size token: '{tok}'")
            try:
                count = int(num, 10)
            except ValueError as exc:
                raise ValueError(f"invalid element-count size token: '{tok}'") from exc
            if count <= 0:
                raise ValueError(f"invalid element-count size token: '{tok}'")
            bytes_per_rank = count * _ELEM_SIZE_BYTES
            out.append(SizeSpec(bytes_per_rank=bytes_per_rank, count_elems=count, token=tok))
            continue

        # Bytes with optional units.
        mul = 1
        num = lower

        def take_suffix(suf: str, factor: int) -> bool:
            nonlocal mul, num
            if num.endswith(suf):
                mul = factor
                num = num[: -len(suf)].strip()
                return True
            return False

        # Order matters (kib before b).
        take_suffix("kib", 1024) or take_suffix("k", 1024) or take_suffix("mib", 1024**2) or take_suffix(
            "m", 1024**2
        ) or take_suffix("gib", 1024**3) or take_suffix("g", 1024**3) or take_suffix("b", 1)

        try:
            base = int(num, 10)
        except ValueError as exc:
            raise ValueError(f"invalid byte size token: '{tok}'") from exc
        if base <= 0:
            raise ValueError(f"invalid byte size token: '{tok}'")

        bytes_per_rank = base * mul
        if bytes_per_rank % _ELEM_SIZE_BYTES != 0:
            raise ValueError(
                f"byte size is not divisible by sizeof(float): '{tok}' (bytes={bytes_per_rank})"
            )
        count_elems = bytes_per_rank // _ELEM_SIZE_BYTES
        if count_elems <= 0:
            raise ValueError(f"element count must be > 0: '{tok}'")

        out.append(SizeSpec(bytes_per_rank=bytes_per_rank, count_elems=count_elems, token=tok))

    return out


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def _make_verify_indices(count_elems: int, verify_samples: int, verify_seed: int) -> List[int]:
    n = min(int(verify_samples), int(count_elems))
    if n <= 0:
        return []

    indices: List[int] = []
    used = set()

    def add(idx: int) -> None:
        if idx not in used:
            used.add(idx)
            indices.append(idx)

    add(0)
    if n > 1:
        add(count_elems - 1)

    # Deterministic sampling using splitmix64; depends on (verify_seed, count_elems).
    x = (int(verify_seed) ^ (int(count_elems) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    while len(indices) < n:
        x = _splitmix64(x)
        add(int(x % count_elems))

    return indices


def _make_rank_seed(verify_seed: int, world_size: int, count_elems: int, rank: int) -> int:
    x = int(verify_seed) & 0xFFFFFFFFFFFFFFFF
    x ^= (int(world_size) + 1) * 0x9E3779B97F4A7C15
    x ^= int(count_elems) * 0xBF58476D1CE4E5B9
    x ^= (int(rank) + 1) * 0x94D049BB133111EB
    return _splitmix64(x)


def _print_table_header() -> None:
    print(
        f"{'world':<6}"
        f"{'bytes':<12}"
        f"{'elems':<12}"
        f"{'result':<14}"
        f"{'avg_ms':<12}"
        f"{'algbw(GB/s)':<14}"
        f"{'busbw(GB/s)':<14}"
        "devices"
    )


def _devices_to_str(devices: Sequence[int]) -> str:
    return "[" + ",".join(str(d) for d in devices) + "]"


def _print_row(
    *,
    world_size: int,
    spec: SizeSpec,
    result: str,
    avg_ms: float,
    algbw: float,
    busbw: float,
    devices: Sequence[int],
) -> None:
    if result == "OK":
        avg_s = f"{avg_ms:.3f}"
        alg_s = f"{algbw:.2f}"
        bus_s = f"{busbw:.2f}"
    else:
        avg_s = "-"
        alg_s = "-"
        bus_s = "-"

    print(
        f"{world_size:<6}"
        f"{_format_bytes_binary(spec.bytes_per_rank):<12}"
        f"{spec.count_elems:<12}"
        f"{result:<14}"
        f"{avg_s:<12}"
        f"{alg_s:<14}"
        f"{bus_s:<14}"
        f"{_devices_to_str(devices)}"
    )


def _maybe_write_csv_header(f) -> None:
    f.write(
        "world_size,dtype,bytes,element_count,warmup_iters,measure_iters,"
        "nccl_algo,nccl_proto,nccl_nchannels,result,avg_ms,algbw_GBps,busbw_GBps,devices\n"
    )


def _write_csv_row(
    *,
    f,
    world_size: int,
    spec: SizeSpec,
    warmup_iters: int,
    measure_iters: int,
    nccl_env: Dict[str, str],
    result: str,
    avg_ms: float,
    algbw: float,
    busbw: float,
    devices: Sequence[int],
) -> None:
    devs = ":".join(str(d) for d in devices)
    nccl_algo = nccl_env.get("NCCL_ALGO", "")
    nccl_proto = nccl_env.get("NCCL_PROTO", "")
    nccl_nch = nccl_env.get("NCCL_NCHANNELS", "")

    f.write(
        f"{world_size},float32,{spec.bytes_per_rank},{spec.count_elems},{warmup_iters},{measure_iters},"
        f"{nccl_algo},{nccl_proto},{nccl_nch},{result}"
    )

    if result == "OK":
        f.write(f",{avg_ms:.6f},{algbw:.6f},{busbw:.6f}")
    else:
        f.write(",,,")

    f.write(f",{devs}\n")
    f.flush()


def _run_case(
    *,
    rank: int,
    world_size: int,
    device: int,
    spec: SizeSpec,
    warmup_iters: int,
    measure_iters: int,
    verify: bool,
    verify_seed: int,
    verify_samples: int,
) -> Tuple[str, float, float, float]:
    # Allocation
    try:
        x = torch.zeros(spec.count_elems, device=device, dtype=torch.float32)
        alloc_ok = True
    except RuntimeError:
        # Likely OOM; keep ranks in sync with a dummy tensor.
        x = torch.zeros(1, device=device, dtype=torch.float32)
        alloc_ok = False

    alloc_flag = torch.tensor([1 if alloc_ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(alloc_flag, op=dist.ReduceOp.MIN)
    if int(alloc_flag.item()) == 0:
        return "OOM", 0.0, 0.0, 0.0

    if verify:
        # Fill random inputs deterministically per rank.
        gen = torch.Generator(device="cuda")
        gen.manual_seed(_make_rank_seed(verify_seed, world_size, spec.count_elems, rank))
        x.uniform_(-1.0, 1.0, generator=gen)

        indices = _make_verify_indices(spec.count_elems, verify_samples, verify_seed)
        idx = torch.tensor(indices, device=device, dtype=torch.int64)

        pre = x.index_select(0, idx).contiguous()
        gathered: List[torch.Tensor] = [torch.empty_like(pre) for _ in range(world_size)]
        dist.all_gather(gathered, pre)

        expected = torch.zeros_like(pre)
        for t in gathered:
            expected.add_(t)

        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        post = x.index_select(0, idx)

        atol = 1e-3
        rtol = 1e-3
        ok_local = torch.allclose(post, expected, atol=atol, rtol=rtol)
        ok_flag = torch.tensor([1 if ok_local else 0], device=device, dtype=torch.int32)
        dist.all_reduce(ok_flag, op=dist.ReduceOp.MIN)

        # Reset to zeros for stable timing.
        x.zero_()

        if int(ok_flag.item()) == 0:
            return "VERIFY_FAIL", 0.0, 0.0, 0.0

    # Warmup
    dist.barrier()
    for _ in range(int(warmup_iters)):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Measure
    dist.barrier()
    torch.cuda.synchronize(device=device)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    lat = torch.empty(1, device=device, dtype=torch.float32)

    total_ms = 0.0
    for _ in range(int(measure_iters)):
        start.record()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        stop.record()
        stop.synchronize()

        elapsed_ms = float(start.elapsed_time(stop))
        lat[0] = elapsed_ms
        dist.all_reduce(lat, op=dist.ReduceOp.MAX)
        total_ms += float(lat.item())

    avg_ms = total_ms / float(measure_iters)
    algbw = float(spec.bytes_per_rank) / (avg_ms / 1.0e3) / 1.0e9
    busbw = algbw * (2.0 * float(world_size - 1) / float(world_size))

    return "OK", avg_ms, algbw, busbw


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    ring_devices: List[int],
    sizes: List[Dict[str, object]],
    warmup_iters: int,
    measure_iters: int,
    verify: bool,
    verify_seed: int,
    verify_samples: int,
    csv_path: Optional[str],
    nccl_env: Dict[str, str],
    result_queue,
) -> None:
    device = int(ring_devices[rank])
    torch.cuda.set_device(device)

    # NCCL env vars must be visible before the process group initializes.
    for k, v in nccl_env.items():
        os.environ[str(k)] = str(v)

    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=int(world_size),
        rank=int(rank),
    )

    csv_f = None
    if rank == 0 and csv_path:
        # Append; write header if new/empty.
        new_file = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        csv_f = open(csv_path, "a", encoding="utf-8", newline="")
        if new_file:
            _maybe_write_csv_header(csv_f)
            csv_f.flush()

    any_bad = False

    for spec_dict in sizes:
        spec = SizeSpec(
            bytes_per_rank=int(spec_dict["bytes_per_rank"]),
            count_elems=int(spec_dict["count_elems"]),
            token=str(spec_dict["token"]),
        )

        result, avg_ms, algbw, busbw = _run_case(
            rank=rank,
            world_size=world_size,
            device=device,
            spec=spec,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            verify=verify,
            verify_seed=verify_seed,
            verify_samples=verify_samples,
        )

        if result != "OK":
            any_bad = True

        if rank == 0:
            _print_row(
                world_size=world_size,
                spec=spec,
                result=result,
                avg_ms=avg_ms,
                algbw=algbw,
                busbw=busbw,
                devices=ring_devices,
            )
            if csv_f:
                _write_csv_row(
                    f=csv_f,
                    world_size=world_size,
                    spec=spec,
                    warmup_iters=warmup_iters,
                    measure_iters=measure_iters,
                    nccl_env=nccl_env,
                    result=result,
                    avg_ms=avg_ms,
                    algbw=algbw,
                    busbw=busbw,
                    devices=ring_devices,
                )

    if csv_f:
        csv_f.close()

    dist.destroy_process_group()

    if rank == 0:
        result_queue.put(bool(any_bad))


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pytorch_allreduce_benchmark.py",
        description="Standalone PyTorch NCCL all-reduce bandwidth benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--world_sizes",
        default="2,4,8",
        help="CSV list of world sizes to benchmark (sequentially)",
    )
    p.add_argument(
        "--sizes",
        default="4KiB,16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB",
        help=(
            "CSV list of per-rank payload sizes. Supports B/KiB/MiB/GiB units, "
            "or element forms like 262144e (float32 elements)."
        ),
    )

    p.add_argument("--warmup_iters", type=int, default=10)
    p.add_argument("--measure_iters", type=int, default=50)

    p.add_argument(
        "--devices",
        default=None,
        help=(
            "Optional explicit device order as CSV. A prefix is used for each world size. "
            "Example: --devices=3,2,1,0,7,6,5,4"
        ),
    )

    p.add_argument("--csv", default=None, help="Optional CSV output path (append; header if new/empty)")

    p.add_argument("--verify", action="store_true", help="Verify numerical correctness (spot-check)")
    p.add_argument("--verify_seed", type=int, default=1)
    p.add_argument("--verify_samples", type=int, default=4096)

    p.add_argument("--nccl_algo", default="Ring", help="Sets NCCL_ALGO")
    p.add_argument("--nccl_proto", default=None, help="Sets NCCL_PROTO")
    p.add_argument("--nccl_nchannels", default=None, help="Sets NCCL_NCHANNELS")

    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    world_sizes = _parse_csv_ints(args.world_sizes, what="world_sizes")
    if not world_sizes:
        raise ValueError("world_sizes must be non-empty")

    sizes = _parse_sizes_float32(args.sizes)
    if not sizes:
        raise ValueError("sizes must be non-empty")

    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be >= 0")
    if args.measure_iters <= 0:
        raise ValueError("measure_iters must be > 0")
    if args.verify_samples <= 0:
        raise ValueError("verify_samples must be > 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (torch.cuda.is_available() is False)")

    device_count = torch.cuda.device_count()

    devices: Optional[List[int]] = None
    if args.devices is not None:
        devices = _parse_csv_ints(args.devices, what="devices")
        if len(set(devices)) != len(devices):
            raise ValueError("device list contains duplicates")
        for d in devices:
            if d < 0 or d >= device_count:
                raise ValueError(f"device id out of range: {d}")

    max_ws = max(world_sizes)
    if max_ws > device_count:
        raise RuntimeError(
            f"Requested max world size {max_ws} but only {device_count} CUDA devices are visible"
        )
    if devices is not None and len(devices) < max_ws:
        raise RuntimeError(
            f"--devices must list at least {max_ws} device IDs (got {len(devices)})"
        )

    nccl_env: Dict[str, str] = {"NCCL_ALGO": str(args.nccl_algo)}
    if args.nccl_proto is not None:
        nccl_env["NCCL_PROTO"] = str(args.nccl_proto)
    if args.nccl_nchannels is not None:
        nccl_env["NCCL_NCHANNELS"] = str(args.nccl_nchannels)

    # Ensure env vars are inherited by spawned children.
    for k, v in nccl_env.items():
        os.environ[str(k)] = str(v)

    _print_table_header()

    any_bad = False

    for ws in world_sizes:
        ring_devices = devices[:ws] if devices is not None else list(range(ws))

        port = _find_free_port()
        init_method = f"tcp://127.0.0.1:{port}"

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        mp.spawn(
            _worker,
            args=(
                int(ws),
                init_method,
                ring_devices,
                [dataclasses.asdict(s) for s in sizes],
                int(args.warmup_iters),
                int(args.measure_iters),
                bool(args.verify),
                int(args.verify_seed),
                int(args.verify_samples),
                args.csv,
                nccl_env,
                result_queue,
            ),
            nprocs=int(ws),
            join=True,
        )

        ws_bad = bool(result_queue.get())
        result_queue.close()
        result_queue.join_thread()

        if bool(args.verify):
            any_bad = any_bad or ws_bad

    if bool(args.verify) and any_bad:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
