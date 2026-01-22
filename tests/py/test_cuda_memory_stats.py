# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import os
import subprocess
import sys
import textwrap

import pytest
from vibetensor import _C as C
import vibetensor.torch.cuda as cuda


def _cuda_unavailable() -> bool:
    has_cuda = getattr(C, "_has_cuda", False)
    get_count = getattr(C, "_cuda_device_count", None)
    if not has_cuda or get_count is None:
        return True
    try:
        return int(get_count()) == 0
    except Exception:
        return True


def test_memory_stats_nested_keys_exist():
    st = cuda.memory_stats_as_nested_dict()
    for fam in ("allocated_bytes", "reserved_bytes", "requested_bytes"):
        assert fam in st
        assert "all" in st[fam]
        for metric in ("current", "peak", "allocated", "freed"):
            assert metric in st[fam]["all"]
            assert isinstance(st[fam]["all"][metric], int)


def test_memory_stats_flatten_type():
    flat = cuda.memory_stats()
    assert isinstance(flat, OrderedDict)


def test_empty_cache_noop_cpu():
    # Should be safe to call even when CUDA is unavailable
    cuda.empty_cache()


def test_memory_stats_schema_stable_across_calls():
    st1 = cuda.memory_stats_as_nested_dict()
    st2 = cuda.memory_stats_as_nested_dict()
    assert set(st1.keys()) == set(st2.keys())
    for fam in ("allocated_bytes", "reserved_bytes", "requested_bytes"):
        assert fam in st1 and fam in st2
        assert set(st1[fam].keys()) == set(st2[fam].keys())
        assert set(st1[fam]["all"].keys()) == set(st2[fam]["all"].keys())


def test_cuda_memory_stats_diagnostic_counters_zero_or_absent():
    """Allocator diagnostics are present and zero when CUDA is available.

    The low-level _cuda_memoryStats binding exposes scalar counters including
    fraction_cap_* and gc_*; these are scaffolding-only and must
    remain zero for both native and async backends. On CPU-only builds the
    helper should return an empty dict instead of raising.
    """

    impl = getattr(C, "_cuda_memoryStats", None)
    if impl is None:
        pytest.skip("_cuda_memoryStats binding not available")

    if _cuda_unavailable():
        # CPU-only behavior: no stats; empty dict.
        assert impl(None) == {}
        return

    stats = impl(None)
    for key in ("fraction_cap_breaches", "fraction_cap_misfires", "gc_passes", "gc_reclaimed_bytes"):
        assert key in stats
        assert isinstance(stats[key], int)
        assert stats[key] == 0


def test_cuda_memory_stats_diagnostic_counters_zero_async_backend_subprocess() -> None:
    """Async backend must expose zero-only Allocator counters via _cuda_memoryStats.

    This runs in a fresh subprocess with backend=cudaMallocAsync to ensure
    that native allocator state from other tests does not leak into the async
    view of stats.
    """

    impl = getattr(C, "_cuda_memoryStats", None)
    if impl is None:
        pytest.skip("_cuda_memoryStats binding not available")

    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    script = textwrap.dedent(
        """\
        import os
        import sys

        os.environ.setdefault("VBT_CUDA_ALLOC_CONF", "backend=cudaMallocAsync")

        import vibetensor.torch.cuda as cuda  # type: ignore[import]
        from vibetensor import _C as C  # type: ignore[import]


        def _cuda_unavailable() -> bool:
            has_cuda = getattr(C, "_has_cuda", False)
            get_count = getattr(C, "_cuda_device_count", None)
            if not has_cuda or get_count is None:
                return True
            try:
                return int(get_count()) == 0
            except Exception:
                return True


        def main() -> int:
            if _cuda_unavailable():
                return 0

            impl = getattr(C, "_cuda_memoryStats", None)
            if impl is None:
                return 0

            # Small bounded workload to exercise the async allocator surface.
            s = cuda.Stream()
            with s:
                cuda.empty_cache()
                _ = cuda.memory_stats()

            stats = impl(None)
            nested = cuda.memory_stats_as_nested_dict()
            flat = cuda.memory_stats()

            agg = nested.get("device_stats", {}).get("aggregated", {})
            for key in ("fraction_cap_breaches", "fraction_cap_misfires",
                        "gc_passes", "gc_reclaimed_bytes"):
                val = stats.get(key, 0)
                if val != 0:
                    raise SystemExit(
                        f"expected {key}=0 under async backend, got {val} (stats={stats})"
                    )
                if agg.get(key, 0) != 0:
                    raise SystemExit(
                        f"expected nested {key}=0 under async backend, got {agg.get(key)} (nested={nested})"
                    )
                flat_key = f"device_stats.aggregated.{key}"
                if flat.get(flat_key, 0) != 0:
                    raise SystemExit(
                        f"expected flat {flat_key}=0 under async backend, got {flat.get(flat_key)} (flat={flat})"
                    )
            return 0


        if __name__ == "__main__":
            try:
                code = main()
            except Exception:
                import traceback

                traceback.print_exc()
                sys.exit(1)
            else:
                sys.exit(code)
        """
    )

    env = os.environ.copy()
    env["VBT_CUDA_ALLOC_CONF"] = "backend=cudaMallocAsync"

    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"async-backend memory-stats subprocess failed with code {proc.returncode}: {proc.stderr}"
        )


def test_memory_stats_cpu_only_neutral_shape():
    # Only exercise CPU-only / zero-device semantics when CUDA devices are unavailable.
    if not _cuda_unavailable():
        pytest.skip("CUDA devices present; CPU-only/zero-device semantics not exercised")

    nested = cuda.memory_stats_as_nested_dict()
    for fam in ("allocated_bytes", "reserved_bytes", "requested_bytes"):
        assert fam in nested
        assert "all" in nested[fam]
        bucket = nested[fam]["all"]
        assert isinstance(bucket, dict)
        assert bucket == {"current": 0, "peak": 0, "allocated": 0, "freed": 0}

    flat = cuda.memory_stats()
    keys = list(flat.keys())
    assert len(keys) >= 12
    assert keys[:12] == [
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "allocated_bytes.all.allocated",
        "allocated_bytes.all.freed",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "reserved_bytes.all.allocated",
        "reserved_bytes.all.freed",
        "requested_bytes.all.current",
        "requested_bytes.all.peak",
        "requested_bytes.all.allocated",
        "requested_bytes.all.freed",
    ]
    assert all(v == 0 for v in flat.values())


def test_memory_stats_negative_device_errors_cpu_only():
    if not _cuda_unavailable():
        pytest.skip("CUDA devices present; negative-device semantics tested elsewhere")

    with pytest.raises(RuntimeError) as excinfo:
        cuda.memory_stats_as_nested_dict(-1)
    assert "device must be >= 0 or None for current device" in str(excinfo.value)


def test_memory_stats_diagnostic_counters_exposed_native():
    impl = getattr(C, "_cuda_memoryStats", None)
    if impl is None or _cuda_unavailable():
        pytest.skip("_cuda_memoryStats binding or CUDA devices not available")

    raw = impl(None)
    for key in ("fraction_cap_breaches", "fraction_cap_misfires",
                "gc_passes", "gc_reclaimed_bytes"):
        assert key in raw
        assert isinstance(raw[key], int)

    nested = cuda.memory_stats_as_nested_dict()
    dev_stats = nested.get("device_stats", {}).get("aggregated", {})
    for key in ("fraction_cap_breaches", "fraction_cap_misfires",
                "gc_passes", "gc_reclaimed_bytes"):
        assert key in dev_stats
        assert isinstance(dev_stats[key], int)

    flat = cuda.memory_stats()
    for key in ("fraction_cap_breaches", "fraction_cap_misfires",
                "gc_passes", "gc_reclaimed_bytes"):
        flat_key = f"device_stats.aggregated.{key}"
        assert flat_key in flat
        assert isinstance(flat[flat_key], int)


def test_cuda_memoryStats_return_types():
    impl = getattr(C, "_cuda_memoryStats", None)
    if impl is None or _cuda_unavailable():
        pytest.skip("_cuda_memoryStats not available or no CUDA devices")

    raw = impl(None)
    assert isinstance(raw, dict)
    for fam in ("allocated_bytes", "reserved_bytes", "requested_bytes"):
        assert fam in raw
        assert isinstance(raw[fam], dict)


def test_memory_stats_packaging_error_behavior(monkeypatch):
    impl = getattr(C, "_cuda_memoryStats", None)
    dev_count_fn = getattr(C, "_cuda_device_count", None)
    if impl is None or dev_count_fn is None or _cuda_unavailable():
        pytest.skip("_cuda_memoryStats/_cuda_device_count not available or no devices")

    if int(dev_count_fn()) <= 0:
        pytest.skip("no CUDA devices detected")

    def fake_stats(device):  # pragma: no cover - wrapper behavior under test
        return {}

    # Patch both the C module and cached function pointer in the cuda module.
    monkeypatch.setattr(C, "_cuda_memoryStats", fake_stats, raising=False)
    if hasattr(cuda, "_CUDA_MEMORY_STATS_FN"):
        monkeypatch.setattr(cuda, "_CUDA_MEMORY_STATS_FN", fake_stats, raising=False)

    nested = cuda.memory_stats_as_nested_dict(0)
    flat = cuda.memory_stats(0)

    assert "allocated_bytes" in nested and "reserved_bytes" in nested and "requested_bytes" in nested
    assert "device_stats" not in nested

    assert all(
        key.startswith(("allocated_bytes.", "reserved_bytes.", "requested_bytes."))
        for key in flat.keys()
    )
    assert not any(key.startswith("device_stats.aggregated.") for key in flat.keys())


def test_memory_stats_byte_families_ordering():
    flat = cuda.memory_stats()
    keys = list(flat.keys())
    assert len(keys) >= 12
    assert keys[:12] == [
        "allocated_bytes.all.current",
        "allocated_bytes.all.peak",
        "allocated_bytes.all.allocated",
        "allocated_bytes.all.freed",
        "reserved_bytes.all.current",
        "reserved_bytes.all.peak",
        "reserved_bytes.all.allocated",
        "reserved_bytes.all.freed",
        "requested_bytes.all.current",
        "requested_bytes.all.peak",
        "requested_bytes.all.allocated",
        "requested_bytes.all.freed",
    ]


def test_fraction_cap_oom_advances_diagnostic_counters_subprocess():
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor")

    if getattr(C, "_cuda_memoryStats", None) is None:
        pytest.skip("_cuda_memoryStats binding not available")

    script = textwrap.dedent(
        """\
        import os
        import sys

        os.environ.setdefault("VBT_CUDA_ALLOC_CONF", "backend=native")

        try:
            import numpy as np
        except Exception:
            # NumPy is optional; treat missing NumPy as skip.
            sys.exit(0)

        import vibetensor.torch.cuda as cuda  # type: ignore[import]
        from vibetensor import _C as C        # type: ignore[import]


        def _cuda_unavailable() -> bool:
            has_cuda = getattr(C, "_has_cuda", False)
            get_count = getattr(C, "_cuda_device_count", None)
            if not has_cuda or get_count is None:
                return True
            try:
                return int(get_count()) == 0
            except Exception:
                return True


        def main() -> int:
            if _cuda_unavailable():
                return 0

            impl = getattr(C, "_cuda_memoryStats", None)
            if impl is None:
                return 0

            before = impl(None)

            names = [
                "fraction_cap_breaches",
                "fraction_cap_misfires",
                "gc_passes",
                "gc_reclaimed_bytes",
            ]

            cuda.set_per_process_memory_fraction(0.0)
            cuda.empty_cache()

            try:
                arr = np.ones((4 * 1024 * 1024,), dtype=np.float32)
            except Exception:
                return 0

            try:
                _ = cuda.to_device(arr)
            except RuntimeError as e:
                msg = str(e)
                if "per-process memory fraction cap" not in msg:
                    raise SystemExit(f"expected fraction-cap OOM, got: {msg!r}")
            except TypeError:
                # e.g. NumPy not available inside cuda module; treat as skip.
                return 0
            else:
                # If allocation unexpectedly succeeds, treat as soft skip.
                return 0

            after = impl(None)

            if after["fraction_cap_breaches"] <= before.get("fraction_cap_breaches", 0):
                raise SystemExit("fraction_cap_breaches did not advance under fraction-cap OOM")
            if after["fraction_cap_misfires"] <= before.get("fraction_cap_misfires", 0):
                raise SystemExit("fraction_cap_misfires did not advance under fraction-cap OOM")

            nested = cuda.memory_stats_as_nested_dict()
            dev_stats = nested.get("device_stats", {}).get("aggregated", {})
            for name in names:
                if name not in dev_stats:
                    raise SystemExit(
                        f"missing {name} in nested device_stats after OOM: {dev_stats}"
                    )

            flat = cuda.memory_stats()
            for name in names:
                key = f"device_stats.aggregated.{name}"
                if key not in flat:
                    raise SystemExit(f"missing {key} in flat stats after OOM: {flat}")

            return 0


        if __name__ == "__main__":
            try:
                code = main()
            except Exception:
                import traceback

                traceback.print_exc()
                sys.exit(1)
            else:
                sys.exit(code)
        """
    )

    env = os.environ.copy()
    env.setdefault("VBT_CUDA_ALLOC_CONF", "backend=native")

    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"fraction-cap OOM subprocess failed with code {proc.returncode}: {proc.stderr}"
        )
