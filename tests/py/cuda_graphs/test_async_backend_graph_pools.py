# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest

from vibetensor import _C as C


def _cuda_unavailable() -> bool:
    has_cuda = getattr(C, "_has_cuda", False)
    get_count = getattr(C, "_cuda_device_count", None)
    if not has_cuda or get_count is None:
        return True
    try:
        return int(get_count()) == 0
    except Exception:
        return True


def test_async_backend_graph_pools_smoke() -> None:
    """Smoke test that graph pools work under the async backend.

    This test spawns a fresh Python process with VBT_CUDA_ALLOC_CONF pointing
    at the cudaMallocAsync backend. Inside that process we run a small
    capture/instantiate/replay sequence using an explicit GraphPoolHandle and
    assert that basic stats are well-formed. The heavy lifting around
    per-pool free lists and busy-pool gating is covered by C++ tests; here we
    only check that the async backend wiring does not regress graph-pool
    observability.
    """

    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    script = textwrap.dedent(
        """\
        import os
        import sys

        os.environ.setdefault("VBT_CUDA_ALLOC_CONF", "backend=cudaMallocAsync")

        import vibetensor.torch.cuda as vc
        from vibetensor import _C as C
        try:
            from vibetensor.torch.cuda import graphs as vgraphs
        except ImportError:
            # CUDA Graphs Python overlay not available in this build; treat as success.
            sys.exit(0)


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
                # Treat CUDA-unavailable as success; outer test has already
                # checked availability but multi-process CI environments may
                # still vary.
                return 0

            pool = vgraphs.graph_pool_handle()
            g = vgraphs.CUDAGraph(keep_graph=True)

            s = vc.Stream()
            with s:
                with vgraphs.graph(g, pool=pool, stream=s):
                    pass
                g.instantiate()
                g.replay()
                s.synchronize()
                g.replay()
                s.synchronize()

            stats = vgraphs.cuda_graphs_stats()
            graphs_stats = stats["graphs"]

            if graphs_stats["graphs_instantiated"] < 1:
                raise SystemExit("expected at least one instantiated graph, got: " + str(graphs_stats))
            if graphs_stats["graphs_replayed"] < 2:
                raise SystemExit("expected at least two replays, got: " + str(graphs_stats))

            pool_stats = vgraphs.graph_pool_stats(pool)
            if not isinstance(pool_stats, list):
                raise SystemExit("graph_pool_stats did not return a list")

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
            f"async-backend subprocess failed with code {proc.returncode}: {proc.stderr}"
        )
