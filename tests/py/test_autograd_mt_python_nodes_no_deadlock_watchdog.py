# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest


def test_autograd_mt_python_nodes_no_deadlock_watchdog() -> None:
    """Ensure mt backward does not deadlock on Python-backed nodes.

    This spawns a subprocess so a regression manifests as a timeout rather than
    hanging the entire pytest run.
    """

    code = r'''
import threading

import vibetensor.autograd as A
import vibetensor.torch as vt

main_tid = threading.get_ident()
bw_tid = None


class Square(A.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        global bw_tid
        bw_tid = threading.get_ident()
        (x,) = ctx.saved_tensors
        return (2.0 * x * grad_out,)


A.set_multithreading_enabled(True)

x = vt.full([], 3.0, dtype="float32")
x.requires_grad = True

y = Square.apply(x)
y.backward()

g = x.grad
assert g is not None
assert float(g.item()) == 6.0

assert bw_tid is not None
assert bw_tid != main_tid
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as e:
        pytest.fail(
            "multithreaded backward hung (likely Python GIL deadlock); "
            f"stdout={getattr(e, 'stdout', b'')!r} stderr={getattr(e, 'stderr', b'')!r}"
        )

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        stdout = result.stdout.decode(errors="replace")
        raise AssertionError(
            f"subprocess failed: exit={result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
