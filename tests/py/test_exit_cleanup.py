# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for clean interpreter exit after registering Python callbacks.

These tests spawn subprocesses to verify that the atexit cleanup handlers
properly clear C++ registries holding Python objects, preventing segfaults
during interpreter finalization.
"""

import subprocess
import sys

import pytest


@pytest.mark.cuda
def test_exit_after_autograd_register():
    """Test clean exit after registering autograd backward functions."""
    code = '''
import numpy as np
import vibetensor._C as C
import vibetensor.torch as vt
import vibetensor.autograd as vag

C.autograd.set_cuda_autograd_enabled(True)

def _bw(grads_in, saved):
    return grads_in

vag.register("vt::add", _bw)

x = vt.from_numpy(np.ones((2, 3), dtype=np.float32)).cuda()
x.requires_grad = True
y = vt.ops.vt.add(x, x)
seed = vt.from_numpy(np.ones((2, 3), dtype=np.float32)).cuda()
y.backward(seed)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Exit code {result.returncode}, stderr: {result.stderr.decode()}"


@pytest.mark.cuda
def test_exit_after_python_override():
    """Test clean exit after registering Python op overrides."""
    code = '''
import vibetensor._C as C

def my_override(*args, **kwargs):
    return args[0]

C._try_register_boxed_python_override("vt::relu", my_override)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Exit code {result.returncode}, stderr: {result.stderr.decode()}"


@pytest.mark.cuda
def test_exit_after_pyfunction():
    """Test clean exit after using autograd.Function."""
    code = '''
import numpy as np
import vibetensor._C as C
import vibetensor.torch as vt
import vibetensor.autograd as vag

C.autograd.set_cuda_autograd_enabled(True)

class IdentityFn(vag.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_out):
        return (grad_out,)

x = vt.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).cuda()
x.requires_grad = True
y = IdentityFn.apply(x)
seed = vt.from_numpy(np.ones(2, dtype=np.float32)).cuda()
y.backward(seed)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Exit code {result.returncode}, stderr: {result.stderr.decode()}"
