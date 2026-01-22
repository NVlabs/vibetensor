# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for VibeTensor-native gather/scatter_add implementation.

These tests verify that the pure VibeTensor implementation (NO PyTorch)
produces correct results.
"""

import pytest
import numpy as np

# Skip all tests if VibeTensor is not available
pytest.importorskip("vibetensor")


@pytest.fixture(scope="module")
def vbt_modules():
    """Initialize VibeTensor and return modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    
    # Check CUDA availability
    if not getattr(_C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if _C._cuda_device_count() <= 0:
        pytest.skip("No CUDA device available")
    
    # Initialize CUDA context
    _ = _C._cuda_empty([1], "float32", 0)
    
    return vt, _C


@pytest.fixture
def idx_ops():
    """Import and return the vbt_native module."""
    import sys
    sys.path.insert(0, "/workspace/terry/nano-cursor/tmp")
    from vibe_kernels.indexing import vbt_native
    return vbt_native


class TestGatherVibeTensor:
    """Tests for gather with pure VibeTensor."""
    
    def test_gather_basic(self, vbt_modules, idx_ops):
        """Basic gather test."""
        vt, _C = vbt_modules
        
        # Create source tensor
        src_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        src = vt.cuda.to_device(src_np, device=0)
        
        # Create index tensor
        idx_np = np.array([1, 3, 0], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        # Run gather
        out = idx_ops.gather(src, 0, idx)
        out_np = vt.cuda.from_device(out)
        
        # Verify
        expected = src_np[[1, 3, 0], :]
        assert out_np.shape == (3, 8)
        assert np.allclose(out_np, expected)
    
    def test_gather_single_index(self, vbt_modules, idx_ops):
        """Gather with single index."""
        vt, _C = vbt_modules
        
        src_np = np.arange(24, dtype=np.float32).reshape(3, 8)
        src = vt.cuda.to_device(src_np, device=0)
        
        idx_np = np.array([2], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        out = idx_ops.gather(src, 0, idx)
        out_np = vt.cuda.from_device(out)
        
        expected = src_np[[2], :]
        assert out_np.shape == (1, 8)
        assert np.allclose(out_np, expected)
    
    def test_gather_all_same_index(self, vbt_modules, idx_ops):
        """Gather with all same indices."""
        vt, _C = vbt_modules
        
        src_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        src = vt.cuda.to_device(src_np, device=0)
        
        idx_np = np.array([1, 1, 1], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        out = idx_ops.gather(src, 0, idx)
        out_np = vt.cuda.from_device(out)
        
        expected = src_np[[1, 1, 1], :]
        assert out_np.shape == (3, 4)
        assert np.allclose(out_np, expected)


class TestScatterAddVibeTensor:
    """Tests for scatter_add with pure VibeTensor."""
    
    def test_scatter_add_basic(self, vbt_modules, idx_ops):
        """Basic scatter_add test."""
        vt, _C = vbt_modules
        
        # Output tensor (zeros)
        out = _C._cuda_zeros([4, 8], "float32", 0)
        
        # Index: scatter to rows 0, 2
        idx_np = np.array([0, 2], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        # Source: ones
        src_np = np.ones((2, 8), dtype=np.float32)
        src = vt.cuda.to_device(src_np, device=0)
        
        idx_ops.scatter_add(out, 0, idx, src)
        out_np = vt.cuda.from_device(out)
        
        expected = np.zeros((4, 8), dtype=np.float32)
        expected[0] = 1.0
        expected[2] = 1.0
        
        assert np.allclose(out_np, expected)
    
    def test_scatter_add_accumulate(self, vbt_modules, idx_ops):
        """Scatter_add with duplicate indices (accumulation)."""
        vt, _C = vbt_modules
        
        out = _C._cuda_zeros([4, 8], "float32", 0)
        
        # Index: [0, 2, 0] - row 0 gets accumulated twice
        idx_np = np.array([0, 2, 0], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        src_np = np.ones((3, 8), dtype=np.float32)
        src = vt.cuda.to_device(src_np, device=0)
        
        idx_ops.scatter_add(out, 0, idx, src)
        out_np = vt.cuda.from_device(out)
        
        expected = np.zeros((4, 8), dtype=np.float32)
        expected[0] = 2.0  # Accumulated twice
        expected[2] = 1.0
        
        assert np.allclose(out_np, expected)
    
    def test_scatter_add_values(self, vbt_modules, idx_ops):
        """Scatter_add with different values."""
        vt, _C = vbt_modules
        
        out = _C._cuda_zeros([3, 4], "float32", 0)
        
        idx_np = np.array([0, 1, 2], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        src_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
        src = vt.cuda.to_device(src_np, device=0)
        
        idx_ops.scatter_add(out, 0, idx, src)
        out_np = vt.cuda.from_device(out)
        
        assert np.allclose(out_np, src_np)


class TestGatherScatterRoundtrip:
    """Tests verifying gather backward = scatter_add."""
    
    def test_roundtrip(self, vbt_modules, idx_ops):
        """Test gather forward + scatter_add backward."""
        vt, _C = vbt_modules
        
        # Forward: gather
        src_np = np.random.randn(4, 8).astype(np.float32)
        src = vt.cuda.to_device(src_np, device=0)
        
        idx_np = np.array([1, 3, 0], dtype=np.int64)
        idx = vt.cuda.to_device(idx_np, device=0)
        
        out = idx_ops.gather(src, 0, idx)
        
        # Backward: scatter_add
        grad_out_np = np.ones((3, 8), dtype=np.float32)
        grad_out = vt.cuda.to_device(grad_out_np, device=0)
        
        grad_src = _C._cuda_zeros([4, 8], "float32", 0)
        idx_ops.scatter_add(grad_src, 0, idx, grad_out)
        grad_src_np = vt.cuda.from_device(grad_src)
        
        # Verify: grad_src[i] = 1 if i in idx, else 0
        expected_grad = np.zeros((4, 8), dtype=np.float32)
        expected_grad[0] = 1.0  # idx[2] = 0
        expected_grad[1] = 1.0  # idx[0] = 1
        expected_grad[3] = 1.0  # idx[1] = 3
        
        assert np.allclose(grad_src_np, expected_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
