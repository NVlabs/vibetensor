# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for gather and scatter_add Triton kernels."""

import pytest
import torch

from vibe_kernels.indexing import gather, gather_with_grad, scatter_add


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestGather:
    """Tests for the gather kernel."""
    
    def test_gather_1d_simple(self, device):
        """Test simple 1D gather."""
        src = torch.arange(10, dtype=torch.float32, device=device).view(10, 1)
        idx = torch.tensor([0, 2, 5, 9], dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        expected = torch.gather(src, 0, idx.view(-1, 1).expand(-1, 1))
        
        assert out.shape == (4, 1)
        torch.testing.assert_close(out, expected)
    
    def test_gather_2d(self, device):
        """Test 2D gather along dim 0."""
        src = torch.randn(8, 16, dtype=torch.float32, device=device)
        idx = torch.tensor([0, 3, 7, 2], dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        
        # Manual verification
        expected = src[idx]
        
        assert out.shape == (4, 16)
        torch.testing.assert_close(out, expected)
    
    def test_gather_3d(self, device):
        """Test 3D gather along dim 0."""
        src = torch.randn(8, 16, 32, dtype=torch.float32, device=device)
        idx = torch.tensor([0, 3, 7], dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        expected = src[idx]
        
        assert out.shape == (3, 16, 32)
        torch.testing.assert_close(out, expected)
    
    def test_gather_fp16(self, device):
        """Test gather with fp16."""
        src = torch.randn(16, 32, dtype=torch.float16, device=device)
        idx = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        expected = src[idx]
        
        assert out.dtype == torch.float16
        torch.testing.assert_close(out, expected)
    
    def test_gather_bf16(self, device):
        """Test gather with bf16."""
        src = torch.randn(16, 32, dtype=torch.bfloat16, device=device)
        idx = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        expected = src[idx]
        
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, expected)
    
    def test_gather_large(self, device):
        """Test gather with larger tensors."""
        src = torch.randn(1024, 256, dtype=torch.float32, device=device)
        idx = torch.randint(0, 1024, (128,), dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        expected = src[idx]
        
        assert out.shape == (128, 256)
        torch.testing.assert_close(out, expected)


class TestScatterAdd:
    """Tests for the scatter_add kernel."""
    
    def test_scatter_add_simple(self, device):
        """Test simple scatter_add."""
        out = torch.zeros(5, 4, dtype=torch.float32, device=device)
        src = torch.ones(3, 4, dtype=torch.float32, device=device)
        idx = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
        
        scatter_add(out, 0, idx, src)
        
        expected = torch.zeros_like(out)
        expected[0] = 1.0
        expected[2] = 1.0
        expected[4] = 1.0
        
        torch.testing.assert_close(out, expected)
    
    def test_scatter_add_accumulate(self, device):
        """Test scatter_add with duplicate indices (accumulation)."""
        out = torch.zeros(4, 8, dtype=torch.float32, device=device)
        src = torch.ones(5, 8, dtype=torch.float32, device=device)
        idx = torch.tensor([0, 0, 1, 0, 2], dtype=torch.int64, device=device)
        
        scatter_add(out, 0, idx, src)
        
        # idx has 0 appearing 3 times, 1 appearing 1 time, 2 appearing 1 time
        expected = torch.zeros_like(out)
        expected[0] = 3.0  # Three 1s added to row 0
        expected[1] = 1.0  # One 1 added to row 1
        expected[2] = 1.0  # One 1 added to row 2
        
        torch.testing.assert_close(out, expected)
    
    def test_scatter_add_values(self, device):
        """Test scatter_add with non-uniform values."""
        out = torch.zeros(4, 4, dtype=torch.float32, device=device)
        src = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ], device=device)
        idx = torch.tensor([0, 0, 2], dtype=torch.int64, device=device)
        
        scatter_add(out, 0, idx, src)
        
        expected = torch.zeros_like(out)
        expected[0] = src[0] + src[1]  # Both scatter to index 0
        expected[2] = src[2]           # Scatters to index 2
        
        torch.testing.assert_close(out, expected)
    
    def test_scatter_add_fp16(self, device):
        """Test scatter_add with fp16."""
        out = torch.zeros(8, 16, dtype=torch.float16, device=device)
        src = torch.ones(4, 16, dtype=torch.float16, device=device)
        idx = torch.tensor([0, 2, 4, 6], dtype=torch.int64, device=device)
        
        scatter_add(out, 0, idx, src)
        
        expected = torch.zeros_like(out)
        expected[0] = 1.0
        expected[2] = 1.0
        expected[4] = 1.0
        expected[6] = 1.0
        
        assert out.dtype == torch.float16
        torch.testing.assert_close(out, expected)
    
    def test_scatter_add_large(self, device):
        """Test scatter_add with larger tensors."""
        out = torch.zeros(1024, 256, dtype=torch.float32, device=device)
        src = torch.randn(512, 256, dtype=torch.float32, device=device)
        idx = torch.randint(0, 1024, (512,), dtype=torch.int64, device=device)
        
        # Use torch reference
        expected = torch.zeros_like(out)
        expected.scatter_add_(0, idx.view(-1, 1).expand(-1, 256), src)
        
        scatter_add(out, 0, idx, src)
        
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


class TestGatherWithGrad:
    """Tests for gather with autograd support."""
    
    def test_gather_backward(self, device):
        """Test gather backward computes correct gradients."""
        src = torch.randn(8, 16, dtype=torch.float32, device=device, requires_grad=True)
        idx = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
        
        out = gather_with_grad(src, 0, idx)
        loss = out.sum()
        loss.backward()
        
        # Reference with torch
        src_ref = src.detach().clone().requires_grad_(True)
        out_ref = src_ref[idx]
        loss_ref = out_ref.sum()
        loss_ref.backward()
        
        torch.testing.assert_close(src.grad, src_ref.grad)
    
    def test_gather_backward_duplicate_idx(self, device):
        """Test gather backward with duplicate indices."""
        src = torch.randn(8, 16, dtype=torch.float32, device=device, requires_grad=True)
        idx = torch.tensor([0, 0, 3, 3, 5], dtype=torch.int64, device=device)
        
        out = gather_with_grad(src, 0, idx)
        loss = out.sum()
        loss.backward()
        
        # Gradient should accumulate at repeated indices
        # idx 0 appears twice, so grad at src[0] should be 2x
        # idx 3 appears twice, so grad at src[3] should be 2x
        assert src.grad[0].sum().item() == pytest.approx(16 * 2, rel=1e-5)
        assert src.grad[3].sum().item() == pytest.approx(16 * 2, rel=1e-5)
        assert src.grad[5].sum().item() == pytest.approx(16 * 1, rel=1e-5)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_gather_single_element(self, device):
        """Test gather with single element."""
        src = torch.randn(10, 8, dtype=torch.float32, device=device)
        idx = torch.tensor([5], dtype=torch.int64, device=device)
        
        out = gather(src, 0, idx)
        expected = src[5:6]
        
        assert out.shape == (1, 8)
        torch.testing.assert_close(out, expected)
    
    def test_scatter_add_empty(self, device):
        """Test scatter_add with no elements to scatter."""
        out = torch.zeros(4, 4, dtype=torch.float32, device=device)
        src = torch.empty(0, 4, dtype=torch.float32, device=device)
        idx = torch.empty(0, dtype=torch.int64, device=device)
        
        # Should not crash
        scatter_add(out, 0, idx, src)
        
        # Output should be unchanged
        expected = torch.zeros_like(out)
        torch.testing.assert_close(out, expected)
    
    def test_gather_int32_index(self, device):
        """Test gather with int32 indices."""
        src = torch.randn(8, 16, dtype=torch.float32, device=device)
        idx = torch.tensor([0, 3, 7], dtype=torch.int32, device=device)
        
        out = gather(src, 0, idx)
        expected = src[idx.long()]
        
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
