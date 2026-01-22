# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def _to_numpy(t):
    # numpy.from_dlpack is patched by vibetensor to return a copy.
    return np.from_dlpack(t).copy()


def test_embedding_forward_ignores_padding_idx_cpu():
    weight = vt.arange(15, dtype="float32").reshape((5, 3))
    idx = vt.tensor([1, 0, 1], dtype="int64")

    out = vt.nn.functional.embedding(idx, weight, padding_idx=1)

    w_np = _to_numpy(weight)
    idx_np = _to_numpy(idx)
    np.testing.assert_allclose(_to_numpy(out), w_np[idx_np])


def test_embedding_padding_idx_normalization_none_vs_negative_one_cpu():
    # padding_idx=None means "no padding row" (sentinel -1).
    weight1 = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight1.requires_grad = True

    idx = vt.tensor([3], dtype="int64")
    out1 = vt.nn.functional.embedding(idx, weight1, padding_idx=None)
    out1.backward(vt.ones((1, 3), dtype="float32"))

    g1 = _to_numpy(weight1.grad)
    expected1 = np.zeros((4, 3), dtype=np.float32)
    expected1[3] = 1.0
    np.testing.assert_allclose(g1, expected1)

    # padding_idx=-1 is normalized to the *last* row.
    weight2 = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight2.requires_grad = True

    out2 = vt.nn.functional.embedding(idx, weight2, padding_idx=-1)
    out2.backward(vt.ones((1, 3), dtype="float32"))

    g2 = _to_numpy(weight2.grad)
    expected2 = np.zeros((4, 3), dtype=np.float32)
    np.testing.assert_allclose(g2, expected2)


def test_embedding_padding_idx_normalization_negative_range_cpu():
    # For V=4, padding_idx=-4 normalizes to 0.
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx = vt.tensor([0], dtype="int64")
    out = vt.nn.functional.embedding(idx, weight, padding_idx=-4)
    out.backward(vt.ones((1, 3), dtype="float32"))

    g = _to_numpy(weight.grad)
    expected = np.zeros((4, 3), dtype=np.float32)
    np.testing.assert_allclose(g, expected)


def test_embedding_padding_idx_positive_skips_grad_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3)).detach()
    weight.requires_grad = True

    idx = vt.tensor([1], dtype="int64")
    out = vt.nn.functional.embedding(idx, weight, padding_idx=1)
    out.backward(vt.ones((1, 3), dtype="float32"))

    g = _to_numpy(weight.grad)
    expected = np.zeros((4, 3), dtype=np.float32)
    np.testing.assert_allclose(g, expected)


def test_embedding_max_norm_cpu_changes_only_referenced_rows():
    assert vt._C._has_op("vt::embedding_renorm_"), "vt::embedding_renorm_ must be registered"

    weight = vt.tensor(
        [
            [3.0, 4.0, 0.0],  # ||.||2 = 5  -> scaled
            [0.1, 0.2, 0.3],  # ||.||2 < 1  -> unchanged
            [0.0, 5.0, 12.0],  # ||.||2 = 13 -> scaled
            [1.0, 0.0, 0.0],  # ||.||2 = 1  -> unchanged (== max_norm)
        ],
        dtype="float32",
    )
    idx = vt.tensor([0, 2, 3, 0], dtype="int64")

    w_before = _to_numpy(weight)
    idx_np = _to_numpy(idx)

    out = vt.nn.functional.embedding(idx, weight, max_norm=1.0, norm_type=2.0)

    w_after = _to_numpy(weight)
    out_np = _to_numpy(out)

    w_expected = w_before.copy()
    for row in np.unique(idx_np):
        r = w_before[int(row)]
        norm = float(np.linalg.norm(r, ord=2))
        if norm > 1.0:
            w_expected[int(row)] = r * (1.0 / (norm + 1e-7))

    np.testing.assert_allclose(w_after, w_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_np, w_expected[idx_np], rtol=1e-6, atol=1e-6)


def test_embedding_padding_idx_out_of_range_raises_assertionerror_cpu():
    weight = vt.arange(12, dtype="float32").reshape((4, 3))
    idx = vt.tensor([0], dtype="int64")

    with pytest.raises(AssertionError, match="Padding_idx must be within num_embeddings"):
        _ = vt.nn.functional.embedding(idx, weight, padding_idx=4)

    with pytest.raises(AssertionError, match="Padding_idx must be within num_embeddings"):
        _ = vt.nn.functional.embedding(idx, weight, padding_idx=-5)
