# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import vibetensor.torch as vt
from vibetensor import _C as C


def test_graph_get_gradient_edge_leaf_returns_accumulategrad():
    x = vt.tensor([1.0], dtype="float32")
    x.requires_grad = True

    fn, input_nr = C.autograd._graph_get_gradient_edge(x)

    assert fn is not None
    assert fn.name == "AccumulateGrad"
    assert input_nr == 0


def test_graph_get_gradient_edge_nonleaf_matches_grad_fn_and_output_nr():
    x = vt.tensor([2.0], dtype="float32")
    x.requires_grad = True

    y = C.vt.mul(x, x)
    assert y.grad_fn is not None

    fn, input_nr = C.autograd._graph_get_gradient_edge(y)

    assert fn is not None
    assert fn.name == y.grad_fn.name

    meta = C.autograd._debug_tensor_meta(y)
    assert input_nr == meta["output_nr"]
