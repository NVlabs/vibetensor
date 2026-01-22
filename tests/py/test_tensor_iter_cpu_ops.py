# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from vibetensor import _C as C


def _make_cpu_tensor_from_numpy(arr: np.ndarray):
  """Helper to build a CPU TensorImpl from a NumPy array via DLPack."""
  cap = arr.__dlpack__()
  return C._from_dlpack(cap)


def _tensor_to_numpy_cpu(t):
  cap = C._to_dlpack(t)
  # NumPy's from_dlpack historically accepted raw DLPack capsules directly,
  # but newer versions expect an object with a __dlpack__ method. Support
  # both by wrapping the capsule when needed.
  try:
    return np.from_dlpack(cap)
  except AttributeError:
    class _CapsuleWrapper:
      def __init__(self, inner):
        self._inner = inner

      def __dlpack__(self):  # pragma: no cover - tiny adapter
        return self._inner

    return np.from_dlpack(_CapsuleWrapper(cap))


def test_vt_add_cpu_broadcast_against_numpy():
  # Shapes (2,1) and (1,3) broadcasting to (2,3)
  a_np = np.arange(2 * 1, dtype=np.float32).reshape(2, 1)
  b_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

  a = _make_cpu_tensor_from_numpy(a_np)
  b = _make_cpu_tensor_from_numpy(b_np)

  out = C.vt.add(a, b)
  out_np = _tensor_to_numpy_cpu(out)

  np.testing.assert_allclose(out_np, a_np + b_np)


def test_add_inplace_cpu_matches_numpy():
  a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
  b_np = np.ones_like(a_np, dtype=np.float32) * 2.5

  expected = a_np + b_np

  a = _make_cpu_tensor_from_numpy(a_np)
  b = _make_cpu_tensor_from_numpy(b_np)

  v0 = a.version()
  a.add_(b)
  assert a.version() == v0 + 1

  out_np = _tensor_to_numpy_cpu(a)
  np.testing.assert_allclose(out_np, expected)


def test_relu_inplace_cpu_matches_numpy():
  a_np = np.array([[-1.0, 0.0, 3.0], [4.0, -2.0, 5.0]], dtype=np.float32)
  a = _make_cpu_tensor_from_numpy(a_np)

  v0 = a.version()
  a.relu_()
  assert a.version() == v0 + 1

  out_np = _tensor_to_numpy_cpu(a)
  np.testing.assert_allclose(out_np, np.maximum(a_np, 0.0))
