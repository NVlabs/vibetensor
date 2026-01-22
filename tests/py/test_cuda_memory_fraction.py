# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import vibetensor.torch.cuda as cuda


def test_memory_fraction_apis_callable_cpu():
    # No exception when CUDA is unavailable
    cuda.set_per_process_memory_fraction(0.5)
    fr = cuda.get_per_process_memory_fraction()
    assert isinstance(fr, float)


def test_memory_fraction_invalid_type():
    with pytest.raises(TypeError):
        cuda.set_per_process_memory_fraction("0.5")  # type: ignore[arg-type]


def test_memory_fraction_out_of_range():
    with pytest.raises(ValueError):
        cuda.set_per_process_memory_fraction(-0.1)
    with pytest.raises(ValueError):
        cuda.set_per_process_memory_fraction(1.1)
