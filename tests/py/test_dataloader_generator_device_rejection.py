# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_dataloader_rejects_cuda_generator_for_shuffle():
    import vibetensor.torch.rng as rng
    import vibetensor.torch.utils.data as vtd
    from vibetensor import _C

    if not bool(getattr(_C, "_has_cuda", False)):
        pytest.skip("CUDA not available")

    class _DS:
        def __len__(self):
            return 4

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    gen = rng.Generator(0)
    assert gen.device == "cuda:0"

    with pytest.raises(ValueError) as exc:
        vtd.DataLoader(ds, shuffle=True, generator=gen)

    assert str(exc.value) == "generator device mismatch: expected cpu, got cuda:0"
