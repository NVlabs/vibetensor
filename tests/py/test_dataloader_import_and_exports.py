# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def test_dataloader_import_and_exports():
    import vibetensor.torch.utils.data as vtd

    # Import path parity: vibetensor.torch.utils.data
    assert hasattr(vtd, "DataLoader")
    assert hasattr(vtd, "Dataset")
    assert hasattr(vtd, "IterableDataset")
    assert hasattr(vtd, "Sampler")
    assert hasattr(vtd, "SequentialSampler")
    assert hasattr(vtd, "RandomSampler")
    assert hasattr(vtd, "BatchSampler")
    assert hasattr(vtd, "default_convert")
    assert hasattr(vtd, "default_collate")
    assert hasattr(vtd, "WorkerInfo")
    assert hasattr(vtd, "get_worker_info")
