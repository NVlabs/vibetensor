# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Thin alias for torch-like ops namespace
# Users can import vibetensor.ops and access the same singleton object
# as vibetensor.torch.ops (module attribute named 'ops').
from vibetensor.torch import ops as ops  # noqa: F401
