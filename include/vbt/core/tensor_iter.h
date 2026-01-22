// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/core/tensor_iterator/core.h"
#include "vbt/core/tensor_iterator/cpu.h"

#if VBT_WITH_CUDA
#  include "vbt/core/tensor_iterator/cuda.h"
#endif
