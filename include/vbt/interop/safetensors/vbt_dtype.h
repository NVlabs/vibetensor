// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "vbt/core/dtype.h"
#include "vbt/interop/safetensors/types.h"

namespace vbt {
namespace interop {
namespace safetensors {

// Map a safetensors dtype token to the closest supported VBT ScalarType.
//
// Note: safetensors supports more dtypes than VBT currently does; those are
// treated as unsupported here (nullopt / UnsupportedDtypeForVbt).
[[nodiscard]] std::optional<vbt::core::ScalarType> to_vbt_scalar_type(DType dt);

// Like `to_vbt_scalar_type`, but throws SafeTensorsError(UnsupportedDtypeForVbt)
// when no mapping exists.
[[nodiscard]] vbt::core::ScalarType require_vbt_scalar_type(DType dt);

} // namespace safetensors
} // namespace interop
} // namespace vbt
