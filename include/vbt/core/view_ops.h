// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "vbt/core/tensor.h"

namespace vbt {
namespace core {

TensorImpl select(const TensorImpl& self, int64_t dim, int64_t index);
TensorImpl narrow(const TensorImpl& self, int64_t dim, int64_t start, int64_t length);

TensorImpl squeeze(const TensorImpl& self);
TensorImpl squeeze(const TensorImpl& self, int64_t dim);
TensorImpl squeeze(const TensorImpl& self, const std::vector<int64_t>& dims);
TensorImpl unsqueeze(const TensorImpl& self, int64_t dim);

TensorImpl permute(const TensorImpl& self, const std::vector<int64_t>& dims);
TensorImpl transpose(const TensorImpl& self, int64_t dim0, int64_t dim1);

TensorImpl expand(const TensorImpl& self, const std::vector<int64_t>& sizes);
TensorImpl view(const TensorImpl& self, const std::vector<int64_t>& sizes);
TensorImpl reshape(const TensorImpl& self, const std::vector<int64_t>& sizes);

TensorImpl view_as_real(const TensorImpl& self);
TensorImpl view_as_complex(const TensorImpl& self);
TensorImpl conj(const TensorImpl& self);
TensorImpl resolve_conj(const TensorImpl& self);

} // namespace core
} // namespace vbt
