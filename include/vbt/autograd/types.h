// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>
#include <vector>

#include "vbt/core/intrusive_ptr.h"

namespace vbt { namespace core { class TensorImpl; }}

namespace vbt { namespace autograd {

class Node; // fwd

// Optional tensor by value (TensorImpl is a value type sharing Storage)
using OptionalTensor = std::optional<vbt::core::TensorImpl>;

struct Edge {
  vbt::core::intrusive_ptr<Node> fn; // may be null (edge to nowhere)
  uint32_t input_nr{0};              // destination slot index at consumer
};

}} // namespace vbt::autograd
