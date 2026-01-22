// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

#include "vbt/core/tensor.h"

namespace vbt {
namespace dispatch {

using BoxedStack = std::vector<vbt::core::TensorImpl>;

class StackGuard {
 public:
  explicit StackGuard(BoxedStack& s, std::size_t nin) : stack_(s), saved_(s), nin_(nin) {}
  void commit() { committed_ = true; }
  ~StackGuard() {
    if (!committed_) {
      stack_ = saved_;
    }
  }
 private:
  BoxedStack& stack_;
  BoxedStack saved_;
  std::size_t nin_;
  bool committed_{false};
};

} // namespace dispatch
} // namespace vbt
