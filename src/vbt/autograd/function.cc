// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/function.h"
#include "vbt/autograd/wrapper.h"

namespace vbt { namespace autograd {

std::vector<OptionalTensor> FunctionNode::apply(std::vector<OptionalTensor>&& grads_in) {
  NoGradGuard ng;
  if (!backward_) return {};
  return backward_(std::move(grads_in));
}

}} // namespace vbt::autograd
