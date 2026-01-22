// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vbt/autograd/types.h"
#include "vbt/autograd/node.h"
#include "vbt/core/tensor.h"

namespace vbt { namespace autograd {

// Forward input metadata captured for validate_outputs
struct InputMeta {
  vbt::core::ScalarType dtype;
  vbt::core::Device device;
  std::vector<int64_t> sizes;
  bool is_strided_dense{true};

  static inline InputMeta from_tensor(const vbt::core::TensorImpl& t) noexcept {
    return InputMeta{t.dtype(), t.device(), t.sizes(), t.is_non_overlapping_and_dense()};
  }
};

// Probe interface for Engine validation
struct ValidatableNode {
  virtual ~ValidatableNode() = default;
  virtual const std::vector<InputMeta>& input_metas() const noexcept = 0;
};

using BackwardFn = std::function<std::vector<OptionalTensor>(std::vector<OptionalTensor>&&)>;

// Minimal Function node that carries InputMeta and delegates to a backward callable
class FunctionNode final : public Node, public ValidatableNode {
 public:
  FunctionNode(std::string name, std::vector<InputMeta> in_meta, BackwardFn backward)
      : in_meta_(std::move(in_meta)), backward_(std::move(backward)) { this->name = std::move(name); }

  uint32_t num_inputs() const noexcept override { return static_cast<uint32_t>(in_meta_.size()); }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override;

  const std::vector<InputMeta>& input_metas() const noexcept override { return in_meta_; }

 private:
  std::vector<InputMeta> in_meta_;
  BackwardFn backward_;
};

// Helper to snapshot input metas from a list of tensors
inline std::vector<InputMeta> snapshot_input_metas(const std::vector<vbt::core::TensorImpl>& inputs) {
  std::vector<InputMeta> m; m.reserve(inputs.size());
  for (const auto& t : inputs) m.emplace_back(InputMeta::from_tensor(t));
  return m;
}

// Ensure edges sized to num_inputs() to satisfy Engine invariants
inline void ensure_next_edges_sized(Node& n) { n.next_edges.resize(n.num_inputs()); }

}} // namespace vbt::autograd
