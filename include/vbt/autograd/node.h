// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/device.h"
#include "vbt/autograd/types.h"

namespace vbt { namespace core { class TensorImpl; } }

namespace vbt { namespace autograd {

struct NodeStreamInfo {
  bool              has_canonical_stream{false};
  vbt::core::Device device{};      // logical execution device (CPU or CUDA)
  std::uint64_t     stream_id{0};  // opaque id; 0 may mean default stream
};

enum class StreamKind : std::uint8_t {
  CpuOnly,
  CudaAllowlisted,
};

// Base autograd node
struct Node : public vbt::core::IntrusiveRefcounted {
  virtual ~Node() = default;
  // Number of inputs (gradient slots) this node expects
  virtual uint32_t num_inputs() const noexcept = 0;

  // Number of incoming gradient slots the engine should track for this node.
  // Defaults to num_inputs() but can be smaller for wrapper nodes that only
  // ever see a single upstream gradient.
  virtual uint32_t num_incoming_grad_slots() const noexcept { return num_inputs(); }

  std::vector<Edge> next_edges;

  // Apply this node: given incoming gradients for this node's output(s),
  // produce gradients for each of this node's inputs (size == num_inputs()).
  virtual std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) = 0;

  virtual StreamKind stream_kind() const noexcept { return StreamKind::CpuOnly; }

  const NodeStreamInfo& stream_info() const noexcept { return stream_info_; }
  NodeStreamInfo&       mutable_stream_info() noexcept { return stream_info_; }

  // Optional debugging label
  std::string name;

 private:
  NodeStreamInfo stream_info_{};
};

}} // namespace vbt::autograd

