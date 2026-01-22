// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vbt/autograd/meta.h"

namespace vbt { namespace autograd {

// AccumulateGrad: sink that accumulates into a target leaf tensor's AutogradMeta.
class AccumulateGrad final : public Node {
 public:
  explicit AccumulateGrad(AutogradMeta* meta) noexcept
      : meta_(meta, /*add_ref=*/true) {
    name = "AccumulateGrad";
  }

  uint32_t num_inputs() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_leaf_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override;

  AutogradMeta* meta() const noexcept { return meta_.get(); }

 private:
  // Owning keepalive handle so leaf metadata outlives the backward graph.
  vbt::core::intrusive_ptr<AutogradMeta> meta_{};
  bool                                   is_cuda_leaf_{false};

  friend void _tag_accumulategrad_cuda_leaf(AccumulateGrad& ag,
                                            const vbt::core::TensorImpl& leaf);
};

}} // namespace vbt::autograd
