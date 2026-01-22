// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/autograd/add_ops.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#endif

#include <stdexcept>
#include <cstring>

namespace vbt { namespace autograd {

using vbt::core::TensorImpl;

void _tag_accumulategrad_cuda_leaf(AccumulateGrad& ag,
                                   const vbt::core::TensorImpl& leaf) {
#if VBT_WITH_CUDA
  auto dev = leaf.device();
  if (dev.type != kDLCUDA) {
    return;
  }

  NodeStreamInfo& si = ag.mutable_stream_info();
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(dev.index);
  Stream S = vbt::cuda::getCurrentStream(dev_idx);

  si.has_canonical_stream = true;
  si.device = dev;
  si.stream_id = S.id();

  ag.is_cuda_leaf_ = true;
#else
  (void)ag;
  (void)leaf;
#endif
}

std::vector<OptionalTensor> AccumulateGrad::apply(std::vector<OptionalTensor>&& grads_in) {
  std::vector<OptionalTensor> out; // sink
  if (!meta_) return out;

  // Look up metadata for the gradient root tensor.
  AutogradMeta* meta = meta_.get();

  bool updated_grad = false;
  std::vector<vbt::core::intrusive_ptr<TensorHook>> hooks_snapshot;
  TensorImpl grad_snapshot;

  {
    std::lock_guard<std::mutex> lk(meta->grad_mutex);

    TensorImpl* incoming = nullptr;
    if (!grads_in.empty() && grads_in[0].has_value()) {
      incoming = &grads_in[0].value();
    }

    if (!meta->grad_ptr || !meta->grad_has) {
      if (incoming) {
        meta->grad_ptr = std::make_unique<TensorImpl>(std::move(*incoming));
        meta->grad_has = true;
        updated_grad = true;
        try {
          set_requires_grad(*meta->grad_ptr, false);
        } catch (...) {
        }
      }
    } else {
      if (incoming) {
        const vbt::core::Device dev = meta->grad_ptr->device();
        autograd_add_inplace_dense(*meta->grad_ptr, *incoming, dev);
        updated_grad = true;
      }
    }

#if VBT_WITH_CUDA
    // For CUDA leaves, associate the stored .grad buffer with the canonical
    // stream so the allocator can fence deferred frees correctly.
    if (updated_grad && meta->grad_ptr && is_cuda_leaf_) {
      const TensorImpl& stored = *meta->grad_ptr;
      auto dev = stored.device();
      const NodeStreamInfo& si = stream_info();
      if (dev.type == kDLCUDA &&
          si.has_canonical_stream &&
          si.device.type == kDLCUDA &&
          si.device.index == dev.index) {
        using vbt::cuda::DeviceIndex;
        using vbt::cuda::Stream;
        const auto dev_idx = static_cast<DeviceIndex>(dev.index);
        Stream S(Stream::UNCHECKED, si.stream_id, dev_idx);
        vbt::cuda::record_stream(stored.storage(), S);
      }
    }
#endif

    if (!updated_grad || !meta->grad_ptr) {
      return out;
    }

    // Snapshot hook list + updated grad under the mutex, but invoke hooks
    // outside to avoid deadlocks.
    hooks_snapshot = meta->hooks;
    if (!hooks_snapshot.empty()) {
      grad_snapshot = vbt::core::clone_contiguous_same_device(*meta->grad_ptr);
      try {
        set_requires_grad(grad_snapshot, false);
      } catch (...) {
      }
    }
  } // unlock

  if (hooks_snapshot.empty()) {
    return out;
  }

  // Hook path: call hooks with detached clones under NoGradGuard.
  try {
    NoGradGuard ng;
    for (const auto& h : hooks_snapshot) {
      if (!h || h->is_removed()) {
        continue;
      }
      // Observer-only: each hook sees its own detached clone so that
      // mutations inside the hook cannot affect other hooks.
      TensorImpl cloned = vbt::core::clone_contiguous_same_device(grad_snapshot);
      try {
        set_requires_grad(cloned, false);
      } catch (...) {
      }
      h->call(cloned);
    }
  } catch (...) {
    throw;
  }

  return out;
}

}} // namespace vbt::autograd
