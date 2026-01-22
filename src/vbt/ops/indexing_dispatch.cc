// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/indexing.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/core/indexing/encoded_index_spec_v0.h"

#include <stdexcept>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/overlap.h"
#include "vbt/core/write_guard.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/view_ops.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#endif
#if VBT_WITH_CUDA
#include "vbt/cuda/guard.h"
#include "vbt/cuda/stream.h"
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::IndexKind;
using vbt::dispatch::BoxedStack;
namespace idx_errors = vbt::core::indexing::errors;

#if VBT_WITH_CUDA
static std::vector<std::int64_t> copy_cuda_int64_to_cpu(const TensorImpl& t) {
  using vbt::cuda::DeviceGuard;
  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;

  if (t.device().type != kDLCUDA ||
      t.dtype() != ScalarType::Int64 ||
      !t.is_contiguous()) {
    throw std::invalid_argument(
        "vt::index_put autograd: expected contiguous Int64 CUDA index tensor");
  }

  const std::int64_t n = t.numel();
  std::vector<std::int64_t> out;
  out.resize(static_cast<std::size_t>(n));
  if (n <= 0) {
    return out;
  }

  DeviceIndex dev_index = static_cast<DeviceIndex>(t.device().index);
  DeviceGuard guard(dev_index);
  Stream stream = vbt::cuda::getCurrentStream(dev_index);

  const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(std::int64_t);

  cudaError_t st = cudaMemcpyAsync(
      out.data(),
      t.data(),
      nbytes,
      cudaMemcpyDeviceToHost,
      reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = "vt::index_put autograd: cudaMemcpyAsync D2H failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = "vt::index_put autograd: cudaStreamSynchronize failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  return out;
}
#endif


static void vt_index_cpu_boxed(BoxedStack& s) {
  using vbt::core::indexing::index;

  if (s.size() != 3) {
    throw std::invalid_argument(
        "vt::index boxed kernel expected 3 arguments, got " +
        std::to_string(s.size()));
  }

  TensorImpl self  = s[0];
  TensorImpl index_t = s[1];
  TensorImpl meta  = s[2];

  const auto dev_type = self.device().type;
  if (dev_type != kDLCPU && dev_type != kDLCUDA) {
    throw std::invalid_argument(
        "vt::index: advanced indexing is only implemented for CPU and CUDA tensors");
  }

  if (index_t.storage().get()) {
    const auto idx_dev = index_t.device();
    if (dev_type == kDLCPU) {
      if (idx_dev != self.device()) {
        throw std::invalid_argument(
            "vt::index: index tensor must be on the same device as self");
      }
    } else {  // kDLCUDA
      if (idx_dev.type == kDLCUDA) {
        if (idx_dev != self.device()) {
          throw std::invalid_argument(
              "vt::index: index tensor must be on CPU or the same device as self");
        }
      } else if (idx_dev.type != kDLCPU) {
        throw std::invalid_argument(
            "vt::index: index tensor must be on CPU or the same device as self");
      }
    }

    const auto dt = index_t.dtype();
    if (dev_type == kDLCPU) {
      if (!(dt == ScalarType::Int32 ||
            dt == ScalarType::Int64 ||
            dt == ScalarType::Bool)) {
        throw std::invalid_argument(
            "vt::index: index tensor must be int32, int64, or bool");
      }
    } else {  // kDLCUDA
      if (idx_dev.type == kDLCPU) {
        if (!(dt == ScalarType::Int32 ||
              dt == ScalarType::Int64)) {
          throw std::invalid_argument(
              "vt::index: CPU index tensor for CUDA must be int32 or int64");
        }
      } else {
        if (!(dt == ScalarType::Int32 ||
              dt == ScalarType::Int64 ||
              dt == ScalarType::Bool)) {
          throw std::invalid_argument(
              "vt::index: index tensor must be int32, int64, or bool on CUDA");
        }
      }
    }
  }

#if VBT_WITH_AUTOGRAD
  const auto hdr =
      vbt::core::indexing::decode_encoded_index_spec_header_v0(meta, "vt::index");
  if (hdr.prefix_len > 0) {
    const bool graph_enabled =
        vbt::autograd::GradMode::is_enabled() &&
        !vbt::autograd::InferenceMode::is_enabled();
    if (graph_enabled &&
        vbt::autograd::any_requires_grad(s, /*in_arity=*/3) &&
        !vbt::autograd::autograd_indexing_v2_enabled()) {
      throw std::invalid_argument(
          idx_errors::kErrVtIndexPrefixRequiresAutogradIndexingV2);
    }
  }
#endif

  IndexSpec spec =
      vbt::core::indexing::decode_encoded_index_spec_v0(index_t, meta, "vt::index");
  TensorImpl out = index(self, spec);

  s.clear();
  s.push_back(std::move(out));
}

static void vt_index_put_cpu_boxed(BoxedStack& s) {
  using vbt::core::indexing::index_put_;

  if (s.size() != 5) {
    throw std::invalid_argument(
        "vt::index_put boxed kernel expected 5 arguments, got " +
        std::to_string(s.size()));
  }

  TensorImpl self  = s[0];
  TensorImpl index_t = s[1];
  TensorImpl value = s[2];
  TensorImpl meta  = s[3];
  TensorImpl acc_t = s[4];

  const auto dev_type = self.device().type;
  if (dev_type != kDLCPU && dev_type != kDLCUDA) {
    throw std::invalid_argument(
        "vt::index_put: advanced indexing is only implemented for CPU and CUDA tensors");
  }
  const auto idx_dev = index_t.device();
  if (dev_type == kDLCPU) {
    if (idx_dev != self.device()) {
      throw std::invalid_argument(
          "vt::index_put: index tensor must be on the same device as self");
    }
  } else {  // kDLCUDA
    if (idx_dev.type == kDLCUDA) {
      if (idx_dev != self.device()) {
        throw std::invalid_argument(
            "vt::index_put: index tensor must be on CPU or the same device as self");
      }
    } else if (idx_dev.type != kDLCPU) {
      throw std::invalid_argument(
          "vt::index_put: index tensor must be on CPU or the same device as self");
    }
  }
  if (value.device() != self.device()) {
    throw std::invalid_argument(
        "vt::index_put: value tensor must be on the same device as self");
  }
  if (value.dtype() != self.dtype()) {
    throw std::invalid_argument(
        "vt::index_put: dtype/device/size mismatch between self and value");
  }

  const auto dt = index_t.dtype();
  if (dev_type == kDLCPU) {
    if (!(dt == ScalarType::Int32 ||
          dt == ScalarType::Int64 ||
          dt == ScalarType::Bool)) {
      throw std::invalid_argument(
          "vt::index_put: index tensor must be int32, int64, or bool");
    }
  } else {  // kDLCUDA
    if (!(dt == ScalarType::Int32 ||
          dt == ScalarType::Int64)) {
      throw std::invalid_argument(
          "vt::index_put: index tensor must be int32 or int64 on CUDA");
    }
  }

  (void)vbt::core::indexing::decode_encoded_index_spec_header_v0(
      meta, "vt::index_put");

  if (acc_t.device().type != kDLCPU ||
      acc_t.dtype() != ScalarType::Bool ||
      !acc_t.sizes().empty()) {
    throw std::invalid_argument(
        "vt::index_put: accumulate flag must be a 0-d bool CPU tensor");
  }
  const bool accumulate = (acc_t.numel() > 0 &&
                           *static_cast<const std::uint8_t*>(acc_t.data()) != 0);

#if VBT_WITH_AUTOGRAD
  const bool graph_enabled =
      vbt::autograd::GradMode::is_enabled() &&
      !vbt::autograd::InferenceMode::is_enabled();
  const bool need_autograd =
      graph_enabled &&
      (vbt::autograd::requires_grad(self) ||
       vbt::autograd::requires_grad(value));
#else
  const bool need_autograd = false;
#endif

  if (!need_autograd) {
    IndexSpec spec =
        vbt::core::indexing::decode_encoded_index_spec_v0(
            index_t, meta, "vt::index_put");
    index_put_(self, spec, value, accumulate);
  } else {
#if VBT_WITH_AUTOGRAD
    // Autograd-enabled in-place advanced write (v2; int indices only).
    if (!vbt::autograd::autograd_indexing_v2_enabled()) {
      throw std::runtime_error(idx_errors::kErrIndexPutAutogradUnsupported);
    }

    if (!vbt::core::indexing::advanced_indexing_enabled()) {
      throw std::runtime_error(idx_errors::kErrAdvDisabledCore);
    }

    // v2 scope: Float32 only.
    if (self.dtype() != ScalarType::Float32) {
      throw std::runtime_error(
          "vt::index_put autograd: only Float32 is supported");
    }

    // v2 scope: int index tensors only (no bool masks).
    if (!(dt == ScalarType::Int32 || dt == ScalarType::Int64)) {
      throw std::runtime_error(idx_errors::kErrIndexPutAutogradUnsupported);
    }

    // In-place autograd only for non-leaf tensors / views.
    if (!(vbt::autograd::is_view(self) || !vbt::autograd::is_leaf(self))) {
      throw std::runtime_error(idx_errors::kErrIndexPutAutogradUnsupported);
    }

    // Pre-flight aliasing checks so failures don't mutate autograd metadata.
    vbt::core::check_writable(self);
    vbt::core::assert_no_internal_overlap(self);
    if (self.storage().get() == value.storage().get()) {
      vbt::core::assert_no_partial_overlap(self, value);
    }

    // Materialize a private contiguous index tensor to avoid mutating user-visible indices.
    TensorImpl index_private = vbt::core::clone_contiguous_same_device(index_t);

    // Decode spec using the private index tensor.
    IndexSpec spec_private =
        vbt::core::indexing::decode_encoded_index_spec_v0(
            index_private, meta, "vt::index_put");

    // Canonicalize indices + validate (no side effects on self).
    vbt::core::indexing::AdvancedIndex info;
    if (dev_type == kDLCPU) {
      info = vbt::core::indexing::make_advanced_index(self, spec_private);
    } else {  // kDLCUDA
#if VBT_WITH_CUDA
      auto r = vbt::core::indexing::cuda_impl::make_advanced_index_cuda(
          self,
          spec_private,
          vbt::core::indexing::cuda_impl::AdvancedIndexCudaMode::Write);
      info = std::move(r.info);
#else
      throw std::runtime_error("vt::index_put autograd: CUDA not built");
#endif
    }
    if (info.indices.size() != 1) {
      throw std::runtime_error(idx_errors::kErrIndexPutAutogradUnsupported);
    }

    TensorImpl index_used = info.indices[0];
    if (!index_used.is_contiguous()) {
      index_used = vbt::core::clone_contiguous_same_device(index_used);
    }

    // AdvancedIndex reshapes index tensors to include dims_before/dims_after
    // singleton dimensions for kernel convenience. For vt::index_put we need to
    // keep the *original* index shape so value broadcasting matches core
    // semantics (e.g., x[:, idx] writes).
    std::vector<std::int64_t> orig_shape(
        index_private.sizes().begin(), index_private.sizes().end());
    index_used = vbt::core::reshape(index_used, orig_shape);

    if (index_used.dtype() != ScalarType::Int64) {
      throw std::runtime_error(
          "vt::index_put autograd: expected canonicalized Int64 index tensor");
    }

    // Reject duplicates under overwrite semantics to avoid nondeterministic gradients.
    if (!accumulate) {
      const std::int64_t n = index_used.numel();
      if (n > 1) {
        std::vector<std::int64_t> tmp;
        if (index_used.device().type == kDLCPU) {
          const auto* p = static_cast<const std::int64_t*>(index_used.data());
          tmp.assign(p, p + n);
        } else if (index_used.device().type == kDLCUDA) {
#if VBT_WITH_CUDA
          tmp = copy_cuda_int64_to_cpu(index_used);
#else
          throw std::runtime_error("vt::index_put autograd: CUDA not built");
#endif
        } else {
          throw std::runtime_error(
              "vt::index_put autograd: unsupported device for duplicate check");
        }

        std::sort(tmp.begin(), tmp.end());
        for (std::size_t i = 1; i < tmp.size(); ++i) {
          if (tmp[i] == tmp[i - 1]) {
            throw std::runtime_error(
                idx_errors::kErrIndexPutAutogradDuplicateIndices);
          }
        }
      }
    }

    // Pre-flight broadcast validation so failures don't mutate autograd metadata.
    (void)vbt::core::indexing::broadcast_to(
        value,
        std::span<const std::int64_t>(
            info.result_shape.data(), info.result_shape.size()));

    // Decode again so the spec uses the canonicalized int64 index tensor.
    IndexSpec spec_used =
        vbt::core::indexing::decode_encoded_index_spec_v0(
            index_used, meta, "vt::index_put");

    // Build and attach the in-place backward node *before* mutation.
    auto node = vbt::autograd::make_index_put_backward_node(
        self, value, spec_used, accumulate);
    node->next_edges.resize(node->num_inputs());
    std::unordered_map<const vbt::autograd::AutogradMeta*,
                       vbt::core::intrusive_ptr<vbt::autograd::Node>> sinks;
    node->next_edges[0] = vbt::autograd::resolve_edge_for_tensor(self, sinks);
    node->next_edges[1] = vbt::autograd::resolve_edge_for_tensor(value, sinks);
    vbt::autograd::rebase_history(self, node);

    // Execute write under NoGradGuard to avoid nested recording.
    vbt::autograd::NoGradGuard ng;
    index_put_(self, spec_used, value, accumulate);
#else
    throw std::runtime_error(idx_errors::kErrIndexPutAutogradUnsupported);
#endif
  }

  TensorImpl result = self;
  s.clear();
  s.push_back(std::move(result));
}

extern "C" void vbt_register_indexing_kernels() {
  using vbt::dispatch::Dispatcher;
  using vbt::dispatch::KernelFunction;

  auto& D = Dispatcher::instance();

  if (!D.has("vt::index")) {
    D.registerLibrary("vt");
    D.def("vt::index(Tensor self, Tensor index, Tensor meta) -> Tensor");
    auto kf = KernelFunction::makeBoxed(/*arity=*/3, &vt_index_cpu_boxed);
    D.registerCpuKernelFunction("vt::index", kf);
    D.registerCudaKernelFunction("vt::index", kf);
  }

  if (!D.has("vt::index_put")) {
    if (!D.has("vt::index")) {
      D.registerLibrary("vt");
    }
    D.def("vt::index_put(Tensor self, Tensor index, Tensor value, Tensor meta, Tensor accumulate) -> Tensor");
    auto kf_put = KernelFunction::makeBoxed(/*arity=*/5, &vt_index_put_cpu_boxed);
    D.registerCpuKernelFunction("vt::index_put", kf_put);
    D.registerCudaKernelFunction("vt::index_put", kf_put);
  }
}
