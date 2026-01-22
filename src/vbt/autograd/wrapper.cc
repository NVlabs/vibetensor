// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/wrapper.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>

#include "vbt/autograd/saved_variable.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/kernel_function.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/core/strided_loop.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/indexing.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/core/indexing/encoded_index_spec_v0.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/autograd/detail/stats_internal.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/forward.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/storage.h"
#include "vbt/cuda/guard.h"
#include <cuda_runtime_api.h>
extern "C" {
vbt::core::TensorImpl vbt_cuda_sum_impl(const vbt::core::TensorImpl&, std::vector<int64_t>, bool);
}
#endif

#if VBT_WITH_CUDA
namespace vbt { namespace ops {
bool embedding_cuda_bounds_check_i64(const std::int64_t* idx,
                                    std::int64_t N,
                                    std::int64_t V,
                                    vbt::cuda::Stream stream,
                                    const char* op_name);
void embedding_cuda_backward_accum_f32(const std::int64_t* idx,
                                      const float* grad,
                                      float* grad_weight,
                                      std::int64_t N,
                                      std::int64_t D,
                                      std::int64_t padding_idx,
                                      vbt::cuda::Stream stream,
                                      const char* op_name);
void embedding_cuda_count_by_freq_i64(const std::int64_t* idx,
                                     std::int64_t N,
                                     std::int64_t padding_idx,
                                     std::int64_t* counts,
                                     vbt::cuda::Stream stream,
                                     const char* op_name);
void embedding_cuda_backward_accum_scaled_f32(const std::int64_t* idx,
                                             const float* grad,
                                             const std::int64_t* counts,
                                             float* grad_weight,
                                             std::int64_t N,
                                             std::int64_t D,
                                             std::int64_t padding_idx,
                                             vbt::cuda::Stream stream,
                                             const char* op_name);
}}  // namespace vbt::ops
#endif

namespace vbt { namespace autograd {

thread_local bool tls_grad_enabled = true;
thread_local bool tls_inference_mode_enabled = false;

using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::OperatorHandle;

namespace {

static bool compute_opt_in_flag_from_env_raw(const char* raw) noexcept {
  if (raw == nullptr) {
    return false;
  }

  // Trim leading whitespace.
  const unsigned char* begin = reinterpret_cast<const unsigned char*>(raw);
  while (*begin && std::isspace(*begin)) {
    ++begin;
  }
  if (*begin == '\0') {
    return false;
  }

  // Find end.
  const unsigned char* end = begin;
  while (*end) {
    ++end;
  }

  // Trim trailing whitespace.
  while (end > begin && std::isspace(*(end - 1))) {
    --end;
  }

  const std::size_t n = static_cast<std::size_t>(end - begin);
  if (n == 0) {
    return false;
  }

  auto equals_ci = [&](const char* lit, std::size_t lit_n) noexcept {
    if (n != lit_n) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const unsigned char c = begin[i];
      const unsigned char want = static_cast<unsigned char>(lit[i]);
      if (static_cast<unsigned char>(std::tolower(c)) != want) {
        return false;
      }
    }
    return true;
  };

  if ((n == 1 && begin[0] == static_cast<unsigned char>('0')) ||
      equals_ci("false", 5) ||
      equals_ci("no", 2) ||
      equals_ci("off", 3)) {
    return false;
  }

  // Any other non-empty value enables the opt-in flag.
  return true;
}

static std::once_flag g_autograd_indexing_v2_once;
static std::atomic<bool> g_autograd_indexing_v2_enabled{false};
static std::atomic<bool> g_autograd_indexing_v2_override_active{false};

static void init_autograd_indexing_v2_from_env() noexcept {
  const char* raw = std::getenv("VBT_AUTOGRAD_INDEXING_V2");
  g_autograd_indexing_v2_enabled.store(
      compute_opt_in_flag_from_env_raw(raw),
      std::memory_order_relaxed);
}

static std::once_flag g_autograd_indexing_v2_negstride_once;
static std::atomic<bool> g_autograd_indexing_v2_negstride_enabled{false};
static std::atomic<bool> g_autograd_indexing_v2_negstride_override_active{false};

static void init_autograd_indexing_v2_negstride_from_env() noexcept {
  const char* raw = std::getenv("VBT_AUTOGRAD_INDEXING_V2_NEGSTRIDE");
  g_autograd_indexing_v2_negstride_enabled.store(
      compute_opt_in_flag_from_env_raw(raw),
      std::memory_order_relaxed);
}

static vbt::core::TensorImpl make_zeros_like_cpu(
    const vbt::core::TensorImpl& like) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;

  auto dev = like.device();
  if (dev.type != kDLCPU) {
    throw std::invalid_argument("make_zeros_like_cpu: expected CPU tensor");
  }

  const auto& sizes = like.sizes();
  std::vector<std::int64_t> sizes_vec(sizes.begin(), sizes.end());

  std::int64_t ne = 1;
  if (!sizes_vec.empty()) {
    ne = 1;
    for (std::int64_t s : sizes_vec) {
      if (s == 0 || s < 0) {
        ne = 0;
        break;
      }
      if (ne > 0) {
        if (ne > std::numeric_limits<std::int64_t>::max() / s) {
          ne = 0;
          break;
        }
        ne *= s;
      }
    }
  }

  const std::size_t item_b = itemsize(like.dtype());
  const std::size_t nbytes =
      static_cast<std::size_t>(ne > 0 ? ne : 0) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
    std::memset(buf, 0, nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides(sizes_vec.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i =
           static_cast<std::ptrdiff_t>(sizes_vec.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes_vec[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return vbt::core::TensorImpl(storage, sizes_vec, strides,
                               /*storage_offset=*/0,
                               like.dtype(), dev);
}

static vbt::core::TensorImpl make_zeros_like(
    const vbt::core::TensorImpl& like) {
#if VBT_WITH_CUDA
  auto dev = like.device();
  if (dev.type == kDLCUDA) {
    using vbt::cuda::DeviceGuard;
    using vbt::cuda::DeviceIndex;

    const auto& sizes = like.sizes();
    std::vector<std::int64_t> sizes_vec(sizes.begin(), sizes.end());

    std::int64_t ne = 1;
    if (!sizes_vec.empty()) {
      ne = 1;
      for (std::int64_t s : sizes_vec) {
        if (s == 0 || s < 0) {
          ne = 0;
          break;
        }
        if (ne > 0) {
          if (ne > std::numeric_limits<std::int64_t>::max() / s) {
            ne = 0;
            break;
          }
          ne *= s;
        }
      }
    }

    const std::size_t item_b = vbt::core::itemsize(like.dtype());
    const std::size_t nbytes =
        static_cast<std::size_t>(ne > 0 ? ne : 0) * item_b;

    auto storage = vbt::cuda::new_cuda_storage(nbytes, dev.index);
    if (nbytes > 0) {
      DeviceIndex dev_index = static_cast<DeviceIndex>(dev.index);
      DeviceGuard guard(dev_index);
      cudaError_t st = cudaMemset(storage->data(), 0, nbytes);
      if (st != cudaSuccess) {
        const char* msg = cudaGetErrorString(st);
        std::string m = "make_zeros_like: cudaMemset failed: ";
        m += (msg ? msg : "");
        throw std::runtime_error(m);
      }
    }

    std::vector<std::int64_t> strides(sizes_vec.size(), 0);
    std::int64_t acc = 1;
    for (std::ptrdiff_t i =
             static_cast<std::ptrdiff_t>(sizes_vec.size()) - 1;
         i >= 0; --i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      strides[idx] = acc;
      const auto sz = sizes_vec[idx];
      acc *= (sz == 0 ? 1 : sz);
    }

    return vbt::core::TensorImpl(storage, sizes_vec, strides,
                                 /*storage_offset=*/0,
                                 like.dtype(), dev);
  }
#endif
  // Fallback to the CPU helper when CUDA is disabled or when like is a
  // CPU tensor.
  return make_zeros_like_cpu(like);
}

static vbt::core::TensorImpl make_zeros_from_meta_cpu(
    vbt::core::ScalarType dtype,
    vbt::core::Device dev,
    const std::vector<std::int64_t>& sizes_vec) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;

  if (dev.type != kDLCPU) {
    throw std::invalid_argument("make_zeros_from_meta_cpu: expected CPU device");
  }

  std::int64_t ne = 1;
  if (!sizes_vec.empty()) {
    ne = 1;
    for (std::int64_t s : sizes_vec) {
      if (s == 0 || s < 0) {
        ne = 0;
        break;
      }
      if (ne > 0) {
        if (ne > std::numeric_limits<std::int64_t>::max() / s) {
          ne = 0;
          break;
        }
        ne *= s;
      }
    }
  }

  const std::size_t item_b = itemsize(dtype);
  const std::size_t nbytes =
      static_cast<std::size_t>(ne > 0 ? ne : 0) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
    std::memset(buf, 0, nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides(sizes_vec.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i =
           static_cast<std::ptrdiff_t>(sizes_vec.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes_vec[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return vbt::core::TensorImpl(storage, sizes_vec, strides,
                               /*storage_offset=*/0,
                               dtype, dev);
}

static vbt::core::TensorImpl make_zeros_from_meta(const InputMeta& meta) {
#if VBT_WITH_CUDA
  if (meta.device.type == kDLCUDA) {
    using vbt::cuda::DeviceGuard;
    using vbt::cuda::DeviceIndex;

    const std::vector<std::int64_t>& sizes_vec = meta.sizes;

    std::int64_t ne = 1;
    if (!sizes_vec.empty()) {
      ne = 1;
      for (std::int64_t s : sizes_vec) {
        if (s == 0 || s < 0) {
          ne = 0;
          break;
        }
        if (ne > 0) {
          if (ne > std::numeric_limits<std::int64_t>::max() / s) {
            ne = 0;
            break;
          }
          ne *= s;
        }
      }
    }

    const std::size_t item_b = vbt::core::itemsize(meta.dtype);
    const std::size_t nbytes =
        static_cast<std::size_t>(ne > 0 ? ne : 0) * item_b;

    auto storage = vbt::cuda::new_cuda_storage(nbytes, meta.device.index);
    if (nbytes > 0) {
      DeviceIndex dev_index = static_cast<DeviceIndex>(meta.device.index);
      DeviceGuard guard(dev_index);
      cudaError_t st = cudaMemset(storage->data(), 0, nbytes);
      if (st != cudaSuccess) {
        const char* msg = cudaGetErrorString(st);
        std::string m = "make_zeros_from_meta: cudaMemset failed: ";
        m += (msg ? msg : "");
        throw std::runtime_error(m);
      }
    }

    std::vector<std::int64_t> strides(sizes_vec.size(), 0);
    std::int64_t acc = 1;
    for (std::ptrdiff_t i =
             static_cast<std::ptrdiff_t>(sizes_vec.size()) - 1;
         i >= 0; --i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      strides[idx] = acc;
      const auto sz = sizes_vec[idx];
      acc *= (sz == 0 ? 1 : sz);
    }

    return vbt::core::TensorImpl(storage, sizes_vec, strides,
                                 /*storage_offset=*/0,
                                 meta.dtype, meta.device);
  }
#endif

  if (meta.device.type == kDLCPU) {
    return make_zeros_from_meta_cpu(meta.dtype, meta.device, meta.sizes);
  }

  throw std::invalid_argument("make_zeros_from_meta: unsupported device");
}

static void init_node_streaminfo_from_metas(
    Node& n, const std::vector<InputMeta>& metas, bool& is_cuda_flag) {
#if VBT_WITH_CUDA
  int cuda_index = -1;
  for (const auto& m : metas) {
    if (m.device.type == kDLCUDA) {
      if (cuda_index == -1) {
        cuda_index = m.device.index;
      } else if (cuda_index != m.device.index) {
        throw std::runtime_error(
            "autograd wrapper: mixed CUDA devices in backward node metadata");
      }
    }
  }
  if (cuda_index < 0) {
    is_cuda_flag = false;
    return;
  }

  using vbt::cuda::DeviceIndex;
  using vbt::cuda::Stream;
  const auto dev_idx = static_cast<DeviceIndex>(cuda_index);
  Stream S = vbt::cuda::getCurrentStream(dev_idx);

  NodeStreamInfo& si = n.mutable_stream_info();
  si.has_canonical_stream = true;
  si.device = vbt::core::Device::cuda(cuda_index);
  si.stream_id = S.id();

  is_cuda_flag = true;
#else
  (void)n;
  (void)metas;
  is_cuda_flag = false;
#endif
}

inline bool is_supported_single_output(const std::string& fqname) {
  return fqname == "vt::relu" || fqname == "vt::add" || fqname == "vt::mul" ||
         fqname == "vt::index" || fqname == "vt::embedding";
}

static int64_t read_cpu_scalar_int64_0d(const vbt::core::TensorImpl& t,
                                       const char* op_name,
                                       const char* what) {
  using vbt::core::ScalarType;
  if (t.device().type != kDLCPU || t.dtype() != ScalarType::Int64 ||
      !t.sizes().empty()) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " must be a CPU int64 scalar (0-d)");
  }
  if (t.data() == nullptr) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " has no data");
  }
  return *static_cast<const int64_t*>(t.data());
}

static bool read_cpu_scalar_bool_0d(const vbt::core::TensorImpl& t,
                                   const char* op_name,
                                   const char* what) {
  using vbt::core::ScalarType;
  if (t.device().type != kDLCPU || t.dtype() != ScalarType::Bool ||
      !t.sizes().empty()) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " must be a CPU bool scalar (0-d)");
  }
  if (t.data() == nullptr) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " has no data");
  }
  const auto* p = static_cast<const unsigned char*>(t.data());
  return (*p != 0);
}

static uint8_t autograd_requires_grad_in_arity_masked(const std::string& fqname,
                                                     uint8_t in_arity) noexcept {
  // Only consider differentiable inputs. This matters because VibeTensor allows
  // requires_grad=True even on integer tensors.
  if (fqname == "vt::add" || fqname == "vt::mul") {
    return static_cast<uint8_t>(std::min<uint8_t>(2, in_arity));
  }
  if (fqname == "vt::relu" || fqname == "vt::index" ||
      fqname == "vt::embedding") {
    return static_cast<uint8_t>(std::min<uint8_t>(1, in_arity));
  }
  return in_arity;
}

class AddBackwardNode final : public Node, public ValidatableNode {
 public:
  explicit AddBackwardNode(std::vector<InputMeta> m)
      : in_meta_(std::move(m)) {
    name = "AddBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }
  uint32_t num_inputs() const noexcept override { return 2; }
  uint32_t num_incoming_grad_slots() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    OptionalTensor g = grads_in.size() > 0 ? std::move(grads_in[0]) : OptionalTensor{};
    std::vector<OptionalTensor> outs(2);
    if (g.has_value()) {
      outs[0] = g;
      outs[1] = g;
    }
    return outs;
  }
  const std::vector<InputMeta>& input_metas() const noexcept override { return in_meta_; }
 private:
  std::vector<InputMeta> in_meta_;
  bool                   is_cuda_{false};
};

class MulBackwardNode final : public Node, public ValidatableNode {
 public:
  MulBackwardNode(SavedVariable a, SavedVariable b, std::vector<InputMeta> m)
      : a_(std::move(a)), b_(std::move(b)), in_meta_(std::move(m)) {
    name = "MulBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }
  uint32_t num_inputs() const noexcept override { return 2; }
  uint32_t num_incoming_grad_slots() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng; // ensure wrappers don’t re-enter autograd
    std::vector<OptionalTensor> outs(2);
    if (!grads_in.empty() && grads_in[0].has_value()) {
      vbt::core::TensorImpl g = grads_in[0].value();
      vbt::core::TensorImpl a = a_.unpack();
      vbt::core::TensorImpl b = b_.unpack();

      // Conjugate Wirtinger convention (PyTorch):
      // da = g * conj(b), db = g * conj(a)
      vbt::core::TensorImpl b_conj =
          vbt::core::resolve_conj(vbt::core::conj(b));
      vbt::core::TensorImpl a_conj =
          vbt::core::resolve_conj(vbt::core::conj(a));

      BoxedStack s1;
      s1.push_back(g);
      s1.push_back(b_conj);
      Dispatcher::instance().callBoxed("vt::mul", s1);

      BoxedStack s2;
      s2.push_back(g);
      s2.push_back(a_conj);
      Dispatcher::instance().callBoxed("vt::mul", s2);

      outs[0] = s1[0];
      outs[1] = s2[0];
    }
    return outs;
  }
  const std::vector<InputMeta>& input_metas() const noexcept override { return in_meta_; }
 private:
  SavedVariable      a_;
  SavedVariable      b_;
  std::vector<InputMeta> in_meta_;
  bool               is_cuda_{false};
};

class ReluBackwardNode final : public Node, public ValidatableNode {
 public:
  explicit ReluBackwardNode(SavedVariable x, std::vector<InputMeta> m)
      : x_(std::move(x)), in_meta_(std::move(m)) {
    name = "ReluBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }
  uint32_t num_inputs() const noexcept override { return 1; }
  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    std::vector<OptionalTensor> outs(1);
    if (!grads_in.empty() && grads_in[0].has_value()) {
      outs[0] = std::move(grads_in[0]);
    }
    (void)x_;
    return outs;
  }
  const std::vector<InputMeta>& input_metas() const noexcept override { return in_meta_; }
 private:
  SavedVariable        x_;
  std::vector<InputMeta> in_meta_;
  bool                 is_cuda_{false};
};

class IdentityBackwardNode final : public Node, public ValidatableNode {
 public:
  explicit IdentityBackwardNode(std::vector<InputMeta> m) : in_meta_(std::move(m)) { name = "IdentityBackward"; }
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    std::vector<OptionalTensor> outs(1);
    if (!grads_in.empty()) outs[0] = std::move(grads_in[0]);
    return outs;
  }
  const std::vector<InputMeta>& input_metas() const noexcept override { return in_meta_; }
 private:
  std::vector<InputMeta> in_meta_;
};

class IndexBackwardNode final : public Node, public ValidatableNode {
 public:
  IndexBackwardNode(SavedVariable self, SavedVariable index, std::vector<InputMeta> m)
      : self_(std::move(self)), index_(std::move(index)), in_meta_(std::move(m)) {
    name = "IndexBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }

  uint32_t num_inputs() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    std::vector<OptionalTensor> outs(1);
    if (grads_in.empty() || !grads_in[0].has_value()) {
      return outs;
    }

    vbt::core::TensorImpl grad_out = std::move(*grads_in[0]);
    vbt::core::TensorImpl self_t   = self_.unpack();
    vbt::core::TensorImpl index_t  = index_.unpack();

    const auto dev_type = self_t.device().type;
    if (dev_type != kDLCPU && dev_type != kDLCUDA) {
      throw std::invalid_argument(
          "IndexBackward: advanced indexing backward is only implemented for CPU and CUDA tensors");
    }
    if (grad_out.device() != self_t.device()) {
      throw std::invalid_argument(
          "IndexBackward: grad_out must be on the same device as self");
    }
    const bool index_ok =
        (index_t.device() == self_t.device()) ||
        (self_t.device().type == kDLCUDA && index_t.device().type == kDLCPU);
    if (!index_ok) {
      throw std::invalid_argument(
          "IndexBackward: index must be on the same device as self (or CPU for CUDA self)");
    }

    if (!vbt::core::indexing::advanced_indexing_enabled()) {
      throw std::runtime_error(
          vbt::core::indexing::errors::kErrAdvDisabledCore);
    }

    vbt::core::TensorImpl grad_self = make_zeros_like(self_t);

    using vbt::core::indexing::IndexSpec;
    using vbt::core::indexing::TensorIndex;

    IndexSpec spec;
    spec.items.emplace_back(TensorIndex(index_t));

    try {
      vbt::core::indexing::index_put_(grad_self, spec, grad_out,
                                      /*accumulate=*/true);
    } catch (const std::runtime_error&) {
      // Surface index_put_ errors (e.g., unsupported dtype/accumulate
      // combinations on CUDA) directly to the caller.
      throw;
    }

    outs[0] = std::move(grad_self);
    return outs;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return in_meta_;
  }

 private:
  SavedVariable      self_;
  SavedVariable      index_;
  std::vector<InputMeta> in_meta_;
  bool               is_cuda_{false};
};

class IndexBackwardNodeV2 final : public Node, public ValidatableNode {
 public:
  IndexBackwardNodeV2(SavedVariable index, SavedVariable meta,
                      std::vector<InputMeta> m)
      : index_(std::move(index)), meta_(std::move(meta)), in_meta_(std::move(m)) {
    name = "IndexBackwardV2";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }

  uint32_t num_inputs() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    std::vector<OptionalTensor> outs(1);
    if (grads_in.empty() || !grads_in[0].has_value()) {
      return outs;
    }

    vbt::core::TensorImpl grad_out = std::move(*grads_in[0]);
    vbt::core::TensorImpl index_t  = index_.unpack();
    vbt::core::TensorImpl meta_t   = meta_.unpack();

    if (in_meta_.empty()) {
      throw std::logic_error("IndexBackwardV2: missing input meta");
    }

    const auto dev_type = in_meta_[0].device.type;
    if (dev_type != kDLCPU && dev_type != kDLCUDA) {
      throw std::invalid_argument(
          "IndexBackward: advanced indexing backward is only implemented for CPU and CUDA tensors");
    }
    if (grad_out.device() != in_meta_[0].device) {
      throw std::invalid_argument(
          "IndexBackward: grad_out must be on the same device as self");
    }
    const bool index_ok =
        (index_t.device() == in_meta_[0].device) ||
        (in_meta_[0].device.type == kDLCUDA && index_t.device().type == kDLCPU);
    if (!index_ok) {
      throw std::invalid_argument(
          "IndexBackward: index must be on the same device as self (or CPU for CUDA self)");
    }

    if (!vbt::core::indexing::advanced_indexing_enabled()) {
      throw std::runtime_error(
          vbt::core::indexing::errors::kErrAdvDisabledCore);
    }

    vbt::core::TensorImpl grad_self = make_zeros_from_meta(in_meta_[0]);

    // Avoid mutating user-visible indices in backward; the core advanced
    // indexing path canonicalizes Int64 indices in-place.
    if (index_t.dtype() == vbt::core::ScalarType::Int64 &&
        index_t.is_contiguous()) {
      index_t = vbt::core::clone_contiguous_same_device(index_t);
    }

    vbt::core::indexing::IndexSpec spec =
        vbt::core::indexing::decode_encoded_index_spec_v0(
            index_t, meta_t, "vt::index");

    try {
      vbt::core::indexing::index_put_(grad_self, spec, grad_out,
                                      /*accumulate=*/true);
    } catch (const std::runtime_error&) {
      // Surface index_put_ errors (e.g., unsupported dtype/accumulate
      // combinations on CUDA) directly to the caller.
      throw;
    }

    outs[0] = std::move(grad_self);
    return outs;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return in_meta_;
  }

 private:
  SavedVariable      index_;
  SavedVariable      meta_;
  std::vector<InputMeta> in_meta_;
  bool               is_cuda_{false};
};

class EmbeddingBackwardNode final : public Node, public ValidatableNode {
 public:
  EmbeddingBackwardNode(SavedVariable weight,
                        SavedVariable indices,
                        int64_t padding_idx,
                        bool scale_grad_by_freq,
                        bool sparse,
                        std::vector<InputMeta> m)
      : weight_(std::move(weight)),
        indices_(std::move(indices)),
        padding_idx_(padding_idx),
        scale_grad_by_freq_(scale_grad_by_freq),
        sparse_(sparse),
        in_meta_(std::move(m)) {
    name = "EmbeddingBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }

  uint32_t num_inputs() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    std::vector<OptionalTensor> outs(1);
    if (grads_in.empty() || !grads_in[0].has_value()) {
      return outs;
    }

    if (sparse_) {
      throw std::runtime_error(
          "vt::embedding: sparse gradients are not supported");
    }

    vbt::core::TensorImpl grad_out   = std::move(*grads_in[0]);
    vbt::core::TensorImpl weight_t   = weight_.unpack();
    vbt::core::TensorImpl indices_t  = indices_.unpack();

    const auto dev_type = weight_t.device().type;
#if VBT_WITH_CUDA
    if (dev_type == kDLCUDA) {
      // CUDA backward: dense grad_weight accumulation.
      // NOTE: atomicAdd accumulation is potentially nondeterministic when indices
      // contain duplicates (the update order is undefined).

      if (indices_t.device() != weight_t.device() ||
          grad_out.device() != weight_t.device()) {
        throw std::invalid_argument(
            "EmbeddingBackward: weight, indices, and grad_out must be on the same device");
      }

      if (weight_t.dtype() != vbt::core::ScalarType::Float32 ||
          weight_t.sizes().size() != 2) {
        throw std::invalid_argument(
            "EmbeddingBackward: expected float32 rank-2 weight");
      }
      if (indices_t.dtype() != vbt::core::ScalarType::Int64) {
        throw std::invalid_argument(
            "EmbeddingBackward: CUDA indices must be int64");
      }
      if (grad_out.dtype() != vbt::core::ScalarType::Float32) {
        throw std::invalid_argument(
            "EmbeddingBackward: grad_out must be float32");
      }

      const int64_t V = weight_t.sizes()[0];
      const int64_t D = weight_t.sizes()[1];

      if (padding_idx_ != -1 && (padding_idx_ < 0 || padding_idx_ >= V)) {
        throw std::out_of_range(
            "EmbeddingBackward: padding_idx out of range");
      }

      // grad_out must match indices.shape + (D,)
      const auto& idx_sizes = indices_t.sizes();
      const auto& go_sizes = grad_out.sizes();
      if (go_sizes.size() != idx_sizes.size() + 1) {
        throw std::invalid_argument(
            "EmbeddingBackward: grad_out shape mismatch");
      }
      for (std::size_t d = 0; d < idx_sizes.size(); ++d) {
        if (go_sizes[d] != idx_sizes[d]) {
          throw std::invalid_argument(
              "EmbeddingBackward: grad_out shape mismatch");
        }
      }
      if (go_sizes.back() != D) {
        throw std::invalid_argument(
            "EmbeddingBackward: grad_out shape mismatch");
      }

      vbt::core::TensorImpl grad_weight = make_zeros_like(weight_t);

      auto checked_numel_from_sizes = [](const std::vector<int64_t>& sizes,
                                         const char* what) -> int64_t {
        if (sizes.empty()) {
          return 1;  // scalar
        }
        int64_t n = 1;
        for (int64_t s : sizes) {
          if (s < 0) {
            throw std::invalid_argument(
                std::string("EmbeddingBackward: ") + what +
                " size must be >= 0");
          }
          if (s == 0) {
            return 0;
          }
          int64_t tmp = 0;
          if (!vbt::core::checked_mul_i64(n, s, tmp)) {
            throw std::overflow_error(
                std::string("EmbeddingBackward: ") + what + " numel overflow");
          }
          n = tmp;
        }
        return n;
      };

      const int64_t N = checked_numel_from_sizes(idx_sizes, "indices");
      if (N == 0) {
        outs[0] = std::move(grad_weight);
        return outs;
      }

      // Canonicalize indices to contiguous for strict bounds checks.
      vbt::core::TensorImpl idx_contig = indices_t;
      if (!idx_contig.is_contiguous()) {
        idx_contig = vbt::core::clone_contiguous_same_device(idx_contig);
      }

      const int64_t* idx_ptr = static_cast<const int64_t*>(idx_contig.data());
      if (!idx_ptr) {
        throw std::invalid_argument(
            "EmbeddingBackward: indices has no data");
      }

      const int dev_index = weight_t.device().index;
      vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(dev_index));
      vbt::cuda::Stream stream = vbt::cuda::getCurrentStream(
          static_cast<vbt::cuda::DeviceIndex>(dev_index));

      // Record before bounds check for exception safety.
      vbt::cuda::record_stream(idx_contig.storage(), stream);

      // NOTE: CUDA bounds checking requires a device-to-host transfer and a stream
      // synchronize, so EmbeddingBackward is not CUDA-graph-capture safe.
      const bool has_oob =
          vbt::ops::embedding_cuda_bounds_check_i64(
              idx_ptr, N, V, stream, "vt::embedding");
      if (has_oob) {
        throw std::out_of_range("vt::embedding: index out of range in self");
      }

      if (D == 0) {
        outs[0] = std::move(grad_weight);
        return outs;
      }

      vbt::core::TensorImpl grad_contig = grad_out;
      if (!grad_contig.is_contiguous()) {
        grad_contig = vbt::core::clone_contiguous_same_device(grad_contig);
      }

      const float* grad_ptr = static_cast<const float*>(grad_contig.data());
      float* gw_ptr = static_cast<float*>(grad_weight.data());
      if (!grad_ptr || !gw_ptr) {
        throw std::invalid_argument(
            "EmbeddingBackward: internal: null data");
      }

      if (!scale_grad_by_freq_) {
        vbt::ops::embedding_cuda_backward_accum_f32(
            idx_ptr, grad_ptr, gw_ptr, N, D, padding_idx_, stream, "vt::embedding");
      } else {
        // Two-pass CUDA scale_grad_by_freq:
        // 1) counts[row] = freq(row) over non-padding occurrences
        // 2) atomicAdd(grad / counts[row]) into grad_weight

        std::size_t counts_bytes = 0;
        if (V > 0) {
          const int64_t item_b_i64 = static_cast<int64_t>(sizeof(int64_t));
          int64_t bytes_i64 = 0;
          if (!vbt::core::checked_mul_i64(V, item_b_i64, bytes_i64)) {
            throw std::overflow_error(
                "EmbeddingBackward: counts byte size overflow");
          }
          if (bytes_i64 < 0 ||
              static_cast<std::uint64_t>(bytes_i64) >
                  static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error(
                "EmbeddingBackward: counts byte size overflow");
          }
          counts_bytes = static_cast<std::size_t>(bytes_i64);
        }

        auto counts_storage = vbt::cuda::new_cuda_storage(counts_bytes, dev_index);
        vbt::core::TensorImpl counts(std::move(counts_storage),
                                     std::vector<int64_t>{V},
                                     std::vector<int64_t>{1},
                                     /*storage_offset=*/0,
                                     vbt::core::ScalarType::Int64,
                                     weight_t.device());

        if (counts_bytes > 0) {
          cudaError_t st = cudaMemsetAsync(
              counts.data(), 0, counts_bytes,
              reinterpret_cast<cudaStream_t>(stream.handle()));
          if (st != cudaSuccess) {
            const char* msg = cudaGetErrorString(st);
            std::string m = "EmbeddingBackward: cudaMemsetAsync(counts) failed: ";
            m += (msg ? msg : "");
            throw std::runtime_error(m);
          }
          vbt::cuda::record_stream(counts.storage(), stream);
        }

        auto* counts_ptr = static_cast<int64_t*>(counts.data());
        if (!counts_ptr && V > 0) {
          throw std::invalid_argument(
              "EmbeddingBackward: internal: counts has no data");
        }

        vbt::ops::embedding_cuda_count_by_freq_i64(
            idx_ptr, N, padding_idx_, counts_ptr, stream, "vt::embedding");

        vbt::ops::embedding_cuda_backward_accum_scaled_f32(
            idx_ptr, grad_ptr, counts_ptr, gw_ptr, N, D, padding_idx_, stream,
            "vt::embedding");

        vbt::cuda::record_stream(counts.storage(), stream);
      }

      vbt::cuda::record_stream(grad_weight.storage(), stream);
      vbt::cuda::record_stream(grad_contig.storage(), stream);
      vbt::cuda::record_stream(idx_contig.storage(), stream);

      outs[0] = std::move(grad_weight);
      return outs;
    }
#endif
    if (dev_type != kDLCPU) {
      throw std::invalid_argument(
          "EmbeddingBackward: only CPU tensors are supported in this build");
    }
    if (indices_t.device() != weight_t.device() ||
        grad_out.device() != weight_t.device()) {
      throw std::invalid_argument(
          "EmbeddingBackward: weight, indices, and grad_out must be on the same device");
    }

    if (weight_t.dtype() != vbt::core::ScalarType::Float32 ||
        weight_t.sizes().size() != 2) {
      throw std::invalid_argument(
          "EmbeddingBackward: expected float32 rank-2 weight");
    }
    if (!(indices_t.dtype() == vbt::core::ScalarType::Int32 ||
          indices_t.dtype() == vbt::core::ScalarType::Int64)) {
      throw std::invalid_argument(
          "EmbeddingBackward: indices must be int32 or int64");
    }
    if (grad_out.dtype() != vbt::core::ScalarType::Float32) {
      throw std::invalid_argument(
          "EmbeddingBackward: grad_out must be float32");
    }

    const int64_t V = weight_t.sizes()[0];
    const int64_t D = weight_t.sizes()[1];

    if (padding_idx_ != -1 && (padding_idx_ < 0 || padding_idx_ >= V)) {
      throw std::out_of_range(
          "EmbeddingBackward: padding_idx out of range");
    }

    // Canonicalize indices and grad_out to contiguous buffers so we can
    // accumulate in simple row-major order.
    vbt::core::TensorImpl idx_contig = indices_t;
    if (!idx_contig.is_contiguous()) {
      idx_contig = vbt::core::clone_cpu(idx_contig);
    }

    vbt::core::TensorImpl grad_contig = grad_out;
    if (!grad_contig.is_contiguous()) {
      grad_contig = vbt::core::clone_cpu(grad_contig);
    }

    // grad_out must match indices.shape + (D,)
    const auto& idx_sizes = indices_t.sizes();
    const auto& go_sizes = grad_contig.sizes();
    if (go_sizes.size() != idx_sizes.size() + 1) {
      throw std::invalid_argument(
          "EmbeddingBackward: grad_out shape mismatch");
    }
    for (std::size_t d = 0; d < idx_sizes.size(); ++d) {
      if (go_sizes[d] != idx_sizes[d]) {
        throw std::invalid_argument(
            "EmbeddingBackward: grad_out shape mismatch");
      }
    }
    if (go_sizes.back() != D) {
      throw std::invalid_argument(
          "EmbeddingBackward: grad_out shape mismatch");
    }

    vbt::core::TensorImpl grad_weight = make_zeros_like(weight_t);

    auto checked_numel_from_sizes = [](const std::vector<int64_t>& sizes,
                                       const char* what) -> int64_t {
      if (sizes.empty()) {
        return 1;  // scalar
      }
      int64_t n = 1;
      for (int64_t s : sizes) {
        if (s < 0) {
          throw std::invalid_argument(
              std::string("EmbeddingBackward: ") + what + " size must be >= 0");
        }
        if (s == 0) {
          return 0;
        }
        int64_t tmp = 0;
        if (!vbt::core::checked_mul_i64(n, s, tmp)) {
          throw std::overflow_error(
              std::string("EmbeddingBackward: ") + what + " numel overflow");
        }
        n = tmp;
      }
      return n;
    };

    const int64_t N = checked_numel_from_sizes(idx_sizes, "indices");
    if (N == 0) {
      outs[0] = std::move(grad_weight);
      return outs;
    }

    const void* idx_data_void = idx_contig.data();
    if (!idx_data_void) {
      throw std::invalid_argument(
          "EmbeddingBackward: indices has no data");
    }

    // Even when D == 0 (empty embeddings), embedding enforces strict bounds.
    if (D == 0) {
      if (idx_contig.dtype() == vbt::core::ScalarType::Int64) {
        const int64_t* idx_data = static_cast<const int64_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = idx_data[i];
          if (idx < 0 || idx >= V) {
            throw std::out_of_range(
                "vt::embedding: index out of range in self");
          }
        }
      } else {
        const int32_t* idx_data = static_cast<const int32_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = static_cast<int64_t>(idx_data[i]);
          if (idx < 0 || idx >= V) {
            throw std::out_of_range(
                "vt::embedding: index out of range in self");
          }
        }
      }

      outs[0] = std::move(grad_weight);
      return outs;
    }

    // Guard N*D pointer arithmetic.
    {
      int64_t out_numel = 0;
      if (!vbt::core::checked_mul_i64(N, D, out_numel)) {
        throw std::overflow_error(
            "EmbeddingBackward: grad_out numel overflow");
      }
      (void)out_numel;
    }

    const void* grad_data_void = grad_contig.data();
    void* gw_data_void = grad_weight.data();
    if (!grad_data_void || !gw_data_void) {
      throw std::invalid_argument(
          "EmbeddingBackward: internal: null data");
    }

    const float* grad_data = static_cast<const float*>(grad_data_void);
    float* gw_data = static_cast<float*>(gw_data_void);

    auto accum_row = [&](int64_t i, int64_t idx) {
      if (padding_idx_ != -1 && idx == padding_idx_) {
        return;
      }

      int64_t dst_off = 0;
      if (!vbt::core::checked_mul_i64(idx, D, dst_off)) {
        throw std::overflow_error(
            "EmbeddingBackward: grad_weight offset overflow");
      }
      int64_t src_off = 0;
      if (!vbt::core::checked_mul_i64(i, D, src_off)) {
        throw std::overflow_error(
            "EmbeddingBackward: grad_out offset overflow");
      }

      float* dst = gw_data + dst_off;
      const float* src = grad_data + src_off;
      for (int64_t j = 0; j < D; ++j) {
        dst[j] += src[j];
      }
    };

    if (!scale_grad_by_freq_) {
      if (idx_contig.dtype() == vbt::core::ScalarType::Int64) {
        const int64_t* idx_data = static_cast<const int64_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = idx_data[i];
          if (idx < 0 || idx >= V) {
            throw std::out_of_range(
                "vt::embedding: index out of range in self");
          }
          accum_row(i, idx);
        }
      } else {
        const int32_t* idx_data = static_cast<const int32_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = static_cast<int64_t>(idx_data[i]);
          if (idx < 0 || idx >= V) {
            throw std::out_of_range(
                "vt::embedding: index out of range in self");
          }
          accum_row(i, idx);
        }
      }
    } else {
      std::vector<int64_t> counts(static_cast<std::size_t>(V), 0);

      auto bump = [&](int64_t idx) {
        if (padding_idx_ != -1 && idx == padding_idx_) {
          return;
        }
        counts[static_cast<std::size_t>(idx)] += 1;
      };

      if (idx_contig.dtype() == vbt::core::ScalarType::Int64) {
        const int64_t* idx_data = static_cast<const int64_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = idx_data[i];
          if (idx < 0 || idx >= V) {
            throw std::out_of_range(
                "vt::embedding: index out of range in self");
          }
          bump(idx);
        }
      } else {
        const int32_t* idx_data = static_cast<const int32_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = static_cast<int64_t>(idx_data[i]);
          if (idx < 0 || idx >= V) {
            throw std::out_of_range(
                "vt::embedding: index out of range in self");
          }
          bump(idx);
        }
      }

      auto accum_row_scaled = [&](int64_t i, int64_t idx) {
        if (padding_idx_ != -1 && idx == padding_idx_) {
          return;
        }

        const int64_t count = counts[static_cast<std::size_t>(idx)];
        if (count <= 0) {
          throw std::runtime_error(
              "EmbeddingBackward: internal: zero count for index");
        }
        const float scale = 1.0f / static_cast<float>(count);

        int64_t dst_off = 0;
        if (!vbt::core::checked_mul_i64(idx, D, dst_off)) {
          throw std::overflow_error(
              "EmbeddingBackward: grad_weight offset overflow");
        }
        int64_t src_off = 0;
        if (!vbt::core::checked_mul_i64(i, D, src_off)) {
          throw std::overflow_error(
              "EmbeddingBackward: grad_out offset overflow");
        }

        float* dst = gw_data + dst_off;
        const float* src = grad_data + src_off;
        for (int64_t j = 0; j < D; ++j) {
          dst[j] += src[j] * scale;
        }
      };

      if (idx_contig.dtype() == vbt::core::ScalarType::Int64) {
        const int64_t* idx_data = static_cast<const int64_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = idx_data[i];
          accum_row_scaled(i, idx);
        }
      } else {
        const int32_t* idx_data = static_cast<const int32_t*>(idx_data_void);
        for (int64_t i = 0; i < N; ++i) {
          const int64_t idx = static_cast<int64_t>(idx_data[i]);
          accum_row_scaled(i, idx);
        }
      }
    }

    outs[0] = std::move(grad_weight);
    return outs;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return in_meta_;
  }

 private:
  SavedVariable         weight_;
  SavedVariable         indices_;
  int64_t               padding_idx_{-1};
  bool                  scale_grad_by_freq_{false};
  bool                  sparse_{false};
  std::vector<InputMeta> in_meta_;
  bool                  is_cuda_{false};
};

static void add_strided_float32_signed_cpu(const vbt::core::TensorImpl& dst,
                                          const vbt::core::TensorImpl& src) {
  using vbt::core::checked_mul_i64;
  using vbt::core::checked_abs_i64_hdr;

  if (dst.sizes() != src.sizes()) {
    throw std::invalid_argument("basic index view backward: size mismatch");
  }
  if (dst.numel() == 0) return;

  const auto& sizes    = dst.sizes();
  const auto& dstrides = dst.strides();
  const auto& sstrides = src.strides();

  const std::size_t ndim = sizes.size();
  const int64_t item_b = static_cast<int64_t>(dst.itemsize());

  std::vector<int64_t> perm;
  perm.reserve(ndim);
  std::vector<int64_t> dst_stride_bytes(ndim, 0);
  std::vector<int64_t> src_stride_bytes(ndim, 0);
  std::vector<int64_t> step_bytes(ndim, 0);

  for (std::size_t i = 0; i < ndim; ++i) {
    perm.push_back(static_cast<int64_t>(i));

    int64_t tmp = 0;
    if (!checked_mul_i64(dstrides[i], item_b, tmp)) {
      throw std::overflow_error(
          "basic index view backward: overflow computing dst stride bytes");
    }
    dst_stride_bytes[i] = tmp;
    int64_t abs_st = 0;
    if (!checked_abs_i64_hdr(tmp, abs_st)) {
      throw std::overflow_error(
          "basic index view backward: overflow computing absolute dst stride bytes");
    }
    step_bytes[i] = abs_st;

    tmp = 0;
    if (!checked_mul_i64(sstrides[i], item_b, tmp)) {
      throw std::overflow_error(
          "basic index view backward: overflow computing src stride bytes");
    }
    src_stride_bytes[i] = tmp;
  }

  std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    const auto ia = static_cast<std::size_t>(a);
    const auto ib = static_cast<std::size_t>(b);
    const int64_t sa = (sizes[ia] == 1 ? std::numeric_limits<int64_t>::max() : step_bytes[ia]);
    const int64_t sb = (sizes[ib] == 1 ? std::numeric_limits<int64_t>::max() : step_bytes[ib]);
    return sa < sb;
  });

  std::vector<int64_t> it_sizes;
  std::vector<int64_t> it_dsteps;
  std::vector<int64_t> it_ssteps;
  it_sizes.reserve(ndim);
  it_dsteps.reserve(ndim);
  it_ssteps.reserve(ndim);
  for (int64_t d : perm) {
    const std::size_t i = static_cast<std::size_t>(d);
    if (sizes[i] == 1) continue;
    it_sizes.push_back(sizes[i]);
    it_dsteps.push_back(dst_stride_bytes[i]);
    it_ssteps.push_back(src_stride_bytes[i]);
  }

  auto* pd = static_cast<std::uint8_t*>(dst.data());
  const std::uint8_t* ps = static_cast<const std::uint8_t*>(src.data());

  const int64_t it_ndim = static_cast<int64_t>(it_sizes.size());
  if (it_ndim == 0) {
    *reinterpret_cast<float*>(pd) += *reinterpret_cast<const float*>(ps);
    return;
  }

  // NOTE: This iterator uses signed step bytes to preserve the logical indexing
  // order of negative-stride views. It must never form out-of-bounds pointers
  // (C++ UB), so we only apply pointer steps when the corresponding index
  // increment stays in-range.
  std::vector<int64_t> idx(static_cast<std::size_t>(it_ndim), 0);
  while (true) {
    *reinterpret_cast<float*>(pd) += *reinterpret_cast<const float*>(ps);

    int d = static_cast<int>(it_ndim - 1);
    for (; d >= 0; --d) {
      const std::size_t di = static_cast<std::size_t>(d);
      const int64_t next = idx[di] + 1;
      if (next < it_sizes[di]) {
        idx[di] = next;
        pd += static_cast<std::ptrdiff_t>(it_dsteps[di]);
        ps += static_cast<std::ptrdiff_t>(it_ssteps[di]);
        break;
      }

      const int64_t wrap_count = it_sizes[di] - 1;
      int64_t delta_d = 0;
      int64_t delta_s = 0;
      if (!checked_mul_i64(it_dsteps[di], wrap_count, delta_d) ||
          !checked_mul_i64(it_ssteps[di], wrap_count, delta_s)) {
        throw std::overflow_error(
            "basic index view backward: overflow during pointer carry");
      }
      pd -= static_cast<std::ptrdiff_t>(delta_d);
      ps -= static_cast<std::ptrdiff_t>(delta_s);
      idx[di] = 0;
    }
    if (d < 0) {
      break;
    }
  }
}

class BasicIndexViewBackwardNode final : public Node, public ValidatableNode {
 public:
  BasicIndexViewBackwardNode(std::vector<InputMeta> m,
                             vbt::core::indexing::IndexSpec spec)
      : in_meta_(std::move(m)), spec_(std::move(spec)) {
    name = "BasicIndexViewBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }

  uint32_t num_inputs() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    std::vector<OptionalTensor> outs(1);
    if (grads_in.empty() || !grads_in[0].has_value()) {
      return outs;
    }

    vbt::core::TensorImpl grad_out = std::move(*grads_in[0]);

    if (grad_out.device() != in_meta_[0].device) {
      throw std::invalid_argument(
          "BasicIndexViewBackward: grad_out device does not match base");
    }

    if (grad_out.dtype() != in_meta_[0].dtype) {
      throw std::invalid_argument(
          "BasicIndexViewBackward: grad_out dtype does not match base");
    }

    // v2 scope: CPU Float32 only.
    if (grad_out.device().type != kDLCPU ||
        grad_out.dtype() != vbt::core::ScalarType::Float32) {
      throw std::runtime_error(
          "basic index view backward: only Float32 CPU is supported");
    }

    // Normalize grad_out to a safe contiguous layout.
    bool needs_clone = !grad_out.is_non_overlapping_and_dense();
    {
      const auto& sizes = grad_out.sizes();
      const auto& strides = grad_out.strides();
      for (std::size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] < 0) {
          needs_clone = true;
          break;
        }
        if (strides[i] == 0 && sizes[i] > 1) {
          needs_clone = true;
          break;
        }
      }
    }
    if (needs_clone) {
      grad_out = vbt::core::clone_contiguous_same_device(grad_out);
    }

    vbt::core::TensorImpl grad_base = make_zeros_from_meta(in_meta_[0]);

    // Build a view into grad_base with the same IndexSpec used in forward.
    vbt::core::TensorImpl dst = vbt::core::indexing::basic_index(grad_base, spec_);

    bool has_negative_stride = false;
    {
      const auto& dsizes = dst.sizes();
      const auto& dstrides = dst.strides();
      for (std::size_t i = 0; i < dstrides.size(); ++i) {
        if (dstrides[i] < 0 && dsizes[i] > 1) {
          has_negative_stride = true;
          break;
        }
      }
    }
    if (has_negative_stride && !autograd_indexing_v2_negstride_enabled()) {
      throw std::runtime_error(
          "basic index view backward: negative strides are not supported");
    }

    if (dst.sizes() != grad_out.sizes()) {
      throw std::invalid_argument(
          "BasicIndexViewBackward: grad_out shape does not match indexed view");
    }

    // Scatter by accumulating into the view.
    if (has_negative_stride) {
      add_strided_float32_signed_cpu(dst, grad_out);
    } else {
      using vbt::core::for_each_1out_1in;
      for_each_1out_1in(dst, grad_out,
                       [](std::uint8_t* pd, const std::uint8_t* ps) {
                         *reinterpret_cast<float*>(pd) +=
                             *reinterpret_cast<const float*>(ps);
                       });
    }

    outs[0] = std::move(grad_base);
    return outs;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return in_meta_;
  }

 private:
  std::vector<InputMeta> in_meta_;
  vbt::core::indexing::IndexSpec spec_;
  bool is_cuda_{false};
};

static bool needs_clone_for_backward(const vbt::core::TensorImpl& t) {
  if (!t.is_non_overlapping_and_dense()) {
    return true;
  }
  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  for (std::size_t i = 0; i < strides.size(); ++i) {
    if (strides[i] < 0) {
      return true;
    }
    if (strides[i] == 0 && sizes[i] > 1) {
      return true;
    }
  }
  return false;
}

static vbt::core::TensorImpl sum_to_shape_cpu_float32(
    const vbt::core::TensorImpl& src,
    const std::vector<std::int64_t>& target_sizes) {
  using vbt::core::ScalarType;

  if (src.device().type != kDLCPU || src.dtype() != ScalarType::Float32) {
    throw std::runtime_error(
        "sum_to_shape: expected Float32 CPU tensor");
  }

  std::vector<std::int64_t> src_sizes(src.sizes().begin(), src.sizes().end());
  if (src_sizes == target_sizes) {
    return src;
  }

  if (!src.is_contiguous()) {
    throw std::runtime_error(
        "sum_to_shape: expected contiguous source");
  }

  const std::size_t src_rank = src_sizes.size();
  const std::size_t tgt_rank = target_sizes.size();
  if (src_rank < tgt_rank) {
    throw std::invalid_argument(
        "sum_to_shape: target rank exceeds source rank");
  }

  const std::size_t lead = src_rank - tgt_rank;
  for (std::size_t d = 0; d < tgt_rank; ++d) {
    const std::int64_t s = src_sizes[lead + d];
    const std::int64_t t = target_sizes[d];
    if (t != 1 && t != s) {
      throw std::invalid_argument(
          "sum_to_shape: target shape is not broadcast-compatible");
    }
  }

  vbt::core::TensorImpl out = make_zeros_from_meta_cpu(
      ScalarType::Float32, vbt::core::Device::cpu(), target_sizes);

  const std::int64_t N = src.numel();
  if (N == 0) {
    return out;
  }

  auto* ps = static_cast<const float*>(src.data());
  auto* po = static_cast<float*>(out.data());

  std::vector<std::int64_t> out_strides(tgt_rank, 0);
  {
    std::int64_t acc = 1;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(tgt_rank) - 1; i >= 0; --i) {
      const std::size_t ui = static_cast<std::size_t>(i);
      out_strides[ui] = acc;
      const auto sz = target_sizes[ui];
      acc *= (sz == 0 ? 1 : sz);
    }
  }

  std::vector<std::int64_t> coords(src_rank, 0);

  for (std::int64_t linear = 0; linear < N; ++linear) {
    std::int64_t tmp = linear;
    for (std::ptrdiff_t di = static_cast<std::ptrdiff_t>(src_rank) - 1; di >= 0; --di) {
      const std::size_t d = static_cast<std::size_t>(di);
      const std::int64_t sz = src_sizes[d];
      std::int64_t c = 0;
      if (sz > 0) {
        c = tmp % sz;
        tmp /= sz;
      }
      coords[d] = c;
    }

    std::int64_t out_off = 0;
    for (std::size_t d = 0; d < src_rank; ++d) {
      if (d < lead) {
        continue;  // reduced leading dim
      }
      const std::size_t td = d - lead;
      const std::int64_t t_sz = target_sizes[td];
      if (t_sz == 1) {
        continue;  // reduced broadcast dim
      }
      out_off += coords[d] * out_strides[td];
    }

    po[out_off] += ps[linear];
  }

  return out;
}

#if VBT_WITH_CUDA
static vbt::core::TensorImpl sum_to_shape_cuda_float32(
    const vbt::core::TensorImpl& src,
    const std::vector<std::int64_t>& target_sizes) {
  using vbt::core::ScalarType;

  if (src.device().type != kDLCUDA || src.dtype() != ScalarType::Float32) {
    throw std::runtime_error(
        "sum_to_shape: expected Float32 CUDA tensor");
  }

  std::vector<std::int64_t> src_sizes(src.sizes().begin(), src.sizes().end());
  if (src_sizes == target_sizes) {
    return src;
  }

  const std::size_t src_rank = src_sizes.size();
  const std::size_t tgt_rank = target_sizes.size();
  if (src_rank < tgt_rank) {
    throw std::invalid_argument(
        "sum_to_shape: target rank exceeds source rank");
  }

  const std::size_t lead = src_rank - tgt_rank;
  for (std::size_t d = 0; d < tgt_rank; ++d) {
    const std::int64_t s = src_sizes[lead + d];
    const std::int64_t t = target_sizes[d];
    if (t != 1 && t != s) {
      throw std::invalid_argument(
          "sum_to_shape: target shape is not broadcast-compatible");
    }
  }

  vbt::core::TensorImpl out = src;

  // Sum over leading broadcast dims by repeatedly reducing dim 0.
  std::size_t cur_rank = src_rank;
  while (cur_rank > tgt_rank) {
    out = ::vbt_cuda_sum_impl(out, std::vector<int64_t>{0}, /*keepdim=*/false);
    --cur_rank;
  }

  // Sum over expanded dims (target size 1) with keepdim=true.
  for (std::size_t d = 0; d < tgt_rank; ++d) {
    if (target_sizes[d] == 1 && out.sizes()[d] != 1) {
      out = ::vbt_cuda_sum_impl(
          out,
          std::vector<int64_t>{static_cast<int64_t>(d)},
          /*keepdim=*/true);
    }
  }

  return out;
}
#endif

static void fill_zero_float32_cpu(const vbt::core::TensorImpl& t) {
  using vbt::core::for_each_1out_inplace;

  if (t.device().type != kDLCPU || t.dtype() != vbt::core::ScalarType::Float32) {
    throw std::runtime_error(
        "fill_zero_float32_cpu: expected Float32 CPU tensor");
  }

  if (t.numel() == 0) {
    return;
  }

  for_each_1out_inplace(t, [](std::uint8_t* p) {
    *reinterpret_cast<float*>(p) = 0.0f;
  });
}

class BasicIndexPutBackwardNode final : public Node, public ValidatableNode {
 public:
  BasicIndexPutBackwardNode(std::vector<InputMeta> m,
                            vbt::core::indexing::IndexSpec spec)
      : in_meta_(std::move(m)), spec_(std::move(spec)) {
    name = "BasicIndexPutBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }

  uint32_t num_inputs() const noexcept override { return 2; }
  uint32_t num_incoming_grad_slots() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;

    std::vector<OptionalTensor> outs(2);
    if (grads_in.empty() || !grads_in[0].has_value()) {
      return outs;
    }

    vbt::core::TensorImpl grad_out = std::move(*grads_in[0]);

    if (in_meta_.size() < 2) {
      throw std::logic_error("BasicIndexPutBackward: missing input meta");
    }

    if (grad_out.device() != in_meta_[0].device) {
      throw std::invalid_argument(
          "BasicIndexPutBackward: grad_out device does not match self");
    }
    if (grad_out.dtype() != in_meta_[0].dtype) {
      throw std::invalid_argument(
          "BasicIndexPutBackward: grad_out dtype does not match self");
    }
    if (grad_out.sizes() != in_meta_[0].sizes) {
      throw std::invalid_argument(
          "BasicIndexPutBackward: grad_out shape does not match self");
    }

    // v2 scope: CPU Float32 only.
    if (grad_out.device().type != kDLCPU ||
        grad_out.dtype() != vbt::core::ScalarType::Float32) {
      throw std::runtime_error(
          "basic index put backward: only Float32 CPU is supported");
    }

    if (needs_clone_for_backward(grad_out)) {
      grad_out = vbt::core::clone_contiguous_same_device(grad_out);
    }

    const bool need_self_grad =
        next_edges.size() >= 1 && static_cast<bool>(next_edges[0].fn);
    const bool need_value_grad =
        next_edges.size() >= 2 && static_cast<bool>(next_edges[1].fn);

    if (need_value_grad) {
      vbt::core::TensorImpl region =
          vbt::core::indexing::basic_index(grad_out, spec_);
      const bool shapes_match = (region.sizes() == in_meta_[1].sizes);
      const bool must_clone =
          needs_clone_for_backward(region) || !region.is_contiguous() ||
          (need_self_grad && shapes_match);
      if (must_clone) {
        region = vbt::core::clone_contiguous_same_device(region);
      }
      vbt::core::TensorImpl grad_value =
          sum_to_shape_cpu_float32(region, in_meta_[1].sizes);
      outs[1] = std::move(grad_value);
    }

    if (need_self_grad) {
      vbt::core::TensorImpl grad_self = std::move(grad_out);
      vbt::core::TensorImpl dst = vbt::core::indexing::basic_index(grad_self, spec_);
      fill_zero_float32_cpu(dst);
      outs[0] = std::move(grad_self);
    }

    return outs;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return in_meta_;
  }

 private:
  std::vector<InputMeta> in_meta_;
  vbt::core::indexing::IndexSpec spec_;
  bool is_cuda_{false};
};

class IndexPutBackwardNode final : public Node, public ValidatableNode {
 public:
  IndexPutBackwardNode(std::vector<InputMeta> m,
                       vbt::core::indexing::IndexSpec spec,
                       bool accumulate)
      : in_meta_(std::move(m)), spec_(std::move(spec)), accumulate_(accumulate) {
    name = "IndexPutBackward";
    init_node_streaminfo_from_metas(*this, in_meta_, is_cuda_);
  }

  uint32_t num_inputs() const noexcept override { return 2; }
  uint32_t num_incoming_grad_slots() const noexcept override { return 1; }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;

    std::vector<OptionalTensor> outs(2);
    if (grads_in.empty() || !grads_in[0].has_value()) {
      return outs;
    }

    vbt::core::TensorImpl grad_out = std::move(*grads_in[0]);

    if (in_meta_.size() < 2) {
      throw std::logic_error("IndexPutBackward: missing input meta");
    }

    if (grad_out.device() != in_meta_[0].device) {
      throw std::invalid_argument(
          "IndexPutBackward: grad_out device does not match self");
    }
    if (grad_out.dtype() != in_meta_[0].dtype) {
      throw std::invalid_argument(
          "IndexPutBackward: grad_out dtype does not match self");
    }
    if (grad_out.sizes() != in_meta_[0].sizes) {
      throw std::invalid_argument(
          "IndexPutBackward: grad_out shape does not match self");
    }

    // v2 scope: Float32 only (CPU/CUDA).
    const auto dev_type = grad_out.device().type;
    if (!(dev_type == kDLCPU || dev_type == kDLCUDA) ||
        grad_out.dtype() != vbt::core::ScalarType::Float32) {
      throw std::runtime_error(
          "index_put backward: only Float32 CPU/CUDA is supported");
    }

    if (needs_clone_for_backward(grad_out)) {
      grad_out = vbt::core::clone_contiguous_same_device(grad_out);
    }

    const bool need_self_grad =
        next_edges.size() >= 1 && static_cast<bool>(next_edges[0].fn);
    const bool need_value_grad =
        next_edges.size() >= 2 && static_cast<bool>(next_edges[1].fn);

    if (need_value_grad) {
      vbt::core::TensorImpl region = vbt::core::indexing::index(grad_out, spec_);
      if (needs_clone_for_backward(region) || !region.is_contiguous()) {
        region = vbt::core::clone_contiguous_same_device(region);
      }
      vbt::core::TensorImpl grad_value;
      if (dev_type == kDLCPU) {
        grad_value = sum_to_shape_cpu_float32(region, in_meta_[1].sizes);
      } else if (dev_type == kDLCUDA) {
#if VBT_WITH_CUDA
        grad_value = sum_to_shape_cuda_float32(region, in_meta_[1].sizes);
#else
        throw std::runtime_error("index_put backward: CUDA not built");
#endif
      } else {
        throw std::runtime_error("index_put backward: unsupported device");
      }
      outs[1] = std::move(grad_value);
    }

    if (need_self_grad) {
      vbt::core::TensorImpl grad_self = std::move(grad_out);

      if (!accumulate_) {
        // Overwrite semantics: zero the overwritten region.
        InputMeta zero_meta;
        zero_meta.dtype = vbt::core::ScalarType::Float32;
        zero_meta.device = in_meta_[0].device;
        zero_meta.sizes = {};
        zero_meta.is_strided_dense = true;

        vbt::core::TensorImpl zero = make_zeros_from_meta(zero_meta);
        vbt::core::indexing::index_put_(grad_self, spec_, zero, /*accumulate=*/false);
      }

      outs[0] = std::move(grad_self);
    }

    return outs;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return in_meta_;
  }

 private:
  std::vector<InputMeta> in_meta_;
  vbt::core::indexing::IndexSpec spec_;
  bool accumulate_{false};
  bool is_cuda_{false};
};

static std::vector<InputMeta> metas_from_snaps(const std::vector<SavedVariable>& snaps, std::size_t n) {
  std::vector<InputMeta> m; m.reserve(n);
  for (std::size_t i = 0; i < n && i < snaps.size(); ++i) {
    const vbt::core::TensorImpl ti = snaps[i].unpack();
    m.emplace_back(InputMeta::from_tensor(ti));
  }
  return m;
}

inline bool is_forward_ad_allowlisted(const std::string& fqname) {
  // Forward-mode allowlist is a subset of the boxed autograd set.
  return fqname == "vt::relu" || fqname == "vt::add" || fqname == "vt::mul";
}

static vbt::core::TensorImpl compute_add_tangent(
    const vbt::core::TensorImpl& a,
    const vbt::core::TensorImpl& b,
    const vbt::core::TensorImpl* ta,
    const vbt::core::TensorImpl* tb) {
  using vbt::core::TensorImpl;
  if (!ta && !tb) {
    return TensorImpl();
  }

  BoxedStack stack;
  NoGradGuard ng;
  SkipAutogradGuard sa;
  auto& D = Dispatcher::instance();

  if (ta && tb) {
    stack.clear();
    stack.push_back(*ta);
    stack.push_back(*tb);
    D.callBoxed("vt::add", stack);
    return stack[0];
  }

  if (ta) {
    TensorImpl zero_b = make_zeros_like(b);
    stack.clear();
    stack.push_back(*ta);
    stack.push_back(zero_b);
    D.callBoxed("vt::add", stack);
    return stack[0];
  }

  // Only tb present.
  TensorImpl zero_a = make_zeros_like(a);
  stack.clear();
  stack.push_back(zero_a);
  stack.push_back(*tb);
  D.callBoxed("vt::add", stack);
  return stack[0];
}

static vbt::core::TensorImpl compute_mul_tangent(
    const vbt::core::TensorImpl& a,
    const vbt::core::TensorImpl& b,
    const vbt::core::TensorImpl* ta,
    const vbt::core::TensorImpl* tb) {
  using vbt::core::TensorImpl;

  if (!ta && !tb) {
    return TensorImpl();
  }

  // Domain: CPU float32 with matching sizes. This mirrors the core
  if (a.device().type != kDLCPU || b.device().type != kDLCPU) {
    throw std::runtime_error(
        "forward-mode mul: expected CPU tensors");
  }
  if (a.sizes() != b.sizes()) {
    throw std::runtime_error(
        "forward-mode mul: expected matching input sizes");
  }

  const auto dtype = a.dtype();
  if (dtype != vbt::core::ScalarType::Float32 ||
      b.dtype() != dtype ||
      (ta && ta->dtype() != dtype) ||
      (tb && tb->dtype() != dtype)) {
    throw std::runtime_error(
        "forward-mode mul: expected Float32 tensors");
  }

  if (!a.is_contiguous() || !b.is_contiguous() ||
      (ta && !ta->is_contiguous()) ||
      (tb && !tb->is_contiguous())) {
    throw std::runtime_error(
        "forward-mode mul: only contiguous tensors are supported");
  }

  TensorImpl out = make_zeros_like(a);

  float* pout = static_cast<float*>(out.data());
  const float* pa = static_cast<const float*>(a.data());
  const float* pb = static_cast<const float*>(b.data());
  const float* pta = ta ? static_cast<const float*>(ta->data()) : nullptr;
  const float* ptb = tb ? static_cast<const float*>(tb->data()) : nullptr;

  const auto N = a.numel();
  for (int64_t i = 0; i < N; ++i) {
    const float ta_v = pta ? pta[i] : 0.0f;
    const float tb_v = ptb ? ptb[i] : 0.0f;
    pout[i] = ta_v * pb[i] + pa[i] * tb_v;
  }

  return out;
}

static vbt::core::TensorImpl compute_relu_tangent(
    const vbt::core::TensorImpl& x,
    const vbt::core::TensorImpl* tx) {
  using vbt::core::TensorImpl;
  if (!tx) {
    return TensorImpl();
  }

  if (x.device().type != kDLCPU || tx->device().type != kDLCPU ||
      x.dtype() != vbt::core::ScalarType::Float32 ||
      tx->dtype() != vbt::core::ScalarType::Float32 ||
      x.sizes() != tx->sizes()) {
    throw std::runtime_error(
        "forward-mode relu: expected CPU float32 tensors with matching sizes");
  }

  TensorImpl out = make_zeros_like(x);

  vbt::core::TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(x);
  cfg.add_input(*tx);
  cfg.check_mem_overlap(true);
  static const vbt::core::IterOpSignature kReluTangentSig{
      "vt::relu_tangent", nullptr, 0};
  cfg.set_op_signature(&kReluTangentSig);
  vbt::core::TensorIter iter = cfg.build();

  auto loop = [](char** data,
                 const std::int64_t* strides,
                 std::int64_t size,
                 void* /*ctx*/) {
    char* out_base = data[0];
    char* x_base   = data[1];
    char* t_base   = data[2];
    const std::int64_t out_stride = strides[0];
    const std::int64_t x_stride   = strides[1];
    const std::int64_t t_stride   = strides[2];
    for (std::int64_t i = 0; i < size; ++i) {
      auto* pout = reinterpret_cast<float*>(out_base + i * out_stride);
      const auto* px =
          reinterpret_cast<const float*>(x_base + i * x_stride);
      const auto* pt =
          reinterpret_cast<const float*>(t_base + i * t_stride);
      const float xv = *px;
      *pout = xv > 0.0f ? *pt : 0.0f;
    }
  };

  vbt::core::for_each_cpu(iter, loop, nullptr);
  return out;
}

} // anonymous namespace

bool autograd_indexing_v2_enabled() noexcept {
  if (!g_autograd_indexing_v2_override_active.load(std::memory_order_relaxed)) {
    std::call_once(g_autograd_indexing_v2_once, init_autograd_indexing_v2_from_env);
  }
  return g_autograd_indexing_v2_enabled.load(std::memory_order_relaxed);
}

void set_autograd_indexing_v2_enabled_for_tests(bool enabled) noexcept {
  g_autograd_indexing_v2_enabled.store(enabled, std::memory_order_relaxed);
  g_autograd_indexing_v2_override_active.store(true, std::memory_order_relaxed);
}

bool autograd_indexing_v2_negstride_enabled() noexcept {
  if (!g_autograd_indexing_v2_negstride_override_active.load(std::memory_order_relaxed)) {
    std::call_once(g_autograd_indexing_v2_negstride_once, init_autograd_indexing_v2_negstride_from_env);
  }
  return g_autograd_indexing_v2_negstride_enabled.load(std::memory_order_relaxed);
}

void set_autograd_indexing_v2_negstride_enabled_for_tests(bool enabled) noexcept {
  g_autograd_indexing_v2_negstride_enabled.store(enabled, std::memory_order_relaxed);
  g_autograd_indexing_v2_negstride_override_active.store(true, std::memory_order_relaxed);
}

vbt::core::intrusive_ptr<Node> make_basic_index_view_backward_node(
    const vbt::core::TensorImpl& base,
    vbt::core::indexing::IndexSpec spec) {
  std::vector<InputMeta> m;
  m.reserve(1);
  m.emplace_back(InputMeta::from_tensor(base));
  return vbt::core::intrusive_ptr<Node>(
      new BasicIndexViewBackwardNode(std::move(m), std::move(spec)),
      /*add_ref=*/true);
}

vbt::core::intrusive_ptr<Node> make_basic_index_put_backward_node(
    const vbt::core::TensorImpl& self,
    const vbt::core::TensorImpl& value,
    vbt::core::indexing::IndexSpec spec) {
  std::vector<InputMeta> m;
  m.reserve(2);
  m.emplace_back(InputMeta::from_tensor(self));
  m.emplace_back(InputMeta::from_tensor(value));
  return vbt::core::intrusive_ptr<Node>(
      new BasicIndexPutBackwardNode(std::move(m), std::move(spec)),
      /*add_ref=*/true);
}

vbt::core::intrusive_ptr<Node> make_index_put_backward_node(
    const vbt::core::TensorImpl& self,
    const vbt::core::TensorImpl& value,
    vbt::core::indexing::IndexSpec spec,
    bool accumulate) {
  std::vector<InputMeta> m;
  m.reserve(2);
  m.emplace_back(InputMeta::from_tensor(self));
  m.emplace_back(InputMeta::from_tensor(value));
  return vbt::core::intrusive_ptr<Node>(
      new IndexPutBackwardNode(std::move(m), std::move(spec), accumulate),
      /*add_ref=*/true);
}

bool any_requires_grad(const BoxedStack& s, uint8_t in_arity) noexcept {
  if (in_arity == 0) return false;
  const std::size_t n = std::min<std::size_t>(in_arity, s.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (vbt::autograd::requires_grad(s[i])) return true;
  }
  return false;
}

static vbt::core::intrusive_ptr<Node> build_backward_node(const std::string& fqname,
                                                          const std::vector<SavedVariable>& snaps) {
  if (fqname == "vt::add") {
    auto m = metas_from_snaps(snaps, 2);
    return vbt::core::intrusive_ptr<Node>(new AddBackwardNode(std::move(m)), /*add_ref=*/true);
  } else if (fqname == "vt::mul") {
    if (snaps.size() >= 2) {
      auto m = metas_from_snaps(snaps, 2);
      return vbt::core::intrusive_ptr<Node>(new MulBackwardNode(snaps[0], snaps[1], std::move(m)), /*add_ref=*/true);
    }
    auto m = metas_from_snaps(snaps, 2);
    return vbt::core::intrusive_ptr<Node>(new MulBackwardNode(SavedVariable{}, SavedVariable{}, std::move(m)), /*add_ref=*/true);
  } else if (fqname == "vt::relu") {
    auto m = metas_from_snaps(snaps, 1);
    if (!snaps.empty()) return vbt::core::intrusive_ptr<Node>(new ReluBackwardNode(snaps[0], std::move(m)), /*add_ref=*/true);
    return vbt::core::intrusive_ptr<Node>(new ReluBackwardNode(SavedVariable{}, std::move(m)), /*add_ref=*/true);
  } else if (fqname == "vt::index") {
    std::vector<InputMeta> m;
    m.reserve(1);
    if (!snaps.empty()) {
      const vbt::core::TensorImpl ti = snaps[0].unpack();
      m.emplace_back(InputMeta::from_tensor(ti));
    } else {
      m.emplace_back(InputMeta{});
    }
    SavedVariable self_sv  = snaps.size() > 0 ? snaps[0] : SavedVariable{};
    SavedVariable index_sv = snaps.size() > 1 ? snaps[1] : SavedVariable{};
    SavedVariable meta_sv  = snaps.size() > 2 ? snaps[2] : SavedVariable{};

    if (autograd_indexing_v2_enabled()) {
      if (!meta_sv.is_initialized()) {
        throw std::logic_error("IndexBackwardV2: missing meta snapshot");
      }
      return vbt::core::intrusive_ptr<Node>(
          new IndexBackwardNodeV2(std::move(index_sv), std::move(meta_sv), std::move(m)),
          /*add_ref=*/true);
    }

    return vbt::core::intrusive_ptr<Node>(
        new IndexBackwardNode(std::move(self_sv), std::move(index_sv), std::move(m)),
        /*add_ref=*/true);
  } else if (fqname == "vt::embedding") {
    std::vector<InputMeta> m;
    m.reserve(1);
    if (!snaps.empty()) {
      const vbt::core::TensorImpl ti = snaps[0].unpack();
      m.emplace_back(InputMeta::from_tensor(ti));
    } else {
      m.emplace_back(InputMeta{});
    }

    SavedVariable weight_sv  = snaps.size() > 0 ? snaps[0] : SavedVariable{};
    SavedVariable indices_sv = snaps.size() > 1 ? snaps[1] : SavedVariable{};

    int64_t padding_idx = -1;
    bool scale_grad_by_freq = false;
    bool sparse = false;
    if (snaps.size() > 2) {
      padding_idx = read_cpu_scalar_int64_0d(snaps[2].unpack(), "vt::embedding",
                                            "padding_idx");
    }
    if (snaps.size() > 3) {
      scale_grad_by_freq = read_cpu_scalar_bool_0d(snaps[3].unpack(), "vt::embedding",
                                                   "scale_grad_by_freq");
    }
    if (snaps.size() > 4) {
      sparse = read_cpu_scalar_bool_0d(snaps[4].unpack(), "vt::embedding", "sparse");
    }

    return vbt::core::intrusive_ptr<Node>(
        new EmbeddingBackwardNode(std::move(weight_sv), std::move(indices_sv),
                                  padding_idx, scale_grad_by_freq, sparse,
                                  std::move(m)),
        /*add_ref=*/true);
  } else {
    // default single-input identity for unsupported in-place ops (e.g., fill_)
    auto m = metas_from_snaps(snaps, 1);
    return vbt::core::intrusive_ptr<Node>(new IdentityBackwardNode(std::move(m)), /*add_ref=*/true);
  }
}

void autograd_fallback_ctx(void* ctx, BoxedStack& s) {
  // Count Stage 1 entry
  _stats_wrapper_invoked();
  // ctx is OperatorEntry*
  auto* entry = static_cast<vbt::dispatch::OperatorEntry*>(ctx);
  if (!entry) throw std::runtime_error("invalid autograd fallback ctx");
  const std::string& name = entry->schema.fqname;
  const uint8_t in_arity = entry->schema.in_arity;

  const bool supported = is_supported_single_output(name);
  const uint8_t grad_arity =
      autograd_requires_grad_in_arity_masked(name, in_arity);
  const bool do_autograd = supported &&
                           GradMode::is_enabled() &&
                           !InferenceMode::is_enabled() &&
                           any_requires_grad(s, grad_arity);

  // Op-specific preflight checks (only when we will actually produce grads).
  if (do_autograd && name == "vt::embedding") {
    const bool sparse = s.size() > 4 ? read_cpu_scalar_bool_0d(s[4], "vt::embedding", "sparse")
                                     : false;
    if (sparse) {
      throw std::runtime_error(
          "vt::embedding: sparse gradients are not supported");
    }
  }

  const std::int64_t level = current_forward_ad_level();
  const bool forward_enabled = is_forward_ad_enabled() && level >= 0;

  // Snapshot input tensors by value for forward-mode tangent propagation.
  std::vector<vbt::core::TensorImpl> inputs;
  inputs.reserve(in_arity);
  const std::size_t n_inputs =
      std::min<std::size_t>(in_arity, s.size());
  for (std::size_t i = 0; i < n_inputs; ++i) {
    inputs.push_back(s[i]);
  }

  bool has_tangent = false;
  std::vector<const vbt::core::TensorImpl*> input_tangents(inputs.size(), nullptr);
  if (forward_enabled) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const vbt::core::TensorImpl* t =
          get_forward_grad_view(inputs[i], level);
      input_tangents[i] = t;
      if (t) has_tangent = true;
    }
  }

  const bool is_fwd_allowlisted = is_forward_ad_allowlisted(name);

  // For boxed but non-allowlisted ops that see any tangent at the active
  // level, fail fast before wiring reverse-mode.
  if (forward_enabled && has_tangent && !is_fwd_allowlisted) {
    throw std::runtime_error(
        "forward-mode is not implemented for op " + name);
  }

  // Prepare reverse-mode snapshots when needed.
  std::vector<SavedVariable> snaps;
  if (do_autograd) {
    snaps.reserve(in_arity);
    const std::size_t n = std::min<std::size_t>(in_arity, s.size());
    for (std::size_t i = 0; i < n; ++i) snaps.emplace_back(SavedVariable(s[i]));
  }

  // Redispatch to base under SkipAutogradGuard
  OperatorHandle op(entry);
  SkipAutogradGuard sa;
  Dispatcher::instance().redispatchBoxed(op, s);

  // Forward-mode tangent propagation for allowlisted ops.
  if (forward_enabled && has_tangent && is_fwd_allowlisted) {
    if (s.size() != 1) {
      throw std::runtime_error(
          "autograd wrapper forward-mode only supports single-output ops: " + name);
    }
    vbt::core::TensorImpl& out = s[0];

    if (name == "vt::add" && inputs.size() >= 2) {
      const vbt::core::TensorImpl* ta =
          input_tangents.size() > 0 ? input_tangents[0] : nullptr;
      const vbt::core::TensorImpl* tb =
          input_tangents.size() > 1 ? input_tangents[1] : nullptr;
      vbt::core::TensorImpl t_out =
          compute_add_tangent(inputs[0], inputs[1], ta, tb);
      set_forward_grad(out, t_out, level);
    } else if (name == "vt::mul" && inputs.size() >= 2) {
      const vbt::core::TensorImpl* ta =
          input_tangents.size() > 0 ? input_tangents[0] : nullptr;
      const vbt::core::TensorImpl* tb =
          input_tangents.size() > 1 ? input_tangents[1] : nullptr;
      vbt::core::TensorImpl t_out =
          compute_mul_tangent(inputs[0], inputs[1], ta, tb);
      set_forward_grad(out, t_out, level);
    } else if (name == "vt::relu" && inputs.size() >= 1) {
      const vbt::core::TensorImpl* tx =
          input_tangents.size() > 0 ? input_tangents[0] : nullptr;
      vbt::core::TensorImpl t_out =
          compute_relu_tangent(inputs[0], tx);
      set_forward_grad(out, t_out, level);
    }
  }

  // Reverse-mode autograd behaves as before.
  if (!do_autograd) return;

  if (s.size() != 1) {
    throw std::runtime_error("autograd wrapper only supports single-output ops: " + name);
  }
  // Build Node and attach to output
  auto node = build_backward_node(name, snaps);
  node->next_edges.resize(node->num_inputs());
  // Wire next edges for differentiable inputs using snaps-only metadata (no live reads)
  {
    std::unordered_map<const AutogradMeta*, vbt::core::intrusive_ptr<Node>> sinks;
    std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(node->num_inputs()), static_cast<std::size_t>(in_arity));
    n = std::min<std::size_t>(n, snaps.size());
    for (std::size_t i = 0; i < n; ++i) {
      const vbt::core::TensorImpl ti = snaps[i].unpack();
      node->next_edges[i] = resolve_edge_for_tensor(ti, sinks);
    }
  }

  // Attach autograd metadata to the output
  vbt::core::TensorImpl& out = s[0];
  set_requires_grad(out, true);
  if (auto* m = get_autograd_meta(out, /*create_if_missing=*/true)) {
    m->is_leaf = false;
    m->output_nr = 0;
    m->grad_fn = node;
  }
}

RegisterResults register_autograd_fallbacks() {
  RegisterResults r{};
  auto& D = Dispatcher::instance();
  auto try_reg = [&](const char* fq){
    if (!D.has(fq)) return false;
    auto h = D.find(fq);
    const uint8_t arity = h.get().schema.in_arity;
    auto kf = vbt::dispatch::KernelFunction::makeBoxedCtx(arity, &autograd_fallback_ctx, nullptr);
    try {
      return D.tryRegisterAutogradFallback(fq, kf);
    } catch (...) {
      return false;
    }
  };
  r.relu = try_reg("vt::relu");
  r.add  = try_reg("vt::add");
  r.mul  = try_reg("vt::mul");
  r.index = try_reg("vt::index");
  r.embedding = try_reg("vt::embedding");
  return r;
}

vbt::core::intrusive_ptr<Node> build_inplace_backward_node(const char* op_fqname,
                                                           const std::vector<SavedVariable>& snaps) {
  std::string name(op_fqname ? op_fqname : "");
  auto n = build_backward_node(name, snaps);
  n->next_edges.resize(n->num_inputs());
  // Wire next edges for differentiable inputs using provided snaps only
  {
    std::unordered_map<const AutogradMeta*, vbt::core::intrusive_ptr<Node>> sinks;
    std::size_t m = std::min<std::size_t>(static_cast<std::size_t>(n->num_inputs()), snaps.size());
    for (std::size_t i = 0; i < m; ++i) {
      const vbt::core::TensorImpl ti = snaps[i].unpack();
      n->next_edges[i] = resolve_edge_for_tensor(ti, sinks);
    }
  }
  return n;
}

}} // namespace vbt::autograd
