// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/dispatcher.h"

namespace {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::DeviceConstraint;
using vbt::dispatch::DevicePolicy;
using vbt::dispatch::ConstraintKind;

static std::int64_t read_cpu_scalar_int64_0d(const TensorImpl& t,
                                             const char* op,
                                             std::size_t arg_index) {
  if (t.device().type != kDLCPU) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] must be a CPU int64 scalar (0-d)");
  }
  if (t.dtype() != ScalarType::Int64) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] must be dtype=int64");
  }
  if (!t.sizes().empty()) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] must be 0-d (scalar)");
  }
  if (!t.data()) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] has no data");
  }
  return *reinterpret_cast<const long long*>(t.data());
}

static bool read_cpu_scalar_bool_0d(const TensorImpl& t,
                                   const char* op,
                                   std::size_t arg_index) {
  if (t.device().type != kDLCPU) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] must be a CPU bool scalar (0-d)");
  }
  if (t.dtype() != ScalarType::Bool) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] must be dtype=bool");
  }
  if (!t.sizes().empty()) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] must be 0-d (scalar)");
  }
  if (!t.data()) {
    throw std::invalid_argument(
        std::string(op) + ": arg[" + std::to_string(arg_index) +
        "] has no data");
  }
  return (*reinterpret_cast<const bool*>(t.data())) != 0;
}

#if VBT_WITH_CUDA
extern "C" TensorImpl vbt_cuda_sum_impl(const TensorImpl& self,
                                       std::vector<int64_t> dims,
                                       bool keepdim);
extern "C" TensorImpl vbt_cuda_max_impl(const TensorImpl& self,
                                       std::vector<int64_t> dims,
                                       bool keepdim);
#endif

static TensorImpl vt_sum_dim_impl(const TensorImpl& self,
                                 const TensorImpl& dim,
                                 const TensorImpl& keepdim) {
  constexpr const char* kOp = "vt::sum_dim";
  const std::int64_t d = read_cpu_scalar_int64_0d(dim, kOp, 1);
  const bool k = read_cpu_scalar_bool_0d(keepdim, kOp, 2);

#if VBT_WITH_CUDA
  if (self.device().type == kDLCUDA) {
    return vbt_cuda_sum_impl(self, std::vector<int64_t>{d}, k);
  }
#endif

  throw std::runtime_error(std::string(kOp) + ": only CUDA tensors supported in this build");
}

static TensorImpl vt_max_dim_impl(const TensorImpl& self,
                                 const TensorImpl& dim,
                                 const TensorImpl& keepdim) {
  constexpr const char* kOp = "vt::max_dim";
  const std::int64_t d = read_cpu_scalar_int64_0d(dim, kOp, 1);
  const bool k = read_cpu_scalar_bool_0d(keepdim, kOp, 2);

#if VBT_WITH_CUDA
  if (self.device().type == kDLCUDA) {
    return vbt_cuda_max_impl(self, std::vector<int64_t>{d}, k);
  }
#endif

  throw std::runtime_error(std::string(kOp) + ": only CUDA tensors supported in this build");
}

}  // namespace

extern "C" void vbt_register_reduction_dim_kernels() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto& D = Dispatcher::instance();

    if (!D.has("vt::sum_dim")) {
      D.def("vt::sum_dim(Tensor, Tensor, Tensor) -> Tensor");
      DeviceConstraint constraints[] = {
          DeviceConstraint{1, ConstraintKind::MustBeCPUScalarInt64_0d},
          DeviceConstraint{2, ConstraintKind::MustBeCPUScalarBool_0d},
      };
      D.set_device_policy("vt::sum_dim",
                          DevicePolicy::MaskedSameDevice,
                          /*dispatch_arg_mask=*/0b001,
                          constraints,
                          /*allow_undefined_mask=*/0);
#if VBT_WITH_CUDA
      D.registerCudaKernel("vt::sum_dim", &vt_sum_dim_impl);
#endif
    }

    if (!D.has("vt::max_dim")) {
      D.def("vt::max_dim(Tensor, Tensor, Tensor) -> Tensor");
      DeviceConstraint constraints[] = {
          DeviceConstraint{1, ConstraintKind::MustBeCPUScalarInt64_0d},
          DeviceConstraint{2, ConstraintKind::MustBeCPUScalarBool_0d},
      };
      D.set_device_policy("vt::max_dim",
                          DevicePolicy::MaskedSameDevice,
                          /*dispatch_arg_mask=*/0b001,
                          constraints,
                          /*allow_undefined_mask=*/0);
#if VBT_WITH_CUDA
      D.registerCudaKernel("vt::max_dim", &vt_max_dim_impl);
#endif
    }
  });
}

