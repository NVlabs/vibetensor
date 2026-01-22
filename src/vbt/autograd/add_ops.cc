// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/add_ops.h"
#include "vbt/autograd/dtype_policy.h"

#include "vbt/core/complex.h"
#include "vbt/core/dtype.h"

#if VBT_WITH_CUDA
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#endif

#include <stdexcept>

#if VBT_WITH_CUDA
extern "C" vbt::core::TensorImpl
vbt_cuda_add_impl(const vbt::core::TensorImpl& a,
                  const vbt::core::TensorImpl& b);
#endif

namespace vbt { namespace autograd {

using vbt::core::TensorImpl;

void autograd_add_inplace_dense(TensorImpl& acc,
                                const TensorImpl& addend,
                                const vbt::core::Device& autograd_device) {
  if (acc.device().type != addend.device().type ||
      acc.device().index != addend.device().index) {
    throw std::logic_error("autograd_add_inplace_dense: device mismatch");
  }
  if (acc.dtype() != addend.dtype()) {
    throw std::logic_error("autograd_add_inplace_dense: dtype mismatch");
  }
  if (acc.sizes() != addend.sizes() ||
      acc.strides() != addend.strides() ||
      !acc.is_non_overlapping_and_dense() ||
      !addend.is_non_overlapping_and_dense()) {
    throw std::logic_error(
        "autograd_add_inplace_dense: metadata mismatch or non-dense");
  }

  const auto N = acc.numel();
  if (N == 0) return;

  if (acc.device().type == kDLCPU) {
    switch (acc.dtype()) {
      case vbt::core::ScalarType::Float32: {
        float* a = static_cast<float*>(acc.data());
        const float* b = static_cast<const float*>(addend.data());
        for (int64_t i = 0; i < N; ++i) a[i] += b[i];
        return;
      }
      case vbt::core::ScalarType::Float64: {
        double* a = static_cast<double*>(acc.data());
        const double* b = static_cast<const double*>(addend.data());
        for (int64_t i = 0; i < N; ++i) a[i] += b[i];
        return;
      }
      case vbt::core::ScalarType::Complex64: {
        auto* a = static_cast<vbt::core::Complex<float>*>(acc.data());
        const auto* b =
            static_cast<const vbt::core::Complex<float>*>(addend.data());
        for (int64_t i = 0; i < N; ++i) {
          a[i].re += b[i].re;
          a[i].im += b[i].im;
        }
        return;
      }
      case vbt::core::ScalarType::Complex128: {
        auto* a = static_cast<vbt::core::Complex<double>*>(acc.data());
        const auto* b =
            static_cast<const vbt::core::Complex<double>*>(addend.data());
        for (int64_t i = 0; i < N; ++i) {
          a[i].re += b[i].re;
          a[i].im += b[i].im;
        }
        return;
      }
      default:
        throw std::runtime_error(
            "autograd_add_inplace_dense: unsupported CPU dtype for accumulation");
    }
  }

#if VBT_WITH_CUDA
  if (acc.device().type == kDLCUDA) {
    if (acc.device().type != autograd_device.type ||
        acc.device().index != autograd_device.index) {
      throw std::logic_error(
          "autograd_add_inplace_dense: gradient device does not match autograd_device");
    }
    if (!is_cuda_autograd_dtype_supported(acc.dtype())) {
      throw std::runtime_error(
          "autograd_add_inplace_dense: unsupported CUDA dtype for accumulation");
    }

    const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(acc.device().index);
    vbt::cuda::Stream s = vbt::cuda::getCurrentStream(dev_idx);
    vbt::cuda::record_stream(acc.storage(), s);
    vbt::cuda::record_stream(addend.storage(), s);

    // Reuse existing CUDA vt::add implementation; copy result back into acc.
    TensorImpl out = vbt_cuda_add_impl(acc, addend);
    acc.set_storage(out.storage());
    acc.set_sizes_and_strides(out.sizes(), out.strides());
    acc.set_storage_offset(out.storage_offset());
    return;
  }
#else
  (void)autograd_device;
#endif

  throw std::runtime_error(
      "autograd_add_inplace_dense: unsupported device for accumulation");
}

}} // namespace vbt::autograd
