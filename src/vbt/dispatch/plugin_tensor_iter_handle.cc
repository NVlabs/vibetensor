// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/plugin/vbt_plugin.h"
#include "vbt/dispatch/plugin_loader.h"
#include "vbt/core/tensor_iterator/core.h"
#include "vbt/core/tensor_iterator/cpu.h"
#include "vbt/core/tensor_iterator/cuda.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

static_assert(::vbt::core::kTensorIterMaxRank == VT_TENSOR_ITER_MAX_RANK,
              "TI: C++ and C MAX_RANK must match");
static_assert(::vbt::core::kTensorIterCudaMaxNdim == VT_TENSOR_ITER_CUDA_MAX_NDIM,
              "TI CUDA: C++ and C CUDA max-ndim must match");

// Global handle struct backing the opaque C type vt_tensor_iter.
struct vt_tensor_iter__ {
  vbt::core::TensorIter iter;   // owns TI state
  DLDeviceType          device_type;
  std::int32_t          ntensors;
  std::int32_t          num_outputs;
  vt_tensor_iter_kind   kind;
};

namespace vbt {
namespace dispatch {
namespace plugin {
namespace detail {

// Shared exception-to-status shim for vt_tensor_iter_* APIs.
// Uses the existing plugin TLS error channel via set_last_error_helper.

template <class F>
vt_status with_tensor_iter_api_errors(const char* api_name, F&& fn) noexcept {
  try {
    set_last_error_helper("");  // clear on entry

    vt_status st = fn();  // fn may return non-OK without throwing

    if (st == VT_STATUS_OK) {
      // Ensure TLS is empty on success.
      set_last_error_helper("");
    }
    return st;

  } catch (const std::invalid_argument& e) {
    set_last_error_helper(e.what());
    return VT_STATUS_INVALID_ARG;

  } catch (const std::logic_error& e) {
    set_last_error_helper(e.what());
    return VT_STATUS_INVALID_ARG;

  } catch (const std::exception& e) {
    set_last_error_helper(
        (e.what() && e.what()[0]) ? e.what()
                                  : "std::exception in vt_tensor_iter API");
    return VT_STATUS_RUNTIME_ERROR;

  } catch (...) {
    set_last_error_helper(
        (api_name && api_name[0]) ? api_name
                                  : "vt_tensor_iter API: unknown non-standard exception");
    return VT_STATUS_RUNTIME_ERROR;
  }
}

}  // namespace detail
}  // namespace plugin
}  // namespace dispatch
}  // namespace vbt

namespace vbt {
namespace dispatch {
namespace plugin_helpers {

using ::vbt::core::TensorImpl;
using ::vbt::core::TensorIter;
using ::vbt::core::TensorIterConfig;
using ::vbt::core::OptionalTensorImplRef;
using ::vbt::core::ScalarType;
using ::vbt::core::Device;
using ::vbt::core::itemsize;
using ::vbt::core::to_dlpack_dtype;
using ::vbt::dispatch::plugin::detail::require_tensor_impl;
using ::vbt::dispatch::plugin::detail::set_last_error_helper;
using ::vbt::dispatch::plugin::detail::with_tensor_iter_api_errors;

namespace {

inline vt_iter_config make_default_cfg() {
  vt_iter_config cfg;
  cfg.max_rank = 0;  // use TI default (kTensorIterMaxRank)
  cfg.check_mem_overlap = VT_ITER_OVERLAP_ENABLE;
  return cfg;
}

inline bool is_supported_dtype(ScalarType dt) {
  switch (dt) {
    case ScalarType::Bool:
    case ScalarType::Int32:
    case ScalarType::Int64:
    case ScalarType::Float16:
    case ScalarType::BFloat16:
    case ScalarType::Float32:
      return true;
  }
  return false;
}

inline bool is_supported_device(const Device& dev) {
  return dev.type == kDLCPU || dev.type == kDLCUDA;
}

}  // anonymous namespace

// ---- Builders -------------------------------------------------------------------------

vt_status vt_tensor_iter_build_elementwise_host(const vt_iter_config* cfg,
                                                int32_t ntensors,
                                                const vt_tensor* tensors,
                                                vt_tensor_iter* out_iter) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_build_elementwise", [&]() -> vt_status {
    if (!out_iter) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: out_iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    *out_iter = nullptr;

    if (ntensors < 2) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: ntensors must be >= 2 (one output + inputs)");
      return VT_STATUS_INVALID_ARG;
    }

    if (!tensors) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: tensors array must not be null");
      return VT_STATUS_INVALID_ARG;
    }

    if (ntensors > VT_TENSOR_ITER_MAX_OPERANDS) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: ntensors exceeds VT_TENSOR_ITER_MAX_OPERANDS");
      return VT_STATUS_UNSUPPORTED;
    }

    vt_iter_config effective = cfg ? *cfg : make_default_cfg();

    const std::int64_t max_rank_limit =
        static_cast<std::int64_t>(vbt::core::kTensorIterMaxRank);
    if (effective.max_rank < 0 || effective.max_rank > max_rank_limit) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: max_rank out of range");
      return VT_STATUS_INVALID_ARG;
    }

    // Translate vt_tensor handles to TensorImpl& and enforce device/dtype.
    std::vector<TensorImpl*> impls;
    impls.reserve(static_cast<std::size_t>(ntensors));
    for (int32_t i = 0; i < ntensors; ++i) {
      impls.push_back(&require_tensor_impl(tensors[i], "tensor"));
    }

    TensorImpl& out_impl = *impls[0];
    const Device dev0 = out_impl.device();
    const ScalarType dt0 = out_impl.dtype();

    if (!is_supported_device(dev0)) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: unsupported device type");
      return VT_STATUS_UNSUPPORTED;
    }

    if (!is_supported_dtype(dt0)) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: unsupported dtype for TI C ABI");
      return VT_STATUS_UNSUPPORTED;
    }

    for (int32_t i = 1; i < ntensors; ++i) {
      TensorImpl& impl = *impls[static_cast<std::size_t>(i)];
      if (impl.device() != dev0) {
        set_last_error_helper(
            "vt_tensor_iter_build_elementwise: all operands must be on the same device");
        return VT_STATUS_INVALID_ARG;
      }
      if (impl.dtype() != dt0) {
        set_last_error_helper(
            "vt_tensor_iter_build_elementwise: all operands must share the same dtype");
        return VT_STATUS_INVALID_ARG;
      }
    }

    TensorIterConfig cfg_cpp;
    cfg_cpp.add_output(OptionalTensorImplRef(&out_impl, /*defined=*/true));
    for (int32_t i = 1; i < ntensors; ++i) {
      cfg_cpp.add_const_input(*impls[static_cast<std::size_t>(i)]);
    }

    cfg_cpp.check_all_same_dtype(true);
    cfg_cpp.check_all_same_device(true);
    cfg_cpp.promote_inputs_to_common_dtype(false);
    cfg_cpp.promote_integer_inputs_to_float(false);
    cfg_cpp.cast_common_dtype_to_outputs(false);
    cfg_cpp.allow_cpu_scalars(false);
    cfg_cpp.is_reduction(false);
    cfg_cpp.resize_outputs(false);

    const bool want_full =
        (effective.check_mem_overlap == VT_ITER_OVERLAP_ENABLE);
    cfg_cpp.check_mem_overlap(want_full);

    if (effective.max_rank != 0) {
      cfg_cpp.set_max_rank(static_cast<std::int64_t>(effective.max_rank));
    }

    TensorIter iter = cfg_cpp.build();  // may throw

    // Allocate handle and move iterator into it.
    vt_tensor_iter handle = nullptr;
    try {
      handle = new vt_tensor_iter__{};
    } catch (const std::bad_alloc&) {
      set_last_error_helper(
          "vt_tensor_iter_build_elementwise: allocation failure for handle");
      return VT_STATUS_NOMEM;
    }

    handle->iter        = std::move(iter);
    handle->device_type = dev0.type;
    handle->ntensors    = handle->iter.ntensors();
    handle->num_outputs = handle->iter.noutputs();
    handle->kind        = VT_TENSOR_ITER_KIND_ELEMENTWISE;

    *out_iter = handle;
    return VT_STATUS_OK;
  });
}

vt_status vt_tensor_iter_build_reduction_host(const vt_iter_config* cfg,
                                              int32_t ntensors,
                                              const vt_tensor* tensors,
                                              int32_t reduce_dim,
                                              vt_tensor_iter* out_iter) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_build_reduction", [&]() -> vt_status {
    if (!out_iter) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: out_iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    *out_iter = nullptr;

    if (ntensors != 2) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: ntensors must be exactly 2 (out, in)");
      return VT_STATUS_INVALID_ARG;
    }

    if (!tensors) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: tensors array must not be null");
      return VT_STATUS_INVALID_ARG;
    }

    vt_iter_config effective = cfg ? *cfg : make_default_cfg();

    const std::int64_t max_rank_limit =
        static_cast<std::int64_t>(vbt::core::kTensorIterMaxRank);
    if (effective.max_rank < 0 || effective.max_rank > max_rank_limit) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: max_rank out of range");
      return VT_STATUS_INVALID_ARG;
    }

    TensorImpl& out_impl = require_tensor_impl(tensors[0], "out");
    TensorImpl& in_impl  = require_tensor_impl(tensors[1], "in");

    const Device dev_in = in_impl.device();
    const ScalarType dt_in = in_impl.dtype();

    if (!is_supported_device(dev_in)) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: unsupported device type");
      return VT_STATUS_UNSUPPORTED;
    }

    if (dev_in.type != kDLCPU) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: reductions currently support CPU tensors only");
      return VT_STATUS_INVALID_ARG;
    }

    if (!is_supported_dtype(dt_in)) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: unsupported dtype for TI C ABI");
      return VT_STATUS_UNSUPPORTED;
    }

    if (out_impl.device() != dev_in) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: output and input must be on the same device");
      return VT_STATUS_INVALID_ARG;
    }
    if (out_impl.dtype() != dt_in) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: output and input must share the same dtype");
      return VT_STATUS_INVALID_ARG;
    }

    const auto& in_sizes = in_impl.sizes();
    const std::int64_t R = static_cast<std::int64_t>(in_sizes.size());
    if (R <= 0) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: cannot build reduction for rank-0 tensors");
      return VT_STATUS_INVALID_ARG;
    }

    std::int64_t dim = static_cast<std::int64_t>(reduce_dim);
    if (dim < 0) {
      dim += R;
    }
    if (dim < 0 || dim >= R) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: reduction dim out of range");
      return VT_STATUS_INVALID_ARG;
    }

    // Infer keepdim from the value output shape (mirrors TensorIter::reduce_op).
    std::vector<std::int64_t> expected_keep(in_sizes.begin(), in_sizes.end());
    expected_keep[static_cast<std::size_t>(dim)] = 1;

    std::vector<std::int64_t> expected_drop;
    expected_drop.reserve(static_cast<std::size_t>(R > 0 ? R - 1 : 0));
    for (std::int64_t d = 0; d < R; ++d) {
      if (d == dim) continue;
      expected_drop.push_back(in_sizes[static_cast<std::size_t>(d)]);
    }

    const auto& out_sizes = out_impl.sizes();
    const bool matches_keep = (out_sizes == expected_keep);
    const bool matches_drop = (out_sizes == expected_drop);

    bool keepdim = false;
    if (matches_keep && !matches_drop) {
      keepdim = true;
    } else if (matches_drop && !matches_keep) {
      keepdim = false;
    } else {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: output shape does not match reduction dim");
      return VT_STATUS_INVALID_ARG;
    }

    TensorIterConfig cfg_cpp;
    cfg_cpp.check_mem_overlap(true);
    cfg_cpp.is_reduction(true);
    cfg_cpp.add_output(OptionalTensorImplRef(&out_impl, /*defined=*/true),
                       vbt::core::IterOperandRole::ReduceOutput);
    cfg_cpp.add_input(in_impl);

    const std::int64_t dim_arr[1] = {dim};
    cfg_cpp.set_reduce_dims(std::span<const std::int64_t>(dim_arr, 1), keepdim);

    cfg_cpp.check_all_same_dtype(true);
    cfg_cpp.check_all_same_device(true);
    cfg_cpp.promote_inputs_to_common_dtype(false);
    cfg_cpp.promote_integer_inputs_to_float(false);
    cfg_cpp.cast_common_dtype_to_outputs(false);
    cfg_cpp.allow_cpu_scalars(false);
    cfg_cpp.resize_outputs(false);

    if (effective.max_rank != 0) {
      cfg_cpp.set_max_rank(static_cast<std::int64_t>(effective.max_rank));
    }

    TensorIter iter = cfg_cpp.build();  // may throw

    vt_tensor_iter handle = nullptr;
    try {
      handle = new vt_tensor_iter__{};
    } catch (const std::bad_alloc&) {
      set_last_error_helper(
          "vt_tensor_iter_build_reduction: allocation failure for handle");
      return VT_STATUS_NOMEM;
    }

    handle->iter        = std::move(iter);
    handle->device_type = dev_in.type;
    handle->ntensors    = handle->iter.ntensors();
    handle->num_outputs = handle->iter.noutputs();
    handle->kind        = VT_TENSOR_ITER_KIND_REDUCTION;

    *out_iter = handle;
    return VT_STATUS_OK;
  });
}

// ---- Query helpers --------------------------------------------------------------------

vt_status vt_tensor_iter_get_kind_host(vt_tensor_iter iter,
                                       vt_tensor_iter_kind* out_kind) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_get_kind", [&]() -> vt_status {
    if (!iter) {
      set_last_error_helper("vt_tensor_iter_get_kind: iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    if (!out_kind) {
      set_last_error_helper(
          "vt_tensor_iter_get_kind: out_kind must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    *out_kind = iter->kind;
    return VT_STATUS_OK;
  });
}

vt_status vt_tensor_iter_export_desc_host(vt_tensor_iter iter,
                                          vt_tensor_iter_desc* out_desc) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_export_desc", [&]() -> vt_status {
    if (!iter) {
      set_last_error_helper(
          "vt_tensor_iter_export_desc: iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    if (!out_desc) {
      set_last_error_helper(
          "vt_tensor_iter_export_desc: out_desc must not be null");
      return VT_STATUS_INVALID_ARG;
    }

    std::memset(out_desc, 0, sizeof(*out_desc));

    const vbt::core::TensorIterBase& base = iter->iter;

    const int R  = base.ndim();
    const int nt = base.ntensors();
    const int no = base.noutputs();

    out_desc->ndim        = static_cast<std::int32_t>(R);
    out_desc->ntensors    = static_cast<std::int32_t>(nt);
    out_desc->num_outputs = static_cast<std::int32_t>(no);

    const auto& shape = base.shape();
    for (int d = 0; d < R && d < VT_TENSOR_ITER_MAX_RANK; ++d) {
      out_desc->sizes[d] = shape[static_cast<std::size_t>(d)];
    }

    const auto& reduce_dims = base.reduce_dims();
    const int num_rd = static_cast<int>(reduce_dims.size());
    out_desc->num_reduce_dims = static_cast<std::int32_t>(num_rd);
    for (int i = 0; i < num_rd && i < VT_TENSOR_ITER_MAX_RANK; ++i) {
      out_desc->reduce_dims[i] = static_cast<std::int32_t>(reduce_dims[
          static_cast<std::size_t>(i)]);
    }

    for (int k = 0; k < nt && k < VT_TENSOR_ITER_MAX_OPERANDS; ++k) {
      const auto& op = base.operand(k);
      const std::size_t idx = static_cast<std::size_t>(k);

      for (int d = 0; d < R && d < VT_TENSOR_ITER_MAX_RANK; ++d) {
        out_desc->strides[idx][static_cast<std::size_t>(d)] =
            op.dim_stride_bytes[static_cast<std::size_t>(d)];
      }

      out_desc->dtypes[idx] = to_dlpack_dtype(op.dtype);
      out_desc->devices[idx] = DLDevice{op.device.type, op.device.index};

      switch (op.role) {
        case vbt::core::IterOperandRole::ReadOnly:
          out_desc->roles[idx] = VT_TENSOR_ITER_ROLE_READONLY;
          break;
        case vbt::core::IterOperandRole::WriteOnly:
          out_desc->roles[idx] = VT_TENSOR_ITER_ROLE_WRITEONLY;
          break;
        case vbt::core::IterOperandRole::ReadWrite:
          out_desc->roles[idx] = VT_TENSOR_ITER_ROLE_READWRITE;
          break;
        case vbt::core::IterOperandRole::ReduceOutput:
          out_desc->roles[idx] = VT_TENSOR_ITER_ROLE_REDUCE_OUTPUT;
          break;
      }
    }

    return VT_STATUS_OK;
  });
}

vt_status vt_tensor_iter_export_alias_info_host(
    vt_tensor_iter iter,
    vt_tensor_iter_alias_info* out_alias) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_export_alias_info",
                                     [&]() -> vt_status {
    if (!iter) {
      set_last_error_helper(
          "vt_tensor_iter_export_alias_info: iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    if (!out_alias) {
      set_last_error_helper(
          "vt_tensor_iter_export_alias_info: out_alias must not be null");
      return VT_STATUS_INVALID_ARG;
    }

    std::memset(out_alias, 0, sizeof(*out_alias));

    const vbt::core::TensorIterBase& base = iter->iter;
    const int out_count = base.noutputs();
    const int in_count  = base.ninputs();

    out_alias->num_outputs = static_cast<std::uint32_t>(out_count);
    out_alias->num_inputs  = static_cast<std::uint32_t>(in_count);

    if (!base.mem_overlap_checked() || out_count == 0 || in_count == 0) {
      // No alias metadata – conservative "may alias" bitmasks.
      out_alias->has_alias_metadata = 0;
      if (out_count > 0) {
        const int bits = (out_count >= 64) ? 64 : out_count;
        out_alias->output_may_alias_input =
            (bits == 64) ? ~UINT64_C(0) : ((UINT64_C(1) << bits) - 1U);
      }
      if (in_count > 0) {
        const int bits = (in_count >= 64) ? 64 : in_count;
        out_alias->input_may_alias_output =
            (bits == 64) ? ~UINT64_C(0) : ((UINT64_C(1) << bits) - 1U);
      }
      out_alias->has_any_output_input_alias =
          (out_count > 0 && in_count > 0) ? 1u : 0u;
      return VT_STATUS_OK;
    }

    out_alias->has_alias_metadata = 1;

    for (int o = 0; o < out_count && o < 64; ++o) {
      for (int i = 0; i < in_count && i < 64; ++i) {
        const vbt::core::MemOverlapStatus st =
            base.alias_status(o, i);
        if (st == vbt::core::MemOverlapStatus::Full) {
          out_alias->output_may_alias_input |=
              (UINT64_C(1) << static_cast<unsigned>(o));
          out_alias->input_may_alias_output |=
              (UINT64_C(1) << static_cast<unsigned>(i));
        }
      }
    }

    if (out_alias->output_may_alias_input != 0 ||
        out_alias->input_may_alias_output != 0) {
      out_alias->has_any_output_input_alias = 1;
    }

    return VT_STATUS_OK;
  });
}

vt_status vt_tensor_iter_export_cuda_desc_host(
    vt_tensor_iter iter,
    int32_t operand_index,
    int32_t max_ndim,
    vt_tensor_iter_cuda_desc* out_desc) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_export_cuda_desc",
                                     [&]() -> vt_status {
    if (!iter) {
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    if (!out_desc) {
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: out_desc must not be null");
      return VT_STATUS_INVALID_ARG;
    }

    if (operand_index < 0 ||
        operand_index >= iter->ntensors) {
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: operand_index out of range");
      return VT_STATUS_INVALID_ARG;
    }

    if (max_ndim < 1 || max_ndim > VT_TENSOR_ITER_CUDA_MAX_NDIM) {
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: max_ndim out of range");
      return VT_STATUS_INVALID_ARG;
    }

    if (iter->device_type != kDLCUDA) {
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: iterator must be on CUDA");
      return VT_STATUS_INVALID_ARG;
    }

    const vbt::core::TensorIterBase& base = iter->iter;
    const vbt::core::IterOperand& op = base.operand(operand_index);
    const vbt::core::TensorImpl* t = op.tensor;
    if (!t) {
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: operand tensor must not be null");
      return VT_STATUS_INTERNAL;
    }

    const auto& sizes   = t->sizes();
    const auto& strides = t->strides();
    const int   R_tensor = static_cast<int>(sizes.size());

    if (R_tensor > max_ndim) {
      // Operand rank exceeds the plugin-visible CUDA descriptor limit.
      // Leave out_desc unchanged and report UNSUPPORTED.
      set_last_error_helper(
          "vt_tensor_iter_export_cuda_desc: iteration rank exceeds max_ndim");
      return VT_STATUS_UNSUPPORTED;
    }

    // Populate CUDA descriptor directly from the operand's logical shape.
    vt_tensor_iter_cuda_desc desc{};
    desc.operand_index = operand_index;
    desc.ndim = R_tensor;

    const int nd = desc.ndim;
    for (int d = 0; d < nd && d < VT_TENSOR_ITER_CUDA_MAX_NDIM; ++d) {
      desc.sizes[d]   = sizes[static_cast<std::size_t>(d)];
      desc.strides[d] = strides[static_cast<std::size_t>(d)];
    }

    // Zero any unused entries beyond desc.ndim.
    for (int d = nd; d < VT_TENSOR_ITER_CUDA_MAX_NDIM; ++d) {
      desc.sizes[d] = 0;
      desc.strides[d] = 0;
    }

    *out_desc = desc;
    return VT_STATUS_OK;
  });
}

// ---- CPU driver -----------------------------------------------------------------------

vt_status vt_tensor_iter_for_each_cpu_host(vt_tensor_iter iter,
                                           vt_tensor_iter_loop1d_fn loop,
                                           void* ctx) noexcept {
  return with_tensor_iter_api_errors("vt_tensor_iter_for_each_cpu", [&]() -> vt_status {
    if (!iter) {
      set_last_error_helper(
          "vt_tensor_iter_for_each_cpu: iter must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    if (!loop) {
      set_last_error_helper(
          "vt_tensor_iter_for_each_cpu: loop callback must not be null");
      return VT_STATUS_INVALID_ARG;
    }
    if (iter->kind != VT_TENSOR_ITER_KIND_ELEMENTWISE) {
      set_last_error_helper(
          "vt_tensor_iter_for_each_cpu: only elementwise iterators are supported");
      return VT_STATUS_INVALID_ARG;
    }
    if (iter->device_type != kDLCPU) {
      set_last_error_helper(
          "vt_tensor_iter_for_each_cpu: iterator must be on CPU");
      return VT_STATUS_INVALID_ARG;
    }

    ::vbt::core::for_each_cpu(iter->iter,
                              reinterpret_cast<::vbt::core::loop1d_t>(loop),
                              ctx);
    return VT_STATUS_OK;
  });
}

// ---- Destroy --------------------------------------------------------------------------

void vt_tensor_iter_destroy_host(vt_tensor_iter iter) noexcept {
  if (!iter) {
    return;
  }
  try {
    delete iter;
  } catch (...) {
    // Destruction must not throw across the C ABI; swallow.
  }
}

}  // namespace plugin_helpers
}  // namespace dispatch
}  // namespace vbt

// ---- C wrappers -----------------------------------------------------------------------

extern "C" vt_status vt_tensor_iter_build_elementwise(const vt_iter_config* cfg,
                                                       int32_t ntensors,
                                                       const vt_tensor* tensors,
                                                       vt_tensor_iter* out_iter) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_build_elementwise_host(
      cfg, ntensors, tensors, out_iter);
}

extern "C" vt_status vt_tensor_iter_build_reduction(const vt_iter_config* cfg,
                                                     int32_t ntensors,
                                                     const vt_tensor* tensors,
                                                     int32_t reduce_dim,
                                                     vt_tensor_iter* out_iter) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_build_reduction_host(
      cfg, ntensors, tensors, reduce_dim, out_iter);
}

extern "C" vt_status vt_tensor_iter_get_kind(vt_tensor_iter iter,
                                              vt_tensor_iter_kind* out_kind) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_get_kind_host(
      iter, out_kind);
}

extern "C" vt_status vt_tensor_iter_export_desc(vt_tensor_iter iter,
                                                 vt_tensor_iter_desc* out_desc) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_export_desc_host(
      iter, out_desc);
}

extern "C" vt_status vt_tensor_iter_export_alias_info(
    vt_tensor_iter iter,
    vt_tensor_iter_alias_info* out_alias) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_export_alias_info_host(
      iter, out_alias);
}

extern "C" vt_status vt_tensor_iter_export_cuda_desc(
    vt_tensor_iter iter,
    int32_t operand_index,
    int32_t max_ndim,
    vt_tensor_iter_cuda_desc* out_desc) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_export_cuda_desc_host(
      iter, operand_index, max_ndim, out_desc);
}

extern "C" vt_status vt_tensor_iter_for_each_cpu(vt_tensor_iter iter,
                                                  vt_tensor_iter_loop1d_fn loop,
                                                  void* ctx) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_for_each_cpu_host(
      iter, loop, ctx);
}

extern "C" void vt_tensor_iter_destroy(vt_tensor_iter iter) {
  ::vbt::dispatch::plugin_helpers::vt_tensor_iter_destroy_host(iter);
}
