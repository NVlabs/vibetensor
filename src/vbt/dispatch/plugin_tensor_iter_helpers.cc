// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/plugin/vbt_plugin.h"
#include "vbt/dispatch/plugin_loader.h"
#include "vbt/core/tensor_iterator/core.h"
#include "vbt/core/tensor_iterator/cpu.h"

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace vbt {
namespace dispatch {
namespace plugin_helpers {

using ::vbt::core::TensorImpl;
using ::vbt::core::TensorIter;
using ::vbt::core::TensorIterConfig;
using ::vbt::core::OptionalTensorImplRef;

static_assert(sizeof(vt_iter_overlap_mode) == sizeof(std::int32_t),
              "vt_iter_overlap_mode must remain 32-bit for ABI stability");
static_assert(offsetof(vt_iter_config, check_mem_overlap) == sizeof(std::int64_t),
              "vt_iter_config::check_mem_overlap must follow max_rank");
static_assert(sizeof(vt_iter_config) >= sizeof(std::int64_t) + sizeof(std::int32_t),
              "vt_iter_config must be at least {int64_t, int32_t} in size");

// Default helper configuration used when cfg == nullptr.
static inline vt_iter_config make_default_cfg() {
  vt_iter_config cfg;
  cfg.max_rank = 0;  // use TI default (kTensorIterMaxRank)
  cfg.check_mem_overlap = VT_ITER_OVERLAP_ENABLE;
  return cfg;
}

// Validate vt_tensor handle and return the underlying TensorImpl reference.
inline TensorImpl& require_tensor_impl(vt_tensor h, const char* arg_name) {
  return ::vbt::dispatch::plugin::detail::require_tensor_impl(h, arg_name);
}

// Shared exception-to-status mapping for TI-backed helpers.
template <class F>
vt_status with_tensor_iter_helper_errors(const char* api_name, F&& fn) noexcept {
  try {
    ::vbt::dispatch::plugin::detail::set_last_error_helper("");  // clear on entry
    fn();                  // may throw
    ::vbt::dispatch::plugin::detail::set_last_error_helper("");  // clear on success
    return VT_STATUS_OK;
  } catch (const std::invalid_argument& e) {
    ::vbt::dispatch::plugin::detail::set_last_error_helper(e.what());
    return VT_STATUS_INVALID_ARG;
  } catch (const std::logic_error& e) {
    // these typically arise from configuration or user-visible contract
    // violations (e.g., unsupported flags, max_rank range).
    ::vbt::dispatch::plugin::detail::set_last_error_helper(e.what());
    return VT_STATUS_INVALID_ARG;
  } catch (const std::exception& e) {
    ::vbt::dispatch::plugin::detail::set_last_error_helper(e.what() ? e.what()
                            : "std::exception in vt_tensor_iter_cpu helper");
    return VT_STATUS_RUNTIME_ERROR;
  } catch (...) {
    ::vbt::dispatch::plugin::detail::set_last_error_helper(api_name && *api_name
                       ? api_name
                       : "vt_tensor_iter_cpu helper: unknown non-standard exception");
    return VT_STATUS_RUNTIME_ERROR;
  }
}

namespace {

struct Loop1DThunkCtx {
  vt_tensor_iter_loop1d_fn fn;
  void*                    user_ctx;
};

static void loop1d_thunk(char** data,
                         const std::int64_t* strides,
                         std::int64_t size,
                         void* ctx) {
  auto* thunk = static_cast<Loop1DThunkCtx*>(ctx);
  thunk->fn(data,
            reinterpret_cast<const std::int64_t*>(strides),
            static_cast<std::int64_t>(size),
            thunk->user_ctx);
}

}  // anonymous namespace

vt_status vt_tensor_iter_unary_cpu_host(const vt_iter_config* cfg,
                                        vt_tensor out_h,
                                        vt_tensor in_h,
                                        vt_tensor_iter_loop1d_fn loop,
                                        void* user_ctx) noexcept {
  return with_tensor_iter_helper_errors("vt_tensor_iter_unary_cpu", [&] {
    if (!loop) {
      throw std::invalid_argument(
          "vt_tensor_iter_unary_cpu: loop callback must not be null");
    }

    TensorImpl& out_impl = require_tensor_impl(out_h, "out");
    TensorImpl& in_impl  = require_tensor_impl(in_h, "in");

    if (out_h == in_h) {
      throw std::invalid_argument(
          "vt_tensor_iter_unary_cpu: out and in must be distinct handles");
    }

    // CPU-only and same device
    if (out_impl.device().type != kDLCPU ||
        in_impl.device() != out_impl.device()) {
      throw std::invalid_argument(
          "vt_tensor_iter_unary_cpu: CPU tensors on same device required");
    }

    // Same dtype
    if (out_impl.dtype() != in_impl.dtype()) {
      throw std::invalid_argument(
          "vt_tensor_iter_unary_cpu: all operands must share dtype");
    }

    vt_iter_config effective = cfg ? *cfg : make_default_cfg();

    const std::int64_t max_rank_limit =
        static_cast<std::int64_t>(::vbt::core::kTensorIterMaxRank);
    if (effective.max_rank < 0 || effective.max_rank > max_rank_limit) {
      throw std::invalid_argument(
          "vt_tensor_iter_unary_cpu: max_rank out of range");
    }

    if (effective.check_mem_overlap != VT_ITER_OVERLAP_DISABLE &&
        effective.check_mem_overlap != VT_ITER_OVERLAP_ENABLE) {
      throw std::invalid_argument(
          "vt_tensor_iter_unary_cpu: invalid vt_iter_overlap_mode");
    }

    TensorIterConfig cfg_cpp;
    cfg_cpp.add_output(OptionalTensorImplRef(&out_impl, /*defined=*/true),
                       ::vbt::core::IterOperandRole::WriteOnly,
                       /*allow_resize=*/false);
    cfg_cpp.add_const_input(in_impl);

    cfg_cpp.check_all_same_dtype(true);
    cfg_cpp.check_all_same_device(true);
    cfg_cpp.promote_inputs_to_common_dtype(false);
    cfg_cpp.promote_integer_inputs_to_float(false);
    cfg_cpp.cast_common_dtype_to_outputs(false);
    cfg_cpp.allow_cpu_scalars(false);
    cfg_cpp.is_reduction(false);
    cfg_cpp.resize_outputs(false);

    const bool want_full = (effective.check_mem_overlap == VT_ITER_OVERLAP_ENABLE);
    cfg_cpp.check_mem_overlap(want_full);

    if (effective.max_rank != 0) {
      cfg_cpp.set_max_rank(static_cast<std::int64_t>(effective.max_rank));
    }

    TensorIter iter = cfg_cpp.build();  // may throw

    Loop1DThunkCtx thunk{loop, user_ctx};
    ::vbt::core::for_each_cpu(iter, &loop1d_thunk, &thunk);
  });
}

vt_status vt_tensor_iter_binary_cpu_host(const vt_iter_config* cfg,
                                         vt_tensor out_h,
                                         vt_tensor a_h,
                                         vt_tensor b_h,
                                         vt_tensor_iter_loop1d_fn loop,
                                         void* user_ctx) noexcept {
  return with_tensor_iter_helper_errors("vt_tensor_iter_binary_cpu", [&] {
    if (!loop) {
      throw std::invalid_argument(
          "vt_tensor_iter_binary_cpu: loop callback must not be null");
    }

    TensorImpl& out_impl = require_tensor_impl(out_h, "out");
    TensorImpl& a_impl   = require_tensor_impl(a_h, "a");
    TensorImpl& b_impl   = require_tensor_impl(b_h, "b");

    if (out_h == a_h || out_h == b_h) {
      throw std::invalid_argument(
          "vt_tensor_iter_binary_cpu: out must be distinct from inputs");
    }

    // CPU-only and same device
    if (out_impl.device().type != kDLCPU ||
        a_impl.device() != out_impl.device() ||
        b_impl.device() != out_impl.device()) {
      throw std::invalid_argument(
          "vt_tensor_iter_binary_cpu: CPU tensors on same device required");
    }

    // Same dtype
    if (out_impl.dtype() != a_impl.dtype() ||
        out_impl.dtype() != b_impl.dtype()) {
      throw std::invalid_argument(
          "vt_tensor_iter_binary_cpu: all operands must share dtype");
    }

    vt_iter_config effective = cfg ? *cfg : make_default_cfg();

    const std::int64_t max_rank_limit =
        static_cast<std::int64_t>(::vbt::core::kTensorIterMaxRank);
    if (effective.max_rank < 0 || effective.max_rank > max_rank_limit) {
      throw std::invalid_argument(
          "vt_tensor_iter_binary_cpu: max_rank out of range");
    }

    if (effective.check_mem_overlap != VT_ITER_OVERLAP_DISABLE &&
        effective.check_mem_overlap != VT_ITER_OVERLAP_ENABLE) {
      throw std::invalid_argument(
          "vt_tensor_iter_binary_cpu: invalid vt_iter_overlap_mode");
    }

    TensorIterConfig cfg_cpp;
    cfg_cpp.add_output(OptionalTensorImplRef(&out_impl, /*defined=*/true),
                       ::vbt::core::IterOperandRole::WriteOnly,
                       /*allow_resize=*/false);
    cfg_cpp.add_const_input(a_impl);
    cfg_cpp.add_const_input(b_impl);

    cfg_cpp.check_all_same_dtype(true);
    cfg_cpp.check_all_same_device(true);
    cfg_cpp.promote_inputs_to_common_dtype(false);
    cfg_cpp.promote_integer_inputs_to_float(false);
    cfg_cpp.cast_common_dtype_to_outputs(false);
    cfg_cpp.allow_cpu_scalars(false);
    cfg_cpp.is_reduction(false);
    cfg_cpp.resize_outputs(false);

    const bool want_full = (effective.check_mem_overlap == VT_ITER_OVERLAP_ENABLE);
    cfg_cpp.check_mem_overlap(want_full);

    if (effective.max_rank != 0) {
      cfg_cpp.set_max_rank(static_cast<std::int64_t>(effective.max_rank));
    }

    TensorIter iter = cfg_cpp.build();  // may throw

    Loop1DThunkCtx thunk{loop, user_ctx};
    ::vbt::core::for_each_cpu(iter, &loop1d_thunk, &thunk);
  });
}

}  // namespace plugin_helpers
}  // namespace dispatch
}  // namespace vbt

extern "C" vt_status vt_tensor_iter_unary_cpu(const vt_iter_config* cfg,
                                               vt_tensor out,
                                               vt_tensor in,
                                               vt_tensor_iter_loop1d_fn loop,
                                               void* ctx) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_unary_cpu_host(
      cfg, out, in, loop, ctx);
}

extern "C" vt_status vt_tensor_iter_binary_cpu(const vt_iter_config* cfg,
                                                vt_tensor out,
                                                vt_tensor a,
                                                vt_tensor b,
                                                vt_tensor_iter_loop1d_fn loop,
                                                void* ctx) {
  return ::vbt::dispatch::plugin_helpers::vt_tensor_iter_binary_cpu_host(
      cfg, out, a, b, loop, ctx);
}
