// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <stdexcept>

#include "vbt/cuda/reduction_env.h"

namespace nb = nanobind;

namespace vbt_py {

void bind_cuda_reduction(nb::module_& m) {
#if VBT_INTERNAL_TESTS
  using namespace vbt::cuda::reduction;

  auto kernel_to_string = [](CudaReductionKernel k) -> const char* {
    switch (k) {
      case CudaReductionKernel::None:
        return "None";
      case CudaReductionKernel::K1Atomic:
        return "K1";
      case CudaReductionKernel::K2:
        return "K2";
      case CudaReductionKernel::K2Multi:
        return "K2Multi";
      case CudaReductionKernel::K3:
        return "K3";
    }
    return "Unknown";
  };

  auto policy_to_string = [](CudaReductionKernelPolicy p) -> const char* {
    switch (p) {
      case CudaReductionKernelPolicy::Auto:
        return "Auto";
      case CudaReductionKernelPolicy::ForceK1:
        return "ForceK1";
      case CudaReductionKernelPolicy::ForceK2IfEligible:
        return "ForceK2IfEligible";
      case CudaReductionKernelPolicy::ForceK2Strict:
        return "ForceK2Strict";
      case CudaReductionKernelPolicy::ForceK2MultiIfEligible:
        return "ForceK2MultiIfEligible";
      case CudaReductionKernelPolicy::ForceK2MultiStrict:
        return "ForceK2MultiStrict";
      case CudaReductionKernelPolicy::ForceK3IfEligible:
        return "ForceK3IfEligible";
      case CudaReductionKernelPolicy::ForceK3Strict:
        return "ForceK3Strict";
    }
    return "Unknown";
  };

  m.def("_cuda_reduction_reset_last_stats_for_tests",
        &reset_cuda_reduction_last_stats_for_tests);

  m.def("_cuda_reduction_last_stats_for_tests", [=]() {
    const CudaReductionLastStats s = get_cuda_reduction_last_stats_for_tests();
    nb::dict d;
    d["selected_kernel"] = nb::str(kernel_to_string(s.selected_kernel));
    d["selected_kernel_id"] = nb::int_(static_cast<int>(s.selected_kernel));
    d["requested_policy"] = nb::str(policy_to_string(s.requested_policy));
    d["requested_policy_id"] = nb::int_(static_cast<int>(s.requested_policy));
    d["grid"] = nb::make_tuple(s.grid_x, s.grid_y, s.grid_z);
    d["block"] = nb::make_tuple(s.block_x, s.block_y, s.block_z);
    d["ineligible_reason"] = nb::int_(static_cast<int>(s.ineligible_reason));
    d["out_numel"] = nb::int_(s.out_numel);
    d["slice_len"] = nb::int_(s.slice_len);
    d["policy_override_active"] = nb::bool_(s.policy_override_active);
    d["want_plan"] = nb::bool_(s.want_plan);
    d["plan_iter_ndim"] = nb::int_(s.plan_iter_ndim);
    d["plan_kept_ndim"] = nb::int_(s.plan_kept_ndim);
    d["plan_red_ndim"] = nb::int_(s.plan_red_ndim);
    d["plan_red_linear_stride"] = nb::int_(s.plan_red_linear_stride);
    d["k2_smem_bytes"] = nb::int_(s.k2_smem_bytes);
    d["k2multi_ctas_per_output"] = nb::int_(s.k2multi_ctas_per_output);
    d["k2multi_workspace_partials_bytes"] =
        nb::int_(static_cast<unsigned long long>(s.k2multi_workspace_partials_bytes));
    d["k2multi_workspace_sema_off"] =
        nb::int_(static_cast<unsigned long long>(s.k2multi_workspace_sema_off));
    d["k2multi_workspace_total_bytes"] =
        nb::int_(static_cast<unsigned long long>(s.k2multi_workspace_total_bytes));
    d["launch_stream_id"] =
        nb::int_(static_cast<unsigned long long>(s.launch_stream_id));
    return d;
  });

  m.def("_cuda_reduction_set_kernel_policy_for_tests", [](int raw) {
    if (raw < 0 || raw > static_cast<int>(CudaReductionKernelPolicy::ForceK3Strict)) {
      throw std::invalid_argument("cuda_reduction: invalid kernel policy");
    }
    set_cuda_reduction_kernel_policy_for_tests(
        static_cast<CudaReductionKernelPolicy>(raw));
  });

  m.def("_cuda_reduction_clear_kernel_policy_override_for_tests",
        &clear_cuda_reduction_kernel_policy_override_for_tests);

  m.def("_cuda_reduction_kernel_policy_override_is_active_for_tests",
        &cuda_reduction_kernel_policy_override_is_active_for_tests);

  m.def(
      "_cuda_reduction_set_grid_x_cap_for_tests",
      [](nb::object cap_obj) {
        std::optional<unsigned int> cap;
        if (!cap_obj.is_none()) {
          cap = nb::cast<unsigned int>(cap_obj);
        }
        set_cuda_reduction_grid_x_cap_for_tests(cap);
      },
      nb::arg("cap").none(true) = nb::none());

  m.def("_cuda_reduction_get_grid_x_cap_for_tests",
        &get_cuda_reduction_grid_x_cap_for_tests);

  m.def(
      "_cuda_reduction_set_k2multi_ctas_per_output_for_tests",
      [](nb::object ctas_obj) {
        std::optional<unsigned int> ctas;
        if (!ctas_obj.is_none()) {
          ctas = nb::cast<unsigned int>(ctas_obj);
        }
        set_cuda_reduction_k2multi_ctas_per_output_for_tests(ctas);
      },
      nb::arg("ctas_per_output").none(true) = nb::none());

  m.def("_cuda_reduction_get_k2multi_ctas_per_output_for_tests",
        &get_cuda_reduction_k2multi_ctas_per_output_for_tests);

  m.def("_cuda_reduction_clear_k2multi_ctas_per_output_override_for_tests",
        &clear_cuda_reduction_k2multi_ctas_per_output_override_for_tests);

  m.def("_cuda_reduction_get_k2multi_fault_mode_for_tests", [=]() {
    return nb::int_(
        static_cast<int>(get_cuda_reduction_k2multi_fault_mode_for_tests()));
  });

  m.def("_cuda_reduction_set_k2multi_fault_mode_for_tests", [](int raw) {
    if (raw < 0 ||
        raw > static_cast<int>(CudaK2MultiFaultMode::SignalButSkipPartialWrite)) {
      throw std::invalid_argument("cuda_reduction: invalid k2multi fault mode");
    }
    set_cuda_reduction_k2multi_fault_mode_for_tests(
        static_cast<CudaK2MultiFaultMode>(raw));
  });

  m.def("_cuda_reduction_clear_k2multi_fault_mode_override_for_tests",
        &clear_cuda_reduction_k2multi_fault_mode_override_for_tests);

  m.def("_cuda_reduction_k2multi_fault_mode_override_is_active_for_tests",
        &cuda_reduction_k2multi_fault_mode_override_is_active_for_tests);

  m.def("_cuda_reduction_get_env_config_for_tests", []() {
    const CudaReductionEnvConfig cfg = get_cuda_reduction_env_config_for_tests();
    nb::dict d;
    d["staged_default"] = nb::bool_(cfg.staged_default);
    d["cuda_max_blocks_cap"] = nb::int_(cfg.cuda_max_blocks_cap);
    d["override_active"] =
        nb::bool_(cuda_reduction_env_config_override_is_active_for_tests());
    return d;
  });

  m.def("_cuda_reduction_set_env_config_for_tests",
        [](bool staged_default, std::int64_t cuda_max_blocks_cap) {
          CudaReductionEnvConfig cfg{};
          cfg.staged_default = staged_default;
          cfg.cuda_max_blocks_cap = cuda_max_blocks_cap;
          reset_cuda_reduction_env_config_for_tests(cfg);
        },
        nb::arg("staged_default"),
        nb::arg("cuda_max_blocks_cap") = 0);

  m.def("_cuda_reduction_clear_env_config_override_for_tests",
        &clear_cuda_reduction_env_config_override_for_tests);

  m.def("_cuda_reduction_env_config_override_is_active_for_tests",
        &cuda_reduction_env_config_override_is_active_for_tests);

  // Simple integer constants for tests (avoid enum binding dependencies).
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_AUTO") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::Auto));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK1));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK2IfEligible));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK2Strict));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_IF_ELIGIBLE") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK2MultiIfEligible));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK2MultiStrict));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_IF_ELIGIBLE") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK3IfEligible));
  m.attr("_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_STRICT") =
      nb::int_(static_cast<int>(CudaReductionKernelPolicy::ForceK3Strict));

  m.attr("_CUDA_REDUCTION_K2MULTI_FAULT_MODE_NONE") =
      nb::int_(static_cast<int>(CudaK2MultiFaultMode::None));
  m.attr("_CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE") =
      nb::int_(static_cast<int>(CudaK2MultiFaultMode::SignalButSkipPartialWrite));

  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_NONE") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::None));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_OUT_NUMEL") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::EmptyOutNumel));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_SLICE") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::EmptySlice));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_OVERFLOW") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::Overflow));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_RED_STRIDE_ZERO") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::RedStrideZero));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::RedNotLinearizable));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_RED_MULTI_DIM_NEGATIVE_STRIDE") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::RedMultiDimNegativeStride));
  m.attr("_CUDA_REDUCTION_INELIGIBLE_REASON_KEPT_NEGATIVE_STRIDE") =
      nb::int_(static_cast<int>(CudaReduceIneligibleReason::KeptNegativeStride));
#endif  // VBT_INTERNAL_TESTS
}

} // namespace vbt_py
