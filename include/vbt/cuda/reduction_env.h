// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

namespace vbt {
namespace cuda {
namespace reduction {

// Cached env-derived configuration for CUDA reductions.
//
// NOTE: staged kernels are default-off; this struct is designed to grow as
// additional CUDA reduction kernels (K2/K2-multi/K3) land.
struct CudaReductionEnvConfig {
  bool         staged_default{false};
  std::int64_t cuda_max_blocks_cap{0};  // >0 => additional cap on blocks, <=0 => none
};

// Return the process-wide cached reduction env config.
const CudaReductionEnvConfig& get_cuda_reduction_env_config() noexcept;

// ABI-locked reason codes used when staged reductions (K2/K2Multi/K3)
// fall back to K1.
//
// NOTE: The numeric values are part of the internal ABI/contract and must
// match design/reduction/README.md §5.5.
enum class CudaReduceIneligibleReason : std::uint8_t {
  None = 0,
  EmptyOutNumel,             // 1
  EmptySlice,                // 2
  Overflow,                  // 3
  RedStrideZero,             // 4
  RedNotLinearizable,        // 5
  RedMultiDimNegativeStride, // 6
  KeptNegativeStride,        // 7
};

static_assert(static_cast<int>(CudaReduceIneligibleReason::None) == 0);
static_assert(static_cast<int>(CudaReduceIneligibleReason::EmptyOutNumel) == 1);
static_assert(static_cast<int>(CudaReduceIneligibleReason::EmptySlice) == 2);
static_assert(static_cast<int>(CudaReduceIneligibleReason::Overflow) == 3);
static_assert(static_cast<int>(CudaReduceIneligibleReason::RedStrideZero) == 4);
static_assert(static_cast<int>(CudaReduceIneligibleReason::RedNotLinearizable) == 5);
static_assert(static_cast<int>(CudaReduceIneligibleReason::RedMultiDimNegativeStride) == 6);
static_assert(static_cast<int>(CudaReduceIneligibleReason::KeptNegativeStride) == 7);
static_assert(sizeof(CudaReduceIneligibleReason) == sizeof(std::uint8_t));
static_assert(alignof(CudaReduceIneligibleReason) == alignof(std::uint8_t));

// Meta-source contract (staged eligibility):
//
// Reason                     Uses in_meta?  Uses out_meta?  Dims           Notes
// -------------------------  ------------  -------------  -------------  ------------------------------
// RedStrideZero               yes           no             reduced        consult *input* strides only
// RedNotLinearizable          yes           no             reduced        consult *input* strides only
// RedMultiDimNegativeStride   yes           no             reduced        consult *input* strides only
// KeptNegativeStride          yes           yes            kept           consult both in/out strides
// EmptyOutNumel/EmptySlice    no            no             n/a            dispatcher-level empty cases
// Overflow                    yes           yes            all            validation / checked-mul fail
//
// Important: TensorIterator reductions set output strides to 0 on reduced dims,
// so consulting out_meta.strides[d] for reduced dims will yield false positives.
//
// Active-dims policy: stride-based checks ignore dims with size == 1.
// (size==0 is handled by empty semantics before staging/eligibility.)
// Size-1 dims do not affect any accessed addresses under TI's index decomposition.

#if VBT_INTERNAL_TESTS
// Internal test-only kernel identifiers and forcing policies.

enum class CudaReductionKernel : std::uint8_t {
  None     = 0,
  K1Atomic = 1,
  K2       = 2,
  K2Multi  = 3,
  K3       = 4,
};

enum class CudaReductionKernelPolicy : std::uint8_t {
  Auto               = 0,
  ForceK1            = 1,
  ForceK2IfEligible  = 2,
  ForceK2Strict      = 3,
  ForceK2MultiIfEligible = 4,
  ForceK2MultiStrict     = 5,
  ForceK3IfEligible  = 6,
  ForceK3Strict      = 7,
};

enum class CudaK2MultiFaultMode : std::uint8_t {
  None = 0,
  SignalButSkipPartialWrite = 1,
};

struct CudaReductionLastStats {
  CudaReductionKernel       selected_kernel{CudaReductionKernel::None};
  CudaReductionKernelPolicy requested_policy{CudaReductionKernelPolicy::Auto};

  unsigned int grid_x{0};
  unsigned int grid_y{0};
  unsigned int grid_z{0};

  unsigned int block_x{0};
  unsigned int block_y{0};
  unsigned int block_z{0};

  CudaReduceIneligibleReason ineligible_reason{CudaReduceIneligibleReason::None};
  std::int64_t out_numel{0};
  std::int64_t slice_len{0};

  // Forcing + planning
  bool policy_override_active{false};
  bool want_plan{false};

  // Plan summary (best-effort; meaningful when want_plan is true)
  std::int32_t plan_iter_ndim{0};
  std::int32_t plan_kept_ndim{0};
  std::int32_t plan_red_ndim{0};
  std::int64_t plan_red_linear_stride{0};

  // K2 scratch
  std::uint32_t k2_smem_bytes{0};

  // K2-multi
  std::uint32_t k2multi_ctas_per_output{0};
  std::uint64_t k2multi_workspace_partials_bytes{0};
  std::uint64_t k2multi_workspace_sema_off{0};
  std::uint64_t k2multi_workspace_total_bytes{0};

  // Stream context
  std::uint64_t launch_stream_id{0};
};

CudaReductionLastStats get_cuda_reduction_last_stats_for_tests() noexcept;
void reset_cuda_reduction_last_stats_for_tests() noexcept;
void set_cuda_reduction_last_stats_for_tests(const CudaReductionLastStats& stats) noexcept;

CudaReductionKernelPolicy get_cuda_reduction_kernel_policy_for_tests() noexcept;
void set_cuda_reduction_kernel_policy_for_tests(CudaReductionKernelPolicy policy) noexcept;
void clear_cuda_reduction_kernel_policy_override_for_tests() noexcept;
bool cuda_reduction_kernel_policy_override_is_active_for_tests() noexcept;

std::optional<unsigned int> get_cuda_reduction_grid_x_cap_for_tests() noexcept;
void set_cuda_reduction_grid_x_cap_for_tests(std::optional<unsigned int> cap);
void clear_cuda_reduction_grid_x_cap_override_for_tests() noexcept;

// Requested K2-multi CTAs-per-output override (pre-clamp).
std::optional<unsigned int> get_cuda_reduction_k2multi_ctas_per_output_for_tests() noexcept;
void set_cuda_reduction_k2multi_ctas_per_output_for_tests(
    std::optional<unsigned int> ctas_per_output);
void clear_cuda_reduction_k2multi_ctas_per_output_override_for_tests() noexcept;

CudaK2MultiFaultMode get_cuda_reduction_k2multi_fault_mode_for_tests() noexcept;
void set_cuda_reduction_k2multi_fault_mode_for_tests(CudaK2MultiFaultMode mode) noexcept;
void clear_cuda_reduction_k2multi_fault_mode_override_for_tests() noexcept;
bool cuda_reduction_k2multi_fault_mode_override_is_active_for_tests() noexcept;

// Deterministic stream-mismatch hazard injection for K2-multi workspace.
//
// When enabled, the CUDA reduction dispatcher allocates any K2-multi workspace on
// the *current* stream (typically default), but launches K2-multi kernels on a
// non-default stream. This creates a stream mismatch that would be unsafe if
// the workspace is not recorded on the launch stream via record_stream().
//
// Used by tests/cpp/cuda_reduction_record_stream_hazard_test.cc.
bool cuda_reduction_k2multi_stream_mismatch_injection_enabled_for_tests() noexcept;
void set_cuda_reduction_k2multi_stream_mismatch_injection_enabled_for_tests(bool enabled) noexcept;
void clear_cuda_reduction_k2multi_stream_mismatch_injection_override_for_tests() noexcept;
bool cuda_reduction_k2multi_stream_mismatch_injection_override_is_active_for_tests() noexcept;

// Optional env-config override helpers (white-box; used by C++ tests).
CudaReductionEnvConfig get_cuda_reduction_env_config_for_tests() noexcept;
void reset_cuda_reduction_env_config_for_tests(const CudaReductionEnvConfig& cfg) noexcept;
void clear_cuda_reduction_env_config_override_for_tests() noexcept;
bool cuda_reduction_env_config_override_is_active_for_tests() noexcept;
#endif  // VBT_INTERNAL_TESTS

} // namespace reduction
} // namespace cuda
} // namespace vbt
