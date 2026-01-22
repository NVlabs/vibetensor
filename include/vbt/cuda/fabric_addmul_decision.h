// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "vbt/core/tensor.h"

namespace vbt { namespace cuda { namespace fabric {

// Decision helper for 2-GPU elementwise Fabric add/mul.
//
// This header is intentionally small: it contains only pure decision logic

enum class FabricAddMulFallbackReason : std::uint8_t {
  kNone = 0,

  // Call-level / argument problems
  kInvalidComputeDevice,
  kNotEnoughDevices,       // reserved: invalid device cardinality / configuration
  kNotCuda,                // non-CUDA tensor present
  kInvalidShapesOrDtypes,  // non-equal shapes or unsupported dtype/layout

  // Fabric-state / topology problems
  kFabricGloballyDisabled, // fabric_enabled_for_ops(fs) == false
  kNotInSameCliqueOrNoP2P, // devices not usable together via is_fabric_usable_for_with_primary

  // Autograd / graph capture
  kRequiresGrad,
  kInBackward,
  kGraphCaptureActive,
};

struct FabricAddMulDecision {
  bool use_fabric{false};
  bool use_copy_fallback{false};
  FabricAddMulFallbackReason reason{FabricAddMulFallbackReason::kNone};

  int primary_device{-1}; // compute device (Dp)
  int other_device{-1};   // remote device (Dr), or -1 when not applicable

  std::int64_t numel{0};  // total elements (for heuristics / stats)
};

FabricAddMulDecision decide_fabric_addmul_2gpu(
    int compute_device,
    const vbt::core::TensorImpl& a,
    const vbt::core::TensorImpl& b,
    bool require_fabric,
    bool use_copy_fallback);

}}} // namespace vbt::cuda::fabric
