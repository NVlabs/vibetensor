// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/fabric_addmul_decision.h"

#include <array>

#include "vbt/cuda/fabric_state.h"
#include "vbt/cuda/fabric_topology.h"
#include "vbt/cuda/graphs.h"

#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/forward.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#endif

namespace vbt { namespace cuda { namespace fabric {

namespace {

inline bool is_supported_dtype(vbt::core::ScalarType t) noexcept {
  using vbt::core::ScalarType;
  switch (t) {
    case ScalarType::Float32:
    case ScalarType::Float16:
    case ScalarType::BFloat16:
    case ScalarType::Int64:
      return true;
    default:
      return false;
  }
}

inline bool allow_copy_fallback_for_reason(FabricAddMulFallbackReason r) noexcept {
  switch (r) {
    case FabricAddMulFallbackReason::kFabricGloballyDisabled:
    case FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P:
    case FabricAddMulFallbackReason::kRequiresGrad:
    case FabricAddMulFallbackReason::kInBackward:
      return true;
    default:
      return false;
  }
}

}  // namespace

FabricAddMulDecision decide_fabric_addmul_2gpu(
    int compute_device,
    const vbt::core::TensorImpl& a,
    const vbt::core::TensorImpl& b,
    bool require_fabric,
    bool use_copy_fallback) {
  FabricAddMulDecision dec;

  auto invalidate_compute = [&]() {
    dec.reason = FabricAddMulFallbackReason::kInvalidComputeDevice;
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
    dec.primary_device = -1;
    dec.other_device = -1;
    dec.numel = 0;
  };

  // checks so that tests can synthesize topologies via FabricTestHooks.
  FabricState& fs = fabric_state();
  const int topo_dc = fs.topology.device_count;
  if (compute_device < 0 || compute_device >= topo_dc) {
    invalidate_compute();
    return dec;
  }

  // For all non-kInvalidComputeDevice outcomes, keep primary_device stable.
  dec.primary_device = compute_device;
  dec.other_device = -1;

  // Fast argument validation (no Fabric gating decisions yet).
  if (a.device().type != kDLCUDA || b.device().type != kDLCUDA) {
    dec.reason = FabricAddMulFallbackReason::kNotCuda;
    return dec;
  }

  // Treat tensors whose device index is outside the current topology as an
  // invalid compute-device configuration.
  const int da = static_cast<int>(a.device().index);
  const int db = static_cast<int>(b.device().index);
  if (da < 0 || db < 0 || da >= topo_dc || db >= topo_dc) {
    invalidate_compute();
    return dec;
  }

  if (a.sizes() != b.sizes() || a.dtype() != b.dtype() || !is_supported_dtype(a.dtype())) {
    dec.reason = FabricAddMulFallbackReason::kInvalidShapesOrDtypes;
    return dec;
  }

  dec.numel = a.numel();

  // Classify device set S = sorted_unique({da, db}).
  if (da == db) {
    // Single-device: this is explicitly treated as a non-Fabric case.
    dec.reason = FabricAddMulFallbackReason::kNone;
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
    dec.primary_device = da;
    dec.other_device = -1;
    return dec;
  }

  // Two-device candidate; ensure compute_device is one of {da, db}.
  if (compute_device != da && compute_device != db) {
    invalidate_compute();
    return dec;
  }

  dec.primary_device = compute_device;
  dec.other_device = (compute_device == da) ? db : da;

  if (!fabric_enabled_for_ops(fs)) {
    dec.reason = FabricAddMulFallbackReason::kFabricGloballyDisabled;
    // action mapping below
  } else {
    // Clique / topology usability for this (primary, other) pair.
    const std::array<int, 2> devices{{dec.primary_device, dec.other_device}};
    if (!is_fabric_usable_for_with_primary(dec.primary_device, devices, fs.topology)) {
      dec.reason = FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P;
      // action mapping below
    } else {
#if VBT_WITH_AUTOGRAD
      if (vbt::autograd::is_in_backward()) {
        dec.reason = FabricAddMulFallbackReason::kInBackward;
      } else if (vbt::autograd::GradMode::is_enabled() ||
                 vbt::autograd::requires_grad(a) ||
                 vbt::autograd::requires_grad(b)) {
        dec.reason = FabricAddMulFallbackReason::kRequiresGrad;
      }
#endif
      // Graph capture guard: Fabric ops are not allowed under capture.
      if (dec.reason == FabricAddMulFallbackReason::kNone) {
        const auto st = vbt::cuda::currentStreamCaptureStatus(
            static_cast<vbt::cuda::DeviceIndex>(dec.primary_device));
        if (st == vbt::cuda::CaptureStatus::Active) {
          dec.reason = FabricAddMulFallbackReason::kGraphCaptureActive;
        }
      }
    }
  }

  // Convert reason + per-call flags to action.
  if (dec.reason == FabricAddMulFallbackReason::kNone) {
    dec.use_fabric = true;
    dec.use_copy_fallback = false;
    return dec;
  }

  // Invalid compute device is always a hard error with no fallback.
  if (dec.reason == FabricAddMulFallbackReason::kInvalidComputeDevice) {
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
    return dec;
  }

  if (dec.reason == FabricAddMulFallbackReason::kGraphCaptureActive) {
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
    return dec;
  }

  // Argument-level failures are never eligible for copy fallback.
  if (!allow_copy_fallback_for_reason(dec.reason)) {
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
    return dec;
  }

  // Policy-controlled fallback for recoverable reasons.
  if (require_fabric) {
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
    return dec;
  }

  if (use_copy_fallback) {
    dec.use_fabric = false;
    dec.use_copy_fallback = true;
  } else {
    dec.use_fabric = false;
    dec.use_copy_fallback = false;
  }

  return dec;
}

}}} // namespace vbt::cuda::fabric
