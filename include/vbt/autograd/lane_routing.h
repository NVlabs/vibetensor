// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>

#include "vbt/autograd/node.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/device.h"

namespace vbt { namespace autograd {

// Execution lanes for the multithreaded autograd engine.
// - CPU lane: executed on the owner thread or a CPU worker.
// - CUDA lane: executed on the single CUDA worker thread for the backward run.
enum class Lane : std::uint8_t {
  CPU,
  CUDA,
};

// Per-backward-run lane routing mode.
// SingleLaneCPU is an incremental rollout safety knob: it forces all nodes onto
// the CPU lane even when autograd_device is CUDA.
enum class LaneMode : std::uint8_t {
  Default,
  SingleLaneCPU,
};

inline Lane lane_for_node(const vbt::core::Device& autograd_device,
                          const Node& n,
                          LaneMode mode = LaneMode::Default) noexcept {
  // LR4: incremental safety override.
  if (mode == LaneMode::SingleLaneCPU) {
    return Lane::CPU;
  }

  // LR1: CPU-only backwards always run on CPU lane.
  if (autograd_device.type == kDLCPU) {
    return Lane::CPU;
  }

  // LR2: AccumulateGrad is owner/CPU-lane only (hooks / leaf grad mutex).
  if (dynamic_cast<const AccumulateGrad*>(&n) != nullptr) {
    return Lane::CPU;
  }

  // LR3: CUDA backwards route allowlisted nodes to the CUDA lane.
  if (autograd_device.type == kDLCUDA) {
    return (n.stream_kind() == StreamKind::CudaAllowlisted) ? Lane::CUDA : Lane::CPU;
  }

  // Unknown device type: assert in debug, fall back to CPU.
  assert(false && "lane_for_node: unsupported autograd device type");
  return Lane::CPU;
}

}} // namespace vbt::autograd
