// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_iterator/core.h"

#include <stdexcept>

namespace vbt {
namespace core {

TensorIter make_fabric_elementwise_2gpu_iter(
    TensorImpl& out,
    const TensorImpl& a,
    const TensorImpl& b,
    Device primary_dev) {
  if (primary_dev.type != kDLCUDA) {
    throw std::logic_error(
        "[Fabric] TI Fabric: primary device must be CUDA");
  }

  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::WriteOnly,
                 /*allow_resize=*/false)
      .add_input(a)
      .add_input(b)
      .check_mem_overlap(true)
      .check_all_same_dtype(true)
      .check_all_same_device(false)  // relaxed for Fabric
      .is_reduction(false)
      .enable_fabric_2gpu_elementwise(primary_dev.index);

  return cfg.build();
}

}  // namespace core
}  // namespace vbt
