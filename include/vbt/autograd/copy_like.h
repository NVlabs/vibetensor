// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace vbt { namespace autograd {

// Marker interface: a node is copy-like iff dynamic_cast succeeds.
struct CopyLikeNode {
  virtual ~CopyLikeNode() = default;
};

}} // namespace vbt::autograd
