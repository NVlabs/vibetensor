// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "vbt/core/storage.h"

namespace vbt { namespace cuda { class Stream; }}

namespace vbt { namespace cpu {

// Allocate CPU memory and wrap it in a Storage with proper deleter.
vbt::core::StoragePtr new_cpu_storage(std::size_t nbytes, bool pinned=false);

// In M_Ext.1, CPU record_stream is a fast no-op.
void record_stream(const vbt::core::StoragePtr& st, const vbt::cuda::Stream& s) noexcept;

}} // namespace vbt::cpu
