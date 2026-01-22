// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "vbt/core/storage.h"
#include "vbt/cuda/stream.h"

namespace vbt { namespace cuda {

// Allocate a CUDA device buffer of nbytes on device_index and return a Storage owning it.
// Semantics:
// - When VBT_WITH_CUDA==0 or nbytes==0, returns an empty Storage (data==nullptr, nbytes==0).
// - When VBT_WITH_CUDA==1 and nbytes>0: routes through the per-device caching Allocator
//   (Allocator::get(device_index).raw_alloc/raw_delete). Throws std::runtime_error on
//   allocation failure. Deleter is noexcept and swallows errors. device_index==0 by default.
vbt::core::StoragePtr new_cuda_storage(std::size_t nbytes, int device_index = 0);

// Record cross-stream use of a Storage on the given CUDA stream. Metadata-only;
// integrates with the per-device caching allocator for deferred free fencing.
void record_stream(const vbt::core::StoragePtr& storage, Stream s) noexcept;

//
// Semantics:
// - has_producer_metadata(S) is true iff the provider can enumerate a complete
//   set of CUDA streams that may still be producing (writing) to S.
// - for_each_producer_stream enumerates those streams in deterministic order.
//
// The provider is conservative: if metadata is missing or incomplete, it must
// report has_producer_metadata(S) == false.
bool has_producer_metadata(const vbt::core::StoragePtr& storage) noexcept;

namespace detail {
std::vector<Stream> producer_streams_snapshot(const vbt::core::StoragePtr& storage);
}

// fn(Stream) -> bool where false stops early.
template <typename F>
void for_each_producer_stream(const vbt::core::StoragePtr& storage, F&& fn) {
  auto streams = detail::producer_streams_snapshot(storage);
  for (const auto& s : streams) {
    if (!fn(s)) break;
  }
}

#if defined(VBT_INTERNAL_TESTS)
// Test-only helper: clears producer metadata for all tracked storages.
void debug_clear_producer_metadata_for_testing() noexcept;
#endif

// Debug helpers used in tests to validate that high-level ops call record_stream
// exactly once when N>0 and never when N==0.
std::size_t debug_record_stream_call_count() noexcept;
void        debug_reset_record_stream_call_count() noexcept;

}} // namespace vbt::cuda
