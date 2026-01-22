// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cpu/storage.h"

#include <stdexcept>

#include "vbt/cpu/allocator.h"

namespace vbt { namespace cpu {

vbt::core::StoragePtr new_cpu_storage(std::size_t nbytes, bool pinned) {
  using vbt::core::DataPtr;
  using vbt::core::Storage;
  if (nbytes == 0) {
    return vbt::core::make_intrusive<Storage>(DataPtr(nullptr, nullptr), 0);
  }
  if (!pinned) {
    // Allocate via Allocator and capture exact reserved for precise stats on free
    std::size_t reserved = 0;
    void* p = Allocator::get().raw_alloc(nbytes, 0, reserved);
    DataPtr dp(p, [nbytes, reserved](void* q) noexcept {
      Allocator::get().raw_delete_exact(q, nbytes, reserved);
    });
    return vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  } else {
    void* p = PinnedHostAllocator::get().raw_alloc(nbytes, 0);
    DataPtr dp(p, [](void* q) noexcept {
      PinnedHostAllocator::get().raw_delete(q);
    });
    return vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  }
}

void record_stream(const vbt::core::StoragePtr& st, const vbt::cuda::Stream& s) noexcept {
  if (!st) return;
  void* p = st->data();
  if (!p) return;
  auto& pha = PinnedHostAllocator::get();
  if (!pha.maybe_owns(p)) return;
  pha.record_stream(p, s);
}

}} // namespace vbt::cpu
