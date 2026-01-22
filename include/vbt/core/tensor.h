// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <utility>
#include <type_traits>
#include <atomic>

#include "vbt/core/storage.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/contiguity.h"
#include "vbt/core/memory_format.h"

#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/meta.h"
#endif

namespace vbt {
namespace core {

struct VersionCounter : public IntrusiveRefcounted {
  std::atomic<int64_t> v{0};
};

using VersionCounterPtr = intrusive_ptr<VersionCounter>;

class TensorImpl {
 public:
  TensorImpl() = default;
  explicit TensorImpl(StoragePtr storage,
                      std::vector<int64_t> sizes,
                      std::vector<int64_t> strides,
                      int64_t storage_offset,
                      ScalarType dtype,
                      Device device)
      : storage_(std::move(storage)),
        sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        storage_offset_(storage_offset),
        dtype_(dtype),
        device_(device),
        flags_(0),
        version_(make_intrusive<VersionCounter>()) {}

  // Internal ctor to share version counter
  TensorImpl(StoragePtr storage,
             std::vector<int64_t> sizes,
             std::vector<int64_t> strides,
             int64_t storage_offset,
             ScalarType dtype,
             Device device,
             VersionCounterPtr version,
             std::uint8_t flags)
      : storage_(std::move(storage)),
        sizes_(std::move(sizes)),
        strides_(std::move(strides)),
        storage_offset_(storage_offset),
        dtype_(dtype),
        device_(device),
        flags_(flags),
        version_(std::move(version)) {}

  const std::vector<int64_t>& sizes() const noexcept { return sizes_; }
  const std::vector<int64_t>& strides() const noexcept { return strides_; }
  int64_t storage_offset() const noexcept { return storage_offset_; }
  ScalarType dtype() const noexcept { return dtype_; }
  Device device() const noexcept { return device_; }
  const StoragePtr& storage() const noexcept { return storage_; }

  std::size_t itemsize() const noexcept { return vbt::core::itemsize(dtype_); }

  int64_t numel() const noexcept {
    if (sizes_.empty()) return 1; // scalar
    int64_t n = 1;
    for (auto s : sizes_) {
      if (s == 0) return 0;
      int64_t tmp = 0;
      if (!vbt::core::checked_mul_i64(n, s, tmp)) {
        return 0;
      }
      n = tmp;
    }
    return n;
  }

  void* data() const noexcept {
    if (!storage_ || numel() == 0) return nullptr;
    auto* base = static_cast<std::uint8_t*>(storage_->data());
    if (!base) return nullptr;
    return static_cast<void*>(base + itemsize() * static_cast<std::size_t>(storage_offset_));
  }

  bool is_contiguous() const noexcept {
    const auto rank = sizes_.size();
    if (rank == 0) return true;
    for (auto s : sizes_) if (s == 0) return true;
    // compute expected row-major strides ignoring size==1 dims
    int64_t expected = 1;
    for (std::size_t i = rank; i-- > 0;) {
      const auto sz = sizes_[i];
      const auto st = strides_[i];
      if (sz == 1) continue; // stride doesn't matter for size-1 dims
      if (st != expected) return false;
      expected *= sz;
    }
    return true;
  }

  bool is_non_overlapping_and_dense() const noexcept {
    return vbt::core::compute_non_overlapping_and_dense(sizes_, strides_);
  }

  // In PyTorch, _or_false may return a symbolic False; here it is identical
  // to is_non_overlapping_and_dense for concrete sizes/strides.
  bool is_non_overlapping_and_dense_or_false() const noexcept {
    return is_non_overlapping_and_dense();
  }

  bool is_conj() const noexcept { return (flags_ & kConj) != 0; }

  bool is_channels_last() const noexcept {
    if (sizes_.size() != 4) return false;
    const int64_t c = sizes_[1];
    const int64_t h = sizes_[2];
    const int64_t w = sizes_[3];
    const int64_t sn = strides_[0];
    const int64_t sc = strides_[1];
    const int64_t sh = strides_[2];
    const int64_t sw = strides_[3];
    
    if (sc != 1) return false;
    if (sw != c) return false;

    int64_t cw = 0;
    if (!vbt::core::checked_mul_i64(c, w, cw)) return false;
    if (sh != cw) return false;

    int64_t cwh = 0;
    if (!vbt::core::checked_mul_i64(cw, h, cwh)) return false;
    if (sn != cwh) return false;

    return true;
  }

  MemoryFormat suggest_memory_format() const noexcept {
    if (is_channels_last()) return MemoryFormat::ChannelsLast;
    return MemoryFormat::Contiguous;
  }

  TensorImpl as_strided(const std::vector<int64_t>& new_sizes,
                        const std::vector<int64_t>& new_strides,
                        int64_t new_storage_offset) const;

  // Internal helper for dtype-changing views. Shares version counter and
  // computes bounds based on itemsize(new_dtype).
  TensorImpl as_strided_dtype_(const std::vector<int64_t>& new_sizes,
                              const std::vector<int64_t>& new_strides,
                              int64_t new_storage_offset,
                              ScalarType new_dtype) const;

  // Versioning
  void bump_version() noexcept { if (version_) version_->v.fetch_add(1, std::memory_order_relaxed); }
  int64_t version() const noexcept { return version_ ? version_->v.load(std::memory_order_relaxed) : 0; }

  void set_sizes_and_strides(std::vector<int64_t> new_sizes, std::vector<int64_t> new_strides) {
    sizes_ = std::move(new_sizes);
    strides_ = std::move(new_strides);
  }
  void set_storage_offset(int64_t new_offset) { storage_offset_ = new_offset; }
  void set_storage(StoragePtr new_storage) { storage_ = std::move(new_storage); }

#if VBT_WITH_AUTOGRAD
  // Accessors for autograd metadata (created on demand by helpers)
  vbt::core::intrusive_ptr<vbt::autograd::AutogradMeta>& _autograd() noexcept { return autograd_meta_; }
  const vbt::core::intrusive_ptr<vbt::autograd::AutogradMeta>& _autograd() const noexcept { return autograd_meta_; }
#endif

 private:
  static constexpr std::uint8_t kConj = 1u << 0;

  friend TensorImpl conj(const TensorImpl& self);

  StoragePtr storage_{};
  std::vector<int64_t> sizes_{};
  std::vector<int64_t> strides_{};
  int64_t storage_offset_{0};
  ScalarType dtype_{ScalarType::Float32};
  Device device_{};
  std::uint8_t flags_{0};
  VersionCounterPtr version_{make_intrusive<VersionCounter>()};
#if VBT_WITH_AUTOGRAD
  vbt::core::intrusive_ptr<vbt::autograd::AutogradMeta> autograd_meta_{};
#endif
};

} // namespace core
} // namespace vbt
