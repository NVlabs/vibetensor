// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <set>
#include <vector>
#include <string>

namespace vbt { namespace cuda { class Stream; }}

namespace vbt { namespace cpu {

struct DeviceStats {
  std::uint64_t allocated_bytes_all_current{0};
  std::uint64_t reserved_bytes_all_current{0};
  std::uint64_t max_allocated_bytes_all{0};
  std::uint64_t max_reserved_bytes_all{0};
};

struct Config {
  std::size_t alignment_bytes{64};
  bool        enable_caching{false};
  std::size_t min_bin_bytes{512};
  std::size_t max_bin_bytes{1ull<<22};
  std::size_t bin_growth{2};
  std::size_t tls_cap_bytes{0};
  std::size_t global_cap_bytes{0};
  bool        oversize_cache{false};
  bool        poison_on_free{false};
  bool        enforce_dlpack_alignment{false};
  std::size_t dlpack_align_min_bytes{64};
  // M_Ext.4 additions (pinned-only semantics)
  bool        mlock{false};
  bool        madvise_hugepage{false};
  std::string numa_policy{"default"};
};

// CPU allocator (singleton). When enable_caching==false, behaves as non-caching aligned allocator.
class Allocator final {
 public:
  static Allocator& get();

  void* raw_alloc(std::size_t nbytes) { return raw_alloc(nbytes, 0); }
  // alignment==0 -> use effective alignment from config()
  void* raw_alloc(std::size_t nbytes, std::size_t alignment);
  // Overload returning exact reserved bytes for precise stats in Storage deleter.
  void* raw_alloc(std::size_t nbytes, std::size_t alignment, std::size_t& reserved_out);
  // noexcept free with stats update; takes requested nbytes for accounting
  void  raw_delete(void* p, std::size_t nbytes) noexcept;
  // Exact reserved variant for callers that tracked reserved at allocation time.
  void  raw_delete_exact(void* p, std::size_t requested, std::size_t reserved) noexcept;

  DeviceStats getDeviceStats() const;
  void resetPeakStats() noexcept;
  void emptyCache() noexcept; // no-op when caching disabled

  // Caching-only ownership queries. Return false/nullptr when caching disabled.
  bool owns(const void* p) const noexcept;
  void* getBaseAllocation(void* p, std::size_t* seg_bytes) const noexcept;

  Config config() const;

 private:
  Allocator(); // parses VBT_CPU_ALLOC_CONF once

  // Helpers
  static std::size_t normalize_pow2_(std::size_t v) noexcept;
  std::size_t effective_alignment_(std::size_t override_alignment) const;

  // OS-specific helpers
  static void* os_alloc_(std::size_t nbytes, std::size_t alignment, std::size_t& reserved_out);
  static void  os_free_(void* p) noexcept;

  // ---- Caching implementation details ----
  struct Block {
    void*  ptr{nullptr};
    std::size_t size{0};       // piece size (aligned)
    std::size_t requested{0};  // last requested user bytes when allocated
    Block* prev{nullptr};      // neighbor within same OS segment
    Block* next{nullptr};
    bool   allocated{false};
    bool   segment_head{false};
    std::size_t segment_size{0};        // valid iff segment_head==true
    Block* segment_head_ptr{nullptr};   // stable head pointer for this segment
  };
  struct CmpSizeAddr {
    bool operator()(const Block* a, const Block* b) const noexcept {
      if (a->size != b->size) return a->size < b->size;
      return a->ptr < b->ptr;
    }
  };

  // Compute size classes vector based on cfg_ and effective alignment. Filled at construction.
  void build_size_classes_();
  std::size_t idx_for_size_(std::size_t need) const noexcept; // minimal i s.t. class_[i] >= need; npos if oversize
  static constexpr std::size_t npos_ = static_cast<std::size_t>(-1);

  // Bitset helpers for non-empty bins (optional fast path)
  void set_bin_bit_(std::size_t k) noexcept;
  void clear_bin_bit_(std::size_t k) noexcept;
  bool test_bin_bit_(std::size_t k) const noexcept;
  std::size_t find_next_set_(std::size_t from) const noexcept; // returns npos_ if none

  // Insert a free block into global containers (assumes under mu_). Attempts local coalescing around b.
  void insert_free_global_(Block* b) noexcept;
  // Remove a free block from global containers (assumes under mu_)
  void erase_free_global_(Block* b) noexcept;
  // Coalesce around free block b (assumes under mu_) within its segment
  void coalesce_around_(Block* b) noexcept;
  // Evict whole segments to honor cap or emptyCache
  void evict_under_cap_(std::uint64_t target_cached_bytes, std::unique_lock<std::mutex>& lk) noexcept; // assumes lk owns mu_, frees OS off-lock

  // Allocate a new OS segment of size sz_os and create head block (allocated=true). reserved_out updated; returns head pointer
  void* os_alloc_segment_(std::size_t sz_os, std::size_t requested, std::size_t& reserved_out);

  // State guarded by mu_
  mutable std::mutex mu_{};
  DeviceStats stats_{};
  Config cfg_{}; // immutable after construction
  bool caching_enabled_{false};

  // Size classes
  std::vector<std::size_t> classes_{};        // class sizes
  std::vector<std::set<Block*, CmpSizeAddr>> free_bins_{}; // per-bin free sets
  std::set<Block*, CmpSizeAddr> oversize_free_{};          // free oversize pieces
  std::unordered_map<void*, Block*> by_ptr_{}; // base ptr -> Block*
  std::unordered_map<void*, Block*> by_end_{}; // end addr -> Block* (only free-global)
  std::vector<std::uint64_t> non_empty_bins_{}; // bitmap words
  std::uint64_t cached_global_bytes_{0};

  // Non-caching reserved-by-pointer map for precise stats (used only when caching disabled)
  std::unordered_map<void*, std::size_t> nonc_reserved_by_ptr_{};

  // TLS cache (per-thread)
  struct TlsCache {
    std::vector<std::vector<Block*>> bins; // LIFO per bin
    std::size_t tls_cached_bytes{0};
  };
  static thread_local TlsCache tls_;
};
// Pinned host allocator facade (M_Ext.3)
using StreamId = std::uint64_t; // must fit vbt::cuda::Stream::id()

class PinnedHostAllocator final {
 public:
  static PinnedHostAllocator& get();

  // alignment==0 -> use vbt::cpu::Allocator::get().config().alignment_bytes
  void* raw_alloc(std::size_t nbytes, std::size_t alignment = 0);
  void  raw_delete(void* p) noexcept;
  void  record_stream(void* p, const vbt::cuda::Stream& s) noexcept;
  bool  maybe_owns(const void* p) const noexcept;

  DeviceStats getDeviceStats() const;
  void resetPeakStats() noexcept;
  void emptyCache() noexcept;

 private:
  PinnedHostAllocator() = default;
};

}} // namespace vbt::cpu
