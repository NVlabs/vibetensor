// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cpu/allocator.h"
#include "vbt/cuda/stream.h"

#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <atomic>

#include <type_traits>
#include <unordered_map>
#include "vbt/logging/logging.h"
#include <unordered_set>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// ASAN detection and interface
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define VBT_HAS_ASAN 1
#  endif
#endif
#if !defined(VBT_HAS_ASAN) && defined(__SANITIZE_ADDRESS__)
#  define VBT_HAS_ASAN 1
#endif
#if defined(VBT_HAS_ASAN)
#  if defined(__has_include)
#    if __has_include(<sanitizer/asan_interface.h>)
#      include <sanitizer/asan_interface.h>
#    else
#      undef VBT_HAS_ASAN
#    endif
#  else
#    include <sanitizer/asan_interface.h>
#  endif
#endif

namespace vbt { namespace cpu {
namespace {
// Helpers (local to this TU)
static inline unsigned char* u8(void* p) noexcept { return reinterpret_cast<unsigned char*>(p); }

static std::size_t normalize_pow2(std::size_t v) noexcept {
  if (v == 0) return 1;
  v -= 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFull
  v |= v >> 32;
#endif
  return v + 1;
}

static std::size_t effective_alignment(std::size_t override_alignment) {
  std::size_t a = override_alignment ? override_alignment : Allocator::get().config().alignment_bytes;
  const std::size_t min_a = alignof(std::max_align_t);
  if (override_alignment) {
    auto is_pow2 = [](std::size_t v) noexcept { return v && ((v & (v - 1)) == 0); };
    if (!is_pow2(a) || a < min_a) {
      throw std::invalid_argument("PinnedHostAllocator::raw_alloc: alignment override must be power-of-two and >= alignof(std::max_align_t)");
    }
  }
  auto norm_pow2 = [](std::size_t v) noexcept {
    if (v == 0) return std::size_t{1};
    v -= 1; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; if constexpr (sizeof(std::size_t) > 4) { v |= v >> 32; } return v + 1;
  };
  if (a < min_a) a = norm_pow2(min_a);
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
  const std::size_t ws = sizeof(void*);
  if (a < ws) a = ws;
#endif
#if defined(VBT_REQUIRE_DLPACK_ALIGNMENT) && VBT_REQUIRE_DLPACK_ALIGNMENT
  if (a < Allocator::get().config().dlpack_align_min_bytes) a = norm_pow2(Allocator::get().config().dlpack_align_min_bytes);
#else
  if (Allocator::get().config().enforce_dlpack_alignment && a < Allocator::get().config().dlpack_align_min_bytes) a = norm_pow2(Allocator::get().config().dlpack_align_min_bytes);
#endif
  return a;
}

static void* os_alloc_reserved(std::size_t nbytes, std::size_t alignment, std::size_t& reserved_out) {
  reserved_out = 0;
  if (nbytes == 0) return nullptr;
  void* p = nullptr;
#if defined(_WIN32)
  p = _aligned_malloc(nbytes, alignment);
  if (!p) throw std::bad_alloc();
  reserved_out = nbytes; return p;
#else
  int rc = posix_memalign(&p, alignment, nbytes);
  if (rc == 0 && p != nullptr) { reserved_out = nbytes; return p; }
  // aligned_alloc requires multiple of alignment; harden against overflow
  if (alignment == 0) throw std::bad_alloc();
  if (nbytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc();
  std::size_t rounded = ((nbytes + alignment - 1) / alignment) * alignment;
  if (rounded < nbytes) throw std::bad_alloc();
  p = std::aligned_alloc(alignment, rounded);
  if (!p) throw std::bad_alloc();
  reserved_out = rounded; return p;
#endif
}

static void os_free(void* p) noexcept {
  if (!p) return;
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

// Bloom-like prefilter parameters
static constexpr std::size_t kBloomBits  = 256;
static constexpr std::size_t kBloomWords = kBloomBits / 64;
static constexpr std::size_t kHashes     = 4;

static inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x = x ^ (x >> 31);
  return x;
}

inline void asan_poison(void* p, std::size_t n) noexcept {
#if defined(VBT_HAS_ASAN)
  if (p && n) ASAN_POISON_MEMORY_REGION(p, n);
#else
  (void)p; (void)n;
#endif
}
inline void asan_unpoison(void* p, std::size_t n) noexcept {
#if defined(VBT_HAS_ASAN)
  if (p && n) ASAN_UNPOISON_MEMORY_REGION(p, n);
#else
  (void)p; (void)n;
#endif
}

#if defined(__linux__)
static inline void* align_down_ptr(void* p, std::size_t a) noexcept {
  std::uintptr_t v = reinterpret_cast<std::uintptr_t>(p);
  return reinterpret_cast<void*>(v & ~(static_cast<std::uintptr_t>(a) - 1u));
}
static inline std::size_t align_up_len(void* base, std::size_t len, std::size_t a) noexcept {
  std::uintptr_t b = reinterpret_cast<std::uintptr_t>(base);
  std::uintptr_t e = b + len;
  std::uintptr_t ea = (e + a - 1) / a * a;
  return static_cast<std::size_t>(ea - b);
}
#endif

} // namespace

// PinnedHostAllocator state (singleton)
struct PinnedState {
  struct Block {
    void*       base_ptr{nullptr};
    void*       user_ptr{nullptr};
    std::size_t requested_size{0};
    std::size_t reserved_size{0};
    std::size_t align{0};
    std::unordered_set<StreamId> stream_uses; // dedup stream ids
    bool        allocated{false};
    // M_Ext.4 metadata
    std::size_t rz_rounded{0};
    bool        mlocked{false};
    bool        hugepage_advised{false};
    std::size_t base_size{0};
  };

  std::mutex mu;
  DeviceStats stats{};
  std::unordered_map<void*, Block> by_user_ptr; // base user ptr → Block
  std::atomic<std::uint64_t> bloom[kBloomWords]{};     // zero-initialized
  std::size_t live_count{0};

  void bloom_add(void* p) noexcept {
    std::uintptr_t v = reinterpret_cast<std::uintptr_t>(p);
    std::uint64_t h = splitmix64(static_cast<std::uint64_t>(v));
    const std::uint64_t seeds[kHashes] = {
      0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull, 0x94d049bb133111ebull, 0x27d4eb2f165667c5ull
    };
    for (std::size_t i=0; i<kHashes; ++i) {
      std::uint64_t hh = splitmix64(h + seeds[i]);
      std::size_t idx = static_cast<std::size_t>(hh) & (kBloomBits - 1);
      std::size_t w = idx >> 6; std::size_t b = idx & 63;
      bloom[w].fetch_or(1ull << b, std::memory_order_release);
    }
  }

  bool bloom_maybe(void* p) const noexcept {
    std::uintptr_t v = reinterpret_cast<std::uintptr_t>(p);
    std::uint64_t h = splitmix64(static_cast<std::uint64_t>(v));
    const std::uint64_t seeds[kHashes] = {
      0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull, 0x94d049bb133111ebull, 0x27d4eb2f165667c5ull
    };
    for (std::size_t i=0; i<kHashes; ++i) {
      std::uint64_t hh = splitmix64(h + seeds[i]);
      std::size_t idx = static_cast<std::size_t>(hh) & (kBloomBits - 1);
      std::size_t w = idx >> 6; std::size_t b = idx & 63;
      if (((bloom[w].load(std::memory_order_acquire) >> b) & 1ull) == 0ull) return false;
    }
    return true;
  }

  void bloom_clear_all_relaxed() noexcept {
    for (std::size_t i=0; i<kBloomWords; ++i) bloom[i].store(0, std::memory_order_relaxed);
  }
};

static PinnedState& pinned_state() {
  static PinnedState* st = nullptr;
  static std::once_flag once;
  std::call_once(once, [](){ st = new PinnedState(); });
  return *st;
}

PinnedHostAllocator& PinnedHostAllocator::get() {
  static PinnedHostAllocator* inst = nullptr;
  static std::once_flag once;
  std::call_once(once, [](){ inst = new PinnedHostAllocator(); });
  return *inst;
}

void* PinnedHostAllocator::raw_alloc(std::size_t nbytes, std::size_t alignment) {
  if (nbytes == 0) return nullptr;
  const std::size_t a = effective_alignment(alignment);
  // ASAN redzones: default 64 bytes rounded to alignment when ASAN present
  constexpr std::size_t kAsanRz = 64;
  std::size_t rz_rounded = 0;
#if defined(VBT_HAS_ASAN)
  // align up to multiple of a
  rz_rounded = (kAsanRz + a - 1) / a * a;
#endif
  if (rz_rounded > 0) {
    if (nbytes > std::numeric_limits<std::size_t>::max() - 2 * rz_rounded) throw std::bad_alloc();
  }
  std::size_t base_size = nbytes + 2 * rz_rounded;
  std::size_t reserved = 0;
  void* base = os_alloc_reserved(base_size, a, reserved);
  void* user = rz_rounded ? static_cast<void*>(u8(base) + rz_rounded) : base;

  bool did_mlock = false;
  bool did_madv = false;
#if defined(__linux__)
  const auto cfg = Allocator::get().config();
  if (cfg.madvise_hugepage) {
    long pg = ::getpagesize(); if (pg <= 0) pg = 4096;
    void* adv_ptr = align_down_ptr(base, static_cast<std::size_t>(pg));
    std::size_t adv_len = align_up_len(adv_ptr, reserved + (reinterpret_cast<std::uintptr_t>(base) - reinterpret_cast<std::uintptr_t>(adv_ptr)), static_cast<std::size_t>(pg));
    int rc = ::madvise(adv_ptr, adv_len, MADV_HUGEPAGE);
    did_madv = (rc == 0);
  }
  if (cfg.mlock) {
    int rc = ::mlock(user, nbytes);
    did_mlock = (rc == 0);
  }
#endif

  // ASAN poison redzones (if any)
  if (rz_rounded > 0) {
    asan_poison(base, rz_rounded);
    asan_poison(static_cast<void*>(u8(user) + nbytes), rz_rounded);
  }

  // RAII cleanup if publication fails
  struct CleanupGuard {
    void* base; void* user; std::size_t n; std::size_t bsz; std::size_t rz; bool mlocked;
    bool committed{false};
    ~CleanupGuard() {
      if (!committed) {
        if (rz > 0) asan_unpoison(base, bsz);
#if defined(__linux__)
        if (mlocked) (void)::munlock(user, n);
#endif
        os_free(base);
      }
    }
  } guard{base, user, nbytes, base_size, rz_rounded, did_mlock, false};

  auto& st = pinned_state();
  {
    std::lock_guard<std::mutex> lg(st.mu);
    auto ins = st.by_user_ptr.emplace(user, PinnedState::Block{base,user,nbytes,/*reserved_size*/reserved,a,{/*empty*/},true,rz_rounded,did_mlock,did_madv,base_size});
    (void)ins;
    st.stats.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
    if (st.stats.allocated_bytes_all_current > st.stats.max_allocated_bytes_all) st.stats.max_allocated_bytes_all = st.stats.allocated_bytes_all_current;
    st.stats.reserved_bytes_all_current += static_cast<std::uint64_t>(reserved);
    if (st.stats.reserved_bytes_all_current > st.stats.max_reserved_bytes_all) st.stats.max_reserved_bytes_all = st.stats.reserved_bytes_all_current;
    ++st.live_count;
    st.bloom_add(user);
    guard.committed = true;
  }
  return user;
}

void PinnedHostAllocator::raw_delete(void* p) noexcept {
  if (!p) return;
  auto& st = pinned_state();
  void* base = nullptr; void* user=nullptr; std::size_t req=0, base_size=0, rz=0; bool mlocked=false; std::size_t reserved=0;
  bool erased = false;
  {
    std::lock_guard<std::mutex> lg(st.mu);
    auto it = st.by_user_ptr.find(p);
    if (it == st.by_user_ptr.end()) return; // strict no-op on unknown pointers
    base = it->second.base_ptr; user = it->second.user_ptr; req = it->second.requested_size; base_size = it->second.base_size; rz = it->second.rz_rounded; mlocked = it->second.mlocked; reserved = it->second.reserved_size;
    st.by_user_ptr.erase(it);
    if (st.stats.allocated_bytes_all_current >= req) st.stats.allocated_bytes_all_current -= static_cast<std::uint64_t>(req);
    else st.stats.allocated_bytes_all_current = 0;
    if (st.stats.reserved_bytes_all_current >= reserved) st.stats.reserved_bytes_all_current -= static_cast<std::uint64_t>(reserved);
    else st.stats.reserved_bytes_all_current = 0;
    erased = true;
    if (st.live_count > 0) {
      --st.live_count;
      if (st.live_count == 0) st.bloom_clear_all_relaxed();
    }
  }
  // Unpoison full region if ASAN redzones used
  if (rz > 0) {
    asan_unpoison(base, base_size);
  }
#if defined(__linux__)
  if (mlocked) {
    (void)::munlock(user, req);
  }
#endif
  if (erased) os_free(base);
}

void PinnedHostAllocator::record_stream(void* p, const vbt::cuda::Stream& s) noexcept {
  if (!p) return;
  if (s.device_index() < 0) return; // ignore sentinel/current-only
  // Once-only runtime warning: metadata-only semantics, no fences
  static std::atomic_flag warned = ATOMIC_FLAG_INIT;
  if (!warned.test_and_set()) {
    VBT_LOG(WARNING) << "PinnedHostAllocator::record_stream is metadata-only; no stream/event fencing is performed";
  }
  auto& st = pinned_state();
  if (!st.bloom_maybe(p)) return; // fast negative
  std::lock_guard<std::mutex> lg(st.mu);
  auto it = st.by_user_ptr.find(p);
  if (it == st.by_user_ptr.end()) return;
  try {
    if (it->second.stream_uses.empty()) it->second.stream_uses.reserve(4);
    it->second.stream_uses.insert(static_cast<StreamId>(s.id()));
  } catch (...) {
    // swallow in noexcept path
  }
}

bool PinnedHostAllocator::maybe_owns(const void* p) const noexcept {
  return pinned_state().bloom_maybe(const_cast<void*>(p));
}

DeviceStats PinnedHostAllocator::getDeviceStats() const {
  auto& st = pinned_state();
  std::lock_guard<std::mutex> lg(st.mu);
  return st.stats;
}

void PinnedHostAllocator::resetPeakStats() noexcept {
  auto& st = pinned_state();
  std::lock_guard<std::mutex> lg(st.mu);
  st.stats.max_allocated_bytes_all = st.stats.allocated_bytes_all_current;
  st.stats.max_reserved_bytes_all = st.stats.reserved_bytes_all_current;
}

void PinnedHostAllocator::emptyCache() noexcept {
  // Facade has no cache; strict no-op
}

// Compile-time sanity: Stream::id() must fit in StreamId
static_assert(std::is_same_v<StreamId, std::uint64_t>, "StreamId must be uint64_t");
static_assert(sizeof(StreamId) >= sizeof(decltype(std::declval<vbt::cuda::Stream>().id())),
              "Stream::id() must fit in StreamId");

}} // namespace vbt::cpu
