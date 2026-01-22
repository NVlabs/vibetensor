// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cpu/allocator.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <string>
#include <utility>
#include <algorithm>
#include <limits>
#include <stdexcept>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#include <malloc.h>
#endif

namespace vbt { namespace cpu {
static inline unsigned char* u8(void* p) noexcept { return reinterpret_cast<unsigned char*>(p); }
static inline const unsigned char* u8c(const void* p) noexcept { return reinterpret_cast<const unsigned char*>(p); }

thread_local Allocator::TlsCache Allocator::tls_{};

std::size_t Allocator::normalize_pow2_(std::size_t v) noexcept {
  if (v == 0) return 1;
  // ceil to next power of two
  v -= 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFFull
  v |= v >> 32;
#endif
  v += 1;
  return v;
}

static Config parse_env_now_() {
  Config cfg;
  const char* env = std::getenv("VBT_CPU_ALLOC_CONF");
  if (!env || !*env) return cfg;
  auto is_space = [](char c){ return c==' '||c=='\t'||c=='\n' || c=='\r'; };
  std::string s(env);
  std::size_t i = 0;
  auto trim = [&](std::string& t){ std::size_t a=0; while (a<t.size() && is_space(t[a])) ++a; std::size_t b=t.size(); while (b>a && is_space(t[b-1])) --b; t = t.substr(a, b-a); };
  auto to_uint = [&](const std::string& t, std::size_t& out)->bool{
    if (t.empty()) return false; char* end=nullptr; errno=0; unsigned long long x = std::strtoull(t.c_str(), &end, 10);
    if (errno!=0 || (end && *end!='\0')) return false; out = static_cast<std::size_t>(x); return true; };
  auto to_bool = [&](const std::string& t, bool& out)->bool{
    std::string u=t; for (auto& c: u) c = (char)std::tolower(c);
    if (u=="1"||u=="true"||u=="yes") { out=true; return true; }
    if (u=="0"||u=="false"||u=="no") { out=false; return true; }
    return false; };
  auto lower_copy = [&](std::string v){ for (auto& c: v) c = (char)std::tolower(c); return v; };
  while (i < s.size()) {
    while (i < s.size() && (is_space(s[i]) || s[i]==',')) ++i; if (i>=s.size()) break;
    std::size_t k0=i; while (i<s.size() && s[i] != '=' && s[i] != ',') ++i; if (i>=s.size()|| s[i] != '=') break; std::string key = s.substr(k0, i-k0); ++i;
    std::size_t v0=i; while (i<s.size() && s[i] != ',') ++i; std::string val = s.substr(v0, i-v0);
    trim(key); trim(val);
    if (key == "alignment_bytes") { std::size_t v=0; if (to_uint(val, v)) cfg.alignment_bytes = v; }
    else if (key == "enforce_dlpack_alignment") { bool b=false; if (to_bool(val, b)) cfg.enforce_dlpack_alignment = b; }
    else if (key == "dlpack_align_min_bytes") { std::size_t v=0; if (to_uint(val, v)) cfg.dlpack_align_min_bytes = v; }
    else if (key == "enable_caching") { bool b=false; if (to_bool(val,b)) cfg.enable_caching = b; }
    else if (key == "min_bin_bytes") { std::size_t v=0; if (to_uint(val, v)) cfg.min_bin_bytes = v; }
    else if (key == "max_bin_bytes") { std::size_t v=0; if (to_uint(val, v)) cfg.max_bin_bytes = v; }
    else if (key == "bin_growth") { std::size_t v=0; if (to_uint(val, v)) cfg.bin_growth = v; }
    else if (key == "tls_cap_bytes") { std::size_t v=0; if (to_uint(val, v)) cfg.tls_cap_bytes = v; }
    else if (key == "global_cap_bytes") { std::size_t v=0; if (to_uint(val, v)) cfg.global_cap_bytes = v; }
    else if (key == "oversize_cache") { bool b=false; if (to_bool(val, b)) cfg.oversize_cache = b; }
    else if (key == "poison_on_free") { bool b=false; if (to_bool(val, b)) cfg.poison_on_free = b; }
    else if (key == "mlock") { bool b=false; if (to_bool(val, b)) cfg.mlock = b; }
    else if (key == "madvise_hugepage") { bool b=false; if (to_bool(val, b)) cfg.madvise_hugepage = b; }
    else if (key == "numa_policy") {
      std::string t = lower_copy(val);
      trim(t); // only trim ends; internal whitespace invalidates bind: form
      if (t == "default" || t == "local" || t == "interleave") {
        cfg.numa_policy = t;
      } else if (t.rfind("bind:", 0) == 0) {
        std::string suf = t.substr(5);
        bool ok=true; if (suf.empty()) ok=false; else {
          for (char c: suf) { if (c < '0' || c > '9') { ok=false; break; } }
        }
        if (ok) {
          // strip leading zeros
          std::size_t j=0; while (j+1 < suf.size() && suf[j] == '0') ++j; std::string canon = suf.substr(j);
          // limit to 32-bit unsigned
          if (canon.size() > 10) ok=false;
          else if (canon.size() == 10 && canon > std::string("4294967295")) ok=false;
          if (ok) cfg.numa_policy = std::string("bind:") + canon; else cfg.numa_policy = "default";
        } else {
          cfg.numa_policy = "default";
        }
      } else {
        cfg.numa_policy = "default";
      }
    }
  }
  return cfg;
}

Allocator::Allocator() : mu_(), stats_(), cfg_() {
  // Parse env once; normalize
  cfg_ = parse_env_now_();
  // normalize alignment: ceil pow2, >= alignof(max_align_t)
  std::size_t a = normalize_pow2_(cfg_.alignment_bytes);
  const std::size_t min_a = alignof(std::max_align_t);
  if (a < min_a) a = normalize_pow2_(min_a);
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
  const std::size_t ws = sizeof(void*);
  if (a < ws) a = normalize_pow2_(ws);
  // ensure multiple of word size
  if (a % ws != 0) {
    std::size_t k = (a + ws - 1) / ws; a = k * ws;
  }
#endif
  cfg_.alignment_bytes = a;
  if (cfg_.enforce_dlpack_alignment) {
    std::size_t d = normalize_pow2_(cfg_.dlpack_align_min_bytes);
    if (d < min_a) d = normalize_pow2_(min_a);
    if (d > cfg_.alignment_bytes) cfg_.alignment_bytes = d;
  }
#if defined(VBT_REQUIRE_DLPACK_ALIGNMENT) && VBT_REQUIRE_DLPACK_ALIGNMENT
  // If build requires alignment, enforce at least dlpack minimum
  std::size_t d = normalize_pow2_(cfg_.dlpack_align_min_bytes);
  const std::size_t min2 = alignof(std::max_align_t);
  if (d < min2) d = normalize_pow2_(min2);
  if (d > cfg_.alignment_bytes) cfg_.alignment_bytes = d;
#endif

  // normalize caching params
  if (cfg_.bin_growth < 2) cfg_.bin_growth = 2;
  if (cfg_.max_bin_bytes < cfg_.min_bin_bytes) cfg_.max_bin_bytes = cfg_.min_bin_bytes;

  caching_enabled_ = cfg_.enable_caching;
  if (caching_enabled_) {
    build_size_classes_();
    free_bins_.assign(classes_.size(), {});
    // Initialize bitmap words
    std::size_t words = (classes_.size() + 63) / 64;
    non_empty_bins_.assign(words, 0);
  }
}

Allocator& Allocator::get() {
  static Allocator* inst = nullptr;
  static std::once_flag once;
  std::call_once(once, [](){ inst = new Allocator(); });
  return *inst;
}

std::size_t Allocator::effective_alignment_(std::size_t override_alignment) const {
  std::size_t a = override_alignment ? override_alignment : cfg_.alignment_bytes;
  const std::size_t min_a = alignof(std::max_align_t);
  if (override_alignment) {
    auto is_pow2 = [](std::size_t v){ return v && ((v & (v-1)) == 0); };
    if (!is_pow2(override_alignment) || override_alignment < min_a) {
      throw std::invalid_argument("Allocator::raw_alloc: alignment override must be a power-of-two and >= alignof(std::max_align_t)");
    }
  }
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
  const std::size_t ws = sizeof(void*);
  if (a < ws) a = normalize_pow2_(ws);
  if (a % ws != 0) { std::size_t k = (a + ws - 1) / ws; a = k * ws; }
#endif
#if defined(VBT_REQUIRE_DLPACK_ALIGNMENT) && VBT_REQUIRE_DLPACK_ALIGNMENT
  if (a < cfg_.dlpack_align_min_bytes) a = normalize_pow2_(cfg_.dlpack_align_min_bytes);
#else
  if (cfg_.enforce_dlpack_alignment && a < cfg_.dlpack_align_min_bytes) a = normalize_pow2_(cfg_.dlpack_align_min_bytes);
#endif
  // Ensure minimum alignment for non-override path
  if (!override_alignment && a < min_a) a = normalize_pow2_(min_a);
  return a;
}

void* Allocator::os_alloc_(std::size_t nbytes, std::size_t alignment, std::size_t& reserved_out) {
  reserved_out = 0;
  if (nbytes == 0) return nullptr;
  void* p = nullptr;
#if defined(_WIN32)
  p = _aligned_malloc(nbytes, alignment);
  if (!p) throw std::bad_alloc();
  reserved_out = nbytes;
  return p;
#else
  // Prefer posix_memalign when available
  int rc = posix_memalign(&p, alignment, nbytes);
  if (rc == 0 && p != nullptr) { reserved_out = nbytes; return p; }
  // Fallback: aligned_alloc requires size multiple of alignment
  if (alignment == 0) throw std::bad_alloc();
  if (nbytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) throw std::bad_alloc();
  std::size_t rounded = ((nbytes + alignment - 1) / alignment) * alignment;
  if (rounded < nbytes) throw std::bad_alloc();
  p = std::aligned_alloc(alignment, rounded);
  if (!p) throw std::bad_alloc();
  reserved_out = rounded; return p;
#endif
}

void Allocator::os_free_(void* p) noexcept {
  if (!p) return;
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

void Allocator::build_size_classes_() {
  classes_.clear();
  std::size_t a = cfg_.alignment_bytes;
  auto align_up = [&](std::size_t x){ return (x + a - 1) / a * a; };
  std::size_t cur = align_up(cfg_.min_bin_bytes);
  if (cur == 0) cur = a; // clamp to at least one alignment quantum
  const std::size_t maxb = cfg_.max_bin_bytes ? cfg_.max_bin_bytes : a;
  while (cur <= maxb) {
    classes_.push_back(cur);
    // guard overflow on multiplication and addition in align_up
    if (cur > (std::numeric_limits<std::size_t>::max() / cfg_.bin_growth)) break; // mul overflow
    std::size_t next = cur * cfg_.bin_growth;
    if (next <= cur) break; // overflow or no progress
    if (a > 0 && next > (std::numeric_limits<std::size_t>::max() - (a - 1))) break; // align_up would overflow
    cur = align_up(next);
  }
  if (classes_.empty()) {
    classes_.push_back(std::max(align_up(maxb), a));
  }
}

std::size_t Allocator::idx_for_size_(std::size_t need) const noexcept {
  if (classes_.empty()) return npos_;
  if (need > classes_.back()) return npos_;
  auto it = std::lower_bound(classes_.begin(), classes_.end(), need);
  return static_cast<std::size_t>(it - classes_.begin());
}

void Allocator::set_bin_bit_(std::size_t k) noexcept {
  if (non_empty_bins_.empty()) return;
  std::size_t w = k / 64, o = k % 64;
  if (w < non_empty_bins_.size()) non_empty_bins_[w] |= (1ull << o);
}
void Allocator::clear_bin_bit_(std::size_t k) noexcept {
  if (non_empty_bins_.empty()) return;
  std::size_t w = k / 64, o = k % 64;
  if (w < non_empty_bins_.size()) non_empty_bins_[w] &= ~(1ull << o);
}
bool Allocator::test_bin_bit_(std::size_t k) const noexcept {
  if (non_empty_bins_.empty()) return false;
  std::size_t w = k / 64, o = k % 64;
  if (w < non_empty_bins_.size()) return (non_empty_bins_[w] >> o) & 1ull;
  return false;
}
std::size_t Allocator::find_next_set_(std::size_t from) const noexcept {
  if (non_empty_bins_.empty()) return npos_;
  std::size_t w = from / 64, o = from % 64;
  while (w < non_empty_bins_.size()) {
    std::uint64_t word = non_empty_bins_[w] & (~0ull << o);
    if (word) {
#if defined(_MSC_VER)
      unsigned long idx = 0;
      unsigned long long w64 = static_cast<unsigned long long>(word);
      _BitScanForward64(&idx, w64);
      return w * 64 + static_cast<std::size_t>(idx);
#else
      return w * 64 + static_cast<std::size_t>(__builtin_ctzll(word));
#endif
    }
    ++w; o = 0;
  }
  return npos_;
}

void* Allocator::os_alloc_segment_(std::size_t sz_os, std::size_t requested, std::size_t& reserved_out) {
  if (sz_os == 0) throw std::bad_alloc();
  const std::size_t a = effective_alignment_(0);
  void* p = os_alloc_(sz_os, a, reserved_out);
  // Create head block (allocated)
  Block* head = nullptr;
  try {
    head = new Block();
  } catch (...) {
    os_free_(p);
    throw;
  }
  head->ptr = p;
  head->size = reserved_out; // piece size equals segment size initially
  head->requested = requested;
  head->prev = nullptr;
  head->next = nullptr;
  head->allocated = true;
  head->segment_head = true;
  head->segment_size = reserved_out;
  head->segment_head_ptr = head;
  {
    std::lock_guard<std::mutex> lg(mu_);
    by_ptr_[p] = head;
    stats_.reserved_bytes_all_current += static_cast<std::uint64_t>(reserved_out);
    if (stats_.reserved_bytes_all_current > stats_.max_reserved_bytes_all) stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
  }
  return p;
}

void Allocator::insert_free_global_(Block* b) noexcept {
  // assumes mu_ held
  // Attempt coalescing with free neighbors before insertion
  auto merged = true;
  while (merged) {
    merged = false;
    // right neighbor: base at end(b)
    auto r_it = by_ptr_.find((void*)(u8(b->ptr) + b->size));
    if (r_it != by_ptr_.end()) {
      Block* r = r_it->second;
      if (!r->allocated && r->segment_head_ptr == b->segment_head_ptr) {
        // Only coalesce with right neighbor if it is present in GLOBAL free containers (not TLS)
        auto rendp = (void*)(u8(r->ptr) + r->size);
        auto in_global = by_end_.find(rendp);
        if (in_global != by_end_.end() && in_global->second == r) {
          // remove r from free containers and maps
          erase_free_global_(r);
          // unlink r
          b->next = r->next;
          if (r->next) r->next->prev = b;
          // grow b
          b->size += r->size;
          // drop r metadata
          by_ptr_.erase(r->ptr);
          delete r;
          merged = true;
        }
      }
    }
    // left neighbor: free whose end equals b->ptr
    auto l_it = by_end_.find(b->ptr);
    if (l_it != by_end_.end()) {
      Block* l = l_it->second;
      if (!l->allocated && l->segment_head_ptr == b->segment_head_ptr) {
        // remove l from free containers and maps
        erase_free_global_(l);
        // link
        l->next = b->next;
        if (b->next) b->next->prev = l;
        // grow l to absorb b
        l->size += b->size;
        // replace b with l as survivor
        by_ptr_.erase(b->ptr);
        delete b;
        b = l;
        merged = true;
      }
    }
  }
  // Finally insert merged b into containers and by_end_
  auto endp = (void*)(u8(b->ptr) + b->size);
  by_end_[endp] = b;
  if (!classes_.empty() && b->size <= classes_.back()) {
    std::size_t k = idx_for_size_(b->size);
    free_bins_[k].insert(b);
    set_bin_bit_(k);
  } else {
    oversize_free_.insert(b);
  }
  cached_global_bytes_ += static_cast<std::uint64_t>(b->size);
}

void Allocator::erase_free_global_(Block* b) noexcept {
  // assumes mu_ held; remove from container and maps, conditionally adjust bytes
  auto endp = (void*)(u8(b->ptr) + b->size);
  bool removed = false;
  if (!classes_.empty() && b->size <= classes_.back()) {
    std::size_t k = idx_for_size_(b->size);
    auto& bin = free_bins_[k];
    auto it = bin.find(b);
    if (it != bin.end()) {
      bin.erase(it);
      removed = true;
    }
    if (bin.empty()) clear_bin_bit_(k);
  } else {
    auto it = oversize_free_.find(b);
    if (it != oversize_free_.end()) {
      oversize_free_.erase(it);
      removed = true;
    }
  }
  if (removed) {
    auto be = by_end_.find(endp);
    if (be != by_end_.end() && be->second == b) by_end_.erase(be);
    if (cached_global_bytes_ >= b->size) cached_global_bytes_ -= static_cast<std::uint64_t>(b->size);
    else cached_global_bytes_ = 0;
  }
}

// Coalescing helper is now integrated into insert_free_global_; keep a no-op placeholder for future direct-global-free paths.
void Allocator::coalesce_around_(Block* /*b*/) noexcept {
  // no-op
}

void Allocator::evict_under_cap_(std::uint64_t target_cached_bytes, std::unique_lock<std::mutex>& lk) noexcept {
  // lk owns mu_ on entry
  while (cached_global_bytes_ > target_cached_bytes) {
    Block* victim = nullptr;
    // prefer oversize heads
    for (auto it = oversize_free_.begin(); it != oversize_free_.end(); ++it) {
      Block* b = *it;
      if (b->segment_head && b->prev == nullptr && b->next == nullptr && !b->allocated) {
        victim = b; oversize_free_.erase(it); break;
      }
    }
    if (!victim) {
      for (std::size_t k=0; k<free_bins_.size() && !victim; ++k) {
        auto& bin = free_bins_[k];
        for (auto it = bin.begin(); it != bin.end(); ++it) {
          Block* b = *it;
          if (b->segment_head && b->prev == nullptr && b->next == nullptr && !b->allocated) {
            victim = b; bin.erase(it); if (bin.empty()) clear_bin_bit_(k); break;
          }
        }
      }
    }
    if (!victim) break; // no evictable segments
    // remove maps and cached bytes for victim
    auto endp = (void*)(u8(victim->ptr) + victim->size);
    by_end_.erase(endp);
    by_ptr_.erase(victim->ptr);
    if (cached_global_bytes_ >= victim->size) cached_global_bytes_ -= static_cast<std::uint64_t>(victim->size);
    else cached_global_bytes_ = 0;
    // free OS outside lock per victim, then update stats
    lk.unlock();
    os_free_(victim->ptr);
    lk.lock();
    if (stats_.reserved_bytes_all_current >= victim->segment_size) stats_.reserved_bytes_all_current -= static_cast<std::uint64_t>(victim->segment_size);
    else stats_.reserved_bytes_all_current = 0;
    delete victim;
  }
}

void* Allocator::raw_alloc(std::size_t nbytes, std::size_t alignment, std::size_t& reserved_out) {
  reserved_out = 0;
  if (nbytes == 0) return nullptr;
  if (!caching_enabled_) {
    const std::size_t a = effective_alignment_(alignment);
    void* p = os_alloc_(nbytes, a, reserved_out);
    {
      std::lock_guard<std::mutex> lg(mu_);
      stats_.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
      if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
      stats_.reserved_bytes_all_current += static_cast<std::uint64_t>(reserved_out);
      if (stats_.reserved_bytes_all_current > stats_.max_reserved_bytes_all) stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
      // track reserved size for non-caching path accounting
      nonc_reserved_by_ptr_[p] = reserved_out;
    }
    return p;
  }

  // caching path
  const std::size_t a = effective_alignment_(alignment);
  auto align_up = [&](std::size_t x){ return (x + a - 1) / a * a; };
  std::size_t need = align_up(nbytes);

  // Oversize path
  if (classes_.empty() || need > classes_.back()) {
    // try reuse from oversize
    {
      std::lock_guard<std::mutex> lg(mu_);
      // find first block with size >= need
      Block key; key.size = need; key.ptr = nullptr;
      auto it = oversize_free_.lower_bound(&key);
      if (it != oversize_free_.end()) {
        Block* b = *it;
        oversize_free_.erase(it);
        auto endp = (void*)(u8(b->ptr) + b->size);
        by_end_.erase(endp);
        if (cached_global_bytes_ >= b->size) cached_global_bytes_ -= static_cast<std::uint64_t>(b->size);
        else cached_global_bytes_ = 0;
        b->allocated = true;
        b->requested = nbytes;
        stats_.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
        if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
        return b->ptr;
      }
    }
    // allocate new OS segment of size need
    void* p = os_alloc_segment_(need, nbytes, reserved_out);
    {
      std::lock_guard<std::mutex> lg(mu_);
      stats_.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
      if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
    }
    return p;
  }

  // Bin-managed path
  std::size_t k = idx_for_size_(need);
  // TLS fast path
  if (tls_.bins.empty()) tls_.bins.resize(classes_.size());
  if (!tls_.bins[k].empty()) {
    Block* b = tls_.bins[k].back();
    tls_.bins[k].pop_back();
    if (tls_.tls_cached_bytes >= b->size) tls_.tls_cached_bytes -= b->size; else tls_.tls_cached_bytes = 0;
    {
      std::lock_guard<std::mutex> lg(mu_);
      b->allocated = true;
      b->requested = nbytes;
      stats_.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
      if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
    }
    return b->ptr;
  }

  // Global best-fit search across bins starting at k
  {
    std::lock_guard<std::mutex> lg(mu_);
    std::size_t j = find_next_set_(k);
    while (j != npos_) {
      auto& bin = free_bins_[j];
      // find lower_bound by need
      Block key; key.size = need; key.ptr = nullptr;
      auto it = bin.lower_bound(&key);
      if (it != bin.end()) {
        Block* b = *it;
        bin.erase(it);
        if (bin.empty()) clear_bin_bit_(j);
        auto endp = (void*)(u8(b->ptr) + b->size);
        by_end_.erase(endp);
        if (cached_global_bytes_ >= b->size) cached_global_bytes_ -= static_cast<std::uint64_t>(b->size);
        else cached_global_bytes_ = 0;
        // split if beneficial
        std::size_t rem = (b->size > need) ? (b->size - need) : 0;
        b->size = need;
        b->allocated = true;
        b->requested = nbytes;
        // create tail if enough remainder
        if (rem >= std::min(cfg_.min_bin_bytes, classes_[k]) && rem >= a) {
          Block* t = new Block();
          t->ptr = (void*)(u8(b->ptr) + need);
          t->size = rem;
          t->requested = 0;
          t->allocated = false;
          t->segment_head = false;
          t->segment_size = 0;
          t->segment_head_ptr = b->segment_head_ptr;
          // link
          t->prev = b;
          t->next = b->next;
          if (b->next) b->next->prev = t;
          b->next = t;
          by_ptr_[t->ptr] = t;
          insert_free_global_(t); // will coalesce around t
        }
        stats_.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
        if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
        return b->ptr;
      }
      j = find_next_set_(j+1);
    }
  }

  // OS allocate a full class size and split tail
  std::size_t sz_os = classes_[k];
  if (sz_os == 0) { sz_os = std::max(need, a); }
  void* p = os_alloc_segment_(sz_os, nbytes, reserved_out);
  // split off tail if any
  if (sz_os > need) {
    std::lock_guard<std::mutex> lg(mu_);
    Block* b = by_ptr_[p];
    // shrink head to requested aligned size
    b->size = need;
    // create tail covering the remainder
    Block* t = new Block();
    t->ptr = (void*)(u8(p) + need);
    t->size = sz_os - need;
    t->requested = 0;
    t->allocated = false;
    t->segment_head = false;
    t->segment_size = 0;
    t->segment_head_ptr = b->segment_head_ptr;
    // link
    t->prev = b;
    t->next = nullptr;
    b->next = t;
    by_ptr_[t->ptr] = t;
    insert_free_global_(t);
  }
  {
    std::lock_guard<std::mutex> lg(mu_);
    stats_.allocated_bytes_all_current += static_cast<std::uint64_t>(nbytes);
    if (stats_.allocated_bytes_all_current > stats_.max_allocated_bytes_all) stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
  }
  return p;
}

void* Allocator::raw_alloc(std::size_t nbytes, std::size_t alignment) {
  std::size_t reserved_add = 0;
  return raw_alloc(nbytes, alignment, reserved_add);
}

void Allocator::raw_delete_exact(void* p, std::size_t requested, std::size_t reserved) noexcept {
  if (!p) return;
  if (!caching_enabled_) {
    {
      std::lock_guard<std::mutex> lg(mu_);
      if (stats_.allocated_bytes_all_current >= requested) stats_.allocated_bytes_all_current -= static_cast<std::uint64_t>(requested);
      else stats_.allocated_bytes_all_current = 0;
      if (stats_.reserved_bytes_all_current >= reserved) stats_.reserved_bytes_all_current -= static_cast<std::uint64_t>(reserved);
      else stats_.reserved_bytes_all_current = 0;
      // Clean up any side-map entry if present (non-caching stats bookkeeping)
      nonc_reserved_by_ptr_.erase(p);
    }
    os_free_(p);
    return;
  }

  Block* b = nullptr;
  {
    std::lock_guard<std::mutex> lg(mu_);
    auto it = by_ptr_.find(p);
    if (it == by_ptr_.end()) {
      // Unknown pointer: strict no-op
      return;
    }
    b = it->second;
    if (b->allocated) {
      b->allocated = false;
      // update allocated stats
      if (stats_.allocated_bytes_all_current >= requested) stats_.allocated_bytes_all_current -= static_cast<std::uint64_t>(requested);
      else stats_.allocated_bytes_all_current = 0;
    }
  }

  // Optional poison (off-lock)
  if (cfg_.poison_on_free) {
    std::memset(p, 0xEF, (b->size < (1u<<20)) ? b->size : (1u<<20));
  }

  // Oversize handling
  if (b->size > (classes_.empty() ? 0 : classes_.back())) {
    if (!cfg_.oversize_cache) {
      // free to OS and update reserved
      {
        std::lock_guard<std::mutex> lg(mu_);
        // Remove maps for this piece (should not be in by_end_ as allocated)
        by_ptr_.erase(b->ptr);
      }
      os_free_(p);
      {
        std::lock_guard<std::mutex> lg(mu_);
        if (stats_.reserved_bytes_all_current >= b->segment_size) stats_.reserved_bytes_all_current -= static_cast<std::uint64_t>(b->segment_size);
        else stats_.reserved_bytes_all_current = 0;
      }
      delete b;
      return;
    } else {
      std::unique_lock<std::mutex> lk(mu_);
      // Insert into oversize free
      auto endp = (void*)(u8(b->ptr) + b->size);
      by_end_[endp] = b;
      oversize_free_.insert(b);
      cached_global_bytes_ += static_cast<std::uint64_t>(b->size);
      if (cfg_.global_cap_bytes > 0 && cached_global_bytes_ > cfg_.global_cap_bytes) {
        evict_under_cap_(cfg_.global_cap_bytes, lk);
      }
      return;
    }
  }

  // Bin-managed: push to TLS and spill if needed
  if (tls_.bins.empty()) tls_.bins.resize(classes_.size());
  tls_.bins[idx_for_size_(b->size)].push_back(b);
  tls_.tls_cached_bytes += b->size;
  if (cfg_.tls_cap_bytes > 0) {
    // Spill from largest bins down until under cap
    for (std::size_t round=0; tls_.tls_cached_bytes > cfg_.tls_cap_bytes && round < classes_.size(); ++round) {
      for (std::size_t kk = classes_.size(); kk-- > 0 && tls_.tls_cached_bytes > cfg_.tls_cap_bytes; ) {
        if (!tls_.bins[kk].empty()) {
          Block* v = tls_.bins[kk].back();
          tls_.bins[kk].pop_back();
          if (tls_.tls_cached_bytes >= v->size) tls_.tls_cached_bytes -= v->size; else tls_.tls_cached_bytes = 0;
          std::unique_lock<std::mutex> lk(mu_);
          insert_free_global_(v);
          if (cfg_.global_cap_bytes > 0 && cached_global_bytes_ > cfg_.global_cap_bytes) {
            evict_under_cap_(cfg_.global_cap_bytes, lk);
          }
        }
      }
    }
  }
}

void Allocator::raw_delete(void* p, std::size_t nbytes) noexcept {
  if (!p) return;
  if (!caching_enabled_) {
    std::size_t reserved_sub = nbytes;
    {
      std::lock_guard<std::mutex> lg(mu_);
      // If we tracked reserved for this pointer, use it and erase entry
      auto it = nonc_reserved_by_ptr_.find(p);
      if (it != nonc_reserved_by_ptr_.end()) {
        reserved_sub = it->second;
        nonc_reserved_by_ptr_.erase(it);
      }
      if (stats_.allocated_bytes_all_current >= nbytes) stats_.allocated_bytes_all_current -= static_cast<std::uint64_t>(nbytes);
      else stats_.allocated_bytes_all_current = 0;
      if (stats_.reserved_bytes_all_current >= reserved_sub) stats_.reserved_bytes_all_current -= static_cast<std::uint64_t>(reserved_sub);
      else stats_.reserved_bytes_all_current = 0;
    }
    os_free_(p);
    return;
  }
  // caching: look up reserved from metadata and route to exact
  Block* b = nullptr;
  {
    std::lock_guard<std::mutex> lg(mu_);
    auto it = by_ptr_.find(p);
    if (it == by_ptr_.end()) return;
    b = it->second;
  }
  raw_delete_exact(p, b ? b->requested : nbytes, b ? b->size : nbytes);
}

DeviceStats Allocator::getDeviceStats() const {
  std::lock_guard<std::mutex> lg(mu_);
  return stats_;
}

void Allocator::resetPeakStats() noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  stats_.max_allocated_bytes_all = stats_.allocated_bytes_all_current;
  stats_.max_reserved_bytes_all = stats_.reserved_bytes_all_current;
}

void Allocator::emptyCache() noexcept {
  if (!caching_enabled_) return; // strict no-op
  // Drain caller TLS to global, then evict whole segments
  if (!tls_.bins.empty()) {
    for (std::size_t k=0; k<tls_.bins.size(); ++k) {
      while (!tls_.bins[k].empty()) {
        Block* v = tls_.bins[k].back();
        tls_.bins[k].pop_back();
        if (tls_.tls_cached_bytes >= v->size) tls_.tls_cached_bytes -= v->size; else tls_.tls_cached_bytes = 0;
        std::lock_guard<std::mutex> lg(mu_);
        insert_free_global_(v);
      }
    }
  }
  std::unique_lock<std::mutex> lk(mu_);
  if (cached_global_bytes_ > 0) {
    evict_under_cap_(0, lk);
  }
}

bool Allocator::owns(const void* p) const noexcept {
  if (!caching_enabled_) return false;
  std::lock_guard<std::mutex> lg(mu_);
  return by_ptr_.find(const_cast<void*>(p)) != by_ptr_.end();
}

void* Allocator::getBaseAllocation(void* p, std::size_t* seg_bytes) const noexcept {
  if (!caching_enabled_) return nullptr;
  std::lock_guard<std::mutex> lg(mu_);
  auto it = by_ptr_.find(p);
  if (it == by_ptr_.end()) return nullptr;
  Block* b = it->second;
  Block* h = b->segment_head_ptr ? b->segment_head_ptr : b;
  if (seg_bytes) *seg_bytes = h->segment_size ? h->segment_size : h->size;
  return h->ptr;
}

Config Allocator::config() const { return cfg_; }

}} // namespace vbt::cpu
