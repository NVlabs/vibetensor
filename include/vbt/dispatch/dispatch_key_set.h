// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "vbt/dispatch/dispatch_key.h"

namespace vbt {
namespace dispatch {

class DispatchKeySet final {
 public:
  using Repr = std::uint64_t;
  static_assert(kNumDispatchKeys <= 64, "DispatchKeySet supports up to 64 keys");

  // NB: Default constructor representation as zero is MANDATORY as use of
  // DispatchKeySet in TLS requires this.
  constexpr DispatchKeySet() = default;

  constexpr explicit DispatchKeySet(DispatchKey k) : bits_(mask(k)) {}

  constexpr Repr raw_repr() const noexcept { return bits_; }

  constexpr bool empty() const noexcept { return bits_ == 0; }

  constexpr bool has(DispatchKey k) const noexcept {
    return (bits_ & mask(k)) != 0;
  }

  constexpr DispatchKeySet add(DispatchKey k) const noexcept {
    return DispatchKeySet(bits_ | mask(k), RawTag{});
  }

  constexpr DispatchKeySet remove(DispatchKey k) const noexcept {
    return DispatchKeySet(bits_ & ~mask(k), RawTag{});
  }

  // Precondition: !empty().
  constexpr DispatchKey highest_priority_key() const noexcept {
    assert(!empty());
    for (std::uint8_t i = 0; i < static_cast<std::uint8_t>(DispatchKey::NumKeys); ++i) {
      DispatchKey k = static_cast<DispatchKey>(i);
      if (has(k)) return k;
    }
    // Unreachable if the precondition holds.
    return DispatchKey::NumKeys;
  }

  friend constexpr bool operator==(DispatchKeySet a, DispatchKeySet b) noexcept {
    return a.bits_ == b.bits_;
  }
  friend constexpr bool operator!=(DispatchKeySet a, DispatchKeySet b) noexcept {
    return a.bits_ != b.bits_;
  }

  friend constexpr DispatchKeySet operator|(DispatchKeySet a, DispatchKeySet b) noexcept {
    return DispatchKeySet(a.bits_ | b.bits_, RawTag{});
  }
  friend constexpr DispatchKeySet operator&(DispatchKeySet a, DispatchKeySet b) noexcept {
    return DispatchKeySet(a.bits_ & b.bits_, RawTag{});
  }
  friend constexpr DispatchKeySet operator~(DispatchKeySet a) noexcept {
    return DispatchKeySet((~a.bits_) & full_mask(), RawTag{});
  }

 private:
  struct RawTag {};

  constexpr DispatchKeySet(Repr bits, RawTag) : bits_(bits) {}

  static constexpr Repr mask(DispatchKey k) noexcept {
    return (k == DispatchKey::NumKeys) ? 0 : (Repr{1} << static_cast<std::uint8_t>(k));
  }

  static constexpr Repr full_mask() noexcept {
    return (static_cast<std::uint8_t>(DispatchKey::NumKeys) >= 64)
               ? ~Repr{0}
               : ((Repr{1} << static_cast<std::uint8_t>(DispatchKey::NumKeys)) - 1);
  }

  Repr bits_{0};
};

static_assert(std::is_trivially_copyable_v<DispatchKeySet>,
              "DispatchKeySet must be trivially copyable");

struct LocalDispatchKeySet {
  DispatchKeySet included;
  DispatchKeySet excluded;

  constexpr DispatchKeySet apply(DispatchKeySet ks) const noexcept {
    // Exclusion wins over inclusion.
    //
    // Backend keys must never be excludable; this is required to preserve stable
    // missing-kernel behavior once key-masking/redispatch exists.
    constexpr DispatchKeySet backend_mask =
        DispatchKeySet{}.add(DispatchKey::CPU).add(DispatchKey::CUDA);
    DispatchKeySet excluded_no_backend = excluded & ~backend_mask;
    return (ks | included) & ~excluded_no_backend;
  }
};

static_assert(std::is_trivially_copyable_v<LocalDispatchKeySet>,
              "LocalDispatchKeySet must be trivially copyable");

extern thread_local LocalDispatchKeySet tls_local_dispatch_key_set;

inline DispatchKeySet apply_tls(DispatchKeySet ks) noexcept {
  return tls_local_dispatch_key_set.apply(ks);
}

struct IncludeDispatchKeysGuard {
  DispatchKeySet prev_;
  explicit IncludeDispatchKeysGuard(DispatchKeySet ks) noexcept
      : prev_(tls_local_dispatch_key_set.included) {
    tls_local_dispatch_key_set.included = tls_local_dispatch_key_set.included | ks;
  }
  ~IncludeDispatchKeysGuard() noexcept { tls_local_dispatch_key_set.included = prev_; }
  IncludeDispatchKeysGuard(const IncludeDispatchKeysGuard&) = delete;
  IncludeDispatchKeysGuard& operator=(const IncludeDispatchKeysGuard&) = delete;
};

struct ExcludeDispatchKeysGuard {
  DispatchKeySet prev_;
  explicit ExcludeDispatchKeysGuard(DispatchKeySet ks) noexcept
      : prev_(tls_local_dispatch_key_set.excluded) {
    ks = ks.remove(DispatchKey::CPU).remove(DispatchKey::CUDA);
    tls_local_dispatch_key_set.excluded = tls_local_dispatch_key_set.excluded | ks;
  }
  ~ExcludeDispatchKeysGuard() noexcept { tls_local_dispatch_key_set.excluded = prev_; }
  ExcludeDispatchKeysGuard(const ExcludeDispatchKeysGuard&) = delete;
  ExcludeDispatchKeysGuard& operator=(const ExcludeDispatchKeysGuard&) = delete;
};

struct LocalDispatchKeySetGuard {
  LocalDispatchKeySet prev_;
  explicit LocalDispatchKeySetGuard(LocalDispatchKeySet ks) noexcept
      : prev_(tls_local_dispatch_key_set) {
    ks.excluded = ks.excluded.remove(DispatchKey::CPU).remove(DispatchKey::CUDA);
    tls_local_dispatch_key_set = ks;
  }
  ~LocalDispatchKeySetGuard() noexcept { tls_local_dispatch_key_set = prev_; }
  LocalDispatchKeySetGuard(const LocalDispatchKeySetGuard&) = delete;
  LocalDispatchKeySetGuard& operator=(const LocalDispatchKeySetGuard&) = delete;
};

static_assert(noexcept(IncludeDispatchKeysGuard(DispatchKeySet{})),
              "IncludeDispatchKeysGuard ctor must be noexcept");
static_assert(std::is_nothrow_destructible_v<IncludeDispatchKeysGuard>,
              "IncludeDispatchKeysGuard dtor must be noexcept");
static_assert(noexcept(ExcludeDispatchKeysGuard(DispatchKeySet{})),
              "ExcludeDispatchKeysGuard ctor must be noexcept");
static_assert(std::is_nothrow_destructible_v<ExcludeDispatchKeysGuard>,
              "ExcludeDispatchKeysGuard dtor must be noexcept");
static_assert(noexcept(LocalDispatchKeySetGuard(LocalDispatchKeySet{})),
              "LocalDispatchKeySetGuard ctor must be noexcept");
static_assert(std::is_nothrow_destructible_v<LocalDispatchKeySetGuard>,
              "LocalDispatchKeySetGuard dtor must be noexcept");

} // namespace dispatch
} // namespace vbt
