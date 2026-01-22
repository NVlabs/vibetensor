// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/rng/kernels_cpu.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "vbt/core/tensor.h"
#include "vbt/core/write_guard.h"
#include "vbt/rng/philox_util.h"

namespace vbt {
namespace rng {
namespace cpu {

using vbt::core::ScalarType;
using vbt::core::TensorImpl;

static constexpr float kTwoPi = 6.28318530717958647692f; // 2*pi in float32

static inline std::size_t itemsize_bytes(ScalarType st) {
  return vbt::core::itemsize(st);
}

// Compute row-major logical index e -> data pointer for arbitrary strided tensor
static inline std::uint8_t* ptr_for_linear_e(const TensorImpl& t, std::uint64_t e_linear) {
  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  const std::size_t ndim = sizes.size();
  std::uintptr_t base = reinterpret_cast<std::uintptr_t>(t.data());
  if (base == 0) return nullptr;
  const std::size_t item_b = itemsize_bytes(t.dtype());
  std::uint64_t rem = e_linear;
  // Last dimension is fastest in row-major logical order
  for (std::size_t rev = 0; rev < ndim; ++rev) {
    const std::size_t i = ndim - 1 - rev;
    const std::uint64_t dim = static_cast<std::uint64_t>(sizes[i]);
    if (dim == 0) return reinterpret_cast<std::uint8_t*>(base); // unreachable for N==0 guards
    const std::uint64_t idx = (dim == 0) ? 0ull : (rem % dim);
    rem = (dim == 0) ? 0ull : (rem / dim);
    const std::int64_t st = strides[i];
    const std::int64_t step_bytes = static_cast<std::int64_t>(item_b) * st;
    base += static_cast<std::intptr_t>(step_bytes) * static_cast<std::intptr_t>(idx);
  }
  return reinterpret_cast<std::uint8_t*>(base);
}

void uniform_(TensorImpl& t, float low, float high, vbt::rng::Generator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Float32) {
    throw std::runtime_error("uniform_: expected dtype=float32");
  }
  if (!std::isfinite(low) || !std::isfinite(high)) {
    throw std::invalid_argument("uniform_: low and high must be finite");
  }
  if (!(low <= high)) {
    throw std::invalid_argument("uniform_: low must be <= high");
  }
  const std::int64_t n64 = t.numel();
  if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = (N + 3ull) / 4ull;
  PhiloxState st = gen.reserve_blocks(total_blocks);
  std::uint32_t key[2]; seed_to_key(st.seed, key);

  // Contiguous fast-path
  if (t.is_contiguous()) {
    float* out = reinterpret_cast<float*>(t.data());
    std::uint64_t e = 0;
    for (std::uint64_t block_rel = 0; block_rel < total_blocks; ++block_rel) {
      std::uint32_t ctr[4]; block_to_counter(st.offset + block_rel, ctr);
      std::uint32_t lanes[4]; philox10(ctr, key, lanes);
      for (std::uint32_t lane = 0; lane < 4u && e < N; ++lane, ++e) {
        float U = u32_to_uniform_f32(lanes[lane]);
        const float scale = static_cast<float>(high - low);
        out[e] = static_cast<float>(low) + scale * U;
      }
    }
    return;
  }

  // Generic strided fallback: map logical e->pointer
  for (std::uint64_t block_rel = 0; block_rel < total_blocks; ++block_rel) {
    std::uint32_t ctr[4]; block_to_counter(st.offset + block_rel, ctr);
    std::uint32_t lanes[4]; philox10(ctr, key, lanes);
    for (std::uint32_t lane = 0; lane < 4u; ++lane) {
      std::uint64_t e = block_rel * 4ull + static_cast<std::uint64_t>(lane);
      if (e >= N) break;
      std::uint8_t* pd = ptr_for_linear_e(t, e);
      float* pf = reinterpret_cast<float*>(pd);
      float U = u32_to_uniform_f32(lanes[lane]);
      const float scale = static_cast<float>(high - low);
      *pf = static_cast<float>(low) + scale * U;
    }
  }
}

void normal_(TensorImpl& t, float mean, float std, vbt::rng::Generator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Float32) {
    throw std::runtime_error("expected floating dtype for normal_");
  }
  const std::int64_t n64 = t.numel();
  if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = (N + 3ull) / 4ull; // P=4
  PhiloxState st = gen.reserve_blocks(total_blocks);
  std::uint32_t key[2]; seed_to_key(st.seed, key);

  auto gen_block = [&](std::uint64_t block_index, std::uint32_t lanes[4]) {
    std::uint32_t ctr[4]; block_to_counter(block_index, ctr);
    philox10(ctr, key, lanes);
  };

  const float m = mean; const float s = std;

  if (t.is_contiguous()) {
    float* out = reinterpret_cast<float*>(t.data());
    std::uint64_t e = 0;
    for (std::uint64_t br = 0; br < total_blocks; ++br) {
      std::uint32_t lanes[4]; gen_block(st.offset + br, lanes);
      // First pair from (u0,u1)
      if (e < N) {
        float U0 = u01_open_open(lanes[0]);
        float U1 = u01_closed_open(lanes[1]);
        float R = std::sqrt(-2.0f * std::log(U0));
        float Theta = kTwoPi * U1; // 2*pi
        float Z0 = R * std::cos(Theta);
        float Z1 = R * std::sin(Theta);
        out[e++] = m + s * Z0;
        if (e < N) out[e++] = m + s * Z1;
      }
      // Second pair from (u2,u3)
      if (e < N) {
        float U0 = u01_open_open(lanes[2]);
        float U1 = u01_closed_open(lanes[3]);
        float R = std::sqrt(-2.0f * std::log(U0));
        float Theta = kTwoPi * U1;
        float Z0 = R * std::cos(Theta);
        float Z1 = R * std::sin(Theta);
        out[e++] = m + s * Z0;
        if (e < N) out[e++] = m + s * Z1;
      }
    }
    return;
  }

  // Strided fallback
  for (std::uint64_t br = 0; br < total_blocks; ++br) {
    std::uint32_t lanes[4]; gen_block(st.offset + br, lanes);
    for (std::uint32_t lane_pair = 0; lane_pair < 2u; ++lane_pair) {
      std::uint32_t u0 = (lane_pair == 0) ? lanes[0] : lanes[2];
      std::uint32_t u1 = (lane_pair == 0) ? lanes[1] : lanes[3];
      float U0 = u01_open_open(u0);
      float U1 = u01_closed_open(u1);
      float R = std::sqrt(-2.0f * std::log(U0));
      float Theta = kTwoPi * U1;
      float Z0 = R * std::cos(Theta);
      float Z1 = R * std::sin(Theta);
      std::uint64_t e0 = br * 4ull + lane_pair * 2ull + 0ull;
      std::uint64_t e1 = br * 4ull + lane_pair * 2ull + 1ull;
      if (e0 < N) {
        float* pf0 = reinterpret_cast<float*>(ptr_for_linear_e(t, e0));
        *pf0 = m + s * Z0;
      }
      if (e1 < N) {
        float* pf1 = reinterpret_cast<float*>(ptr_for_linear_e(t, e1));
        *pf1 = m + s * Z1;
      }
    }
  }
}

void bernoulli_(TensorImpl& t, float p, vbt::rng::Generator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Float32) {
    throw std::runtime_error("expected floating dtype for bernoulli_");
  }
  if (!(p >= 0.0f && p <= 1.0f)) {
    throw std::runtime_error("bernoulli_: p must be in [0, 1]");
  }
  const std::int64_t n64 = t.numel();
  if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = (N + 3ull) / 4ull; // P=4
  PhiloxState st = gen.reserve_blocks(total_blocks);
  std::uint32_t key[2]; seed_to_key(st.seed, key);

  if (t.is_contiguous()) {
    float* out = reinterpret_cast<float*>(t.data());
    std::uint64_t e = 0;
    for (std::uint64_t br = 0; br < total_blocks; ++br) {
      std::uint32_t ctr[4]; block_to_counter(st.offset + br, ctr);
      std::uint32_t lanes[4]; philox10(ctr, key, lanes);
      for (std::uint32_t lane = 0; lane < 4u && e < N; ++lane, ++e) {
        float U = u32_to_uniform_f32(lanes[lane]);
        out[e] = (U < p) ? 1.0f : 0.0f;
      }
    }
    return;
  }

  for (std::uint64_t br = 0; br < total_blocks; ++br) {
    std::uint32_t ctr[4]; block_to_counter(st.offset + br, ctr);
    std::uint32_t lanes[4]; philox10(ctr, key, lanes);
    for (std::uint32_t lane = 0; lane < 4u; ++lane) {
      std::uint64_t e = br * 4ull + static_cast<std::uint64_t>(lane);
      if (e >= N) break;
      float* pf = reinterpret_cast<float*>(ptr_for_linear_e(t, e));
      float U = u32_to_uniform_f32(lanes[lane]);
      *pf = (U < p) ? 1.0f : 0.0f;
    }
  }
}

// 128-bit mul helpers for randint
static inline void mul_64x64_128(std::uint64_t a, std::uint64_t b, std::uint64_t& lo, std::uint64_t& hi) {
#if defined(_MSC_VER) && defined(_M_X64)
  unsigned __int64 _lo, _hi;
  _lo = _umul128(a, b, &_hi);
  lo = static_cast<std::uint64_t>(_lo);
  hi = static_cast<std::uint64_t>(_hi);
#elif defined(__SIZEOF_INT128__)
  unsigned __int128 prod = (static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b));
  lo = static_cast<std::uint64_t>(prod & 0xFFFFFFFFFFFFFFFFull);
  hi = static_cast<std::uint64_t>(prod >> 64);
#else
  // Portable split-32 fallback
  const std::uint64_t a_lo = a & 0xFFFFFFFFull;
  const std::uint64_t a_hi = a >> 32;
  const std::uint64_t b_lo = b & 0xFFFFFFFFull;
  const std::uint64_t b_hi = b >> 32;
  std::uint64_t p0 = a_lo * b_lo;
  std::uint64_t p1 = a_lo * b_hi;
  std::uint64_t p2 = a_hi * b_lo;
  std::uint64_t p3 = a_hi * b_hi;
  std::uint64_t mid1 = (p0 >> 32) + (p1 & 0xFFFFFFFFull) + (p2 & 0xFFFFFFFFull);
  std::uint64_t mid2 = (mid1 >> 32) + (p1 >> 32) + (p2 >> 32) + (p3 & 0xFFFFFFFFull);
  lo = (p0 & 0xFFFFFFFFull) | (mid1 << 32);
  hi = (p3 >> 32) + mid2;
#endif
}

void randint_(TensorImpl& t, std::int64_t low, std::int64_t high, vbt::rng::Generator& gen) {
  vbt::core::check_writable(t);
  if (t.dtype() != ScalarType::Int64) {
    throw std::runtime_error("randint: output dtype must be int64");
  }
  if (!(low < high)) {
    throw std::runtime_error("randint: require low < high and (high - low) in [1, 2^63 - 1]");
  }
  const std::uint64_t lo_u = static_cast<std::uint64_t>(low);
  const std::uint64_t hi_u = static_cast<std::uint64_t>(high);
  const std::uint64_t n = hi_u - lo_u; // unsigned subtraction avoids UB on signed overflow
  if (n == 0 || n > 0x7FFFFFFFFFFFFFFFull) {
    throw std::runtime_error("randint: require low < high and (high - low) in [1, 2^63 - 1]");
  }
  const std::int64_t n64 = t.numel();
  if (n64 <= 0) return;
  const std::uint64_t N = static_cast<std::uint64_t>(n64);
  const std::uint64_t total_blocks = (N + 1ull) / 2ull; // P_fixed=2
  PhiloxState st = gen.reserve_blocks(total_blocks);
  std::uint32_t key[2]; seed_to_key(st.seed, key);
  const std::uint64_t threshold = (static_cast<std::uint64_t>(0) - static_cast<std::uint64_t>(n)) % static_cast<std::uint64_t>(n);

  auto candidate_for = [&](std::uint64_t B, std::uint32_t lane, std::uint32_t attempt) -> std::uint64_t {
    std::uint32_t ctr[4]; block_to_counter(B, ctr); ctr[2] = attempt;
    std::uint32_t lanes[4]; philox10(ctr, key, lanes);
    if (lane == 0u) {
      return pack_u64(lanes[0], lanes[1]);
    } else {
      return pack_u64(lanes[2], lanes[3]);
    }
  };

  if (t.is_contiguous()) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(t.data());
    for (std::uint64_t e = 0; e < N; ++e) {
      const std::uint64_t lane = e & 1ull; // e % 2
      const std::uint64_t block_rel = e >> 1; // e / 2
      const std::uint64_t B = st.offset + block_rel;
      std::uint32_t attempt = 0u;
      while (true) {
        std::uint64_t R = candidate_for(B, static_cast<std::uint32_t>(lane), attempt);
        std::uint64_t lo, hi; mul_64x64_128(R, n, lo, hi);
        if (lo < threshold) { ++attempt; continue; }
        out[e] = static_cast<std::int64_t>(static_cast<std::uint64_t>(low) + hi);
        break;
      }
    }
    return;
  }

  for (std::uint64_t e = 0; e < N; ++e) {
    const std::uint64_t lane = e & 1ull;
    const std::uint64_t block_rel = e >> 1;
    const std::uint64_t B = st.offset + block_rel;
    std::uint32_t attempt = 0u;
    while (true) {
      std::uint64_t R = candidate_for(B, static_cast<std::uint32_t>(lane), attempt);
      std::uint64_t lo, hi; mul_64x64_128(R, n, lo, hi);
      if (lo < threshold) { ++attempt; continue; }
      std::int64_t* pd = reinterpret_cast<std::int64_t*>(ptr_for_linear_e(t, e));
      *pd = static_cast<std::int64_t>(static_cast<std::uint64_t>(low) + hi);
      break;
    }
  }
}

} // namespace cpu
} // namespace rng
} // namespace vbt
