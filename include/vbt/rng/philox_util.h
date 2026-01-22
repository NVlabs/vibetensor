// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace vbt {
namespace rng {

#if defined(__CUDACC__)
#define VBT_RNG_HD __host__ __device__
#else
#define VBT_RNG_HD
#endif

// Integer ceil_div for 64-bit unsigned values.
VBT_RNG_HD inline std::uint64_t ceil_div_u64(std::uint64_t a, std::uint64_t b) {
  return (b == 0) ? 0ull : (a / b) + ((a % b) != 0 ? 1ull : 0ull);
}

// Map 64-bit seed to 2x32-bit key (low32, high32)
VBT_RNG_HD inline void seed_to_key(std::uint64_t seed, std::uint32_t key[2]) {
  key[0] = static_cast<std::uint32_t>(seed & 0xFFFFFFFFull);
  key[1] = static_cast<std::uint32_t>((seed >> 32) & 0xFFFFFFFFull);
}

// Map 64-bit block index to 4x32-bit counter {lo, hi, attempt, 0}
VBT_RNG_HD inline void block_to_counter(std::uint64_t block_index, std::uint32_t ctr[4]) {
  ctr[0] = static_cast<std::uint32_t>(block_index & 0xFFFFFFFFull);
  ctr[1] = static_cast<std::uint32_t>((block_index >> 32) & 0xFFFFFFFFull);
  ctr[2] = 0u; // attempt dimension (unused for uniform)
  ctr[3] = 0u;
}

static constexpr std::uint32_t kPhilox10A = 0x9E3779B9u;
static constexpr std::uint32_t kPhilox10B = 0xBB67AE85u;
static constexpr std::uint32_t kPhiloxSA  = 0xD2511F53u;
static constexpr std::uint32_t kPhiloxSB  = 0xCD9E8D57u;

VBT_RNG_HD inline std::uint32_t mulhilo32(std::uint32_t a, std::uint32_t b, std::uint32_t* hi) {
  std::uint64_t prod = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
  *hi = static_cast<std::uint32_t>(prod >> 32);
  return static_cast<std::uint32_t>(prod);
}

VBT_RNG_HD inline void single_round(std::uint32_t ctr[4], const std::uint32_t key[2], std::uint32_t out[4]) {
  std::uint32_t hi0 = 0, hi1 = 0;
  std::uint32_t lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
  std::uint32_t lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
  out[0] = hi1 ^ ctr[1] ^ key[0];
  out[1] = lo1;
  out[2] = hi0 ^ ctr[3] ^ key[1];
  out[3] = lo0;
}

VBT_RNG_HD inline void philox10(const std::uint32_t ctr_in[4], const std::uint32_t key_in[2], std::uint32_t out[4]) {
  std::uint32_t ctr[4] = {ctr_in[0], ctr_in[1], ctr_in[2], ctr_in[3]};
  std::uint32_t key[2] = {key_in[0], key_in[1]};
  for (std::uint32_t r = 0; r < 9u; ++r) {
    std::uint32_t tmp[4];
    single_round(ctr, key, tmp);
    ctr[0] = tmp[0]; ctr[1] = tmp[1]; ctr[2] = tmp[2]; ctr[3] = tmp[3];
    key[0] += kPhilox10A; key[1] += kPhilox10B;
  }
  single_round(ctr, key, out);
}

VBT_RNG_HD inline float u32_to_uniform_f32(std::uint32_t u) {
  // Mask to 31 bits then scale by exact 2^-31 in float32.
  // This must stay bitwise-aligned with _kat_uniform in tests.
  constexpr float kInvPow2_31 = 0x1p-31f; // 1 / 2^31
  float x = static_cast<float>(u & 0x7FFFFFFFu);
  return static_cast<float>(x * kInvPow2_31);
}

// Closed-open [0,1) uniform using full 32-bit range scaled to float32
VBT_RNG_HD inline float u01_closed_open(std::uint32_t u) {
  // Use 32-bit scaling to [0,1); match u32_to_uniform_f32
  return u32_to_uniform_f32(u);
}

// Open-open (0,1): ensure strictly inside (0,1) to avoid log(0)
VBT_RNG_HD inline float u01_open_open(std::uint32_t u) {
  // Construct from closed-open by avoiding endpoints: scale to (2^-32, 1-2^-32]
  // Using float math: Uo = Uc * (1 - 2^-32) + 2^-32
  // 2^-32 in float32
  constexpr float kInvPow2_32 = 0x1p-32f;
  float Uc = u01_closed_open(u); // in [0,1)
  return Uc * (1.0f - kInvPow2_32) + kInvPow2_32;
}

// Pack two 32-bit lanes into 64-bit (lo,hi) -> (hi<<32)|lo
VBT_RNG_HD inline std::uint64_t pack_u64(std::uint32_t lo, std::uint32_t hi) {
  return (static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo);
}

#undef VBT_RNG_HD

} // namespace rng
} // namespace vbt
