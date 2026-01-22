// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <cstdint>
#include <cstring>
#include <random>
#ifdef __linux__
#include <sys/random.h>
#include <unistd.h>
#endif
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
#include <stdlib.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#ifndef BCRYPT_SUCCESS
#define BCRYPT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif
#pragma comment(lib, "bcrypt.lib")
#endif

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/rng/generator.h"
#include "vbt/rng/kernels_cpu.h"
#include "vbt/rng/kernels_cuda.h"
#include "vbt/cuda/device.h"
#include "vbt/rng/graph_capture.h"

namespace nb = nanobind;

namespace vbt_py {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::rng::Generator;
using vbt::rng::PhiloxState;

static inline std::uint64_t nondet_seed64() {
  std::uint64_t s = 0;
#if defined(__linux__)
  ssize_t n = getrandom(&s, sizeof(s), 0);
  if (n == (ssize_t)sizeof(s)) return s;
#endif
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
  arc4random_buf(&s, sizeof(s));
  return s;
#endif
#if defined(_WIN32)
  NTSTATUS st = BCryptGenRandom(nullptr,
                                reinterpret_cast<PUCHAR>(&s),
                                static_cast<ULONG>(sizeof(s)),
                                BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  if (BCRYPT_SUCCESS(st)) {
    return s;
  }
#endif
  std::random_device rd;
  std::uint64_t a = static_cast<std::uint64_t>(rd());
  std::uint64_t b = static_cast<std::uint64_t>(rd());
  return (a << 32) | b;
}

static inline nb::bytes pack_state(const PhiloxState& st) {
  unsigned char buf[16];
  std::uint64_t seed = st.seed;
  std::uint64_t off = st.offset;
  for (int i = 0; i < 8; ++i) buf[i] = (unsigned char)((seed >> (8 * i)) & 0xFFu);
  for (int i = 0; i < 8; ++i) buf[8 + i] = (unsigned char)((off >> (8 * i)) & 0xFFu);
  return nb::bytes(reinterpret_cast<const char*>(buf), 16);
}

static inline vbt::rng::PhiloxState unpack_state(const nb::bytes& b) {
  size_t n = (size_t) b.size();
  if (n != 16) {
    throw nb::value_error("state must be 16 bytes: {seed:u64, offset:u64}");
  }
  vbt::rng::PhiloxState st{};
  const unsigned char* p = reinterpret_cast<const unsigned char*>(b.c_str());
  std::uint64_t seed = 0, off = 0;
  for (int i = 0; i < 8; ++i) seed |= (std::uint64_t)p[i] << (8 * i);
  for (int i = 0; i < 8; ++i) off  |= (std::uint64_t)p[8 + i] << (8 * i);
  st.seed = seed; st.offset = off;
  return st;
}

namespace {

static inline void guard_cuda_generator_mutation(vbt::rng::CudaGenerator& gen) {
  if (vbt::rng::graph_capture::is_generator_capture_active(gen)) {
    throw std::runtime_error(
        vbt::rng::graph_capture::kErrCudaRngMutationDuringCapture);
  }
}

} // anonymous namespace

void bind_rng(nb::module_& m) {
  // State and seeding APIs (CPU default generator)
  m.def("_rng_manual_seed", [](std::uint64_t seed){ vbt::rng::default_cpu().manual_seed(seed); return seed; });
  m.def("_rng_seed", [](){ std::uint64_t s = nondet_seed64(); vbt::rng::default_cpu().manual_seed(s); return s; });
  m.def("_rng_initial_seed", [](){ return vbt::rng::default_cpu().initial_seed(); });
  m.def("_rng_get_state", [](){ return pack_state(vbt::rng::default_cpu().get_state()); });
  m.def("_rng_set_state", [](nb::bytes b){ vbt::rng::PhiloxState st = unpack_state(b); vbt::rng::default_cpu().set_state(st.seed, st.offset); });

  // Expose canonical CUDA RNG mutation guard error text for Python tests.
  m.attr("_ERR_CUDA_RNG_MUTATION_DURING_CAPTURE") = nb::str(
      vbt::rng::graph_capture::kErrCudaRngMutationDuringCapture);

  // CUDA device-indexed RNG state and seeding APIs (available when built with CUDA)
#if VBT_WITH_CUDA
  m.def("_cuda_rng_manual_seed", [](int device_index, std::uint64_t seed){
    int n = vbt::cuda::device_count();
    if (device_index < 0 || device_index >= n) {
      std::string msg = std::string("invalid cuda device index: ") + std::to_string(device_index);
      throw nb::value_error(msg.c_str());
    }
    auto& gen = vbt::rng::default_cuda(device_index);
    guard_cuda_generator_mutation(gen);
    gen.manual_seed(seed);
    return seed;
  });
  m.def("_cuda_rng_initial_seed", [](int device_index){
    int n = vbt::cuda::device_count();
    if (device_index < 0 || device_index >= n) {
      std::string msg = std::string("invalid cuda device index: ") + std::to_string(device_index);
      throw nb::value_error(msg.c_str());
    }
    return vbt::rng::default_cuda(device_index).initial_seed();
  });
  m.def("_cuda_rng_get_state", [](int device_index){
    int n = vbt::cuda::device_count();
    if (device_index < 0 || device_index >= n) {
      std::string msg = std::string("invalid cuda device index: ") + std::to_string(device_index);
      throw nb::value_error(msg.c_str());
    }
    return pack_state(vbt::rng::default_cuda(device_index).get_state());
  });
  m.def("_cuda_rng_set_state", [](int device_index, nb::bytes b){
    int n = vbt::cuda::device_count();
    if (device_index < 0 || device_index >= n) {
      std::string msg = std::string("invalid cuda device index: ") + std::to_string(device_index);
      throw nb::value_error(msg.c_str());
    }
    vbt::rng::PhiloxState st = unpack_state(b);
    auto& gen = vbt::rng::default_cuda(device_index);
    guard_cuda_generator_mutation(gen);
    gen.set_state(st.seed, st.offset);
  });

  // Internal helper for tests: query whether RNG capture is active for
  // the default CUDA generator on this device.
  m.def("_cuda_rng_is_capture_active_for_device", [](int device_index) {
    int n = vbt::cuda::device_count();
    if (device_index < 0 || device_index >= n) {
      std::string msg = std::string("invalid cuda device index: ") + std::to_string(device_index);
      throw nb::value_error(msg.c_str());
    }
    auto& gen = vbt::rng::default_cuda(device_index);
    return vbt::rng::graph_capture::is_generator_capture_active(gen);
  });
#else
  m.def("_cuda_rng_manual_seed", [](int, std::uint64_t){ throw nb::runtime_error("cuda rng: CUDA backend not available"); });
  m.def("_cuda_rng_initial_seed", [](int){ throw nb::runtime_error("cuda rng: CUDA backend not available"); });
  m.def("_cuda_rng_get_state", [](int){ throw nb::runtime_error("cuda rng: CUDA backend not available"); });
  m.def("_cuda_rng_set_state", [](int, nb::bytes){ throw nb::runtime_error("cuda rng: CUDA backend not available"); });
#endif

  // In-place uniform for float32 (CPU + CUDA)
  m.def("_uniform_", [](TensorImpl& t, float low, float high){
    if (t.dtype() != ScalarType::Float32) {
      throw nb::type_error("uniform_: expected dtype=float32");
    }
    auto dev = t.device();
    if (dev.type == kDLCUDA) {
#if VBT_WITH_CUDA
      vbt::rng::cuda::uniform_(t, low, high, vbt::rng::default_cuda(dev.index));
      return t;
#else
      throw nb::runtime_error("uniform_: CUDA backend not available");
#endif
    }
    vbt::rng::cpu::uniform_(t, low, high, vbt::rng::default_cpu());
    return t;
  });

  // In-place normal for float32 (CPU + CUDA)
  m.def("_normal_", [](TensorImpl& t, float mean, float std){
    if (t.dtype() != ScalarType::Float32) {
      throw nb::type_error("expected floating dtype for normal_");
    }
    auto dev = t.device();
    if (dev.type == kDLCUDA) {
#if VBT_WITH_CUDA
      vbt::rng::cuda::normal_(t, mean, std, vbt::rng::default_cuda(dev.index));
      return t;
#else
      throw nb::runtime_error("normal_: CUDA backend not available");
#endif
    }
    vbt::rng::cpu::normal_(t, mean, std, vbt::rng::default_cpu());
    return t;
  });

  // In-place bernoulli for float32 (CPU + CUDA)
  m.def("_bernoulli_", [](TensorImpl& t, float p){
    if (t.dtype() != ScalarType::Float32) {
      throw nb::type_error("expected floating dtype for bernoulli_");
    }
    if (!(p >= 0.0f && p <= 1.0f)) {
      throw nb::value_error("bernoulli_: p must be in [0, 1]");
    }
    auto dev = t.device();
    if (dev.type == kDLCUDA) {
#if VBT_WITH_CUDA
      vbt::rng::cuda::bernoulli_(t, p, vbt::rng::default_cuda(dev.index));
      return t;
#else
      throw nb::runtime_error("bernoulli_: CUDA backend not available");
#endif
    }
    vbt::rng::cpu::bernoulli_(t, p, vbt::rng::default_cpu());
    return t;
  });

  // In-place randint for int64 (CPU + CUDA)
  m.def("_randint_", [](TensorImpl& t, std::int64_t low, std::int64_t high){
    if (t.dtype() != ScalarType::Int64) {
      throw nb::type_error("randint: output dtype must be int64");
    }
    if (!(low < high)) {
      throw nb::value_error("randint: require low < high and (high - low) in [1, 2^63 - 1]");
    }
    const std::uint64_t lo_u = static_cast<std::uint64_t>(low);
    const std::uint64_t hi_u = static_cast<std::uint64_t>(high);
    const std::uint64_t n = hi_u - lo_u; // unsigned subtraction avoids UB on signed overflow
    if (n == 0 || n > 0x7FFFFFFFFFFFFFFFull) {
      throw nb::value_error("randint: require low < high and (high - low) in [1, 2^63 - 1]");
    }
    auto dev = t.device();
    if (dev.type == kDLCUDA) {
#if VBT_WITH_CUDA
      vbt::rng::cuda::randint_(t, low, high, vbt::rng::default_cuda(dev.index));
      return t;
#else
      throw nb::runtime_error("randint_: CUDA backend not available");
#endif
    }
    vbt::rng::cpu::randint_(t, low, high, vbt::rng::default_cpu());
    return t;
  });
}

} // namespace vbt_py
