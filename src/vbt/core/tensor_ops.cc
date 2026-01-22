// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_ops.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/core/checked_math.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/storage.h"
#include "vbt/core/complex.h"

namespace vbt {
namespace core {

namespace {

[[nodiscard]] inline int64_t numel_or_throw_from_sizes(
    const std::vector<int64_t>& sizes,
    const char* ctx) {
  if (sizes.empty()) {
    return 1;  // scalar
  }

  int64_t n = 1;
  for (int64_t s : sizes) {
    if (s < 0) {
      throw std::invalid_argument(std::string(ctx) + ": negative size");
    }
    if (s == 0) {
      return 0;
    }
    int64_t tmp = 0;
    if (!checked_mul_i64(n, s, tmp)) {
      throw std::overflow_error(std::string(ctx) + ": numel overflow");
    }
    n = tmp;
  }

  return n;
}

[[nodiscard]] inline int64_t numel_or_throw(const TensorImpl& self,
                                           const char* ctx) {
  return numel_or_throw_from_sizes(self.sizes(), ctx);
}

[[nodiscard]] inline std::size_t nbytes_or_throw(std::size_t item_b,
                                                 int64_t numel,
                                                 const char* ctx) {
  if (numel < 0) {
    throw std::invalid_argument(std::string(ctx) + ": negative numel");
  }
  if (numel == 0) {
    return 0;
  }
  if (item_b == 0) {
    throw std::invalid_argument(
        std::string(ctx) + ": itemsize is zero for non-empty tensor");
  }

  if (item_b > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(std::string(ctx) + ": itemsize too large");
  }
  const int64_t item_b_i64 = static_cast<int64_t>(item_b);

  int64_t bytes_i64 = 0;
  if (!checked_mul_i64(numel, item_b_i64, bytes_i64)) {
    throw std::overflow_error(std::string(ctx) + ": nbytes overflow");
  }
  if (bytes_i64 < 0) {
    throw std::overflow_error(std::string(ctx) + ": nbytes underflow");
  }

  const auto bytes_u64 = static_cast<std::uint64_t>(bytes_i64);
  if (bytes_u64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(ctx) + ": nbytes too large");
  }

  return static_cast<std::size_t>(bytes_u64);
}

[[nodiscard]] inline std::vector<int64_t> compute_contiguous_strides_checked(
    const std::vector<int64_t>& sizes,
    const char* ctx) {
  std::vector<int64_t> strides_out(sizes.size(), 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0;
       --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides_out[idx] = acc;

    int64_t dim = sizes[idx];
    if (dim < 0) {
      throw std::invalid_argument(std::string(ctx) + ": negative size");
    }
    if (dim == 0) {
      dim = 1;
    }

    int64_t tmp = 0;
    if (!checked_mul_i64(acc, dim, tmp)) {
      throw std::overflow_error(std::string(ctx) + ": contiguous stride overflow");
    }
    acc = tmp;
  }
  return strides_out;
}

}  // namespace

TensorImpl clone_contiguous_same_device(const TensorImpl& self) {
  const auto dev = self.device();
  if (dev.type == kDLCPU) {
    return clone_cpu(self);
  }
#if VBT_WITH_CUDA
  if (dev.type == kDLCUDA) {
    return clone_cuda(self);
  }
#endif
  throw std::invalid_argument(
      "clone_contiguous_same_device: unsupported device");
}

TensorImpl clone_cpu(const TensorImpl& self) {
  if (self.device().type != kDLCPU) {
    throw std::invalid_argument("clone_cpu: expected a CPU tensor");
  }

  const auto item_b = static_cast<std::size_t>(self.itemsize());
  const int64_t ne = numel_or_throw(self, "clone_cpu");
  const std::size_t nbytes = nbytes_or_throw(item_b, ne, "clone_cpu");

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<int64_t> out_sizes = self.sizes();
  std::vector<int64_t> out_strides =
      compute_contiguous_strides_checked(out_sizes, "clone_cpu");

  TensorImpl out(storage,
                 std::move(out_sizes),
                 std::move(out_strides),
                 /*storage_offset=*/0,
                 self.dtype(),
                 Device::cpu(self.device().index));

  if (ne == 0) {
    return out;
  }

  const ScalarType dt = self.dtype();
  auto maybe_materialize_conj = [&]() {
    if (!self.is_conj()) {
      return;
    }
    if (dt == ScalarType::Complex64) {
      auto* p = static_cast<Complex64*>(out.data());
      for (int64_t i = 0; i < ne; ++i) {
        p[i].im = -p[i].im;
      }
      return;
    }
    if (dt == ScalarType::Complex128) {
      auto* p = static_cast<Complex128*>(out.data());
      for (int64_t i = 0; i < ne; ++i) {
        p[i].im = -p[i].im;
      }
      return;
    }
  };

  const StoragePtr& in_storage = self.storage();
  if (!in_storage) {
    throw std::runtime_error("clone_cpu: input has no storage");
  }
  void* base_void = in_storage->data();
  if (!base_void) {
    throw std::runtime_error("clone_cpu: input storage has no data");
  }
  const auto* storage_base = static_cast<const std::uint8_t*>(base_void);
  const std::size_t in_nbytes = in_storage->nbytes();

  if (item_b > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error("clone_cpu: itemsize too large");
  }
  const int64_t item_b_i64 = static_cast<int64_t>(item_b);

  // Fast path for contiguous inputs: a single memcpy (validated).
  if (self.is_contiguous()) {
    int64_t base_bytes_i64 = 0;
    if (!checked_mul_i64(self.storage_offset(), item_b_i64, base_bytes_i64)) {
      throw std::overflow_error("clone_cpu: base byte offset overflow");
    }
    if (base_bytes_i64 < 0) {
      throw std::out_of_range("clone_cpu: negative base byte offset");
    }

    const std::size_t base_bytes = static_cast<std::size_t>(base_bytes_i64);
    if (base_bytes > in_nbytes || nbytes > (in_nbytes - base_bytes)) {
      throw std::out_of_range(
          "clone_cpu: input storage too small for contiguous copy");
    }

    std::memcpy(out.data(), storage_base + base_bytes, nbytes);
    maybe_materialize_conj();
    return out;
  }

  // General strided -> contiguous elementwise copy (negative strides supported).
  const auto& in_sizes = self.sizes();
  const auto& in_strides = self.strides();
  const std::size_t nd = in_sizes.size();

  std::vector<int64_t> idx(nd, 0);
  auto* out_ptr = static_cast<std::uint8_t*>(out.data());

  for (int64_t linear = 0; linear < ne; ++linear) {
    int64_t abs_off_elems = self.storage_offset();

    for (std::size_t d = 0; d < nd; ++d) {
      int64_t term = 0;
      if (!checked_mul_i64(idx[d], in_strides[d], term)) {
        throw std::overflow_error("clone_cpu: index*stride overflow");
      }
      if (!checked_add_i64(abs_off_elems, term, abs_off_elems)) {
        throw std::overflow_error("clone_cpu: offset accumulation overflow");
      }
    }

    if (abs_off_elems < 0) {
      throw std::out_of_range("clone_cpu: computed negative element offset");
    }

    int64_t byte_off_i64 = 0;
    if (!checked_mul_i64(abs_off_elems, item_b_i64, byte_off_i64)) {
      throw std::overflow_error("clone_cpu: byte offset overflow");
    }
    if (byte_off_i64 < 0) {
      throw std::out_of_range("clone_cpu: computed negative byte offset");
    }

    const std::size_t byte_off = static_cast<std::size_t>(byte_off_i64);
    if (byte_off > in_nbytes || item_b > (in_nbytes - byte_off)) {
      throw std::out_of_range("clone_cpu: input storage out of bounds");
    }

    const auto* src = storage_base + byte_off;
    std::memcpy(out_ptr + static_cast<std::size_t>(linear) * item_b, src, item_b);

    // Increment idx (row-major from last dimension)
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nd) - 1; d >= 0; --d) {
      const auto ud = static_cast<std::size_t>(d);
      idx[ud] += 1;
      if (idx[ud] < in_sizes[ud]) {
        break;
      }
      idx[ud] = 0;
    }
  }

  maybe_materialize_conj();
  return out;
}

}  // namespace core
}  // namespace vbt
