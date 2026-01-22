// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "vbt/core/checked_math.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor.h"
#include "vbt/cpu/storage.h"

namespace nb = nanobind;

namespace vbt_py {

using vbt::core::Device;
using vbt::core::TensorImpl;

[[nodiscard]] static inline int64_t numel_or_throw(const TensorImpl& t, const char* ctx) {
  const auto& sizes = t.sizes();
  if (sizes.empty()) {
    return 1;  // scalar
  }
  int64_t n = 1;
  for (int64_t s : sizes) {
    if (s < 0) {
      throw nb::value_error((std::string(ctx) + ": negative size").c_str());
    }
    if (s == 0) {
      return 0;
    }
    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(n, s, tmp)) {
      throw std::overflow_error(std::string(ctx) + ": numel overflow");
    }
    n = tmp;
  }
  return n;
}

[[nodiscard]] static inline std::size_t nbytes_or_throw(std::size_t item_b,
                                                       int64_t numel,
                                                       const char* ctx) {
  if (numel < 0) {
    throw nb::value_error((std::string(ctx) + ": negative numel").c_str());
  }
  if (item_b > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(std::string(ctx) + ": itemsize too large");
  }
  const int64_t item_b_i64 = static_cast<int64_t>(item_b);
  int64_t nbytes_i64 = 0;
  if (!vbt::core::checked_mul_i64(numel, item_b_i64, nbytes_i64)) {
    throw std::overflow_error(std::string(ctx) + ": numel*itemsize overflow");
  }
  if (nbytes_i64 < 0) {
    throw std::out_of_range(std::string(ctx) + ": negative nbytes");
  }
  return static_cast<std::size_t>(nbytes_i64);
}

[[nodiscard]] static inline std::vector<int64_t> make_contig_strides_checked(
    const std::vector<int64_t>& sizes,
    const char* ctx) {
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t ui = static_cast<std::size_t>(i);
    strides[ui] = acc;
    const int64_t dim = sizes[ui];
    if (dim < 0) {
      throw nb::value_error((std::string(ctx) + ": negative size").c_str());
    }
    if (dim != 0) {
      int64_t tmp = 0;
      if (!vbt::core::checked_mul_i64(acc, dim, tmp)) {
        throw std::overflow_error(std::string(ctx) + ": stride overflow");
      }
      acc = tmp;
    }
  }
  return strides;
}

void bind_pinned_memory(nb::module_& m) {
  m.def("_cpu_pin_memory", [](const TensorImpl& t) {
    if (t.device().type != kDLCPU) {
      throw nb::type_error("_cpu_pin_memory: expected a CPU tensor");
    }

    nb::gil_scoped_release release;

    const std::size_t item_b = static_cast<std::size_t>(t.itemsize());
    const int64_t ne = numel_or_throw(t, "_cpu_pin_memory");
    const std::size_t nbytes = nbytes_or_throw(item_b, ne, "_cpu_pin_memory");

    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/true);

    std::vector<int64_t> out_sizes = t.sizes();
    std::vector<int64_t> out_strides = make_contig_strides_checked(out_sizes, "_cpu_pin_memory");

    TensorImpl out(storage,
                   std::move(out_sizes),
                   std::move(out_strides),
                   /*storage_offset=*/0,
                   t.dtype(),
                   Device::cpu(t.device().index));

    if (ne == 0) {
      return out;
    }

    const auto& in_storage = t.storage();
    if (!in_storage) {
      throw std::runtime_error("_cpu_pin_memory: input has no storage");
    }
    void* base_void = in_storage->data();
    if (!base_void) {
      throw std::runtime_error("_cpu_pin_memory: input storage has no data");
    }
    const auto* storage_base = static_cast<const std::uint8_t*>(base_void);
    const std::size_t in_nbytes = in_storage->nbytes();

    if (item_b > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::overflow_error("_cpu_pin_memory: itemsize too large");
    }
    const int64_t item_b_i64 = static_cast<int64_t>(item_b);

    // Fast path for contiguous inputs: a single memcpy (validated).
    if (t.is_contiguous()) {
      int64_t base_bytes_i64 = 0;
      if (!vbt::core::checked_mul_i64(t.storage_offset(), item_b_i64, base_bytes_i64)) {
        throw std::overflow_error("_cpu_pin_memory: base byte offset overflow");
      }
      if (base_bytes_i64 < 0) {
        throw std::out_of_range("_cpu_pin_memory: negative base byte offset");
      }

      const std::size_t base_bytes = static_cast<std::size_t>(base_bytes_i64);
      if (base_bytes > in_nbytes || nbytes > (in_nbytes - base_bytes)) {
        throw std::out_of_range("_cpu_pin_memory: input storage too small for contiguous copy");
      }

      std::memcpy(out.data(), storage_base + base_bytes, nbytes);
      return out;
    }

    // General strided -> contiguous elementwise copy (negative strides supported).
    const auto& in_sizes = t.sizes();
    const auto& in_strides = t.strides();
    const std::size_t nd = in_sizes.size();

    std::vector<int64_t> idx(nd, 0);
    auto* out_ptr = static_cast<std::uint8_t*>(out.data());

    for (int64_t linear = 0; linear < ne; ++linear) {
      int64_t abs_off_elems = t.storage_offset();

      for (std::size_t d = 0; d < nd; ++d) {
        int64_t term = 0;
        if (!vbt::core::checked_mul_i64(idx[d], in_strides[d], term)) {
          throw std::overflow_error("_cpu_pin_memory: index*stride overflow");
        }
        if (!vbt::core::checked_add_i64(abs_off_elems, term, abs_off_elems)) {
          throw std::overflow_error("_cpu_pin_memory: offset accumulation overflow");
        }
      }

      if (abs_off_elems < 0) {
        throw std::out_of_range("_cpu_pin_memory: computed negative element offset");
      }

      int64_t byte_off_i64 = 0;
      if (!vbt::core::checked_mul_i64(abs_off_elems, item_b_i64, byte_off_i64)) {
        throw std::overflow_error("_cpu_pin_memory: byte offset overflow");
      }
      if (byte_off_i64 < 0) {
        throw std::out_of_range("_cpu_pin_memory: computed negative byte offset");
      }

      const std::size_t byte_off = static_cast<std::size_t>(byte_off_i64);
      if (byte_off > in_nbytes || item_b > (in_nbytes - byte_off)) {
        throw std::out_of_range("_cpu_pin_memory: input storage out of bounds");
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

    return out;
  });
}

} // namespace vbt_py
