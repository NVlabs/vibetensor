// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "vbt/core/indexing.h"
#include "vbt/core/indexing/index_errors.h"

namespace vbt {
namespace core {
namespace indexing {

struct EncodedIndexSpecHeader {
  std::int64_t version;
  std::int64_t adv_kind;
  std::int64_t adv_param;
  std::int64_t prefix_len;
};

namespace detail {

inline void throw_meta_invalid_shape(const char* op_name) {
  std::string msg(op_name);
  msg += ": ";
  msg += vbt::core::indexing::errors::kErrMetaInvalidShape;
  throw std::invalid_argument(msg);
}

inline const std::uint8_t* validated_meta_bytes(const vbt::core::TensorImpl& meta,
                                                const char* op_name) {
  if (meta.device().type != kDLCPU ||
      meta.dtype() != vbt::core::ScalarType::Int64 ||
      meta.sizes().size() != 1 ||
      !meta.is_contiguous() ||
      meta.numel() < 4) {
    throw_meta_invalid_shape(op_name);
  }

  const auto storage = meta.storage();
  if (!storage) {
    throw_meta_invalid_shape(op_name);
  }

  const std::int64_t off_el_i64 = meta.storage_offset();
  if (off_el_i64 < 0) {
    throw_meta_invalid_shape(op_name);
  }

  const std::size_t off_el = static_cast<std::size_t>(off_el_i64);
  const std::size_t n_el = static_cast<std::size_t>(meta.numel());

  constexpr std::size_t item_b = sizeof(std::int64_t);

  if (off_el > (std::numeric_limits<std::size_t>::max() / item_b) ||
      n_el > (std::numeric_limits<std::size_t>::max() / item_b)) {
    throw_meta_invalid_shape(op_name);
  }

  const std::size_t off_bytes = off_el * item_b;
  const std::size_t data_bytes = n_el * item_b;
  if (off_bytes > std::numeric_limits<std::size_t>::max() - data_bytes) {
    throw_meta_invalid_shape(op_name);
  }
  const std::size_t end = off_bytes + data_bytes;
  if (end > storage->nbytes()) {
    throw_meta_invalid_shape(op_name);
  }

  const auto* base = static_cast<const std::uint8_t*>(storage->data());
  if (!base && end > 0) {
    throw_meta_invalid_shape(op_name);
  }

  return base + off_bytes;
}

inline std::int64_t load_i64_le(const std::uint8_t* p) {
  std::int64_t out = 0;
  std::memcpy(&out, p, sizeof(out));
  return out;
}

} // namespace detail

inline EncodedIndexSpecHeader decode_encoded_index_spec_header_v0(
    const vbt::core::TensorImpl& meta, const char* op_name) {
  const std::uint8_t* base = detail::validated_meta_bytes(meta, op_name);

  EncodedIndexSpecHeader h{
      detail::load_i64_le(base + 0 * sizeof(std::int64_t)),
      detail::load_i64_le(base + 1 * sizeof(std::int64_t)),
      detail::load_i64_le(base + 2 * sizeof(std::int64_t)),
      detail::load_i64_le(base + 3 * sizeof(std::int64_t)),
  };

  // Basic prefix_len / length validation (overflow-safe)
  if (h.prefix_len < 0) {
    detail::throw_meta_invalid_shape(op_name);
  }

  std::int64_t expected_elems = 4;
  if (h.prefix_len > 0) {
    const std::int64_t max_prefix =
        (std::numeric_limits<std::int64_t>::max() - 4) / 4;
    if (h.prefix_len > max_prefix) {
      detail::throw_meta_invalid_shape(op_name);
    }
    expected_elems = 4 + 4 * h.prefix_len;
  }

  if (meta.numel() != expected_elems) {
    detail::throw_meta_invalid_shape(op_name);
  }

  if (h.version != 0) {
    std::string msg(op_name);
    msg += ": ";
    msg += vbt::core::indexing::errors::kErrMetaUnsupportedVersion;
    throw std::invalid_argument(msg);
  }

  if (h.adv_kind != 0 && h.adv_kind != 1) {
    std::string msg(op_name);
    msg += ": invalid advanced_kind in meta";
    throw std::invalid_argument(msg);
  }

  if (h.adv_kind == 0 && h.adv_param != 0 && h.adv_param != 1) {
    std::string msg(op_name);
    msg += ": invalid ScalarBool adv_param in meta";
    throw std::invalid_argument(msg);
  }

  if (h.adv_kind == 1 && h.adv_param != 0) {
    std::string msg(op_name);
    msg += ": invalid Tensor adv_param in meta";
    throw std::invalid_argument(msg);
  }

  return h;
}

inline IndexSpec decode_encoded_index_spec_v0(
    const vbt::core::TensorImpl& index_tensor,
    const vbt::core::TensorImpl& meta,
    const char* op_name) {
  const EncodedIndexSpecHeader hdr =
      decode_encoded_index_spec_header_v0(meta, op_name);

  const std::uint8_t* base = detail::validated_meta_bytes(meta, op_name);

  constexpr std::int64_t kSentinel = std::numeric_limits<std::int64_t>::min();

  auto decode_optional = [&](std::int64_t v) -> std::optional<std::int64_t> {
    if (v == kSentinel) return std::nullopt;
    return v;
  };

  IndexSpec spec;
  spec.items.reserve(static_cast<std::size_t>(hdr.prefix_len + 1));

  for (std::int64_t i = 0; i < hdr.prefix_len; ++i) {
    const std::size_t rec_base = static_cast<std::size_t>(4 + 4 * i);

    const std::int64_t kind_tag = detail::load_i64_le(
        base + (rec_base + 0) * sizeof(std::int64_t));
    const std::int64_t a = detail::load_i64_le(
        base + (rec_base + 1) * sizeof(std::int64_t));
    const std::int64_t b = detail::load_i64_le(
        base + (rec_base + 2) * sizeof(std::int64_t));
    const std::int64_t c = detail::load_i64_le(
        base + (rec_base + 3) * sizeof(std::int64_t));

    if (kind_tag == 0) {
      spec.items.emplace_back(TensorIndex(nullptr));
      continue;
    }

    if (kind_tag == 1) {
      spec.items.emplace_back(TensorIndex(a));
      continue;
    }

    if (kind_tag == 2) {
      Slice s;
      s.start = decode_optional(a);
      s.stop = decode_optional(b);
      s.step = decode_optional(c);
      spec.items.emplace_back(TensorIndex(s));
      continue;
    }

    std::string msg(op_name);
    msg += ": invalid prefix index kind in meta";
    throw std::invalid_argument(msg);
  }

  if (hdr.adv_kind == 0) {
    // Scalar-bool advanced indices are intentionally unsupported in the
    // invalid encoded specs so that core scalar-bool paths are unreachable
    // from Python surfaces.
    std::string msg(op_name);
    msg += ": scalar-bool advanced indexing is not supported";
    throw std::invalid_argument(msg);
  }

  // adv_kind == 1: Tensor advanced index; index_tensor must be defined.
  if (!index_tensor.storage().get()) {
    std::string msg(op_name);
    msg += ": index tensor must be defined for tensor advanced index";
    throw std::invalid_argument(msg);
  }

  spec.items.emplace_back(TensorIndex(index_tensor));
  return spec;
}

}}} // namespace vbt::core::indexing
