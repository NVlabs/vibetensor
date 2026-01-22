// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/interop/safetensors/safetensors.h"

#include "serialize_prep.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if VBT_WITH_SAFETENSORS
#include <nlohmann/json.hpp>
#endif

namespace vbt {
namespace interop {
namespace safetensors {

#if VBT_WITH_SAFETENSORS

namespace {
[[nodiscard]] inline bool checked_add_size(std::size_t a, std::size_t b, std::size_t& out) {
  if (a > (std::numeric_limits<std::size_t>::max() - b)) return false;
  out = a + b;
  return true;
}

[[nodiscard]] inline bool checked_mul_size(std::size_t a, std::size_t b, std::size_t& out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > (std::numeric_limits<std::size_t>::max() / b)) return false;
  out = a * b;
  return true;
}

[[nodiscard]] inline bool checked_product(std::span<const std::size_t> xs, std::size_t& out) {
  std::size_t prod = 1;
  for (const std::size_t x : xs) {
    std::size_t tmp = 0;
    if (!checked_mul_size(prod, x, tmp)) return false;
    prod = tmp;
  }
  out = prod;
  return true;
}

[[nodiscard]] inline bool is_valid_utf8(std::span<const std::byte> bytes) {
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const std::size_t n = bytes.size();

  std::size_t i = 0;
  while (i < n) {
    const unsigned char c = p[i];
    if (c <= 0x7F) {
      ++i;
      continue;
    }

    // 2-byte sequence.
    if (c >= 0xC2 && c <= 0xDF) {
      if (i + 1 >= n) return false;
      const unsigned char c1 = p[i + 1];
      if ((c1 & 0xC0) != 0x80) return false;
      i += 2;
      continue;
    }

    // 3-byte sequences.
    if (c == 0xE0) {
      if (i + 2 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      if (c1 < 0xA0 || c1 > 0xBF) return false; // disallow overlongs
      if ((c2 & 0xC0) != 0x80) return false;
      i += 3;
      continue;
    }
    if (c >= 0xE1 && c <= 0xEC) {
      if (i + 2 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      if ((c1 & 0xC0) != 0x80) return false;
      if ((c2 & 0xC0) != 0x80) return false;
      i += 3;
      continue;
    }
    if (c == 0xED) {
      if (i + 2 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      if (c1 < 0x80 || c1 > 0x9F) return false; // disallow UTF-16 surrogate halves
      if ((c2 & 0xC0) != 0x80) return false;
      i += 3;
      continue;
    }
    if (c >= 0xEE && c <= 0xEF) {
      if (i + 2 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      if ((c1 & 0xC0) != 0x80) return false;
      if ((c2 & 0xC0) != 0x80) return false;
      i += 3;
      continue;
    }

    // 4-byte sequences.
    if (c == 0xF0) {
      if (i + 3 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      const unsigned char c3 = p[i + 3];
      if (c1 < 0x90 || c1 > 0xBF) return false; // disallow overlongs
      if ((c2 & 0xC0) != 0x80) return false;
      if ((c3 & 0xC0) != 0x80) return false;
      i += 4;
      continue;
    }
    if (c >= 0xF1 && c <= 0xF3) {
      if (i + 3 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      const unsigned char c3 = p[i + 3];
      if ((c1 & 0xC0) != 0x80) return false;
      if ((c2 & 0xC0) != 0x80) return false;
      if ((c3 & 0xC0) != 0x80) return false;
      i += 4;
      continue;
    }
    if (c == 0xF4) {
      if (i + 3 >= n) return false;
      const unsigned char c1 = p[i + 1];
      const unsigned char c2 = p[i + 2];
      const unsigned char c3 = p[i + 3];
      if (c1 < 0x80 || c1 > 0x8F) return false; // max codepoint U+10FFFF
      if ((c2 & 0xC0) != 0x80) return false;
      if ((c3 & 0xC0) != 0x80) return false;
      i += 4;
      continue;
    }

    return false;
  }

  return true;
}

inline void write_u64_le(std::uint64_t v, std::vector<std::byte>& out) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
  }
}

[[nodiscard]] inline std::string_view dtype_token(DType dt) {
  switch (dt) {
    case DType::BOOL: return "BOOL";
    case DType::F4: return "F4";
    case DType::F6_E2M3: return "F6_E2M3";
    case DType::F6_E3M2: return "F6_E3M2";
    case DType::U8: return "U8";
    case DType::I8: return "I8";
    case DType::F8_E5M2: return "F8_E5M2";
    case DType::F8_E4M3: return "F8_E4M3";
    case DType::F8_E8M0: return "F8_E8M0";
    case DType::I16: return "I16";
    case DType::U16: return "U16";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::I32: return "I32";
    case DType::U32: return "U32";
    case DType::F32: return "F32";
    case DType::C64: return "C64";
    case DType::F64: return "F64";
    case DType::I64: return "I64";
    case DType::U64: return "U64";
  }
  return "";
}

inline std::vector<std::pair<std::string, std::string>> sorted_user_metadata(
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata) {
  if (!user_metadata.has_value()) return {};

  std::vector<std::pair<std::string, std::string>> out = *user_metadata;
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
  });

  for (std::size_t i = 1; i < out.size(); ++i) {
    if (out[i - 1].first == out[i].first) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: duplicate __metadata__ key");
    }
  }

  return out;
}
} // namespace

namespace detail {

[[nodiscard]] PreparedSerialization prepare_serialization(
    std::span<const TensorEntry> tensors,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata,
    SerializeOptions opts) {
  const std::size_t max_header = std::min(opts.max_header_size_bytes, kUpstreamMaxHeaderSizeBytes);

  // Detect duplicate tensor names early (JSON objects cannot represent duplicates
  // unambiguously).
  {
    std::vector<std::string_view> names;
    names.reserve(tensors.size());
    for (const auto& t : tensors) {
      names.push_back(t.name);
    }
    std::sort(names.begin(), names.end());
    const auto dup = std::adjacent_find(names.begin(), names.end());
    if (dup != names.end()) {
      throw SafeTensorsError(ErrorCode::TensorInvalidInfo, "safetensors: duplicate tensor name",
                             *dup);
    }
  }

  // Determine serialization order.
  std::vector<std::size_t> order;
  order.reserve(tensors.size());
  for (std::size_t i = 0; i < tensors.size(); ++i) order.push_back(i);

  if (opts.sort_by_dtype_alignment_then_name) {
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      const TensorEntry& ta = tensors[a];
      const TensorEntry& tb = tensors[b];
      if (ta.info.dtype != tb.info.dtype) return ta.info.dtype > tb.info.dtype;
      return ta.name < tb.name;
    });
  }

  // Compute offsets.
  std::vector<PreparedTensor> prepared;
  prepared.reserve(order.size());

  std::size_t offset = 0;
  std::size_t actual_data_bytes = 0;

  for (const std::size_t idx : order) {
    const TensorEntry& t = tensors[idx];

    if (t.name == "__metadata__") {
      throw SafeTensorsError(ErrorCode::TensorInvalidInfo,
                             "safetensors: tensor name '__metadata__' is reserved", t.name);
    }

    std::size_t nelems = 0;
    if (!checked_product(std::span<const std::size_t>(t.info.shape.data(), t.info.shape.size()),
                         nelems)) {
      throw SafeTensorsError(ErrorCode::ValidationOverflow,
                             "safetensors: tensor element count overflow", t.name);
    }

    std::size_t nbits = 0;
    if (!checked_mul_size(nelems, dtype_bits(t.info.dtype), nbits)) {
      throw SafeTensorsError(ErrorCode::ValidationOverflow, "safetensors: tensor bit size overflow",
                             t.name);
    }
    if ((nbits % 8) != 0) {
      throw SafeTensorsError(ErrorCode::MisalignedSlice,
                             "safetensors: misaligned sub-byte tensor slice");
    }

    const std::size_t nbytes = nbits / 8;
    if (opts.validate_tensor_sizes && t.data.size() != nbytes) {
      throw SafeTensorsError(ErrorCode::TensorInvalidInfo,
                             "safetensors: tensor data size does not match dtype and shape",
                             t.name);
    }

    std::size_t end = 0;
    if (!checked_add_size(offset, nbytes, end)) {
      throw SafeTensorsError(ErrorCode::ValidationOverflow, "safetensors: tensor offsets overflow",
                             t.name);
    }

    std::size_t tmp_actual = 0;
    if (!checked_add_size(actual_data_bytes, t.data.size(), tmp_actual)) {
      throw SafeTensorsError(ErrorCode::ValidationOverflow, "safetensors: output size overflow");
    }
    actual_data_bytes = tmp_actual;

    PreparedTensor pt;
    pt.entry = &t;
    pt.begin = offset;
    pt.end = end;
    prepared.push_back(pt);
    offset = end;
  }

  // Construct header JSON.
  using ordered_json = nlohmann::ordered_json;

  ordered_json root = ordered_json::object();

  if (user_metadata.has_value()) {
    ordered_json meta = ordered_json::object();
    for (const auto& kv : sorted_user_metadata(user_metadata)) {
      meta[kv.first] = kv.second;
    }
    root["__metadata__"] = std::move(meta);
  }

  for (const auto& pt : prepared) {
    const TensorEntry& t = *pt.entry;

    ordered_json tinfo = ordered_json::object();
    tinfo["dtype"] = std::string(dtype_token(t.info.dtype));
    tinfo["shape"] = t.info.shape;
    tinfo["data_offsets"] = {pt.begin, pt.end};

    root[t.name] = std::move(tinfo);
  }

  std::string header_str;
  try {
    header_str = root.dump();
  } catch (...) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                           "safetensors: header JSON serialization failed");
  }

  std::vector<std::byte> header_bytes;
  header_bytes.reserve(header_str.size());
  for (const char ch : header_str) {
    header_bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(ch)));
  }

  if (!is_valid_utf8(std::span<const std::byte>(header_bytes.data(), header_bytes.size()))) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderUtf8,
                           "safetensors: invalid UTF-8 in serialized header");
  }

  // Pad to a multiple of 8 bytes with spaces.
  std::size_t padded_header_size = header_bytes.size();
  const std::size_t rem = padded_header_size % kHeaderPadBytes;
  if (rem != 0) {
    const std::size_t pad = kHeaderPadBytes - rem;
    if (!checked_add_size(padded_header_size, pad, padded_header_size)) {
      throw SafeTensorsError(ErrorCode::HeaderTooLarge, "safetensors: header too large");
    }
  }
  header_bytes.resize(padded_header_size, static_cast<std::byte>(0x20));

  if (padded_header_size > max_header) {
    throw SafeTensorsError(ErrorCode::HeaderTooLarge, "safetensors: header too large");
  }

  PreparedSerialization out;
  out.header_bytes = std::move(header_bytes);
  out.tensors = std::move(prepared);
  out.expected_data_bytes = offset;
  out.actual_data_bytes = actual_data_bytes;
  return out;
}

} // namespace detail

std::vector<std::byte> serialize(
    std::span<const TensorEntry> tensors,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata,
    SerializeOptions opts) {
  detail::PreparedSerialization prep = detail::prepare_serialization(tensors, user_metadata, opts);

  // Pre-size checks (overflow safety).
  std::size_t total_size = 0;
  std::size_t tmp = 0;
  if (!checked_add_size(static_cast<std::size_t>(8), prep.header_bytes.size(), tmp) ||
      !checked_add_size(tmp, prep.actual_data_bytes, total_size)) {
    throw SafeTensorsError(ErrorCode::ValidationOverflow, "safetensors: output size overflow");
  }

  std::vector<std::byte> out;
  out.reserve(total_size);

  write_u64_le(static_cast<std::uint64_t>(prep.header_bytes.size()), out);
  out.insert(out.end(), prep.header_bytes.begin(), prep.header_bytes.end());

  for (const auto& pt : prep.tensors) {
    const auto& data = pt.entry->data;
    out.insert(out.end(), data.begin(), data.end());
  }

  return out;
}

#else

std::vector<std::byte> serialize(
    std::span<const TensorEntry> /*tensors*/,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& /*user_metadata*/,
    SerializeOptions /*opts*/) {
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: support disabled at build time");
}

#endif

} // namespace safetensors
} // namespace interop
} // namespace vbt
