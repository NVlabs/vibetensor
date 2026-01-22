// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/interop/safetensors/safetensors.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#if VBT_WITH_SAFETENSORS
#include <nlohmann/json.hpp>
#endif

namespace vbt {
namespace interop {
namespace safetensors {

#if VBT_WITH_SAFETENSORS

namespace {
inline std::uint64_t read_u64_le(const std::byte* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<std::uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i));
  }
  return v;
}

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

[[nodiscard]] inline bool parse_dtype_token(std::string_view token, DType& out) {
  if (token == "BOOL") {
    out = DType::BOOL;
    return true;
  }
  if (token == "F4") {
    out = DType::F4;
    return true;
  }
  if (token == "F6_E2M3") {
    out = DType::F6_E2M3;
    return true;
  }
  if (token == "F6_E3M2") {
    out = DType::F6_E3M2;
    return true;
  }
  if (token == "U8") {
    out = DType::U8;
    return true;
  }
  if (token == "I8") {
    out = DType::I8;
    return true;
  }
  if (token == "F8_E5M2") {
    out = DType::F8_E5M2;
    return true;
  }
  if (token == "F8_E4M3") {
    out = DType::F8_E4M3;
    return true;
  }
  if (token == "F8_E8M0") {
    out = DType::F8_E8M0;
    return true;
  }
  if (token == "I16") {
    out = DType::I16;
    return true;
  }
  if (token == "U16") {
    out = DType::U16;
    return true;
  }
  if (token == "F16") {
    out = DType::F16;
    return true;
  }
  if (token == "BF16") {
    out = DType::BF16;
    return true;
  }
  if (token == "I32") {
    out = DType::I32;
    return true;
  }
  if (token == "U32") {
    out = DType::U32;
    return true;
  }
  if (token == "F32") {
    out = DType::F32;
    return true;
  }
  if (token == "C64") {
    out = DType::C64;
    return true;
  }
  if (token == "F64") {
    out = DType::F64;
    return true;
  }
  if (token == "I64") {
    out = DType::I64;
    return true;
  }
  if (token == "U64") {
    out = DType::U64;
    return true;
  }
  return false;
}

[[nodiscard]] inline bool json_to_size_t(const nlohmann::json& j, std::size_t& out) {
  if (j.is_number_unsigned()) {
    const std::uint64_t v = j.get<std::uint64_t>();
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) return false;
    out = static_cast<std::size_t>(v);
    return true;
  }
  if (j.is_number_integer()) {
    const std::int64_t v = j.get<std::int64_t>();
    if (v < 0) return false;
    if (static_cast<std::uint64_t>(v) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return false;
    }
    out = static_cast<std::size_t>(v);
    return true;
  }
  return false;
}

[[nodiscard]] inline Metadata parse_header_metadata(std::span<const std::byte> header_bytes,
                                                   ParseOptions opts) {
  using json = nlohmann::json;

  // Parse directly from the header byte range to avoid an extra copy.
  const char* begin =
      header_bytes.empty() ? "" : reinterpret_cast<const char*>(header_bytes.data());
  const char* end = begin + header_bytes.size();

  json root = json::parse(begin, end, nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded()) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                           "safetensors: invalid JSON in header");
  }
  if (!root.is_object()) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                           "safetensors: header root must be a JSON object");
  }

  Metadata md;

  std::size_t tensor_count = 0;
  for (auto it = root.begin(); it != root.end(); ++it) {
    const std::string& name = it.key();
    const json& value = it.value();

    if (name == "__metadata__") {
      if (!value.is_object()) {
        throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                               "safetensors: __metadata__ must be an object");
      }

      std::vector<std::pair<std::string, std::string>> user;
      user.reserve(value.size());
      for (auto mit = value.begin(); mit != value.end(); ++mit) {
        if (!mit.value().is_string()) {
          throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                                 "safetensors: __metadata__ values must be strings");
        }
        user.emplace_back(mit.key(), mit.value().get<std::string>());
      }
      std::sort(user.begin(), user.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
      });
      md.user_metadata = std::move(user);
      continue;
    }

    ++tensor_count;
    if (opts.max_tensors && tensor_count > *opts.max_tensors) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: too many tensors in header");
    }

    if (!value.is_object()) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: tensor entry must be an object", name);
    }

    if (opts.reject_unknown_tensor_fields) {
      for (auto field_it = value.begin(); field_it != value.end(); ++field_it) {
        const std::string& field = field_it.key();
        if (field != "dtype" && field != "shape" && field != "data_offsets") {
          throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                                 "safetensors: unknown field in tensor entry", name);
        }
      }
    }

    const auto dtype_it = value.find("dtype");
    if (dtype_it == value.end() || !dtype_it->is_string()) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: missing or invalid dtype in tensor entry", name);
    }
    const std::string& dtype_token = dtype_it->get_ref<const std::string&>();
    DType dtype;
    if (!parse_dtype_token(dtype_token, dtype)) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: unknown dtype token", name);
    }

    const auto shape_it = value.find("shape");
    if (shape_it == value.end() || !shape_it->is_array()) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: missing or invalid shape in tensor entry", name);
    }
    if (opts.max_rank && shape_it->size() > *opts.max_rank) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: tensor rank exceeds configured maximum", name);
    }

    std::vector<std::size_t> shape;
    shape.reserve(shape_it->size());
    for (const auto& dim : *shape_it) {
      std::size_t v = 0;
      if (!json_to_size_t(dim, v)) {
        throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                               "safetensors: invalid shape element in tensor entry", name);
      }
      shape.push_back(v);
    }

    const auto offsets_it = value.find("data_offsets");
    if (offsets_it == value.end() || !offsets_it->is_array() || offsets_it->size() != 2) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: missing or invalid data_offsets in tensor entry",
                             name);
    }

    std::size_t begin = 0;
    std::size_t end = 0;
    if (!json_to_size_t((*offsets_it)[0], begin) || !json_to_size_t((*offsets_it)[1], end)) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderDeserialization,
                             "safetensors: invalid data_offsets in tensor entry", name);
    }

    TensorEntry entry;
    entry.name = name;
    entry.info.dtype = dtype;
    entry.info.shape = std::move(shape);
    entry.info.data_offsets = {begin, end};
    md.tensors_by_offset.push_back(std::move(entry));
  }

  std::sort(md.tensors_by_offset.begin(), md.tensors_by_offset.end(),
            [](const TensorEntry& a, const TensorEntry& b) {
              const auto [a0, a1] = a.info.data_offsets;
              const auto [b0, b1] = b.info.data_offsets;
              if (a0 != b0) return a0 < b0;
              if (a1 != b1) return a1 < b1;
              return a.name < b.name;
            });

  return md;
}

inline void validate_metadata(Metadata& md, std::size_t data_bytes_size) {
  std::size_t start = 0;
  for (const TensorEntry& entry : md.tensors_by_offset) {
    const auto [s, e] = entry.info.data_offsets;
    if (s != start || e < s) {
      throw SafeTensorsError(ErrorCode::InvalidOffset, "safetensors: invalid data_offsets", entry.name);
    }

    std::size_t nelems = 0;
    if (!checked_product(std::span<const std::size_t>(entry.info.shape.data(), entry.info.shape.size()),
                         nelems)) {
      throw SafeTensorsError(ErrorCode::ValidationOverflow,
                             "safetensors: tensor element count overflow", entry.name);
    }

    std::size_t nbits = 0;
    if (!checked_mul_size(nelems, dtype_bits(entry.info.dtype), nbits)) {
      throw SafeTensorsError(ErrorCode::ValidationOverflow, "safetensors: tensor bit size overflow",
                             entry.name);
    }
    if ((nbits % 8) != 0) {
      throw SafeTensorsError(ErrorCode::MisalignedSlice,
                             "safetensors: misaligned sub-byte tensor slice");
    }

    const std::size_t nbytes = nbits / 8;
    if ((e - s) != nbytes) {
      throw SafeTensorsError(ErrorCode::TensorInvalidInfo,
                             "safetensors: tensor info does not match data_offsets span", entry.name);
    }

    start = e;
  }

  md.data_len_bytes = start;
  if (md.data_len_bytes != data_bytes_size) {
    throw SafeTensorsError(ErrorCode::MetadataIncompleteBuffer,
                           "safetensors: metadata does not cover the full data buffer");
  }

  md.name_index.clear();
  md.name_index.reserve(md.tensors_by_offset.size());
  for (std::size_t i = 0; i < md.tensors_by_offset.size(); ++i) {
    md.name_index.emplace_back(md.tensors_by_offset[i].name, i);
  }
  std::sort(md.name_index.begin(), md.name_index.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second < b.second;
  });
}
} // namespace

ParsedMetadata read_metadata(std::span<const std::byte> file_bytes, ParseOptions opts) {
  if (file_bytes.size() < 8) {
    throw SafeTensorsError(ErrorCode::HeaderTooSmall, "safetensors: header too small");
  }

  const std::uint64_t n_u64 = read_u64_le(file_bytes.data());
  if (n_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw SafeTensorsError(ErrorCode::HeaderTooLarge, "safetensors: header length exceeds SIZE_MAX");
  }

  const std::size_t n = static_cast<std::size_t>(n_u64);
  const std::size_t max_header = std::min(opts.max_header_size_bytes, kUpstreamMaxHeaderSizeBytes);
  if (n > max_header) {
    throw SafeTensorsError(ErrorCode::HeaderTooLarge, "safetensors: header too large");
  }

  std::size_t header_end = 0;
  if (!checked_add_size(static_cast<std::size_t>(8), n, header_end)) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderLength, "safetensors: invalid header length");
  }
  if (header_end > file_bytes.size()) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderLength, "safetensors: invalid header length");
  }

  const auto header_bytes = file_bytes.subspan(8, n);
  if (!is_valid_utf8(header_bytes)) {
    throw SafeTensorsError(ErrorCode::InvalidHeaderUtf8, "safetensors: invalid UTF-8 in header");
  }
  if (opts.require_header_start_curly) {
    if (header_bytes.empty() ||
        static_cast<unsigned char>(header_bytes[0]) != static_cast<unsigned char>('{')) {
      throw SafeTensorsError(ErrorCode::InvalidHeaderStart, "safetensors: invalid start character in header");
    }
  }

  ParsedMetadata out;
  out.header_n_bytes = n;
  out.header_end = header_end;
  out.metadata = parse_header_metadata(header_bytes, opts);
  const auto data_bytes = file_bytes.subspan(header_end);
  validate_metadata(out.metadata, data_bytes.size());
  return out;
}

#else

ParsedMetadata read_metadata(std::span<const std::byte> /*file_bytes*/, ParseOptions /*opts*/) {
  throw SafeTensorsError(ErrorCode::IoError, "safetensors: support disabled at build time");
}

#endif

SafeTensorsView SafeTensorsView::deserialize(std::span<const std::byte> file_bytes,
                                            ParseOptions opts) {
  ParsedMetadata parsed = read_metadata(file_bytes, opts);
  SafeTensorsView out;
  out.metadata_ = std::move(parsed.metadata);
  out.data_ = file_bytes.subspan(parsed.header_end);
  return out;
}

const Metadata& SafeTensorsView::metadata() const noexcept { return metadata_; }

std::span<const std::byte> SafeTensorsView::data() const noexcept { return data_; }

TensorView SafeTensorsView::tensor(std::string_view name) const {
  const auto& idx = metadata_.name_index;
  auto it = std::lower_bound(idx.begin(), idx.end(), name,
                             [](const auto& kv, std::string_view key) {
                               return kv.first < key;
                             });
  if (it == idx.end() || it->first != name) {
    throw SafeTensorsError(ErrorCode::TensorNotFound, "safetensors: tensor not found", name);
  }

  const std::size_t tensor_idx = it->second;
  const TensorEntry& entry = metadata_.tensors_by_offset[tensor_idx];
  const auto [begin, end] = entry.info.data_offsets;
  if (begin > end || end > data_.size()) {
    throw SafeTensorsError(ErrorCode::InvalidOffset, "safetensors: invalid data_offsets", entry.name);
  }

  TensorView out;
  out.dtype = entry.info.dtype;
  out.shape = entry.info.shape;
  out.data_offsets = entry.info.data_offsets;
  out.data = data_.subspan(begin, end - begin);
  return out;
}

} // namespace safetensors
} // namespace interop
} // namespace vbt
