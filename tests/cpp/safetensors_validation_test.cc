// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vbt/interop/safetensors.h"

using vbt::interop::safetensors::ErrorCode;
using vbt::interop::safetensors::ParsedMetadata;
using vbt::interop::safetensors::SafeTensorsError;
using vbt::interop::safetensors::read_metadata;

namespace {
std::vector<std::byte> build_file(std::string_view header) {
  std::vector<std::byte> out;
  out.reserve(static_cast<std::size_t>(8) + header.size());

  const std::uint64_t n = static_cast<std::uint64_t>(header.size());
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((n >> (8 * i)) & 0xFF));
  }
  for (char ch : header) {
    out.push_back(static_cast<std::byte>(static_cast<unsigned char>(ch)));
  }
  return out;
}

void append_zeros(std::vector<std::byte>& buf, std::size_t n) {
  buf.insert(buf.end(), n, static_cast<std::byte>(0));
}

void expect_code(std::span<const std::byte> file_bytes,
                 ErrorCode expected,
                 std::optional<std::string_view> expected_tensor_name = std::nullopt) {
  try {
    (void)read_metadata(file_bytes);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), expected);
    if (expected_tensor_name.has_value()) {
      EXPECT_EQ(e.tensor_name(), *expected_tensor_name);
    }
  }
}

void expect_ok(std::span<const std::byte> file_bytes, ParsedMetadata& out) {
  try {
    out = read_metadata(file_bytes);
  } catch (const SafeTensorsError& e) {
    FAIL() << "unexpected SafeTensorsError: " << static_cast<int>(e.code()) << " (" << e.what()
           << ")";
  }
}
} // namespace

TEST(SafeTensorsValidationTest, TensorInvalidInfoSizeMismatch) {
  // DType I32 + shape [1] requires 4 bytes, but offsets span is empty.
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[1],"data_offsets":[0,0]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::TensorInvalidInfo, "t");
}

TEST(SafeTensorsValidationTest, MetadataIncompleteBufferMissingData) {
  // Metadata says there should be 4 bytes of data, but the file ends at the header.
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::MetadataIncompleteBuffer,
              std::string_view{});
}

TEST(SafeTensorsValidationTest, MetadataIncompleteBufferTrailingData) {
  // Metadata covers 4 bytes but file has extra trailing bytes.
  auto buf = build_file(R"({"t":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})");
  append_zeros(buf, 8);
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::MetadataIncompleteBuffer,
              std::string_view{});
}

TEST(SafeTensorsValidationTest, InvalidOffsetGap) {
  // Second tensor begins at 8 but the previous ended at 4.
  const auto buf = build_file(
      R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"I32","shape":[1],"data_offsets":[8,12]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidOffset, "b");
}

TEST(SafeTensorsValidationTest, InvalidOffsetOverlap) {
  // Second tensor overlaps (starts at 4) but previous ended at 8.
  const auto buf = build_file(
      R"({"a":{"dtype":"I32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"I32","shape":[1],"data_offsets":[4,8]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidOffset, "b");
}

TEST(SafeTensorsValidationTest, InvalidOffsetReversedOffsets) {
  // Second tensor starts at the correct location (4) but ends before it begins.
  const auto buf = build_file(
      R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"I32","shape":[1],"data_offsets":[4,3]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidOffset, "b");
}

TEST(SafeTensorsValidationTest, InvalidOffsetFirstTensorStartsNonZero) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[1],"data_offsets":[1,5]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidOffset, "t");
}

TEST(SafeTensorsValidationTest, MisalignedSliceSubByteF4) {
  // F4 uses 4 bits per element; 1 element is not byte-aligned.
  const auto buf = build_file(R"({"t":{"dtype":"F4","shape":[1],"data_offsets":[0,0]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::MisalignedSlice,
              std::string_view{});
}

TEST(SafeTensorsValidationTest, ValidationOverflow) {
  // shape product overflows size_t.
  const std::string header =
      std::string("{\"t\":{\"dtype\":\"U8\",\"shape\":[") +
      std::to_string(std::numeric_limits<std::size_t>::max()) +
      ",2],\"data_offsets\":[0,0]}}";
  const auto buf = build_file(header);
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::ValidationOverflow, "t");
}

TEST(SafeTensorsValidationTest, ValidationOverflowBitSize) {
  // shape fits, but (nelems * dtype_bits) overflows size_t.
  const std::string header =
      std::string("{\"t\":{\"dtype\":\"U16\",\"shape\":[") +
      std::to_string(std::numeric_limits<std::size_t>::max()) +
      "],\"data_offsets\":[0,0]}}";
  const auto buf = build_file(header);
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::ValidationOverflow, "t");
}

TEST(SafeTensorsValidationTest, AlignedSubByteF4IsAccepted) {
  auto buf = build_file(R"({"t":{"dtype":"F4","shape":[2],"data_offsets":[0,1]}})");
  append_zeros(buf, 1);

  ParsedMetadata parsed;
  expect_ok(std::span<const std::byte>(buf.data(), buf.size()), parsed);
  EXPECT_EQ(parsed.metadata.data_len_bytes, 1u);
  ASSERT_EQ(parsed.metadata.tensors_by_offset.size(), 1u);
  EXPECT_EQ(parsed.metadata.tensors_by_offset[0].name, "t");
}
