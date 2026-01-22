// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

#include "vbt/interop/safetensors.h"

using vbt::interop::safetensors::DType;
using vbt::interop::safetensors::ErrorCode;
using vbt::interop::safetensors::ParseOptions;
using vbt::interop::safetensors::ParsedMetadata;
using vbt::interop::safetensors::SafeTensorsError;
using vbt::interop::safetensors::read_metadata;

namespace {
std::vector<std::byte> bytes(std::initializer_list<unsigned char> xs) {
  std::vector<std::byte> out;
  out.reserve(xs.size());
  for (auto x : xs) out.push_back(static_cast<std::byte>(x));
  return out;
}

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

void expect_code(std::span<const std::byte> file_bytes, ErrorCode expected, ParseOptions opts = {}) {
  try {
    (void)read_metadata(file_bytes, opts);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), expected);
  }
}

void expect_ok(std::span<const std::byte> file_bytes, ParsedMetadata& out, ParseOptions opts = {}) {
  try {
    out = read_metadata(file_bytes, opts);
  } catch (const SafeTensorsError& e) {
    FAIL() << "unexpected SafeTensorsError: " << static_cast<int>(e.code()) << " (" << e.what() << ")";
  }
}
} // namespace

TEST(SafeTensorsFramingTest, HeaderTooSmall) {
  std::vector<std::byte> buf;
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::HeaderTooSmall);
}

TEST(SafeTensorsFramingTest, HeaderTooLarge) {
  // Mirrors upstream test vector: tensor.rs:test_header_too_large.
  const auto buf = bytes({0x3c, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff});
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::HeaderTooLarge);
}

TEST(SafeTensorsFramingTest, InvalidHeaderLength) {
  // Header length (N) == 60, but file is only 8 bytes long.
  const auto buf = bytes({0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderLength);
}

TEST(SafeTensorsHeaderTest, InvalidHeaderUtf8) {
  // Mirrors upstream test vector: tensor.rs:test_invalid_header_non_utf8.
  const auto buf = bytes({
      0x01,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0xff,
  });
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderUtf8);
}

TEST(SafeTensorsHeaderTest, InvalidHeaderUtf8TruncatedTwoByte) {
  const auto buf = bytes({
      0x01,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0xc2,
  });
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderUtf8);
}

TEST(SafeTensorsHeaderTest, InvalidHeaderUtf8OverlongThreeByte) {
  const auto buf = bytes({
      0x03,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0xe0,
      0x80,
      0x80,
  });
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderUtf8);
}

TEST(SafeTensorsHeaderTest, InvalidHeaderJson) {
  // Mirrors upstream test vector: tensor.rs:test_invalid_header_not_json.
  const auto buf = bytes({
      0x01,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x7b,
  });
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}

TEST(SafeTensorsHeaderTest, WhitespacePaddedHeader) {
  // Mirrors upstream test vector: tensor.rs:test_whitespace_padded_header.
  const auto buf = bytes({
      0x06,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x00,
      0x7b,
      0x7d,
      0x0d,
      0x20,
      0x09,
      0x0a,
  });

  ParsedMetadata parsed;
  expect_ok(std::span<const std::byte>(buf.data(), buf.size()), parsed);
  EXPECT_EQ(parsed.metadata.tensors_by_offset.size(), 0u);
  EXPECT_FALSE(parsed.metadata.user_metadata.has_value());
}

TEST(SafeTensorsHeaderTest, ExtractsMetadataAndTensorSchema) {
  const auto buf = build_file(
      R"({"__metadata__":{"foo":"bar"},"t":{"dtype":"I32","shape":[2,0],"data_offsets":[0,0]}})");

  ParsedMetadata parsed;
  expect_ok(std::span<const std::byte>(buf.data(), buf.size()), parsed);

  ASSERT_TRUE(parsed.metadata.user_metadata.has_value());
  ASSERT_EQ(parsed.metadata.user_metadata->size(), 1u);
  EXPECT_EQ((*parsed.metadata.user_metadata)[0].first, "foo");
  EXPECT_EQ((*parsed.metadata.user_metadata)[0].second, "bar");

  ASSERT_EQ(parsed.metadata.tensors_by_offset.size(), 1u);
  const auto& t = parsed.metadata.tensors_by_offset[0];
  EXPECT_EQ(t.name, "t");
  EXPECT_EQ(t.info.dtype, DType::I32);
  EXPECT_EQ(t.info.shape, (std::vector<std::size_t>{2u, 0u}));
  EXPECT_EQ(t.info.data_offsets.first, 0u);
  EXPECT_EQ(t.info.data_offsets.second, 0u);
}

TEST(SafeTensorsHeaderTest, HeaderStartCurlyOption) {
  const auto buf = build_file("\n{}");

  ParsedMetadata parsed;
  expect_ok(std::span<const std::byte>(buf.data(), buf.size()), parsed);

  ParseOptions strict;
  strict.require_header_start_curly = true;
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderStart, strict);
}

TEST(SafeTensorsHeaderTest, RejectUnknownTensorFieldsOption) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[0],"data_offsets":[0,0],"extra":1}})");

  // Default (upstream parity): ignore unknown fields.
  ParsedMetadata parsed;
  expect_ok(std::span<const std::byte>(buf.data(), buf.size()), parsed);

  ParseOptions strict;
  strict.reject_unknown_tensor_fields = true;
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization, strict);
}

TEST(SafeTensorsHeaderTest, MaxRankOption) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[1,2],"data_offsets":[0,0]}})");
  ParseOptions opts;
  opts.max_rank = 1;
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization, opts);
}

TEST(SafeTensorsHeaderTest, MaxTensorsOption) {
  const auto buf = build_file(
      R"({"a":{"dtype":"I32","shape":[1],"data_offsets":[0,0]},"b":{"dtype":"I32","shape":[1],"data_offsets":[0,0]}})");
  ParseOptions opts;
  opts.max_tensors = 1;
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization, opts);
}

TEST(SafeTensorsHeaderTest, RootMustBeObject) {
  const auto buf = build_file("[]");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}

TEST(SafeTensorsHeaderTest, MetadataMustBeObject) {
  const auto buf = build_file(R"({"__metadata__":1})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}

TEST(SafeTensorsHeaderTest, MetadataValuesMustBeStrings) {
  const auto buf = build_file(R"({"__metadata__":{"k":1}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}

TEST(SafeTensorsHeaderTest, UnknownDtypeToken) {
  const auto buf = build_file(R"({"t":{"dtype":"I666","shape":[1],"data_offsets":[0,0]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}

TEST(SafeTensorsHeaderTest, ShapeMustBeNonNegativeIntegers) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[-1],"data_offsets":[0,0]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}

TEST(SafeTensorsHeaderTest, DataOffsetsMustHaveTwoElements) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[1],"data_offsets":[0]}})");
  expect_code(std::span<const std::byte>(buf.data(), buf.size()), ErrorCode::InvalidHeaderDeserialization);
}
