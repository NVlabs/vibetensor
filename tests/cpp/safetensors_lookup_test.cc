// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vbt/interop/safetensors.h"

using vbt::interop::safetensors::DType;
using vbt::interop::safetensors::ErrorCode;
using vbt::interop::safetensors::SafeTensorsError;
using vbt::interop::safetensors::SafeTensorsView;

namespace {
std::vector<std::byte> build_file(std::string_view header,
                                 std::initializer_list<unsigned char> data_bytes) {
  std::vector<std::byte> out;
  out.reserve(static_cast<std::size_t>(8) + header.size() + data_bytes.size());

  const std::uint64_t n = static_cast<std::uint64_t>(header.size());
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::byte>((n >> (8 * i)) & 0xFF));
  }
  for (char ch : header) {
    out.push_back(static_cast<std::byte>(static_cast<unsigned char>(ch)));
  }
  for (unsigned char b : data_bytes) {
    out.push_back(static_cast<std::byte>(b));
  }

  return out;
}
} // namespace

TEST(SafeTensorsLookupTest, TensorLookupByNameReturnsTensorView) {
  const auto buf = build_file(
      R"({"b":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"I32","shape":[1],"data_offsets":[4,8]}})",
      {1, 2, 3, 4, 5, 6, 7, 8});

  SafeTensorsView view =
      SafeTensorsView::deserialize(std::span<const std::byte>(buf.data(), buf.size()));

  EXPECT_EQ(view.data().size(), 8u);

  ASSERT_EQ(view.metadata().tensors_by_offset.size(), 2u);
  EXPECT_EQ(view.metadata().tensors_by_offset[0].name, "b");
  EXPECT_EQ(view.metadata().tensors_by_offset[1].name, "a");

  ASSERT_EQ(view.metadata().name_index.size(), 2u);
  EXPECT_EQ(view.metadata().name_index[0].first, "a");
  EXPECT_EQ(view.metadata().name_index[0].second, 1u);
  EXPECT_EQ(view.metadata().name_index[1].first, "b");
  EXPECT_EQ(view.metadata().name_index[1].second, 0u);

  const auto ta = view.tensor("a");
  EXPECT_EQ(ta.dtype, DType::I32);
  EXPECT_EQ(ta.shape, (std::vector<std::size_t>{1u}));
  EXPECT_EQ(ta.data_offsets.first, 4u);
  EXPECT_EQ(ta.data_offsets.second, 8u);
  ASSERT_EQ(ta.data.size(), 4u);
  EXPECT_EQ(static_cast<unsigned char>(ta.data[0]), 5);
  EXPECT_EQ(static_cast<unsigned char>(ta.data[1]), 6);
  EXPECT_EQ(static_cast<unsigned char>(ta.data[2]), 7);
  EXPECT_EQ(static_cast<unsigned char>(ta.data[3]), 8);

  const auto tb = view.tensor("b");
  EXPECT_EQ(tb.dtype, DType::I32);
  EXPECT_EQ(tb.shape, (std::vector<std::size_t>{1u}));
  EXPECT_EQ(tb.data_offsets.first, 0u);
  EXPECT_EQ(tb.data_offsets.second, 4u);
  ASSERT_EQ(tb.data.size(), 4u);
  EXPECT_EQ(static_cast<unsigned char>(tb.data[0]), 1);
  EXPECT_EQ(static_cast<unsigned char>(tb.data[1]), 2);
  EXPECT_EQ(static_cast<unsigned char>(tb.data[2]), 3);
  EXPECT_EQ(static_cast<unsigned char>(tb.data[3]), 4);
}

TEST(SafeTensorsLookupTest, TensorNotFoundThrows) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[0],"data_offsets":[0,0]}})", {});

  SafeTensorsView view =
      SafeTensorsView::deserialize(std::span<const std::byte>(buf.data(), buf.size()));

  try {
    (void)view.tensor("missing");
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::TensorNotFound);
    EXPECT_EQ(e.tensor_name(), "missing");
  }
}

TEST(SafeTensorsLookupTest, TensorNotFoundNameIsTruncated) {
  const auto buf = build_file(R"({"t":{"dtype":"I32","shape":[0],"data_offsets":[0,0]}})", {});

  SafeTensorsView view =
      SafeTensorsView::deserialize(std::span<const std::byte>(buf.data(), buf.size()));

  const std::string long_name(1000, 'x');
  try {
    (void)view.tensor(long_name);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::TensorNotFound);
    EXPECT_EQ(e.tensor_name().size(), 256u);
    EXPECT_EQ(std::string(e.tensor_name()), long_name.substr(0, 256));
  }
}
