// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vbt/interop/safetensors.h"

using vbt::interop::safetensors::DType;
using vbt::interop::safetensors::ErrorCode;
using vbt::interop::safetensors::FileLoadMode;
using vbt::interop::safetensors::FileOpenOptions;
using vbt::interop::safetensors::FileWriteOptions;
using vbt::interop::safetensors::SafeTensorsError;
using vbt::interop::safetensors::SafeTensorsFile;
using vbt::interop::safetensors::TensorEntry;
using vbt::interop::safetensors::serialize;
using vbt::interop::safetensors::serialize_to_file;

namespace {

std::vector<std::byte> bytes(std::initializer_list<unsigned char> xs) {
  std::vector<std::byte> out;
  out.reserve(xs.size());
  for (auto x : xs) out.push_back(static_cast<std::byte>(x));
  return out;
}

TensorEntry make_tensor(std::string name,
                        DType dtype,
                        std::vector<std::size_t> shape,
                        const std::vector<std::byte>& data) {
  TensorEntry t;
  t.name = std::move(name);
  t.info.dtype = dtype;
  t.info.shape = std::move(shape);
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(data.data(), data.size());
  return t;
}

std::filesystem::path make_temp_dir(const std::string& prefix) {
  namespace fs = std::filesystem;
  const fs::path base = fs::temp_directory_path();
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();

  for (int i = 0; i < 100; ++i) {
    fs::path p = base / (prefix + std::to_string(now) + "_" + std::to_string(i));
    std::error_code ec;
    if (fs::create_directory(p, ec)) {
      return p;
    }
  }
  throw std::runtime_error("failed to create temp dir");
}

std::vector<std::byte> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open file");
  }
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);

  if (n < 0) {
    throw std::runtime_error("invalid file size");
  }

  std::vector<std::byte> out(static_cast<std::size_t>(n));
  if (n > 0) {
    f.read(reinterpret_cast<char*>(out.data()), n);
  }
  if (!f) {
    throw std::runtime_error("failed to read file");
  }
  return out;
}

} // namespace

TEST(SafeTensorsFileIoTest, SerializeToFileAndOpenReadAllRoundTrip) {
  namespace fs = std::filesystem;
  const fs::path dir = make_temp_dir("vbt_safetensors_io_");

  struct DirGuard {
    fs::path p;
    explicit DirGuard(fs::path p_) : p(std::move(p_)) {}
    ~DirGuard() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } guard(dir);

  const std::vector<std::byte> a_data = bytes({1, 2, 3, 4});
  const std::vector<std::byte> b_data = bytes({5, 6, 7, 8});

  std::vector<TensorEntry> tensors;
  tensors.push_back(make_tensor("b", DType::I32, {1}, b_data));
  tensors.push_back(make_tensor("a", DType::I32, {1}, a_data));

  const fs::path file = dir / "model.safetensors";
  const std::string file_str = file.string();

  FileWriteOptions wopts;
  wopts.preallocate = true;
  wopts.buf_bytes = 64;

  serialize_to_file(file_str.c_str(), std::span<const TensorEntry>(tensors.data(), tensors.size()),
                    std::nullopt, {}, wopts);

  SafeTensorsFile st = SafeTensorsFile::open(file_str.c_str(), FileLoadMode::ReadAll);
  ASSERT_EQ(st.metadata().tensors_by_offset.size(), 2u);

  const auto ta = st.tensor("a");
  EXPECT_EQ(ta.dtype, DType::I32);
  EXPECT_EQ(std::vector<std::byte>(ta.data.begin(), ta.data.end()), a_data);

  const auto tb = st.tensor("b");
  EXPECT_EQ(tb.dtype, DType::I32);
  EXPECT_EQ(std::vector<std::byte>(tb.data.begin(), tb.data.end()), b_data);

  // MMapIfAvailable uses mmap when possible, otherwise falls back to ReadAll.
  SafeTensorsFile st_maybe =
      SafeTensorsFile::open(file_str.c_str(), FileLoadMode::MMapIfAvailable);
  ASSERT_EQ(st_maybe.metadata().tensors_by_offset.size(), 2u);

  const auto ta2 = st_maybe.tensor("a");
  EXPECT_EQ(ta2.dtype, DType::I32);
  EXPECT_EQ(std::vector<std::byte>(ta2.data.begin(), ta2.data.end()), a_data);

  const auto tb2 = st_maybe.tensor("b");
  EXPECT_EQ(tb2.dtype, DType::I32);
  EXPECT_EQ(std::vector<std::byte>(tb2.data.begin(), tb2.data.end()), b_data);

  // File contents match in-memory serialize output.
  const std::vector<std::byte> expected =
      serialize(std::span<const TensorEntry>(tensors.data(), tensors.size()), std::nullopt);
  const std::vector<std::byte> on_disk = read_file_bytes(file);
  EXPECT_EQ(on_disk, expected);
}

TEST(SafeTensorsFileIoTest, SerializeToFileAndOpenMMapRoundTrip) {
  namespace fs = std::filesystem;
  const fs::path dir = make_temp_dir("vbt_safetensors_io_mmap_");

  struct DirGuard {
    fs::path p;
    explicit DirGuard(fs::path p_) : p(std::move(p_)) {}
    ~DirGuard() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } guard(dir);

  const std::vector<std::byte> a_data = bytes({1, 2, 3, 4});
  const std::vector<std::byte> b_data = bytes({5, 6, 7, 8});

  std::vector<TensorEntry> tensors;
  tensors.push_back(make_tensor("b", DType::I32, {1}, b_data));
  tensors.push_back(make_tensor("a", DType::I32, {1}, a_data));

  const fs::path file = dir / "model.safetensors";
  const std::string file_str = file.string();

  serialize_to_file(file_str.c_str(), std::span<const TensorEntry>(tensors.data(), tensors.size()));

#if VBT_SAFETENSORS_ENABLE_MMAP
  SafeTensorsFile st = SafeTensorsFile::open(file_str.c_str(), FileLoadMode::MMap);
  ASSERT_EQ(st.metadata().tensors_by_offset.size(), 2u);

  const auto ta = st.tensor("a");
  EXPECT_EQ(ta.dtype, DType::I32);
  EXPECT_EQ(std::vector<std::byte>(ta.data.begin(), ta.data.end()), a_data);

  const auto tb = st.tensor("b");
  EXPECT_EQ(tb.dtype, DType::I32);
  EXPECT_EQ(std::vector<std::byte>(tb.data.begin(), tb.data.end()), b_data);
#else
  try {
    (void)SafeTensorsFile::open(file_str.c_str(), FileLoadMode::MMap);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::IoError);
  }
#endif
}

TEST(SafeTensorsFileIoTest, OpenNotRegularFileThrows) {
  namespace fs = std::filesystem;
  const fs::path dir = make_temp_dir("vbt_safetensors_io_dir_");

  struct DirGuard {
    fs::path p;
    explicit DirGuard(fs::path p_) : p(std::move(p_)) {}
    ~DirGuard() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } guard(dir);

  const std::string dir_str = dir.string();

  try {
    (void)SafeTensorsFile::open(dir_str.c_str(), FileLoadMode::ReadAll);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::NotRegularFile);
  }

  try {
    (void)SafeTensorsFile::open(dir_str.c_str(), FileLoadMode::MMapIfAvailable);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::NotRegularFile);
  }

#if VBT_SAFETENSORS_ENABLE_MMAP
  try {
    (void)SafeTensorsFile::open(dir_str.c_str(), FileLoadMode::MMap);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::NotRegularFile);
  }
#else
  try {
    (void)SafeTensorsFile::open(dir_str.c_str(), FileLoadMode::MMap);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::IoError);
  }
#endif
}

TEST(SafeTensorsFileIoTest, OpenFileTooLargeRespectsOpenOptions) {
  namespace fs = std::filesystem;
  const fs::path dir = make_temp_dir("vbt_safetensors_io_limit_");

  struct DirGuard {
    fs::path p;
    explicit DirGuard(fs::path p_) : p(std::move(p_)) {}
    ~DirGuard() {
      std::error_code ec;
      fs::remove_all(p, ec);
    }
  } guard(dir);

  const std::vector<std::byte> a_data = bytes({1, 2, 3, 4});

  std::vector<TensorEntry> tensors;
  tensors.push_back(make_tensor("a", DType::I32, {1}, a_data));

  const fs::path file = dir / "small.safetensors";
  const std::string file_str = file.string();

  serialize_to_file(file_str.c_str(), std::span<const TensorEntry>(tensors.data(), tensors.size()));

  const std::uintmax_t file_size = fs::file_size(file);
  ASSERT_GT(file_size, 0u);

  FileOpenOptions oopts;
  oopts.max_file_size_bytes = static_cast<std::size_t>(file_size - 1);

  try {
    (void)SafeTensorsFile::open(file_str.c_str(), FileLoadMode::ReadAll, {}, oopts);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::FileTooLarge);
  }

  try {
    (void)SafeTensorsFile::open(file_str.c_str(), FileLoadMode::MMapIfAvailable, {}, oopts);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::FileTooLarge);
  }

#if VBT_SAFETENSORS_ENABLE_MMAP
  try {
    (void)SafeTensorsFile::open(file_str.c_str(), FileLoadMode::MMap, {}, oopts);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::FileTooLarge);
  }
#else
  try {
    (void)SafeTensorsFile::open(file_str.c_str(), FileLoadMode::MMap, {}, oopts);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::IoError);
  }
#endif
}

TEST(SafeTensorsFileIoTest, SerializeToFileNullPathThrows) {
  const std::vector<std::byte> empty;
  TensorEntry t;
  t.name = "t";
  t.info.dtype = DType::I32;
  t.info.shape = {0};
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(empty.data(), empty.size());

  try {
    serialize_to_file(nullptr, std::span<const TensorEntry>(&t, 1));
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::IoError);
  }
}
