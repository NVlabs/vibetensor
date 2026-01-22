// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vbt/interop/safetensors/safetensors.h"

namespace vbt {
namespace interop {
namespace safetensors {

enum class FileLoadMode {
  // Read the entire file into an owned byte buffer.
  ReadAll,

  // Memory-map the file read-only (POSIX only; guarded by VBT_SAFETENSORS_ENABLE_MMAP).
  // When mmap support is disabled at build time, selecting MMap will throw IoError.
  // Note: the file must not be modified/truncated while mapped (may SIGBUS on access).
  MMap,

  // Use MMap when available (and enabled), otherwise fall back to ReadAll.
  MMapIfAvailable,
};

struct FileOpenOptions {
  std::optional<std::size_t> max_file_size_bytes;
  bool require_regular_file = true;
};

class SafeTensorsFile {
 public:
  static SafeTensorsFile open(const char* path,
                              FileLoadMode mode = FileLoadMode::MMapIfAvailable,
                              ParseOptions parse_opts = {},
                              FileOpenOptions open_opts = {});

  [[nodiscard]] const Metadata& metadata() const noexcept { return view_.metadata(); }
  [[nodiscard]] std::span<const std::byte> data() const noexcept { return view_.data(); }
  [[nodiscard]] TensorView tensor(std::string_view name) const { return view_.tensor(name); }

 private:
  std::shared_ptr<const std::byte> owner_;
  std::size_t size_ = 0;
  SafeTensorsView view_;
};

struct FileWriteOptions {
  bool preallocate = true;
  std::size_t buf_bytes = 1024 * 1024;
};

// Serialize and write a `.safetensors` file to disk.
//
// Note: this helper overwrites `path` in-place (open + truncate + write). It does
// not attempt an atomic replace.
void serialize_to_file(
    const char* path,
    std::span<const TensorEntry> tensors,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata =
        std::nullopt,
    SerializeOptions sopts = {},
    FileWriteOptions fopts = {});

} // namespace safetensors
} // namespace interop
} // namespace vbt
