// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vbt/interop/safetensors/error.h"
#include "vbt/interop/safetensors/types.h"

namespace vbt {
namespace interop {
namespace safetensors {

inline constexpr std::size_t kUpstreamMaxHeaderSizeBytes = 100'000'000;

struct ParseOptions {
  std::size_t max_header_size_bytes = kUpstreamMaxHeaderSizeBytes;

  // Upstream parity default: upstream code currently does NOT enforce startswith('{').
  bool require_header_start_curly = false;

  // Optional post-parse guardrails (hardening):
  std::optional<std::size_t> max_tensors;
  std::optional<std::size_t> max_rank;

  // If true, reject unknown fields inside per-tensor objects.
  // Default false for upstream parity.
  bool reject_unknown_tensor_fields = false;

  static ParseOptions Hardened() {
    ParseOptions opts;
    opts.require_header_start_curly = true;
    opts.reject_unknown_tensor_fields = true;
    // Conservative defaults; callers may override for their workloads.
    opts.max_tensors = 1'000'000;
    opts.max_rank = 128;
    return opts;
  }
};

struct ParsedMetadata {
  std::size_t header_n_bytes = 0;
  std::size_t header_end = 0;
  Metadata metadata;
};

[[nodiscard]] ParsedMetadata read_metadata(std::span<const std::byte> file_bytes,
                                          ParseOptions opts = {});

// A zero-copy view over a `.safetensors` file buffer.
//
// Lifetime: `file_bytes` passed to `deserialize()` must outlive the returned
// `SafeTensorsView` and any `TensorView::data` spans it produces.
class SafeTensorsView {
 public:
  [[nodiscard]] static SafeTensorsView deserialize(std::span<const std::byte> file_bytes,
                                                  ParseOptions opts = {});

  [[nodiscard]] const Metadata& metadata() const noexcept;
  [[nodiscard]] std::span<const std::byte> data() const noexcept;
  [[nodiscard]] TensorView tensor(std::string_view name) const;

 private:
  Metadata metadata_;
  std::span<const std::byte> data_;
};

inline constexpr std::size_t kHeaderPadBytes = 8;

struct SerializeOptions {
  // Enforced cap: min(max_header_size_bytes, kUpstreamMaxHeaderSizeBytes)
  std::size_t max_header_size_bytes = kUpstreamMaxHeaderSizeBytes;

  // Deterministic output: follow upstream layout.
  bool sort_by_dtype_alignment_then_name = true;

  // If true, enforce that dtype/shape match the provided data length.
  // If false, callers must ensure consistency; otherwise the serialized buffer
  // may be invalid.
  bool validate_tensor_sizes = true;
};

// Serialize to an owned `.safetensors` byte buffer.
//
// Note: `TensorEntry::info.data_offsets` is ignored on input; offsets are computed
// from `TensorEntry::info.dtype`/`shape` (expected byte sizes). When
// `SerializeOptions::validate_tensor_sizes` is enabled, `TensorEntry::data` is
// validated to match.
[[nodiscard]] std::vector<std::byte> serialize(
    std::span<const TensorEntry> tensors,
    const std::optional<std::vector<std::pair<std::string, std::string>>>& user_metadata =
        std::nullopt,
    SerializeOptions opts = {});

} // namespace safetensors
} // namespace interop
} // namespace vbt
