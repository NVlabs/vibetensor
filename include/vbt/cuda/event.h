// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace vbt { namespace cuda {

class Stream;  // forward decl
using DeviceIndex = int16_t;

class Event final {
 public:
  explicit Event(bool enable_timing = false) noexcept;
  Event(Event&&) noexcept;
  Event& operator=(Event&&) noexcept;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() noexcept;

  bool is_created() const noexcept { return is_created_; }
  bool query() const noexcept;  // true if not created or completed

  void record(const Stream& stream);
  void wait(const Stream& stream) const;
  void synchronize() const;

 private:
  unsigned int flags_{0};
  bool         is_created_{false};
  bool         was_recorded_{false};
  DeviceIndex  device_index_{-1};
  void*        event_{nullptr}; // cudaEvent_t
};

}} // namespace vbt::cuda
