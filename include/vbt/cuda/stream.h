// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace vbt { namespace cuda {

// Signed small type for device indices; -1 means "current device".
using DeviceIndex = int16_t;
inline constexpr int kMaxCompileTimePriorities = 4;

class Stream final {
 public:
  enum Unchecked { UNCHECKED };
  Stream(Unchecked, uint64_t packed_id, DeviceIndex device) noexcept;
  explicit Stream(int priority = 0, DeviceIndex device = -1);

  [[nodiscard]] DeviceIndex device_index() const noexcept { return device_index_; }
  [[nodiscard]] uint64_t id() const noexcept { return id_; }
  [[nodiscard]] uintptr_t handle() const noexcept { return handle_; }

  bool query() const noexcept;
  void synchronize() const;
  int  priority() const;

  bool operator==(const Stream& other) const noexcept {
    return device_index_ == other.device_index_ && id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept { return !(*this == other); }

 private:
  uint64_t    id_{0};       // 0 == default stream
  DeviceIndex device_index_{0};
  uintptr_t   handle_{0};   // cudaStream_t value as integer; 0 for default (nullptr)

  friend Stream getDefaultStream(DeviceIndex);
  friend Stream getCurrentStream(DeviceIndex);
  friend void   setCurrentStream(Stream);
  friend Stream getStreamFromPool(bool, DeviceIndex);
  friend Stream getStreamFromPool(int, DeviceIndex);
};

Stream getDefaultStream(DeviceIndex device = -1);
Stream getCurrentStream(DeviceIndex device = -1);
void   setCurrentStream(Stream s);

Stream getStreamFromPool(bool high_priority = false, DeviceIndex device = -1);
Stream getStreamFromPool(int priority, DeviceIndex device = -1);

std::pair<int,int> priority_range();

std::string to_string(const Stream& s);

}} // namespace vbt::cuda
