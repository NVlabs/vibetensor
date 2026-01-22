// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

namespace vbt { namespace autograd {

enum class AutogradDeviceMode : std::uint8_t {
  SingleDevice = 0,
  MultiDeviceExperimental = 1,
};

AutogradDeviceMode get_device_mode() noexcept;
void set_device_mode(AutogradDeviceMode mode) noexcept;

bool is_multithreading_enabled() noexcept;
void set_multithreading_enabled(bool v) noexcept;

bool is_view_replay_enabled() noexcept;
void set_view_replay_enabled(bool v) noexcept;

bool is_streaming_backwards_enabled() noexcept;
void set_streaming_backwards_enabled(bool enabled) noexcept;

}} // namespace vbt::autograd
