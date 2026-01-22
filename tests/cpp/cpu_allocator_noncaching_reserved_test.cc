// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include "vbt/cpu/allocator.h"

using vbt::cpu::Allocator;
using vbt::cpu::DeviceStats;

static void set_conf(const char* conf) {
#if defined(_WIN32)
  _putenv_s("VBT_CPU_ALLOC_CONF", conf);
#else
  setenv("VBT_CPU_ALLOC_CONF", conf, 1);
#endif
}

TEST(CPUAllocatorNonCachingTest, ReservedAccountingNoDriftAcrossCycles) {
  // Non-caching mode; alignment set but behavior should be consistent
  set_conf("enable_caching=0,alignment_bytes=256");

  for (int i = 0; i < 2; ++i) {
    DeviceStats s0 = Allocator::get().getDeviceStats();
    std::size_t reserved = 0;
    void* p = Allocator::get().raw_alloc(1500, 0, reserved);
    ASSERT_NE(p, nullptr);
    DeviceStats s1 = Allocator::get().getDeviceStats();
    // reserved must increase by the reserved amount we observed
    ASSERT_GE(s1.reserved_bytes_all_current, s0.reserved_bytes_all_current);
    EXPECT_EQ(s1.reserved_bytes_all_current - s0.reserved_bytes_all_current, static_cast<std::uint64_t>(reserved));

    // Free using requested bytes only; allocator must subtract the recorded reserved amount internally
    Allocator::get().raw_delete(p, 1500);
    DeviceStats s2 = Allocator::get().getDeviceStats();
    // Reserved should return to the baseline
    EXPECT_EQ(s2.reserved_bytes_all_current, s0.reserved_bytes_all_current);
  }
}
