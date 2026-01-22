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

TEST(CPUAllocatorCachingTest, TLSReuseAndEmptyCache) {
  // Enable caching with reasonable bins; ensure fresh process state
  set_conf("enable_caching=1,min_bin_bytes=256,max_bin_bytes=4096,bin_growth=2,tls_cap_bytes=1048576,global_cap_bytes=0,alignment_bytes=64");
  // Allocate A
  std::size_t r1=0;
  void* p1 = Allocator::get().raw_alloc(1000, 0, r1);
  ASSERT_NE(p1, nullptr);
  EXPECT_GE(r1, (size_t)1000);
  // Free A
  Allocator::get().raw_delete_exact(p1, 1000, r1);
  // Allocate same size, expect TLS hit and same pointer
  std::size_t r2=0;
  void* p2 = Allocator::get().raw_alloc(1000, 0, r2);
  ASSERT_EQ(p2, p1);
  // reserved_out should be 0 on cache reuse
  EXPECT_EQ(r2, (size_t)0);
  // Free again
  Allocator::get().raw_delete_exact(p2, 1000, r1);
  // Reserved should still be >0 (segment retained)
  DeviceStats s = Allocator::get().getDeviceStats();
  EXPECT_GT(s.reserved_bytes_all_current, (uint64_t)0);
  // emptyCache should free OS segments and reduce reserved to 0
  Allocator::get().emptyCache();
  s = Allocator::get().getDeviceStats();
  EXPECT_EQ(s.reserved_bytes_all_current, (uint64_t)0);
}

TEST(CPUAllocatorCachingTest, OwnsAndBaseAllocation) {
  set_conf("enable_caching=1,min_bin_bytes=256,max_bin_bytes=4096,bin_growth=2,alignment_bytes=64");
  std::size_t r=0;
  void* p = Allocator::get().raw_alloc(1500, 0, r);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(Allocator::get().owns(p));
  std::size_t seg=0;
  void* base = Allocator::get().getBaseAllocation(p, &seg);
  ASSERT_EQ(base, p);
  EXPECT_GE(seg, (size_t)1500);
  Allocator::get().raw_delete_exact(p, 1500, r);
  Allocator::get().emptyCache();
}
