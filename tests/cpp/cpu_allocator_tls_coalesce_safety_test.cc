// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include "vbt/cpu/allocator.h"

using vbt::cpu::Allocator;

static void set_conf(const char* conf) {
#if defined(_WIN32)
  _putenv_s("VBT_CPU_ALLOC_CONF", conf);
#else
  setenv("VBT_CPU_ALLOC_CONF", conf, 1);
#endif
}

TEST(CPUAllocatorCachingFixes, RightNeighborInTlsIsNotCoalesced) {
  // Configure caching with small TLS cap so spilling can move only larger blocks
  set_conf("enable_caching=1,min_bin_bytes=128,max_bin_bytes=4096,bin_growth=2,alignment_bytes=64,tls_cap_bytes=256,global_cap_bytes=0");

  // Step 1: Allocate ~768 bytes -> class 1024, split OS segment into head(768) + tail(256 inserted to global)
  std::size_t r0 = 0;
  void* p_head = Allocator::get().raw_alloc(768, 0, r0);
  ASSERT_NE(p_head, nullptr);
  ASSERT_GE(r0, static_cast<std::size_t>(768));

  // Step 2: Allocate 256 bytes; this should come from the global tail created above
  std::size_t r1 = 0;
  void* p_right = Allocator::get().raw_alloc(256, 0, r1);
  ASSERT_NE(p_right, nullptr);
  EXPECT_EQ(r1, static_cast<std::size_t>(0));

  // Step 3: Free right neighbor first: goes into TLS (not global)
  Allocator::get().raw_delete(p_right, 256);

  // Step 4: Free left block; TLS now exceeds cap, so the larger (left) is spilled to global,
  // inserting into global containers while right neighbor remains in TLS.
  Allocator::get().raw_delete(p_head, 768);

  // Step 5: Trigger spill of the remaining TLS block (right neighbor) to global by freeing
  // a small block so TLS exceeds cap and the largest (256) gets spilled.
  std::size_t r2 = 0;
  void* p_small = Allocator::get().raw_alloc(100, 0, r2); // need 128
  ASSERT_NE(p_small, nullptr);
  // Whether from cache or OS is not critical here
  Allocator::get().raw_delete(p_small, 100);

  // Cleanup and drain caches; test passes if no crash/UAF occurred during coalescing decisions
  Allocator::get().emptyCache();
  auto stats = Allocator::get().getDeviceStats();
  // After emptyCache, reserved may be zero or non-zero depending on segment heads retained, but should be non-negative by type.
  (void)stats;
}
