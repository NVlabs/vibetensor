// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "vbt/dispatch/dispatch_key_set.h"

using vbt::dispatch::DispatchKey;
using vbt::dispatch::DispatchKeySet;
using vbt::dispatch::ExcludeDispatchKeysGuard;
using vbt::dispatch::IncludeDispatchKeysGuard;
using vbt::dispatch::tls_local_dispatch_key_set;

TEST(DispatchKeySetTest, AddRemoveAndPriorityOrdering) {
  DispatchKeySet ks;
  EXPECT_TRUE(ks.empty());

  ks = ks.add(DispatchKey::CPU);
  EXPECT_TRUE(ks.has(DispatchKey::CPU));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::CPU);

  ks = ks.add(DispatchKey::CUDA);
  EXPECT_TRUE(ks.has(DispatchKey::CUDA));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::CUDA);

  ks = ks.add(DispatchKey::Python);
  EXPECT_TRUE(ks.has(DispatchKey::Python));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::Python);

  ks = ks.add(DispatchKey::Autograd);
  EXPECT_TRUE(ks.has(DispatchKey::Autograd));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::Autograd);

  ks = ks.remove(DispatchKey::Autograd);
  EXPECT_FALSE(ks.has(DispatchKey::Autograd));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::Python);

  ks = ks.remove(DispatchKey::Python);
  EXPECT_FALSE(ks.has(DispatchKey::Python));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::CUDA);

  ks = ks.remove(DispatchKey::CUDA);
  EXPECT_FALSE(ks.has(DispatchKey::CUDA));
  EXPECT_EQ(ks.highest_priority_key(), DispatchKey::CPU);

  ks = ks.remove(DispatchKey::CPU);
  EXPECT_TRUE(ks.empty());
}

TEST(DispatchKeySetTest, TLSIncludeExcludeExclusionWins) {
  // Ensure each test starts from a clean TLS state.
  tls_local_dispatch_key_set = {};

  DispatchKeySet base;
  base = base.add(DispatchKey::CUDA);

  // No TLS modifications.
  {
    DispatchKeySet applied = tls_local_dispatch_key_set.apply(base);
    EXPECT_EQ(applied, base);
  }

  // Include Python.
  {
    IncludeDispatchKeysGuard include_python{DispatchKeySet{}.add(DispatchKey::Python)};
    DispatchKeySet applied = tls_local_dispatch_key_set.apply(base);
    EXPECT_TRUE(applied.has(DispatchKey::CUDA));
    EXPECT_TRUE(applied.has(DispatchKey::Python));

    // Exclude Python (exclusion wins).
    {
      ExcludeDispatchKeysGuard exclude_python{DispatchKeySet{}.add(DispatchKey::Python).add(DispatchKey::CUDA)};
      DispatchKeySet applied2 = tls_local_dispatch_key_set.apply(base);
      EXPECT_TRUE(applied2.has(DispatchKey::CUDA));
      EXPECT_FALSE(applied2.has(DispatchKey::Python));
      EXPECT_FALSE(tls_local_dispatch_key_set.excluded.has(DispatchKey::CUDA));
    }

    // Exclude guard should restore excluded set.
    DispatchKeySet applied3 = tls_local_dispatch_key_set.apply(base);
    EXPECT_TRUE(applied3.has(DispatchKey::Python));
  }

  // Include guard should restore included set.
  EXPECT_TRUE(tls_local_dispatch_key_set.included.empty());
  EXPECT_TRUE(tls_local_dispatch_key_set.excluded.empty());
}
