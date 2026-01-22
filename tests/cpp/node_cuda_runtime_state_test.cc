// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <thread>

#include "vbt/node/cuda_napi.h"

using vbt::node::CudaRuntimeState;
using vbt::node::IsOnMainThreadFromState;

// Basic unit tests for the threading helper used by the Node N-API CUDA
// bindings. These tests do not touch N-API; they only exercise the pure
// helper that checks thread affinity against CudaRuntimeState.

TEST(NodeCudaRuntimeStateTest, IsOnMainThreadReturnsFalseWhenUninitialized) {
  CudaRuntimeState rt;
  rt.initialized = false;
  rt.main_thread_id = std::thread::id{};

  // Regardless of the thread id, an uninitialized runtime must report false.
  EXPECT_FALSE(IsOnMainThreadFromState(rt, std::this_thread::get_id()));

  std::thread::id other;
  EXPECT_FALSE(IsOnMainThreadFromState(rt, other));
}

TEST(NodeCudaRuntimeStateTest, IsOnMainThreadMatchesConfiguredThreadId) {
  CudaRuntimeState rt;
  rt.initialized = true;
  const std::thread::id main_id = std::this_thread::get_id();
  rt.main_thread_id = main_id;

  EXPECT_TRUE(IsOnMainThreadFromState(rt, main_id));

  // A different thread id must report false.
  std::thread::id other_id;
  if (other_id == main_id) {
    std::thread tmp([] {});
    other_id = tmp.get_id();
    tmp.join();
  }
  EXPECT_FALSE(IsOnMainThreadFromState(rt, other_id));
}
