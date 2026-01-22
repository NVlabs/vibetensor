// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

#include "vbt/rng/generator.h"
#include "vbt/rng/graph_capture.h"
#include "vbt/cuda/device.h"

using vbt::rng::CudaGenerator;
using vbt::rng::PhiloxState;

namespace {

constexpr int kDeviceIndex = 0;

bool has_cuda_device() {
#if VBT_WITH_CUDA
  return vbt::cuda::device_count() > 0;
#else
  return false;
#endif
}

} // anonymous namespace

TEST(RngCudaGraphCaptureCore, BeginReserveEndSingleSlice) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(/*seed=*/123ull, /*offset=*/5ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  PhiloxState st = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/7ull,
      /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/true);

  EXPECT_EQ(st.seed, 123ull);
  EXPECT_EQ(st.offset, 5ull);

  auto summary = vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
  EXPECT_EQ(summary.base_state.seed, 123ull);
  EXPECT_EQ(summary.base_state.offset, 5ull);
  EXPECT_EQ(summary.total_blocks, 7ull);

  auto final_state = gen.get_state();

  // Non-graph reference: same seed/offset and a single reserve_blocks call.
  gen.set_state(/*seed=*/123ull, /*offset=*/5ull);
  PhiloxState st_ref = gen.reserve_blocks(7ull);
  auto final_state_ref = gen.get_state();

  EXPECT_EQ(st_ref.seed, 123ull);
  EXPECT_EQ(st_ref.offset, 5ull);
  EXPECT_EQ(final_state.seed, final_state_ref.seed);
  EXPECT_EQ(final_state.offset, final_state_ref.offset);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, BeginReserveEndMultipleSlicesEquivalence) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  const std::uint64_t seed = 42ull;
  const std::uint64_t base_offset = 10ull;
  gen.set_state(seed, base_offset);

  const std::uint64_t blocks1 = 3ull;
  const std::uint64_t blocks2 = 10ull;

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  PhiloxState st1 = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen, blocks1, /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/true);
  PhiloxState st2 = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen, blocks2, /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Normal,
      /*stream_is_capturing=*/true);

  auto summary = vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
  EXPECT_EQ(summary.base_state.seed, seed);
  EXPECT_EQ(summary.base_state.offset, base_offset);
  EXPECT_EQ(summary.total_blocks, blocks1 + blocks2);

  auto final_state_capture = gen.get_state();

  // Reference path: perform the two reservations without the capture helper.
  gen.set_state(seed, base_offset);
  PhiloxState ref1 = gen.reserve_blocks(blocks1);
  PhiloxState ref2 = gen.reserve_blocks(blocks2);
  auto final_state_ref = gen.get_state();

  EXPECT_EQ(st1.seed, ref1.seed);
  EXPECT_EQ(st1.offset, ref1.offset);
  EXPECT_EQ(st2.seed, ref2.seed);
  EXPECT_EQ(st2.offset, ref2.offset);

  EXPECT_EQ(final_state_capture.seed, final_state_ref.seed);
  EXPECT_EQ(final_state_capture.offset, final_state_ref.offset);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, OverlappingBeginThrows) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(1ull, 0ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  try {
    vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);
    FAIL() << "Expected std::runtime_error for overlapping capture";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngOverlappingCapture);
  } catch (...) {
    FAIL() << "Unexpected exception type";
  }

  vbt::rng::graph_capture::on_cuda_graph_capture_abort(gen);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, EndWithoutBeginThrows) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(5ull, 0ull);

  try {
    (void)vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
    FAIL() << "Expected std::logic_error when ending without begin";
  } catch (const std::logic_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngCaptureEndWithoutActiveCapture);
  } catch (...) {
    FAIL() << "Unexpected exception type";
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, EndFromWrongThreadThrows) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(7ull, 0ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  std::exception_ptr eptr;
  std::thread t([&]() {
    try {
      (void)vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
    } catch (...) {
      eptr = std::current_exception();
    }
  });
  t.join();

  ASSERT_TRUE(eptr != nullptr);
  try {
    std::rethrow_exception(eptr);
  } catch (const std::logic_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngCaptureEndWrongThread);
  } catch (...) {
    FAIL() << "Unexpected exception type from wrong-thread end";
  }

  // Abort from the owner thread to clean up.
  vbt::rng::graph_capture::on_cuda_graph_capture_abort(gen);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, CrossThreadReserveThrowsConcurrentUseError) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(9ull, 0ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  std::exception_ptr eptr;
  std::thread t([&]() {
    try {
      (void)vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
          gen,
          /*total_blocks=*/1ull,
          /*outputs_per_block=*/4u,
          vbt::rng::graph_capture::RngOpTag::Uniform,
          /*stream_is_capturing=*/true);
    } catch (...) {
      eptr = std::current_exception();
    }
  });
  t.join();

  ASSERT_TRUE(eptr != nullptr);
  try {
    std::rethrow_exception(eptr);
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngConcurrentUseDuringCapture);
  } catch (...) {
    FAIL() << "Unexpected exception type from cross-thread reserve";
  }

  vbt::rng::graph_capture::on_cuda_graph_capture_abort(gen);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, NonCaptureStreamThrowsUseOnNonCaptureStream) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(11ull, 0ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  try {
    (void)vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
        gen,
        /*total_blocks=*/4ull,
        /*outputs_per_block=*/4u,
        vbt::rng::graph_capture::RngOpTag::Uniform,
        /*stream_is_capturing=*/false);
    FAIL() << "Expected std::runtime_error for non-capture stream";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngUseOnNonCaptureStream);
  } catch (...) {
    FAIL() << "Unexpected exception type";
  }

  vbt::rng::graph_capture::on_cuda_graph_capture_abort(gen);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, MutationDuringCaptureDetectedAtFinalize) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(13ull, 0ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  (void)vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/2ull,
      /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/true);

  // Mutate the generator directly during capture.
  gen.manual_seed(999ull);

  try {
    (void)vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
    FAIL() << "Expected std::logic_error for mutation during capture";
  } catch (const std::logic_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngGeneratorStateMutatedDuringCapture);
  } catch (...) {
    FAIL() << "Unexpected exception type";
  }

  // Capture should be torn down and generator state should remain mutated.
  EXPECT_FALSE(vbt::rng::graph_capture::is_generator_capture_active(gen));
  auto st = gen.get_state();
  EXPECT_EQ(st.seed, 999ull);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, ZeroBlocksBypassCaptureState) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);

  // No active capture.
  gen.set_state(21ull, 5ull);
  auto before = gen.get_state();
  PhiloxState st = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/0ull,
      /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/false);
  auto after = gen.get_state();
  EXPECT_EQ(st.seed, before.seed);
  EXPECT_EQ(st.offset, before.offset);
  EXPECT_EQ(after.seed, before.seed);
  EXPECT_EQ(after.offset, before.offset);

  // With an active capture.
  gen.set_state(22ull, 7ull);
  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);
  auto base_state = gen.get_state();

  PhiloxState st2 = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/0ull,
      /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/true);
  EXPECT_EQ(st2.seed, base_state.seed);
  EXPECT_EQ(st2.offset, base_state.offset);

  auto summary = vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
  EXPECT_EQ(summary.base_state.seed, base_state.seed);
  EXPECT_EQ(summary.base_state.offset, base_state.offset);
  EXPECT_EQ(summary.total_blocks, 0ull);

  auto final_state = gen.get_state();
  EXPECT_EQ(final_state.seed, base_state.seed);
  EXPECT_EQ(final_state.offset, base_state.offset);

  EXPECT_FALSE(vbt::rng::graph_capture::is_generator_capture_active(gen));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, NonDefaultGeneratorRejectedAtReserve) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  // Construct a non-default generator instance for the same device.
  CudaGenerator other(kDeviceIndex);
  other.set_state(33ull, 0ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(other);

  try {
    (void)vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
        other,
        /*total_blocks=*/1ull,
        /*outputs_per_block=*/4u,
        vbt::rng::graph_capture::RngOpTag::Uniform,
        /*stream_is_capturing=*/true);
    FAIL() << "Expected std::runtime_error for non-default generator";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngNonDefaultGeneratorInGraph);
  } catch (...) {
    FAIL() << "Unexpected exception type";
  }

  vbt::rng::graph_capture::on_cuda_graph_capture_abort(other);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, CaptureBlocksOverflowGuard) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  const std::uint64_t seed = 555ull;
  const std::uint64_t near_max = std::numeric_limits<std::uint64_t>::max() - 3ull;
  gen.set_state(seed, near_max);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  try {
    (void)vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
        gen,
        /*total_blocks=*/10ull,
        /*outputs_per_block=*/4u,
        vbt::rng::graph_capture::RngOpTag::Uniform,
        /*stream_is_capturing=*/true);
    FAIL() << "Expected overflow guard runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), vbt::rng::graph_capture::kErrCudaRngCaptureBlocksOverflow);
  } catch (...) {
    FAIL() << "Unexpected exception type";
  }

  // Generator state should be unchanged; capture should still be active until abort.
  auto st = gen.get_state();
  EXPECT_EQ(st.seed, seed);
  EXPECT_EQ(st.offset, near_max);
  EXPECT_TRUE(vbt::rng::graph_capture::is_generator_capture_active(gen));

  vbt::rng::graph_capture::on_cuda_graph_capture_abort(gen);
  EXPECT_FALSE(vbt::rng::graph_capture::is_generator_capture_active(gen));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(RngCudaGraphCaptureCore, IsGeneratorCaptureActiveLifecycle) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(777ull, 0ull);

  EXPECT_FALSE(vbt::rng::graph_capture::is_generator_capture_active(gen));

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);
  EXPECT_TRUE(vbt::rng::graph_capture::is_generator_capture_active(gen));

  (void)vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/4ull,
      /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/true);

  auto summary = vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);
  (void)summary;
  EXPECT_FALSE(vbt::rng::graph_capture::is_generator_capture_active(gen));

  // Begin + abort sequence.
  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);
  EXPECT_TRUE(vbt::rng::graph_capture::is_generator_capture_active(gen));
  vbt::rng::graph_capture::on_cuda_graph_capture_abort(gen);
  EXPECT_FALSE(vbt::rng::graph_capture::is_generator_capture_active(gen));
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

#ifdef VBT_INTERNAL_TESTS
TEST(RngCudaGraphCaptureCore, DebugSummaryStoresSlices) {
#if VBT_WITH_CUDA
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA device";
  }

  auto& gen = vbt::rng::default_cuda(kDeviceIndex);
  gen.set_state(100ull, 1ull);

  vbt::rng::graph_capture::on_cuda_graph_capture_begin(gen);

  PhiloxState s1 = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/2ull,
      /*outputs_per_block=*/4u,
      vbt::rng::graph_capture::RngOpTag::Uniform,
      /*stream_is_capturing=*/true);
  PhiloxState s2 = vbt::rng::graph_capture::reserve_blocks_for_graph_aware_cuda_op(
      gen,
      /*total_blocks=*/5ull,
      /*outputs_per_block=*/2u,
      vbt::rng::graph_capture::RngOpTag::Randint,
      /*stream_is_capturing=*/true);

  auto summary = vbt::rng::graph_capture::on_cuda_graph_capture_end_success(gen);

  auto dbg = vbt::rng::graph_capture::debug_last_capture_summary_for_cuda_device(kDeviceIndex);
  ASSERT_TRUE(dbg.has_value());
  EXPECT_EQ(dbg->base_state.seed, summary.base_state.seed);
  EXPECT_EQ(dbg->base_state.offset, summary.base_state.offset);
  EXPECT_EQ(dbg->total_blocks, summary.total_blocks);

  ASSERT_EQ(dbg->slices.size(), 2u);
  EXPECT_EQ(dbg->slices[0].state.seed, s1.seed);
  EXPECT_EQ(dbg->slices[0].state.offset, s1.offset);
  EXPECT_EQ(dbg->slices[0].total_blocks, 2ull);
  EXPECT_EQ(dbg->slices[0].outputs_per_block, 4u);
  EXPECT_EQ(dbg->slices[0].op_tag, vbt::rng::graph_capture::RngOpTag::Uniform);

  EXPECT_EQ(dbg->slices[1].state.seed, s2.seed);
  EXPECT_EQ(dbg->slices[1].state.offset, s2.offset);
  EXPECT_EQ(dbg->slices[1].total_blocks, 5ull);
  EXPECT_EQ(dbg->slices[1].outputs_per_block, 2u);
  EXPECT_EQ(dbg->slices[1].op_tag, vbt::rng::graph_capture::RngOpTag::Randint);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
#endif  // VBT_INTERNAL_TESTS
