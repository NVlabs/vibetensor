// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/cpu/storage.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/graphs.h"
#include "vbt/rng/generator.h"
#include "vbt/rng/kernels_cuda.h"
#include "vbt/rng/graph_capture.h"
#include "vbt/rng/philox_util.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;

namespace gc = vbt::rng::graph_capture;

namespace {

static TensorImpl make_cuda_tensor(std::int64_t N, int dev) {
#if VBT_WITH_CUDA
  std::vector<int64_t> sizes{N};
  std::vector<int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>((N > 0 ? N : 0)) * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  return TensorImpl(storage,
                    std::move(sizes),
                    std::move(strides),
                    /*storage_offset=*/0,
                    ScalarType::Float32,
                    Device::cuda(dev));
#else
  (void)N; (void)dev;
  throw std::runtime_error("CUDA not built");
#endif
}

static std::vector<float> copy_cuda_tensor_to_host(const TensorImpl& t) {
#if VBT_WITH_CUDA
  const std::int64_t N = t.numel();
  std::vector<float> out(static_cast<std::size_t>(N > 0 ? N : 0));
  if (N > 0) {
    cudaError_t st = cudaMemcpy(out.data(), t.data(), out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy D2H failed");
    }
  }
  return out;
#else
  (void)t;
  throw std::runtime_error("CUDA not built");
#endif
}

struct RngCudaGraphCaptureScope {
  vbt::rng::CudaGenerator& gen;
  vbt::cuda::CUDAGraph&    graph;
  vbt::cuda::Stream        stream;
  vbt::cuda::CUDAStreamGuard stream_guard;
  bool graph_capturing{false};

  RngCudaGraphCaptureScope(vbt::rng::CudaGenerator& g,
                           vbt::cuda::CUDAGraph&    gr,
                           vbt::cuda::Stream        s)
      : gen(g), graph(gr), stream(s), stream_guard(s) {
    graph.capture_begin(stream);
    graph_capturing = true;
  }

  void end_success() {
    graph.capture_end();
    graph_capturing = false;
  }

  ~RngCudaGraphCaptureScope() {
    if (graph_capturing) {
      try {
        graph.capture_end();
      } catch (...) {
        try {
          graph.reset();
        } catch (...) {
        }
      }
      graph_capturing = false;
    }
  }
};

} // anonymous namespace

TEST(RngCudaUniformGraphCaptureIntegration, UniformGraphCaptureParityBasic) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  auto& gen = vbt::rng::default_cuda(dev);
  const std::uint64_t seed = 1234ull;
  const float low = 0.0f;
  const float high = 1.0f;

  const std::int64_t sizes[] = {0, 1, 2, 4, 31, 32, 33, 1001};

  for (std::int64_t N64 : sizes) {
    TensorImpl t_eager = make_cuda_tensor(N64, dev);
    TensorImpl t_cap = make_cuda_tensor(N64, dev);

    // Eager run (no graphs)
    gen.set_state(seed, 0ull);
    vbt::rng::cuda::uniform_(t_eager, low, high, gen);
    std::vector<float> eager = copy_cuda_tensor_to_host(t_eager);
    auto st_eager = gen.get_state();

    // Graph capture run
    gen.set_state(seed, 0ull);
    vbt::cuda::Stream s(/*priority=*/0, static_cast<vbt::cuda::DeviceIndex>(dev));
    vbt::cuda::CUDAGraph graph;

    {
      RngCudaGraphCaptureScope scope(gen, graph, s);
      vbt::rng::cuda::uniform_(t_cap, low, high, gen);
      scope.end_success();
    }

    graph.instantiate();
    graph.replay(s);

    cudaError_t sync_st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(s.handle()));
    ASSERT_EQ(sync_st, cudaSuccess);

    std::vector<float> cap = copy_cuda_tensor_to_host(t_cap);
    auto st_cap = gen.get_state();

    ASSERT_EQ(eager.size(), cap.size());
    for (std::size_t i = 0; i < eager.size(); ++i) {
      EXPECT_EQ(eager[i], cap[i]);
    }

    EXPECT_EQ(st_eager.seed, st_cap.seed);

    const std::uint64_t N = static_cast<std::uint64_t>(N64 > 0 ? N64 : 0);
    const std::uint64_t expected_blocks = vbt::rng::ceil_div_u64(N, 4ull);
    EXPECT_EQ(st_eager.offset, expected_blocks);
    EXPECT_EQ(st_cap.offset, expected_blocks);

#ifdef VBT_INTERNAL_TESTS
    auto dbg = gc::debug_last_capture_summary_for_cuda_device(dev);
    ASSERT_TRUE(dbg.has_value());
    EXPECT_EQ(dbg->base_state.seed, seed);
    EXPECT_EQ(dbg->base_state.offset, 0ull);
    EXPECT_EQ(dbg->total_blocks, expected_blocks);
    if (N64 > 0) {
      ASSERT_EQ(dbg->slices.size(), 1u);
      EXPECT_EQ(dbg->slices[0].total_blocks, expected_blocks);
      EXPECT_EQ(dbg->slices[0].outputs_per_block, 4u);
      EXPECT_EQ(dbg->slices[0].op_tag, gc::RngOpTag::Uniform);
    }
#endif
  }
#endif
}

TEST(RngCudaUniformGraphCaptureIntegration, UniformGraphCaptureErrorsWrongStream) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  auto& gen = vbt::rng::default_cuda(dev);
  const std::uint64_t seed = 2024ull;

  // Non-empty tensor
  TensorImpl t = make_cuda_tensor(/*N=*/32, dev);

  gen.set_state(seed, 0ull);
  vbt::cuda::Stream s_main(/*priority=*/0, static_cast<vbt::cuda::DeviceIndex>(dev));
  vbt::cuda::CUDAGraph graph;

  {
    RngCudaGraphCaptureScope scope(gen, graph, s_main);

    // Use a different stream as the current stream while RNG wrapper executes.
    vbt::cuda::Stream s_other(/*priority=*/0, static_cast<vbt::cuda::DeviceIndex>(dev));
    vbt::cuda::CUDAStreamGuard guard_other(s_other);

    try {
      vbt::rng::cuda::uniform_(t, 0.0f, 1.0f, gen);
      FAIL() << "Expected runtime_error for RNG on non-capture stream";
    } catch (const std::runtime_error& e) {
      EXPECT_STREQ(e.what(), gc::kErrCudaRngUseOnNonCaptureStream);
    } catch (...) {
      FAIL() << "Unexpected exception type";
    }

    // Finalize capture successfully; there were no recorded RNG slices.
    scope.end_success();
#ifdef VBT_INTERNAL_TESTS
    auto dbg = gc::debug_last_capture_summary_for_cuda_device(dev);
    ASSERT_TRUE(dbg.has_value());
    EXPECT_EQ(dbg->total_blocks, 0ull);
#endif
  }
#endif
}

TEST(RngCudaUniformGraphCaptureIntegration, UniformGraphCaptureErrorsCrossThread) {
#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA required";
#else
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  auto& gen = vbt::rng::default_cuda(dev);
  const std::uint64_t seed = 3033ull;

  TensorImpl t = make_cuda_tensor(/*N=*/16, dev);

  gen.set_state(seed, 0ull);
  vbt::cuda::Stream s(/*priority=*/0, static_cast<vbt::cuda::DeviceIndex>(dev));
  vbt::cuda::CUDAGraph graph;

  std::exception_ptr eptr;

  {
    RngCudaGraphCaptureScope scope(gen, graph, s);

    std::thread worker([&]() {
      try {
        vbt::rng::cuda::uniform_(t, 0.0f, 1.0f, gen);
      } catch (...) {
        eptr = std::current_exception();
      }
    });

    worker.join();

    ASSERT_TRUE(eptr != nullptr);
    try {
      std::rethrow_exception(eptr);
    } catch (const std::runtime_error& e) {
      EXPECT_STREQ(e.what(), gc::kErrCudaRngConcurrentUseDuringCapture);
    } catch (...) {
      FAIL() << "Unexpected exception type from cross-thread uniform_";
    }

    scope.end_success();
#ifdef VBT_INTERNAL_TESTS
    auto dbg = gc::debug_last_capture_summary_for_cuda_device(dev);
    ASSERT_TRUE(dbg.has_value());
    // No successful RNG slices should have been recorded, so total_blocks==0.
    EXPECT_EQ(dbg->total_blocks, 0ull);
#endif
  }
#endif
}
