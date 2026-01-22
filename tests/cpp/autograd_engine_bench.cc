// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::AccumulateGrad;
using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::autograd::run_backward;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

namespace {

// Minimal CPU Float32 dense tensor helper (copied from existing tests).
TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  std::size_t nbytes = ne * sizeof(float);
  void* buf = nullptr;
  if (nbytes > 0) buf = ::operator new(nbytes);
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]);
  }
  TensorImpl t(st, sizes, strides, /*offset=*/0, ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

// Simple pass-through node used to build synthetic chains.
struct ChainNode final : Node {
  uint32_t num_inputs() const noexcept override { return 1; }
  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    std::vector<OptionalTensor> out(1);
    if (!grads_in.empty()) {
      out[0] = std::move(grads_in[0]);
    }
    return out;
  }
};

struct ChainGraph {
  intrusive_ptr<Node> root;
  TensorImpl          leaf;
};

ChainGraph make_chain_graph(std::size_t depth) {
  if (depth == 0) {
    throw std::invalid_argument("depth must be > 0");
  }

  // Leaf + AccumulateGrad sink.
  TensorImpl leaf = make_cpu_dense_f32({1}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  // Build a simple linear chain of ChainNode instances.
  std::vector<intrusive_ptr<ChainNode>> nodes;
  nodes.reserve(depth);
  for (std::size_t i = 0; i < depth; ++i) {
    nodes.push_back(vbt::core::make_intrusive<ChainNode>());
  }

  for (std::size_t i = 0; i < depth; ++i) {
    nodes[i]->next_edges.resize(1);
    if (i + 1 < depth) {
      nodes[i]->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(nodes[i + 1].get()), 0};
    } else {
      nodes[i]->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<Node>(acc.get()), 0};
    }
  }

  ChainGraph g;
  g.root = intrusive_ptr<Node>(nodes[0].get());
  g.leaf = std::move(leaf);
  return g;
}

void run_bench(std::size_t depth, std::size_t iters) {
  ChainGraph graph = make_chain_graph(depth);

  // Seed gradient for the root node.
  std::vector<OptionalTensor> seed(1);
  TensorImpl seed_impl = make_cpu_dense_f32({1}, 1.0f);
  seed[0] = seed_impl;

  auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iters; ++i) {
    run_backward(graph.root, seed, {});
  }
  auto end = std::chrono::steady_clock::now();

  auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double elapsed_ms = static_cast<double>(elapsed_ns) / 1.0e6;
  double avg_us = (iters > 0)
                      ? static_cast<double>(elapsed_ns) / static_cast<double>(iters) / 1.0e3
                      : 0.0;

  std::cout << "VibeTensor autograd bench" << "\n"
            << "  depth=" << depth << " nodes" << "\n"
            << "  iters=" << iters << "\n"
            << "  elapsed_ms=" << elapsed_ms << "\n"
            << "  avg_us_per_backward=" << avg_us << "\n";
}

} // anonymous namespace

int main(int argc, char** argv) {
  std::size_t depth = 64;
  std::size_t iters = 200;

  if (argc >= 2) {
    depth = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
  }
  if (argc >= 3) {
    iters = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
  }

  try {
    run_bench(depth, iters);
  } catch (const std::exception& ex) {
    std::cerr << "autograd_engine_bench: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
