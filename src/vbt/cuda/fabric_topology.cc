// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/fabric_topology.h"

#include <algorithm>
#include <queue>

#include "vbt/cuda/device.h"
#include "vbt/cuda/fabric_state.h"

#ifndef VBT_WITH_CUDA
#  define VBT_WITH_CUDA 0
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

namespace vbt { namespace cuda { namespace fabric {

namespace {

constexpr const char* kTopologyCudaErrorMessage =
    "[Fabric] CUDA error during topology discovery; Fabric is disabled";

}  // namespace

void build_fabric_topology_from_runtime(
    FabricTopology& topo,
    FabricInitStats* stats,
    std::string* disable_reason) {
  // Reset topology to an empty state first.
  topo.device_count = 0;
  topo.can_access_peer.clear();
  topo.p2p_enabled.clear();
  topo.clique_id.clear();
  topo.clique_size.clear();

  if (stats) {
    ++stats->topology_build_attempts;
  }

#if !VBT_WITH_CUDA
  (void)disable_reason;
  // CPU-only build: leave device_count == 0 and matrices empty.
  return;
#else

#  if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
  // Test hook: allow synthetic topologies without touching CUDA APIs.
  if (auto& hooks = fabric_test_hooks(); hooks.fake_topology_builder) {
    try {
      hooks.fake_topology_builder(topo);
    } catch (...) {
      if (stats) {
        ++stats->topology_build_failures;
      }
      if (disable_reason && disable_reason->empty()) {
        *disable_reason = kTopologyCudaErrorMessage;
      }
      throw;
    }
    return;
  }
#  endif  // VBT_INTERNAL_TESTS

  try {
    int count = vbt::cuda::device_count();
    if (count < 0) {
      count = 0;
    }
    topo.device_count = count;
    if (count == 0) {
      return;
    }

    topo.can_access_peer.assign(count, std::vector<bool>(count, false));
    topo.p2p_enabled.assign(count, std::vector<bool>(count, false));

    // Query P2P capability for each ordered pair (src, dst), src != dst.
    for (int src = 0; src < count; ++src) {
      for (int dst = 0; dst < count; ++dst) {
        if (src == dst) continue;
        int can_access = 0;
        cudaError_t st = cudaDeviceCanAccessPeer(&can_access, src, dst);
        if (st == cudaSuccess) {
          topo.can_access_peer[src][dst] = (can_access != 0);
        } else {
          // Treat errors as lack of P2P capability but do not throw.
          topo.can_access_peer[src][dst] = false;
        }
      }
    }

    // Build cliques over the symmetric capability graph where an undirected
    // edge exists iff can_access_peer[src][dst] && can_access_peer[dst][src].
    const int n = topo.device_count;
    topo.clique_id.assign(n, -1);
    topo.clique_size.clear();

    int num_cliques = 0;
    std::vector<int> queue;
    queue.reserve(n);

    for (int d = 0; d < n; ++d) {
      if (topo.clique_id[d] != -1) continue;
      // Start a new clique at d.
      const int cid = num_cliques++;
      int size = 0;
      queue.clear();
      queue.push_back(d);
      topo.clique_id[d] = cid;
      for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        const int cur = queue[qi];
        ++size;
        for (int other = 0; other < n; ++other) {
          if (cur == other) continue;
          if (topo.clique_id[other] != -1) continue;
          if (topo.can_access_peer[cur][other] &&
              topo.can_access_peer[other][cur]) {
            topo.clique_id[other] = cid;
            queue.push_back(other);
          }
        }
      }
      topo.clique_size.push_back(size);
    }

  } catch (...) {
    if (stats) {
      ++stats->topology_build_failures;
    }
    if (disable_reason && disable_reason->empty()) {
      *disable_reason = kTopologyCudaErrorMessage;
    }
    throw;
  }
#endif  // VBT_WITH_CUDA
}

bool in_same_fabric(int primary, int remote, const FabricTopology& topo) noexcept {
  if (primary == remote) return true;
  if (primary < 0 || remote < 0 ||
      primary >= topo.device_count || remote >= topo.device_count) {
    return false;
  }
  if (topo.clique_id.empty()) return false;
  const int cid_p = topo.clique_id[primary];
  const int cid_r = topo.clique_id[remote];
  if (cid_p < 0 || cid_r < 0) return false;
  return cid_p == cid_r;
}

bool is_fabric_usable_for_with_primary(
    int primary,
    std::span<const int> devices,
    const FabricTopology& topo) noexcept {
  if (primary < 0 || primary >= topo.device_count) return false;
  if (topo.clique_id.empty()) return false;

  // Build deduplicated set S = {primary} U devices.
  std::vector<int> uniq;
  uniq.reserve(devices.size() + 1);
  auto add = [&](int d) {
    if (d < 0 || d >= topo.device_count) return;
    if (std::find(uniq.begin(), uniq.end(), d) == uniq.end()) {
      uniq.push_back(d);
    }
  };
  add(primary);
  for (int d : devices) add(d);

  if (uniq.empty()) return false;

  const int cid_primary = topo.clique_id[primary];
  if (cid_primary < 0) return false;

  for (int d : uniq) {
    if (d < 0 || d >= topo.device_count) return false;
    if (topo.clique_id[d] != cid_primary) return false;
  }
  return true;
}

}}} // namespace vbt::cuda::fabric
