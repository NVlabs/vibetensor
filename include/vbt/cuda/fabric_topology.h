// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vbt { namespace cuda { namespace fabric {

// Forward declaration from fabric_state.h; used only for stats pointers
// in topology helpers. Full definition lives in fabric_state.h.
struct FabricInitStats;

//
// - device_count >= 0.
// - If device_count == 0, matrices and clique vectors may be empty.
// - If device_count > 0:
//   - can_access_peer.size() == device_count and each inner vector has
//     device_count entries.
//   - p2p_enabled has the same shape as can_access_peer.
//   - clique_id.size() == device_count.
//   - clique_size.size() == num_cliques, with num_cliques >= 1.
//   - For each device d, 0 <= clique_id[d] < clique_size.size().
struct FabricTopology {
  int device_count{0};

  // Capability: result of cudaDeviceCanAccessPeer(src, dst). Size is either
  // 0x0 (device_count == 0) or device_count x device_count.
  std::vector<std::vector<bool>> can_access_peer;  // [src][dst]

  // Runtime peer enablement status for each ordered pair (src, dst).
  //
  // Fabric initialization best-effort enables peer access for every pair with
  // mutual can_access_peer capability and marks the symmetric entries true.
  // Entries remain false when capability is absent or enablement failed.
  std::vector<std::vector<bool>> p2p_enabled;      // [src][dst]

  // Connected components over the symmetric capability graph:
  // edge(src,dst) iff can_access_peer[src][dst] && can_access_peer[dst][src].
  std::vector<int> clique_id;    // length == device_count; -1 before assignment
  std::vector<int> clique_size;  // length == num_cliques; >= 1 per clique
};

// by dispatcher/TensorIterator integration.
bool in_same_fabric(int primary, int remote, const FabricTopology& topo) noexcept;

bool is_fabric_usable_for_with_primary(
    int primary,
    std::span<const int> devices,
    const FabricTopology& topo) noexcept;

// Build a FabricTopology snapshot from the CUDA runtime. Called exactly once
// per process in production code (from fabric_state initialization) and
// potentially multiple times in tests via reset hooks.
//
// On CPU-only builds (VBT_WITH_CUDA == 0), this leaves topo.device_count == 0
// and does not touch any CUDA APIs.
//
// When stats is non-null, increments topology_build_attempts and, on
// exceptional failures, topology_build_failures.
//
// disable_reason, when non-null, may be populated with a short, human-readable
// description of any fatal initialization error; it is left unchanged on
// success.
void build_fabric_topology_from_runtime(
    FabricTopology& topo,
    FabricInitStats* stats,
    std::string* disable_reason);

}}} // namespace vbt::cuda::fabric
