// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <optional>
#include <cstdio>
#include <cstdlib>

#if !defined(VBT_WITH_CUDA)
#  define VBT_WITH_CUDA 0
#endif
#if VBT_WITH_CUDA
#  include <cuda_runtime_api.h>
#endif

#include "vbt/cuda/stream.h"
#include "vbt/cuda/event_pool.h"
#include "vbt/cuda/graphs.h"

namespace vbt { namespace cuda {

struct RoundPolicy {
  // Small pool uses power-of-two classes up to and including 1 MiB
  static constexpr std::size_t kSmallClasses[12] = {
      512ull, 1024ull, 2048ull, 4096ull, 8192ull, 16384ull,
      32768ull, 65536ull, 131072ull, 262144ull, 524288ull, 1048576ull};
  static constexpr std::size_t kSmallSize    = (1ull << 20); // 1 MiB
  static constexpr std::size_t kLargeQuantum = (2ull << 20); // 2 MiB
};

enum class PoolKind { Small, Large };

enum class BackendKind { Native, Async }; // allocator backend selector

enum class GcReason : std::uint8_t { Oom, FractionCap };

inline std::size_t round_small(std::size_t n) noexcept {
  if (n == 0) return 0;
  for (std::size_t i = 0; i < 12; ++i) {
    if (n <= RoundPolicy::kSmallClasses[i]) return RoundPolicy::kSmallClasses[i];
  }
  return RoundPolicy::kSmallSize; // fallback
}

inline std::size_t round_large(std::size_t n) noexcept {
  auto q = RoundPolicy::kLargeQuantum;
  return ((n + q - 1) / q) * q;
}

inline std::size_t round_size(std::size_t n) noexcept {
  if (n == 0) return 0;
  if (n <= RoundPolicy::kSmallSize) {
    return round_small(n);
  } else {
    return round_large(n);
  }
}

inline PoolKind classify(std::size_t rounded) noexcept {
  return (rounded == 0 || rounded <= RoundPolicy::kSmallSize) ? PoolKind::Small : PoolKind::Large;
}

using StreamId = std::uint64_t;

struct DeviceStats {
  // Gauges (all pools)
  std::uint64_t allocated_bytes_all_current{0};
  std::uint64_t reserved_bytes_all_current{0};
  std::uint64_t max_allocated_bytes_all{0};
  std::uint64_t max_reserved_bytes_all{0};

  std::uint64_t requested_bytes_all_current{0};
  std::uint64_t max_requested_bytes_all{0};

  // Minimal event counters
  std::uint64_t num_alloc_retries{0};
  std::uint64_t num_ooms{0};
  std::uint64_t num_device_alloc{0};
  std::uint64_t num_device_free{0};

  // Extended counters (parity scaffolding)
  std::uint64_t tolerance_fills_count{0};
  std::uint64_t tolerance_fills_bytes{0};
  std::uint64_t deferred_flush_attempts{0};
  std::uint64_t deferred_flush_successes{0};
  std::uint64_t num_prev_owner_fences{0};

  // Fragmentation/fraction/GC diagnostics.
  std::uint64_t inactive_split_blocks_all{0};
  std::uint64_t inactive_split_bytes_all{0};
  std::uint64_t fraction_cap_breaches{0};
  std::uint64_t fraction_cap_misfires{0};
  std::uint64_t gc_passes{0};
  std::uint64_t gc_reclaimed_bytes{0};

  std::uint64_t graphs_pools_created{0};
  std::uint64_t graphs_pools_released{0};
  std::uint64_t graphs_pools_active{0};
};

struct MemorySegmentSnapshot {
  DeviceIndex   device;         // CUDA device index
  std::uint64_t pool_id;        // 0 = global; >0 = graph pool id
  std::uint64_t bytes_reserved; // sum of Block::size across the segment
  std::uint64_t bytes_active;   // sum of Block::size for allocated blocks
  std::uint64_t blocks;         // number of Blocks in the segment
};

std::vector<MemorySegmentSnapshot>
snapshot_memory_segments(std::optional<DeviceIndex> device_filter = std::nullopt);

struct GraphPoolSnapshot {
  MempoolId id;                 // (device, pool_id); id.id != 0
  std::uint64_t segments{0};    // # allocator segments owned by this pool
  std::uint64_t blocks{0};      // total # blocks across all segments
  std::uint64_t bytes_reserved{0}; // sum of reserved bytes across segments
  std::uint64_t bytes_active{0};   // sum of active (allocated) bytes
};

// Snapshot all graph-private pools, optionally filtered by device or id.
std::vector<GraphPoolSnapshot>
snapshot_graph_pools(std::optional<MempoolId> filter = std::nullopt);

#if defined(VBT_INTERNAL_TESTS)
namespace testonly {

struct GraphPoolDebugInfo {
  MempoolId    id;                    // (device, pool_id)
  std::uint64_t refcnt{0};           // internal reference count
  std::uint64_t active_capture_count{0};
  std::uint64_t active_replay_count{0};
  std::uint64_t prewarm_in_progress{0};
};

std::vector<GraphPoolDebugInfo> debug_graph_pools(DeviceIndex dev);

// Force GC/demotion of a quiescent graph pool back to the global pool.
void allocator_debug_gc_pool_now(DeviceIndex dev, MempoolId id);

// Invariant-checked wrapper around snapshot_memory_segments for a single device.
std::vector<MemorySegmentSnapshot> allocator_debug_snapshot(DeviceIndex dev);

} // namespace testonly
#endif

#if VBT_WITH_CUDA && defined(VBT_INTERNAL_TESTS)
// Test hook type for overriding cudaMemGetInfo in allocator constructor.
using CudaMemGetInfoFn = cudaError_t (*)(size_t*, size_t*);

// Install a test hook for cudaMemGetInfo; nullptr restores the default.
void debug_set_cudaMemGetInfo_hook_for_testing(CudaMemGetInfoFn fn) noexcept;

// RAII guard that installs a temporary cudaMemGetInfo hook for the
// lifetime of the guard, restoring the previous hook on destruction.
struct CudaMemGetInfoGuard final {
  CudaMemGetInfoFn prev_{nullptr};
  explicit CudaMemGetInfoGuard(CudaMemGetInfoFn fn) noexcept;
  ~CudaMemGetInfoGuard() noexcept;
};

// calls. These helpers are only available in internal test builds and are
// no-ops in production.
std::uint64_t debug_get_cudaMalloc_call_count_for_testing() noexcept;
void          debug_reset_cudaMalloc_calls_for_testing() noexcept;
#endif

#if defined(VBT_INTERNAL_TESTS)
// Debug wrappers for tracking Allocator::mu_ holdings in internal tests.
struct MuLockGuardDebug;
struct MuUniqueLockDebug;
#endif

class Allocator final {
public:
  static Allocator& get(DeviceIndex dev);

  // Grant snapshot helpers access to internal allocator state for
  // diagnostic graph-pool and segment snapshots.
  friend std::vector<GraphPoolSnapshot>
  snapshot_graph_pools(std::optional<MempoolId> filter);

  friend std::vector<MemorySegmentSnapshot>
  snapshot_memory_segments(std::optional<DeviceIndex> device_filter);

  class AllocateToPoolGuard final {
   public:
    AllocateToPoolGuard() noexcept = default;
    ~AllocateToPoolGuard() noexcept;
    AllocateToPoolGuard(AllocateToPoolGuard&&) noexcept;
    AllocateToPoolGuard& operator=(AllocateToPoolGuard&&) noexcept;
    AllocateToPoolGuard(const AllocateToPoolGuard&) = delete;
    AllocateToPoolGuard& operator=(const AllocateToPoolGuard&) = delete;

    void end() noexcept;
    void cancel() noexcept;
    [[nodiscard]] bool active() const noexcept { return engaged_; }

   private:
    friend class Allocator;
    AllocateToPoolGuard(Allocator* a, MempoolId id) noexcept;

    Allocator* alloc_{nullptr};
    MempoolId  id_{};
    bool       engaged_{false};
  };

  static MempoolId           create_pool_id(DeviceIndex dev);
  static void                retain_pool(DeviceIndex dev, MempoolId id);
  static void                release_pool(DeviceIndex dev, MempoolId id) noexcept;
  static AllocateToPoolGuard begin_allocate_to_pool(DeviceIndex dev, MempoolId id);
  static void                end_allocate_to_pool(DeviceIndex dev, MempoolId id) noexcept;
  static void                cancel_allocate_to_pool(DeviceIndex dev, MempoolId id) noexcept;
  static void                mark_pool_replay_begin(DeviceIndex dev, MempoolId id);
  static void                mark_pool_replay_end(DeviceIndex dev, MempoolId id) noexcept;
  static void                prewarm_graph_pool_for_stream(DeviceIndex dev, MempoolId id,
                                                           Stream stream,
                                                           std::size_t min_total_bytes,
                                                           int min_blocks);

  void* raw_alloc(std::size_t nbytes);
  void* raw_alloc(std::size_t nbytes, Stream s);
  void  raw_delete(void* ptr) noexcept;

  // Metadata-only; records cross-stream use for deferred free fencing.
  void  record_stream(void* ptr, Stream s) noexcept;

  // Opportunistically poll events and reclaim ready blocks. Non-blocking; performs CUDA queries off-lock.
  void  process_events(int max_pops = -1) noexcept;

  void  emptyCache() noexcept; // trims EventPool and frees cached whole segments and free-list blocks

  DeviceIndex device() const noexcept { return dev_; }
  std::size_t debug_cached_segments() const noexcept;

  DeviceStats getDeviceStats() const;
  // Reset peak counters to current values.
  void resetPeakStats() noexcept;
  // Reset accumulated event counters without touching gauges or peaks.
  void resetAccumulatedStats() noexcept;

  bool owns(const void* ptr) const noexcept;
  void* getBaseAllocation(void* ptr, std::size_t* size) const noexcept;

#ifdef VBT_INTERNAL_TESTS
  std::vector<testonly::GraphPoolDebugInfo>
  debug_graph_pools_for_testing() const;

  std::vector<MemorySegmentSnapshot>
  debug_allocator_snapshot_for_testing(DeviceIndex dev) const;

  void debug_gc_pool_now_for_testing(MempoolId id);

  BackendKind debug_backend_kind_for_testing() const noexcept {
    return cfg_.backend;
  }

  // Debug/testing helper: return graph_pool_id tag for a tracked block (0 if unknown)
  std::uint64_t debug_block_pool_id(void* ptr) const noexcept;
  // Debug/testing helper: is TLS routing active for this allocator/device
  bool debug_tls_routing_active() const noexcept;
  bool debug_device_routing_active() const noexcept;

  bool debug_cfg_enable_block_splitting() const noexcept {
    return cfg_.enable_block_splitting;
  }

  bool debug_cfg_enable_cross_stream_fallback_for_testing() const noexcept {
    return cfg_.enable_cross_stream_fallback;
  }

  bool debug_split_enabled() const noexcept {
    return split_enabled();
  }

  // Unordered snapshot of all currently tracked allocation pointers.
  std::vector<void*> debug_tracked_block_ptrs() const;
  // Safe read of Block::is_split_tail for a tracked pointer.
  bool debug_block_is_split_tail(void* ptr) const noexcept;
  // Safe read of Block::gc_age for a tracked pointer (0 if unknown).
  std::uint32_t debug_block_gc_age(void* ptr) const noexcept;

  std::size_t debug_deferred_size_for_testing() const;
  bool        debug_is_in_free_list_for_testing(const void* ptr) const noexcept;

  bool debug_is_oversize_size_for_testing(std::size_t sz) const noexcept;
  bool debug_is_oversize_request_for_testing(std::size_t req) const noexcept;

  bool debug_should_split_for_testing(std::size_t block_size,
                                      bool        is_split_tail,
                                      std::size_t req) const noexcept;

  enum class DebugCandidateDecision : std::uint8_t { Reject, TakeWhole, Split };
  struct DebugCandidateResult {
    DebugCandidateDecision decision;
    bool                   counted_as_tolerance_fill;
    std::size_t            tolerance_waste_bytes;
  };

  DebugCandidateResult debug_evaluate_candidate_for_testing(
      std::size_t block_size,
      bool        is_split_tail,
      std::size_t req,
      std::size_t N_override,
      std::size_t T_override);

  struct DebugBlockInfo {
    void*        ptr{nullptr};
    std::size_t  size{0};
    bool         allocated{false};
    bool         segment_head{false};
    bool         is_split_tail{false};
    void*        prev_ptr{nullptr};
    void*        next_ptr{nullptr};
    std::uint64_t graph_pool_id{0};
    StreamId     owner_stream{0};
  };

  struct DebugSplitResult {
    DebugBlockInfo front_before;   // front block before split
    DebugBlockInfo front_after;    // front block after split
    DebugBlockInfo tail_after;     // tail after split (ptr == nullptr if no tail)

    std::uint64_t inactive_blocks_before{0};
    std::uint64_t inactive_blocks_after{0};
    std::uint64_t inactive_bytes_before{0};
    std::uint64_t inactive_bytes_after{0};

    bool front_in_per_stream_free{false};
    bool front_in_cross_stream_free{false};
    bool tail_in_per_stream_free{false};
    bool tail_in_cross_stream_free{false};
  };

  DebugSplitResult debug_split_block_unlocked_for_testing(
      std::size_t   block_size,
      std::size_t   take_size,
      StreamId      owner_sid,
      std::uint64_t graph_pool_id) noexcept;

  void debug_trigger_split_block_assert_for_testing(
      std::size_t   block_size,
      std::size_t   take_size,
      bool          mark_as_split_tail,
      bool          make_oversize,
      std::uint64_t graph_pool_id) noexcept;

  struct DebugCoalesceScenario {
    // Layout flags and sizes
    bool        has_left{false};
    bool        has_right{false};
    std::size_t left_size{0};
    std::size_t self_size{0};
    std::size_t right_size{0};

    // Pools and owner streams
    std::uint64_t left_graph_pool_id{0};
    std::uint64_t self_graph_pool_id{0};
    std::uint64_t right_graph_pool_id{0};
    StreamId      left_owner_stream{0};
    StreamId      self_owner_stream{0};
    StreamId      right_owner_stream{0};

    // Neighbor allocation & limbo state
    bool left_allocated{false};
    bool right_allocated{false};
    int  left_event_count{0};
    int  right_event_count{0};

    // Tail flags
    bool left_is_split_tail{false};
    bool self_is_split_tail{false};
    bool right_is_split_tail{false};

    // Whether neighbors start in free indices
    bool left_in_free_indices{true};
    bool right_in_free_indices{true};

    // Optional oversize threshold override for this scenario (0 = leave cfg_ as-is).
    std::size_t oversize_threshold_bytes{0};
  };

  struct DebugCoalesceResult {
    DebugBlockInfo left_before;
    DebugBlockInfo self_before;
    DebugBlockInfo right_before;

    DebugBlockInfo head_after;
    DebugBlockInfo left_after;
    DebugBlockInfo self_after;
    DebugBlockInfo right_after;

    std::uint64_t inactive_blocks_before{0};
    std::uint64_t inactive_blocks_after{0};
    std::uint64_t inactive_bytes_before{0};
    std::uint64_t inactive_bytes_after{0};
  };

  DebugCoalesceResult debug_coalesce_neighbors_unlocked_for_testing(
      const DebugCoalesceScenario& scenario) noexcept;

  void debug_trigger_coalesce_neighbors_assert_for_testing(
      bool mark_center_allocated,
      bool make_center_eventful,
      bool insert_center_into_free_indices,
      bool mismatch_neighbor_graph_pool) noexcept;

  std::vector<const char*> debug_allocation_sites_for_testing() const;

  struct DebugTailGaugeSnapshot {
    std::uint64_t stats_blocks{0};
    std::uint64_t stats_bytes{0};
    std::uint64_t recomputed_blocks{0};
    std::uint64_t recomputed_bytes{0};
    bool          consistent{true};
    bool          undercount_only{false};
  };

  DebugTailGaugeSnapshot debug_tail_gauge_snapshot_for_testing(
      bool force_scan = false) const;

  struct DebugStatsSnapshotConsistency {
    bool          native_backend{false};
    bool          split_enabled{false};
    std::uint64_t stats_reserved{0};
    std::uint64_t stats_allocated{0};
    std::uint64_t segs_reserved{0};
    std::uint64_t segs_active{0};
    bool          stats_vs_segments_ok{true};
    bool          stats_vs_pools_ok{true};
  };

  DebugStatsSnapshotConsistency debug_stats_snapshot_consistency_for_testing(
      DeviceIndex dev) const;

  struct DebugFragmentationConfig {
    std::uint64_t seed{0};
    std::size_t   steps{0};
    std::size_t   max_block_size{0};
  };

  void debug_run_fragmentation_fuzzer_for_testing(
      const DebugFragmentationConfig& cfg);

  struct DebugInvariantsReport {
    bool                   ok{true};
    const char*            failed_check{nullptr};
    DebugTailGaugeSnapshot tails_snapshot{};
    std::uint64_t          inactive_split_blocks_gauge{0};
    std::uint64_t          inactive_split_bytes_gauge{0};
  };

  DebugInvariantsReport debug_check_invariants_for_testing() const;

  bool debug_cap_exceeded_for_testing(std::size_t rounded,
                                      std::size_t reserved,
                                      std::size_t limit) const noexcept;
  std::size_t debug_safe_prospective_reserved_for_testing(
      std::size_t rounded,
      std::size_t reserved) const noexcept;

  std::size_t debug_run_gc_pass_for_testing(std::size_t gc_target_bytes,
                                            GcReason    reason) noexcept;

  std::uint64_t debug_raw_alloc_nostream_calls_for_testing() const noexcept;
  std::uint64_t debug_raw_alloc_stream_calls_for_testing() const noexcept;
  void          debug_reset_raw_alloc_call_counters_for_testing() noexcept;
#endif

#if VBT_WITH_CUDA
  // memcpy/peer API
  cudaError_t memcpyAsync(void* dst, int dstDev, const void* src, int srcDev,
                          std::size_t bytes, Stream s, bool p2p_enabled) noexcept;
  static cudaError_t enablePeerAccess(int dev, int peer) noexcept;
  // Memory fraction APIs (CUDA-only)
  // Throws std::invalid_argument if f is outside [0,1].
  void setMemoryFraction(double f);
  double getMemoryFraction() const noexcept {
    return memory_fraction_.load(std::memory_order_relaxed);
  }

#ifdef VBT_INTERNAL_TESTS
  // Pure, lock-free view of the current per-device native fraction cap
  std::size_t debug_current_limit_bytes_for_testing() const noexcept {
    return current_limit_bytes();
  }
#endif
#endif

private:
  explicit Allocator(DeviceIndex dev);

#ifdef VBT_INTERNAL_TESTS
  // Internal debug helper: assert that Allocator::mu_ is not held in this
  // thread when entering GC paths.
  void debug_assert_mu_not_held_for_gc() const noexcept;
  using MuLockGuard  = MuLockGuardDebug;
  using MuUniqueLock = MuUniqueLockDebug;
#else
  using MuLockGuard  = std::lock_guard<std::mutex>;
  using MuUniqueLock = std::unique_lock<std::mutex>;
#endif

  struct Block;
  struct CmpSizeAddr {
    bool operator()(const Block* a, const Block* b) const;
  };
  struct LimboEntry { std::uint64_t token{0}; PooledEvent ev{}; Block* b{nullptr}; };

  struct Config {
    // Backend selection and async mempool knobs
    BackendKind backend{BackendKind::Native};
    double      per_process_memory_fraction{1.0};
    std::size_t release_threshold_bytes{static_cast<std::size_t>(-1)}; // UINT64_MAX by default
    bool        reuse_follow_event_deps{true};
    bool        reuse_allow_opportunistic{true};
    bool        reuse_allow_internal_deps{true};
    // Native backend knobs
    std::size_t max_split_size_bytes{static_cast<std::size_t>(-1)}; // SIZE_MAX
    std::size_t roundup_tolerance_bytes{0};
    std::size_t max_non_split_rounding_bytes{0};
    std::size_t roundup_power2_divisions{0};
    double      garbage_collection_threshold{0.0};
    // Event pool knobs
    std::size_t event_pool_cap{1024};
    std::size_t event_pool_prewarm{0};
    // OOM retry knobs
    std::size_t oom_retry_count{0};
    std::size_t oom_retry_sleep_ms{0};
    // Debug cadence and cross-stream fallback
    std::size_t process_events_every_frees{0};
    bool        enable_cross_stream_fallback{false};
    bool        enable_block_splitting{false};
  } cfg_{};

  static Config parse_env_now_();

  bool split_enabled() const noexcept;

  enum class CandidateDecision : std::uint8_t { Reject, TakeWhole, Split };

  bool is_oversize_block(const Block* b) const noexcept;
  bool is_oversize_request(std::size_t req) const noexcept;

  // Pure heuristic; requires external locking (mu_) as noted in the impl.
  bool should_split_unlocked(const Block* b, std::size_t req) const noexcept;

  CandidateDecision evaluate_candidate(
      Block* b,
      std::size_t req,
      std::size_t M,
      std::size_t N,
      std::size_t T) noexcept;

  // reclaimed bytes (≤ gc_target_bytes). Callers must not hold mu_.
  std::size_t run_gc_pass_if_eligible(std::size_t gc_target_bytes,
                                      GcReason    reason) noexcept;

#if VBT_WITH_CUDA
  // to a per-device native fraction cap in bytes. Pure and lock-free.
  std::size_t current_limit_bytes() const noexcept;

  void maybe_run_fraction_gate(std::size_t nbytes,
                               std::size_t rounded);

  std::vector<Block*> find_candidate_heads_locked() noexcept;
  void detach_segment_for_gc_locked(
      Block*               head,
      std::vector<void*>&  free_ptrs,
      std::vector<Block*>& delete_blocks,
      std::size_t&         freed_bytes) noexcept;
#else
  // Non-CUDA builds: GC helpers are hard no-ops.
  std::size_t current_limit_bytes() const noexcept { return 0; }
#endif

  MempoolId           create_pool_id_();
  void                retain_pool_(MempoolId id);
  void                release_pool_(MempoolId id) noexcept;
  AllocateToPoolGuard begin_allocate_to_pool_(MempoolId id);
  void                end_allocate_to_pool_(MempoolId id) noexcept;
  void                cancel_allocate_to_pool_(MempoolId id) noexcept;
  void                mark_pool_replay_begin_(MempoolId id);
  void                mark_pool_replay_end_(MempoolId id) noexcept;
  void                prewarm_graph_pool_for_stream_(MempoolId id, Stream stream,
                                                     std::size_t min_total_bytes,
                                                     int min_blocks);

  // Internal helpers (not thread-safe; require external locking as noted in impl)
  Block* try_take_free_block_unlocked(StreamId sid, std::size_t rounded_size, std::uint64_t target_pool_id) noexcept;
  Block* try_take_from_cross_stream_unlocked(std::size_t rounded_size, std::uint64_t target_pool_id) noexcept;
  void   remove_from_free_indices_unlocked(Block* b) noexcept;
  void   insert_free_block_unlocked(Block* b, StreamId sid) noexcept;
  void   on_reuse_from_free_list(Block* b,
                                 std::size_t nbytes,
                                 StreamId alloc_sid,
                                 std::uint64_t target_pool_id) noexcept;
  Block* split_block_unlocked(Block* b, std::size_t take_size) noexcept; // returns allocated part (b), inserts tail to free lists
  Block* coalesce_neighbors_unlocked(Block* b) noexcept; // returns merged block pointer

#ifdef VBT_INTERNAL_TESTS
  void debug_note_allocation_transition(Block* b, const char* site);
#endif

  DeviceIndex dev_{};
  mutable std::mutex mu_{};  // allocator mutex (global order: mu_ -> events_.pool_.mu)
  EventPool events_;

  struct Segment { void* base{nullptr}; std::size_t size{0}; bool idle{true}; };
  std::vector<Segment> idle_segments_;

  struct Block {
    DeviceIndex device{0};
    StreamId    alloc_stream{0};
    StreamId    owner_stream{0};
    std::unordered_set<StreamId> stream_uses{};
    void*       ptr{nullptr};
    std::size_t size{0};
    std::size_t requested_size{0};
    bool        allocated{false};
    bool        mapped{false};
    Block*      prev{nullptr};
    Block*      next{nullptr};
    bool        segment_head{false};
    bool        is_split_tail{false};
    int         event_count{0};
    std::uint64_t graph_pool_id{0};
    std::uint32_t gc_age{0};  // GC pass age; meaningful only for fully idle global segment heads
  };

  std::unordered_map<void*, Block*> by_ptr_;
  std::unordered_set<Block*>        active_blocks_;
  std::unordered_map<StreamId, std::set<Block*, CmpSizeAddr>> per_stream_free_;
  std::set<Block*, CmpSizeAddr>     cross_stream_free_;
  std::unordered_map<StreamId, std::deque<LimboEntry>> limbo_;
  std::uint64_t limbo_token_seq_{0};

  struct DeferredFree { Block* b{nullptr}; StreamId owner_sid{0}; std::vector<StreamId> streams; };
  std::deque<DeferredFree> deferred_;

  struct GraphPrivatePool final {
    std::uint32_t refcnt{0};
    std::uint32_t active_capture_count{0};
    std::uint32_t active_replay_count{0};
    std::uint64_t begins{0};
    std::uint64_t ends{0};
    std::uint64_t cancels{0};
    std::uint32_t prewarm_in_progress{0};
  };
  std::unordered_map<std::uint64_t, GraphPrivatePool> graph_pools_;
  std::uint64_t next_graph_pool_id_{1};

  void gc_pool_locked(std::uint64_t id_val, GraphPrivatePool& pool);

  DeviceStats stats_{};

#ifdef VBT_INTERNAL_TESTS
  std::vector<const char*> debug_allocation_sites_{};
#endif

  // Debug cadence counter
  std::atomic<std::size_t> free_count_{0};
  std::atomic<double>      memory_fraction_{1.0};

  // Total memory on dev_ in bytes, as reported by cudaMemGetInfo.
  // 0 means "unknown" (CPU-only build, no devices, or query failed).
  std::size_t              device_total_bytes_{0};

  std::atomic<bool>        routing_active_flag_{false};
};

#if defined(VBT_INTERNAL_TESTS) && defined(VBT_PARANOID_RUNTIME_CHECKS)
  // Debug-only invariant scan; do not call while holding mu_.
  #define VBT_DEBUG_CHECK_INVARIANTS(alloc)                                   \
    do {                                                                      \
      ::vbt::cuda::Allocator::DebugInvariantsReport rep =                    \
          (alloc).debug_check_invariants_for_testing();                      \
      if (!rep.ok) {                                                          \
        std::fprintf(stderr,                                                  \
          "[vbt::cuda::Allocator] invariant failed: %s (tails=(%llu,%llu), " \
          "gauges=(%llu,%llu)\n",                                         \
          rep.failed_check ? rep.failed_check : "unknown",                  \
          static_cast<unsigned long long>(                                   \
              rep.tails_snapshot.recomputed_blocks),                          \
          static_cast<unsigned long long>(                                   \
              rep.tails_snapshot.recomputed_bytes),                           \
          static_cast<unsigned long long>(rep.inactive_split_blocks_gauge),   \
          static_cast<unsigned long long>(rep.inactive_split_bytes_gauge));   \
        std::abort();                                                         \
      }                                                                       \
    } while (0)
#else
  #define VBT_DEBUG_CHECK_INVARIANTS(alloc) do { } while (0)
#endif

}} // namespace vbt::cuda
