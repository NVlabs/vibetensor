// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Internal branded handle types used by the CUDA and DLPack overlays. These
// are opaque to user code; only core.ts/cuda.ts/dlpack.ts know how to
// interact with them.

export interface CudaStreamHandle {
  readonly __brand: 'CudaStreamHandle';
}

export interface CudaEventHandle {
  readonly __brand: 'CudaEventHandle';
}

declare const DlpackCapsuleBrand: unique symbol;
export type DlpackCapsule = { readonly [DlpackCapsuleBrand]: 'DlpackCapsule' } & object;

export interface FromDlpackOptions {
  /**
   * Destination logical device index.
   *
   * Semantics are capsule-type–dependent; see design/napi/README.md §5.4.
   *
   * CPU capsules (`kDLCPU`, `device_id == 0`):
   *   - `undefined` or `0` → CPU tensor (zero-copy alias).
   *   - `n > 0`           → CUDA tensor on `cuda:n` via CPU→CUDA copy.
   *
   * CUDA device capsules (`kDLCUDA` / `kDLCUDAManaged`):
   *   - `undefined`       → CUDA tensor on provider device `cuda:dev_id`.
   *   - `n === dev_id`    → same-device import (alias or copy per `copy`).
   *   - `n !== dev_id`    → rejected as cross-device (no cross-GPU import).
   *
   * CUDA host capsules (`kDLCUDAHost`):
   *   - `undefined`       → copy to `cuda:dev_id`.
   *   - `n >= 0`          → copy to `cuda:n`.
   */
  device?: number;

  /**
   * Copy vs alias hint for CUDA capsules.
   *
   * CPU capsules (`kDLCPU`):
   *   - Ignored; imports are always zero-copy aliases.
   *
   * CUDA device capsules:
   *   - `copy === false` → alias-only import on same device via `from_dlpack`.
   *                         If aliasing fails (e.g., alignment), import fails;
   *                         there is no fallback to copy.
   *   - `copy === true`  → copy import via `from_dlpack_cuda_copy` on same device.
   *   - `copy` omitted   → default is copy (`true`) for CUDA device capsules.
   *
   * CUDA host capsules (`kDLCUDAHost`):
   *   - Always copied via `from_dlpack_cuda_copy`; `copy` is ignored.
   */
  copy?: boolean;
}

// Subset of dtypes supported by the H2D/D2H helpers.
export type H2DDType = 'float32' | 'int32' | 'int64' | 'bool';

export interface H2DOptions {
  /** CUDA device index; default: current CUDA device for this JS thread. */
  device?: number;
  /** Data type; must match the TypedArray element type. */
  dtype: H2DDType;
}

// ---- External memory stats & CUDA allocator observability -----------------

/** Snapshot of JS-visible external memory accounting state. */
export interface ExternalMemoryStats {
  /**
   * Approximate count of live ExternalMemoryOwner instances whose `released`
   * flag is still false (including zero-byte owners).
   */
  ownersAlive: number;

  /**
   * Total bytes currently accounted to V8 via napi_adjust_external_memory.
   * This is the sum of `bytes` for which +bytes has been applied but -bytes
   * has not, and is GC-oriented rather than a precise allocator meter.
   */
  bytesAccounted: number;
}

/** Family of allocator gauges for a given byte-class (allocated, reserved, requested). */
export interface CudaMemoryGaugeFamily {
  all: {
    current: number;   // `<family>.all.current`
    peak: number;      // `<family>.all.peak`
    allocated: number; // `<family>.all.allocated`
    freed: number;     // `<family>.all.freed`
  };
}

export interface CudaDeviceAggregatedStats {
  [name: string]: number;  // e.g. num_alloc_retries, num_ooms, gc_passes
}

/** Nested view corresponding to Python memory_stats_as_nested_dict. */
export interface CudaMemoryStatsNested {
  allocated_bytes: CudaMemoryGaugeFamily;
  reserved_bytes: CudaMemoryGaugeFamily;
  requested_bytes: CudaMemoryGaugeFamily;
  device_stats: {
    aggregated: CudaDeviceAggregatedStats;
  };
  // Allow future extensions without breaking consumers.
  [family: string]: any;
}

/** Flat key→value view, like Python memory_stats. */
export type CudaMemoryStatsFlat = Readonly<Record<string, number>>;

export interface CudaMemorySegmentSnapshot {
  device: number;      // logical CUDA device index
  poolId: number;      // allocator-specific pool identifier
  bytesReserved: number;
  bytesActive: number;
  blocks: number;      // number of blocks in this segment
}

// ---- Logging types ---------------------------------------------------------

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogCategory =
  | 'dispatcher'
  | 'cuda-runtime'
  | 'cuda-allocator'
  | 'dlpack'
  | 'h2d'
  | 'd2h'
  | 'external-memory';

export interface LogEntry {
  ts: number;              // milliseconds since process start (or epoch)
  level: LogLevel;
  category: LogCategory;
  message: string;
  data?: Record<string, string | number | boolean | null>;
}

export type LogSink = (entry: LogEntry) => void;

export interface LogConfig {
  level: LogLevel;            // required; overlay applies default 'info'
  categories?: LogCategory[]; // default: all categories
  maxEntriesPerDrain?: number;// soft cap per drain; default 256
}

// ---- Async metrics ---------------------------------------------------------

export interface AsyncCategoryStats {
  started: number;
  completed: number;
  failed: number;
  bytes: number;      // total bytes scheduled for transfer (H2D/D2H only)
}

export interface DebugAsyncStats {
  inflight: number;       // current g_inflight_ops
  peakInflight: number;   // max g_inflight_ops observed
  maxInflight: number;    // g_max_inflight_ops
  dispatcher: {
    totalCalls: number;
    failedCalls: number;
  };
  h2d: AsyncCategoryStats;
  d2h: AsyncCategoryStats;
}

// ---- Fabric stats & events -----------------------------------------------

export interface FabricStatsReasonsSnapshot {
  no_p2p: bigint;
  requires_grad: bigint;
  in_backward: bigint;
  small_tensor: bigint;
}

export interface FabricPerDeviceStatsSnapshot {
  device_index: number;
  ops_as_primary: bigint;
  ops_as_remote: bigint;
  remote_bytes_read: bigint;
  remote_bytes_written: bigint;
}

export interface FabricStatsSnapshot {
  mesh_builds: bigint;
  p2p_pairs_enabled: bigint;
  p2p_pairs_failed: bigint;

  fabric_ops_attempted: bigint;
  fabric_ops_hit: bigint;
  fabric_ops_fallback: bigint;

  remote_bytes_read: bigint;
  remote_bytes_written: bigint;

  inflight_ops_current: bigint;
  inflight_ops_peak: bigint;

  event_queue_len_peak: bigint;
  event_dropped_total: bigint;
  event_failures_total: bigint;

  mode_enable_calls: bigint;
  mode_disable_calls: bigint;
  mode_set_failures: bigint;

  reasons: FabricStatsReasonsSnapshot;
  per_device: FabricPerDeviceStatsSnapshot[];
}

export type FabricEventKind =
  | 'op_enqueue'
  | 'op_complete'
  | 'op_fallback'
  | 'op_error'
  | 'mode_changed'
  | 'event_lifetime_toggled'
  | 'events_mode_changed'
  | 'unknown';

export type FabricEventLevel = 'debug' | 'info' | 'warn' | 'error' | 'unknown';

export interface FabricEvent {
  seq: bigint;
  t_ns: bigint;
  primary_device: number;
  other_device: number;
  kind: FabricEventKind;
  level: FabricEventLevel;
  op_id: bigint;
  numel: bigint;
  bytes: bigint;
  reason_raw: number;
  message: string | null;
}

export interface FabricEventSnapshot {
  base_seq: bigint;
  next_seq: bigint;
  dropped_total: bigint;
  capacity: number;
  events: FabricEvent[];
}
