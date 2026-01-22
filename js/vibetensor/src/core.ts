// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { addon, CallOptions, NativeTensorHandle } from './addon.js';
import type {
  CudaStreamHandle,
  CudaEventHandle,
  DlpackCapsule,
  FromDlpackOptions,
  H2DOptions,
  ExternalMemoryStats,
  CudaMemoryGaugeFamily,
  CudaMemoryStatsNested,
  CudaMemoryStatsFlat,
  CudaMemorySegmentSnapshot,
  DebugAsyncStats,
  LogEntry,
  LogLevel,
  LogCategory,
  FabricStatsSnapshot,
  FabricEventSnapshot,
} from './internal/types.js';

const hasNativeDlpack =
  typeof (addon as any).toDlpack === 'function' &&
  typeof (addon as any).fromDlpackCapsule === 'function';

const hasNativeFabric =
  typeof (addon as any).fabricStatsSnapshot === 'function' &&
  typeof (addon as any).fabricEventsSnapshot === 'function';

export type DeviceString = 'cpu' | `cuda:${number}`;

export type DTypeString = 'float32' | 'int64' | 'bool'; // minimal set for Node demo metadata scalars

export interface TensorHandle extends NativeTensorHandle {
  readonly __brand: 'TensorHandle';
}

// ---- CUDA runtime basics ----------------------------------------------------

/** Cached CUDA availability flag, mirroring Python's _has_cuda attr. */
export function cudaIsAvailable(): boolean {
  return Boolean(addon.hasCuda());
}

/** Cached CUDA device count from addon initialization. */
export function cudaDeviceCount(): number {
  return Number(addon.cudaDeviceCount());
}

/** Backwards-compatible helper kept from earlier tests. */
export function hasCuda(): boolean {
  return cudaIsAvailable();
}

/**
 * Returns the current CUDA device index for this JS thread.
 *
 * Throws an Error with code "ENOCUDA" when CUDA is unavailable.
 */
export function cudaCurrentDevice(): number {
  return Number(addon.cudaCurrentDevice());
}

/**
 * Sets the current CUDA device for this JS thread.
 *
 * Validation and error mapping are handled inside the addon:
 *  - TypeError with code "EINVAL" for invalid or out-of-range indices.
 *  - Error with code "ENOCUDA" when CUDA is unavailable.
 *  - Error with code "ERUNTIME" on driver failures.
 */
export function cudaSetDevice(idx: number): void {
  addon.cudaSetDevice(idx);
}

// ---- Tensor factories and dispatcher wrappers -------------------------------

export interface ZerosOptions {
  device?: DeviceString;
  dtype?: DTypeString;
}

export function zeros(
  sizes: readonly number[],
  opts: ZerosOptions = {},
): TensorHandle {
  const device = opts.device ?? 'cpu';
  const dtype: DTypeString = opts.dtype ?? 'float32';
  if (device !== 'cpu') {
    throw new Error('zeros: only cpu device supported');
  }
  if (dtype !== 'float32') {
    throw new Error("zeros: only dtype 'float32' supported");
  }
  return addon.zeros(Array.from(sizes), { device, dtype }) as TensorHandle;
}

export function scalarInt64(value: number): TensorHandle {
  return addon.scalarInt64(value) as TensorHandle;
}

export function scalarBool(value: boolean): TensorHandle {
  return addon.scalarBool(value) as TensorHandle;
}

export function scalarFloat32(value: number): TensorHandle {
  return addon.scalarFloat32(value) as TensorHandle;
}

// ---- DLPack helpers ----------------------------------------------------------

export function toDlpackCapsule(handle: TensorHandle): DlpackCapsule {
  // Addon throws TypeError(EINVAL) on forged handles when available.
  if (!hasNativeDlpack) {
    const err: any = new Error(
      'vibetensor: Node DLPack bindings are unavailable; rebuild with VBT_BUILD_NODE and Node.js headers',
    );
    err.code = 'ENOSYS';
    throw err;
  }
  return (addon as any).toDlpack(handle as NativeTensorHandle) as unknown as DlpackCapsule;
}

export function fromDlpackCapsule(
  cap: DlpackCapsule,
  opts: FromDlpackOptions = {},
): Promise<TensorHandle> {
  const { device, copy } = opts;

  if (!hasNativeDlpack) {
    const err: any = new Error(
      'vibetensor: Node DLPack bindings are unavailable; rebuild with VBT_BUILD_NODE and Node.js headers',
    );
    err.code = 'ENOSYS';
    return Promise.reject(err);
  }

  let targetDevice: number | undefined;
  if (typeof device !== 'undefined') {
    if (typeof device !== 'number' || !Number.isFinite(device)) {
      const err: any = new TypeError(
        'dlpack.fromDlpack: device must be a finite number or undefined',
      );
      err.code = 'EINVAL';
      return Promise.reject(err);
    }
    if (!Number.isInteger(device) || device < 0) {
      const err: any = new TypeError(
        'dlpack.fromDlpack: device index must be a non-negative integer',
      );
      err.code = 'EINVAL';
      return Promise.reject(err);
    }
    targetDevice = device;

    // ENOCUDA pre-check for explicit CUDA targets in Node.
    if (targetDevice > 0 && !cudaIsAvailable()) {
      const err: any = new Error(
        'dlpack.fromDlpack: CUDA target device requested but CUDA is unavailable',
      );
      err.code = 'ENOCUDA';
      return Promise.reject(err);
    }
  }

  return (addon as any)
    .fromDlpackCapsule(
      cap as unknown as object,
      {
        copy,
        device: targetDevice,
      },
    )
    .then((h: any) => h as TensorHandle);
}

export function callOpSync(
  name: string,
  args: any[],
  opts?: CallOptions,
): any {
  return addon.callOpSync(name, args, opts);
}

export function callOp(
  name: string,
  args: any[],
  opts?: CallOptions,
): Promise<any> {
  return addon.callOp(name, args, opts);
}

export function callOpNoOverride(
  name: string,
  args: any[],
  opts?: CallOptions,
): Promise<any> {
  return addon.callOpNoOverride(name, args, opts);
}

export function _debugDispatcherStats() {
  return addon._debugDispatcherStats();
}

// ---- Internal CUDA handle helpers used by the public cuda overlay ----------

export function createCudaStream(deviceIndex: number): CudaStreamHandle {
  // The addon constructor performs validation and throws TypeError(EINVAL)
  // or Error(ENOCUDA/ERUNTIME) as appropriate.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ctor: any = (addon as any).CudaStream;
  if (typeof ctor !== 'function') {
    throw new Error('vibetensor: native CudaStream constructor is unavailable');
  }
  return new ctor(deviceIndex) as CudaStreamHandle;
}

export function createCudaEvent(
  opts?: { enableTiming?: boolean },
): CudaEventHandle {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ctor: any = (addon as any).CudaEvent;
  if (typeof ctor !== 'function') {
    throw new Error('vibetensor: native CudaEvent constructor is unavailable');
  }
  return new ctor(opts ?? {}) as CudaEventHandle;
}

export function cudaStreamDeviceIndex(handle: CudaStreamHandle): number {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.deviceIndex !== 'function') {
    throw new TypeError('cudaStreamDeviceIndex: invalid Stream handle');
  }
  return Number(anyHandle.deviceIndex());
}

export function cudaStreamQuery(handle: CudaStreamHandle): boolean {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.query !== 'function') {
    throw new TypeError('cudaStreamQuery: invalid Stream handle');
  }
  return Boolean(anyHandle.query());
}

export function cudaStreamToString(handle: CudaStreamHandle): string {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.toString !== 'function') {
    throw new TypeError('cudaStreamToString: invalid Stream handle');
  }
  return String(anyHandle.toString());
}

export function cudaStreamSynchronizeAsync(
  handle: CudaStreamHandle,
): Promise<void> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.synchronize !== 'function') {
    return Promise.reject(
      new TypeError('cudaStreamSynchronizeAsync: invalid Stream handle'),
    );
  }
  return anyHandle.synchronize();
}

export function cudaEventIsCreated(handle: CudaEventHandle): boolean {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.isCreated !== 'function') {
    throw new TypeError('cudaEventIsCreated: invalid Event handle');
  }
  return Boolean(anyHandle.isCreated());
}

export function cudaEventQuery(handle: CudaEventHandle): boolean {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.query !== 'function') {
    throw new TypeError('cudaEventQuery: invalid Event handle');
  }
  return Boolean(anyHandle.query());
}

export function cudaEventToString(handle: CudaEventHandle): string {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.toString !== 'function') {
    throw new TypeError('cudaEventToString: invalid Event handle');
  }
  return String(anyHandle.toString());
}

export function cudaEventSynchronizeAsync(
  handle: CudaEventHandle,
): Promise<void> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const anyHandle: any = handle;
  if (!anyHandle || typeof anyHandle.synchronize !== 'function') {
    return Promise.reject(
      new TypeError('cudaEventSynchronizeAsync: invalid Event handle'),
    );
  }
  return anyHandle.synchronize();
}

export async function cudaH2DAsync(
  src: ArrayBufferView,
  sizes: readonly number[],
  opts: H2DOptions,
): Promise<TensorHandle> {
  if (!cudaIsAvailable()) {
    const err: any = new Error('cuda.h2d: CUDA not available');
    err.code = 'ENOCUDA';
    return Promise.reject(err);
  }

  const { device, dtype } = opts;
  return (addon as any)
    ._cudaH2DAsync(src, Array.from(sizes), { device, dtype })
    .then((h: any) => h as TensorHandle);
}

export async function cudaD2HAsync(
  handle: TensorHandle,
): Promise<ArrayBufferView> {
  if (!cudaIsAvailable()) {
    const err: any = new Error('cuda.d2h: CUDA not available');
    err.code = 'ENOCUDA';
    return Promise.reject(err);
  }

  return (addon as any)._cudaD2HAsync(handle as any);
}


export interface ExternalMemoryStatsRaw {
  ownersAlive: number;
  bytesAccounted: number;
}

export function externalMemoryStatsRaw(): ExternalMemoryStatsRaw {
  const val = (addon as any)._external_memory_stats?.() ?? {};
  const owners = Number((val as any).ownersAlive ?? 0);
  const bytes = Number((val as any).bytesAccounted ?? 0);
  return {
    ownersAlive: Number.isFinite(owners) && owners >= 0 ? owners : 0,
    bytesAccounted: Number.isFinite(bytes) && bytes >= 0 ? bytes : 0,
  };
}

function zeroFamily(): CudaMemoryGaugeFamily {
  return { all: { current: 0, peak: 0, allocated: 0, freed: 0 } };
}

export function cudaMemoryStatsAsNested(device?: number): CudaMemoryStatsNested {
  const rawVal = (addon as any)._cudaMemoryStatsAsNested?.(device);
  const raw = (rawVal && typeof rawVal === 'object' ? rawVal : {}) as Partial<CudaMemoryStatsNested>;

  const allocated = raw.allocated_bytes ?? zeroFamily();
  const reserved = raw.reserved_bytes ?? zeroFamily();
  const requested = raw.requested_bytes ?? zeroFamily();

  const ds = (raw.device_stats && typeof raw.device_stats === 'object'
    ? raw.device_stats
    : { aggregated: {} as Record<string, number> }) as { aggregated?: Record<string, number> };

  const aggregated = ds.aggregated ?? {};

  return {
    ...raw,
    allocated_bytes: allocated,
    reserved_bytes: reserved,
    requested_bytes: requested,
    device_stats: { aggregated },
  } as CudaMemoryStatsNested;
}

export function cudaMemoryStats(device?: number): CudaMemoryStatsFlat {
  const flatVal = (addon as any)._cudaMemoryStats?.(device);
  const src = flatVal && typeof flatVal === 'object' ? (flatVal as Record<string, unknown>) : {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(src)) {
    const n = Number(v);
    // Collapse NaN/Infinity to 0 for robustness.
    out[k] = Number.isFinite(n) ? n : 0;
  }
  return out;
}

export function cudaMemorySnapshot(device?: number): CudaMemorySegmentSnapshot[] {
  const arrVal = (addon as any)._cudaMemorySnapshot?.(device);
  const arr = Array.isArray(arrVal) ? (arrVal as CudaMemorySegmentSnapshot[]) : [];
  // Return a shallow copy to avoid accidental aliasing.
  return arr.slice();
}

export function _debugAsyncStats(): DebugAsyncStats {
  const raw = (addon as any)._debugAsyncStats?.() ?? {};
  // Minimal normalization: coerce to numbers, clamp negatives to 0.
  function nn(x: unknown): number {
    const n = Number(x);
    return Number.isFinite(n) && n >= 0 ? n : 0;
  }
  return {
    inflight: nn((raw as any).inflight),
    peakInflight: nn((raw as any).peakInflight),
    maxInflight: nn((raw as any).maxInflight),
    dispatcher: {
      totalCalls: nn((raw as any).dispatcher?.totalCalls),
      failedCalls: nn((raw as any).dispatcher?.failedCalls),
    },
    h2d: {
      started: nn((raw as any).h2d?.started),
      completed: nn((raw as any).h2d?.completed),
      failed: nn((raw as any).h2d?.failed),
      bytes: nn((raw as any).h2d?.bytes),
    },
    d2h: {
      started: nn((raw as any).d2h?.started),
      completed: nn((raw as any).d2h?.completed),
      failed: nn((raw as any).d2h?.failed),
      bytes: nn((raw as any).d2h?.bytes),
    },
  };
}

export function drainNativeLogs(maxEntries = 256): LogEntry[] {
  const n = Number.isFinite(maxEntries) && maxEntries > 0 ? Math.floor(maxEntries) : 256;
  const raw = (addon as any)._drainLogs?.(n);
  if (!Array.isArray(raw)) return [];

  const out: LogEntry[] = [];
  for (const item of raw as any[]) {
    if (!item || typeof item !== 'object') continue;
    const r: any = item;
    const tsVal = Number(r.ts ?? 0);
    const ts = Number.isFinite(tsVal) && tsVal >= 0 ? tsVal : 0;
    const level = String(r.level ?? 'info') as LogLevel;
    const category = String(r.category ?? 'dispatcher') as LogCategory;
    const message = String(r.message ?? '');
    const dataRaw = r.data;
    const data =
      dataRaw && typeof dataRaw === 'object'
        ? (dataRaw as Record<string, string | number | boolean | null>)
        : undefined;
    out.push({ ts, level, category, message, data });
  }
  return out;
}

export function setNativeLoggingEnabled(
  enabled: boolean,
  level: LogLevel,
  categories?: LogCategory[],
): void {
  // level/category validation happens natively; JS only forwards.
  (addon as any)._setLoggingEnabled?.(Boolean(enabled), String(level), categories?.slice());
}


export function getFabricStats(): FabricStatsSnapshot {
  if (!hasNativeFabric) {
    const err: any = new Error(
      'vibetensor: Fabric observability bindings are unavailable; rebuild with VBT_BUILD_NODE and Node.js headers',
    );
    err.code = 'ENOSYS';
    throw err;
  }
  return (addon as any).fabricStatsSnapshot() as FabricStatsSnapshot;
}

export function getFabricEvents(
  minSeq: bigint = 0n,
  maxEvents: number = 1024,
): FabricEventSnapshot {
  if (typeof minSeq !== 'bigint') {
    const err: any = new TypeError('fabric.events: minSeq must be a BigInt');
    err.code = 'EINVAL';
    throw err;
  }
  if (minSeq < 0n) {
    const err: any = new TypeError('fabric.events: minSeq must be >= 0');
    err.code = 'EINVAL';
    throw err;
  }

  if (typeof maxEvents !== 'number' || !Number.isFinite(maxEvents) || maxEvents < 0) {
    const err: any = new TypeError('fabric.events: maxEvents must be a non-negative number');
    err.code = 'EINVAL';
    throw err;
  }
  if (!Number.isInteger(maxEvents)) {
    const err: any = new TypeError('fabric.events: maxEvents must be an integer');
    err.code = 'EINVAL';
    throw err;
  }

  if (!hasNativeFabric) {
    const err: any = new Error(
      'vibetensor: Fabric observability bindings are unavailable; rebuild with VBT_BUILD_NODE and Node.js headers',
    );
    err.code = 'ENOSYS';
    throw err;
  }

  return (addon as any).fabricEventsSnapshot(minSeq, maxEvents) as FabricEventSnapshot;
}
