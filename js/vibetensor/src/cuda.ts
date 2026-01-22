// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as core from './core.js';
import { Tensor } from './tensor.js';
import type {
  CudaStreamHandle,
  CudaEventHandle,
  H2DOptions,
  ExternalMemoryStats,
  CudaMemoryStatsNested,
  CudaMemoryStatsFlat,
  CudaMemorySegmentSnapshot,
} from './internal/types.js';

/**
 * Returns true when CUDA is available in this process.
 *
 * Mirrors vibetensor.torch.cuda.is_available() semantics.
 */
export function isAvailable(): boolean {
  return core.cudaIsAvailable();
}

/**
 * Returns the number of CUDA devices visible to VibeTensor.
 */
export function deviceCount(): number {
  return core.cudaDeviceCount();
}

/**
 * Returns the current CUDA device for this JS thread.
 *
 * Control-plane convenience only; cache within hot loops if needed.
 */
export function currentDevice(): number {
  return core.cudaCurrentDevice();
}

/**
 * Sets the current CUDA device for this JS thread.
 *
 * May perform one-time context initialization and is **not** suitable for
 * tight inner loops.
 */
export function setDevice(idx: number): void {
  core.cudaSetDevice(idx);
}

export class Stream {
  constructor(private readonly handle: CudaStreamHandle) {}

  static create(deviceIndex: number = 0): Stream {
    return new Stream(core.createCudaStream(deviceIndex));
  }

  get deviceIndex(): number {
    return core.cudaStreamDeviceIndex(this.handle);
  }

  /** Non-blocking; true when no pending work on the stream. */
  query(): boolean {
    return core.cudaStreamQuery(this.handle);
  }

  /** Human-readable description with no pointers or paths. */
  toString(): string {
    return core.cudaStreamToString(this.handle);
  }

  /**
   * Asynchronously waits for all work on this stream to complete.
   *
   * Implemented via napi_async_work; never blocks the event loop.
   */
  async synchronize(): Promise<void> {
    return core.cudaStreamSynchronizeAsync(this.handle);
  }
}

export class Event {
  constructor(private readonly handle: CudaEventHandle) {}

  static create(opts?: { enableTiming?: boolean }): Event {
    return new Event(core.createCudaEvent(opts));
  }

  isCreated(): boolean {
    return core.cudaEventIsCreated(this.handle);
  }

  /** Non-blocking; true if the event has completed or was never recorded. */
  query(): boolean {
    return core.cudaEventQuery(this.handle);
  }

  toString(): string {
    return core.cudaEventToString(this.handle);
  }

  /**
   * Asynchronously waits for this event to complete.
   *
   * Implemented via napi_async_work; never blocks the event loop.
   */
  async synchronize(): Promise<void> {
    return core.cudaEventSynchronizeAsync(this.handle);
  }
}


export function externalMemoryStats(): ExternalMemoryStats {
  return core.externalMemoryStatsRaw();
}

export function memoryStatsAsNested(device?: number): CudaMemoryStatsNested {
  return core.cudaMemoryStatsAsNested(device);
}

export function memoryStats(device?: number): CudaMemoryStatsFlat {
  return core.cudaMemoryStats(device);
}

export function memorySnapshot(device?: number): CudaMemorySegmentSnapshot[] {
  return core.cudaMemorySnapshot(device);
}

export async function h2d(
  src: ArrayBufferView,
  sizes: readonly number[],
  opts: H2DOptions,
): Promise<Tensor> {
  const handle = await core.cudaH2DAsync(src, sizes, opts);
  return new Tensor(handle);
}

export async function d2h(t: Tensor): Promise<ArrayBufferView> {
  return core.cudaD2HAsync(t.__vbt_handle);
}
