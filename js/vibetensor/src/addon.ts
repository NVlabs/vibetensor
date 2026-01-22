// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from 'module';
import type { FabricEventSnapshot, FabricStatsSnapshot } from './internal/types.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);

export interface CallOptions {
  /** Reserved for tracing/metrics only; MUST NOT affect behavior. */
  fastPath?: boolean;
}

export interface NativeTensorHandle {
  sizes(): readonly number[];
  dtype(): string;
  device(): string;
}

export interface NativeCudaStream {
  deviceIndex(): number;
  query(): boolean;
  toString(): string;
  synchronize(): Promise<void>;
}

export interface NativeCudaEvent {
  isCreated(): boolean;
  query(): boolean;
  toString(): string;
  synchronize(): Promise<void>;
}

export interface NativeBindings {
  hasCuda(): boolean;
  cudaDeviceCount(): number;
  cudaCurrentDevice(): number;
  cudaSetDevice(idx: number): void;

  zeros(
    sizes: number[],
    opts?: { dtype?: string; device?: string },
  ): NativeTensorHandle;

  scalarInt64(value: number): NativeTensorHandle;
  scalarBool(value: boolean): NativeTensorHandle;
  scalarFloat32(value: number): NativeTensorHandle;

  // dlpack bindings can still function; JS layer checks for presence.
  toDlpack?(handle: NativeTensorHandle): object;
  fromDlpackCapsule?(
    cap: object,
    opts?: { copy?: boolean; device?: number },
  ): Promise<NativeTensorHandle>;

  _cudaH2DAsync?(
    src: ArrayBufferView,
    sizes: number[],
    opts: { device?: number; dtype: string },
  ): Promise<NativeTensorHandle>;

  _cudaD2HAsync?(handle: NativeTensorHandle): Promise<ArrayBufferView>;

  callOpSync(name: string, args: any[], opts?: CallOptions): any;
  callOp(name: string, args: any[], opts?: CallOptions): Promise<any>;
  callOpNoOverride(
    name: string,
    args: any[],
    opts?: CallOptions,
  ): Promise<any>;

  // without Fabric hooks can still load; core.ts checks for presence.
  fabricStatsSnapshot?(): FabricStatsSnapshot;
  fabricEventsSnapshot?(minSeq: bigint, maxEvents: number): FabricEventSnapshot;

  _debugDispatcherStats(): {
    totalCalls: number;
    failedCalls: number;
    inflight: number;
    peakInflight: number;
    maxInflight: number;
  };

  // CUDA runtime classes for streams/events (internal/advanced surface).
  CudaStream?: {
    new (deviceIndex?: number): NativeCudaStream;
  };
  CudaEvent?: {
    new (opts?: { enableTiming?: boolean }): NativeCudaEvent;
  };

  _dummyAdd?(a: number, b: number): number;
}

function resolveAddonPath(): string {
  const fromEnv = process.env.VBT_NODE_ADDON_PATH;
  if (fromEnv && fromEnv.length > 0) return fromEnv;

  const here = path.dirname(fileURLToPath(import.meta.url));
  const candidate = path.join(here, '..', 'vbt_napi.node');
  if (fs.existsSync(candidate)) return candidate;

  throw new Error(
    'vibetensor: could not locate vbt_napi.node. ' +
      'Set VBT_NODE_ADDON_PATH or build with VBT_BUILD_NODE=ON.',
  );
}

// eslint-disable-next-line @typescript-eslint/no-var-requires
export const addon: NativeBindings = require(resolveAddonPath()) as NativeBindings;
