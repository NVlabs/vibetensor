// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as core from './core.js';

export class Tensor {
  /** Opaque handle into the native addon. */
  readonly __vbt_handle: core.TensorHandle;

  constructor(handle: core.TensorHandle) {
    if (!handle) {
      throw new TypeError(
        'Tensor constructor is internal-only; use vibetensor.zeros() to create tensors',
      );
    }
    this.__vbt_handle = handle;
  }

  get sizes(): readonly number[] {
    const handle = this.__vbt_handle;
    if (typeof handle.sizes !== 'function') {
      throw new Error('Tensor handle missing sizes() method');
    }
    return handle.sizes();
  }

  get dtype(): 'float32' {
    const handle = this.__vbt_handle;
    if (typeof handle.dtype !== 'function') {
      throw new Error('Tensor handle missing dtype() method');
    }
    return handle.dtype() as 'float32';
  }

  get device(): core.DeviceString {
    const handle = this.__vbt_handle;
    if (typeof handle.device !== 'function') {
      throw new Error('Tensor handle missing device() method');
    }
    const raw = String(handle.device());
    if (raw === 'cpu' || raw === 'cpu:0') return 'cpu';
    if (raw === 'cuda') return 'cuda:0';
    if (raw.startsWith('cuda:')) {
      const idx = Number(raw.slice('cuda:'.length));
      if (Number.isInteger(idx) && idx >= 0) return `cuda:${idx}`;
    }
    throw new Error(`Unexpected device string from addon: ${raw}`);
  }
}

export interface TensorOptions {
  device?: core.DeviceString;
  dtype?: 'float32';
}

export async function zeros(
  sizes: readonly number[],
  opts?: TensorOptions,
): Promise<Tensor> {
  const handle = core.zeros(sizes, opts ?? {});
  return new Tensor(handle);
}

export async function scalarInt64(value: number): Promise<Tensor> {
  return new Tensor(core.scalarInt64(value));
}

export async function scalarBool(value: boolean): Promise<Tensor> {
  return new Tensor(core.scalarBool(value));
}

export async function scalarFloat32(value: number): Promise<Tensor> {
  return new Tensor(core.scalarFloat32(value));
}
