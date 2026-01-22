// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Tensor } from './tensor.js';
import * as core from './core.js';
import type { DlpackCapsule, FromDlpackOptions } from './internal/types.js';

export type DlpackProvider = {
  __dlpack__: (stream?: unknown) => DlpackCapsule | unknown;
  __dlpack_device__?: () => [number, number];
};

export type DlpackInput = DlpackCapsule | DlpackProvider;

export function toDlpack(t: Tensor): DlpackCapsule {
  return core.toDlpackCapsule(t.__vbt_handle);
}

export async function fromDlpack(
  x: DlpackInput,
  opts: FromDlpackOptions = {},
): Promise<Tensor> {
  const capsule = await toCapsule(x);
  const handle = await core.fromDlpackCapsule(capsule, opts);
  return new Tensor(handle);
}

async function toCapsule(x: DlpackInput): Promise<DlpackCapsule> {
  // Provider path has priority, mirroring Python.
  const maybeProvider = x as any;
  if (maybeProvider && typeof maybeProvider.__dlpack__ === 'function') {
    const devFn = maybeProvider.__dlpack_device__;
    if (typeof devFn === 'function') {
      try {
        const [devType, devId] = devFn();
        if (devType !== 1 /* kDLCPU */ || devId !== 0) {
          const err: any = new Error(
            'dlpack.fromDlpack: unsupported provider device type',
          );
          err.code = 'ERUNTIME';
          throw err;
        }
      } catch (e: any) {
        const err: any = new Error(
          'dlpack.fromDlpack: provider error: ' + (e?.message ?? String(e)),
        );
        err.code = 'ERUNTIME';
        throw err;
      }
    }

    try {
      const cap = maybeProvider.__dlpack__(undefined);
      // We only accept capsules produced by our addon; if a foreign provider
      // returns something else, UnwrapDlpackCapsule in C++ will reject it.
      return cap as DlpackCapsule;
    } catch (e: any) {
      const err: any = new Error(
        'dlpack.fromDlpack: provider error: ' + (e?.message ?? String(e)),
      );
      err.code = 'ERUNTIME';
      throw err;
    }
  }

  // Capsule path.
  return x as DlpackCapsule;
}
