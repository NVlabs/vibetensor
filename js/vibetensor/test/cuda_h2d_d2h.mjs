// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';

import { Tensor, zeros, cuda } from '../dist/index.js';
import { addon } from '../dist/addon.js';

const hasCuda = cuda.isAvailable();
const hasHelpers =
  typeof (addon)._cudaH2DAsync === 'function' &&
  typeof (addon)._cudaD2HAsync === 'function';

// Happy-path tests only run when CUDA is available and the native helpers
// are present. ENOCUDA behavior is covered in separate tests that run when
// cuda.isAvailable() is false.
const skipCuda = !hasCuda || !hasHelpers;
const skipNoCuda = hasCuda;

// Helper to read the raw device string from the internal tensor handle
// without going through Tensor.device(), which is currently CPU-only on
// the JS surface.
function getRawDeviceString(t) {
  const handle = /** @type {any} */ (t).__vbt_handle;
  if (!handle || typeof handle.device !== 'function') {
    throw new Error('Tensor handle missing device() method');
  }
  return String(handle.device());
}

// ---- Happy-path H2D/D2H round-trips ---------------------------------------

test('cuda.h2d/d2h float32 round-trip', { concurrency: false, skip: skipCuda }, async () => {
  const src = new Float32Array([0.5, -1.25, 3.0, 4.5]);
  const t = await cuda.h2d(src, [2, 2], { dtype: 'float32' });
  assert.ok(t instanceof Tensor);

  const dst = await cuda.d2h(t);
  assert.ok(dst instanceof Float32Array);
  assert.deepEqual(Array.from(dst), Array.from(src));
});


test('cuda.h2d/d2h int32 round-trip', { concurrency: false, skip: skipCuda }, async () => {
  const src = new Int32Array([1, -2, 3, -4]);
  const t = await cuda.h2d(src, [2, 2], { dtype: 'int32' });
  assert.ok(t instanceof Tensor);

  const dst = await cuda.d2h(t);
  assert.ok(dst instanceof Int32Array);
  assert.deepEqual(Array.from(dst), Array.from(src));
});


test('cuda.h2d/d2h int64 round-trip', { concurrency: false, skip: skipCuda }, async () => {
  const src = new BigInt64Array([1n, -2n, 3n, -4n]);
  const t = await cuda.h2d(src, [4], { dtype: 'int64' });
  assert.ok(t instanceof Tensor);

  const dst = await cuda.d2h(t);
  assert.ok(dst instanceof BigInt64Array);
  assert.deepEqual(Array.from(dst), Array.from(src));
});


test('cuda.h2d/d2h bool round-trip', { concurrency: false, skip: skipCuda }, async () => {
  const src = new Uint8Array([0, 1, 1, 0]);
  const t = await cuda.h2d(src, [2, 2], { dtype: 'bool' });
  assert.ok(t instanceof Tensor);

  const dst = await cuda.d2h(t);
  assert.ok(dst instanceof Uint8Array);
  assert.deepEqual(Array.from(dst), Array.from(src));
});


// ---- Device resolution semantics ------------------------------------------

test('cuda.h2d uses current device when device option is omitted', { concurrency: false, skip: skipCuda }, async () => {
  const count = cuda.deviceCount();
  assert.ok(count > 0);

  // Ensure a known current device.
  cuda.setDevice(0);
  const current = cuda.currentDevice();

  const src = new Float32Array([1, 2, 3, 4]);
  const t = await cuda.h2d(src, [4], { dtype: 'float32' });
  const devString = getRawDeviceString(t);
  assert.equal(devString, `cuda:${current}`);
});


// ---- Error cases for D2H --------------------------------------------------

test('cuda.d2h rejects CPU tensor with TypeError(EINVAL)', { concurrency: false, skip: skipCuda }, async () => {
  const cpu = await zeros([2, 2]);
  await assert.rejects(
    cuda.d2h(cpu),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      assert.match(String(e.message), /CUDA tensor/i);
      return true;
    },
  );
});


// ---- ENOCUDA behavior when CUDA is unavailable ----------------------------

test('cuda.h2d rejects with Error(ENOCUDA) when CUDA is unavailable', { concurrency: false, skip: skipNoCuda }, async () => {
  const src = new Float32Array([1, 2, 3, 4]);
  await assert.rejects(
    cuda.h2d(src, [4], { dtype: 'float32' }),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );
});


test('cuda.d2h rejects with Error(ENOCUDA) when CUDA is unavailable', { concurrency: false, skip: skipNoCuda }, async () => {
  const cpu = await zeros([1]);
  await assert.rejects(
    cuda.d2h(cpu),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );
});
