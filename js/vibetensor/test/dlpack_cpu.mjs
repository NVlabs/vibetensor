// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';

import { Tensor, zeros, dlpack, hasCuda } from '../dist/index.js';
import * as core from '../dist/core.js';
import { addon } from '../dist/addon.js';

const hasNativeDlpack =
  typeof addon.toDlpack === 'function' &&
  typeof addon.fromDlpackCapsule === 'function';
const cudaAvailable = hasCuda();

// ---------------------------------------------------------------------------
// 9.1 CPU round-trip & aliasing (metadata-only checks here)
// ---------------------------------------------------------------------------

test('dlpack CPU round-trip basic metadata', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const t = await zeros([2, 3]);
  const cap = dlpack.toDlpack(t);
  assert.ok(cap);

  const u = await dlpack.fromDlpack(cap);
  assert.ok(u instanceof Tensor);
  assert.deepEqual(u.sizes, t.sizes);
  assert.equal(u.dtype, t.dtype);
  assert.equal(u.device, t.device);
});


test('dlpack zero-size CPU tensor round-trip', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const t = await zeros([0, 3]);
  const cap = dlpack.toDlpack(t);
  const u = await dlpack.fromDlpack(cap);
  assert.ok(u instanceof Tensor);
  assert.deepEqual(u.sizes, t.sizes);
  assert.equal(u.dtype, t.dtype);
  assert.equal(u.device, t.device);
});

// ---------------------------------------------------------------------------
// 9.3 Capsule reuse & foreign capsules
// ---------------------------------------------------------------------------

test('dlpack capsule reuse is rejected with Error(ERUNTIME)', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const t = await zeros([2, 2]);
  const cap = dlpack.toDlpack(t);

  const first = await dlpack.fromDlpack(cap);
  assert.ok(first instanceof Tensor);

  await assert.rejects(
    dlpack.fromDlpack(cap),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ERUNTIME');
      assert.match(String(e.message), /capsule already consumed/i);
      return true;
    },
  );
});


test('dlpack.fromDlpack rejects non-capsule objects with TypeError(EINVAL)', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  await assert.rejects(
    dlpack.fromDlpack({}),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      assert.match(String(e.message), /DlpackCapsule/i);
      return true;
    },
  );
});


test('dlpack.fromDlpack rejects providers that return foreign capsules', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const provider = {
    __dlpack__: () => ({ not: 'a real capsule' }),
    __dlpack_device__: () => [1, 0],
  };

  await assert.rejects(
    dlpack.fromDlpack(provider),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      assert.match(String(e.message), /DlpackCapsule/i);
      return true;
    },
  );
});

// ---------------------------------------------------------------------------
// 9.4 Provider protocol (CPU-only)
// ---------------------------------------------------------------------------

test('dlpack.fromDlpack accepts CPU providers', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const base = await zeros([1, 4]);

  const provider = {
    __dlpack__: () => dlpack.toDlpack(base),
    __dlpack_device__: () => [1, 0],
  };

  const out = await dlpack.fromDlpack(provider);
  assert.ok(out instanceof Tensor);
  assert.deepEqual(out.sizes, base.sizes);
  assert.equal(out.dtype, base.dtype);
  assert.equal(out.device, base.device);
});


test('dlpack.fromDlpack rejects non-CPU providers with Error(ERUNTIME)', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const provider = {
    __dlpack__: () => ({ bogus: true }),
    __dlpack_device__: () => [2, 0], // kDLCUDA
  };

  await assert.rejects(
    dlpack.fromDlpack(provider),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ERUNTIME');
      assert.match(
        String(e.message),
        /unsupported provider device type/i,
      );
      return true;
    },
  );
});

// ---------------------------------------------------------------------------
// 9.6 Device-option surface
// ---------------------------------------------------------------------------

test('dlpack.fromDlpack accepts device:0 in options', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const t = await zeros([1]);
  const cap = dlpack.toDlpack(t);
  const u = await dlpack.fromDlpack(cap, { device: 0 });
  assert.ok(u instanceof Tensor);
  assert.deepEqual(u.sizes, t.sizes);
});


test('dlpack.fromDlpack rejects non-zero device with Error(ENOCUDA)', { concurrency: false, skip: !hasNativeDlpack || cudaAvailable }, async () => {
  const t = await zeros([1]);
  const cap = dlpack.toDlpack(t);

  await assert.rejects(
    dlpack.fromDlpack(cap, { device: 1 }),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );
});


test('dlpack.fromDlpack rejects non-integer device with TypeError(EINVAL)', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const t = await zeros([1]);
  const cap = dlpack.toDlpack(t);

  await assert.rejects(
    dlpack.fromDlpack(cap, { device: 0.5 }),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      return true;
    },
  );
});


test('dlpack.fromDlpack rejects non-numeric device with TypeError(EINVAL)', { concurrency: false, skip: !hasNativeDlpack }, async () => {
  const t = await zeros([1]);
  const cap = dlpack.toDlpack(t);

  await assert.rejects(
    // @ts-expect-error deliberate misuse in JS test
    dlpack.fromDlpack(cap, { device: 'cuda:0' }),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      return true;
    },
  );
});
