// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';

import { Tensor, zeros, hasCuda, cudaDeviceCount } from '../dist/index.js';
import { addon } from '../dist/addon.js';

test('cuda metadata basics', () => {
  const has = hasCuda();
  const count = cudaDeviceCount();
  assert.equal(typeof has, 'boolean');
  assert.equal(typeof count, 'number');
  assert.ok(count >= 0);
  assert.equal(has, count > 0);
});

test('zeros factory and Tensor metadata', async () => {
  const t = await zeros([2, 3]);
  assert.ok(t instanceof Tensor);
  assert.deepEqual(t.sizes, [2, 3]);
  assert.equal(t.dtype, 'float32');
  assert.equal(t.device, 'cpu');
});

test('zeros validation errors', async () => {
  await assert.rejects(async () => zeros([-1]), /sizes/i);
  await assert.rejects(async () => zeros([1], { dtype: 'int32' }), /dtype/i);
  await assert.rejects(async () => zeros([1], { device: 'cuda:0' }), /cpu/i);
});

test('_dummyAdd basic behavior', () => {
  const sum = addon._dummyAdd(2, 3);
  assert.equal(sum, 5);
  assert.throws(() => addon._dummyAdd('a', 3), { name: 'TypeError' });
});
