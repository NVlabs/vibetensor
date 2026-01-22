// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';

import { Tensor, zeros, cuda, ops } from '../dist/index.js';
import * as core from '../dist/core.js';
import * as logging from '../dist/internal/logging.js';

// Basic helper to assert non-negative finite numbers.
function assertNonNegativeNumber(value, msg) {
  assert.equal(typeof value, 'number', msg);
  assert.ok(Number.isFinite(value), msg);
  assert.ok(value >= 0, msg);
}

test('externalMemoryStats basic shape', { concurrency: false }, () => {
  const stats = cuda.externalMemoryStats();
  assertNonNegativeNumber(stats.ownersAlive, 'ownersAlive');
  assertNonNegativeNumber(stats.bytesAccounted, 'bytesAccounted');
});

test('cuda.memoryStats* and memorySnapshot basic structure', { concurrency: false }, () => {
  const nested = cuda.memoryStatsAsNested();
  const flat = cuda.memoryStats();
  const snaps = cuda.memorySnapshot();

  const families = ['allocated_bytes', 'reserved_bytes', 'requested_bytes'];
  for (const name of families) {
    const fam = nested[name];
    assert.ok(fam && typeof fam === 'object', `missing family ${name}`);
    assert.ok(fam.all && typeof fam.all === 'object', `missing all for ${name}`);
    for (const key of ['current', 'peak', 'allocated', 'freed']) {
      assertNonNegativeNumber(fam.all[key], `${name}.all.${key}`);
    }
  }

  assert.ok(nested.device_stats && typeof nested.device_stats === 'object');
  const agg = nested.device_stats.aggregated;
  if (agg && typeof agg === 'object') {
    for (const v of Object.values(agg)) {
      assertNonNegativeNumber(v, 'device_stats.aggregated value');
    }
  }

  assert.ok(flat && typeof flat === 'object');
  for (const v of Object.values(flat)) {
    assertNonNegativeNumber(v, 'flat stats value');
  }

  assert.ok(Array.isArray(snaps));
  for (const seg of snaps) {
    assert.equal(typeof seg.device, 'number');
    assert.equal(typeof seg.poolId, 'number');
    assert.equal(typeof seg.bytesReserved, 'number');
    assert.equal(typeof seg.bytesActive, 'number');
    assert.equal(typeof seg.blocks, 'number');
    assert.ok(seg.bytesReserved >= 0);
    assert.ok(seg.bytesActive >= 0);
    assert.ok(seg.blocks >= 0);
  }

  // CPU-only builds should synthesize zeros/empty structures.
  if (!cuda.isAvailable()) {
    assert.equal(snaps.length, 0);
    for (const v of Object.values(flat)) {
      assert.equal(v, 0);
    }
  }
});

test('native logging produces structured entries when enabled', { concurrency: false }, async () => {
  // Ensure deterministic start state.
  core.setNativeLoggingEnabled(false, 'info');
  core.drainNativeLogs(1024);

  core.setNativeLoggingEnabled(true, 'debug', ['dispatcher']);

  const a = await zeros([2, 2]);
  const b = await zeros([2, 2]);
  const c = await ops.vt.add(a, b);
  assert.ok(c instanceof Tensor);

  const entries = core.drainNativeLogs(1024);
  core.setNativeLoggingEnabled(false, 'info');

  assert.ok(Array.isArray(entries));

  for (const e of entries) {
    if (!e) continue;
    assertNonNegativeNumber(e.ts, 'log.ts');
    assert.equal(typeof e.level, 'string');
    assert.equal(typeof e.category, 'string');
    assert.equal(typeof e.message, 'string');
    if (e.data !== undefined) {
      assert.ok(e.data && typeof e.data === 'object');
    }
  }
});

test('async metrics surface returns non-negative counters', { concurrency: false }, () => {
  const stats = logging.getAsyncMetrics();

  assertNonNegativeNumber(stats.inflight, 'inflight');
  assertNonNegativeNumber(stats.peakInflight, 'peakInflight');
  assertNonNegativeNumber(stats.maxInflight, 'maxInflight');
  assert.ok(stats.maxInflight >= stats.peakInflight);
  assert.ok(stats.peakInflight >= stats.inflight);

  assertNonNegativeNumber(stats.dispatcher.totalCalls, 'dispatcher.totalCalls');
  assertNonNegativeNumber(stats.dispatcher.failedCalls, 'dispatcher.failedCalls');
  assert.ok(stats.dispatcher.totalCalls >= stats.dispatcher.failedCalls);

  for (const [name, cat] of [['h2d', stats.h2d], ['d2h', stats.d2h]]) {
    assert.ok(cat && typeof cat === 'object', `missing ${name} stats`);
    assertNonNegativeNumber(cat.started, `${name}.started`);
    assertNonNegativeNumber(cat.completed, `${name}.completed`);
    assertNonNegativeNumber(cat.failed, `${name}.failed`);
    assertNonNegativeNumber(cat.bytes, `${name}.bytes`);
  }
});
