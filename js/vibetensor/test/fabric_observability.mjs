// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';

// Enable BASIC mode before Fabric initializes so the event ring contains
// a deterministic events_mode_changed entry.
process.env.VBT_FABRIC_EVENTS_MODE = 'basic';

const { fabric } = await import('../dist/index.js');

function assertBigInt(value, msg) {
  assert.equal(typeof value, 'bigint', msg);
}

test('Fabric stats snapshot basic shape (BigInt counters)', { concurrency: false }, () => {
  const s = fabric.stats();

  for (const k of [
    'mesh_builds',
    'p2p_pairs_enabled',
    'p2p_pairs_failed',
    'fabric_ops_attempted',
    'fabric_ops_hit',
    'fabric_ops_fallback',
    'remote_bytes_read',
    'remote_bytes_written',
    'inflight_ops_current',
    'inflight_ops_peak',
    'event_queue_len_peak',
    'event_dropped_total',
    'event_failures_total',
    'mode_enable_calls',
    'mode_disable_calls',
    'mode_set_failures',
  ]) {
    assertBigInt(s[k], `stats.${k}`);
    assert.ok(s[k] >= 0n, `stats.${k} must be >= 0`);
  }

  assert.ok(s.reasons && typeof s.reasons === 'object');
  for (const k of ['no_p2p', 'requires_grad', 'in_backward', 'small_tensor']) {
    assertBigInt(s.reasons[k], `reasons.${k}`);
    assert.ok(s.reasons[k] >= 0n, `reasons.${k} must be >= 0`);
  }

  assert.ok(Array.isArray(s.per_device), 'per_device must be an array');
  for (const d of s.per_device) {
    assert.equal(typeof d.device_index, 'number', 'device_index');
    assertBigInt(d.ops_as_primary, 'ops_as_primary');
    assertBigInt(d.ops_as_remote, 'ops_as_remote');
    assertBigInt(d.remote_bytes_read, 'remote_bytes_read');
    assertBigInt(d.remote_bytes_written, 'remote_bytes_written');
  }
});

test('Fabric events snapshot basic shape (poll-only; no waits)', { concurrency: false }, () => {
  // Ensure Fabric init has happened in this process.
  fabric.stats();

  const snap = fabric.events(0n, 128);
  assertBigInt(snap.base_seq, 'base_seq');
  assertBigInt(snap.next_seq, 'next_seq');
  assertBigInt(snap.dropped_total, 'dropped_total');
  assert.equal(typeof snap.capacity, 'number', 'capacity');
  assert.ok(snap.capacity >= 0);

  assert.ok(Array.isArray(snap.events), 'events must be an array');

  for (const e of snap.events) {
    assertBigInt(e.seq, 'event.seq');
    assertBigInt(e.t_ns, 'event.t_ns');
    assert.equal(typeof e.primary_device, 'number', 'event.primary_device');
    assert.equal(typeof e.other_device, 'number', 'event.other_device');
    assert.equal(typeof e.kind, 'string', 'event.kind');
    assert.equal(typeof e.level, 'string', 'event.level');
    assertBigInt(e.op_id, 'event.op_id');
    assertBigInt(e.numel, 'event.numel');
    assertBigInt(e.bytes, 'event.bytes');
    assert.equal(typeof e.reason_raw, 'number', 'event.reason_raw');
    assert.ok(e.message === null || typeof e.message === 'string', 'event.message');
  }

  // When BASIC mode is enabled before init, we expect to see at least one
  // events_mode_changed marker.
  assert.ok(
    snap.events.some((e) => e.kind === 'events_mode_changed'),
    'expected an events_mode_changed event',
  );
});
