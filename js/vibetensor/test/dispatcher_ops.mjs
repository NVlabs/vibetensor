// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';

import { Tensor, zeros, ops, cuda } from '../dist/index.js';
import * as core from '../dist/core.js';
import { addon } from '../dist/addon.js';

// Configure dispatcher env for this process before any callOp* usage.
// Small inflight cap keeps tests simple while exercising the limit path.
if (!process.env.VBT_NODE_MAX_INFLIGHT_OPS) {
  process.env.VBT_NODE_MAX_INFLIGHT_OPS = '1';
}
delete process.env.VBT_NODE_ALLOW_SYNC_DANGER; // keep callOpSync gated in these tests

// Helper: unwrap the internal handle from a Tensor.
function handleOf(t) {
  return t.__vbt_handle;
}

// ---------------------------------------------------------------------------
// 6.1 Happy-path tests – ops.vt.*
// ---------------------------------------------------------------------------

test('ops.vt.add/mul/relu basic metadata', { concurrency: false }, async () => {
  const a = await zeros([2, 3]);
  const b = await zeros([2, 3]);

  const cAdd = await ops.vt.add(a, b);
  assert.ok(cAdd instanceof Tensor);
  assert.deepEqual(cAdd.sizes, [2, 3]);
  assert.equal(cAdd.dtype, 'float32');
  assert.equal(cAdd.device, 'cpu');

  const cMul = await ops.vt.mul(a, b);
  assert.ok(cMul instanceof Tensor);
  assert.deepEqual(cMul.sizes, [2, 3]);
  assert.equal(cMul.dtype, 'float32');
  assert.equal(cMul.device, 'cpu');

  const cRelu = await ops.vt.relu(a);
  assert.ok(cRelu instanceof Tensor);
  assert.deepEqual(cRelu.sizes, [2, 3]);
  assert.equal(cRelu.dtype, 'float32');
  assert.equal(cRelu.device, 'cpu');
});

// ---------------------------------------------------------------------------
// 6.2 Error-path tests
// ---------------------------------------------------------------------------

test('core.callOp invalid tensor args throw TypeError(EINVAL)', { concurrency: false }, () => {
  assert.throws(
    () => {
      // Pass plain objects instead of Tensor handles; this should fail
      // synchronously during input validation, before any Promise is created.
      core.callOp('vt::add', [{}, {}]);
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      assert.match(String(e.message), /Tensor handle/i);
      return true;
    },
  );
});


test('ops.vt.add rejects invalid arguments with TypeError(EINVAL)', { concurrency: false }, async () => {
  const a = await zeros([1]);

  await assert.rejects(
    ops.vt.add(a, {}),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      assert.match(String(e.message), /Tensor handle/i);
      return true;
    },
  );
});


test('core.callOp unknown op rejects with Error(ERUNTIME)', { concurrency: false }, async () => {
  const t = await zeros([1]);

  await assert.rejects(
    core.callOp('vt::does_not_exist', [handleOf(t)]),
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ERUNTIME');
      assert.match(String(e.message), /unknown/i);
      assert.match(String(e.message), /vt::does_not_exist/);
      return true;
    },
  );
});


test('callOpSync is gated by VBT_NODE_ALLOW_SYNC_DANGER', { concurrency: false }, () => {
  // With VBT_NODE_ALLOW_SYNC_DANGER unset/false, callOpSync must throw
  // synchronously without touching the dispatcher.
  assert.throws(
    () => {
      core.callOpSync('vt::add', []);
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ERUNTIME');
      assert.match(String(e.message), /callOpSync is disabled/i);
      return true;
    },
  );
});


test('inflight cap exhaustion produces ERUNTIME error and sane stats', { concurrency: false }, async () => {
  const a = await zeros([1024, 1024]);
  const b = await zeros([1024, 1024]);

  // Kick off one long-ish op and intentionally do not await it yet.
  const p1 = core.callOp('vt::add', [handleOf(a), handleOf(b)]);

  // Second call should fail synchronously due to VBT_NODE_MAX_INFLIGHT_OPS=1.
  assert.throws(
    () => {
      core.callOp('vt::add', [handleOf(a), handleOf(b)]);
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ERUNTIME');
      assert.match(String(e.message), /too many inflight ops/i);
      return true;
    },
  );

  await p1;

  const stats = core._debugDispatcherStats();
  assert.ok(stats.inflight <= stats.maxInflight);
  assert.ok(stats.peakInflight >= stats.inflight);
  assert.equal(stats.maxInflight, Number(process.env.VBT_NODE_MAX_INFLIGHT_OPS));
});

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

test('cuda overlay mirrors core hasCuda/deviceCount', { concurrency: false }, () => {
  const has = core.hasCuda();
  const count = core.cudaDeviceCount();
  assert.equal(typeof has, 'boolean');
  assert.equal(typeof count, 'number');
  assert.ok(count >= 0);
  assert.equal(has, count > 0);

  assert.equal(cuda.isAvailable(), has);
  assert.equal(cuda.deviceCount(), count);
});


test('cuda.currentDevice and setDevice happy-path when CUDA is available', {
  concurrency: false,
  skip: !cuda.isAvailable(),
}, () => {
  const count = cuda.deviceCount();
  assert.ok(count >= 1);

  const cur = cuda.currentDevice();
  assert.ok(Number.isInteger(cur));
  assert.ok(cur >= 0 && cur < count);

  // setDevice to the current device is a no-op.
  cuda.setDevice(cur);

  // Out-of-range index yields TypeError(EINVAL).
  assert.throws(
    () => {
      cuda.setDevice(count);
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.equal(e.code, 'EINVAL');
      return true;
    },
  );
});


test('cuda Stream and Event basic semantics', {
  concurrency: false,
  skip: !cuda.isAvailable(),
}, async () => {
  const stream = cuda.Stream.create(0);
  assert.equal(stream.deviceIndex, 0);
  assert.equal(typeof stream.query(), 'boolean');
  const desc = stream.toString();
  assert.equal(typeof desc, 'string');
  assert.match(desc, /Stream/i);

  await stream.synchronize();

  const ev = cuda.Event.create();
  assert.equal(ev.isCreated(), false);
  assert.equal(ev.query(), true);
  assert.equal(typeof ev.toString(), 'string');
  await ev.synchronize();
});


test('cuda ENOCUDA semantics when CUDA is unavailable', {
  concurrency: false,
  skip: cuda.isAvailable(),
}, () => {
  assert.equal(cuda.deviceCount(), 0);

  assert.throws(
    () => {
      cuda.currentDevice();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );

  assert.throws(
    () => {
      cuda.setDevice(0);
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );

  assert.throws(
    () => {
      cuda.Stream.create(0);
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );

  const ev = cuda.Event.create();
  assert.throws(
    () => {
      ev.query();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.code, 'ENOCUDA');
      return true;
    },
  );
});

test('forged cuda Stream/Event handles throw TypeError(EINVAL)', {
  concurrency: false,
}, () => {
  if (!addon.CudaStream || !addon.CudaEvent) {
    // If the native classes are unavailable (e.g., addon build disabled),
    // treat this as a no-op; other tests already validate that setup.
    return;
  }

  const { CudaStream, CudaEvent } = addon;

  const fakeStream = Object.create(CudaStream.prototype);
  assert.throws(
    () => {
      fakeStream.deviceIndex();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.match(String(e.message), /Illegal invocation/i);
      return true;
    },
  );
  assert.throws(
    () => {
      fakeStream.query();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.match(String(e.message), /Illegal invocation/i);
      return true;
    },
  );
  assert.throws(
    () => {
      fakeStream.synchronize();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.match(String(e.message), /Illegal invocation/i);
      return true;
    },
  );

  const fakeEvent = Object.create(CudaEvent.prototype);
  assert.throws(
    () => {
      fakeEvent.isCreated();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.match(String(e.message), /Illegal invocation/i);
      return true;
    },
  );
  assert.throws(
    () => {
      fakeEvent.query();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.match(String(e.message), /Illegal invocation/i);
      return true;
    },
  );
  assert.throws(
    () => {
      fakeEvent.synchronize();
    },
    (err) => {
      const e = /** @type {any} */ (err);
      assert.equal(e.name, 'TypeError');
      assert.match(String(e.message), /Illegal invocation/i);
      return true;
    },
  );
});

const workerScriptUrl = new URL('./worker_cuda_smoke.mjs', import.meta.url);

test('cuda.isAvailable/deviceCount work in worker_threads', {
  concurrency: false,
}, async () => {
  const mainHas = cuda.isAvailable();
  const mainCount = cuda.deviceCount();

  await new Promise((resolve, reject) => {
    const worker = new Worker(workerScriptUrl);

    worker.once('message', (msg) => {
      try {
        const e = /** @type {any} */ (msg);
        assert.equal(typeof e.available, 'boolean');
        assert.equal(typeof e.count, 'number');
        assert.ok(e.count >= 0);
        assert.equal(e.available, e.count > 0);

        // Main thread and worker should agree on CUDA availability and count.
        assert.equal(e.available, mainHas);
        assert.equal(e.count, mainCount);

        resolve();
      } catch (err) {
        reject(err);
      }
    });

    worker.once('error', reject);
  });
});
