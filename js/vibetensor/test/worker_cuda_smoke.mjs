// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parentPort } from 'node:worker_threads';

import { cuda } from '../dist/index.js';

const available = cuda.isAvailable();
const count = cuda.deviceCount();

if (parentPort) {
  parentPort.postMessage({ available, count });
}
