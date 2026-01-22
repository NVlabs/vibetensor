// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as core from './core.js';
import type { FabricEventSnapshot, FabricStatsSnapshot } from './internal/types.js';

/** Snapshot the process-global Fabric stats counters (best-effort). */
export function stats(): FabricStatsSnapshot {
  return core.getFabricStats();
}

/**
 * Snapshot the process-global Fabric event ring (poll-only).
 *
 * Enable recording by setting `VBT_FABRIC_EVENTS_MODE=basic` before importing vibetensor.
 */
export function events(
  minSeq: bigint = 0n,
  maxEvents: number = 1024,
): FabricEventSnapshot {
  return core.getFabricEvents(minSeq, maxEvents);
}
