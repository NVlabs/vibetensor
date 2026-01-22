// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as core from '../core.js';
import type {
  LogConfig,
  LogEntry,
  LogSink,
  LogLevel,
  LogCategory,
  DebugAsyncStats,
} from './types.js';

let currentSink: LogSink | null = null;
let pumpScheduled = false;
let currentConfig: LogConfig | null = null;

function schedulePump() {
  if (pumpScheduled || !currentSink) return;
  pumpScheduled = true;
  queueMicrotask(pumpOnce);
}

function pumpOnce() {
  pumpScheduled = false;
  if (!currentSink) return;
  const max = currentConfig?.maxEntriesPerDrain ?? 256;
  const entries: LogEntry[] = core.drainNativeLogs(max);
  for (const e of entries) {
    try {
      currentSink(e);
    } catch {
      // Sink errors are swallowed; native side tracks dropped entries.
    }
  }
  // Reschedule if there might be more logs.
  if (entries.length === max) {
    schedulePump();
  }
}

export function setLogSink(sink: LogSink, config?: Partial<LogConfig>): void {
  const level: LogLevel = config?.level ?? 'info';
  const cats: LogCategory[] | undefined = config?.categories;
  currentSink = sink;
  currentConfig = {
    level,
    categories: cats,
    maxEntriesPerDrain: config?.maxEntriesPerDrain,
  } as LogConfig;
  core.setNativeLoggingEnabled(true, level, cats);
  schedulePump();
}

export function clearLogSink(): void {
  currentSink = null;
  currentConfig = null;
  core.setNativeLoggingEnabled(false, 'info', undefined);
}

export function drainLogsOnce(maxEntries = 256): LogEntry[] {
  return core.drainNativeLogs(maxEntries);
}

export function getAsyncMetrics(): DebugAsyncStats {
  return core._debugAsyncStats();
}
