// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

// A dummy vt::add CPU kernel (won't be used; registration is enough to test rollback)
static vt_status vt_add_cpu_fail(vt_stream /*s*/, vt_tensor a, vt_tensor b, vt_tensor* out) {
  (void)a; (void)b; (void)out;
  // Return an error to make it obvious if this ever executes
  if (g_host && g_host->set_last_error) g_host->set_last_error("vt_add_cpu_fail should not execute");
  return VT_STATUS_INTERNAL;
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_fail_rollback";
  // Override vt::add CPU kernel then deliberately fail init to trigger rollback
  if (host->register_cpu_kernel2) {
    vt_status st = host->register_cpu_kernel2("vt::add", &vt_add_cpu_fail);
    if (st != VT_STATUS_OK) return st;
  }
  if (host->set_last_error) host->set_last_error("init failure for rollback test");
  return VT_STATUS_INVALID_ARG;
}
