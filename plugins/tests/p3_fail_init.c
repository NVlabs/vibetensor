// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

// A CPU kernel for vt::p3_fail. It should never execute because the plugin init
// deliberately fails.
static vt_status vt_p3_fail_cpu(vt_stream /*s*/, vt_tensor a, vt_tensor b, vt_tensor* out) {
  (void)a;
  (void)b;
  if (!g_host || !out) return VT_STATUS_INTERNAL;
  *out = NULL;
  if (g_host->set_last_error) g_host->set_last_error("vt_p3_fail_cpu should not execute");
  return VT_STATUS_INTERNAL;
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;

  *out_api = (struct vbt_plugin_api){0};
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "p3_fail_init";

  if (host->def) {
    vt_status st = host->def("vt::p3_fail(Tensor, Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  }

  if (host->register_cpu_kernel2) {
    vt_status st = host->register_cpu_kernel2("vt::p3_fail", &vt_p3_fail_cpu);
    if (st != VT_STATUS_OK) return st;
  }

  // Fail init after registering an op + kernel.
  if (host->set_last_error) host->set_last_error("plugin init failed");
  return VT_STATUS_INTERNAL;
}
