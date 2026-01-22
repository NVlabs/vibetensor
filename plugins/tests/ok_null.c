// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <dlpack/dlpack.h>
#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

static vt_status ok_null_boxed(vt_stream /*s*/, const vt_tensor* args, size_t nargs, vt_tensor* out) {
  (void)args; (void)nargs; (void)out;
  // Deliberately return OK without setting *out
  return VT_STATUS_OK;
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_ok_null";
  if (host->register_library) host->register_library("ext");
  if (host->def) {
    vt_status st = host->def("ext::bad(Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  }
  if (host->register_kernel_boxed) {
    return host->register_kernel_boxed("ext::bad", kDLCPU, &ok_null_boxed);
  }
  return VT_STATUS_UNSUPPORTED;
}
