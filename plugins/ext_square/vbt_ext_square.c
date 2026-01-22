// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <dlpack/dlpack.h>
#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

static int is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == (uint8_t)kDLFloat && dt.bits == 32;
}

static vt_status square_boxed(vt_stream /*s*/, const vt_tensor* args, size_t nargs, vt_tensor* out) {
  if (!g_host) return VT_STATUS_INTERNAL;
  if (!out) { if (g_host->set_last_error) g_host->set_last_error("null out"); return VT_STATUS_INVALID_ARG; }
  *out = NULL;
  if (nargs != 1) { if (g_host->set_last_error) g_host->set_last_error("arity must be 1"); return VT_STATUS_INVALID_ARG; }
  vt_tensor a = args[0];
  DLDevice da = g_host->tensor_device(a);
  if (da.device_type != kDLCPU) { if (g_host->set_last_error) g_host->set_last_error("device not CPU"); return VT_STATUS_UNSUPPORTED; }
  if (!is_float32(g_host->tensor_dtype(a))) { if (g_host->set_last_error) g_host->set_last_error("dtype must be float32"); return VT_STATUS_UNSUPPORTED; }
  if (!g_host->tensor_is_contiguous(a)) { if (g_host->set_last_error) g_host->set_last_error("non-contiguous input"); return VT_STATUS_UNSUPPORTED; }
  vt_status st = g_host->tensor_new_dense_like(a, out);
  if (st != VT_STATUS_OK) return st;
  int64_t n = g_host->tensor_numel(a);
  const float* pa = (const float*)g_host->tensor_data(a);
  float* pc = (float*)g_host->tensor_mutable_data(*out);
  for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] * pa[i];
  return VT_STATUS_OK;
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_ext_square";
  if (host->register_library) host->register_library("ext");
  if (host->def) {
    vt_status st = host->def("ext::square(Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  }
  if (host->register_kernel_boxed) {
    return host->register_kernel_boxed("ext::square", kDLCPU, &square_boxed);
  }
  return VT_STATUS_UNSUPPORTED;
}
