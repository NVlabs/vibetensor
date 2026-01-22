// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

static int is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == (uint8_t)kDLFloat && dt.bits == 32;
}

static int device_is_cpu(DLDevice d) { return d.device_type == kDLCPU; }

static vt_status vt_p3_set_device_policy_cpu(vt_stream /*s*/,
                                             vt_tensor a,
                                             vt_tensor b,
                                             vt_tensor* out) {
  if (!out) {
    if (g_host && g_host->set_last_error) g_host->set_last_error("null out");
    return VT_STATUS_INVALID_ARG;
  }
  *out = NULL;
  if (!g_host) {
    return VT_STATUS_INTERNAL;
  }

  if (!device_is_cpu(g_host->tensor_device(a)) ||
      !device_is_cpu(g_host->tensor_device(b))) {
    if (g_host->set_last_error) g_host->set_last_error("device not CPU");
    return VT_STATUS_UNSUPPORTED;
  }
  if (!is_float32(g_host->tensor_dtype(a)) ||
      !is_float32(g_host->tensor_dtype(b))) {
    if (g_host->set_last_error) g_host->set_last_error("dtype must be float32");
    return VT_STATUS_UNSUPPORTED;
  }

  size_t na = g_host->tensor_ndim(a);
  size_t nb = g_host->tensor_ndim(b);
  if (na != nb) {
    if (g_host->set_last_error) g_host->set_last_error("size mismatch");
    return VT_STATUS_INVALID_ARG;
  }
  const int64_t* sa = g_host->tensor_sizes(a);
  const int64_t* sb = g_host->tensor_sizes(b);
  for (size_t i = 0; i < na; ++i) {
    if (sa[i] != sb[i]) {
      if (g_host->set_last_error) g_host->set_last_error("size mismatch");
      return VT_STATUS_INVALID_ARG;
    }
  }

  vt_tensor c = NULL;
  vt_status st = g_host->tensor_new_dense_like(a, &c);
  if (st != VT_STATUS_OK) return st;

  int64_t n = g_host->tensor_numel(a);
  const float* pa = (const float*)g_host->tensor_data(a);
  const float* pb = (const float*)g_host->tensor_data(b);
  float* pc = (float*)g_host->tensor_mutable_data(c);
  if (!pc) {
    // host_tensor_mutable_data sets a TLS error message.
    return VT_STATUS_INTERNAL;
  }
  for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] + pb[i];

  *out = c;
  return VT_STATUS_OK;
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host,
                          struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "p3_set_device_policy";

  if (host->register_library) host->register_library("vt");

  if (host->def) {
    vt_status st = host->def("vt::p3_set_device_policy(Tensor, Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  } else {
    return VT_STATUS_UNSUPPORTED;
  }

  if (host->host_abi_major != 1 || host->host_abi_minor < 4 ||
      !host->set_device_policy) {
    if (host->set_last_error) {
      host->set_last_error(
          "host does not provide set_device_policy (need ABI >= 1.4)");
    }
    return VT_STATUS_UNSUPPORTED;
  }

  vt_device_constraint c = (vt_device_constraint){0};
  c.kind = VT_CONSTRAINT_DEFER_TO_KERNEL;
  c.index = 1;

  vt_status st = host->set_device_policy(
      "vt::p3_set_device_policy",
      VT_DEVICE_POLICY_MASKED_SAME_DEVICE,
      /*dispatch_arg_mask=*/1ULL,
      /*constraints=*/&c,
      /*nconstraints=*/1,
      /*allow_undefined_mask=*/2ULL);
  if (st != VT_STATUS_OK) return st;

  if (host->register_cpu_kernel2) {
    st = host->register_cpu_kernel2("vt::p3_set_device_policy",
                                    &vt_p3_set_device_policy_cpu);
    if (st != VT_STATUS_OK) return st;
  } else {
    return VT_STATUS_UNSUPPORTED;
  }

  // Avoid leaving a stale error string on a successful init.
  if (host->set_last_error) host->set_last_error("");
  return VT_STATUS_OK;
}
