// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

static vt_status fabric_stream_boxed(vt_stream s,
                                    const vt_tensor* args,
                                    size_t nargs,
                                    vt_tensor* out) {
  if (!out) return VT_STATUS_INVALID_ARG;
  *out = NULL;

  if (!g_host || !args) return VT_STATUS_INVALID_ARG;
  if (nargs != 5) {
    if (g_host->set_last_error) g_host->set_last_error("fabric_stream_boxed: expected nargs==5");
    return VT_STATUS_INVALID_ARG;
  }

  // Extract compute_device (arg2) as CPU scalar int64.
  if (!g_host->tensor_data) return VT_STATUS_INTERNAL;
  const void* p = g_host->tensor_data(args[2]);
  if (!p) {
    if (g_host->set_last_error) g_host->set_last_error("fabric_stream_boxed: compute_device data is NULL");
    return VT_STATUS_INVALID_ARG;
  }
  const int64_t compute_dev = *(const int64_t*)p;

  // Verify that the stream handle passed by the host matches the current stream
  // for the compute device.
  vt_stream expected = 0;
  if (g_host->current_cuda_stream) {
    expected = g_host->current_cuda_stream((int32_t)compute_dev);
  }
  if (s != expected) {
    if (g_host->set_last_error) g_host->set_last_error("fabric_stream_boxed: stream mismatch");
    return VT_STATUS_RUNTIME_ERROR;
  }

  // Allocate an output tensor on the compute device.
  if (!g_host->tensor_device || !g_host->tensor_new_dense_like) {
    return VT_STATUS_UNSUPPORTED;
  }

  vt_tensor like = args[0];
  const DLDevice d0 = g_host->tensor_device(args[0]);
  const DLDevice d1 = g_host->tensor_device(args[1]);

  if (d1.device_type == kDLCUDA && d1.device_id == (int32_t)compute_dev) {
    like = args[1];
  } else if (d0.device_type == kDLCUDA && d0.device_id == (int32_t)compute_dev) {
    like = args[0];
  } else {
    if (g_host->set_last_error) g_host->set_last_error("fabric_stream_boxed: no operand on compute device");
    return VT_STATUS_INVALID_ARG;
  }

  return g_host->tensor_new_dense_like(like, out);
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;

  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_fabric_stream_check";

  // This plugin expects the generalized boxed registration API.
  if (!host->register_kernel_boxed) {
    if (host->set_last_error) host->set_last_error("host missing register_kernel_boxed");
    return VT_STATUS_UNSUPPORTED;
  }

  // The op must already be defined and marked as Fabric by the host tests.
  return host->register_kernel_boxed(
      "fabric_testlib::plugin_stream_check",
      kDLCUDA,
      &fabric_stream_boxed);
}
