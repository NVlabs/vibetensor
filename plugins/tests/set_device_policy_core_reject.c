// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

static vt_status post_def_boxed(vt_stream /*s*/,
                                const vt_tensor* /*args*/,
                                size_t /*nargs*/,
                                vt_tensor* out) {
  if (!g_host || !g_host->def) return VT_STATUS_INTERNAL;
  if (out) *out = NULL;
  // Attempt to grow the dispatcher registry after init.
  return g_host->def("p2_reject::late_dummy(Tensor) -> Tensor");
}

static vt_status post_policy_boxed(vt_stream /*s*/,
                                   const vt_tensor* /*args*/,
                                   size_t /*nargs*/,
                                   vt_tensor* out) {
  if (!g_host || !g_host->set_device_policy) return VT_STATUS_INTERNAL;
  if (out) *out = NULL;
  // Attempt to mutate device policy metadata after init.
  return g_host->set_device_policy(
      "p2_reject::post_policy",
      VT_DEVICE_POLICY_ALL_SAME_DEVICE,
      /*dispatch_arg_mask=*/0ULL,
      /*constraints=*/NULL,
      /*nconstraints=*/0,
      /*allow_undefined_mask=*/0ULL);
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host,
                          struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;

  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_set_device_policy_core_reject";

  const char* mode = getenv("VBT_SET_DEVICE_POLICY_CORE_REJECT_MODE");
  if (mode && strcmp(mode, "def_null") == 0) {
    if (!host->def) return VT_STATUS_UNSUPPORTED;
    return host->def(NULL);
  }
  if (mode && strcmp(mode, "def_empty") == 0) {
    if (!host->def) return VT_STATUS_UNSUPPORTED;
    return host->def("");
  }
  if (mode && strcmp(mode, "policy_not_owned") == 0) {
    if (host->host_abi_major != 1 || host->host_abi_minor < 4 ||
        !host->set_device_policy) {
      if (host->set_last_error) {
        host->set_last_error(
            "host does not provide set_device_policy (need ABI >= 1.4)");
      }
      return VT_STATUS_INVALID_ARG;
    }

    // Attempt to set policy on a core op. Host must reject this as not owned.
    return host->set_device_policy(
        "vt::add",
        VT_DEVICE_POLICY_ALL_SAME_DEVICE,
        /*dispatch_arg_mask=*/0ULL,
        /*constraints=*/NULL,
        /*nconstraints=*/0,
        /*allow_undefined_mask=*/0ULL);
  }

  if (host->register_library) {
    (void)host->register_library("p2_reject");
  }

  if (host->def) {
    vt_status st = host->def("p2_reject::post_def(Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
    st = host->def("p2_reject::post_policy(Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  }

  if (host->register_kernel_boxed) {
    vt_status st = host->register_kernel_boxed(
        "p2_reject::post_def", kDLCPU, &post_def_boxed);
    if (st != VT_STATUS_OK) return st;
    st = host->register_kernel_boxed(
        "p2_reject::post_policy", kDLCPU, &post_policy_boxed);
    if (st != VT_STATUS_OK) return st;
  } else {
    return VT_STATUS_UNSUPPORTED;
  }

  // Avoid leaving a stale error string on a successful init.
  if (host->set_last_error) host->set_last_error("");
  return VT_STATUS_OK;
}
