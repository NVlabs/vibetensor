// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "vbt/plugin/vbt_plugin.h"

uint32_t vbt_plugin_get_abi_version(void) {
  return VBT_PLUGIN_ABI_ENCODE(1, 0);
}

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  (void)host;
  if (!out_api) return VT_STATUS_INVALID_ARG;
  out_api->abi_version = VBT_PLUGIN_ABI_ENCODE(1, 0);
  out_api->name = "vbt_minor0_compat";
  return VT_STATUS_OK;
}
