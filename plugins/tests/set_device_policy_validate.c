// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;

  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_set_device_policy_validate";

  if (host->host_abi_major != 1 || host->host_abi_minor < 4 || !host->set_device_policy) {
    if (host->set_last_error) {
      host->set_last_error("host does not provide set_device_policy (need ABI >= 1.4)");
    }
    return VT_STATUS_INVALID_ARG;
  }

  // Define a dummy op to attach policy metadata to.
  if (host->register_library) {
    (void)host->register_library("p2_test");
  }
  if (host->def) {
    vt_status st = host->def("p2_test::dummy(Tensor, Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  }

  const char* fqname = "p2_test::dummy";

  // 1) Valid call should succeed.
  {
    vt_device_constraint c = {1, VT_CONSTRAINT_CPU_I64_SCALAR_0D, {0}};
    vt_status st = host->set_device_policy(
        fqname,
        VT_DEVICE_POLICY_MASKED_SAME_DEVICE,
        /*dispatch_arg_mask=*/1ULL,
        &c,
        /*nconstraints=*/1,
        /*allow_undefined_mask=*/0ULL);
    if (st != VT_STATUS_OK) {
      if (host->set_last_error) host->set_last_error("valid set_device_policy call failed");
      return st;
    }
  }

  // 2) Duplicate constraint indices must be rejected.
  {
    vt_device_constraint dup[2];
    for (size_t i = 0; i < 2; ++i) {
      dup[i].index = 1;
      dup[i].kind = VT_CONSTRAINT_CPU_I64_SCALAR_0D;
      for (size_t j = 0; j < sizeof(dup[i].reserved); ++j) dup[i].reserved[j] = 0;
    }

    vt_status st = host->set_device_policy(
        fqname,
        VT_DEVICE_POLICY_MASKED_SAME_DEVICE,
        /*dispatch_arg_mask=*/1ULL,
        dup,
        /*nconstraints=*/2,
        /*allow_undefined_mask=*/0ULL);
    if (st != VT_STATUS_INVALID_ARG) {
      if (host->set_last_error) host->set_last_error("expected VT_STATUS_INVALID_ARG for duplicate indices");
      return VT_STATUS_INVALID_ARG;
    }
  }

  // 3) Unknown enum values must be rejected.
  {
    vt_status st = host->set_device_policy(
        fqname,
        (vt_device_policy)99,
        /*dispatch_arg_mask=*/0ULL,
        NULL,
        /*nconstraints=*/0,
        /*allow_undefined_mask=*/0ULL);
    if (st != VT_STATUS_INVALID_ARG) {
      if (host->set_last_error) host->set_last_error("expected VT_STATUS_INVALID_ARG for unknown policy enum");
      return VT_STATUS_INVALID_ARG;
    }
  }

  // 4) nconstraints > 64 must be rejected.
  {
    static vt_device_constraint many[65];
    for (size_t i = 0; i < 65; ++i) {
      many[i].index = (uint8_t)0;
      many[i].kind = VT_CONSTRAINT_MUST_MATCH_DISPATCH_IF_DEFINED;
      for (size_t j = 0; j < sizeof(many[i].reserved); ++j) many[i].reserved[j] = 0;
    }

    vt_status st = host->set_device_policy(
        fqname,
        VT_DEVICE_POLICY_ALL_SAME_DEVICE,
        /*dispatch_arg_mask=*/0ULL,
        many,
        /*nconstraints=*/65,
        /*allow_undefined_mask=*/0ULL);
    if (st != VT_STATUS_INVALID_ARG) {
      if (host->set_last_error) host->set_last_error("expected VT_STATUS_INVALID_ARG for nconstraints > 64");
      return VT_STATUS_INVALID_ARG;
    }
  }

  // 5) Fabric5Arg policy is core-only (must be rejected as unsupported).
  {
    vt_status st = host->set_device_policy(
        fqname,
        VT_DEVICE_POLICY_FABRIC5ARG,
        /*dispatch_arg_mask=*/0ULL,
        NULL,
        /*nconstraints=*/0,
        /*allow_undefined_mask=*/0ULL);
    if (st != VT_STATUS_UNSUPPORTED) {
      if (host->set_last_error) host->set_last_error("expected VT_STATUS_UNSUPPORTED for Fabric5Arg policy");
      return VT_STATUS_INVALID_ARG;
    }
  }

  // Avoid leaving a stale error string on a successful init.
  if (host->set_last_error) host->set_last_error("");
  return VT_STATUS_OK;
}
