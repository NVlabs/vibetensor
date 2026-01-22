// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Intentionally duplicated between p3_dup_def_a.c and p3_dup_def_b.c.
//
// These two plugins build into distinct .so files but both attempt to:
//   def("vt::p3_dup_def(Tensor, Tensor) -> Tensor")
//
// This enables deterministic duplicate-def concurrency tests for the
// atomic plugin commit path.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

static void write_ok_marker(const char* filename) {
#if defined(__unix__) || defined(__APPLE__)
  const char* dir = getenv("VBT_P3_DUP_DEF_MARKER_DIR");
  if (!dir || *dir == '\0') return;

  char path[512];
  const int n = snprintf(path, sizeof(path), "%s/%s", dir, filename);
  if (n <= 0 || (size_t)n >= sizeof(path)) return;

  FILE* f = fopen(path, "w");
  if (!f) return;
  fputs("ok\n", f);
  fclose(f);
#else
  (void)filename;
#endif
}

static int is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == (uint8_t)kDLFloat && dt.bits == 32;
}

static int device_is_cpu(DLDevice d) { return d.device_type == kDLCPU; }

static vt_status vt_p3_dup_def_cpu(vt_stream /*s*/,
                                  vt_tensor a,
                                  vt_tensor b,
                                  vt_tensor* out) {
  if (!out) {
    if (g_host && g_host->set_last_error) g_host->set_last_error("null out");
    return VT_STATUS_INVALID_ARG;
  }
  *out = NULL;
  if (!g_host) return VT_STATUS_INTERNAL;

  if (!device_is_cpu(g_host->tensor_device(a)) || !device_is_cpu(g_host->tensor_device(b))) {
    if (g_host->set_last_error) g_host->set_last_error("device not CPU");
    return VT_STATUS_UNSUPPORTED;
  }
  if (!is_float32(g_host->tensor_dtype(a)) || !is_float32(g_host->tensor_dtype(b))) {
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
  out_api->name = "p3_dup_def_b";

#if defined(__unix__) || defined(__APPLE__)
  // Increase overlap for concurrent-load tests.
  usleep(50 * 1000);
#endif

  if (host->register_library) host->register_library("vt");

  vt_status st = VT_STATUS_INTERNAL;
  if (host->def) {
    st = host->def("vt::p3_dup_def(Tensor, Tensor) -> Tensor");
    if (st != VT_STATUS_OK) return st;
  }

  if (host->register_cpu_kernel2) {
    st = host->register_cpu_kernel2("vt::p3_dup_def", &vt_p3_dup_def_cpu);
    if (st != VT_STATUS_OK) return st;
  }

  write_ok_marker("p3_dup_def_b.ok");
  return VT_STATUS_OK;
}
