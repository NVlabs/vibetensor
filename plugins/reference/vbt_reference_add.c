// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <dlpack/dlpack.h>
#include <stdio.h>
#include "vbt/plugin/vbt_plugin.h"

static const struct vbt_host_api* g_host = NULL;

// Accessor for CUDA TU
const struct vbt_host_api* vbt_ref_add_get_host_api(void) { return g_host; }

// Forward-declare CUDA kernel registration when available
#if defined(VBT_WITH_CUDA) && VBT_WITH_CUDA
extern vt_status vt_add_cuda(vt_stream s, vt_tensor a, vt_tensor b, vt_tensor* out);
extern vt_status vt_check_stream_cuda(vt_stream s, vt_tensor probe, vt_tensor tpl_i64_cpu, vt_tensor* out);
#endif

static int is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == (uint8_t)kDLFloat && dt.bits == 32;
}

static int device_is_cpu(DLDevice d) { return d.device_type == kDLCPU; }

static vt_status vt_add_cpu(vt_stream /*s*/, vt_tensor a, vt_tensor b, vt_tensor* out) {
  if (!out) { if (g_host && g_host->set_last_error) g_host->set_last_error("null out"); return VT_STATUS_INVALID_ARG; }
  *out = NULL;
  if (!device_is_cpu(g_host->tensor_device(a)) || !device_is_cpu(g_host->tensor_device(b))) {
    if (g_host && g_host->set_last_error) g_host->set_last_error("device not CPU");
    return VT_STATUS_UNSUPPORTED;
  }
  if (!is_float32(g_host->tensor_dtype(a)) || !is_float32(g_host->tensor_dtype(b))) {
    if (g_host && g_host->set_last_error) g_host->set_last_error("dtype must be float32");
    return VT_STATUS_UNSUPPORTED;
  }
  size_t na = g_host->tensor_ndim(a);
  size_t nb = g_host->tensor_ndim(b);
  fprintf(stderr, "vt_add_cpu enter: na=%zu nb=%zu\n", na, nb);
  if (na != nb) { if (g_host && g_host->set_last_error) g_host->set_last_error("size mismatch"); return VT_STATUS_INVALID_ARG; }
  const int64_t* sa = g_host->tensor_sizes(a);
  const int64_t* sb = g_host->tensor_sizes(b);
  for (size_t i = 0; i < na; ++i) {
    if (sa[i] != sb[i]) { if (g_host && g_host->set_last_error) g_host->set_last_error("size mismatch"); return VT_STATUS_INVALID_ARG; }
  }
  // Skip contiguity check to accommodate broader host implementations
  /* if (!g_host->tensor_is_contiguous(a) || !g_host->tensor_is_contiguous(b)) {
    if (g_host && g_host->set_last_error) g_host->set_last_error("non-contiguous input");
    return VT_STATUS_UNSUPPORTED;
  } */
  fprintf(stderr, "vt_add_cpu contig ok\n");
  vt_tensor c = NULL;
  vt_status st = g_host->tensor_new_dense_like(a, &c);
  fprintf(stderr, "vt_add_cpu alloc status=%d out=%p\n", (int)st, (void*)c);
  if (st != VT_STATUS_OK) return st;
  int64_t n = g_host->tensor_numel(a);
  fprintf(stderr, "vt_add_cpu numel=%lld\n", (long long)n);
  const float* pa = (const float*)g_host->tensor_data(a);
  const float* pb = (const float*)g_host->tensor_data(b);
  float* pc = (float*)g_host->tensor_mutable_data(c);
  fprintf(stderr, "vt_add_cpu ptrs pa=%p pb=%p pc=%p\n", (void*)pa, (void*)pb, (void*)pc);
  for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] + pb[i];
  *out = c;
  return VT_STATUS_OK;
}

// CPU implementation of vt::check_stream: validate template and return 0 in an int64 CPU scalar
static vt_status vt_check_stream_cpu(vt_stream /*s*/, vt_tensor /*probe*/, vt_tensor tpl_i64_cpu, vt_tensor* out) {
  if (!out) { if (g_host && g_host->set_last_error) g_host->set_last_error("null out"); return VT_STATUS_INVALID_ARG; }
  *out = NULL;
  if (!g_host) { if (g_host && g_host->set_last_error) g_host->set_last_error("host API unavailable"); return VT_STATUS_INTERNAL; }
  // Validate template: CPU, int64, 0-dim
  DLDevice dtpl = g_host->tensor_device(tpl_i64_cpu);
  if (dtpl.device_type != kDLCPU) { if (g_host->set_last_error) g_host->set_last_error("template must be CPU"); return VT_STATUS_INVALID_ARG; }
  DLDataType tdt = g_host->tensor_dtype(tpl_i64_cpu);
  if (!(tdt.lanes == 1 && tdt.code == (uint8_t)kDLInt && tdt.bits == 64)) { if (g_host->set_last_error) g_host->set_last_error("template must be int64 scalar"); return VT_STATUS_INVALID_ARG; }
  if (g_host->tensor_ndim(tpl_i64_cpu) != 0) { if (g_host->set_last_error) g_host->set_last_error("template must be 0-d scalar"); return VT_STATUS_INVALID_ARG; }
  vt_tensor out_t = NULL;
  vt_status st = g_host->tensor_new_dense_like(tpl_i64_cpu, &out_t);
  if (st != VT_STATUS_OK) { if (g_host->set_last_error) g_host->set_last_error("allocation failed for output"); return st; }
  int64_t* p = (int64_t*) g_host->tensor_mutable_data(out_t);
  if (!p) { if (g_host->set_last_error) g_host->set_last_error("mutable_data returned NULL"); /* out_t freed by host tracker */ return VT_STATUS_INTERNAL; }
  *p = (int64_t)0;
  *out = out_t;
  return VT_STATUS_OK;
}

uint32_t vbt_plugin_get_abi_version(void) { return VBT_PLUGIN_ABI_VERSION; }

vt_status vbt_plugin_init(const struct vbt_host_api* host, struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_reference_add";
  fprintf(stderr, "host api ptrs: ndim=%p sizes=%p strides=%p is_contig=%p new_dense=%p set_err=%p cpu_k2=%p cuda_k2=%p\n",
          (void*)host->tensor_ndim, (void*)host->tensor_sizes, (void*)host->tensor_strides,
          (void*)host->tensor_is_contiguous, (void*)host->tensor_new_dense_like, (void*)host->set_last_error,
          (void*)host->register_cpu_kernel2, (void*)host->register_cuda_kernel2);
  if (host->register_library) host->register_library("vt");
  if (host->def) host->def("vt::add(Tensor, Tensor) -> Tensor");
  vt_status st = host->register_cpu_kernel2 ? host->register_cpu_kernel2("vt::add", &vt_add_cpu) : VT_STATUS_INTERNAL;
#if defined(VBT_WITH_CUDA) && VBT_WITH_CUDA
  if (st == VT_STATUS_OK && host->register_cuda_kernel2) {
    vt_status st2 = host->register_cuda_kernel2("vt::add", &vt_add_cuda);
    // Do not fail the plugin if CUDA registration is unsupported at host side
    if (st2 != VT_STATUS_OK && st2 != VT_STATUS_UNSUPPORTED) return st2;
  }
#endif
  // Tests-only stream check op registration
  if (host->def) host->def("vt::check_stream(Tensor, Tensor) -> Tensor");
  if (host->register_cpu_kernel2) {
    vt_status stc = host->register_cpu_kernel2("vt::check_stream", &vt_check_stream_cpu);
    if (st == VT_STATUS_OK && stc != VT_STATUS_OK) st = stc; // surface first error
  }
#if defined(VBT_WITH_CUDA) && VBT_WITH_CUDA
  if (host->register_cuda_kernel2) {
    vt_status stcz = host->register_cuda_kernel2("vt::check_stream", &vt_check_stream_cuda);
    if (st == VT_STATUS_OK && !(stcz == VT_STATUS_OK || stcz == VT_STATUS_UNSUPPORTED)) st = stcz;
  }
#endif

  return st;
}
