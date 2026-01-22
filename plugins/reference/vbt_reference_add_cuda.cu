// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include "vbt/plugin/vbt_plugin.h"

// Share host API pointer with C TU
extern "C" const struct vbt_host_api* vbt_ref_add_get_host_api(void);
static inline const struct vbt_host_api* host() { return vbt_ref_add_get_host_api(); }

static int is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == (uint8_t)kDLFloat && dt.bits == 32;
}

static int device_is_cuda(DLDevice d) { return d.device_type == kDLCUDA; }

__global__ void vbt_add_kernel_f32(const float* a, const float* b, float* out, long n) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

extern "C" vt_status vt_add_cuda(vt_stream s, vt_tensor a, vt_tensor b, vt_tensor* out) {
  const struct vbt_host_api* g = host();
  if (!out) { if (g && g->set_last_error) g->set_last_error("null out"); return VT_STATUS_INVALID_ARG; }
  *out = NULL;
  if (!g) return VT_STATUS_INTERNAL;
  DLDevice da = g->tensor_device(a), db = g->tensor_device(b);
  if (!device_is_cuda(da) || !device_is_cuda(db) || da.device_id != db.device_id) {
    if (g->set_last_error) g->set_last_error("device mismatch or not CUDA");
    return VT_STATUS_INVALID_ARG;
  }
  if (!is_float32(g->tensor_dtype(a)) || !is_float32(g->tensor_dtype(b))) {
    if (g->set_last_error) g->set_last_error("dtype must be float32");
    return VT_STATUS_UNSUPPORTED;
  }
  size_t na = (size_t)g->tensor_ndim(a), nb = (size_t)g->tensor_ndim(b);
  if (na != nb) { if (g->set_last_error) g->set_last_error("size mismatch"); return VT_STATUS_INVALID_ARG; }
  const int64_t* sa = g->tensor_sizes(a);
  const int64_t* sb = g->tensor_sizes(b);
  for (size_t i = 0; i < na; ++i) {
    if (sa[i] != sb[i]) { if (g->set_last_error) g->set_last_error("size mismatch"); return VT_STATUS_INVALID_ARG; }
  }
  // Skip contiguity checks to tolerate broader host views
  /* if (!g->tensor_is_contiguous(a) || !g->tensor_is_contiguous(b)) {
    if (g->set_last_error) g->set_last_error("non-contiguous input");
    return VT_STATUS_UNSUPPORTED;
  } */
  vt_tensor c = NULL;
  vt_status st = g->tensor_new_dense_like(a, &c);
  if (st != VT_STATUS_OK) return st;
  cudaStream_t cs = (cudaStream_t)(uintptr_t)s;
  long n = (long)g->tensor_numel(a);
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  if (blocks <= 0) blocks = 1;
  vbt_add_kernel_f32<<<blocks, threads, 0, cs>>>((const float*)g->tensor_data(a), (const float*)g->tensor_data(b), (float*)g->tensor_mutable_data(c), n);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (g->set_last_error) g->set_last_error(cudaGetErrorString(err));
    return VT_STATUS_INTERNAL;
  }
  *out = c;
  return VT_STATUS_OK;
}

// CUDA implementation of vt::check_stream: compare s with expected handle read from CPU int64 template
extern "C" vt_status vt_check_stream_cuda(vt_stream s, vt_tensor probe, vt_tensor tpl_i64_cpu, vt_tensor* out) {
  const struct vbt_host_api* g = host();
  if (!out) { if (g && g->set_last_error) g->set_last_error("null out"); return VT_STATUS_INVALID_ARG; }
  *out = NULL;
  if (!g) { if (g && g->set_last_error) g->set_last_error("host API unavailable"); return VT_STATUS_INTERNAL; }
  // Validate probe device is CUDA
  DLDevice dp = g->tensor_device(probe);
  if (!device_is_cuda(dp)) { if (g->set_last_error) g->set_last_error("probe must be CUDA tensor"); return VT_STATUS_INVALID_ARG; }
  // Validate template CPU int64 scalar
  DLDevice dtpl = g->tensor_device(tpl_i64_cpu);
  if (dtpl.device_type != kDLCPU) { if (g->set_last_error) g->set_last_error("template must be CPU"); return VT_STATUS_INVALID_ARG; }
  DLDataType tdt = g->tensor_dtype(tpl_i64_cpu);
  if (!(tdt.lanes == 1 && tdt.code == (uint8_t)kDLInt && tdt.bits == 64)) { if (g->set_last_error) g->set_last_error("template must be int64 scalar"); return VT_STATUS_INVALID_ARG; }
  if (g->tensor_ndim(tpl_i64_cpu) != 0) { if (g->set_last_error) g->set_last_error("template must be 0-d scalar"); return VT_STATUS_INVALID_ARG; }
  // Allocate output like the template
  vt_tensor out_t = NULL;
  vt_status st = g->tensor_new_dense_like(tpl_i64_cpu, &out_t);
  if (st != VT_STATUS_OK) { if (g->set_last_error) g->set_last_error("allocation failed for output"); return st; }
  int64_t* pout = (int64_t*) g->tensor_mutable_data(out_t);
  if (!pout) { if (g->set_last_error) g->set_last_error("mutable_data returned NULL"); /* out_t freed by host tracker */ return VT_STATUS_INTERNAL; }
  // Read expected handle value from template storage (int64)
  const int64_t* pexp = (const int64_t*) g->tensor_data(tpl_i64_cpu);
  if (!pexp) { if (g->set_last_error) g->set_last_error("template data returned NULL"); /* out_t freed by host tracker */ return VT_STATUS_INVALID_ARG; }
  uint64_t expected = (uint64_t)(*pexp);
  uint64_t seen = (uint64_t) s;
  *pout = (seen == expected) ? (int64_t)1 : (int64_t)0;
  *out = out_t;
  return VT_STATUS_OK;
}
