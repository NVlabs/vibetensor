// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <dlpack/dlpack.h>

#include "vbt/plugin/vbt_plugin.h"

// Reference plugin that implements a simple add kernel on CPU
// using the TI-backed helpers vt_tensor_iter_binary_cpu.

static const struct vbt_host_api* g_host = NULL;

// Test-only accessor to inspect the host API from C++ tests.
const struct vbt_host_api* vbt_reference_ti_add_get_host_api(void) {
  return g_host;
}

static int is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == (uint8_t)kDLFloat && dt.bits == 32;
}

static int device_is_cpu(DLDevice d) {
  return d.device_type == kDLCPU;
}

struct TiAddLoopCtx {
  // Counter used in tests to validate tiling behavior if desired.
  int64_t tiles;
};

static void ti_add_loop1d(char** data,
                          const int64_t* strides,
                          int64_t size,
                          void* ctx_void) {
  struct TiAddLoopCtx* ctx = (struct TiAddLoopCtx*)ctx_void;
  if (ctx) {
    ++ctx->tiles;
  }

  char* out_base = data[0];
  char* a_base   = data[1];
  char* b_base   = data[2];
  const int64_t out_stride = strides[0];
  const int64_t a_stride   = strides[1];
  const int64_t b_stride   = strides[2];

  for (int64_t i = 0; i < size; ++i) {
    float* out = (float*)(out_base + i * out_stride);
    const float* a = (const float*)(a_base + i * a_stride);
    const float* b = (const float*)(b_base + i * b_stride);
    *out = *a + *b;
  }
}

// CPU kernel registered under vt::ti_add. It allocates an output tensor
// using tensor_new_dense_like(a) and then runs vt_tensor_iter_binary_cpu
// to perform the elementwise addition.
static vt_status vt_ti_add_cpu(vt_stream /*s*/,
                               vt_tensor a,
                               vt_tensor b,
                               vt_tensor* out) {
  if (!out) {
    if (g_host && g_host->set_last_error) {
      g_host->set_last_error("null out");
    }
    return VT_STATUS_INVALID_ARG;
  }
  *out = NULL;

  if (!g_host) {
    return VT_STATUS_INTERNAL;
  }

  if (!g_host->tensor_device || !g_host->tensor_dtype ||
      !g_host->tensor_new_dense_like || !g_host->vt_tensor_iter_binary_cpu) {
    if (g_host->set_last_error) {
      g_host->set_last_error("host API incomplete for TI helpers");
    }
    return VT_STATUS_INTERNAL;
  }

  DLDevice da = g_host->tensor_device(a);
  DLDevice db = g_host->tensor_device(b);
  if (!device_is_cpu(da) || !device_is_cpu(db)) {
    if (g_host->set_last_error) {
      g_host->set_last_error("device must be CPU");
    }
    return VT_STATUS_INVALID_ARG;
  }

  DLDataType ta = g_host->tensor_dtype(a);
  DLDataType tb = g_host->tensor_dtype(b);
  if (!is_float32(ta) || memcmp(&ta, &tb, sizeof(DLDataType)) != 0) {
    if (g_host->set_last_error) {
      g_host->set_last_error("dtype must be float32 and match");
    }
    return VT_STATUS_INVALID_ARG;
  }

  vt_tensor out_h = NULL;
  vt_status st = g_host->tensor_new_dense_like(a, &out_h);
  if (st != VT_STATUS_OK) {
    // TLS error, if any, is set by tensor_new_dense_like.
    return st;
  }

  struct TiAddLoopCtx ctx;
  ctx.tiles = 0;

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  st = g_host->vt_tensor_iter_binary_cpu(&cfg,
                                         out_h,
                                         a,
                                         b,
                                         &ti_add_loop1d,
                                         &ctx);
  if (st != VT_STATUS_OK) {
    // On failure, the host-side AllocTracker will reclaim out_h.
    return st;
  }

  *out = out_h;
  return VT_STATUS_OK;
}

uint32_t vbt_plugin_get_abi_version(void) {
  return VBT_PLUGIN_ABI_VERSION;
}

vt_status vbt_plugin_init(const struct vbt_host_api* host,
                          struct vbt_plugin_api* out_api) {
  if (!host || !out_api) {
    return VT_STATUS_INVALID_ARG;
  }

  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_reference_ti_add";

  if (!host->vt_tensor_iter_binary_cpu) {
    if (host->set_last_error) {
      host->set_last_error("vt_tensor_iter_binary_cpu helper unavailable");
    }
    return VT_STATUS_UNSUPPORTED;
  }

  // Register a test-only op vt::ti_add(Tensor, Tensor) -> Tensor that
  // uses vt_tensor_iter_binary_cpu inside vt_ti_add_cpu.
  if (host->register_library) {
    host->register_library("vt");
  }
  if (host->def) {
    host->def("vt::ti_add(Tensor, Tensor) -> Tensor");
  }

  if (host->register_cpu_kernel2) {
    return host->register_cpu_kernel2("vt::ti_add", &vt_ti_add_cpu);
  }

  if (host->set_last_error) {
    host->set_last_error("register_cpu_kernel2 unavailable");
  }
  return VT_STATUS_UNSUPPORTED;
}
