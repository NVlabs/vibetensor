// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Minimal stub of Node-API headers used only for C++ unit tests
// that depend on vbt/node/* headers but do not actually invoke
// any Node-API functions. This avoids a hard dependency on the
// full Node.js development headers when building the core C++
// test suite.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct napi_env__* napi_env;
typedef struct napi_value__* napi_value;
typedef struct napi_ref__* napi_ref;
typedef struct napi_callback_info__* napi_callback_info;

typedef uint32_t napi_status;

enum {
  napi_ok = 0,
  napi_invalid_arg = 1,
};

#define NAPI_AUTO_LENGTH ((size_t)-1)

#ifdef __cplusplus
}  // extern "C"
#endif
