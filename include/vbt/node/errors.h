// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <node_api.h>

namespace vbt {
namespace node {

// Strip raw pointer-like substrings (e.g., 0x1234abcd) from messages.
std::string SanitizePointers(const std::string& in);

// Strip absolute filesystem paths from messages.
std::string SanitizePaths(const std::string& in);

// Construct a JS Error or TypeError with a stable string code.
//  - message: human‑readable text; will be sanitized for pointers and paths.
//  - code: string tag such as "EINVAL", "ERUNTIME", "EOOM", "EUNKNOWN",
//          or "ENOCUDA". When null, defaults to "ERUNTIME".
//  - type_error: when true, creates a TypeError; otherwise a generic Error.
//
// On internal failures this aborts the process via napi_fatal_error.
napi_value MakeErrorWithCode(napi_env env,
                             const std::string& message,
                             const char* code,
                             bool type_error);

// Helper that throws the Error created by MakeErrorWithCode.
inline void ThrowErrorWithCode(napi_env env,
                               const std::string& message,
                               const char* code,
                               bool type_error) {
  napi_value err = MakeErrorWithCode(env, message, code, type_error);
  napi_throw(env, err);
}

}  // namespace node
}  // namespace vbt
