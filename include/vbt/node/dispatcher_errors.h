// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <node_api.h>

#include "vbt/node/dispatcher.h"

namespace vbt {
namespace node {

// Construct a JS Error/TypeError for dispatcher failures.
// - Chooses class + .code based on CallOpErrorKind.
// - Sanitizes raw_message to strip pointers and absolute paths.
// - Message shape: "callOp(" + op_name + "): " + sanitized_message.
napi_value MakeDispatchError(
    napi_env env,
    CallOpErrorKind kind,
    const std::string& op_name,
    const std::string& raw_message);

// Throw a dispatch error created by MakeDispatchError.
inline void ThrowDispatchError(
    napi_env env,
    CallOpErrorKind kind,
    const std::string& op_name,
    const std::string& raw_message) {
  napi_value err = MakeDispatchError(env, kind, op_name, raw_message);
  napi_throw(env, err);
}

} // namespace node
} // namespace vbt
