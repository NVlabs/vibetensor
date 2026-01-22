// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/dispatcher_errors.h"

#include <node_api.h>

#include <cctype>
#include <cstdlib>
#include <string>

#include "vbt/node/errors.h"
#include "vbt/node/util.h"

namespace vbt {
namespace node {
namespace {

bool IsOpsCompatEnabled() {
  const char* v = std::getenv("VBT_NODE_OPS_COMPAT");
  if (!v || *v == '\0') return false;
  std::string s(v);
  for (auto& ch : s) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return s == "1" || s == "true" || s == "yes";
}

std::string MaybeRewriteForCompat(const std::string& op_name,
                                  const std::string& msg) {
  if (!IsOpsCompatEnabled()) return msg;

  // Very small set of normalizations mirroring Python wording.
  if (msg.rfind("unknown op: ", 0) == 0) {
    return std::string("unknown operator: ") + op_name;
  }
  return msg;
}

const char* ErrorCodeForKind(CallOpErrorKind kind) {
  switch (kind) {
    case CallOpErrorKind::InvalidArg: return "EINVAL";
    case CallOpErrorKind::BadAlloc: return "EOOM";
    case CallOpErrorKind::Runtime:
    case CallOpErrorKind::UnknownOp:
    case CallOpErrorKind::AsyncWorkFailure: return "ERUNTIME";
    case CallOpErrorKind::None:
    case CallOpErrorKind::Unknown:
    default: return "EUNKNOWN";
  }
}

bool IsTypeError(CallOpErrorKind kind) {
  return kind == CallOpErrorKind::InvalidArg;
}

}  // namespace

napi_value MakeDispatchError(
    napi_env env,
    CallOpErrorKind kind,
    const std::string& op_name,
    const std::string& raw_message) {
  std::string msg = raw_message;
  msg = MaybeRewriteForCompat(op_name, msg);

  std::string full = "callOp(" + op_name + "): " + msg;

  const char* code = ErrorCodeForKind(kind);
  const bool is_type_error = IsTypeError(kind);

  return MakeErrorWithCode(env, full, code, is_type_error);
}

}  // namespace node
}  // namespace vbt
