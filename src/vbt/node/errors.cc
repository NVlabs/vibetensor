// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/errors.h"

#include <cctype>
#include <string>

namespace vbt {
namespace node {
namespace {

[[noreturn]] void Fatal(napi_env env, const char* msg) {
  (void)env;  // unused in napi_fatal_error path
  napi_fatal_error("vbt.node.MakeErrorWithCode", NAPI_AUTO_LENGTH,
                   msg, NAPI_AUTO_LENGTH);
}

}  // namespace

std::string SanitizePointers(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size();) {
    if (i + 2 <= in.size() && in[i] == '0' &&
        (in[i + 1] == 'x' || in[i + 1] == 'X')) {
      // Consume 0x[0-9a-fA-F]+
      std::size_t j = i + 2;
      while (j < in.size() &&
             std::isxdigit(static_cast<unsigned char>(in[j]))) {
        ++j;
      }
      out.append("0x…");
      i = j;
    } else {
      out.push_back(in[i]);
      ++i;
    }
  }
  return out;
}

std::string SanitizePaths(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size();) {
    if (in[i] == '/') {
      // Treat as absolute path until whitespace.
      out.append("<path>");
      while (i < in.size() &&
             !std::isspace(static_cast<unsigned char>(in[i]))) {
        ++i;
      }
    } else {
      out.push_back(in[i]);
      ++i;
    }
  }
  return out;
}

napi_value MakeErrorWithCode(napi_env env,
                             const std::string& message,
                             const char* code,
                             bool type_error) {
  const char* effective_code = code ? code : "ERUNTIME";

  std::string msg = SanitizePointers(message);
  msg = SanitizePaths(msg);

  napi_value js_msg;
  napi_status st = napi_create_string_utf8(env, msg.c_str(), msg.size(), &js_msg);
  if (st != napi_ok) {
    Fatal(env, "failed to create error message string");
  }

  napi_value err;
  if (type_error) {
    st = napi_create_type_error(env, nullptr, js_msg, &err);
  } else {
    st = napi_create_error(env, nullptr, js_msg, &err);
  }
  if (st != napi_ok) {
    Fatal(env, "failed to create JS Error object");
  }

  napi_value js_code;
  st = napi_create_string_utf8(env, effective_code, NAPI_AUTO_LENGTH, &js_code);
  if (st != napi_ok) {
    Fatal(env, "failed to create error code string");
  }

  st = napi_set_named_property(env, err, "code", js_code);
  if (st != napi_ok) {
    Fatal(env, "failed to assign error.code property");
  }

  return err;
}

}  // namespace node
}  // namespace vbt
