// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/dispatch/kernel_function.h"

#include <stdexcept>

namespace vbt {
namespace dispatch {

void KernelFunction::callBoxed(const std::string& fqname, BoxedStack& stack) const {
  StackGuard guard(stack, arity);
  if (stack.size() != arity) {
    throw std::invalid_argument("arity mismatch " + fqname + ": expected " + std::to_string(arity) + ", got " + std::to_string(stack.size()));
  }
  if (mode == Mode::Boxed) {
    if (!boxed) throw std::runtime_error("invalid kernel mode: " + fqname);
    boxed(stack);
    if (stack.size() != 1) {
      throw std::runtime_error("boxed kernel did not produce exactly one result: " + fqname);
    }
    guard.commit();
    return;
  }
  if (mode == Mode::Unboxed) {
    vbt::core::TensorImpl out;
    if (arity == 0) {
      auto fn = reinterpret_cast<Unboxed0>(unboxed_ptr);
      if (!fn) throw std::runtime_error("no CPU kernel registered: " + fqname);
      out = fn();
      stack.clear();
      stack.push_back(out);
      guard.commit();
      return;
    } else if (arity == 1) {
      auto fn = reinterpret_cast<Unboxed1>(unboxed_ptr);
      if (!fn) throw std::runtime_error("no CPU kernel registered: " + fqname);
      out = fn(stack[0]);
      stack.clear();
      stack.push_back(out);
      guard.commit();
      return;
    } else if (arity == 2) {
      auto fn = reinterpret_cast<Unboxed2>(unboxed_ptr);
      if (!fn) throw std::runtime_error("no CPU kernel registered: " + fqname);
      out = fn(stack[0], stack[1]);
      stack.clear();
      stack.push_back(out);
      guard.commit();
      return;
    } else if (arity == 3) {
      auto fn = reinterpret_cast<Unboxed3>(unboxed_ptr);
      if (!fn) throw std::runtime_error("no CPU kernel registered: " + fqname);
      out = fn(stack[0], stack[1], stack[2]);
      stack.clear();
      stack.push_back(out);
      guard.commit();
      return;
    } else if (arity == 4) {
      auto fn = reinterpret_cast<Unboxed4>(unboxed_ptr);
      if (!fn) throw std::runtime_error("no CPU kernel registered: " + fqname);
      out = fn(stack[0], stack[1], stack[2], stack[3]);
      stack.clear();
      stack.push_back(out);
      guard.commit();
      return;
    }
  }
  if (mode == Mode::BoxedWithCtx) {
    if (!boxed_ctx) throw std::runtime_error("invalid kernel mode: " + fqname);
    boxed_ctx(ctx, stack);
    if (stack.size() != 1) {
      throw std::runtime_error("boxed kernel did not produce exactly one result: " + fqname);
    }
    guard.commit();
    return;
  }
  throw std::runtime_error("invalid kernel mode: " + fqname);
}

} // namespace dispatch
} // namespace vbt
