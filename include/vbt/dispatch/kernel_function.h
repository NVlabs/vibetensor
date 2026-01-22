// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include "vbt/dispatch/boxed.h"

namespace vbt {
namespace dispatch {

class KernelFunction {
 public:
  enum class Mode : uint8_t { Boxed, Unboxed, BoxedWithCtx };
  using BoxedFn = void(*)(BoxedStack&);
  using BoxedWithCtxFn = void(*)(void*, BoxedStack&);
  using Unboxed0 = vbt::core::TensorImpl(*)();
  using Unboxed1 = vbt::core::TensorImpl(*)(const vbt::core::TensorImpl&);
  using Unboxed2 = vbt::core::TensorImpl(*)(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
  using Unboxed3 = vbt::core::TensorImpl(*)(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
  using Unboxed4 = vbt::core::TensorImpl(*)(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&, const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);

  KernelFunction() = default;
  static KernelFunction makeBoxed(uint8_t arity, BoxedFn fn) {
    KernelFunction k; k.arity = arity; k.mode = Mode::Boxed; k.boxed = fn; return k;
  }
  static KernelFunction makeBoxedCtx(uint8_t arity, BoxedWithCtxFn fn, void* ctx) {
    KernelFunction k; k.arity = arity; k.mode = Mode::BoxedWithCtx; k.boxed_ctx = fn; k.ctx = ctx; return k;
  }
  static KernelFunction makeUnboxed0(Unboxed0 fn) { KernelFunction k; k.arity=0; k.mode=Mode::Unboxed; k.unboxed_ptr = reinterpret_cast<void*>(fn); return k; }
  static KernelFunction makeUnboxed1(Unboxed1 fn) { KernelFunction k; k.arity=1; k.mode=Mode::Unboxed; k.unboxed_ptr = reinterpret_cast<void*>(fn); return k; }
  static KernelFunction makeUnboxed2(Unboxed2 fn) { KernelFunction k; k.arity=2; k.mode=Mode::Unboxed; k.unboxed_ptr = reinterpret_cast<void*>(fn); return k; }
  static KernelFunction makeUnboxed3(Unboxed3 fn) { KernelFunction k; k.arity=3; k.mode=Mode::Unboxed; k.unboxed_ptr = reinterpret_cast<void*>(fn); return k; }
  static KernelFunction makeUnboxed4(Unboxed4 fn) { KernelFunction k; k.arity=4; k.mode=Mode::Unboxed; k.unboxed_ptr = reinterpret_cast<void*>(fn); return k; }

  void callBoxed(const std::string& fqname, BoxedStack& stack) const;

  uint8_t arity{0};
  Mode mode{Mode::Boxed};
  BoxedFn boxed{nullptr};
  BoxedWithCtxFn boxed_ctx{nullptr};
  void* ctx{nullptr};
  void* unboxed_ptr{nullptr};
};

} // namespace dispatch
} // namespace vbt
