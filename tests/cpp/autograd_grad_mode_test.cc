// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>

#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/meta.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/kernel_function.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"

using vbt::dispatch::Dispatcher;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::KernelFunction;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

static TensorImpl make_cpu_dense_f32(const std::vector<std::int64_t>& sizes, float fill) {
  using vbt::core::StoragePtr;
  using vbt::core::make_intrusive;

  std::size_t ne = 1;
  for (std::int64_t s : sizes) {
    ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  }
  const std::size_t nbytes = ne * sizeof(float);

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
    float* p = static_cast<float*>(buf);
    for (std::size_t i = 0; i < ne; ++i) {
      p[i] = fill;
    }
  }

  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides(sizes.size());
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  return TensorImpl(st, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
}

// Minimal CPU vt::add kernel used only for gating tests.
static TensorImpl vt_add_impl(const TensorImpl& a, const TensorImpl& b) {
  (void)b;
  return a;
}

TEST(AutogradGradModeTest, GatingMatrixMatchesDesign) {
  auto& D = Dispatcher::instance();
  const char* op = "vt::add";

  // Ensure a simple vt::add schema and CPU kernel exist.
  try { D.registerLibrary("vt"); } catch (...) {}
  try { D.def("vt::add(Tensor, Tensor) -> Tensor"); } catch (...) {}
  try { D.registerCpuKernel(op, &vt_add_impl); } catch (...) {}

  // Install boxed autograd fallback for vt::add using the real wrapper.
  KernelFunction kf = KernelFunction::makeBoxedCtx(/*arity=*/2, &vbt::autograd::autograd_fallback_ctx, /*ctx=*/nullptr);
  try {
    (void)D.tryRegisterAutogradFallback(op, kf);
  } catch (...) {
  }

  ASSERT_TRUE(D.has(op));

  auto run_case = [&](bool raw_grad, bool inf, bool any_req_grad) {
    // Configure TLS flags.
    vbt::autograd::GradMode::set_enabled(raw_grad);
    vbt::autograd::InferenceMode::set_enabled(inf);

    // Fresh inputs each time so metadata does not leak across cases.
    TensorImpl a = make_cpu_dense_f32({2}, 1.0f);
    TensorImpl b = make_cpu_dense_f32({2}, 2.0f);

    if (any_req_grad) {
      vbt::autograd::set_requires_grad(a, true);
    }

    BoxedStack s;
    s.push_back(a);
    s.push_back(b);

    D.callBoxed(op, s);
    ASSERT_EQ(s.size(), 1u);
    TensorImpl& out = s[0];

    const vbt::autograd::AutogradMeta* meta = vbt::autograd::get_autograd_meta(out);
    const bool has_grad_fn = meta && static_cast<bool>(meta->grad_fn);

    const bool expected_do_autograd = raw_grad && !inf && any_req_grad;

    EXPECT_EQ(has_grad_fn, expected_do_autograd)
        << "raw_grad=" << raw_grad
        << " inf=" << inf
        << " any_req_grad=" << any_req_grad;
  };

  const bool B[2] = {false, true};
  for (bool raw : B) {
    for (bool inf : B) {
      for (bool any : B) {
        run_case(raw, inf, any);
      }
    }
  }
}
