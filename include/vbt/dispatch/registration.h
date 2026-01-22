// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <type_traits>

#include "vbt/dispatch/dispatcher.h"

namespace vbt {
namespace dispatch {

struct LibraryRegistrar { explicit LibraryRegistrar(const char* ns) { (void)ns; Dispatcher::instance().registerLibrary(ns); } };
struct OpRegistrar { explicit OpRegistrar(const char* def) { Dispatcher::instance().def(def); } };

template<class Fn>
struct CpuImplRegistrar { CpuImplRegistrar(const char* name, Fn* fn) { Dispatcher::instance().registerCpuKernel<Fn>(name, fn); } };

template<class Fn>
struct CudaImplRegistrar { CudaImplRegistrar(const char* name, Fn* fn) { Dispatcher::instance().registerCudaKernel<Fn>(name, fn); } };

#define VBT_CONCAT2(a,b) a##b
#define VBT_CONCAT(a,b) VBT_CONCAT2(a,b)

#define VBT_LIBRARY(ns) static ::vbt::dispatch::LibraryRegistrar VBT_CONCAT(_vbt_lib_reg_, __LINE__){#ns};
#define VBT_OP(def) static ::vbt::dispatch::OpRegistrar VBT_CONCAT(_vbt_op_reg_, __LINE__){def};
#define VBT_IMPL_CPU(name, fn) static ::vbt::dispatch::CpuImplRegistrar<std::remove_pointer_t<decltype(fn)>> VBT_CONCAT(_vbt_impl_reg_cpu_, __LINE__){name, fn};
#define VBT_IMPL_CUDA(name, fn) static ::vbt::dispatch::CudaImplRegistrar<std::remove_pointer_t<decltype(fn)>> VBT_CONCAT(_vbt_impl_reg_cuda_, __LINE__){name, fn};

} // namespace dispatch
} // namespace vbt
