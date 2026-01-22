// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <node_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <exception>

#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor.h"
#include "vbt/core/checked_math.h"
#include "vbt/cpu/storage.h"
#include "vbt/node/tensor.h"
#include "vbt/node/util.h"

namespace vbt::node {
namespace {

std::vector<int64_t> make_contig_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    std::size_t idx = static_cast<std::size_t>(i);
    st[idx] = acc;
    int64_t dim = sizes[idx];
    if (dim != 0) {
      int64_t tmp = 0;
      if (!vbt::core::checked_mul_i64(acc, dim, tmp)) {
        // Overflow should have been caught earlier by NumelBytes; keep acc.
        tmp = acc;
      }
      acc = tmp;
    }
  }
  return st;
}

void JsTensorFinalizer(napi_env env, void* data, void* /*hint*/) {
  auto* jt = static_cast<JsTensor*>(data);
  if (!jt) return;

  if (jt->is_owner && jt->owner) {
    ReleaseExternalOwner(env, *jt->owner);
  }

  delete jt;
}

napi_value JsTensorSizes(napi_env env, napi_callback_info info) {
  napi_value this_arg;
  size_t argc = 0;
  napi_status st = napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "JsTensorSizes/get_cb_info");

  JsTensor* jt = nullptr;
  if (!UnwrapJsTensor(env, this_arg, &jt)) return nullptr;

  const auto& sizes = jt->impl.sizes();
  napi_value arr;
  st = napi_create_array_with_length(env, static_cast<uint32_t>(sizes.size()), &arr);
  CHECK_NAPI_OK(env, st, "JsTensorSizes/create_array");

  for (std::size_t i = 0; i < sizes.size(); ++i) {
    napi_value v;
    st = napi_create_int64(env, sizes[i], &v);
    CHECK_NAPI_OK(env, st, "JsTensorSizes/create_int64");
    st = napi_set_element(env, arr, static_cast<uint32_t>(i), v);
    CHECK_NAPI_OK(env, st, "JsTensorSizes/set_element");
  }

  return arr;
}

napi_value JsTensorDtype(napi_env env, napi_callback_info info) {
  napi_value this_arg;
  size_t argc = 0;
  napi_status st = napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "JsTensorDtype/get_cb_info");

  JsTensor* jt = nullptr;
  if (!UnwrapJsTensor(env, this_arg, &jt)) return nullptr;

  const vbt::core::ScalarType stype = jt->impl.dtype();
  const char* name = "float32";
  switch (stype) {
    case vbt::core::ScalarType::Float32: name = "float32"; break;
    case vbt::core::ScalarType::Float16: name = "float16"; break;
    case vbt::core::ScalarType::BFloat16: name = "bfloat16"; break;
    case vbt::core::ScalarType::Float64: name = "float64"; break;
    case vbt::core::ScalarType::Complex64: name = "complex64"; break;
    case vbt::core::ScalarType::Complex128: name = "complex128"; break;
    case vbt::core::ScalarType::Int32: name = "int32"; break;
    case vbt::core::ScalarType::Int64: name = "int64"; break;
    case vbt::core::ScalarType::Bool: name = "bool"; break;
    case vbt::core::ScalarType::Undefined: name = "undefined"; break;
    default: name = "unknown"; break;
  }

  napi_value out;
  st = napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &out);
  CHECK_NAPI_OK(env, st, "JsTensorDtype/create_string");
  return out;
}

napi_value JsTensorDevice(napi_env env, napi_callback_info info) {
  napi_value this_arg;
  size_t argc = 0;
  napi_status st = napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "JsTensorDevice/get_cb_info");

  JsTensor* jt = nullptr;
  if (!UnwrapJsTensor(env, this_arg, &jt)) return nullptr;

  std::string dev = jt->impl.device().to_string();
  napi_value out;
  st = napi_create_string_utf8(env, dev.c_str(), static_cast<size_t>(dev.size()), &out);
  CHECK_NAPI_OK(env, st, "JsTensorDevice/create_string");
  return out;
}

}  // namespace

napi_value WrapJsTensor(napi_env env, JsTensor* jt) {
  napi_value obj = nullptr;
  if (!TryWrapJsTensor(env, jt, &obj)) {
    // TryWrapJsTensor already deleted jt and set an exception.
    return nullptr;
  }
  return obj;
}

bool TryWrapJsTensor(napi_env env, JsTensor* jt, napi_value* out) {
  if (!jt || !out) return false;

  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  if (st != napi_ok) {
    delete jt;
    return false;
  }

  napi_property_descriptor props[] = {
      {"sizes", nullptr, JsTensorSizes, nullptr, nullptr, nullptr, napi_default,
       nullptr},
      {"dtype", nullptr, JsTensorDtype, nullptr, nullptr, nullptr, napi_default,
       nullptr},
      {"device", nullptr, JsTensorDevice, nullptr, nullptr, nullptr, napi_default,
       nullptr},
  };

  st = napi_define_properties(env, obj, sizeof(props) / sizeof(props[0]), props);
  if (st != napi_ok) {
    delete jt;
    return false;
  }

  st = napi_wrap(env, obj, jt, JsTensorFinalizer, nullptr, nullptr);
  if (st != napi_ok) {
    delete jt;
    return false;
  }

  *out = obj;
  return true;
}

napi_value WrapTensorImplAsJsTensor(napi_env env,
                                    vbt::core::TensorImpl impl) {
  napi_value obj = nullptr;
  if (!TryWrapTensorImplAsJsTensor(env, std::move(impl), &obj)) {
    return nullptr;
  }
  return obj;
}

bool TryWrapTensorImplAsJsTensor(napi_env env,
                                 vbt::core::TensorImpl impl,
                                 napi_value* out) {
  if (!out) return false;

  const vbt::core::Device dev = impl.device();
  const bool is_cuda_device =
      (dev.type == vbt::core::Device::cuda().type);
  const int32_t device_index = is_cuda_device ? dev.index : -1;

  auto owner_pair = AttachExternalOwnerForStorage(impl, is_cuda_device,
                                                  device_index);
  std::shared_ptr<ExternalMemoryOwner> owner = std::move(owner_pair.first);
  const bool is_new_owner = owner_pair.second;

  auto* jt = new JsTensor{};
  jt->impl = std::move(impl);
  jt->nbytes = owner ? owner->bytes : 0;
  jt->owner = std::move(owner);
  jt->is_owner = false;

  if (jt->owner && is_new_owner && jt->nbytes > 0) {
    jt->is_owner = true;
    AccountExternalBytes(env, *jt->owner);
  }

  return TryWrapJsTensor(env, jt, out);
}

bool UnwrapJsTensor(napi_env env, napi_value value, JsTensor** out) {
  if (!out) return false;
  JsTensor* jt = nullptr;
  napi_status st = napi_unwrap(env, value, reinterpret_cast<void**>(&jt));
  if (st != napi_ok || jt == nullptr) {
    ThrowTypeError(env, "expected Tensor handle");
    return false;
  }
  *out = jt;
  return true;
}

napi_value Zeros(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "Zeros/get_cb_info");

  if (argc < 1) {
    return ThrowTypeError(env, "zeros(sizes, opts?) expects at least 1 argument");
  }

  std::vector<int64_t> sizes;
  if (!ParseSizes(env, args[0], &sizes)) return nullptr;  // sets TypeError

  napi_value opts = (argc >= 2) ? args[1] : nullptr;
  napi_value js_dtype = nullptr;
  napi_value js_device = nullptr;
  if (opts) {
    napi_valuetype t;
    st = napi_typeof(env, opts, &t);
    CHECK_NAPI_OK(env, st, "Zeros/opts_typeof");
    if (t == napi_object) {
      st = napi_get_named_property(env, opts, "dtype", &js_dtype);
      if (st != napi_ok) js_dtype = nullptr;  // optional
      st = napi_get_named_property(env, opts, "device", &js_device);
      if (st != napi_ok) js_device = nullptr;  // optional
    }
  }

  auto stype_opt = ParseDType(env, js_dtype);
  if (!stype_opt.has_value()) return nullptr;  // error already set
  auto dev_opt = ParseDeviceCpuOnly(env, js_device);
  if (!dev_opt.has_value()) return nullptr;  // error already set

  std::size_t nbytes = 0;
  if (!NumelBytes(sizes, *stype_opt, &nbytes)) {
    return ThrowTypeError(env, "zeros: numel overflow");
  }

  if (nbytes > (static_cast<std::size_t>(1) << 31)) {
    return ThrowRuntimeError(env, "zeros: requested allocation too large");
  }

  try {
    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
    if (nbytes > 0) std::memset(storage->data(), 0, nbytes);

    auto strides = make_contig_strides(sizes);
    vbt::core::TensorImpl impl(storage, sizes, strides, /*storage_offset=*/0,
                               *stype_opt, *dev_opt);

    return WrapTensorImplAsJsTensor(env, std::move(impl));
  } catch (const std::bad_alloc&) {
    return ThrowRuntimeError(env, "zeros: allocation failed");
  } catch (const std::exception&) {
    return ThrowRuntimeError(env, "zeros: internal error");
  } catch (...) {
    return ThrowRuntimeError(env, "zeros: unknown internal error");
  }
}

static napi_value MakeCpuScalarTensor(napi_env env,
                                      vbt::core::ScalarType dtype,
                                      const void* src_bytes,
                                      std::size_t nbytes) {
  try {
    auto storage = vbt::cpu::new_cpu_storage(nbytes, /*pinned=*/false);
    if (nbytes > 0 && src_bytes != nullptr) {
      std::memcpy(storage->data(), src_bytes, nbytes);
    }
    vbt::core::TensorImpl impl(storage,
                               /*sizes=*/{},
                               /*strides=*/{},
                               /*storage_offset=*/0,
                               dtype,
                               vbt::core::Device::cpu());
    return WrapTensorImplAsJsTensor(env, std::move(impl));
  } catch (const std::bad_alloc&) {
    return ThrowRuntimeError(env, "scalar: allocation failed");
  } catch (const std::exception&) {
    return ThrowRuntimeError(env, "scalar: internal error");
  } catch (...) {
    return ThrowRuntimeError(env, "scalar: unknown internal error");
  }
}

napi_value ScalarInt64(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "ScalarInt64/get_cb_info");

  if (argc != 1) {
    return ThrowTypeError(env, "scalarInt64(value) expects 1 argument");
  }

  std::int64_t v = 0;
  st = napi_get_value_int64(env, args[0], &v);
  if (st != napi_ok) {
    return ThrowTypeError(env, "scalarInt64: expected a finite int64 number");
  }

  const long long vv = static_cast<long long>(v);
  return MakeCpuScalarTensor(env,
                             vbt::core::ScalarType::Int64,
                             &vv,
                             sizeof(vv));
}

napi_value ScalarBool(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "ScalarBool/get_cb_info");

  if (argc != 1) {
    return ThrowTypeError(env, "scalarBool(value) expects 1 argument");
  }

  bool v = false;
  st = napi_get_value_bool(env, args[0], &v);
  if (st != napi_ok) {
    return ThrowTypeError(env, "scalarBool: expected a boolean");
  }

  const bool vv = v;
  return MakeCpuScalarTensor(env,
                             vbt::core::ScalarType::Bool,
                             &vv,
                             sizeof(vv));
}

napi_value ScalarFloat32(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "ScalarFloat32/get_cb_info");

  if (argc != 1) {
    return ThrowTypeError(env, "scalarFloat32(value) expects 1 argument");
  }

  double v = 0.0;
  st = napi_get_value_double(env, args[0], &v);
  if (st != napi_ok) {
    return ThrowTypeError(env, "scalarFloat32: expected a finite number");
  }

  const float vv = static_cast<float>(v);
  return MakeCpuScalarTensor(env,
                             vbt::core::ScalarType::Float32,
                             &vv,
                             sizeof(vv));
}

}  // namespace vbt::node
