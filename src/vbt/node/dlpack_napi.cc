// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/dlpack_napi.h"

#include <node_api.h>
#include <dlpack/dlpack.h>

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <cmath>
#include <vector>

#include "vbt/interop/dlpack.h"
#include "vbt/core/tensor.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/stream.h"
#include "vbt/node/dispatcher.h"
#include "vbt/node/errors.h"
#include "vbt/node/tensor.h"
#include "vbt/node/util.h"
#include "vbt/node/logging.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace vbt {
namespace node {

namespace {

using vbt::core::TensorImpl;
using vbt::interop::from_dlpack;
using vbt::interop::to_dlpack;
#if VBT_WITH_CUDA
using vbt::interop::from_dlpack_cuda_copy;
#endif

using ::DLManagedTensor;

#if VBT_WITH_CUDA
inline bool cuda_available_globally() noexcept {
  try {
    return vbt::cuda::device_count() > 0;
  } catch (...) {
    return false;
  }
}
#else
inline bool cuda_available_globally() noexcept { return false; }
#endif

// Finalizer for DlpackCapsule JS objects. Ensures the provider's deleter runs
// exactly once on capsules that were never imported or that hit an error
// before ownership was transferred to TensorImpl.
void DlpackCapsuleFinalizer(napi_env /*env*/, void* data, void* /*hint*/) {
  auto* owner = static_cast<NodeDlpackOwner*>(data);
  if (!owner) return;

  DLManagedTensor* mt = nullptr;
  {
    std::lock_guard<std::mutex> lock(owner->mu);
    if ((owner->state == DlpackState::kNew ||
         owner->state == DlpackState::kErrored) &&
        owner->mt != nullptr) {
      mt = owner->mt;
      owner->mt = nullptr;
      owner->state = DlpackState::kConsumed;
    }
  }

  if (mt != nullptr) {
    auto* del = std::exchange(mt->deleter, nullptr);
    if (del != nullptr) {
      try {
        del(mt);
      } catch (...) {
        // Swallow provider exceptions in finalizers.
      }
    }
  }

  delete owner;
}

// Helper used by importer and finalizer to consume a capsule on error paths.
// It transitions the owner to kErrored and invokes the provider deleter once.
void ConsumeCapsuleOnError(NodeDlpackOwner* owner, DLManagedTensor* mt) {
  if (!owner || !mt) return;

  {
    std::lock_guard<std::mutex> lock(owner->mu);
    if (owner->mt != mt) {
      // Either already cleared by another path, or never owned.
      return;
    }
    owner->mt = nullptr;
    owner->state = DlpackState::kErrored;
  }

  auto* del = std::exchange(mt->deleter, nullptr);
  if (del != nullptr) {
    try {
      del(mt);
    } catch (...) {
      // Best-effort only; deleter exceptions must not escape to JS.
    }
  }
}

// Mark a DLPack capsule as errored after a core importer exception.
//
// Once the core importer starts running, it may invoke the provider deleter on
// failures (and the deleter may free `mt`). Therefore, this helper must not
// dereference `mt`.
void MarkCapsuleErroredAfterCore(NodeDlpackOwner* owner, DLManagedTensor* mt) {
  if (!owner) return;
  std::lock_guard<std::mutex> lock(owner->mu);
  if (owner->mt != mt) {
    // Either already cleared by another path, or never owned.
    return;
  }
  owner->mt = nullptr;
  owner->state = DlpackState::kErrored;
}

// Unwrap a DlpackCapsule JS object into a NodeDlpackOwner*.
bool UnwrapDlpackCapsule(napi_env env,
                         napi_value js_capsule,
                         NodeDlpackOwner** out_owner) {
  using vbt::node::MakeErrorWithCode;

  if (!out_owner) return false;
  void* raw = nullptr;
  napi_status st = napi_unwrap(env, js_capsule, &raw);
  if (st != napi_ok || raw == nullptr) {
    napi_value err = MakeErrorWithCode(
        env,
        "fromDlpack: expected DlpackCapsule handle",
        "EINVAL",
        /*type_error=*/true);
    napi_throw(env, err);
    return false;
  }

  *out_owner = static_cast<NodeDlpackOwner*>(raw);
  return true;
}

// Wrap a freshly allocated NodeDlpackOwner* into a JS object using
// DlpackCapsuleFinalizer. On failure, deletes the owner and sets a JS error.
napi_value WrapDlpackCapsule(napi_env env, NodeDlpackOwner* owner) {
  if (!owner) return nullptr;

  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  if (st != napi_ok) {
    delete owner;
    vbt::node::CheckNapiOkImpl(env, st, "WrapDlpackCapsule/create_object");
    return nullptr;
  }

  st = napi_wrap(env, obj, owner, DlpackCapsuleFinalizer, nullptr, nullptr);
  if (st != napi_ok) {
    delete owner;
    vbt::node::CheckNapiOkImpl(env, st, "WrapDlpackCapsule/wrap");
    return nullptr;
  }

  return obj;
}

enum class DlpackErrorKind {
  None,
  InvalidArg,     // std::invalid_argument
  Runtime,        // std::runtime_error and other runtime failures
  BadAlloc,       // std::bad_alloc
  AsyncWorkFailure,
  CudaUnavailable, // CUDA not built or globally unavailable
};

struct DlpackImportJob {
  NodeDlpackOwner* owner{nullptr};
  DLManagedTensor* mt{nullptr};    // Snapshot of owner->mt while owned
  ImportOpts       opts{};

  bool             success{false};
  TensorImpl       output;
  DlpackErrorKind  error_kind{DlpackErrorKind::None};
  std::string      error_message;  // short, sanitized description

  napi_deferred    deferred{nullptr};
  napi_async_work  work{nullptr};
  napi_ref         capsule_ref{nullptr}; // prevents GC during async job
};

// Any N-API failure results in a thrown error and returns false.
bool ParseImportOpts(napi_env env, napi_value js_opts, ImportOpts* out_opts) {
  using vbt::node::ThrowErrorWithCode;

  if (!out_opts) return false;
  *out_opts = ImportOpts{};
  if (!js_opts) return true;

  napi_valuetype t;
  napi_status st = napi_typeof(env, js_opts, &t);
  if (!vbt::node::CheckNapiOkImpl(env, st,
                                  "FromDlpackCapsuleAsync/opts_typeof")) {
    return false;
  }

  if (t == napi_undefined || t == napi_null) return true;

  if (t != napi_object) {
    ThrowErrorWithCode(env,
                       "fromDlpack: options must be an object",
                       "EINVAL",
                       /*type_error=*/true);
    return false;
  }

  // copy
  napi_value v_copy;
  st = napi_get_named_property(env, js_opts, "copy", &v_copy);
  if (st == napi_ok) {
    napi_valuetype t_copy;
    st = napi_typeof(env, v_copy, &t_copy);
    if (!vbt::node::CheckNapiOkImpl(
            env, st,
            "FromDlpackCapsuleAsync/opts_copy_typeof")) {
      return false;
    }
    if (t_copy == napi_boolean) {
      bool b = false;
      st = napi_get_value_bool(env, v_copy, &b);
      if (!vbt::node::CheckNapiOkImpl(
              env, st,
              "FromDlpackCapsuleAsync/opts_copy_get")) {
        return false;
      }
      out_opts->has_copy = true;
      out_opts->copy = b;
    }
  } else if (st != napi_invalid_arg) {
    if (!vbt::node::CheckNapiOkImpl(env, st,
                                    "FromDlpackCapsuleAsync/opts_copy")) {
      return false;
    }
  }

  // device
  napi_value v_device;
  st = napi_get_named_property(env, js_opts, "device", &v_device);
  if (st == napi_ok) {
    napi_valuetype t_dev;
    st = napi_typeof(env, v_device, &t_dev);
    if (!vbt::node::CheckNapiOkImpl(
            env, st,
            "FromDlpackCapsuleAsync/opts_device_typeof")) {
      return false;
    }
    if (t_dev == napi_undefined || t_dev == napi_null) {
      // No explicit target device override.
      return true;
    }
    if (t_dev != napi_number) {
      ThrowErrorWithCode(env,
                         "dlpack.fromDlpack: device must be a finite number or undefined",
                         "EINVAL",
                         /*type_error=*/true);
      return false;
    }

    double dv = 0.0;
    st = napi_get_value_double(env, v_device, &dv);
    if (!vbt::node::CheckNapiOkImpl(
            env, st,
            "FromDlpackCapsuleAsync/opts_device_get_value")) {
      return false;
    }

    if (!std::isfinite(dv)) {
      ThrowErrorWithCode(env,
                         "dlpack.fromDlpack: device must be a finite number or undefined",
                         "EINVAL",
                         /*type_error=*/true);
      return false;
    }

    long long ll = static_cast<long long>(dv);
    if (static_cast<double>(ll) != dv || ll < 0) {
      ThrowErrorWithCode(env,
                         "dlpack.fromDlpack: device index must be a non-negative integer",
                         "EINVAL",
                         /*type_error=*/true);
      return false;
    }

    out_opts->has_target_device = true;
    out_opts->target_device_id = static_cast<int>(ll);
  } else if (st != napi_invalid_arg) {
    if (!vbt::node::CheckNapiOkImpl(env, st,
                                    "FromDlpackCapsuleAsync/opts_device")) {
      return false;
    }
  }

  return true;
}

// Helper: standard contiguous row-major strides for a shape.
std::vector<int64_t> make_contig_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i =
           static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    st[idx] = acc;
    const int64_t dim = sizes[idx];
    if (dim != 0) {
      // Best-effort overflow handling; mirror tensor.cc behavior.
      const int64_t next = acc * dim;
      acc = next;
    }
  }
  return st;
}

// CPU-only import helper.
void execute_cpu_alias_import(DlpackImportJob* job) {
  DLManagedTensor* mt = job->mt;
  if (!mt) {
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "null DLManagedTensor in job";
    return;
  }

  try {
    job->output = from_dlpack(mt);
    // Success: core importer steals deleter into DataPtr.
    {
      std::lock_guard<std::mutex> lock(job->owner->mu);
      job->owner->mt = nullptr;
      job->owner->state = DlpackState::kConsumed;
    }
    job->mt = nullptr;
    job->success = true;
  } catch (const std::invalid_argument& e) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::InvalidArg;
    job->error_message = e.what() ? e.what() : "invalid argument";
  } catch (const std::bad_alloc& e) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::BadAlloc;
    job->error_message = e.what() ? e.what() : "allocation failed";
  } catch (const std::runtime_error& e) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "runtime error";
  } catch (...) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "unknown DLPack import error";
  }
}

#if VBT_WITH_CUDA
// CPU→CUDA helper used for kDLCPU:0 capsules with CUDA targets.
void execute_cpu_to_cuda_import(DlpackImportJob* job, int target_device) {
  DLManagedTensor* mt = job->mt;
  if (!mt) {
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "null DLManagedTensor in job";
    return;
  }

  // Validate target device index against CUDA device count. At this point,
  // CUDA is known to be globally available, so bad indices should surface as
  // EINVAL rather than generic ERUNTIME.
  int device_count = vbt::cuda::device_count();
  if (device_count <= 0 || target_device < 0 || target_device >= device_count) {
    // Treat invalid device indices as immediate argument errors while still
    // honoring DLPack deleter invariants: once FromDlpackCapsuleAsync has
    // taken ownership (state == kImported), any error after that must either
    // transfer the deleter into TensorImpl or consume the capsule via
    // ConsumeCapsuleOnError.
    ConsumeCapsuleOnError(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::InvalidArg;
    job->error_message = "CPU->CUDA import: device index out of range";
    return;
  }

  TensorImpl cpu_tensor;
  try {
    cpu_tensor = from_dlpack(mt);
    {
      std::lock_guard<std::mutex> lock(job->owner->mu);
      job->owner->mt = nullptr;
      job->owner->state = DlpackState::kConsumed;
    }
    job->mt = nullptr;
  } catch (const std::invalid_argument& e) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::InvalidArg;
    job->error_message = e.what() ? e.what() : "invalid argument";
    return;
  } catch (const std::bad_alloc& e) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::BadAlloc;
    job->error_message = e.what() ? e.what() : "allocation failed";
    return;
  } catch (const std::runtime_error& e) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "runtime error";
    return;
  } catch (...) {
    MarkCapsuleErroredAfterCore(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "unknown DLPack import error";
    return;
  }

  try {
    if (!cpu_tensor.is_non_overlapping_and_dense()) {
      job->success = false;
      job->error_kind = DlpackErrorKind::InvalidArg;
      job->error_message =
          "CPU->CUDA import requires dense-contiguous tensors";
      return;
    }

    const std::size_t nbytes =
        static_cast<std::size_t>(cpu_tensor.itemsize()) *
        static_cast<std::size_t>(cpu_tensor.numel());

    int prev_dev = 0;
    cudaError_t st = cudaGetDevice(&prev_dev);
    if (st != cudaSuccess) {
      job->success = false;
      job->error_kind = DlpackErrorKind::Runtime;
      job->error_message = "CPU->CUDA import: cudaGetDevice failed";
      return;
    }

    if (target_device != prev_dev) {
      st = cudaSetDevice(target_device);
      if (st != cudaSuccess) {
        job->success = false;
        job->error_kind = DlpackErrorKind::Runtime;
        job->error_message = "CPU->CUDA import: cudaSetDevice failed";
        return;
      }
    }

    auto storage = vbt::cuda::new_cuda_storage(nbytes, target_device);
    auto stream = vbt::cuda::getCurrentStream(
        static_cast<vbt::cuda::DeviceIndex>(target_device));
    vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get(
        static_cast<vbt::cuda::DeviceIndex>(target_device));

    if (nbytes > 0) {
      cudaError_t st2 = alloc.memcpyAsync(
          storage->data(), target_device,
          cpu_tensor.data(), -1 /* host */, nbytes,
          stream, /*p2p_enabled=*/false);
      if (st2 != cudaSuccess) {
        const char* msg = cudaGetErrorString(st2);
        job->success = false;
        job->error_kind = DlpackErrorKind::Runtime;
        job->error_message =
            std::string("CPU->CUDA import: memcpyAsync failed: ") +
            (msg ? msg : "");
        if (target_device != prev_dev) {
          (void)cudaSetDevice(prev_dev);
        }
        return;
      }
      st2 = cudaStreamSynchronize(
          reinterpret_cast<cudaStream_t>(stream.handle()));
      if (st2 != cudaSuccess) {
        const char* msg = cudaGetErrorString(st2);
        job->success = false;
        job->error_kind = DlpackErrorKind::Runtime;
        job->error_message =
            std::string("CPU->CUDA import: cudaStreamSynchronize failed: ") +
            (msg ? msg : "");
        if (target_device != prev_dev) {
          (void)cudaSetDevice(prev_dev);
        }
        return;
      }
    }

    std::vector<int64_t> sizes = cpu_tensor.sizes();
    auto strides = make_contig_strides(sizes);
    job->output = TensorImpl(
        storage, std::move(sizes), std::move(strides),
        /*storage_offset=*/0,
        cpu_tensor.dtype(),
        vbt::core::Device::cuda(target_device));
    job->success = true;

    if (target_device != prev_dev) {
      (void)cudaSetDevice(prev_dev);
    }
  } catch (const std::bad_alloc& e) {
    job->success = false;
    job->error_kind = DlpackErrorKind::BadAlloc;
    job->error_message = e.what() ? e.what() : "allocation failed";
  } catch (const std::runtime_error& e) {
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "runtime error";
  } catch (const std::exception& e) {
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "error";
  } catch (...) {
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "unknown CPU->CUDA import error";
  }
}

// CUDA alias/copy helpers for CUDA-family capsules.
void execute_cuda_alias_import(DlpackImportJob* job, int dev_id) {
  DLManagedTensor* mt = job->mt;
  bool core_started = false;

  auto on_error = [&]() {
    if (core_started) {
      MarkCapsuleErroredAfterCore(job->owner, mt);
    } else {
      ConsumeCapsuleOnError(job->owner, mt);
    }
  };

  try {
    int prev_dev = 0;
    cudaError_t st = cudaGetDevice(&prev_dev);
    if (st != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice failed");
    }
    if (dev_id != prev_dev) {
      st = cudaSetDevice(dev_id);
      if (st != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed");
      }
    }

    core_started = true;
    job->output = from_dlpack(mt);
    {
      std::lock_guard<std::mutex> lock(job->owner->mu);
      job->owner->mt = nullptr;
      job->owner->state = DlpackState::kConsumed;
    }
    job->mt = nullptr;
    job->success = true;

    if (dev_id != prev_dev) {
      (void)cudaSetDevice(prev_dev);
    }
  } catch (const std::invalid_argument& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::InvalidArg;
    job->error_message = e.what() ? e.what() : "invalid argument";
  } catch (const std::bad_alloc& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::BadAlloc;
    job->error_message = e.what() ? e.what() : "allocation failed";
  } catch (const std::runtime_error& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "runtime error";
  } catch (const std::exception& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "error";
  } catch (...) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "unknown DLPack import error";
  }
}

void execute_cuda_copy_import(DlpackImportJob* job, int dev_id) {
  DLManagedTensor* mt = job->mt;
  bool core_started = false;

  auto on_error = [&]() {
    if (core_started) {
      MarkCapsuleErroredAfterCore(job->owner, mt);
    } else {
      ConsumeCapsuleOnError(job->owner, mt);
    }
  };

  try {
    int prev_dev = 0;
    cudaError_t st = cudaGetDevice(&prev_dev);
    if (st != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice failed");
    }
    if (dev_id != prev_dev) {
      st = cudaSetDevice(dev_id);
      if (st != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed");
      }
    }

    core_started = true;
    job->output = from_dlpack_cuda_copy(mt);
    {
      std::lock_guard<std::mutex> lock(job->owner->mu);
      job->owner->mt = nullptr;
      job->owner->state = DlpackState::kConsumed;
    }
    job->mt = nullptr;
    job->success = true;

    if (dev_id != prev_dev) {
      (void)cudaSetDevice(prev_dev);
    }
  } catch (const std::invalid_argument& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::InvalidArg;
    job->error_message = e.what() ? e.what() : "invalid argument";
  } catch (const std::bad_alloc& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::BadAlloc;
    job->error_message = e.what() ? e.what() : "allocation failed";
  } catch (const std::runtime_error& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "runtime error";
  } catch (const std::exception& e) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = e.what() ? e.what() : "error";
  } catch (...) {
    on_error();
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "unknown DLPack import error";
  }
}
#endif  // VBT_WITH_CUDA

// Worker callback: performs CPU and CUDA DLPack import on a libuv worker thread.
void ExecuteDlpackImport(napi_env /*env*/, void* data) {
  auto* job = static_cast<DlpackImportJob*>(data);
  DLManagedTensor* mt = job->mt;

  if (!mt) {
    // Defensive: this should be unreachable if the main-thread setup logic
    // always sets job->mt for imported capsules. Fail closed by attempting to
    // consume any live capsule owned by the NodeDlpackOwner.
    if (job->owner) {
      ConsumeCapsuleOnError(job->owner, job->owner->mt);
    }
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::Runtime;
    job->error_message = "null DLManagedTensor in job";
    return;
  }

  const DLDevice dev = mt->dl_tensor.device;
  const int dev_type = dev.device_type;
  const int dev_id = dev.device_id;

  // Case 1: CPU sources.
  if (dev_type == kDLCPU) {
    if (dev_id == 0) {
      if (!job->opts.has_target_device || job->opts.target_device_id == 0) {
        execute_cpu_alias_import(job);
        return;
      }
#if !VBT_WITH_CUDA
      ConsumeCapsuleOnError(job->owner, mt);
      job->mt = nullptr;
      job->success = false;
      job->error_kind = DlpackErrorKind::CudaUnavailable;
      job->error_message =
          "CUDA DLPack import requested but CUDA support is not built";
      return;
#else
      if (!cuda_available_globally()) {
        ConsumeCapsuleOnError(job->owner, mt);
        job->mt = nullptr;
        job->success = false;
        job->error_kind = DlpackErrorKind::CudaUnavailable;
        job->error_message =
            "CUDA DLPack import requested but CUDA is unavailable";
        return;
      }
      execute_cpu_to_cuda_import(job, job->opts.target_device_id);
      return;
#endif
    }

    // Exotic CPU devices (kDLCPU with dev_id != 0) are not supported in Node.
    ConsumeCapsuleOnError(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::InvalidArg;
    job->error_message =
        "unsupported CPU device_id for Node DLPack import (expected kDLCPU:0)";
    return;
  }

  // Case 2+: Non-CPU sources require CUDA.
#if !VBT_WITH_CUDA
  ConsumeCapsuleOnError(job->owner, mt);
  job->mt = nullptr;
  job->success = false;
  job->error_kind = DlpackErrorKind::CudaUnavailable;
  job->error_message =
      "CUDA DLPack import requested but CUDA support is not built";
  return;
#else
  if (!cuda_available_globally()) {
    ConsumeCapsuleOnError(job->owner, mt);
    job->mt = nullptr;
    job->success = false;
    job->error_kind = DlpackErrorKind::CudaUnavailable;
    job->error_message =
        "CUDA DLPack import requested but CUDA is unavailable";
    return;
  }

  const int target = job->opts.has_target_device
                         ? job->opts.target_device_id
                         : dev_id;  // default to provider device

  if (dev_type == kDLCUDA || dev_type == kDLCUDAManaged) {
    if (target != dev_id) {
      ConsumeCapsuleOnError(job->owner, mt);
      job->mt = nullptr;
      job->success = false;
      job->error_kind = DlpackErrorKind::Runtime;
      job->error_message =
          "cross-GPU DLPack import not supported";
      return;
    }

    const bool do_copy =
        job->opts.has_copy ? job->opts.copy : true;  // default copy
    if (!do_copy) {
      execute_cuda_alias_import(job, dev_id);
    } else {
      execute_cuda_copy_import(job, dev_id);
    }
    return;
  }

  if (dev_type == kDLCUDAHost) {
    const int host_target = target;  // default to dev_id when !has_target_device
    execute_cuda_copy_import(job, host_target);
    return;
  }

  // Unsupported device types.
  ConsumeCapsuleOnError(job->owner, mt);
  job->mt = nullptr;
  job->success = false;
  job->error_kind = DlpackErrorKind::InvalidArg;
  job->error_message =
      "unsupported DLPack device type for Node CUDA import";
#endif  // VBT_WITH_CUDA
}

// Complete callback: runs on the main thread and resolves/rejects the Promise.
void CompleteDlpackImport(napi_env env, napi_status status, void* data) {
  auto* job = static_cast<DlpackImportJob*>(data);

  // Decrement inflight counter first.
  g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

  if (job->capsule_ref) {
    napi_delete_reference(env, job->capsule_ref);
    job->capsule_ref = nullptr;
  }
  if (job->work) {
    napi_delete_async_work(env, job->work);
    job->work = nullptr;
  }

  if (status != napi_ok) {
    if (job->mt != nullptr) {
      // Worker may not have run; we still own the capsule.
      ConsumeCapsuleOnError(job->owner, job->mt);
      job->mt = nullptr;
    }
    job->success = false;
    job->error_kind = DlpackErrorKind::AsyncWorkFailure;
    job->error_message = "async worker failed";
  }

  if (job->success) {
    // Wrap TensorImpl into JsTensor and JS Tensor handle.
    napi_value js_tensor = nullptr;
    if (!TryWrapTensorImplAsJsTensor(env, std::move(job->output), &js_tensor)) {
      napi_value err = vbt::node::MakeErrorWithCode(
          env,
          "fromDlpack: failed to wrap tensor result",
          "ERUNTIME",
          /*type_error=*/false);
      napi_reject_deferred(env, job->deferred, err);
    } else {
      napi_resolve_deferred(env, job->deferred, js_tensor);
    }
  } else {
    const char* code = "ERUNTIME";
    bool type_err = false;
    switch (job->error_kind) {
      case DlpackErrorKind::InvalidArg:
        code = "EINVAL";
        type_err = true;
        break;
      case DlpackErrorKind::BadAlloc:
        code = "EOOM";
        break;
      case DlpackErrorKind::CudaUnavailable:
        code = "ENOCUDA";
        break;
      case DlpackErrorKind::AsyncWorkFailure:
      case DlpackErrorKind::Runtime:
      case DlpackErrorKind::None:
      default:
        code = "ERUNTIME";
        break;
    }

    napi_value err = vbt::node::MakeErrorWithCode(
        env,
        std::string("fromDlpack: ") + job->error_message,
        code,
        type_err);
    napi_reject_deferred(env, job->deferred, err);
  }

  delete job;
}

}  // namespace

napi_value ToDlpack(napi_env env, napi_callback_info info) {
  using vbt::node::MakeErrorWithCode;

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "ToDlpack/get_cb_info");

  if (argc < 1) {
    napi_value err = MakeErrorWithCode(
        env,
        "toDlpack: tensor argument is required",
        "EINVAL",
        /*type_error=*/true);
    napi_throw(env, err);
    return nullptr;
  }

  JsTensor* jt = nullptr;
  if (!UnwrapJsTensor(env, args[0], &jt)) {
    // UnwrapJsTensor throws TypeError(EINVAL) for forged objects.
    return nullptr;
  }

  try {
    struct DlpackGuard {
      void operator()(DLManagedTensor* mt) const {
        if (mt && mt->deleter) {
          try {
            mt->deleter(mt);
          } catch (...) {
            // Best-effort only; provider exceptions must not escape to JS.
          }
        }
      }
    };

    std::unique_ptr<DLManagedTensor, DlpackGuard> guard(
        to_dlpack(jt->impl).release());

    // Allocate owner and wrap capsule before transferring ownership of mt.
    auto* owner = new NodeDlpackOwner{};
    {
      std::lock_guard<std::mutex> lock(owner->mu);
      owner->state = DlpackState::kNew;
      owner->mt = nullptr;  // set after wrap succeeds
    }

    napi_value cap = WrapDlpackCapsule(env, owner);
    if (cap == nullptr) {
      // WrapDlpackCapsule already deleted owner and set an exception.
      return nullptr;
    }

    {
      std::lock_guard<std::mutex> lock(owner->mu);
      owner->mt = guard.release();  // owner now controls mt and its deleter
    }

    return cap;
  } catch (const std::invalid_argument& e) {
    napi_value err = MakeErrorWithCode(
        env,
        e.what() ? e.what() : "toDlpack: invalid argument",
        "EINVAL",
        /*type_error=*/true);
    napi_throw(env, err);
    return nullptr;
  } catch (const std::bad_alloc& e) {
    napi_value err = MakeErrorWithCode(
        env,
        e.what() ? e.what() : "toDlpack: allocation failed",
        "EOOM",
        /*type_error=*/false);
    napi_throw(env, err);
    return nullptr;
  } catch (const std::runtime_error& e) {
    napi_value err = MakeErrorWithCode(
        env,
        e.what() ? e.what() : "toDlpack: runtime error",
        "ERUNTIME",
        /*type_error=*/false);
    napi_throw(env, err);
    return nullptr;
  } catch (...) {
    napi_value err = MakeErrorWithCode(
        env,
        "toDlpack: unknown internal error",
        "EUNKNOWN",
        /*type_error=*/false);
    napi_throw(env, err);
    return nullptr;
  }
}

napi_value FromDlpackCapsuleAsync(napi_env env, napi_callback_info info) {
  using vbt::node::ThrowErrorWithCode;

  // Ensure env config (VBT_NODE_MAX_INFLIGHT_OPS) has been read.
  (void)max_inflight_ops();

  size_t argc = 2;
  napi_value args[2];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "FromDlpackCapsuleAsync/get_cb_info");

  if (argc < 1) {
    ThrowErrorWithCode(env,
                       "fromDlpack: capsule argument is required",
                       "EINVAL",
                       /*type_error=*/true);
    return nullptr;
  }

  napi_value js_capsule = args[0];
  napi_value js_opts = (argc >= 2) ? args[1] : nullptr;

  NodeDlpackOwner* owner = nullptr;
  if (!UnwrapDlpackCapsule(env, js_capsule, &owner)) {
    return nullptr;  // TypeError(EINVAL) already thrown
  }

  auto* job = new DlpackImportJob();
  job->owner = owner;

  // Take logical ownership of the capsule under mutex.
  {
    std::lock_guard<std::mutex> lock(owner->mu);

    if (owner->state != DlpackState::kNew || owner->mt == nullptr) {
      delete job;
      ThrowErrorWithCode(env,
                         "fromDlpack: capsule already consumed",
                         "ERUNTIME",
                         /*type_error=*/false);
      return nullptr;
    }

    job->mt = owner->mt;
    owner->state = DlpackState::kImported;  // one-shot from this point
  }

  // Enforce inflight cap shared with dispatcher and CUDA runtime.
  const std::uint32_t prev =
      g_inflight_ops.fetch_add(1, std::memory_order_acq_rel);
  const std::uint32_t current = prev + 1;
  if (current > g_max_inflight_ops) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    ConsumeCapsuleOnError(job->owner, job->mt);
    job->mt = nullptr;
    delete job;
    LogIfEnabled(LogLevel::kWarn,
                 LogCategory::kDlpack,
                 "fromDlpack: too many inflight jobs",
                 {{"source", "capsule"}});
    ThrowErrorWithCode(
        env,
        "fromDlpack: too many inflight jobs (see VBT_NODE_MAX_INFLIGHT_OPS)",
        "ERUNTIME",
        /*type_error=*/false);
    return nullptr;
  }

  UpdateAsyncPeakInflight(current);

  napi_value promise;
  st = napi_create_promise(env, &job->deferred, &promise);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    napi_fatal_error("vbt.node.FromDlpackCapsuleAsync", NAPI_AUTO_LENGTH,
                     "napi_create_promise failed", NAPI_AUTO_LENGTH);
    return nullptr;  // unreachable
  }

  // Parse ImportOpts (best-effort).
  if (js_opts) {
    if (!ParseImportOpts(env, js_opts, &job->opts)) {
      g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
      ConsumeCapsuleOnError(job->owner, job->mt);
      job->mt = nullptr;
      delete job;
      return nullptr;  // ParseImportOpts already set an error.
    }
  }

  // Hold a reference to the capsule to prevent its finalizer from running
  // while the async job is in flight.
  if (napi_create_reference(env, js_capsule, 1, &job->capsule_ref) !=
      napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    ConsumeCapsuleOnError(job->owner, job->mt);
    job->mt = nullptr;
    delete job;
    ThrowErrorWithCode(env,
                       "fromDlpack: failed to capture capsule reference",
                       "ERUNTIME",
                       /*type_error=*/false);
    return promise;  // Promise is created but an exception is pending.
  }

  napi_value resource_name;
  st = napi_create_string_utf8(env,
                               "vbt_fromDlpackCapsule",
                               NAPI_AUTO_LENGTH,
                               &resource_name);
  if (!vbt::node::CheckNapiOkImpl(
          env, st,
          "FromDlpackCapsuleAsync/create_resource_name")) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    napi_delete_reference(env, job->capsule_ref);
    job->capsule_ref = nullptr;
    ConsumeCapsuleOnError(job->owner, job->mt);
    job->mt = nullptr;
    delete job;
    return nullptr;
  }

  st = napi_create_async_work(env,
                              /*async_resource=*/nullptr,
                              resource_name,
                              ExecuteDlpackImport,
                              CompleteDlpackImport,
                              job,
                              &job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    napi_delete_reference(env, job->capsule_ref);
    job->capsule_ref = nullptr;
    ConsumeCapsuleOnError(job->owner, job->mt);
    job->mt = nullptr;
    delete job;
    CHECK_NAPI_OK(env, st,
                  "FromDlpackCapsuleAsync/create_async_work");
    return promise;  // CHECK_NAPI_OK already threw.
  }

  st = napi_queue_async_work(env, job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    napi_delete_reference(env, job->capsule_ref);
    job->capsule_ref = nullptr;
    if (job->work) {
      napi_delete_async_work(env, job->work);
      job->work = nullptr;
    }
    ConsumeCapsuleOnError(job->owner, job->mt);
    job->mt = nullptr;
    delete job;
    CHECK_NAPI_OK(env, st,
                  "FromDlpackCapsuleAsync/queue_async_work");
    return promise;
  }

  LogIfEnabled(LogLevel::kInfo,
               LogCategory::kDlpack,
               "fromDlpack: scheduled import",
               {{"source", "capsule"}});

  return promise;
}

napi_value RegisterDlpackBindings(napi_env env, napi_value exports) {
  napi_property_descriptor props[] = {
      {"toDlpack", nullptr, ToDlpack, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"fromDlpackCapsule", nullptr, FromDlpackCapsuleAsync, nullptr, nullptr,
       nullptr, napi_default, nullptr},
  };

  napi_status st = napi_define_properties(
      env, exports, sizeof(props) / sizeof(props[0]), props);
  CHECK_NAPI_OK(env, st, "RegisterDlpackBindings/define_properties");

  return exports;
}

}  // namespace node
}  // namespace vbt
