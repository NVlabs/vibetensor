// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/interop/dlpack.h"

#include <memory>
#include <new>
#include <cstring>
#include <limits>
#include <atomic>
#include <utility>

#include <dlpack/dlpack.h>
#ifndef kDLGPU
#define kDLGPU kDLCUDA
#endif

#include "vbt/core/checked_math.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/intrusive_ptr.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include "vbt/cuda/stream.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/allocator.h"
#endif

namespace vbt {
namespace interop {

namespace {
struct ManagerCtx {
  vbt::core::StoragePtr storage; // keepalive
  std::atomic<bool> freed{false};
};

struct DlpackDeleterGuard {
  DLManagedTensor* mt{nullptr};
  void (*del)(DLManagedTensor*){nullptr};
  bool active{false};

  DlpackDeleterGuard(DLManagedTensor* mt_, void (*del_)(DLManagedTensor*)) noexcept
      : mt(mt_), del(del_), active(true) {}

  DlpackDeleterGuard(const DlpackDeleterGuard&) = delete;
  DlpackDeleterGuard& operator=(const DlpackDeleterGuard&) = delete;

  ~DlpackDeleterGuard() noexcept {
    if (active && del && mt) {
      try { del(mt); } catch (...) {}
    }
  }

  void release() noexcept {
    active = false;
    mt = nullptr;
    del = nullptr;
  }

  void call_now_and_release() noexcept {
    if (active && del && mt) {
      try { del(mt); } catch (...) {}
    }
    active = false;
    mt = nullptr;
    del = nullptr;
  }
};

void idempotent_deleter(DLManagedTensor* mt) {
  if (!mt) return;
  auto* ctx = static_cast<ManagerCtx*>(mt->manager_ctx);
  if (ctx) {
    bool expected = false;
    if (ctx->freed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      if (mt->dl_tensor.shape) {
        delete[] mt->dl_tensor.shape;
        mt->dl_tensor.shape = nullptr;
      }
      if (mt->dl_tensor.strides) {
        delete[] mt->dl_tensor.strides;
        mt->dl_tensor.strides = nullptr;
      }
      delete ctx;
      delete mt;
    }
    return; // already freed → no-op
  }
  // No ctx: best-effort delete
  delete mt;
}

inline void ensure(bool cond, const char* msg) {
  if (!cond) throw std::runtime_error(msg);
}
inline std::uintptr_t checked_ptr_add(std::uintptr_t base, std::uint64_t add, bool& ok) {
  if (add > std::numeric_limits<std::uintptr_t>::max() - base) { ok = false; return 0; }
  ok = true; return base + static_cast<std::uintptr_t>(add);
}

inline std::uintptr_t checked_ptr_add_signed(std::uintptr_t base, std::int64_t off, bool& ok) {
  if (off >= 0) {
    return checked_ptr_add(base, static_cast<std::uint64_t>(off), ok);
  }
  // off is negative; avoid UB for off == INT64_MIN.
  const std::uint64_t neg = static_cast<std::uint64_t>(-(off + 1)) + 1;
  if (neg > static_cast<std::uint64_t>(base)) { ok = false; return 0; }
  ok = true; return base - static_cast<std::uintptr_t>(neg);
}

} // namespace

std::unique_ptr<DLManagedTensor, void(*)(DLManagedTensor*)>
to_dlpack(const vbt::core::TensorImpl& t) {
  using namespace vbt::core;
  // Device mapping: support CPU:0 and CUDA devices
  int dev_type = 0;
  int dev_id = 0;
  if (t.device().type == kDLCPU) {
    if (t.device().index != 0) {
      throw std::runtime_error("to_dlpack: CPU device index must be 0");
    }
    dev_type = static_cast<int>(kDLCPU);
    dev_id = 0;
  } else if (t.device().type == kDLCUDA) {
    // Allow any non-negative CUDA device index; no runtime CUDA calls here
    if (t.device().index < 0) {
      throw std::runtime_error("to_dlpack: CUDA device index must be >= 0");
    }
    dev_type = static_cast<int>(kDLCUDA);
    dev_id = static_cast<int>(t.device().index);
  } else {
    throw std::runtime_error("to_dlpack: unsupported device type");
  }

  if (t.is_conj()) {
    throw std::runtime_error(
        "to_dlpack: cannot export conjugated complex tensor; call resolve_conj()");
  }

  // Map dtype
  DLDataType dt = to_dlpack_dtype(t.dtype());
  ensure(dt.lanes == 1, "to_dlpack: lanes must be 1");
  ensure(dt.bits != 0, "to_dlpack: unsupported dtype mapping");

  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  const int64_t ndim64 = static_cast<int64_t>(sizes.size());
  ensure(ndim64 <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()), "to_dlpack: ndim exceeds int32_t");

  for (auto st : strides) {
    if (st == std::numeric_limits<int64_t>::min()) {
      throw std::runtime_error("to_dlpack: unexpected INT64_MIN stride (internal guard)");
    }
  }

  DLManagedTensor* mt = nullptr;
  ManagerCtx* ctx = nullptr;
  try {
    mt = new DLManagedTensor{};
    ctx = new ManagerCtx{};
  } catch (const std::bad_alloc&) {
    throw std::runtime_error("to_dlpack: allocation failure (shape/strides/capsule)");
  }
  mt->manager_ctx = ctx;
  ctx->storage = t.storage();
  mt->deleter = idempotent_deleter;

  DLTensor& dl = mt->dl_tensor;
  dl.device = DLDevice{.device_type = static_cast<DLDeviceType>(dev_type), .device_id = dev_id};
  dl.dtype = dt;
  dl.ndim = static_cast<int32_t>(ndim64);

  // shape
  if (ndim64 == 0) {
    dl.shape = nullptr;
  } else {
    int64_t* shape = nullptr;
    try {
      shape = new int64_t[ndim64];
    } catch (const std::bad_alloc&) {
      idempotent_deleter(mt);
      throw std::runtime_error("to_dlpack: allocation failure (shape/strides/capsule)");
    }
    for (int64_t i = 0; i < ndim64; ++i) shape[i] = sizes[static_cast<std::size_t>(i)];
    dl.shape = shape;
  }

  // strides: always materialize explicit strides for ndim>0 to maximize
  // consumer compatibility (e.g., NumPy from_dlpack writability) while
  // keeping the legacy contiguous contract when strides==nullptr only for
  // true scalars.
  if (ndim64 == 0) {
    dl.strides = nullptr;
  } else {
    int64_t* st = nullptr;
    try {
      st = new int64_t[ndim64];
    } catch (const std::bad_alloc&) {
      idempotent_deleter(mt);
      throw std::runtime_error("to_dlpack: allocation failure (shape/strides/capsule)");
    }
    for (int64_t i = 0; i < ndim64; ++i) st[i] = strides[static_cast<std::size_t>(i)];
    dl.strides = st;
  }

  // data and byte_offset
  if (t.numel() == 0) {
    dl.data = nullptr;
    dl.byte_offset = 0;
  } else {
    auto* data = static_cast<std::uint8_t*>(t.data());
    if (!data) {
      dl.data = nullptr;
    } else {
      dl.data = data - (t.itemsize() * static_cast<std::size_t>(t.storage_offset()));
    }
    const auto item_b = static_cast<int64_t>(t.itemsize());
    if (t.storage_offset() < 0) throw std::runtime_error("to_dlpack: negative storage_offset (internal)");
    auto off = static_cast<int64_t>(t.storage_offset());
    if (off > 0 && item_b > std::numeric_limits<int64_t>::max() / off) {
      idempotent_deleter(mt);
      throw std::runtime_error("to_dlpack: storage_offset byte_offset overflow");
    }
    dl.byte_offset = static_cast<uint64_t>(off * item_b);
  }

  return {mt, idempotent_deleter};
}

vbt::core::TensorImpl from_dlpack(DLManagedTensor* mt) {
  using namespace vbt::core;
  ensure(mt != nullptr, "from_dlpack: invalid ndim");

  // Enforce one-shot semantics and acquire the provider deleter up-front.
  auto* del = std::exchange(mt->deleter, nullptr);
  if (del == nullptr) throw std::runtime_error("from_dlpack: capsule already consumed");
  DlpackDeleterGuard guard(mt, del);

  const DLTensor& dl = mt->dl_tensor;
  // ndim and shape
  if (dl.ndim < 0) throw std::runtime_error("from_dlpack: invalid ndim");
  std::vector<int64_t> sizes;
  sizes.reserve(static_cast<std::size_t>(dl.ndim));
  bool has_zero = false;
  if (dl.ndim == 0) {
    // Accept either NULL or non-NULL shape/strides for scalars; treat as 0-d tensor
  } else {
    if (dl.shape == nullptr) throw std::runtime_error("from_dlpack: shape==NULL with ndim>0");
    for (int i = 0; i < dl.ndim; ++i) {
      int64_t s = dl.shape[i];
      if (s < 0) throw std::runtime_error("from_dlpack: negative dimension size");
      if (s == 0) has_zero = true;
      sizes.push_back(s);
    }
  }

  // Device
  int dev_type = dl.device.device_type;
  int dev_id = dl.device.device_id;
  Device device = Device::cpu();
  
  if (dev_type == kDLCPU) {
    if (dev_id != 0) throw std::runtime_error("from_dlpack: unsupported device, expected kDLCPU:0");
    device = Device::cpu();
  } else if (dev_type == kDLCUDA) {
#if VBT_WITH_CUDA
    int cur_dev = 0;
    if (cudaGetDevice(&cur_dev) == cudaSuccess) {
      if (dev_id != cur_dev) {
        throw std::runtime_error(std::string("Expected all tensors to be on the same device: got cuda:") +
                                 std::to_string(dev_id) + " vs cuda:" + std::to_string(cur_dev));
      }
    }
#endif
    device = Device::cuda(dev_id);
  } else {
    throw std::runtime_error("from_dlpack: unsupported device type (expected kDLCPU or kDLCUDA)");
  }

  // Mixed-device check for CUDA tensors (zero-copy path)
#if VBT_WITH_CUDA
  if (dev_type == kDLCUDA) {
    int cur_dev = 0; (void)cudaGetDevice(&cur_dev);
    if (dev_id != cur_dev) {
      throw std::runtime_error(std::string("Expected all tensors to be on the same device: got cuda:") + std::to_string(dev_id) + " vs cuda:" + std::to_string(cur_dev));
    }
  }
#endif

  // Dtype
  auto stype_opt = from_dlpack_dtype(dl.dtype);
  if (!stype_opt.has_value()) {
    throw std::runtime_error("from_dlpack: unsupported dtype (code/bits/lanes); lanes must be 1");
  }
  auto stype = stype_opt.value();
  if (stype == vbt::core::ScalarType::Float64 ||
      stype == vbt::core::ScalarType::Complex64 ||
      stype == vbt::core::ScalarType::Complex128) {
    throw std::runtime_error("from_dlpack: unsupported dtype (code/bits/lanes); lanes must be 1");
  }
  const std::size_t item_b = itemsize(stype);

  // Strides
  std::vector<int64_t> strides;
  strides.reserve(static_cast<std::size_t>(dl.ndim));
  if (dl.ndim > 0) {
    if (dl.strides == nullptr) {
      // synthesize contiguous
      int64_t acc = 1;
      for (int i = dl.ndim - 1; i >= 0; --i) {
        strides.insert(strides.begin(), acc);
        int64_t next = 0;
        if (!checked_mul_i64(acc, dl.shape[i] == 0 ? 1 : dl.shape[i], next)) {
          throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
        }
        acc = next;
      }
    } else {
      for (int i = 0; i < dl.ndim; ++i) {
        int64_t st = dl.strides[i];
        if (st == std::numeric_limits<int64_t>::min()) throw std::runtime_error("from_dlpack: INT64_MIN stride is invalid");
        strides.push_back(st);
      }
    }
  }

  // Zero-size fast path
  if (dl.ndim > 0 && has_zero) {
    if (dl.data != nullptr || dl.byte_offset != 0) {
      throw std::runtime_error("from_dlpack: zero-size requires NULL data and byte_offset=0");
    }
    // For zero-size, invoke the provider deleter immediately (no aliasing).
    guard.call_now_and_release();

    auto storage = make_intrusive<Storage>(DataPtr(nullptr, nullptr), 0);
    return TensorImpl(storage, sizes, strides, 0, stype, device);
  }

  // numel > 0 path (includes scalar)
  if (dl.ndim == 0 || !has_zero) {
    if (dl.data == nullptr) throw std::runtime_error("from_dlpack: numel>0 requires non-null data");
    // effective pointer
    bool ok = true;
    auto p_eff = checked_ptr_add(reinterpret_cast<std::uintptr_t>(dl.data), dl.byte_offset, ok);
    if (!ok) throw std::runtime_error("from_dlpack: byte_offset overflow");
    // alignment check (optional via CMake)
#if VBT_REQUIRE_DLPACK_ALIGNMENT
    if ((p_eff % static_cast<std::uintptr_t>(item_b)) != 0) {
      throw std::runtime_error("from_dlpack: effective pointer is not aligned to itemsize");
    }
#endif
    int64_t min_elem_off = 0, max_elem_off = 0;
    for (int i = 0; i < dl.ndim; ++i) {
      int64_t n = (dl.ndim == 0) ? 1 : dl.shape[i];
      int64_t d = n > 0 ? (n - 1) : 0;
      if (d == 0) continue;
      int64_t st = (dl.ndim == 0) ? 1 : strides[static_cast<std::size_t>(i)];
      int64_t term = 0;
      if (!checked_mul_i64(st, d, term)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
      if (st >= 0) {
        int64_t tmp = 0; if (!checked_add_i64(max_elem_off, term, tmp)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)"); max_elem_off = tmp;
      } else {
        int64_t tmp = 0; if (!checked_add_i64(min_elem_off, term, tmp)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)"); min_elem_off = tmp;
      }
    }
    int64_t max_plus_one = 0; if (!checked_add_i64(max_elem_off, 1, max_plus_one)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
    int64_t lower_bytes = 0; if (!checked_mul_i64(min_elem_off, static_cast<int64_t>(item_b), lower_bytes)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
    int64_t upper_excl = 0; if (!checked_mul_i64(max_plus_one, static_cast<int64_t>(item_b), upper_excl)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
    int64_t req_bytes = 0; if (!checked_add_i64(upper_excl, -lower_bytes, req_bytes)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");

    // base pointer
    auto base_ptr = checked_ptr_add_signed(p_eff, lower_bytes, ok);
    if (!ok) throw std::runtime_error("from_dlpack: pointer base overflow");
    int64_t storage_offset_elems = -min_elem_off;

    // Transfer deleter ownership into the returned Storage's DataPtr.
    vbt::core::DataPtr dp(reinterpret_cast<void*>(base_ptr),
                          [mt, del](void*) { if (del) del(mt); });
    guard.release();
    auto storage = make_intrusive<Storage>(std::move(dp), static_cast<std::size_t>(req_bytes));
    return TensorImpl(storage, sizes, strides, storage_offset_elems, stype, device);
  }

  // Unreachable
  return TensorImpl{};
}

#if VBT_WITH_CUDA
vbt::core::TensorImpl from_dlpack_cuda_copy(DLManagedTensor* mt) {
  using namespace vbt::core;
  ensure(mt != nullptr, "from_dlpack: invalid ndim");

  // Enforce one-shot semantics and acquire the provider deleter up-front.
  auto* del = std::exchange(mt->deleter, nullptr);
  if (del == nullptr) throw std::runtime_error("from_dlpack: capsule already consumed");
  DlpackDeleterGuard guard(mt, del);

  const DLTensor& dl = mt->dl_tensor;
  // Device-type check (Python binding may perform an earlier Stage A check).
  int t_dev_type = dl.device.device_type;
  if (t_dev_type != kDLCUDA && t_dev_type != kDLCUDAHost && t_dev_type != kDLCUDAManaged) {
    throw std::invalid_argument("from_dlpack: unsupported device type for CUDA import");
  }

  // dtype mapping
  auto stype_opt = from_dlpack_dtype(dl.dtype);
  if (!stype_opt.has_value()) {
    throw std::invalid_argument("from_dlpack: unsupported dtype (code/bits/lanes); lanes must be 1");
  }
  auto stype = stype_opt.value();
  if (stype == vbt::core::ScalarType::Float64 ||
      stype == vbt::core::ScalarType::Complex64 ||
      stype == vbt::core::ScalarType::Complex128) {
    throw std::invalid_argument("from_dlpack: unsupported dtype (code/bits/lanes); lanes must be 1");
  }
  const std::size_t item_b = itemsize(stype);

  // sizes and strides
  std::vector<int64_t> sizes;
  sizes.reserve(static_cast<std::size_t>(dl.ndim));
  bool has_zero = false;
  if (dl.ndim == 0) {
    // Accept either NULL or non-NULL shape/strides for scalars
  } else {
    if (dl.shape == nullptr) throw std::runtime_error("from_dlpack: shape==NULL with ndim>0");
    for (int i = 0; i < dl.ndim; ++i) {
      int64_t s = dl.shape[i];
      if (s < 0) throw std::runtime_error("from_dlpack: negative dimension size");
      if (s == 0) has_zero = true;
      sizes.push_back(s);
    }
  }
  std::vector<int64_t> strides;
  strides.reserve(static_cast<std::size_t>(dl.ndim));
  if (dl.ndim > 0) {
    if (dl.strides == nullptr) {
      int64_t acc = 1;
      for (int i = dl.ndim - 1; i >= 0; --i) {
        strides.insert(strides.begin(), acc);
        int64_t next = 0;
        if (!checked_mul_i64(acc, dl.shape[i] == 0 ? 1 : dl.shape[i], next)) {
          throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
        }
        acc = next;
      }
    } else {
      for (int i = 0; i < dl.ndim; ++i) {
        int64_t st = dl.strides[i];
        if (st == std::numeric_limits<int64_t>::min()) throw std::runtime_error("from_dlpack: INT64_MIN stride is invalid");
        strides.push_back(st);
      }
    }
  }

  // Zero-size fast path (CUDA)
  if (dl.ndim > 0 && has_zero) {
    if (dl.data != nullptr || dl.byte_offset != 0) {
      throw std::runtime_error("from_dlpack: zero-size requires NULL data and byte_offset=0");
    }
    int cur_dev = 0; (void)cudaGetDevice(&cur_dev);
    auto storage = vbt::cuda::new_cuda_storage(0, cur_dev);
    // No device work was enqueued; invoke provider deleter immediately.
    guard.call_now_and_release();
    return TensorImpl(storage, sizes, strides, 0, stype, Device::cuda(cur_dev));
  }

  if (dl.data == nullptr) {
    throw std::runtime_error("from_dlpack: numel>0 requires non-null data");
  }

  // compute span and base pointer
  bool ok = true;
  auto p_eff = checked_ptr_add(reinterpret_cast<std::uintptr_t>(dl.data), dl.byte_offset, ok);
  if (!ok) throw std::runtime_error("from_dlpack: byte_offset overflow");
#if VBT_REQUIRE_DLPACK_ALIGNMENT
  if ((p_eff % static_cast<std::uintptr_t>(item_b)) != 0) {
    throw std::runtime_error("from_dlpack: effective pointer is not aligned to itemsize");
  }
#endif

  int64_t min_elem_off = 0, max_elem_off = 0;
  if (dl.ndim > 0) {
    for (int i = 0; i < dl.ndim; ++i) {
      int64_t n = (dl.ndim == 0) ? 1 : dl.shape[i];
      int64_t d = n > 0 ? (n - 1) : 0;
      if (d == 0) continue;
      int64_t st = (dl.ndim == 0) ? 1 : strides[static_cast<std::size_t>(i)];
      int64_t term = 0;
      if (!checked_mul_i64(st, d, term)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
      if (st >= 0) {
        int64_t tmp = 0; if (!checked_add_i64(max_elem_off, term, tmp)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)"); max_elem_off = tmp;
      } else {
        int64_t tmp = 0; if (!checked_add_i64(min_elem_off, term, tmp)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)"); min_elem_off = tmp;
      }
    }
  }
  int64_t max_plus_one = 0; if (!checked_add_i64(max_elem_off, 1, max_plus_one)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
  int64_t lower_bytes = 0; if (!checked_mul_i64(min_elem_off, static_cast<int64_t>(item_b), lower_bytes)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
  int64_t upper_excl = 0; if (!checked_mul_i64(max_plus_one, static_cast<int64_t>(item_b), upper_excl)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");
  int64_t req_bytes = 0; if (!checked_add_i64(upper_excl, -lower_bytes, req_bytes)) throw std::runtime_error("from_dlpack: span overflow (stride*extent/add)");

  auto base_ptr = checked_ptr_add_signed(p_eff, lower_bytes, ok);
  if (!ok) throw std::runtime_error("from_dlpack: pointer base overflow");
  int64_t storage_offset_elems = -min_elem_off;

  // destination allocation on current device
  int cur_dev = 0; cudaGetDevice(&cur_dev);
  // Mixed-device check for device tensors
  const int dev_type2 = dl.device.device_type;
  if (dev_type2 == kDLCUDA || dev_type2 == kDLCUDAManaged) {
    if (dl.device.device_id != cur_dev) {
      throw std::runtime_error(std::string("Expected all tensors to be on the same device: got cuda:") + std::to_string(dl.device.device_id) + " vs cuda:" + std::to_string(cur_dev));
    }
  }
  auto storage = vbt::cuda::new_cuda_storage(static_cast<std::size_t>(req_bytes), cur_dev);

  // Launch copy on current stream (async)
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(cur_dev));
  cudaError_t st;
  vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get(static_cast<vbt::cuda::DeviceIndex>(cur_dev));
  if (dev_type2 == kDLCUDA || dev_type2 == kDLCUDAManaged) {
    st = alloc.memcpyAsync(storage->data(), cur_dev, reinterpret_cast<void*>(base_ptr), dl.device.device_id,
                           static_cast<std::size_t>(req_bytes), stream, /*p2p_enabled=*/true);
  } else if (dev_type2 == kDLCUDAHost) {
    st = alloc.memcpyAsync(storage->data(), cur_dev, reinterpret_cast<void*>(base_ptr), -1,
                           static_cast<std::size_t>(req_bytes), stream, /*p2p_enabled=*/false);
  } else {
    throw std::invalid_argument("from_dlpack: unsupported device type for CUDA import");
  }
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(std::string("from_dlpack: Allocator::memcpyAsync failed: ") + (msg ? msg : ""));
  }

  // Ensure the copy completes and provider deleter runs on host synchronously to avoid lifetime hazards
  st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(std::string("from_dlpack: cudaStreamSynchronize failed: ") + (msg ? msg : ""));
  }

  guard.call_now_and_release();

  // Return Tensor owning CUDA storage
  return TensorImpl(storage, sizes, strides, storage_offset_elems, stype, Device::cuda(cur_dev));
}
#endif

} // namespace interop
} // namespace vbt
