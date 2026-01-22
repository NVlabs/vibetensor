// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <string>
#include <limits>
#include <cstdlib>
#include <complex>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/complex.h"
#include "vbt/core/device.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/event.h"
#include "vbt/cuda/device.h"
#endif

namespace nb = nanobind;

namespace vbt_py {

#if VBT_WITH_CUDA

static_assert(sizeof(vbt::core::Complex64) == sizeof(std::complex<float>),
              "Complex64 must match std::complex<float> layout for NumPy interop");
static_assert(sizeof(vbt::core::Complex128) == sizeof(std::complex<double>),
              "Complex128 must match std::complex<double> layout for NumPy interop");
static_assert(alignof(vbt::core::Complex64) >= alignof(std::complex<float>),
              "Complex64 alignment must be >= std::complex<float> alignment");
static_assert(alignof(vbt::core::Complex128) >= alignof(std::complex<double>),
              "Complex128 alignment must be >= std::complex<double> alignment");

static constexpr const char* kErrComplexDisabled =
    "complex dtypes are disabled; set VBT_ENABLE_COMPLEX=1";

static inline bool complex_enabled_from_env() noexcept {
  const char* raw = std::getenv("VBT_ENABLE_COMPLEX");
  return raw && raw[0] == '1' && raw[1] == '\0';
}

static inline void throw_if_complex_disabled(vbt::core::ScalarType st) {
  if ((st == vbt::core::ScalarType::Complex64 || st == vbt::core::ScalarType::Complex128) &&
      !complex_enabled_from_env()) {
    throw nb::type_error(kErrComplexDisabled);
  }
}

static inline vbt::core::ScalarType h2d_dtype_from_token(const std::string& token) {
  using vbt::core::ScalarType;
  if (token == "float32") return ScalarType::Float32;
  if (token == "float64") return ScalarType::Float64;
  if (token == "complex64") { throw_if_complex_disabled(ScalarType::Complex64); return ScalarType::Complex64; }
  if (token == "complex128") { throw_if_complex_disabled(ScalarType::Complex128); return ScalarType::Complex128; }
  if (token == "int32") return ScalarType::Int32;
  if (token == "int64") return ScalarType::Int64;
  if (token == "bool") return ScalarType::Bool;
  if (token == "float16") return ScalarType::Float16;
#if VBT_HAS_DLPACK_BF16
  if (token == "bfloat16") return ScalarType::BFloat16;
#endif
  throw nb::type_error("unsupported dtype for H2D copy");
}

static inline std::vector<int64_t> contig_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = (std::ptrdiff_t)sizes.size() - 1; i >= 0; --i) {
    st[(std::size_t)i] = acc;
    int64_t dim = sizes[(std::size_t)i];
    if (dim != 0) acc *= dim;
  }
  return st;
}

static inline int resolve_device_index(const std::optional<int>& device_opt) {
  int n = vbt::cuda::device_count();
  if (!device_opt.has_value()) {
    if (n == 0) throw std::runtime_error("CUDA unavailable: no devices");
    int dev = 0; {
      nb::gil_scoped_release r;
      cudaError_t st = cudaGetDevice(&dev);
      if (st != cudaSuccess) {
        const char* msg = cudaGetErrorString(st);
        std::string m = "cudaGetDevice failed: "; m += (msg ? msg : "");
        throw std::runtime_error(m);
      }
    }
    return dev;
  }
  int dev = *device_opt;
  if (dev < 0) throw nb::value_error("device must be >= 0 or None");
  if (n == 0) throw std::runtime_error("CUDA unavailable: no devices");
  if (dev >= n) throw nb::value_error("device index out of range");
  return dev;
}

struct AsyncPinnedState {
  void* ptr{nullptr};
  int   dev{-1};
  vbt::cuda::Event ev{false};
};

static inline void maybe_wait_for_producer_streams(
    const vbt::core::StoragePtr& storage,
    const vbt::cuda::Stream&     dst_stream) {
  // If producer metadata is missing, we conservatively do nothing and rely on
  // legacy stream semantics (callers may still synchronize explicitly).
  if (!vbt::cuda::has_producer_metadata(storage)) {
    return;
  }

  vbt::cuda::for_each_producer_stream(storage, [&](const vbt::cuda::Stream& prod) {
    // Producer streams are tracked per-storage and should be same-device, but
    // keep this defensive.
    if (prod.device_index() != dst_stream.device_index()) {
      return true;
    }
    if (prod.id() == dst_stream.id()) {
      return true;
    }

    // Record an event on the producer stream and make the destination stream
    // wait on it. This establishes correct cross-stream ordering for D2H copies
    // even when the producing kernel ran on a different stream than the current
    // one.
    vbt::cuda::Event ev(false);
    ev.record(prod);
    ev.wait(dst_stream);
    return true;
  });
}

#endif

void bind_cuda_copy(nb::module_& m) {
#if VBT_WITH_CUDA
  using vbt::core::TensorImpl;
  using vbt::core::ScalarType;
  using vbt::core::Device;


  // NumPy -> CUDA tensor (alloc + copy). Uses current stream of destination device.
  // Using generic Python Buffer Protocol to avoid nanobind ndarray type matching issues.
  m.def("_cuda_h2d_alloc_copy",
        [=](nb::object arr_obj, std::string dtype_token, int device, bool non_blocking) {
          PyObject* obj = arr_obj.ptr();
          Py_buffer view;
          if (PyObject_GetBuffer(obj, &view, PyBUF_C_CONTIGUOUS | PyBUF_ND) != 0) {
             throw nb::type_error("expected a C-contiguous buffer/array");
          }
          
          // RAII to release buffer
          struct BufferGuard {
             Py_buffer* v;
             ~BufferGuard() { PyBuffer_Release(v); }
          } guard{&view};

          const void* src = view.buf;
          if (!src) throw nb::value_error("buffer has no data");

          auto stype = h2d_dtype_from_token(dtype_token);
          std::size_t nbytes = (std::size_t) view.len;

          const std::size_t dtype_itemsize = (std::size_t) vbt::core::itemsize(stype);
          if (view.itemsize <= 0) {
            throw nb::value_error("buffer has invalid itemsize");
          }
          if ((std::size_t) view.itemsize != dtype_itemsize) {
            throw nb::value_error("dtype does not match buffer itemsize");
          }
          std::size_t numel = 1;
          for (int i = 0; i < view.ndim; ++i) {
            const Py_ssize_t dim = view.shape[i];
            if (dim < 0) throw nb::value_error("buffer has negative dimension");
            if (dim == 0) { numel = 0; break; }
            if (numel > std::numeric_limits<std::size_t>::max() / (std::size_t) dim) {
              throw nb::value_error("buffer shape overflow");
            }
            numel *= (std::size_t) dim;
          }
          if (dtype_itemsize != 0 &&
              numel > std::numeric_limits<std::size_t>::max() / dtype_itemsize) {
            throw nb::value_error("buffer size overflow");
          }
          const std::size_t expected_nbytes = numel * dtype_itemsize;
          if (expected_nbytes != nbytes) {
            throw nb::value_error("buffer size does not match shape and dtype");
          }

          // device is passed as int
          int dev = device;
          if (dev < 0) { 
             dev = resolve_device_index(std::nullopt);
          } else {
             resolve_device_index(dev);
          }

          // Allocate device storage on destination device
          auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);

          // Launch H2D async copy on destination device current stream
          auto stream = vbt::cuda::getCurrentStream((vbt::cuda::DeviceIndex) dev);
          vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get((vbt::cuda::DeviceIndex) dev);
          
          void* dst = storage->data();
          if (nbytes > 0) {
            // Release GIL during CUDA copy/wait
            nb::gil_scoped_release r;
            auto st = alloc.memcpyAsync(dst, dev, src, -1, nbytes, stream, /*p2p_enabled=*/false);
            if (st != cudaSuccess) {
              const char* msg = cudaGetErrorString(st);
              std::string m = "_cuda_h2d_alloc_copy: memcpyAsync failed: "; m += (msg ? msg : "");
              throw std::runtime_error(m);
            }
            if (!non_blocking) {
              vbt::cuda::DeviceGuard dg((vbt::cuda::DeviceIndex) dev);
              cudaError_t sync_st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
              if (sync_st != cudaSuccess) {
                const char* msg = cudaGetErrorString(sync_st);
                std::string m = "_cuda_h2d_alloc_copy: cudaStreamSynchronize failed: "; m += (msg ? msg : "");
                throw std::runtime_error(m);
              }
            }
          }

          // Build sizes/strides vectors from view
          std::vector<int64_t> sizes(view.ndim);
          for (int i = 0; i < view.ndim; ++i)
            sizes[i] = (int64_t) view.shape[i];
          auto strides = contig_strides(sizes);

          TensorImpl out(storage, sizes, strides, 0, stype, Device::cuda(dev));

          // For non-blocking transfers, keep the source buffer alive for the
          // lifetime of the returned Tensor to avoid UAF hazards.
          nb::object out_obj = nb::cast(out);
          if (non_blocking) {
            out_obj.attr("_h2d_source_ref") = arr_obj;
          }
          return out_obj;
        }, nb::arg("array"), nb::arg("dtype"), nb::arg("device"), nb::arg("non_blocking") = false);

  // D2H (synchronous copy)
  m.def("_cuda_d2h_copy_numpy_sync",
        [](const TensorImpl& t) -> nb::object {
          auto dev = t.device();
          if (dev.type != kDLCUDA) throw nb::value_error("expected a CUDA tensor");
          if (!t.is_non_overlapping_and_dense()) throw nb::value_error("tensor must be dense-contiguous");
          ScalarType st = t.dtype();

          std::size_t nbytes = (std::size_t) t.itemsize() * (std::size_t) t.numel();

          // Zero-size: construct an empty NumPy array of the requested dtype
          if (nbytes == 0) {
            nb::object np = nb::module_::import_("numpy");
            nb::list shape_list;
            for (auto s : t.sizes()) shape_list.append(nb::int_(s));
            const char* dt = nullptr;
            switch (st) {
              case ScalarType::Float32: dt = "float32"; break;
              case ScalarType::Float64: dt = "float64"; break;
              case ScalarType::Complex64: dt = "complex64"; break;
              case ScalarType::Complex128: dt = "complex128"; break;
              case ScalarType::Int32:   dt = "int32";   break;
              case ScalarType::Int64:   dt = "int64";   break;
              case ScalarType::Bool:    dt = "bool";    break;
              case ScalarType::Float16: dt = "float16"; break;
              case ScalarType::BFloat16: dt = "bfloat16"; break;
            }
            if (dt == nullptr) throw nb::value_error("unsupported dtype for D2H copy");
            return np.attr("empty")(nb::tuple(shape_list), nb::str(dt));
          }

          // For all dtypes, copy into user-visible NumPy array allocated on Python side
          nb::object np = nb::module_::import_("numpy");
          nb::list shape_list;
          for (auto s : t.sizes()) shape_list.append(nb::int_(s));
          const char* dt = nullptr;
          switch (st) {
            case ScalarType::Float32: dt = "float32"; break;
            case ScalarType::Float64: dt = "float64"; break;
            case ScalarType::Complex64: dt = "complex64"; break;
            case ScalarType::Complex128: dt = "complex128"; break;
            case ScalarType::Int32:   dt = "int32";   break;
            case ScalarType::Int64:   dt = "int64";   break;
            case ScalarType::Bool:    dt = "bool";    break;
            case ScalarType::Float16: dt = "float16"; break;
            case ScalarType::BFloat16: dt = "bfloat16"; break;
          }
          if (dt == nullptr) throw nb::value_error("unsupported dtype for D2H copy");

          // Validate that NumPy supports the dtype (especially for bfloat16)
          try { (void) np.attr("dtype")(nb::str(dt)); }
          catch (...) { throw nb::value_error("unsupported dtype for D2H copy"); }

          nb::object arr_obj = np.attr("empty")(nb::tuple(shape_list), nb::str(dt));
          // Wrap as ndarray to get data pointer
          nb::ndarray<> arr = nb::cast<nb::ndarray<>>(arr_obj);

          int src_dev = (int) dev.index;
          auto stream = vbt::cuda::getCurrentStream((vbt::cuda::DeviceIndex) src_dev);
          vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get((vbt::cuda::DeviceIndex) src_dev);
          {
            nb::gil_scoped_release r;
            maybe_wait_for_producer_streams(t.storage(), stream);
            auto st_code = alloc.memcpyAsync(arr.data(), -1, t.data(), src_dev, nbytes, stream, /*p2p_enabled=*/false);
            if (st_code != cudaSuccess) {
              const char* msg = cudaGetErrorString(st_code);
              std::string m = "_cuda_d2h_copy_numpy_sync: memcpyAsync failed: "; m += (msg ? msg : "");
              throw std::runtime_error(m);
            }
            vbt::cuda::DeviceGuard dg((vbt::cuda::DeviceIndex) src_dev);
            cudaError_t sync_st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.handle()));
            if (sync_st != cudaSuccess) {
              const char* msg = cudaGetErrorString(sync_st);
              std::string m = "_cuda_d2h_copy_numpy_sync: cudaStreamSynchronize failed: "; m += (msg ? msg : "");
              throw std::runtime_error(m);
            }
          }

          return arr_obj;
        });

  // CUDA tensor -> (NumPy ndarray, Event) async. Allocate pinned host and return immediately.
  m.def("_cuda_d2h_copy_numpy_async",
        [](const TensorImpl& t) -> nb::tuple {
          auto dev = t.device();
          if (dev.type != kDLCUDA) throw nb::value_error("expected a CUDA tensor");
          if (!t.is_non_overlapping_and_dense()) throw nb::value_error("tensor must be dense-contiguous");
          ScalarType st = t.dtype();

          if (!(st == ScalarType::Float32 || st == ScalarType::Float64 ||
                st == ScalarType::Complex64 || st == ScalarType::Complex128 ||
                st == ScalarType::Int32 || st == ScalarType::Int64 ||
                st == ScalarType::Bool)) {
            throw nb::value_error("unsupported dtype for D2H copy");
          }

          std::size_t nbytes = (std::size_t) t.itemsize() * (std::size_t) t.numel();

          int src_dev = (int) dev.index;
          auto stream = vbt::cuda::getCurrentStream((vbt::cuda::DeviceIndex) src_dev);

          if (nbytes == 0) {
            vbt::cuda::Event ev(false);
            {
              nb::gil_scoped_release r;
              ev.record(stream);
            }
            std::vector<size_t> shape(t.sizes().size());
            for (size_t i = 0; i < shape.size(); ++i) shape[i] = (size_t) t.sizes()[i];
            switch (st) {
              case ScalarType::Float32: return nb::make_tuple(nb::ndarray<nb::numpy, const float>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              case ScalarType::Float64: return nb::make_tuple(nb::ndarray<nb::numpy, const double>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              case ScalarType::Complex64: return nb::make_tuple(nb::ndarray<nb::numpy, const std::complex<float>>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              case ScalarType::Complex128: return nb::make_tuple(nb::ndarray<nb::numpy, const std::complex<double>>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              case ScalarType::Int32:   return nb::make_tuple(nb::ndarray<nb::numpy, const int32_t>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              case ScalarType::Int64:   return nb::make_tuple(nb::ndarray<nb::numpy, const int64_t>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              case ScalarType::Bool:    return nb::make_tuple(nb::ndarray<nb::numpy, const bool>(nullptr, shape.size(), shape.data(), nb::handle()), std::move(ev));
              default: throw nb::value_error("unsupported dtype for D2H copy");
            }
          }

          void* host = nullptr;
          cudaError_t st_alloc = cudaHostAlloc(&host, nbytes, cudaHostAllocPortable);
          if (st_alloc != cudaSuccess) {
            const char* msg = cudaGetErrorString(st_alloc);
            std::string m = "_cuda_d2h_copy_numpy_async: cudaHostAlloc failed: "; m += (msg ? msg : "");
            throw std::runtime_error(m);
          }

          vbt::cuda::Allocator& alloc = vbt::cuda::Allocator::get((vbt::cuda::DeviceIndex) src_dev);
          vbt::cuda::Event ev(false);
          {
            nb::gil_scoped_release r;
            maybe_wait_for_producer_streams(t.storage(), stream);
            auto st_code = alloc.memcpyAsync(host, -1, t.data(), src_dev, nbytes, stream, /*p2p_enabled=*/false);
            if (st_code != cudaSuccess) {
              const char* msg = cudaGetErrorString(st_code);
              std::string m = "_cuda_d2h_copy_numpy_async: memcpyAsync failed: "; m += (msg ? msg : "");
              cudaFreeHost(host);
              throw std::runtime_error(m);
            }

            // Prevent allocator reuse of the source storage while the async copy is in flight.
            vbt::cuda::record_stream(t.storage(), stream);

            // Record an event on the stream to fence the copy
            ev.record(stream);
          }

          // Capsule state that synchronizes the event before freeing pinned memory on GC
          auto* state = new AsyncPinnedState{host, src_dev, std::move(ev)};
          nb::capsule owner(state, [](void* p) noexcept {
            auto* s = reinterpret_cast<AsyncPinnedState*>(p);
            if (!s) return;
            // Wait for completion without holding the GIL; swallow any exceptions to honor noexcept
            try { nb::gil_scoped_release r; s->ev.synchronize(); } catch (...) {}
            cudaError_t st = cudaFreeHost(s->ptr);
            if (st != cudaSuccess) { (void)cudaGetLastError(); }
            delete s;
          });

          // Build shape array
          std::vector<size_t> shape(t.sizes().size());
          for (size_t i = 0; i < shape.size(); ++i) shape[i] = (size_t) t.sizes()[i];

          nb::object ndarray_obj;
          switch (st) {
            case ScalarType::Float32: ndarray_obj = nb::ndarray<nb::numpy, const float>(host, shape.size(), shape.data(), owner).cast(); break;
            case ScalarType::Float64: ndarray_obj = nb::ndarray<nb::numpy, const double>(host, shape.size(), shape.data(), owner).cast(); break;
            case ScalarType::Complex64: ndarray_obj = nb::ndarray<nb::numpy, const std::complex<float>>(host, shape.size(), shape.data(), owner).cast(); break;
            case ScalarType::Complex128: ndarray_obj = nb::ndarray<nb::numpy, const std::complex<double>>(host, shape.size(), shape.data(), owner).cast(); break;
            case ScalarType::Int32:   ndarray_obj = nb::ndarray<nb::numpy, const int32_t>(host, shape.size(), shape.data(), owner).cast(); break;
            case ScalarType::Int64:   ndarray_obj = nb::ndarray<nb::numpy, const int64_t>(host, shape.size(), shape.data(), owner).cast(); break;
            case ScalarType::Bool:    ndarray_obj = nb::ndarray<nb::numpy, const bool>(host, shape.size(), shape.data(), owner).cast(); break;
            default: cudaFreeHost(host); throw nb::value_error("unsupported dtype for D2H copy");
          }

          // Return ndarray and a separate event recorded on the same stream for the caller to wait on
          vbt::cuda::Event ev2(false);
          {
            nb::gil_scoped_release r;
            ev2.record(stream);
          }
          return nb::make_tuple(ndarray_obj, std::move(ev2));
        });
#else
  // Define stub functions so attributes exist even without CUDA
  m.def("_cuda_h2d_alloc_copy",
        [](nb::object, nb::object, nb::object, bool){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); },
        nb::arg("array"), nb::arg("dtype"), nb::arg("device").none(true) = nb::none(), nb::arg("non_blocking") = false);
  m.def("_cuda_d2h_copy_numpy_sync",
        [](nb::object){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
  m.def("_cuda_d2h_copy_numpy_async",
        [](nb::object){ throw nb::type_error("CUDA is not available or not built for VibeTensor"); });
#endif
}

} // namespace vbt_py
