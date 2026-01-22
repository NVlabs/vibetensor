// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/allocator_async.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"
#include <memory>

#if VBT_WITH_CUDA

namespace vbt { namespace cuda {

namespace {


static cudaStream_t sid_to_stream(StreamId sid) {
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(sid));
}
}

AsyncBackend& AsyncBackend::get(DeviceIndex dev) {
  static std::mutex g_mu;
  static std::vector<std::unique_ptr<AsyncBackend>> backends;
  std::lock_guard<std::mutex> lk(g_mu);
  if (backends.empty()) {
    backends.resize(static_cast<std::size_t>(std::max(0, device_count())));
  }
  auto idx = static_cast<std::size_t>(dev < 0 ? 0 : dev);
  if (idx >= backends.size()) backends.resize(idx + 1);
  if (!backends[idx]) backends[idx] = std::unique_ptr<AsyncBackend>(new AsyncBackend(dev));
  return *backends[idx];
}

AsyncBackend::AsyncBackend(DeviceIndex dev) : dev_(dev) {}

void AsyncBackend::configure(double /*per_process_fraction*/,
                             std::size_t release_threshold_bytes,
                             bool reuse_follow_event_deps,
                             bool reuse_allow_opportunistic,
                             bool reuse_allow_internal_deps) {
  std::lock_guard<std::mutex> lg(mu_);
  // Fraction is owned by set_memory_fraction(); ignore the argument.
  release_threshold_bytes_ = release_threshold_bytes;
  reuse_follow_event_deps_ = reuse_follow_event_deps;
  reuse_allow_opportunistic_ = reuse_allow_opportunistic;
  reuse_allow_internal_deps_ = reuse_allow_internal_deps;
  inited_ = false; // re-init on next use
}

void AsyncBackend::set_memory_fraction(double f) {
  // Clamp into [0,1] for robustness against non-Python callers.
  if (f < 0.0) f = 0.0;
  if (f > 1.0) f = 1.0;

  std::lock_guard<std::mutex> lg(mu_);
  per_process_fraction_ = f;

  if (!inited_) {
    // lazy_init_() will compute limit_bytes_ when first allocation happens.
    return;
  }

  std::size_t freeB = 0, totalB = 0;
  {
    DeviceGuard g(dev_);
    cudaMemGetInfo(&freeB, &totalB);
  }
  limit_bytes_ = static_cast<std::size_t>(f * static_cast<double>(totalB));
}

void AsyncBackend::lazy_init_() {
  if (inited_) return;
  DeviceGuard g(dev_);
  // compute limit
  size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB, &totalB);
  limit_bytes_ = static_cast<std::size_t>(per_process_fraction_ * static_cast<double>(totalB));
  // set mempool attributes
  cudaMemPool_t mempool = nullptr;
  cudaDeviceGetDefaultMemPool(&mempool, static_cast<int>(dev_));
  uint64_t threshold = (release_threshold_bytes_ == static_cast<std::size_t>(-1)) ? UINT64_MAX : static_cast<uint64_t>(release_threshold_bytes_);
  cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
  int enable = 1, disable = 0;
  cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseFollowEventDependencies, reuse_follow_event_deps_ ? &enable : &disable);
  cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowOpportunistic, reuse_allow_opportunistic_ ? &enable : &disable);
  cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowInternalDependencies, reuse_allow_internal_deps_ ? &enable : &disable);
  if (!unifying_stream_) {
    cudaStreamCreateWithFlags(&unifying_stream_, cudaStreamNonBlocking);
  }
  inited_ = true;
}

void AsyncBackend::mallocAsync_(void** devPtr, DeviceIndex device, std::size_t size, cudaStream_t stream) {
  DeviceGuard g(device);
  // pre-check deferred frees: see if any can be freed now
  if (!deferred_.empty()) {
    std::vector<void*> kept;
    kept.reserve(deferred_.size());
    for (void* p : deferred_) {
      auto it = ptrs_.find(p);
      if (it == ptrs_.end()) continue;
      if (any_stream_capturing_(it->second, it->second.creation_stream)) {
        kept.push_back(p);
      } else {
        free_impl_(p);
      }
    }
    deferred_.swap(kept);
  }
  auto it_used = ptrs_.end(); // not used here; just to clarify control flow
  // enforce per-process fraction
  if (used_bytes_ + size > limit_bytes_) {
    // emulate OOM
    size_t device_free=0, device_total=0; cudaMemGetInfo(&device_free, &device_total);
    throw std::runtime_error("CUDA out of memory (async backend): allocation would exceed per-process limit");
  }
  cudaError_t err = cudaMallocAsync(devPtr, size, stream);
  if (err != cudaSuccess) {
    (void)cudaGetLastError(); // clear error for retry later by caller
    size_t device_free=0, device_total=0; cudaMemGetInfo(&device_free, &device_total);
    throw std::runtime_error("CUDA out of memory (async backend)");
  }
  // track
  PtrUsage u; u.creation_stream = reinterpret_cast<StreamId>(stream); u.size = size;
  ptrs_.emplace(*devPtr, std::move(u));
  used_bytes_ += size;
}

void* AsyncBackend::raw_alloc(std::size_t nbytes) {
  if (nbytes == 0) return nullptr;
  std::lock_guard<std::mutex> lg(mu_);
  lazy_init_();
  DeviceIndex device = dev_;
  cudaStream_t s = sid_to_stream(getCurrentStream(dev_).id());
  void* p=nullptr; mallocAsync_( &p, device, nbytes, s );
  return p;
}

void* AsyncBackend::raw_alloc(std::size_t nbytes, Stream s) {
  if (nbytes == 0) return nullptr;
  std::lock_guard<std::mutex> lg(mu_);
  lazy_init_();
  DeviceIndex device = dev_;
  cudaStream_t cs = sid_to_stream(s.id());
  void* p=nullptr; mallocAsync_( &p, device, nbytes, cs );
  return p;
}

bool AsyncBackend::any_stream_capturing_(const PtrUsage& u, StreamId owner) const noexcept {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(sid_to_stream(owner), &st);
  if (st != cudaStreamCaptureStatusNone) return true;
  for (auto sid : u.recorded_streams) {
    st = cudaStreamCaptureStatusNone; cudaStreamIsCapturing(sid_to_stream(sid), &st);
    if (st != cudaStreamCaptureStatusNone) return true;
  }
  return false;
}

void AsyncBackend::free_impl_(void* ptr) {
  auto it = ptrs_.find(ptr);
  if (it == ptrs_.end()) return;
  PtrUsage u = it->second;
  // if only creation stream, free there
  cudaStream_t creation = sid_to_stream(u.creation_stream);
  if (u.recorded_streams.empty()) {
    cudaFreeAsync(ptr, creation);
  } else {
    // sync all usage streams into unifying stream
    // record event on creation and wait in unifying
    cudaEvent_t ev = nullptr; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, creation);
    cudaStreamWaitEvent(unifying_stream_, ev, 0);
    cudaEventDestroy(ev);
    for (auto sid : u.recorded_streams) {
      cudaEvent_t ev2 = nullptr; cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming);
      cudaEventRecord(ev2, sid_to_stream(sid));
      cudaStreamWaitEvent(unifying_stream_, ev2, 0);
      cudaEventDestroy(ev2);
    }
    cudaFreeAsync(ptr, unifying_stream_);
  }
  used_bytes_ -= u.size;
  ptrs_.erase(it);
}

void AsyncBackend::raw_delete(void* ptr) noexcept {
  if (!ptr) return;
  std::lock_guard<std::mutex> lg(mu_);
  auto it = ptrs_.find(ptr);
  if (it == ptrs_.end()) return;
  if (any_stream_capturing_(it->second, it->second.creation_stream)) {
    deferred_.push_back(ptr);
    return;
  }
  free_impl_(ptr);
}

void AsyncBackend::record_stream(void* ptr, Stream s) noexcept {
  if (!ptr) return;
  std::lock_guard<std::mutex> lg(mu_);
  auto it = ptrs_.find(ptr);
  if (it == ptrs_.end()) return;
  StreamId sid = s.id();
  if (sid != it->second.creation_stream) it->second.recorded_streams.insert(sid);
}

void AsyncBackend::emptyCache() noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  lazy_init_();
  DeviceGuard g(dev_);
  cudaMemPool_t mempool = nullptr; cudaDeviceGetDefaultMemPool(&mempool, static_cast<int>(dev_));
  cudaDeviceSynchronize();
  cudaMemPoolTrimTo(mempool, 0);
}

DeviceStats AsyncBackend::getDeviceStats() const noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  DeviceStats st{};
  DeviceGuard g(dev_);
  cudaMemPool_t mempool = nullptr; cudaDeviceGetDefaultMemPool(&mempool, static_cast<int>(dev_));
  uint64_t r_cur=0, r_high=0, u_cur=0, u_high=0;
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &r_cur);
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemHigh, &r_high);
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &u_cur);
  cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemHigh, &u_high);
  st.reserved_bytes_all_current = static_cast<std::size_t>(r_cur);
  st.max_reserved_bytes_all = static_cast<std::size_t>(r_high);
  st.allocated_bytes_all_current = static_cast<std::size_t>(u_cur);
  st.max_allocated_bytes_all = static_cast<std::size_t>(u_high);
  return st;
}

void AsyncBackend::resetPeakStats() noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  DeviceGuard g(dev_);
  cudaMemPool_t mempool = nullptr; cudaDeviceGetDefaultMemPool(&mempool, static_cast<int>(dev_));
  uint64_t zero=0; cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReservedMemHigh, &zero);
  cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrUsedMemHigh, &zero);
}

bool AsyncBackend::owns(const void* p) const noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  return ptrs_.find(const_cast<void*>(p)) != ptrs_.end();
}

void* AsyncBackend::getBaseAllocation(void* p, std::size_t* size) const noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  auto it = ptrs_.find(p);
  if (it == ptrs_.end()) { if (size) *size = 0; return nullptr; }
  if (size) *size = it->second.size;
  return p;
}

#ifdef VBT_INTERNAL_TESTS
double AsyncBackend::debug_fraction() const noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  return per_process_fraction_;
}

std::size_t AsyncBackend::debug_limit_bytes() const noexcept {
  std::lock_guard<std::mutex> lg(mu_);
  return limit_bytes_;
}
#endif

cudaError_t AsyncBackend::memcpyAsync(void* dst, int dstDev, const void* src, int srcDev,
                                      std::size_t bytes, Stream s, bool p2p_enabled) noexcept {
  if (bytes == 0) return cudaSuccess;
  if (!dst || !src) return cudaErrorInvalidValue;
  auto is_host = [](int d){ return d < 0; };
  if (is_host(dstDev) && is_host(srcDev)) return cudaErrorInvalidValue;
  auto guard_dev_for = [&](cudaMemcpyKind kind)->DeviceIndex{
    switch (kind) {
      case cudaMemcpyHostToDevice: return static_cast<DeviceIndex>(dstDev);
      case cudaMemcpyDeviceToHost: return static_cast<DeviceIndex>(srcDev);
      case cudaMemcpyDeviceToDevice:
      case cudaMemcpyDefault: return static_cast<DeviceIndex>(dstDev);
      default: return static_cast<DeviceIndex>(dstDev);
    }
  };
  auto do_async = [&](cudaMemcpyKind kind)->cudaError_t{
    DeviceGuard dg(guard_dev_for(kind));
    return cudaMemcpyAsync(dst, src, bytes, kind, reinterpret_cast<cudaStream_t>(s.handle()));
  };
  auto do_peer = [&]()->cudaError_t{
    DeviceGuard dg(static_cast<DeviceIndex>(dstDev));
    return cudaMemcpyPeerAsync(dst, dstDev, src, srcDev, bytes, reinterpret_cast<cudaStream_t>(s.handle()));
  };
  if (is_host(srcDev) && !is_host(dstDev)) {
    return do_async(cudaMemcpyHostToDevice);
  } else if (!is_host(srcDev) && is_host(dstDev)) {
    return do_async(cudaMemcpyDeviceToHost);
  } else if (!is_host(srcDev) && !is_host(dstDev)) {
    if (dstDev == srcDev) {
      return do_async(cudaMemcpyDeviceToDevice);
    }
    if (p2p_enabled) {
      cudaError_t st = do_peer();
      if (st != cudaSuccess) { (void)cudaGetLastError(); return do_async(cudaMemcpyDefault); }
      return st;
    } else {
      cudaError_t st = do_peer();
      if (st != cudaSuccess) { (void)cudaGetLastError(); return do_async(cudaMemcpyDefault); }
      return st;
    }
  }
  return do_async(cudaMemcpyDefault);
}

cudaError_t AsyncBackend::enablePeerAccess(int dev, int peer) noexcept {
  DeviceGuard dg(static_cast<DeviceIndex>(dev));
  cudaMemPool_t mempool = nullptr; cudaDeviceGetDefaultMemPool(&mempool, peer);
  cudaMemAccessDesc desc{}; desc.location.type = cudaMemLocationTypeDevice; desc.location.id = dev; desc.flags = cudaMemAccessFlagsProtReadWrite;
  return cudaMemPoolSetAccess(mempool, &desc, 1);
}

}} // namespace vbt::cuda

#endif // VBT_WITH_CUDA
