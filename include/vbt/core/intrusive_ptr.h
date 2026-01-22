// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <utility>
#include <type_traits>

namespace vbt {
namespace core {

// Simple intrusive refcounted base (thread-safe). Objects start with refcount=0.
struct IntrusiveRefcounted {
  mutable std::atomic<std::size_t> refcount_{0};
  void retain() const noexcept { refcount_.fetch_add(1, std::memory_order_acq_rel); }
  void release() const noexcept {
    if (refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete this;
    }
  }
protected:
  virtual ~IntrusiveRefcounted() = default;
};

// Minimal intrusive_ptr implementation (like boost::intrusive_ptr).
template <class T>
class intrusive_ptr {
 public:
  using element_type = T;
  constexpr intrusive_ptr() noexcept : ptr_(nullptr) {}
  constexpr intrusive_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

  // Construct from raw pointer; add_ref indicates whether to retain.
  explicit intrusive_ptr(T* p, bool add_ref = true) noexcept : ptr_(p) {
    if (ptr_ && add_ref) ptr_->retain();
  }

  intrusive_ptr(const intrusive_ptr& other) noexcept : ptr_(other.ptr_) {
    if (ptr_) ptr_->retain();
  }
  intrusive_ptr(intrusive_ptr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  intrusive_ptr& operator=(const intrusive_ptr& other) noexcept {
    if (this != &other) {
      if (other.ptr_) other.ptr_->retain();
      if (ptr_) ptr_->release();
      ptr_ = other.ptr_;
    }
    return *this;
  }
  intrusive_ptr& operator=(intrusive_ptr&& other) noexcept {
    if (this != &other) {
      if (ptr_) ptr_->release();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  ~intrusive_ptr() { if (ptr_) ptr_->release(); }

  void reset(T* p = nullptr, bool add_ref = true) noexcept {
    if (ptr_) ptr_->release();
    ptr_ = p;
    if (ptr_ && add_ref) ptr_->retain();
  }

  T* get() const noexcept { return ptr_; }
  T& operator*() const noexcept { return *ptr_; }
  T* operator->() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  T* ptr_;
};

// Factory to allocate a new intrusive object and wrap it.
// Requires T to be derived from IntrusiveRefcounted and constructible with Args...
template <class T, class... Args>
inline intrusive_ptr<T> make_intrusive(Args&&... args) {
  static_assert(std::is_base_of<IntrusiveRefcounted, T>::value, "T must be IntrusiveRefcounted");
  T* raw = new T(std::forward<Args>(args)...);
  return intrusive_ptr<T>(raw, /*add_ref=*/true);
}

} // namespace core
} // namespace vbt
