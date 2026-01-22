// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace vbt {
namespace core {

// Owning pointer with custom deleter that can capture context.
class DataPtr {
 public:
  using Deleter = std::function<void(void*)>;

  DataPtr() noexcept = default;
  DataPtr(void* data, Deleter d) noexcept : data_(data), deleter_(std::move(d)) {}
  DataPtr(const DataPtr&) = delete;
  DataPtr& operator=(const DataPtr&) = delete;
  DataPtr(DataPtr&& other) noexcept(noexcept(std::declval<DataPtr&>().swap(other))) { swap(other); }
  DataPtr& operator=(DataPtr&& other) noexcept(noexcept(std::declval<DataPtr&>().swap(other))) { if (this != &other) swap(other); return *this; }
  ~DataPtr() { reset(); }

  void* get() const noexcept { return data_; }
  explicit operator bool() const noexcept { return data_ != nullptr; }

  void reset(void* p = nullptr, Deleter d = nullptr) noexcept {
    if (data_) {
      if (deleter_) {
        try { deleter_(data_); } catch (...) { /* swallow */ }
      }
    }
    data_ = p; deleter_ = std::move(d);
  }

  void swap(DataPtr& other) noexcept(noexcept(std::swap(data_, other.data_)) && noexcept(deleter_.swap(other.deleter_))) {
    std::swap(data_, other.data_);
    deleter_.swap(other.deleter_);
  }

 private:
  void* data_{nullptr};
  Deleter deleter_{};
};

} // namespace core
} // namespace vbt
