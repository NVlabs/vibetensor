// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"

namespace vbt {
namespace core {

class Storage : public IntrusiveRefcounted {
 public:
  Storage() = default;
  explicit Storage(DataPtr data, std::size_t nbytes) : data_(std::move(data)), nbytes_(nbytes) {}
  virtual ~Storage() = default;

  void release_resources() noexcept;

  std::size_t nbytes() const noexcept { return nbytes_; }
  void* data() const noexcept { return data_.get(); }

 private:
  DataPtr data_{};
  std::size_t nbytes_{0};
};

using StoragePtr = intrusive_ptr<Storage>;

} // namespace core
} // namespace vbt
