// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/storage.h"

namespace vbt {
namespace core {

void Storage::release_resources() noexcept {
  data_.reset();
  nbytes_ = 0;
}

} // namespace core
} // namespace vbt
