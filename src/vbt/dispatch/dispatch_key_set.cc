// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/dispatch/dispatch_key_set.h"

namespace vbt {
namespace dispatch {

thread_local LocalDispatchKeySet tls_local_dispatch_key_set{};

} // namespace dispatch
} // namespace vbt
