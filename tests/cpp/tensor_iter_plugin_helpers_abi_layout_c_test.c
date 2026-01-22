// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>

#include "vbt/plugin/vbt_plugin.h"

// C-side ABI layout checks for TI helpers.

_Static_assert(sizeof(enum vt_iter_overlap_mode) == sizeof(int32_t),
               "vt_iter_overlap_mode must remain 32-bit for ABI");

_Static_assert(offsetof(struct vt_iter_config, check_mem_overlap) == sizeof(int64_t),
               "vt_iter_config::check_mem_overlap must follow max_rank");

_Static_assert(sizeof(struct vt_iter_config) >= sizeof(int64_t) + sizeof(int32_t),
               "vt_iter_config must be at least {int64_t, int32_t} in size");

// No tests; this TU is compiled and linked into a gtest binary so that
// the static assertions run at compile time.
