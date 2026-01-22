// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/***************************************************************************************************
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "../common/cutlass_unit_test.h"

#include "cutlass/experimental/distributed/collective/ring_allreduce_types.hpp"

namespace {

using cutlass::distributed::collective::merge_status;
using cutlass::distributed::collective::RingAllreduceError;

static inline int precedence_rank(RingAllreduceError e) {
  // Matches ring_allreduce/design/README.md (design_v11).
  switch (e) {
    case RingAllreduceError::kOk: return 0;
    case RingAllreduceError::kAbortObserved: return 1;
    case RingAllreduceError::kTimeout: return 2;
    case RingAllreduceError::kInvalidParams: return 3;
  }
  // Silence compiler warnings; unknown values are treated as lowest precedence.
  return -1;
}

static inline RingAllreduceError expected_merge(RingAllreduceError a, RingAllreduceError b) {
  int ra = precedence_rank(a);
  int rb = precedence_rank(b);
  return (ra >= rb) ? a : b;
}

} // namespace

TEST(RingAllreduceStatus, MergeStatusTable) {
  // Table from ring_allreduce/design/README.md.
  EXPECT_EQ(merge_status(RingAllreduceError::kOk, RingAllreduceError::kOk),
            RingAllreduceError::kOk);

  EXPECT_EQ(merge_status(RingAllreduceError::kAbortObserved, RingAllreduceError::kOk),
            RingAllreduceError::kAbortObserved);

  EXPECT_EQ(merge_status(RingAllreduceError::kTimeout, RingAllreduceError::kAbortObserved),
            RingAllreduceError::kTimeout);

  EXPECT_EQ(merge_status(RingAllreduceError::kInvalidParams, RingAllreduceError::kTimeout),
            RingAllreduceError::kInvalidParams);

  EXPECT_EQ(merge_status(RingAllreduceError::kInvalidParams, RingAllreduceError::kAbortObserved),
            RingAllreduceError::kInvalidParams);
}

TEST(RingAllreduceStatus, MergeStatusAllPairs) {
  constexpr RingAllreduceError kAll[] = {
      RingAllreduceError::kOk,
      RingAllreduceError::kAbortObserved,
      RingAllreduceError::kTimeout,
      RingAllreduceError::kInvalidParams,
  };

  for (RingAllreduceError a : kAll) {
    for (RingAllreduceError b : kAll) {
      RingAllreduceError got = merge_status(a, b);
      RingAllreduceError exp = expected_merge(a, b);

      EXPECT_EQ(got, exp) << "a=" << uint32_t(a) << " b=" << uint32_t(b);
      EXPECT_EQ(got, merge_status(b, a)) << "merge must be commutative";

      for (RingAllreduceError c : kAll) {
        RingAllreduceError lhs = merge_status(merge_status(a, b), c);
        RingAllreduceError rhs = merge_status(a, merge_status(b, c));
        EXPECT_EQ(lhs, rhs) << "merge must be associative"
                            << " (a=" << uint32_t(a)
                            << " b=" << uint32_t(b)
                            << " c=" << uint32_t(c) << ")";
      }
    }
  }
}
