// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>

#include "vbt/plugin/vbt_plugin.h"

//
// These tests ensure that:
//  - The C prototypes for vt_tensor_iter_unary_cpu / _binary_cpu and
//    vt_iter_config match the design in README_v2.
//  - The README_v2.md design document continues to contain the expected
//    code block describing these APIs.
//
// This gives a lightweight guard that the public C ABI and the design
// documentation stay in sync as the code evolves.

namespace {

// Expected function pointer types derived from the design in
// design/tensor_iter/README_v2.md §4.2.2.
using UnaryHelperFnFromDoc = vt_status (*)(const vt_iter_config*,
                                           vt_tensor,
                                           vt_tensor,
                                           vt_tensor_iter_loop1d_fn,
                                           void*);

using BinaryHelperFnFromDoc = vt_status (*)(const vt_iter_config*,
                                            vt_tensor,
                                            vt_tensor,
                                            vt_tensor,
                                            vt_tensor_iter_loop1d_fn,
                                            void*);

static_assert(std::is_same<UnaryHelperFnFromDoc,
                           decltype(&vt_tensor_iter_unary_cpu)>::value,
              "vt_tensor_iter_unary_cpu prototype diverged from design doc");

static_assert(std::is_same<BinaryHelperFnFromDoc,
                           decltype(&vt_tensor_iter_binary_cpu)>::value,
              "vt_tensor_iter_binary_cpu prototype diverged from design doc");

// Raw snippet from design/tensor_iter/README_v2.md that describes the
// vt_tensor_iter_* helpers and vt_iter_config. If this block changes,
// either the design has evolved (and this test should be updated in
// lock-step) or the documentation has drifted from the canonical ABI.
static const char kHelpersSnippet[] = R"DOC(```c
// User callback; mirrors TensorIterBase::loop1d_t but is pure C.
typedef void (*vt_tensor_iter_loop1d_fn)(char** data,
                                         const int64_t* strides,
                                         int64_t size,
                                         void* ctx);

// Overlap mode and lightweight configuration for TI-backed helpers.
typedef enum vt_iter_overlap_mode {
  VT_ITER_OVERLAP_DISABLE = 0,
  VT_ITER_OVERLAP_ENABLE  = 1,
} vt_iter_overlap_mode;

typedef struct vt_iter_config {
  int64_t max_rank;                // 0 ⇒ TI default (64); [1,64] explicit cap.
  vt_iter_overlap_mode check_mem_overlap;
} vt_iter_config;

// TI-backed CPU helpers (also exposed via vbt_host_api function pointers).
vt_status vt_tensor_iter_unary_cpu(const vt_iter_config* cfg,
                                   vt_tensor out,
                                   vt_tensor in,
                                   vt_tensor_iter_loop1d_fn loop,
                                   void* ctx);

vt_status vt_tensor_iter_binary_cpu(const vt_iter_config* cfg,
                                    vt_tensor out,
                                    vt_tensor a,
                                    vt_tensor b,
                                    vt_tensor_iter_loop1d_fn loop,
                                    void* ctx);
```)DOC";

std::string load_file(const std::string& path) {
  std::ifstream in(path.c_str());
  if (!in.is_open()) {
    return std::string();
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  return oss.str();
}

}  // namespace

TEST(TensorIterDocSyncTest, READMEContainsHelperSnippet) {
#ifndef VBT_PROJECT_SOURCE_DIR
  GTEST_SKIP() << "VBT_PROJECT_SOURCE_DIR not defined; doc-sync test disabled";
#else
  const std::string root = VBT_PROJECT_SOURCE_DIR;
  const std::string readme_path = root + "/design/tensor_iter/README_v2.md";

  const std::string contents = load_file(readme_path);
  if (contents.empty()) {
    GTEST_SKIP() << "README_v2.md not present at " << readme_path;
  }

  const std::string snippet(kHelpersSnippet);
  EXPECT_NE(contents.find(snippet), std::string::npos)
      << "design/tensor_iter/README_v2.md is missing the expected helper "
      << "snippet; update the design doc or tensor_iter_doc_sync_test.cc in "
      << "lock-step.";
#endif
}
