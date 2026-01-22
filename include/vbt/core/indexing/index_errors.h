// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

//
// These constants are the single source of truth for pinned error
// substrings used by the m_index basic/advanced indexing subsystem.
// Production code must reference these instead of duplicating literals.
//
// the normative registry and tests mapping.

namespace vbt {
namespace core {
namespace indexing {
namespace errors {

// E-B*: basic indexing structural errors.
inline constexpr const char* kErrTooManyIndices =
    "too many indices for tensor of dimension";

inline constexpr const char* kErrMultipleEllipsis =
    "index: at most one ellipsis is allowed";

inline constexpr const char* kErrInvalidZeroDim =
    "invalid index of a 0-dim tensor";

// E-A*: advanced indexing feature-flag / policy errors.
inline constexpr const char* kErrAdvDisabledCore =
    "advanced indexing disabled";

// E-A*: advanced indexing range / DoS errors.
inline constexpr const char* kErrIndexOutOfRange =
    "advanced indexing: index out of range for dimension with size";

inline constexpr const char* kErrAdvIndexTooLarge =
    "advanced indexing too large";

inline constexpr const char* kErrAdvResultTooLarge =
    "advanced indexing result too large";

inline constexpr const char* kErrAdvTooManyIndexDims =
    "advanced indexing too many index dims";

// E-V*: vt/meta validation errors.
inline constexpr const char* kErrMetaInvalidShape =
    "meta must be 1-D CPU int64 with at least 4 elements";

inline constexpr const char* kErrMetaUnsupportedVersion =
    "unsupported meta version";

// CUDA caps and policy errors.
inline constexpr const char* kErrCudaAdvResultTooLarge =
    "index: CUDA advanced indexing result too large";

inline constexpr const char* kErrCudaAdvBoolMaskUnsupported =
    "CUDA advanced indexing does not support boolean mask indices";

// CUDA memcpy/sync operational failures.
inline constexpr const char* kErrCudaAdvCopyD2HFailed =
    "CUDA advanced indexing D2H copy failed";

inline constexpr const char* kErrCudaAdvSyncFailed =
    "CUDA advanced indexing D2H sync failed";

inline constexpr const char* kErrCudaAdvCopyH2DFailed =
    "CUDA advanced indexing H2D copy failed";

// Autograd policy errors.
inline constexpr const char* kErrIndexPutAutogradUnsupported =
    "vt::index_put: autograd for in-place advanced indexing is not supported";

// Autograd policy for index_put_ overwrite semantics:
// reject duplicate indices when accumulate=False under autograd to avoid
// nondeterministic "last write wins" gradients.
inline constexpr const char* kErrIndexPutAutogradDuplicateIndices =
    "vt::index_put: duplicate indices are not supported when accumulate=False under autograd";

inline constexpr const char* kErrVtIndexPrefixRequiresAutogradIndexingV2 =
    "vt::index: prefix meta requires VBT_AUTOGRAD_INDEXING_V2";

} // namespace errors
} // namespace indexing
} // namespace core
} // namespace vbt
