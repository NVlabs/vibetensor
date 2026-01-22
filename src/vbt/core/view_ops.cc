// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_ops.h"

#include "vbt/core/checked_math.h"
#include "vbt/core/type_promotion.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace vbt {
namespace core {

static inline int64_t wrap_dim(int64_t dim, int64_t ndims) {
  if (ndims <= 0) return 0;
  if (dim < 0) dim += ndims;
  if (dim < 0 || dim >= ndims) {
    throw std::out_of_range("Dimension out of range (expected to be in range of [" +
                            std::to_string(-ndims) + ", " + std::to_string(ndims-1) +
                            "], but got " + std::to_string(dim) + ")");
  }
  return dim;
}

TensorImpl select(const TensorImpl& self, int64_t dim, int64_t index) {
  const auto& sz = self.sizes();
  const auto& st = self.strides();
  const int64_t nd = static_cast<int64_t>(sz.size());
  if (nd == 0) {
    throw std::invalid_argument("select() cannot be applied to a 0-dim tensor.");
  }
  dim = wrap_dim(dim, nd);
  const int64_t size_dim = sz[static_cast<std::size_t>(dim)];
  int64_t idx = index;
  if (idx < -size_dim || idx >= size_dim) {
    std::string sizes_str = "[";
    for (size_t i = 0; i < sz.size(); ++i) { sizes_str += std::to_string(sz[i]); if (i+1<sz.size()) sizes_str += ", "; }
    sizes_str += "]";
    throw std::out_of_range("select(): index " + std::to_string(index) +
                            " out of range for tensor of size " + sizes_str +
                            " at dimension " + std::to_string(dim));
  }
  if (idx < 0) idx += size_dim;

  std::vector<int64_t> new_sizes; new_sizes.reserve(sz.size()>0?sz.size()-1:0);
  std::vector<int64_t> new_strides; new_strides.reserve(st.size()>0?st.size()-1:0);
  for (int64_t d = 0; d < nd; ++d) {
    if (d == dim) continue;
    new_sizes.push_back(sz[static_cast<std::size_t>(d)]);
    new_strides.push_back(st[static_cast<std::size_t>(d)]);
  }
  const int64_t new_offset = self.storage_offset() + idx * st[static_cast<std::size_t>(dim)];
  return self.as_strided(new_sizes, new_strides, new_offset);
}

TensorImpl narrow(const TensorImpl& self, int64_t dim, int64_t start, int64_t length) {
  const auto& sz = self.sizes();
  const auto& st = self.strides();
  const int64_t nd = static_cast<int64_t>(sz.size());
  if (nd == 0) {
    throw std::invalid_argument("narrow() cannot be applied to a 0-dim tensor.");
  }
  if (length < 0) {
    throw std::invalid_argument("narrow(): length must be non-negative.");
  }
  dim = wrap_dim(dim, nd);
  auto cur_size = sz[static_cast<std::size_t>(dim)];
  if (!( -cur_size <= start && start <= cur_size )) {
    throw std::out_of_range("start out of range (expected to be in range of [" +
                            std::to_string(-cur_size) + ", " + std::to_string(cur_size) +
                            "], but got " + std::to_string(start) + ")");
  }
  int64_t s = start;
  if (s < 0) s = s + cur_size;
  if (s + length > cur_size) {
    throw std::out_of_range("start (" + std::to_string(start) + ") + length (" + std::to_string(length) + ") exceeds dimension size (" + std::to_string(cur_size) + ").");
  }
  std::vector<int64_t> new_sizes = sz;
  new_sizes[static_cast<std::size_t>(dim)] = length;
  const int64_t new_offset = self.storage_offset() + s * st[static_cast<std::size_t>(dim)];
  return self.as_strided(new_sizes, st, new_offset);
}

TensorImpl squeeze(const TensorImpl& self) {
  const auto& sz = self.sizes();
  const auto& st = self.strides();
  std::vector<int64_t> ns; ns.reserve(sz.size());
  std::vector<int64_t> nt; nt.reserve(st.size());
  for (size_t i = 0; i < sz.size(); ++i) {
    if (sz[i] == 1) continue;
    ns.push_back(sz[i]);
    nt.push_back(st[i]);
  }
  return self.as_strided(ns, nt, self.storage_offset());
}

TensorImpl squeeze(const TensorImpl& self, int64_t dim) {
  const int64_t nd = static_cast<int64_t>(self.sizes().size());
  dim = wrap_dim(dim, nd);
  if (self.sizes()[static_cast<std::size_t>(dim)] != 1) {
    return self;
  }
  std::vector<int64_t> ns; ns.reserve(self.sizes().size()-1);
  std::vector<int64_t> nt; nt.reserve(self.strides().size()-1);
  for (int64_t d = 0; d < nd; ++d) {
    if (d == dim) continue;
    ns.push_back(self.sizes()[static_cast<std::size_t>(d)]);
    nt.push_back(self.strides()[static_cast<std::size_t>(d)]);
  }
  return self.as_strided(ns, nt, self.storage_offset());
}

TensorImpl squeeze(const TensorImpl& self, const std::vector<int64_t>& dims) {
  std::unordered_set<int64_t> keep;
  const int64_t nd = static_cast<int64_t>(self.sizes().size());
  keep.reserve(dims.size());
  for (auto d : dims) {
    int64_t wd = wrap_dim(d, nd);
    keep.insert(wd);
  }
  std::vector<int64_t> ns; ns.reserve(self.sizes().size());
  std::vector<int64_t> nt; nt.reserve(self.strides().size());
  for (int64_t d = 0; d < nd; ++d) {
    if (self.sizes()[static_cast<std::size_t>(d)] == 1 && keep.count(d)) continue;
    ns.push_back(self.sizes()[static_cast<std::size_t>(d)]);
    nt.push_back(self.strides()[static_cast<std::size_t>(d)]);
  }
  return self.as_strided(ns, nt, self.storage_offset());
}

TensorImpl unsqueeze(const TensorImpl& self, int64_t dim) {
  const int64_t nd = static_cast<int64_t>(self.sizes().size());
  if (dim < 0) dim += (nd + 1);
  if (dim < 0 || dim > nd) {
    throw std::out_of_range("Dimension out of range (expected to be in range of [" +
                            std::to_string(-(nd+1)) + ", " + std::to_string(nd) +
                            "], but got " + std::to_string(dim) + ")");
  }
  const auto& sz = self.sizes();
  const auto& st = self.strides();
  std::vector<int64_t> ns = sz; ns.insert(ns.begin() + dim, 1);
  std::vector<int64_t> nt = st;
  int64_t new_stride = 1;
  if (!nt.empty()) {
    if (dim < static_cast<int64_t>(st.size())) {
      new_stride = sz[static_cast<std::size_t>(dim)] * st[static_cast<std::size_t>(dim)];
    } else {
      new_stride = 1;
    }
  }
  nt.insert(nt.begin() + dim, new_stride);
  return self.as_strided(ns, nt, self.storage_offset());
}

TensorImpl permute(const TensorImpl& self, const std::vector<int64_t>& dims) {
  const int64_t nd = static_cast<int64_t>(self.sizes().size());
  if (static_cast<int64_t>(dims.size()) != nd) {
    throw std::invalid_argument("permute(): invalid dims size");
  }
  std::vector<int64_t> used(nd, 0);
  std::vector<int64_t> ns(nd), nt(nd);
  for (int64_t i = 0; i < nd; ++i) {
    int64_t d = dims[static_cast<std::size_t>(i)];
    d = wrap_dim(d, nd);
    if (used[static_cast<std::size_t>(d)]) {
      throw std::invalid_argument("permute(): duplicate dims are not allowed.");
    }
    used[static_cast<std::size_t>(d)] = 1;
    ns[static_cast<std::size_t>(i)] = self.sizes()[static_cast<std::size_t>(d)];
    nt[static_cast<std::size_t>(i)] = self.strides()[static_cast<std::size_t>(d)];
  }
  return self.as_strided(ns, nt, self.storage_offset());
}

TensorImpl transpose(const TensorImpl& self, int64_t dim0, int64_t dim1) {
  const int64_t nd = static_cast<int64_t>(self.sizes().size());
  dim0 = wrap_dim(dim0, nd);
  dim1 = wrap_dim(dim1, nd);
  if (dim0 == dim1) return self;
  std::vector<int64_t> ns = self.sizes();
  std::vector<int64_t> nt = self.strides();
  std::swap(ns[static_cast<std::size_t>(dim0)], ns[static_cast<std::size_t>(dim1)]);
  std::swap(nt[static_cast<std::size_t>(dim0)], nt[static_cast<std::size_t>(dim1)]);
  return self.as_strided(ns, nt, self.storage_offset());
}

TensorImpl expand(const TensorImpl& self, const std::vector<int64_t>& sizes) {
  const auto& in_sizes = self.sizes();
  const auto& in_strides = self.strides();
  const int64_t in_nd = static_cast<int64_t>(in_sizes.size());
  const int64_t out_nd = static_cast<int64_t>(sizes.size());
  if (out_nd < in_nd) {
    throw std::invalid_argument("expand(Tensor,{" + std::to_string(in_nd) + " dims}, size=" + std::to_string(out_nd) + "): the number of sizes provided (" + std::to_string(out_nd) + ") must be greater or equal to the number of dimensions in the tensor (" + std::to_string(in_nd) + ")");
  }
  std::vector<int64_t> out_sizes = sizes;
  std::vector<int64_t> out_strides(out_nd, 0);
  int64_t in_i = in_nd - 1;
  for (int64_t out_i = out_nd - 1; out_i >= 0; --out_i) {
    const int64_t target = sizes[static_cast<std::size_t>(out_i)];
    if (in_i >= 0) {
      const int64_t cur = in_sizes[static_cast<std::size_t>(in_i)];
      if (target == -1) {
        out_sizes[static_cast<std::size_t>(out_i)] = cur;
        out_strides[static_cast<std::size_t>(out_i)] = in_strides[static_cast<std::size_t>(in_i)];
      } else if (cur == target) {
        out_strides[static_cast<std::size_t>(out_i)] = in_strides[static_cast<std::size_t>(in_i)];
      } else if (cur == 1 && target > 1) {
        out_strides[static_cast<std::size_t>(out_i)] = 0;
      } else {
        std::string t_sizes = "["; for (size_t i=0;i<sizes.size();++i){ t_sizes += std::to_string(sizes[i]); if (i+1<sizes.size()) t_sizes += ", "; } t_sizes += "]";
        std::string s_sizes = "["; for (size_t i=0;i<in_sizes.size();++i){ s_sizes += std::to_string(in_sizes[i]); if (i+1<in_sizes.size()) s_sizes += ", "; } s_sizes += "]";
        throw std::invalid_argument("The expanded size of the tensor (" + std::to_string(target) + ") must match the existing size (" + std::to_string(cur) + ") at non-singleton dimension " + std::to_string(in_i) + ".  Target sizes: " + t_sizes + ".  Tensor sizes: " + s_sizes);
      }
      --in_i;
    } else {
      if (target == -1) {
        throw std::invalid_argument("The expanded size of the tensor (-1) isn't allowed in a leading, non-existing dimension " + std::to_string(out_i));
      }
      out_strides[static_cast<std::size_t>(out_i)] = 0;
    }
  }
  return self.as_strided(out_sizes, out_strides, self.storage_offset());
}

static constexpr const char* kViewSizeNotCompatibleMsg =
    "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.";

[[nodiscard]] static inline int64_t numel_or_throw(const TensorImpl& self,
                                                  const char* ctx) {
  const auto& sz = self.sizes();
  if (sz.empty()) {
    return 1;  // scalar
  }

  int64_t n = 1;
  for (int64_t s : sz) {
    if (s < 0) {
      throw std::invalid_argument(std::string(ctx) + ": negative size");
    }
    if (s == 0) {
      return 0;
    }
    int64_t tmp = 0;
    if (!checked_mul_i64(n, s, tmp)) {
      throw std::overflow_error(std::string(ctx) + ": numel overflow");
    }
    n = tmp;
  }

  return n;
}

[[nodiscard]] static inline std::vector<int64_t> compute_contiguous_strides_checked(
    const std::vector<int64_t>& sizes,
    const char* ctx) {
  std::vector<int64_t> st(sizes.size(), 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0;
       --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    st[idx] = acc;

    int64_t dim = sizes[idx];
    if (dim < 0) {
      throw std::invalid_argument(std::string(ctx) + ": negative size");
    }
    if (dim == 0) {
      dim = 1;
    }

    int64_t tmp = 0;
    if (!checked_mul_i64(acc, dim, tmp)) {
      throw std::overflow_error(std::string(ctx) + ": contiguous stride overflow");
    }
    acc = tmp;
  }
  return st;
}

[[nodiscard]] static inline std::string format_shape(
    const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i + 1 < shape.size()) s += ", ";
  }
  s += "]";
  return s;
}

[[nodiscard]] static inline std::vector<int64_t> infer_view_sizes_checked(
    int64_t numel,
    const std::vector<int64_t>& sizes,
    const char* /*ctx*/) {
  const std::string shape_str = format_shape(sizes);
  std::vector<int64_t> out_sizes = sizes;

  int64_t newsize = 1;
  std::optional<std::size_t> infer_dim;

  for (std::size_t i = 0; i < sizes.size(); ++i) {
    const int64_t s = sizes[i];
    if (s == -1) {
      if (infer_dim.has_value()) {
        throw std::invalid_argument("only one dimension can be inferred");
      }
      infer_dim = i;
      continue;
    }
    if (s < -1) {
      throw std::invalid_argument("invalid shape dimension " + std::to_string(s) +
                                  " at index " + std::to_string(i) +
                                  " of shape " + shape_str);
    }
    int64_t tmp = 0;
    if (!checked_mul_i64(newsize, s, tmp)) {
      throw std::overflow_error("shape product overflow");
    }
    newsize = tmp;
  }

  auto throw_invalid_shape = [&]() {
    throw std::invalid_argument("shape '" + shape_str +
                                "' is invalid for input of size " +
                                std::to_string(numel));
  };

  if (infer_dim.has_value()) {
    // `newsize` is the product of known sizes.
    if (!((newsize > 0 && (numel % newsize) == 0) || numel == newsize)) {
      throw_invalid_shape();
    }

    // Match PyTorch error text for ambiguous `-1` inference with numel==0.
    if (newsize == 0) {
      throw std::invalid_argument(
          "cannot reshape tensor of 0 elements into shape " + shape_str +
          " because the unspecified dimension size -1 can be any value and is ambiguous");
    }

    out_sizes[*infer_dim] = numel / newsize;
    return out_sizes;
  }

  if (newsize != numel) {
    throw_invalid_shape();
  }

  return out_sizes;
}

[[nodiscard]] static inline std::optional<std::vector<int64_t>>
compute_view_strides_checked(const std::vector<int64_t>& oldshape,
                            const std::vector<int64_t>& oldstride,
                            const std::vector<int64_t>& newshape,
                            int64_t numel,
                            const char* ctx) {
  // Port of PyTorch at::detail::computeStride_impl
  // (aten/src/ATen/TensorUtils.cpp).

  if (oldshape.size() != oldstride.size()) {
    return std::nullopt;
  }

  if (oldshape.empty()) {
    return std::vector<int64_t>(newshape.size(), 1);
  }

  // NOTE: stride is arbitrary in the numel() == 0 case; match NumPy/PyTorch:
  // copy old strides when the shape matches, otherwise behave like resize.
  if (numel == 0) {
    if (oldshape == newshape) {
      return oldstride;
    }
    return compute_contiguous_strides_checked(newshape, ctx);
  }

  std::vector<int64_t> newstride(newshape.size(), 0);

  int64_t view_d = static_cast<int64_t>(newshape.size()) - 1;
  // Stride for each subspace in the chunk.
  int64_t chunk_base_stride = oldstride.back();
  // Numel in current chunk.
  int64_t tensor_numel = 1;
  int64_t view_numel = 1;

  for (int64_t tensor_d = static_cast<int64_t>(oldshape.size()) - 1;
       tensor_d >= 0;
       --tensor_d) {
    int64_t tmp = 0;
    if (!checked_mul_i64(tensor_numel, oldshape[static_cast<std::size_t>(tensor_d)], tmp)) {
      return std::nullopt;
    }
    tensor_numel = tmp;

    bool end_of_chunk = false;
    if (tensor_d == 0) {
      end_of_chunk = true;
    } else if (oldshape[static_cast<std::size_t>(tensor_d - 1)] != 1) {
      int64_t expected = 0;
      const bool ok = checked_mul_i64(tensor_numel, chunk_base_stride, expected);
      if (!ok || oldstride[static_cast<std::size_t>(tensor_d - 1)] != expected) {
        end_of_chunk = true;
      }
    }

    if (end_of_chunk) {
      while (view_d >= 0 &&
             (view_numel < tensor_numel ||
              newshape[static_cast<std::size_t>(view_d)] == 1)) {
        int64_t st = 0;
        if (!checked_mul_i64(view_numel, chunk_base_stride, st)) {
          return std::nullopt;
        }
        newstride[static_cast<std::size_t>(view_d)] = st;

        if (!checked_mul_i64(view_numel,
                             newshape[static_cast<std::size_t>(view_d)],
                             tmp)) {
          return std::nullopt;
        }
        view_numel = tmp;
        --view_d;
      }

      if (view_numel != tensor_numel) {
        return std::nullopt;
      }

      if (tensor_d > 0) {
        chunk_base_stride = oldstride[static_cast<std::size_t>(tensor_d - 1)];
        tensor_numel = 1;
        view_numel = 1;
      }
    }
  }

  if (view_d != -1) {
    return std::nullopt;
  }

  return newstride;
}

TensorImpl view(const TensorImpl& self, const std::vector<int64_t>& sizes) {
  const int64_t ne = numel_or_throw(self, "view");
  std::vector<int64_t> out_sizes = infer_view_sizes_checked(ne, sizes, "view");

  auto st = compute_view_strides_checked(
      self.sizes(), self.strides(), out_sizes, ne, "view");
  if (!st.has_value()) {
    throw std::invalid_argument(kViewSizeNotCompatibleMsg);
  }

  return self.as_strided(out_sizes, *st, self.storage_offset());
}

TensorImpl reshape(const TensorImpl& self, const std::vector<int64_t>& sizes) {
  const int64_t ne = numel_or_throw(self, "reshape");
  std::vector<int64_t> out_sizes = infer_view_sizes_checked(ne, sizes, "reshape");

  auto st = compute_view_strides_checked(
      self.sizes(), self.strides(), out_sizes, ne, "reshape");
  if (st.has_value()) {
    return self.as_strided(out_sizes, *st, self.storage_offset());
  }

  TensorImpl tmp = vbt::core::clone_contiguous_same_device(self);
  auto dense_strides = compute_contiguous_strides_checked(out_sizes, "reshape");
  return tmp.as_strided(out_sizes, dense_strides, tmp.storage_offset());
}

TensorImpl view_as_real(const TensorImpl& self) {
  const ScalarType in_dtype = self.dtype();
  if (!is_complex(in_dtype)) {
    throw std::invalid_argument("view_as_real: expected a complex tensor");
  }
  if (self.is_conj()) {
    throw std::invalid_argument(
        "view_as_real: cannot view a conjugated complex tensor; call resolve_conj()"
    );
  }

  const auto& in_sizes = self.sizes();
  const auto& in_strides = self.strides();
  const std::size_t nd = in_sizes.size();

  std::vector<int64_t> out_sizes = in_sizes;
  out_sizes.push_back(2);

  std::vector<int64_t> out_strides;
  out_strides.reserve(nd + 1);
  for (std::size_t i = 0; i < nd; ++i) {
    int64_t tmp = 0;
    if (!checked_mul_i64(in_strides[i], 2, tmp)) {
      throw std::overflow_error("view_as_real: stride*2 overflow");
    }
    out_strides.push_back(tmp);
  }
  out_strides.push_back(1);

  int64_t out_offset = 0;
  if (!checked_mul_i64(self.storage_offset(), 2, out_offset)) {
    throw std::overflow_error("view_as_real: storage_offset*2 overflow");
  }

  const ScalarType out_dtype = to_real_value_type(in_dtype);
  return self.as_strided_dtype_(out_sizes, out_strides, out_offset, out_dtype);
}

TensorImpl view_as_complex(const TensorImpl& self) {
  const ScalarType in_dtype = self.dtype();
  if (!(in_dtype == ScalarType::Float32 || in_dtype == ScalarType::Float64)) {
    throw std::invalid_argument("view_as_complex: expected a float32 or float64 tensor");
  }

  const auto& in_sizes = self.sizes();
  const auto& in_strides = self.strides();
  if (in_sizes.empty()) {
    throw std::invalid_argument("view_as_complex: expected tensor with rank >= 1");
  }
  if (in_sizes.back() != 2) {
    throw std::invalid_argument("view_as_complex: last dimension size must be 2");
  }
  if (in_strides.back() != 1) {
    throw std::invalid_argument("view_as_complex: last dimension stride must be 1");
  }

  std::vector<int64_t> out_sizes(in_sizes.begin(), in_sizes.end() - 1);
  std::vector<int64_t> out_strides;
  out_strides.reserve(in_sizes.size() - 1);
  for (std::size_t i = 0; i + 1 < in_strides.size(); ++i) {
    const int64_t st = in_strides[i];
    if ((st % 2) != 0) {
      throw std::invalid_argument("view_as_complex: stride must be divisible by 2");
    }
    out_strides.push_back(st / 2);
  }

  const int64_t in_off = self.storage_offset();
  if ((in_off % 2) != 0) {
    throw std::invalid_argument("view_as_complex: storage_offset must be divisible by 2");
  }
  const int64_t out_off = in_off / 2;

  const ScalarType out_dtype = to_complex_type(in_dtype);
  if (out_dtype == ScalarType::Undefined) {
    throw std::invalid_argument("view_as_complex: unsupported dtype");
  }

  // Alignment check (only for non-empty output).
  const int64_t out_numel = self.numel() / 2;
  if (out_numel > 0) {
    const void* p = self.data();
    if (!p) {
      throw std::runtime_error(
          "view_as_complex: null data pointer for non-empty tensor");
    }
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
    const std::size_t align = required_alignment_bytes(out_dtype);
    if (align != 0 && (addr % static_cast<std::uintptr_t>(align)) != 0) {
      throw std::runtime_error(
          "view_as_complex: data pointer is not aligned for complex dtype");
    }
  }

  return self.as_strided_dtype_(out_sizes, out_strides, out_off, out_dtype);
}

TensorImpl conj(const TensorImpl& self) {
  const ScalarType dt = self.dtype();
  if (!is_complex(dt)) {
    // Conjugation is a no-op for real tensors.
    return self;
  }

  TensorImpl out = self.as_strided(self.sizes(), self.strides(), self.storage_offset());
  out.flags_ ^= TensorImpl::kConj;
  return out;
}

TensorImpl resolve_conj(const TensorImpl& self) {
  if (!is_complex(self.dtype()) || !self.is_conj()) {
    return self;
  }
  // clone_* is expected to materialize the conjugation and clear the conj bit.
  return vbt::core::clone_contiguous_same_device(self);
}

} // namespace core
} // namespace vbt
