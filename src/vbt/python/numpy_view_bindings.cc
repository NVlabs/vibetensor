// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

  // Alternate entry that takes ndarray directly (NumPy) to avoid cast issues
  m.def("_cpu_tensor_from_numpy_view_alt", [](const nb::ndarray<nb::numpy>& arr, std::string dtype_token){
    if (!arr.is_valid()) throw nb::value_error("expected a valid NumPy ndarray");
    // For warnings and dtype/stride attributes, borrow the original object via cast
    nb::object arr_obj = arr.cast<nb::object>();

    static std::atomic<bool> warned{false};
    bool writeable = false;
    try { writeable = nb::cast<bool>(arr_obj.attr("flags").attr("writeable")); } catch (...) { writeable = false; }
    if (!writeable) {
      bool expected = false;
      if (warned.compare_exchange_strong(expected, true)) {
        try {
          auto warnings = nb::module_::import_("warnings");
          auto builtins = nb::module_::import_("builtins");
          warnings.attr("warn")(nb::str(
            "The given NumPy array is not writable, and VibeTensor does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program."
          ), builtins.attr("UserWarning"), 2);
        } catch (...) { /* best-effort only */ }
      }
    }
    ScalarType st = dtype_from_token(dtype_token);
    std::size_t item_b = (std::size_t) vbt::core::itemsize(st);
    std::size_t got = 0;
    try { got = (std::size_t) nb::cast<int>(arr_obj.attr("dtype").attr("itemsize")); } catch (...) { got = 0; }
    if (item_b != got) throw nb::value_error("dtype itemsize mismatch");

    const size_t ndim = arr.ndim();
    std::vector<int64_t> sizes(ndim);
    for (size_t i = 0; i < ndim; ++i) sizes[i] = (int64_t) arr.shape(i);

    // Byte order enforcement
    try {
      std::string bo = nb::cast<std::string>(arr_obj.attr("dtype").attr("byteorder"));
      if (!bo.empty()) {
        char c = bo[0];
        if (c != '|' && c != '=') {
          bool little = true;
          try { auto sys = nb::module_::import_("sys"); std::string sb = nb::cast<std::string>(sys.attr("byteorder")); little = (sb == "little"); } catch (...) { little = true; }
          if ((c == '<' && !little) || (c == '>' && little)) {
            throw nb::value_error("given numpy array has byte order different from the native byte order. Conversion between byte orders is currently not supported.");
          }
        }
      }
    } catch (const nb::python_error&) { /* ignore */ }

    std::vector<int64_t> stride_bytes; stride_bytes.reserve(ndim);
    try {
      nb::object pystrides = arr_obj.attr("strides");
      if (PyTuple_Check(pystrides.ptr()) || PyList_Check(pystrides.ptr())) {
        nb::sequence seq = nb::borrow<nb::sequence>(pystrides);
        for (size_t i = 0; i < ndim; ++i) { int64_t sb = nb::cast<int64_t>(seq[i]); stride_bytes.push_back(sb); }
      } else { throw nb::type_error("bad strides"); }
    } catch (...) { for (size_t i = 0; i < ndim; ++i) stride_bytes.push_back((int64_t) arr.stride(i)); }

    bool has_zero = false; for (auto s : sizes) { if (s == 0) { has_zero = true; break; } }
    if (has_zero) {
      auto storage = vbt::core::make_intrusive<vbt::core::Storage>(vbt::core::DataPtr(nullptr, nullptr), 0);
      std::vector<int64_t> elem_strides(ndim); for (size_t i = 0; i < ndim; ++i) elem_strides[i] = (int64_t)(stride_bytes[i] / (int64_t)item_b);
      return TensorImpl(storage, sizes, elem_strides, 0, st, Device::cpu());
    }

    for (size_t i = 0; i < ndim; ++i) { int64_t sb = stride_bytes[i]; if ((sb % (int64_t) item_b) != 0) { throw nb::value_error("given numpy array strides not a multiple of the element byte size. Copy the numpy array to reallocate the memory."); } }
    for (size_t i = 0; i < ndim; ++i) { if (stride_bytes[i] < 0) { throw nb::value_error("At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) "); } }

    std::vector<int64_t> elem_strides(ndim); for (size_t i = 0; i < ndim; ++i) elem_strides[i] = (int64_t)(stride_bytes[i] / (int64_t)item_b);
    bool ok = true; int64_t min_elem_off = 0, max_elem_off = 0;
    for (size_t i = 0; i < ndim; ++i) { int64_t n = sizes[i]; int64_t d = n > 0 ? (n - 1) : 0; if (d == 0) continue; int64_t st_e = elem_strides[i]; int64_t term = 0; if (!vbt::core::checked_mul_i64(st_e, d, term)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)"); if (st_e >= 0) { int64_t tmp = 0; if (!vbt::core::checked_add_i64(max_elem_off, term, tmp)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)"); max_elem_off = tmp; } else { int64_t tmp = 0; if (!vbt::core::checked_add_i64(min_elem_off, term, tmp)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)"); min_elem_off = tmp; } }
    int64_t max_plus_one = 0; if (!vbt::core::checked_add_i64(max_elem_off, 1, max_plus_one)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)");
    int64_t lower_bytes = 0; if (!vbt::core::checked_mul_i64(min_elem_off, (int64_t)item_b, lower_bytes)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)");
    int64_t upper_excl = 0; if (!vbt::core::checked_mul_i64(max_plus_one, (int64_t)item_b, upper_excl)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)");
    int64_t req_bytes = 0; if (!vbt::core::checked_add_i64(upper_excl, -lower_bytes, req_bytes)) throw nb::value_error("from_numpy: span overflow (stride*extent/add)");

    auto p_eff = reinterpret_cast<std::uintptr_t>(arr.data());
    auto base_ptr = checked_ptr_add_signed(p_eff, lower_bytes, ok);
    if (!ok) throw nb::value_error("from_numpy: pointer base overflow");
#if VBT_REQUIRE_NUMPY_ALIGNMENT
    if ((base_ptr % (std::uintptr_t)item_b) != 0) { throw nb::value_error("from_numpy: effective pointer is not aligned to itemsize (NumPy)"); }
#endif
    int64_t storage_offset_elems = -min_elem_off;
    nb::object owner = arr_obj;
    vbt::core::DataPtr dp(reinterpret_cast<void*>(base_ptr), [owner](void*) mutable { nb::gil_scoped_acquire gil; owner = nb::object(); });
    auto storage = vbt::core::make_intrusive<vbt::core::Storage>(std::move(dp), (std::size_t) req_bytes);
    return TensorImpl(storage, sizes, elem_strides, storage_offset_elems, st, Device::cpu());
  }, nb::arg("array"), nb::arg("dtype"));
