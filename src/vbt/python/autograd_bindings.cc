// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <string>

#include "vbt/core/complex.h"
#include "vbt/core/tensor.h"
#include "vbt/core/tensor_ops.h"
#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/autograd/engine.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/dtype_policy.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/forward.h"
#include "vbt/autograd/saved_variable.h"
#include "vbt/autograd/detail/stats_internal.h"
#endif

#if VBT_WITH_CUDA
#include "vbt/cuda/event.h"
#include "vbt/cuda/graphs.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#endif

namespace nb = nanobind;

#if VBT_WITH_AUTOGRAD
namespace vbt { namespace autograd {

using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

struct PyFunctionPayload {
  nb::object fn_cls;                 // Python Function subclass
  nb::dict ctx_state;                // non-tensor ctx attributes
  std::vector<SavedVariable> saved;  // tensors from save_for_backward
  std::vector<bool> needs_input_grad;      // per positional arg
  std::vector<std::int64_t> edge_index_by_arg; // positional -> input index or -1
};

class PyFunctionNode final : public Node, public ValidatableNode {
 public:
  PyFunctionNode(std::string name,
                 std::vector<InputMeta> in_meta,
                 PyFunctionPayload payload,
                 bool is_cuda,
                 int output_cuda_index = -1)
      : input_meta_(std::move(in_meta)), payload_(std::move(payload)), is_cuda_(is_cuda) {
    this->name = std::move(name);
#if VBT_WITH_CUDA
    if (is_cuda_) {
      // Match wrapper-node semantics: snapshot the current stream on the node's
      // canonical CUDA device at creation time.
      int cuda_index = -1;
      for (const auto& m : input_meta_) {
        if (m.device.type != kDLCUDA) {
          continue;
        }
        if (cuda_index < 0) {
          cuda_index = m.device.index;
        } else if (cuda_index != m.device.index) {
          throw std::runtime_error(
              "PyFunctionNode: mixed CUDA devices in backward node metadata");
        }
      }

      if (output_cuda_index >= 0) {
        if (cuda_index < 0) {
          cuda_index = output_cuda_index;
        } else if (cuda_index != output_cuda_index) {
          throw std::runtime_error(
              "PyFunctionNode: output CUDA device does not match input metadata");
        }
      }

      if (cuda_index < 0) {
        throw std::runtime_error(
            "PyFunctionNode: CUDA node missing CUDA device index");
      }

      using vbt::cuda::DeviceIndex;
      using vbt::cuda::Stream;
      const auto dev_idx = static_cast<DeviceIndex>(cuda_index);
      Stream S = vbt::cuda::getCurrentStream(dev_idx);

      NodeStreamInfo& si = this->mutable_stream_info();
      si.has_canonical_stream = true;
      si.device = vbt::core::Device::cuda(cuda_index);
      si.stream_id = S.id();
    }
#else
    (void)output_cuda_index;
#endif
  }

  std::uint32_t num_inputs() const noexcept override {
    return static_cast<std::uint32_t>(input_meta_.size());
  }

  // PyFunctionNode always has a single output, so only 1 incoming grad slot
  std::uint32_t num_incoming_grad_slots() const noexcept override {
    return 1;
  }

  const std::vector<InputMeta>& input_metas() const noexcept override {
    return input_meta_;
  }

  StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? StreamKind::CudaAllowlisted : StreamKind::CpuOnly;
  }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    NoGradGuard ng;
    _stats_py_function_node_applied(1);

    try {
      nb::gil_scoped_acquire gil;

      // Convert incoming gradient to Python object (single output).
      nb::object grad_output_py;
      if (grads_in.empty() || !grads_in[0].has_value()) {
        grad_output_py = nb::none();
      } else {
        grad_output_py = nb::cast(grads_in[0].value());
      }

      // Reconstruct ctx for backward.
      nb::object autograd_mod = nb::module_::import_("vibetensor.autograd");
      nb::object ctx_cls = autograd_mod.attr("_FunctionCtx");

      const std::size_t num_args = payload_.needs_input_grad.size();
      nb::list needs_list;
      for (std::size_t i = 0; i < num_args; ++i) {
        needs_list.append(nb::bool_(payload_.needs_input_grad[i]));
      }

      nb::object ctx_obj = ctx_cls(nb::tuple(needs_list), nb::str("backward"));

      // Unpack SavedVariables into ctx._saved_unpacked.
      nb::list saved_list;
      for (std::size_t i = 0; i < payload_.saved.size(); ++i) {
        TensorImpl t = payload_.saved[i].unpack();
        saved_list.append(nb::cast(t));
      }
      ctx_obj.attr("_saved_unpacked") = nb::tuple(saved_list);
      ctx_obj.attr("_saved_called") = nb::bool_(!payload_.saved.empty());

      // Restore non-tensor attributes from ctx_state.
      for (auto item : payload_.ctx_state) {
        nb::object key = nb::cast<nb::object>(item.first);
        nb::object value = nb::cast<nb::object>(item.second);
        std::string attr_name = nb::cast<std::string>(key);
        ctx_obj.attr(attr_name.c_str()) = value;
      }

      // Call Python backward.
      nb::object backward_fn = payload_.fn_cls.attr("backward");
      nb::object result = backward_fn(ctx_obj, grad_output_py);

      // Normalize Python gradients to a flat sequence.
      nb::list py_list;
      if (result.is_none()) {
        py_list = nb::list();
      } else if (nb::isinstance<nb::tuple>(result) || nb::isinstance<nb::list>(result)) {
        py_list = nb::list(result);
      } else {
        std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
        throw std::runtime_error(
            "Custom Function backward for " + cls_name +
            " must return a tuple or list of gradients");
      }

      const std::size_t got = static_cast<std::size_t>(py_list.size());
      const std::size_t expected = num_args;
      if (got != expected) {
        std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
        throw std::runtime_error(
            "Custom Function backward for " + cls_name + " returned " +
            std::to_string(got) + " gradients for " + std::to_string(expected) + " inputs");
      }

      std::vector<OptionalTensor> grads_out(input_meta_.size());

      // Map per-arg gradients to differentiable tensor inputs.
      for (std::size_t arg_idx = 0; arg_idx < num_args; ++arg_idx) {
        const std::int64_t edge_idx =
            (arg_idx < payload_.edge_index_by_arg.size())
                ? payload_.edge_index_by_arg[arg_idx]
                : static_cast<std::int64_t>(-1);
        if (edge_idx < 0) {
          continue;  // non-tensor or non-differentiable input
        }

        if (!payload_.needs_input_grad[arg_idx]) {
          continue;  // gradient ignored for non-differentiable inputs
        }

        nb::object g_obj = py_list[arg_idx];
        if (g_obj.is_none()) {
          continue;  // treat as zero gradient
        }

        if (!nb::isinstance<TensorImpl>(g_obj)) {
          std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
          throw std::runtime_error(
              "Custom Function backward for " + cls_name +
              " must return VibeTensor tensors or None for tensor inputs");
        }

        TensorImpl g = nb::cast<TensorImpl>(g_obj);
        const std::size_t idx = static_cast<std::size_t>(edge_idx);
        if (idx >= input_meta_.size()) {
          throw std::runtime_error(
              "Custom Function internal error: edge_index_by_arg out of range");
        }

        const InputMeta& m = input_meta_[idx];
        if (g.dtype() != m.dtype) {
          std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
          throw std::runtime_error(
              "Custom Function backward for " + cls_name +
              " produced a gradient with wrong dtype for input " +
              std::to_string(arg_idx));
        }
        if (g.device() != m.device) {
          std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
          throw std::runtime_error(
              "Custom Function backward for " + cls_name +
              " produced a gradient with wrong device for input " +
              std::to_string(arg_idx));
        }
        if (m.is_strided_dense && !g.is_non_overlapping_and_dense()) {
          std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
          throw std::runtime_error(
              "Custom Function backward for " + cls_name +
              " produced a gradient with non-dense layout for input " +
              std::to_string(arg_idx));
        }
        if (g.sizes() != m.sizes) {
          std::string cls_name = nb::cast<std::string>(payload_.fn_cls.attr("__name__"));
          throw std::runtime_error(
              "Custom Function backward for " + cls_name +
              " produced a gradient with mismatched shape for input " +
              std::to_string(arg_idx));
        }

        grads_out[idx] = std::move(g);
      }

      return grads_out;
    } catch (...) {
      _stats_py_function_backward_failed(1);
      throw;
    }
  }

 private:
  std::vector<InputMeta> input_meta_;
  PyFunctionPayload payload_;
  bool is_cuda_{false};
};

void create_py_function_node(
    nb::object fn_cls,
    const nb::tuple& tensor_inputs,
    TensorImpl& output_tensor,
    const nb::tuple& saved_tensors,
    const nb::tuple& needs_input_grad,
    const nb::tuple& edge_index_by_arg,
    const nb::dict& ctx_state) {

  // Domain: CPU float32 tensors; CUDA float32/float16 when CUDA autograd is enabled.
#if VBT_WITH_CUDA
  const bool cuda_autograd_enabled = is_streaming_backwards_enabled();
#else
  const bool cuda_autograd_enabled = false;
#endif

  const auto out_dev = output_tensor.device();
  const bool output_is_cuda = out_dev.type == kDLCUDA;
  const int output_cuda_index = output_is_cuda ? out_dev.index : -1;

  if (output_is_cuda) {
    if (!cuda_autograd_enabled) {
      throw std::runtime_error(
          "vibetensor.autograd.Function: CUDA output requires CUDA autograd to be enabled");
    }
    if (!is_cuda_autograd_dtype_supported(output_tensor.dtype())) {
      throw std::runtime_error(
          "vibetensor.autograd.Function: CUDA output must be float32/float16");
    }
  } else if (out_dev.type == kDLCPU) {
    if (output_tensor.dtype() != vbt::core::ScalarType::Float32) {
      throw std::runtime_error(
          "vibetensor.autograd.Function: CPU output must be float32");
    }
  } else {
    throw std::runtime_error(
        "vibetensor.autograd.Function: output tensor must be CPU or CUDA");
  }

  std::vector<TensorImpl> inputs_impl;
  inputs_impl.reserve(tensor_inputs.size());
  for (std::size_t i = 0; i < tensor_inputs.size(); ++i) {
    TensorImpl t = nb::cast<TensorImpl>(tensor_inputs[i]);
    if (output_is_cuda) {
      if (!is_cuda_autograd_dtype_supported(t.dtype())) {
        throw std::runtime_error(
            "vibetensor.autograd.Function: differentiable CUDA inputs must be float32/float16");
      }
    } else {
      if (t.dtype() != vbt::core::ScalarType::Float32) {
        throw std::runtime_error(
            "vibetensor.autograd.Function: differentiable CPU inputs must be float32");
      }
    }

    const auto in_dev = t.device();
    if (in_dev.type != kDLCPU && in_dev.type != kDLCUDA) {
      throw std::runtime_error(
          "vibetensor.autograd.Function: differentiable inputs must be CPU or CUDA");
    }
    if (in_dev != out_dev) {
      throw std::runtime_error(
          "vibetensor.autograd.Function: differentiable inputs must be on the same device as output");
    }

    inputs_impl.push_back(std::move(t));
  }

  const bool is_cuda = output_is_cuda;

  std::vector<InputMeta> in_meta;
  in_meta.reserve(inputs_impl.size());
  for (const auto& t : inputs_impl) {
    in_meta.emplace_back(InputMeta::from_tensor(t));
  }

  // Snapshot saved tensors (if any).
  std::vector<SavedVariable> snaps;
  snaps.reserve(saved_tensors.size());
  for (std::size_t i = 0; i < saved_tensors.size(); ++i) {
    TensorImpl t = nb::cast<TensorImpl>(saved_tensors[i]);
    const auto dev = t.device();
    if (dev.type == kDLCUDA) {
      if (!output_is_cuda || dev.index != output_cuda_index) {
        throw std::runtime_error(
            "vibetensor.autograd.Function: CUDA saved tensors must be on the same device as differentiable inputs");
      }
    } else if (dev.type != kDLCPU) {
      throw std::runtime_error(
          "vibetensor.autograd.Function: saved tensors must be CPU or CUDA");
    }
    snaps.emplace_back(SavedVariable(t));
  }

  // Convert needs_input_grad / edge_index_by_arg to C++ vectors.
  std::vector<bool> needs_vec;
  needs_vec.reserve(needs_input_grad.size());
  for (std::size_t i = 0; i < needs_input_grad.size(); ++i) {
    needs_vec.push_back(nb::cast<bool>(needs_input_grad[i]));
  }

  std::vector<std::int64_t> edge_vec;
  edge_vec.reserve(edge_index_by_arg.size());
  for (std::size_t i = 0; i < edge_index_by_arg.size(); ++i) {
    edge_vec.push_back(nb::cast<std::int64_t>(edge_index_by_arg[i]));
  }

  PyFunctionPayload payload;
  payload.fn_cls = std::move(fn_cls);
  payload.ctx_state = ctx_state;
  payload.saved = std::move(snaps);
  payload.needs_input_grad = std::move(needs_vec);
  payload.edge_index_by_arg = std::move(edge_vec);

  auto node = intrusive_ptr<Node>(
      new PyFunctionNode(
          nb::cast<std::string>(payload.fn_cls.attr("__name__")) + "Backward",
          std::move(in_meta),
          std::move(payload),
          is_cuda,
          output_cuda_index),
      /*add_ref=*/true);

  // Ensure next_edges sized to number of differentiable tensor inputs.
  node->next_edges.resize(node->num_inputs());

  // Wire next edges for differentiable inputs.
  {
    std::unordered_map<const AutogradMeta*, intrusive_ptr<Node>> sinks;
    const std::size_t n = inputs_impl.size();
    for (std::size_t i = 0; i < n; ++i) {
      const TensorImpl& ti = inputs_impl[i];
      node->next_edges[i] = resolve_edge_for_tensor(ti, sinks);
    }
  }

  // Attach node to output via rebase_history and mark requires_grad.
  set_requires_grad(output_tensor, true);
  rebase_history(output_tensor, node);

  _stats_py_function_node_created(1);
}

}} // namespace vbt::autograd
#endif

namespace vbt_py {

using vbt::core::TensorImpl;
#if VBT_WITH_AUTOGRAD
using vbt::core::intrusive_ptr;

struct FunctionShim { };

// Opaque handle types surfaced to Python via _C.autograd.
struct GradFnHandle {
  intrusive_ptr<vbt::autograd::Node> fn;
};

struct HookHandle {
  intrusive_ptr<vbt::autograd::TensorHook> hook;
};

// Python-backed TensorHook implementation that wraps a callable.
struct PythonTensorHook final : public vbt::autograd::TensorHook {
  nb::object cb;
  explicit PythonTensorHook(nb::object callable) : cb(std::move(callable)) {}

  void call(const TensorImpl& grad) override {
    nb::gil_scoped_acquire gil;
    if (!cb.is_valid()) {
      return;
    }

    nb::object arg;
    try {
      // Prefer to surface a torch.Tensor to hooks when torch is
      // available so that numpy.from_dlpack() sees a writable view of
      // the gradient, matching PyTorch behavior.
      nb::object vt_mod = nb::module_::import_("vibetensor.torch");
      nb::object torch_mod = nb::module_::import_("torch");
      nb::object to_dlpack = vt_mod.attr("to_dlpack");
      nb::object cap = to_dlpack(grad);
      nb::object from_dlpack = torch_mod.attr("from_dlpack");
      nb::object t = from_dlpack(cap);
      // Ensure requires_grad is False on the hook tensor.
      try {
        if (nb::cast<bool>(t.attr("requires_grad"))) {
          t.attr("requires_grad_")(false);
        }
      } catch (...) {
      }
      arg = std::move(t);
    } catch (...) {
      // Best-effort fallback: expose the raw VibeTensor tensor when
      // torch or the DLPack bridge are unavailable.
      arg = nb::cast(grad);
    }

    cb(arg);
  }
};

// Thread-local guard to prevent nested backward() on the same thread.
static thread_local bool g_backward_in_progress = false;

inline bool complex_autograd_enabled_from_env() noexcept {
  const char* complex_raw = std::getenv("VBT_ENABLE_COMPLEX");
  if (!(complex_raw && complex_raw[0] == '1' && complex_raw[1] == '\0')) {
    return false;
  }
  const char* raw = std::getenv("VBT_ENABLE_COMPLEX_AUTOGRAD");
  return raw && raw[0] == '1' && raw[1] == '\0';
}

// Internal helper implementing the core backward logic without the
// re-entrancy guard.
static void tensor_backward_impl(TensorImpl& self, nb::object gradient, bool /*retain_graph*/) {
  // If no grad_fn attached, nothing to do
  auto* meta = vbt::autograd::get_autograd_meta(self, /*create_if_missing=*/false);
  if (!meta || !meta->grad_fn) return;

  auto dev = self.device();

  // Seed gradient tensor g on the same device as self.
  vbt::core::TensorImpl g;

  if (dev.type == kDLCPU) {
    const auto dtype = self.dtype();
    const bool is_complex =
        dtype == vbt::core::ScalarType::Complex64 ||
        dtype == vbt::core::ScalarType::Complex128;

    if (is_complex && !complex_autograd_enabled_from_env()) {
      PyErr_SetString(
          PyExc_ValueError,
          "backward(): complex autograd is disabled; set VBT_ENABLE_COMPLEX_AUTOGRAD=1");
      throw nb::python_error();
    }

    if (dtype != vbt::core::ScalarType::Float32 && !is_complex) {
      PyErr_SetString(PyExc_ValueError,
                      "backward(): accumulation only supported for Float32 CPU in this build");
      throw nb::python_error();
    }

    if (gradient.is_none()) {
      // Implicit seed is only supported for roots with exactly one element
      // (scalars or tensors with numel == 1). All other non-scalar roots
      // must provide an explicit gradient.
      if (!(self.sizes().empty() || self.numel() == 1)) {
        PyErr_SetString(PyExc_ValueError,
                        "backward(): gradient dtype/device/shape must match tensor");
        throw nb::python_error();
      }

      // Build a CPU tensor shaped like `self` and filled with 1.
      vbt::core::TensorImpl g_like =
          vbt::core::clone_contiguous_same_device(self);
      const std::size_t ne =
          static_cast<std::size_t>(g_like.numel() < 0 ? 0 : g_like.numel());

      if (dtype == vbt::core::ScalarType::Float32) {
        float* p = static_cast<float*>(g_like.data());
        for (std::size_t i = 0; i < ne; ++i) {
          p[i] = 1.0f;
        }
      } else if (dtype == vbt::core::ScalarType::Complex64) {
        auto* p = static_cast<vbt::core::Complex<float>*>(g_like.data());
        for (std::size_t i = 0; i < ne; ++i) {
          p[i].re = 1.0f;
          p[i].im = 0.0f;
        }
      } else if (dtype == vbt::core::ScalarType::Complex128) {
        auto* p = static_cast<vbt::core::Complex<double>*>(g_like.data());
        for (std::size_t i = 0; i < ne; ++i) {
          p[i].re = 1.0;
          p[i].im = 0.0;
        }
      }

      g = std::move(g_like);
    } else {
      TensorImpl grad = nb::cast<TensorImpl>(gradient);
      if (grad.dtype() != self.dtype() || grad.device() != self.device() ||
          grad.sizes() != self.sizes()) {
        PyErr_SetString(PyExc_ValueError,
                        "backward(): gradient dtype/device/shape must match tensor");
        throw nb::python_error();
      }
      g = grad;
    }
  }
#if VBT_WITH_CUDA
  else if (dev.type == kDLCUDA) {
    // CUDA: supported floating dtypes only and CUDA autograd must be enabled.
    if (!vbt::autograd::is_cuda_autograd_dtype_supported(self.dtype())) {
      PyErr_SetString(PyExc_ValueError,
                      "backward(): accumulation only supported for Float32/Float16 CUDA in this build");
      throw nb::python_error();
    }
    if (!vbt::autograd::is_streaming_backwards_enabled()) {
      PyErr_SetString(PyExc_ValueError,
                      "backward(): accumulation only supported for Float32/Float16 CUDA when CUDA autograd is enabled in this build");
      throw nb::python_error();
    }

    if (gradient.is_none()) {
      PyErr_SetString(PyExc_ValueError,
                      "backward(): gradient dtype/device/shape must match tensor");
      throw nb::python_error();
    }

    TensorImpl grad = nb::cast<TensorImpl>(gradient);
    if (grad.dtype() != self.dtype() || grad.device() != self.device() ||
        grad.sizes() != self.sizes()) {
      PyErr_SetString(PyExc_ValueError,
                      "backward(): gradient dtype/device/shape must match tensor");
      throw nb::python_error();
    }
    g = grad;
  }
#endif
  else {
    PyErr_SetString(PyExc_ValueError,
                    "backward(): accumulation only supported for Float32 CPU in this build");
    throw nb::python_error();
  }

  // Build initial_grads vector for a single-root backward run. The
  // engine expects initial_grads.size() == root->num_inputs(); we seed
  // exactly the slot indicated by AutogradMeta::output_nr and leave the
  // others disengaged.
  auto root = meta->grad_fn;
  std::vector<vbt::autograd::OptionalTensor> initial;
  const std::size_t n = static_cast<std::size_t>(root->num_inputs());
  initial.resize(n);
  const std::size_t slot = static_cast<std::size_t>(meta->output_nr);
  if (slot < n) {
    initial[slot] = g;
  }

#if VBT_WITH_CUDA
  // CUDA: if backward() is invoked from a different stream than the root node's
  // canonical stream, bridge the call stream to the root stream so the explicit
  // gradient seed is visible before autograd starts consuming it.
  //
  // Note: this is intentionally a conservative barrier on the entire call
  // stream (not just the gradient's producer), since VibeTensor does not yet
  // track producer-stream metadata for explicit gradient tensors.
  //
  // Contract: the explicit gradient tensor is assumed ready on the call stream
  // at backward() entry; if it was produced on a different stream, the caller
  // must synchronize before calling backward().
  bool              need_bridge_back = false;
  vbt::cuda::Stream S_call(vbt::cuda::Stream::UNCHECKED, 0u, static_cast<vbt::cuda::DeviceIndex>(dev.index));
  vbt::cuda::Stream S_root(vbt::cuda::Stream::UNCHECKED, 0u, static_cast<vbt::cuda::DeviceIndex>(dev.index));
  if (dev.type == kDLCUDA &&
      root &&
      root->stream_kind() == vbt::autograd::StreamKind::CudaAllowlisted) {
    const vbt::autograd::NodeStreamInfo& si = root->stream_info();
    if (si.has_canonical_stream) {
      if (si.device.type != kDLCUDA || si.device.index != dev.index) {
        throw std::runtime_error(
            "VibeTensor CUDA autograd internal error: missing or mismatched canonical stream on CUDA node");
      }

      const auto dev_idx = static_cast<vbt::cuda::DeviceIndex>(dev.index);
      S_call = vbt::cuda::getCurrentStream(dev_idx);
      S_root = vbt::cuda::Stream(vbt::cuda::Stream::UNCHECKED, si.stream_id, dev_idx);

      if (S_call.id() != S_root.id()) {
        // Keep the CUDA-graphs invariant: backward must not emit any CUDA work
        // while the current stream is under capture.
        vbt::cuda::assert_not_capturing_backward_stream(dev);

        vbt::cuda::Event ev(false);
        ev.record(S_call);
        ev.wait(S_root);
        need_bridge_back = true;
      }
    }
  }
#endif

  if (vbt::autograd::is_multithreading_enabled()) {
    nb::gil_scoped_release release;
    vbt::autograd::run_backward(root, initial, {});
  } else {
    vbt::autograd::run_backward(root, initial, {});
  }

#if VBT_WITH_CUDA
  if (need_bridge_back) {
    // Ensure that, after backward returns, subsequent work issued on the call
    // stream observes all CUDA-side effects of the backward run (which executes
    // on the root node's canonical stream).
    vbt::cuda::Event ev_done(false);
    ev_done.record(S_root);
    ev_done.wait(S_call);
  }
#endif
}

// User-facing requires_grad setter with leaf-only, non-view restriction.
static void tensor_set_requires_grad_user(TensorImpl& self, bool v) {
  if (!vbt::autograd::is_leaf(self) || vbt::autograd::is_view(self)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Tensor.requires_grad: may only be set on non-view leaf tensors; for intermediates use y = y.detach(); y.requires_grad = value instead");
    throw nb::python_error();
  }
  vbt::autograd::set_requires_grad(self, v);
}

// Public Tensor.grad/grad_tensor helper that presents a PyTorch-like
// gradient surface. It prefers the stored leaf grad buffer when its
// shape matches the tensor, and falls back to a simple 1D
static nb::object tensor_public_grad(const TensorImpl& self) {
  auto* m = vbt::autograd::get_autograd_meta(
      const_cast<TensorImpl&>(self), /*create_if_missing=*/false);
  if (!m) {
    return nb::object(nb::none());
  }

  TensorImpl stored;
  {
    std::lock_guard<std::mutex> lk(m->grad_mutex);
    if (!m->grad_ptr || !m->grad_has) {
      return nb::object(nb::none());
    }
    stored = *m->grad_ptr;
  }

#if VBT_WITH_CUDA
  // Record stream usage for safe cross-stream deallocation of .grad.
  if (stored.device().type == kDLCUDA && stored.numel() > 0) {
    const auto dev_idx =
        static_cast<vbt::cuda::DeviceIndex>(stored.device().index);
    vbt::cuda::Stream s = vbt::cuda::getCurrentStream(dev_idx);
    vbt::cuda::record_stream(stored.storage(), s);
  }
#endif
  const auto& stored_sizes = stored.sizes();
  const auto& self_sizes = self.sizes();

  // Fast path: shapes already match; expose the underlying buffer
  // directly. This keeps existing semantics for non-view leaves.
  if (stored_sizes == self_sizes) {
    return nb::cast(stored);
  }

  // gradients flowing into the base of a view chain (e.g., x[0:1]).
  if (self.device().type != kDLCPU ||
      stored.device().type != kDLCPU ||
      self.dtype() != vbt::core::ScalarType::Float32 ||
      stored.dtype() != vbt::core::ScalarType::Float32 ||
      self_sizes.size() != 1 ||
      stored_sizes.size() != 1) {
    return nb::cast(stored);
  }

  const std::int64_t n_self = self_sizes[0];
  const std::int64_t n_stored = stored_sizes[0];
  if (n_stored <= 0 || n_stored > n_self) {
    return nb::cast(stored);
  }

  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;
  using vbt::core::Device;

  const std::size_t item_b =
      static_cast<std::size_t>(itemsize(vbt::core::ScalarType::Float32));
  const std::size_t n_elems = static_cast<std::size_t>(n_self < 0 ? 0 : n_self);
  const std::size_t nbytes = n_elems * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
    float* p = static_cast<float*>(buf);
    for (std::size_t i = 0; i < n_elems; ++i) {
      p[i] = 0.0f;
    }
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> sizes_vec;
  sizes_vec.push_back(n_self);
  std::vector<std::int64_t> strides_vec;
  strides_vec.push_back(1);

  TensorImpl logical(storage, sizes_vec, strides_vec,
                     /*storage_offset=*/0,
                     vbt::core::ScalarType::Float32,
                     Device::cpu());

  // Copy the stored gradient into the logical prefix; the remainder
  // stays zero.
  const float* src = static_cast<const float*>(stored.data());
  float* dst = static_cast<float*>(logical.data());
  for (std::int64_t i = 0; i < n_stored; ++i) {
    dst[i] = src[i];
  }

  // Ensure the exposed gradient never requires grad.
  try {
    vbt::autograd::set_requires_grad(logical, false);
  } catch (...) {
  }

  return nb::cast(logical);
}

#endif

void bind_autograd(nb::module_& m) {
#if VBT_WITH_AUTOGRAD
  // Reopen Tensor class to add autograd properties
  auto Tensor = nb::borrow<nb::class_<TensorImpl>>(m.attr("Tensor"));
  Tensor
      .def("requires_grad",
           [](const TensorImpl& self) {
             return vbt::autograd::requires_grad(self);
           })
      .def("set_requires_grad",
           [](TensorImpl& self, bool v) {
             tensor_set_requires_grad_user(self, v);
           })
      .def("grad",
           [](const TensorImpl& self) {
             return tensor_public_grad(self);
           })
      .def("grad_tensor",
           [](const TensorImpl& self) {
             return tensor_public_grad(self);
           })
      .def("is_leaf",
           [](const TensorImpl& self) {
             return vbt::autograd::is_leaf(self);
           })
      .def("_grad_fn_handle",
           [](const TensorImpl& self) {
             intrusive_ptr<vbt::autograd::Node> fn = vbt::autograd::get_grad_fn(self);
             if (!fn) {
               return nb::object(nb::none());
             }
             GradFnHandle h{std::move(fn)};
             return nb::cast(h);
           })
      .def("detach",
           [](const TensorImpl& self) {
             return vbt::autograd::detach_copy(self);
           })
      .def("detach_",
           [](TensorImpl& self) {
             // Only allow on leaf tensors without grad_fn
             if (!vbt::autograd::is_leaf(self) ||
                 vbt::autograd::get_grad_fn(self)) {
               PyErr_SetString(
                   PyExc_RuntimeError,
                   "detach_(): only supported on leaf tensors without grad_fn; use detach() instead");
               throw nb::python_error();
             }
             vbt::autograd::detach_inplace(self);
             return &self;
           },
           nb::rv_policy::reference_internal)
      .def("retain_grad",
           [](TensorImpl& self) {
             if (!vbt::autograd::is_leaf(self) ||
                 vbt::autograd::is_view(self)) {
               PyErr_SetString(
                   PyExc_RuntimeError,
                   "retain_grad(): only supported on non-view leaf tensors");
               throw nb::python_error();
             }
             return &self;
           },
           nb::rv_policy::reference_internal)
      .def("register_hook",
           [](TensorImpl& self, nb::object hook) {
             if (!PyCallable_Check(hook.ptr())) {
               PyErr_SetString(PyExc_TypeError, "hook must be callable");
               throw nb::python_error();
             }
             if (!vbt::autograd::is_leaf(self) ||
                 vbt::autograd::is_view(self) ||
                 !vbt::autograd::requires_grad(self)) {
               PyErr_SetString(
                   PyExc_RuntimeError,
                   "register_hook(): only supported on non-view leaf tensors that require grad");
               throw nb::python_error();
             }
             auto th_impl = vbt::core::make_intrusive<PythonTensorHook>(hook);
             auto th = vbt::core::intrusive_ptr<vbt::autograd::TensorHook>(th_impl.get(), /*add_ref=*/true);
             vbt::autograd::register_leaf_hook(self, th);
             HookHandle handle{th};
             return nb::cast(handle);
           })
      .def("backward",
           [](TensorImpl& self, nb::object gradient, bool retain_graph) {
             (void) retain_graph;
             if (g_backward_in_progress) {
               PyErr_SetString(
                   PyExc_RuntimeError,
                   "nested backward() from inside backward is not supported");
               throw nb::python_error();
             }
             struct Guard {
               Guard() { g_backward_in_progress = true; }
               ~Guard() { g_backward_in_progress = false; }
             } guard;
             tensor_backward_impl(self, std::move(gradient), false);
           },
           nb::arg("gradient") = nb::none(), nb::arg("retain_graph") = false);

  // Extend autograd submodule with GradMode controls and Function API
  try {
    nb::module_ ag = nb::cast<nb::module_>(m.attr("autograd"));
    // Direct GradMode toggles and helpers
    ag.def("_clear_tensor_grad", [](TensorImpl& t) { vbt::autograd::clear_tensor_grad(t); });
    ag.def("_debug_tensor_meta", [](TensorImpl& t) {
      nb::dict out;
      const vbt::autograd::AutogradMeta* m = vbt::autograd::get_autograd_meta(t);
      if (!m) {
        return out;
      }
      out["requires_grad"] = nb::bool_(m->requires_grad);
      out["is_leaf"] = nb::bool_(m->is_leaf);
      out["output_nr"] = nb::int_(m->output_nr);
      out["has_grad_fn"] = nb::bool_(static_cast<bool>(m->grad_fn));
      bool grad_has = false;
      bool has_grad_ptr = false;
      {
        std::lock_guard<std::mutex> lk(m->grad_mutex);
        grad_has = m->grad_has;
        has_grad_ptr = static_cast<bool>(m->grad_ptr);
      }
      out["grad_has"] = nb::bool_(grad_has);
      out["has_grad_ptr"] = nb::bool_(has_grad_ptr);
      return out;
    });

    ag.def("is_inference_mode_enabled", []() {
      return vbt::autograd::InferenceMode::is_enabled();
    }, "Return True if inference mode is enabled in this thread.");

    // Internal setter; used only by Python context managers.
    ag.def("_set_inference_mode_enabled", [](bool v) {
      vbt::autograd::InferenceMode::set_enabled(v);
    });

    ag.def("is_multithreading_enabled",
           &vbt::autograd::is_multithreading_enabled,
           "Return whether multithreaded backward scheduling is enabled.");
    ag.def("set_multithreading_enabled",
           &vbt::autograd::set_multithreading_enabled,
           "Enable or disable multithreaded backward scheduling.");

    ag.def("is_view_replay_enabled",
           &vbt::autograd::is_view_replay_enabled,
           "Return stub view‑replay flag (currently unused).");
    ag.def("set_view_replay_enabled",
           &vbt::autograd::set_view_replay_enabled,
           "Set stub view‑replay flag (currently unused).");

    ag.def("get_device_mode", []() -> std::string {
      switch (vbt::autograd::get_device_mode()) {
        case vbt::autograd::AutogradDeviceMode::SingleDevice:
          return "single_device";
        case vbt::autograd::AutogradDeviceMode::MultiDeviceExperimental:
          return "multi_device_experimental";
      }
      return "single_device";
    });

    ag.def("set_device_mode",
           [](const std::string& mode) {
             if (mode == "single_device") {
               vbt::autograd::set_device_mode(
                   vbt::autograd::AutogradDeviceMode::SingleDevice);
               return;
             }
             if (mode == "multi_device_experimental") {
               vbt::autograd::set_device_mode(
                   vbt::autograd::AutogradDeviceMode::MultiDeviceExperimental);
               return;
             }
             throw std::runtime_error(
                 "set_device_mode: expected 'single_device' or 'multi_device_experimental'");
           },
           nb::arg("mode"),
           "Set the global autograd device mode for future backward runs.");

    ag.def("is_cuda_autograd_enabled",
           &vbt::autograd::is_streaming_backwards_enabled,
           "Return whether CUDA autograd/streaming backward is enabled.");

    ag.def("set_cuda_autograd_enabled",
           &vbt::autograd::set_streaming_backwards_enabled,
           nb::arg("enabled"),
           "Enable or disable CUDA autograd/streaming backward for future calls.");

    ag.def("_enter_dual_level", []() {
      return vbt::autograd::enter_forward_ad_level();
    });

    ag.def("_exit_dual_level", [](std::int64_t level_id) {
      vbt::autograd::exit_forward_ad_level(level_id);
    });

    ag.def("_current_dual_level", []() {
      return vbt::autograd::current_forward_ad_level();
    });

    ag.def("_set_forward_grad",
           [](TensorImpl& primal,
              const TensorImpl& tangent,
              std::int64_t level_id) {
             if (level_id < 0 ||
                 level_id != vbt::autograd::current_forward_ad_level()) {
               throw std::runtime_error(
                   "_set_forward_grad: level id must equal current forward-ad level");
             }
             vbt::autograd::set_forward_grad(primal, tangent, level_id);
           });

    ag.def("_get_forward_grad",
           [](const TensorImpl& primal,
              std::int64_t level_id) -> nb::object {
             if (level_id < 0) {
               throw std::runtime_error(
                   "_get_forward_grad: level id must be >= 0");
             }
             if (!vbt::autograd::has_forward_grad(primal, level_id)) {
               return nb::object(nb::none());
             }
             vbt::core::TensorImpl copy =
                 vbt::autograd::get_forward_grad_copy(primal, level_id);
             return nb::cast(copy);
           });

    ag.def("_has_forward_grad",
           [](const TensorImpl& primal) {
             return vbt::autograd::has_any_forward_grad(primal);
           });

    ag.def("is_in_backward", []() {
      return vbt::autograd::is_in_backward();
    });

    ag.def("_set_backward_complete_callback",
           [](nb::object cb) {
             if (cb.is_none()) {
               vbt::autograd::set_backward_complete_callback(
                   vbt::autograd::BackwardCompleteCallback{});
               return;
             }
             if (!PyCallable_Check(cb.ptr())) {
               PyErr_SetString(
                   PyExc_TypeError,
                   "_set_backward_complete_callback: callback must be callable or None");
               throw nb::python_error();
             }
             vbt::autograd::set_backward_complete_callback(
                 [cb = nb::object(std::move(cb))]() {
                   nb::gil_scoped_acquire gil;
                   cb();
                 });
           },
           nb::arg("cb").none(true));

    // GradFn and hook handle wrappers surfaced to Python
    nb::class_<GradFnHandle>(ag, "GradFn")
        .def_prop_ro("name", [](const GradFnHandle& h) {
          return h.fn ? nb::str(h.fn->name.c_str()) : nb::str("");
        });

    ag.def("_grad_fn_from_tensor", [](TensorImpl& t) {
      intrusive_ptr<vbt::autograd::Node> fn = vbt::autograd::get_grad_fn(t);
      if (!fn) {
        throw std::runtime_error(
            "_grad_fn_from_tensor: tensor has no grad_fn");
      }
      GradFnHandle h{std::move(fn)};
      vbt::autograd::_stats_graph_nodes_exposed(1);
      return h;
    });

    ag.def("_grad_fn_next_edges", [](const GradFnHandle& h) {
      std::vector<std::pair<nb::object, int>> result;
      if (!h.fn) {
        return result;
      }
      const auto& edges = h.fn->next_edges;
      result.reserve(edges.size());
      for (const auto& e : edges) {
        if (e.fn) {
          GradFnHandle child{e.fn};
          result.emplace_back(nb::cast(child), static_cast<int>(e.input_nr));
        } else {
          result.emplace_back(nb::object(nb::none()), static_cast<int>(e.input_nr));
        }
      }
      if (!edges.empty()) {
        vbt::autograd::_stats_graph_edges_exposed(
            static_cast<std::uint64_t>(edges.size()));
      }
      return result;
    });

    ag.def("_grad_fn_debug_metadata", [](const GradFnHandle& h) {
      nb::dict out;
      if (!h.fn) {
        out["num_inputs"] = nb::int_(0);
        out["has_input_meta"] = nb::bool_(false);
        out["debug_id"] = nb::int_(0);
        return out;
      }
      vbt::autograd::Node* n = h.fn.get();
      auto* valid = dynamic_cast<const vbt::autograd::ValidatableNode*>(n);
      out["num_inputs"] = nb::int_(n->num_inputs());
      out["has_input_meta"] = nb::bool_(valid != nullptr);
      auto dbg = reinterpret_cast<std::uintptr_t>(n);
      out["debug_id"] = nb::int_(static_cast<std::uint64_t>(dbg));
      return out;
    });

    ag.def("_grad_fn_stream_info", [](const GradFnHandle& h) {
      nb::dict out;
      if (!h.fn) {
        out["stream_kind"] = nb::str("");
        out["has_canonical_stream"] = nb::bool_(false);
        out["device_type"] = nb::int_(0);
        out["device_index"] = nb::int_(0);
        out["stream_id"] = nb::int_(0);
        return out;
      }

      const vbt::autograd::Node& n = *h.fn;
      const char* kind = "";
      switch (n.stream_kind()) {
        case vbt::autograd::StreamKind::CpuOnly:
          kind = "cpu_only";
          break;
        case vbt::autograd::StreamKind::CudaAllowlisted:
          kind = "cuda_allowlisted";
          break;
        default:
          kind = "unknown";
          break;
      }

      const vbt::autograd::NodeStreamInfo& si = n.stream_info();
      out["stream_kind"] = nb::str(kind);
      out["has_canonical_stream"] = nb::bool_(si.has_canonical_stream);
      out["device_type"] = nb::int_(static_cast<int>(si.device.type));
      out["device_index"] = nb::int_(static_cast<int>(si.device.index));
      out["stream_id"] = nb::int_(static_cast<std::uint64_t>(si.stream_id));
      return out;
    });

    ag.def("_graph_get_gradient_edge", [](TensorImpl& t) {
      if (!vbt::autograd::requires_grad(t)) {
        throw std::runtime_error(
            "_graph_get_gradient_edge: tensor does not require grad");
      }

      std::unordered_map<const vbt::autograd::AutogradMeta*,
                         intrusive_ptr<vbt::autograd::Node>>
          sinks;
      const vbt::autograd::Edge e =
          vbt::autograd::resolve_edge_for_tensor(t, sinks);
      if (!e.fn) {
        throw std::runtime_error(
            "_graph_get_gradient_edge: internal resolve_edge_for_tensor returned null");
      }

      GradFnHandle h{e.fn};
      vbt::autograd::_stats_graph_nodes_exposed(1);
      return nb::make_tuple(h, nb::int_(static_cast<int>(e.input_nr)));
    });

    nb::class_<HookHandle>(ag, "HookHandle")
        .def("remove", [](HookHandle& h) {
          if (h.hook) {
            h.hook->set_removed();
          }
        });

    ag.def("_push_saved_tensors_hooks",
           [](nb::object pack, nb::object unpack, bool trusted_builtin) {
             vbt::autograd::SavedTensorHookState& st = vbt::autograd::saved_tensor_hooks_tls();
             if (st.disabled) {
               const std::string& msg = st.disabled_error.empty()
                                           ? std::string("saved_tensors_hooks are disabled")
                                           : st.disabled_error;
               throw std::runtime_error(msg);
             }
             vbt::autograd::SavedTensorHookPair pair;
             pair.pack = std::move(pack);
             pair.unpack = std::move(unpack);
             pair.trusted_builtin = trusted_builtin;
             st.stack.push_back(std::move(pair));
           });

    ag.def("_pop_saved_tensors_hooks", []() {
      vbt::autograd::SavedTensorHookState& st = vbt::autograd::saved_tensor_hooks_tls();
      if (st.stack.empty()) {
        throw std::runtime_error(
            "saved_tensors_hooks: internal stack underflow");
      }
      st.stack.pop_back();
    });

    ag.def("_set_saved_tensors_disabled",
           [](bool disabled, nb::object error_message) {
             vbt::autograd::SavedTensorHookState& st = vbt::autograd::saved_tensor_hooks_tls();
             st.disabled = disabled;
             if (error_message.is_none()) {
               st.disabled_error.clear();
             } else {
               st.disabled_error = nb::cast<std::string>(error_message);
             }
           });

    ag.def("_create_py_function_node",
           [](nb::object fn_cls,
              nb::tuple tensor_inputs,
              TensorImpl& output_tensor,
              nb::tuple saved_tensors,
              nb::tuple needs_input_grad,
              nb::tuple edge_index_by_arg,
              nb::dict ctx_state) {
             vbt::autograd::create_py_function_node(
                 std::move(fn_cls),
                 tensor_inputs,
                 output_tensor,
                 saved_tensors,
                 needs_input_grad,
                 edge_index_by_arg,
                 ctx_state);
           });

    // Minimal autograd.Function with static apply(backward, inputs)
    nb::class_<FunctionShim>(ag, "Function")
        .def_static("apply", [](nb::object backward, nb::tuple inputs){
          std::vector<vbt::autograd::InputMeta> metas;
          metas.reserve(inputs.size());
          for (size_t i = 0; i < inputs.size(); ++i) {
            TensorImpl t = nb::cast<TensorImpl>(inputs[i]);
            metas.emplace_back(vbt::autograd::InputMeta::from_tensor(t));
          }
          vbt::autograd::BackwardFn fn = [backward](std::vector<vbt::autograd::OptionalTensor>&& /*grads_in*/){
            nb::gil_scoped_acquire gil;
            nb::object out = backward();
            std::vector<vbt::autograd::OptionalTensor> grads;
            if (nb::isinstance<nb::tuple>(out) || nb::isinstance<nb::list>(out)) {
              size_t n = static_cast<size_t>(nb::len(out));
              grads.resize(n);
              if (nb::isinstance<nb::tuple>(out)) {
                nb::tuple tup = nb::cast<nb::tuple>(out);
                for (size_t i = 0; i < n; ++i) {
                  nb::object oi = tup[i];
                  if (!oi.is_none()) grads[i] = nb::cast<TensorImpl>(oi);
                }
              } else {
                nb::list li = nb::cast<nb::list>(out);
                for (size_t i = 0; i < n; ++i) {
                  nb::object oi = li[i];
                  if (!oi.is_none()) grads[i] = nb::cast<TensorImpl>(oi);
                }
              }
            }
            return grads;
          };
          auto node = vbt::core::intrusive_ptr<vbt::autograd::Node>(new vbt::autograd::FunctionNode("FunctionBackward", std::move(metas), std::move(fn)), /*add_ref=*/true);
          vbt::autograd::ensure_next_edges_sized(*node);
          std::vector<vbt::autograd::OptionalTensor> initial; initial.resize(static_cast<std::size_t>(node->num_inputs()));
          if (vbt::autograd::is_multithreading_enabled()) {
            nb::gil_scoped_release release;
            vbt::autograd::run_backward(node, initial, {});
          } else {
            vbt::autograd::run_backward(node, initial, {});
          }
          return nb::none();
        });

    // RAII guards as Python context managers
    nb::class_<vbt::autograd::NoGradGuard>(ag, "_NoGradGuard")
        .def(nb::init<>())
        .def("__enter__", [](vbt::autograd::NoGradGuard& self){ return &self; }, nb::rv_policy::reference)
        .def("__exit__", [](vbt::autograd::NoGradGuard&, nb::object, nb::object, nb::object){ return nb::bool_(false); });
    ag.def("no_grad", [](){ return vbt::autograd::NoGradGuard(); });

    nb::class_<vbt::autograd::EnableGradGuard>(ag, "_EnableGradGuard")
        .def(nb::init<>())
        .def("__enter__", [](vbt::autograd::EnableGradGuard& self){ return &self; }, nb::rv_policy::reference)
        .def("__exit__", [](vbt::autograd::EnableGradGuard&, nb::object, nb::object, nb::object){ return nb::bool_(false); });
    ag.def("enable_grad", [](){ return vbt::autograd::EnableGradGuard(); });

  } catch (...) {
    // Ignore if submodule missing (should have been created in bindings.cpp)
  }
#else
  // When autograd disabled, expose no-ops
  auto Tensor = nb::borrow<nb::class_<TensorImpl>>(m.attr("Tensor"));
  Tensor
      .def("requires_grad", [](const TensorImpl&){ return false; })
      .def("set_requires_grad", [](TensorImpl&, bool){})
      .def("grad", [](const TensorImpl&){ return nb::none(); })
      .def("grad_tensor", [](const TensorImpl&){ return nb::none(); })
      .def("backward", [](TensorImpl&, nb::object, bool){ /* no-op */ },
           nb::arg("gradient") = nb::none(), nb::arg("retain_graph") = false);
#endif
}

} // namespace vbt_py
