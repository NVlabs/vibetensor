// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "vbt/hello.h"
#include "vbt/logging/logging.h"
#include "vbt/cuda/device.h"
#include "vbt/dispatch/plugin_loader.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/core/indexing.h"
#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/stats.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/autograd/saved_variable.h"
#include "vbt/autograd/detail/stats_internal.h"
#endif

#include <cstdlib>
#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>
#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include "vbt/cuda/stream.h"
#endif

namespace nb = nanobind;

// Forward declarations of binder functions
namespace vbt_py {
void bind_tensor(nb::module_& m);
void bind_dlpack(nb::module_& m);
void bind_ops(nb::module_& m);
void bind_cuda_streams(nb::module_& m);
void bind_cuda_events(nb::module_& m);
void bind_cuda_memory(nb::module_& m);
void bind_cpu_memory(nb::module_& m);
void bind_pinned_memory(nb::module_& m);
void bind_cuda_copy(nb::module_& m);
void bind_cuda_graphs(nb::module_& m);
void bind_cuda_reduction(nb::module_& m);
void bind_factories(nb::module_& m);
void bind_autograd(nb::module_& m);
void bind_cuda_driver(nb::module_& m);
void bind_rng(nb::module_& m);
void bind_fabric(nb::module_& m);
}

extern "C" void vbt_register_default_kernels();
extern "C" void vbt_register_indexing_kernels();
extern "C" void vbt_register_embedding_kernels();
#if VBT_WITH_CUDA
extern "C" void vbt_register_cuda_elementwise_kernels();
extern "C" void vbt_register_cuda_reduction_kernels();
extern "C" void vbt_register_fabric_kernels();
#endif

namespace {
// Python override registry keyed by OperatorEntry*
using OverrideMap = std::unordered_map<const vbt::dispatch::OperatorEntry*, nb::object>;
static std::shared_ptr<const OverrideMap> g_overrides = std::make_shared<OverrideMap>();
// TLS for kwargs forwarding into Python override trampoline
static thread_local PyObject* g_tls_kwargs = nullptr;
// Env-gated compatibility for error strings
static const bool k_ops_compat = [](){ const char* v = std::getenv("VBT_OPS_COMPAT"); return v && std::string(v) == "1"; }();

#if VBT_WITH_AUTOGRAD
// Python autograd backward registry keyed by fully-qualified op name
using AutogradPyMap = std::unordered_map<std::string, nb::object>;
static std::shared_ptr<const AutogradPyMap> g_autograd_py = std::make_shared<AutogradPyMap>();

static inline bool any_requires_grad(const vbt::dispatch::BoxedStack& s, uint8_t in_arity) noexcept {
  if (in_arity == 0) return false;
  const std::size_t n = std::min<std::size_t>(in_arity, s.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (vbt::autograd::requires_grad(s[i])) return true;
  }
  return false;
}

// Helper to initialize CUDA stream info for Python backward nodes
static void init_python_node_streaminfo(
    vbt::autograd::Node& n,
    const std::vector<vbt::autograd::InputMeta>& metas,
    bool& is_cuda_flag) {
#if VBT_WITH_CUDA
  int cuda_index = -1;
  for (const auto& m : metas) {
    if (m.device.type == kDLCUDA) {
      if (cuda_index == -1) cuda_index = m.device.index;
      else if (cuda_index != m.device.index)
        throw std::runtime_error("PythonBackwardNode: mixed CUDA devices");
    }
  }
  if (cuda_index < 0) { is_cuda_flag = false; return; }
  auto S = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(cuda_index));
  auto& si = n.mutable_stream_info();
  si.has_canonical_stream = true;
  si.device = vbt::core::Device::cuda(cuda_index);
  si.stream_id = S.id();
  is_cuda_flag = true;
#else
  (void)n; (void)metas;  // Avoid unused parameter warnings
  is_cuda_flag = false;
#endif
}

class PythonBackwardNode final : public vbt::autograd::Node, public vbt::autograd::ValidatableNode {
 public:
  PythonBackwardNode(nb::object py_fn,
                     std::vector<vbt::autograd::SavedVariable> snaps,
                     std::vector<vbt::autograd::InputMeta> metas)
    : py_backward_(std::move(py_fn)), snaps_(std::move(snaps)), metas_(std::move(metas)) {
      name = "PythonBackward";
      init_python_node_streaminfo(*this, metas_, is_cuda_);
    }
  uint32_t num_inputs() const noexcept override { return 1; }
  const std::vector<vbt::autograd::InputMeta>& input_metas() const noexcept override { return metas_; }
  vbt::autograd::StreamKind stream_kind() const noexcept override {
    return is_cuda_ ? vbt::autograd::StreamKind::CudaAllowlisted
                    : vbt::autograd::StreamKind::CpuOnly;
  }
  std::vector<vbt::autograd::OptionalTensor> apply(std::vector<vbt::autograd::OptionalTensor>&& grads_in) override {
    vbt::autograd::NoGradGuard ng;
    nb::gil_scoped_acquire gil;
    nb::list gin;
    for (size_t i = 0; i < grads_in.size(); ++i) {
      if (grads_in[i].has_value()) gin.append(nb::cast(*grads_in[i])); else gin.append(nb::none());
    }
    nb::list saved;
    for (size_t i = 0; i < snaps_.size(); ++i) saved.append(nb::cast(snaps_[i].unpack()));
    nb::object out = py_backward_(nb::tuple(gin), nb::tuple(saved));
    std::vector<vbt::autograd::OptionalTensor> outs;
    if (nb::isinstance<nb::tuple>(out)) {
      nb::tuple tup = nb::cast<nb::tuple>(out);
      outs.resize(static_cast<size_t>(tup.size()));
      for (size_t i = 0; i < outs.size(); ++i) {
        nb::object oi = tup[i];
        if (!oi.is_none()) outs[i] = nb::cast<vbt::core::TensorImpl>(oi);
      }
    } else if (nb::isinstance<nb::list>(out)) {
      nb::list li = nb::cast<nb::list>(out);
      outs.resize(static_cast<size_t>(li.size()));
      for (size_t i = 0; i < outs.size(); ++i) {
        nb::object oi = li[i];
        if (!oi.is_none()) outs[i] = nb::cast<vbt::core::TensorImpl>(oi);
      }
    }
    return outs;
  }
 private:
  nb::object py_backward_;
  std::vector<vbt::autograd::SavedVariable> snaps_;
  std::vector<vbt::autograd::InputMeta> metas_;
  bool is_cuda_{false};
};

static void autograd_py_fallback_ctx(void* ctx, vbt::dispatch::BoxedStack& s) {
  auto* entry = static_cast<vbt::dispatch::OperatorEntry*>(ctx);
  if (!entry) throw std::runtime_error("invalid autograd fallback ctx");
  const std::string& name = entry->schema.fqname;
  const uint8_t in_arity = entry->schema.in_arity;

  const bool do_autograd = vbt::autograd::GradMode::is_enabled() &&
                           !vbt::autograd::InferenceMode::is_enabled() &&
                           any_requires_grad(s, in_arity);

  std::vector<vbt::autograd::SavedVariable> snaps;
  if (do_autograd) {
    snaps.reserve(in_arity);
    const std::size_t n = std::min<std::size_t>(in_arity, s.size());
    for (std::size_t i = 0; i < n; ++i) snaps.emplace_back(vbt::autograd::SavedVariable(s[i]));
  }

  // Redispatch to base
  vbt::dispatch::OperatorHandle op(entry);
  vbt::autograd::SkipAutogradGuard sa;
  vbt::dispatch::Dispatcher::instance().callBoxed(op, s);

  if (!do_autograd) return;
  if (s.size() != 1) {
    throw std::runtime_error(std::string("autograd wrapper only supports single-output ops: ") + name);
  }

  // Lookup Python backward (GIL required for nb::object copy which calls Py_INCREF)
  nb::object pyb;
  {
    nb::gil_scoped_acquire gil;
    auto reg = g_autograd_py;
    auto it = reg->find(name);
    if (it == reg->end()) return; // not registered
    pyb = it->second;
  }

  // Build metas
  std::vector<vbt::autograd::InputMeta> metas; metas.reserve(snaps.size());
  for (size_t i = 0; i < snaps.size(); ++i) metas.emplace_back(vbt::autograd::InputMeta::from_tensor(snaps[i].unpack()));

  // Build node
  auto node = vbt::core::intrusive_ptr<vbt::autograd::Node>(new PythonBackwardNode(pyb, snaps, metas), /*add_ref=*/true);
  node->next_edges.resize(metas.size());
  // Wire next edges
  const std::size_t nwire = std::min<std::size_t>(static_cast<std::size_t>(in_arity), snaps.size());
  {
    std::unordered_map<const vbt::autograd::AutogradMeta*, vbt::core::intrusive_ptr<vbt::autograd::Node>> sinks;
    for (std::size_t i = 0; i < nwire; ++i) {
      const vbt::core::TensorImpl ti = snaps[i].unpack();
      node->next_edges[i] = vbt::autograd::resolve_edge_for_tensor(ti, sinks);
    }
  }

  // Attach to output
  vbt::core::TensorImpl& out = s[0];
  vbt::autograd::set_requires_grad(out, true);
  if (auto* m = vbt::autograd::get_autograd_meta(out, /*create_if_missing=*/true)) {
    m->is_leaf = false;
    m->output_nr = 0;
    m->grad_fn = node;
  }
}
#endif

static void boxed_python_trampoline(vbt::dispatch::BoxedStack& s) {
  auto* op = vbt::dispatch::Dispatcher::tls_current_op_;
  if (!op) {
    throw std::runtime_error("python override called without current op context");
  }
  nb::gil_scoped_acquire gil;
  auto overrides = g_overrides;
  auto it = overrides->find(op);
  if (it == overrides->end()) {
    throw std::runtime_error("no python override registered for operator");
  }
  nb::object fn = it->second;
  nb::list args;
  for (auto& t : s) args.append(nb::cast(t));
  nb::object result;
  if (g_tls_kwargs != nullptr) {
    result = nb::steal<nb::object>(PyObject_Call(fn.ptr(), nb::tuple(args).ptr(), g_tls_kwargs));
  } else {
    result = fn(*nb::tuple(args));
  }
  // Enforce return type strictly: must be vibetensor._C.Tensor
  if (!nb::isinstance<vbt::core::TensorImpl>(result)) {
    PyErr_SetString(PyExc_TypeError, "Python override must return vibetensor._C.Tensor");
    throw nb::python_error();
  }
  // Copy result tensor into stack (keep result alive until after push)
  vbt::core::TensorImpl out = nb::cast<vbt::core::TensorImpl>(result);
  if (s.size() != 1) s.resize(1);
  s[0] = out;
}
} // anonymous

NB_MODULE(_C, m) {
  // Nanobind's leak checker is very strict and can be noisy in test runs.
  // Default to disabling it unless explicitly enabled.
  const char* nb_leak = std::getenv("VBT_NB_LEAK_WARNINGS");
  const bool enable_nb_leak = (nb_leak && nb_leak[0] == '1' && nb_leak[1] == '\0');
  if (!enable_nb_leak) {
    nb::set_leak_warnings(false);
  }

  // Ensure default and indexing kernels are registered for Python wrappers
  vbt_register_default_kernels();
  vbt_register_indexing_kernels();
  vbt_register_embedding_kernels();

  m.def("_vbt_hello", &vbt::HelloString);
  m.def("_init_logging", &vbt::InitLogging, nb::arg("min_log_level").none(true));

#if VBT_WITH_CUDA
  m.attr("_has_cuda") = nb::bool_(true);
  // Ensure CUDA elementwise kernels TU is linked and static registrars run
  vbt_register_cuda_elementwise_kernels();
  vbt_register_cuda_reduction_kernels();
  vbt_register_fabric_kernels();
#else
  m.attr("_has_cuda") = nb::bool_(false);
#endif
  m.def("_cuda_device_count", [](){ return vbt::cuda::device_count(); });
#if VBT_HAS_DLPACK_BF16
  m.attr("_has_dlpack_bf16") = nb::bool_(true);
#else
  m.attr("_has_dlpack_bf16") = nb::bool_(false);
#endif
  // Single source of truth helper for BF16 capability
  m.def("_has_bf16", [](){
#if VBT_HAS_DLPACK_BF16
    return true;
#else
    return false;
#endif
  });

  // Default-device helper used by printing tests and suffix rules

#if VBT_WITH_CUDA
  m.def("_cuda_current_device", [](){ int d = -1; cudaError_t st = cudaGetDevice(&d); if (st != cudaSuccess) return -1; return d; });
#else
  m.def("_cuda_current_device", [](){ return -1; });
#endif
  m.def("_get_default_device_type", []() {
    const char* env = std::getenv("VBT_DEFAULT_DEVICE_TYPE");
    if (env) {
      std::string s(env);
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
      if (s == "cuda") return std::string("cuda");
    }
    return std::string("cpu");
  });
  // Advanced indexing feature flag helpers (test-only; used from Python tests)
  m.def("_advanced_indexing_enabled", [](){
    return vbt::core::indexing::advanced_indexing_enabled();
  });
  m.def("_set_advanced_indexing_enabled_for_tests", [](bool enabled){
    vbt::core::indexing::set_advanced_indexing_enabled_for_tests(enabled);
  });
  // Autograd indexing v2 gate (test-only; used from Python tests)
  m.def("_autograd_indexing_v2_enabled", [](){
    return vbt::autograd::autograd_indexing_v2_enabled();
  });
  m.def("_set_autograd_indexing_v2_enabled_for_tests", [](bool enabled){
    vbt::autograd::set_autograd_indexing_v2_enabled_for_tests(enabled);
  });
  m.def("_autograd_indexing_v2_negstride_enabled", [](){
    return vbt::autograd::autograd_indexing_v2_negstride_enabled();
  });
  m.def("_set_autograd_indexing_v2_negstride_enabled_for_tests", [](bool enabled){
    vbt::autograd::set_autograd_indexing_v2_negstride_enabled_for_tests(enabled);
  });
#if VBT_WITH_AUTOGRAD
  (void)vbt::autograd::register_autograd_fallbacks();
  // Extend autograd stats and later autograd Tensor methods/controls
  nb::module_ ag = m.def_submodule("autograd");
  ag.def("stats", [](){
    auto s = vbt::autograd::stats();
    nb::dict d;
    d["engine_runs"] = nb::int_(s.engine_runs);
    d["engine_nodes_processed"] = nb::int_(s.engine_nodes_processed);
    d["engine_edges_processed"] = nb::int_(s.engine_edges_processed);
    d["engine_duplicates_coalesced"] = nb::int_(s.engine_duplicates_coalesced);
    d["engine_callbacks_run"] = nb::int_(s.engine_callbacks_run);
    d["wrapper_invocations"] = nb::int_(s.wrapper_invocations);
    d["wrapper_guard_skips"] = nb::int_(s.wrapper_guard_skips);
    d["graph_nodes_exposed"] = nb::int_(s.graph_nodes_exposed);
    d["graph_edges_exposed"] = nb::int_(s.graph_edges_exposed);
    d["saved_tensors_packed"] = nb::int_(s.saved_tensors_packed);
    d["saved_tensors_unpacked"] = nb::int_(s.saved_tensors_unpacked);
    d["saved_tensors_hook_violations"] = nb::int_(s.saved_tensors_hook_violations);
    d["multi_grad_hooks_registered"] = nb::int_(s.multi_grad_hooks_registered);
    d["multi_grad_hooks_fired_all"] = nb::int_(s.multi_grad_hooks_fired_all);
    d["multi_grad_hooks_fired_any"] = nb::int_(s.multi_grad_hooks_fired_any);
    d["py_function_nodes_created"] = nb::int_(s.py_function_nodes_created);
    d["py_function_nodes_applied"] = nb::int_(s.py_function_nodes_applied);
    d["py_function_backward_failures"] = nb::int_(s.py_function_backward_failures);
    return d;
  }, "Return a best‑effort snapshot of autograd counters (not atomic across fields).");
  ag.def("reset_stats", &vbt::autograd::reset_stats, "Reset all global autograd counters to zero; other threads may bump concurrently.");

  ag.def("_stats_multi_grad_registered", []() {
    vbt::autograd::_stats_multi_grad_registered(1);
  });
  ag.def("_stats_multi_grad_fired_all", []() {
    vbt::autograd::_stats_multi_grad_fired_all(1);
  });
  ag.def("_stats_multi_grad_fired_any", []() {
    vbt::autograd::_stats_multi_grad_fired_any(1);
  });

  ag.def("_raw_grad_mode_enabled", []() {
    return vbt::autograd::GradMode::is_enabled();
  });

  ag.def("is_grad_enabled", []() {
    // Graph‑enabled state: respects inference‑mode.
    return vbt::autograd::GradMode::is_enabled() &&
           !vbt::autograd::InferenceMode::is_enabled();
  });

  ag.def("set_grad_enabled", [](bool v) {
    vbt::autograd::GradMode::set_enabled(v);
  });
  // Python-facing autograd fallback registration (boxed ctx)
  m.def("_try_register_boxed_autograd_fallback", [](const std::string& opname, nb::object py_backward){
    using namespace vbt::dispatch;
    auto& D = Dispatcher::instance();
    OperatorHandle h = D.find(opname);
    const uint8_t arity = h.get().schema.in_arity;
    auto kf = vbt::dispatch::KernelFunction::makeBoxedCtx(arity, &autograd_py_fallback_ctx, nullptr);
    bool ok = false;
    try {
      ok = D.tryRegisterAutogradFallback(opname, kf);
    } catch (const std::runtime_error& e) {
      std::string msg = e.what();
      if (msg.rfind("duplicate autograd fallback: ", 0) == 0) {
        ok = false;
      } else {
        throw std::runtime_error(e.what());
      }
    }
    if (ok) {
      nb::gil_scoped_acquire gil;
      auto next = std::make_shared<AutogradPyMap>(*(g_autograd_py));
      (*next)[h.get().schema.fqname] = py_backward;
      g_autograd_py = next;
    }
    return ok;
  });
  m.def("_reset_autograd_py", [](){
    nb::gil_scoped_acquire gil;
    g_autograd_py = std::make_shared<AutogradPyMap>();
  });
#endif

  // Plugin loader
  m.def("_load_library", [](const std::string& path){
    vt_status st = vbt::dispatch::plugin::load_library(path.c_str());
    if (st != VT_STATUS_OK) {
      throw nb::value_error(vbt::dispatch::plugin::get_last_error());
    }
  });

  // Loader diagnostics
  m.def("_loaded_libraries", [](){
    return vbt::dispatch::plugin::loaded_libraries();
  });
  m.def("_is_library_loaded", [](const std::string& p){ return vbt::dispatch::plugin::is_library_loaded(p); });

  // Python override registration
  m.def("_register_boxed_python_override", [](const std::string& opname, nb::object fn){
    using namespace vbt::dispatch;
    auto& D = Dispatcher::instance();
    OperatorHandle h = D.find(opname);
    D.registerBoxedOverride(opname, &boxed_python_trampoline);
    nb::gil_scoped_acquire gil;
    auto next = std::make_shared<OverrideMap>(*(g_overrides));
    (*next)[&h.get()] = fn;
    g_overrides = next;
  });

  // Python override registration (idempotent try)
  m.def("_try_register_boxed_python_override", [](const std::string& opname, nb::object fn){
    using namespace vbt::dispatch;
    auto& D = Dispatcher::instance();
    OperatorHandle h = D.find(opname);
    try {
      D.registerBoxedOverride(opname, &boxed_python_trampoline);
    } catch (const std::runtime_error& e) {
      std::string msg = e.what();
      if (msg.rfind("duplicate CPU impl (boxed): ", 0) == 0) {
        return false; // already registered
      }
      throw std::runtime_error(e.what());
    }
    nb::gil_scoped_acquire gil;
    auto next = std::make_shared<OverrideMap>(*(g_overrides));
    (*next)[&h.get()] = fn;
    g_overrides = next;
    return true;
  });

  // Define a dispatcher schema (Python-only ops supported)
  m.def("def", [](const std::string& def_string){
    using namespace vbt::dispatch;
    try {
      (void)Dispatcher::instance().def(def_string);
    } catch (const std::invalid_argument& e) {
      throw nb::value_error(e.what());
    }
  });

  m.def("_has_op", [](const std::string& name) {
    return vbt::dispatch::Dispatcher::instance().has(name);
  });

  // Generic boxed call by name; requires exactly one Tensor result
  m.def("_call_op", [](const std::string& name, nb::args args){
    using namespace vbt::dispatch;
    BoxedStack s; s.reserve(args.size());
    for (auto o : args) s.push_back(nb::cast<vbt::core::TensorImpl>(o));
    try {
      Dispatcher::instance().callBoxed(name, s);
    } catch (const std::invalid_argument& e) {
      throw nb::value_error(e.what());
    } catch (const std::out_of_range& /*e*/) {
      // Preserve std::out_of_range so nanobind raises IndexError.
      throw;
    } catch (const std::overflow_error& /*e*/) {
      // Preserve std::overflow_error so nanobind raises OverflowError.
      throw;
    } catch (const std::runtime_error& e) {
      if (k_ops_compat) {
        std::string msg = e.what();
        if (msg.rfind("unknown op: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("unknown op: "));
          throw std::runtime_error(("Didn't find operator '" + fq + "'").c_str());
        }
        if (msg.rfind("no CPU kernel registered: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("no CPU kernel registered: "));
          throw std::runtime_error(("No kernel found for dispatch key CPU for operator '" + fq + "'").c_str());
        }
        if (msg.rfind("no CUDA kernel registered: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("no CUDA kernel registered: "));
          throw std::runtime_error(("No kernel found for dispatch key CUDA for operator '" + fq + "'").c_str());
        }
      }
      throw std::runtime_error(e.what());
    }
    if (s.size() != 1) {
      std::string msg = std::string("_call_op: kernel did not produce exactly one result: ") + name;
      throw nb::value_error(msg.c_str());
    }
    return s[0];
  });

  // Generic boxed call by name with kwargs; requires exactly one Tensor result; only supported for Python overrides
  m.def("_call_op_kwargs", [](const std::string& name, nb::args args, nb::kwargs kwargs){
    using namespace vbt::dispatch;
    OperatorHandle h;
    try {
      h = Dispatcher::instance().find(name);
    } catch (const std::runtime_error& e) {
      if (k_ops_compat) {
        std::string msg = e.what();
        if (msg.rfind("unknown op: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("unknown op: "));
          throw std::runtime_error(("Didn't find operator '" + fq + "'").c_str());
        }
      }
      throw;
    }
    if (h.get().boxed_override == nullptr) {
      throw nb::type_error("kwargs not supported for base kernels");
    }
    BoxedStack s; s.reserve(args.size());
    for (auto o : args) s.push_back(nb::cast<vbt::core::TensorImpl>(o));
    struct KwargsGuard { PyObject* prev; PyObject* kw_; explicit KwargsGuard(PyObject* kw): prev(g_tls_kwargs), kw_(kw) { g_tls_kwargs = kw_; Py_XINCREF(kw_);} ~KwargsGuard(){ Py_XDECREF(kw_); g_tls_kwargs = prev; } };
    KwargsGuard guard(kwargs.ptr());
    try {
      Dispatcher::instance().callBoxed(h, s);
    } catch (const std::invalid_argument& e) {
      throw nb::value_error(e.what());
    } catch (const std::out_of_range& /*e*/) {
      // Preserve std::out_of_range so nanobind raises IndexError.
      throw;
    } catch (const std::overflow_error& /*e*/) {
      // Preserve std::overflow_error so nanobind raises OverflowError.
      throw;
    } catch (const std::runtime_error& e) {
      if (k_ops_compat) {
        std::string msg = e.what();
        if (msg.rfind("unknown op: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("unknown op: "));
          throw std::runtime_error(("Didn't find operator '" + fq + "'").c_str());
        }
        if (msg.rfind("no CPU kernel registered: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("no CPU kernel registered: "));
          throw std::runtime_error(("No kernel found for dispatch key CPU for operator '" + fq + "'").c_str());
        }
        if (msg.rfind("no CUDA kernel registered: ", 0) == 0) {
          std::string fq = msg.substr(std::strlen("no CUDA kernel registered: "));
          throw std::runtime_error(("No kernel found for dispatch key CUDA for operator '" + fq + "'").c_str());
        }
      }
      throw;
    }
    if (s.size() != 1) {
      std::string msg = std::string("_call_op_kwargs: kernel did not produce exactly one result: ") + name;
      throw nb::value_error(msg.c_str());
    }
    return s[0];
  });

  // Redispatch to base implementation for the current override
  m.def("_redispatch_boxed_current", [](nb::args args){
    using namespace vbt::dispatch;
    BoxedStack s; s.reserve(args.size());
    for (auto o : args) s.push_back(nb::cast<vbt::core::TensorImpl>(o));
    try {
      Dispatcher::instance().redispatchBoxedCurrent(s);
    } catch (const std::invalid_argument& e) {
      throw nb::value_error(e.what());
    }
    if (s.size() != 1) {
      throw nb::value_error("_redispatch_boxed_current: base kernel did not produce exactly one result");
    }
    return s[0];
  });

#if VBT_WITH_CUDA
  // CUDA runtime bindings (streams/events/memory/copy/graphs)
  vbt_py::bind_cuda_streams(m);
  vbt_py::bind_cuda_events(m);
  vbt_py::bind_cuda_memory(m);
  vbt_py::bind_cuda_copy(m);
  vbt_py::bind_cuda_graphs(m);
  vbt_py::bind_cuda_reduction(m);
#endif

  // CUDA driver APIs (always bound; provide CPU stubs when CUDA is disabled)
  vbt_py::bind_cuda_driver(m);

  // CPU memory bindings are always available
  vbt_py::bind_cpu_memory(m);
  vbt_py::bind_pinned_memory(m);
  // Bind core tensor and interop
  vbt_py::bind_tensor(m);
#if VBT_REQUIRE_DLPACK_ALIGNMENT
  m.attr("_REQUIRE_DLPACK_ALIGNMENT") = nb::bool_(true);
#else
  m.attr("_REQUIRE_DLPACK_ALIGNMENT") = nb::bool_(false);
#endif
  vbt_py::bind_dlpack(m);

  // CPU factories
  vbt_py::bind_factories(m);
  // RNG bindings
  vbt_py::bind_rng(m);
  vbt_py::bind_fabric(m);

  // NumPy zero-copy view binding (CPU)
  // vbt_py::bind_numpy_view(m);

  // Extend autograd surface (Tensor methods/backward, GradMode setters).
  vbt_py::bind_autograd(m);
  // Administrative: clear Python overrides at interpreter shutdown (called via atexit in Python)
  m.def("_reset_boxed_python_overrides", [](){
    nb::gil_scoped_acquire gil;
    g_overrides = std::make_shared<OverrideMap>();
  });

  // Dispatcher-backed ops under submodule vt
  nb::module_ vt = m.def_submodule("vt");
  vbt_py::bind_ops(vt);

#if VBT_WITH_CUDA
  // Note: avoid registering atexit synchronizer to prevent teardown hazards in tests
#endif
}
