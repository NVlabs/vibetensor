// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/dispatch/dispatcher.h"

#include "vbt/plugin/vbt_plugin.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "vbt/cuda/device.h"
#include "vbt/cuda/guard.h"

#if VBT_WITH_AUTOGRAD
#include "vbt/autograd/detail/stats_internal.h"
#endif

namespace vbt {
namespace dispatch {

thread_local const OperatorEntry* Dispatcher::tls_redispatch_op_ = nullptr;
thread_local const OperatorEntry* Dispatcher::tls_current_op_ = nullptr;
thread_local bool Dispatcher::tls_skip_autograd_ = false;

namespace {

#if VBT_INTERNAL_TESTS
std::atomic<int> g_dispatch_v2_fabric_no_cuda_calls_override{-1};  // -1=no override, 0=force off, 1=force on
#endif

std::once_flag g_dispatch_v2_fabric_no_cuda_calls_once;
bool g_dispatch_v2_fabric_no_cuda_calls_cached = false;

bool IsTruthyEnv(const char* v) {
  if (!v || *v == '\0') return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "1" || s == "true" || s == "yes";
}

std::atomic<std::uint64_t> g_dispatch_v2_state_version{0};

std::vector<std::unique_ptr<OpDispatchStateV2>> g_dispatch_v2_states;

std::unique_ptr<OpDispatchStateV2> compute_state_v2_locked(const OperatorEntry& entry) {
  auto st = std::make_unique<OpDispatchStateV2>();
  st->fqname = entry.schema.fqname;
  st->in_arity = entry.schema.in_arity;

  st->device_policy = entry.device_policy;
  st->dispatch_arg_mask = entry.dispatch_arg_mask;
  st->allow_undefined_mask = entry.allow_undefined_mask;
  std::copy_n(entry.constraint_kind_by_index,
              kV2DevicePolicyMaxArity,
              st->constraint_kind_by_index);
  st->allow_multi_device_fabric = entry.allow_multi_device_fabric;

  DispatchKeySet wrappers;
  if (entry.autograd_fallback.boxed_ctx != nullptr) {
    wrappers = wrappers.add(DispatchKey::Autograd);
    st->autograd_fallback = entry.autograd_fallback;
  }
  if (entry.boxed_override != nullptr) {
    wrappers = wrappers.add(DispatchKey::Python);
    st->boxed_override = entry.boxed_override;
  }
  st->present_wrappers = wrappers;

  st->has_cpu = entry.cpu_base.has_value();
  if (st->has_cpu) st->cpu_base = *entry.cpu_base;

  st->has_cuda = entry.cuda_base.has_value();
  if (st->has_cuda) st->cuda_base = *entry.cuda_base;

  st->version =
      g_dispatch_v2_state_version.fetch_add(1, std::memory_order_relaxed) + 1;
  return st;
}

void publish_state_v2_locked(OperatorEntry& entry) {
  std::unique_ptr<OpDispatchStateV2> st = compute_state_v2_locked(entry);
  const OpDispatchStateV2* raw = st.get();
  g_dispatch_v2_states.emplace_back(std::move(st));
  entry.state_v2.store(raw, std::memory_order_release);
}


}  // namespace


// NOTE: This flag is only consulted in the dispatcher v2 Fabric bypass helper.
bool dispatch_v2_fabric_no_cuda_calls() {
#if VBT_INTERNAL_TESTS
  int ov = g_dispatch_v2_fabric_no_cuda_calls_override.load(
      std::memory_order_acquire);
  if (ov != -1) return ov == 1;
#endif
  std::call_once(g_dispatch_v2_fabric_no_cuda_calls_once, [] {
    const char* env = std::getenv("VBT_DISPATCH_V2_FABRIC_NO_CUDA_CALLS");
    g_dispatch_v2_fabric_no_cuda_calls_cached =
        (env == nullptr) ? true : IsTruthyEnv(env);
  });
  return g_dispatch_v2_fabric_no_cuda_calls_cached;
}

#if VBT_INTERNAL_TESTS
DispatchV2ModeGuard::DispatchV2ModeGuard(bool enabled) : prev_(enabled ? 1 : 0) {
  if (!enabled) {
    throw std::invalid_argument(
        "DispatchV2ModeGuard(false) is unsupported: dispatcher v1 retired");
  }
}

DispatchV2ModeGuard::~DispatchV2ModeGuard() = default;

DispatchV2FabricNoCudaCallsGuard::DispatchV2FabricNoCudaCallsGuard(bool enabled)
    : prev_(g_dispatch_v2_fabric_no_cuda_calls_override.exchange(
          enabled ? 1 : 0, std::memory_order_acq_rel)) {}

DispatchV2FabricNoCudaCallsGuard::~DispatchV2FabricNoCudaCallsGuard() {
  g_dispatch_v2_fabric_no_cuda_calls_override.store(prev_,
                                                    std::memory_order_release);
}
#endif  // VBT_INTERNAL_TESTS

Dispatcher& Dispatcher::instance() {
  static Dispatcher inst;
  return inst;
}

template <class Fn>
decltype(auto) Dispatcher::mutate_and_publish_locked(
    const std::string& fqname,
    const char* undefined_msg_prefix,
    bool undefined_is_invalid_argument,
    Fn&& fn) {
  std::lock_guard<std::mutex> lg(mu_);
  auto it = registry_.find(fqname);
  if (it == registry_.end() || !it->second) {
    std::string msg = std::string(undefined_msg_prefix) + fqname;
    if (undefined_is_invalid_argument) throw std::invalid_argument(msg);
    throw std::runtime_error(msg);
  }

  OperatorEntry& entry = *it->second;

  if constexpr (std::is_void_v<std::invoke_result_t<Fn, OperatorEntry&>>) {
    std::forward<Fn>(fn)(entry);
    publish_state_v2_locked(entry);
  } else {
    auto out = std::forward<Fn>(fn)(entry);
    publish_state_v2_locked(entry);
    return out;
  }
}

void Dispatcher::registerLibrary(const std::string& /*ns*/) {
}

struct ParsedSignature {
  uint8_t in_arity{0};
  bool has_non_tensor_args{false};
};

static ParsedSignature parse_signature(const std::string& def) {
  auto l = def.find('(');
  auto r = def.find(')');
  if (l == std::string::npos || r == std::string::npos || r < l) {
    throw std::runtime_error("malformed def: missing '()'");
  }
  auto arrow = def.find("->");
  if (arrow == std::string::npos) throw std::runtime_error("malformed def: missing '->'");
  auto ret = def.substr(arrow + 2);
  if (ret.find("Tensor") == std::string::npos) throw std::runtime_error("malformed def: return must be Tensor");

  std::string args = def.substr(l + 1, r - l - 1);
  ParsedSignature out;

  auto is_space = [](char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };

  std::size_t start = 0;
  while (start <= args.size()) {
    std::size_t end = args.find(',', start);
    if (end == std::string::npos) end = args.size();

    // Trim [start, end)
    std::size_t a = start;
    while (a < end && is_space(args[a])) ++a;
    std::size_t b = end;
    while (b > a && is_space(args[b - 1])) --b;

    if (b > a) {
      std::string token = args.substr(a, b - a);
      if (token.rfind("Tensor", 0) == 0) {
        // Treat "Tensor" and "Tensor?"-style tokens as tensor args.
        ++out.in_arity;
      } else {
        out.has_non_tensor_args = true;
      }
    }

    if (end == args.size()) break;
    start = end + 1;
  }

  return out;
}

OperatorHandle Dispatcher::def(const std::string& def_string) {
  std::lock_guard<std::mutex> lg(mu_);
  auto p = def_string.find('(');
  if (p == std::string::npos) throw std::runtime_error("malformed def: missing '->'");
  std::string fqname = def_string.substr(0, p);
  if (registry_.count(fqname)) throw std::runtime_error("duplicate def: " + fqname);
  auto entry = std::make_unique<OperatorEntry>();
  entry->schema.fqname = fqname;
  auto sig = parse_signature(def_string);
  entry->schema.in_arity = sig.in_arity;
  entry->schema.has_non_tensor_args = sig.has_non_tensor_args;
  //
  // These ops participate in mixed-device + metadata patterns; encode them in
  // the v2 snapshot so v2 dispatch does not rely on name allowlists.
  if (fqname == "vt::check_stream") {
    if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 2) {
      throw std::runtime_error(
          "vt::check_stream must have schema (Tensor, Tensor) -> Tensor");
    }
    entry->device_policy = DevicePolicy::MaskedSameDevice;
    entry->dispatch_arg_mask = 0b01;
    entry->allow_undefined_mask = 0;
    entry->constraint_kind_by_index[1] =
        ConstraintKind::MustBeCPUScalarInt64_0d;
  } else if (fqname == "vt::index") {
    if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 3) {
      throw std::runtime_error(
          "vt::index must have schema (Tensor, Tensor, Tensor) -> Tensor");
    }
    entry->device_policy = DevicePolicy::MaskedSameDevice;
    entry->dispatch_arg_mask = 0b001;
    entry->allow_undefined_mask = 0b010;
    entry->constraint_kind_by_index[1] = ConstraintKind::DeferToKernel;
    entry->constraint_kind_by_index[2] = ConstraintKind::DeferToKernel;
  } else if (fqname == "vt::index_put") {
    if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 5) {
      throw std::runtime_error(
          "vt::index_put must have schema (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    }
    entry->device_policy = DevicePolicy::MaskedSameDevice;
    entry->dispatch_arg_mask = 0b00001;
    entry->allow_undefined_mask = 0;
    entry->constraint_kind_by_index[1] = ConstraintKind::DeferToKernel;
    entry->constraint_kind_by_index[2] = ConstraintKind::DeferToKernel;
    entry->constraint_kind_by_index[3] = ConstraintKind::DeferToKernel;
    entry->constraint_kind_by_index[4] = ConstraintKind::DeferToKernel;
  } else if (fqname == "vt::embedding") {
    if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 5) {
      throw std::runtime_error(
          "vt::embedding must have schema (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    }
    entry->device_policy = DevicePolicy::MaskedSameDevice;
    entry->dispatch_arg_mask = 0b00011;
    entry->allow_undefined_mask = 0;
    entry->constraint_kind_by_index[2] = ConstraintKind::MustBeCPUScalarInt64_0d;
    entry->constraint_kind_by_index[3] = ConstraintKind::MustBeCPUScalarBool_0d;
    entry->constraint_kind_by_index[4] = ConstraintKind::MustBeCPUScalarBool_0d;
  } else if (fqname == "vt::embedding_renorm_") {
    if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 4) {
      throw std::runtime_error(
          "vt::embedding_renorm_ must have schema (Tensor, Tensor, Tensor, Tensor) -> Tensor");
    }
    entry->device_policy = DevicePolicy::MaskedSameDevice;
    entry->dispatch_arg_mask = 0b00011;
    entry->allow_undefined_mask = 0;
    entry->constraint_kind_by_index[2] = ConstraintKind::DeferToKernel;
    entry->constraint_kind_by_index[3] = ConstraintKind::DeferToKernel;
  }

  publish_state_v2_locked(*entry);
  auto* raw = entry.get();
  registry_.emplace(fqname, std::move(entry));
  return OperatorHandle{raw};
}

vt_status Dispatcher::apply_plugin_commit_plan(const PluginCommitPlan& plan,
                                              std::string* out_err) noexcept {
  if (out_err) out_err->clear();

  try {
    std::lock_guard<std::mutex> lg(mu_);

    const std::size_t pre_states = g_dispatch_v2_states.size();
    const std::uint64_t pre_version =
        g_dispatch_v2_state_version.load(std::memory_order_relaxed);

    struct PrevEntry {
      OperatorEntry* entry{nullptr};
      bool saved_cpu{false};
      bool saved_cuda{false};
      bool saved_policy{false};

      std::optional<KernelFunction> cpu_prev;
      std::optional<KernelFunction> cuda_prev;

      DevicePolicy policy_prev{DevicePolicy::AllSameDevice};
      std::uint64_t dispatch_arg_mask_prev{0};
      std::uint64_t allow_undefined_mask_prev{0};
      ConstraintKind constraint_kind_prev[kV2DevicePolicyMaxArity]{};
    };

    struct RollbackState {
      Dispatcher* self;
      std::size_t pre_states;
      std::uint64_t pre_version;
      std::vector<OperatorEntry*> inserted;
      std::vector<PrevEntry> prev;
      bool published{false};

      RollbackState(Dispatcher* d,
                    std::size_t pre_states_,
                    std::uint64_t pre_version_) noexcept
          : self(d), pre_states(pre_states_), pre_version(pre_version_) {}

      RollbackState(const RollbackState&) = delete;
      RollbackState& operator=(const RollbackState&) = delete;

      void rollback() noexcept {
        if (published || !self) return;
        try {
          // 1) Restore base kernels for mutated existing ops.
          for (auto& pe : prev) {
            if (!pe.entry) continue;
            if (pe.saved_cpu) pe.entry->cpu_base = pe.cpu_prev;
            if (pe.saved_cuda) pe.entry->cuda_base = pe.cuda_prev;
            if (pe.saved_policy) {
              pe.entry->device_policy = pe.policy_prev;
              pe.entry->dispatch_arg_mask = pe.dispatch_arg_mask_prev;
              pe.entry->allow_undefined_mask = pe.allow_undefined_mask_prev;
              std::copy_n(pe.constraint_kind_prev,
                          kV2DevicePolicyMaxArity,
                          pe.entry->constraint_kind_by_index);
            }
          }

          // 2) Remove newly inserted ops.
          if (!inserted.empty()) {
            for (auto it = self->registry_.begin();
                 it != self->registry_.end();) {
              OperatorEntry* e = it->second.get();
              if (std::find(inserted.begin(), inserted.end(), e) !=
                  inserted.end()) {
                it = self->registry_.erase(it);
              } else {
                ++it;
              }
            }
          }

          // 3) Restore published snapshot owner vector + version counter.
          if (g_dispatch_v2_states.size() > pre_states) {
            g_dispatch_v2_states.resize(pre_states);
          }
          g_dispatch_v2_state_version.store(pre_version,
                                           std::memory_order_relaxed);
        } catch (...) {
          // Best-effort rollback.
        }
      }

      ~RollbackState() noexcept { rollback(); }
    };

    RollbackState rb(this, pre_states, pre_version);
    rb.inserted.reserve(plan.defs.size());
    rb.prev.reserve(plan.kernels.size() + plan.policies.size());

    std::vector<OperatorEntry*> affected;
    affected.reserve(plan.defs.size() + plan.policies.size() + plan.kernels.size());

    auto set_err = [&](const std::string& msg) noexcept {
      if (!out_err) return;
      try {
        *out_err = msg;
      } catch (...) {
        // drop
      }
    };

    auto mark_affected = [&](OperatorEntry* e) {
      if (!e) return;
      if (std::find(affected.begin(), affected.end(), e) == affected.end()) {
        affected.push_back(e);
      }
    };

    auto in_defs = [&](const std::string& fqname) {
      for (const auto& d : plan.defs) {
        if (d.fqname == fqname) return true;
      }
      return false;
    };

    // --- 1) Validate plan (no mutation) ---
    for (std::size_t i = 0; i < plan.defs.size(); ++i) {
      const auto& d = plan.defs[i];
      if (d.fqname.empty()) {
        set_err("def: fqname is empty");
        return VT_STATUS_INVALID_ARG;
      }
      if (registry_.count(d.fqname)) {
        set_err("duplicate def: " + d.fqname);
        return VT_STATUS_INVALID_ARG;
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (plan.defs[j].fqname == d.fqname) {
          set_err("duplicate def: " + d.fqname);
          return VT_STATUS_INVALID_ARG;
        }
      }
    }

    for (const auto& k : plan.kernels) {
      if (k.key != DispatchKey::CPU && k.key != DispatchKey::CUDA) {
        set_err("unsupported dispatch key for plugin kernel: " + k.fqname);
        return VT_STATUS_UNSUPPORTED;
      }
      if (registry_.count(k.fqname) == 0 && !in_defs(k.fqname)) {
        set_err(std::string(k.key == DispatchKey::CPU
                                ? "undefined op in VBT_IMPL_CPU: "
                                : "undefined op in VBT_IMPL_CUDA: ") +
                k.fqname);
        return VT_STATUS_INVALID_ARG;
      }
    }

    for (std::size_t i = 0; i < plan.policies.size(); ++i) {
      const auto& p = plan.policies[i];
      if (p.fqname.empty()) {
        set_err("set_device_policy: fqname is empty");
        return VT_STATUS_INVALID_ARG;
      }
      if (!in_defs(p.fqname)) {
        set_err("set_device_policy: non-owned op");
        return VT_STATUS_UNSUPPORTED;
      }
      for (std::size_t j = 0; j < i; ++j) {
        if (plan.policies[j].fqname == p.fqname) {
          set_err("set_device_policy: duplicate policy: " + p.fqname);
          return VT_STATUS_INVALID_ARG;
        }
      }
    }

    // --- 2) Apply libraries ---
    //
    // NOTE: Dispatcher::registerLibrary is currently a no-op. If it ever becomes
    // stateful, rollback must be extended to restore that state on failure.
    for (const auto& ns : plan.libraries) {
      registerLibrary(ns);
    }

    // --- 3) Apply defs (no v2 publish yet) ---
    for (const auto& d : plan.defs) {
      try {
        const std::string& def_string = d.def_string;
        auto p = def_string.find('(');
        if (p == std::string::npos) {
          set_err("malformed def: missing '->'");
          return VT_STATUS_INVALID_ARG;
        }
        std::string fqname = def_string.substr(0, p);
        if (!d.fqname.empty() && d.fqname != fqname) {
          set_err("def: fqname mismatch");
          return VT_STATUS_INVALID_ARG;
        }

        auto entry = std::make_unique<OperatorEntry>();
        entry->schema.fqname = fqname;
        auto sig = parse_signature(def_string);
        entry->schema.in_arity = sig.in_arity;
        entry->schema.has_non_tensor_args = sig.has_non_tensor_args;

        if (fqname == "vt::check_stream") {
          if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 2) {
            throw std::runtime_error(
                "vt::check_stream must have schema (Tensor, Tensor) -> Tensor");
          }
          entry->device_policy = DevicePolicy::MaskedSameDevice;
          entry->dispatch_arg_mask = 0b01;
          entry->allow_undefined_mask = 0;
          entry->constraint_kind_by_index[1] =
              ConstraintKind::MustBeCPUScalarInt64_0d;
        } else if (fqname == "vt::index") {
          if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 3) {
            throw std::runtime_error(
                "vt::index must have schema (Tensor, Tensor, Tensor) -> Tensor");
          }
          entry->device_policy = DevicePolicy::MaskedSameDevice;
          entry->dispatch_arg_mask = 0b001;
          entry->allow_undefined_mask = 0b010;
          entry->constraint_kind_by_index[1] = ConstraintKind::DeferToKernel;
          entry->constraint_kind_by_index[2] = ConstraintKind::DeferToKernel;
        } else if (fqname == "vt::index_put") {
          if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 5) {
            throw std::runtime_error(
                "vt::index_put must have schema (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
          }
          entry->device_policy = DevicePolicy::MaskedSameDevice;
          entry->dispatch_arg_mask = 0b00001;
          entry->allow_undefined_mask = 0;
          entry->constraint_kind_by_index[1] = ConstraintKind::DeferToKernel;
          entry->constraint_kind_by_index[2] = ConstraintKind::DeferToKernel;
          entry->constraint_kind_by_index[3] = ConstraintKind::DeferToKernel;
          entry->constraint_kind_by_index[4] = ConstraintKind::DeferToKernel;
        } else if (fqname == "vt::embedding") {
          if (entry->schema.has_non_tensor_args || entry->schema.in_arity != 5) {
            throw std::runtime_error(
                "vt::embedding must have schema (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
          }
          entry->device_policy = DevicePolicy::MaskedSameDevice;
          entry->dispatch_arg_mask = 0b00011;
          entry->allow_undefined_mask = 0;
          entry->constraint_kind_by_index[2] = ConstraintKind::MustBeCPUScalarInt64_0d;
          entry->constraint_kind_by_index[3] = ConstraintKind::MustBeCPUScalarBool_0d;
          entry->constraint_kind_by_index[4] = ConstraintKind::MustBeCPUScalarBool_0d;
        }

        OperatorEntry* raw = entry.get();
        registry_.emplace(fqname, std::move(entry));
        rb.inserted.push_back(raw);
        mark_affected(raw);
      } catch (const std::bad_alloc&) {
        throw;
      } catch (const std::exception& e) {
        set_err(e.what());
        return VT_STATUS_INVALID_ARG;
      }
    }

    auto get_prev_entry = [&](OperatorEntry* entry) -> PrevEntry* {
      for (auto& pe : rb.prev) {
        if (pe.entry == entry) return &pe;
      }
      rb.prev.push_back(PrevEntry{entry});
      return &rb.prev.back();
    };

    auto ensure_prev_saved = [&](OperatorEntry* entry, DispatchKey key) {
      PrevEntry* pe = get_prev_entry(entry);
      if (key == DispatchKey::CPU) {
        if (!pe->saved_cpu) {
          pe->saved_cpu = true;
          pe->cpu_prev = entry->cpu_base;
        }
      } else {
        if (!pe->saved_cuda) {
          pe->saved_cuda = true;
          pe->cuda_prev = entry->cuda_base;
        }
      }
    };

    auto ensure_policy_prev_saved = [&](OperatorEntry* entry) {
      PrevEntry* pe = get_prev_entry(entry);
      if (!pe->saved_policy) {
        pe->saved_policy = true;
        pe->policy_prev = entry->device_policy;
        pe->dispatch_arg_mask_prev = entry->dispatch_arg_mask;
        pe->allow_undefined_mask_prev = entry->allow_undefined_mask;
        std::copy_n(entry->constraint_kind_by_index,
                    kV2DevicePolicyMaxArity,
                    pe->constraint_kind_prev);
      }
    };

    // --- 4) Apply policies (no v2 publish yet) ---
    for (const auto& p : plan.policies) {
      auto it = registry_.find(p.fqname);
      if (it == registry_.end() || !it->second) {
        set_err("set_device_policy: unknown op: " + p.fqname);
        return VT_STATUS_INVALID_ARG;
      }
      OperatorEntry* entry = it->second.get();
      ensure_policy_prev_saved(entry);

      entry->device_policy = p.policy;
      entry->dispatch_arg_mask = p.dispatch_arg_mask;
      entry->allow_undefined_mask = p.allow_undefined_mask;
      std::copy_n(p.constraint_kind_by_index,
                  kV2DevicePolicyMaxArity,
                  entry->constraint_kind_by_index);
      mark_affected(entry);
    }

    auto validate_and_install = [&](OperatorEntry& entry,
                                   const std::string& fqname,
                                   DispatchKey key,
                                   const KernelFunction& kf) -> bool {
      const char* null_prefix =
          (key == DispatchKey::CPU) ? "null function in VBT_IMPL_CPU: "
                                    : "null function in VBT_IMPL_CUDA: ";
      const char* arity_prefix =
          (key == DispatchKey::CPU) ? "arity mismatch in VBT_IMPL_CPU: "
                                    : "arity mismatch in VBT_IMPL_CUDA: ";

      // Null checks by mode.
      if (kf.mode == KernelFunction::Mode::Unboxed && kf.unboxed_ptr == nullptr) {
        set_err(std::string(null_prefix) + fqname);
        return false;
      }
      if (kf.mode == KernelFunction::Mode::Boxed && kf.boxed == nullptr) {
        set_err(std::string(null_prefix) + fqname);
        return false;
      }
      if (kf.mode == KernelFunction::Mode::BoxedWithCtx &&
          kf.boxed_ctx == nullptr) {
        set_err(std::string(null_prefix) + fqname);
        return false;
      }
      if (kf.mode == KernelFunction::Mode::Unboxed &&
          entry.schema.has_non_tensor_args) {
        set_err(
            "unboxed registration not allowed for ops with non-Tensor args: " +
            fqname +
            " (use boxed kernel or tensor-encoded metadata schema)");
        return false;
      }
      if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.in_arity > 4) {
        set_err("malformed def: args must be 0-4 Tensor");
        return false;
      }
      if (kf.arity != entry.schema.in_arity) {
        set_err(std::string(arity_prefix) + fqname);
        return false;
      }

      KernelFunction copy = kf;
      copy.arity = entry.schema.in_arity;

      ensure_prev_saved(&entry, key);
      if (key == DispatchKey::CPU) {
        entry.cpu_base = copy;
      } else {
        entry.cuda_base = copy;
      }
      return true;
    };

    // --- 5) Apply kernels (no v2 publish yet) ---
    for (const auto& k : plan.kernels) {
      auto it = registry_.find(k.fqname);
      if (it == registry_.end() || !it->second) {
        set_err(std::string(k.key == DispatchKey::CPU
                                ? "undefined op in VBT_IMPL_CPU: "
                                : "undefined op in VBT_IMPL_CUDA: ") +
                k.fqname);
        return VT_STATUS_INVALID_ARG;
      }
      OperatorEntry* entry = it->second.get();

      if (!validate_and_install(*entry, k.fqname, k.key, k.kf)) {
        return VT_STATUS_INVALID_ARG;
      }
      mark_affected(entry);
    }

    // --- 6) Compute + publish new v2 snapshots (publish-last) ---
    std::vector<std::unique_ptr<OpDispatchStateV2>> new_states;
    new_states.reserve(affected.size());
    std::vector<const OpDispatchStateV2*> raw_states;
    raw_states.reserve(affected.size());

    for (OperatorEntry* e : affected) {
      auto st = compute_state_v2_locked(*e);
      raw_states.push_back(st.get());
      new_states.emplace_back(std::move(st));
    }

    g_dispatch_v2_states.reserve(g_dispatch_v2_states.size() +
                                new_states.size());
    for (auto& st : new_states) {
      g_dispatch_v2_states.emplace_back(std::move(st));
    }

    for (std::size_t i = 0; i < affected.size(); ++i) {
      affected[i]->state_v2.store(raw_states[i], std::memory_order_release);
    }

    rb.published = true;
    return VT_STATUS_OK;
  } catch (const std::bad_alloc&) {
    if (out_err) {
      try {
        *out_err = "out of memory";
      } catch (...) {
        // drop
      }
    }
    return VT_STATUS_NOMEM;
  } catch (const std::exception& e) {
    if (out_err) {
      try {
        *out_err = e.what();
      } catch (...) {
        // drop
      }
    }
    return VT_STATUS_INTERNAL;
  } catch (...) {
    if (out_err) {
      try {
        *out_err = "internal error";
      } catch (...) {
        // drop
      }
    }
    return VT_STATUS_INTERNAL;
  }
}


void Dispatcher::registerBoxedOverride(const std::string& fqname, KernelFunction::BoxedFn fn) {
  mutate_and_publish_locked(
      fqname,
      "undefined op for boxed override: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) {
        if (entry.boxed_override != nullptr) {
          throw std::runtime_error("duplicate CPU impl (boxed): " + fqname);
        }
        entry.boxed_override = fn;
      });
}

bool Dispatcher::tryRegisterBoxedOverride(const std::string& fqname, KernelFunction::BoxedFn fn) {
  return mutate_and_publish_locked(
      fqname,
      "undefined op for boxed override: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) -> bool {
        if (entry.boxed_override != nullptr) return false;
        entry.boxed_override = fn;
        return true;
      });
}

void Dispatcher::registerCpuKernelFunction(const std::string& fqname, const KernelFunction& kf) {
  mutate_and_publish_locked(
      fqname,
      "undefined op in VBT_IMPL_CPU: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) {
        if (entry.cpu_base.has_value()) throw std::runtime_error("duplicate CPU impl (base): " + fqname);
        // Null checks by mode
        if (kf.mode == KernelFunction::Mode::Unboxed && kf.unboxed_ptr == nullptr) throw std::runtime_error("null function in VBT_IMPL_CPU: " + fqname);
        if (kf.mode == KernelFunction::Mode::Boxed && kf.boxed == nullptr) throw std::runtime_error("null function in VBT_IMPL_CPU: " + fqname);
        if (kf.mode == KernelFunction::Mode::BoxedWithCtx && kf.boxed_ctx == nullptr) throw std::runtime_error("null function in VBT_IMPL_CPU: " + fqname);
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.has_non_tensor_args) {
          throw std::runtime_error(
              "unboxed registration not allowed for ops with non-Tensor args: " + fqname +
              " (use boxed kernel or tensor-encoded metadata schema)");
        }
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.in_arity > 4) {
          throw std::runtime_error("malformed def: args must be 0-4 Tensor");
        }
        if (kf.arity != entry.schema.in_arity) throw std::runtime_error("arity mismatch in VBT_IMPL_CPU: " + fqname);
        KernelFunction copy = kf; copy.arity = entry.schema.in_arity; entry.cpu_base = copy;
      });
}

void Dispatcher::registerCudaKernelFunction(const std::string& fqname, const KernelFunction& kf) {
  mutate_and_publish_locked(
      fqname,
      "undefined op in VBT_IMPL_CUDA: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) {
        if (entry.cuda_base.has_value()) throw std::runtime_error("duplicate CUDA impl (base): " + fqname);
        if (kf.mode == KernelFunction::Mode::Unboxed && kf.unboxed_ptr == nullptr) throw std::runtime_error("null function in VBT_IMPL_CUDA: " + fqname);
        if (kf.mode == KernelFunction::Mode::Boxed && kf.boxed == nullptr) throw std::runtime_error("null function in VBT_IMPL_CUDA: " + fqname);
        if (kf.mode == KernelFunction::Mode::BoxedWithCtx && kf.boxed_ctx == nullptr) throw std::runtime_error("null function in VBT_IMPL_CUDA: " + fqname);
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.has_non_tensor_args) {
          throw std::runtime_error(
              "unboxed registration not allowed for ops with non-Tensor args: " + fqname +
              " (use boxed kernel or tensor-encoded metadata schema)");
        }
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.in_arity > 4) {
          throw std::runtime_error("malformed def: args must be 0-4 Tensor");
        }
        if (kf.arity != entry.schema.in_arity) throw std::runtime_error("arity mismatch in VBT_IMPL_CUDA: " + fqname);
        KernelFunction copy = kf; copy.arity = entry.schema.in_arity; entry.cuda_base = copy;
      });
}

std::optional<KernelFunction> Dispatcher::replaceCpuKernelFunction(const std::string& fqname, const KernelFunction& kf) {
  return mutate_and_publish_locked(
      fqname,
      "undefined op in VBT_IMPL_CPU: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) -> std::optional<KernelFunction> {
        if (kf.mode == KernelFunction::Mode::Unboxed && kf.unboxed_ptr == nullptr) throw std::runtime_error("null function in VBT_IMPL_CPU: " + fqname);
        if (kf.mode == KernelFunction::Mode::Boxed && kf.boxed == nullptr) throw std::runtime_error("null function in VBT_IMPL_CPU: " + fqname);
        if (kf.mode == KernelFunction::Mode::BoxedWithCtx && kf.boxed_ctx == nullptr) throw std::runtime_error("null function in VBT_IMPL_CPU: " + fqname);
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.has_non_tensor_args) {
          throw std::runtime_error(
              "unboxed registration not allowed for ops with non-Tensor args: " + fqname +
              " (use boxed kernel or tensor-encoded metadata schema)");
        }
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.in_arity > 4) {
          throw std::runtime_error("malformed def: args must be 0-4 Tensor");
        }
        if (kf.arity != entry.schema.in_arity) throw std::runtime_error("arity mismatch in VBT_IMPL_CPU: " + fqname);
        std::optional<KernelFunction> prev = entry.cpu_base;
        KernelFunction copy = kf; copy.arity = entry.schema.in_arity; entry.cpu_base = copy;
        return prev;
      });
}

std::optional<KernelFunction> Dispatcher::replaceCudaKernelFunction(const std::string& fqname, const KernelFunction& kf) {
  return mutate_and_publish_locked(
      fqname,
      "undefined op in VBT_IMPL_CUDA: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) -> std::optional<KernelFunction> {
        if (kf.mode == KernelFunction::Mode::Unboxed && kf.unboxed_ptr == nullptr) throw std::runtime_error("null function in VBT_IMPL_CUDA: " + fqname);
        if (kf.mode == KernelFunction::Mode::Boxed && kf.boxed == nullptr) throw std::runtime_error("null function in VBT_IMPL_CUDA: " + fqname);
        if (kf.mode == KernelFunction::Mode::BoxedWithCtx && kf.boxed_ctx == nullptr) throw std::runtime_error("null function in VBT_IMPL_CUDA: " + fqname);
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.has_non_tensor_args) {
          throw std::runtime_error(
              "unboxed registration not allowed for ops with non-Tensor args: " + fqname +
              " (use boxed kernel or tensor-encoded metadata schema)");
        }
        if (kf.mode == KernelFunction::Mode::Unboxed && entry.schema.in_arity > 4) {
          throw std::runtime_error("malformed def: args must be 0-4 Tensor");
        }
        if (kf.arity != entry.schema.in_arity) throw std::runtime_error("arity mismatch in VBT_IMPL_CUDA: " + fqname);
        std::optional<KernelFunction> prev = entry.cuda_base;
        KernelFunction copy = kf; copy.arity = entry.schema.in_arity; entry.cuda_base = copy;
        return prev;
      });
}

std::optional<KernelFunction> Dispatcher::uninstallCpuKernelFunction(const std::string& fqname) {
  return mutate_and_publish_locked(
      fqname,
      "undefined op in VBT_IMPL_CPU: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) -> std::optional<KernelFunction> {
        std::optional<KernelFunction> prev = entry.cpu_base;
        entry.cpu_base.reset();
        return prev;
      });
}

std::optional<KernelFunction> Dispatcher::uninstallCudaKernelFunction(const std::string& fqname) {
  return mutate_and_publish_locked(
      fqname,
      "undefined op in VBT_IMPL_CUDA: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) -> std::optional<KernelFunction> {
        std::optional<KernelFunction> prev = entry.cuda_base;
        entry.cuda_base.reset();
        return prev;
      });
}

void Dispatcher::registerAutogradFallback(const std::string& fqname, const KernelFunction& kf) {
  mutate_and_publish_locked(
      fqname,
      "undefined op for autograd fallback: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) {
        if (entry.is_fabric_op) {
          throw std::runtime_error(
              "[Fabric] Fabric ops must not register autograd fallbacks: " + fqname);
        }
        if (entry.schema.in_arity == 0) throw std::runtime_error("autograd fallback not supported for nullary ops: " + fqname);
        if (entry.autograd_fallback.boxed_ctx != nullptr) throw std::runtime_error("duplicate autograd fallback: " + fqname);
        if (kf.mode != KernelFunction::Mode::BoxedWithCtx || kf.boxed_ctx == nullptr) throw std::runtime_error("autograd fallback must be BoxedWithCtx: " + fqname);
        if (kf.arity != entry.schema.in_arity) throw std::runtime_error("arity mismatch in autograd fallback: " + fqname);
        KernelFunction copy = kf; copy.arity = entry.schema.in_arity; copy.ctx = static_cast<void*>(&entry); entry.autograd_fallback = copy;
      });
}

bool Dispatcher::tryRegisterAutogradFallback(const std::string& fqname, const KernelFunction& kf) {
  return mutate_and_publish_locked(
      fqname,
      "undefined op for autograd fallback: ",
      /*undefined_is_invalid_argument=*/false,
      [&](OperatorEntry& entry) -> bool {
        if (entry.is_fabric_op) {
          throw std::runtime_error(
              "[Fabric] Fabric ops must not register autograd fallbacks: " + fqname);
        }
        if (entry.schema.in_arity == 0) throw std::runtime_error("autograd fallback not supported for nullary ops: " + fqname);
        if (entry.autograd_fallback.boxed_ctx != nullptr) return false;
        if (kf.mode != KernelFunction::Mode::BoxedWithCtx || kf.boxed_ctx == nullptr) throw std::runtime_error("autograd fallback must be BoxedWithCtx: " + fqname);
        if (kf.arity != entry.schema.in_arity) throw std::runtime_error("arity mismatch in autograd fallback: " + fqname);
        KernelFunction copy = kf; copy.arity = entry.schema.in_arity; copy.ctx = static_cast<void*>(&entry); entry.autograd_fallback = copy;
        return true;
      });
}

namespace {

static inline std::int64_t extract_cpu_scalar_int64(const vbt::core::TensorImpl& t,
                                                    const std::string& fqname,
                                                    const char* argname) {
  using vbt::core::ScalarType;
  if (t.device().type != kDLCPU) {
    throw std::runtime_error(std::string("[Fabric] ") + fqname + ": " + argname +
                             " must be a CPU scalar int64 tensor");
  }
  if (t.numel() != 1) {
    throw std::runtime_error(std::string("[Fabric] ") + fqname + ": " + argname +
                             " must have numel()==1");
  }
  if (t.dtype() != ScalarType::Int64) {
    throw std::runtime_error(std::string("[Fabric] ") + fqname + ": " + argname +
                             " must have dtype int64");
  }
  const void* p = t.data();
  if (!p) {
    throw std::runtime_error(std::string("[Fabric] ") + fqname + ": " + argname +
                             " has no data");
  }
  return *static_cast<const std::int64_t*>(p);
}



enum class DeviceCheckMode { OpAllowsMixed, StrictSameDevice };

static void call_boxed_fabric_cuda_base_v2(const OpDispatchStateV2& st, BoxedStack& s) {
  //   (Tensor a, Tensor b, compute_device, require_fabric, use_copy_fallback)
  // where the last three arguments are represented as 0-d CPU scalar int64
  // tensors.
  if (st.in_arity != 5) {
    throw std::runtime_error(
        std::string("[Fabric] ") + st.fqname +
        ": allow_multi_device_fabric requires arity==5 (2 tensor operands + 3 CPU scalar metadata tensors)");
  }
  if (s.size() != 5) {
    throw std::runtime_error(
        std::string("[Fabric] ") + st.fqname +
        ": internal error: stack size mismatch for Fabric dispatch");
  }

  constexpr std::size_t idx_compute = 2;
  const std::int64_t compute_device_index =
      extract_cpu_scalar_int64(s[idx_compute], st.fqname, "compute_device");
  const std::int64_t require_fabric =
      extract_cpu_scalar_int64(s[idx_compute + 1], st.fqname, "require_fabric");
  const std::int64_t use_copy_fallback =
      extract_cpu_scalar_int64(s[idx_compute + 2], st.fqname, "use_copy_fallback");

  if (!(require_fabric == 0 || require_fabric == 1)) {
    throw std::runtime_error(
        std::string("[Fabric] ") + st.fqname +
        ": require_fabric must be 0 or 1");
  }
  if (!(use_copy_fallback == 0 || use_copy_fallback == 1)) {
    throw std::runtime_error(
        std::string("[Fabric] ") + st.fqname +
        ": use_copy_fallback must be 0 or 1");
  }

  const bool no_cuda_calls = dispatch_v2_fabric_no_cuda_calls();
  if (!no_cuda_calls) {
    // Default behavior: validate visible CUDA devices and pin the current device.
    const int dc = vbt::cuda::device_count();
    if (compute_device_index < 0 || compute_device_index >= dc) {
      throw std::runtime_error(
          std::string("[Fabric] ") + st.fqname +
          ": compute_device=" + std::to_string(compute_device_index) +
          " is out of range for visible CUDA devices (device_count=" +
          std::to_string(dc) + ")");
    }

    const vbt::core::Device primary = vbt::core::Device::cuda(
        static_cast<std::int32_t>(compute_device_index));

    if (s[0].device().type != kDLCUDA || s[1].device().type != kDLCUDA) {
      throw std::runtime_error(
          std::string("[Fabric] ") + st.fqname +
          ": Fabric ops require CUDA tensor operands");
    }

    const bool a_on_primary = (s[0].device() == primary);
    const bool b_on_primary = (s[1].device() == primary);
    if (!a_on_primary && !b_on_primary) {
      throw std::runtime_error(
          std::string("[Fabric] ") + st.fqname +
          ": compute_device must match one of the operand devices");
    }

    if (!st.has_cuda) {
      throw std::runtime_error(
          std::string("[Fabric] Fabric op has no CUDA kernel registered: ") +
          st.fqname);
    }

    // Guard the active CUDA device so downstream CUDA kernels that use the
    // current stream/device run on the intended primary.
    vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(compute_device_index));

    st.cuda_base.callBoxed(st.fqname, s);
    return;
  }

  // No-CUDA-calls mode: avoid direct CUDA calls; kernels are responsible for
  // pinning the current CUDA device as needed.
  if (s[0].device().type != kDLCUDA || s[1].device().type != kDLCUDA) {
    throw std::runtime_error(
        std::string("[Fabric] ") + st.fqname +
        ": Fabric ops require CUDA tensor operands");
  }

  const std::int64_t max_i32 =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
  // Also cap by DeviceIndex::max() to keep compute_device representable in the
  // type used by DeviceGuard and other CUDA helpers (int16_t today), even though
  // this branch does not construct a DeviceGuard.
  const std::int64_t max_dev_index =
      static_cast<std::int64_t>(std::numeric_limits<vbt::cuda::DeviceIndex>::max());
  const std::int64_t max_allowed = std::min(max_i32, max_dev_index);

  if (compute_device_index < 0 || compute_device_index > max_allowed) {
    throw std::runtime_error(std::string("[Fabric] ") + st.fqname +
                             ": compute_device=" +
                             std::to_string(compute_device_index) +
                             " is invalid (expected CUDA device index)");
  }

  const std::int64_t a_idx = static_cast<std::int64_t>(s[0].device().index);
  const std::int64_t b_idx = static_cast<std::int64_t>(s[1].device().index);
  const bool match_a =
      (s[0].device().type == kDLCUDA) && (compute_device_index == a_idx);
  const bool match_b =
      (s[1].device().type == kDLCUDA) && (compute_device_index == b_idx);
  if (!match_a && !match_b) {
    throw std::runtime_error(
        std::string("[Fabric] ") + st.fqname +
        ": compute_device must match one of the operand devices");
  }

  if (!st.has_cuda) {
    throw std::runtime_error(
        std::string("[Fabric] Fabric op has no CUDA kernel registered: ") +
        st.fqname);
  }

  // In no-CUDA-calls mode, the dispatcher does not set the current CUDA device.
  // Allowlisted kernels must be robust to current_device != compute_device
  // (e.g., pin internally with DeviceGuard).
  st.cuda_base.callBoxed(st.fqname, s);
}

static inline bool is_defined(const vbt::core::TensorImpl& t) noexcept {
  return t.storage().get() != nullptr;
}

[[noreturn]] static void throw_mixed_device_error(const vbt::core::Device& a,
                                                  const vbt::core::Device& b) {
  // Mixed-device error with parity substring.
  throw std::runtime_error(
      std::string("Expected all tensors to be on the same device, but found ") +
      a.to_string() + " and " + b.to_string());
}

[[noreturn]] static void throw_cannot_determine_dispatch_device(
    const std::string& fqname) {
  throw std::invalid_argument(fqname + ": cannot determine dispatch device");
}

[[noreturn]] static void throw_scalar_constraint_mismatch(
    const std::string& fqname,
    std::size_t index,
    const char* expected) {
  throw std::invalid_argument(fqname + ": arg[" + std::to_string(index) +
                              "] expected " + expected);
}

static void validate_cpu_scalar_0d(const vbt::core::TensorImpl& t,
                                  const std::string& fqname,
                                  std::size_t index,
                                  vbt::core::ScalarType expected_dtype,
                                  const char* expected_str,
                                  bool allow_undefined) {
  if (!is_defined(t)) {
    if (allow_undefined) return;
    throw_scalar_constraint_mismatch(fqname, index, expected_str);
  }

  if (t.device().type != kDLCPU) {
    throw_scalar_constraint_mismatch(fqname, index, expected_str);
  }
  if (t.dtype() != expected_dtype) {
    throw_scalar_constraint_mismatch(fqname, index, expected_str);
  }
  if (!t.sizes().empty()) {
    throw_scalar_constraint_mismatch(fqname, index, expected_str);
  }
  if (t.numel() > 0 && t.data() == nullptr) {
    throw_scalar_constraint_mismatch(fqname, index, expected_str);
  }
}

static void validate_defer_to_kernel(const vbt::core::TensorImpl& t,
                                    const std::string& fqname,
                                    std::size_t index,
                                    bool allow_undefined) {
  if (!is_defined(t)) {
    if (allow_undefined) return;
    throw std::invalid_argument(fqname + ": arg[" + std::to_string(index) +
                                "] is undefined");
  }
  // Defensive null-data guard: several kernels assume data()!=nullptr when
  // numel()>0 and will dereference.
  if (t.numel() > 0 && t.data() == nullptr) {
    throw std::invalid_argument(fqname + ": arg[" + std::to_string(index) +
                                "] has no data");
  }
}

static vbt::core::Device select_dispatch_device_all_same_device(
    const OpDispatchStateV2& st,
    const BoxedStack& s) {
  for (const auto& t : s) {
    if (is_defined(t)) {
      return t.device();
    }
  }
  throw_cannot_determine_dispatch_device(st.fqname);
}

static void callBoxed_v2_base(const OpDispatchStateV2& st,
                              BoxedStack& s,
                              DeviceCheckMode mode) {
  // 1) Arity guard (must be first; stack may be empty).
  if (s.size() != st.in_arity) {
    throw std::invalid_argument("arity mismatch " + st.fqname + ": expected " +
                                std::to_string(st.in_arity) + ", got " +
                                std::to_string(s.size()));
  }

  // 2) Nullary carve-out: CPU-only.
  //
  // dispatch device" failure for in_arity==0.
  if (st.in_arity == 0) {
    if (!st.has_cpu) {
      throw std::runtime_error("no CPU kernel registered: " + st.fqname);
    }
    st.cpu_base.callBoxed(st.fqname, s);
    return;
  }

  if (st.allow_multi_device_fabric) {
    call_boxed_fabric_cuda_base_v2(st, s);
    return;
  }

  // 4) Determine dispatch device.
  //
  // - AllSameDevice: first defined arg (undefined ignored)
  // - MaskedSameDevice: arg0 (must be defined)
  vbt::core::Device dispatch_device;
  if (st.device_policy == DevicePolicy::MaskedSameDevice) {
    if (!is_defined(s[0])) {
      throw_cannot_determine_dispatch_device(st.fqname);
    }
    dispatch_device = s[0].device();
  } else {
    dispatch_device = select_dispatch_device_all_same_device(st, s);
  }

  // Defense-in-depth: MaskedSameDevice uses 64-bit masks and a 64-entry
  // constraint table.
  if (st.device_policy == DevicePolicy::MaskedSameDevice &&
      st.in_arity > kV2DevicePolicyMaxArity) {
    throw std::logic_error(
        "internal error: MaskedSameDevice requires in_arity <= 64: " + st.fqname);
  }

  // 5) Device validation.
  if (mode == DeviceCheckMode::StrictSameDevice) {
    // Strict mode overrides per-op mixed-device exceptions.
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (!is_defined(s[i])) continue;
      if (s[i].device() != dispatch_device) {
        throw_mixed_device_error(dispatch_device, s[i].device());
      }
    }
  } else if (st.device_policy == DevicePolicy::AllSameDevice ||
             st.device_policy == DevicePolicy::Fabric5Arg) {
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (!is_defined(s[i])) continue;
      if (s[i].device() != dispatch_device) {
        throw_mixed_device_error(dispatch_device, s[i].device());
      }
    }
  } else if (st.device_policy == DevicePolicy::MaskedSameDevice) {
    // Validate dispatch args by mask; validate non-dispatch args by constraint.
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (!is_defined(s[i])) continue;
      const std::uint64_t bit = static_cast<std::uint64_t>(1) << i;
      if ((st.dispatch_arg_mask & bit) != 0) {
        if (s[i].device() != dispatch_device) {
          throw_mixed_device_error(dispatch_device, s[i].device());
        }
      } else if (st.constraint_kind_by_index[i] ==
                 ConstraintKind::MustMatchDispatchDeviceIfDefined) {
        if (s[i].device() != dispatch_device) {
          throw_mixed_device_error(dispatch_device, s[i].device());
        }
      }
    }
  } else {
    throw std::runtime_error("internal error: unknown device policy in dispatcher: " +
                             st.fqname);
  }

  // 6) Constraint validation (MaskedSameDevice only).
  if (st.device_policy == DevicePolicy::MaskedSameDevice) {
    using vbt::core::ScalarType;

    for (std::size_t i = 0; i < s.size(); ++i) {
      const std::uint64_t bit = static_cast<std::uint64_t>(1) << i;
      if ((st.dispatch_arg_mask & bit) != 0) continue;

      const bool allow_undef = (st.allow_undefined_mask & bit) != 0;
      switch (st.constraint_kind_by_index[i]) {
        case ConstraintKind::MustMatchDispatchDeviceIfDefined:
          // Already validated by device check above.
          break;

        case ConstraintKind::MustBeCPUScalarInt64_0d:
          validate_cpu_scalar_0d(s[i], st.fqname, i, ScalarType::Int64,
                                 "CPU int64 scalar (0-d)", allow_undef);
          break;

        case ConstraintKind::MustBeCPUScalarBool_0d:
          validate_cpu_scalar_0d(s[i], st.fqname, i, ScalarType::Bool,
                                 "CPU bool scalar (0-d)", allow_undef);
          break;

        case ConstraintKind::DeferToKernel:
          validate_defer_to_kernel(s[i], st.fqname, i, allow_undef);
          break;

        default:
          throw std::invalid_argument(st.fqname + ": arg[" + std::to_string(i) +
                                      "] has unknown ConstraintKind");
      }
    }
  }

  // 7) Backend selection by dispatch device.
  if (dispatch_device.type == kDLCPU) {
    if (!st.has_cpu) {
      throw std::runtime_error("no CPU kernel registered: " + st.fqname);
    }
    st.cpu_base.callBoxed(st.fqname, s);
    return;
  } else if (dispatch_device.type == kDLCUDA) {
    if (!st.has_cuda) {
      throw std::runtime_error("no CUDA kernel registered: " + st.fqname);
    }
    st.cuda_base.callBoxed(st.fqname, s);
    return;
  } else {
    throw std::runtime_error("unsupported device in dispatcher: " + st.fqname);
  }
}

static void callBoxed_v2(const OperatorEntry& entry,
                         const OpDispatchStateV2& st,
                         BoxedStack& s,
                         const OperatorEntry* redispatch_op) {
  // 0) Build keyset without inspecting stack elements.
  DispatchKeySet ks;
  if (st.present_wrappers.has(DispatchKey::Autograd)) {
    ks = ks.add(DispatchKey::Autograd);
  }
  if (st.present_wrappers.has(DispatchKey::Python)) {
    ks = ks.add(DispatchKey::Python);
  }
  // Base sentinel: always present and never excludable.
  ks = ks.add(DispatchKey::CPU);

  // 1) Apply TLS include/exclude.
  ks = apply_tls(ks);

  // 2) Mask out wrapper keys that are not present.
  // Allow base keys even if TLS included them.
  constexpr DispatchKeySet base_keys = DispatchKeySet{}
                                          .add(DispatchKey::CPU)
                                          .add(DispatchKey::CUDA)
                                          .add(DispatchKey::Fabric);
  ks = ks & (st.present_wrappers | base_keys);

  // 3) v1 redispatch safety net: if RedispatchGuard active for this entry,
  // wrappers MUST be skipped.
  if (redispatch_op == &entry) {
    ks = ks.remove(DispatchKey::Autograd).remove(DispatchKey::Python);
  }

  // 4) SkipAutogradGuard parity + stats hook.
  const bool autograd_candidate = ks.has(DispatchKey::Autograd);
  if (st.present_wrappers.has(DispatchKey::Autograd) &&
      Dispatcher::tls_skip_autograd_ &&
      autograd_candidate &&
      redispatch_op != &entry) {
#if VBT_WITH_AUTOGRAD
    vbt::autograd::_stats_wrapper_guard_skipped();
#endif
  }
  if (Dispatcher::tls_skip_autograd_) {
    ks = ks.remove(DispatchKey::Autograd);
  }

  // 5) Select and execute.
  DispatchKey k = ks.highest_priority_key();
  switch (k) {
    case DispatchKey::Autograd:
      st.autograd_fallback.callBoxed(st.fqname, s);
      return;

    case DispatchKey::Python: {
      StackGuard guard(s, st.in_arity);
      Dispatcher::CurrentOpGuard cg(&entry);
      st.boxed_override(s);
      if (s.size() != 1) {
        throw std::runtime_error(
            "boxed kernel did not produce exactly one result: " + st.fqname);
      }
      guard.commit();
      return;
    }

    default:
      callBoxed_v2_base(st, s, DeviceCheckMode::OpAllowsMixed);
      return;
  }
}



} // namespace

OperatorHandle Dispatcher::find(const std::string& name) {
  std::lock_guard<std::mutex> lg(mu_);
  auto it = registry_.find(name);
  if (it == registry_.end()) throw std::runtime_error("unknown op: " + name);
  return OperatorHandle{it->second.get()};
}
void Dispatcher::mark_fabric_op(const std::string& fqname,
                              bool is_fabric_op,
                              bool allow_multi_device_fabric) {
  mutate_and_publish_locked(
      fqname,
      "[Fabric] mark_fabric_op: unknown op: ",
      /*undefined_is_invalid_argument=*/true,
      [&](OperatorEntry& entry) {
        if (is_fabric_op && entry.autograd_fallback.boxed_ctx != nullptr) {
          throw std::runtime_error(
              "[Fabric] Fabric ops must not register autograd fallbacks: " + fqname);
        }

        if (allow_multi_device_fabric && !is_fabric_op) {
          throw std::runtime_error(
              "[Fabric] allow_multi_device_fabric requires is_fabric_op=true: " + fqname);
        }

        // Enforce Fabric invariant at the source of truth.
        if (allow_multi_device_fabric) {
          if (entry.schema.in_arity != 5) {
            throw std::runtime_error(
                "[Fabric] mark_fabric_op: expected arity==5, got " +
                std::to_string(entry.schema.in_arity) + ": " + fqname);
          }
          if (entry.schema.has_non_tensor_args) {
            throw std::runtime_error(
                "[Fabric] mark_fabric_op: expected has_non_tensor_args==false: " +
                fqname);
          }
          // Keep v2 metadata consistent with the bypass state.
          entry.device_policy = DevicePolicy::Fabric5Arg;
          entry.dispatch_arg_mask = 0;
          entry.allow_undefined_mask = 0;
          std::fill_n(entry.constraint_kind_by_index,
                      kV2DevicePolicyMaxArity,
                      ConstraintKind::MustMatchDispatchDeviceIfDefined);
        } else {
          // Deterministic restore: avoid a confusing "Fabric5Arg policy but no
          // bypass" state.
          entry.device_policy = DevicePolicy::AllSameDevice;
          entry.dispatch_arg_mask = 0;
          entry.allow_undefined_mask = 0;
          std::fill_n(entry.constraint_kind_by_index,
                      kV2DevicePolicyMaxArity,
                      ConstraintKind::MustMatchDispatchDeviceIfDefined);
        }

        entry.is_fabric_op = is_fabric_op;
        entry.allow_multi_device_fabric = allow_multi_device_fabric;
      });
}

void Dispatcher::set_device_policy(const std::string& fqname,
                                  DevicePolicy policy,
                                  std::uint64_t dispatch_arg_mask,
                                  std::span<const DeviceConstraint> constraints,
                                  std::uint64_t allow_undefined_mask) {
  mutate_and_publish_locked(
      fqname,
      "set_device_policy: unknown op: ",
      /*undefined_is_invalid_argument=*/true,
      [&](OperatorEntry& entry) {
        auto throw_bad = [&](const char* reason) {
          throw std::invalid_argument(
              "set_device_policy: " + fqname + ": " + reason);
        };

        if (entry.is_fabric_op) {
          throw_bad("fabric op");
        }

        const std::size_t in_arity = entry.schema.in_arity;

        // `in_arity > 64` contract (design/dispatcher/p2 §4.2).
        if (in_arity > kV2DevicePolicyMaxArity) {
          if (policy != DevicePolicy::AllSameDevice ||
              dispatch_arg_mask != 0 ||
              allow_undefined_mask != 0 ||
              !constraints.empty()) {
            throw_bad("in_arity > 64");
          }

          entry.device_policy = policy;
          entry.dispatch_arg_mask = 0;
          entry.allow_undefined_mask = 0;
          std::fill_n(entry.constraint_kind_by_index,
                      kV2DevicePolicyMaxArity,
                      ConstraintKind::MustMatchDispatchDeviceIfDefined);
          return;
        }

        std::uint64_t allowed_mask = 0;
        if (in_arity == 64) {
          allowed_mask = ~static_cast<std::uint64_t>(0);
        } else if (in_arity > 0) {
          allowed_mask = (static_cast<std::uint64_t>(1) << in_arity) - 1;
        }

        if ((dispatch_arg_mask & ~allowed_mask) != 0) {
          throw_bad("mask out of range");
        }
        if ((allow_undefined_mask & ~allowed_mask) != 0) {
          throw_bad("allow_undefined_mask out of range");
        }

        ConstraintKind table[kV2DevicePolicyMaxArity]{};
        std::uint64_t seen = 0;
        for (const auto& c : constraints) {
          const std::size_t idx = c.index;
          if (idx >= in_arity || idx >= kV2DevicePolicyMaxArity) {
            throw_bad("constraint index out of range");
          }

          const std::uint64_t bit = static_cast<std::uint64_t>(1) << idx;
          if (seen & bit) {
            throw_bad("duplicate constraint index");
          }
          seen |= bit;

          table[idx] = c.kind;
        }

        // `MustMatchDispatchDeviceIfDefined` does not use allow_undefined_mask and
        // rejects allow_undefined_mask bits for indices with the default kind.
        std::uint64_t allow_undefined_disallowed = 0;
        for (std::size_t i = 0; i < in_arity; ++i) {
          if (table[i] == ConstraintKind::MustMatchDispatchDeviceIfDefined) {
            allow_undefined_disallowed |= static_cast<std::uint64_t>(1) << i;
          }
        }
        if ((allow_undefined_mask & allow_undefined_disallowed) != 0) {
          throw_bad("allow_undefined_mask out of range");
        }

        entry.device_policy = policy;
        entry.dispatch_arg_mask = dispatch_arg_mask;
        entry.allow_undefined_mask = allow_undefined_mask;
        std::copy_n(table, kV2DevicePolicyMaxArity, entry.constraint_kind_by_index);
      });
}


void Dispatcher::callBoxed(const OperatorHandle& op, BoxedStack& s) {
  auto& entry = op.get();
  const OpDispatchStateV2* st =
      entry.state_v2.load(std::memory_order_acquire);
  if (!st) {
#ifndef NDEBUG
    assert(st != nullptr);
#endif
    throw std::logic_error("internal error: state_v2 is null");
  }
  callBoxed_v2(entry, *st, s, tls_redispatch_op_);
}

void Dispatcher::callBoxed(const std::string& name, BoxedStack& s) {
  callBoxed(find(name), s);
}

void Dispatcher::redispatchBoxed(const OperatorHandle& op, BoxedStack& s) {
  auto& entry = op.get();
  const OpDispatchStateV2* st =
      entry.state_v2.load(std::memory_order_acquire);
  if (!st) {
#ifndef NDEBUG
    assert(st != nullptr);
#endif
    throw std::logic_error("internal error: state_v2 is null");
  }

  RedispatchGuard g(&entry);
  callBoxed_v2_base(*st, s, DeviceCheckMode::OpAllowsMixed);
}

void Dispatcher::redispatchBoxed(const std::string& name, BoxedStack& s) {
  redispatchBoxed(find(name), s);
}

void Dispatcher::redispatchBoxedCurrent(BoxedStack& s) {
  auto* entry = tls_current_op_;
  if (!entry) {
    throw std::runtime_error(
        "_redispatch_boxed_current() must be called from within a Python override");
  }

  const OpDispatchStateV2* st =
      entry->state_v2.load(std::memory_order_acquire);
  if (!st) {
#ifndef NDEBUG
    assert(st != nullptr);
#endif
    throw std::logic_error("internal error: state_v2 is null");
  }

  RedispatchGuard g(entry);
  callBoxed_v2_base(*st, s, DeviceCheckMode::StrictSameDevice);
}

} // namespace dispatch
} // namespace vbt
