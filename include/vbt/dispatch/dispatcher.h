// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "vbt/plugin/vbt_plugin.h"

#include "vbt/dispatch/dispatch_key.h"
#include "vbt/dispatch/dispatch_key_set.h"
#include "vbt/dispatch/kernel_function.h"

namespace vbt {
namespace dispatch {

// Dispatcher v2.
// - Build flag: VBT_WITH_DISPATCH_V2 (mandatory; dispatcher v1 retired).
// - Runtime selection flag removed: v2 is always used.
// - Tests may still construct DispatchV2ModeGuard(true) as a clarity marker.

#if VBT_INTERNAL_TESTS
class DispatchV2ModeGuard {
 public:
  explicit DispatchV2ModeGuard(bool enabled);
  ~DispatchV2ModeGuard();
  DispatchV2ModeGuard(const DispatchV2ModeGuard&) = delete;
  DispatchV2ModeGuard& operator=(const DispatchV2ModeGuard&) = delete;

 private:
  int prev_;
};
#endif  // VBT_INTERNAL_TESTS

// Env: VBT_DISPATCH_V2_FABRIC_NO_CUDA_CALLS (cached; default ON).
// Only affects the Fabric bypass helper.
bool dispatch_v2_fabric_no_cuda_calls();

#if VBT_INTERNAL_TESTS
class DispatchV2FabricNoCudaCallsGuard {
 public:
  explicit DispatchV2FabricNoCudaCallsGuard(bool enabled);
  ~DispatchV2FabricNoCudaCallsGuard();
  DispatchV2FabricNoCudaCallsGuard(const DispatchV2FabricNoCudaCallsGuard&) = delete;
  DispatchV2FabricNoCudaCallsGuard& operator=(const DispatchV2FabricNoCudaCallsGuard&) = delete;

 private:
  int prev_;
};
#endif  // VBT_INTERNAL_TESTS

//
constexpr std::size_t kV2DevicePolicyMaxArity = 64;

enum class DevicePolicy : std::uint8_t {
  AllSameDevice = 0,
  MaskedSameDevice = 1,
  Fabric5Arg = 2,
};

enum class ConstraintKind : std::uint8_t {
  MustMatchDispatchDeviceIfDefined = 0,
  MustBeCPUScalarInt64_0d = 1,
  MustBeCPUScalarBool_0d = 2,
  DeferToKernel = 3,
};

static_assert(static_cast<std::uint8_t>(DevicePolicy::AllSameDevice) == 0,
              "DevicePolicy::AllSameDevice must remain 0 for default-init");
static_assert(static_cast<std::uint8_t>(ConstraintKind::MustMatchDispatchDeviceIfDefined) == 0,
              "ConstraintKind::MustMatchDispatchDeviceIfDefined must remain 0 for default-init");

struct DeviceConstraint {
  std::uint8_t index{0};
  ConstraintKind kind{ConstraintKind::MustMatchDispatchDeviceIfDefined};
};

struct OpSchema {
  std::string fqname;
  uint8_t in_arity{0};             // number of Tensor inputs
  bool has_non_tensor_args{false}; // schema contains any non-Tensor args
};

// NOTE: Internal implementation detail; not ABI-stable.
struct OpDispatchStateV2 final {
  // Identity / schema (copied from v1; stable after def)
  std::string fqname;
  std::uint8_t in_arity{0};

  DevicePolicy device_policy{DevicePolicy::AllSameDevice};
  std::uint64_t dispatch_arg_mask{0};
  std::uint64_t allow_undefined_mask{0};
  ConstraintKind constraint_kind_by_index[kV2DevicePolicyMaxArity]{};

  bool allow_multi_device_fabric{false};

  // Wrapper presence (subset of {Autograd, Python}).
  DispatchKeySet present_wrappers;

  // Wrapper call targets.
  KernelFunction autograd_fallback;             // valid iff present_wrappers has Autograd
  KernelFunction::BoxedFn boxed_override{nullptr}; // valid iff present_wrappers has Python

  // Base kernels (copied from v1 optionals)
  bool has_cpu{false};
  bool has_cuda{false};
  KernelFunction cpu_base;   // valid iff has_cpu
  KernelFunction cuda_base;  // valid iff has_cuda

  // Debug
  std::uint64_t version{0};
};


struct OperatorEntry {
  OpSchema schema;
  std::optional<KernelFunction> cpu_base;
  std::optional<KernelFunction> cuda_base;
  KernelFunction::BoxedFn boxed_override{nullptr};
  // Registration normalizes arity to schema.in_arity and ctx to &entry.
  KernelFunction autograd_fallback;

  // Informational: true only for allowlisted Fabric ops.
  bool is_fabric_op{false};
  // When true, Dispatcher may bypass the generic same-device invariant and
  // directly dispatch to a CUDA base kernel (with additional validation).
  bool allow_multi_device_fabric{false};

  DevicePolicy device_policy{DevicePolicy::AllSameDevice};
  std::uint64_t dispatch_arg_mask{0};
  std::uint64_t allow_undefined_mask{0};
  ConstraintKind constraint_kind_by_index[kV2DevicePolicyMaxArity]{};

  // states (no reclamation yet).
  std::atomic<const OpDispatchStateV2*> state_v2{nullptr};
};

class OperatorHandle {
 public:
  OperatorHandle() = default;
  explicit OperatorHandle(OperatorEntry* e) : entry_(e) {}
  OperatorEntry& get() const { if (!entry_) throw std::runtime_error("invalid handle"); return *entry_; }
  bool valid() const { return entry_ != nullptr; }
 private:
  OperatorEntry* entry_{nullptr};
};

// Internal-only (not ABI-stable).
struct PluginCommitPlan final {
  struct Def final {
    std::string fqname;
    std::string def_string;
  };

  struct Policy final {
    std::string fqname;
    DevicePolicy policy{DevicePolicy::AllSameDevice};
    std::uint64_t dispatch_arg_mask{0};
    std::uint64_t allow_undefined_mask{0};
    ConstraintKind constraint_kind_by_index[kV2DevicePolicyMaxArity]{};
  };

  struct Kernel final {
    std::string fqname;
    DispatchKey key{DispatchKey::CPU};  // must be CPU or CUDA
    KernelFunction kf;
  };

  std::vector<std::string> libraries;
  std::vector<Def> defs;
  std::vector<Policy> policies;
  std::vector<Kernel> kernels;
};

// Registration/mutation APIs are synchronized with a coarse mutex; call paths are lock-free.
// Registration-before-use is required; concurrent registration with in-flight calls may race in V1.
// This remains a minimal design; no RCU snapshotting or unload semantics.
class Dispatcher {
 public:
  static Dispatcher& instance();
  void registerLibrary(const std::string& ns);
  OperatorHandle def(const std::string& def_string);
  template<class Fn>
  void registerCpuKernel(const std::string& fqname, Fn* fn) {
    using FnPtr = Fn*;
    constexpr int fn_arity =
      std::is_same_v<FnPtr, KernelFunction::Unboxed0> ? 0 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed1> ? 1 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed2> ? 2 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed3> ? 3 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed4> ? 4 :
      -1;
    static_assert(fn_arity != -1,
                  "registerCpuKernel only supports TensorImpl(*)(const TensorImpl&...) signatures (0-4 Tensor args)");

    if constexpr (fn_arity == 0) registerCpuKernelFunction(fqname, KernelFunction::makeUnboxed0(fn));
    else if constexpr (fn_arity == 1) registerCpuKernelFunction(fqname, KernelFunction::makeUnboxed1(fn));
    else if constexpr (fn_arity == 2) registerCpuKernelFunction(fqname, KernelFunction::makeUnboxed2(fn));
    else if constexpr (fn_arity == 3) registerCpuKernelFunction(fqname, KernelFunction::makeUnboxed3(fn));
    else if constexpr (fn_arity == 4) registerCpuKernelFunction(fqname, KernelFunction::makeUnboxed4(fn));
  }
  template<class Fn>
  void registerCudaKernel(const std::string& fqname, Fn* fn) {
    using FnPtr = Fn*;
    constexpr int fn_arity =
      std::is_same_v<FnPtr, KernelFunction::Unboxed0> ? 0 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed1> ? 1 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed2> ? 2 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed3> ? 3 :
      std::is_same_v<FnPtr, KernelFunction::Unboxed4> ? 4 :
      -1;
    static_assert(fn_arity != -1,
                  "registerCudaKernel only supports TensorImpl(*)(const TensorImpl&...) signatures (0-4 Tensor args)");

    if constexpr (fn_arity == 0) registerCudaKernelFunction(fqname, KernelFunction::makeUnboxed0(fn));
    else if constexpr (fn_arity == 1) registerCudaKernelFunction(fqname, KernelFunction::makeUnboxed1(fn));
    else if constexpr (fn_arity == 2) registerCudaKernelFunction(fqname, KernelFunction::makeUnboxed2(fn));
    else if constexpr (fn_arity == 3) registerCudaKernelFunction(fqname, KernelFunction::makeUnboxed3(fn));
    else if constexpr (fn_arity == 4) registerCudaKernelFunction(fqname, KernelFunction::makeUnboxed4(fn));
  }
  // Register a boxed override for an existing op
  void registerBoxedOverride(const std::string& fqname, KernelFunction::BoxedFn fn);
  // Idempotent registration; returns false when duplicate
  bool tryRegisterBoxedOverride(const std::string& fqname, KernelFunction::BoxedFn fn);

  void registerCpuKernelFunction(const std::string& fqname, const KernelFunction& kf);
  void registerCudaKernelFunction(const std::string& fqname, const KernelFunction& kf);
  std::optional<KernelFunction> replaceCpuKernelFunction(const std::string& fqname, const KernelFunction& kf);
  std::optional<KernelFunction> replaceCudaKernelFunction(const std::string& fqname, const KernelFunction& kf);
  // Uninstall helpers used by loader rollback to clear base kernels
  std::optional<KernelFunction> uninstallCpuKernelFunction(const std::string& fqname);
  std::optional<KernelFunction> uninstallCudaKernelFunction(const std::string& fqname);

  void registerAutogradFallback(const std::string& fqname, const KernelFunction& kf);
  bool tryRegisterAutogradFallback(const std::string& fqname, const KernelFunction& kf);

  void mark_fabric_op(const std::string& fqname,
                      bool is_fabric_op,
                      bool allow_multi_device_fabric);

  void set_device_policy(const std::string& fqname,
                         DevicePolicy policy,
                         std::uint64_t dispatch_arg_mask,
                         std::span<const DeviceConstraint> constraints,
                         std::uint64_t allow_undefined_mask);

  // for a plugin load and publish updated state_v2 snapshots last.
  vt_status apply_plugin_commit_plan(const PluginCommitPlan& plan,
                                     std::string* out_err) noexcept;

  bool has(const std::string& name) const {
    std::lock_guard<std::mutex> lg(mu_);
    return registry_.count(name) != 0;
  }
  OperatorHandle find(const std::string& name);
  void callBoxed(const OperatorHandle& op, BoxedStack& s);
  void callBoxed(const std::string& name, BoxedStack& s);
  void redispatchBoxed(const OperatorHandle& op, BoxedStack& s);
  void redispatchBoxed(const std::string& name, BoxedStack& s);
  void redispatchBoxedCurrent(BoxedStack& s);

 private:
  std::unordered_map<std::string, std::unique_ptr<OperatorEntry>> registry_{};
  static thread_local const OperatorEntry* tls_redispatch_op_;

 public:
  // Exposed for Python override trampolines to locate the current op
  static thread_local const OperatorEntry* tls_current_op_;
  static thread_local bool tls_skip_autograd_;
  struct RedispatchGuard {
    const OperatorEntry* prev;
    explicit RedispatchGuard(const OperatorEntry* op) : prev(Dispatcher::tls_redispatch_op_) { Dispatcher::tls_redispatch_op_ = op; }
    ~RedispatchGuard() { Dispatcher::tls_redispatch_op_ = prev; }
  };
  struct CurrentOpGuard {
    const OperatorEntry* prev;
    explicit CurrentOpGuard(const OperatorEntry* op) : prev(Dispatcher::tls_current_op_) { Dispatcher::tls_current_op_ = op; }
    ~CurrentOpGuard() { Dispatcher::tls_current_op_ = prev; }
  };
  // Mutation guard
  mutable std::mutex mu_;

 private:
  template<class Fn>
  decltype(auto) mutate_and_publish_locked(const std::string& fqname,
                                          const char* undefined_msg_prefix,
                                          bool undefined_is_invalid_argument,
                                          Fn&& fn);
};

} // namespace dispatch
} // namespace vbt
