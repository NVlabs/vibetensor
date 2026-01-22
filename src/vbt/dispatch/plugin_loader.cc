// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/dispatch/plugin_loader.h"

#include <dlfcn.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <new>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cerrno>
#include <limits>
#include <limits.h>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <optional>
#include <atomic>
#include <cctype>
#ifdef __linux__
#include <sys/syscall.h>
#ifdef SYS_openat2
#include <linux/openat2.h>
#endif
#endif

#include "vbt/dispatch/dispatcher.h"
#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"

// Opaque vt_tensor backing struct (host-side definition).
struct vt_tensor__ {
  vbt::core::TensorImpl* impl;  // non-owning for borrowed, owning for allocated
  bool owned;                   // true if allocated via host allocator
};

namespace vbt {
namespace dispatch {
namespace plugin_helpers {

vt_status vt_tensor_iter_unary_cpu_host(const vt_iter_config* cfg,
                                        vt_tensor out_h,
                                        vt_tensor in_h,
                                        vt_tensor_iter_loop1d_fn loop,
                                        void* user_ctx) noexcept;

vt_status vt_tensor_iter_binary_cpu_host(const vt_iter_config* cfg,
                                         vt_tensor out_h,
                                         vt_tensor a_h,
                                         vt_tensor b_h,
                                         vt_tensor_iter_loop1d_fn loop,
                                         void* user_ctx) noexcept;

vt_status vt_tensor_iter_build_elementwise_host(const vt_iter_config* cfg,
                                                int32_t ntensors,
                                                const vt_tensor* tensors,
                                                vt_tensor_iter* out_iter) noexcept;

vt_status vt_tensor_iter_build_reduction_host(const vt_iter_config* cfg,
                                              int32_t ntensors,
                                              const vt_tensor* tensors,
                                              int32_t reduce_dim,
                                              vt_tensor_iter* out_iter) noexcept;

vt_status vt_tensor_iter_get_kind_host(vt_tensor_iter iter,
                                       vt_tensor_iter_kind* out_kind) noexcept;

vt_status vt_tensor_iter_export_desc_host(vt_tensor_iter iter,
                                          vt_tensor_iter_desc* out_desc) noexcept;

vt_status vt_tensor_iter_export_alias_info_host(vt_tensor_iter iter,
                                                vt_tensor_iter_alias_info* out_alias) noexcept;

vt_status vt_tensor_iter_export_cuda_desc_host(vt_tensor_iter iter,
                                               int32_t operand_index,
                                               int32_t max_ndim,
                                               vt_tensor_iter_cuda_desc* out_desc) noexcept;

vt_status vt_tensor_iter_for_each_cpu_host(vt_tensor_iter iter,
                                           vt_tensor_iter_loop1d_fn loop,
                                           void* ctx) noexcept;

void vt_tensor_iter_destroy_host(vt_tensor_iter iter) noexcept;
}  // namespace plugin_helpers
}  // namespace dispatch
}  // namespace vbt

namespace vbt {
namespace dispatch {
namespace plugin {

namespace {

// Thread-local error string
static thread_local std::string tl_err;

// kernel typedefs
using kernel2_t = vt_status(*)(vt_stream, vt_tensor, vt_tensor, vt_tensor*);

// Track allocations during one plugin call to free on failure (optional safety)
static thread_local struct AllocTracker* tls_tracker = nullptr;

struct AllocTracker {
  std::vector<vt_tensor> owned;
  AllocTracker() : prev(tls_tracker) { tls_tracker = this; }
  ~AllocTracker() {
    tls_tracker = prev;
    for (auto h : owned) {
      if (h) { delete h->impl; delete h; }
    }
  }
  void disarm(vt_tensor h) {
    auto it = std::find(owned.begin(), owned.end(), h);
    if (it != owned.end()) { *it = nullptr; }
  }
private:
  AllocTracker* prev;
};

// Helpers to convert between TensorImpl and vt_tensor borrowed handle
static inline vt_tensor make_borrowed_handle(const vbt::core::TensorImpl& t) {
  vt_tensor h = new vt_tensor__{const_cast<vbt::core::TensorImpl*>(&t), false};
  return h;
}

static inline vbt::core::TensorImpl adopt_owned_and_free(vt_tensor h) {
  if (!h || !h->owned || h->impl == nullptr) {
    throw std::runtime_error("plugin bridge: invalid output handle");
  }
  // Move out TensorImpl value and free wrapper
  vbt::core::TensorImpl out = std::move(*h->impl);
  delete h->impl;
  delete h;
  return out;
}

// Plugin handle & scaffolding
struct PluginKernelCtx;
struct PluginHandle {
  void* dl_handle{nullptr};
  std::string path;
  uint16_t abi_major{0}, abi_minor{0};
  std::atomic<uint64_t> inflight{0};
  std::atomic<bool> unloading{false};
  std::optional<vbt::dispatch::KernelFunction> prev_cpu_add_base{};
  struct AddCpuCtx { PluginHandle* handle; kernel2_t fn; const char* fqname; };
  std::unique_ptr<AddCpuCtx> add_cpu_ctx{};
  // CUDA bridge
  std::optional<vbt::dispatch::KernelFunction> prev_cuda_add_base{};
  struct AddCudaCtx { PluginHandle* handle; kernel2_t fn; const char* fqname; };
  std::unique_ptr<AddCudaCtx> add_cuda_ctx{};
  // Generalized per-fqname tracking for M_EXT2.2
  struct InstalledPrev { std::optional<vbt::dispatch::KernelFunction> cpu_prev; std::optional<vbt::dispatch::KernelFunction> cuda_prev; };
  std::unordered_map<std::string, InstalledPrev> prev_by_fqname;
  std::unordered_map<std::string, std::unique_ptr<PluginKernelCtx>> cpu_ctx_by_fqname;
  std::unordered_map<std::string, std::unique_ptr<PluginKernelCtx>> cuda_ctx_by_fqname;

  // Ops defined by this plugin during vbt_plugin_init (used to scope set_device_policy).
  std::unordered_set<std::string> defined_ops;
};

//
// In atomic mode, registration APIs stage into a loader-local transaction during
// vbt_plugin_init. After init returns VT_STATUS_OK, the loader commits staged
// defs + kernels to the Dispatcher.
struct ParsedDef {
  std::string fqname;
  uint8_t in_arity{0};
  bool has_non_tensor_args{false};
  std::string def_string;
};

enum class StagedKernelKind { Boxed, Arity2 };

struct StagedKernel {
  std::string fqname;
  vt_dispatch_key key{kDLCPU};
  StagedKernelKind kind{StagedKernelKind::Arity2};
  vt_kernel_boxed_fn boxed{nullptr};
  kernel2_t k2{nullptr};
};

struct StagedPolicy {
  std::string fqname;
  vbt::dispatch::DevicePolicy policy{vbt::dispatch::DevicePolicy::AllSameDevice};
  uint64_t dispatch_arg_mask{0};
  uint64_t allow_undefined_mask{0};
  vbt::dispatch::ConstraintKind constraint_kind_by_index[vbt::dispatch::kV2DevicePolicyMaxArity]{};
};

struct PluginInitTxn {
  const void* txn_id{nullptr};
  std::vector<std::string> libraries;
  std::unordered_map<std::string, ParsedDef> defs_by_fqname;
  std::vector<StagedKernel> kernels;
  std::unordered_map<std::string, StagedPolicy> policy_by_fqname;

  // Names reserved globally (fqname -> txn_id) to prevent concurrent staging of
  // the same new op.
  std::vector<std::string> reserved_fqnames;
};

static thread_local PluginInitTxn* tls_plugin_txn = nullptr; // only set during vbt_plugin_init in atomic mode

struct ActiveCallsGuard {
  std::atomic<uint64_t>* ctr;
  explicit ActiveCallsGuard(std::atomic<uint64_t>& c) : ctr(&c) { ctr->fetch_add(1, std::memory_order_acq_rel); }
  ~ActiveCallsGuard() { ctr->fetch_sub(1, std::memory_order_acq_rel); }
};

// Persistent registry to keep plugin handles alive for the process lifetime
static std::mutex g_loader_mu;
static std::unordered_map<std::string, std::shared_ptr<PluginHandle>> g_loaded_handles;
static std::unordered_set<std::string> g_loaded_devino;

// In legacy (non-atomic) mode, on plugin init failure we keep the dlopen handle
// and ctx objects alive to avoid stale dispatch snapshots or trampolines calling
// into unmapped code (no dlclose mitigation).
//
// In atomic commit mode (VBT_PLUGIN_ATOMIC_COMMIT), init failures stage
// registration without publishing dispatch-visible state, so dlclose on failure
static std::unordered_map<std::string, std::shared_ptr<PluginHandle>> g_failed_init_handles;
static std::unordered_set<std::string> g_failed_init_devino;

// Guarded by g_loader_mu.
static std::unordered_map<std::string, const void*> g_reserved_fqnames;

// Single host API struct whose address is stable for the process lifetime.
static vbt_host_api g_host_api{};
static std::once_flag g_host_api_init_once;

static thread_local PluginHandle* g_current_handle_for_registration = nullptr; // only valid during vbt_plugin_init

struct RegistrationTLSGuard {
  PluginInitTxn* prev_txn{nullptr};
  PluginHandle* prev_handle{nullptr};

  explicit RegistrationTLSGuard(PluginHandle* handle, PluginInitTxn* txn) noexcept
      : prev_txn(tls_plugin_txn), prev_handle(g_current_handle_for_registration) {
    tls_plugin_txn = txn;
    g_current_handle_for_registration = handle;
  }

  RegistrationTLSGuard(const RegistrationTLSGuard&) = delete;
  RegistrationTLSGuard& operator=(const RegistrationTLSGuard&) = delete;

  ~RegistrationTLSGuard() noexcept {
    g_current_handle_for_registration = prev_handle;
    tls_plugin_txn = prev_txn;
  }
};

// Error helpers
static inline void set_loader_error(const std::string& msg) { tl_err = std::string("loader: ") + msg; }

// Map vt_status to C++ exceptions
static inline void throw_from_status(const char* fqname, vt_status st, const std::string& detail) {
  switch (st) {
    case VT_STATUS_OK:
      return;
    case VT_STATUS_INVALID_ARG:
      throw std::invalid_argument(detail.empty() ? std::string(fqname) + ": invalid argument" : detail);
    case VT_STATUS_UNSUPPORTED:
      throw std::runtime_error(std::string("unsupported: ") + fqname);
    case VT_STATUS_NOT_FOUND:
      throw std::runtime_error(detail.empty() ? std::string("plugin kernel failed: ") + fqname
                                              : std::string("plugin kernel failed: ") + fqname + ": " + detail);
    case VT_STATUS_INTERNAL:
      throw std::runtime_error("plugin internal error");
    case VT_STATUS_ABI_MISMATCH:
      throw std::runtime_error("plugin ABI mismatch");
    case VT_STATUS_NOMEM:
      throw std::runtime_error("out of memory");
    case VT_STATUS_RUNTIME_ERROR:
      throw std::runtime_error(detail.empty() ? std::string("plugin runtime error: ") + fqname : detail);
    default:
      throw std::runtime_error("plugin runtime error");
  }
}

// Env helpers
static bool IsTruthyEnv(const char* v) {
  if (!v || *v == '\0') return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "1" || s == "true" || s == "yes";
}

static bool plugin_atomic_commit_requested() {
  static std::once_flag once;
  static bool cached = false;
  std::call_once(once, [] {
    cached = IsTruthyEnv(std::getenv("VBT_PLUGIN_ATOMIC_COMMIT"));
  });
  return cached;
}

struct ParsedSignature {
  uint8_t in_arity{0};
  bool has_non_tensor_args{false};
};

static ParsedSignature parse_signature_for_staging(const std::string& def) {
  auto l = def.find('(');
  auto r = def.find(')');
  if (l == std::string::npos || r == std::string::npos || r < l) {
    throw std::runtime_error("malformed def: missing '()'");
  }
  auto arrow = def.find("->");
  if (arrow == std::string::npos) {
    throw std::runtime_error("malformed def: missing '->'");
  }
  auto ret = def.substr(arrow + 2);
  if (ret.find("Tensor") == std::string::npos) {
    throw std::runtime_error("malformed def: return must be Tensor");
  }

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

static void validate_special_schema_for_staging(const ParsedDef& def) {
  if (def.fqname == "vt::check_stream") {
    if (def.has_non_tensor_args || def.in_arity != 2) {
      throw std::runtime_error(
          "vt::check_stream must have schema (Tensor, Tensor) -> Tensor");
    }
  } else if (def.fqname == "vt::index") {
    if (def.has_non_tensor_args || def.in_arity != 3) {
      throw std::runtime_error(
          "vt::index must have schema (Tensor, Tensor, Tensor) -> Tensor");
    }
  } else if (def.fqname == "vt::index_put") {
    if (def.has_non_tensor_args || def.in_arity != 5) {
      throw std::runtime_error(
          "vt::index_put must have schema (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    }
  }
}

static ParsedDef parse_def_for_staging(const char* def_str) {
  // Mirror Dispatcher::def parsing + signature checks.
  std::string s(def_str);
  auto p = s.find('(');
  if (p == std::string::npos) {
    throw std::runtime_error("malformed def: missing '->'");
  }

  ParsedDef out;
  out.fqname = s.substr(0, p);
  out.def_string = s;

  auto sig = parse_signature_for_staging(s);
  out.in_arity = sig.in_arity;
  out.has_non_tensor_args = sig.has_non_tensor_args;

  validate_special_schema_for_staging(out);
  return out;
}

static void release_fqname_reservations(PluginInitTxn* txn) noexcept {
  if (!txn || txn->txn_id == nullptr || txn->reserved_fqnames.empty()) return;
  try {
    std::lock_guard<std::mutex> lg(g_loader_mu);
    for (const auto& fq : txn->reserved_fqnames) {
      auto it = g_reserved_fqnames.find(fq);
      if (it != g_reserved_fqnames.end() && it->second == txn->txn_id) {
        g_reserved_fqnames.erase(it);
      }
    }
    txn->reserved_fqnames.clear();
  } catch (...) {
    // Best-effort; leak reservations on OOM/exception.
  }
}

// Secure open helpers
static int secure_open_readonly(const char* path) {
  // Validate input
  if (!path || *path == '\0') { set_loader_error("null path"); errno = EINVAL; return -1; }

#ifdef __linux__
#ifdef SYS_openat2
  // Try openat2 first with strict resolution
  {
    struct stat st_parent{};
    // quick parent world-writable check using string path (best-effort)
    std::string p(path);
    auto pos = p.find_last_of('/');
    if (pos != std::string::npos) {
      std::string parent = p.substr(0, pos == 0 ? 1 : pos);
      if (::stat(parent.c_str(), &st_parent) == 0) {
        if ((st_parent.st_mode & S_IWOTH) && !(st_parent.st_mode & S_ISVTX)) {
          const char* allow = std::getenv("VBT_ALLOW_WORLD_WRITABLE");
          if (!allow || std::string(allow) != "1") {
            set_loader_error("world-writable parent disallowed");
            errno = EPERM;
            return -1;
          }
        }
      }
    }

    struct open_how how{};
    how.flags = O_PATH | O_CLOEXEC;
    how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS | RESOLVE_NO_XDEV;
    int fd = static_cast<int>(syscall(SYS_openat2, AT_FDCWD, path, &how, sizeof(how)));
    if (fd >= 0) {
      struct stat st{};
      if (fstat(fd, &st) != 0) { int e = errno; close(fd); errno = e; set_loader_error("not a regular file"); return -1; }
      if (!S_ISREG(st.st_mode)) { close(fd); set_loader_error("not a regular file"); errno = EINVAL; return -1; }
      return fd;
    } else {
      // If unsupported or other failure, fall back to secure-walk. Only hard-fail on explicit symlink loops.
      if (errno == ELOOP) { set_loader_error("symlink in path rejected"); return -1; }
      // fallthrough: secure-walk below
    }
  }
#endif
#endif
  // Fallback: secure-walk path components without following symlinks
  int dfd = AT_FDCWD;
  int leaf_fd = -1;
  std::vector<int> dirfds_to_close;
  auto bail = [&](const char* msg){ if (msg) set_loader_error(msg); for (int fd : dirfds_to_close) if (fd >= 0 && fd != AT_FDCWD) close(fd); if (leaf_fd >= 0) close(leaf_fd); return -1; };
  // Split path into components
  std::string p(path);
  size_t start = 0;
  bool absolute = !p.empty() && p[0] == '/';
  if (absolute) { start = 1; }
  size_t next = start;
  dev_t root_dev = 0;
  bool root_dev_set = false;
  std::string comp;
  // derive parent dir for world-writable check later
  std::vector<std::string> components;
  while (next <= p.size()) {
    size_t slash = p.find('/', next);
    if (slash == std::string::npos) slash = p.size();
    comp = p.substr(next, slash - next);
    if (!comp.empty() && comp != ".") {
      if (comp == "..") { errno = EPERM; return bail("symlink in path rejected"); }
      components.push_back(comp);
    }
    if (slash == p.size()) break;
    next = slash + 1;
  }
  if (components.empty()) return bail("not a regular file");

  // Walk directories
  for (size_t i = 0; i + 1 < components.size(); ++i) {
    int nfd = openat(dfd, components[i].c_str(), O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (nfd < 0) {
      if (errno == ELOOP) return bail("symlink in path rejected");
      return bail(std::strerror(errno));
    }
    struct stat st{};
    if (fstat(nfd, &st) != 0) { int e = errno; close(nfd); errno = e; return bail(std::strerror(errno)); }
    if (!root_dev_set) { root_dev = st.st_dev; root_dev_set = true; }
    else {
      if (st.st_dev != root_dev) {
        const char* allow = std::getenv("VBT_LOADER_ALLOW_XDEV");
        if (!allow || std::string(allow) != "1") {
          close(nfd);
          return bail("cross-device traversal disallowed");
        }
      }
    }
    if (dfd != AT_FDCWD) dirfds_to_close.push_back(dfd);
    dfd = nfd;
  }
  // World-writable parent policy
  struct stat stp{};
  if (fstat(dfd, &stp) == 0) {
    if ((stp.st_mode & S_IWOTH) && !(stp.st_mode & S_ISVTX)) {
      const char* allow = std::getenv("VBT_ALLOW_WORLD_WRITABLE");
      if (!allow || std::string(allow) != "1") {
        return bail("world-writable parent disallowed");
      }
    }
  }
  // Open leaf
  const std::string& leaf = components.back();
  leaf_fd = openat(dfd, leaf.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (leaf_fd < 0) {
    if (errno == ELOOP) return bail("symlink in path rejected");
    return bail(std::strerror(errno));
  }
  struct stat st{};
  if (fstat(leaf_fd, &st) != 0) return bail(std::strerror(errno));
  if (!S_ISREG(st.st_mode)) return bail("not a regular file");
  // Close dirfds
  for (int fd : dirfds_to_close) if (fd >= 0 && fd != AT_FDCWD) close(fd);
  if (dfd != AT_FDCWD) close(dfd);
  return leaf_fd;
}

// Host API implementations
static vt_status host_register_library(const char* ns) {
  tl_err.clear();

  // In atomic commit mode, stage library registrations so plugin init performs no
  // dispatcher mutation.
  if (tls_plugin_txn) {
    try {
      std::string s = ns ? std::string(ns) : std::string();
      auto& libs = tls_plugin_txn->libraries;
      if (std::find(libs.begin(), libs.end(), s) == libs.end()) {
        libs.push_back(std::move(s));
      }
      return VT_STATUS_OK;
    } catch (const std::exception& e) {
      tl_err = e.what();
      return VT_STATUS_INVALID_ARG;
    }
  }

  try {
    vbt::dispatch::Dispatcher::instance().registerLibrary(
        ns ? std::string(ns) : std::string());
    return VT_STATUS_OK;
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INVALID_ARG;
  }
}

static vt_status host_def(const char* def_str) {
  tl_err.clear();

  if (!g_current_handle_for_registration) {
    tl_err = "def: init-only";
    return VT_STATUS_INVALID_ARG;
  }
  if (!def_str) {
    tl_err = "def: def_string is null";
    return VT_STATUS_INVALID_ARG;
  }
  if (*def_str == '\0') {
    tl_err = "def: def_string is empty";
    return VT_STATUS_INVALID_ARG;
  }

  try {
    if (tls_plugin_txn) {
      // Atomic mode: stage def parsing and fqname reservation.
      PluginInitTxn& txn = *tls_plugin_txn;
      ParsedDef parsed = parse_def_for_staging(def_str);
      std::string fqname = parsed.fqname;

      if (txn.defs_by_fqname.count(fqname) != 0) {
        tl_err = std::string("duplicate def: ") + fqname;
        return VT_STATUS_INVALID_ARG;
      }

      // Reserve fqname across concurrent plugin inits.
      bool inserted_reservation = false;
      {
        std::lock_guard<std::mutex> lg(g_loader_mu);
        auto it = g_reserved_fqnames.find(fqname);
        if (it != g_reserved_fqnames.end() && it->second != txn.txn_id) {
          tl_err = std::string("duplicate def: ") + fqname;
          return VT_STATUS_INVALID_ARG;
        }
        if (it == g_reserved_fqnames.end()) {
          g_reserved_fqnames.emplace(fqname, txn.txn_id);
          inserted_reservation = true;
        }
      }
      if (inserted_reservation) {
        try {
          txn.reserved_fqnames.push_back(fqname);
        } catch (const std::bad_alloc&) {
          // Undo reservation (best-effort) on tracking failure.
          std::lock_guard<std::mutex> lg(g_loader_mu);
          auto it = g_reserved_fqnames.find(fqname);
          if (it != g_reserved_fqnames.end() && it->second == txn.txn_id) {
            g_reserved_fqnames.erase(it);
          }
          tl_err = "out of memory";
          return VT_STATUS_NOMEM;
        } catch (const std::exception& e) {
          // Undo reservation (best-effort) on unexpected tracking failure.
          std::lock_guard<std::mutex> lg(g_loader_mu);
          auto it = g_reserved_fqnames.find(fqname);
          if (it != g_reserved_fqnames.end() && it->second == txn.txn_id) {
            g_reserved_fqnames.erase(it);
          }
          tl_err = e.what();
          return VT_STATUS_INTERNAL;
        }
      }

      // Duplicate check against the dispatcher registry (no loader lock).
      if (vbt::dispatch::Dispatcher::instance().has(fqname)) {
        // Undo reservation (best-effort) for this txn.
        {
          std::lock_guard<std::mutex> lg(g_loader_mu);
          auto it = g_reserved_fqnames.find(fqname);
          if (it != g_reserved_fqnames.end() && it->second == txn.txn_id) {
            g_reserved_fqnames.erase(it);
          }
        }
        tl_err = std::string("duplicate def: ") + fqname;
        return VT_STATUS_INVALID_ARG;
      }

      txn.defs_by_fqname.emplace(fqname, std::move(parsed));
      g_current_handle_for_registration->defined_ops.insert(fqname);
      tl_err.clear();
      return VT_STATUS_OK;
    }

    auto h = vbt::dispatch::Dispatcher::instance().def(std::string(def_str));
    if (h.valid()) {
      g_current_handle_for_registration->defined_ops.insert(h.get().schema.fqname);
    }
    tl_err.clear();
    return VT_STATUS_OK;
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INVALID_ARG;
  }
}

static vt_status host_set_device_policy(const char* fqname,
                                        vt_device_policy policy,
                                        uint64_t dispatch_arg_mask,
                                        const vt_device_constraint* constraints,
                                        size_t nconstraints,
                                        uint64_t allow_undefined_mask) {
  tl_err.clear();

  try {
    if (!g_current_handle_for_registration) {
    tl_err = "set_device_policy: init-only";
    return VT_STATUS_INVALID_ARG;
  }
  if (!fqname) {
    tl_err = "set_device_policy: fqname is null";
    return VT_STATUS_INVALID_ARG;
  }
  if (*fqname == '\0') {
    tl_err = "set_device_policy: fqname is empty";
    return VT_STATUS_INVALID_ARG;
  }

  if (tls_plugin_txn) {
    PluginInitTxn& txn = *tls_plugin_txn;
    // Atomic commit mode: plugins may only set policy for ops defined in the same
    // init transaction.
    if (txn.defs_by_fqname.count(fqname) == 0) {
      tl_err = "set_device_policy: non-owned op";
      return VT_STATUS_UNSUPPORTED;
    }
    if (txn.policy_by_fqname.count(fqname) != 0) {
      tl_err = std::string("set_device_policy: duplicate policy: ") + fqname;
      return VT_STATUS_INVALID_ARG;
    }
  } else {
    if (g_current_handle_for_registration->defined_ops.count(fqname) == 0) {
      tl_err = "set_device_policy: op not owned";
      return VT_STATUS_INVALID_ARG;
    }
  }

  if ((constraints == nullptr) != (nconstraints == 0)) {
    tl_err = "set_device_policy: constraints pointer/length mismatch";
    return VT_STATUS_INVALID_ARG;
  }

  if (nconstraints > vbt::dispatch::kV2DevicePolicyMaxArity) {
    tl_err = "set_device_policy: nconstraints > 64";
    return VT_STATUS_INVALID_ARG;
  }

  // Validate policy before casting.
  if (policy > VT_DEVICE_POLICY_FABRIC5ARG) {
    tl_err = "set_device_policy: unknown policy";
    return VT_STATUS_INVALID_ARG;
  }
  // Fabric5Arg is core-only; plugins must not claim it.
  if (policy == VT_DEVICE_POLICY_FABRIC5ARG) {
    tl_err = "set_device_policy: Fabric5Arg is core-only";
    return VT_STATUS_UNSUPPORTED;
  }

  // Look up schema arity and fabric flag.
  auto& D = vbt::dispatch::Dispatcher::instance();
  uint8_t in_arity = 0;
  bool is_fabric_op = false;
  try {
    if (tls_plugin_txn) {
      auto it = tls_plugin_txn->defs_by_fqname.find(fqname);
      if (it == tls_plugin_txn->defs_by_fqname.end()) {
        tl_err = "set_device_policy: non-owned op";
        return VT_STATUS_UNSUPPORTED;
      }
      in_arity = it->second.in_arity;
      is_fabric_op = false;
    } else {
      auto h = D.find(fqname);
      in_arity = h.get().schema.in_arity;
      is_fabric_op = h.get().is_fabric_op;
    }
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INVALID_ARG;
  }

  if (is_fabric_op) {
    tl_err = "set_device_policy: fabric op";
    return VT_STATUS_UNSUPPORTED;
  }

  // `in_arity > 64` contract (design/dispatcher/p2 §4.2).
  if (in_arity > vbt::dispatch::kV2DevicePolicyMaxArity) {
    if (policy != VT_DEVICE_POLICY_ALL_SAME_DEVICE ||
        dispatch_arg_mask != 0 ||
        allow_undefined_mask != 0 ||
        nconstraints != 0) {
      tl_err = "set_device_policy: in_arity > 64";
      return VT_STATUS_INVALID_ARG;
    }

    if (tls_plugin_txn) {
      StagedPolicy sp;
      sp.fqname = fqname;
      sp.policy = vbt::dispatch::DevicePolicy::AllSameDevice;
      sp.dispatch_arg_mask = 0;
      sp.allow_undefined_mask = 0;
      tls_plugin_txn->policy_by_fqname.emplace(sp.fqname, std::move(sp));
      tl_err.clear();
      return VT_STATUS_OK;
    }

    // Only the default policy is supported at arity>64; let Dispatcher
    // redundantly validate and publish.
    try {
      D.set_device_policy(
          fqname,
          vbt::dispatch::DevicePolicy::AllSameDevice,
          /*dispatch_arg_mask=*/0,
          std::span<const vbt::dispatch::DeviceConstraint>{},
          /*allow_undefined_mask=*/0);
      tl_err.clear();
      return VT_STATUS_OK;
    } catch (const std::invalid_argument& e) {
      tl_err = e.what();
      return VT_STATUS_INVALID_ARG;
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      tl_err = e.what();
      return VT_STATUS_INTERNAL;
    }
  }

  uint64_t allowed_mask = 0;
  if (in_arity == 64) {
    allowed_mask = ~static_cast<uint64_t>(0);
  } else if (in_arity > 0) {
    allowed_mask = (static_cast<uint64_t>(1) << in_arity) - 1;
  }

  if ((dispatch_arg_mask & ~allowed_mask) != 0) {
    tl_err = "set_device_policy: mask out of range";
    return VT_STATUS_INVALID_ARG;
  }
  if ((allow_undefined_mask & ~allowed_mask) != 0) {
    tl_err = "set_device_policy: allow_undefined_mask out of range";
    return VT_STATUS_INVALID_ARG;
  }

  vbt::dispatch::ConstraintKind table[vbt::dispatch::kV2DevicePolicyMaxArity]{};
  uint64_t seen = 0;
  vbt::dispatch::DeviceConstraint cs[vbt::dispatch::kV2DevicePolicyMaxArity]{};
  size_t cs_size = 0;

  for (size_t i = 0; i < nconstraints; ++i) {
    const vt_device_constraint c = constraints[i];

    for (uint8_t r : c.reserved) {
      if (r != 0) {
        tl_err = "set_device_policy: constraint.reserved must be zero";
        return VT_STATUS_INVALID_ARG;
      }
    }

    if (c.kind > VT_CONSTRAINT_DEFER_TO_KERNEL) {
      tl_err = "set_device_policy: unknown constraint kind";
      return VT_STATUS_INVALID_ARG;
    }

    const uint8_t idx = c.index;
    if (idx >= in_arity || idx >= vbt::dispatch::kV2DevicePolicyMaxArity) {
      tl_err = "set_device_policy: constraint index out of range";
      return VT_STATUS_INVALID_ARG;
    }

    const uint64_t bit = static_cast<uint64_t>(1) << idx;
    if (seen & bit) {
      tl_err = "set_device_policy: duplicate constraint index";
      return VT_STATUS_INVALID_ARG;
    }
    seen |= bit;

    const auto kind = static_cast<vbt::dispatch::ConstraintKind>(c.kind);
    table[idx] = kind;
    cs[cs_size++] = vbt::dispatch::DeviceConstraint{idx, kind};
  }

  uint64_t disallowed = 0;
  for (size_t i = 0; i < in_arity; ++i) {
    if (table[i] == vbt::dispatch::ConstraintKind::MustMatchDispatchDeviceIfDefined) {
      disallowed |= static_cast<uint64_t>(1) << i;
    }
  }
  if ((allow_undefined_mask & disallowed) != 0) {
    tl_err = "set_device_policy: allow_undefined_mask out of range";
    return VT_STATUS_INVALID_ARG;
  }

  vbt::dispatch::DevicePolicy pol = vbt::dispatch::DevicePolicy::AllSameDevice;
  if (policy == VT_DEVICE_POLICY_MASKED_SAME_DEVICE) {
    pol = vbt::dispatch::DevicePolicy::MaskedSameDevice;
  }

  if (tls_plugin_txn) {
    StagedPolicy sp;
    sp.fqname = fqname;
    sp.policy = pol;
    sp.dispatch_arg_mask = dispatch_arg_mask;
    sp.allow_undefined_mask = allow_undefined_mask;
    std::copy_n(table,
                vbt::dispatch::kV2DevicePolicyMaxArity,
                sp.constraint_kind_by_index);
    tls_plugin_txn->policy_by_fqname.emplace(sp.fqname, std::move(sp));
    tl_err.clear();
    return VT_STATUS_OK;
  }

  try {
    D.set_device_policy(
        fqname,
        pol,
        dispatch_arg_mask,
        std::span<const vbt::dispatch::DeviceConstraint>(cs, cs_size),
        allow_undefined_mask);
    tl_err.clear();
    return VT_STATUS_OK;
  } catch (const std::invalid_argument& e) {
    tl_err = e.what();
    return VT_STATUS_INVALID_ARG;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INTERNAL;
  }
  } catch (const std::bad_alloc&) {
    try {
      tl_err = "set_device_policy: OOM";
    } catch (...) {
      // drop
    }
    return VT_STATUS_NOMEM;
  } catch (const std::exception& e) {
    try {
      tl_err = e.what();
    } catch (...) {
      // drop
    }
    return VT_STATUS_INTERNAL;
  } catch (...) {
    try {
      tl_err = "set_device_policy: internal error";
    } catch (...) {
      // drop
    }
    return VT_STATUS_INTERNAL;
  }
}

// Generalized plugin trampolines (M_EXT2.2)
struct PluginKernelCtx {
  PluginHandle* handle;
  std::string fqname;

  // Stable for process lifetime (OperatorEntry is not reclaimed).
  const vbt::dispatch::OperatorEntry* entry{nullptr};

  uint8_t arity;

  // Snapshot at registration time (debug only). Runtime behavior uses `entry`
  // state, not this bool.
  bool allow_multi_device_fabric_snapshot{false};

  vt_kernel_boxed_fn boxed; // optional
  kernel2_t k2;             // optional (arity must be 2)
};

static void plugin_cpu_trampoline(void* vctx, vbt::dispatch::BoxedStack& s) {
  auto* ctx = reinterpret_cast<PluginKernelCtx*>(vctx);
  if (!ctx || !ctx->handle) throw std::runtime_error("invalid kernel ctx");
  ActiveCallsGuard acg(ctx->handle->inflight);
  AllocTracker track;
  tl_err.clear();
  vt_tensor tout = nullptr;
  const size_t nargs = s.size();
  std::vector<vt_tensor> handles; handles.reserve(nargs);
  for (size_t i = 0; i < nargs; ++i) handles.push_back(make_borrowed_handle(s[i]));
  vt_status st = VT_STATUS_INTERNAL;
  if (ctx->boxed) {
    st = ctx->boxed(/*s*/0, handles.data(), nargs, &tout);
  } else if (ctx->k2) {
    if (nargs != 2) {
      for (auto h : handles) delete h; // cleanup borrowed wrappers
      throw std::invalid_argument(std::string("arity mismatch ") + ctx->fqname + ": expected 2, got " + std::to_string(nargs));
    }
    st = ctx->k2(/*s*/0, handles[0], handles[1], &tout);
  } else {
    for (auto h : handles) delete h;
    throw std::runtime_error("invalid kernel fn");
  }
  for (auto h : handles) delete h;
  if (st != VT_STATUS_OK) {
    const std::string detail = tl_err;
    throw_from_status(ctx->fqname.c_str(), st, detail);
  }
  if (tout == nullptr) {
    throw std::runtime_error(std::string("plugin kernel returned OK but out==NULL: ") + ctx->fqname);
  }
  track.disarm(tout);
  vbt::core::TensorImpl out = adopt_owned_and_free(tout);
  s.clear();
  s.push_back(out);
}

static void plugin_cuda_trampoline(void* vctx, vbt::dispatch::BoxedStack& s) {
  auto* ctx = reinterpret_cast<PluginKernelCtx*>(vctx);
  if (!ctx || !ctx->handle) throw std::runtime_error("invalid kernel ctx");
  ActiveCallsGuard acg(ctx->handle->inflight);
  AllocTracker track;
  tl_err.clear();
  vt_tensor tout = nullptr;
  const size_t nargs = s.size();
  vt_stream vs = 0;

  bool fabric_effective = ctx->allow_multi_device_fabric_snapshot;
  if (ctx->entry) {
    const auto* st = ctx->entry->state_v2.load(std::memory_order_acquire);
    if (st) fabric_effective = st->allow_multi_device_fabric;
  }

#if VBT_WITH_CUDA
  std::optional<vbt::cuda::DeviceGuard> dg;

  auto extract_cpu_scalar_int64 = [&](const vbt::core::TensorImpl& t,
                                      const char* argname) -> std::int64_t {
    using vbt::core::ScalarType;
    if (t.device().type != kDLCPU) {
      throw std::runtime_error(std::string("[Fabric] ") + ctx->fqname + ": " + argname +
                               " must be a CPU scalar int64 tensor");
    }
    if (t.numel() != 1) {
      throw std::runtime_error(std::string("[Fabric] ") + ctx->fqname + ": " + argname +
                               " must have numel()==1");
    }
    if (t.dtype() != ScalarType::Int64) {
      throw std::runtime_error(std::string("[Fabric] ") + ctx->fqname + ": " + argname +
                               " must have dtype int64");
    }
    const void* p = t.data();
    if (!p) {
      throw std::runtime_error(std::string("[Fabric] ") + ctx->fqname + ": " + argname +
                               " has no data");
    }
    return *static_cast<const std::int64_t*>(p);
  };

  if (fabric_effective) {
    // Fabric plugin trampoline contract: compute-device-correct stream selection
    // is defined only for the canonical 5-arg Fabric schema.
    if (nargs != 5) {
      throw std::runtime_error("[Fabric] plugin trampoline: expected nargs==5");
    }

    const std::int64_t compute_device_i64 =
        extract_cpu_scalar_int64(s[2], "compute_device");
    const std::int64_t max_device_i64 =
        static_cast<std::int64_t>(std::numeric_limits<vbt::cuda::DeviceIndex>::max());
    if (compute_device_i64 < 0 || compute_device_i64 > max_device_i64) {
      throw std::runtime_error(
          std::string("[Fabric] ") + ctx->fqname +
          ": compute_device out of range: " + std::to_string(compute_device_i64));
    }
    const auto compute_device =
        static_cast<vbt::cuda::DeviceIndex>(compute_device_i64);

    // Ensure the current CUDA device matches the stream handle we pass.
    // Many CUDA runtime APIs require the current device to match the stream's
    // device.
    dg.emplace(compute_device);
    auto cs = vbt::cuda::getCurrentStream(compute_device);
    vs = static_cast<vt_stream>(cs.handle());
  } else {
    // Non-Fabric path: infer the CUDA device from the first CUDA tensor arg.
    try {
      vbt::cuda::DeviceIndex dev_index = -1;
      for (size_t i = 0; i < nargs; ++i) {
        const auto& d = s[i].device();
        if (d.type == kDLCUDA) {
          dev_index = static_cast<vbt::cuda::DeviceIndex>(d.index);
          break;
        }
      }
      if (dev_index >= 0) {
        dg.emplace(dev_index);
        auto cs = vbt::cuda::getCurrentStream(dev_index);
        vs = static_cast<vt_stream>(cs.handle());
      }
    } catch (...) {
      vs = 0;
      dg.reset();
    }
  }
#endif
  std::vector<vt_tensor> handles; handles.reserve(nargs);
  for (size_t i = 0; i < nargs; ++i) handles.push_back(make_borrowed_handle(s[i]));
  vt_status st = VT_STATUS_INTERNAL;
  if (ctx->boxed) {
    st = ctx->boxed(vs, handles.data(), nargs, &tout);
  } else if (ctx->k2) {
    if (nargs != 2) {
      for (auto h : handles) delete h;
      throw std::invalid_argument(std::string("arity mismatch ") + ctx->fqname + ": expected 2, got " + std::to_string(nargs));
    }
    st = ctx->k2(vs, handles[0], handles[1], &tout);
  } else {
    for (auto h : handles) delete h;
    throw std::runtime_error("invalid kernel fn");
  }
  for (auto h : handles) delete h;
  if (st != VT_STATUS_OK) {
    const std::string detail = tl_err;
    throw_from_status(ctx->fqname.c_str(), st, detail);
  }
  if (tout == nullptr) {
    throw std::runtime_error(std::string("plugin kernel returned OK but out==NULL: ") + ctx->fqname);
  }
  track.disarm(tout);
  vbt::core::TensorImpl out = adopt_owned_and_free(tout);
  s.clear();
  s.push_back(out);
}

static vt_status install_plugin_kernel(PluginHandle* handle,
                                      const char* fqname,
                                      vt_dispatch_key key,
                                      StagedKernelKind kind,
                                      vt_kernel_boxed_fn boxed,
                                      kernel2_t k2) {
  if (!handle) {
    tl_err = "loader internal: no handle";
    return VT_STATUS_INTERNAL;
  }

  auto& D = vbt::dispatch::Dispatcher::instance();
  uint8_t arity = 0;
  const vbt::dispatch::OperatorEntry* entry = nullptr;
  bool allow_multi_device_fabric_snapshot = false;

  try {
    auto h = D.find(fqname);
    entry = &h.get();
    arity = entry->schema.in_arity;
    // Snapshot for debugging only; runtime Fabric behavior consults `entry`.
    allow_multi_device_fabric_snapshot = entry->allow_multi_device_fabric;
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INVALID_ARG;
  }

  if (kind == StagedKernelKind::Arity2 && arity != 2) {
    tl_err = std::string("arity mismatch: ") + fqname + " expected 2";
    return VT_STATUS_INVALID_ARG;
  }

  const uint8_t effective_arity =
      (kind == StagedKernelKind::Arity2) ? 2 : arity;

  auto ctx = std::make_unique<PluginKernelCtx>();
  ctx->handle = handle;
  ctx->fqname = fqname;
  ctx->entry = entry;
  ctx->arity = effective_arity;
  ctx->allow_multi_device_fabric_snapshot = allow_multi_device_fabric_snapshot;
  ctx->boxed = (kind == StagedKernelKind::Boxed) ? boxed : nullptr;
  ctx->k2 = (kind == StagedKernelKind::Arity2) ? k2 : nullptr;

  vbt::dispatch::KernelFunction kf =
      vbt::dispatch::KernelFunction::makeBoxedCtx(
          effective_arity,
          key == kDLCPU ? &plugin_cpu_trampoline : &plugin_cuda_trampoline,
          ctx.get());

  try {
    const std::string fq(fqname);
    if (key == kDLCPU) {
      auto prev = D.replaceCpuKernelFunction(fq, kf);
      handle->prev_by_fqname[fq].cpu_prev = prev;
      handle->cpu_ctx_by_fqname[fq] = std::move(ctx);
    } else {
      auto prev = D.replaceCudaKernelFunction(fq, kf);
      handle->prev_by_fqname[fq].cuda_prev = prev;
      handle->cuda_ctx_by_fqname[fq] = std::move(ctx);
    }
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INVALID_ARG;
  }

  return VT_STATUS_OK;
}

static vt_status host_register_kernel_boxed(const char* fqname, vt_dispatch_key key, vt_kernel_boxed_fn fn) {
  tl_err.clear();
  if (!g_current_handle_for_registration) { tl_err = "loader internal: no handle"; return VT_STATUS_INTERNAL; }
  if (!fqname) { tl_err = "invalid argument: fqname is null"; return VT_STATUS_INVALID_ARG; }
  if (*fqname == '\0') { tl_err = "invalid argument: fqname is empty"; return VT_STATUS_INVALID_ARG; }
  if (!fn) { tl_err = "null kernel"; return VT_STATUS_INVALID_ARG; }
  if (key != kDLCPU && key != kDLCUDA) { tl_err = std::string("unsupported dispatch key: ") + std::to_string((int)key); return VT_STATUS_UNSUPPORTED; }

  if (std::strcmp(fqname, "vt::add") == 0) {
    return VT_STATUS_OK;
  }

  if (tls_plugin_txn) {
    PluginInitTxn& txn = *tls_plugin_txn;
    const std::string fq(fqname);

    // Ensure the op exists either in this init transaction or in the dispatcher.
    if (txn.defs_by_fqname.count(fq) == 0 &&
        !vbt::dispatch::Dispatcher::instance().has(fq)) {
      tl_err = std::string(key == kDLCPU ? "undefined op in VBT_IMPL_CPU: "
                                         : "undefined op in VBT_IMPL_CUDA: ") +
               fq;
      return VT_STATUS_INVALID_ARG;
    }

    StagedKernel sk;
    sk.fqname = fq;
    sk.key = key;
    sk.kind = StagedKernelKind::Boxed;
    sk.boxed = fn;
    sk.k2 = nullptr;
    txn.kernels.push_back(std::move(sk));
    return VT_STATUS_OK;
  }

  return install_plugin_kernel(g_current_handle_for_registration,
                               fqname,
                               key,
                               StagedKernelKind::Boxed,
                               fn,
                               nullptr);
}

static vt_status host_register_kernel2(const char* fqname, vt_dispatch_key key, kernel2_t fn) {
  tl_err.clear();
  if (!g_current_handle_for_registration) { tl_err = "loader internal: no handle"; return VT_STATUS_INTERNAL; }
  if (!fqname) { tl_err = "invalid argument: fqname is null"; return VT_STATUS_INVALID_ARG; }
  if (*fqname == '\0') { tl_err = "invalid argument: fqname is empty"; return VT_STATUS_INVALID_ARG; }
  if (!fn) { tl_err = "null kernel"; return VT_STATUS_INVALID_ARG; }
  if (key != kDLCPU && key != kDLCUDA) { tl_err = std::string("unsupported dispatch key: ") + std::to_string((int)key); return VT_STATUS_UNSUPPORTED; }

  if (std::strcmp(fqname, "vt::add") == 0) {
    return VT_STATUS_OK;
  }

  if (tls_plugin_txn) {
    PluginInitTxn& txn = *tls_plugin_txn;
    const std::string fq(fqname);

    uint8_t arity = 0;
    auto it = txn.defs_by_fqname.find(fq);
    if (it != txn.defs_by_fqname.end()) {
      arity = it->second.in_arity;
    } else {
      auto& D = vbt::dispatch::Dispatcher::instance();
      if (!D.has(fq)) {
        tl_err = std::string(key == kDLCPU ? "undefined op in VBT_IMPL_CPU: "
                                           : "undefined op in VBT_IMPL_CUDA: ") +
                 fq;
        return VT_STATUS_INVALID_ARG;
      }
      try {
        auto h = D.find(fq);
        arity = h.get().schema.in_arity;
      } catch (const std::exception& e) {
        tl_err = e.what();
        return VT_STATUS_INVALID_ARG;
      }
    }

    if (arity != 2) {
      tl_err = std::string("arity mismatch: ") + fq + " expected 2";
      return VT_STATUS_INVALID_ARG;
    }

    StagedKernel sk;
    sk.fqname = fq;
    sk.key = key;
    sk.kind = StagedKernelKind::Arity2;
    sk.boxed = nullptr;
    sk.k2 = fn;
    txn.kernels.push_back(std::move(sk));
    return VT_STATUS_OK;
  }

  return install_plugin_kernel(g_current_handle_for_registration,
                               fqname,
                               key,
                               StagedKernelKind::Arity2,
                               nullptr,
                               fn);
}

// Forward trampoline for vt::add CPU boxed-with-ctx
static void add_cpu_trampoline(void* vctx, vbt::dispatch::BoxedStack& s) {
  using namespace vbt::core;
  auto* ctx = reinterpret_cast<PluginHandle::AddCpuCtx*>(vctx);
  if (!ctx || !ctx->fn || !ctx->handle) throw std::runtime_error("invalid kernel mode: vt::add");
  ActiveCallsGuard acg(ctx->handle->inflight);
  AllocTracker track;
  // Rely on Dispatcher for arity/device-uniformity; preserve dtype error substring like default kernel
  if (s.size() != 2) {
    throw std::invalid_argument(std::string("arity mismatch ") + ctx->fqname + ": expected 2, got " + std::to_string(s.size()));
  }
  if (s[0].dtype() != s[1].dtype()) {
    throw std::invalid_argument("vt::add: dtype mismatch");
  }
  // Prepare call
  tl_err.clear();
  vt_tensor ta = make_borrowed_handle(s[0]);
  vt_tensor tb = make_borrowed_handle(s[1]);
  vt_tensor tout = nullptr;
  vt_status st = ctx->fn(/*vt_stream*/0, ta, tb, &tout);
  delete ta; delete tb;
  if (st != VT_STATUS_OK) {
    // Map statuses
    const std::string detail = tl_err;
    throw_from_status(ctx->fqname, st, detail);
  }
  if (tout == nullptr) {
    throw std::runtime_error("plugin kernel returned OK but out==NULL: vt::add");
  }
  // Disarm tracker before adopting (ownership transferred)
  track.disarm(tout);
  vbt::core::TensorImpl out = adopt_owned_and_free(tout);
  s.clear();
  s.push_back(out);
}

// CUDA trampoline for vt::add arity-2
static void add_cuda_trampoline(void* vctx, vbt::dispatch::BoxedStack& s) {
  using namespace vbt::core;
  auto* ctx = reinterpret_cast<PluginHandle::AddCudaCtx*>(vctx);
  if (!ctx || !ctx->fn || !ctx->handle) throw std::runtime_error("invalid kernel mode: vt::add");
  ActiveCallsGuard acg(ctx->handle->inflight);
  AllocTracker track;
  // Prepare call: dispatcher has checked device uniformity
  // Early guards for parity with CPU path
  if (s.size() != 2) {
    throw std::invalid_argument(std::string("arity mismatch ") + ctx->fqname + ": expected 2, got " + std::to_string(s.size()));
  }
  if (s[0].dtype() != s[1].dtype()) {
    throw std::invalid_argument("vt::add: dtype mismatch");
  }
  tl_err.clear();
  vt_tensor ta = make_borrowed_handle(s[0]);
  vt_tensor tb = make_borrowed_handle(s[1]);
  vt_tensor tout = nullptr;
  // Use current VBT CUDA stream handle (0 if default)
  vt_stream vs = 0;
#if VBT_WITH_CUDA
  std::optional<vbt::cuda::DeviceGuard> dg;
  try {
    vbt::cuda::DeviceIndex dev_index =
        static_cast<vbt::cuda::DeviceIndex>(s[0].device().index);
    dg.emplace(dev_index);
    auto cs = vbt::cuda::getCurrentStream(dev_index);
    vs = static_cast<vt_stream>(cs.handle());
  } catch (...) {
    vs = 0;
    dg.reset();
  }
#endif
  vt_status st = ctx->fn(vs, ta, tb, &tout);
  delete ta; delete tb;
  if (st != VT_STATUS_OK) {
    const std::string detail = tl_err;
    throw_from_status(ctx->fqname, st, detail);
  }
  if (tout == nullptr) {
    throw std::runtime_error("plugin kernel returned OK but out==NULL: vt::add");
  }
  track.disarm(tout);
  vbt::core::TensorImpl out = adopt_owned_and_free(tout);
  s.clear();
  s.push_back(out);
}

static vt_status host_register_cpu_kernel2(const char* fqname, kernel2_t fn) {
  return host_register_kernel2(fqname, kDLCPU, fn);
}

static vt_status host_register_cuda_kernel2(const char* fqname, kernel2_t fn) {
  return host_register_kernel2(fqname, kDLCUDA, fn);
}

// Tensor queries and allocators
static int64_t host_tensor_numel(vt_tensor t) {
  return static_cast<int64_t>(t->impl->numel());
}
static DLDataType host_tensor_dtype(vt_tensor t) {
  return vbt::core::to_dlpack_dtype(t->impl->dtype());
}
static DLDevice host_tensor_device(vt_tensor t) {
  DLDevice d{t->impl->device().type, t->impl->device().index};
  return d;
}
static size_t host_tensor_ndim(vt_tensor t) { return t->impl->sizes().size(); }
static const int64_t* host_tensor_sizes(vt_tensor t) { return t->impl->sizes().empty() ? nullptr : t->impl->sizes().data(); }
static const int64_t* host_tensor_strides(vt_tensor t) { return t->impl->strides().empty() ? nullptr : t->impl->strides().data(); }
static int64_t host_tensor_storage_offset(vt_tensor t) { return static_cast<int64_t>(t->impl->storage_offset()); }
static size_t host_tensor_itemsize(vt_tensor t) { return vbt::core::itemsize(t->impl->dtype()); }
static const void* host_tensor_data(vt_tensor t) { return t->impl->data(); }
static void* host_tensor_mutable_data(vt_tensor t) {
  // Only allow mutable access for host-allocated output tensors
  if (!t->owned) {
    tl_err = "host.tensor_mutable_data: input tensor is read-only";
    return nullptr;
  }
  return const_cast<void*>(t->impl->data());
}
static int host_tensor_is_contiguous(vt_tensor t) { return t->impl->is_contiguous() ? 1 : 0; }

static vt_status host_tensor_new_dense_like(vt_tensor like, vt_tensor* out) {
  if (!out) { tl_err = "tensor_new_dense_like: out is NULL"; return VT_STATUS_INVALID_ARG; }
  *out = nullptr;
  auto dev = like->impl->device();
  try {
    using namespace vbt::core;
    // Allocate new storage for contiguous tensor with same dtype and sizes
    const auto& sizes = like->impl->sizes();
    const auto& dtype = like->impl->dtype();
    std::vector<int64_t> strides;
    strides.resize(sizes.size());
    // Compute row-major strides
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) { strides[i] = stride; stride *= sizes[i]; }
    size_t nbytes = static_cast<size_t>(stride) * itemsize(dtype);
    if (dev.type == kDLCPU) {
      void* buf = ::operator new(nbytes);
      vbt::core::DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
      auto storage = vbt::core::make_intrusive<vbt::core::Storage>(std::move(dp), nbytes);
      auto* impl = new vbt::core::TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype, dev);
      vt_tensor h = new vt_tensor__{impl, true};
      if (tls_tracker) tls_tracker->owned.push_back(h);
      *out = h;
      return VT_STATUS_OK;
    } else if (dev.type == kDLCUDA) {
#if VBT_WITH_CUDA
      auto storage = vbt::cuda::new_cuda_storage(nbytes, dev.index);
      auto* impl = new vbt::core::TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype, dev);
      vt_tensor h = new vt_tensor__{impl, true};
      if (tls_tracker) tls_tracker->owned.push_back(h);
      *out = h;
      return VT_STATUS_OK;
#else
      (void)nbytes;
      tl_err = "tensor_new_dense_like: CUDA unavailable";
      return VT_STATUS_UNSUPPORTED;
#endif
    } else {
      tl_err = "tensor_new_dense_like: unsupported device";
      return VT_STATUS_UNSUPPORTED;
    }
  } catch (const std::exception& e) {
    tl_err = e.what();
    return VT_STATUS_INTERNAL;
  }
}

static vt_stream host_current_cuda_stream(int32_t device_index) {
#if VBT_WITH_CUDA
  try {
    auto s = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(device_index));
    return static_cast<vt_stream>(s.handle());
  } catch (...) {
    return 0ULL;
  }
#else
  (void)device_index;
  return 0ULL;
#endif
}
static void host_set_last_error(const char* msg) { tl_err = (msg ? msg : ""); }

} // namespace

namespace detail {

::vbt::core::TensorImpl& require_tensor_impl(vt_tensor h,
                                             const char* arg_name) {
  if (!h) {
    throw std::invalid_argument(std::string(arg_name) + " must be non-null");
  }
  if (!h->impl) {
    throw std::invalid_argument(std::string(arg_name) + "->impl must be non-null");
  }
  return *h->impl;
}

vt_tensor make_borrowed_handle_for_tests(const ::vbt::core::TensorImpl& t) {
  return make_borrowed_handle(t);
}

void destroy_borrowed_handle_for_tests(vt_tensor h) {
  if (!h) {
    return;
  }
  if (h->owned) {
    throw std::logic_error(
        "destroy_borrowed_handle_for_tests: expected non-owned handle");
  }
  delete h;
}

}  // namespace detail

// Public API
vt_status load_library(const char* path) noexcept {
  tl_err.clear();
  try {
    if (!path) { set_loader_error("null path"); return VT_STATUS_INVALID_ARG; }

    const bool atomic_requested = plugin_atomic_commit_requested();

    // Lightweight upfront policy checks to catch obvious bad paths before dlopen
    // 1) World-writable parent without sticky bit (unless env override)
    {
#if defined(__unix__) || defined(__APPLE__)
      std::string p(path);
      auto pos = p.find_last_of('/');
      if (pos != std::string::npos) {
        std::string parent = p.substr(0, pos == 0 ? 1 : pos);
        struct stat stp{};
        if (::stat(parent.c_str(), &stp) == 0) {
          if ((stp.st_mode & S_IWOTH) && !(stp.st_mode & S_ISVTX)) {
            const char* allow = std::getenv("VBT_ALLOW_WORLD_WRITABLE");
            if (!allow || std::string(allow) != "1") {
              set_loader_error("world-writable parent disallowed");
              return VT_STATUS_NOT_FOUND;
            }
          }
        }
      } else {
        // Relative path: apply policy to current directory
        struct stat stp{};
        if (::stat(".", &stp) == 0) {
          if ((stp.st_mode & S_IWOTH) && !(stp.st_mode & S_ISVTX)) {
            const char* allow = std::getenv("VBT_ALLOW_WORLD_WRITABLE");
            if (!allow || std::string(allow) != "1") {
              set_loader_error("world-writable parent disallowed");
              return VT_STATUS_NOT_FOUND;
            }
          }
        }
      }
      // 2) Leaf symlink rejection
      struct stat lst{};
      if (lstat(path, &lst) == 0) {
        if (S_ISLNK(lst.st_mode)) {
          set_loader_error("symlink in path rejected");
          return VT_STATUS_NOT_FOUND;
        }
      }
#endif
    }

    // NOTE: We do not use RTLD_NOLOAD to reject duplicates here. A plugin library
    // can be mapped into the process (e.g., linked into a test binary) without
    // having had vbt_plugin_init() executed. Duplicates are handled by the
    // canonical-path/devino guards below.


#if defined(__linux__)
    // Linux: direct dlopen after policy checks
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) { const char* em = dlerror(); set_loader_error(std::string(em ? em : "dlopen failed") + ": " + path); return VT_STATUS_NOT_FOUND; }
    // Compute dev/inode key for duplicate guard
#if defined(__unix__) || defined(__APPLE__)
    struct stat lst{}; dev_t dev = 0; ino_t ino = 0;
    if (lstat(path, &lst) == 0) { dev = lst.st_dev; ino = lst.st_ino; }
    char canon_buf[PATH_MAX];
    const char* canon = realpath(path, canon_buf);
    const std::string canon_path = canon ? std::string(canon) : std::string(path);
#else
    dev_t dev = 0; ino_t ino = 0;
    const std::string canon_path(path);
#endif
#else
    // Non-Linux: direct dlopen after policy checks
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) { const char* em = dlerror(); set_loader_error(std::string(em ? em : "dlopen failed") + ": " + path); return VT_STATUS_NOT_FOUND; }
#endif

    auto sym_get_abi = (uint32_t(*)()) dlsym(handle, "vbt_plugin_get_abi_version");
    if (!sym_get_abi) { tl_err = "missing symbol: vbt_plugin_get_abi_version"; dlclose(handle); return VT_STATUS_NOT_FOUND; }
    auto sym_init = (vt_status(*)(const struct vbt_host_api*, struct vbt_plugin_api*)) dlsym(handle, "vbt_plugin_init");
    if (!sym_init) { tl_err = "missing symbol: vbt_plugin_init"; dlclose(handle); return VT_STATUS_NOT_FOUND; }
    uint32_t abi = sym_get_abi();
    uint32_t host = VBT_PLUGIN_ABI_VERSION;
    uint32_t pmaj = (abi >> 16) & 0xFFFFu, pmin = abi & 0xFFFFu;
    uint32_t hmaj = (host >> 16) & 0xFFFFu, hmin = host & 0xFFFFu;
    if (pmaj != hmaj || pmin > hmin) {
      tl_err = std::string("ABI mismatch: host ") + std::to_string(hmaj) + "." + std::to_string(hmin) + " vs plugin " + std::to_string(pmaj) + "." + std::to_string(pmin);
      dlclose(handle);
      return VT_STATUS_ABI_MISMATCH;
    }
    // Build host api (address stable for the process lifetime)
    std::call_once(g_host_api_init_once, [&]() {
      g_host_api.host_abi_major = hmaj;
      g_host_api.host_abi_minor = hmin;
      g_host_api.register_library = &host_register_library;
      g_host_api.def = &host_def;
      g_host_api.register_cpu_kernel2 = &host_register_cpu_kernel2;
      // Tensor query and allocation functions must match vbt_plugin.h layout exactly
      g_host_api.tensor_numel = &host_tensor_numel;
      g_host_api.tensor_dtype = &host_tensor_dtype;
      g_host_api.tensor_device = &host_tensor_device;
      g_host_api.tensor_ndim = &host_tensor_ndim;
      g_host_api.tensor_sizes = &host_tensor_sizes;
      g_host_api.tensor_strides = &host_tensor_strides;
      g_host_api.tensor_storage_offset = &host_tensor_storage_offset;
      g_host_api.tensor_itemsize = &host_tensor_itemsize;
      g_host_api.tensor_data = &host_tensor_data;
      g_host_api.tensor_mutable_data = &host_tensor_mutable_data;
      g_host_api.tensor_is_contiguous = &host_tensor_is_contiguous;
      g_host_api.tensor_new_dense_like = &host_tensor_new_dense_like;
      g_host_api.set_last_error = &host_set_last_error;
      // Append-only fields in ABI order
      g_host_api.register_cuda_kernel2 = &host_register_cuda_kernel2;
      g_host_api.current_cuda_stream = &host_current_cuda_stream;
      g_host_api.register_kernel_boxed = &host_register_kernel_boxed;
      g_host_api.register_kernel2 = &host_register_kernel2;
      g_host_api.vt_tensor_iter_unary_cpu =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_unary_cpu_host;
      g_host_api.vt_tensor_iter_binary_cpu =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_binary_cpu_host;
      g_host_api.vt_tensor_iter_build_elementwise =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_build_elementwise_host;
      g_host_api.vt_tensor_iter_build_reduction =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_build_reduction_host;
      g_host_api.vt_tensor_iter_get_kind =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_get_kind_host;
      g_host_api.vt_tensor_iter_export_desc =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_export_desc_host;
      g_host_api.vt_tensor_iter_export_alias_info =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_export_alias_info_host;
      g_host_api.vt_tensor_iter_export_cuda_desc =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_export_cuda_desc_host;
      g_host_api.vt_tensor_iter_for_each_cpu =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_for_each_cpu_host;
      g_host_api.vt_tensor_iter_destroy =
          &::vbt::dispatch::plugin_helpers::vt_tensor_iter_destroy_host;
      // Append-only additions for ABI v1.4 (dispatcher v2 device policy setter)
      g_host_api.set_device_policy = &host_set_device_policy;
    });

    vbt_plugin_api plug_api{};
    // Create persistent handle and keep it alive in a global registry
    auto handle_sp = std::make_shared<PluginHandle>();
    handle_sp->dl_handle = handle;
    handle_sp->path = canon_path;
    handle_sp->abi_major = static_cast<uint16_t>(hmaj);
    handle_sp->abi_minor = static_cast<uint16_t>(hmin);

    // Reject duplicate loads of the same path to avoid replacing live trampolines without quiescent unpublish
    {
      std::lock_guard<std::mutex> lg(g_loader_mu);
      // path-string guard
      if (g_loaded_handles.find(canon_path) != g_loaded_handles.end() ||
          g_failed_init_handles.find(canon_path) != g_failed_init_handles.end()) {
        dlclose(handle);
        set_loader_error(std::string("plugin already loaded: ") + canon_path);
        return VT_STATUS_INVALID_ARG;
      }
      // dev/inode guard (best-effort; empty key when lstat failed)
      if (dev != 0 || ino != 0) {
        std::string key = std::to_string((unsigned long long)dev) + ":" + std::to_string((unsigned long long)ino);
        if (g_loaded_devino.count(key) || g_failed_init_devino.count(key)) {
          dlclose(handle);
          set_loader_error(std::string("plugin already loaded: ") + canon_path);
          return VT_STATUS_INVALID_ARG;
        }
      }
      // Reserve the slot by path to reduce races during init
      g_loaded_handles[canon_path] = nullptr;
    }
    PluginInitTxn txn;
    if (atomic_requested) {
      txn.txn_id = &txn;
    }

    struct ReservationReleaseGuard {
      PluginInitTxn* txn;
      explicit ReservationReleaseGuard(PluginInitTxn* t) noexcept : txn(t) {}
      ~ReservationReleaseGuard() noexcept { release_fqname_reservations(txn); }
    } reservation_release_guard(atomic_requested ? &txn : nullptr);

    vt_status st = VT_STATUS_INTERNAL;
    try {
      RegistrationTLSGuard tls_guard(handle_sp.get(), atomic_requested ? &txn : nullptr);
      st = sym_init(&g_host_api, &plug_api);
    } catch (const std::bad_alloc&) {
      tl_err = "out of memory";
      st = VT_STATUS_NOMEM;
    } catch (const std::exception& e) {
      tl_err = e.what();
      st = VT_STATUS_INTERNAL;
    }

    if (atomic_requested) {
      // Commit staged defs + kernels after init returns OK (publish-last bulk commit).
      if (st == VT_STATUS_OK) {
        try {
          auto& D = vbt::dispatch::Dispatcher::instance();
          vbt::dispatch::PluginCommitPlan plan;

          // 1) libraries
          std::sort(txn.libraries.begin(), txn.libraries.end());
          txn.libraries.erase(std::unique(txn.libraries.begin(),
                                          txn.libraries.end()),
                              txn.libraries.end());
          plan.libraries = txn.libraries;

          // 2) defs
          std::vector<std::string> defs;
          defs.reserve(txn.defs_by_fqname.size());
          for (const auto& kv : txn.defs_by_fqname) defs.push_back(kv.first);
          std::sort(defs.begin(), defs.end());

          plan.defs.reserve(defs.size());
          for (const auto& fq : defs) {
            const auto it = txn.defs_by_fqname.find(fq);
            if (it == txn.defs_by_fqname.end()) continue;
            vbt::dispatch::PluginCommitPlan::Def d;
            d.fqname = it->second.fqname;
            d.def_string = it->second.def_string;
            plan.defs.push_back(std::move(d));
          }

          // 3) policies
          std::vector<std::string> pols;
          pols.reserve(txn.policy_by_fqname.size());
          for (const auto& kv : txn.policy_by_fqname) pols.push_back(kv.first);
          std::sort(pols.begin(), pols.end());

          plan.policies.reserve(pols.size());
          for (const auto& fq : pols) {
            const auto it = txn.policy_by_fqname.find(fq);
            if (it == txn.policy_by_fqname.end()) continue;
            vbt::dispatch::PluginCommitPlan::Policy p;
            p.fqname = it->second.fqname;
            p.policy = it->second.policy;
            p.dispatch_arg_mask = it->second.dispatch_arg_mask;
            p.allow_undefined_mask = it->second.allow_undefined_mask;
            std::copy_n(it->second.constraint_kind_by_index,
                        vbt::dispatch::kV2DevicePolicyMaxArity,
                        p.constraint_kind_by_index);
            plan.policies.push_back(std::move(p));
          }

          // 4) kernels (sort + last-wins dedupe by fqname/key)
          auto key_rank = [](vt_dispatch_key k) noexcept -> int {
            return (k == kDLCPU) ? 0 : 1;
          };
          std::stable_sort(txn.kernels.begin(), txn.kernels.end(),
                           [&](const StagedKernel& a, const StagedKernel& b) {
                             if (a.fqname != b.fqname) return a.fqname < b.fqname;
                             return key_rank(a.key) < key_rank(b.key);
                           });

          std::vector<StagedKernel> kernels;
          kernels.reserve(txn.kernels.size());
          for (const auto& sk : txn.kernels) {
            if (!kernels.empty() && kernels.back().fqname == sk.fqname &&
                kernels.back().key == sk.key) {
              // last-wins; stable_sort preserves init call order within ties.
              kernels.back() = sk;
            } else {
              kernels.push_back(sk);
            }
          }

          plan.kernels.reserve(kernels.size());
          for (const auto& sk : kernels) {
            // Determine schema arity and, when the op already exists, capture its
            // entry pointer for runtime Fabric behavior.
            uint8_t arity = 0;
            const vbt::dispatch::OperatorEntry* entry = nullptr;
            bool allow_multi_device_fabric_snapshot = false;

            auto def_it = txn.defs_by_fqname.find(sk.fqname);
            if (def_it != txn.defs_by_fqname.end()) {
              arity = def_it->second.in_arity;
            } else {
              auto h = D.find(sk.fqname);
              entry = &h.get();
              arity = entry->schema.in_arity;
              allow_multi_device_fabric_snapshot = entry->allow_multi_device_fabric;
            }

            if (sk.kind == StagedKernelKind::Arity2 && arity != 2) {
              tl_err = std::string("arity mismatch: ") + sk.fqname + " expected 2";
              st = VT_STATUS_INVALID_ARG;
              break;
            }

            const uint8_t effective_arity =
                (sk.kind == StagedKernelKind::Arity2) ? 2 : arity;

            auto ctx = std::make_unique<PluginKernelCtx>();
            ctx->handle = handle_sp.get();
            ctx->fqname = sk.fqname;
            ctx->entry = entry;
            ctx->arity = effective_arity;
            ctx->allow_multi_device_fabric_snapshot =
                allow_multi_device_fabric_snapshot;
            ctx->boxed =
                (sk.kind == StagedKernelKind::Boxed) ? sk.boxed : nullptr;
            ctx->k2 =
                (sk.kind == StagedKernelKind::Arity2) ? sk.k2 : nullptr;

            vbt::dispatch::DispatchKey dk;
            vbt::dispatch::KernelFunction::BoxedWithCtxFn tramp = nullptr;
            if (sk.key == kDLCPU) {
              dk = vbt::dispatch::DispatchKey::CPU;
              tramp = &plugin_cpu_trampoline;
            } else if (sk.key == kDLCUDA) {
              dk = vbt::dispatch::DispatchKey::CUDA;
              tramp = &plugin_cuda_trampoline;
            } else {
              tl_err = std::string("unsupported dispatch key: ") +
                       std::to_string((int)sk.key);
              st = VT_STATUS_UNSUPPORTED;
              break;
            }

            vbt::dispatch::KernelFunction kf =
                vbt::dispatch::KernelFunction::makeBoxedCtx(
                    effective_arity, tramp, ctx.get());

            vbt::dispatch::PluginCommitPlan::Kernel pk;
            pk.fqname = sk.fqname;
            pk.key = dk;
            pk.kf = kf;
            plan.kernels.push_back(std::move(pk));

            // Keep ctx alive for process lifetime.
            if (dk == vbt::dispatch::DispatchKey::CPU) {
              handle_sp->cpu_ctx_by_fqname[sk.fqname] = std::move(ctx);
            } else {
              handle_sp->cuda_ctx_by_fqname[sk.fqname] = std::move(ctx);
            }
          }

          if (st == VT_STATUS_OK) {
            std::string err;
            const vt_status cst = D.apply_plugin_commit_plan(plan, &err);
            if (cst != VT_STATUS_OK) {
              st = cst;
              if (!err.empty()) tl_err = err;
            }
          }
        } catch (const std::bad_alloc&) {
          tl_err = "out of memory";
          st = VT_STATUS_NOMEM;
        } catch (const std::exception& e) {
          tl_err = e.what();
          st = VT_STATUS_INTERNAL;
        }

      }
    }

    if (st != VT_STATUS_OK) {
      // Roll back any installed trampolines to their previous state
      try {
        auto& D = vbt::dispatch::Dispatcher::instance();
        for (const auto& kv : handle_sp->prev_by_fqname) {
          const std::string& fq = kv.first;
          const auto& prevs = kv.second;
          try {
            if (handle_sp->cpu_ctx_by_fqname.count(fq)) {
              if (prevs.cpu_prev.has_value()) (void)D.replaceCpuKernelFunction(fq, *prevs.cpu_prev);
              else (void)D.uninstallCpuKernelFunction(fq);
            }
            if (handle_sp->cuda_ctx_by_fqname.count(fq)) {
              if (prevs.cuda_prev.has_value()) (void)D.replaceCudaKernelFunction(fq, *prevs.cuda_prev);
              else (void)D.uninstallCudaKernelFunction(fq);
            }
          } catch (...) { /* swallow */ }
        }
      } catch (...) {
        // Best-effort rollback; continue to error reporting
      }
      if (tl_err.empty()) tl_err = "plugin init failed";

      // registration is staged without publishing dispatch-visible pointers.
      // dlclose() on init failure is therefore safe.
      //
      // NOTE: Plugins are trusted; if a failing plugin spawns background threads
      // that keep executing plugin code after returning failure, dlclose() may
      // still be unsafe.
      if (atomic_requested) {
        {
          std::lock_guard<std::mutex> lg(g_loader_mu);
          // Drop the reserved slot.
          auto it = g_loaded_handles.find(canon_path);
          if (it != g_loaded_handles.end() && it->second == nullptr) {
            g_loaded_handles.erase(it);
          }
        }
        if (handle_sp->dl_handle) {
          dlclose(handle_sp->dl_handle);
          handle_sp->dl_handle = nullptr;
        }
        return st;
      }

      // Legacy behavior: On init failure we intentionally do NOT dlclose(). With
      // append-only dispatch snapshots, stale function pointers may still be
      // reachable by concurrent callers; keeping the library + ctx objects alive
      // prevents UAF.
      {
        std::lock_guard<std::mutex> lg(g_loader_mu);
        // Drop the reserved slot.
        auto it = g_loaded_handles.find(canon_path);
        if (it != g_loaded_handles.end() && it->second == nullptr) {
          g_loaded_handles.erase(it);
        }
        // Park the handle so trampoline ctx pointers and the dlopen() handle stay
        // valid for the process lifetime.
        (void)g_failed_init_handles.emplace(canon_path, handle_sp);
        if (dev != 0 || ino != 0) {
          std::string key = std::to_string((unsigned long long)dev) + ":" +
                            std::to_string((unsigned long long)ino);
          g_failed_init_devino.insert(key);
        }
      }
      return st;
    }

    // Persist the handle so trampoline ctx pointers remain valid for the process lifetime
    {
      std::lock_guard<std::mutex> lg(g_loader_mu);
      // Re-check duplicate before publish
      if (g_loaded_handles.find(canon_path) != g_loaded_handles.end() && g_loaded_handles[canon_path] != nullptr) {
        // already published by another thread
        if (handle_sp->dl_handle) dlclose(handle_sp->dl_handle);
        set_loader_error(std::string("plugin already loaded: ") + canon_path);
        return VT_STATUS_INVALID_ARG;
      }
      if (dev != 0 || ino != 0) {
        std::string key = std::to_string((unsigned long long)dev) + ":" + std::to_string((unsigned long long)ino);
        if (g_loaded_devino.count(key)) {
          if (handle_sp->dl_handle) dlclose(handle_sp->dl_handle);
          set_loader_error(std::string("plugin already loaded: ") + canon_path);
          return VT_STATUS_INVALID_ARG;
        }
        g_loaded_devino.insert(key);
      }
      // Publish handle
      g_loaded_handles[canon_path] = handle_sp;
    }
    (void)plug_api;
    return VT_STATUS_OK;
  } catch (const std::exception& e) {
    set_loader_error(e.what());
    return VT_STATUS_INTERNAL;
  } catch (...) {
    set_loader_error("internal error");
    return VT_STATUS_INTERNAL;
  }
}

const char* get_last_error() noexcept { return tl_err.c_str(); }
void set_last_error(const char* msg) noexcept { try { tl_err = (msg ? msg : ""); } catch (...) { /* drop */ } }

std::vector<std::string> loaded_libraries() {
  std::lock_guard<std::mutex> lg(g_loader_mu);
  std::vector<std::string> out;
  out.reserve(g_loaded_handles.size());
  for (const auto& kv : g_loaded_handles) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

bool is_library_loaded(const std::string& path) {
  struct stat st{};
  // Use stat() to follow symlinks for query
  if (stat(path.c_str(), &st) != 0) return false;
  std::string key = std::to_string((unsigned long long)st.st_dev) + ":" + std::to_string((unsigned long long)st.st_ino);
  std::lock_guard<std::mutex> lg(g_loader_mu);
  return g_loaded_devino.count(key) != 0 || g_failed_init_devino.count(key) != 0;
}

} // namespace plugin
} // namespace dispatch
} // namespace vbt
