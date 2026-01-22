// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef VBT_PLUGIN_VBT_PLUGIN_H_
#define VBT_PLUGIN_VBT_PLUGIN_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <dlpack/dlpack.h>

// Opaque tensor handle (borrowed for inputs; owned for outputs created by host allocators)
typedef struct vt_tensor__* vt_tensor;

// Opaque stream handle. 0 == CUDA default stream (nullptr)
typedef uint64_t vt_stream;

// Opaque TensorIterator handle
typedef struct vt_tensor_iter__* vt_tensor_iter;

// Status codes for host/plugin calls
typedef enum vt_status {
  VT_STATUS_OK = 0,
  VT_STATUS_INVALID_ARG = 1,
  VT_STATUS_UNSUPPORTED = 2,
  VT_STATUS_NOT_FOUND = 3,
  VT_STATUS_INTERNAL = 4,
  VT_STATUS_ABI_MISMATCH = 5,
  VT_STATUS_NOMEM = 6,
  VT_STATUS_RUNTIME_ERROR = 7
} vt_status;

// ABI versioning (semantic: major must match, plugin minor <= host minor)
#define VBT_PLUGIN_ABI_VERSION_MAJOR 1
#define VBT_PLUGIN_ABI_VERSION_MINOR 4
#define VBT_PLUGIN_ABI_ENCODE(maj, min) (((uint32_t)(maj) << 16) | ((uint32_t)(min)))
#define VBT_PLUGIN_ABI_VERSION \
  VBT_PLUGIN_ABI_ENCODE(VBT_PLUGIN_ABI_VERSION_MAJOR, VBT_PLUGIN_ABI_VERSION_MINOR)

// Prototype for an arity-2 kernel (CPU/CUDA). Host passes vt_stream (0 for CPU/default)
typedef vt_status (*vt_kernel2_fn)(vt_stream s, vt_tensor a, vt_tensor b, vt_tensor* out);

// Dispatch key: reuse DLPack device enum for parity (kDLCPU=1, kDLCUDA=2)
typedef DLDeviceType vt_dispatch_key;

// Dispatcher v2 device policy C ABI (v1.4, design/dispatcher/p2)
typedef uint8_t vt_device_policy;
#define VT_DEVICE_POLICY_ALL_SAME_DEVICE    ((uint8_t)0)
#define VT_DEVICE_POLICY_MASKED_SAME_DEVICE ((uint8_t)1)
#define VT_DEVICE_POLICY_FABRIC5ARG         ((uint8_t)2)  /* core-only; host thunk rejects */

typedef uint8_t vt_constraint_kind;
#define VT_CONSTRAINT_MUST_MATCH_DISPATCH_IF_DEFINED ((uint8_t)0)
#define VT_CONSTRAINT_CPU_I64_SCALAR_0D              ((uint8_t)1)
#define VT_CONSTRAINT_CPU_BOOL_SCALAR_0D             ((uint8_t)2)
#define VT_CONSTRAINT_DEFER_TO_KERNEL                ((uint8_t)3)

typedef struct vt_device_constraint {
  uint8_t index;        /* arg index */
  uint8_t kind;         /* vt_constraint_kind */
  uint8_t reserved[6];  /* must be zero; ensures sizeof==8 */
} vt_device_constraint;

#ifdef __cplusplus
static_assert(sizeof(vt_device_constraint) == 8,
              "vt_device_constraint must be 8 bytes");
#else
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(sizeof(vt_device_constraint) == 8,
               "vt_device_constraint must be 8 bytes");
#else
typedef char vt_device_constraint_must_be_8_bytes[
    (sizeof(vt_device_constraint) == 8) ? 1 : -1];
#endif
#endif

// Boxed variable-arity kernel (Tensor-only)
typedef vt_status (*vt_kernel_boxed_fn)(vt_stream s,
                                        const vt_tensor* args,
                                        size_t nargs,
                                        vt_tensor* out);

// User callback; mirrors vbt::core::TensorIterBase::loop1d_t but is pure C.
typedef void (*vt_tensor_iter_loop1d_fn)(char** data,
                                         const int64_t* strides,
                                         int64_t size,
                                         void* ctx);

// Overlap-checking mode for TensorIterator-based helpers.
typedef enum vt_iter_overlap_mode {
  // Disable cross-tensor alias analysis (TensorIterConfig::check_mem_overlap(false)).
  // Expert-only: TI may accept aliasing patterns that FULL would reject.
  VT_ITER_OVERLAP_DISABLE = 0,

  // Enable full TI cross-tensor alias analysis (check_mem_overlap(true)),
  // enforcing the alias and overlap invariants documented for TI.
  VT_ITER_OVERLAP_ENABLE  = 1,
} vt_iter_overlap_mode;

// Lightweight configuration for plugin TI helpers.
typedef struct vt_iter_config {
  // Effective max iteration rank. Valid values:
  //   - 0: use TI default (kTensorIterMaxRank == 64).
  //   - [1, kTensorIterMaxRank]: explicitly cap iteration rank.
  // Any other value results in VT_STATUS_INVALID_ARG.
  int64_t max_rank;

  // Overlap-checking policy; see vt_iter_overlap_mode.
  vt_iter_overlap_mode check_mem_overlap;
} vt_iter_config;

// Default configuration: TI default max_rank and enabled overlap checks.
#define VT_ITER_CONFIG_DEFAULT_INIT {0, VT_ITER_OVERLAP_ENABLE}

// === vt_tensor_iter handle and metadata C ABI ==========================================

// Mirrors vbt::core::kTensorIterMaxRank and TI limits.
#define VT_TENSOR_ITER_MAX_RANK       64
// Maximum number of operands exposed through the C ABI.
#define VT_TENSOR_ITER_MAX_OPERANDS   64
// Practical CUDA offset-calculator limit (mirrors PyTorch OffsetCalculator::MAX_DIMS).
#define VT_TENSOR_ITER_CUDA_MAX_NDIM  25

// Kind of TensorIterator represented by a handle.
typedef enum vt_tensor_iter_kind {
  VT_TENSOR_ITER_KIND_ELEMENTWISE = 0,
  VT_TENSOR_ITER_KIND_REDUCTION   = 1,
} vt_tensor_iter_kind;

// Logical role of an operand in the iterator.
typedef enum vt_tensor_iter_operand_role {
  VT_TENSOR_ITER_ROLE_READONLY      = 0,
  VT_TENSOR_ITER_ROLE_WRITEONLY     = 1,
  VT_TENSOR_ITER_ROLE_READWRITE     = 2,
  VT_TENSOR_ITER_ROLE_REDUCE_OUTPUT = 3,
} vt_tensor_iter_operand_role;

// Logical iteration descriptor for an entire TensorIterator instance.
typedef struct vt_tensor_iter_desc {
  int32_t ndim;            // logical iteration rank (0..VT_TENSOR_ITER_MAX_RANK)
  int32_t ntensors;        // total operands (1..VT_TENSOR_ITER_MAX_OPERANDS)
  int32_t num_outputs;     // outputs at front (1..ntensors)
  int32_t num_reduce_dims; // 0 for elementwise; >0 for reductions

  // Reduction dims are logical indices into the input tensor’s logical shape.
  int32_t reduce_dims[VT_TENSOR_ITER_MAX_RANK];

  // Iteration shape; sizes[d] for d>=ndim are zeroed.
  int64_t sizes[VT_TENSOR_ITER_MAX_RANK];

  // Byte strides per operand per dim; strides[k][d] for d>=ndim are zeroed.
  int64_t strides[VT_TENSOR_ITER_MAX_OPERANDS][VT_TENSOR_ITER_MAX_RANK];

  // Per-operand properties, indexed like TI operands:
  //   outputs: [0, num_outputs)
  //   inputs:  [num_outputs, ntensors)
  DLDataType                 dtypes[VT_TENSOR_ITER_MAX_OPERANDS];
  DLDevice                   devices[VT_TENSOR_ITER_MAX_OPERANDS];
  vt_tensor_iter_operand_role roles[VT_TENSOR_ITER_MAX_OPERANDS];
} vt_tensor_iter_desc;

// Coarse alias information projected from TI’s alias metadata.
typedef struct vt_tensor_iter_alias_info {
  uint32_t num_outputs;  // == desc.num_outputs
  uint32_t num_inputs;   // == desc.ntensors - desc.num_outputs
  uint8_t  has_alias_metadata;         // 0 if TI did not run alias analysis
  uint8_t  has_any_output_input_alias; // 1 if any output<->input full alias
  uint8_t  reserved[2];                // must be zeroed by host

  // Bit i set in output_may_alias_input means output i may FULLY alias an input.
  uint64_t output_may_alias_input;

  // Bit j set in input_may_alias_output means input j may FULLY alias an output.
  uint64_t input_may_alias_output;
} vt_tensor_iter_alias_info;

// CUDA-oriented descriptor for a single operand, derived from DeviceStrideMeta.
typedef struct vt_tensor_iter_cuda_desc {
  int32_t ndim;          // 0..VT_TENSOR_ITER_CUDA_MAX_NDIM
  int32_t operand_index; // which operand this describes (0-based)
  int64_t sizes[VT_TENSOR_ITER_CUDA_MAX_NDIM];
  int64_t strides[VT_TENSOR_ITER_CUDA_MAX_NDIM]; // element strides
} vt_tensor_iter_cuda_desc;

// === Host API passed to plugins at init time ===========================================

struct vbt_host_api {
  uint32_t host_abi_major; // = VBT_PLUGIN_ABI_VERSION_MAJOR
  uint32_t host_abi_minor; // = VBT_PLUGIN_ABI_VERSION_MINOR
  // Registration subset
  vt_status (*register_library)(const char* ns);
  vt_status (*def)(const char* def_string);
  vt_status (*register_cpu_kernel2)(const char* fqname, vt_kernel2_fn fn);
  // Tensor queries (borrowed views; valid only during the call)
  int64_t       (*tensor_numel)(vt_tensor t);
  DLDataType    (*tensor_dtype)(vt_tensor t);
  DLDevice      (*tensor_device)(vt_tensor t);
  size_t        (*tensor_ndim)(vt_tensor t);
  const int64_t* (*tensor_sizes)(vt_tensor t);   // NULL if scalar
  const int64_t* (*tensor_strides)(vt_tensor t); // NULL if scalar
  int64_t       (*tensor_storage_offset)(vt_tensor t);
  size_t        (*tensor_itemsize)(vt_tensor t);
  const void*   (*tensor_data)(vt_tensor t);
  void*         (*tensor_mutable_data)(vt_tensor t);
  int           (*tensor_is_contiguous)(vt_tensor t); // 0/1
  vt_status (*tensor_new_dense_like)(vt_tensor like, vt_tensor* out);
  // Streams and error channel
  void (*set_last_error)(const char* msg); // thread-local; NULL or "" clears
  vt_status (*register_cuda_kernel2)(const char* fqname, vt_kernel2_fn fn);
  vt_stream (*current_cuda_stream)(int32_t device_index); // returns 0 if default/unavailable
  // Append-only additions for ABI v1.1 (M_EXT2.2)
  vt_status (*register_kernel_boxed)(const char* fqname,
                                     vt_dispatch_key key,
                                     vt_kernel_boxed_fn fn);
  vt_status (*register_kernel2)(const char* fqname,
                                vt_dispatch_key key,
                                vt_kernel2_fn fn);
  vt_status (*vt_tensor_iter_unary_cpu)(const vt_iter_config* cfg,
                                        vt_tensor out,
                                        vt_tensor in,
                                        vt_tensor_iter_loop1d_fn loop,
                                        void* ctx);
  vt_status (*vt_tensor_iter_binary_cpu)(const vt_iter_config* cfg,
                                         vt_tensor out,
                                         vt_tensor a,
                                         vt_tensor b,
                                         vt_tensor_iter_loop1d_fn loop,
                                         void* ctx);
  vt_status (*vt_tensor_iter_build_elementwise)(const vt_iter_config* cfg,
                                                int32_t ntensors,
                                                const vt_tensor* tensors,
                                                vt_tensor_iter* out_iter);
  vt_status (*vt_tensor_iter_build_reduction)(const vt_iter_config* cfg,
                                              int32_t ntensors,
                                              const vt_tensor* tensors,
                                              int32_t reduce_dim,
                                              vt_tensor_iter* out_iter);
  vt_status (*vt_tensor_iter_get_kind)(vt_tensor_iter iter,
                                       vt_tensor_iter_kind* out_kind);
  vt_status (*vt_tensor_iter_export_desc)(vt_tensor_iter iter,
                                          vt_tensor_iter_desc* out_desc);
  vt_status (*vt_tensor_iter_export_alias_info)(vt_tensor_iter iter,
                                                vt_tensor_iter_alias_info* out_alias);
  vt_status (*vt_tensor_iter_export_cuda_desc)(vt_tensor_iter iter,
                                               int32_t operand_index,
                                               int32_t max_ndim,
                                               vt_tensor_iter_cuda_desc* out_desc);
  vt_status (*vt_tensor_iter_for_each_cpu)(vt_tensor_iter iter,
                                           vt_tensor_iter_loop1d_fn loop,
                                           void* ctx);
  void      (*vt_tensor_iter_destroy)(vt_tensor_iter iter);
  // Append-only additions for ABI v1.4 (dispatcher v2 device policy setter)
  vt_status (*set_device_policy)(
      const char* fqname,
      vt_device_policy policy,
      uint64_t dispatch_arg_mask,
      const vt_device_constraint* constraints,
      size_t nconstraints,
      uint64_t allow_undefined_mask);
};

// Plugin API returned at init
typedef struct vbt_plugin_api {
  uint32_t abi_version;   // VBT_PLUGIN_ABI_VERSION
  const char* name;       // plugin display name
} vbt_plugin_api;

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

vt_status vt_tensor_iter_build_elementwise(const vt_iter_config* cfg,
                                           int32_t ntensors,
                                           const vt_tensor* tensors,
                                           vt_tensor_iter* out_iter);

vt_status vt_tensor_iter_build_reduction(const vt_iter_config* cfg,
                                         int32_t ntensors,
                                         const vt_tensor* tensors,
                                         int32_t reduce_dim,
                                         vt_tensor_iter* out_iter);

vt_status vt_tensor_iter_get_kind(vt_tensor_iter iter,
                                  vt_tensor_iter_kind* out_kind);

vt_status vt_tensor_iter_export_desc(vt_tensor_iter iter,
                                     vt_tensor_iter_desc* out_desc);

vt_status vt_tensor_iter_export_alias_info(vt_tensor_iter iter,
                                           vt_tensor_iter_alias_info* out_alias);

vt_status vt_tensor_iter_export_cuda_desc(vt_tensor_iter iter,
                                          int32_t operand_index,
                                          int32_t max_ndim,
                                          vt_tensor_iter_cuda_desc* out_desc);

vt_status vt_tensor_iter_for_each_cpu(vt_tensor_iter iter,
                                      vt_tensor_iter_loop1d_fn loop,
                                      void* ctx);

void vt_tensor_iter_destroy(vt_tensor_iter iter);

// Required plugin symbols
uint32_t vbt_plugin_get_abi_version(void);
vt_status vbt_plugin_init(const struct vbt_host_api* host,
                          struct vbt_plugin_api* out_api);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VBT_PLUGIN_VBT_PLUGIN_H_
