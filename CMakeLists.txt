cmake_minimum_required(VERSION 3.26)
project(vibetensor LANGUAGES C CXX)

# C++ standard and reproducibility flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fno-record-gcc-switches)
  add_link_options(-Wl,--build-id)
endif()

# Runtime search paths (build tree only):
#
# When CMake finds Python from a Conda environment, it frequently injects the
# Conda lib directory into RPATH for all binaries that link against Python.
# Conda also commonly ships its own libstdc++/libgcc_s, which can be older than
# the system toolchain used to build VibeTensor and lead to runtime failures like
# "GLIBCXX_3.4.30 not found".
#
# To keep this mitigation scoped and portable, we *optionally* prepend the
# toolchain's runtime library directory to the *build* RPATH so system
# libstdc++/libgcc_s win over any Conda-provided copies.
set(VBT_FIX_CONDA_RPATH_SHADOWING_DEFAULT OFF)
if(DEFINED ENV{CONDA_PREFIX})
  set(VBT_FIX_CONDA_RPATH_SHADOWING_DEFAULT ON)
elseif(DEFINED CMAKE_CXX_COMPILER_ARG1 AND CMAKE_CXX_COMPILER_ARG1 MATCHES "compiler_compat")
  # Conda toolchains often inject a compiler wrapper via "compiler_compat".
  # This heuristic makes the RPATH fix robust under PEP517 build isolation
  # where CONDA_PREFIX may not be forwarded.
  set(VBT_FIX_CONDA_RPATH_SHADOWING_DEFAULT ON)
endif()
option(VBT_FIX_CONDA_RPATH_SHADOWING "Prepend toolchain runtime lib dir to build RPATH to avoid Conda libstdc++ shadowing" ${VBT_FIX_CONDA_RPATH_SHADOWING_DEFAULT})
unset(VBT_FIX_CONDA_RPATH_SHADOWING_DEFAULT)

if(VBT_FIX_CONDA_RPATH_SHADOWING AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so.6
    OUTPUT_VARIABLE VBT_TOOLCHAIN_LIBSTDCXX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(EXISTS "${VBT_TOOLCHAIN_LIBSTDCXX}")
    get_filename_component(VBT_TOOLCHAIN_LIBSTDCXX_DIR "${VBT_TOOLCHAIN_LIBSTDCXX}" DIRECTORY)
    list(PREPEND CMAKE_BUILD_RPATH "${VBT_TOOLCHAIN_LIBSTDCXX_DIR}")
  endif()

  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libgcc_s.so.1
    OUTPUT_VARIABLE VBT_TOOLCHAIN_LIBGCC
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(EXISTS "${VBT_TOOLCHAIN_LIBGCC}")
    get_filename_component(VBT_TOOLCHAIN_LIBGCC_DIR "${VBT_TOOLCHAIN_LIBGCC}" DIRECTORY)
    list(PREPEND CMAKE_BUILD_RPATH "${VBT_TOOLCHAIN_LIBGCC_DIR}")
  endif()

  list(REMOVE_DUPLICATES CMAKE_BUILD_RPATH)
  # Keep install RPATH aligned with build RPATH so editable installs and
  # subprocess-spawned tests can import the extension under Conda.
  set(CMAKE_INSTALL_RPATH "${CMAKE_BUILD_RPATH}")

  unset(VBT_TOOLCHAIN_LIBSTDCXX)
  unset(VBT_TOOLCHAIN_LIBSTDCXX_DIR)
  unset(VBT_TOOLCHAIN_LIBGCC)
  unset(VBT_TOOLCHAIN_LIBGCC_DIR)
endif()

# Options
option(VBT_USE_CUDA "Build with CUDA support" ON)
option(VBT_BUILD_TESTS "Build tests" ON)
option(VBT_BUILD_NODE "Build Node.js N-API addon" ON)
option(VBT_BUILD_RING_ALLREDUCE_PLUGIN "Build CUTLASS ring allreduce plugin (sm103a)" ON)
option(VBT_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(VBT_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(VBT_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(VBT_REQUIRE_DLPACK_ALIGNMENT "Require effective pointer alignment to itemsize in from_dlpack" ON)
option(VBT_REQUIRE_NUMPY_ALIGNMENT "Require effective pointer alignment to itemsize in NumPy view importer" ON)
# Require presence of BF16 (kDLBfloat) in DLPack headers when ON
option(VBT_REQUIRE_BF16 "Require DLPack BF16 (kDLBfloat) in headers" ON)
option(VBT_WITH_AUTOGRAD "Build Autograd scaffolding" ON)
option(VBT_WITH_DISPATCH_V2 "Build dispatcher v2 scaffolding" ON)
if(NOT VBT_WITH_DISPATCH_V2)
  message(FATAL_ERROR "Dispatcher v1 has been removed; VBT_WITH_DISPATCH_V2 must be ON.")
endif()

option(VBT_WITH_SAFETENSORS "Build safetensors utilities" ON)
option(VBT_SAFETENSORS_ENABLE_MMAP "Enable POSIX mmap loader" ON)

option(VBT_ENABLE_TI_STATS "Enable TensorIterator stats collection" ON)
if(VBT_ENABLE_TI_STATS)
  add_compile_definitions(VBT_TI_STATS=1)
endif()

# Enforce CUDA-only builds
if(NOT VBT_USE_CUDA)
  message(FATAL_ERROR "CUDA is mandatory: build without CUDA is disabled. Set -DVBT_USE_CUDA=ON and ensure the NVIDIA CUDA toolkit and a GPU are available.")
endif()

# Detect presence of DLPack BF16 enum (kDLBfloat)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include")
check_cxx_source_compiles("#include <dlpack/dlpack.h>
int main(){ int x = (int)kDLBfloat; (void)x; return 0; }" VBT_HAS_DLPACK_BF16_TEST)
if(VBT_HAS_DLPACK_BF16_TEST)
  add_compile_definitions(VBT_HAS_DLPACK_BF16=1)
else()
  if(VBT_REQUIRE_BF16)
    message(FATAL_ERROR "VBT_REQUIRE_BF16 is ON but kDLBfloat is missing in DLPack headers. Set -DVBT_REQUIRE_BF16=OFF to build without BF16.")
  endif()
  add_compile_definitions(VBT_HAS_DLPACK_BF16=0)
endif()

# Compile-only probes (CUB wrapper hygiene)
set(_vbt_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(_vbt_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_vbt_saved_required_definitions "${CMAKE_REQUIRED_DEFINITIONS}")

set(CMAKE_REQUIRED_INCLUDES "${PROJECT_SOURCE_DIR}/include")
set(CMAKE_REQUIRED_DEFINITIONS "-DVBT_WITH_CUDA=0" "-DVBT_INTERNAL_TESTS=0")
check_cxx_source_compiles("#include <vbt/cuda/cub.h>\nint main(){return 0;}" VBT_CUB_HEADER_COMPILES_NO_CUDA)
if(NOT VBT_CUB_HEADER_COMPILES_NO_CUDA)
  message(FATAL_ERROR "include/vbt/cuda/cub.h must compile with VBT_WITH_CUDA=0")
endif()

# Test-only APIs must not be visible when VBT_INTERNAL_TESTS=0.
check_cxx_source_compiles("#include <vbt/cuda/cub.h>\nint main(){ (void)sizeof(&vbt::cuda::cub::testonly::temp_storage_bytes0_requires_nonnull); return 0; }" VBT_CUB_TESTONLY_VISIBLE_WHEN_DISABLED)
if(VBT_CUB_TESTONLY_VISIBLE_WHEN_DISABLED)
  message(FATAL_ERROR "include/vbt/cuda/cub.h leaked test-only APIs when VBT_INTERNAL_TESTS=0")
endif()

set(CMAKE_REQUIRED_INCLUDES "${_vbt_saved_required_includes}")
set(CMAKE_REQUIRED_DEFINITIONS "${_vbt_saved_required_definitions}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_vbt_saved_try_compile_target_type}")
unset(_vbt_saved_try_compile_target_type)
unset(_vbt_saved_required_includes)
unset(_vbt_saved_required_definitions)

# Ensure Python development headers are available for nanobind
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Third-party (vendored)
# Avoid installing googletest/gmock headers/libs into the Python wheel
set(BUILD_GMOCK OFF CACHE BOOL "Disable GoogleMock build" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest installation" FORCE)
add_subdirectory(3rdparty/abseil-cpp)
add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
add_subdirectory(3rdparty/nanobind)
add_subdirectory(3rdparty/dlpack)

if(VBT_WITH_SAFETENSORS)
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  set(JSON_Install OFF CACHE BOOL "" FORCE)
  add_subdirectory(3rdparty/json EXCLUDE_FROM_ALL)
endif()

# Core library
add_library(vbt_core STATIC
  src/vbt/hello.cc
  src/vbt/logging/logging.cc
  src/vbt/core/storage.cc
  src/vbt/core/tensor.cc
  src/vbt/core/overlap.cc
  src/vbt/core/strided_loop.cc
  src/vbt/core/broadcast.cc
  src/vbt/core/write_guard.cc
  src/vbt/core/contiguity.cc
  src/vbt/core/view_ops.cc
  src/vbt/core/indexing_basic.cc
  src/vbt/core/indexing_advanced.cc
  src/vbt/core/indexing_advanced_stats.cc
  src/vbt/core/tensor_ops.cc
  src/vbt/core/tensor_iterator/core.cc
  src/vbt/core/tensor_iterator/fabric_elementwise_2gpu.cc
  src/vbt/cpu/tensor_iterator/for_each_cpu.cc
  src/vbt/cpu/tensor_iterator/reduction.cc
  src/vbt/core/tensor_iter.cc
  src/vbt/dispatch/dispatcher.cc
  src/vbt/dispatch/dispatch_key_set.cc
  src/vbt/dispatch/kernel_function.cc
  src/vbt/dispatch/registration.cc
  src/vbt/interop/dlpack.cc
  src/vbt/ops/default_kernels.cc
  src/vbt/ops/reduction_dim_dispatch.cc
  src/vbt/ops/indexing_dispatch.cc
  src/vbt/ops/embedding_dispatch.cc
  src/vbt/cpu/allocator.cc
  src/vbt/cpu/pinned_allocator.cc
  src/vbt/cpu/storage.cc
  src/vbt/dispatch/plugin_loader.cc
  src/vbt/dispatch/plugin_tensor_iter_helpers.cc
  src/vbt/dispatch/plugin_tensor_iter_handle.cc
  src/vbt/rng/generator.cc
  src/vbt/rng/cpu_kernels.cc
  src/vbt/rng/graph_capture.cc
)


target_include_directories(vbt_core PUBLIC include 3rdparty/dlpack/include 3rdparty/abseil-cpp 3rdparty/nanobind/include ${Python_INCLUDE_DIRS})
target_include_directories(vbt_core PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(vbt_core PUBLIC absl::log absl::log_initialize ${CMAKE_DL_LIBS} Python::Python)

target_sources(vbt_core PRIVATE src/vbt/interop/safetensors/safetensors_read.cc src/vbt/interop/safetensors/safetensors_write.cc src/vbt/interop/safetensors/safetensors_file.cc src/vbt/interop/safetensors/vbt_dtype.cc)

if(VBT_WITH_SAFETENSORS)
  target_link_libraries(vbt_core PRIVATE nlohmann_json::nlohmann_json)
endif()

# Propagate CUDA macro and optional CUDA sources
target_compile_definitions(vbt_core PUBLIC VBT_WITH_CUDA=$<IF:$<BOOL:${VBT_USE_CUDA}>,1,0> VBT_REQUIRE_NUMPY_ALIGNMENT=$<BOOL:${VBT_REQUIRE_NUMPY_ALIGNMENT}> VBT_WITH_AUTOGRAD=$<BOOL:${VBT_WITH_AUTOGRAD}> VBT_WITH_DISPATCH_V2=$<BOOL:${VBT_WITH_DISPATCH_V2}> VBT_WITH_SAFETENSORS=$<BOOL:${VBT_WITH_SAFETENSORS}> VBT_SAFETENSORS_ENABLE_MMAP=$<BOOL:${VBT_SAFETENSORS_ENABLE_MMAP}>)
target_sources(vbt_core PRIVATE src/vbt/cuda/device_count.cc src/vbt/cuda/fabric_topology.cc src/vbt/cuda/fabric_state.cc src/vbt/cuda/fabric_events.cc src/vbt/cuda/fabric_addmul_decision.cc src/vbt/rng/cuda_policy.cc src/vbt/autograd/meta.cc src/vbt/autograd/saved_variable.cc src/vbt/autograd/node.cc src/vbt/autograd/engine.cc src/vbt/autograd/forward.cc src/vbt/autograd/wrapper.cc src/vbt/autograd/function.cc src/vbt/autograd/stats.cc src/vbt/autograd/engine_toggles.cc src/vbt/autograd/add_ops.cc)
target_sources(vbt_core PRIVATE src/vbt/cuda/cub_stub.cc)

# CUDA toolchain and linkage (when enabled)
if(VBT_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # CCCL (CUB/Thrust) headers - CUDA 13+ uses cccl/ subdirectory, CUDA 12.x has them at root
  if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/cccl/cub")
    # CUDA 13+ layout - use parent cccl/ dir so <cuda/...>, <cub/...>, <thrust/...> resolve correctly
    target_include_directories(vbt_core SYSTEM PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS}/cccl
    )
    message(STATUS "Using CUDA 13+ CCCL layout: ${CUDAToolkit_INCLUDE_DIRS}/cccl/")
  else()
    # CUDA 12.x layout - CUB/Thrust at include root
    message(STATUS "Using CUDA 12.x layout: ${CUDAToolkit_INCLUDE_DIRS}/")
  endif()

  add_library(vbt_cuda_probe STATIC src/vbt/cuda/build_probe.cu)
  set_target_properties(vbt_cuda_probe PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(vbt_core PUBLIC CUDA::cudart CUDA::cuda_driver)
  set_property(TARGET vbt_core PROPERTY CUDA_ARCHITECTURES native)
  target_sources(vbt_core PRIVATE src/vbt/cuda/stream.cc src/vbt/cuda/guard.cc src/vbt/cuda/event.cc src/vbt/cuda/event_pool.cc src/vbt/cuda/allocator.cc src/vbt/cuda/allocator_async.cc src/vbt/cuda/storage.cc src/vbt/cuda/graphs.cc src/vbt/cuda/cub_wrappers.cu src/vbt/cuda/cub_radix_sort_pairs_u64_f32.cu src/vbt/cuda/cub_radix_sort_pairs_u64_i64.cu src/vbt/cuda/device_caps.cc src/vbt/cuda/reduction_env.cc src/vbt/cuda/reduction_plan.cc src/vbt/cuda/fabric_lifetime.cc src/vbt/ops/cuda_elementwise.cu src/vbt/ops/cuda_reduction.cu src/vbt/ops/embedding_cuda.cu src/vbt/cuda/fabric_elementwise.cu src/vbt/core/tensor_ops_cuda.cu src/vbt/cuda/tensor_iterator/device_meta.cu src/vbt/core/tensor_iter_cuda.cu src/vbt/core/indexing_advanced_cuda.cu src/vbt/rng/cuda_kernels.cu)
  # Deterministic math for RNG CUDA kernels
  set_source_files_properties(src/vbt/rng/cuda_kernels.cu PROPERTIES COMPILE_OPTIONS "--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false")
endif()

# Sanitizers (opt-in)
function(vbt_apply_sanitizers target)
  if(VBT_ENABLE_ASAN)
    target_compile_options(${target} PRIVATE -fsanitize=address)
    target_link_options(${target} PRIVATE -fsanitize=address)
  endif()

  if(VBT_ENABLE_UBSAN)
    target_compile_options(${target} PRIVATE -fsanitize=undefined)
    target_link_options(${target} PRIVATE -fsanitize=undefined)
  endif()

  if(VBT_ENABLE_TSAN)
    target_compile_options(${target} PRIVATE -fsanitize=thread)
    target_link_options(${target} PRIVATE -fsanitize=thread)
  endif()

  if(VBT_ENABLE_ASAN OR VBT_ENABLE_UBSAN OR VBT_ENABLE_TSAN)
    target_compile_options(${target} PRIVATE -fno-omit-frame-pointer)
  endif()
endfunction()

vbt_apply_sanitizers(vbt_core)
target_compile_definitions(vbt_core PUBLIC VBT_INTERNAL_TESTS=$<BOOL:${VBT_BUILD_TESTS}> VBT_AUTOGRAD_TESTING=$<BOOL:${VBT_BUILD_TESTS}>)

# Python extension via nanobind
nanobind_add_module(_C
  src/vbt/python/bindings.cpp
  src/vbt/python/tensor_bindings.cc
  src/vbt/python/dlpack_bindings.cc
  src/vbt/python/ops_bindings.cc
  src/vbt/python/cpu_memory_bindings.cc
  src/vbt/python/pinned_memory_bindings.cc
  src/vbt/python/factory_bindings.cc
  src/vbt/python/autograd_bindings.cc
  src/vbt/python/rng_bindings.cc
  src/vbt/python/fabric_bindings.cc
)
set_target_properties(_C PROPERTIES
  OUTPUT_NAME "_C"
  # Ensure the built extension lives under a vibetensor/ subtree in the
  # build directory so that editable installs report a sensible
  # C.__file__ location (tests rely on "vibetensor" appearing in the
  # path). The install() rule below still controls the final wheel
  # layout.
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/vibetensor"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/vibetensor"
)
target_link_libraries(_C PRIVATE vbt_core)
# Link core library against nanobind runtime so shared autograd helpers
# (e.g., SavedVariable hooks) can safely call into Python.
target_link_libraries(vbt_core PUBLIC nanobind-static)
vbt_apply_sanitizers(_C)
# Expose CUDA macro to bindings
target_compile_definitions(_C PRIVATE VBT_WITH_CUDA=$<IF:$<BOOL:${VBT_USE_CUDA}>,1,0> VBT_REQUIRE_NUMPY_ALIGNMENT=$<BOOL:${VBT_REQUIRE_NUMPY_ALIGNMENT}> VBT_WITH_AUTOGRAD=$<BOOL:${VBT_WITH_AUTOGRAD}>)
# Always include CUDA driver bindings (they provide CPU stubs when CUDA is disabled)
target_sources(_C PRIVATE src/vbt/python/cuda_driver_bindings.cc)
# Add CUDA runtime bindings sources
if(VBT_USE_CUDA)
  target_sources(_C PRIVATE src/vbt/python/cuda_stream_bindings.cc src/vbt/python/cuda_event_bindings.cc src/vbt/python/cuda_memory_bindings.cc src/vbt/python/cuda_copy_bindings.cc src/vbt/python/cuda_graph_bindings.cc src/vbt/python/cuda_reduction_bindings.cc)
endif()

install(TARGETS _C
  LIBRARY DESTINATION vibetensor
  RUNTIME DESTINATION vibetensor
)

add_library(missing_init SHARED plugins/tests/missing_init.c)
target_include_directories(missing_init PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
add_library(bad_abi SHARED plugins/tests/bad_abi.c)
target_include_directories(bad_abi PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
add_library(vbt_reference_add SHARED plugins/reference/vbt_reference_add.c)
target_include_directories(vbt_reference_add PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
# Propagate CUDA compile definition and add CUDA source when enabled
target_compile_definitions(vbt_reference_add PRIVATE VBT_WITH_CUDA=$<IF:$<BOOL:${VBT_USE_CUDA}>,1,0>)
if(VBT_USE_CUDA)
  target_sources(vbt_reference_add PRIVATE plugins/reference/vbt_reference_add_cuda.cu)
  target_link_libraries(vbt_reference_add PRIVATE CUDA::cudart)
  set_property(TARGET vbt_reference_add PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
endif()

add_library(vbt_reference_ti_add SHARED plugins/reference/vbt_reference_ti_add.c)
target_include_directories(vbt_reference_ti_add PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
target_compile_definitions(vbt_reference_ti_add PRIVATE VBT_WITH_CUDA=$<IF:$<BOOL:${VBT_USE_CUDA}>,1,0>)

add_library(vbt_ext_square SHARED plugins/ext_square/vbt_ext_square.c)
target_include_directories(vbt_ext_square PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
add_library(minor0_compat SHARED plugins/tests/minor0_compat.c)
target_include_directories(minor0_compat PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
add_library(ok_null SHARED plugins/tests/ok_null.c)
target_include_directories(ok_null PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
# Test plugin that fails during init to validate rollback behavior
add_library(fail_rollback SHARED plugins/tests/fail_rollback.c)
target_include_directories(fail_rollback PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(p3_fail_init SHARED plugins/tests/p3_fail_init.c)
target_include_directories(p3_fail_init PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(p3_stage_sleep SHARED plugins/tests/p3_stage_sleep.c)
target_include_directories(p3_stage_sleep PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(p3_set_device_policy SHARED plugins/tests/p3_set_device_policy.c)
target_include_directories(p3_set_device_policy PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(p3_dup_def_a SHARED plugins/tests/p3_dup_def_a.c)
target_include_directories(p3_dup_def_a PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
add_library(p3_dup_def_b SHARED plugins/tests/p3_dup_def_b.c)
target_include_directories(p3_dup_def_b PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(set_device_policy_validate SHARED plugins/tests/set_device_policy_validate.c)
target_include_directories(set_device_policy_validate PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(set_device_policy_core_reject SHARED plugins/tests/set_device_policy_core_reject.c)
target_include_directories(set_device_policy_core_reject PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

add_library(fabric_stream_check SHARED plugins/tests/fabric_stream_check.c)
target_include_directories(fabric_stream_check PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)

# CUTLASS Blackwell ring allreduce plugin (best-effort; requires sm103a-capable toolchain)
if(VBT_BUILD_RING_ALLREDUCE_PLUGIN)
  if(NOT VBT_USE_CUDA)
    message(STATUS "VBT_BUILD_RING_ALLREDUCE_PLUGIN=ON but VBT_USE_CUDA=OFF; skipping ring_allreduce plugin")
  else()
    set(_vbt_ring_allreduce_can_build TRUE)

    if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
      message(STATUS "Skipping ring_allreduce plugin: requires CUDA toolkit >= 13.0 for sm103a")
      set(_vbt_ring_allreduce_can_build FALSE)
    endif()

    if(_vbt_ring_allreduce_can_build)
      # Non-fatal probe: some toolchains may not accept 103a even when CUDA is enabled.
      set(_vbt_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
      set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

      set(_vbt_ring_allreduce_probe_dir "${CMAKE_BINARY_DIR}/cmake/vbt_ring_allreduce_probe_sm103a")
      file(MAKE_DIRECTORY "${_vbt_ring_allreduce_probe_dir}")
      file(WRITE "${_vbt_ring_allreduce_probe_dir}/probe.cu" "__global__ void vbt_ring_allreduce_probe_kernel() {}\n")

      try_compile(VBT_RING_ALLREDUCE_SM103A_OK
        "${_vbt_ring_allreduce_probe_dir}/build"
        "${_vbt_ring_allreduce_probe_dir}/probe.cu"
        CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES=103a"
        OUTPUT_VARIABLE _vbt_ring_allreduce_probe_log
      )

      set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_vbt_saved_try_compile_target_type}")
      unset(_vbt_saved_try_compile_target_type)

      if(NOT VBT_RING_ALLREDUCE_SM103A_OK)
        message(STATUS "Skipping ring_allreduce plugin: CUDA toolchain does not accept sm103a (103a)\n${_vbt_ring_allreduce_probe_log}")
        set(_vbt_ring_allreduce_can_build FALSE)
      endif()
    endif()

    if(_vbt_ring_allreduce_can_build)
      add_library(vbt_ring_allreduce SHARED
        ring_allreduce_plugin/vbt_ring_allreduce/vbt_ring_allreduce_plugin.cu
      )
      target_include_directories(vbt_ring_allreduce PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include
        ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
        ${PROJECT_SOURCE_DIR}/ring_allreduce_plugin/94_blackwell_ring_allreduce
      )
      target_link_libraries(vbt_ring_allreduce PRIVATE CUDA::cudart)
      set_property(TARGET vbt_ring_allreduce PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
      set_property(TARGET vbt_ring_allreduce PROPERTY CUDA_ARCHITECTURES 103a)
      # Ensure editable installs build the plugin alongside the Python extension.
      if(TARGET _C)
        add_dependencies(_C vbt_ring_allreduce)
      endif()
    endif()

    unset(_vbt_ring_allreduce_can_build)
    unset(_vbt_ring_allreduce_probe_dir)
    unset(_vbt_ring_allreduce_probe_log)
  endif()
endif()

if(VBT_BUILD_NODE)
  # Best-effort discovery; never REQUIRED. When Node/NPM/headers are missing we
  # simply skip building the addon and its tests.
  find_program(NODE_EXECUTABLE node)
  if(NOT NODE_EXECUTABLE)
    message(STATUS "VBT_BUILD_NODE=ON but 'node' not found; Node overlay build/tests will be skipped.")
  endif()

  find_program(NPM_EXECUTABLE npm)
  if(NOT NPM_EXECUTABLE)
    message(STATUS "VBT_BUILD_NODE=ON but 'npm' not found; Node overlay build/tests will be skipped.")
  endif()

  # Allow both -DNODEJS_INCLUDE_DIR=... (preferred) and -DNODE_INCLUDE_DIR=...
  # as overrides, and fall back to auto-detecting nvm/system installs.
  set(NODEJS_INCLUDE_DIR "" CACHE PATH "Path to Node.js N-API headers (node_api.h)")
  if(NOT NODEJS_INCLUDE_DIR AND DEFINED NODE_INCLUDE_DIR)
    # Support the more generic NODE_INCLUDE_DIR cache variable used in many
    # small N-API examples.
    set(NODEJS_INCLUDE_DIR "${NODE_INCLUDE_DIR}" CACHE PATH "Path to Node.js N-API headers (node_api.h)" FORCE)
  endif()

  # Environment override, e.g. NODE_INCLUDE_DIR=/path/to/include/node
  if(NOT NODEJS_INCLUDE_DIR AND DEFINED ENV{NODE_INCLUDE_DIR})
    set(NODEJS_INCLUDE_DIR "$ENV{NODE_INCLUDE_DIR}" CACHE PATH "Path to Node.js N-API headers (node_api.h)" FORCE)
  endif()

  # If still unset, try to infer headers from the Node executable location.
  # For nvm-managed installs this typically yields:
  #   $NVM_DIR/versions/node/vX.Y.Z/include/node
  if(NOT NODEJS_INCLUDE_DIR AND NODE_EXECUTABLE)
    get_filename_component(_VBT_NODE_BIN_DIR "${NODE_EXECUTABLE}" DIRECTORY)
    get_filename_component(_VBT_NODE_PREFIX "${_VBT_NODE_BIN_DIR}" DIRECTORY)
    set(_VBT_NODE_CANDIDATE "${_VBT_NODE_PREFIX}/include/node")
    if(EXISTS "${_VBT_NODE_CANDIDATE}/node_api.h")
      set(NODEJS_INCLUDE_DIR "${_VBT_NODE_CANDIDATE}" CACHE PATH "Path to Node.js N-API headers (node_api.h)" FORCE)
      message(STATUS "VBT_BUILD_NODE: inferred NODEJS_INCLUDE_DIR='${NODEJS_INCLUDE_DIR}' from '${NODE_EXECUTABLE}'")
    endif()
  endif()

  set(_VBT_NODE_CAN_BUILD TRUE)
  if(NOT NODE_EXECUTABLE OR NOT NPM_EXECUTABLE)
    set(_VBT_NODE_CAN_BUILD FALSE)
  endif()
  if(NOT NODEJS_INCLUDE_DIR OR NOT EXISTS "${NODEJS_INCLUDE_DIR}/node_api.h")
    message(STATUS "VBT_BUILD_NODE=ON but Node.js N-API headers not found (NODEJS_INCLUDE_DIR='${NODEJS_INCLUDE_DIR}'); Node addon build/tests will be skipped.")
    set(_VBT_NODE_CAN_BUILD FALSE)
  endif()

  if(_VBT_NODE_CAN_BUILD)
    add_library(vbt_napi MODULE
      src/vbt/node/addon.cc
      src/vbt/node/tensor.cc
      src/vbt/node/dispatcher.cc
      src/vbt/node/dispatcher_errors.cc
      src/vbt/node/errors.cc
      src/vbt/node/dlpack_napi.cc
      src/vbt/node/cuda_napi.cc
      src/vbt/node/external_memory.cc
      src/vbt/node/logging_napi.cc
      src/vbt/node/fabric_napi.cc
    )

    target_link_libraries(vbt_napi PRIVATE vbt_core)
    target_include_directories(vbt_napi PRIVATE
      ${NODEJS_INCLUDE_DIR}
      ${PROJECT_SOURCE_DIR}/include
    )

    target_compile_definitions(vbt_napi PRIVATE NAPI_VERSION=8)

    # Node addons must not have the "lib" prefix and should live next to the
    # JS overlay sources for local development.
    set_target_properties(vbt_napi PROPERTIES
      PREFIX ""
      SUFFIX ".node"
      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/js/vibetensor"
    )

    vbt_apply_sanitizers(vbt_napi)
  endif()
endif()

if(VBT_BUILD_TESTS)
  target_compile_definitions(vbt_core PRIVATE VBT_TI_ENABLE_TEST_HOOKS=1)
  add_compile_definitions(VBT_TI_ENABLE_TEST_HOOKS=1)
  enable_testing()

  if(VBT_BUILD_NODE)
    if(NODE_EXECUTABLE AND NPM_EXECUTABLE AND TARGET vbt_napi)
      add_custom_target(vbt_node_js ALL
        COMMAND ${CMAKE_COMMAND} -E env NODE_ENV=development
                ${NPM_EXECUTABLE} ci
        COMMAND ${CMAKE_COMMAND} -E env NODE_ENV=development
                ${NPM_EXECUTABLE} run build
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/js/vibetensor"
        COMMENT "Installing and building VibeTensor Node.js overlay (TS → JS)"
      )
      add_dependencies(vbt_node_js vbt_napi)

      add_test(NAME vbt_node_basic
        COMMAND "${NODE_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/js/vibetensor/test/basic_smoke.mjs"
      )
      set_tests_properties(vbt_node_basic PROPERTIES
        ENVIRONMENT "VBT_NODE_ADDON_PATH=$<TARGET_FILE:vbt_napi>"
      )

      add_test(NAME vbt_node_dispatcher_ops
        COMMAND "${NODE_EXECUTABLE}" --test test/dispatcher_ops.mjs
      )
      set_tests_properties(vbt_node_dispatcher_ops PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/js/vibetensor"
        ENVIRONMENT "VBT_NODE_ADDON_PATH=$<TARGET_FILE:vbt_napi>"
      )

      add_test(NAME vbt_node_dlpack_cpu
        COMMAND "${NODE_EXECUTABLE}" --test test/dlpack_cpu.mjs
      )
      set_tests_properties(vbt_node_dlpack_cpu PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/js/vibetensor"
        ENVIRONMENT "VBT_NODE_ADDON_PATH=$<TARGET_FILE:vbt_napi>"
      )

      add_test(NAME vbt_node_cuda_h2d_d2h
        COMMAND "${NODE_EXECUTABLE}" --test test/cuda_h2d_d2h.mjs
      )
      set_tests_properties(vbt_node_cuda_h2d_d2h PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/js/vibetensor"
        ENVIRONMENT "VBT_NODE_ADDON_PATH=$<TARGET_FILE:vbt_napi>"
      )

      add_test(NAME vbt_node_fabric_observability
        COMMAND "${NODE_EXECUTABLE}" --test test/fabric_observability.mjs
      )
      set_tests_properties(vbt_node_fabric_observability PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/js/vibetensor"
        ENVIRONMENT "VBT_NODE_ADDON_PATH=$<TARGET_FILE:vbt_napi>"
      )
    elseif(NODE_EXECUTABLE AND NOT NPM_EXECUTABLE)
      message(STATUS "VBT_BUILD_NODE=ON and VBT_BUILD_TESTS=ON but 'npm' not found; Node overlay build/tests skipped.")
    elseif(NOT NODE_EXECUTABLE)
      message(STATUS "VBT_BUILD_NODE=ON and VBT_BUILD_TESTS=ON but 'node' not found; Node overlay build/tests skipped.")
    elseif(NOT TARGET vbt_napi)
      message(STATUS "VBT_BUILD_NODE=ON and VBT_BUILD_TESTS=ON but vbt_napi was not built (missing Node.js headers?); Node overlay build/tests skipped.")
    endif()
  endif()

  add_executable(vbt_hello_test tests/cpp/hello_test.cc)
  target_link_libraries(vbt_hello_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_hello_test)
  add_test(NAME vbt_hello_test COMMAND "$<TARGET_FILE:vbt_hello_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_hello_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_hello_test>")

  add_executable(vbt_logging_test tests/cpp/logging_test.cc)
  target_link_libraries(vbt_logging_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_logging_test)
  add_test(NAME vbt_logging_test COMMAND "$<TARGET_FILE:vbt_logging_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_logging_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_logging_test>")

  add_executable(vbt_storage_tensor_invariants_test tests/cpp/storage_tensor_invariants_test.cc)
  target_link_libraries(vbt_storage_tensor_invariants_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_storage_tensor_invariants_test)
  add_test(NAME vbt_storage_tensor_invariants_test COMMAND "$<TARGET_FILE:vbt_storage_tensor_invariants_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_storage_tensor_invariants_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_storage_tensor_invariants_test>")

  add_executable(vbt_checked_math_test tests/cpp/checked_math_test.cc)
  target_link_libraries(vbt_checked_math_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_checked_math_test)
  add_test(NAME vbt_checked_math_test COMMAND "$<TARGET_FILE:vbt_checked_math_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_checked_math_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_checked_math_test>")

  if(VBT_WITH_SAFETENSORS)
    add_executable(vbt_safetensors_framing_test tests/cpp/safetensors_framing_test.cc)
    target_link_libraries(vbt_safetensors_framing_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_safetensors_framing_test)
    add_test(NAME vbt_safetensors_framing_test COMMAND "$<TARGET_FILE:vbt_safetensors_framing_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_safetensors_framing_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_safetensors_framing_test>")

    add_executable(vbt_safetensors_validation_test tests/cpp/safetensors_validation_test.cc)
    target_link_libraries(vbt_safetensors_validation_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_safetensors_validation_test)
    add_test(NAME vbt_safetensors_validation_test COMMAND "$<TARGET_FILE:vbt_safetensors_validation_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_safetensors_validation_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_safetensors_validation_test>")

    add_executable(vbt_safetensors_lookup_test tests/cpp/safetensors_lookup_test.cc)
    target_link_libraries(vbt_safetensors_lookup_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_safetensors_lookup_test)
    add_test(NAME vbt_safetensors_lookup_test COMMAND "$<TARGET_FILE:vbt_safetensors_lookup_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_safetensors_lookup_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_safetensors_lookup_test>")

    add_executable(vbt_safetensors_serialize_test tests/cpp/safetensors_serialize_test.cc)
    target_link_libraries(vbt_safetensors_serialize_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_safetensors_serialize_test)
    add_test(NAME vbt_safetensors_serialize_test COMMAND "$<TARGET_FILE:vbt_safetensors_serialize_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_safetensors_serialize_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_safetensors_serialize_test>")

    add_executable(vbt_safetensors_file_io_test tests/cpp/safetensors_file_io_test.cc)
    target_link_libraries(vbt_safetensors_file_io_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_safetensors_file_io_test)
    add_test(NAME vbt_safetensors_file_io_test COMMAND "$<TARGET_FILE:vbt_safetensors_file_io_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_safetensors_file_io_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_safetensors_file_io_test>")

    add_executable(vbt_safetensors_vbt_dtype_test tests/cpp/safetensors_vbt_dtype_test.cc)
    target_link_libraries(vbt_safetensors_vbt_dtype_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_safetensors_vbt_dtype_test)
    add_test(NAME vbt_safetensors_vbt_dtype_test COMMAND "$<TARGET_FILE:vbt_safetensors_vbt_dtype_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_safetensors_vbt_dtype_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_safetensors_vbt_dtype_test>")
  endif()

  add_executable(vbt_node_cuda_runtime_state_test tests/cpp/node_cuda_runtime_state_test.cc)
  target_link_libraries(vbt_node_cuda_runtime_state_test PRIVATE vbt_core GTest::gtest_main)
  target_include_directories(vbt_node_cuda_runtime_state_test PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${NODEJS_INCLUDE_DIR}
  )
  vbt_apply_sanitizers(vbt_node_cuda_runtime_state_test)
  add_test(NAME vbt_node_cuda_runtime_state_test COMMAND "$<TARGET_FILE:vbt_node_cuda_runtime_state_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_node_cuda_runtime_state_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_node_cuda_runtime_state_test>")

  add_executable(vbt_saved_variable_version_check_test tests/cpp/saved_variable_version_check_test.cc)
  target_link_libraries(vbt_saved_variable_version_check_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_saved_variable_version_check_test)
  add_test(NAME vbt_saved_variable_version_check_test COMMAND "$<TARGET_FILE:vbt_saved_variable_version_check_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_saved_variable_version_check_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_saved_variable_version_check_test>")

  add_executable(vbt_autograd_meta_basic_test tests/cpp/autograd_meta_basic_test.cc)
  target_link_libraries(vbt_autograd_meta_basic_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_meta_basic_test)
  add_test(NAME vbt_autograd_meta_basic_test COMMAND "$<TARGET_FILE:vbt_autograd_meta_basic_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_meta_basic_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_meta_basic_test>")

  add_executable(vbt_autograd_accumulategrad_thread_safety_test tests/cpp/autograd_accumulategrad_thread_safety_test.cc)
  target_link_libraries(vbt_autograd_accumulategrad_thread_safety_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_accumulategrad_thread_safety_test)
  add_test(NAME vbt_autograd_accumulategrad_thread_safety_test COMMAND "$<TARGET_FILE:vbt_autograd_accumulategrad_thread_safety_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_accumulategrad_thread_safety_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_accumulategrad_thread_safety_test>")

  add_executable(vbt_autograd_meta_concurrent_hooks_test tests/cpp/autograd_meta_concurrent_hooks_test.cc)
  target_link_libraries(vbt_autograd_meta_concurrent_hooks_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_meta_concurrent_hooks_test)
  add_test(NAME vbt_autograd_meta_concurrent_hooks_test COMMAND "$<TARGET_FILE:vbt_autograd_meta_concurrent_hooks_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_meta_concurrent_hooks_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_meta_concurrent_hooks_test>")

  add_executable(vbt_autograd_concurrent_task_queue_test tests/cpp/autograd_concurrent_task_queue_test.cc)
  target_link_libraries(vbt_autograd_concurrent_task_queue_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_concurrent_task_queue_test)
  add_test(NAME vbt_autograd_concurrent_task_queue_test COMMAND "$<TARGET_FILE:vbt_autograd_concurrent_task_queue_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_concurrent_task_queue_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_concurrent_task_queue_test>")

  add_executable(vbt_autograd_owner_mailbox_test tests/cpp/autograd_owner_mailbox_test.cc)
  target_link_libraries(vbt_autograd_owner_mailbox_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_owner_mailbox_test)
  add_test(NAME vbt_autograd_owner_mailbox_test COMMAND "$<TARGET_FILE:vbt_autograd_owner_mailbox_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_owner_mailbox_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_owner_mailbox_test>")

  add_executable(vbt_autograd_engine_lane_routing_test tests/cpp/autograd_engine_lane_routing_test.cc)
  target_link_libraries(vbt_autograd_engine_lane_routing_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_lane_routing_test)
  add_test(NAME vbt_autograd_engine_lane_routing_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_lane_routing_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_lane_routing_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_lane_routing_test>")

  add_executable(vbt_autograd_tensor_surface_test tests/cpp/autograd_tensor_surface_test.cc)
  target_link_libraries(vbt_autograd_tensor_surface_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_tensor_surface_test)
  add_test(NAME vbt_autograd_tensor_surface_test COMMAND "$<TARGET_FILE:vbt_autograd_tensor_surface_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_tensor_surface_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_tensor_surface_test>")

  add_executable(vbt_engine_duplicate_edge_and_callbacks_test tests/cpp/engine_duplicate_edge_and_callbacks_test.cc)
  target_link_libraries(vbt_engine_duplicate_edge_and_callbacks_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_engine_duplicate_edge_and_callbacks_test)
  add_test(NAME vbt_engine_duplicate_edge_and_callbacks_test COMMAND "$<TARGET_FILE:vbt_engine_duplicate_edge_and_callbacks_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_engine_duplicate_edge_and_callbacks_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_engine_duplicate_edge_and_callbacks_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  add_executable(vbt_engine_negative_test tests/cpp/engine_negative_test.cc)
  target_link_libraries(vbt_engine_negative_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_engine_negative_test)
  add_test(NAME vbt_engine_negative_test COMMAND "$<TARGET_FILE:vbt_engine_negative_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_engine_negative_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_engine_negative_test>")

  add_executable(vbt_autograd_engine_test tests/cpp/autograd_engine_test.cc)
  target_link_libraries(vbt_autograd_engine_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_test)
  add_test(NAME vbt_autograd_engine_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_test>")

  add_executable(vbt_autograd_engine_inputbuffer_test tests/cpp/autograd_engine_inputbuffer_test.cc)
  target_link_libraries(vbt_autograd_engine_inputbuffer_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_inputbuffer_test)
  add_test(NAME vbt_autograd_engine_inputbuffer_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_inputbuffer_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_inputbuffer_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_inputbuffer_test>")

  add_executable(vbt_autograd_engine_stream_metadata_test tests/cpp/autograd_engine_stream_metadata_test.cc)
  target_link_libraries(vbt_autograd_engine_stream_metadata_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_stream_metadata_test)
  add_test(NAME vbt_autograd_engine_stream_metadata_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_stream_metadata_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_stream_metadata_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_stream_metadata_test>")

  add_executable(vbt_autograd_engine_device_toggle_test tests/cpp/autograd_engine_device_toggle_test.cc)
  target_link_libraries(vbt_autograd_engine_device_toggle_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_device_toggle_test)
  add_test(NAME vbt_autograd_engine_device_toggle_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_device_toggle_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_device_toggle_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_device_toggle_test>")

  add_executable(vbt_autograd_engine_device_mode_toggle_test tests/cpp/autograd_engine_device_mode_toggle_test.cc)
  target_link_libraries(vbt_autograd_engine_device_mode_toggle_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_device_mode_toggle_test)
  add_test(NAME vbt_autograd_engine_device_mode_toggle_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_device_mode_toggle_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_device_mode_toggle_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_device_mode_toggle_test>")

  add_executable(vbt_autograd_engine_multidevice_single_gpu_test tests/cpp/autograd_engine_multidevice_single_gpu_test.cc)
  target_link_libraries(vbt_autograd_engine_multidevice_single_gpu_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_multidevice_single_gpu_test)
  add_test(NAME vbt_autograd_engine_multidevice_single_gpu_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_multidevice_single_gpu_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_multidevice_single_gpu_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_multidevice_single_gpu_test>")

  add_executable(vbt_autograd_engine_mt_toggle_thread_safety_test tests/cpp/autograd_engine_mt_toggle_thread_safety_test.cc)
  target_link_libraries(vbt_autograd_engine_mt_toggle_thread_safety_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_mt_toggle_thread_safety_test)
  add_test(NAME vbt_autograd_engine_mt_toggle_thread_safety_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_mt_toggle_thread_safety_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_mt_toggle_thread_safety_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_mt_toggle_thread_safety_test>")

  add_executable(vbt_autograd_engine_deps_single_lane_test tests/cpp/autograd_engine_deps_single_lane_test.cc)
  target_link_libraries(vbt_autograd_engine_deps_single_lane_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_deps_single_lane_test)
  add_test(NAME vbt_autograd_engine_deps_single_lane_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_deps_single_lane_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_deps_single_lane_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_deps_single_lane_test>")

  add_executable(vbt_autograd_engine_cpu_parallelism_test tests/cpp/autograd_engine_cpu_parallelism_test.cc)
  target_link_libraries(vbt_autograd_engine_cpu_parallelism_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_cpu_parallelism_test)
  add_test(NAME vbt_autograd_engine_cpu_parallelism_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_cpu_parallelism_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_cpu_parallelism_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_cpu_parallelism_test>")

  add_executable(vbt_autograd_engine_abort_on_error_test tests/cpp/autograd_engine_abort_on_error_test.cc)
  target_link_libraries(vbt_autograd_engine_abort_on_error_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_abort_on_error_test)
  add_test(NAME vbt_autograd_engine_abort_on_error_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_abort_on_error_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_abort_on_error_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_abort_on_error_test>")

  add_executable(vbt_autograd_engine_gate_raii_test tests/cpp/autograd_engine_gate_raii_test.cc)
  target_link_libraries(vbt_autograd_engine_gate_raii_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_gate_raii_test)
  add_test(NAME vbt_autograd_engine_gate_raii_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_gate_raii_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_gate_raii_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_gate_raii_test>")

  add_executable(vbt_autograd_engine_nested_backward_test tests/cpp/autograd_engine_nested_backward_test.cc)
  target_link_libraries(vbt_autograd_engine_nested_backward_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_nested_backward_test)
  add_test(NAME vbt_autograd_engine_nested_backward_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_nested_backward_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_nested_backward_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_nested_backward_test>")

  add_executable(vbt_autograd_engine_capture_guard_test tests/cpp/autograd_engine_capture_guard_test.cc)
  target_link_libraries(vbt_autograd_engine_capture_guard_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_capture_guard_test)
  add_test(NAME vbt_autograd_engine_capture_guard_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_capture_guard_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_capture_guard_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_capture_guard_test>")

  add_executable(vbt_autograd_engine_guard_lifetime_test tests/cpp/autograd_engine_guard_lifetime_test.cc)
  target_link_libraries(vbt_autograd_engine_guard_lifetime_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_guard_lifetime_test)
  add_test(NAME vbt_autograd_engine_guard_lifetime_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_guard_lifetime_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_guard_lifetime_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_guard_lifetime_test>")

  add_executable(vbt_autograd_engine_cpu_regression_test tests/cpp/autograd_engine_cpu_regression_test.cc)
  target_link_libraries(vbt_autograd_engine_cpu_regression_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_engine_cpu_regression_test)
  add_test(NAME vbt_autograd_engine_cpu_regression_test COMMAND "$<TARGET_FILE:vbt_autograd_engine_cpu_regression_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_engine_cpu_regression_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_engine_cpu_regression_test>")

  add_executable(vbt_autograd_engine_bench tests/cpp/autograd_engine_bench.cc)
  target_link_libraries(vbt_autograd_engine_bench PRIVATE vbt_core)
  vbt_apply_sanitizers(vbt_autograd_engine_bench)

  add_executable(vbt_validate_outputs_parity_test tests/cpp/validate_outputs_parity_test.cc)
  target_link_libraries(vbt_validate_outputs_parity_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_validate_outputs_parity_test)
  add_test(NAME vbt_validate_outputs_parity_test COMMAND "$<TARGET_FILE:vbt_validate_outputs_parity_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_validate_outputs_parity_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_validate_outputs_parity_test>")
  add_executable(vbt_autograd_dispatch_stage_order_test tests/cpp/autograd_dispatch_stage_order_test.cc)
  target_link_libraries(vbt_autograd_dispatch_stage_order_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_dispatch_stage_order_test)
  add_test(NAME vbt_autograd_dispatch_stage_order_test COMMAND "$<TARGET_FILE:vbt_autograd_dispatch_stage_order_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_dispatch_stage_order_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_dispatch_stage_order_test>")

  add_executable(vbt_skip_autograd_guard_raii_test tests/cpp/skip_autograd_guard_raii_test.cc)
  target_link_libraries(vbt_skip_autograd_guard_raii_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_skip_autograd_guard_raii_test)
  add_test(NAME vbt_skip_autograd_guard_raii_test COMMAND "$<TARGET_FILE:vbt_skip_autograd_guard_raii_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_skip_autograd_guard_raii_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_skip_autograd_guard_raii_test>")

  add_executable(vbt_autograd_forward_ad_test tests/cpp/autograd_forward_ad_test.cc)
  target_link_libraries(vbt_autograd_forward_ad_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_autograd_forward_ad_test)
  add_test(NAME vbt_autograd_forward_ad_test COMMAND "$<TARGET_FILE:vbt_autograd_forward_ad_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_autograd_forward_ad_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_autograd_forward_ad_test>")

  add_executable(vbt_contiguity_nod_test tests/cpp/contiguity_nod_test.cc)
  target_link_libraries(vbt_contiguity_nod_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_contiguity_nod_test)
  add_test(NAME vbt_contiguity_nod_test COMMAND "$<TARGET_FILE:vbt_contiguity_nod_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_contiguity_nod_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_contiguity_nod_test>")

  
  add_executable(vbt_clone_invariants_test tests/cpp/clone_invariants_test.cc)
  target_link_libraries(vbt_clone_invariants_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_clone_invariants_test)
  add_test(NAME vbt_clone_invariants_test COMMAND "$<TARGET_FILE:vbt_clone_invariants_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_clone_invariants_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_clone_invariants_test>")

  add_executable(vbt_views_versioning_and_writes_test tests/cpp/views_versioning_and_writes_test.cc)
  target_link_libraries(vbt_views_versioning_and_writes_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_views_versioning_and_writes_test)
  add_test(NAME vbt_views_versioning_and_writes_test COMMAND "$<TARGET_FILE:vbt_views_versioning_and_writes_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_views_versioning_and_writes_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_views_versioning_and_writes_test>")

  add_executable(vbt_indexing_basic_test tests/cpp/indexing_basic_test.cc)
  target_link_libraries(vbt_indexing_basic_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_basic_test)
  add_test(NAME vbt_indexing_basic_test COMMAND "$<TARGET_FILE:vbt_indexing_basic_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_basic_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_basic_test>")

  add_executable(vbt_indexing_advanced_make_test tests/cpp/indexing_advanced_make_advanced_index_test.cc)
  target_link_libraries(vbt_indexing_advanced_make_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_make_test)
  add_test(NAME vbt_indexing_advanced_make_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_make_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_make_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_make_test>")

  add_executable(vbt_indexing_advanced_flag_env_test tests/cpp/indexing_advanced_flag_env_test.cc)
  target_link_libraries(vbt_indexing_advanced_flag_env_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_flag_env_test)
  add_test(NAME vbt_indexing_advanced_flag_env_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_flag_env_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_flag_env_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_flag_env_test>")

  add_executable(vbt_indexing_advanced_cpu_kernel_test tests/cpp/indexing_advanced_cpu_kernel_test.cc)
  target_link_libraries(vbt_indexing_advanced_cpu_kernel_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_cpu_kernel_test)
  add_test(NAME vbt_indexing_advanced_cpu_kernel_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_cpu_kernel_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_cpu_kernel_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_cpu_kernel_test>")

  add_executable(vbt_indexing_advanced_cuda_kernel_test tests/cpp/indexing_advanced_cuda_kernel_test.cc)
  target_link_libraries(vbt_indexing_advanced_cuda_kernel_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_cuda_kernel_test)
  add_test(NAME vbt_indexing_advanced_cuda_kernel_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_cuda_kernel_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_cuda_kernel_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_cuda_kernel_test>")

  add_executable(vbt_indexing_advanced_entrypoints_test tests/cpp/indexing_advanced_entrypoints_test.cc)
  target_link_libraries(vbt_indexing_advanced_entrypoints_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_entrypoints_test)
  add_test(NAME vbt_indexing_advanced_entrypoints_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_entrypoints_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_entrypoints_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_entrypoints_test>")

  add_executable(vbt_indexing_advanced_entrypoints_dispatch_test tests/cpp/indexing_advanced_entrypoints_dispatch_test.cc)
  target_link_libraries(vbt_indexing_advanced_entrypoints_dispatch_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_entrypoints_dispatch_test)
  add_test(NAME vbt_indexing_advanced_entrypoints_dispatch_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_entrypoints_dispatch_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_entrypoints_dispatch_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_entrypoints_dispatch_test>")

  add_executable(vbt_indexing_advanced_entrypoints_autograd_test tests/cpp/indexing_advanced_entrypoints_autograd_test.cc)
  target_link_libraries(vbt_indexing_advanced_entrypoints_autograd_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_indexing_advanced_entrypoints_autograd_test)
  add_test(NAME vbt_indexing_advanced_entrypoints_autograd_test COMMAND "$<TARGET_FILE:vbt_indexing_advanced_entrypoints_autograd_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_indexing_advanced_entrypoints_autograd_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_indexing_advanced_entrypoints_autograd_test>")

  add_executable(vbt_overlap_policy_test tests/cpp/overlap_policy_test.cc)
  target_link_libraries(vbt_overlap_policy_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_overlap_policy_test)
  add_test(NAME vbt_overlap_policy_test COMMAND "$<TARGET_FILE:vbt_overlap_policy_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_overlap_policy_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_overlap_policy_test>")

  add_executable(vbt_overlap_cross_tensor_test tests/cpp/overlap_cross_tensor_test.cc)
  target_link_libraries(vbt_overlap_cross_tensor_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_overlap_cross_tensor_test)
  add_test(NAME vbt_overlap_cross_tensor_test COMMAND "$<TARGET_FILE:vbt_overlap_cross_tensor_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_overlap_cross_tensor_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_overlap_cross_tensor_test>")
  add_executable(vbt_strided_loop_test tests/cpp/strided_loop_test.cc)
  target_link_libraries(vbt_strided_loop_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_strided_loop_test)
  add_test(NAME vbt_strided_loop_test COMMAND "$<TARGET_FILE:vbt_strided_loop_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_strided_loop_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_strided_loop_test>")

  add_executable(vbt_tensor_iter_scaffold_test tests/cpp/tensor_iter_scaffold_test.cc)
  target_link_libraries(vbt_tensor_iter_scaffold_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_scaffold_test)
  add_test(NAME vbt_tensor_iter_scaffold_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_scaffold_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_scaffold_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_scaffold_test>")

  add_executable(vbt_tensor_iter_basic_test tests/cpp/tensor_iter_basic_test.cc)
  target_link_libraries(vbt_tensor_iter_basic_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_basic_test)
  add_test(NAME vbt_tensor_iter_basic_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_basic_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_basic_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_basic_test>")

  add_executable(vbt_tensor_iter_broadcast_test tests/cpp/tensor_iter_broadcast_test.cc)
  target_link_libraries(vbt_tensor_iter_broadcast_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_broadcast_test)
  add_test(NAME vbt_tensor_iter_broadcast_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_broadcast_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_broadcast_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_broadcast_test>")

  add_executable(vbt_tensor_iter_reduction_test tests/cpp/tensor_iter_reduction_test.cc)
  target_link_libraries(vbt_tensor_iter_reduction_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_reduction_test)
  add_test(NAME vbt_tensor_iter_reduction_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_reduction_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_reduction_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_reduction_test>")

  add_executable(vbt_tensor_iter_overlap_test tests/cpp/tensor_iter_overlap_test.cc)
  target_link_libraries(vbt_tensor_iter_overlap_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_overlap_test)
  add_test(NAME vbt_tensor_iter_overlap_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_overlap_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_overlap_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_overlap_test>")

  add_executable(vbt_tensor_iter_cuda_meta_test tests/cpp/tensor_iter_cuda_meta_test.cc)
  target_link_libraries(vbt_tensor_iter_cuda_meta_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_cuda_meta_test)
  add_test(NAME vbt_tensor_iter_cuda_meta_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_cuda_meta_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_cuda_meta_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_cuda_meta_test>")

  add_executable(vbt_tensor_iter_cuda_stream_test tests/cpp/tensor_iter_cuda_stream_test.cc)
  target_link_libraries(vbt_tensor_iter_cuda_stream_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_cuda_stream_test)
  add_test(NAME vbt_tensor_iter_cuda_stream_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_cuda_stream_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_cuda_stream_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_cuda_stream_test>")

  add_executable(vbt_tensor_iter_fabric_test tests/cpp/tensor_iter_fabric_test.cc)
  target_link_libraries(vbt_tensor_iter_fabric_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_fabric_test)
  add_test(NAME vbt_tensor_iter_fabric_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_fabric_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_fabric_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_fabric_test>")

  add_executable(vbt_tensor_iter_test_hooks_test tests/cpp/tensor_iter_test_hooks_test.cc)
  target_link_libraries(vbt_tensor_iter_test_hooks_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_test_hooks_test)
  add_test(NAME vbt_tensor_iter_test_hooks_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_test_hooks_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_test_hooks_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_test_hooks_test>")

  add_executable(vbt_tensor_iter_numel_maxrank_test tests/cpp/tensor_iter_numel_maxrank_test.cc)
  target_link_libraries(vbt_tensor_iter_numel_maxrank_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_numel_maxrank_test)
  add_test(NAME vbt_tensor_iter_numel_maxrank_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_numel_maxrank_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_numel_maxrank_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_numel_maxrank_test>")

  add_executable(vbt_tensor_iter_promotion_test tests/cpp/tensor_iter_promotion_test.cc)
  target_link_libraries(vbt_tensor_iter_promotion_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_promotion_test)
  add_test(NAME vbt_tensor_iter_promotion_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_promotion_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_promotion_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_promotion_test>")

  add_executable(vbt_tensor_iter_indexing32_test tests/cpp/tensor_iter_indexing32_test.cc)
  target_link_libraries(vbt_tensor_iter_indexing32_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_indexing32_test)
  add_test(NAME vbt_tensor_iter_indexing32_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_indexing32_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_indexing32_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_indexing32_test>")

  add_executable(vbt_tensor_iter_resize_outputs_test tests/cpp/tensor_iter_resize_outputs_test.cc)
  target_link_libraries(vbt_tensor_iter_resize_outputs_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_resize_outputs_test)
  add_test(NAME vbt_tensor_iter_resize_outputs_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_resize_outputs_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_resize_outputs_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_resize_outputs_test>")

  add_executable(vbt_tensor_iter_stats_test tests/cpp/tensor_iter_stats_test.cc)
  target_link_libraries(vbt_tensor_iter_stats_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_stats_test)
  add_test(NAME vbt_tensor_iter_stats_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_stats_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_stats_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_stats_test>")

  add_executable(vbt_tensor_iter_coalescing_test tests/cpp/tensor_iter_coalescing_test.cc)
  target_link_libraries(vbt_tensor_iter_coalescing_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_coalescing_test)
  add_test(NAME vbt_tensor_iter_coalescing_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_coalescing_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_coalescing_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_coalescing_test>")

  add_executable(vbt_tensor_iter_fastpath_cpu_test tests/cpp/tensor_iter_fastpath_cpu_test.cc)
  target_link_libraries(vbt_tensor_iter_fastpath_cpu_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_fastpath_cpu_test)
  add_test(NAME vbt_tensor_iter_fastpath_cpu_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_fastpath_cpu_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_fastpath_cpu_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_fastpath_cpu_test>")

  add_executable(vbt_tensor_iter_plugin_cpu_helper_test tests/cpp/tensor_iter_plugin_cpu_helper_test.cc)
  target_link_libraries(vbt_tensor_iter_plugin_cpu_helper_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_cpu_helper_test)
  add_test(NAME vbt_tensor_iter_plugin_cpu_helper_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_cpu_helper_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_cpu_helper_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_cpu_helper_test>")

  add_executable(vbt_tensor_iter_plugin_helpers_abi_layout_test tests/cpp/tensor_iter_plugin_helpers_abi_layout_test.cc)
  target_link_libraries(vbt_tensor_iter_plugin_helpers_abi_layout_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_helpers_abi_layout_test)
  add_test(NAME vbt_tensor_iter_plugin_helpers_abi_layout_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_helpers_abi_layout_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_helpers_abi_layout_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_helpers_abi_layout_test>")

  add_executable(vbt_tensor_iter_plugin_helpers_abi_layout_c_test tests/cpp/tensor_iter_plugin_helpers_abi_layout_c_test.c)
  target_link_libraries(vbt_tensor_iter_plugin_helpers_abi_layout_c_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_helpers_abi_layout_c_test)
  add_test(NAME vbt_tensor_iter_plugin_helpers_abi_layout_c_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_helpers_abi_layout_c_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_helpers_abi_layout_c_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_helpers_abi_layout_c_test>")

  add_executable(vbt_tensor_iter_plugin_handle_abi_layout_test tests/cpp/tensor_iter_plugin_handle_abi_layout_test.cc)
  target_link_libraries(vbt_tensor_iter_plugin_handle_abi_layout_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_handle_abi_layout_test)
  add_test(NAME vbt_tensor_iter_plugin_handle_abi_layout_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_handle_abi_layout_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_handle_abi_layout_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_handle_abi_layout_test>")

  add_executable(vbt_tensor_iter_plugin_handle_abi_layout_c_test tests/cpp/tensor_iter_plugin_handle_abi_layout_c_test.c)
  target_link_libraries(vbt_tensor_iter_plugin_handle_abi_layout_c_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_handle_abi_layout_c_test)
  add_test(NAME vbt_tensor_iter_plugin_handle_abi_layout_c_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_handle_abi_layout_c_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_handle_abi_layout_c_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_handle_abi_layout_c_test>")

  add_executable(vbt_tensor_iter_plugin_handle_test tests/cpp/tensor_iter_plugin_handle_test.cc)
  target_link_libraries(vbt_tensor_iter_plugin_handle_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_handle_test)
  add_test(NAME vbt_tensor_iter_plugin_handle_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_handle_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_handle_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_handle_test>")

  add_executable(vbt_tensor_iter_plugin_ti_reference_test tests/cpp/tensor_iter_plugin_ti_reference_test.cc)
  target_link_libraries(vbt_tensor_iter_plugin_ti_reference_test PRIVATE vbt_core vbt_reference_ti_add GTest::gtest_main)
  target_compile_definitions(vbt_tensor_iter_plugin_ti_reference_test PRIVATE VBT_TI_REF_PLUGIN_PATH="$<TARGET_FILE:vbt_reference_ti_add>")
  vbt_apply_sanitizers(vbt_tensor_iter_plugin_ti_reference_test)
  add_test(NAME vbt_tensor_iter_plugin_ti_reference_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_plugin_ti_reference_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_plugin_ti_reference_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_plugin_ti_reference_test>")

  add_executable(vbt_tensor_iter_doc_sync_test tests/cpp/tensor_iter_doc_sync_test.cc)
  target_link_libraries(vbt_tensor_iter_doc_sync_test PRIVATE vbt_core GTest::gtest_main)
  target_compile_definitions(vbt_tensor_iter_doc_sync_test PRIVATE VBT_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  vbt_apply_sanitizers(vbt_tensor_iter_doc_sync_test)
  add_test(NAME vbt_tensor_iter_doc_sync_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_doc_sync_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_doc_sync_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_doc_sync_test>")

  add_executable(vbt_dispatch_key_set_test tests/cpp/dispatch_key_set_test.cc)
  target_link_libraries(vbt_dispatch_key_set_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_key_set_test)
  add_test(NAME vbt_dispatch_key_set_test COMMAND "$<TARGET_FILE:vbt_dispatch_key_set_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_key_set_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_key_set_test>")

  add_executable(vbt_dispatch_v2_flags_test tests/cpp/dispatch_v2_flags_test.cc)
  target_link_libraries(vbt_dispatch_v2_flags_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_flags_test)
  add_test(NAME vbt_dispatch_v2_flags_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_flags_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_flags_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_flags_test>")

  add_executable(vbt_dispatch_v2_fabric_no_cuda_calls_test tests/cpp/dispatch_v2_fabric_no_cuda_calls_test.cc)
  target_link_libraries(vbt_dispatch_v2_fabric_no_cuda_calls_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_fabric_no_cuda_calls_test)
  add_test(NAME vbt_dispatch_v2_fabric_no_cuda_calls_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_fabric_no_cuda_calls_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_fabric_no_cuda_calls_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_fabric_no_cuda_calls_test>")

  add_executable(vbt_dispatch_v2_snapshot_def_test tests/cpp/dispatch_v2_snapshot_def_test.cc)
  target_link_libraries(vbt_dispatch_v2_snapshot_def_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_snapshot_def_test)
  add_test(NAME vbt_dispatch_v2_snapshot_def_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_snapshot_def_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_snapshot_def_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_snapshot_def_test>")

  add_executable(vbt_dispatch_v2_no_name_allowlists_test tests/cpp/dispatch_v2_no_name_allowlists_test.cc)
  target_link_libraries(vbt_dispatch_v2_no_name_allowlists_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_no_name_allowlists_test)
  add_test(NAME vbt_dispatch_v2_no_name_allowlists_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_no_name_allowlists_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_no_name_allowlists_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_no_name_allowlists_test>")

  add_executable(vbt_dispatch_v2_publish_on_mutation_test tests/cpp/dispatch_v2_publish_on_mutation_test.cc)
  target_link_libraries(vbt_dispatch_v2_publish_on_mutation_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_publish_on_mutation_test)
  add_test(NAME vbt_dispatch_v2_publish_on_mutation_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_publish_on_mutation_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_publish_on_mutation_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_publish_on_mutation_test>")

  add_executable(vbt_dispatch_v2_fabric_mark_invariant_test tests/cpp/dispatch_v2_fabric_mark_invariant_test.cc)
  target_link_libraries(vbt_dispatch_v2_fabric_mark_invariant_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_fabric_mark_invariant_test)
  add_test(NAME vbt_dispatch_v2_fabric_mark_invariant_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_fabric_mark_invariant_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_fabric_mark_invariant_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_fabric_mark_invariant_test>")

  add_executable(vbt_dispatch_v2_device_policy_setter_test tests/cpp/dispatch_v2_device_policy_setter_test.cc)
  target_link_libraries(vbt_dispatch_v2_device_policy_setter_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_device_policy_setter_test)
  add_test(NAME vbt_dispatch_v2_device_policy_setter_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_device_policy_setter_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_device_policy_setter_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_device_policy_setter_test>")

  add_executable(vbt_dispatch_v2_validator_test tests/cpp/dispatch_v2_validator_test.cc)
  target_link_libraries(vbt_dispatch_v2_validator_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatch_v2_validator_test)
  add_test(NAME vbt_dispatch_v2_validator_test COMMAND "$<TARGET_FILE:vbt_dispatch_v2_validator_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatch_v2_validator_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatch_v2_validator_test>")

  add_executable(vbt_dispatcher_cpu_test tests/cpp/dispatcher_cpu_test.cc)
  target_link_libraries(vbt_dispatcher_cpu_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dispatcher_cpu_test)
  add_test(NAME vbt_dispatcher_cpu_test COMMAND "$<TARGET_FILE:vbt_dispatcher_cpu_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dispatcher_cpu_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatcher_cpu_test>")
  add_executable(vbt_cpu_allocator_tls_coalesce_safety_test tests/cpp/cpu_allocator_tls_coalesce_safety_test.cc)
  target_link_libraries(vbt_cpu_allocator_tls_coalesce_safety_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_cpu_allocator_tls_coalesce_safety_test)
  add_test(NAME vbt_cpu_allocator_tls_coalesce_safety_test COMMAND "$<TARGET_FILE:vbt_cpu_allocator_tls_coalesce_safety_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_cpu_allocator_tls_coalesce_safety_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cpu_allocator_tls_coalesce_safety_test>")
  add_executable(vbt_cpu_allocator_noncaching_reserved_test tests/cpp/cpu_allocator_noncaching_reserved_test.cc)
  target_link_libraries(vbt_cpu_allocator_noncaching_reserved_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_cpu_allocator_noncaching_reserved_test)
  add_test(NAME vbt_cpu_allocator_noncaching_reserved_test COMMAND "$<TARGET_FILE:vbt_cpu_allocator_noncaching_reserved_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_cpu_allocator_noncaching_reserved_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cpu_allocator_noncaching_reserved_test>")

  # FP16/BF16 dtype + DLPack mapping tests
  add_executable(vbt_dtype_dlpack_fp16_bf16_test tests/cpp/dtype_dlpack_fp16_bf16_test.cc)
  target_link_libraries(vbt_dtype_dlpack_fp16_bf16_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dtype_dlpack_fp16_bf16_test)
  add_test(NAME vbt_dtype_dlpack_fp16_bf16_test COMMAND "$<TARGET_FILE:vbt_dtype_dlpack_fp16_bf16_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dtype_dlpack_fp16_bf16_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dtype_dlpack_fp16_bf16_test>")

  # Float64/Complex dtype + DLPack mapping tests
  add_executable(vbt_dtype_complex_test tests/cpp/dtype_complex_test.cc)
  target_link_libraries(vbt_dtype_complex_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_dtype_complex_test)
  add_test(NAME vbt_dtype_complex_test COMMAND "$<TARGET_FILE:vbt_dtype_complex_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_dtype_complex_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dtype_complex_test>")

  add_executable(vbt_plugin_loader_abi_test tests/cpp/vbt_plugin_loader_abi_test.cc)
  target_link_libraries(vbt_plugin_loader_abi_test PRIVATE vbt_core GTest::gtest_main)
  target_compile_definitions(vbt_plugin_loader_abi_test PRIVATE PLUGIN_MISSING_INIT_PATH="$<TARGET_FILE:missing_init>" PLUGIN_BAD_ABI_PATH="$<TARGET_FILE:bad_abi>")
  vbt_apply_sanitizers(vbt_plugin_loader_abi_test)
  add_test(NAME vbt_plugin_loader_abi_test COMMAND "$<TARGET_FILE:vbt_plugin_loader_abi_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_loader_abi_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_loader_abi_test>")

  add_executable(vbt_plugin_init_failure_no_dlclose_test tests/cpp/plugin_init_failure_no_dlclose_test.cc)
  target_link_libraries(vbt_plugin_init_failure_no_dlclose_test PRIVATE vbt_core GTest::gtest_main)
  target_compile_definitions(vbt_plugin_init_failure_no_dlclose_test PRIVATE PLUGIN_FAIL_ROLLBACK_PATH="$<TARGET_FILE:fail_rollback>")
  vbt_apply_sanitizers(vbt_plugin_init_failure_no_dlclose_test)
  add_test(NAME vbt_plugin_init_failure_no_dlclose_test COMMAND "$<TARGET_FILE:vbt_plugin_init_failure_no_dlclose_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_init_failure_no_dlclose_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_init_failure_no_dlclose_test>")

  add_executable(vbt_plugin_atomic_commit_stage_only_test tests/cpp/plugin_atomic_commit_stage_only_test.cc)
  target_link_libraries(vbt_plugin_atomic_commit_stage_only_test PRIVATE vbt_core GTest::gtest_main)
  add_dependencies(vbt_plugin_atomic_commit_stage_only_test p3_stage_sleep)
  target_compile_definitions(vbt_plugin_atomic_commit_stage_only_test PRIVATE PLUGIN_P3_STAGE_SLEEP_PATH="$<TARGET_FILE:p3_stage_sleep>")
  vbt_apply_sanitizers(vbt_plugin_atomic_commit_stage_only_test)
  add_test(NAME vbt_plugin_atomic_commit_stage_only_test COMMAND "$<TARGET_FILE:vbt_plugin_atomic_commit_stage_only_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_atomic_commit_stage_only_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_atomic_commit_stage_only_test>")

  add_executable(vbt_plugin_atomic_commit_fail_init_test tests/cpp/plugin_atomic_commit_fail_init_test.cc)
  target_link_libraries(vbt_plugin_atomic_commit_fail_init_test PRIVATE vbt_core GTest::gtest_main)
  add_dependencies(vbt_plugin_atomic_commit_fail_init_test p3_fail_init)
  target_compile_definitions(vbt_plugin_atomic_commit_fail_init_test PRIVATE PLUGIN_P3_FAIL_INIT_PATH="$<TARGET_FILE:p3_fail_init>")
  vbt_apply_sanitizers(vbt_plugin_atomic_commit_fail_init_test)
  add_test(NAME vbt_plugin_atomic_commit_fail_init_test COMMAND "$<TARGET_FILE:vbt_plugin_atomic_commit_fail_init_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_atomic_commit_fail_init_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_atomic_commit_fail_init_test>")

  add_executable(vbt_plugin_atomic_commit_set_device_policy_test tests/cpp/plugin_atomic_commit_set_device_policy_test.cc)
  target_link_libraries(vbt_plugin_atomic_commit_set_device_policy_test PRIVATE vbt_core GTest::gtest_main)
  add_dependencies(vbt_plugin_atomic_commit_set_device_policy_test p3_set_device_policy set_device_policy_core_reject)
  target_compile_definitions(vbt_plugin_atomic_commit_set_device_policy_test PRIVATE
    PLUGIN_P3_SET_DEVICE_POLICY_PATH="$<TARGET_FILE:p3_set_device_policy>"
    PLUGIN_SET_DEVICE_POLICY_CORE_REJECT_PATH="$<TARGET_FILE:set_device_policy_core_reject>")
  vbt_apply_sanitizers(vbt_plugin_atomic_commit_set_device_policy_test)
  add_test(NAME vbt_plugin_atomic_commit_set_device_policy_test COMMAND "$<TARGET_FILE:vbt_plugin_atomic_commit_set_device_policy_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_atomic_commit_set_device_policy_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_atomic_commit_set_device_policy_test>")

  add_executable(vbt_plugin_atomic_commit_determinism_concurrency_test
    tests/cpp/plugin_atomic_commit_determinism_concurrency_test.cc)
  target_link_libraries(vbt_plugin_atomic_commit_determinism_concurrency_test PRIVATE vbt_core GTest::gtest_main)
  add_dependencies(vbt_plugin_atomic_commit_determinism_concurrency_test
    p3_dup_def_a p3_dup_def_b p3_stage_sleep)
  target_compile_definitions(vbt_plugin_atomic_commit_determinism_concurrency_test PRIVATE
    PLUGIN_P3_DUP_DEF_A_PATH="$<TARGET_FILE:p3_dup_def_a>"
    PLUGIN_P3_DUP_DEF_B_PATH="$<TARGET_FILE:p3_dup_def_b>"
    PLUGIN_P3_STAGE_SLEEP_PATH="$<TARGET_FILE:p3_stage_sleep>")
  vbt_apply_sanitizers(vbt_plugin_atomic_commit_determinism_concurrency_test)
  add_test(NAME vbt_plugin_atomic_commit_determinism_concurrency_test
    COMMAND "$<TARGET_FILE:vbt_plugin_atomic_commit_determinism_concurrency_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_atomic_commit_determinism_concurrency_test PROPERTIES
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_atomic_commit_determinism_concurrency_test>")

  add_executable(vbt_plugin_device_policy_and_stream_test tests/cpp/plugin_device_policy_and_stream_test.cc)
  target_link_libraries(vbt_plugin_device_policy_and_stream_test PRIVATE vbt_core GTest::gtest_main)
  add_dependencies(vbt_plugin_device_policy_and_stream_test set_device_policy_validate fabric_stream_check)
  target_compile_definitions(vbt_plugin_device_policy_and_stream_test PRIVATE
    PLUGIN_SET_DEVICE_POLICY_VALIDATE_PATH="$<TARGET_FILE:set_device_policy_validate>"
    PLUGIN_FABRIC_STREAM_CHECK_PATH="$<TARGET_FILE:fabric_stream_check>")
  vbt_apply_sanitizers(vbt_plugin_device_policy_and_stream_test)
  add_test(NAME vbt_plugin_device_policy_and_stream_test COMMAND "$<TARGET_FILE:vbt_plugin_device_policy_and_stream_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_plugin_device_policy_and_stream_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_plugin_device_policy_and_stream_test>")

  add_executable(vbt_tensor_iter_memory_format_test tests/cpp/tensor_iter_memory_format_test.cc)
  target_link_libraries(vbt_tensor_iter_memory_format_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_memory_format_test)
  add_test(NAME vbt_tensor_iter_memory_format_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_memory_format_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_memory_format_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_memory_format_test>")

  add_executable(vbt_tensor_iter_loops_test tests/cpp/tensor_iter_loops_test.cc)
  target_link_libraries(vbt_tensor_iter_loops_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_loops_test)
  add_test(NAME vbt_tensor_iter_loops_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_loops_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_loops_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_loops_test>")

  add_executable(vbt_tensor_iter_reduction_multidim_test tests/cpp/tensor_iter_reduction_multidim_test.cc)
  target_link_libraries(vbt_tensor_iter_reduction_multidim_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_reduction_multidim_test)
  add_test(NAME vbt_tensor_iter_reduction_multidim_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_reduction_multidim_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_reduction_multidim_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_reduction_multidim_test>")

  add_executable(vbt_tensor_iter_opmath_test tests/cpp/tensor_iter_opmath_test.cc)
  target_link_libraries(vbt_tensor_iter_opmath_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_opmath_test)
  add_test(NAME vbt_tensor_iter_opmath_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_opmath_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_opmath_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_opmath_test>")

  add_executable(vbt_tensor_iter_static_declarations_test tests/cpp/tensor_iter_static_declarations_test.cc)
  target_link_libraries(vbt_tensor_iter_static_declarations_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_tensor_iter_static_declarations_test)
  add_test(NAME vbt_tensor_iter_static_declarations_test COMMAND "$<TARGET_FILE:vbt_tensor_iter_static_declarations_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_tensor_iter_static_declarations_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_tensor_iter_static_declarations_test>")

  add_executable(vbt_ops_comparison_test tests/cpp/ops_comparison_test.cc)
  target_link_libraries(vbt_ops_comparison_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_ops_comparison_test)
  add_test(NAME vbt_ops_comparison_test COMMAND "$<TARGET_FILE:vbt_ops_comparison_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_ops_comparison_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_ops_comparison_test>")


  add_executable(vbt_ops_arithmetic_test tests/cpp/ops_arithmetic_test.cc)
  target_link_libraries(vbt_ops_arithmetic_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_ops_arithmetic_test)
  add_test(NAME vbt_ops_arithmetic_test COMMAND "$<TARGET_FILE:vbt_ops_arithmetic_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_ops_arithmetic_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_ops_arithmetic_test>")

  add_executable(vbt_cuda_fabric_test tests/cpp/cuda_fabric_test.cc)
  target_link_libraries(vbt_cuda_fabric_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_cuda_fabric_test)
  add_test(NAME vbt_cuda_fabric_test COMMAND "$<TARGET_FILE:vbt_cuda_fabric_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_cuda_fabric_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_test>")

  if(VBT_USE_CUDA)
    add_executable(vbt_cuda_event_test tests/cpp/cuda_event_test.cc)
    target_link_libraries(vbt_cuda_event_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_event_test)
    add_test(NAME vbt_cuda_event_test COMMAND "$<TARGET_FILE:vbt_cuda_event_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_event_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_event_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_stream_test tests/cpp/cuda_stream_test.cc)
    target_link_libraries(vbt_cuda_stream_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_stream_test)
    add_test(NAME vbt_cuda_stream_test COMMAND "$<TARGET_FILE:vbt_cuda_stream_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_stream_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_stream_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_device_count_test tests/cpp/cuda_device_count_test.cc)
    target_link_libraries(vbt_cuda_device_count_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_device_count_test)
    add_test(NAME vbt_cuda_device_count_test COMMAND "$<TARGET_FILE:vbt_cuda_device_count_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_device_count_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_device_count_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_tensor_device_and_data_test tests/cpp/cuda_tensor_device_and_data_test.cc)
    target_link_libraries(vbt_cuda_tensor_device_and_data_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_tensor_device_and_data_test)
    add_test(NAME vbt_cuda_tensor_device_and_data_test COMMAND "$<TARGET_FILE:vbt_cuda_tensor_device_and_data_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_tensor_device_and_data_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_tensor_device_and_data_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_elementwise_test tests/cpp/cuda_elementwise_test.cc)
    target_link_libraries(vbt_cuda_elementwise_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_elementwise_test)
    add_test(NAME vbt_cuda_elementwise_test COMMAND "$<TARGET_FILE:vbt_cuda_elementwise_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_elementwise_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_elementwise_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_elementwise_stream_safety_test tests/cpp/cuda_elementwise_stream_safety_test.cc)
    target_link_libraries(vbt_cuda_elementwise_stream_safety_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_elementwise_stream_safety_test)
    add_test(NAME vbt_cuda_elementwise_stream_safety_test COMMAND "$<TARGET_FILE:vbt_cuda_elementwise_stream_safety_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_elementwise_stream_safety_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_elementwise_stream_safety_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_multigpu_smoke_test tests/cpp/cuda_multigpu_smoke_test.cc)
    target_link_libraries(vbt_cuda_multigpu_smoke_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_multigpu_smoke_test)
    add_test(NAME vbt_cuda_multigpu_smoke_test COMMAND "$<TARGET_FILE:vbt_cuda_multigpu_smoke_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_multigpu_smoke_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_multigpu_smoke_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_fabric_manual_p2p_test
      tests/cpp/cuda_fabric_manual_p2p_test.cc)

    # This test includes a small CUDA kernel. Compile the TU with NVCC even though
    # it uses a .cc extension to keep the file map stable.
    set_property(SOURCE tests/cpp/cuda_fabric_manual_p2p_test.cc PROPERTY LANGUAGE CUDA)

    target_link_libraries(vbt_cuda_fabric_manual_p2p_test
      PRIVATE vbt_core GTest::gtest_main)

    vbt_apply_sanitizers(vbt_cuda_fabric_manual_p2p_test)

    add_test(
      NAME vbt_cuda_fabric_manual_p2p_native_test
      COMMAND "$<TARGET_FILE:vbt_cuda_fabric_manual_p2p_test>" --gtest_also_run_disabled_tests)

    set_tests_properties(vbt_cuda_fabric_manual_p2p_native_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_manual_p2p_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=backend=native"
      LABELS "cuda;fabric;allocator")

    add_test(
      NAME vbt_cuda_fabric_manual_p2p_async_test
      COMMAND "$<TARGET_FILE:vbt_cuda_fabric_manual_p2p_test>" --gtest_also_run_disabled_tests)

    set_tests_properties(vbt_cuda_fabric_manual_p2p_async_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_manual_p2p_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=backend=cudamallocasync"
      LABELS "cuda;fabric;allocator")

    add_executable(vbt_cuda_fabric_decide_test
      tests/cpp/cuda_fabric_decide_test.cc)
    target_link_libraries(vbt_cuda_fabric_decide_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_fabric_decide_test)
    add_test(NAME vbt_cuda_fabric_decide_test COMMAND "$<TARGET_FILE:vbt_cuda_fabric_decide_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_fabric_decide_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_decide_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;fabric")

    add_executable(vbt_cuda_fabric_stats_v2_test
      tests/cpp/cuda_fabric_stats_v2_test.cc)
    target_link_libraries(vbt_cuda_fabric_stats_v2_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_fabric_stats_v2_test)
    add_test(NAME vbt_cuda_fabric_stats_v2_test COMMAND "$<TARGET_FILE:vbt_cuda_fabric_stats_v2_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_fabric_stats_v2_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_stats_v2_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;fabric")

    add_executable(vbt_cuda_fabric_events_ring_test
      tests/cpp/cuda_fabric_events_ring_test.cc)
    target_link_libraries(vbt_cuda_fabric_events_ring_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_fabric_events_ring_test)
    add_test(NAME vbt_cuda_fabric_events_ring_test COMMAND "$<TARGET_FILE:vbt_cuda_fabric_events_ring_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_fabric_events_ring_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_events_ring_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;fabric;events")

    add_executable(vbt_cuda_fabric_correctness_test
      tests/cpp/cuda_fabric_correctness_test.cc)
    target_link_libraries(vbt_cuda_fabric_correctness_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_fabric_correctness_test)
    add_test(NAME vbt_cuda_fabric_correctness_test COMMAND "$<TARGET_FILE:vbt_cuda_fabric_correctness_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_fabric_correctness_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_correctness_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;fabric")

  add_executable(vbt_cuda_fabric_lifetime_test
    tests/cpp/cuda_fabric_lifetime_test.cc)
  target_link_libraries(vbt_cuda_fabric_lifetime_test PRIVATE vbt_core GTest::gtest_main)
  vbt_apply_sanitizers(vbt_cuda_fabric_lifetime_test)
  add_test(NAME vbt_cuda_fabric_lifetime_test COMMAND "$<TARGET_FILE:vbt_cuda_fabric_lifetime_test>" --gtest_also_run_disabled_tests)
  set_tests_properties(vbt_cuda_fabric_lifetime_test PROPERTIES
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_fabric_lifetime_test>"
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
    LABELS "cuda;fabric")

    add_executable(vbt_cuda_event_pool_test tests/cpp/cuda_event_pool_test.cc)
    target_link_libraries(vbt_cuda_event_pool_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_event_pool_test)
    add_test(NAME vbt_cuda_event_pool_test COMMAND "$<TARGET_FILE:vbt_cuda_event_pool_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_event_pool_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_event_pool_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_cub_invalid_stream_device_test
      tests/cpp/cuda_cub_invalid_stream_device_test.cc)
    target_link_libraries(vbt_cuda_cub_invalid_stream_device_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_cub_invalid_stream_device_test)
    add_test(NAME vbt_cuda_cub_invalid_stream_device_test COMMAND "$<TARGET_FILE:vbt_cuda_cub_invalid_stream_device_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_cub_invalid_stream_device_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_cub_invalid_stream_device_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;cub")

    add_executable(vbt_cuda_cub_temp_zero_test
      tests/cpp/cuda_cub_temp_zero_test.cc)
    target_link_libraries(vbt_cuda_cub_temp_zero_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_cub_temp_zero_test)
    add_test(NAME vbt_cuda_cub_temp_zero_test COMMAND "$<TARGET_FILE:vbt_cuda_cub_temp_zero_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_cub_temp_zero_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_cub_temp_zero_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;cub")

    add_executable(vbt_cuda_cub_bool_mask_wrappers_test
      tests/cpp/cuda_cub_bool_mask_wrappers_test.cc)
    target_link_libraries(vbt_cuda_cub_bool_mask_wrappers_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_cub_bool_mask_wrappers_test)
    add_test(NAME vbt_cuda_cub_bool_mask_wrappers_test COMMAND "$<TARGET_FILE:vbt_cuda_cub_bool_mask_wrappers_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_cub_bool_mask_wrappers_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_cub_bool_mask_wrappers_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;cub")

    add_executable(vbt_cuda_cub_reduce_all_sum_test
      tests/cpp/cuda_cub_reduce_all_sum_test.cc)
    target_link_libraries(vbt_cuda_cub_reduce_all_sum_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_cub_reduce_all_sum_test)
    add_test(NAME vbt_cuda_cub_reduce_all_sum_test COMMAND "$<TARGET_FILE:vbt_cuda_cub_reduce_all_sum_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_cub_reduce_all_sum_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_cub_reduce_all_sum_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;cub")

    add_executable(vbt_cuda_reduction_slice_len_overflow_stats_test
      tests/cpp/cuda_reduction_slice_len_overflow_stats_test.cc)
    target_link_libraries(vbt_cuda_reduction_slice_len_overflow_stats_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_reduction_slice_len_overflow_stats_test)
    add_test(NAME vbt_cuda_reduction_slice_len_overflow_stats_test COMMAND "$<TARGET_FILE:vbt_cuda_reduction_slice_len_overflow_stats_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_reduction_slice_len_overflow_stats_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_reduction_slice_len_overflow_stats_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;reduction")

    add_executable(vbt_cuda_reduction_plan_noalloc_test
      tests/cpp/cuda_reduction_plan_noalloc_test.cc)
    target_link_libraries(vbt_cuda_reduction_plan_noalloc_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_reduction_plan_noalloc_test)
    add_test(NAME vbt_cuda_reduction_plan_noalloc_test COMMAND "$<TARGET_FILE:vbt_cuda_reduction_plan_noalloc_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_reduction_plan_noalloc_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_reduction_plan_noalloc_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;reduction")

    add_executable(vbt_cuda_reduction_k2multi_workspace_layout_test
      tests/cpp/cuda_reduction_k2multi_workspace_layout_test.cc)
    target_link_libraries(vbt_cuda_reduction_k2multi_workspace_layout_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_reduction_k2multi_workspace_layout_test)
    add_test(NAME vbt_cuda_reduction_k2multi_workspace_layout_test COMMAND "$<TARGET_FILE:vbt_cuda_reduction_k2multi_workspace_layout_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_reduction_k2multi_workspace_layout_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_reduction_k2multi_workspace_layout_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;reduction")

    add_executable(vbt_cuda_reduction_record_stream_hazard_test
      tests/cpp/cuda_reduction_record_stream_hazard_test.cc)
    target_link_libraries(vbt_cuda_reduction_record_stream_hazard_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_reduction_record_stream_hazard_test)
    add_test(NAME vbt_cuda_reduction_record_stream_hazard_test COMMAND "$<TARGET_FILE:vbt_cuda_reduction_record_stream_hazard_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_reduction_record_stream_hazard_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_reduction_record_stream_hazard_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;reduction")

    add_executable(vbt_cuda_cub_scan_test
      tests/cpp/cuda_cub_scan_test.cc)
    target_link_libraries(vbt_cuda_cub_scan_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_cub_scan_test)
    add_test(NAME vbt_cuda_cub_scan_test COMMAND "$<TARGET_FILE:vbt_cuda_cub_scan_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_cub_scan_test PROPERTIES
      WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_cub_scan_test>"
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0"
      LABELS "cuda;cub")

    if (EXISTS "${PROJECT_SOURCE_DIR}/tools/check_cub_include_allowlist.py")
      add_test(NAME vbt_cub_include_allowlist_test
        COMMAND "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/check_cub_include_allowlist.py")
      set_tests_properties(vbt_cub_include_allowlist_test PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        LABELS "cub")
    else()
      message(STATUS "Skipping vbt_cub_include_allowlist_test: tools/check_cub_include_allowlist.py not found")
    endif()

    add_executable(vbt_cuda_allocator_skeleton_test tests/cpp/cuda_allocator_skeleton_test.cc)
    target_link_libraries(vbt_cuda_allocator_skeleton_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_skeleton_test)
    add_test(NAME vbt_cuda_allocator_skeleton_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_skeleton_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_skeleton_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_skeleton_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_rounding_test tests/cpp/cuda_allocator_rounding_test.cc)
    target_link_libraries(vbt_cuda_allocator_rounding_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_rounding_test)
    add_test(NAME vbt_cuda_allocator_rounding_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_rounding_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_rounding_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_rounding_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_memory_snapshot_test tests/cpp/cuda_allocator_memory_snapshot_test.cc)
    target_link_libraries(vbt_cuda_allocator_memory_snapshot_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_memory_snapshot_test)
    add_test(NAME vbt_cuda_allocator_memory_snapshot_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_memory_snapshot_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_memory_snapshot_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_memory_snapshot_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_cross_stream_test tests/cpp/cuda_allocator_cross_stream_test.cc)
    target_link_libraries(vbt_cuda_allocator_cross_stream_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_cross_stream_test)
    add_test(NAME vbt_cuda_allocator_cross_stream_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_cross_stream_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_cross_stream_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_cross_stream_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=enable_cross_stream_fallback=0")

    add_test(NAME vbt_cuda_allocator_cross_stream_fallback_enabled_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_cross_stream_test>" --gtest_filter=CudaAllocatorCrossStreamTest.CrossStreamFallbackRespectsEnvToggle --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_cross_stream_fallback_enabled_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_cross_stream_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=enable_cross_stream_fallback=1")

    add_executable(vbt_cuda_allocator_coalesce_test tests/cpp/cuda_allocator_coalesce_test.cc)
    target_link_libraries(vbt_cuda_allocator_coalesce_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_coalesce_test)
    add_test(NAME vbt_cuda_allocator_coalesce_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_coalesce_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_coalesce_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_coalesce_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=enable_block_splitting=1")

    add_test(NAME vbt_cuda_allocator_coalesce_gateoff_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_coalesce_test>" --gtest_filter=CudaAllocatorCoalesceTest.GateOffIdentityBehavior --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_coalesce_gateoff_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_coalesce_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=enable_block_splitting=0")

    add_executable(vbt_cuda_allocator_stats_invariants_test tests/cpp/cuda_allocator_stats_invariants_test.cc)
    target_link_libraries(vbt_cuda_allocator_stats_invariants_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_stats_invariants_test)
    add_test(NAME vbt_cuda_allocator_stats_invariants_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_stats_invariants_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_stats_invariants_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_stats_invariants_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=enable_block_splitting=1")

    add_executable(vbt_cuda_allocator_fragmentation_fuzzer_test tests/cpp/cuda_allocator_fragmentation_fuzzer_test.cc)
    target_link_libraries(vbt_cuda_allocator_fragmentation_fuzzer_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_fragmentation_fuzzer_test)
    add_test(NAME vbt_cuda_allocator_fragmentation_fuzzer_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fragmentation_fuzzer_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fragmentation_fuzzer_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fragmentation_fuzzer_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=enable_block_splitting=1")

    add_executable(vbt_cuda_allocator_record_stream_test tests/cpp/cuda_allocator_record_stream_test.cc)
    target_link_libraries(vbt_cuda_allocator_record_stream_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_record_stream_test)
    add_test(NAME vbt_cuda_allocator_record_stream_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_record_stream_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_record_stream_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_record_stream_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_fraction_env_and_api_test tests/cpp/cuda_allocator_fraction_env_and_api_test.cc)
    target_link_libraries(vbt_cuda_allocator_fraction_env_and_api_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_fraction_env_and_api_test)
    add_test(NAME vbt_cuda_allocator_fraction_env_default_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fraction_env_and_api_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fraction_env_default_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fraction_env_and_api_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=")

    add_test(NAME vbt_cuda_allocator_fraction_env_valid_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fraction_env_and_api_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fraction_env_valid_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fraction_env_and_api_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=per_process_memory_fraction=0.25")

    add_test(NAME vbt_cuda_allocator_fraction_env_invalid_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fraction_env_and_api_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fraction_env_invalid_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fraction_env_and_api_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=per_process_memory_fraction=invalid")

    add_executable(vbt_cuda_allocator_fraction_async_integration_test tests/cpp/cuda_allocator_fraction_async_integration_test.cc)
    target_link_libraries(vbt_cuda_allocator_fraction_async_integration_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_fraction_async_integration_test)
    add_test(NAME vbt_cuda_allocator_fraction_async_integration_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fraction_async_integration_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fraction_async_integration_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fraction_async_integration_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;VBT_CUDA_ALLOC_CONF=backend=cudamallocasync,per_process_memory_fraction=0.5")

    add_executable(vbt_cuda_allocator_gc_age_test tests/cpp/cuda_allocator_gc_age_test.cc)
    target_link_libraries(vbt_cuda_allocator_gc_age_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_gc_age_test)
    add_test(NAME vbt_cuda_allocator_gc_age_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_gc_age_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_gc_age_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_gc_age_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_fraction_cap_scaffolding_test tests/cpp/cuda_allocator_fraction_cap_scaffolding_test.cc)
    target_link_libraries(vbt_cuda_allocator_fraction_cap_scaffolding_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_fraction_cap_scaffolding_test)
    add_test(NAME vbt_cuda_allocator_fraction_cap_scaffolding_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fraction_cap_scaffolding_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fraction_cap_scaffolding_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fraction_cap_scaffolding_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_cap_exceeded_test tests/cpp/cuda_allocator_cap_exceeded_test.cc)
    target_link_libraries(vbt_cuda_allocator_cap_exceeded_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_cap_exceeded_test)
    add_test(NAME vbt_cuda_allocator_cap_exceeded_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_cap_exceeded_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_cap_exceeded_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_cap_exceeded_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_allocator_fraction_gate_test tests/cpp/cuda_allocator_fraction_gate_test.cc)
    target_link_libraries(vbt_cuda_allocator_fraction_gate_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_fraction_gate_test)
    add_test(NAME vbt_cuda_allocator_fraction_gate_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_fraction_gate_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_fraction_gate_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_fraction_gate_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_dispatcher_cuda_selection_test tests/cpp/dispatcher_cuda_selection_test.cc)
    target_link_libraries(vbt_dispatcher_cuda_selection_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_dispatcher_cuda_selection_test)
    add_test(NAME vbt_dispatcher_cuda_selection_test COMMAND "$<TARGET_FILE:vbt_dispatcher_cuda_selection_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_dispatcher_cuda_selection_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatcher_cuda_selection_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_dispatcher_fabric_test tests/cpp/dispatcher_fabric_test.cc)
    target_link_libraries(vbt_dispatcher_fabric_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_dispatcher_fabric_test)
    add_test(NAME vbt_dispatcher_fabric_test COMMAND "$<TARGET_FILE:vbt_dispatcher_fabric_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_dispatcher_fabric_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_dispatcher_fabric_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_clone_test tests/cpp/cuda_clone_test.cc)
    target_link_libraries(vbt_cuda_clone_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_clone_test)
    add_test(NAME vbt_cuda_clone_test COMMAND "$<TARGET_FILE:vbt_cuda_clone_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_clone_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_clone_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_status_helper_test tests/cpp/cuda_graphs_status_helper_test.cc)
    target_link_libraries(vbt_cuda_graphs_status_helper_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_status_helper_test)
    add_test(NAME vbt_cuda_graphs_status_helper_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_status_helper_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_status_helper_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_status_helper_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_allocator_pool_lifecycle_test tests/cpp/cuda_graphs_allocator_pool_lifecycle_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_pool_lifecycle_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_pool_lifecycle_test)
    add_test(NAME vbt_cuda_graphs_allocator_pool_lifecycle_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_pool_lifecycle_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_pool_lifecycle_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_pool_lifecycle_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_allocator_graph_pools_snapshot_test tests/cpp/cuda_graphs_allocator_graph_pools_snapshot_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_graph_pools_snapshot_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_graph_pools_snapshot_test)
    add_test(NAME vbt_cuda_graphs_allocator_graph_pools_snapshot_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_graph_pools_snapshot_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_graph_pools_snapshot_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_graph_pools_snapshot_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_allocator_capture_routing_test tests/cpp/cuda_graphs_allocator_capture_routing_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_capture_routing_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_capture_routing_test)
    add_test(NAME vbt_cuda_graphs_allocator_capture_routing_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_capture_routing_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_capture_routing_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_capture_routing_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")


    add_executable(vbt_cuda_graphs_allocator_async_denial_test tests/cpp/cuda_graphs_allocator_async_denial_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_async_denial_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_async_denial_test)
    add_test(NAME vbt_cuda_graphs_allocator_async_denial_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_async_denial_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_async_denial_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_async_denial_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_allocator_stats_snapshot_test tests/cpp/cuda_graphs_allocator_stats_snapshot_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_stats_snapshot_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_stats_snapshot_test)
    add_test(NAME vbt_cuda_graphs_allocator_stats_snapshot_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_stats_snapshot_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_stats_snapshot_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_stats_snapshot_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_allocator_capture_denial_test tests/cpp/cuda_graphs_allocator_capture_denial_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_capture_denial_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_capture_denial_test)
    add_test(NAME vbt_cuda_graphs_allocator_capture_denial_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_capture_denial_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_capture_denial_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_capture_denial_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_allocator_pool_invariants_test tests/cpp/cuda_graphs_allocator_pool_invariants_test.cc)
    target_link_libraries(vbt_cuda_graphs_allocator_pool_invariants_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_allocator_pool_invariants_test)
    add_test(NAME vbt_cuda_graphs_allocator_pool_invariants_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_allocator_pool_invariants_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_allocator_pool_invariants_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_allocator_pool_invariants_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    # Allocator+CUDA graphs perf harnesses (C++ side).
    add_executable(vbt_cuda_allocator_native_perf_test
      tests/cpp/cuda_allocator_native_perf_test.cc
      tests/cpp/cuda_allocator_perf.cc)
    target_link_libraries(vbt_cuda_allocator_native_perf_test PRIVATE vbt_core)
    vbt_apply_sanitizers(vbt_cuda_allocator_native_perf_test)

    add_executable(vbt_cuda_allocator_async_perf_test
      tests/cpp/cuda_allocator_async_perf_test.cc
      tests/cpp/cuda_allocator_perf.cc)
    target_link_libraries(vbt_cuda_allocator_async_perf_test PRIVATE vbt_core)
    vbt_apply_sanitizers(vbt_cuda_allocator_async_perf_test)

    add_executable(vbt_cuda_allocator_perf_smoke_test
      tests/cpp/cuda_allocator_perf_smoke_test.cc
      tests/cpp/cuda_allocator_perf.cc)
    target_link_libraries(vbt_cuda_allocator_perf_smoke_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_perf_smoke_test)
    add_test(NAME vbt_cuda_allocator_perf_smoke_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_perf_smoke_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_perf_smoke_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_perf_smoke_test>" LABELS "perf;allocator;cuda;graphs")

    add_executable(vbt_cuda_allocator_tsan_smoke_test tests/cpp/cuda_allocator_tsan_smoke_test.cc)
    target_link_libraries(vbt_cuda_allocator_tsan_smoke_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_allocator_tsan_smoke_test)
    add_test(NAME vbt_cuda_allocator_tsan_smoke_test COMMAND "$<TARGET_FILE:vbt_cuda_allocator_tsan_smoke_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_allocator_tsan_smoke_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_allocator_tsan_smoke_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_deferred_free_during_capture_test tests/cpp/cuda_graphs_deferred_free_during_capture_test.cc)
    target_link_libraries(vbt_cuda_graphs_deferred_free_during_capture_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_deferred_free_during_capture_test)
    add_test(NAME vbt_cuda_graphs_deferred_free_during_capture_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_deferred_free_during_capture_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_deferred_free_during_capture_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_deferred_free_during_capture_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_status_mode_rejection_test tests/cpp/cuda_graphs_status_mode_rejection_test.cc)
    target_link_libraries(vbt_cuda_graphs_status_mode_rejection_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_status_mode_rejection_test)
    add_test(NAME vbt_cuda_graphs_status_mode_rejection_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_status_mode_rejection_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_status_mode_rejection_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_status_mode_rejection_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_dtor_fallback_ends_capture_test tests/cpp/cuda_graphs_dtor_fallback_ends_capture_test.cc)
    target_link_libraries(vbt_cuda_graphs_dtor_fallback_ends_capture_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_dtor_fallback_ends_capture_test)
    add_test(NAME vbt_cuda_graphs_dtor_fallback_ends_capture_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_dtor_fallback_ends_capture_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_dtor_fallback_ends_capture_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_dtor_fallback_ends_capture_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_instantiate_and_replay_test tests/cpp/cuda_graphs_instantiate_and_replay_test.cc)
    target_link_libraries(vbt_cuda_graphs_instantiate_and_replay_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_instantiate_and_replay_test)
    add_test(NAME vbt_cuda_graphs_instantiate_and_replay_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_instantiate_and_replay_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_instantiate_and_replay_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_instantiate_and_replay_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_graphs_replay_nesting_guard_test tests/cpp/cuda_graphs_replay_nesting_guard_test.cc)
    target_link_libraries(vbt_cuda_graphs_replay_nesting_guard_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_graphs_replay_nesting_guard_test)
    add_test(NAME vbt_cuda_graphs_replay_nesting_guard_test COMMAND "$<TARGET_FILE:vbt_cuda_graphs_replay_nesting_guard_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_graphs_replay_nesting_guard_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_graphs_replay_nesting_guard_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

  if(VBT_ENABLE_ASAN)
    # Mark CUDA Graphs tests as unsupported under ASAN due to CUDA runtime stream creation failures.
    set(_VBT_CUDA_GRAPHS_ASAN_UNSUPPORTED
        vbt_cuda_graphs_allocator_capture_routing_test
        vbt_cuda_graphs_allocator_async_denial_test
        vbt_cuda_graphs_allocator_stats_snapshot_test
        vbt_cuda_graphs_allocator_capture_denial_test
        vbt_cuda_graphs_allocator_pool_invariants_test
        vbt_cuda_graphs_deferred_free_during_capture_test
        vbt_cuda_graphs_status_mode_rejection_test
        vbt_cuda_graphs_dtor_fallback_ends_capture_test
        vbt_cuda_graphs_instantiate_and_replay_test
        vbt_cuda_graphs_replay_nesting_guard_test)
    foreach(t IN LISTS _VBT_CUDA_GRAPHS_ASAN_UNSUPPORTED)
      set_tests_properties(${t} PROPERTIES LABELS "asan_unsupported;cuda_graphs" DISABLED TRUE)
    endforeach()
  endif()

    add_executable(vbt_cuda_rng_uniform_test tests/cpp/rng_cuda_uniform_test.cc)
    target_link_libraries(vbt_cuda_rng_uniform_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_uniform_test)
    add_test(NAME vbt_cuda_rng_uniform_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_uniform_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_uniform_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_uniform_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_rng_normal_test tests/cpp/rng_cuda_normal_test.cc)
    target_link_libraries(vbt_cuda_rng_normal_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_normal_test)
    add_test(NAME vbt_cuda_rng_normal_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_normal_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_normal_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_normal_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_rng_bernoulli_test tests/cpp/rng_cuda_bernoulli_test.cc)
    target_link_libraries(vbt_cuda_rng_bernoulli_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_bernoulli_test)
    add_test(NAME vbt_cuda_rng_bernoulli_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_bernoulli_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_bernoulli_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_bernoulli_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_rng_randint_test tests/cpp/rng_cuda_randint_test.cc)
    target_link_libraries(vbt_cuda_rng_randint_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_randint_test)
    add_test(NAME vbt_cuda_rng_randint_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_randint_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_randint_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_randint_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_rng_guard_error_text_test tests/cpp/rng_cuda_guard_error_text_test.cc)
    target_link_libraries(vbt_cuda_rng_guard_error_text_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_guard_error_text_test)
    add_test(NAME vbt_cuda_rng_guard_error_text_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_guard_error_text_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_guard_error_text_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_guard_error_text_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_rng_graph_capture_core_test tests/cpp/rng_cuda_graph_capture_core_test.cc)
    target_link_libraries(vbt_cuda_rng_graph_capture_core_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_graph_capture_core_test)
    add_test(NAME vbt_cuda_rng_graph_capture_core_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_graph_capture_core_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_graph_capture_core_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_graph_capture_core_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")

    add_executable(vbt_cuda_rng_uniform_graph_capture_integration_test tests/cpp/rng_cuda_uniform_graph_capture_integration_test.cc)
    target_link_libraries(vbt_cuda_rng_uniform_graph_capture_integration_test PRIVATE vbt_core GTest::gtest_main)
    vbt_apply_sanitizers(vbt_cuda_rng_uniform_graph_capture_integration_test)
    add_test(NAME vbt_cuda_rng_uniform_graph_capture_integration_test COMMAND "$<TARGET_FILE:vbt_cuda_rng_uniform_graph_capture_integration_test>" --gtest_also_run_disabled_tests)
    set_tests_properties(vbt_cuda_rng_uniform_graph_capture_integration_test PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:vbt_cuda_rng_uniform_graph_capture_integration_test>" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endif()

  # Force build of specific tests when building _C (for dev iteration via pip)
  add_dependencies(_C vbt_tensor_iter_resize_outputs_test vbt_tensor_iter_coalescing_test)
endif()
