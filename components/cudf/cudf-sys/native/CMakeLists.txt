# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cudf/cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cuda)
include(rapids-find)
include(rapids-cpm)
rapids_cpm_init()

rapids_cuda_init_architectures(CUDF_RUST)

project(
  CUDF_RUST
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES C CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

# ##################################################################################################
# * lint configuration ----------------------------------------------------------------------------

set(CUDF_RUST_CLANG_TIDY_EXE
    "${CUDF_RUST_CLANG_TIDY_EXE}"
    CACHE FILEPATH "Path to clang-tidy executable used for cudf-sys/native"
)
if(NOT CUDF_RUST_CLANG_TIDY_EXE)
  find_program(
    CUDF_RUST_CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable used for cudf-sys/native" REQUIRED
  )
endif()

if(NOT CUDF_RUST_CLANG_TIDY_CONFIG_FILE)
  set(CUDF_RUST_CLANG_TIDY_CONFIG_FILE "${PROJECT_SOURCE_DIR}/.clang-tidy")
endif()
set(CUDF_RUST_CLANG_TIDY_CONFIG_FILE
    "${CUDF_RUST_CLANG_TIDY_CONFIG_FILE}"
    CACHE FILEPATH "Path to the cudf-sys/native clang-tidy configuration"
)
if(NOT EXISTS "${CUDF_RUST_CLANG_TIDY_CONFIG_FILE}")
  message(
    FATAL_ERROR
      "cudf-sys/native requires a clang-tidy config file at ${CUDF_RUST_CLANG_TIDY_CONFIG_FILE}"
  )
endif()

function(enable_clang_tidy target)
  set_target_properties(
    ${target}
    PROPERTIES CXX_CLANG_TIDY
                 "${CUDF_RUST_CLANG_TIDY_EXE};--config-file=${CUDF_RUST_CLANG_TIDY_CONFIG_FILE};--extra-arg=-Qunused-arguments"
  )
endfunction()

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

rapids_find_package(CUDAToolkit REQUIRED)

if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM CUDF_USE_PER_THREAD_DEFAULT_STREAM)
endif()

rapids_cmake_build_type("Release")

# ##################################################################################################
# * CUDF ------------------------------------------------------------------------------------------

set(CUDF_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../cudf/cpp")

# Search for libcudf in CUDF_INSTALL_DIR if specified, otherwise in CUDF_CPP_BUILD_DIR.
if(DEFINED ENV{CUDF_INSTALL_DIR})
  set(CUDF_INSTALL_DIR
      "$ENV{CUDF_INSTALL_DIR}"
      CACHE STRING "path to libcudf install root"
  )
  set(cudf_ROOT "${CUDF_INSTALL_DIR}")
  list(PREPEND CMAKE_PREFIX_PATH ${CUDF_INSTALL_DIR} "${CUDF_INSTALL_DIR}/lib/cmake" "${CUDF_INSTALL_DIR}/lib64/cmake")
  set(CUDF_RUST_CUDF_DISCOVERY_MODE "install")
  message(STATUS "Using dependencies from libcudf installed path: ${CUDF_INSTALL_DIR}")
elseif(DEFINED ENV{CUDF_CPP_BUILD_DIR})
  set(CUDF_CPP_BUILD_DIR
      "$ENV{CUDF_CPP_BUILD_DIR}"
      CACHE STRING "path to libcudf build root"
  )
  set(cudf_ROOT "${CUDF_CPP_BUILD_DIR}")
  set(CUDF_RUST_CUDF_DISCOVERY_MODE "build")
  message(STATUS "Using pre-configured libcudf from build path: ${CUDF_CPP_BUILD_DIR}")
else()
  message(FATAL_ERROR "Either CUDF_INSTALL_DIR or CUDF_CPP_BUILD_DIR must be defined.")
endif()

rapids_find_package(cudf REQUIRED)

get_target_property(CUDF_CXX_FLAGS cudf::cudf CUDF_CXX_FLAGS)
get_target_property(CUDF_CUDA_FLAGS cudf::cudf CUDF_CUDA_FLAGS)
get_target_property(CUDF_CXX_DEFINITIONS cudf::cudf CUDF_CXX_DEFINITIONS)
get_target_property(CUDF_CUDA_DEFINITIONS cudf::cudf CUDF_CUDA_DEFINITIONS)

if("${CUDF_CXX_FLAGS}" STREQUAL "CUDF_CXX_FLAGS-NOTFOUND")
  set(CUDF_CXX_FLAGS "")
endif()
if("${CUDF_CUDA_FLAGS}" STREQUAL "CUDF_CUDA_FLAGS-NOTFOUND")
  set(CUDF_CUDA_FLAGS "")
endif()
if("${CUDF_CXX_DEFINITIONS}" STREQUAL "CUDF_CXX_DEFINITIONS-NOTFOUND")
  set(CUDF_CXX_DEFINITIONS "")
endif()
if("${CUDF_CUDA_DEFINITIONS}" STREQUAL "CUDF_CUDA_DEFINITIONS-NOTFOUND")
  set(CUDF_CUDA_DEFINITIONS "")
endif()

# ##################################################################################################
# * library target --------------------------------------------------------------------------------

add_library(
  cudf_rust SHARED
  src/error.cpp
  src/arrow_device.cpp
  src/types.cpp
  src/owning_handle.cpp
  src/column.cpp
  src/prepared_import.cpp
  src/device_buffer.cpp
  src/table.cpp
  src/scalar.cpp
  src/stream.cpp
  src/event.cpp
  src/rmm.cpp
  src/execution_context.cpp
  src/io_parquet.cpp
  src/io_csv.cpp
  src/io_json.cpp
  src/io_test_hooks.cpp
  src/sorting.cpp
  src/filter.cpp
  src/groupby.cpp
  src/join.cpp
  src/unary.cpp
  src/binary.cpp
  src/reduction.cpp
  src/concatenate.cpp
  src/dictionary.cpp
  src/hashing.cpp
  src/replace.cpp
  src/search.cpp
  src/copying.cpp
  src/strings.cpp
  src/stream_compaction.cpp
  src/device.cpp
  src/abi_checks.cpp
  src/c_header_smoke.c
)
enable_clang_tidy(cudf_rust)

target_include_directories(
  cudf_rust PUBLIC "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/src"
)

set_target_properties(
  cudf_rust
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudf_rust PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_compile_definitions(
  cudf_rust PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>"
)

target_link_options(cudf_rust PRIVATE "-Wl,--no-undefined")

target_link_libraries(cudf_rust PUBLIC cudf::cudf)

# ##################################################################################################
# * cudart options --------------------------------------------------------------------------------

if(CUDA_STATIC_RUNTIME)
  set(CUDF_RUST_CUDA_RUNTIME_MODE "static")
  target_link_libraries(cudf_rust PRIVATE CUDA::cudart_static)
else()
  set(CUDF_RUST_CUDA_RUNTIME_MODE "shared")
  target_link_libraries(cudf_rust PRIVATE CUDA::cudart)
endif()

# ##################################################################################################
# * Cargo link manifest ---------------------------------------------------------------------------

set(CUDF_RUST_CARGO_LINK_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/cargo-link.txt")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cargo-link.txt.in" "${CUDF_RUST_CARGO_LINK_MANIFEST}" @ONLY)

# ##################################################################################################
# * install ---------------------------------------------------------------------------------------

install(TARGETS cudf_rust LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES "${CUDF_RUST_CARGO_LINK_MANIFEST}" DESTINATION share/cudf_rust)
