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
  LANGUAGES CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuDF Rust shared libraries" ON)
option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

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
  set(CMAKE_PREFIX_PATH ${CUDF_INSTALL_DIR} "${CUDF_INSTALL_DIR}/lib/cmake" "${CUDF_INSTALL_DIR}/lib64/cmake")
  message(STATUS "Using dependencies from libcudf installed path: ${CUDF_INSTALL_DIR}")
elseif(DEFINED ENV{CUDF_CPP_BUILD_DIR})
  set(CUDF_CPP_BUILD_DIR
      "$ENV{CUDF_CPP_BUILD_DIR}"
      CACHE STRING "path to libcudf build root"
  )
  set(cudf_ROOT ${CUDF_CPP_BUILD_DIR})
  message(STATUS "Using pre-configured libcudf from build path: ${CUDF_CPP_BUILD_DIR}")
else()
  message(FATAL_ERROR "Either CUDF_INSTALL_DIR or CUDF_CPP_BUILD_DIR must be defined.")
endif()

rapids_find_package(cudf REQUIRED)

get_target_property(CUDF_CXX_FLAGS cudf::cudf CUDF_CXX_FLAGS)
get_target_property(CUDF_CUDA_FLAGS cudf::cudf CUDF_CUDA_FLAGS)
get_target_property(CUDF_CXX_DEFINITIONS cudf::cudf CUDF_CXX_DEFINITIONS)
get_target_property(CUDF_CUDA_DEFINITIONS cudf::cudf CUDF_CUDA_DEFINITIONS)

# ##################################################################################################
# * library target --------------------------------------------------------------------------------

add_library(
  cudf_rust
  src/error.cpp
  src/types.cpp
  src/column.cpp
  src/table.cpp
  src/scalar.cpp
  src/stream.cpp
  src/rmm.cpp
  src/io_parquet.cpp
  src/io_csv.cpp
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
  src/stream_compaction.cpp
  src/abi_checks.cpp
  src/c_header_smoke.c
)

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
  target_link_libraries(cudf_rust PRIVATE CUDA::cudart_static)
else()
  target_link_libraries(cudf_rust PRIVATE CUDA::cudart)
endif()

# ##################################################################################################
# * install ---------------------------------------------------------------------------------------

install(TARGETS cudf_rust LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
