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
option(CUDF_RUST_ENABLE_CLANG_TIDY "Run clang-tidy for cudf-sys/native while building" OFF)

# ##################################################################################################
# * lint configuration ----------------------------------------------------------------------------

if(CUDF_RUST_ENABLE_CLANG_TIDY)
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
endif()

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
function(cudf_rust_cache_value cache_path cache_key output_var)
  file(STRINGS "${cache_path}" cache_line REGEX "^${cache_key}:[^=]*=")
  if(cache_line)
    list(GET cache_line 0 cache_line_first)
    string(REGEX REPLACE "^[^=]*=" "" cache_value "${cache_line_first}")
    set(${output_var}
        "${cache_value}"
        PARENT_SCOPE
    )
  else()
    set(${output_var}
        ""
        PARENT_SCOPE
    )
  endif()
endfunction()

set(CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE "")
set(CUDF_RUST_CUDF_BUILD_DIR_EXPLICIT OFF)
if(DEFINED ENV{CUDF_CPP_BUILD_DIR})
  set(CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE "$ENV{CUDF_CPP_BUILD_DIR}")
  set(CUDF_RUST_CUDF_BUILD_DIR_EXPLICIT ON)
elseif(EXISTS "${CUDF_SOURCE_DIR}/build/CMakeCache.txt")
  set(CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE "${CUDF_SOURCE_DIR}/build")
endif()

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
  list(PREPEND CMAKE_PREFIX_PATH ${CUDF_CPP_BUILD_DIR} "${CUDF_CPP_BUILD_DIR}/_deps/kvikio-build")
  set(CUDF_RUST_CUDF_DISCOVERY_MODE "build")
  message(STATUS "Using pre-configured libcudf from build path: ${CUDF_CPP_BUILD_DIR}")
else()
  message(FATAL_ERROR "Either CUDF_INSTALL_DIR or CUDF_CPP_BUILD_DIR must be defined.")
endif()

set(CUDF_RUST_CUDF_BUILD_DIR "")
if(CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE
   AND EXISTS "${CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE}/CMakeCache.txt"
)
  set(CUDF_RUST_USE_CUDF_BUILD_DIR_METADATA OFF)
  if(CUDF_RUST_CUDF_DISCOVERY_MODE STREQUAL "build")
    set(CUDF_RUST_USE_CUDF_BUILD_DIR_METADATA ON)
  elseif(CUDF_RUST_CUDF_DISCOVERY_MODE STREQUAL "install")
    cudf_rust_cache_value(
      "${CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE}/CMakeCache.txt"
      "CMAKE_INSTALL_PREFIX"
      CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX
    )
    if(CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX)
      get_filename_component(CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX_REAL
                             "${CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX}" REALPATH
      )
      get_filename_component(CUDF_RUST_CUDF_INSTALL_DIR_REAL "${CUDF_INSTALL_DIR}" REALPATH)
      if(CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX_REAL STREQUAL CUDF_RUST_CUDF_INSTALL_DIR_REAL)
        set(CUDF_RUST_USE_CUDF_BUILD_DIR_METADATA ON)
      elseif(CUDF_RUST_CUDF_BUILD_DIR_EXPLICIT)
        message(
          FATAL_ERROR
            "CUDF_CPP_BUILD_DIR CMakeCache.txt has CMAKE_INSTALL_PREFIX=${CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX}, which does not match CUDF_INSTALL_DIR=${CUDF_INSTALL_DIR}"
        )
      else()
        message(
          STATUS
            "Ignoring sibling libcudf build metadata because CMAKE_INSTALL_PREFIX=${CUDF_RUST_CUDF_BUILD_INSTALL_PREFIX} does not match CUDF_INSTALL_DIR=${CUDF_INSTALL_DIR}"
        )
      endif()
    elseif(CUDF_RUST_CUDF_BUILD_DIR_EXPLICIT)
      message(
        FATAL_ERROR
          "CUDF_CPP_BUILD_DIR CMakeCache.txt does not contain CMAKE_INSTALL_PREFIX; cannot use it as install-prefix capability evidence"
      )
    endif()
  endif()

  if(CUDF_RUST_USE_CUDF_BUILD_DIR_METADATA)
    set(CUDF_RUST_CUDF_BUILD_DIR "${CUDF_RUST_CUDF_BUILD_DIR_CANDIDATE}")
    list(PREPEND CMAKE_PREFIX_PATH ${CUDF_RUST_CUDF_BUILD_DIR} "${CUDF_RUST_CUDF_BUILD_DIR}/_deps/kvikio-build")
  endif()
endif()

rapids_find_package(cudf REQUIRED)
find_package(kvikio CONFIG QUIET)

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

set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "unknown")
if(DEFINED CUDF_KVIKIO_REMOTE_IO)
  if(CUDF_KVIKIO_REMOTE_IO)
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "enabled")
  else()
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "disabled")
  endif()
elseif(CUDF_RUST_CUDF_BUILD_DIR AND EXISTS "${CUDF_RUST_CUDF_BUILD_DIR}/CMakeCache.txt")
  cudf_rust_cache_value(
    "${CUDF_RUST_CUDF_BUILD_DIR}/CMakeCache.txt"
    "CUDF_KVIKIO_REMOTE_IO"
    CUDF_RUST_KVIKIO_REMOTE_IO_CACHE
  )
  if(CUDF_RUST_KVIKIO_REMOTE_IO_CACHE MATCHES "^(ON|TRUE|1)$")
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "enabled")
  elseif(CUDF_RUST_KVIKIO_REMOTE_IO_CACHE MATCHES "^(OFF|FALSE|0)$")
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "disabled")
  endif()
elseif(DEFINED KvikIO_REMOTE_SUPPORT)
  if(KvikIO_REMOTE_SUPPORT)
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "enabled")
  else()
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "disabled")
  endif()
elseif(DEFINED KvikIO_REMOTE_SUPPORT)
  if(KvikIO_REMOTE_SUPPORT)
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "enabled")
  else()
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "disabled")
  endif()
elseif(DEFINED KvikIO_REMOTE_SUPPORT)
  if(KvikIO_REMOTE_SUPPORT)
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "enabled")
  else()
    set(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS "disabled")
  endif()
endif()
if(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS STREQUAL "enabled" AND NOT TARGET kvikio::kvikio)
  message(
    FATAL_ERROR
      "libcudf reports KvikIO remote IO enabled, but the kvikio::kvikio CMake target was not found"
  )
endif()
message(STATUS "libcudf KvikIO remote IO build status: ${CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS}")

# ##################################################################################################
# * library target --------------------------------------------------------------------------------

add_library(
  cudf_rust SHARED
  src/error.cpp
  src/handle_provenance.cpp
  src/ast.cpp
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
  src/partitioning.cpp
  src/filter.cpp
  src/gds.cpp
  src/groupby.cpp
  src/join_hash.cpp
  src/join_conditional.cpp
  src/join_mixed.cpp
  src/join_filter.cpp
  src/unary.cpp
  src/binary.cpp
  src/reduction.cpp
  src/concatenate.cpp
  src/dictionary.cpp
  src/datetime.cpp
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
if(CUDF_RUST_ENABLE_CLANG_TIDY)
  enable_clang_tidy(cudf_rust)
endif()

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
if(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS STREQUAL "enabled")
  target_compile_definitions(cudf_rust PRIVATE CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED)
elseif(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_STATUS STREQUAL "disabled")
  target_compile_definitions(cudf_rust PRIVATE CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_DISABLED)
endif()

target_link_options(cudf_rust PRIVATE "-Wl,--no-undefined")

target_link_libraries(cudf_rust PUBLIC cudf::cudf)
if(TARGET kvikio::kvikio)
  target_link_libraries(cudf_rust PUBLIC kvikio::kvikio)
endif()
target_link_libraries(cudf_rust PRIVATE ${CMAKE_DL_LIBS})

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
