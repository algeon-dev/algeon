cmake_minimum_required(VERSION 3.26)
project(cuml_rust LANGUAGES C CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../../cuml/cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-find)
include(rapids-cuda)
include(rapids-cpm)
rapids_cpm_init()
rapids_cmake_install_lib_dir(lib_dir)
set(CUML_INSTALL_DIR "$ENV{CUML_INSTALL_DIR}" CACHE PATH "Canonical cuML install prefix")
set(cuml_ROOT "${CUML_INSTALL_DIR}" CACHE PATH "Canonical cuML install prefix")
set(CUDF_INSTALL_DIR "$ENV{CUDF_INSTALL_DIR}" CACHE PATH "Canonical cuDF install prefix")
set(CUVS_INSTALL_DIR "$ENV{CUVS_INSTALL_DIR}" CACHE PATH "Canonical cuVS install prefix")
set(CMAKE_PREFIX_PATH "${CUML_INSTALL_DIR};${CUDF_INSTALL_DIR};${CUVS_INSTALL_DIR};${CMAKE_PREFIX_PATH}" CACHE STRING "RAPIDS package prefixes" FORCE)
rapids_find_package(cuml REQUIRED)
if(NOT EXISTS "${CUML_INSTALL_DIR}/share/algeon-cuml/openmp-disabled")
  message(FATAL_ERROR "CUML_INSTALL_DIR must contain share/algeon-cuml/openmp-disabled from the OpenMP-disabled cuML build")
endif()
rapids_find_package(CUDAToolkit REQUIRED)

set(CUDF_INSTALL_DIR "$ENV{CUDF_INSTALL_DIR}" CACHE PATH "Canonical cuDF install prefix")
find_path(CUDF_RUST_INCLUDE_DIR cudf_rust/error_boundary.hpp
  PATHS "${CUDF_INSTALL_DIR}/include" NO_DEFAULT_PATH)
if(NOT CUDF_RUST_INCLUDE_DIR OR IS_SYMLINK "${CUDF_RUST_INCLUDE_DIR}/cudf_rust/error_boundary.hpp")
  message(FATAL_ERROR "CUDF_INSTALL_DIR must provide a non-symlink cudf_rust/error_boundary.hpp")
endif()
find_library(CUDF_RUST_RMM_INTEROP_LIB cudf_rust_rmm_interop
  PATHS "${CUDF_INSTALL_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

option(CUML_RUST_ENABLE_TEST_HOOKS "Build native test-only hooks" OFF)
set(CUML_RUST_SOURCES
  src/error.cpp src/handle.cpp src/abi_checks.cpp src/c_header_smoke.c
  src/pca.cpp src/tsvd.cpp src/glm.cpp src/hdbscan.cpp src/umap.cpp
  src/random_forest.cpp src/forest_inference.cpp)
if(CUML_RUST_ENABLE_TEST_HOOKS)
  list(APPEND CUML_RUST_SOURCES src/test_support.cpp)
endif()

add_library(cuml_rust SHARED ${CUML_RUST_SOURCES})
target_compile_features(cuml_rust PRIVATE cxx_std_20)
set_target_properties(cuml_rust PROPERTIES POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "$ORIGIN;${CUML_INSTALL_DIR}/lib;${CUDF_INSTALL_DIR}/lib;${CUVS_INSTALL_DIR}/lib;${CUDAToolkit_LIBRARY_DIR}"
  INSTALL_RPATH "$ORIGIN;${CUML_INSTALL_DIR}/lib;${CUDF_INSTALL_DIR}/lib;${CUVS_INSTALL_DIR}/lib;${CUDAToolkit_LIBRARY_DIR}")
target_include_directories(cuml_rust PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CUDF_RUST_INCLUDE_DIR}")
# libcuml.so resolves libcuvs.so only through an $ORIGIN RUNPATH, and a
# RUNPATH is not inherited by a library's children. Record libcuvs.so as a
# direct shim dependency so the shim's RUNPATH resolves the canonical copy.
find_library(CUML_RUST_CUVS_LIB cuvs PATHS "${CUVS_INSTALL_DIR}/lib" "${CUVS_INSTALL_DIR}/lib64"
  NO_DEFAULT_PATH REQUIRED)
target_link_libraries(cuml_rust PUBLIC cuml::cuml PRIVATE CUDA::cudart CUDA::cuda_driver "${CUDF_RUST_RMM_INTEROP_LIB}"
  nvforest::nvforest treelite::treelite
  "-Wl,--push-state,--no-as-needed" "${CUML_RUST_CUVS_LIB}" "-Wl,--pop-state")
target_link_options(cuml_rust PRIVATE "-Wl,--no-undefined" "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuml-rust.map")

install(TARGETS cuml_rust LIBRARY DESTINATION "${lib_dir}")
install(DIRECTORY include/cuml_rust/ DESTINATION include/cuml_rust
  PATTERN "test_support.h" EXCLUDE)
if(CUML_RUST_ENABLE_TEST_HOOKS)
  install(FILES include/cuml_rust/test_support.h DESTINATION include/cuml_rust)
endif()
configure_file(cmake/cargo-link.txt.in "${CMAKE_CURRENT_BINARY_DIR}/cargo-link.txt" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cargo-link.txt" DESTINATION share/cuml_rust)
