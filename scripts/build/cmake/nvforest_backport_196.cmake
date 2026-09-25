# Injected after project(NVForest) through CMAKE_PROJECT_NVForest_INCLUDE.
# nvForest release/26.08, pinned by cuML 26.08, copies decision_forest buffers
# on the legacy default stream, which the query domain refuses. Upstream #196
# (bfc77ff, first in 26.10) moves them instead. The importer is header-only, so
# the patched headers are what the cuML shim instantiates. Delete this hook,
# the patch, and the build_libcuml.sh header check on the 26.10 bump.
set(_nvforest_patch "${CMAKE_CURRENT_LIST_DIR}/../patches/nvforest-196-move-buffers.patch")
file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.." _nvforest_source)
file(REAL_PATH "${CMAKE_BINARY_DIR}/_deps" _nvforest_deps)
cmake_path(IS_PREFIX _nvforest_deps "${_nvforest_source}" NORMALIZE _nvforest_in_build_tree)
if(NOT _nvforest_in_build_tree)
  message(FATAL_ERROR "nvForest source '${_nvforest_source}' is outside '${_nvforest_deps}'; refusing to patch a shared checkout")
endif()
execute_process(COMMAND git -C "${_nvforest_source}" apply --check --reverse "${_nvforest_patch}"
                RESULT_VARIABLE _nvforest_applied OUTPUT_QUIET ERROR_QUIET)
if(NOT _nvforest_applied EQUAL 0)
  execute_process(COMMAND git -C "${_nvforest_source}" apply --check "${_nvforest_patch}"
                  RESULT_VARIABLE _nvforest_applicable)
  if(NOT _nvforest_applicable EQUAL 0)
    message(FATAL_ERROR "nvForest backport #196 no longer applies; if upstream includes it, remove this hook")
  endif()
  execute_process(COMMAND git -C "${_nvforest_source}" apply "${_nvforest_patch}" COMMAND_ERROR_IS_FATAL ANY)
endif()
