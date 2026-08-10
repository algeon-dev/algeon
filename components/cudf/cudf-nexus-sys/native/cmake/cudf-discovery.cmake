# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

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

function(cudf_rust_build_dir_metadata_matches_install build_dir install_dir build_dir_explicit output_var)
  set(use_metadata OFF)
  cudf_rust_cache_value("${build_dir}/CMakeCache.txt" "CMAKE_INSTALL_PREFIX"
                        cudf_rust_build_install_prefix
  )
  if(cudf_rust_build_install_prefix)
    get_filename_component(cudf_rust_build_install_prefix_real "${cudf_rust_build_install_prefix}"
                           REALPATH
    )
    get_filename_component(cudf_rust_install_dir_real "${install_dir}" REALPATH)
    if(cudf_rust_build_install_prefix_real STREQUAL cudf_rust_install_dir_real)
      set(use_metadata ON)
    elseif(build_dir_explicit)
      message(
        FATAL_ERROR
          "CUDF_CPP_BUILD_DIR CMakeCache.txt has CMAKE_INSTALL_PREFIX=${cudf_rust_build_install_prefix}, which does not match CUDF_INSTALL_DIR=${install_dir}"
      )
    else()
      message(
        STATUS
          "Ignoring sibling libcudf build metadata because CMAKE_INSTALL_PREFIX=${cudf_rust_build_install_prefix} does not match CUDF_INSTALL_DIR=${install_dir}"
      )
    endif()
  elseif(build_dir_explicit)
    message(
      FATAL_ERROR
        "CUDF_CPP_BUILD_DIR CMakeCache.txt does not contain CMAKE_INSTALL_PREFIX; cannot use it as install-prefix capability evidence"
    )
  endif()

  set(${output_var}
      "${use_metadata}"
      PARENT_SCOPE
  )
endfunction()

function(cudf_rust_collect_package_prefixes cache_path output_var)
  set(package_prefixes "")
  file(STRINGS "${cache_path}" cache_lines REGEX "^[A-Za-z0-9_]+_BINARY_DIR:[^=]*=")
  foreach(cache_line IN LISTS cache_lines)
    string(REGEX REPLACE "^[^=]*=" "" candidate "${cache_line}")
    if(NOT IS_DIRECTORY "${candidate}")
      continue()
    endif()

    file(GLOB package_configs LIST_DIRECTORIES FALSE "${candidate}/*Config.cmake"
         "${candidate}/*config.cmake"
    )
    if(package_configs)
      list(APPEND package_prefixes "${candidate}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES package_prefixes)
  set(${output_var}
      "${package_prefixes}"
      PARENT_SCOPE
  )
endfunction()
