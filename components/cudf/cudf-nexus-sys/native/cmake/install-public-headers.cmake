foreach(required_variable IN ITEMS
        CUDF_RUST_PUBLIC_HEADER_SOURCE_DIR
        CUDF_RUST_PUBLIC_HEADER_INSTALL_PREFIX
        CUDF_RUST_PUBLIC_HEADER_TEST_HOOKS)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "${required_variable} must be defined before installing cudf_rust public headers")
  endif()
endforeach()

if(NOT IS_DIRECTORY "${CUDF_RUST_PUBLIC_HEADER_SOURCE_DIR}")
  message(
    FATAL_ERROR
      "cudf_rust public header source '${CUDF_RUST_PUBLIC_HEADER_SOURCE_DIR}' is not a directory"
  )
endif()

file(GLOB CUDF_RUST_PUBLIC_HEADERS LIST_DIRECTORIES false "${CUDF_RUST_PUBLIC_HEADER_SOURCE_DIR}/*")
list(SORT CUDF_RUST_PUBLIC_HEADERS)
if(NOT CUDF_RUST_PUBLIC_HEADERS)
  message(FATAL_ERROR "cudf_rust public header source '${CUDF_RUST_PUBLIC_HEADER_SOURCE_DIR}' is empty")
endif()

set(CUDF_RUST_TEST_SUPPORT_HEADER "${CUDF_RUST_PUBLIC_HEADER_SOURCE_DIR}/test_support.h")
if(NOT CUDF_RUST_PUBLIC_HEADER_TEST_HOOKS)
  list(REMOVE_ITEM CUDF_RUST_PUBLIC_HEADERS "${CUDF_RUST_TEST_SUPPORT_HEADER}")
endif()

set(CUDF_RUST_PUBLIC_HEADER_DESTINATION "${CUDF_RUST_PUBLIC_HEADER_INSTALL_PREFIX}/include/cudf_rust")
file(MAKE_DIRECTORY "${CUDF_RUST_PUBLIC_HEADER_DESTINATION}")

foreach(header IN LISTS CUDF_RUST_PUBLIC_HEADERS)
  get_filename_component(header_name "${header}" NAME)
  # Remove first so an obsolete target-local symlink cannot survive an otherwise up-to-date install.
  file(REMOVE
       "${CUDF_RUST_PUBLIC_HEADER_INSTALL_PREFIX}/include/${header_name}"
       "${CUDF_RUST_PUBLIC_HEADER_DESTINATION}/${header_name}")
endforeach()

if(NOT CUDF_RUST_PUBLIC_HEADER_TEST_HOOKS)
  file(REMOVE
       "${CUDF_RUST_PUBLIC_HEADER_INSTALL_PREFIX}/include/test_support.h"
       "${CUDF_RUST_PUBLIC_HEADER_DESTINATION}/test_support.h")
endif()

file(
  INSTALL
  DESTINATION "${CUDF_RUST_PUBLIC_HEADER_DESTINATION}"
  TYPE FILE
  FILES ${CUDF_RUST_PUBLIC_HEADERS}
)
