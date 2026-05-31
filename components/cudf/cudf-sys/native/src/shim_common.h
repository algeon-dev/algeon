#ifndef CUDF_RUST_SHIM_COMMON_H
#define CUDF_RUST_SHIM_COMMON_H

#include "error.h"
#include <string>

// Defined in error.cpp
void set_error(const std::string& msg);
void set_error(const char* msg);
cudf_status_t translate_exception();

// Exception-catching macro pair for C shim functions.
//
// Allocation failures must always surface as CUDF_ERROR_OUT_OF_MEMORY.
// Most shim code relies on exception translation for this, but explicit OOM
// checks are also allowed when they provide clearer context to Rust callers.
#define CUDF_RUST_TRY  \
  cudf_clear_error();  \
  try

#define CUDF_RUST_CATCH                  \
  catch (...) {                          \
    return translate_exception();        \
  }

// Null-check macro that sets error and returns.
#define CUDF_RUST_NULL_CHECK(ptr, msg)            \
  do {                                            \
    if ((ptr) == nullptr) {                       \
      set_error(msg);                             \
      return CUDF_ERROR_NULL;                     \
    }                                             \
  } while (0)

#endif  // CUDF_RUST_SHIM_COMMON_H
