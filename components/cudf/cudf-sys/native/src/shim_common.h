#ifndef CUDF_RUST_SHIM_COMMON_H
#define CUDF_RUST_SHIM_COMMON_H

#include "error.h"
#include <cstdint>
#include <string>

// Defined in error.cpp
void set_error(const std::string& msg);
void set_error(const char* msg);
cudf_status_t translate_exception();
cudf_status_t translate_io_exception();

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

#define CUDF_RUST_IO_CATCH               \
  catch (...) {                          \
    return translate_io_exception();     \
  }

// Null-check macro that sets error and returns.
//
// Project rule for C-shim output parameters:
// once an output pointer itself has been validated, initialize it immediately
// so any later failure leaves caller-visible outputs in a predictable state.
#define CUDF_RUST_NULL_CHECK(ptr, msg)            \
  do {                                            \
    if ((ptr) == nullptr) {                       \
      set_error(msg);                             \
      return CUDF_ERROR_NULL;                     \
    }                                             \
  } while (0)

inline cudf_status_t validate_c_string_array(char const* const* values,
                                             int32_t count,
                                             char const* arg_name) {
  for (int32_t i = 0; i < count; ++i) {
    if (values[i] == nullptr) {
      set_error(std::string(arg_name) + " entry is null");
      return CUDF_ERROR_NULL;
    }
  }
  return CUDF_SUCCESS;
}

#endif  // CUDF_RUST_SHIM_COMMON_H
