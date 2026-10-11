#ifndef CUDF_RUST_SHIM_COMMON_H
#define CUDF_RUST_SHIM_COMMON_H

#include "cudf_rust/error.h"
#include "cudf_rust/types.h"
#include "owning_handle_fwd.hpp"

#include <cudf/types.hpp>

#include <cstdint>
#include <string>

// Defined in error.cpp
void set_error(std::string const& msg);

void set_error(char const* msg);

// Records `status` in the copyable error detail. Nonthrowing producers use it
// when the returned or recorded status classifies a failure, and the message
// plays no part in that classification.
void set_error(cudf_status_t status, char const* msg) noexcept;

cudf_status_t translate_exception();

cudf_status_t translate_io_exception();

// Destructor-boundary translation that is itself guaranteed not to throw.
cudf_status_t cudf_translate_cleanup_exception_noexcept() noexcept;

// Exception-catching macro pair for C shim functions.
//
// Allocation failures must always surface as CUDF_ERROR_OUT_OF_MEMORY.
// Most shim code relies on exception translation for this, but explicit OOM
// checks are also allowed when they provide clearer context to Rust callers.
#define CUDF_RUST_TRY \
  cudf_clear_error(); \
  try

#define CUDF_RUST_CATCH \
  catch (...) { return translate_exception(); }

#define CUDF_RUST_IO_CATCH \
  catch (...) { return translate_io_exception(); }

// Null-check macro that sets error and returns.
//
// Under the project rule for C-shim output parameters, an output pointer is
// initialized right after it passes validation. Then any later failure leaves
// caller-visible outputs in a predictable state.
#define CUDF_RUST_NULL_CHECK(ptr, msg) \
  do {                                 \
    if ((ptr) == nullptr) {            \
      set_error(msg);                  \
      return CUDF_ERROR_NULL;          \
    }                                  \
  } while (0)

inline cudf_status_t validate_c_string_array(char const* const* values,
                                             int32_t count,
                                             char const* arg_name)
{
  for (int32_t i = 0; i < count; ++i) {
    if (values[i] == nullptr) {
      set_error(std::string(arg_name) + " entry is null");
      return CUDF_ERROR_NULL;
    }
  }
  return CUDF_SUCCESS;
}

inline cudf::type_id to_cudf_type_id(cudf_type_id_t id) { return static_cast<cudf::type_id>(id); }

#endif  // CUDF_RUST_SHIM_COMMON_H
