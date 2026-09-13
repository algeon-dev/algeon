#ifndef CUML_RUST_SHIM_COMMON_HPP
#define CUML_RUST_SHIM_COMMON_HPP

#include "cuml_rust/error.h"

#include <exception>
#include <stdexcept>

struct cuml_invalid_state final : std::runtime_error {
  using std::runtime_error::runtime_error;
};

cuml_status_t cuml_translate_exception() noexcept;
cuml_status_t cuml_translate_cleanup_exception_noexcept() noexcept;

#define CUML_RUST_TRY \
  cuml_clear_error(); \
  try
#define CUML_RUST_CATCH \
  catch (...) { return cuml_translate_exception(); }

#endif
