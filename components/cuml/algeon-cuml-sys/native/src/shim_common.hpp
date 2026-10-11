#ifndef CUML_RUST_SHIM_COMMON_HPP
#define CUML_RUST_SHIM_COMMON_HPP

#include "cuml_rust/error.h"

#include <cuml/common/logger.hpp>

#include <exception>
#include <stdexcept>

struct cuml_invalid_state final : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Upstream QN and UMAP solvers call ML::default_logger().set_level(...) with
// their per-call verbosity on every fit/transform, overwriting the
// process-global cuML logger level. Every shim call site that sets a
// verbosity or level must pass this same level, or upstream re-enables
// trace/info logging regardless of the server's configured log level.
inline constexpr rapids_logger::level_enum cuml_native_log_level = rapids_logger::level_enum::warn;

cuml_status_t cuml_translate_exception() noexcept;
cuml_status_t cuml_translate_cleanup_exception_noexcept() noexcept;

#define CUML_RUST_TRY \
  cuml_clear_error(); \
  try
#define CUML_RUST_CATCH \
  catch (...) { return cuml_translate_exception(); }

#endif
