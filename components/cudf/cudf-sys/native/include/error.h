#ifndef CUDF_RUST_ERROR_H
#define CUDF_RUST_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/// Error codes returned by C shim functions.
/// This enum is part of the Rust FFI ABI and must remain 32-bit compatible
/// with Rust `#[repr(i32)]`. Keep `native/src/abi_checks.cpp` in sync.
typedef enum {
  CUDF_SUCCESS                 = 0,
  CUDF_ERROR_INVALID_OPERATION = 1,
  CUDF_ERROR_CUDA              = 2,
  CUDF_ERROR_OUT_OF_MEMORY     = 3,
  CUDF_ERROR_IO                = 4,
  CUDF_ERROR_TYPE              = 5,
  CUDF_ERROR_NULL              = 6,
  CUDF_ERROR_INDEX             = 7,
  CUDF_ERROR_UNKNOWN           = 8,
  CUDF_ERROR_FATAL_CUDA        = 9,
  CUDF_ERROR_INVALID_ARGUMENT  = 10,
  CUDF_ERROR_UNSUPPORTED       = 11,
  CUDF_ERROR_SIZE_OVERFLOW     = 12,
} cudf_status_t;

/// Retrieve the thread-local error message from the last failed call.
/// Returns NULL if no error is pending. The returned pointer is valid until the
/// next C shim call on the same thread.
const char* cudf_last_error_message(void);

/// Clear the thread-local error state.
void cudf_clear_error(void);

/// Testing-only: make the next provenance-bound destroy on this thread return
/// `status` without running native cleanup. Pass CUDF_SUCCESS to disarm.
void cudf_test_fail_next_provenance_destroy(cudf_status_t status);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_ERROR_H
