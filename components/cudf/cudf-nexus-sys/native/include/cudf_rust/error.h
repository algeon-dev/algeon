#ifndef CUDF_RUST_ERROR_H
#define CUDF_RUST_ERROR_H

#include <stdint.h>

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

/// Fixed capacity of the message copied into a per-call error detail.
///
/// The payload is deliberately self-contained: Rust copies it before making
/// another shim call, so no callback or worker-thread state is authoritative.
enum { CUDF_ERROR_MESSAGE_CAPACITY = 512 };

/// Provenance of an allocation failure captured by the shim boundary.
typedef enum {
  CUDF_ERROR_FAILURE_SOURCE_PHYSICAL          = 0,
  CUDF_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT = 1,
  CUDF_ERROR_FAILURE_SOURCE_UNKNOWN           = 2,
} cudf_error_failure_source_t;

/// Fixed, copyable payload for the last failed shim call on this thread.
///
/// Zero-valued identity and allocation fields mean that the throwing owner did
/// not provide that fact. `cuda_error == -1` means that no CUDA status was
/// available at the exception boundary.
typedef struct {
  cudf_status_t status;
  cudf_error_failure_source_t failure_source;
  uint64_t domain_id;
  uint64_t member_id;
  uint64_t generation;
  uint64_t requested_bytes;
  uint64_t alignment;
  int32_t cuda_error;
  char message[CUDF_ERROR_MESSAGE_CAPACITY];
} cudf_error_detail_t;

/// Retrieve the fixed per-call error payload from the last failed call.
/// Returns NULL if no error is pending. The pointer is valid until the next C
/// shim call on the same thread; Rust must copy it immediately.
cudf_error_detail_t const* cudf_last_error_detail(void);

/// Retrieve the thread-local error message from the last failed call.
/// Returns NULL if no error is pending. The returned pointer is valid until the
/// next C shim call on the same thread.
char const* cudf_last_error_message(void);

/// Clear the thread-local error state.
void cudf_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_ERROR_H
