#ifndef CUDF_RUST_TYPES_H
#define CUDF_RUST_TYPES_H

#include "error.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Mirrors cudf::type_id with identical discriminant values.
/// This enum is part of the Rust FFI ABI and must remain 32-bit compatible
/// with Rust `#[repr(i32)]`. Keep `native/src/abi_checks.cpp` in sync.
typedef enum {
  CUDF_TYPE_EMPTY = 0,
  CUDF_TYPE_INT8,
  CUDF_TYPE_INT16,
  CUDF_TYPE_INT32,
  CUDF_TYPE_INT64,
  CUDF_TYPE_UINT8,
  CUDF_TYPE_UINT16,
  CUDF_TYPE_UINT32,
  CUDF_TYPE_UINT64,
  CUDF_TYPE_FLOAT32,
  CUDF_TYPE_FLOAT64,
  CUDF_TYPE_BOOL8,
  CUDF_TYPE_TIMESTAMP_DAYS,
  CUDF_TYPE_TIMESTAMP_SECONDS,
  CUDF_TYPE_TIMESTAMP_MILLISECONDS,
  CUDF_TYPE_TIMESTAMP_MICROSECONDS,
  CUDF_TYPE_TIMESTAMP_NANOSECONDS,
  CUDF_TYPE_DURATION_DAYS,
  CUDF_TYPE_DURATION_SECONDS,
  CUDF_TYPE_DURATION_MILLISECONDS,
  CUDF_TYPE_DURATION_MICROSECONDS,
  CUDF_TYPE_DURATION_NANOSECONDS,
  CUDF_TYPE_DICTIONARY32,
  CUDF_TYPE_STRING,
  CUDF_TYPE_LIST,
  CUDF_TYPE_DECIMAL32,
  CUDF_TYPE_DECIMAL64,
  CUDF_TYPE_DECIMAL128,
  CUDF_TYPE_STRUCT,
} cudf_type_id_t;

/// Opaque handle to a cudf::data_type.
typedef struct cudf_data_type_handle_state* cudf_data_type_handle;

/// Create a data_type from type id (scale=0).
cudf_status_t cudf_data_type_create(cudf_type_id_t type_id, cudf_data_type_handle* out);

/// Create a data_type with explicit scale (for DECIMAL types).
cudf_status_t cudf_data_type_create_fixed_point(cudf_type_id_t type_id,
                                                int32_t scale,
                                                cudf_data_type_handle* out);

/// Destroy a data_type handle. Ownership is consumed on call; success means
/// cleanup completed and failure quarantines the handle.
cudf_status_t cudf_data_type_destroy(cudf_data_type_handle handle);

/// Get the type_id from a data_type.
cudf_status_t cudf_data_type_id(cudf_data_type_handle handle, cudf_type_id_t* out);

/// Get the scale from a data_type.
cudf_status_t cudf_data_type_scale(cudf_data_type_handle handle, int32_t* out);

/// Return the size in bytes of a fixed-width type. Returns 0 for variable-width.
cudf_status_t cudf_type_size(cudf_type_id_t type_id, int32_t* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_TYPES_H
