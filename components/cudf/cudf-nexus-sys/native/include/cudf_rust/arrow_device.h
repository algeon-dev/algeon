#ifndef CUDF_RUST_ARROW_DEVICE_H
#define CUDF_RUST_ARROW_DEVICE_H

#include "column.h"
#include "execution_context.h"
#include "table.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudf_arrow_schema_handle_state* cudf_arrow_schema_handle;
typedef struct cudf_arrow_device_array_handle_state* cudf_arrow_device_array_handle;

/// Arrow C Device device-type codes used by this shim.
///
/// These are raw Arrow ABI values. Callers must preserve unknown values until
/// a higher-level boundary validates or interprets them.
typedef enum cudf_arrow_device_type_t {
  CUDF_ARROW_DEVICE_TYPE_CPU          = 1,
  CUDF_ARROW_DEVICE_TYPE_CUDA         = 2,
  CUDF_ARROW_DEVICE_TYPE_CUDA_HOST    = 3,
  CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED = 13,
} cudf_arrow_device_type_t;

typedef enum cudf_arrow_device_export_materialization_t {
  CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW   = 0,
  CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY = 1,
} cudf_arrow_device_export_materialization_t;

typedef enum cudf_arrow_device_import_materialization_t {
  CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW    = 0,
  CUDF_ARROW_DEVICE_IMPORT_BRIDGE_OWNED_DEVICE_COPY = 1,
} cudf_arrow_device_import_materialization_t;

/// Export one borrowed cudf column view as Arrow C Device ABI.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned ArrowDeviceArray when bridge-owned device storage is reported.
///
/// This slice supports recursive fixed-width (including decimal), STRING,
/// LIST, STRUCT, and DICTIONARY32 column layouts.
/// Any `BOOL8` leaf exports through a bridge-owned device copy because Arrow
/// uses bitmap layout. Zero-row STRING leaves also export through bridge-owned
/// device storage because Arrow requires a length + 1 offsets buffer.
cudf_status_t cudf_arrow_device_column_export(
  cudf_column_view_handle view,
  cudf_exec_options_t const* options,
  cudf_arrow_schema_handle* out_schema,
  cudf_arrow_device_array_handle* out_array,
  cudf_arrow_device_export_materialization_t* out_materialization);

/// Export one borrowed cudf table view as Arrow C Device ABI.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned ArrowDeviceArray when bridge-owned device storage is reported.
///
/// This slice supports tables whose top-level columns are all recursive
/// fixed-width (including decimal), STRING, LIST, STRUCT, or DICTIONARY32
/// layouts. Any `BOOL8` leaf exports through bridge-owned copied device
/// storage because Arrow uses bitmap layout. Zero-row STRING leaves also
/// export through bridge-owned device storage because Arrow requires a
/// length + 1 offsets buffer.
///
/// When `column_names` is non-NULL, it must point to `num_column_names`
/// top-level UTF-8 column names. Passing NULL with `num_column_names == 0`
/// exports unnamed top-level columns.
cudf_status_t cudf_arrow_device_table_export(
  cudf_table_view_handle view,
  cudf_exec_options_t const* options,
  char const* const* column_names,
  int32_t num_column_names,
  cudf_arrow_schema_handle* out_schema,
  cudf_arrow_device_array_handle* out_array,
  cudf_arrow_device_export_materialization_t* out_materialization);

/// Import one Arrow C Device ABI column into a cudf column handle using the
/// provided explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned column when bridge-owned device storage is materialized.
///
/// When `array->sync_event` is present, this shim first performs one
/// observable `cudaStreamWaitEvent` on the destination stream and then forwards
/// a temporary `sync_event = nullptr` copy to libcudf so ordering stays honest
/// without double-waiting.
cudf_status_t cudf_arrow_device_column_import(
  cudf_arrow_schema_handle schema,
  cudf_arrow_device_array_handle array,
  cudf_exec_options_t const* options,
  cudf_column_handle* out,
  cudf_arrow_device_import_materialization_t* out_materialization);

/// Import one Arrow C Device ABI table into a cudf table handle using the
/// provided explicit execution options.
///
/// `options` must be non-NULL and must name an explicit valid device, a
/// non-NULL stream, and a non-NULL memory resource. `options->mr` must outlive
/// the returned table when bridge-owned device storage is materialized.
///
/// When `array->sync_event` is present, this shim first performs one
/// observable `cudaStreamWaitEvent` on the destination stream and then forwards
/// a temporary `sync_event = nullptr` copy to libcudf so ordering stays honest
/// without double-waiting.
cudf_status_t cudf_arrow_device_table_import(
  cudf_arrow_schema_handle schema,
  cudf_arrow_device_array_handle array,
  cudf_exec_options_t const* options,
  cudf_table_handle* out,
  cudf_arrow_device_import_materialization_t* out_materialization);

/// Destroy one heap-owned ArrowSchema handle returned by this shim. Ownership
/// is consumed on call; failure quarantines the root and must not be retried.
cudf_status_t cudf_arrow_device_schema_destroy(cudf_arrow_schema_handle schema);

/// Destroy one heap-owned ArrowDeviceArray handle returned by this shim.
/// Ownership is consumed on call; failure quarantines the root.
cudf_status_t cudf_arrow_device_array_destroy(cudf_arrow_device_array_handle array);

/// Read the ArrowDeviceArray device id.
cudf_status_t cudf_arrow_device_array_device_id(cudf_arrow_device_array_handle array, int64_t* out);

/// Read the ArrowDeviceArray device type.
cudf_status_t cudf_arrow_device_array_device_type(cudf_arrow_device_array_handle array,
                                                  int32_t* out);

/// Report whether the ArrowDeviceArray currently carries a non-null sync_event.
cudf_status_t cudf_arrow_device_array_has_sync_event(cudf_arrow_device_array_handle array,
                                                     int32_t* out);

/// Borrow the raw ArrowDeviceArray sync_event pointer without transferring
/// ownership. The returned pointer stays valid for the lifetime of `array`.
cudf_status_t cudf_arrow_device_array_sync_event_raw(cudf_arrow_device_array_handle array,
                                                     void** out);

/// Return the logical fixed-width values buffer pointer for supported
/// first-slice column exports/imports.
cudf_status_t cudf_arrow_device_column_fixed_width_values_ptr(cudf_arrow_schema_handle schema,
                                                              cudf_arrow_device_array_handle array,
                                                              void const** out);

/// Return the number of top-level columns described by an Arrow table schema.
cudf_status_t cudf_arrow_device_table_num_columns(cudf_arrow_schema_handle schema, int64_t* out);

/// Return the borrowed top-level Arrow column name pointer for one table
/// schema child. The returned pointer stays valid for the lifetime of `schema`.
cudf_status_t cudf_arrow_device_table_column_name(cudf_arrow_schema_handle schema,
                                                  int64_t index,
                                                  char const** out);

/// Return the logical fixed-width values buffer pointer for one supported
/// first-slice top-level column in a table export/import.
cudf_status_t cudf_arrow_device_table_fixed_width_values_ptr(cudf_arrow_schema_handle schema,
                                                             cudf_arrow_device_array_handle array,
                                                             int64_t index,
                                                             void const** out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_ARROW_DEVICE_H
