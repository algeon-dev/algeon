#include "cudf_rust.h"

int cudf_rust_c_header_smoke(void)
{
  cudf_status_t (*f1)(cudf_stream_handle*) = &cudf_stream_create;
  void (*f2)(cudf_stream_handle)           = &cudf_stream_destroy;
  cudf_status_t (*f3)(cudf_memory_resource_handle*) = &cudf_memory_resource_create_cuda;
  void (*f4)(cudf_memory_resource_handle)           = &cudf_memory_resource_destroy;

  cudf_status_t (*f5)(cudf_type_id_t, cudf_data_type_handle*) = &cudf_data_type_create;
  void (*f6)(cudf_data_type_handle)                           = &cudf_data_type_destroy;

  cudf_status_t (*f7)(cudf_column_handle, cudf_column_view_handle*) = &cudf_column_view;
  void (*f8)(cudf_column_view_handle)                               = &cudf_column_view_destroy;

  cudf_status_t (*f9)(cudf_table_handle, cudf_table_view_handle*) = &cudf_table_view;
  void (*f10)(cudf_table_view_handle)                             = &cudf_table_view_destroy;

  cudf_status_t (*f11)(cudf_column_view_handle, cudf_type_id_t, cudf_column_handle*) =
    &cudf_dictionary_encode;
  cudf_status_t (*f12)(cudf_table_view_handle, uint32_t, cudf_column_handle*) =
    &cudf_hashing_murmurhash3_x86_32;
  cudf_status_t (*f13)(cudf_column_handle, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_copy_ex;
  cudf_status_t (*f14)(cudf_column_view_handle, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_copy_from_view_ex;
  cudf_status_t (*f15)(cudf_type_id_t, const void*, int32_t, const uint8_t*, int32_t, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_create_numeric_ex;
  cudf_status_t (*f16)(cudf_table_handle, const cudf_exec_options_t*, cudf_table_handle*) =
    &cudf_table_copy_ex;
  cudf_status_t (*f17)(cudf_column_view_handle, void*, size_t, const cudf_exec_options_t*) =
    &cudf_column_view_to_host_ex;
  cudf_status_t (*f18)(cudf_type_id_t, const void*, int32_t, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_create_from_device_data_ex;
  cudf_status_t (*f19)(cudf_column_view_handle, cudf_scalar_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_replace_nulls_scalar_ex;
  cudf_status_t (*f20)(cudf_column_view_handle, cudf_column_view_handle, cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_replace_nulls_column_ex;
  cudf_status_t (*f21)(cudf_table_view_handle, cudf_column_view_handle, cudf_table_handle*, const cudf_exec_options_t*) =
    &cudf_apply_boolean_mask_ex;
  cudf_status_t (*f22)(cudf_table_view_handle, cudf_column_view_handle, cudf_out_of_bounds_policy_t, cudf_table_handle*, const cudf_exec_options_t*) =
    &cudf_gather_ex;
  cudf_exec_options_t opts = CUDF_EXEC_OPTIONS_INITIALIZER;

  cudf_status_t (*f23)(size_t, cudf_device_buffer_handle*) = &cudf_device_buffer_create;
  void (*f24)(cudf_device_buffer_handle) = &cudf_device_buffer_destroy;
  cudf_status_t (*f25)(cudf_device_buffer_handle, void**) = &cudf_device_buffer_data;
  cudf_status_t (*f26)(cudf_device_buffer_handle, size_t*) = &cudf_device_buffer_size;
  cudf_status_t (*f27)(cudf_device_buffer_handle, const void*, size_t) = &cudf_device_buffer_copy_from_host;
  cudf_status_t (*f28)(cudf_type_id_t, cudf_device_buffer_handle, int32_t, cudf_column_handle*) =
    &cudf_column_adopt_device_buffer;
  cudf_status_t (*f29)(cudf_type_id_t, cudf_device_buffer_handle, cudf_device_buffer_handle,
                       int32_t, int32_t, cudf_column_handle*) =
    &cudf_column_adopt_device_buffer_nullable;

  (void)f1;
  (void)f2;
  (void)f3;
  (void)f4;
  (void)f5;
  (void)f6;
  (void)f7;
  (void)f8;
  (void)f9;
  (void)f10;
  (void)f11;
  (void)f12;
  (void)f13;
  (void)f14;
  (void)f15;
  (void)f16;
  (void)f17;
  (void)f18;
  (void)f19;
  (void)f20;
  (void)f21;
  (void)f22;
  (void)f23;
  (void)f24;
  (void)f25;
  (void)f26;
  (void)f27;
  (void)f28;
  (void)f29;
  (void)opts;
  return 0;
}
