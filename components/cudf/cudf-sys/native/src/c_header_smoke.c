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
  cudf_status_t (*f14)(cudf_column_view_handle, void*, size_t, const cudf_exec_options_t*) =
    &cudf_column_view_to_host_ex;
  cudf_exec_options_t opts = CUDF_EXEC_OPTIONS_INITIALIZER;

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
  (void)opts;
  return 0;
}
