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
  cudf_status_t (*f14b)(cudf_column_handle, size_t*) = &cudf_column_payload_size;
  cudf_status_t (*f14c)(cudf_column_handle, const cudf_exec_options_t*, size_t*) =
    &cudf_column_payload_size_ex;
  cudf_status_t (*f14d)(cudf_column_handle, size_t*) = &cudf_column_device_allocated_size;
  cudf_status_t (*f15)(cudf_type_id_t, const void*, int32_t, const uint8_t*, int32_t, const cudf_exec_options_t*, cudf_column_handle*) =
    &cudf_column_create_numeric_ex;
  cudf_status_t (*f16)(cudf_table_handle, const cudf_exec_options_t*, cudf_table_handle*) =
    &cudf_table_copy_ex;
  cudf_status_t (*f16b)(cudf_table_handle, size_t*) = &cudf_table_payload_size;
  cudf_status_t (*f16c)(cudf_table_handle, const cudf_exec_options_t*, size_t*) =
    &cudf_table_payload_size_ex;
  cudf_status_t (*f16d)(cudf_table_handle, size_t*) = &cudf_table_device_allocated_size;
  cudf_status_t (*f17)(cudf_column_view_handle, void*, size_t, const cudf_exec_options_t*) =
    &cudf_column_view_to_host_ex;
  cudf_status_t (*f17b)(cudf_column_view_handle, int32_t*, size_t, char*, size_t, int32_t*, const cudf_exec_options_t*) =
    &cudf_column_view_strings_to_host_ex;
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
  cudf_status_t (*f22b)(cudf_type_id_t, const void*, int32_t, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_create_numeric_ex;
  cudf_status_t (*f22c)(const char*, int32_t, const cudf_exec_options_t*, cudf_scalar_handle*) =
    &cudf_scalar_create_string_ex;
  cudf_status_t (*f22d)(cudf_scalar_handle, int32_t*, const cudf_exec_options_t*) =
    &cudf_scalar_is_valid_ex;
  cudf_status_t (*f22e)(cudf_scalar_handle, void*, const cudf_exec_options_t*) =
    &cudf_scalar_value_ex;
  cudf_status_t (*f22f)(cudf_scalar_handle, char**, int32_t*, const cudf_exec_options_t*) =
    &cudf_scalar_string_value_ex;
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

  /* strings.h */
  cudf_status_t (*f30)(cudf_column_view_handle, cudf_column_handle*) =
    &cudf_strings_to_lower;
  cudf_status_t (*f31)(cudf_column_view_handle, cudf_column_handle*,
                       const cudf_exec_options_t*) =
    &cudf_strings_to_lower_ex;
  cudf_status_t (*f32)(cudf_column_view_handle, cudf_column_handle*) =
    &cudf_strings_to_upper;
  cudf_status_t (*f33)(cudf_column_view_handle, cudf_column_handle*,
                       const cudf_exec_options_t*) =
    &cudf_strings_to_upper_ex;
  cudf_status_t (*f34)(cudf_column_view_handle, cudf_scalar_handle,
                       cudf_column_handle*) =
    &cudf_strings_contains;
  cudf_status_t (*f35)(cudf_column_view_handle, cudf_scalar_handle,
                       cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_contains_ex;
  cudf_status_t (*f36)(cudf_column_view_handle, cudf_scalar_handle,
                       cudf_column_handle*) =
    &cudf_strings_starts_with;
  cudf_status_t (*f37)(cudf_column_view_handle, cudf_scalar_handle,
                       cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_starts_with_ex;
  cudf_status_t (*f38)(cudf_column_view_handle, cudf_scalar_handle,
                       cudf_column_handle*) =
    &cudf_strings_ends_with;
  cudf_status_t (*f39)(cudf_column_view_handle, cudf_scalar_handle,
                       cudf_column_handle*, const cudf_exec_options_t*) =
    &cudf_strings_ends_with_ex;
  cudf_status_t (*f40)(cudf_column_view_handle, cudf_strip_side_t,
                       cudf_scalar_handle, cudf_column_handle*) =
    &cudf_strings_strip;
  cudf_status_t (*f41)(cudf_column_view_handle, cudf_strip_side_t,
                       cudf_scalar_handle, cudf_column_handle*,
                       const cudf_exec_options_t*) =
    &cudf_strings_strip_ex;
  cudf_status_t (*f42)(cudf_column_view_handle, cudf_column_handle*) =
    &cudf_strings_count_characters;
  cudf_status_t (*f43)(cudf_column_view_handle, cudf_column_handle*,
                       const cudf_exec_options_t*) =
    &cudf_strings_count_characters_ex;
  cudf_status_t (*f44)(cudf_column_view_handle, cudf_column_handle*) =
    &cudf_strings_count_bytes;
  cudf_status_t (*f45)(cudf_column_view_handle, cudf_column_handle*,
                       const cudf_exec_options_t*) =
    &cudf_strings_count_bytes_ex;

  /* device.h */
  cudf_status_t (*f46)(int32_t*) = &cudf_cuda_get_device;
  cudf_status_t (*f47)(int32_t)  = &cudf_cuda_set_device;
  cudf_status_t (*f48)(int32_t*) = &cudf_cuda_get_device_count;

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
  (void)f14b;
  (void)f14c;
  (void)f14d;
  (void)f15;
  (void)f16;
  (void)f16b;
  (void)f16c;
  (void)f16d;
  (void)f17;
  (void)f17b;
  (void)f18;
  (void)f19;
  (void)f20;
  (void)f21;
  (void)f22;
  (void)f22b;
  (void)f22c;
  (void)f22d;
  (void)f22e;
  (void)f22f;
  (void)f23;
  (void)f24;
  (void)f25;
  (void)f26;
  (void)f27;
  (void)f28;
  (void)f29;
  (void)f30;
  (void)f31;
  (void)f32;
  (void)f33;
  (void)f34;
  (void)f35;
  (void)f36;
  (void)f37;
  (void)f38;
  (void)f39;
  (void)f40;
  (void)f41;
  (void)f42;
  (void)f43;
  (void)f44;
  (void)f45;
  (void)f46;
  (void)f47;
  (void)f48;
  (void)opts;
  return 0;
}
