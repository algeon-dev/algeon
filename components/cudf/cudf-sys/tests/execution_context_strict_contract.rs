mod source_contract_support;

use source_contract_support::contains_normalized;

const EXECUTION_CONTEXT_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/execution_context.cpp"));
const ARROW_DEVICE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/arrow_device.cpp"));
const BINARY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/binary.h"));
const BINARY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/binary.cpp"));
const CONCATENATE_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/concatenate.h"));
const CONCATENATE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/concatenate.cpp"));
const COLUMN_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/column.h"));
const COLUMN_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/column.cpp"));
const COPYING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/copying.h"));
const COPYING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/copying.cpp"));
const DATETIME_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/datetime.h"));
const DATETIME_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/datetime.cpp"));
const DEVICE_BUFFER_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/device_buffer.h"));
const DEVICE_BUFFER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/device_buffer.cpp"));
const DICTIONARY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/dictionary.h"));
const DICTIONARY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/dictionary.cpp"));
const FILTER_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/filter.h"));
const FILTER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/filter.cpp"));
const GROUPBY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/groupby.h"));
const GROUPBY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/groupby.cpp"));
const HASHING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/hashing.h"));
const HASHING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/hashing.cpp"));
const IO_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/io.h"));
const IO_CSV_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_csv.cpp"));
const IO_JSON_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_json.cpp"));
const IO_PARQUET_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_parquet.cpp"));
const JOIN_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/join.h"));
const JOIN_COMMON_HPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_common.hpp"));
const JOIN_HASH_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_hash.cpp"));
const JOIN_CONDITIONAL_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_conditional.cpp"));
const JOIN_FILTER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_filter.cpp"));
const JOIN_MIXED_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_mixed.cpp"));
const PARTITIONING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/partitioning.h"));
const PARTITIONING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/partitioning.cpp"));
const PREPARED_IMPORT_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/prepared_import.cpp"));
const REPLACE_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/replace.h"));
const REPLACE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/replace.cpp"));
const REDUCTION_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/reduction.h"));
const REDUCTION_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/reduction.cpp"));
const SEARCH_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/search.h"));
const SEARCH_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/search.cpp"));
const SCALAR_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/scalar.h"));
const SCALAR_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/scalar.cpp"));
const SORTING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/sorting.h"));
const SORTING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/sorting.cpp"));
const STREAM_COMPACTION_H: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/stream_compaction.h"));
const STREAM_COMPACTION_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/stream_compaction.cpp"));
const STRINGS_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/strings.h"));
const STRINGS_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/strings.cpp"));
const TABLE_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/table.h"));
const TABLE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/table.cpp"));
const UNARY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/unary.h"));
const UNARY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/unary.cpp"));
const BINARY_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/binary.rs"));
const CONCATENATE_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/concatenate.rs"));
const COLUMN_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/column.rs"));
const COPYING_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/copying.rs"));
const DATETIME_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/datetime.rs"));
const DEVICE_BUFFER_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/device_buffer.rs"));
const DICTIONARY_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/dictionary.rs"));
const FILTER_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/filter.rs"));
const GROUPBY_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/groupby.rs"));
const HASHING_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/hashing.rs"));
const IO_CSV_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/io/csv.rs"));
const IO_JSON_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/io/json.rs"));
const IO_PARQUET_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/io/parquet.rs"));
const JOIN_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/join.rs"));
const PARTITIONING_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/partitioning.rs"));
const REPLACE_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/replace.rs"));
const REDUCTION_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/reduction.rs"));
const SEARCH_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/search.rs"));
const SCALAR_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/scalar.rs"));
const SORTING_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/sorting.rs"));
const STREAM_COMPACTION_FFI_RS: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/stream_compaction.rs"));
const STRINGS_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/strings.rs"));
const TABLE_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/table.rs"));
const UNARY_FFI_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi/unary.rs"));

#[test]
fn required_exec_options_resolver_rejects_ambient_execution_and_does_not_install_current_mr() {
    let body = function_region(
        EXECUTION_CONTEXT_CPP,
        "cudf_resolve_required_exec_options",
        "void cudf_record_synchronized_stream",
    );

    for required_marker in [
        "options == nullptr",
        "CUDF_EXEC_FLAG_DEVICE_EXPLICIT) == 0",
        "options->stream == nullptr",
        "options->mr == nullptr",
        "options->reserved",
        "guard.activate(options->device)",
    ] {
        assert!(
            body.contains(required_marker),
            "strict resolver must validate `{required_marker}` before production execution"
        );
    }

    assert!(
        source_contract_support::contains_normalized(body, "record_exec_context((*out)->stream, (*out)->mr_raw);"),
        "strict resolver must observe the resolved stream and memory resource before production execution"
    );

    for forbidden_marker in [
        "cudf_make_exec_context",
        "get_current_device_resource",
        "get_current_device_resource_ref",
        "install_current_device_resource",
        "cudf::get_default_stream",
    ] {
        assert!(
            !body.contains(forbidden_marker),
            "strict resolver must not use ambient/default execution state via `{forbidden_marker}`"
        );
    }
}

#[test]
fn execution_context_memory_resource_ref_is_not_type_erased_through_any_resource() {
    assert!(
        !EXECUTION_CONTEXT_CPP.contains("cuda::mr::any_resource"),
        "execution-context memory-resource refs must be built directly from the raw RMM resource; \
         type-erasing through cuda::mr::any_resource makes copied refs depend on a separate \
         object lifetime"
    );
    assert!(
        !EXECUTION_CONTEXT_CPP.contains("device_memory_resource_view"),
        "execution-context code should rely on rmm::device_async_resource_ref's direct raw-resource \
         constructor so copied refs rebuild their own device_memory_resource_view"
    );
}

#[test]
fn binary_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_binary_operation_vv_ex", "cudf_status_t cudf_binary_operation_vs_ex"),
        ("cudf_binary_operation_vs_ex", "cudf_status_t cudf_binary_operation_sv_ex"),
        ("cudf_binary_operation_sv_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(BINARY_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn binary_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("binary native header", BINARY_H, "cudf_status_t cudf_binary_operation_vv("),
        ("binary native source", BINARY_CPP, "cudf_status_t cudf_binary_operation_vv("),
        ("binary Rust FFI", BINARY_FFI_RS, "pub fn cudf_binary_operation_vv("),
        ("binary native header", BINARY_H, "cudf_status_t cudf_binary_operation_vs("),
        ("binary native source", BINARY_CPP, "cudf_status_t cudf_binary_operation_vs("),
        ("binary Rust FFI", BINARY_FFI_RS, "pub fn cudf_binary_operation_vs("),
        ("binary native header", BINARY_H, "cudf_status_t cudf_binary_operation_sv("),
        ("binary native source", BINARY_CPP, "cudf_status_t cudf_binary_operation_sv("),
        ("binary Rust FFI", BINARY_FFI_RS, "pub fn cudf_binary_operation_sv("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_binary_operation_vv_ex(lhs, rhs, op, output_type, out, nullptr)",
        "cudf_binary_operation_vs_ex(lhs, rhs, op, output_type, out, nullptr)",
        "cudf_binary_operation_sv_ex(lhs, rhs, op, output_type, out, nullptr)",
    ] {
        assert!(
            !BINARY_CPP.contains(marker),
            "binary native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn copying_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_gather_ex", "cudf_status_t cudf_gather_device_allocated_size_ex"),
        ("cudf_gather_device_allocated_size_ex", "cudf_status_t cudf_scatter_ex"),
        ("cudf_scatter_ex", "cudf_status_t cudf_copy_if_else_vv_ex"),
        ("cudf_copy_if_else_vv_ex", "cudf_status_t cudf_shift_ex"),
        ("cudf_shift_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(COPYING_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn column_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("create_column_from_device_data_impl", "extern \"C\""),
        ("cudf_column_create_empty_ex", "cudf_status_t cudf_column_create_numeric_ex"),
        ("cudf_column_create_numeric_ex", "cudf_status_t cudf_column_create_fixed_width_from_host_ex"),
        ("cudf_column_create_fixed_width_from_host_ex", "cudf_status_t cudf_column_create_sequence_ex"),
        ("cudf_column_create_sequence_ex", "cudf_status_t cudf_column_copy_ex"),
        ("cudf_column_copy_ex", "cudf_status_t cudf_column_copy_from_view_ex"),
        ("cudf_column_copy_from_view_ex", "cudf_status_t cudf_column_copy_from_borrowed_view_ex"),
        ("cudf_column_copy_from_borrowed_view_ex", "cudf_status_t cudf_column_create_from_owned_device_transfer"),
        ("cudf_column_payload_size_ex", "cudf_status_t cudf_column_device_allocated_size"),
        ("cudf_column_view_null_count_ex", "cudf_status_t cudf_column_view_nullable"),
        ("cudf_column_view_list_sliced_child_ex", "cudf_status_t cudf_column_view_data_ptr"),
        ("cudf_column_view_string_chars_ptr_and_size_ex", "cudf_status_t cudf_column_view_null_mask_ptr"),
        ("cudf_column_view_slice_ex", "cudf_status_t cudf_column_view_to_host_ex"),
        ("cudf_column_view_to_host_ex", "cudf_column_view_null_mask_to_host_ex"),
        ("cudf_column_view_null_mask_to_host_ex", "cudf_status_t cudf_column_create_from_device_data("),
        ("cudf_column_create_strings_ex", "cudf_status_t cudf_column_view_strings_to_host_ex"),
        ("cudf_column_view_strings_to_host_ex", "cudf_status_t cudf_column_from_scalar_ex"),
        ("cudf_column_from_scalar_ex", "} // extern \"C\""),
    ] {
        let body = function_region_by_function_name(COLUMN_CPP, function, end_marker);
        assert!(
            contains_normalized(body, "cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn column_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_create_empty("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_create_empty("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_create_empty("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_create_numeric("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_create_numeric("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_create_numeric("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_create_fixed_width_from_host("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_create_fixed_width_from_host("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_create_fixed_width_from_host("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_create_sequence("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_create_sequence("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_create_sequence("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_copy("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_copy("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_copy("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_copy_from_view("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_copy_from_view("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_copy_from_view("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_copy_from_borrowed_view("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_copy_from_borrowed_view("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_copy_from_borrowed_view("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_payload_size("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_payload_size("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_payload_size("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_view_list_sliced_child("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_view_list_sliced_child("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_view_list_sliced_child("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_view_string_chars_ptr_and_size("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_view_string_chars_ptr_and_size("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_view_string_chars_ptr_and_size("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_view_slice("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_view_slice("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_view_slice("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_view_to_host("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_view_to_host("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_view_to_host("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_view_null_mask_to_host("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_view_null_mask_to_host("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_view_null_mask_to_host("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_create_strings("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_create_strings("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_create_strings("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_view_strings_to_host("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_view_strings_to_host("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_view_strings_to_host("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_from_scalar("),
        ("column native source", COLUMN_CPP, "cudf_status_t cudf_column_from_scalar("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_from_scalar("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_column_create_numeric_ex(type_id, data, size, null_mask, null_count, nullptr, out)",
        "cudf_column_create_fixed_width_from_host_ex(type_id, scale, data, size, null_mask, null_count, nullptr, out)",
        "cudf_column_create_sequence_ex(init, step, size, nullptr, out)",
        "cudf_column_copy_ex(src, nullptr, out)",
        "cudf_column_copy_from_view_ex(src, nullptr, out)",
        "cudf_column_copy_from_borrowed_view_ex(type_id, scale, size, offset, data, null_mask, null_count, nullptr, out)",
        "cudf_column_payload_size_ex(handle, nullptr, out)",
        "cudf_column_view_list_sliced_child_ex(handle, nullptr, out)",
        "cudf_column_view_string_chars_ptr_and_size_ex(handle, nullptr, out_ptr, out_size)",
        "cudf_column_view_slice_ex(handle, offset, length, nullptr, out)",
        "cudf_column_view_to_host_ex(handle, host_buf, buf_size, nullptr)",
        "cudf_column_view_null_mask_to_host_ex(handle, host_buf, buf_size, nullptr)",
        "cudf_column_create_strings_ex(strings, lengths, num_strings, null_mask, null_count, nullptr, out)",
        "cudf_column_view_strings_to_host_ex(handle, offsets_buf, offsets_buf_size, chars_buf, chars_buf_size, total_chars_bytes, nullptr)",
        "cudf_column_from_scalar_ex(scalar, size, out, nullptr)",
    ] {
        assert!(
            !contains_normalized(COLUMN_CPP, marker),
            "column native source must not delegate to strict `_ex` with null options via `{marker}`"
        );
    }
}

#[test]
fn unary_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_unary_operation_ex", "cudf_status_t cudf_round_half_up_ex"),
        ("cudf_round_half_up_ex", "cudf_status_t cudf_is_null_ex"),
        ("cudf_is_null_ex", "cudf_status_t cudf_is_valid_ex"),
        ("cudf_is_valid_ex", "cudf_status_t cudf_cast_ex"),
        ("cudf_cast_ex", "cudf_status_t cudf_is_supported_cast("),
        ("cudf_is_nan_ex", "cudf_status_t cudf_is_not_nan_ex"),
        ("cudf_is_not_nan_ex", "} // extern \"C\""),
    ] {
        let body = function_region(UNARY_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn unary_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("unary native header", UNARY_H, "cudf_status_t cudf_unary_operation("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_unary_operation("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_unary_operation("),
        ("unary native header", UNARY_H, "cudf_status_t cudf_round_half_up("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_round_half_up("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_round_half_up("),
        ("unary native header", UNARY_H, "cudf_status_t cudf_is_null("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_is_null("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_is_null("),
        ("unary native header", UNARY_H, "cudf_status_t cudf_is_valid("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_is_valid("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_is_valid("),
        ("unary native header", UNARY_H, "cudf_status_t cudf_cast("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_cast("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_cast("),
        ("unary native header", UNARY_H, "cudf_status_t cudf_is_nan("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_is_nan("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_is_nan("),
        ("unary native header", UNARY_H, "cudf_status_t cudf_is_not_nan("),
        ("unary native source", UNARY_CPP, "cudf_status_t cudf_is_not_nan("),
        ("unary Rust FFI", UNARY_FFI_RS, "pub fn cudf_is_not_nan("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_unary_operation_ex(input, op, out, nullptr)",
        "cudf_round_half_up_ex(input, decimal_places, out, nullptr)",
        "cudf_is_null_ex(input, out, nullptr)",
        "cudf_is_valid_ex(input, out, nullptr)",
        "cudf_cast_ex(input, output_type, out, nullptr)",
        "cudf_is_nan_ex(input, out, nullptr)",
        "cudf_is_not_nan_ex(input, out, nullptr)",
    ] {
        assert!(!UNARY_CPP.contains(marker), "unary native source must not delegate to strict `_ex` with null options");
    }
}

#[test]
fn strings_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_strings_to_lower_ex", "cudf_status_t cudf_strings_to_upper_ex"),
        ("cudf_strings_to_upper_ex", "cudf_status_t cudf_strings_contains_ex"),
        ("cudf_strings_contains_ex", "cudf_status_t cudf_strings_starts_with_ex"),
        ("cudf_strings_starts_with_ex", "cudf_status_t cudf_strings_ends_with_ex"),
        ("cudf_strings_ends_with_ex", "cudf_status_t cudf_strings_like_ex"),
        ("cudf_strings_like_ex", "cudf_status_t cudf_strings_replace_with_backrefs_ex"),
        ("cudf_strings_replace_with_backrefs_ex", "cudf_status_t cudf_strings_slice_ex"),
        ("cudf_strings_slice_ex", "cudf_status_t cudf_strings_concatenate_ex"),
        ("cudf_strings_concatenate_ex", "cudf_status_t cudf_strings_strip_ex"),
        ("cudf_strings_strip_ex", "cudf_status_t cudf_strings_count_characters_ex"),
        ("cudf_strings_count_characters_ex", "cudf_status_t cudf_strings_count_bytes_ex"),
        ("cudf_strings_count_bytes_ex", "cudf_status_t cudf_strings_try_to_integers_ex"),
        ("cudf_strings_try_to_integers_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(STRINGS_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn strings_duplicate_non_ex_production_apis_are_removed() {
    for function in [
        "cudf_strings_to_lower",
        "cudf_strings_to_upper",
        "cudf_strings_contains",
        "cudf_strings_starts_with",
        "cudf_strings_ends_with",
        "cudf_strings_like",
        "cudf_strings_replace_with_backrefs",
        "cudf_strings_slice",
        "cudf_strings_concatenate",
        "cudf_strings_strip",
        "cudf_strings_count_characters",
        "cudf_strings_count_bytes",
        "cudf_strings_try_to_integers",
    ] {
        for (name, source, marker) in [
            ("strings native header", STRINGS_H, format!("cudf_status_t {function}(")),
            ("strings native source", STRINGS_CPP, format!("cudf_status_t {function}(")),
            ("strings Rust FFI", STRINGS_FFI_RS, format!("pub fn {function}(")),
        ] {
            assert!(!source.contains(&marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
        }
    }

    for marker in [
        "cudf_strings_to_lower_ex(input, out, nullptr)",
        "cudf_strings_to_upper_ex(input, out, nullptr)",
        "cudf_strings_contains_ex(input, target, out, nullptr)",
        "cudf_strings_starts_with_ex(input, target, out, nullptr)",
        "cudf_strings_ends_with_ex(input, target, out, nullptr)",
        "input, pattern, pattern_len, escape_character, escape_character_len, out, nullptr",
        "input, pattern, pattern_len, replacement, replacement_len, out, nullptr",
        "cudf_strings_slice_ex(input, start, start_valid, stop, stop_valid, step, out, nullptr)",
        "cudf_strings_concatenate_ex(columns, num_columns, out, nullptr)",
        "cudf_strings_strip_ex(input, side, to_strip, out, nullptr)",
        "cudf_strings_count_characters_ex(input, out, nullptr)",
        "cudf_strings_count_bytes_ex(input, out, nullptr)",
        "cudf_strings_try_to_integers_ex(input, output_type, out, nullptr)",
    ] {
        assert!(
            !STRINGS_CPP.contains(marker),
            "strings native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn copying_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("copying native header", COPYING_H, "cudf_status_t cudf_gather("),
        ("copying native source", COPYING_CPP, "cudf_status_t cudf_gather("),
        ("copying Rust FFI", COPYING_FFI_RS, "pub fn cudf_gather("),
        ("copying native header", COPYING_H, "cudf_status_t cudf_scatter("),
        ("copying native source", COPYING_CPP, "cudf_status_t cudf_scatter("),
        ("copying Rust FFI", COPYING_FFI_RS, "pub fn cudf_scatter("),
        ("copying native header", COPYING_H, "cudf_status_t cudf_copy_if_else_vv("),
        ("copying native source", COPYING_CPP, "cudf_status_t cudf_copy_if_else_vv("),
        ("copying Rust FFI", COPYING_FFI_RS, "pub fn cudf_copy_if_else_vv("),
        ("copying native header", COPYING_H, "cudf_status_t cudf_shift("),
        ("copying native source", COPYING_CPP, "cudf_status_t cudf_shift("),
        ("copying Rust FFI", COPYING_FFI_RS, "pub fn cudf_shift("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_gather_ex(input, gather_map, bounds_policy, out, nullptr)",
        "cudf_scatter_ex(source, scatter_map, target, out, nullptr)",
        "cudf_copy_if_else_vv_ex(lhs, rhs, boolean_mask, out, nullptr)",
        "cudf_shift_ex(input, offset, fill_value, out, nullptr)",
    ] {
        assert!(
            !COPYING_CPP.contains(marker),
            "copying native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn groupby_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_groupby_aggregate_ex", "cudf_status_t cudf_groupby_scan_ex"),
        ("cudf_groupby_scan_ex", "cudf_status_t cudf_groupby_shift_ex"),
        ("cudf_groupby_shift_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(GROUPBY_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn groupby_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("groupby native header", GROUPBY_H, "cudf_status_t cudf_groupby_aggregate("),
        ("groupby native source", GROUPBY_CPP, "cudf_status_t cudf_groupby_aggregate("),
        ("groupby Rust FFI", GROUPBY_FFI_RS, "pub fn cudf_groupby_aggregate("),
        ("groupby native header", GROUPBY_H, "cudf_status_t cudf_groupby_scan("),
        ("groupby native source", GROUPBY_CPP, "cudf_status_t cudf_groupby_scan("),
        ("groupby Rust FFI", GROUPBY_FFI_RS, "pub fn cudf_groupby_scan("),
        ("groupby native header", GROUPBY_H, "cudf_status_t cudf_groupby_shift("),
        ("groupby native source", GROUPBY_CPP, "cudf_status_t cudf_groupby_shift("),
        ("groupby Rust FFI", GROUPBY_FFI_RS, "pub fn cudf_groupby_shift("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "out_total_columns, nullptr",
        "cudf_groupby_shift_ex(keys, values, offsets, fill_values, num_value_columns, keys_null_policy, keys_are_sorted, out_keys, out_values, nullptr)",
    ] {
        assert!(
            !GROUPBY_CPP.contains(marker),
            "groupby native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn hash_join_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_inner_join_ex", "cudf_status_t cudf_left_join_ex"),
        ("cudf_left_join_ex", "cudf_status_t cudf_full_join_ex"),
        ("cudf_full_join_ex", "cudf_status_t cudf_cross_join_ex"),
        ("cudf_cross_join_ex", "cudf_status_t cudf_projected_cross_join_ex"),
        ("cudf_projected_cross_join_ex", "cudf_status_t cudf_projected_cross_join_device_allocated_size_ex"),
        ("cudf_projected_cross_join_device_allocated_size_ex", "cudf_status_t cudf_inner_join_indices_ex"),
        ("cudf_inner_join_indices_ex", "cudf_status_t cudf_left_join_indices_ex"),
        ("cudf_left_join_indices_ex", "cudf_status_t cudf_full_join_indices_ex"),
        ("cudf_full_join_indices_ex", "cudf_status_t cudf_left_semi_join_indices_ex"),
        ("cudf_left_semi_join_indices_ex", "cudf_status_t cudf_left_anti_join_indices_ex"),
        ("cudf_left_anti_join_indices_ex", "cudf_status_t cudf_distinct_hash_join_create_ex"),
        ("cudf_distinct_hash_join_create_ex", "void cudf_distinct_hash_join_destroy"),
        ("cudf_distinct_hash_join_left_join_ex", "cudf_status_t cudf_hash_join_create_ex"),
        ("cudf_hash_join_create_ex", "cudf_status_t cudf_hash_join_pre_build_reservation_size_ex"),
        ("cudf_hash_join_pre_build_reservation_size_ex", "void cudf_hash_join_destroy"),
        ("cudf_hash_join_inner_join_ex", "cudf_status_t cudf_hash_join_inner_join_with_output_size_ex"),
        ("cudf_hash_join_inner_join_with_output_size_ex", "cudf_status_t cudf_hash_join_inner_join_size_ex"),
        ("cudf_hash_join_left_join_ex", "cudf_status_t cudf_hash_join_left_join_with_output_size_ex"),
        ("cudf_hash_join_left_join_with_output_size_ex", "cudf_status_t cudf_hash_join_left_join_size_ex"),
        ("cudf_hash_join_full_join_ex", "cudf_status_t cudf_hash_join_full_join_with_output_size_ex"),
        ("cudf_hash_join_full_join_with_output_size_ex", "cudf_status_t cudf_hash_join_full_join_size_ex"),
    ] {
        let body = function_region(JOIN_HASH_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }

    let body = function_region(
        JOIN_COMMON_HPP,
        "inline cudf_status_t hash_join_size_impl",
        "inline cudf_status_t filtered_join_create_impl",
    );
    assert!(
        body.contains("cudf_resolve_required_exec_options(options, &exec)"),
        "hash_join_size_impl must use the strict execution-options resolver"
    );
    assert!(
        !body.contains("cudf_resolve_exec_options(options"),
        "hash_join_size_impl must not accept permissive/ambient execution options"
    );
    assert!(
        !body.contains("cudf_make_exec_context"),
        "hash_join_size_impl must not seed execution from ambient defaults"
    );
}

#[test]
fn join_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("join native header", JOIN_H, "cudf_status_t cudf_inner_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_inner_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_inner_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_left_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_left_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_left_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_full_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_full_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_full_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_cross_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_cross_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_cross_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_projected_cross_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_projected_cross_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_projected_cross_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_inner_join_indices("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_inner_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_inner_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_left_join_indices("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_left_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_left_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_full_join_indices("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_full_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_full_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_left_semi_join_indices("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_left_semi_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_left_semi_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_left_anti_join_indices("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_left_anti_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_left_anti_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_distinct_hash_join_create("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_distinct_hash_join_create("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_distinct_hash_join_create("),
        ("join native header", JOIN_H, "cudf_status_t cudf_distinct_hash_join_left_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_distinct_hash_join_left_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_distinct_hash_join_left_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_create("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_create("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_create("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_inner_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_inner_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_inner_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_inner_join_with_output_size("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_inner_join_with_output_size("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_inner_join_with_output_size("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_inner_join_size("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_inner_join_size("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_inner_join_size("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_left_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_left_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_left_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_left_join_with_output_size("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_left_join_with_output_size("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_left_join_with_output_size("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_left_join_size("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_left_join_size("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_left_join_size("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_full_join("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_full_join("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_full_join("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_full_join_with_output_size("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_full_join_with_output_size("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_full_join_with_output_size("),
        ("join native header", JOIN_H, "cudf_status_t cudf_hash_join_full_join_size("),
        ("join hash source", JOIN_HASH_CPP, "cudf_status_t cudf_hash_join_full_join_size("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_hash_join_full_join_size("),
        ("join native header", JOIN_H, "cudf_status_t cudf_conditional_inner_join_indices("),
        ("join conditional source", JOIN_CONDITIONAL_CPP, "cudf_status_t cudf_conditional_inner_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_conditional_inner_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_conditional_left_join_indices("),
        ("join conditional source", JOIN_CONDITIONAL_CPP, "cudf_status_t cudf_conditional_left_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_conditional_left_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_conditional_full_join_indices("),
        ("join conditional source", JOIN_CONDITIONAL_CPP, "cudf_status_t cudf_conditional_full_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_conditional_full_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_conditional_left_semi_join_indices("),
        ("join conditional source", JOIN_CONDITIONAL_CPP, "cudf_status_t cudf_conditional_left_semi_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_conditional_left_semi_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_conditional_left_anti_join_indices("),
        ("join conditional source", JOIN_CONDITIONAL_CPP, "cudf_status_t cudf_conditional_left_anti_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_conditional_left_anti_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_filter_join_indices("),
        ("join filter source", JOIN_FILTER_CPP, "cudf_status_t cudf_filter_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_filter_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_mixed_inner_join_indices("),
        ("join mixed source", JOIN_MIXED_CPP, "cudf_status_t cudf_mixed_inner_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_mixed_inner_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_mixed_left_join_indices("),
        ("join mixed source", JOIN_MIXED_CPP, "cudf_status_t cudf_mixed_left_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_mixed_left_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_mixed_full_join_indices("),
        ("join mixed source", JOIN_MIXED_CPP, "cudf_status_t cudf_mixed_full_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_mixed_full_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_mixed_left_semi_join_indices("),
        ("join mixed source", JOIN_MIXED_CPP, "cudf_status_t cudf_mixed_left_semi_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_mixed_left_semi_join_indices("),
        ("join native header", JOIN_H, "cudf_status_t cudf_mixed_left_anti_join_indices("),
        ("join mixed source", JOIN_MIXED_CPP, "cudf_status_t cudf_mixed_left_anti_join_indices("),
        ("join Rust FFI", JOIN_FFI_RS, "pub fn cudf_mixed_left_anti_join_indices("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }
}

#[test]
fn join_common_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("conditional_pair_join_indices_impl", "inline cudf_status_t conditional_single_side_join_indices_impl"),
        ("conditional_single_side_join_indices_impl", "inline cudf_status_t conditional_join_size_impl"),
        ("conditional_join_size_impl", "inline cudf_status_t mixed_pair_join_indices_impl"),
        ("mixed_pair_join_indices_impl", "inline cudf_status_t conditional_full_join_size_data_impl"),
        (
            "conditional_full_join_size_data_impl",
            "inline cudf_status_t conditional_full_join_size_data_materialize_indices_impl",
        ),
        (
            "conditional_full_join_size_data_materialize_indices_impl",
            "inline cudf_status_t mixed_single_side_join_indices_impl",
        ),
        ("mixed_single_side_join_indices_impl", "inline cudf_status_t mixed_join_size_data_impl"),
        ("mixed_join_size_data_impl", "inline cudf_status_t mixed_join_size_data_materialize_indices_impl"),
        (
            "mixed_join_size_data_materialize_indices_impl",
            "inline cudf_status_t mixed_join_size_data_materialize_single_indices_impl",
        ),
        ("mixed_join_size_data_materialize_single_indices_impl", "inline cudf_status_t filter_join_indices_impl"),
        ("filter_join_indices_impl", "inline cudf_status_t hash_join_size_impl"),
        ("filtered_join_create_impl", "inline cudf_status_t filtered_join_size_impl"),
        ("filtered_join_size_impl", "inline cudf_status_t filtered_join_indices_with_output_size_impl"),
        ("filtered_join_indices_with_output_size_impl", "inline cudf_status_t filtered_join_begin_probe_impl"),
        ("filtered_join_begin_probe_impl", "inline cudf_status_t filtered_join_probe_materialize_indices_impl"),
        ("filtered_join_probe_materialize_indices_impl", "}  // namespace cudf_rust::join"),
    ] {
        let body = function_region(JOIN_COMMON_HPP, &format!("inline cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn sorting_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("is_sorted_impl", "extern \"C\""),
        ("cudf_sort_ex", "cudf_status_t cudf_stable_sort_ex"),
        ("cudf_stable_sort_ex", "cudf_status_t cudf_sort_by_key_ex"),
        ("cudf_sort_by_key_ex", "cudf_status_t cudf_stable_sort_by_key_ex"),
        ("cudf_stable_sort_by_key_ex", "cudf_status_t cudf_sorted_order_ex"),
        ("cudf_sorted_order_ex", "cudf_status_t cudf_is_sorted_ex"),
    ] {
        let body = function_region_by_function_name(SORTING_CPP, function, end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn concatenate_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_concatenate_columns_ex", "cudf_status_t cudf_concatenate_tables_ex"),
        ("cudf_concatenate_tables_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(CONCATENATE_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn concatenate_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("concatenate native header", CONCATENATE_H, "cudf_status_t cudf_concatenate_columns("),
        ("concatenate native source", CONCATENATE_CPP, "cudf_status_t cudf_concatenate_columns("),
        ("concatenate Rust FFI", CONCATENATE_FFI_RS, "pub fn cudf_concatenate_columns("),
        ("concatenate native header", CONCATENATE_H, "cudf_status_t cudf_concatenate_tables("),
        ("concatenate native source", CONCATENATE_CPP, "cudf_status_t cudf_concatenate_tables("),
        ("concatenate Rust FFI", CONCATENATE_FFI_RS, "pub fn cudf_concatenate_tables("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_concatenate_columns_ex(columns, num_columns, nullptr, out)",
        "cudf_concatenate_tables_ex(tables, num_tables, nullptr, out)",
    ] {
        assert!(
            !CONCATENATE_CPP.contains(marker),
            "concatenate native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn sorting_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("sorting native header", SORTING_H, "cudf_status_t cudf_sort("),
        ("sorting native source", SORTING_CPP, "cudf_status_t cudf_sort("),
        ("sorting Rust FFI", SORTING_FFI_RS, "pub fn cudf_sort("),
        ("sorting native header", SORTING_H, "cudf_status_t cudf_stable_sort("),
        ("sorting native source", SORTING_CPP, "cudf_status_t cudf_stable_sort("),
        ("sorting Rust FFI", SORTING_FFI_RS, "pub fn cudf_stable_sort("),
        ("sorting native header", SORTING_H, "cudf_status_t cudf_sort_by_key("),
        ("sorting native source", SORTING_CPP, "cudf_status_t cudf_sort_by_key("),
        ("sorting Rust FFI", SORTING_FFI_RS, "pub fn cudf_sort_by_key("),
        ("sorting native header", SORTING_H, "cudf_status_t cudf_stable_sort_by_key("),
        ("sorting native source", SORTING_CPP, "cudf_status_t cudf_stable_sort_by_key("),
        ("sorting Rust FFI", SORTING_FFI_RS, "pub fn cudf_stable_sort_by_key("),
        ("sorting native header", SORTING_H, "cudf_status_t cudf_sorted_order("),
        ("sorting native source", SORTING_CPP, "cudf_status_t cudf_sorted_order("),
        ("sorting Rust FFI", SORTING_FFI_RS, "pub fn cudf_sorted_order("),
        ("sorting native header", SORTING_H, "cudf_status_t cudf_is_sorted("),
        ("sorting native source", SORTING_CPP, "cudf_status_t cudf_is_sorted("),
        ("sorting Rust FFI", SORTING_FFI_RS, "pub fn cudf_is_sorted("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_sort_ex(input, column_order, null_precedence, num_keys, out, nullptr)",
        "cudf_stable_sort_ex(input, column_order, null_precedence, num_keys, out, nullptr)",
        "cudf_sort_by_key_ex(values, keys, column_order, null_precedence, num_key_columns, out, nullptr)",
        "cudf_stable_sort_by_key_ex(values, keys, column_order, null_precedence, num_key_columns, out, nullptr)",
        "cudf_sorted_order_ex(input, column_order, null_precedence, num_keys, out, nullptr)",
        "is_sorted_impl(input, column_order, null_precedence, num_keys, out, nullptr)",
    ] {
        assert!(
            !SORTING_CPP.contains(marker),
            "sorting native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn stream_compaction_ex_entries_use_required_exec_options() {
    for (function, end_marker) in
        [("cudf_drop_nulls_ex", "cudf_status_t cudf_distinct_ex"), ("cudf_distinct_ex", "}  // extern \"C\"")]
    {
        let body = function_region(STREAM_COMPACTION_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn stream_compaction_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("stream compaction native header", STREAM_COMPACTION_H, "cudf_status_t cudf_drop_nulls("),
        ("stream compaction native source", STREAM_COMPACTION_CPP, "cudf_status_t cudf_drop_nulls("),
        ("stream compaction Rust FFI", STREAM_COMPACTION_FFI_RS, "pub fn cudf_drop_nulls("),
        ("stream compaction native header", STREAM_COMPACTION_H, "cudf_status_t cudf_distinct("),
        ("stream compaction native source", STREAM_COMPACTION_CPP, "cudf_status_t cudf_distinct("),
        ("stream compaction Rust FFI", STREAM_COMPACTION_FFI_RS, "pub fn cudf_distinct("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_drop_nulls_ex(input, keys, num_keys, keep_threshold, out, nullptr)",
        "cudf_distinct_ex(input, keys, num_keys, keep, null_equality, nan_equality, out, nullptr)",
    ] {
        assert!(
            !STREAM_COMPACTION_CPP.contains(marker),
            "stream compaction native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn filter_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_apply_boolean_mask_ex", "cudf_status_t cudf_filter_rows_by_predicate_ex"),
        ("cudf_filter_rows_by_predicate_ex", "void cudf_test_row_predicate_reset_observation("),
    ] {
        let body = function_region(FILTER_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn filter_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("filter native header", FILTER_H, "cudf_status_t cudf_apply_boolean_mask("),
        ("filter native source", FILTER_CPP, "cudf_status_t cudf_apply_boolean_mask("),
        ("filter Rust FFI", FILTER_FFI_RS, "pub fn cudf_apply_boolean_mask("),
        ("filter native header", FILTER_H, "cudf_status_t cudf_filter_rows_by_predicate("),
        ("filter native source", FILTER_CPP, "cudf_status_t cudf_filter_rows_by_predicate("),
        ("filter Rust FFI", FILTER_FFI_RS, "pub fn cudf_filter_rows_by_predicate("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_apply_boolean_mask_ex(input, mask, out, nullptr)",
        "cudf_filter_rows_by_predicate_ex(input, nodes, num_nodes, root_index, out, nullptr)",
    ] {
        assert!(
            !FILTER_CPP.contains(marker),
            "filter native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn hashing_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_hashing_murmurhash3_x86_32_ex", "cudf_status_t cudf_hashing_xxhash_64_ex"),
        ("cudf_hashing_xxhash_64_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(HASHING_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn search_ex_entries_use_required_exec_options() {
    let body = function_region(SEARCH_CPP, "cudf_status_t cudf_contains_ex", "}  // extern \"C\"");
    assert!(
        body.contains("cudf_resolve_required_exec_options(options, &exec)"),
        "cudf_contains_ex must use the strict execution-options resolver"
    );
    assert!(
        !body.contains("cudf_resolve_exec_options(options"),
        "cudf_contains_ex must not accept permissive/ambient execution options"
    );
    assert!(!body.contains("cudf_make_exec_context"), "cudf_contains_ex must not seed execution from ambient defaults");
}

#[test]
fn search_and_hashing_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("search native header", SEARCH_H, "cudf_status_t cudf_contains("),
        ("search native source", SEARCH_CPP, "cudf_status_t cudf_contains("),
        ("search Rust FFI", SEARCH_FFI_RS, "pub fn cudf_contains("),
        ("hashing native header", HASHING_H, "cudf_status_t cudf_hashing_murmurhash3_x86_32("),
        ("hashing native source", HASHING_CPP, "cudf_status_t cudf_hashing_murmurhash3_x86_32("),
        ("hashing Rust FFI", HASHING_FFI_RS, "pub fn cudf_hashing_murmurhash3_x86_32("),
        ("hashing native header", HASHING_H, "cudf_status_t cudf_hashing_xxhash_64("),
        ("hashing native source", HASHING_CPP, "cudf_status_t cudf_hashing_xxhash_64("),
        ("hashing Rust FFI", HASHING_FFI_RS, "pub fn cudf_hashing_xxhash_64("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for (name, source, marker) in [
        ("search native source", SEARCH_CPP, "cudf_contains_ex(haystack, needles, out, nullptr)"),
        ("hashing native source", HASHING_CPP, "cudf_hashing_murmurhash3_x86_32_ex(input, seed, out, nullptr)"),
        ("hashing native source", HASHING_CPP, "cudf_hashing_xxhash_64_ex(input, seed, out, nullptr)"),
    ] {
        assert!(!source.contains(marker), "{name} must not delegate to strict `_ex` with null options");
    }
}

#[test]
fn replace_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_replace_nulls_scalar_ex", "cudf_status_t cudf_replace_nulls_column_ex"),
        ("cudf_replace_nulls_column_ex", "cudf_status_t cudf_replace_nulls_policy_ex"),
        ("cudf_replace_nulls_policy_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(REPLACE_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn replace_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("replace native header", REPLACE_H, "cudf_status_t cudf_replace_nulls_scalar("),
        ("replace native source", REPLACE_CPP, "cudf_status_t cudf_replace_nulls_scalar("),
        ("replace Rust FFI", REPLACE_FFI_RS, "pub fn cudf_replace_nulls_scalar("),
        ("replace native header", REPLACE_H, "cudf_status_t cudf_replace_nulls_column("),
        ("replace native source", REPLACE_CPP, "cudf_status_t cudf_replace_nulls_column("),
        ("replace Rust FFI", REPLACE_FFI_RS, "pub fn cudf_replace_nulls_column("),
        ("replace native header", REPLACE_H, "cudf_status_t cudf_replace_nulls_policy("),
        ("replace native source", REPLACE_CPP, "cudf_status_t cudf_replace_nulls_policy("),
        ("replace Rust FFI", REPLACE_FFI_RS, "pub fn cudf_replace_nulls_policy("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_replace_nulls_scalar_ex(input, replacement, out, nullptr)",
        "cudf_replace_nulls_column_ex(input, replacement, out, nullptr)",
        "cudf_replace_nulls_policy_ex(input, policy, out, nullptr)",
    ] {
        assert!(
            !REPLACE_CPP.contains(marker),
            "replace native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn partitioning_ex_entries_use_required_exec_options() {
    let body = function_region(PARTITIONING_CPP, "cudf_status_t cudf_hash_partition_ex", "}  // extern \"C\"");
    assert!(
        body.contains("cudf_resolve_required_exec_options(options, &exec)"),
        "cudf_hash_partition_ex must use the strict execution-options resolver"
    );
    assert!(
        !body.contains("cudf_resolve_exec_options(options"),
        "cudf_hash_partition_ex must not accept permissive/ambient execution options"
    );
    assert!(
        !body.contains("cudf_make_exec_context"),
        "cudf_hash_partition_ex must not seed execution from ambient defaults"
    );
}

#[test]
fn partitioning_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("partitioning native header", PARTITIONING_H, "cudf_status_t cudf_hash_partition("),
        ("partitioning native source", PARTITIONING_CPP, "cudf_status_t cudf_hash_partition("),
        ("partitioning Rust FFI", PARTITIONING_FFI_RS, "pub fn cudf_hash_partition("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    assert!(
        !PARTITIONING_CPP.contains("cudf_hash_partition_ex(input,"),
        "partitioning native source must not delegate to strict `_ex` with null options"
    );
}

#[test]
fn datetime_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_datetime_extract_component_ex", "cudf_status_t cudf_datetime_floor_ex"),
        ("cudf_datetime_floor_ex", "cudf_status_t cudf_datetime_add_calendrical_months_scalar_ex"),
        ("cudf_datetime_add_calendrical_months_scalar_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(DATETIME_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn datetime_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("datetime native header", DATETIME_H, "cudf_status_t cudf_datetime_extract_component("),
        ("datetime native source", DATETIME_CPP, "cudf_status_t cudf_datetime_extract_component("),
        ("datetime Rust FFI", DATETIME_FFI_RS, "pub fn cudf_datetime_extract_component("),
        ("datetime native header", DATETIME_H, "cudf_status_t cudf_datetime_floor("),
        ("datetime native source", DATETIME_CPP, "cudf_status_t cudf_datetime_floor("),
        ("datetime Rust FFI", DATETIME_FFI_RS, "pub fn cudf_datetime_floor("),
        ("datetime native header", DATETIME_H, "cudf_status_t cudf_datetime_add_calendrical_months_scalar("),
        ("datetime native source", DATETIME_CPP, "cudf_status_t cudf_datetime_add_calendrical_months_scalar("),
        ("datetime Rust FFI", DATETIME_FFI_RS, "pub fn cudf_datetime_add_calendrical_months_scalar("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_datetime_extract_component_ex(input, component, out, nullptr)",
        "cudf_datetime_floor_ex(input, frequency, out, nullptr)",
        "cudf_datetime_add_calendrical_months_scalar_ex(input, months, out, nullptr)",
    ] {
        assert!(
            !DATETIME_CPP.contains(marker),
            "datetime native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn reduction_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_reduce_ex", "cudf_status_t cudf_scan_ex"),
        ("cudf_scan_ex", "cudf_status_t cudf_minmax_ex"),
        ("cudf_minmax_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(REDUCTION_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn reduction_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("reduction native header", REDUCTION_H, "cudf_status_t cudf_reduce("),
        ("reduction native source", REDUCTION_CPP, "cudf_status_t cudf_reduce("),
        ("reduction Rust FFI", REDUCTION_FFI_RS, "pub fn cudf_reduce("),
        ("reduction native header", REDUCTION_H, "cudf_status_t cudf_scan("),
        ("reduction native source", REDUCTION_CPP, "cudf_status_t cudf_scan("),
        ("reduction Rust FFI", REDUCTION_FFI_RS, "pub fn cudf_scan("),
        ("reduction native header", REDUCTION_H, "cudf_status_t cudf_minmax("),
        ("reduction native source", REDUCTION_CPP, "cudf_status_t cudf_minmax("),
        ("reduction Rust FFI", REDUCTION_FFI_RS, "pub fn cudf_minmax("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_reduce_ex(input, agg, output_type, out, nullptr)",
        "cudf_scan_ex(input, agg, scan_type, null_policy, out, nullptr)",
        "cudf_minmax_ex(input, out_min, out_max, nullptr)",
    ] {
        assert!(
            !REDUCTION_CPP.contains(marker),
            "reduction native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn dictionary_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_dictionary_encode_ex", "cudf_status_t cudf_dictionary_decode_ex"),
        ("cudf_dictionary_decode_ex", "cudf_status_t cudf_dictionary_keys("),
        ("cudf_dictionary_set_keys_ex", "cudf_status_t cudf_dictionary_match_dictionaries_ex"),
        ("cudf_dictionary_match_dictionaries_ex", "}  // extern \"C\""),
    ] {
        let body = function_region(DICTIONARY_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn dictionary_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("dictionary native header", DICTIONARY_H, "cudf_status_t cudf_dictionary_encode("),
        ("dictionary native source", DICTIONARY_CPP, "cudf_status_t cudf_dictionary_encode("),
        ("dictionary Rust FFI", DICTIONARY_FFI_RS, "pub fn cudf_dictionary_encode("),
        ("dictionary native header", DICTIONARY_H, "cudf_status_t cudf_dictionary_decode("),
        ("dictionary native source", DICTIONARY_CPP, "cudf_status_t cudf_dictionary_decode("),
        ("dictionary Rust FFI", DICTIONARY_FFI_RS, "pub fn cudf_dictionary_decode("),
        ("dictionary native header", DICTIONARY_H, "cudf_status_t cudf_dictionary_set_keys("),
        ("dictionary native source", DICTIONARY_CPP, "cudf_status_t cudf_dictionary_set_keys("),
        ("dictionary Rust FFI", DICTIONARY_FFI_RS, "pub fn cudf_dictionary_set_keys("),
        ("dictionary native header", DICTIONARY_H, "cudf_status_t cudf_dictionary_match_dictionaries("),
        ("dictionary native source", DICTIONARY_CPP, "cudf_status_t cudf_dictionary_match_dictionaries("),
        ("dictionary Rust FFI", DICTIONARY_FFI_RS, "pub fn cudf_dictionary_match_dictionaries("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_dictionary_encode_ex(input, indices_type_id, out, nullptr)",
        "cudf_dictionary_decode_ex(dict_column, out, nullptr)",
        "cudf_dictionary_set_keys_ex(dict_column, keys, out, nullptr)",
        "dict_columns, num_columns, out_columns, out_count, nullptr",
    ] {
        assert!(
            !DICTIONARY_CPP.contains(marker),
            "dictionary native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn is_sorted_ex_is_the_strict_production_entry() {
    let body = function_region_by_function_name(SORTING_CPP, "cudf_is_sorted_ex", "} // extern \"C\"");

    assert!(
        source_contract_support::contains_normalized(
            body,
            "return is_sorted_impl(input, column_order, null_precedence, num_keys, out, options);"
        ),
        "cudf_is_sorted_ex must require strict execution options"
    );
}

#[test]
fn io_ex_entries_use_required_exec_options() {
    for (source, function, end_marker) in [
        (IO_CSV_CPP, "cudf_io_read_csv_ex", "cudf_status_t cudf_io_write_csv_ex"),
        (IO_CSV_CPP, "cudf_io_write_csv_ex", "}  // extern \"C\""),
        (IO_JSON_CPP, "cudf_io_read_json_ex", "cudf_status_t cudf_io_read_json_with_metadata_ex"),
        (IO_JSON_CPP, "cudf_io_read_json_with_metadata_ex", "cudf_status_t cudf_io_write_json_ex"),
        (IO_JSON_CPP, "cudf_io_write_json_ex", "}  // extern \"C\""),
        (IO_PARQUET_CPP, "cudf_io_read_parquet_ex", "cudf_status_t cudf_io_read_parquet_from_host_buffer_ex"),
        (IO_PARQUET_CPP, "cudf_io_read_parquet_from_host_buffer_ex", "cudf_status_t cudf_io_write_parquet_ex"),
        (IO_PARQUET_CPP, "cudf_io_write_parquet_ex", "cudf_status_t cudf_io_read_parquet_with_metadata_ex"),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_with_metadata_ex",
            "cudf_status_t cudf_io_read_parquet_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_remote_parquet_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_remote_parquet_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_parquet_multi_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_multi_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex",
            "cudf_status_t cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex",
        ),
        (
            IO_PARQUET_CPP,
            "cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex",
            "int32_t cudf_io_remote_parquet_kvikio_remote_io_enabled",
        ),
    ] {
        let body = function_region(source, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(exec_options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(exec_options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn remote_parquet_device_read_orders_external_device_write_after_requested_stream() {
    let async_body =
        function_region(IO_PARQUET_CPP, "std::future<size_t> device_read_async", "size_t device_read(size_t offset,");
    let sync_pointer_body =
        function_region(IO_PARQUET_CPP, "size_t device_read(size_t offset,", "std::unique_ptr<buffer> device_read");
    let owned_buffer_body =
        function_region(IO_PARQUET_CPP, "std::unique_ptr<buffer> device_read", "[[nodiscard]] size_t size()");

    assert!(
        source_contract_support::contains_normalized(
            async_body,
            "stream.synchronize(); cudf_record_synchronized_stream(stream); return handle_.pread(dst, read_size, offset, read_size);"
        ),
        "remote Parquet device_read_async must wait for prior requested-stream work before KvikIO writes into the supplied device pointer"
    );
    assert!(
        source_contract_support::contains_normalized(
            sync_pointer_body,
            "return device_read_async(offset, size, dst, stream).get();"
        ) && !sync_pointer_body.contains("handle_.pread"),
        "remote Parquet synchronous pointer device_read must delegate through synchronized device_read_async"
    );
    assert!(
        source_contract_support::contains_normalized(
            owned_buffer_body,
            "device_read(offset, read_size, reinterpret_cast<uint8_t*>(out.data()), stream);"
        ) && !owned_buffer_body.contains("handle_.pread"),
        "remote Parquet owned-buffer device_read must delegate through synchronized pointer device_read"
    );
}

#[test]
fn io_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_csv("),
        ("CSV native source", IO_CSV_CPP, "cudf_status_t cudf_io_read_csv("),
        ("CSV Rust FFI", IO_CSV_FFI_RS, "pub fn cudf_io_read_csv("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_write_csv("),
        ("CSV native source", IO_CSV_CPP, "cudf_status_t cudf_io_write_csv("),
        ("CSV Rust FFI", IO_CSV_FFI_RS, "pub fn cudf_io_write_csv("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_json("),
        ("JSON native source", IO_JSON_CPP, "cudf_status_t cudf_io_read_json("),
        ("JSON Rust FFI", IO_JSON_FFI_RS, "pub fn cudf_io_read_json("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_json_with_metadata("),
        ("JSON native source", IO_JSON_CPP, "cudf_status_t cudf_io_read_json_with_metadata("),
        ("JSON Rust FFI", IO_JSON_FFI_RS, "pub fn cudf_io_read_json_with_metadata("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_write_json("),
        ("JSON native source", IO_JSON_CPP, "cudf_status_t cudf_io_write_json("),
        ("JSON Rust FFI", IO_JSON_FFI_RS, "pub fn cudf_io_write_json("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_parquet("),
        ("Parquet native source", IO_PARQUET_CPP, "cudf_status_t cudf_io_read_parquet("),
        ("Parquet Rust FFI", IO_PARQUET_FFI_RS, "pub fn cudf_io_read_parquet("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_read_parquet_with_metadata("),
        ("Parquet native source", IO_PARQUET_CPP, "cudf_status_t cudf_io_read_parquet_with_metadata("),
        ("Parquet Rust FFI", IO_PARQUET_FFI_RS, "pub fn cudf_io_read_parquet_with_metadata("),
        ("IO native header", IO_H, "cudf_status_t cudf_io_write_parquet("),
        ("Parquet native source", IO_PARQUET_CPP, "cudf_status_t cudf_io_write_parquet("),
        ("Parquet Rust FFI", IO_PARQUET_FFI_RS, "pub fn cudf_io_write_parquet("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for (source_name, source, marker) in [
        ("CSV native source", IO_CSV_CPP, "cudf_io_read_csv_ex(filepath, delimiter, has_header, out, nullptr)"),
        (
            "CSV native source",
            IO_CSV_CPP,
            "cudf_io_write_csv_ex(table, filepath, column_names, num_columns, delimiter, include_header, nullptr)",
        ),
        ("JSON native source", IO_JSON_CPP, "cudf_io_read_json_ex(filepath, lines, out, nullptr)"),
        (
            "JSON native source",
            IO_JSON_CPP,
            "cudf_io_read_json_with_metadata_ex(filepath, lines, out_table, out_names, out_num_names, nullptr)",
        ),
        (
            "JSON native source",
            IO_JSON_CPP,
            "cudf_io_write_json_ex(table, filepath, column_names, num_columns, lines, include_nulls, nullptr)",
        ),
        (
            "Parquet native source",
            IO_PARQUET_CPP,
            "cudf_io_write_parquet_ex(table, filepath, column_names, num_columns, row_group_size_rows, dictionary_policy, nullptr)",
        ),
    ] {
        assert!(
            !contains_normalized(source, marker),
            "{source_name} must not delegate to strict `_ex` with null options via `{marker}`"
        );
    }
}

#[test]
fn device_buffer_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_device_buffer_create_ex", "void cudf_device_buffer_destroy"),
        ("cudf_device_buffer_copy_from_host_ex", "cudf_status_t cudf_column_adopt_device_buffer("),
    ] {
        let body = function_region_by_function_name(DEVICE_BUFFER_CPP, function, end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn device_buffer_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("device buffer native header", DEVICE_BUFFER_H, "cudf_status_t cudf_device_buffer_create("),
        ("device buffer native source", DEVICE_BUFFER_CPP, "cudf_status_t cudf_device_buffer_create("),
        ("device buffer Rust FFI", DEVICE_BUFFER_FFI_RS, "pub fn cudf_device_buffer_create("),
        ("device buffer native header", DEVICE_BUFFER_H, "cudf_device_buffer_copy_from_host("),
        ("device buffer native source", DEVICE_BUFFER_CPP, "cudf_device_buffer_copy_from_host("),
        ("device buffer Rust FFI", DEVICE_BUFFER_FFI_RS, "pub fn cudf_device_buffer_copy_from_host("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_device_buffer_create_ex(size, nullptr, out)",
        "cudf_device_buffer_copy_from_host_ex(handle, host_data, size, nullptr)",
    ] {
        assert!(
            !DEVICE_BUFFER_CPP.contains(marker),
            "device buffer native source must not delegate to strict `_ex` with null options"
        );
    }
}

#[test]
fn table_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_table_copy_ex", "cudf_status_t cudf_table_create_from_owned_device_transfer"),
        ("cudf_table_payload_size_ex", "cudf_status_t cudf_table_device_allocated_size"),
    ] {
        let body = function_region(TABLE_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn table_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("table native header", TABLE_H, "cudf_status_t cudf_table_copy("),
        ("table native source", TABLE_CPP, "cudf_status_t cudf_table_copy("),
        ("table Rust FFI", TABLE_FFI_RS, "pub fn cudf_table_copy("),
        ("table native header", TABLE_H, "cudf_status_t cudf_table_payload_size("),
        ("table native source", TABLE_CPP, "cudf_status_t cudf_table_payload_size("),
        ("table Rust FFI", TABLE_FFI_RS, "pub fn cudf_table_payload_size("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in ["cudf_table_copy_ex(src, nullptr, out)", "cudf_table_payload_size_ex(handle, nullptr, out)"] {
        assert!(!TABLE_CPP.contains(marker), "table native source must not delegate to strict `_ex` with null options");
    }
}

#[test]
fn prepared_import_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        (
            "cudf_column_copy_from_prepared_device_export_ex",
            "cudf_status_t cudf_table_copy_from_prepared_device_export_ex(",
        ),
        (
            "cudf_table_copy_from_prepared_device_export_ex",
            "cudf_status_t cudf_column_copy_from_prepared_host_export_ex(",
        ),
        (
            "cudf_column_copy_from_prepared_host_export_ex",
            "cudf_status_t cudf_table_copy_from_prepared_host_export_ex(",
        ),
        ("cudf_table_copy_from_prepared_host_export_ex", "} // extern \"C\""),
    ] {
        let body = function_region(PREPARED_IMPORT_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            contains_normalized(body, "cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn prepared_import_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_copy_from_prepared_device_export("),
        ("prepared import source", PREPARED_IMPORT_CPP, "cudf_status_t cudf_column_copy_from_prepared_device_export("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_copy_from_prepared_device_export("),
        ("table native header", TABLE_H, "cudf_status_t cudf_table_copy_from_prepared_device_export("),
        ("prepared import source", PREPARED_IMPORT_CPP, "cudf_status_t cudf_table_copy_from_prepared_device_export("),
        ("table Rust FFI", TABLE_FFI_RS, "pub fn cudf_table_copy_from_prepared_device_export("),
        ("column native header", COLUMN_H, "cudf_status_t cudf_column_copy_from_prepared_host_export("),
        ("prepared import source", PREPARED_IMPORT_CPP, "cudf_status_t cudf_column_copy_from_prepared_host_export("),
        ("column Rust FFI", COLUMN_FFI_RS, "pub fn cudf_column_copy_from_prepared_host_export("),
        ("table native header", TABLE_H, "cudf_status_t cudf_table_copy_from_prepared_host_export("),
        ("prepared import source", PREPARED_IMPORT_CPP, "cudf_status_t cudf_table_copy_from_prepared_host_export("),
        ("table Rust FFI", TABLE_FFI_RS, "pub fn cudf_table_copy_from_prepared_host_export("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }
}

#[test]
fn arrow_device_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_arrow_device_column_export", "cudf_status_t cudf_arrow_device_table_export"),
        ("cudf_arrow_device_table_export", "cudf_status_t cudf_arrow_device_column_import"),
        ("cudf_arrow_device_column_import", "cudf_status_t cudf_arrow_device_table_import"),
        ("cudf_arrow_device_table_import", "void cudf_arrow_device_schema_destroy"),
    ] {
        let body = function_region(ARROW_DEVICE_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn scalar_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_scalar_create_numeric_ex", "cudf_status_t cudf_scalar_create_decimal_raw_ex"),
        ("cudf_scalar_create_decimal_raw_ex", "cudf_status_t cudf_scalar_create_timestamp_raw_ex"),
        ("cudf_scalar_create_timestamp_raw_ex", "cudf_status_t cudf_scalar_create_duration_raw_ex"),
        ("cudf_scalar_create_duration_raw_ex", "cudf_status_t cudf_scalar_create_string_ex"),
        ("cudf_scalar_create_string_ex", "cudf_status_t cudf_scalar_copy_ex"),
        ("cudf_scalar_copy_ex", "void cudf_scalar_destroy"),
        ("cudf_scalar_is_valid_ex", "cudf_status_t cudf_scalar_value_ex"),
        ("cudf_scalar_value_ex", "cudf_status_t cudf_scalar_string_value_ex"),
        ("cudf_scalar_string_value_ex", "void cudf_string_free"),
    ] {
        let body = function_region(SCALAR_CPP, &format!("cudf_status_t {function}"), end_marker);
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(
            !body.contains("cudf_resolve_exec_options(options"),
            "{function} must not accept permissive/ambient execution options"
        );
        assert!(!body.contains("cudf_make_exec_context"), "{function} must not seed execution from ambient defaults");
    }
}

#[test]
fn scalar_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_create_numeric("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_create_numeric("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_create_numeric("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_create_decimal_raw("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_create_decimal_raw("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_create_decimal_raw("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_create_timestamp_raw("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_create_timestamp_raw("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_create_timestamp_raw("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_create_duration_raw("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_create_duration_raw("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_create_duration_raw("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_create_string("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_create_string("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_create_string("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_copy("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_copy("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_copy("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_is_valid("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_is_valid("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_is_valid("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_value("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_value("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_value("),
        ("scalar native header", SCALAR_H, "cudf_status_t cudf_scalar_string_value("),
        ("scalar native source", SCALAR_CPP, "cudf_status_t cudf_scalar_string_value("),
        ("scalar Rust FFI", SCALAR_FFI_RS, "pub fn cudf_scalar_string_value("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }

    for marker in [
        "cudf_scalar_create_numeric_ex(type_id, value, is_valid, nullptr, out)",
        "cudf_scalar_create_decimal_raw_ex(type_id, value, scale, is_valid, nullptr, out)",
        "cudf_scalar_create_timestamp_raw_ex(type_id, value, is_valid, nullptr, out)",
        "cudf_scalar_create_duration_raw_ex(type_id, value, is_valid, nullptr, out)",
        "cudf_scalar_create_string_ex(value, is_valid, nullptr, out)",
        "cudf_scalar_copy_ex(src, nullptr, out)",
        "cudf_scalar_is_valid_ex(handle, out, nullptr)",
        "cudf_scalar_value_ex(handle, out, nullptr)",
        "cudf_scalar_string_value_ex(handle, out, len_out, nullptr)",
    ] {
        assert!(
            !contains_normalized(SCALAR_CPP, marker),
            "scalar native source must not delegate to strict `_ex` with null options via `{marker}`"
        );
    }
}

fn function_region<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source.find(start_marker).expect("start marker must exist");
    let tail = &source[start..];
    let end = tail.find(end_marker).expect("end marker must exist");
    &tail[..end]
}

fn function_region_by_function_name<'a>(source: &'a str, function: &str, end_marker: &str) -> &'a str {
    for (start, _) in source.match_indices(function) {
        let prefix = &source[start.saturating_sub(96)..start];
        if !prefix.contains("cudf_status_t") {
            continue;
        }
        let tail = &source[start..];
        let end = tail.find(end_marker).expect("end marker must exist");
        return &tail[..end];
    }
    panic!("{function} definition marker must exist");
}
