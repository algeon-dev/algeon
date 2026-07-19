use super::*;

#[test]
fn scalar_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_scalar_create_numeric_ex", "cudf_status_t cudf_scalar_create_decimal_raw_ex"),
        ("cudf_scalar_create_decimal_raw_ex", "cudf_status_t cudf_scalar_create_timestamp_raw_ex"),
        ("cudf_scalar_create_timestamp_raw_ex", "cudf_status_t cudf_scalar_create_duration_raw_ex"),
        ("cudf_scalar_create_duration_raw_ex", "cudf_status_t cudf_scalar_create_string_ex"),
        ("cudf_scalar_create_string_ex", "cudf_status_t cudf_scalar_copy_ex"),
        ("cudf_scalar_copy_ex", "cudf_status_t cudf_scalar_destroy"),
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
