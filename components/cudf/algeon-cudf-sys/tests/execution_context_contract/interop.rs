use super::*;

#[test]
fn device_buffer_ex_entries_use_required_exec_options() {
    for (function, end_marker) in [
        ("cudf_device_buffer_create_ex", "cudf_status_t cudf_device_buffer_destroy"),
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
        ("cudf_table_copy_ex", "cudf_status_t cudf_table_payload_size_ex"),
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
        ("cudf_arrow_device_table_import", "cudf_arrow_device_schema_destroy"),
    ] {
        let body = function_region_by_function_name(ARROW_DEVICE_CPP, function, end_marker);
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
fn arrow_host_entries_use_required_exec_options() {
    let body = function_region(ARROW_HOST_CPP, "cudf_status_t cudf_arrow_host_column_import_ex", "CUDF_RUST_CATCH");
    assert!(
        body.contains("cudf_resolve_required_exec_options(options, &exec)"),
        "cudf_arrow_host_column_import_ex must use the strict execution-options resolver"
    );
    assert!(
        !body.contains("cudf_resolve_exec_options(options"),
        "cudf_arrow_host_column_import_ex must not accept permissive/ambient execution options"
    );
    assert!(
        !body.contains("cudf_make_exec_context"),
        "cudf_arrow_host_column_import_ex must not seed execution from ambient defaults"
    );
}

#[test]
fn arrow_host_duplicate_non_ex_production_apis_are_removed() {
    for (name, source, marker) in [
        ("arrow host native header", ARROW_HOST_H, "cudf_status_t cudf_arrow_host_column_import("),
        ("arrow host native source", ARROW_HOST_CPP, "cudf_status_t cudf_arrow_host_column_import("),
        ("arrow host Rust FFI", ARROW_HOST_FFI_RS, "pub fn cudf_arrow_host_column_import("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }
}
