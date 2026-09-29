use super::*;

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
        ("cudf_column_copy_from_view_ex", "cudf_status_t cudf_column_payload_size_ex"),
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
        ("cudf_strings_replace_with_backrefs_ex", "cudf_status_t cudf_strings_replace_with_backrefs_output_size_ex"),
        (
            "cudf_strings_replace_with_backrefs_output_size_ex",
            "cudf_status_t cudf_strings_prepare_replace_with_backrefs_ex",
        ),
        (
            "cudf_strings_prepare_replace_with_backrefs_ex",
            "cudf_status_t cudf_strings_prepared_replace_with_backrefs_output_size_facts",
        ),
        (
            "cudf_strings_execute_replace_with_backrefs_ex",
            "cudf_status_t cudf_strings_prepared_replace_with_backrefs_destroy",
        ),
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
fn replace_backrefs_reservation_preflight_is_measured_at_the_admission_boundary() {
    let body = function_region(
        STRINGS_CPP,
        "cudf_status_t cudf_strings_replace_with_backrefs_output_size_ex",
        "cudf_status_t cudf_strings_prepare_replace_with_backrefs_ex",
    );
    assert!(contains_normalized(
        body,
        "cudf::rust::reservation::statistics_resource_adaptor measurement{exec->mr_raw};"
    ));
    assert!(contains_normalized(
        body,
        "std::max(measurement.peak_aligned_bytes(), checked_add_reservation_bytes(measurement.current_aligned_bytes(), facts.execute_reservation_required_bytes))"
    ));
    assert!(contains_normalized(RESERVATION_ACCOUNTING_HPP, "return rmm::align_up(bytes, alignment);"));
    assert!(
        RESERVATION_CPP.contains("cudf::rust::reservation::allocation_charge(bytes, alignment)"),
        "reservation admission and preflight observation must share one per-allocation charge helper"
    );
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
        ("cudf_groupby_shift_ex", "} // extern \"C\""),
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
        ("cudf_distinct_hash_join_create_ex", "cudf_status_t cudf_distinct_hash_join_destroy"),
        ("cudf_distinct_hash_join_left_join_ex", "cudf_status_t cudf_hash_join_create_ex"),
        ("cudf_hash_join_create_ex", "cudf_status_t cudf_hash_join_pre_build_reservation_size_ex"),
        ("cudf_hash_join_pre_build_reservation_size_ex", "cudf_status_t cudf_hash_join_destroy"),
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
fn filtered_join_shim_uses_2606_right_filter_constructor() {
    for (name, source) in [("join hash source", JOIN_HASH_CPP), ("join common source", JOIN_COMMON_HPP)] {
        assert!(
            !source.contains("set_as_build_table"),
            "{name} must use the cuDF 26.06 filtered_join constructors that always treat the retained table as right/filter"
        );
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
fn reshape_ex_entry_uses_required_exec_options() {
    let body = function_region(RESHAPE_CPP, "cudf_status_t cudf_interleave_columns_ex", "}  // extern \"C\"");
    assert!(
        body.contains("cudf_resolve_required_exec_options(options, &exec)"),
        "cudf_interleave_columns_ex must use the strict execution-options resolver"
    );
    assert!(
        !body.contains("cudf_resolve_exec_options(options"),
        "cudf_interleave_columns_ex must not accept permissive/ambient execution options"
    );
    assert!(
        !body.contains("cudf_make_exec_context"),
        "cudf_interleave_columns_ex must not seed execution from ambient defaults"
    );
}

#[test]
fn reshape_ex_entry_rejects_oversized_output_before_execution() {
    let body = function_region(RESHAPE_CPP, "cudf_status_t cudf_interleave_columns_ex", "}  // extern \"C\"");
    let size_check = body
        .find("std::numeric_limits<cudf::size_type>::max()")
        .expect("reshape native entry must enforce the cudf size_type output limit");
    let resolve = body
        .find("cudf_resolve_required_exec_options(options, &exec)")
        .expect("reshape native entry must resolve explicit execution options");
    let launch = body.find("cudf::interleave_columns").expect("reshape native entry must launch libcudf interleave");

    assert!(body.contains("static_cast<int64_t>(views.front().size())"));
    assert!(body.contains("rows * static_cast<int64_t>(num_columns)"));
    assert!(size_check < resolve, "reshape output-size admission must precede execution resolution");
    assert!(size_check < launch, "reshape output-size admission must precede the libcudf launch");
}

#[test]
fn reshape_duplicate_non_ex_production_api_is_absent() {
    for (name, source, marker) in [
        ("reshape native header", RESHAPE_H, "cudf_status_t cudf_interleave_columns("),
        ("reshape native source", RESHAPE_CPP, "cudf_status_t cudf_interleave_columns("),
        ("reshape Rust FFI", RESHAPE_FFI_RS, "pub fn cudf_interleave_columns("),
    ] {
        assert!(!source.contains(marker), "{name} must not retain duplicate non-_ex production API `{marker}`");
    }
    assert!(
        !RESHAPE_CPP.contains("cudf_interleave_columns_ex(columns, num_columns, nullptr, out)"),
        "reshape native source must not delegate to strict `_ex` with null options"
    );
}

#[test]
fn finite_validation_reduces_on_explicit_stream_to_one_bounded_scalar() {
    let helper =
        function_region(VALIDATION_CU, "int32_t count_invalid_values", "cudf_status_t validate_raw_element_count");
    for required in [
        "cub::DeviceReduce::TransformReduce",
        "cudf::detail::device_scalar<int32_t>",
        "result.value(exec.stream)",
        "*exec.mr",
    ] {
        assert!(helper.contains(required), "bounded validation reduction must retain `{required}`");
    }
    for forbidden in ["cudaDeviceSynchronize", "cudf_column_view_to_host", "std::vector<float>"] {
        assert!(!VALIDATION_CU.contains(forbidden), "finite validation must not use `{forbidden}`");
    }
    for function in [
        "cudf_float32_non_finite_count_ex",
        "cudf_float32_buffer_non_finite_count_ex",
        "cudf_int64_buffer_out_of_range_count_ex",
    ] {
        let start = VALIDATION_CU.find(&format!("cudf_status_t {function}")).expect("validation entry must exist");
        let tail = &VALIDATION_CU[start..];
        let end = tail
            .find("CUDF_RUST_CATCH")
            .map(|index| index + "CUDF_RUST_CATCH".len())
            .expect("validation entry must retain exception boundary");
        let body = &tail[..end];
        assert!(
            body.contains("cudf_resolve_required_exec_options(options, &exec)"),
            "{function} must use the strict execution-options resolver"
        );
        assert!(body.contains("count_invalid_values"), "{function} must use the bounded device reduction helper");
        assert!(
            body.contains("cudf_record_synchronized_stream(exec->stream)"),
            "{function} must record exact-stream synchronization"
        );
    }
    assert!(VALIDATION_FFI_RS.contains("pub fn cudf_float32_non_finite_count_ex"));
    assert!(VALIDATION_FFI_RS.contains("pub fn cudf_float32_buffer_non_finite_count_ex"));
    assert!(VALIDATION_FFI_RS.contains("pub fn cudf_int64_buffer_out_of_range_count_ex"));
    assert!(VALIDATION_FFI_RS.contains("options: *const cudf_exec_options_t"));
}

#[test]
fn finite_column_validation_passes_a_device_view_by_value_to_the_gpu_transform() {
    assert!(
        VALIDATION_CU.contains("cudf::column_device_view input;"),
        "the device transform must own a column_device_view value rather than a host pointer"
    );
    assert!(VALIDATION_CU.contains("input.is_null(index)"));
    assert!(VALIDATION_CU.contains("input.element<T>(index)"));
    for element in ["float", "double"] {
        assert!(
            VALIDATION_CU.contains(&format!("column_non_finite_count_transform<{element}>{{*device_view}}")),
            "the {element} device transform must copy the device view into its kernel argument"
        );
        assert!(
            !VALIDATION_CU.contains(&format!("column_non_finite_count_transform<{element}>{{device_view.get()}}")),
            "the {element} device transform must not dereference a host-owned device-view pointer on the GPU"
        );
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
        ("cudf_filter_rows_by_predicate_ex", "} // extern \"C\""),
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
