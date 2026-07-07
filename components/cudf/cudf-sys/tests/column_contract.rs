mod source_contract_support;

const COLUMN_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/column.cpp"));

#[test]
fn column_slice_resolves_active_range_null_count_with_required_exec_options() {
    let null_count_helper_body = function_region(
        COLUMN_CPP,
        "cudf_status_t resolve_column_view_null_count",
        "class host_source_copy_sync_guard",
    );
    let slice_ex_body = function_region(
        COLUMN_CPP,
        "cudf_status_t cudf_column_view_slice_ex",
        "cudf_status_t cudf_column_view_to_host_ex",
    );

    assert!(
        source_contract_support::contains_normalized(
            null_count_helper_body,
            "*out = view.null_count(0, view.size(), stream);"
        ),
        "expected unknown column-view null counts to use libcudf's range-aware null_count API on the resolved stream"
    );
    assert!(
        source_contract_support::contains_normalized(
            slice_ex_body,
            "cudf_resolve_required_exec_options(options, &exec);"
        ),
        "expected column_view_slice_ex to require explicit execution options before resolving active-range null counts"
    );
    assert!(
        source_contract_support::contains_normalized(slice_ex_body, "parent_null_count == 0"),
        "expected column_view_slice_ex to avoid active-range null-count work when parent metadata proves no nulls"
    );
    assert!(
        source_contract_support::contains_normalized(slice_ex_body, "parent_null_count == view->size()"),
        "expected column_view_slice_ex to avoid active-range null-count work when parent metadata proves all nulls"
    );
    assert!(
        source_contract_support::contains_normalized(
            slice_ex_body,
            "null_count = view->null_count(start, start + len, exec->stream);"
        ),
        "expected column_view_slice_ex to resolve mixed/unknown active-range null counts on the resolved execution stream"
    );
    assert!(
        !source_contract_support::contains_normalized(
            slice_ex_body,
            "cudf::detail::null_count(view->null_mask(), active_begin, active_end, exec_stream)"
        ),
        "column.cpp still depends on the non-exported cudf::detail::null_count symbol"
    );
    assert!(
        !slice_ex_body.contains("cudf_current_device_resource_mutex"),
        "column_view_slice must not serialize null-count computation on the current-device-resource mutex"
    );
    assert!(
        !slice_ex_body.contains("cudf_current_device_resource_guard"),
        "column_view_slice must not install the process-global current device resource for null-count computation"
    );
}

fn function_region<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source.find(start_marker).expect("start marker must exist");
    let tail = &source[start..];
    let end = tail.find(end_marker).expect("end marker must exist");
    &tail[..end]
}
