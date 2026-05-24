mod source_contract_support;

const COLUMN_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/column.cpp"));

#[test]
fn column_slice_uses_public_column_view_null_count_api() {
    let slice_ex_body = function_region(
        COLUMN_CPP,
        "cudf_status_t cudf_column_view_slice_ex",
        "cudf_status_t cudf_column_view_to_host_ex",
    );

    assert!(
        source_contract_support::contains_normalized(
            slice_ex_body,
            "auto const null_count = view->nullable() ? view->null_count(start, start + len, exec->stream) : 0;"
        ),
        "expected column.cpp to use the public column_view::null_count overload with the resolved execution stream for sliced views"
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
