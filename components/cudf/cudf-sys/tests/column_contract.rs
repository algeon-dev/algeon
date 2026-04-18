mod source_contract_support;

const COLUMN_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/column.cpp"));

#[test]
fn column_slice_uses_public_column_view_null_count_api() {
    assert!(
        source_contract_support::contains_normalized(
            COLUMN_CPP,
            "auto const null_count = view->nullable() ? view->null_count(start, start + len, exec_stream) : 0;"
        ),
        "expected column.cpp to use the public column_view::null_count overload for sliced views"
    );
    assert!(
        !source_contract_support::contains_normalized(
            COLUMN_CPP,
            "cudf::detail::null_count(view->null_mask(), active_begin, active_end, exec_stream)"
        ),
        "column.cpp still depends on the non-exported cudf::detail::null_count symbol"
    );
}
