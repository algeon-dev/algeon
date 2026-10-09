//! `resolve` and `reassemble` are `pub(crate)`, so these tests verify them
//! **indirectly** through public DataFrame operations that exercise the
//! internal helpers.
//!
//! Direct unit tests for the helpers live inside `src/internal/resolve.rs`.

use cudf::{
    DataFrame, DuplicateKeepOption, ErrorCode, ErrorKind, Exec,
    core::{
        Table,
        error::{FactKeyName, FactValue},
    },
    frame::dataframe::DropDuplicatesOptions,
    testing,
};

use crate::common::{
    assert_duplicate_name_error, assert_missing_column_error, column_from_slice, dataframe_from_columns,
    error_context_value, series,
};

// Borrowed TableView construction is exercised via filter, sort, and drop_duplicates.
// These operations internally build table views from temporary column views.

fn filter_preserves_row_and_column_counts() {
    // filter internally uses build_borrowed_table_view_from_columns.
    let ids = series::<i32>("ids", &[1, 2, 3, 4, 5]).unwrap();
    let vals = series::<f64>("vals", &[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("val", vals)]).unwrap();

    // Boolean mask: keep rows 0, 2, 4.
    let mask = series::<bool>("mask", &[true, false, true, false, true]).unwrap();
    let filtered = df.on(Exec::RETAINED).filter(&mask).unwrap();

    assert_eq!(filtered.num_columns(), 2, "column count preserved");
    assert_eq!(filtered.num_rows(), 3, "filtered row count");
    assert_eq!(filtered.column_names(), vec!["id", "val"], "schema names preserved");
}

fn sort_preserves_schema_names_and_lengths() {
    // sort internally resolves column names and builds table views.
    let names_col = series::<i32>("a", &[3, 1, 2]).unwrap();
    let vals_col = series::<i64>("b", &[30, 10, 20]).unwrap();
    let df = DataFrame::from_series([("a", names_col), ("b", vals_col)]).unwrap();

    let sorted = df.on(Exec::RETAINED).sort_values(["a"], cudf::SortOptions::new()).unwrap();

    assert_eq!(sorted.num_columns(), 2);
    assert_eq!(sorted.num_rows(), 3);
    assert_eq!(sorted.column_names(), vec!["a", "b"]);

    let a_view = sorted.series("a").unwrap();
    let a_vals = a_view.to_vec_strict::<i32>().unwrap();
    assert_eq!(a_vals, vec![1, 2, 3]);
}

fn reassembly_after_drop_duplicates_preserves_schema() {
    // drop_duplicates internally uses resolve + reassemble helpers.
    let a = series::<i32>("x", &[1, 1, 2, 2, 3]).unwrap();
    let b = series::<i64>("y", &[10, 10, 20, 20, 30]).unwrap();
    let df = DataFrame::from_series([("x", a), ("y", b)]).unwrap();

    let distinct = df
        .on(Exec::RETAINED)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(&["x", "y"]).keep(DuplicateKeepOption::KeepFirst).build(),
        )
        .unwrap();

    assert_eq!(distinct.num_columns(), 2, "schema columns preserved");
    assert_eq!(distinct.num_rows(), 3, "distinct row count");
    assert_eq!(distinct.column_names(), vec!["x", "y"], "schema names preserved");
}

fn sort_with_missing_column_returns_error() {
    // resolve_column_names should produce a clear error for unknown columns.
    let a = series::<i32>("a", &[1, 2, 3]).unwrap();
    let df = DataFrame::from_series([("a", a)]).unwrap();

    let err = df.on(Exec::RETAINED).sort_values(["nonexistent"], cudf::SortOptions::new()).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

// Reassembly goes through `from_table_with_names` for join.

fn join_reassembly_preserves_schema_names() {
    // join internally uses reassembly helpers to rebuild DataFrames.
    let left_key = series::<i32>("key", &[1, 2, 3]).unwrap();
    let left_val = series::<f64>("lval", &[10.0, 20.0, 30.0]).unwrap();
    let left = DataFrame::from_series([("key", left_key), ("lval", left_val)]).unwrap();

    let right_key = series::<i32>("key", &[2, 3, 4]).unwrap();
    let right_val = series::<f64>("rval", &[200.0, 300.0, 400.0]).unwrap();
    let right = DataFrame::from_series([("key", right_key), ("rval", right_val)]).unwrap();

    let opts = cudf::MergeOptions::new().on(["key"]).how_inner();
    let joined = left.on(Exec::RETAINED).merge(&right, opts).unwrap();

    // Inner join on [2,3] → 2 rows.
    assert_eq!(joined.num_rows(), 2);
    assert_eq!(joined.column_names(), vec!["key", "lval", "rval"]);
}

fn reassembly_name_count_mismatch_uses_canonical_count_contract() {
    let table = Table::from_columns(vec![column_from_slice(&[1i32, 2]).unwrap()]).unwrap();

    let err = testing::dataframe::reassemble_dataframe_from_table_and_names_for_test(
        table,
        vec!["only".into(), "extra".into()],
    )
    .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "internal::reassemble::dataframe_from_table_and_names");
    assert_eq!(err.fact_value(FactKeyName::ActualCount), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::TableColumnCount), Some(&FactValue::Usize(1)));
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("mismatch_with_table_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure column_name_count matches table_column_count")
    );
    assert_eq!(error_context_value(&err, "names_count").as_deref(), None);
    assert_eq!(error_context_value(&err, "columns_count").as_deref(), None);
}

// `validate_no_duplicate_names` is exercised via DataFrame constructors.

fn from_series_rejects_duplicate_names() {
    let a = series::<i32>("a", &[1]).unwrap();
    let b = series::<i32>("b", &[2]).unwrap();

    let err = DataFrame::from_series(vec![("x".to_string(), a), ("x".to_string(), b)]).unwrap_err();
    assert_duplicate_name_error(&err, "x");
}

fn dataframe_new_rejects_duplicate_names() {
    let c1 = column_from_slice(&[1i32]).unwrap();
    let c2 = column_from_slice(&[2i32]).unwrap();
    let err = dataframe_from_columns(vec!["dup".into(), "dup".into()], vec![c1, c2]).unwrap_err();
    assert_duplicate_name_error(&err, "dup");
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("internal_tests::filter_preserves_row_and_column_counts", filter_preserves_row_and_column_counts),
    ("internal_tests::sort_preserves_schema_names_and_lengths", sort_preserves_schema_names_and_lengths),
    (
        "internal_tests::reassembly_after_drop_duplicates_preserves_schema",
        reassembly_after_drop_duplicates_preserves_schema,
    ),
    ("internal_tests::sort_with_missing_column_returns_error", sort_with_missing_column_returns_error),
    ("internal_tests::join_reassembly_preserves_schema_names", join_reassembly_preserves_schema_names),
    (
        "internal_tests::reassembly_name_count_mismatch_uses_canonical_count_contract",
        reassembly_name_count_mismatch_uses_canonical_count_contract,
    ),
    ("internal_tests::from_series_rejects_duplicate_names", from_series_rejects_duplicate_names),
    ("internal_tests::dataframe_new_rejects_duplicate_names", dataframe_new_rejects_duplicate_names),
];
