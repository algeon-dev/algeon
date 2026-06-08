//! Tests asserting that interop consumer import methods use the
//! `try_import_from_*_into_runtime` naming convention established by
//! the public API layering interop alignment pass.

use cudf::{
    DataFrame, Runtime, Series,
    core::{Column, Table},
    interop::{BorrowedColumnView, BorrowedTableView, InteropError, PreparedColumnExport, PreparedTableExport},
};

use crate::common;

// ── Compile-time signature assertions ──────────────────────────────────

#[test]
fn consumer_import_naming_column_borrowed() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a BorrowedColumnView<'b>, &'a Runtime) -> Result<Column, InteropError>) {}
    assert_sig(Column::try_import_from_borrowed_column_view_into_runtime);
}

#[test]
fn consumer_import_naming_column_prepared() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a PreparedColumnExport<'b>, &'a Runtime) -> Result<Column, InteropError>) {}
    assert_sig(Column::try_import_from_prepared_column_export_into_runtime);
}

#[test]
fn consumer_import_naming_table_borrowed() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a BorrowedTableView<'b>, &'a Runtime) -> Result<Table, InteropError>) {}
    assert_sig(Table::try_import_from_borrowed_table_view_into_runtime);
}

#[test]
fn consumer_import_naming_table_prepared() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a PreparedTableExport<'b>, &'a Runtime) -> Result<Table, InteropError>) {}
    assert_sig(Table::try_import_from_prepared_table_export_into_runtime);
}

#[test]
fn consumer_import_naming_series_borrowed() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a BorrowedColumnView<'b>, &'a Runtime) -> Result<Series, InteropError>) {}
    assert_sig(Series::try_import_from_borrowed_column_view_into_runtime);
}

#[test]
fn consumer_import_naming_series_prepared() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a PreparedColumnExport<'b>, &'a Runtime) -> Result<Series, InteropError>) {}
    assert_sig(Series::try_import_from_prepared_column_export_into_runtime);
}

#[test]
fn consumer_import_naming_dataframe_borrowed_with_names() {
    fn assert_sig(
        _f: for<'a, 'b> fn(&'a BorrowedTableView<'b>, &'a Runtime, Vec<String>) -> Result<DataFrame, InteropError>,
    ) {
    }
    assert_sig(DataFrame::try_import_from_borrowed_table_view_into_runtime_and_names);
}

#[test]
fn consumer_import_naming_dataframe_prepared() {
    fn assert_sig(_f: for<'a, 'b> fn(&'a PreparedTableExport<'b>, &'a Runtime) -> Result<DataFrame, InteropError>) {}
    assert_sig(DataFrame::try_import_from_prepared_table_export_into_runtime);
}

#[test]
fn consumer_import_naming_dataframe_prepared_with_names() {
    fn assert_sig(
        _f: for<'a, 'b> fn(&'a PreparedTableExport<'b>, &'a Runtime, Vec<String>) -> Result<DataFrame, InteropError>,
    ) {
    }
    assert_sig(DataFrame::try_import_from_prepared_table_export_into_runtime_and_names);
}

// ── Runtime round-trip smoke tests ─────────────────────────────────────

#[test]
fn column_import_into_runtime_round_trips_data() {
    let source_runtime = Runtime::builder().build().unwrap();
    let source = source_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let export = source.try_prepare_device_column_export().unwrap();
    let dest = common::sample_runtime();

    let imported = Column::try_import_from_prepared_column_export_into_runtime(&export, &dest).unwrap();

    assert_eq!(imported.view().unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(imported.__debug_execution_domain_stream_value(), dest.stream().__debug_raw_handle_value());
}

#[test]
fn table_import_into_runtime_round_trips_data() {
    let source_runtime = Runtime::builder().build().unwrap();
    let first = source_runtime.column_from_slice(&[10i32, 20]).unwrap();
    let second = source_runtime.column_from_slice(&[100i64, 200]).unwrap();
    let source = Table::from_columns(vec![first, second]).unwrap();
    let view = source.view().unwrap();
    let borrowed = view.try_borrow_table_view().unwrap();
    let dest = common::sample_runtime();

    let imported = Table::try_import_from_borrowed_table_view_into_runtime(&borrowed, &dest).unwrap();

    let iv = imported.view().unwrap();
    assert_eq!(iv.column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 20]);
    assert_eq!(iv.column(1).unwrap().to_vec::<i64>().unwrap(), vec![100, 200]);
}

#[test]
fn series_import_into_runtime_round_trips_data() {
    let source_runtime = Runtime::builder().build().unwrap();
    let source = source_runtime.column_from_slice(&[5i32, 6]).unwrap();
    let view = source.view().unwrap();
    let borrowed = view.try_borrow_column_view().unwrap();
    let dest = common::sample_runtime();

    let imported = Series::try_import_from_borrowed_column_view_into_runtime(&borrowed, &dest).unwrap();

    assert_eq!(imported.to_vec_strict::<i32>().unwrap(), vec![5, 6]);
}

#[test]
fn dataframe_import_into_runtime_round_trips_data() {
    let source_runtime = Runtime::builder().build().unwrap();
    let col = source_runtime.column_from_slice(&[7i32, 8]).unwrap();
    let source = Table::from_columns(vec![col]).unwrap();
    let view = source.view().unwrap();
    let borrowed = view.try_borrow_table_view().unwrap();
    let dest = common::sample_runtime();

    let imported = DataFrame::try_import_from_borrowed_table_view_into_runtime_and_names(
        &borrowed,
        &dest,
        vec!["value".to_string()],
    )
    .unwrap();

    assert_eq!(imported.column_names(), vec!["value"]);
    assert_eq!(imported.series("value").unwrap().to_vec_strict::<i32>().unwrap(), vec![7, 8]);
}
