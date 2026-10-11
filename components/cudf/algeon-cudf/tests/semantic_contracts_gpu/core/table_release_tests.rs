use cudf::{ErrorCode, Runtime, TypeId, core::Table, testing::prelude::*};
use cudf_sys::test_support;

use crate::common::{column_from_slice, column_from_slice_nullable, column_from_strs, empty_column};

fn test_into_columns_single_column() {
    let c1 = column_from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 1);

    let view = columns[0].testing_view().unwrap();
    assert_eq!(view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(view.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

fn test_into_columns_multiple_columns() {
    let c1 = column_from_slice(&[1i32, 2, 3]).unwrap();
    let c2 = column_from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
    let c3 = column_from_slice(&[7i64, 8, 9]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 3);

    let v0 = columns[0].testing_view().unwrap();
    assert_eq!(v0.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(v0.to_vec::<i32>().unwrap(), vec![1, 2, 3]);

    let v1 = columns[1].testing_view().unwrap();
    assert_eq!(v1.data_type().unwrap().id(), TypeId::Float64);
    assert_eq!(v1.to_vec::<f64>().unwrap(), vec![4.0, 5.0, 6.0]);

    let v2 = columns[2].testing_view().unwrap();
    assert_eq!(v2.data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(v2.to_vec::<i64>().unwrap(), vec![7, 8, 9]);
}

fn test_into_columns_empty_table() {
    let table = Runtime::new().unwrap().empty_table().unwrap();
    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 0);
}

fn test_into_columns_preserves_column_order() {
    let c1 = column_from_slice(&[100i32]).unwrap();
    let c2 = column_from_slice(&[200i64]).unwrap();
    let c3 = column_from_slice(&[3.25f32]).unwrap();
    let table = Table::from_columns(vec![c1, c2, c3]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 3);

    assert_eq!(columns[0].testing_view().unwrap().data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(columns[1].testing_view().unwrap().data_type().unwrap().id(), TypeId::Int64);
    assert_eq!(columns[2].testing_view().unwrap().data_type().unwrap().id(), TypeId::Float32);
}

fn test_into_columns_nullable_column() {
    let c1 = column_from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 1);

    let view = columns[0].testing_view().unwrap();
    assert_eq!(view.to_vec::<i32>().unwrap()[0], 1);
    assert_eq!(view.to_vec::<i32>().unwrap()[2], 3);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 1);

    let validity = view.to_host_validity().unwrap().unwrap();
    assert_eq!(validity, vec![true, false, true]);
}

fn test_into_columns_roundtrip() {
    let c1 = column_from_slice(&[10i32, 20]).unwrap();
    let c2 = column_from_slice(&[1.5f64, 2.5]).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 2);

    let table2 = Table::from_columns(columns).unwrap();
    let view = table2.testing_view().unwrap();
    assert_eq!(view.num_columns().unwrap(), 2);
    assert_eq!(view.num_rows().unwrap(), 2);

    let col0 = view.column(0).unwrap();
    assert_eq!(col0.to_vec::<i32>().unwrap(), vec![10, 20]);

    let col1 = view.column(1).unwrap();
    assert_eq!(col1.to_vec::<f64>().unwrap(), vec![1.5, 2.5]);
}

fn test_into_columns_with_string_column() {
    let c1 = column_from_strs(&["hello", "world"]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 1);

    let view = columns[0].testing_view().unwrap();
    assert_eq!(view.data_type().unwrap().id(), TypeId::String);
    assert_eq!(view.to_strings_lossy().unwrap(), vec!["hello", "world"]);
}

fn test_into_columns_with_empty_rows() {
    let c1 = empty_column(TypeId::Int32).unwrap();
    let c2 = empty_column(TypeId::Float64).unwrap();
    let table = Table::from_columns(vec![c1, c2]).unwrap();

    let columns = table.into_columns().unwrap();
    assert_eq!(columns.len(), 2);

    let v0 = columns[0].testing_view().unwrap();
    assert_eq!(v0.len().unwrap(), 0);
    assert_eq!(v0.data_type().unwrap().id(), TypeId::Int32);

    let v1 = columns[1].testing_view().unwrap();
    assert_eq!(v1.len().unwrap(), 0);
    assert_eq!(v1.data_type().unwrap().id(), TypeId::Float64);
}

#[test]
fn into_columns_copies_typed_failure_before_cleanup_clears_native_error_state() {
    let c1 = column_from_slice(&[10i32, 20]).unwrap();
    let table = Table::from_columns(vec![c1]).unwrap();

    unsafe { test_support::cudf_test_fail_next_table_release_columns_handle_array_alloc() };
    let error = table.into_columns().unwrap_err();

    assert_eq!(error.code(), ErrorCode::OutOfMemory);
    let detail = error.ffi_detail().expect("table release failure must preserve the native record before cleanup");
    assert_eq!(detail.status, cudf_sys::CUDF_ERROR_OUT_OF_MEMORY);
    assert!(detail.message.contains("failed to allocate column handle array"));
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("table_release_tests::test_into_columns_single_column", test_into_columns_single_column),
    ("table_release_tests::test_into_columns_multiple_columns", test_into_columns_multiple_columns),
    ("table_release_tests::test_into_columns_empty_table", test_into_columns_empty_table),
    ("table_release_tests::test_into_columns_preserves_column_order", test_into_columns_preserves_column_order),
    ("table_release_tests::test_into_columns_nullable_column", test_into_columns_nullable_column),
    ("table_release_tests::test_into_columns_roundtrip", test_into_columns_roundtrip),
    ("table_release_tests::test_into_columns_with_string_column", test_into_columns_with_string_column),
    ("table_release_tests::test_into_columns_with_empty_rows", test_into_columns_with_empty_rows),
];
