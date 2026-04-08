use cudf::{
    DataFrame, Runtime, Series, TypeId,
    core::{Column, Table},
    interop::{ColumnLayout, TryBorrowColumnView, TryBorrowTableView},
    ops::dictionary,
};
use rapids_interop::{InteropErrorKind, NullMaskKind, TypeId as InteropTypeId};
use static_assertions::assert_impl_all;

assert_impl_all!(Column: TryBorrowColumnView);
assert_impl_all!(Table: TryBorrowTableView);

#[test]
fn column_view_borrow_preserves_fixed_width_metadata_and_explicit_binding() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let view = column.view().unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();
    let expected_mr = column.__debug_execution_domain_mr_value();

    let borrowed = view.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::Int32);
    assert_eq!(borrowed.len(), 3);
    assert_eq!(borrowed.offset(), 0);
    assert_eq!(borrowed.data_ptr(), view.device_data_ptr().unwrap());
    assert!(matches!(borrowed.null_mask().kind(), NullMaskKind::Absent));

    let execution = borrowed.execution();
    assert_eq!(execution.device(), runtime.device().ordinal() as i32);
    assert_eq!(execution.cuda_stream().as_raw() as usize, expected_stream);
    assert_eq!(execution.cuda_stream().device(), runtime.device().ordinal() as i32);
    assert_eq!(execution.memory_resource().as_raw() as usize, expected_mr);
}

#[test]
fn nullable_column_distinguishes_present_mask_with_zero_null_count() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_nullable_with_runtime(&runtime, &[10i32, 20, 30], &[false, true, true]).unwrap();
    let view = column.slice_view(1, 2).unwrap();
    let borrowed = view.try_borrow_column_view().unwrap();

    match borrowed.null_mask().kind() {
        NullMaskKind::Present { ptr, bit_len, null_count } => {
            let _ = ptr.as_ptr();
            assert_eq!(*null_count, 0);
            assert!(*bit_len >= borrowed.offset() + borrowed.len());
        },
        NullMaskKind::Absent => panic!("expected a present null mask for nullable column"),
    }
}

#[test]
fn sliced_column_view_preserves_offset_semantics() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[11i32, 22, 33, 44]).unwrap();
    let base = column.view().unwrap();
    let sliced = column.slice_view(1, 2).unwrap();

    let borrowed = sliced.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::Int32);
    assert_eq!(borrowed.len(), 2);
    assert_eq!(borrowed.offset(), 1);
    assert_eq!(borrowed.data_ptr(), base.device_data_ptr().unwrap());
}

#[test]
fn table_view_borrow_preserves_shape_and_explicit_execution_binding() {
    let runtime = Runtime::builder().build().unwrap();
    let a = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let b = Column::from_slice_with_runtime(&runtime, &[4i32, 5, 6]).unwrap();
    let table = Table::from_columns(vec![a, b]).unwrap();
    let view = table.view().unwrap();
    let expected_stream = table.__debug_execution_domain_stream_value();
    let expected_mr = table.__debug_execution_domain_mr_value();

    let borrowed = view.try_borrow_table_view().unwrap();

    assert_eq!(borrowed.num_rows(), 3);
    assert_eq!(borrowed.num_columns(), 2);

    let execution = borrowed.execution();
    assert_eq!(execution.device(), runtime.device().ordinal() as i32);
    assert_eq!(execution.cuda_stream().as_raw() as usize, expected_stream);
    assert_eq!(execution.cuda_stream().device(), runtime.device().ordinal() as i32);
    assert_eq!(execution.memory_resource().as_raw() as usize, expected_mr);
}

#[test]
fn owned_column_borrow_matches_underlying_column_view_projection() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[5i32, 6, 7]).unwrap();
    let view = column.view().unwrap();

    let borrowed_from_column = column.try_borrow_column_view().unwrap();
    let borrowed_from_view = view.try_borrow_column_view().unwrap();

    assert_eq!(borrowed_from_column, borrowed_from_view);
}

#[test]
fn owned_table_borrow_matches_underlying_table_view_projection() {
    let runtime = Runtime::builder().build().unwrap();
    let a = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let b = Column::from_slice_with_runtime(&runtime, &[4i32, 5, 6]).unwrap();
    let table = Table::from_columns(vec![a, b]).unwrap();
    let view = table.view().unwrap();

    let borrowed_from_table = table.try_borrow_table_view().unwrap();
    let borrowed_from_view = view.try_borrow_table_view().unwrap();

    assert_eq!(borrowed_from_table, borrowed_from_view);
}

#[test]
fn series_and_dataframe_convenience_producers_match_underlying_low_level_views() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids.clone(), vals.clone()]).unwrap();

    let series_borrowed = ids.try_borrow_column_view().unwrap();
    let frame_borrowed = frame.try_borrow_table_view().unwrap();

    assert_eq!(frame_borrowed.num_rows(), 3);
    assert_eq!(frame_borrowed.num_columns(), 2);
    assert_eq!(frame_borrowed.columns()[0], series_borrowed);
    assert_eq!(frame_borrowed.columns()[1], vals.try_borrow_column_view().unwrap());
}

#[test]
fn ambient_column_view_borrow_is_rejected() {
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let err = column.view().unwrap().try_borrow_column_view().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
    assert!(
        err.context()
            .iter()
            .any(|(key, value)| *key == "operation" && value == "interop::ColumnView::try_borrow_column_view")
    );
}

#[test]
fn ambient_dataframe_borrow_is_rejected() {
    let ids = Series::new("id", &[1i32, 2, 3]).unwrap();
    let vals = Series::new("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();

    let err = frame.try_borrow_table_view().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
    assert!(
        err.context()
            .iter()
            .any(|(key, value)| *key == "operation" && value == "interop::DataFrame::try_borrow_table_view")
    );
}

#[test]
fn string_columns_borrow_as_binary_like_layouts() {
    let runtime = Runtime::builder().build().unwrap();
    let column =
        Column::from_strings_nullable_with_runtime(&runtime, &["zero", "one", "two"], &[true, false, true]).unwrap();
    let sliced = column.slice_view(1, 2).unwrap();
    let borrowed = sliced.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::String);
    assert_eq!(borrowed.len(), 2);
    assert_eq!(borrowed.offset(), 1);
    assert_eq!(borrowed.data_ptr(), std::ptr::null());
    assert_eq!(borrowed.semantic_columns().len(), 1);
    assert!(matches!(borrowed.column().unwrap().layout(), ColumnLayout::BinaryLike { .. }));

    match borrowed.null_mask().kind() {
        NullMaskKind::Present { ptr, bit_len, null_count } => {
            let _ = ptr.as_ptr();
            assert_eq!(*null_count, 1);
            assert!(*bit_len >= borrowed.offset() + borrowed.len());
        },
        NullMaskKind::Absent => panic!("expected a present null mask for nullable string column"),
    }
}

#[test]
fn dictionary_columns_borrow_with_recursive_string_keys() {
    let runtime = Runtime::builder().build().unwrap();
    let base = Column::from_strings_with_runtime(&runtime, &["beta", "alpha", "beta", "gamma"]).unwrap();
    let dict = dictionary::encode(&base.view().unwrap(), TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    let borrowed = dict_view.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::Dictionary32);
    assert!(matches!(borrowed.column().unwrap().layout(), ColumnLayout::Dictionary { .. }));
    assert_eq!(borrowed.semantic_columns().len(), 2);

    let keys_id = match borrowed.column().unwrap().layout() {
        ColumnLayout::Dictionary { keys, .. } => keys,
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };
    let keys_column = borrowed
        .semantic_columns()
        .iter()
        .find(|column| column.id() == *keys_id)
        .expect("dictionary keys semantic column should be present");
    assert_eq!(keys_column.data_type().id(), InteropTypeId::String);
    assert!(matches!(keys_column.layout(), ColumnLayout::BinaryLike { .. }));
}

// ---------------------------------------------------------------------------
// Nested-type constructor guards
//
// These tests verify that List and Struct columns cannot be created through the
// normal Column / Series construction APIs (the C++ layer rejects them in
// make_empty_column).  Because no current API produces a runtime-backed nested
// Column, we cannot directly exercise the borrowed-interop-v1 type guard
// (borrow_from_materialized_view) for List/Struct.  That guard independently
// rejects non-fixed-width types and child-bearing layouts; it is covered for
// other variable-width types by the string and dictionary tests above.
//
// If a future API (e.g. a runtime-aware JSON reader) makes it possible to
// produce runtime-backed nested columns, a test should be added here that
// constructs such a column and asserts that try_borrow_column_view() returns
// InteropErrorKind::UnsupportedType.
// ---------------------------------------------------------------------------

#[test]
fn nested_list_column_construction_is_rejected_before_interop() {
    let err = match Column::empty(cudf::TypeId::List) {
        Ok(_) => panic!("expected nested list column construction to fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
    assert!(err.to_string().contains("nested types"));
}

#[test]
fn nested_struct_series_construction_is_rejected_before_interop() {
    let err = match Series::empty("nested", cudf::TypeId::Struct) {
        Ok(_) => panic!("expected nested struct series construction to fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
    assert!(err.to_string().contains("nested types"));
}
