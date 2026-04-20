use cudf::{
    DataFrame, Runtime, TypeId,
    core::{Column, Table},
    frame::dataframe::JsonReadOptions,
    ops::dictionary,
    testing::prelude::*,
};
use rapids_interop::{ColumnLayout, NullMaskKind, TryBorrowColumnView, TryBorrowTableView, TypeId as InteropTypeId};
use static_assertions::assert_impl_all;
use tempfile::TempDir;

use crate::common::column_from_slice;

assert_impl_all!(Column: TryBorrowColumnView);
assert_impl_all!(Table: TryBorrowTableView);

fn read_json_dataframe(filename: &str, rows: &str) -> DataFrame {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    std::fs::write(&path, rows).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::read_json_with_options(JsonReadOptions::builder().filepath(&path).with_runtime(&runtime).build())
        .unwrap()
}

#[test]
fn column_view_borrow_preserves_fixed_width_metadata_and_explicit_binding() {
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let view = column.testing_view().unwrap();
    let expected_stream = column.testing_execution_domain_stream_value();
    let expected_mr = column.testing_execution_domain_mr_value();

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
    let column = runtime.column_from_slice_nullable(&[10i32, 20, 30], &[false, true, true]).unwrap();
    let view = column.testing_slice_view(1, 2).unwrap();
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
    let column = runtime.column_from_slice(&[11i32, 22, 33, 44]).unwrap();
    let base = column.testing_view().unwrap();
    let sliced = column.testing_slice_view(1, 2).unwrap();

    let borrowed = sliced.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::Int32);
    assert_eq!(borrowed.len(), 2);
    assert_eq!(borrowed.offset(), 1);
    assert_eq!(borrowed.data_ptr(), base.device_data_ptr().unwrap());
}

#[test]
fn table_view_borrow_preserves_shape_and_explicit_execution_binding() {
    let runtime = Runtime::builder().build().unwrap();
    let a = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let b = runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
    let table = Table::from_columns(vec![a, b]).unwrap();
    let view = table.testing_view().unwrap();
    let expected_stream = table.testing_execution_domain_stream_value();
    let expected_mr = table.testing_execution_domain_mr_value();

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
    let column = runtime.column_from_slice(&[5i32, 6, 7]).unwrap();
    let view = column.testing_view().unwrap();

    let borrowed_from_column = column.try_borrow_column_view().unwrap();
    let borrowed_from_view = view.try_borrow_column_view().unwrap();

    assert_eq!(borrowed_from_column, borrowed_from_view);
}

#[test]
fn owned_table_borrow_matches_underlying_table_view_projection() {
    let runtime = Runtime::builder().build().unwrap();
    let a = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let b = runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
    let table = Table::from_columns(vec![a, b]).unwrap();
    let view = table.testing_view().unwrap();

    let borrowed_from_table = table.try_borrow_table_view().unwrap();
    let borrowed_from_view = view.try_borrow_table_view().unwrap();

    assert_eq!(borrowed_from_table, borrowed_from_view);
}

#[test]
fn series_and_dataframe_convenience_producers_match_underlying_low_level_views() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series(&[1i32, 2, 3]).unwrap();
    let vals = runtime.series(&[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_columns([("id", ids.clone()), ("value", vals.clone())]).unwrap();

    let series_borrowed = ids.try_borrow_column_view().unwrap();
    let frame_borrowed = frame.try_borrow_table_view().unwrap();

    assert_eq!(frame_borrowed.num_rows(), 3);
    assert_eq!(frame_borrowed.num_columns(), 2);
    assert_eq!(frame_borrowed.columns()[0], series_borrowed);
    assert_eq!(frame_borrowed.columns()[1], vals.try_borrow_column_view().unwrap());
}

#[test]
fn helper_built_column_view_borrow_uses_explicit_test_runtime_binding() {
    crate::common::with_test_runtime(|runtime| {
        let column = column_from_slice(&[1i32, 2, 3]).unwrap();
        let view = column.testing_view().unwrap();
        let borrowed = view.try_borrow_column_view().unwrap();

        assert_eq!(column.testing_execution_domain_kind(), "explicit");
        assert_eq!(borrowed.execution().device(), runtime.device().ordinal() as i32);
        assert_eq!(borrowed.execution().cuda_stream().as_raw() as usize, runtime.stream().testing_raw_handle_value());
        assert_eq!(
            borrowed.execution().memory_resource().as_raw() as usize,
            runtime.memory_resource().testing_raw_handle_value()
        );
    });
}

#[test]
fn helper_built_dataframe_borrow_uses_explicit_test_runtime_binding() {
    crate::common::with_test_runtime(|runtime| {
        let ids = crate::common::series("id", &[1i32, 2, 3]).unwrap();
        let vals = crate::common::series("value", &[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_columns([("id", ids), ("value", vals)]).unwrap();
        let borrowed = frame.try_borrow_table_view().unwrap();

        assert_eq!(frame.testing_execution_domain_kind(), "explicit");
        assert_eq!(borrowed.execution().device(), runtime.device().ordinal() as i32);
        assert_eq!(borrowed.execution().cuda_stream().as_raw() as usize, runtime.stream().testing_raw_handle_value());
        assert_eq!(
            borrowed.execution().memory_resource().as_raw() as usize,
            runtime.memory_resource().testing_raw_handle_value()
        );
    });
}

#[test]
fn string_columns_borrow_as_binary_like_layouts() {
    let runtime = Runtime::builder().build().unwrap();
    let column = runtime.column_from_strs_nullable(&["zero", "one", "two"], &[true, false, true]).unwrap();
    let sliced = column.testing_slice_view(1, 2).unwrap();
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
    let base = runtime.column_from_strs(&["beta", "alpha", "beta", "gamma"]).unwrap();
    let dict = dictionary::encode(&base.testing_view().unwrap(), TypeId::Int32).unwrap();
    let dict_view = dict.testing_view().unwrap();
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

#[test]
fn list_columns_borrow_as_recursive_list_layouts() {
    let frame = read_json_dataframe(
        "borrowed_lists.jsonl",
        "{\"nested\":[[1,2],[3]]}
\
         {\"nested\":[[],[4,5]]}
\
         {\"nested\":[[6]]}
",
    );
    let series = frame.series("nested").unwrap();

    let borrowed = series.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::List);
    assert_eq!(borrowed.data_ptr(), std::ptr::null());
    assert_eq!(borrowed.semantic_columns().len(), 3);

    let nested_child = match borrowed.column().unwrap().layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected list layout, got {layout:?}"),
    };
    let nested_list = borrowed
        .semantic_columns()
        .iter()
        .find(|column| column.id() == nested_child)
        .expect("nested list child should be present");
    let leaf = match nested_list.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let leaf_column =
        borrowed.semantic_columns().iter().find(|column| column.id() == leaf).expect("nested leaf should be present");

    assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
    assert_eq!(leaf_column.data_type().id(), InteropTypeId::Int64);
}

#[test]
fn struct_columns_borrow_as_recursive_struct_layouts() {
    let frame = read_json_dataframe(
        "borrowed_structs.jsonl",
        "{\"point\":{\"coords\":[1,2],\"label\":\"one\"}}
\
         {\"point\":{\"coords\":[],\"label\":\"two\"}}
\
         {\"point\":{\"coords\":[3],\"label\":\"three\"}}
",
    );
    let series = frame.series("point").unwrap();

    let borrowed = series.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), InteropTypeId::Struct);
    assert_eq!(borrowed.data_ptr(), std::ptr::null());

    let children = match borrowed.column().unwrap().layout() {
        ColumnLayout::Struct { children } => children.to_vec(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };
    assert_eq!(children.len(), 2);

    let coords = borrowed
        .semantic_columns()
        .iter()
        .find(|column| column.id() == children[0])
        .expect("coords child should be present");
    let label = borrowed
        .semantic_columns()
        .iter()
        .find(|column| column.id() == children[1])
        .expect("label child should be present");

    assert!(matches!(coords.layout(), ColumnLayout::List { .. }));
    assert!(matches!(label.layout(), ColumnLayout::BinaryLike { .. }));
}

#[test]
fn dataframe_borrow_supports_list_and_struct_columns() {
    let frame = read_json_dataframe(
        "borrowed_nested_frame.jsonl",
        "{\"nested\":[[1,2],[3]],\"point\":{\"coords\":[1,2],\"label\":\"one\"},\"id\":10}
\
         {\"nested\":[[],[4,5]],\"point\":{\"coords\":[],\"label\":\"two\"},\"id\":20}
\
         {\"nested\":[[6]],\"point\":{\"coords\":[3],\"label\":\"three\"},\"id\":30}
",
    );

    let borrowed = frame.try_borrow_table_view().unwrap();

    assert_eq!(borrowed.num_rows(), 3);
    assert_eq!(borrowed.num_columns(), 3);
    assert!(matches!(borrowed.columns()[0].column().unwrap().layout(), ColumnLayout::List { .. }));
    assert!(matches!(borrowed.columns()[1].column().unwrap().layout(), ColumnLayout::Struct { .. }));
    assert!(matches!(borrowed.columns()[2].column().unwrap().layout(), ColumnLayout::FixedWidth { .. }));
}

#[test]
fn nested_list_column_construction_is_rejected_before_interop() {
    let err = match crate::common::empty_column(cudf::TypeId::List) {
        Ok(_) => panic!("expected nested list column construction to fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
    assert!(err.to_string().contains("nested types"));
}

#[test]
fn nested_struct_series_construction_is_rejected_before_interop() {
    let err = match crate::common::empty_series("nested", cudf::TypeId::Struct) {
        Ok(_) => panic!("expected nested struct series construction to fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
    assert!(err.to_string().contains("nested types"));
}
