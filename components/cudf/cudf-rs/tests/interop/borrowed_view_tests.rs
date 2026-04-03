use cudf::{Column, DataFrame, DeviceId, Runtime, Series, Table};
use rapids_interop::{ExecutionBinding, NullMaskKind, TypeId};

#[test]
fn column_view_borrow_preserves_fixed_width_metadata_and_absent_mask() {
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = column.view().unwrap();

    let borrowed = view.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), TypeId::Int32);
    assert_eq!(borrowed.len(), 3);
    assert_eq!(borrowed.offset(), 0);
    assert_eq!(borrowed.data_ptr(), view.device_data_ptr().unwrap());
    assert!(matches!(borrowed.null_mask().kind(), NullMaskKind::Absent));

    match borrowed.execution() {
        ExecutionBinding::AmbientDefault { device_snapshot } => {
            assert_eq!(device_snapshot, DeviceId::current().unwrap().ordinal() as i32)
        },
        ExecutionBinding::Explicit { .. } => panic!("expected ambient execution binding"),
    }
}

#[test]
fn nullable_column_distinguishes_present_mask_with_zero_null_count() {
    let column = Column::from_slice_nullable(&[10i32, 20, 30], &[false, true, true]).unwrap();
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
    let column = Column::from_slice(&[11i32, 22, 33, 44]).unwrap();
    let base = column.view().unwrap();
    let sliced = column.slice_view(1, 2).unwrap();

    let borrowed = sliced.try_borrow_column_view().unwrap();

    assert_eq!(borrowed.type_id(), TypeId::Int32);
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

    match borrowed.execution() {
        ExecutionBinding::Explicit { device, cuda_stream, mr } => {
            assert_eq!(device, runtime.device().ordinal() as i32);
            assert_eq!(cuda_stream.as_raw() as usize, expected_stream);
            assert_eq!(mr.as_raw() as usize, expected_mr);
        },
        ExecutionBinding::AmbientDefault { .. } => panic!("expected explicit execution binding"),
    }
}

#[test]
fn series_and_dataframe_convenience_producers_match_underlying_low_level_views() {
    let ids = Series::new("id", &[1i32, 2, 3]).unwrap();
    let vals = Series::new("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_columns([("id", ids.clone()), ("value", vals.clone())]).unwrap();

    let series_borrowed = ids.try_borrow_column_view().unwrap();
    let frame_borrowed = frame.try_borrow_table_view().unwrap();

    assert_eq!(frame_borrowed.num_rows(), 3);
    assert_eq!(frame_borrowed.num_columns(), 2);
    assert_eq!(frame_borrowed.columns()[0], series_borrowed);
    assert_eq!(frame_borrowed.columns()[1], vals.try_borrow_column_view().unwrap());
}

#[test]
fn unsupported_string_columns_fail_clearly() {
    let column = Column::from_strings(&["alice", "bob"]).unwrap();
    let err = column.view().unwrap().try_borrow_column_view().unwrap_err();

    assert_eq!(err.kind(), rapids_interop::InteropErrorKind::UnsupportedType);
}
