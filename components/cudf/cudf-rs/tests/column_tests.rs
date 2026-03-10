//! Tests for Column, ColumnView, and DeviceSlice.

use cudf::column::Column;
use cudf::types::{DataType, TypeId};

#[test]
fn test_create_empty_column() {
    let col = Column::empty(TypeId::Int32).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 0);
    assert!(view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap(), DataType::new(TypeId::Int32));
}

#[test]
fn test_create_i32_column() {
    let data = vec![1i32, 2, 3, 4, 5];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 5);
    assert!(!view.is_empty().unwrap());
    assert_eq!(view.data_type().unwrap().id(), TypeId::Int32);
    assert_eq!(view.null_count().unwrap(), 0);
    assert!(!view.nullable().unwrap());

    let host_data = view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, data);
}

#[test]
fn test_create_f64_column() {
    let data = vec![1.5f64, 2.5, 3.5];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.data_type().unwrap().id(), TypeId::Float64);

    let host_data = view.to_vec::<f64>().unwrap();
    assert_eq!(host_data, data);
}

#[test]
fn test_nullable_column() {
    let data = vec![10i32, 20, 30, 40];
    let validity = vec![true, false, true, false];
    let col = Column::from_slice_nullable(&data, &validity).unwrap();
    let view = col.view().unwrap();

    assert_eq!(view.len().unwrap(), 4);
    assert!(view.nullable().unwrap());
    assert_eq!(view.null_count().unwrap(), 2);

    let host_data = view.to_vec::<i32>().unwrap();
    assert_eq!(host_data, data);

    let mask = view.validity().unwrap().unwrap();
    assert_eq!(mask, validity);
}

#[test]
fn test_column_copy() {
    let data = vec![100i64, 200, 300];
    let col = Column::from_slice(&data).unwrap();
    let copy = col.copy().unwrap();
    let view = copy.view().unwrap();

    assert_eq!(view.len().unwrap(), 3);
    assert_eq!(view.to_vec::<i64>().unwrap(), data);
}

#[test]
fn test_type_mismatch_error() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();

    let result = view.to_vec::<f64>();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), cudf::ErrorKind::TypeError);
}

#[test]
fn test_u8_column() {
    let data = vec![0u8, 128, 255];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.to_vec::<u8>().unwrap(), data);
}

#[test]
fn test_mismatched_validity_length() {
    let data = vec![1i32, 2, 3];
    let validity = vec![true, false]; // wrong length
    let result = Column::from_slice_nullable(&data, &validity);
    assert!(result.is_err());
}

#[test]
fn test_validity_none_for_non_nullable_column() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    assert!(view.validity().unwrap().is_none());
}

#[test]
fn test_various_numeric_types() {
    // i8
    let col = Column::from_slice(&[1i8, -1, 127]).unwrap();
    assert_eq!(
        col.view().unwrap().to_vec::<i8>().unwrap(),
        vec![1i8, -1, 127]
    );

    // i16
    let col = Column::from_slice(&[1i16, -1, 32767]).unwrap();
    assert_eq!(
        col.view().unwrap().to_vec::<i16>().unwrap(),
        vec![1i16, -1, 32767]
    );

    // u16
    let col = Column::from_slice(&[0u16, 1000, 65535]).unwrap();
    assert_eq!(
        col.view().unwrap().to_vec::<u16>().unwrap(),
        vec![0u16, 1000, 65535]
    );

    // u32
    let col = Column::from_slice(&[0u32, 42, u32::MAX]).unwrap();
    assert_eq!(
        col.view().unwrap().to_vec::<u32>().unwrap(),
        vec![0u32, 42, u32::MAX]
    );

    // u64
    let col = Column::from_slice(&[0u64, 99]).unwrap();
    assert_eq!(col.view().unwrap().to_vec::<u64>().unwrap(), vec![0u64, 99]);

    // f32
    let col = Column::from_slice(&[1.0f32, -0.5, 3.125]).unwrap();
    let vals = col.view().unwrap().to_vec::<f32>().unwrap();
    assert!((vals[2] - 3.125).abs() < 0.001);
}

#[test]
fn test_device_data_ptr_i32() {
    let data = vec![10i32, 20, 30];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let ptr = view.device_data_ptr().unwrap();
    assert!(!ptr.is_null(), "device data pointer for i32 column must be non-null");
}

#[test]
fn test_device_data_ptr_f64() {
    let data = vec![1.5f64, 2.5, 3.5];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let ptr = view.device_data_ptr().unwrap();
    assert!(!ptr.is_null(), "device data pointer for f64 column must be non-null");
}

#[test]
fn test_device_data_ptr_consistent_with_to_vec() {
    // Two views of the same column should return the same device pointer.
    let col = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
    let view1 = col.view().unwrap();
    let view2 = col.view().unwrap();

    let ptr1 = view1.device_data_ptr().unwrap();
    let ptr2 = view2.device_data_ptr().unwrap();
    assert_eq!(ptr1, ptr2, "same column should yield same device pointer");
}

#[test]
fn test_offset_for_non_sliced_column() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.offset().unwrap(), 0);
}

#[test]
fn test_device_data_ptr_empty_column() {
    let col = Column::from_slice::<i32>(&[]).unwrap();
    let view = col.view().unwrap();
    // Empty columns may or may not have a non-null pointer — just ensure no crash.
    let _ptr = view.device_data_ptr().unwrap();
}

// --- DeviceSlice tests ---

#[test]
fn test_device_slice_i32() {
    let col = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let view = col.view().unwrap();

    let slice = view.device_slice::<i32>().unwrap();
    assert_eq!(slice.len(), 5);
    assert!(!slice.is_empty());
    assert_eq!(slice.type_id(), TypeId::Int32);
    assert_eq!(slice.elem_size(), 4);
    assert!(!slice.as_ptr().is_null());
}

#[test]
fn test_device_slice_f64() {
    let col = Column::from_slice(&[1.5f64, 2.5]).unwrap();
    let view = col.view().unwrap();

    let slice = view.device_slice::<f64>().unwrap();
    assert_eq!(slice.len(), 2);
    assert_eq!(slice.type_id(), TypeId::Float64);
    assert_eq!(slice.elem_size(), 8);
}

#[test]
fn test_device_slice_type_mismatch() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();

    let err = view.device_slice::<f64>().unwrap_err();
    assert_eq!(err.kind(), cudf::ErrorKind::TypeError);
}

#[test]
fn test_device_slice_split_at() {
    let col = Column::from_slice(&[10i32, 20, 30, 40, 50, 60]).unwrap();
    let view = col.view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    let (left, right) = slice.split_at(2);
    assert_eq!(left.len(), 2);
    assert_eq!(right.len(), 4);
    assert_eq!(left.type_id(), TypeId::Int32);
    assert_eq!(right.type_id(), TypeId::Int32);

    // Left pointer should equal the original pointer.
    assert_eq!(left.as_ptr(), slice.as_ptr());

    // Right pointer should be offset by 2 elements = 8 bytes.
    let expected_right = unsafe {
        (slice.as_ptr() as *const u8).add(2 * 4) as *mut std::ffi::c_void
    };
    assert_eq!(right.as_ptr(), expected_right);
}

#[test]
fn test_device_slice_split_at_zero() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    let (left, right) = slice.split_at(0);
    assert_eq!(left.len(), 0);
    assert_eq!(right.len(), 3);
}

#[test]
fn test_device_slice_split_at_end() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    let (left, right) = slice.split_at(3);
    assert_eq!(left.len(), 3);
    assert_eq!(right.len(), 0);
}

#[test]
#[should_panic(expected = "split_at: mid (4) > len (3)")]
fn test_device_slice_split_at_out_of_bounds() {
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    let slice = view.device_slice::<i32>().unwrap();

    slice.split_at(4); // should panic
}

#[test]
fn test_device_slice_empty_column() {
    let col = Column::from_slice::<i32>(&[]).unwrap();
    let view = col.view().unwrap();

    let slice = view.device_slice::<i32>().unwrap();
    assert_eq!(slice.len(), 0);
    assert!(slice.is_empty());
}

#[test]
fn test_device_slice_from_dictionary_indices() {
    // Dictionary-encode and get the indices as a DeviceSlice — the exact
    // workflow that gpu-app needs.
    let data = vec![10i32, 20, 10, 30, 20, 10];
    let col = Column::from_slice(&data).unwrap();
    let view = col.view().unwrap();

    let dict = cudf::dictionary::encode(&view, TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    let indices_view = cudf::dictionary::indices(&dict_view).unwrap();

    let slice = indices_view.device_slice::<i32>().unwrap();
    assert_eq!(slice.len(), 6);
    assert_eq!(slice.type_id(), TypeId::Int32);
    assert!(!slice.as_ptr().is_null());

    // Split into two halves (simulating src/dst edge split).
    let (src, dst) = slice.split_at(3);
    assert_eq!(src.len(), 3);
    assert_eq!(dst.len(), 3);
}
