//! Integration tests for `DeviceArrayView::to_typed_vec`.

use cugraph::{
    array::{DeviceArray, device_array_from_host},
    error::ErrorKind,
    handle::ResourceHandle,
    types::DataType,
};

#[test]
fn to_typed_vec_i32_roundtrip() {
    let handle = ResourceHandle::new().expect("handle");
    let src: Vec<i32> = vec![10, 20, 30, 40, 50];
    let arr = device_array_from_host(&handle, &src, DataType::Int32).expect("create");

    let result: Vec<i32> = arr.view().to_typed_vec(&handle).expect("to_typed_vec");
    assert_eq!(result, src);
}

#[test]
fn to_typed_vec_f32_roundtrip() {
    let handle = ResourceHandle::new().expect("handle");
    let src: Vec<f32> = vec![1.0, 2.5, 3.75];
    let arr = device_array_from_host(&handle, &src, DataType::Float32).expect("create");

    let result: Vec<f32> = arr.view().to_typed_vec(&handle).expect("to_typed_vec");
    assert_eq!(result, src);
}

#[test]
fn to_typed_vec_f64_roundtrip() {
    let handle = ResourceHandle::new().expect("handle");
    let src: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let arr = device_array_from_host(&handle, &src, DataType::Float64).expect("create");

    let result: Vec<f64> = arr.view().to_typed_vec(&handle).expect("to_typed_vec");
    assert_eq!(result, src);
}

#[test]
fn to_typed_vec_type_size_mismatch_returns_error() {
    let handle = ResourceHandle::new().expect("handle");
    let src: Vec<i32> = vec![1, 2, 3];
    let arr = device_array_from_host(&handle, &src, DataType::Int32).expect("create");

    // f64 is 8 bytes, Int32 is 4 bytes — must fail
    let err = arr.view().to_typed_vec::<f64>(&handle).expect_err("must reject size mismatch");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.operation(), "to_typed_vec");
    assert!(err.message().contains("size mismatch"));
}

#[test]
fn to_typed_vec_empty_array_returns_empty_vec() {
    let handle = ResourceHandle::new().expect("handle");
    let arr = DeviceArray::new(&handle, 0, DataType::Int32).expect("create empty");

    let result: Vec<i32> = arr.view().to_typed_vec(&handle).expect("to_typed_vec");
    assert!(result.is_empty());
}

#[test]
fn to_typed_vec_i64_roundtrip() {
    let handle = ResourceHandle::new().expect("handle");
    let src: Vec<i64> = vec![100_000_000_000, -42, 0, i64::MAX, i64::MIN];
    let arr = device_array_from_host(&handle, &src, DataType::Int64).expect("create");

    let result: Vec<i64> = arr.view().to_typed_vec(&handle).expect("to_typed_vec");
    assert_eq!(result, src);
}
