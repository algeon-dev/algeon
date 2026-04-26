//! Integration tests for `DeviceArrayView::to_vec`.

use cugraph::{array::DeviceArray, handle::ResourceHandle};

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(cudf::Runtime::new().expect("runtime"));
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

#[test]
fn to_vec_i64_roundtrip() {
    let handle = test_handle();
    let src: Vec<i64> = vec![100_000_000_000, -42, 0, i64::MAX, i64::MIN];
    let arr = DeviceArray::from_slice(&handle, &src).expect("create");

    let result: Vec<i64> = arr.view().expect("view").to_vec(&handle).expect("to_vec");
    assert_eq!(result, src);
}

#[test]
fn to_vec_empty_array_returns_empty_vec() {
    let handle = test_handle();
    let arr = DeviceArray::<i32>::new(&handle, 0).expect("create empty");

    let result: Vec<i32> = arr.view().expect("view").to_vec(&handle).expect("to_vec");
    assert!(result.is_empty());
}
