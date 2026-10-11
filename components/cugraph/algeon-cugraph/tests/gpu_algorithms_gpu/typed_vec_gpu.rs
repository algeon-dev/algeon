//! Integration tests for `DeviceArrayView::to_vec`.

use cugraph::{array::DeviceArray, handle::ResourceHandle};

use crate::common;

fn test_handle() -> ResourceHandle {
    let runtime = std::sync::Arc::new(cudf::Runtime::new().expect("runtime"));
    ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).expect("handle")
}

pub(super) fn to_vec_i64_roundtrip() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("to_vec_i64_roundtrip").expect("entered");
    let src: Vec<i64> = vec![100_000_000_000, -42, 0, i64::MAX, i64::MIN];
    let arr = DeviceArray::from_slice(&entered, &src).expect("create");

    let result: Vec<i64> = arr.view().expect("view").to_vec(&entered).expect("to_vec");
    assert_eq!(result, src);
}

pub(super) fn to_vec_empty_array_returns_empty_vec() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let handle = test_handle();
    let entered = handle.enter("to_vec_empty_array_returns_empty_vec").expect("entered");
    let arr = DeviceArray::<i32>::new(&entered, 0).expect("create empty");

    let result: Vec<i32> = arr.view().expect("view").to_vec(&entered).expect("to_vec");
    assert!(result.is_empty());
}
