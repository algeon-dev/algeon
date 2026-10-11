use cugraph::DeviceArray;

use crate::test_handle;

pub(super) fn test_device_array_roundtrip() {
    let handle = test_handle();
    let entered = handle.enter("test_device_array_roundtrip").expect("entered");
    let arr = DeviceArray::from_slice(&entered, &[10i32, 20, 30, 40, 50]).expect("create");

    let view = arr.view().expect("view");
    assert_eq!(view.len(), 5);

    let result: Vec<i32> = view.to_vec(&entered).expect("to_vec");
    assert_eq!(result, vec![10, 20, 30, 40, 50]);
}

pub(super) fn test_device_array_f32_roundtrip() {
    let handle = test_handle();
    let entered = handle.enter("test_device_array_f32_roundtrip").expect("entered");
    let arr = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0]).expect("create");

    let view = arr.view().expect("view");
    let result: Vec<f32> = view.to_vec(&entered).expect("to_vec");
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

pub(super) fn test_to_vec_roundtrip_f64() {
    let handle = test_handle();
    let entered = handle.enter("test_to_vec_roundtrip_f64").expect("entered");
    let src: Vec<f64> = vec![1.5, 2.5, 3.5];
    let arr = DeviceArray::from_slice(&entered, &src).expect("create");
    let result: Vec<f64> = arr.view().expect("view").to_vec(&entered).expect("roundtrip");
    assert_eq!(result, src);
}
