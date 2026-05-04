//! Runtime device integration tests — single-GPU (always run).

use cudf::{DataFrame, DeviceId, Runtime, SortOptions};

#[test]
fn runtime_device_accessor_returns_build_device() {
    let rt = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    assert_eq!(rt.device(), DeviceId::new(0).unwrap());
}

#[test]
fn dataframe_ops_with_device_runtime() {
    let _rt = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();

    // Create data and seed with runtime's provenance.
    let ids = crate::common::series::<i32>("id", &[3, 1, 4, 1, 5]).unwrap();
    let scores = crate::common::series::<f64>("score", &[30.0, 10.0, 40.0, 15.0, 50.0]).unwrap();
    let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();

    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    let result: Vec<i32> = sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(result, vec![1, 1, 3, 4, 5]);
}
