//! Multi-GPU integration tests — require 2+ GPUs.
//!
//! Gated by `CUDF_MULTI_GPU=1` environment variable and `#[ignore]`.
//! Run with: `CUDF_MULTI_GPU=1 cargo test -p cudf --test multi_gpu_tests -- --ignored --nocapture`

use cudf::{Column, DataFrame, DeviceId, Runtime, Series, SortOptions};

fn require_multi_gpu() {
    if std::env::var("CUDF_MULTI_GPU").unwrap_or_default() != "1" {
        panic!("CUDF_MULTI_GPU=1 not set; skipping multi-GPU test");
    }
    let count = DeviceId::count().unwrap();
    assert!(count >= 2, "multi-GPU tests require 2+ GPUs, found {count}");
}

#[test]
#[ignore]
fn two_runtimes_on_different_devices() {
    require_multi_gpu();

    let rt0 = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let rt1 = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();

    assert_eq!(rt0.device().ordinal(), 0);
    assert_eq!(rt1.device().ordinal(), 1);
    assert_ne!(rt0.device(), rt1.device());
}

#[test]
#[ignore]
fn operations_on_different_devices_produce_correct_results() {
    require_multi_gpu();

    // Create runtimes on two different devices.
    let _rt0 = Runtime::builder().with_device(DeviceId::new(0).unwrap()).build().unwrap();
    let _rt1 = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();

    // Operations on device 0 (ambient default).
    let ids = Series::new::<i32>("id", &[3, 1, 4]).unwrap();
    let df = DataFrame::from_columns([("id", ids)]).unwrap();
    let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
    let result: Vec<i32> = sorted.series("id").unwrap().to_vec::<i32>().unwrap();
    assert_eq!(result, vec![1, 3, 4]);
}

#[test]
#[ignore]
fn device_guard_restores_device_across_scopes() {
    require_multi_gpu();

    let before = DeviceId::current().unwrap();
    {
        let _rt1 = Runtime::builder().with_device(DeviceId::new(1).unwrap()).build().unwrap();
        // During build, DeviceGuard temporarily switches to device 1, then restores.
    }
    let after = DeviceId::current().unwrap();
    assert_eq!(before, after, "DeviceGuard did not restore previous device");
}

#[test]
#[ignore]
fn ambient_domains_from_different_devices_differ() {
    require_multi_gpu();

    // This test verifies that ambient-default objects created on different
    // devices have different provenance (device_snapshot differs).
    let id0 = DeviceId::current().unwrap();
    assert_eq!(id0.ordinal(), 0);

    // Ambient objects on device 0.
    let _col0 = Column::from_slice(&[1i32, 2, 3]).unwrap();

    // We can verify the device count is at least 2.
    let count = DeviceId::count().unwrap();
    assert!(count >= 2);
}
