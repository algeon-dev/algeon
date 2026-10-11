//! Installs memory managers and reads Parquet concurrently on up to four
//! GPUs, so it runs alone.
use std::{
    sync::{Arc, Barrier},
    thread,
};

use cudf::{
    DataFrame, DeviceId, DeviceMemoryManager, Exec, InstallSpec, ParquetReadOptions, Runtime, testing::prelude::*,
};
use tempfile::TempDir;

#[test]
#[ignore = "requires at least 2 GPUs"]
fn concurrent_parquet_reads_across_devices_do_not_need_a_process_global_io_lock() {
    let device_count = DeviceId::count().unwrap().min(4);
    assert!(device_count >= 2, "requires at least 2 GPUs");

    let dir = TempDir::new().unwrap();
    let mut fixtures = Vec::new();
    for ordinal in 0..device_count {
        let device = DeviceId::new(ordinal).unwrap();
        DeviceMemoryManager::install(device, InstallSpec::default()).unwrap();
        let runtime = Runtime::builder().with_device(device).build().unwrap();
        let values = (0..1024).map(|value| (ordinal as i32 * 10_000) + value).collect::<Vec<_>>();
        let frame = DataFrame::from_series([("id", runtime.series(&values).unwrap())]).unwrap();
        let path = dir.path().join(format!("device_{ordinal}.parquet"));
        frame.write_parquet(&path).unwrap();
        fixtures.push((device, path, values));
    }

    let workers = device_count * 2;
    let barrier = Arc::new(Barrier::new(workers));
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker_index in 0..workers {
            let (device, path, expected) = &fixtures[worker_index % fixtures.len()];
            let barrier = Arc::clone(&barrier);
            handles.push(scope.spawn(move || {
                let runtime = Runtime::builder().with_device(*device).build().unwrap();
                barrier.wait();
                for _ in 0..3 {
                    let frame = DataFrame::read_parquet(
                        ParquetReadOptions::builder().filepath(path).with_runtime(&runtime).build(),
                        Exec::RETAINED,
                    )
                    .unwrap();
                    assert_eq!(frame.testing_execution_domain_id(), runtime.testing_execution_domain_id());
                    assert_eq!(frame.num_rows(), expected.len());
                    let actual = frame.series("id").unwrap().to_vec_strict::<i32>().unwrap();
                    assert_eq!(actual.as_slice(), expected.as_slice());
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
    });
}
