use std::thread;

use cudf::{
    DataFrame, DeviceId, Runtime,
    core::{Column, column::DeviceSlice},
    interop::{
        BorrowedColumnView, BorrowedTableView, CompletionWitnessKind, OwnedColumnTransfer, OwnedTableTransfer,
        PartitionedRuntimeOwnedTableTransferReleaser, PreparedColumnExport, PreparedTableExport, ProducerReadiness,
        RuntimeCompletionAuthority, RuntimeLeaseCompletionResult, RuntimeOwnedColumnTransferReleaser,
        RuntimeOwnedTableTransferReleaser, RuntimeReadStartAuthority, TryActivateRuntimeColumnReadLease,
        TryActivateRuntimeTableReadLease,
    },
    runtime::{CudaEvent, CudaStream},
};
use static_assertions::{assert_impl_all, assert_not_impl_any};

use crate::common;

const SKIP_MULTI_GPU_ENV: &str = "CUDF_SKIP_MULTI_GPU";

assert_impl_all!(DeviceSlice<'static>: Send, Sync);
assert_impl_all!(ProducerReadiness<'static>: Send, Sync);
assert_impl_all!(OwnedColumnTransfer: Send, Sync);
assert_impl_all!(OwnedTableTransfer: Send, Sync);
assert_impl_all!(RuntimeOwnedColumnTransferReleaser: Send, Sync);
assert_impl_all!(RuntimeOwnedTableTransferReleaser: Send, Sync);
assert_impl_all!(PartitionedRuntimeOwnedTableTransferReleaser: Send, Sync);
assert_impl_all!(PreparedColumnExport<'static>: Send, Sync);
assert_impl_all!(PreparedTableExport<'static>: Send, Sync);

assert_not_impl_any!(BorrowedColumnView<'static>: Send, Sync);
assert_not_impl_any!(BorrowedTableView<'static>: Send, Sync);

fn should_run_multi_gpu_test() -> bool {
    if std::env::var(SKIP_MULTI_GPU_ENV).unwrap_or_default() == "1" {
        eprintln!("skipping multi-GPU test because {SKIP_MULTI_GPU_ENV}=1");
        return false;
    }

    let count = DeviceId::count().expect("failed to query visible CUDA devices");
    if count < 2 {
        eprintln!("skipping multi-GPU test because only {count} CUDA device is visible");
        return false;
    }

    true
}

fn available_test_devices() -> Vec<DeviceId> {
    let count = DeviceId::count().expect("failed to query visible CUDA devices");
    (0..count).map(|ordinal| DeviceId::new(ordinal).expect("device count returned an invalid ordinal")).collect()
}

#[test]
fn owned_column_transfer_round_trips_across_thread_boundary() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[7i32, 8, 9]).unwrap();
    let expected_ptr = column.view().unwrap().device_data_ptr().unwrap() as usize;
    let expected_stream = column.__debug_execution_domain_stream_value();
    let expected_mr = column.__debug_execution_domain_mr_value();

    let transfer = column.try_into_owned_device_column_transfer().unwrap();
    let (values, ptr, stream, mr) = thread::spawn(move || {
        let imported = Column::try_from_owned_device_column_transfer(transfer).unwrap();
        let view = imported.view().unwrap();
        let values = view.to_vec::<i32>().unwrap();
        let ptr = view.device_data_ptr().unwrap() as usize;
        let stream = imported.__debug_execution_domain_stream_value();
        let mr = imported.__debug_execution_domain_mr_value();
        (values, ptr, stream, mr)
    })
    .join()
    .unwrap();

    assert_eq!(values, vec![7, 8, 9]);
    assert_eq!(ptr, expected_ptr);
    assert_eq!(stream, expected_stream);
    assert_eq!(mr, expected_mr);
}

#[test]
fn prepared_column_export_can_activate_and_complete_on_worker_thread() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    thread::scope(|scope| {
        scope
            .spawn(move || {
                let lease = export
                    .try_activate_runtime_read_lease(
                        RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                        RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
                    )
                    .unwrap();

                match lease.complete().unwrap() {
                    RuntimeLeaseCompletionResult::Completed(witness) => {
                        assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
                        assert_eq!(witness.device(), producer_runtime.device());
                        witness.synchronize().unwrap();
                        assert!(witness.is_ready().unwrap());
                    },
                    RuntimeLeaseCompletionResult::FallbackSynchronized => {
                        panic!("expected producer-owned completion witness for same-device stream completion")
                    },
                }
            })
            .join()
            .unwrap();
    });
}

#[test]
fn prepared_table_export_survives_cross_thread_cross_device_completion_stress() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let _device_guard = common::DeviceRestoreGuard::new();
    let devices = available_test_devices();
    let producer_device = devices[0];
    let consumer_devices = &devices[1..];
    let rounds = consumer_devices.len().max(1) * 2;

    for round in 0..rounds {
        let consumer_device = consumer_devices[round % consumer_devices.len()];

        common::set_current_device(producer_device);
        let producer_runtime = Runtime::builder().with_device(producer_device).build().unwrap();
        let ids = producer_runtime.series("id", &[1i32, 2, 3, round as i32]).unwrap();
        let vals = producer_runtime.series("value", &[10i64, 20, 30, 40]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(producer_runtime.stream()).unwrap();
        let export = frame.try_prepare_device_table_export_with_event(&producer_ready).unwrap();

        thread::scope(|scope| {
            scope
                .spawn(move || {
                    common::set_current_device(consumer_device);
                    let consumer_stream = CudaStream::new().unwrap();
                    let consumer_runtime = Runtime::builder()
                        .with_device(consumer_device)
                        .with_stream(consumer_stream.clone())
                        .build()
                        .unwrap();
                    let _consumer_work = consumer_runtime.series("tmp", &[7i32, 8, 9, round as i32]).unwrap();
                    let consumer_event = CudaEvent::new().unwrap();
                    consumer_event.record(&consumer_stream).unwrap();

                    let lease = export
                        .try_activate_runtime_read_lease(
                            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                            RuntimeCompletionAuthority::cross_device_event(&consumer_event),
                        )
                        .unwrap();

                    match lease.complete().unwrap() {
                        RuntimeLeaseCompletionResult::Completed(witness) => {
                            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
                            assert_eq!(witness.device(), producer_device);
                            witness.synchronize().unwrap();
                            assert!(witness.is_ready().unwrap());
                        },
                        RuntimeLeaseCompletionResult::FallbackSynchronized => {
                            panic!("expected producer-owned completion witness for cross-device event completion")
                        },
                    }
                })
                .join()
                .unwrap();
        });
    }
}
