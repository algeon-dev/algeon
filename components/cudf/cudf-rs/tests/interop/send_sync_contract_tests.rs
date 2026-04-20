use std::thread;

use cudf::{
    DataFrame, DeviceId, Runtime,
    core::{Column, column::DeviceSlice},
    interop::{
        arrow_device::{
            DeviceArrowColumnExport, DeviceArrowColumnOwnedAbiParts, DeviceArrowColumnRawParts, DeviceArrowTableExport,
            DeviceArrowTableOwnedAbiParts, DeviceArrowTableRawParts, ForeignDeviceArrowColumn, ForeignDeviceArrowTable,
        },
        runtime_release::{
            OwnedColumnTransferReleaser, OwnedTableTransferReleaser, PartitionedOwnedTableTransferReleaser,
        },
    },
    runtime::{CudaEvent, CudaStream},
    testing::prelude::*,
};
use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, CompletionAuthority, CompletionWitnessKind, LeaseCompletionResult,
    OwnedColumnTransfer, OwnedTableTransfer, PreparedColumnExport, PreparedTableExport, ProducerReadiness,
    ReadStartAuthority,
};
use static_assertions::{assert_impl_all, assert_not_impl_any};

use crate::common;

const SKIP_MULTI_GPU_ENV: &str = "CUDF_SKIP_MULTI_GPU";

assert_impl_all!(DeviceSlice<'static>: Send, Sync);
assert_impl_all!(ProducerReadiness<'static>: Send, Sync);
assert_impl_all!(OwnedColumnTransfer: Send, Sync);
assert_impl_all!(OwnedTableTransfer: Send, Sync);
assert_impl_all!(OwnedColumnTransferReleaser: Send, Sync);
assert_impl_all!(OwnedTableTransferReleaser: Send, Sync);
assert_impl_all!(PartitionedOwnedTableTransferReleaser: Send, Sync);
assert_impl_all!(PreparedColumnExport<'static>: Send, Sync);
assert_impl_all!(PreparedTableExport<'static>: Send, Sync);
assert_impl_all!(DeviceArrowColumnExport: Send, Sync);
assert_impl_all!(DeviceArrowColumnRawParts: Send, Sync);
assert_impl_all!(DeviceArrowColumnOwnedAbiParts: Send, Sync);
assert_impl_all!(ForeignDeviceArrowColumn: Send, Sync);
assert_impl_all!(DeviceArrowTableExport: Send, Sync);
assert_impl_all!(DeviceArrowTableRawParts: Send, Sync);
assert_impl_all!(DeviceArrowTableOwnedAbiParts: Send, Sync);
assert_impl_all!(ForeignDeviceArrowTable: Send, Sync);

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
    let column = runtime.column_from_slice(&[7i32, 8, 9]).unwrap();
    let expected_ptr = column.testing_view().unwrap().device_data_ptr().unwrap() as usize;
    let expected_stream = column.testing_execution_domain_stream_value();
    let expected_mr = column.testing_execution_domain_mr_value();

    let transfer = column.try_into_owned_transfer().unwrap();
    let (values, ptr, stream, mr) = thread::spawn(move || {
        let imported = Column::try_import_from_owned_device_column_transfer_into_runtime(transfer, &runtime).unwrap();
        let view = imported.testing_view().unwrap();
        let values = view.to_vec::<i32>().unwrap();
        let ptr = view.device_data_ptr().unwrap() as usize;
        let stream = imported.testing_execution_domain_stream_value();
        let mr = imported.testing_execution_domain_mr_value();
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
    let _observation_lock = common::lock_event_observation();
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
                    .try_activate_read_lease(
                        ReadStartAuthority::ConsumerStream {
                            stream: cudf::interop::cuda_stream_handle(&consumer_stream).unwrap(),
                        },
                        CompletionAuthority::SameDeviceStream {
                            stream: cudf::interop::cuda_stream_handle(&consumer_stream).unwrap(),
                        },
                    )
                    .unwrap();

                match lease.complete().unwrap() {
                    LeaseCompletionResult::Completed(witness) => {
                        assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
                        witness.synchronize().unwrap();
                        assert!(witness.is_ready().unwrap());
                    },
                    LeaseCompletionResult::FallbackSynchronized => {
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

    let _observation_lock = common::lock_event_observation();
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
        let frame = DataFrame::from_columns(::std::iter::IntoIterator::into_iter([ids, vals]).map(|series| {
            let name = series.name().expect("explicit test migration requires named Series inputs").to_string();
            (name, series)
        }))
        .unwrap();
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
                        .try_activate_read_lease(
                            ReadStartAuthority::ConsumerStream {
                                stream: cudf::interop::cuda_stream_handle(&consumer_stream).unwrap(),
                            },
                            CompletionAuthority::CrossDeviceEvent {
                                event: cudf::interop::cuda_event_handle(&consumer_event).unwrap(),
                            },
                        )
                        .unwrap();

                    match lease.complete().unwrap() {
                        LeaseCompletionResult::Completed(witness) => {
                            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
                            witness.synchronize().unwrap();
                            assert!(witness.is_ready().unwrap());
                        },
                        LeaseCompletionResult::FallbackSynchronized => {
                            panic!("expected producer-owned completion witness for cross-device event completion")
                        },
                    }
                })
                .join()
                .unwrap();
        });
    }
}
