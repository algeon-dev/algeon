use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use cudf::{
    DeviceId, ErrorCode, ErrorKind, Runtime,
    core::{
        Table,
        column::Column,
        error::{FactKeyName, FactValue},
    },
    interop::{PartitionedDataFrameExport, runtime_release::TransferCompletionAuthority},
    runtime::CudaStream,
    testing::prelude::*,
};
use rapids_interop::{
    CompletionAuthority, ExportProfile, InteropErrorKind, LeaseCompletionResult, Placement, ProducerReadiness,
    ReadStartAuthority, ReleaseSynchronization, TransferCompletionRule,
};

use crate::common::{
    assert_duplicate_name_error, column_from_slice, error_context_value, interop_error_context_value,
    lock_event_observation,
};

const SKIP_MULTI_GPU_ENV: &str = "CUDF_SKIP_MULTI_GPU";

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

fn first_two_test_devices() -> (DeviceId, DeviceId) {
    (DeviceId::new(0).unwrap(), DeviceId::new(1).unwrap())
}

fn reset_event_observation() {
    unsafe { cudf_sys::cudf_test_event_reset_observation() };
}

fn observed_event_synchronize_call_count() -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_synchronize_call_count(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read event synchronize observation count");
    out
}

fn observed_raw_record_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream raw event-record observation count");
    out
}

fn block_event_synchronize() {
    unsafe { cudf_sys::cudf_test_event_block_synchronize() };
}

fn unblock_event_synchronize() {
    unsafe { cudf_sys::cudf_test_event_unblock_synchronize() };
}

struct EventSynchronizeBlockGuard;

impl EventSynchronizeBlockGuard {
    fn new() -> Self {
        block_event_synchronize();
        Self
    }
}

impl Drop for EventSynchronizeBlockGuard {
    fn drop(&mut self) {
        unblock_event_synchronize();
    }
}

fn wait_for_condition(timeout: Duration, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        thread::yield_now();
    }

    assert!(predicate(), "condition was not met within {:?}", timeout);
}

#[test]
fn partitioned_dataframe_export_groups_columns_by_retained_provenance_in_first_appearance_order() {
    let _observation_lock = lock_event_observation();
    let left = column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(7001).unwrap();
    let middle = column_from_slice(&[10i32, 20, 30]).unwrap().testing_rebind_execution_domain_explicit(7002).unwrap();
    let right = column_from_slice(&[100i32, 200, 300]).unwrap().testing_rebind_execution_domain_explicit(7001).unwrap();

    let producer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap();
    let export = producer.try_prepare_device_partitioned_table_export().unwrap();

    assert_eq!(export.num_rows(), 3);
    assert_eq!(export.schema_names().unwrap(), &["left".to_string(), "middle".to_string(), "right".to_string()]);
    assert_eq!(export.partitions().len(), 2);
    assert_eq!(export.partitions()[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(export.partitions()[1].top_level_column_indices(), &[1]);
    assert_eq!(export.partitions()[0].export().columns().len(), 2);
    assert_eq!(export.partitions()[1].export().columns().len(), 1);
    assert_eq!(export.partitions()[0].export().diagnostics().profile(), ExportProfile::PreparedDeviceExport);
    assert_eq!(export.partitions()[1].export().diagnostics().profile(), ExportProfile::PreparedDeviceExport);

    let first = export.partitions()[0]
        .export()
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    let second = export.partitions()[1]
        .export()
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(first.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    assert_eq!(second.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_dataframe_export_partition_wrapper_activates_directly() {
    let _observation_lock = lock_event_observation();
    let left = column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(7003).unwrap();
    let middle = column_from_slice(&[10i32, 20, 30]).unwrap().testing_rebind_execution_domain_explicit(7004).unwrap();
    let right = column_from_slice(&[100i32, 200, 300]).unwrap().testing_rebind_execution_domain_explicit(7003).unwrap();

    let producer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap();
    let export = producer.try_prepare_device_partitioned_table_export().unwrap();

    let first = export.partitions()[0]
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    let second = export.partitions()[1]
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(first.top_level_column_indices(), &[0, 2]);
    assert_eq!(second.top_level_column_indices(), &[1]);
    assert_eq!(first.export().columns().len(), 2);
    assert_eq!(second.export().columns().len(), 1);
    assert_eq!(first.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    assert_eq!(second.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_prepared_table_export_imports_one_local_table_per_partition_runtime() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_runtime = Runtime::builder().with_device(device0).build().unwrap();
    let right_runtime = Runtime::builder().with_device(device1).build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let middle = right_runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let right = left_runtime.column_from_slice(&[100i32, 200, 300]).unwrap();

    let producer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap();
    let export = producer.try_prepare_device_partitioned_table_export().unwrap();
    let _event_lock = lock_event_observation();

    let imported = Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes(
        &export,
        &[&left_runtime, &right_runtime],
    )
    .unwrap();

    assert_eq!(imported.num_rows(), 3);
    assert_eq!(imported.schema_names().unwrap(), &["left".to_string(), "middle".to_string(), "right".to_string()]);

    let partitions = imported.into_partitions();
    assert_eq!(partitions.len(), 2);

    let first = &partitions[0];
    assert_eq!(first.top_level_column_indices(), &[0, 2]);
    assert_eq!(first.table().testing_view().unwrap().num_columns().unwrap(), 2);
    assert_eq!(first.table().testing_execution_domain_id(), left_runtime.testing_execution_domain_id());
    assert_eq!(first.table().testing_execution_domain_stream_value(), left_runtime.stream().testing_raw_handle_value());

    let first_view = first.table().testing_view().unwrap();
    assert_eq!(first_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(first_view.column(1).unwrap().to_vec::<i32>().unwrap(), vec![100, 200, 300]);

    let second = &partitions[1];
    assert_eq!(second.top_level_column_indices(), &[1]);
    assert_eq!(second.table().testing_view().unwrap().num_columns().unwrap(), 1);
    assert_eq!(second.table().testing_execution_domain_id(), right_runtime.testing_execution_domain_id());
    assert_eq!(
        second.table().testing_execution_domain_stream_value(),
        right_runtime.stream().testing_raw_handle_value()
    );

    let second_view = second.table().testing_view().unwrap();
    assert_eq!(second_view.column(0).unwrap().to_vec::<i32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn partitioned_prepared_table_export_import_rejects_runtime_count_mismatch_with_stable_contract() {
    let left_runtime = Runtime::builder().build().unwrap();
    let right_runtime = Runtime::builder().build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let right = right_runtime.column_from_slice(&[10i32, 20, 30]).unwrap();

    let producer = PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![left, right]).unwrap();
    let export = producer.try_prepare_device_partitioned_table_export().unwrap();

    let err =
        Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes(&export, &[&left_runtime])
            .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ExecutionBindingMismatch);
    assert_eq!(
        interop_error_context_value(&err, "operation").as_deref(),
        Some("interop::Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes"),
    );
    assert_eq!(interop_error_context_value(&err, "resource").as_deref(), Some("execution_bindings"));
    assert_eq!(interop_error_context_value(&err, "reason").as_deref(), Some("partition_runtime_count_mismatch"));
    assert_eq!(interop_error_context_value(&err, "expected_count").as_deref(), Some("2"));
    assert_eq!(interop_error_context_value(&err, "actual_count").as_deref(), Some("1"));
    assert_eq!(
        interop_error_context_value(&err, "remedy").as_deref(),
        Some("provide exactly one destination runtime per partition"),
    );
    assert_eq!(interop_error_context_value(&err, "partition_count").as_deref(), None);
    assert_eq!(interop_error_context_value(&err, "runtime_count").as_deref(), None);
}

#[test]
fn partitioned_prepared_table_export_import_rejects_runtime_on_wrong_device_for_a_partition() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let left_runtime = Runtime::builder().with_device(device0).build().unwrap();
    let right_runtime = Runtime::builder().with_device(device1).build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let middle = right_runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let right = left_runtime.column_from_slice(&[100i32, 200, 300]).unwrap();

    let producer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap();
    let export = producer.try_prepare_device_partitioned_table_export().unwrap();

    let err = Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes(
        &export,
        &[&right_runtime, &left_runtime],
    )
    .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ExecutionBindingMismatch);
    assert!(
        err.context().iter().any(|(key, value)| *key == "partition_index" && value == "0"),
        "expected partition-local mismatch context, got {:?}",
        err.context()
    );
    assert_eq!(
        interop_error_context_value(&err, "left_device").as_deref(),
        Some(device0.ordinal().to_string().as_str())
    );
    assert_eq!(
        interop_error_context_value(&err, "right_device").as_deref(),
        Some(device1.ordinal().to_string().as_str())
    );
    assert_eq!(interop_error_context_value(&err, "producer_device").as_deref(), None);
    assert_eq!(interop_error_context_value(&err, "runtime_device").as_deref(), None);
}

#[test]
fn partitioned_dataframe_export_prepare_host_table_export_returns_one_host_visible_root() {
    let left = column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(7101).unwrap();
    let right = column_from_slice(&[4i32, 5, 6]).unwrap().testing_rebind_execution_domain_explicit(7102).unwrap();

    let export = PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![left, right]).unwrap();
    let prepared = export.try_prepare_host_table_export().unwrap();

    assert_eq!(prepared.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(prepared.table().num_rows(), 3);
    assert_eq!(prepared.table().names().unwrap(), &["left".to_string(), "right".to_string()]);
    assert_eq!(prepared.columns().len(), 2);
    assert!(prepared.bindings().iter().all(|binding| matches!(binding.placement(), Placement::Host)));
    assert!(matches!(prepared.readiness(), ProducerReadiness::HostVisible));

    let lease = prepared
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn partitioned_dataframe_export_owned_transfer_returns_one_partition_per_retained_provenance() {
    let left = column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(7201).unwrap();
    let middle = column_from_slice(&[10i32, 20, 30]).unwrap().testing_rebind_execution_domain_explicit(7202).unwrap();
    let right = column_from_slice(&[100i32, 200, 300]).unwrap().testing_rebind_execution_domain_explicit(7201).unwrap();

    let transfer = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap()
    .try_into_owned_transfer()
    .unwrap();

    assert_eq!(transfer.num_rows(), 3);
    assert_eq!(transfer.schema_names().unwrap(), &["left".to_string(), "middle".to_string(), "right".to_string()]);
    assert_eq!(transfer.partitions().len(), 2);
    assert_eq!(transfer.partitions()[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(transfer.partitions()[1].top_level_column_indices(), &[1]);
    assert_eq!(transfer.partitions()[0].transfer().table().num_rows(), 3);
    assert_eq!(transfer.partitions()[0].transfer().columns().len(), 2);
    assert_eq!(transfer.partitions()[1].transfer().columns().len(), 1);
}

#[test]
fn partitioned_dataframe_export_runtime_release_owned_transfer_preserves_async_contract_per_partition() {
    let left_runtime = Runtime::builder().build().unwrap();
    let right_runtime = Runtime::builder().build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let middle = right_runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let right = left_runtime.column_from_slice(&[100i32, 200, 300]).unwrap();

    let (transfer, releaser) = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap()
    .try_into_owned_transfer_with_release()
    .unwrap();

    assert_eq!(transfer.num_rows(), 3);
    assert_eq!(transfer.partitions().len(), 2);
    assert_eq!(transfer.partitions()[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(transfer.partitions()[1].top_level_column_indices(), &[1]);
    for partition in transfer.partitions() {
        assert_eq!(
            partition.transfer().release_contract().completion(),
            TransferCompletionRule::ReleaseWaitsOnConsumerSignal
        );
        assert_eq!(partition.transfer().release_contract().synchronization(), ReleaseSynchronization::Asynchronous);
    }

    let releaser_partitions = releaser.into_partitions();
    assert_eq!(releaser_partitions.len(), 2);
    assert_eq!(releaser_partitions[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(releaser_partitions[1].top_level_column_indices(), &[1]);
}

#[test]
fn partitioned_dataframe_export_runtime_release_synchronization_preserves_blocking_signal_contract_per_partition() {
    let left_runtime = Runtime::builder().build().unwrap();
    let right_runtime = Runtime::builder().build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let middle = right_runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let right = left_runtime.column_from_slice(&[100i32, 200, 300]).unwrap();

    let (transfer, releaser) = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap()
    .try_into_owned_transfer_with_release_synchronization(ReleaseSynchronization::MaySynchronizeInternally)
    .unwrap();

    assert_eq!(transfer.num_rows(), 3);
    assert_eq!(transfer.partitions().len(), 2);
    for partition in transfer.partitions() {
        assert_eq!(
            partition.transfer().release_contract().completion(),
            TransferCompletionRule::ReleaseWaitsOnConsumerSignal
        );
        assert_eq!(
            partition.transfer().release_contract().synchronization(),
            ReleaseSynchronization::MaySynchronizeInternally
        );
    }

    let releaser_partitions = releaser.into_partitions();
    assert_eq!(releaser_partitions.len(), 2);
    assert_eq!(releaser_partitions[0].top_level_column_indices(), &[0, 2]);
    assert_eq!(releaser_partitions[1].top_level_column_indices(), &[1]);
}

#[test]
fn partitioned_dataframe_export_runtime_release_accepts_per_partition_release_without_blocking_caller_threads() {
    let _observation_lock = lock_event_observation();
    let left_runtime = Runtime::builder().build().unwrap();
    let right_runtime = Runtime::builder().build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let middle = right_runtime.column_from_slice(&[10i32, 20, 30]).unwrap();
    let right = left_runtime.column_from_slice(&[100i32, 200, 300]).unwrap();

    let (transfer, releaser) = PartitionedDataFrameExport::new(
        vec!["left".into(), "middle".into(), "right".into()],
        vec![left, middle, right],
    )
    .unwrap()
    .try_into_owned_transfer_with_release()
    .unwrap();
    drop(transfer);

    let first_consumer_stream = CudaStream::new().unwrap();
    let first_consumer_stream_raw = first_consumer_stream.testing_raw_handle_value();
    let second_consumer_stream = CudaStream::new().unwrap();
    let second_consumer_stream_raw = second_consumer_stream.testing_raw_handle_value();

    reset_event_observation();
    let _sync_block = EventSynchronizeBlockGuard::new();
    let baseline_sync = observed_event_synchronize_call_count();
    let baseline_first_record = observed_raw_record_call_count_for_stream_raw(first_consumer_stream_raw);
    let baseline_second_record = observed_raw_record_call_count_for_stream_raw(second_consumer_stream_raw);

    let mut releaser_partitions = releaser.into_partitions().into_iter();
    let first_partition = releaser_partitions.next().expect("missing first partition releaser");
    let second_partition = releaser_partitions.next().expect("missing second partition releaser");
    assert!(releaser_partitions.next().is_none());

    let (tx, rx) = mpsc::channel();
    let second_tx = tx.clone();
    let first_release_thread = thread::spawn(move || {
        let result = first_partition.release(TransferCompletionAuthority::same_device_stream(&first_consumer_stream));
        tx.send(result).unwrap();
    });
    let second_release_thread = thread::spawn(move || {
        let result = second_partition.release(TransferCompletionAuthority::same_device_stream(&second_consumer_stream));
        second_tx.send(result).unwrap();
    });

    rx.recv_timeout(Duration::from_secs(2))
        .expect("first partition runtime-backed release did not return promptly")
        .unwrap();
    rx.recv_timeout(Duration::from_secs(2))
        .expect("second partition runtime-backed release did not return promptly")
        .unwrap();

    wait_for_condition(Duration::from_secs(2), || {
        observed_raw_record_call_count_for_stream_raw(first_consumer_stream_raw) > baseline_first_record
    });
    wait_for_condition(Duration::from_secs(2), || {
        observed_raw_record_call_count_for_stream_raw(second_consumer_stream_raw) > baseline_second_record
    });
    wait_for_condition(Duration::from_secs(2), || observed_event_synchronize_call_count() >= baseline_sync + 2);

    first_release_thread.join().unwrap();
    second_release_thread.join().unwrap();
}

#[test]
fn partitioned_dataframe_export_runtime_release_import_and_release_remain_mutually_exclusive_per_partition() {
    let _observation_lock = lock_event_observation();
    let left_runtime = Runtime::builder().build().unwrap();
    let right_runtime = Runtime::builder().build().unwrap();
    let left = left_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
    let right = right_runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
    let left_expected_ptr = left.testing_view().unwrap().device_data_ptr().unwrap();

    let (transfer, releaser) = PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![left, right])
        .unwrap()
        .try_into_owned_transfer_with_release()
        .unwrap();

    let mut transfer_partitions = transfer.into_partitions().into_iter();
    let left_transfer = transfer_partitions.next().expect("missing left partition transfer").into_transfer();
    let right_transfer = transfer_partitions.next().expect("missing right partition transfer").into_transfer();
    assert!(transfer_partitions.next().is_none());

    let mut releaser_partitions = releaser.into_partitions().into_iter();
    let left_releaser = releaser_partitions.next().expect("missing left partition releaser");
    let right_releaser = releaser_partitions.next().expect("missing right partition releaser");
    assert!(releaser_partitions.next().is_none());

    let imported_left =
        Table::try_import_from_owned_device_table_transfer_into_runtime(left_transfer, &left_runtime).unwrap();
    let imported_left_view = imported_left.testing_view().unwrap();
    assert_eq!(imported_left_view.num_columns().unwrap(), 1);
    assert_eq!(imported_left_view.column(0).unwrap().device_data_ptr().unwrap(), left_expected_ptr);

    let consumer_stream = CudaStream::new().unwrap();
    let left_release_err =
        left_releaser.release(TransferCompletionAuthority::same_device_stream(&consumer_stream)).unwrap_err();
    assert_eq!(left_release_err.kind(), InteropErrorKind::ProfileUnsupported);

    let _sync_block = EventSynchronizeBlockGuard::new();
    right_releaser.release(TransferCompletionAuthority::same_device_stream(&consumer_stream)).unwrap();
    let right_import_err =
        match Table::try_import_from_owned_device_table_transfer_into_runtime(right_transfer, &right_runtime) {
            Ok(_) => panic!("expected partition import to fail after release was accepted"),
            Err(err) => err,
        };
    assert_eq!(right_import_err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn ordinary_dataframe_rejects_mixed_provenance_but_partitioned_export_accepts_it() {
    // Ordinary DataFrame rejects mixed provenance at construction time.
    let left = column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(7301).unwrap();
    let right = column_from_slice(&[4i32, 5, 6]).unwrap().testing_rebind_execution_domain_explicit(7302).unwrap();
    let frame = crate::common::dataframe_from_columns(vec!["left".into(), "right".into()], vec![left, right]);
    assert!(frame.is_err(), "DataFrame::new should reject mixed-provenance columns");

    // The same provenance layout succeeds through the partitioned export path,
    // which is designed to handle heterogeneous provenance by grouping columns
    // into per-provenance partitions.
    let left2 = column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(7301).unwrap();
    let right2 = column_from_slice(&[4i32, 5, 6]).unwrap().testing_rebind_execution_domain_explicit(7302).unwrap();
    let export = PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![left2, right2])
        .expect("PartitionedDataFrameExport should accept mixed-provenance columns");

    let prepared = export.try_prepare_device_partitioned_table_export().unwrap();
    assert_eq!(prepared.partitions().len(), 2, "mixed provenance should produce two partitions");
}

#[test]
fn partitioned_dataframe_export_rejects_duplicate_names_in_wide_schema() {
    let names: Vec<String> = (0..12).map(|idx| if idx == 11 { "c3".to_string() } else { format!("c{idx}") }).collect();
    let columns: Vec<Column> = (0..12)
        .map(|idx| {
            column_from_slice(&[idx, idx + 100])
                .unwrap()
                .testing_rebind_execution_domain_explicit(7400 + idx as u64)
                .unwrap()
        })
        .collect();

    let result = PartitionedDataFrameExport::new(names, columns);
    let err = match result {
        Ok(_) => panic!("expected duplicate column name error"),
        Err(err) => err,
    };
    assert_duplicate_name_error(&err, "c3");
}

#[test]
fn partitioned_dataframe_export_rejects_name_count_mismatch_with_canonical_context() {
    let column = column_from_slice(&[1i32, 2]).unwrap();

    let err = match PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![column]) {
        Ok(_) => panic!("name/column count mismatch should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "interop::PartitionedDataFrameExport::new");
    assert_eq!(error_context_value(&err, "column_name_count").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "table_column_count").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "resource").as_deref(), Some("column_name_count"));
    assert_eq!(error_context_value(&err, "reason").as_deref(), Some("mismatch_with_table_column_count"));
    assert_eq!(
        error_context_value(&err, "remedy").as_deref(),
        Some("ensure column_name_count matches table_column_count")
    );
    assert_eq!(error_context_value(&err, "names_count").as_deref(), None);
    assert_eq!(error_context_value(&err, "columns_count").as_deref(), None);
}

#[test]
fn partitioned_dataframe_export_rejects_row_count_mismatch_with_canonical_facts() {
    let left = column_from_slice(&[1i32, 2, 3]).unwrap();
    let right = column_from_slice(&[10i32, 20]).unwrap();

    let err = match PartitionedDataFrameExport::new(vec!["left".into(), "right".into()], vec![left, right]) {
        Ok(_) => panic!("row-count mismatch should fail"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert_eq!(err.code(), ErrorCode::ColumnLengthMismatch);
    assert_eq!(err.operation(), "interop::PartitionedDataFrameExport::new");
    assert_eq!(err.fact_value(FactKeyName::ColumnIndex), Some(&FactValue::Usize(1)));
    assert_eq!(err.fact_value(FactKeyName::ActualRows), Some(&FactValue::Usize(2)));
    assert_eq!(err.fact_value(FactKeyName::ExpectedRows), Some(&FactValue::Usize(3)));
    assert_eq!(error_context_value(&err, "column_index").as_deref(), Some("1"));
    assert_eq!(error_context_value(&err, "actual_rows").as_deref(), Some("2"));
    assert_eq!(error_context_value(&err, "expected_rows").as_deref(), Some("3"));
    assert_eq!(err.detail_value("column_index"), None);
    assert_eq!(err.detail_value("actual_rows"), None);
    assert_eq!(err.detail_value("expected_rows"), None);
}
