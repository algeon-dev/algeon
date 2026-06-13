//! Query-backend substrate contract tests.
//!
//! These tests lock the current library-level execution and host/device
//! boundary semantics before the execution resolver refactor centralizes the
//! implementation.

use std::sync::{Arc, Mutex};

use arrow::{
    array::{Array, Int32Array},
    datatypes::{DataType as ArrowDataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::{
    BoundaryKind, DataFrame, ExecutionOverrides, Runtime,
    ops::expr::col,
    testing::{self, prelude::*},
};
use rapids_interop::{CompletionAuthority, ReadStartAuthority};

use crate::common::{
    last_exec_mr_value, last_exec_stream_value, last_sync_stream_value, lock_event_observation, reset_exec_observation,
    reset_sync_observation, sample_runtime,
};

fn assert_frame_residency(df: &DataFrame, expected_id: u64, expected_stream: usize, expected_mr: usize) {
    assert_eq!(df.testing_execution_domain_kind(), "explicit");
    assert_eq!(df.testing_execution_domain_id(), expected_id);
    assert_eq!(df.testing_execution_domain_stream_value(), expected_stream);
    assert_eq!(df.testing_execution_domain_mr_value(), expected_mr);
}

fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array")
}

fn sample_record_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int32, false),
        Field::new("value", ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as Arc<dyn Array>,
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as Arc<dyn Array>,
        ],
    )
    .unwrap()
}

#[test]
fn host_record_batch_import_and_export_are_observable_boundaries() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let runtime_stream = runtime.stream().testing_raw_handle_value();
    let runtime_mr = runtime.memory_resource().testing_raw_handle_value();

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let import_count_before = testing::arrow::arrow_host_import_call_count_for_test();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let import_count_after = testing::arrow::arrow_host_import_call_count_for_test();
    let import_observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(import_count_after >= import_count_before + 1);
    assert_eq!(import_observation.kind, BoundaryKind::HostToDevice);
    assert_eq!(import_observation.placement, runtime.placement());
    assert_eq!(import_observation.rows, 4);
    assert_eq!(import_observation.columns, 2);
    assert!(!import_observation.synchronized_for_host);
    assert_frame_residency(&df, runtime.testing_execution_domain_id(), runtime_stream, runtime_mr);
    assert_eq!(df.column_names(), vec!["id", "value"]);

    reset_sync_observation();
    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let export_count_before = testing::arrow::arrow_host_export_call_count_for_test();
    let exported = df.to_record_batch().unwrap();
    let export_observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(testing::arrow::arrow_host_export_call_count_for_test() >= export_count_before + 1);
    assert_eq!(export_observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(export_observation.placement, df.placement());
    assert_eq!(export_observation.rows, 4);
    assert_eq!(export_observation.columns, 2);
    assert!(export_observation.synchronized_for_host);
    assert_eq!(last_sync_stream_value(), runtime_stream);
    assert_eq!(int32_array(&exported, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(int32_array(&exported, 1).values().as_ref(), &[10, 20, 30, 40]);
}

#[test]
fn runtime_boundary_observer_receives_opt_in_observations() {
    let _observation_lock = lock_event_observation();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();

    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let _exported = df.to_record_batch().unwrap();

    let observations = observed.lock().unwrap();
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].kind, BoundaryKind::HostToDevice);
    assert_eq!(observations[0].placement, runtime.placement());
    assert_eq!(observations[0].rows, 4);
    assert_eq!(observations[0].columns, 2);
    assert!(!observations[0].synchronized_for_host);
    assert_eq!(observations[1].kind, BoundaryKind::DeviceToHost);
    assert_eq!(observations[1].placement, df.placement());
    assert_eq!(observations[1].rows, 4);
    assert_eq!(observations[1].columns, 2);
    assert!(observations[1].synchronized_for_host);
}

#[test]
fn query_style_chain_keeps_one_override_until_host_materialization() {
    let _observation_lock = lock_event_observation();
    let data_runtime = sample_runtime();
    let ctx_runtime = sample_runtime();
    let ctx_stream = ctx_runtime.stream().testing_raw_handle_value();
    let ctx_mr = ctx_runtime.memory_resource().testing_raw_handle_value();
    let ctx =
        ExecutionOverrides::new().with_stream(ctx_runtime.stream()).with_memory_resource(ctx_runtime.memory_resource());

    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let import_count_before = testing::arrow::arrow_host_import_call_count_for_test();
    let imported = DataFrame::from_record_batch(&data_runtime, &sample_record_batch()).unwrap();
    assert!(testing::arrow::arrow_host_import_call_count_for_test() >= import_count_before + 1);
    assert_frame_residency(
        &imported,
        data_runtime.testing_execution_domain_id(),
        data_runtime.stream().testing_raw_handle_value(),
        data_runtime.memory_resource().testing_raw_handle_value(),
    );

    reset_exec_observation();
    let filtered = imported.with_execution(&ctx).filter_expr(col("id").ge(2i32)).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&filtered, imported.testing_execution_domain_id(), ctx_stream, ctx_mr);

    reset_exec_observation();
    let projected = filtered.with_execution(&ctx).with_columns([("total", col("id") + col("value"))]).unwrap();
    assert_eq!(last_exec_stream_value(), ctx_stream);
    assert_eq!(last_exec_mr_value(), ctx_mr);
    assert_frame_residency(&projected, imported.testing_execution_domain_id(), ctx_stream, ctx_mr);
    assert_eq!(projected.series("total").unwrap().to_vec_strict::<i32>().unwrap(), vec![22, 33, 44]);

    reset_sync_observation();
    testing::arrow::reset_arrow_host_interop_observation_for_test();
    let export_count_before = testing::arrow::arrow_host_export_call_count_for_test();
    let exported = projected.to_record_batch().unwrap();
    let export_observation = testing::arrow::last_arrow_host_boundary_observation_for_test().unwrap();

    assert!(testing::arrow::arrow_host_export_call_count_for_test() >= export_count_before + 1);
    assert_eq!(export_observation.kind, BoundaryKind::DeviceToHost);
    assert_eq!(export_observation.placement, projected.placement());
    assert_eq!(export_observation.rows, 3);
    assert_eq!(export_observation.columns, 3);
    assert!(export_observation.synchronized_for_host);
    assert_eq!(last_sync_stream_value(), ctx_stream);
    assert_eq!(exported.column(0).len(), 3);
    assert_eq!(int32_array(&exported, 0).values().as_ref(), &[2, 3, 4]);
    assert_eq!(int32_array(&exported, 1).values().as_ref(), &[20, 30, 40]);
    assert_eq!(int32_array(&exported, 2).values().as_ref(), &[22, 33, 44]);
}

#[test]
fn explicit_runtime_relocation_is_an_observable_boundary() {
    let _observation_lock = lock_event_observation();
    let source_runtime = sample_runtime();
    let target_runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&source_runtime, &sample_record_batch()).unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let relocated = df.copy_into_runtime(&target_runtime).unwrap();
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::DeviceToDeviceCopy);
    assert_eq!(observation.placement, target_runtime.placement());
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(relocated.placement(), target_runtime.placement());
}

#[test]
fn arrow_c_device_export_and_import_are_observable_boundaries() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = sample_runtime();
    let consumer_runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&producer_runtime, &sample_record_batch()).unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let export = df.try_prepare_device_arrow_export().unwrap();
    let export_observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(export_observation.kind, BoundaryKind::DeviceAbiHandoff);
    assert_eq!(export_observation.placement, df.placement());
    assert_eq!(export_observation.rows, 4);
    assert_eq!(export_observation.columns, 2);
    assert!(!export_observation.synchronized_for_host);

    let foreign = export.into_foreign();
    let imported = DataFrame::try_import_from_device_arrow_table_into_runtime(&foreign, &consumer_runtime).unwrap();
    let import_observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(import_observation.kind, BoundaryKind::DeviceAbiHandoff);
    assert_eq!(import_observation.placement, consumer_runtime.placement());
    assert_eq!(import_observation.rows, 4);
    assert_eq!(import_observation.columns, 2);
    assert!(!import_observation.synchronized_for_host);
    assert_eq!(imported.placement(), consumer_runtime.placement());
}

#[test]
fn prepared_interop_lease_activation_is_an_observable_boundary() {
    let _observation_lock = lock_event_observation();
    let observed = Arc::new(Mutex::new(Vec::new()));
    let observer_records = Arc::clone(&observed);
    let runtime = Runtime::builder()
        .with_boundary_observer(move |observation| {
            observer_records.lock().unwrap().push(observation);
        })
        .build()
        .unwrap();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let export = df.try_prepare_device_table_export().unwrap();

    testing::boundary::reset_boundary_observation_for_test();
    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::InteropLease);
    assert_eq!(observation.placement, df.placement());
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(*observed.lock().unwrap().last().unwrap(), observation);
    lease.complete().unwrap();
}

#[test]
fn owned_transfer_export_is_an_observable_boundary() {
    let _observation_lock = lock_event_observation();
    let runtime = sample_runtime();
    let df = DataFrame::from_record_batch(&runtime, &sample_record_batch()).unwrap();
    let source_placement = df.placement();

    testing::boundary::reset_boundary_observation_for_test();
    let transfer = df.try_into_owned_transfer().unwrap();
    let observation = testing::boundary::last_boundary_observation_for_test().unwrap();

    assert_eq!(observation.kind, BoundaryKind::OwnedTransfer);
    assert_eq!(observation.placement, source_placement);
    assert_eq!(observation.rows, 4);
    assert_eq!(observation.columns, 2);
    assert!(!observation.synchronized_for_host);
    assert_eq!(transfer.table().num_rows(), 4);
    assert_eq!(transfer.table().columns().len(), 2);
}
