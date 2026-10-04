//! Contract tests for prepared export interop surfaces.
//!
//! This suite keeps representative coverage for lease activation/completion,
//! host vs device export profiles, metadata preservation, and recursive nested
//! projections without repeating the same matrix for every shared/runtime and
//! column/table permutation.
use std::{
    collections::BTreeSet,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudf::{
    DataFrame, DeviceId, ErrorCode, Runtime, TypeId,
    core::Table,
    interop::cuda_stream_handle,
    runtime::{CudaEvent, CudaStream},
    testing::{
        dictionary,
        interop::{
            column_export_with_lease_backend_for_test, trigger_interop_column_index_limit_for_test,
            trigger_interop_projection_active_range_overflow_for_test,
            trigger_interop_projection_dictionary_indices_width_mismatch_for_test,
            trigger_interop_projection_host_offsets_too_small_for_test,
            trigger_interop_projection_list_offsets_width_mismatch_for_test,
            trigger_interop_projection_missing_null_mask_for_test,
            trigger_interop_projection_missing_string_offsets_child_for_test,
            trigger_interop_projection_negative_offset_for_test,
            trigger_interop_projection_struct_missing_child_for_test,
        },
        prelude::*,
    },
};
use cudf_sys::{
    CUDF_SUCCESS,
    test_support::{
        cudf_test_event_raw_record_call_count_for_stream_raw, cudf_test_event_reset_observation,
        cudf_test_event_stream_wait_call_count_for_event_raw, cudf_test_event_stream_wait_call_count_for_stream_raw,
    },
};
use rapids_interop::{
    ActiveLeaseBackend, BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority, CompletionWitness,
    InteropError, InteropErrorKind, LeaseBackend, PhysicalValidity, Placement, PreparedAccess, PreparedColumnExport,
    PreparedExportProfile, PreparedTableExport, ProducerReadiness, ReadStartAuthority, SemanticColumn,
    TypeId as InteropTypeId,
};

use crate::common::{
    self, column_from_slice, dataframe_from_columns, interop_error_context_value, lock_event_observation,
    run_named_cases, series, with_test_runtime,
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
    let first = DeviceId::new(0).unwrap();
    let second = DeviceId::new(1).unwrap();
    (first, second)
}

fn reset_event_observation() {
    unsafe { cudf_test_event_reset_observation() };
}

fn observed_stream_wait_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, CUDF_SUCCESS, "failed to read per-stream wait observation count");
    out
}

fn observed_stream_wait_call_count_for_event_raw(event_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_test_event_stream_wait_call_count_for_event_raw(event_raw, &mut out) };
    assert_eq!(status, CUDF_SUCCESS, "failed to read per-event wait observation count");
    out
}

fn observed_raw_record_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, CUDF_SUCCESS, "failed to read per-stream raw event-record observation count");
    out
}

fn shared_same_device_completion(stream: &CudaStream) -> CompletionAuthority<'_> {
    CompletionAuthority::SameDeviceStream { stream: cuda_stream_handle(stream).unwrap() }
}

fn shared_stream_read_start(stream: &CudaStream) -> ReadStartAuthority<'_> {
    ReadStartAuthority::ConsumerStream { stream: cuda_stream_handle(stream).unwrap() }
}

fn binding<'a>(bindings: &'a [BufferBinding<'a>], id: BufferId) -> &'a BufferBinding<'a> {
    bindings.iter().find(|binding| binding.id() == id).unwrap()
}

fn semantic_column(columns: &[SemanticColumn], id: ColumnId) -> &SemanticColumn {
    columns.iter().find(|column| column.id() == id).unwrap()
}

fn binding_bytes(binding: &BufferBinding<'_>) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()) }.to_vec()
}

fn expected_i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn expected_i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn host_string_values(bindings: &[BufferBinding<'_>], offsets: BufferId, data: BufferId) -> Vec<String> {
    let offsets = binding_bytes(binding(bindings, offsets))
        .as_chunks::<{ std::mem::size_of::<i32>() }>()
        .0
        .iter()
        .map(|chunk| i32::from_ne_bytes(*chunk) as usize)
        .collect::<Vec<_>>();
    let data = binding_bytes(binding(bindings, data));
    offsets.windows(2).map(|bounds| String::from_utf8(data[bounds[0]..bounds[1]].to_vec()).unwrap()).collect()
}

fn expected_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn assert_device_binding_on(binding: &BufferBinding<'_>, device: DeviceId) {
    assert_eq!(binding.placement(), &Placement::CudaDevice { device: device.ordinal() as i32 });
}

fn assert_device_stream_readiness(access: &PreparedAccess<'_>, expected_stream_raw: usize, expected_device: DeviceId) {
    match access {
        PreparedAccess::Device { readiness: ProducerReadiness::DeviceStreamOrdered { stream }, .. } => {
            assert_eq!(stream.as_raw() as usize, expected_stream_raw);
            assert_eq!(stream.device(), expected_device.ordinal() as i32);
        },
        access => panic!("expected device-stream readiness, got {access:?}"),
    }
}

fn assert_host_column_basics(
    export: &PreparedColumnExport<'_>,
    expected_type: InteropTypeId,
    expected_len: usize,
    expected_offset: usize,
) {
    assert_eq!(export.column().data_type().id(), expected_type);
    assert_eq!(export.column().len(), expected_len);
    assert_eq!(export.column().offset(), expected_offset);
    assert_eq!(export.profile(), PreparedExportProfile::Host);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));
}

fn assert_host_table_basics(export: &PreparedTableExport<'_>, expected_names_slice: &[&str], expected_rows: usize) {
    assert_eq!(export.table().num_rows(), expected_rows);
    assert_eq!(export.table().names().unwrap(), expected_names(expected_names_slice).as_slice());
    assert_eq!(export.profile(), PreparedExportProfile::Host);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));
}

fn assert_host_binding_bytes(bindings: &[BufferBinding<'_>], id: BufferId, expected: Vec<u8>) {
    let binding = binding(bindings, id);
    assert_eq!(binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(binding), expected);
}

fn recursive_supported_frame() -> DataFrame {
    let nested = vec![vec![vec![1, 2], vec![3]], vec![vec![], vec![4, 5]], vec![vec![6]]];
    let runtime = Runtime::builder().build().unwrap();
    let frame = common::nested_dataframe(
        &runtime,
        vec![
            ("category", common::utf8_array(&["beta", "alpha", "beta"])),
            ("nested", common::nested_int64_list_array(&nested)),
            (
                "point",
                common::struct_array(vec![
                    (
                        "meta",
                        common::struct_array(vec![
                            ("x", common::int64_array(&[1, 2, 3])),
                            ("name", common::utf8_array(&["one", "two", "three"])),
                        ]),
                    ),
                    ("coords", common::nested_int64_list_array(&nested)),
                ]),
            ),
            ("id", common::int64_array(&[10, 20, 30])),
        ],
    );

    let category = frame.series("category").unwrap();
    let nested = frame.series("nested").unwrap();
    let point = frame.series("point").unwrap();
    let id = frame.series("id").unwrap();
    let encoded = dictionary::encode(&category.testing_view().unwrap(), TypeId::Int32).unwrap();
    let category = cudf::Series::from_column(encoded).unwrap();
    DataFrame::from_series([("category", category), ("nested", nested), ("point", point), ("id", id)]).unwrap()
}

mod lease_contracts {
    use super::*;

    fn shared_synchronous_completion_falls_back_by_return() {
        let _observation_lock = lock_event_observation();
        let runtime = Runtime::builder().build().unwrap();
        let series = runtime.series(&[1i32, 2, 3]).unwrap();

        let export = series.try_prepare_device_column_export().unwrap();
        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();

        assert!(lease.complete().unwrap().is_ready().unwrap());
    }

    fn shared_stream_lease_orders_consumer_stream_and_returns_witness() {
        let _observation_lock = lock_event_observation();
        let producer_stream = CudaStream::new().unwrap();
        let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
        let ids = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let vals = producer_runtime.series(&[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_series([("id", ids), ("value", vals)]).unwrap();
        let export = frame.try_prepare_device_table_export().unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let producer_stream_raw = producer_stream.testing_raw_handle_value();
        let consumer_stream_raw = consumer_stream.testing_raw_handle_value();

        reset_event_observation();
        let baseline_producer_record = observed_raw_record_call_count_for_stream_raw(producer_stream_raw);
        let baseline_consumer_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);
        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        assert!(observed_raw_record_call_count_for_stream_raw(producer_stream_raw) > baseline_producer_record);
        assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_consumer_wait);

        let mut completion = lease.complete().unwrap();
        completion.synchronize().unwrap();
        assert!(completion.is_ready().unwrap());
    }

    fn shared_event_ready_stream_lease_waits_producer_event_before_consumer_stream() {
        let _observation_lock = lock_event_observation();
        let producer_stream = CudaStream::new().unwrap();
        let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(&producer_stream).unwrap();
        let export = series.try_prepare_device_column_export_with_event(&producer_ready).unwrap();

        match export.access() {
            PreparedAccess::Device { readiness: ProducerReadiness::DeviceEventOrdered { event }, .. } => {
                assert_eq!(event.device(), producer_runtime.device().ordinal() as i32);
                assert!(!event.as_raw().is_null());
            },
            access => panic!("expected device-event readiness, got {access:?}"),
        }

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let producer_stream_raw = producer_stream.testing_raw_handle_value();
        let producer_event_raw = producer_ready.testing_interop_raw_handle_value();
        let consumer_stream_raw = consumer_stream.testing_raw_handle_value();

        reset_event_observation();
        let baseline_producer_event_wait = observed_stream_wait_call_count_for_event_raw(producer_event_raw);
        let baseline_producer_record = observed_raw_record_call_count_for_stream_raw(producer_stream_raw);
        let baseline_consumer_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);
        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        assert!(observed_stream_wait_call_count_for_event_raw(producer_event_raw) > baseline_producer_event_wait);
        assert_eq!(observed_raw_record_call_count_for_stream_raw(producer_stream_raw), baseline_producer_record);
        assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_consumer_wait);

        let mut completion = lease.complete().unwrap();
        completion.synchronize().unwrap();
    }

    fn into_prepared_column_export_with_event_retains_the_readiness_event() {
        let _observation_lock = lock_event_observation();
        let producer_stream = CudaStream::new().unwrap();
        let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
        let column = producer_runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(&producer_stream).unwrap();
        let producer_event_raw = producer_ready.testing_interop_raw_handle_value();

        let export = column.try_into_prepared_device_column_export_with_event(&producer_ready).unwrap();
        // The consuming `'static` export stores only the raw event pointer in
        // its readiness, so it must co-own the event: dropping the caller's
        // handle must leave the readiness event alive for lease activation.
        drop(producer_ready);

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();

        reset_event_observation();
        let baseline = observed_stream_wait_call_count_for_event_raw(producer_event_raw);
        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();
        assert!(observed_stream_wait_call_count_for_event_raw(producer_event_raw) > baseline);

        let mut completion = lease.complete().unwrap();
        completion.synchronize().unwrap();
    }

    fn into_prepared_table_export_with_event_retains_the_readiness_event() {
        let _observation_lock = lock_event_observation();
        let producer_stream = CudaStream::new().unwrap();
        let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
        let column = producer_runtime.column_from_slice(&[4i32, 5, 6]).unwrap();
        let table = Table::from_columns(vec![column]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(&producer_stream).unwrap();
        let producer_event_raw = producer_ready.testing_interop_raw_handle_value();

        let export = table.try_into_prepared_device_table_export_with_event(&producer_ready).unwrap();
        drop(producer_ready);

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();

        reset_event_observation();
        let baseline = observed_stream_wait_call_count_for_event_raw(producer_event_raw);
        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();
        assert!(observed_stream_wait_call_count_for_event_raw(producer_event_raw) > baseline);

        let mut completion = lease.complete().unwrap();
        completion.synchronize().unwrap();
    }

    fn shared_lease_drop_synchronizes_consumer_stream() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let export = series.try_prepare_device_column_export().unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let _consumer_work = consumer_runtime.series(&[10i32, 20, 30]).unwrap();
        let consumer_done = CudaEvent::new().unwrap();

        {
            let _lease = export
                .try_activate_read_lease(
                    shared_stream_read_start(&consumer_stream),
                    shared_same_device_completion(&consumer_stream),
                )
                .unwrap();
            consumer_done.record(&consumer_stream).unwrap();
        }

        assert!(consumer_done.is_ready().unwrap());
    }

    #[test]
    fn device_event_readiness_must_match_the_producer_device() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(producer_runtime.stream()).unwrap();

        let export = series.try_prepare_device_column_export_with_event(&producer_ready).unwrap();
        match export.access() {
            PreparedAccess::Device { readiness: ProducerReadiness::DeviceEventOrdered { event }, .. } => {
                assert_eq!(event.device(), producer_runtime.device().ordinal() as i32);
                assert!(!event.as_raw().is_null());
            },
            access => panic!("expected device-event readiness, got {access:?}"),
        }

        if !should_run_multi_gpu_test() {
            return;
        }
        let _device_guard = common::DeviceRestoreGuard::new();
        let (device0, device1) = first_two_test_devices();
        common::set_current_device(device0);
        let producer_runtime = Runtime::builder().with_device(device0).build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();

        common::set_current_device(device1);
        let wrong_device_event = CudaEvent::new().unwrap();
        let wrong_device_stream = CudaStream::new().unwrap();
        wrong_device_event.record(&wrong_device_stream).unwrap();

        common::set_current_device(device0);
        let err = series.try_prepare_device_column_export_with_event(&wrong_device_event).unwrap_err();
        assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
        assert_eq!(interop_error_context_value(&err, "left_device"), Some(device0.ordinal().to_string().as_str()));
        assert_eq!(interop_error_context_value(&err, "right_device"), Some(device1.ordinal().to_string().as_str()));
        assert_eq!(interop_error_context_value(&err, "producer_device"), None);
        assert_eq!(interop_error_context_value(&err, "event_device"), None);
    }

    fn runtime_stream_lease_records_one_completion_event_and_returns_witness() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let export = series.try_prepare_device_column_export().unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let consumer_stream_raw = consumer_stream.testing_raw_handle_value();

        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        reset_event_observation();
        let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);
        let mut completion = lease.complete().unwrap();
        assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record + 1);
        completion.synchronize().unwrap();
        assert!(completion.is_ready().unwrap());
        assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record + 1);
    }

    fn runtime_custom_backend_complete_path_is_used() {
        let _observation_lock = lock_event_observation();
        struct CountingBackend {
            complete_calls: Arc<AtomicUsize>,
        }

        impl ActiveLeaseBackend for CountingBackend {
            fn complete(
                &self,
                _readiness: ProducerReadiness<'_>,
                _completion: CompletionAuthority<'_>,
            ) -> std::result::Result<Option<CompletionWitness>, InteropError> {
                self.complete_calls.fetch_add(1, Ordering::SeqCst);
                Ok(None)
            }

            fn drop_incomplete(
                &self,
                _readiness: ProducerReadiness<'_>,
                _completion: CompletionAuthority<'_>,
            ) -> std::result::Result<(), InteropError> {
                Ok(())
            }
        }

        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let complete_calls = Arc::new(AtomicUsize::new(0));
        let export = column_export_with_lease_backend_for_test(
            &series.try_prepare_device_column_export().unwrap(),
            LeaseBackend::new(CountingBackend { complete_calls: Arc::clone(&complete_calls) }),
        )
        .unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();

        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        assert!(lease.complete().unwrap().is_ready().unwrap());
        assert_eq!(complete_calls.load(Ordering::SeqCst), 1);
    }

    fn shared_leases_can_overlap_and_reacquire_one_export_root() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
        let export = series.try_prepare_device_column_export().unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();

        let shared_lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        let device_stream_lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        assert!(shared_lease.complete().unwrap().is_ready().unwrap());

        let reacquired_device_stream_lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        let mut completion = device_stream_lease.complete().unwrap();
        completion.synchronize().unwrap();
        let mut completion = reacquired_device_stream_lease.complete().unwrap();
        completion.synchronize().unwrap();
    }

    #[test]
    fn prepared_export_tests_cases_lease_contracts() {
        run_named_cases(&[
            ("shared_synchronous_completion_falls_back_by_return", shared_synchronous_completion_falls_back_by_return),
            (
                "shared_stream_lease_orders_consumer_stream_and_returns_witness",
                shared_stream_lease_orders_consumer_stream_and_returns_witness,
            ),
            (
                "shared_event_ready_stream_lease_waits_producer_event_before_consumer_stream",
                shared_event_ready_stream_lease_waits_producer_event_before_consumer_stream,
            ),
            (
                "into_prepared_column_export_with_event_retains_the_readiness_event",
                into_prepared_column_export_with_event_retains_the_readiness_event,
            ),
            (
                "into_prepared_table_export_with_event_retains_the_readiness_event",
                into_prepared_table_export_with_event_retains_the_readiness_event,
            ),
            ("shared_lease_drop_synchronizes_consumer_stream", shared_lease_drop_synchronizes_consumer_stream),
            (
                "runtime_stream_lease_records_one_completion_event_and_returns_witness",
                runtime_stream_lease_records_one_completion_event_and_returns_witness,
            ),
            ("runtime_custom_backend_complete_path_is_used", runtime_custom_backend_complete_path_is_used),
            (
                "shared_leases_can_overlap_and_reacquire_one_export_root",
                shared_leases_can_overlap_and_reacquire_one_export_root,
            ),
        ]);
    }
}

mod device_export_contracts {
    use super::*;

    fn projection_active_range_overflow_hook_uses_row_count_context() {
        let err = trigger_interop_projection_active_range_overflow_for_test().unwrap_err();
        let expected = usize::MAX.to_string();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::active_range_overflow")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("row_range"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("end_overflowed_usize"));
        assert_eq!(interop_error_context_value(&err, "offset"), Some(expected.as_str()));
        assert_eq!(interop_error_context_value(&err, "row_count"), Some("1"));
        assert_eq!(interop_error_context_value(&err, "len"), None);
    }

    fn projection_negative_offset_hook_uses_offset_contract() {
        let err = trigger_interop_projection_negative_offset_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::negative_offset")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("offset"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("negative_value"));
        assert_eq!(
            interop_error_context_value(&err, "remedy"),
            Some("normalize the view so offset >= 0 before projection")
        );
        assert_eq!(interop_error_context_value(&err, "offset"), Some("-1"));
    }

    fn projection_missing_string_offsets_child_hook_uses_child_count_contract() {
        let err = trigger_interop_projection_missing_string_offsets_child_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::missing_string_offsets_child")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("child_count"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("requires_offsets_child"));
        assert_eq!(interop_error_context_value(&err, "type_id"), Some("string"));
        assert_eq!(interop_error_context_value(&err, "child_count"), Some("0"));
        assert_eq!(interop_error_context_value(&err, "required_child_count"), Some("1"));
        assert_eq!(interop_error_context_value(&err, "num_children"), None);
    }

    fn projection_list_offsets_width_mismatch_hook_uses_layout_contract() {
        let err = trigger_interop_projection_list_offsets_width_mismatch_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::list_offsets_width_mismatch")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("offsets_layout"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("requires_i32_width"));
        assert_eq!(interop_error_context_value(&err, "actual_width_bytes"), Some("8"));
        assert_eq!(interop_error_context_value(&err, "required_width_bytes"), Some("4"));
        assert_eq!(interop_error_context_value(&err, "offsets_elem_size"), None);
    }

    fn projection_dictionary_indices_width_mismatch_hook_uses_layout_contract() {
        let err = trigger_interop_projection_dictionary_indices_width_mismatch_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::dictionary_indices_width_mismatch")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("indices_layout"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("requires_i32_width"));
        assert_eq!(interop_error_context_value(&err, "actual_width_bytes"), Some("2"));
        assert_eq!(interop_error_context_value(&err, "required_width_bytes"), Some("4"));
        assert_eq!(interop_error_context_value(&err, "indices_elem_size"), None);
    }

    fn projection_host_offsets_too_small_hook_uses_offsets_range_contract() {
        let err = trigger_interop_projection_host_offsets_too_small_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::host_offsets_too_small")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("offsets_range"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("too_small_for_active_range"));
        assert_eq!(interop_error_context_value(&err, "offset"), Some("3"));
        assert_eq!(interop_error_context_value(&err, "row_count"), Some("2"));
        assert_eq!(interop_error_context_value(&err, "offset_count"), Some("5"));
        assert_eq!(interop_error_context_value(&err, "offsets_len"), None);
    }

    fn projection_missing_null_mask_hook_uses_null_mask_contract() {
        let err = trigger_interop_projection_missing_null_mask_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::missing_null_mask")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("null_mask_buffer"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("missing_for_nullable_column"));
        assert_eq!(interop_error_context_value(&err, "row_count"), Some("3"));
    }

    fn projection_struct_missing_child_hook_uses_child_count_contract() {
        let err = trigger_interop_projection_struct_missing_child_for_test().unwrap_err();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(
            interop_error_context_value(&err, "operation"),
            Some("testing::interop::projection::missing_struct_child")
        );
        assert_eq!(interop_error_context_value(&err, "resource"), Some("child_count"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("requires_child_column"));
        assert_eq!(
            interop_error_context_value(&err, "remedy"),
            Some("expose at least one child column before projecting struct data")
        );
        assert_eq!(interop_error_context_value(&err, "type_id"), Some("struct"));
        assert_eq!(interop_error_context_value(&err, "child_count"), Some("0"));
        assert_eq!(interop_error_context_value(&err, "required_child_count"), Some("1"));
    }

    fn interop_column_index_limit_hook_uses_shared_i32_limit_contract() {
        let err = trigger_interop_column_index_limit_for_test().unwrap_err();
        let expected_column_index = usize::MAX.to_string();

        assert_eq!(err.kind(), InteropErrorKind::InvalidView);
        assert_eq!(interop_error_context_value(&err, "operation"), Some("testing::interop::column_index_limit"));
        assert_eq!(interop_error_context_value(&err, "resource"), Some("column_index"));
        assert_eq!(interop_error_context_value(&err, "reason"), Some("exceeds_i32_max"));
        assert_eq!(interop_error_context_value(&err, "remedy"), Some("reduce column_index to fit within i32::MAX"));
        assert_eq!(interop_error_context_value(&err, "column_index"), Some(expected_column_index.as_str()));
        assert_eq!(interop_error_context_value(&err, "index"), None);
    }

    #[test]
    fn helper_built_exports_use_explicit_runtime_binding_and_cross_device_inputs_are_rejected() {
        with_test_runtime(|runtime| {
            let column = column_from_slice(&[1i32, 2, 3]).unwrap();
            let view = column.testing_view().unwrap();
            let column_export = view.try_prepare_device_column_export().unwrap();
            assert_eq!(column_export.profile(), PreparedExportProfile::Device);
            assert_device_stream_readiness(
                column_export.access(),
                runtime.stream().testing_raw_handle_value(),
                runtime.device(),
            );

            let ids = series("id", &[1i32, 2, 3]).unwrap();
            let vals = series("value", &[10i64, 20, 30]).unwrap();
            let frame = DataFrame::from_series([("id", ids), ("value", vals)]).unwrap();
            let frame_export = frame.try_prepare_device_table_export().unwrap();
            assert_eq!(frame_export.profile(), PreparedExportProfile::Device);
            assert_eq!(frame_export.table().names().unwrap(), expected_names(&["id", "value"]).as_slice());
            assert_device_stream_readiness(
                frame_export.access(),
                runtime.stream().testing_raw_handle_value(),
                runtime.device(),
            );
        });

        if !should_run_multi_gpu_test() {
            return;
        }

        let (device0, device1) = first_two_test_devices();
        let rt0 = Runtime::builder().with_device(device0).build().unwrap();
        let rt1 = Runtime::builder().with_device(device1).build().unwrap();

        let left = rt0.column_from_slice(&[1i32, 2, 3]).unwrap();
        let right = rt1.column_from_slice(&[4i32, 5, 6]).unwrap();
        let frame_err = dataframe_from_columns(vec!["left".into(), "right".into()], vec![left, right]).unwrap_err();
        assert_eq!(frame_err.code(), ErrorCode::CrossProvenanceInputs);

        let _left_domain_seed =
            rt0.column_from_slice(&[1i32, 2, 3]).unwrap().testing_rebind_execution_domain_explicit(9001).unwrap();
        let rebind_err =
            rt1.column_from_slice(&[4i32, 5, 6]).unwrap().testing_rebind_execution_domain_explicit(9001).unwrap_err();
        assert_eq!(rebind_err.code(), ErrorCode::CrossDeviceInputs);
    }

    fn device_fixed_width_exports_preserve_metadata_validity_and_slice_offset() {
        let runtime = Runtime::builder().build().unwrap();
        let nullable = runtime.column_from_slice_nullable(&[10i32, 20, 30], &[true, false, true]).unwrap();
        let nullable_view = nullable.testing_view().unwrap();
        let expected_stream = nullable.testing_execution_domain_stream_value();
        let nullable_export = nullable_view.try_prepare_device_column_export().unwrap();

        assert_eq!(nullable_export.column().data_type().id(), InteropTypeId::Int32);
        assert_eq!(nullable_export.column().len(), 3);
        assert_eq!(nullable_export.column().offset(), 0);
        assert_eq!(nullable_export.profile(), PreparedExportProfile::Device);

        let values = match nullable_export.column().layout() {
            &ColumnLayout::FixedWidth { values } => values,
            layout => panic!("expected fixed-width layout, got {layout:?}"),
        };
        let values_binding = binding(nullable_export.bindings(), values);
        assert_eq!(values_binding.bytes().base_ptr(), nullable_view.device_data_ptr().unwrap());
        assert_device_binding_on(values_binding, runtime.device());

        let mask = match nullable_export.column().validity() {
            PhysicalValidity::ValidityMask { mask, null_count } => {
                assert_eq!(*null_count, Some(1));
                *mask
            },
            PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
        };
        assert_device_binding_on(binding(nullable_export.bindings(), mask), runtime.device());
        assert_device_stream_readiness(nullable_export.access(), expected_stream, runtime.device());

        let base = runtime.column_from_slice(&[11i32, 22, 33, 44]).unwrap();
        let base_view = base.testing_view().unwrap();
        let sliced = base.testing_slice_view(1, 2).unwrap();
        let sliced_export = sliced.try_prepare_device_column_export().unwrap();
        assert_eq!(sliced_export.column().len(), 2);
        assert_eq!(sliced_export.column().offset(), 1);

        let values = match sliced_export.column().layout() {
            &ColumnLayout::FixedWidth { values } => values,
            layout => panic!("expected fixed-width layout, got {layout:?}"),
        };
        assert_eq!(binding(sliced_export.bindings(), values).bytes().base_ptr(), base_view.device_data_ptr().unwrap());
    }

    fn owned_low_level_column_consuming_prepared_export_retains_owner() {
        let runtime = Runtime::builder().build().unwrap();
        let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();

        let export = column.try_into_prepared_device_column_export().unwrap();

        assert!(export.retained_owner().is_some());
        assert_eq!(export.profile(), PreparedExportProfile::Device);
        assert_device_stream_readiness(export.access(), runtime.stream().testing_raw_handle_value(), runtime.device());
    }

    fn borrowed_low_level_column_prepared_export_does_not_fabricate_owner_retention() {
        let runtime = Runtime::builder().build().unwrap();
        let column = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();

        let export = column.try_prepare_device_column_export().unwrap();

        assert!(export.retained_owner().is_none());
    }

    fn device_table_export_preserves_names_nullability_readiness_and_low_level_equivalence() {
        let runtime = Runtime::builder().build().unwrap();
        let ids = runtime.series(&[1i32, 2, 3]).unwrap();
        let vals = runtime.nullable_series(&[10i64, 0, 30], &[true, false, true]).unwrap();
        let frame = DataFrame::from_series([("id", ids), ("value", vals)]).unwrap();
        let expected_stream = frame.testing_execution_domain_stream_value();

        let frame_export = frame.try_prepare_device_table_export().unwrap();
        let table_view = frame.testing_view().unwrap();
        let table_export = table_view.try_prepare_device_table_export().unwrap();

        assert_eq!(frame_export.table().num_rows(), 3);
        assert_eq!(frame_export.table().names().unwrap(), expected_names(&["id", "value"]).as_slice());
        assert_eq!(frame_export.columns()[0].validity(), &PhysicalValidity::NoValidityBuffer);
        assert!(matches!(frame_export.columns()[1].validity(), PhysicalValidity::ValidityMask { .. }));
        assert_eq!(frame_export.columns(), table_export.columns());
        assert_eq!(frame_export.bindings(), table_export.bindings());
        assert_device_stream_readiness(frame_export.access(), expected_stream, runtime.device());
        assert_device_stream_readiness(table_export.access(), expected_stream, runtime.device());
    }

    fn owned_low_level_table_consuming_prepared_export_retains_owner() {
        let runtime = Runtime::builder().build().unwrap();
        let left = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let right = runtime.column_from_slice(&[10i64, 20, 30]).unwrap();
        let table = Table::from_columns(vec![left, right]).unwrap();

        let export = table.try_into_prepared_device_table_export().unwrap();

        assert!(export.retained_owner().is_some());
        assert_eq!(export.profile(), PreparedExportProfile::Device);
        assert_device_stream_readiness(export.access(), runtime.stream().testing_raw_handle_value(), runtime.device());
    }

    fn borrowed_low_level_table_prepared_export_does_not_fabricate_owner_retention() {
        let runtime = Runtime::builder().build().unwrap();
        let left = runtime.column_from_slice(&[1i32, 2, 3]).unwrap();
        let right = runtime.column_from_slice(&[10i64, 20, 30]).unwrap();
        let table = Table::from_columns(vec![left, right]).unwrap();

        let export = table.try_prepare_device_table_export().unwrap();

        assert!(export.retained_owner().is_none());
    }

    fn device_string_exports_preserve_binary_like_metadata_and_active_slice_span() {
        let runtime = Runtime::builder().build().unwrap();
        let strings = runtime.column_from_strs(&["alice", "bob"]).unwrap();
        let strings_view = strings.testing_view().unwrap();
        let offsets_view = strings_view.child(0).unwrap();
        let expected_stream = strings.testing_execution_domain_stream_value();
        let export = strings_view.try_prepare_device_column_export().unwrap();

        assert_eq!(export.column().data_type().id(), InteropTypeId::String);
        assert_eq!(export.column().len(), 2);
        assert_eq!(export.column().offset(), 0);

        let (offsets, data) = match export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        let offsets_binding = binding(export.bindings(), *offsets);
        assert_eq!(offsets_binding.bytes().base_ptr(), offsets_view.device_data_ptr().unwrap());
        assert_device_binding_on(offsets_binding, runtime.device());
        let data_binding = binding(export.bindings(), *data);
        assert_device_binding_on(data_binding, runtime.device());
        assert!(data_binding.bytes().byte_len() >= "alicebob".len());
        assert_device_stream_readiness(export.access(), expected_stream, runtime.device());

        let base = runtime.column_from_strs(&["zero", "one", "two", "three"]).unwrap();
        let base_view = base.testing_view().unwrap();
        let base_export = base_view.try_prepare_device_column_export().unwrap();
        let sliced = base.testing_slice_view(1, 2).unwrap();
        let sliced_offsets_view = sliced.child(0).unwrap();
        let sliced_export = sliced.try_prepare_device_column_export().unwrap();

        assert_eq!(sliced_export.column().len(), 2);
        assert_eq!(sliced_export.column().offset(), 1);

        let (offsets, data) = match sliced_export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        let (_, base_data) = match base_export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        let base_data_binding = binding(base_export.bindings(), *base_data);

        let offsets_binding = binding(sliced_export.bindings(), *offsets);
        assert_eq!(offsets_binding.bytes().base_ptr(), sliced_offsets_view.device_data_ptr().unwrap());
        assert_eq!(offsets_binding.bytes().byte_len(), 4 * std::mem::size_of::<i32>());
        let data_binding = binding(sliced_export.bindings(), *data);
        assert_eq!(data_binding.bytes().base_ptr(), base_data_binding.bytes().base_ptr());
        // The data binding starts at the chars BASE and the exported offsets
        // are not rebased (rows index chars[4..10] for "one"/"two"), so the
        // declared span must reach the last referenced byte.
        assert_eq!(data_binding.bytes().byte_len(), "zeroonetwo".len());
    }

    fn device_recursive_nested_export_preserves_supported_projection_shape() {
        let frame = recursive_supported_frame();
        let export = frame.try_prepare_device_table_export().unwrap();

        assert_eq!(export.table().num_rows(), 3);
        assert_eq!(export.table().names().unwrap(), expected_names(&["category", "nested", "point", "id"]).as_slice(),);
        assert!(matches!(export.columns()[0].layout(), ColumnLayout::Dictionary { .. }));
        assert!(matches!(export.columns()[1].layout(), ColumnLayout::List { .. }));
        assert!(matches!(export.columns()[2].layout(), ColumnLayout::Struct { .. }));
        assert!(matches!(export.columns()[3].layout(), ColumnLayout::FixedWidth { .. }));

        let dictionary_keys = match export.columns()[0].layout() {
            ColumnLayout::Dictionary { keys, .. } => keys,
            layout => panic!("expected dictionary layout, got {layout:?}"),
        };
        let dictionary_keys_column = semantic_column(export.semantic_columns(), *dictionary_keys);
        assert_eq!(dictionary_keys_column.data_type().id(), InteropTypeId::String);
        assert!(matches!(dictionary_keys_column.layout(), ColumnLayout::BinaryLike { .. }));

        let nested_child = match export.columns()[1].layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected list layout, got {layout:?}"),
        };
        let nested_list = semantic_column(export.semantic_columns(), *nested_child);
        let nested_leaf = match nested_list.layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
        assert_eq!(semantic_column(export.semantic_columns(), *nested_leaf).data_type().id(), InteropTypeId::Int64);

        let struct_children = match export.columns()[2].layout() {
            ColumnLayout::Struct { children } => children.clone(),
            layout => panic!("expected struct layout, got {layout:?}"),
        };
        let struct_meta = struct_children
            .iter()
            .find_map(|id| {
                export
                    .semantic_columns()
                    .iter()
                    .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::Struct)
            })
            .unwrap();
        let meta_children = match struct_meta.layout() {
            ColumnLayout::Struct { children } => children.clone(),
            layout => panic!("expected nested struct layout, got {layout:?}"),
        };
        assert!(
            meta_children
                .iter()
                .any(|id| semantic_column(export.semantic_columns(), *id).data_type().id() == InteropTypeId::Int64)
        );
        assert!(
            meta_children
                .iter()
                .any(|id| semantic_column(export.semantic_columns(), *id).data_type().id() == InteropTypeId::String)
        );

        let struct_list = struct_children
            .iter()
            .find_map(|id| {
                export
                    .semantic_columns()
                    .iter()
                    .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::List)
            })
            .unwrap();
        let struct_nested = match struct_list.layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        let struct_leaf = match semantic_column(export.semantic_columns(), *struct_nested).layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        assert_eq!(semantic_column(export.semantic_columns(), *struct_leaf).data_type().id(), InteropTypeId::Int64);

        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert!(lease.complete().unwrap().is_ready().unwrap());
    }

    #[test]
    fn prepared_export_tests_cases_device_export_contracts() {
        run_named_cases(&[
            (
                "projection_active_range_overflow_hook_uses_row_count_context",
                projection_active_range_overflow_hook_uses_row_count_context,
            ),
            (
                "projection_negative_offset_hook_uses_offset_contract",
                projection_negative_offset_hook_uses_offset_contract,
            ),
            (
                "projection_missing_string_offsets_child_hook_uses_child_count_contract",
                projection_missing_string_offsets_child_hook_uses_child_count_contract,
            ),
            (
                "projection_list_offsets_width_mismatch_hook_uses_layout_contract",
                projection_list_offsets_width_mismatch_hook_uses_layout_contract,
            ),
            (
                "projection_dictionary_indices_width_mismatch_hook_uses_layout_contract",
                projection_dictionary_indices_width_mismatch_hook_uses_layout_contract,
            ),
            (
                "projection_host_offsets_too_small_hook_uses_offsets_range_contract",
                projection_host_offsets_too_small_hook_uses_offsets_range_contract,
            ),
            (
                "projection_missing_null_mask_hook_uses_null_mask_contract",
                projection_missing_null_mask_hook_uses_null_mask_contract,
            ),
            (
                "projection_struct_missing_child_hook_uses_child_count_contract",
                projection_struct_missing_child_hook_uses_child_count_contract,
            ),
            (
                "interop_column_index_limit_hook_uses_shared_i32_limit_contract",
                interop_column_index_limit_hook_uses_shared_i32_limit_contract,
            ),
            (
                "device_fixed_width_exports_preserve_metadata_validity_and_slice_offset",
                device_fixed_width_exports_preserve_metadata_validity_and_slice_offset,
            ),
            (
                "owned_low_level_column_consuming_prepared_export_retains_owner",
                owned_low_level_column_consuming_prepared_export_retains_owner,
            ),
            (
                "borrowed_low_level_column_prepared_export_does_not_fabricate_owner_retention",
                borrowed_low_level_column_prepared_export_does_not_fabricate_owner_retention,
            ),
            (
                "device_table_export_preserves_names_nullability_readiness_and_low_level_equivalence",
                device_table_export_preserves_names_nullability_readiness_and_low_level_equivalence,
            ),
            (
                "owned_low_level_table_consuming_prepared_export_retains_owner",
                owned_low_level_table_consuming_prepared_export_retains_owner,
            ),
            (
                "borrowed_low_level_table_prepared_export_does_not_fabricate_owner_retention",
                borrowed_low_level_table_prepared_export_does_not_fabricate_owner_retention,
            ),
            (
                "device_string_exports_preserve_binary_like_metadata_and_active_slice_span",
                device_string_exports_preserve_binary_like_metadata_and_active_slice_span,
            ),
            (
                "device_recursive_nested_export_preserves_supported_projection_shape",
                device_recursive_nested_export_preserves_supported_projection_shape,
            ),
        ]);
    }
}

mod host_export_contracts {
    use super::*;

    fn host_fixed_width_exports_materialize_active_range_and_pack_validity() {
        let runtime = Runtime::builder().build().unwrap();

        let base = runtime.column_from_slice(&[11i32, 22, 33, 44]).unwrap();
        let sliced = base.testing_slice_view(1, 2).unwrap();
        let sliced_export = sliced.try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&sliced_export, InteropTypeId::Int32, 2, 0);
        let values = match sliced_export.column().layout() {
            ColumnLayout::FixedWidth { values } => values,
            layout => panic!("expected fixed-width layout, got {layout:?}"),
        };
        assert_host_binding_bytes(sliced_export.bindings(), *values, expected_i32_bytes(&[22, 33]));

        let nullable = runtime.column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, false]).unwrap();
        let nullable_export = nullable.testing_view().unwrap().try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&nullable_export, InteropTypeId::Int32, 4, 0);
        let mask = match nullable_export.column().validity() {
            PhysicalValidity::ValidityMask { mask, null_count } => {
                assert_eq!(*null_count, Some(2));
                *mask
            },
            PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
        };
        let expected_mask = nullable.testing_view().unwrap().to_host_null_mask_bytes().unwrap().unwrap();
        assert_host_binding_bytes(nullable_export.bindings(), mask, expected_mask);
    }

    fn host_table_export_is_host_visible_and_rejects_cuda_completion() {
        let runtime = Runtime::builder().build().unwrap();
        let ids = runtime.series(&[1i32, 2, 3]).unwrap();
        let vals = runtime.nullable_series(&[10i64, 0, 30], &[true, false, true]).unwrap();
        let frame = DataFrame::from_series([("id", ids), ("value", vals)]).unwrap();

        let export = frame.try_prepare_host_table_export().unwrap();
        assert_host_table_basics(&export, &["id", "value"], 3);
        assert_eq!(export.columns()[0].validity(), &PhysicalValidity::NoValidityBuffer);
        assert!(matches!(export.columns()[1].validity(), PhysicalValidity::ValidityMask { .. }));

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime =
            Runtime::builder().with_device(runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
        let err = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap_err();
        assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
    }

    fn host_string_exports_materialize_binary_like_bytes_rebase_slice_and_pack_validity() {
        let runtime = Runtime::builder().build().unwrap();

        let strings = runtime.column_from_strs(&["alice", "bob"]).unwrap();
        let export = strings.testing_view().unwrap().try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&export, InteropTypeId::String, 2, 0);
        let (offsets, data) = match export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), *offsets, expected_i32_bytes(&[0, 5, 8]));
        assert_host_binding_bytes(export.bindings(), *data, b"alicebob".to_vec());

        let base = runtime.column_from_strs(&["zero", "one", "two", "three"]).unwrap();
        let sliced = base.testing_slice_view(1, 2).unwrap();
        let sliced_export = sliced.try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&sliced_export, InteropTypeId::String, 2, 0);
        let (offsets, data) = match sliced_export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        assert_host_binding_bytes(sliced_export.bindings(), *offsets, expected_i32_bytes(&[0, 3, 6]));
        assert_host_binding_bytes(sliced_export.bindings(), *data, b"onetwo".to_vec());

        let nullable =
            runtime.nullable_series(&["alice", "placeholder", "bob", "unused"], &[true, false, true, false]).unwrap();
        let nullable_export = nullable.try_prepare_host_column_export().unwrap();
        let mask = match nullable_export.column().validity() {
            PhysicalValidity::ValidityMask { mask, null_count } => {
                assert_eq!(*null_count, Some(2));
                *mask
            },
            PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
        };
        let expected_mask = nullable.as_column().testing_view().unwrap().to_host_null_mask_bytes().unwrap().unwrap();
        assert_host_binding_bytes(nullable_export.bindings(), mask, expected_mask);
    }

    fn host_list_slice_export_rebases_offsets_and_child_range() {
        let runtime = Runtime::builder().build().unwrap();
        let frame = common::nested_dataframe(
            &runtime,
            vec![("lists", common::int64_list_array(&[vec![1, 2], vec![3], vec![], vec![4, 5, 6]]))],
        );
        let series = frame.series("lists").unwrap();
        let sliced = series.as_column().testing_slice_view(1, 2).unwrap();

        let export = sliced.try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&export, InteropTypeId::List, 2, 0);

        let (offsets, child) = match export.column().layout() {
            ColumnLayout::List { offsets, child } => (offsets, child),
            layout => panic!("expected list layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), *offsets, expected_i32_bytes(&[0, 1, 1]));

        let child_column = semantic_column(export.semantic_columns(), *child);
        assert_eq!(child_column.data_type().id(), InteropTypeId::Int64);
        assert_eq!(child_column.len(), 1);
        assert_eq!(child_column.offset(), 0);
        let values = match child_column.layout() {
            ColumnLayout::FixedWidth { values } => values,
            layout => panic!("expected fixed-width child layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), *values, expected_i64_bytes(&[3]));

        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert!(lease.complete().unwrap().is_ready().unwrap());
    }

    fn host_recursive_nested_export_materializes_supported_projections() {
        let frame = recursive_supported_frame();
        let export = frame.try_prepare_host_table_export().unwrap();
        assert_host_table_basics(&export, &["category", "nested", "point", "id"], 3);

        let dictionary_keys = match export.columns()[0].layout() {
            ColumnLayout::Dictionary { keys, .. } => keys,
            layout => panic!("expected dictionary layout, got {layout:?}"),
        };
        let dictionary_keys_column = semantic_column(export.semantic_columns(), *dictionary_keys);
        assert_eq!(dictionary_keys_column.data_type().id(), InteropTypeId::String);
        let (key_offsets, key_data) = match dictionary_keys_column.layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like dictionary keys, got {layout:?}"),
        };
        assert_eq!(
            host_string_values(export.bindings(), *key_offsets, *key_data).into_iter().collect::<BTreeSet<_>>(),
            BTreeSet::from(["alpha".to_string(), "beta".to_string()])
        );

        let nested_child = match export.columns()[1].layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected list layout, got {layout:?}"),
        };
        let nested_list = semantic_column(export.semantic_columns(), *nested_child);
        let (nested_offsets, nested_leaf) = match nested_list.layout() {
            ColumnLayout::List { offsets, child } => (offsets, child),
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        let root_offsets = match export.columns()[1].layout() {
            ColumnLayout::List { offsets, .. } => offsets,
            layout => panic!("expected list layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), *root_offsets, expected_i32_bytes(&[0, 2, 4, 5]));
        assert_host_binding_bytes(export.bindings(), *nested_offsets, expected_i32_bytes(&[0, 2, 3, 3, 5, 6]));
        let nested_leaf_column = semantic_column(export.semantic_columns(), *nested_leaf);
        assert_eq!(nested_leaf_column.data_type().id(), InteropTypeId::Int64);
        let leaf_values = match nested_leaf_column.layout() {
            ColumnLayout::FixedWidth { values } => values,
            layout => panic!("expected fixed-width nested leaf, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), *leaf_values, expected_i64_bytes(&[1, 2, 3, 4, 5, 6]));

        let struct_children = match export.columns()[2].layout() {
            ColumnLayout::Struct { children } => children.clone(),
            layout => panic!("expected struct layout, got {layout:?}"),
        };
        let struct_meta = struct_children
            .iter()
            .find_map(|id| {
                export
                    .semantic_columns()
                    .iter()
                    .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::Struct)
            })
            .unwrap();
        let meta_children = match struct_meta.layout() {
            ColumnLayout::Struct { children } => children.clone(),
            layout => panic!("expected nested struct layout, got {layout:?}"),
        };
        assert!(
            meta_children
                .iter()
                .any(|id| semantic_column(export.semantic_columns(), *id).data_type().id() == InteropTypeId::Int64)
        );
        assert!(
            meta_children
                .iter()
                .any(|id| semantic_column(export.semantic_columns(), *id).data_type().id() == InteropTypeId::String)
        );

        let struct_list = struct_children
            .iter()
            .find_map(|id| {
                export
                    .semantic_columns()
                    .iter()
                    .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::List)
            })
            .unwrap();
        let struct_nested = match struct_list.layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        let struct_leaf = match semantic_column(export.semantic_columns(), *struct_nested).layout() {
            ColumnLayout::List { child, .. } => child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        assert_eq!(semantic_column(export.semantic_columns(), *struct_leaf).data_type().id(), InteropTypeId::Int64);

        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert!(lease.complete().unwrap().is_ready().unwrap());
    }

    #[test]
    fn prepared_export_tests_cases_host_export_contracts() {
        run_named_cases(&[
            (
                "host_fixed_width_exports_materialize_active_range_and_pack_validity",
                host_fixed_width_exports_materialize_active_range_and_pack_validity,
            ),
            (
                "host_table_export_is_host_visible_and_rejects_cuda_completion",
                host_table_export_is_host_visible_and_rejects_cuda_completion,
            ),
            (
                "host_string_exports_materialize_binary_like_bytes_rebase_slice_and_pack_validity",
                host_string_exports_materialize_binary_like_bytes_rebase_slice_and_pack_validity,
            ),
            (
                "host_list_slice_export_rebases_offsets_and_child_range",
                host_list_slice_export_rebases_offsets_and_child_range,
            ),
            (
                "host_recursive_nested_export_materializes_supported_projections",
                host_recursive_nested_export_materializes_supported_projections,
            ),
        ]);
    }
}
