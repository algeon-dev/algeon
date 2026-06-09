//! Contract tests for prepared export interop surfaces.
//!
//! This suite keeps representative coverage for lease activation/completion,
//! host vs device export profiles, metadata preservation, and recursive nested
//! projections without repeating the same matrix for every shared/runtime and
//! column/table permutation.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use cudf::{
    DataFrame, DeviceId, JsonReadOptions, Runtime, TypeId,
    core::Column,
    interop::{
        ActiveLeaseBackend, BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority, CompletionBinding,
        CompletionWitness, CompletionWitnessKind, CudaEventHandle, CudaStreamHandle, ExportProfile,
        IncompleteDropBehavior, InteropError, InteropErrorKind, InteropTypeId, LeaseBackend, LeaseCompletionResult,
        LeaseExportContext, PhysicalValidity, Placement, PreparedColumnExport, PreparedTableExport, ProducerReadiness,
        ReadStartAuthority, RuntimeCompletionAuthority, RuntimeCompletionWitness, RuntimeLeaseCompletionResult,
        RuntimeReadStartAuthority, SchemaNullability, SemanticColumn, TryActivateRuntimeColumnReadLease,
        TryActivateRuntimeTableReadLease, cuda_explicit_completion_required_lease_backend,
    },
    ops::dictionary,
    runtime::{CudaEvent, CudaStream},
};
use tempfile::TempDir;

use crate::common::{self, lock_event_observation};

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
    unsafe { cudf_sys::cudf_test_event_reset_observation() };
}

fn observed_stream_wait_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream wait observation count");
    out
}

fn observed_stream_wait_call_count_for_event_raw(event_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_stream_wait_call_count_for_event_raw(event_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-event wait observation count");
    out
}

fn observed_raw_record_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream raw event-record observation count");
    out
}

fn shared_same_device_completion(stream: &CudaStream) -> CompletionAuthority<'static> {
    CompletionAuthority::SameDeviceStream {
        stream: CudaStreamHandle::from_raw_on_device(
            stream.__debug_raw_handle_value() as *mut std::ffi::c_void,
            stream.device().ordinal() as i32,
        )
        .unwrap(),
    }
}

fn shared_stream_read_start(stream: &CudaStream) -> ReadStartAuthority<'static> {
    ReadStartAuthority::ConsumerStream {
        stream: CudaStreamHandle::from_raw_on_device(
            stream.__debug_raw_handle_value() as *mut std::ffi::c_void,
            stream.device().ordinal() as i32,
        )
        .unwrap(),
    }
}

fn shared_cross_device_completion(event: &CudaEvent) -> CompletionAuthority<'static> {
    CompletionAuthority::CrossDeviceEvent {
        event: CudaEventHandle::from_raw_on_device(
            event.__debug_interop_raw_handle_value() as *mut std::ffi::c_void,
            event.device().ordinal() as i32,
        )
        .unwrap(),
    }
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

fn expected_names(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn assert_device_binding_on(binding: &BufferBinding<'_>, device: DeviceId) {
    assert_eq!(binding.placement(), &Placement::CudaDevice { device: device.ordinal() as i32 });
}

fn assert_device_stream_readiness(
    readiness: &ProducerReadiness<'_>,
    expected_stream_raw: usize,
    expected_device: DeviceId,
) {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream_raw);
            assert_eq!(stream.device(), expected_device.ordinal() as i32);
        },
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
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
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));
}

fn assert_host_table_basics(export: &PreparedTableExport<'_>, expected_names_slice: &[&str], expected_rows: usize) {
    assert_eq!(export.table().num_rows(), expected_rows);
    assert_eq!(export.table().names().unwrap(), expected_names(expected_names_slice).as_slice());
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));
}

fn assert_host_binding_bytes(bindings: &[BufferBinding<'_>], id: BufferId, expected: Vec<u8>) {
    let binding = binding(bindings, id);
    assert_eq!(binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(binding), expected);
}

fn expect_shared_witness(result: LeaseCompletionResult, expected_kind: CompletionWitnessKind) -> CompletionWitness {
    match result {
        LeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), expected_kind);
            witness
        },
        LeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness")
        },
    }
}

fn expect_runtime_witness(
    result: RuntimeLeaseCompletionResult,
    expected_kind: CompletionWitnessKind,
) -> RuntimeCompletionWitness {
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), expected_kind);
            witness
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned runtime completion witness")
        },
    }
}

fn read_json_dataframe(filename: &str, rows: &str) -> DataFrame {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join(filename);
    std::fs::write(&path, rows).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    DataFrame::read_json_with_options(JsonReadOptions::builder().filepath(&path).with_runtime(&runtime).build())
        .unwrap()
}

fn read_list_dataframe(rows: &str) -> DataFrame {
    read_json_dataframe("lists.jsonl", rows)
}

fn recursive_supported_frame() -> DataFrame {
    let frame = read_json_dataframe(
        "recursive_supported.jsonl",
        "{\"category\":\"beta\",\"nested\":[[1,2],[3]],\"point\":{\"meta\":{\"x\":1,\"name\":\"one\"},\"coords\":[[1,2],[3]]},\"id\":10}\n\
         {\"category\":\"alpha\",\"nested\":[[],[4,5]],\"point\":{\"meta\":{\"x\":2,\"name\":\"two\"},\"coords\":[[],[4,5]]},\"id\":20}\n\
         {\"category\":\"beta\",\"nested\":[[6]],\"point\":{\"meta\":{\"x\":3,\"name\":\"three\"},\"coords\":[[6]]},\"id\":30}\n",
    );

    let category = frame.series("category").unwrap();
    let nested = frame.series("nested").unwrap();
    let point = frame.series("point").unwrap();
    let id = frame.series("id").unwrap();
    let encoded = dictionary::encode(&category.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let category = cudf::Series::from_column(encoded).unwrap().rename("category");
    DataFrame::from_series([category, nested, point, id]).unwrap()
}

mod lease_contracts {
    use super::*;

    #[test]
    fn shared_synchronous_completion_falls_back_by_return() {
        let _observation_lock = lock_event_observation();
        let runtime = Runtime::builder().build().unwrap();
        let series = runtime.series("id", &[1i32, 2, 3]).unwrap();

        let export = series.try_prepare_device_column_export().unwrap();
        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();

        assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    }

    #[test]
    fn shared_stream_lease_orders_consumer_stream_and_returns_witness() {
        let _observation_lock = lock_event_observation();
        let producer_stream = CudaStream::new().unwrap();
        let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
        let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let export = frame.try_prepare_device_table_export().unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let producer_stream_raw = producer_stream.__debug_raw_handle_value();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

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

        let witness = expect_shared_witness(lease.complete().unwrap(), CompletionWitnessKind::ConsumerStream);
        witness.synchronize().unwrap();
        assert!(witness.is_ready().unwrap());
    }

    #[test]
    fn shared_event_ready_stream_lease_waits_producer_event_before_consumer_stream() {
        let _observation_lock = lock_event_observation();
        let producer_stream = CudaStream::new().unwrap();
        let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(&producer_stream).unwrap();
        let export = series.try_prepare_device_column_export_with_event(&producer_ready).unwrap();

        match export.readiness() {
            ProducerReadiness::DeviceEventOrdered { event } => {
                assert_eq!(event.device(), producer_runtime.device().ordinal() as i32);
                assert!(!event.as_raw().is_null());
            },
            readiness => panic!("expected device-event readiness, got {readiness:?}"),
        }

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let producer_stream_raw = producer_stream.__debug_raw_handle_value();
        let producer_event_raw = producer_ready.__debug_interop_raw_handle_value();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

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

        let witness = expect_shared_witness(lease.complete().unwrap(), CompletionWitnessKind::ConsumerStream);
        witness.synchronize().unwrap();
    }

    #[test]
    fn guaranteed_fallback_shared_drop_synchronizes_consumer_stream() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let export = series.try_prepare_device_column_export().unwrap();
        assert_eq!(export.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);

        let consumer_stream = CudaStream::new().unwrap();
        let consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let _consumer_work = consumer_runtime.series("tmp", &[10i32, 20, 30]).unwrap();
        let consumer_done = CudaEvent::new().unwrap();

        {
            let lease = export
                .try_activate_read_lease(
                    shared_stream_read_start(&consumer_stream),
                    shared_same_device_completion(&consumer_stream),
                )
                .unwrap();
            assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
            consumer_done.record(&consumer_stream).unwrap();
        }

        assert!(consumer_done.is_ready().unwrap());
    }

    #[test]
    fn device_event_readiness_must_match_the_producer_device() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let producer_ready = CudaEvent::new().unwrap();
        producer_ready.record(producer_runtime.stream()).unwrap();

        let export = series.try_prepare_device_column_export_with_event(&producer_ready).unwrap();
        match export.readiness() {
            ProducerReadiness::DeviceEventOrdered { event } => {
                assert_eq!(event.device(), producer_runtime.device().ordinal() as i32);
                assert!(!event.as_raw().is_null());
            },
            readiness => panic!("expected device-event readiness, got {readiness:?}"),
        }

        if !should_run_multi_gpu_test() {
            return;
        }
        let _device_guard = common::DeviceRestoreGuard::new();
        let (device0, device1) = first_two_test_devices();
        common::set_current_device(device0);
        let producer_runtime = Runtime::builder().with_device(device0).build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();

        common::set_current_device(device1);
        let wrong_device_event = CudaEvent::new().unwrap();
        let wrong_device_stream = CudaStream::new().unwrap();
        wrong_device_event.record(&wrong_device_stream).unwrap();

        common::set_current_device(device0);
        let err = series.try_prepare_device_column_export_with_event(&wrong_device_event).unwrap_err();
        assert_eq!(err.kind(), InteropErrorKind::ReadinessMismatch);
    }

    #[test]
    fn explicit_shared_backend_requires_explicit_completion_and_releases_guard_on_drop() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let export = series
            .try_prepare_device_column_export()
            .unwrap()
            .with_lease_backend(cuda_explicit_completion_required_lease_backend());
        let cloned = export.clone();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();
        assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);

        reset_event_observation();
        let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);
        drop(lease);
        assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record);

        let reacquired = cloned
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert_eq!(reacquired.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    }

    #[test]
    fn runtime_stream_lease_records_one_completion_event_and_returns_witness() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let export = series.try_prepare_device_column_export().unwrap();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

        let lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();
        assert_eq!(lease.export().incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);

        reset_event_observation();
        let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);
        let witness = expect_runtime_witness(lease.complete().unwrap(), CompletionWitnessKind::ConsumerStream);
        assert_eq!(witness.device(), producer_runtime.device());
        assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record + 1);
        witness.synchronize().unwrap();
        assert!(witness.is_ready().unwrap());
        assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record + 1);
    }

    #[test]
    fn runtime_custom_backend_complete_path_is_used() {
        let _observation_lock = lock_event_observation();
        struct CountingFallbackBackend {
            complete_calls: Arc<AtomicUsize>,
        }

        impl ActiveLeaseBackend for CountingFallbackBackend {
            fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
                IncompleteDropBehavior::ExplicitCompletionRequired
            }

            fn complete(
                &self,
                _export: &LeaseExportContext,
                _completion: CompletionBinding,
            ) -> std::result::Result<LeaseCompletionResult, InteropError> {
                self.complete_calls.fetch_add(1, Ordering::SeqCst);
                Ok(LeaseCompletionResult::FallbackSynchronized)
            }

            fn drop_incomplete(
                &self,
                _export: &LeaseExportContext,
                _completion: CompletionBinding,
            ) -> std::result::Result<(), InteropError> {
                Ok(())
            }
        }

        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let complete_calls = Arc::new(AtomicUsize::new(0));
        let export = series
            .try_prepare_device_column_export()
            .unwrap()
            .with_lease_backend(LeaseBackend::new(CountingFallbackBackend { complete_calls: complete_calls.clone() }));

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();

        let lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();

        assert!(matches!(lease.complete().unwrap(), RuntimeLeaseCompletionResult::FallbackSynchronized));
        assert_eq!(complete_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn runtime_explicit_backend_drop_does_not_synchronize_and_releases_guard() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let export = frame
            .try_prepare_device_table_export()
            .unwrap()
            .with_lease_backend(cuda_explicit_completion_required_lease_backend());
        let cloned = export.clone();

        let consumer_stream = CudaStream::new().unwrap();
        let _consumer_runtime = Runtime::builder()
            .with_device(producer_runtime.device())
            .with_stream(consumer_stream.clone())
            .build()
            .unwrap();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

        let lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();
        assert_eq!(lease.export().incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);

        reset_event_observation();
        let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);
        drop(lease);
        assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record);

        let reacquired = cloned
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();
        let witness = expect_runtime_witness(reacquired.complete().unwrap(), CompletionWitnessKind::ConsumerStream);
        witness.synchronize().unwrap();
    }

    #[test]
    fn shared_and_runtime_leases_can_overlap_and_reacquire_one_export_root() {
        let _observation_lock = lock_event_observation();
        let producer_runtime = Runtime::builder().build().unwrap();
        let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
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
        let runtime_lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();

        assert_eq!(shared_lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);

        let reacquired_runtime_lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();

        let witness = expect_runtime_witness(runtime_lease.complete().unwrap(), CompletionWitnessKind::ConsumerStream);
        witness.synchronize().unwrap();
        let witness =
            expect_runtime_witness(reacquired_runtime_lease.complete().unwrap(), CompletionWitnessKind::ConsumerStream);
        witness.synchronize().unwrap();
    }

    #[test]
    fn shared_cross_device_event_lease_orders_stream_then_waits_completion_event() {
        let _observation_lock = lock_event_observation();
        if !should_run_multi_gpu_test() {
            return;
        }
        let _device_guard = common::DeviceRestoreGuard::new();

        let (device0, device1) = first_two_test_devices();
        common::set_current_device(device0);
        let producer_runtime = Runtime::builder().with_device(device0).build().unwrap();
        let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let export = frame
            .try_prepare_device_table_export()
            .unwrap()
            .with_lease_backend(cuda_explicit_completion_required_lease_backend());

        common::set_current_device(device1);
        let consumer_stream = CudaStream::new().unwrap();
        let consumer_runtime =
            Runtime::builder().with_device(device1).with_stream(consumer_stream.clone()).build().unwrap();
        let _consumer_work = consumer_runtime.series("tmp", &[7i32, 8, 9]).unwrap();
        let consumer_event = CudaEvent::new().unwrap();
        consumer_event.record(&consumer_stream).unwrap();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();
        let consumer_event_raw = consumer_event.__debug_interop_raw_handle_value();

        common::set_current_device(device0);
        reset_event_observation();
        let baseline_activation_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);
        let baseline_completion_wait = observed_stream_wait_call_count_for_event_raw(consumer_event_raw);

        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_cross_device_completion(&consumer_event),
            )
            .unwrap();

        assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_activation_wait);
        assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_completion_wait);

        let witness = expect_shared_witness(lease.complete().unwrap(), CompletionWitnessKind::ConsumerEvent);
        assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_completion_wait + 1);
        witness.synchronize().unwrap();
    }

    #[test]
    fn runtime_cross_device_event_lease_orders_stream_then_waits_completion_event() {
        let _observation_lock = lock_event_observation();
        if !should_run_multi_gpu_test() {
            return;
        }
        let _device_guard = common::DeviceRestoreGuard::new();

        let (device0, device1) = first_two_test_devices();
        common::set_current_device(device0);
        let producer_runtime = Runtime::builder().with_device(device0).build().unwrap();
        let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
        let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let export = frame
            .try_prepare_device_table_export()
            .unwrap()
            .with_lease_backend(cuda_explicit_completion_required_lease_backend());

        common::set_current_device(device1);
        let consumer_stream = CudaStream::new().unwrap();
        let consumer_runtime =
            Runtime::builder().with_device(device1).with_stream(consumer_stream.clone()).build().unwrap();
        let _consumer_work = consumer_runtime.series("tmp", &[7i32, 8, 9]).unwrap();
        let consumer_event = CudaEvent::new().unwrap();
        consumer_event.record(&consumer_stream).unwrap();
        let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();
        let consumer_event_raw = consumer_event.__debug_interop_raw_handle_value();

        common::set_current_device(device0);
        reset_event_observation();
        let baseline_activation_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);
        let baseline_completion_wait = observed_stream_wait_call_count_for_event_raw(consumer_event_raw);

        let lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::cross_device_event(&consumer_event),
            )
            .unwrap();

        assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_activation_wait);
        assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_completion_wait);

        let witness = expect_runtime_witness(lease.complete().unwrap(), CompletionWitnessKind::ConsumerEvent);
        assert_eq!(witness.device(), device0);
        assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_completion_wait + 1);
        witness.synchronize().unwrap();
    }
}

mod device_export_contracts {
    use super::*;

    #[test]
    fn ambient_and_mixed_device_exports_are_rejected() {
        let column = Column::from_slice(&[1i32, 2, 3]).unwrap();
        let err = column.view().unwrap().try_prepare_device_column_export().unwrap_err();
        assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
        assert!(err.context().iter().any(
            |(key, value)| *key == "operation" && value == "interop::ColumnView::try_prepare_device_column_export"
        ));

        let ids = cudf::Series::new("id", &[1i32, 2, 3]).unwrap();
        let vals = cudf::Series::new("value", &[10i64, 20, 30]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let err = frame.try_prepare_device_table_export().unwrap_err();
        assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
        assert!(
            err.context()
                .iter()
                .any(|(key, value)| *key == "operation"
                    && value == "interop::DataFrame::try_prepare_device_table_export")
        );

        if !should_run_multi_gpu_test() {
            return;
        }

        let (device0, device1) = first_two_test_devices();
        let rt0 = Runtime::builder().with_device(device0).build().unwrap();
        let rt1 = Runtime::builder().with_device(device1).build().unwrap();
        let left =
            rt0.column_from_slice(&[1i32, 2, 3]).unwrap().__debug_rebind_execution_domain_explicit(9001).unwrap();
        let right =
            rt1.column_from_slice(&[4i32, 5, 6]).unwrap().__debug_rebind_execution_domain_explicit(9001).unwrap();
        let frame =
            crate::common::dataframe_from_columns(vec!["left".into(), "right".into()], vec![left, right]).unwrap();

        let err = frame.try_prepare_device_table_export().unwrap_err();
        assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
    }

    #[test]
    fn device_fixed_width_exports_preserve_metadata_validity_and_slice_offset() {
        let runtime = Runtime::builder().build().unwrap();
        let nullable = runtime.column_from_slice_nullable(&[10i32, 20, 30], &[true, false, true]).unwrap();
        let nullable_view = nullable.view().unwrap();
        let expected_stream = nullable.__debug_execution_domain_stream_value();
        let nullable_export = nullable_view.try_prepare_device_column_export().unwrap();

        assert_eq!(nullable_export.column().data_type().id(), InteropTypeId::Int32);
        assert_eq!(nullable_export.column().len(), 3);
        assert_eq!(nullable_export.column().offset(), 0);
        assert_eq!(nullable_export.diagnostics().profile(), ExportProfile::PreparedDeviceExport);

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
        assert_device_stream_readiness(nullable_export.readiness(), expected_stream, runtime.device());

        let base = runtime.column_from_slice(&[11i32, 22, 33, 44]).unwrap();
        let base_view = base.view().unwrap();
        let sliced = base.slice_view(1, 2).unwrap();
        let sliced_export = sliced.try_prepare_device_column_export().unwrap();
        assert_eq!(sliced_export.column().len(), 2);
        assert_eq!(sliced_export.column().offset(), 1);

        let values = match sliced_export.column().layout() {
            &ColumnLayout::FixedWidth { values } => values,
            layout => panic!("expected fixed-width layout, got {layout:?}"),
        };
        assert_eq!(binding(sliced_export.bindings(), values).bytes().base_ptr(), base_view.device_data_ptr().unwrap());
    }

    #[test]
    fn device_table_export_preserves_names_nullability_readiness_and_low_level_equivalence() {
        let runtime = Runtime::builder().build().unwrap();
        let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
        let vals = runtime.nullable_series("value", &[10i64, 0, 30], &[true, false, true]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();
        let expected_stream = frame.__debug_execution_domain_stream_value();

        let frame_export = frame.try_prepare_device_table_export().unwrap();
        let table_view = frame.table_view().unwrap();
        let table_export = table_view.table().try_prepare_device_table_export().unwrap();

        assert_eq!(frame_export.table().num_rows(), 3);
        assert_eq!(frame_export.table().names().unwrap(), expected_names(&["id", "value"]).as_slice());
        assert_eq!(frame_export.columns()[0].schema_nullability(), SchemaNullability::NonNullable);
        assert_eq!(frame_export.columns()[1].schema_nullability(), SchemaNullability::Nullable);
        assert_eq!(frame_export.columns(), table_export.columns());
        assert_eq!(frame_export.bindings(), table_export.bindings());
        assert_eq!(frame_export.readiness(), table_export.readiness());
        assert_device_stream_readiness(frame_export.readiness(), expected_stream, runtime.device());
    }

    #[test]
    fn device_string_exports_preserve_binary_like_metadata_and_active_slice_span() {
        let runtime = Runtime::builder().build().unwrap();
        let strings = runtime.column_from_strs(&["alice", "bob"]).unwrap();
        let strings_view = strings.view().unwrap();
        let offsets_view = strings_view.child(0).unwrap();
        let expected_stream = strings.__debug_execution_domain_stream_value();
        let export = strings_view.try_prepare_device_column_export().unwrap();

        assert_eq!(export.column().data_type().id(), InteropTypeId::String);
        assert_eq!(export.column().len(), 2);
        assert_eq!(export.column().offset(), 0);

        let (offsets, data) = match export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        let offsets_binding = binding(export.bindings(), offsets);
        assert_eq!(offsets_binding.bytes().base_ptr(), offsets_view.device_data_ptr().unwrap());
        assert_device_binding_on(offsets_binding, runtime.device());
        let data_binding = binding(export.bindings(), data);
        assert_device_binding_on(data_binding, runtime.device());
        assert!(data_binding.bytes().byte_len() >= "alicebob".len());
        assert_device_stream_readiness(export.readiness(), expected_stream, runtime.device());

        let base = runtime.column_from_strs(&["zero", "one", "two", "three"]).unwrap();
        let base_view = base.view().unwrap();
        let base_export = base_view.try_prepare_device_column_export().unwrap();
        let sliced = base.slice_view(1, 2).unwrap();
        let sliced_offsets_view = sliced.child(0).unwrap();
        let sliced_export = sliced.try_prepare_device_column_export().unwrap();

        assert_eq!(sliced_export.column().len(), 2);
        assert_eq!(sliced_export.column().offset(), 1);

        let (offsets, data) = match sliced_export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        let (_, base_data) = match base_export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        let base_data_binding = binding(base_export.bindings(), *base_data);

        let offsets_binding = binding(sliced_export.bindings(), offsets);
        assert_eq!(offsets_binding.bytes().base_ptr(), sliced_offsets_view.device_data_ptr().unwrap());
        assert_eq!(offsets_binding.bytes().byte_len(), 4 * std::mem::size_of::<i32>());
        let data_binding = binding(sliced_export.bindings(), data);
        assert_eq!(data_binding.bytes().base_ptr(), base_data_binding.bytes().base_ptr());
        assert_eq!(data_binding.bytes().byte_len(), "onetwo".len());
    }

    #[test]
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
            ColumnLayout::Dictionary { keys, .. } => *keys,
            layout => panic!("expected dictionary layout, got {layout:?}"),
        };
        let dictionary_keys_column = semantic_column(export.semantic_columns(), dictionary_keys);
        assert_eq!(dictionary_keys_column.data_type().id(), InteropTypeId::String);
        assert!(matches!(dictionary_keys_column.layout(), ColumnLayout::BinaryLike { .. }));

        let nested_child = match export.columns()[1].layout() {
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected list layout, got {layout:?}"),
        };
        let nested_list = semantic_column(export.semantic_columns(), nested_child);
        let nested_leaf = match nested_list.layout() {
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
        assert_eq!(semantic_column(export.semantic_columns(), nested_leaf).data_type().id(), InteropTypeId::Int64);

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
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        let struct_leaf = match semantic_column(export.semantic_columns(), struct_nested).layout() {
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        assert_eq!(semantic_column(export.semantic_columns(), struct_leaf).data_type().id(), InteropTypeId::Int64);

        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    }
}

mod host_export_contracts {
    use super::*;

    #[test]
    fn host_fixed_width_exports_materialize_active_range_and_pack_validity() {
        let runtime = Runtime::builder().build().unwrap();

        let base = runtime.column_from_slice(&[11i32, 22, 33, 44]).unwrap();
        let sliced = base.slice_view(1, 2).unwrap();
        let sliced_export = sliced.try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&sliced_export, InteropTypeId::Int32, 2, 0);
        let values = match sliced_export.column().layout() {
            ColumnLayout::FixedWidth { values } => *values,
            layout => panic!("expected fixed-width layout, got {layout:?}"),
        };
        assert_host_binding_bytes(sliced_export.bindings(), values, expected_i32_bytes(&[22, 33]));

        let nullable = runtime.column_from_slice_nullable(&[10i32, 20, 30, 40], &[true, false, true, false]).unwrap();
        let nullable_export = nullable.view().unwrap().try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&nullable_export, InteropTypeId::Int32, 4, 0);
        let mask = match nullable_export.column().validity() {
            PhysicalValidity::ValidityMask { mask, null_count } => {
                assert_eq!(*null_count, Some(2));
                *mask
            },
            PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
        };
        assert_host_binding_bytes(nullable_export.bindings(), mask, vec![0b0000_0101]);
    }

    #[test]
    fn host_table_export_is_host_visible_and_rejects_cuda_completion() {
        let runtime = Runtime::builder().build().unwrap();
        let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
        let vals = runtime.nullable_series("value", &[10i64, 0, 30], &[true, false, true]).unwrap();
        let frame = DataFrame::from_series([ids, vals]).unwrap();

        let export = frame.try_prepare_host_table_export().unwrap();
        assert_host_table_basics(&export, &["id", "value"], 3);
        assert_eq!(export.columns()[0].schema_nullability(), SchemaNullability::NonNullable);
        assert_eq!(export.columns()[1].schema_nullability(), SchemaNullability::Nullable);

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

    #[test]
    fn host_string_exports_materialize_binary_like_bytes_rebase_slice_and_pack_validity() {
        let runtime = Runtime::builder().build().unwrap();

        let strings = runtime.column_from_strs(&["alice", "bob"]).unwrap();
        let export = strings.view().unwrap().try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&export, InteropTypeId::String, 2, 0);
        let (offsets, data) = match export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), offsets, expected_i32_bytes(&[0, 5, 8]));
        assert_host_binding_bytes(export.bindings(), data, b"alicebob".to_vec());

        let base = runtime.column_from_strs(&["zero", "one", "two", "three"]).unwrap();
        let sliced = base.slice_view(1, 2).unwrap();
        let sliced_export = sliced.try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&sliced_export, InteropTypeId::String, 2, 0);
        let (offsets, data) = match sliced_export.column().layout() {
            ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
            layout => panic!("expected binary-like layout, got {layout:?}"),
        };
        assert_host_binding_bytes(sliced_export.bindings(), offsets, expected_i32_bytes(&[0, 3, 6]));
        assert_host_binding_bytes(sliced_export.bindings(), data, b"onetwo".to_vec());

        let nullable = runtime
            .nullable_str_series("name", &["alice", "placeholder", "bob", "unused"], &[true, false, true, false])
            .unwrap();
        let nullable_export = nullable.try_prepare_host_column_export().unwrap();
        let mask = match nullable_export.column().validity() {
            PhysicalValidity::ValidityMask { mask, null_count } => {
                assert_eq!(*null_count, Some(2));
                *mask
            },
            PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
        };
        assert_host_binding_bytes(nullable_export.bindings(), mask, vec![0b0000_0101]);
    }

    #[test]
    fn host_list_slice_export_rebases_offsets_and_child_range() {
        let frame = read_list_dataframe("{\"lists\":[1,2]}\n{\"lists\":[3]}\n{\"lists\":[]}\n{\"lists\":[4,5,6]}\n");
        let series = frame.series("lists").unwrap();
        let sliced = series.as_column().slice_view(1, 2).unwrap();

        let export = sliced.try_prepare_host_column_export().unwrap();
        assert_host_column_basics(&export, InteropTypeId::List, 2, 0);

        let (offsets, child) = match export.column().layout() {
            ColumnLayout::List { offsets, child } => (*offsets, *child),
            layout => panic!("expected list layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), offsets, expected_i32_bytes(&[0, 1, 1]));

        let child_column = semantic_column(export.semantic_columns(), child);
        assert_eq!(child_column.data_type().id(), InteropTypeId::Int64);
        assert_eq!(child_column.len(), 1);
        assert_eq!(child_column.offset(), 0);
        let values = match child_column.layout() {
            ColumnLayout::FixedWidth { values } => *values,
            layout => panic!("expected fixed-width child layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), values, expected_i64_bytes(&[3]));

        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    }

    #[test]
    fn host_recursive_nested_export_materializes_supported_projections() {
        let frame = recursive_supported_frame();
        let export = frame.try_prepare_host_table_export().unwrap();
        assert_host_table_basics(&export, &["category", "nested", "point", "id"], 3);

        let dictionary_keys = match export.columns()[0].layout() {
            ColumnLayout::Dictionary { keys, .. } => *keys,
            layout => panic!("expected dictionary layout, got {layout:?}"),
        };
        let dictionary_keys_column = semantic_column(export.semantic_columns(), dictionary_keys);
        assert_eq!(dictionary_keys_column.data_type().id(), InteropTypeId::String);
        let (key_offsets, key_data) = match dictionary_keys_column.layout() {
            ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
            layout => panic!("expected binary-like dictionary keys, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), key_offsets, expected_i32_bytes(&[0, 5, 9]));
        assert_host_binding_bytes(export.bindings(), key_data, b"alphabeta".to_vec());

        let nested_child = match export.columns()[1].layout() {
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected list layout, got {layout:?}"),
        };
        let nested_list = semantic_column(export.semantic_columns(), nested_child);
        let (nested_offsets, nested_leaf) = match nested_list.layout() {
            ColumnLayout::List { offsets, child } => (*offsets, *child),
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        let root_offsets = match export.columns()[1].layout() {
            ColumnLayout::List { offsets, .. } => *offsets,
            layout => panic!("expected list layout, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), root_offsets, expected_i32_bytes(&[0, 2, 4, 5]));
        assert_host_binding_bytes(export.bindings(), nested_offsets, expected_i32_bytes(&[0, 2, 3, 3, 5, 6]));
        let nested_leaf_column = semantic_column(export.semantic_columns(), nested_leaf);
        assert_eq!(nested_leaf_column.data_type().id(), InteropTypeId::Int64);
        let leaf_values = match nested_leaf_column.layout() {
            ColumnLayout::FixedWidth { values } => *values,
            layout => panic!("expected fixed-width nested leaf, got {layout:?}"),
        };
        assert_host_binding_bytes(export.bindings(), leaf_values, expected_i64_bytes(&[1, 2, 3, 4, 5, 6]));

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
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        let struct_leaf = match semantic_column(export.semantic_columns(), struct_nested).layout() {
            ColumnLayout::List { child, .. } => *child,
            layout => panic!("expected nested list layout, got {layout:?}"),
        };
        assert_eq!(semantic_column(export.semantic_columns(), struct_leaf).data_type().id(), InteropTypeId::Int64);

        let lease = export
            .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
            .unwrap();
        assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    }
}
