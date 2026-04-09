use std::sync::{
    Arc, Mutex, MutexGuard, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use cudf::{
    DataFrame, DeviceId, JsonReadOptions, Runtime, TypeId,
    core::Column,
    interop::{
        ActiveLeaseBackend, BufferBinding, ColumnLayout, CompletionAuthority, CompletionBinding, CompletionWitnessKind,
        CudaEventHandle, CudaStreamHandle, ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind,
        InteropTypeId, LeaseBackend, LeaseCompletionResult, LeaseExportContext, PhysicalValidity, Placement,
        ProducerReadiness, ReadStartAuthority, RuntimeCompletionAuthority, RuntimeLeaseCompletionResult,
        RuntimeReadStartAuthority, SchemaNullability, TryActivateRuntimeColumnReadLease,
        TryActivateRuntimeTableReadLease, cuda_explicit_completion_required_lease_backend,
    },
    ops::dictionary,
    runtime::{CudaEvent, CudaStream},
};
use tempfile::TempDir;

use crate::common;

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

fn lock_event_observation() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
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

fn binding_bytes(binding: &BufferBinding<'_>) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(binding.bytes().base_ptr() as *const u8, binding.bytes().byte_len()) }.to_vec()
}

fn expected_i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn expected_i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
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

fn read_struct_dataframe(rows: &str) -> DataFrame {
    read_json_dataframe("structs.jsonl", rows)
}

#[test]
fn series_prepare_device_export_can_activate_synchronous_read_lease() {
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("id", &[1i32, 2, 3]).unwrap();

    let export = series.try_prepare_device_column_export().unwrap();
    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn dataframe_prepare_device_export_can_activate_stream_read_lease() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();

    let export = frame.try_prepare_device_table_export().unwrap();
    // Test-only shortcut: reusing the producer's stream as the consumer's
    // completion stream. Real consumers should supply their own stream to keep
    // producer readiness and consumer completion authority separate.
    let (read_start, completion) = match export.readiness() {
        ProducerReadiness::DeviceStreamOrdered { stream } => (
            ReadStartAuthority::ConsumerStream { stream: *stream },
            CompletionAuthority::SameDeviceStream { stream: *stream },
        ),
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
    };

    let lease = export.try_activate_read_lease(read_start, completion).unwrap();

    let witness = match lease.complete().unwrap() {
        LeaseCompletionResult::Completed(witness) => witness,
        LeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    };
    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    witness.synchronize().unwrap();
    assert!(witness.is_ready().unwrap());
}

#[test]
fn shared_table_read_lease_orders_consumer_stream_after_stream_ordered_producer_readiness() {
    let _observation_lock = lock_event_observation();
    let producer_stream = CudaStream::new().unwrap();
    let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
    let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();
    let export = frame.try_prepare_device_table_export().unwrap();
    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
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

    let witness = match lease.complete().unwrap() {
        LeaseCompletionResult::Completed(witness) => witness,
        LeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for shared same-device stream completion")
        },
    };
    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    witness.synchronize().unwrap();
}

#[test]
fn shared_column_read_lease_with_event_readiness_orders_consumer_stream_after_producer_event() {
    let _observation_lock = lock_event_observation();
    let producer_stream = CudaStream::new().unwrap();
    let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let producer_ready = CudaEvent::new().unwrap();
    producer_ready.record(&producer_stream).unwrap();
    let export = series.try_prepare_device_column_export_with_event(&producer_ready).unwrap();
    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
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

    let witness = match lease.complete().unwrap() {
        LeaseCompletionResult::Completed(witness) => witness,
        LeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for shared event-ready stream completion")
        },
    };
    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    witness.synchronize().unwrap();
}

#[test]
fn shared_stream_read_lease_drop_synchronizes_consumer_stream() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let _consumer_work = consumer_runtime.series("tmp", &[10i32, 20, 30]).unwrap();
    let consumer_done = CudaEvent::new().unwrap();

    {
        let _lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();

        // Record the event *after* lease activation so it captures stream-ordering
        // work enqueued by the lease. Before the lease is dropped (and its fallback
        // synchronization runs), the event must still be pending.
        consumer_done.record(&consumer_stream).unwrap();
    }
    // The lease drop performed fallback synchronization on the consumer stream,
    // which must have completed all prior work including the recorded event.
    assert!(consumer_done.is_ready().unwrap());
}

#[test]
fn series_prepare_device_export_can_activate_runtime_stream_read_lease() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    let result = lease.complete().unwrap();
    match result {
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
}

#[test]
fn runtime_column_read_lease_preserves_accessor_and_debug_surface() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    assert!(std::ptr::eq(lease.export(), &export));
    match lease.read_start() {
        RuntimeReadStartAuthority::ConsumerStream { stream } => {
            assert!(std::ptr::eq(*stream, &consumer_stream));
        },
        read_start => panic!("expected consumer-stream read_start, got {read_start:?}"),
    }
    match lease.completion() {
        RuntimeCompletionAuthority::SameDeviceStream { stream } => {
            assert!(std::ptr::eq(*stream, &consumer_stream));
        },
        completion => panic!("expected same-device-stream completion, got {completion:?}"),
    }

    let debug = format!("{lease:?}");
    assert!(debug.contains("RuntimeActiveColumnReadLease"));
    assert!(debug.contains("read_start"));
    assert!(debug.contains("completion"));
    assert!(debug.contains("producer_device"));

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    };
    witness.synchronize().unwrap();
}

#[test]
fn runtime_stream_read_lease_complete_does_not_fallback_synchronize_consumer_stream() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    reset_event_observation();
    let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    };

    assert_eq!(
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw),
        baseline_consumer_record + 1,
        "runtime complete() should record exactly one shared completion event on the consumer stream",
    );

    witness.synchronize().unwrap();
    assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record + 1,);
}

#[test]
fn series_prepare_device_export_defaults_to_cuda_guaranteed_fallback_backend() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    assert_eq!(export.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
    let lease = export
        .try_activate_read_lease(
            shared_stream_read_start(&consumer_stream),
            shared_same_device_completion(&consumer_stream),
        )
        .unwrap();
    assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::GuaranteedFallback);
}

#[test]
fn shared_cuda_explicit_backend_reports_behavior_and_releases_guard_on_drop() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series
        .try_prepare_device_column_export()
        .unwrap()
        .with_lease_backend(cuda_explicit_completion_required_lease_backend());
    let cloned = export.clone();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    {
        let lease = export
            .try_activate_read_lease(
                shared_stream_read_start(&consumer_stream),
                shared_same_device_completion(&consumer_stream),
            )
            .unwrap();
        assert_eq!(lease.incomplete_drop_behavior(), IncompleteDropBehavior::ExplicitCompletionRequired);
    }

    let reacquired = cloned
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(reacquired.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn shared_cuda_explicit_stream_read_lease_drop_does_not_hidden_fallback_synchronize_consumer_stream() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series
        .try_prepare_device_column_export()
        .unwrap()
        .with_lease_backend(cuda_explicit_completion_required_lease_backend());

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();
    let lease = export
        .try_activate_read_lease(
            shared_stream_read_start(&consumer_stream),
            shared_same_device_completion(&consumer_stream),
        )
        .unwrap();

    reset_event_observation();
    let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);
    drop(lease);

    assert_eq!(
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw),
        baseline_consumer_record,
        "explicit-completion shared drop must not record a hidden fallback event on the consumer stream",
    );
}

#[test]
fn runtime_table_read_lease_preserves_accessor_and_debug_surface() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();
    let export = frame.try_prepare_device_table_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    assert!(std::ptr::eq(lease.export(), &export));
    match lease.read_start() {
        RuntimeReadStartAuthority::ConsumerStream { stream } => {
            assert!(std::ptr::eq(*stream, &consumer_stream));
        },
        read_start => panic!("expected consumer-stream read_start, got {read_start:?}"),
    }
    match lease.completion() {
        RuntimeCompletionAuthority::SameDeviceStream { stream } => {
            assert!(std::ptr::eq(*stream, &consumer_stream));
        },
        completion => panic!("expected same-device-stream completion, got {completion:?}"),
    }

    let debug = format!("{lease:?}");
    assert!(debug.contains("RuntimeActiveTableReadLease"));
    assert!(debug.contains("read_start"));
    assert!(debug.contains("completion"));
    assert!(debug.contains("producer_device"));

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    };
    witness.synchronize().unwrap();
}

#[test]
fn runtime_stream_read_lease_with_explicit_backend_complete_returns_witness_via_shared_backend() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series
        .try_prepare_device_column_export()
        .unwrap()
        .with_lease_backend(cuda_explicit_completion_required_lease_backend());

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    reset_event_observation();
    let baseline_consumer_record = observed_raw_record_call_count_for_stream_raw(consumer_stream_raw);

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    };

    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    assert_eq!(
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw),
        baseline_consumer_record + 1,
        "runtime complete() with explicit backend should record exactly one shared completion event on the consumer stream",
    );

    witness.synchronize().unwrap();
    assert_eq!(observed_raw_record_call_count_for_stream_raw(consumer_stream_raw), baseline_consumer_record + 1,);
}

#[test]
fn runtime_complete_invokes_custom_shared_backend_complete_path() {
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
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

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
fn runtime_stream_read_lease_with_explicit_backend_drop_does_not_synchronize_and_releases_guard() {
    let _observation_lock = lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series
        .try_prepare_device_column_export()
        .unwrap()
        .with_lease_backend(cuda_explicit_completion_required_lease_backend());
    let cloned = export.clone();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
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

    assert_eq!(
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw),
        baseline_consumer_record,
        "runtime drop with explicit backend must not hidden-fallback synchronize the consumer stream",
    );

    let reacquired = cloned
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();
    let witness = match reacquired.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness after reacquiring explicit runtime lease")
        },
    };
    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    witness.synchronize().unwrap();
}

#[test]
fn runtime_table_read_lease_with_explicit_backend_drop_does_not_synchronize_and_releases_guard() {
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
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
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

    assert_eq!(
        observed_raw_record_call_count_for_stream_raw(consumer_stream_raw),
        baseline_consumer_record,
        "runtime table drop with explicit backend must not hidden-fallback synchronize the consumer stream",
    );

    let reacquired = cloned
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();
    let witness = match reacquired.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness after reacquiring explicit runtime table lease")
        },
    };
    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
    witness.synchronize().unwrap();
}

#[test]
fn runtime_cross_device_event_read_lease_with_explicit_backend_complete_returns_witness_without_extra_wait() {
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
    let consumer_event_raw = consumer_event.__debug_interop_raw_handle_value();

    common::set_current_device(device0);
    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::cross_device_event(&consumer_event),
        )
        .unwrap();

    reset_event_observation();
    let baseline_consumer_event_wait = observed_stream_wait_call_count_for_event_raw(consumer_event_raw);

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for cross-device event completion")
        },
    };

    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
    assert_eq!(witness.device(), device0);
    assert_eq!(
        observed_stream_wait_call_count_for_event_raw(consumer_event_raw),
        baseline_consumer_event_wait + 1,
        "runtime complete() with explicit backend must wait on the consumer event exactly once",
    );

    witness.synchronize().unwrap();
    assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_consumer_event_wait + 1,);
}

#[test]
fn runtime_stream_read_lease_drop_synchronizes_consumer_stream() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let _consumer_work = consumer_runtime.series("tmp", &[10i32, 20, 30]).unwrap();
    let consumer_done = CudaEvent::new().unwrap();

    {
        let _lease = export
            .try_activate_runtime_read_lease(
                RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
                RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
            )
            .unwrap();

        // Record the event *after* lease activation so it captures stream-ordering
        // work enqueued by the lease. Before the lease is dropped (and its fallback
        // synchronization runs), the event must still be pending.
        consumer_done.record(&consumer_stream).unwrap();
    }
    // The lease drop performed fallback synchronization on the consumer stream,
    // which must have completed all prior work including the recorded event.
    assert!(consumer_done.is_ready().unwrap());
}

#[test]
fn runtime_stream_read_lease_orders_consumer_stream_after_stream_ordered_producer_readiness() {
    let _observation_lock = lock_event_observation();
    let producer_stream = CudaStream::new().unwrap();
    let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();
    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let producer_stream_raw = producer_stream.__debug_raw_handle_value();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

    reset_event_observation();
    let baseline_producer_record = observed_raw_record_call_count_for_stream_raw(producer_stream_raw);
    let baseline_consumer_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);
    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    assert!(observed_raw_record_call_count_for_stream_raw(producer_stream_raw) > baseline_producer_record);
    assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_consumer_wait);

    let result = lease.complete().unwrap();
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
            witness.synchronize().unwrap();
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    }
}

#[test]
fn runtime_table_read_lease_orders_consumer_stream_after_stream_ordered_producer_readiness() {
    let _observation_lock = lock_event_observation();
    let producer_stream = CudaStream::new().unwrap();
    let producer_runtime = Runtime::builder().with_stream(producer_stream.clone()).build().unwrap();
    let ids = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = producer_runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();
    let export = frame.try_prepare_device_table_export().unwrap();
    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();
    let producer_stream_raw = producer_stream.__debug_raw_handle_value();
    let consumer_stream_raw = consumer_stream.__debug_raw_handle_value();

    reset_event_observation();
    let baseline_producer_record = observed_raw_record_call_count_for_stream_raw(producer_stream_raw);
    let baseline_consumer_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);
    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    assert!(observed_raw_record_call_count_for_stream_raw(producer_stream_raw) > baseline_producer_record);
    assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_consumer_wait);

    let result = lease.complete().unwrap();
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
            witness.synchronize().unwrap();
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for same-device stream completion")
        },
    }
}

#[test]
fn runtime_and_shared_read_leases_can_overlap_on_one_export_root() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();
    let cloned = export.clone();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    let runtime_lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    let shared_lease = cloned
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(shared_lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
    let result = runtime_lease.complete().unwrap();
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
            witness.synchronize().unwrap();
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected runtime lease completion to keep using the runtime witness path")
        },
    }
}

#[test]
fn shared_read_lease_allows_runtime_activation_before_completion() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

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
    let first_result = runtime_lease.complete().unwrap();
    match first_result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
            witness.synchronize().unwrap();
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected overlapping runtime activation to stay on the runtime witness path")
        },
    }
    let result = reacquired_runtime_lease.complete().unwrap();
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerStream);
            witness.synchronize().unwrap();
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected runtime activation to reacquire the export root after shared completion")
        },
    }
}

#[test]
fn dataframe_prepare_device_export_can_activate_runtime_cross_device_event_read_lease() {
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
    let export = frame.try_prepare_device_table_export().unwrap();

    common::set_current_device(device1);
    let consumer_stream = CudaStream::new().unwrap();
    let consumer_runtime =
        Runtime::builder().with_device(device1).with_stream(consumer_stream.clone()).build().unwrap();
    let _consumer_work = consumer_runtime.series("tmp", &[7i32, 8, 9]).unwrap();
    let consumer_event = CudaEvent::new().unwrap();
    consumer_event.record(&consumer_stream).unwrap();

    common::set_current_device(device0);
    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::cross_device_event(&consumer_event),
        )
        .unwrap();

    let result = lease.complete().unwrap();
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
            assert_eq!(witness.device(), device0);
            witness.synchronize().unwrap();
            assert!(witness.is_ready().unwrap());
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for cross-device event completion")
        },
    }
}

#[test]
fn shared_cross_device_read_lease_orders_consumer_stream_on_activation_then_waits_consumer_event_on_complete() {
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

    assert!(
        observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_activation_wait,
        "activation must order the consumer stream after producer readiness for cross-device reads",
    );
    assert_eq!(
        observed_stream_wait_call_count_for_event_raw(consumer_event_raw),
        baseline_completion_wait,
        "activation must not wait on the consumer completion event",
    );

    let witness = match lease.complete().unwrap() {
        LeaseCompletionResult::Completed(witness) => witness,
        LeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for cross-device event completion")
        },
    };

    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
    assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_completion_wait + 1);
    witness.synchronize().unwrap();
}

#[test]
fn runtime_cross_device_event_read_lease_complete_does_not_fallback_synchronize_consumer_event() {
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
    let export = frame.try_prepare_device_table_export().unwrap();

    common::set_current_device(device1);
    let consumer_stream = CudaStream::new().unwrap();
    let consumer_runtime =
        Runtime::builder().with_device(device1).with_stream(consumer_stream.clone()).build().unwrap();
    let _consumer_work = consumer_runtime.series("tmp", &[7i32, 8, 9]).unwrap();
    let consumer_event = CudaEvent::new().unwrap();
    consumer_event.record(&consumer_stream).unwrap();
    let consumer_event_raw = consumer_event.__debug_interop_raw_handle_value();

    common::set_current_device(device0);
    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::cross_device_event(&consumer_event),
        )
        .unwrap();

    reset_event_observation();
    let baseline_consumer_event_wait = observed_stream_wait_call_count_for_event_raw(consumer_event_raw);

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for cross-device event completion")
        },
    };

    assert_eq!(
        observed_stream_wait_call_count_for_event_raw(consumer_event_raw),
        baseline_consumer_event_wait + 1,
        "runtime complete() must wait on the consumer event exactly once",
    );

    witness.synchronize().unwrap();
    assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_consumer_event_wait + 1,);
}

#[test]
fn runtime_cross_device_read_lease_orders_consumer_stream_on_activation_then_waits_consumer_event_on_complete() {
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

    assert!(
        observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_activation_wait,
        "runtime activation must order the consumer stream after producer readiness for cross-device reads",
    );
    assert_eq!(
        observed_stream_wait_call_count_for_event_raw(consumer_event_raw),
        baseline_completion_wait,
        "runtime activation must not wait on the consumer completion event",
    );

    let witness = match lease.complete().unwrap() {
        RuntimeLeaseCompletionResult::Completed(witness) => witness,
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for cross-device event completion")
        },
    };

    assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
    assert_eq!(observed_stream_wait_call_count_for_event_raw(consumer_event_raw), baseline_completion_wait + 1);
    witness.synchronize().unwrap();
}

#[test]
fn series_prepare_device_export_with_event_preserves_event_readiness() {
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
}

#[test]
fn series_prepare_device_export_with_event_can_activate_runtime_stream_read_lease() {
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series("id", &[1i32, 2, 3]).unwrap();
    let producer_ready = CudaEvent::new().unwrap();
    producer_ready.record(producer_runtime.stream()).unwrap();
    let export = series.try_prepare_device_column_export_with_event(&producer_ready).unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::same_device_stream(&consumer_stream),
        )
        .unwrap();

    let result = lease.complete().unwrap();
    match result {
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
}

#[test]
fn dataframe_prepare_device_export_with_event_can_activate_runtime_cross_device_event_read_lease() {
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
    let producer_ready = CudaEvent::new().unwrap();
    producer_ready.record(producer_runtime.stream()).unwrap();
    let export = frame.try_prepare_device_table_export_with_event(&producer_ready).unwrap();

    common::set_current_device(device1);
    let consumer_stream = CudaStream::new().unwrap();
    let consumer_runtime =
        Runtime::builder().with_device(device1).with_stream(consumer_stream.clone()).build().unwrap();
    let _consumer_work = consumer_runtime.series("tmp", &[7i32, 8, 9]).unwrap();
    let consumer_event = CudaEvent::new().unwrap();
    consumer_event.record(&consumer_stream).unwrap();

    common::set_current_device(device0);
    let lease = export
        .try_activate_runtime_read_lease(
            RuntimeReadStartAuthority::consumer_stream(&consumer_stream),
            RuntimeCompletionAuthority::cross_device_event(&consumer_event),
        )
        .unwrap();

    let result = lease.complete().unwrap();
    match result {
        RuntimeLeaseCompletionResult::Completed(witness) => {
            assert_eq!(witness.kind(), CompletionWitnessKind::ConsumerEvent);
            assert_eq!(witness.device(), device0);
            witness.synchronize().unwrap();
            assert!(witness.is_ready().unwrap());
        },
        RuntimeLeaseCompletionResult::FallbackSynchronized => {
            panic!("expected producer-owned completion witness for cross-device event completion")
        },
    }
}

#[test]
fn series_prepare_device_export_with_event_rejects_mismatched_event_device() {
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
fn column_view_prepare_device_export_preserves_fixed_width_metadata() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let view = column.view().unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();

    let export = view.try_prepare_device_column_export().unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::Int32);
    assert_eq!(export.column().len(), 3);
    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedDeviceExport);

    let values = match export.column().layout() {
        &ColumnLayout::FixedWidth { values } => values,
        layout => panic!("expected fixed-width layout, got {layout:?}"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(binding.bytes().base_ptr(), view.device_data_ptr().unwrap());
    assert_eq!(binding.placement(), &Placement::CudaDevice { device: runtime.device().ordinal() as i32 });

    match export.readiness() {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream);
            assert_eq!(stream.device(), runtime.device().ordinal() as i32);
        },
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
    }
}

#[test]
fn nullable_column_prepare_device_export_preserves_present_validity_mask() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_nullable_with_runtime(&runtime, &[10i32, 20, 30], &[true, false, true]).unwrap();
    let view = column.view().unwrap();

    let export = view.try_prepare_device_column_export().unwrap();

    let mask = match export.column().validity() {
        PhysicalValidity::ValidityMask { mask, null_count } => {
            assert_eq!(*null_count, Some(1));
            *mask
        },
        PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == mask).unwrap();
    assert_eq!(binding.placement(), &Placement::CudaDevice { device: runtime.device().ordinal() as i32 });
    assert!(binding.bytes().byte_len() >= 1);
}

#[test]
fn sliced_column_prepare_device_export_preserves_base_pointer_and_offset() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[11i32, 22, 33, 44]).unwrap();
    let base = column.view().unwrap();
    let sliced = column.slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_device_column_export().unwrap();

    assert_eq!(export.column().offset(), 1);
    assert_eq!(export.column().len(), 2);

    let values = match export.column().layout() {
        &ColumnLayout::FixedWidth { values } => values,
        layout => panic!("expected fixed-width layout, got {layout:?}"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(binding.bytes().base_ptr(), base.device_data_ptr().unwrap());
}

#[test]
fn table_view_prepare_device_export_preserves_num_rows_names_and_uniform_readiness() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();
    let expected_stream = frame.__debug_execution_domain_stream_value();

    let frame_export = frame.try_prepare_device_table_export().unwrap();
    let (table, _) = frame.clone().try_into_table().unwrap();
    let table_view = table.view().unwrap();
    let table_export = table_view.try_prepare_device_table_export().unwrap();

    assert_eq!(table_export.table().num_rows(), 3);
    assert_eq!(frame_export.table().names().unwrap(), &["id".to_string(), "value".to_string()]);
    assert!(
        table_export
            .bindings()
            .iter()
            .all(|binding| binding.placement() == &Placement::CudaDevice { device: runtime.device().ordinal() as i32 })
    );

    match table_export.readiness() {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream);
            assert_eq!(stream.device(), runtime.device().ordinal() as i32);
        },
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
    }
}

#[test]
fn table_view_prepare_device_export_preserves_schema_nullability_metadata() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.nullable_series("value", &[10i64, 0, 30], &[true, false, true]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();

    let export = frame.try_prepare_device_table_export().unwrap();

    assert_eq!(export.columns()[0].schema_nullability(), SchemaNullability::NonNullable);
    assert_eq!(export.columns()[1].schema_nullability(), SchemaNullability::Nullable);
}

#[test]
fn series_prepare_host_export_can_activate_synchronous_read_lease() {
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("id", &[1i32, 2, 3]).unwrap();

    let export = series.try_prepare_host_column_export().unwrap();
    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();

    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn column_view_prepare_host_export_materializes_fixed_width_bytes() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[1i32, 2, 3]).unwrap();
    let view = column.view().unwrap();

    let export = view.try_prepare_host_column_export().unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::Int32);
    assert_eq!(export.column().len(), 3);
    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);

    let values = match export.column().layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width layout, got {layout:?}"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(binding), expected_i32_bytes(&[1, 2, 3]));
}

#[test]
fn sliced_column_prepare_host_export_resets_offset_and_materializes_active_range_only() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_slice_with_runtime(&runtime, &[11i32, 22, 33, 44]).unwrap();
    let sliced = column.slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_host_column_export().unwrap();

    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.column().len(), 2);

    let values = match export.column().layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width layout, got {layout:?}"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(binding), expected_i32_bytes(&[22, 33]));
}

#[test]
fn nullable_column_prepare_host_export_preserves_packed_validity_mask() {
    let runtime = Runtime::builder().build().unwrap();
    let column =
        Column::from_slice_nullable_with_runtime(&runtime, &[10i32, 20, 30, 40], &[true, false, true, false]).unwrap();
    let view = column.view().unwrap();

    let export = view.try_prepare_host_column_export().unwrap();

    let mask = match export.column().validity() {
        &PhysicalValidity::ValidityMask { mask, null_count } => {
            assert_eq!(null_count, Some(2));
            mask
        },
        PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == mask).unwrap();
    assert_eq!(binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(binding), vec![0b0000_0101]);
}

#[test]
fn dataframe_prepare_host_export_preserves_names_and_host_profile() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();

    let export = frame.try_prepare_host_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(export.table().names().unwrap(), &["id".to_string(), "value".to_string()]);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));
}

#[test]
fn dataframe_prepare_host_export_preserves_schema_nullability_metadata() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.nullable_series("value", &[10i64, 0, 30], &[true, false, true]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();

    let export = frame.try_prepare_host_table_export().unwrap();

    assert_eq!(export.columns()[0].schema_nullability(), SchemaNullability::NonNullable);
    assert_eq!(export.columns()[1].schema_nullability(), SchemaNullability::Nullable);
}

#[test]
fn series_prepare_host_export_rejects_cuda_completion_authority() {
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    let export = series.try_prepare_host_column_export().unwrap();
    let err = export
        .try_activate_read_lease(
            shared_stream_read_start(&consumer_stream),
            shared_same_device_completion(&consumer_stream),
        )
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ProfileUnsupported);
}

#[test]
fn struct_column_prepare_host_export_materializes_supported_child_projections() {
    let frame = read_struct_dataframe(
        "{\"point\":{\"x\":1,\"name\":\"one\",\"coords\":[1,2]}}\n\
         {\"point\":{\"x\":2,\"name\":\"two\",\"coords\":[]}}\n\
         {\"point\":{\"x\":3,\"name\":\"three\",\"coords\":[3]}}\n",
    );
    let series = frame.series("point").unwrap();

    let export = series.try_prepare_host_column_export().unwrap();
    let children = match export.column().layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::Struct);
    assert_eq!(export.column().len(), 3);
    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));

    let x_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::Int64)
        })
        .expect("expected fixed-width struct child");
    assert_eq!(x_child.len(), 3);
    assert_eq!(x_child.offset(), 0);
    let x_values = match x_child.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width child layout, got {layout:?}"),
    };
    let x_values_binding = export.bindings().iter().find(|binding| binding.id() == x_values).unwrap();
    assert_eq!(binding_bytes(x_values_binding), expected_i64_bytes(&[1, 2, 3]));

    let name_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::String)
        })
        .expect("expected string struct child");
    assert_eq!(name_child.len(), 3);
    assert_eq!(name_child.offset(), 0);
    let (name_offsets, name_data) = match name_child.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like child layout, got {layout:?}"),
    };
    let name_offsets_binding = export.bindings().iter().find(|binding| binding.id() == name_offsets).unwrap();
    assert_eq!(binding_bytes(name_offsets_binding), expected_i32_bytes(&[0, 3, 6, 11]));
    let name_data_binding = export.bindings().iter().find(|binding| binding.id() == name_data).unwrap();
    assert_eq!(binding_bytes(name_data_binding), b"onetwothree".to_vec());

    let coords_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && matches!(column.layout(), ColumnLayout::List { .. }))
        })
        .expect("expected list struct child");
    let (coords_offsets, coords_values) = match coords_child.layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list child layout, got {layout:?}"),
    };
    let coords_offsets_binding = export.bindings().iter().find(|binding| binding.id() == coords_offsets).unwrap();
    assert_eq!(binding_bytes(coords_offsets_binding), expected_i32_bytes(&[0, 2, 2, 3]));
    let coords_values_column =
        export.semantic_columns().iter().find(|column| column.id() == coords_values).expect("expected list leaf child");
    let coords_values_binding = match coords_values_column.layout() {
        ColumnLayout::FixedWidth { values } => {
            export.bindings().iter().find(|binding| binding.id() == *values).unwrap()
        },
        layout => panic!("expected fixed-width list leaf, got {layout:?}"),
    };
    assert_eq!(binding_bytes(coords_values_binding), expected_i64_bytes(&[1, 2, 3]));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_struct_column_prepare_host_export_resets_root_and_child_offsets_to_active_range() {
    let frame = read_struct_dataframe(
        "{\"point\":{\"x\":1,\"name\":\"one\",\"coords\":[1,2]}}\n\
         {\"point\":{\"x\":2,\"name\":\"two\",\"coords\":[]}}\n\
         {\"point\":{\"x\":3,\"name\":\"three\",\"coords\":[3]}}\n",
    );
    let series = frame.series("point").unwrap();
    let sliced = series.as_column().slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_host_column_export().unwrap();
    let children = match export.column().layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::Struct);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 0);

    let x_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::Int64)
        })
        .expect("expected fixed-width struct child");
    assert_eq!(x_child.len(), 2);
    assert_eq!(x_child.offset(), 0);
    let x_values = match x_child.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width child layout, got {layout:?}"),
    };
    let x_values_binding = export.bindings().iter().find(|binding| binding.id() == x_values).unwrap();
    assert_eq!(binding_bytes(x_values_binding), expected_i64_bytes(&[2, 3]));

    let name_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::String)
        })
        .expect("expected string struct child");
    assert_eq!(name_child.len(), 2);
    assert_eq!(name_child.offset(), 0);
    let (name_offsets, name_data) = match name_child.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like child layout, got {layout:?}"),
    };
    let name_offsets_binding = export.bindings().iter().find(|binding| binding.id() == name_offsets).unwrap();
    assert_eq!(binding_bytes(name_offsets_binding), expected_i32_bytes(&[0, 3, 8]));
    let name_data_binding = export.bindings().iter().find(|binding| binding.id() == name_data).unwrap();
    assert_eq!(binding_bytes(name_data_binding), b"twothree".to_vec());

    let coords_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && matches!(column.layout(), ColumnLayout::List { .. }))
        })
        .expect("expected list struct child");
    assert_eq!(coords_child.len(), 2);
    assert_eq!(coords_child.offset(), 0);
    let (coords_offsets, coords_values) = match coords_child.layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list child layout, got {layout:?}"),
    };
    let coords_offsets_binding = export.bindings().iter().find(|binding| binding.id() == coords_offsets).unwrap();
    assert_eq!(binding_bytes(coords_offsets_binding), expected_i32_bytes(&[0, 0, 1]));
    let coords_values_column =
        export.semantic_columns().iter().find(|column| column.id() == coords_values).expect("expected list leaf child");
    assert_eq!(coords_values_column.len(), 1);
    assert_eq!(coords_values_column.offset(), 0);
    let coords_values_binding = match coords_values_column.layout() {
        ColumnLayout::FixedWidth { values } => {
            export.bindings().iter().find(|binding| binding.id() == *values).unwrap()
        },
        layout => panic!("expected fixed-width list leaf, got {layout:?}"),
    };
    assert_eq!(binding_bytes(coords_values_binding), expected_i64_bytes(&[3]));
}

#[test]
fn dictionary_column_prepare_host_export_materializes_indices_and_string_keys() {
    let runtime = Runtime::builder().build().unwrap();
    let base = runtime.string_series("category", &["gamma", "alpha", "gamma", "beta"]).unwrap();
    let dict = dictionary::encode(&base.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let export = dict.try_prepare_host_column_export().unwrap();
    let (indices, keys) = match export.column().layout() {
        ColumnLayout::Dictionary { indices, keys } => (*indices, *keys),
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::Dictionary32);
    assert_eq!(export.column().len(), 4);
    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));

    let indices_binding = export.bindings().iter().find(|binding| binding.id() == indices).unwrap();
    assert_eq!(binding_bytes(indices_binding), expected_i32_bytes(&[2, 0, 2, 1]));

    let keys_column =
        export.semantic_columns().iter().find(|column| column.id() == keys).expect("expected dictionary keys column");
    assert_eq!(keys_column.data_type().id(), InteropTypeId::String);
    assert_eq!(keys_column.len(), 3);
    assert_eq!(keys_column.offset(), 0);
    let (key_offsets, key_data) = match keys_column.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like dictionary keys, got {layout:?}"),
    };
    let key_offsets_binding = export.bindings().iter().find(|binding| binding.id() == key_offsets).unwrap();
    assert_eq!(binding_bytes(key_offsets_binding), expected_i32_bytes(&[0, 5, 9, 14]));
    let key_data_binding = export.bindings().iter().find(|binding| binding.id() == key_data).unwrap();
    assert_eq!(binding_bytes(key_data_binding), b"alphabetagamma".to_vec());

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_dictionary_column_prepare_host_export_resets_root_offset_and_materializes_active_indices_only() {
    let runtime = Runtime::builder().build().unwrap();
    let base = runtime.string_series("category", &["gamma", "alpha", "gamma", "beta"]).unwrap();
    let dict = dictionary::encode(&base.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let sliced = dict.slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_host_column_export().unwrap();
    let (indices, keys) = match export.column().layout() {
        ColumnLayout::Dictionary { indices, keys } => (*indices, *keys),
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::Dictionary32);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 0);

    let indices_binding = export.bindings().iter().find(|binding| binding.id() == indices).unwrap();
    assert_eq!(binding_bytes(indices_binding), expected_i32_bytes(&[0, 2]));

    let keys_column =
        export.semantic_columns().iter().find(|column| column.id() == keys).expect("expected dictionary keys column");
    assert_eq!(keys_column.data_type().id(), InteropTypeId::String);
    assert_eq!(keys_column.len(), 3);
    assert_eq!(keys_column.offset(), 0);
    let (key_offsets, key_data) = match keys_column.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like dictionary keys, got {layout:?}"),
    };
    let key_offsets_binding = export.bindings().iter().find(|binding| binding.id() == key_offsets).unwrap();
    assert_eq!(binding_bytes(key_offsets_binding), expected_i32_bytes(&[0, 5, 9, 14]));
    let key_data_binding = export.bindings().iter().find(|binding| binding.id() == key_data).unwrap();
    assert_eq!(binding_bytes(key_data_binding), b"alphabetagamma".to_vec());
}

#[test]
fn list_of_lists_prepare_host_export_materializes_recursive_child_projection() {
    let frame = read_list_dataframe("{\"lists\":[[1,2],[3]]}\n{\"lists\":[[],[4,5]]}\n{\"lists\":[[6]]}\n");
    let series = frame.series("lists").unwrap();

    let export = series.try_prepare_host_column_export().unwrap();

    let (root_offsets, nested_list_id) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    let root_offsets_binding = export.bindings().iter().find(|binding| binding.id() == root_offsets).unwrap();
    assert_eq!(root_offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(root_offsets_binding), expected_i32_bytes(&[0, 2, 4, 5]));

    let nested_list = export
        .semantic_columns()
        .iter()
        .find(|column| column.id() == nested_list_id)
        .expect("expected nested list child");
    assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
    assert_eq!(nested_list.len(), 5);
    assert_eq!(nested_list.offset(), 0);

    let (nested_offsets, leaf_id) = match nested_list.layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let nested_offsets_binding = export.bindings().iter().find(|binding| binding.id() == nested_offsets).unwrap();
    assert_eq!(nested_offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(nested_offsets_binding), expected_i32_bytes(&[0, 2, 3, 3, 5, 6]));

    let leaf =
        export.semantic_columns().iter().find(|column| column.id() == leaf_id).expect("expected nested list leaf");
    assert_eq!(leaf.data_type().id(), InteropTypeId::Int64);
    assert_eq!(leaf.len(), 6);
    assert_eq!(leaf.offset(), 0);
    let values = match leaf.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width leaf layout, got {layout:?}"),
    };
    let values_binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(values_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(values_binding), expected_i64_bytes(&[1, 2, 3, 4, 5, 6]));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn host_export_accepts_broader_recursive_nested_columns() {
    let frame = read_json_dataframe(
        "host_recursive_nested.jsonl",
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
    let frame = DataFrame::from_series([category, nested, point, id]).unwrap();

    let export = frame.try_prepare_host_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(
        export.table().names().unwrap(),
        &["category".to_string(), "nested".to_string(), "point".to_string(), "id".to_string()]
    );
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::Dictionary { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::List { .. }));
    assert!(matches!(export.columns()[2].layout(), ColumnLayout::Struct { .. }));
    assert!(matches!(export.columns()[3].layout(), ColumnLayout::FixedWidth { .. }));

    let nested_child = match export.columns()[1].layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected list layout, got {layout:?}"),
    };
    let nested_list = export.semantic_columns().iter().find(|column| column.id() == nested_child).unwrap();
    let nested_leaf = match nested_list.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let nested_leaf_column = export.semantic_columns().iter().find(|column| column.id() == nested_leaf).unwrap();
    assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
    assert_eq!(nested_leaf_column.data_type().id(), InteropTypeId::Int64);

    let struct_children = match export.columns()[2].layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };
    let struct_meta_child = struct_children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::Struct)
        })
        .expect("expected nested struct child");
    let meta_children = match struct_meta_child.layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected nested struct layout, got {layout:?}"),
    };
    assert!(meta_children.iter().any(|id| {
        export
            .semantic_columns()
            .iter()
            .any(|column| column.id() == *id && column.data_type().id() == InteropTypeId::Int64)
    }));
    assert!(meta_children.iter().any(|id| {
        export
            .semantic_columns()
            .iter()
            .any(|column| column.id() == *id && column.data_type().id() == InteropTypeId::String)
    }));

    let struct_list_child = struct_children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::List)
        })
        .expect("expected struct child with list layout");
    let struct_nested_list = match struct_list_child.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let struct_nested_list_column = export
        .semantic_columns()
        .iter()
        .find(|column| column.id() == struct_nested_list)
        .expect("expected nested list child");
    let struct_leaf = match struct_nested_list_column.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let struct_leaf_column = export.semantic_columns().iter().find(|column| column.id() == struct_leaf).unwrap();
    assert_eq!(struct_nested_list_column.data_type().id(), InteropTypeId::List);
    assert_eq!(struct_leaf_column.data_type().id(), InteropTypeId::Int64);

    let dictionary_keys = match export.columns()[0].layout() {
        ColumnLayout::Dictionary { keys, .. } => *keys,
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };
    let dictionary_keys_column = export
        .semantic_columns()
        .iter()
        .find(|column| column.id() == dictionary_keys)
        .expect("expected dictionary keys");
    assert_eq!(dictionary_keys_column.data_type().id(), InteropTypeId::String);
    assert!(matches!(dictionary_keys_column.layout(), ColumnLayout::BinaryLike { .. }));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn string_column_prepare_host_export_materializes_binary_like_bytes() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_strings_with_runtime(&runtime, &["alice", "bob"]).unwrap();
    let view = column.view().unwrap();

    let export = view.try_prepare_host_column_export().unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::String);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);

    let (offsets, data) = match export.column().layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like layout, got {layout:?}"),
    };

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(offsets_binding), expected_i32_bytes(&[0, 5, 8]));

    let data_binding = export.bindings().iter().find(|binding| binding.id() == data).unwrap();
    assert_eq!(data_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(data_binding), b"alicebob".to_vec());

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_string_column_prepare_host_export_resets_offset_and_materializes_active_range_only() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_strings_with_runtime(&runtime, &["zero", "one", "two", "three"]).unwrap();
    let sliced = column.slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_host_column_export().unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::String);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 0);

    let (offsets, data) = match export.column().layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like layout, got {layout:?}"),
    };

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(offsets_binding), expected_i32_bytes(&[0, 3, 6]));

    let data_binding = export.bindings().iter().find(|binding| binding.id() == data).unwrap();
    assert_eq!(data_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(data_binding), b"onetwo".to_vec());
}

#[test]
fn nullable_string_column_prepare_host_export_preserves_packed_validity_mask() {
    let runtime = Runtime::builder().build().unwrap();
    let series = runtime
        .nullable_string_series("name", &["alice", "placeholder", "bob", "unused"], &[true, false, true, false])
        .unwrap();

    let export = series.try_prepare_host_column_export().unwrap();

    let mask = match export.column().validity() {
        &PhysicalValidity::ValidityMask { mask, null_count } => {
            assert_eq!(null_count, Some(2));
            mask
        },
        PhysicalValidity::NoValidityBuffer => panic!("expected validity mask"),
    };

    let binding = export.bindings().iter().find(|binding| binding.id() == mask).unwrap();
    assert_eq!(binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(binding), vec![0b0000_0101]);
}

#[test]
fn dataframe_prepare_host_export_accepts_string_columns() {
    let runtime = Runtime::builder().build().unwrap();
    let names = runtime.string_series("name", &["alice", "bob", "cara"]).unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let frame = DataFrame::from_series([names, ids]).unwrap();

    let export = frame.try_prepare_host_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(export.table().names().unwrap(), &["name".to_string(), "id".to_string()]);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::BinaryLike { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn list_column_prepare_host_export_materializes_offsets_and_fixed_width_child_bytes() {
    let frame = read_list_dataframe("{\"lists\":[1,2]}\n{\"lists\":[3]}\n{\"lists\":[]}\n{\"lists\":[4,5,6]}\n");
    let series = frame.series("lists").unwrap();

    let export = series.try_prepare_host_column_export().unwrap();

    let (offsets, child) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::List);
    assert_eq!(export.column().len(), 4);
    assert_eq!(export.column().offset(), 0);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(offsets_binding), expected_i32_bytes(&[0, 2, 3, 3, 6]));

    let child_column = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child_column.data_type().id(), InteropTypeId::Int64);
    assert_eq!(child_column.len(), 6);
    assert_eq!(child_column.offset(), 0);

    let values = match child_column.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width child layout, got {layout:?}"),
    };
    let values_binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(values_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(values_binding), expected_i64_bytes(&[1, 2, 3, 4, 5, 6]));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_list_column_prepare_host_export_resets_offsets_and_materializes_active_child_range_only() {
    let frame = read_list_dataframe("{\"lists\":[1,2]}\n{\"lists\":[3]}\n{\"lists\":[]}\n{\"lists\":[4,5,6]}\n");
    let series = frame.series("lists").unwrap();
    let sliced = series.as_column().slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_host_column_export().unwrap();

    let (offsets, child) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::List);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 0);

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(offsets_binding), expected_i32_bytes(&[0, 1, 1]));

    let child_column = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child_column.data_type().id(), InteropTypeId::Int64);
    assert_eq!(child_column.len(), 1);
    assert_eq!(child_column.offset(), 0);

    let values = match child_column.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => panic!("expected fixed-width child layout, got {layout:?}"),
    };
    let values_binding = export.bindings().iter().find(|binding| binding.id() == values).unwrap();
    assert_eq!(values_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(values_binding), expected_i64_bytes(&[3]));
}

#[test]
fn list_of_strings_prepare_host_export_materializes_binary_like_child_bytes() {
    let frame = read_list_dataframe("{\"lists\":[\"alice\",\"bob\"]}\n{\"lists\":[\"cara\"]}\n");
    let series = frame.series("lists").unwrap();

    let export = series.try_prepare_host_column_export().unwrap();

    let (offsets, child) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(offsets_binding), expected_i32_bytes(&[0, 2, 3]));

    let child_column = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child_column.data_type().id(), InteropTypeId::String);
    assert_eq!(child_column.len(), 3);
    assert_eq!(child_column.offset(), 0);

    let (child_offsets, child_data) = match child_column.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like child layout, got {layout:?}"),
    };
    let child_offsets_binding = export.bindings().iter().find(|binding| binding.id() == child_offsets).unwrap();
    assert_eq!(child_offsets_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(child_offsets_binding), expected_i32_bytes(&[0, 5, 8, 12]));

    let child_data_binding = export.bindings().iter().find(|binding| binding.id() == child_data).unwrap();
    assert_eq!(child_data_binding.placement(), &Placement::Host);
    assert_eq!(binding_bytes(child_data_binding), b"alicebobcara".to_vec());
}

#[test]
fn dataframe_prepare_host_export_accepts_list_columns_with_supported_child_projection() {
    let frame = read_json_dataframe(
        "host_lists.jsonl",
        "{\"lists\":[1,2],\"id\":10}\n{\"lists\":[3],\"id\":20}\n{\"lists\":[],\"id\":30}\n",
    );

    let export = frame.try_prepare_host_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(export.table().names().unwrap(), &["lists".to_string(), "id".to_string()]);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::List { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn dataframe_prepare_host_export_accepts_struct_and_dictionary_columns_with_supported_projections() {
    let frame = read_json_dataframe(
        "host_struct_dictionary.jsonl",
        "{\"point\":{\"x\":1,\"name\":\"one\",\"coords\":[1,2]},\"category\":\"gamma\",\"id\":10}\n\
         {\"point\":{\"x\":2,\"name\":\"two\",\"coords\":[]},\"category\":\"alpha\",\"id\":20}\n\
         {\"point\":{\"x\":3,\"name\":\"three\",\"coords\":[3]},\"category\":\"gamma\",\"id\":30}\n\
         {\"point\":{\"x\":4,\"name\":\"four\",\"coords\":[4,5]},\"category\":\"beta\",\"id\":40}\n",
    );

    let point = frame.series("point").unwrap();
    let id = frame.series("id").unwrap();
    let category = frame.series("category").unwrap();
    let encoded = dictionary::encode(&category.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let category = cudf::Series::from_column(encoded).unwrap().rename("category");
    let frame = DataFrame::from_series([point, category, id]).unwrap();

    let export = frame.try_prepare_host_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 4);
    assert_eq!(export.table().names().unwrap(), &["point".to_string(), "category".to_string(), "id".to_string()]);
    assert_eq!(export.diagnostics().profile(), ExportProfile::PreparedHostExport);
    assert_eq!(export.readiness(), &ProducerReadiness::HostVisible);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::Struct { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::Dictionary { .. }));
    assert!(matches!(export.columns()[2].layout(), ColumnLayout::FixedWidth { .. }));
    assert!(export.bindings().iter().all(|binding| binding.placement() == &Placement::Host));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn series_and_dataframe_prepare_device_export_match_underlying_low_level_exports() {
    let runtime = Runtime::builder().build().unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let vals = runtime.series("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids.clone(), vals.clone()]).unwrap();

    let series_export = ids.try_prepare_device_column_export().unwrap();
    let column_export = ids.as_column().try_prepare_device_column_export().unwrap();
    assert_eq!(series_export, column_export);

    let frame_export = frame.try_prepare_device_table_export().unwrap();
    let table_view = frame.table_view().unwrap();
    let table_export = table_view.table().try_prepare_device_table_export().unwrap();

    assert_eq!(frame_export.table().num_rows(), table_export.table().num_rows());
    assert_eq!(frame_export.columns(), table_export.columns());
    assert_eq!(frame_export.bindings(), table_export.bindings());
    assert_eq!(frame_export.readiness(), table_export.readiness());
}

#[test]
fn ambient_column_prepare_device_export_is_rejected() {
    let column = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let err = column.view().unwrap().try_prepare_device_column_export().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
    assert!(
        err.context()
            .iter()
            .any(|(key, value)| *key == "operation" && value == "interop::ColumnView::try_prepare_device_column_export")
    );
}

#[test]
fn ambient_dataframe_prepare_device_export_is_rejected() {
    let ids = cudf::Series::new("id", &[1i32, 2, 3]).unwrap();
    let vals = cudf::Series::new("value", &[10i64, 20, 30]).unwrap();
    let frame = DataFrame::from_series([ids, vals]).unwrap();

    let err = frame.try_prepare_device_table_export().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::AmbientInteropUnsupported);
    assert!(
        err.context()
            .iter()
            .any(|(key, value)| *key == "operation" && value == "interop::DataFrame::try_prepare_device_table_export")
    );
}

#[test]
fn string_column_prepare_device_export_preserves_binary_like_metadata() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_strings_with_runtime(&runtime, &["alice", "bob"]).unwrap();
    let view = column.view().unwrap();
    let offsets_view = view.child(0).unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();

    let export = view.try_prepare_device_column_export().unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::String);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 0);

    let (offsets, data) = match export.column().layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like layout, got {layout:?}"),
    };

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.bytes().base_ptr(), offsets_view.device_data_ptr().unwrap());
    assert_eq!(offsets_binding.placement(), &Placement::CudaDevice { device: runtime.device().ordinal() as i32 });
    assert!(offsets_binding.bytes().byte_len() >= 3 * std::mem::size_of::<i32>());

    let data_binding = export.bindings().iter().find(|binding| binding.id() == data).unwrap();
    assert_eq!(data_binding.placement(), &Placement::CudaDevice { device: runtime.device().ordinal() as i32 });
    assert!(data_binding.bytes().byte_len() >= "alicebob".len());

    match export.readiness() {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream);
            assert_eq!(stream.device(), runtime.device().ordinal() as i32);
        },
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
    }
}

#[test]
fn sliced_string_column_prepare_device_export_preserves_active_binary_like_span() {
    let runtime = Runtime::builder().build().unwrap();
    let column = Column::from_strings_with_runtime(&runtime, &["zero", "one", "two", "three"]).unwrap();
    let base = column.view().unwrap();
    let base_export = base.try_prepare_device_column_export().unwrap();
    let sliced = column.slice_view(1, 2).unwrap();
    let offsets_view = sliced.child(0).unwrap();
    let expected_stream = column.__debug_execution_domain_stream_value();

    let export = sliced.try_prepare_device_column_export().unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::String);
    assert_eq!(export.column().len(), 2);
    assert_eq!(export.column().offset(), 1);

    let (offsets, data) = match export.column().layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like layout, got {layout:?}"),
    };
    let (_, base_data) = match base_export.column().layout() {
        ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
        layout => panic!("expected binary-like layout, got {layout:?}"),
    };
    let base_data_binding = base_export.bindings().iter().find(|binding| binding.id() == *base_data).unwrap();

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.bytes().base_ptr(), offsets_view.device_data_ptr().unwrap());
    assert_eq!(offsets_binding.placement(), &Placement::CudaDevice { device: runtime.device().ordinal() as i32 });
    assert_eq!(offsets_binding.bytes().byte_len(), 4 * std::mem::size_of::<i32>());

    let data_binding = export.bindings().iter().find(|binding| binding.id() == data).unwrap();
    assert_eq!(data_binding.bytes().base_ptr(), base_data_binding.bytes().base_ptr());
    assert_eq!(data_binding.placement(), &Placement::CudaDevice { device: runtime.device().ordinal() as i32 });
    assert_eq!(data_binding.bytes().byte_len(), "zeroonetwo".len());

    match export.readiness() {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            assert_eq!(stream.as_raw() as usize, expected_stream);
            assert_eq!(stream.device(), runtime.device().ordinal() as i32);
        },
        readiness => panic!("expected device-stream readiness, got {readiness:?}"),
    }

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn dataframe_prepare_device_export_accepts_string_columns() {
    let runtime = Runtime::builder().build().unwrap();
    let names = runtime.string_series("name", &["alice", "bob", "cara"]).unwrap();
    let ids = runtime.series("id", &[1i32, 2, 3]).unwrap();
    let frame = DataFrame::from_series([names, ids]).unwrap();

    let export = frame.try_prepare_device_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(export.table().names().unwrap(), &["name".to_string(), "id".to_string()]);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::BinaryLike { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn struct_column_prepare_device_export_preserves_child_metadata() {
    let frame = read_struct_dataframe(
        "{\"point\":{\"x\":1,\"name\":\"one\"}}\n{\"point\":{\"x\":2,\"name\":\"two\"}}\n{\"point\":{\"x\":3,\"name\":\"three\"}}\n",
    );
    let series = frame.series("point").unwrap();
    let view = series.as_column_view().unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let children = match export.column().layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };
    assert_eq!(export.column().data_type().id(), InteropTypeId::Struct);
    assert_eq!(export.column().len(), view.len().unwrap());
    assert_eq!(export.column().offset(), 0);
    assert_eq!(children.len(), 2);

    let fixed_view = view.child(0).unwrap();
    let string_view = view.child(1).unwrap();
    let fixed_child = export.semantic_columns().iter().find(|column| column.id() == children[0]).unwrap();
    let string_child = export.semantic_columns().iter().find(|column| column.id() == children[1]).unwrap();

    assert!(matches!(fixed_child.layout(), ColumnLayout::FixedWidth { .. }));
    assert_eq!(fixed_child.len(), fixed_view.len().unwrap());
    assert_eq!(fixed_child.offset(), usize::try_from(fixed_view.offset().unwrap()).unwrap());

    assert!(matches!(string_child.layout(), ColumnLayout::BinaryLike { .. }));
    assert_eq!(string_child.data_type().id(), InteropTypeId::String);
    assert_eq!(string_child.len(), string_view.len().unwrap());
    assert_eq!(string_child.offset(), usize::try_from(string_view.offset().unwrap()).unwrap());

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_struct_column_prepare_device_export_preserves_child_offsets() {
    let frame = read_struct_dataframe(
        "{\"point\":{\"x\":1,\"name\":\"one\"}}\n{\"point\":{\"x\":2,\"name\":\"two\"}}\n{\"point\":{\"x\":3,\"name\":\"three\"}}\n{\"point\":{\"x\":4,\"name\":\"four\"}}\n",
    );
    let series = frame.series("point").unwrap();
    let sliced = series.as_column().slice_view(1, 2).unwrap();
    let export = sliced.try_prepare_device_column_export().unwrap();

    let children = match export.column().layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };
    let fixed_view = sliced.child(0).unwrap();
    let string_view = sliced.child(1).unwrap();
    let fixed_child = export.semantic_columns().iter().find(|column| column.id() == children[0]).unwrap();
    let string_child = export.semantic_columns().iter().find(|column| column.id() == children[1]).unwrap();

    assert_eq!(export.column().data_type().id(), InteropTypeId::Struct);
    assert_eq!(export.column().offset(), usize::try_from(sliced.offset().unwrap()).unwrap());
    assert_eq!(export.column().len(), usize::try_from(sliced.len().unwrap()).unwrap());

    assert!(matches!(fixed_child.layout(), ColumnLayout::FixedWidth { .. }));
    assert_eq!(fixed_child.offset(), usize::try_from(fixed_view.offset().unwrap()).unwrap());
    assert_eq!(fixed_child.len(), fixed_view.len().unwrap());

    assert!(matches!(string_child.layout(), ColumnLayout::BinaryLike { .. }));
    assert_eq!(string_child.offset(), usize::try_from(string_view.offset().unwrap()).unwrap());
    assert_eq!(string_child.len(), string_view.len().unwrap());
}

#[test]
fn dataframe_prepare_device_export_accepts_struct_columns() {
    let frame = read_struct_dataframe(
        "{\"point\":{\"x\":1,\"name\":\"one\"},\"id\":10}\n{\"point\":{\"x\":2,\"name\":\"two\"},\"id\":20}\n",
    );

    let export = frame.try_prepare_device_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 2);
    assert_eq!(export.table().names().unwrap(), &["point".to_string(), "id".to_string()]);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::Struct { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn list_column_prepare_device_export_preserves_offsets_and_child_metadata() {
    let frame = read_list_dataframe("{\"lists\":[1,2]}\n{\"lists\":[3]}\n{\"lists\":[]}\n{\"lists\":[4,5,6]}\n");
    let series = frame.series("lists").unwrap();
    let view = series.as_column_view().unwrap();
    let offsets_view = view.child(0).unwrap();
    let child_view = view.child(1).unwrap();

    let export = series.try_prepare_device_column_export().unwrap();
    let (offsets, child) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::List);
    assert_eq!(export.column().len(), view.len().unwrap());
    assert_eq!(export.column().offset(), 0);

    let child_column = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child_column.len(), child_view.len().unwrap());
    assert_eq!(child_column.offset(), usize::try_from(child_view.offset().unwrap()).unwrap());
    assert!(matches!(child_column.layout(), ColumnLayout::FixedWidth { .. }));

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.bytes().base_ptr(), offsets_view.device_data_ptr().unwrap());
    assert_eq!(offsets_binding.bytes().byte_len(), 5 * std::mem::size_of::<i32>());
    assert!(matches!(offsets_binding.placement(), Placement::CudaDevice { .. }));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_list_column_prepare_device_export_preserves_root_offset_and_child_slice() {
    let frame = read_list_dataframe("{\"lists\":[1,2]}\n{\"lists\":[3]}\n{\"lists\":[]}\n{\"lists\":[4,5,6]}\n");
    let series = frame.series("lists").unwrap();
    let base = series.as_column().view().unwrap();
    let sliced = series.as_column().slice_view(1, 2).unwrap();
    let offsets_view = base.child(0).unwrap();
    let host_offsets = offsets_view.to_vec::<i32>().unwrap();

    let export = sliced.try_prepare_device_column_export().unwrap();
    let (offsets, child) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    let expected_root_offset = usize::try_from(sliced.offset().unwrap()).unwrap();
    let expected_root_len = usize::try_from(sliced.len().unwrap()).unwrap();
    let expected_child_offset = usize::try_from(host_offsets[expected_root_offset]).unwrap();
    let expected_child_end = usize::try_from(host_offsets[expected_root_offset + expected_root_len]).unwrap();

    assert_eq!(export.column().offset(), expected_root_offset);
    assert_eq!(export.column().len(), expected_root_len);

    let child_column = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child_column.offset(), expected_child_offset);
    assert_eq!(child_column.len(), expected_child_end - expected_child_offset);
    assert!(matches!(child_column.layout(), ColumnLayout::FixedWidth { .. }));

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.bytes().base_ptr(), offsets_view.device_data_ptr().unwrap());
    assert_eq!(offsets_binding.bytes().byte_len(), 4 * std::mem::size_of::<i32>());
}

#[test]
fn sliced_list_of_strings_prepare_device_export_preserves_binary_like_child_slice() {
    let frame = read_list_dataframe(
        "{\"lists\":[\"zero\",\"one\"]}\n{\"lists\":[\"two\"]}\n{\"lists\":[]}\n{\"lists\":[\"three\",\"four\"]}\n",
    );
    let series = frame.series("lists").unwrap();
    let base = series.as_column().view().unwrap();
    let base_offsets_view = base.child(0).unwrap();
    let host_root_offsets = base_offsets_view.to_vec::<i32>().unwrap();
    let base_child_view = base.child(1).unwrap();
    let base_child_export = base_child_view.try_prepare_device_column_export().unwrap();
    let child_offsets_view = base_child_view.child(0).unwrap();
    let host_child_offsets = child_offsets_view.to_vec::<i32>().unwrap();
    let sliced = series.as_column().slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_device_column_export().unwrap();
    let (list_offsets, child) = match export.column().layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        layout => panic!("expected list layout, got {layout:?}"),
    };

    let expected_root_offset = usize::try_from(sliced.offset().unwrap()).unwrap();
    let expected_root_len = usize::try_from(sliced.len().unwrap()).unwrap();
    let expected_child_offset = usize::try_from(host_root_offsets[expected_root_offset]).unwrap();
    let expected_child_end = usize::try_from(host_root_offsets[expected_root_offset + expected_root_len]).unwrap();

    assert_eq!(export.column().offset(), expected_root_offset);
    assert_eq!(export.column().len(), expected_root_len);

    let child_column = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    assert_eq!(child_column.data_type().id(), InteropTypeId::String);
    assert_eq!(child_column.offset(), expected_child_offset);
    assert_eq!(child_column.len(), expected_child_end - expected_child_offset);

    let (child_offsets, child_data) = match child_column.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like child layout, got {layout:?}"),
    };
    let (_, base_child_data) = match base_child_export.column().layout() {
        ColumnLayout::BinaryLike { offsets, data } => (offsets, data),
        layout => panic!("expected binary-like child layout, got {layout:?}"),
    };

    let list_offsets_binding = export.bindings().iter().find(|binding| binding.id() == list_offsets).unwrap();
    assert_eq!(list_offsets_binding.bytes().base_ptr(), base_offsets_view.device_data_ptr().unwrap());
    assert_eq!(list_offsets_binding.bytes().byte_len(), 4 * std::mem::size_of::<i32>());

    let child_offsets_binding = export.bindings().iter().find(|binding| binding.id() == child_offsets).unwrap();
    assert_eq!(child_offsets_binding.bytes().base_ptr(), child_offsets_view.device_data_ptr().unwrap());
    assert_eq!(child_offsets_binding.bytes().byte_len(), (expected_child_end + 1) * std::mem::size_of::<i32>());

    let base_child_data_binding =
        base_child_export.bindings().iter().find(|binding| binding.id() == *base_child_data).unwrap();
    let child_data_binding = export.bindings().iter().find(|binding| binding.id() == child_data).unwrap();
    assert_eq!(child_data_binding.bytes().base_ptr(), base_child_data_binding.bytes().base_ptr());
    assert_eq!(child_data_binding.bytes().byte_len(), usize::try_from(host_child_offsets[expected_child_end]).unwrap());

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn dictionary_column_prepare_device_export_preserves_indices_and_keys_metadata() {
    let runtime = Runtime::builder().build().unwrap();
    let base = Column::from_slice_with_runtime(&runtime, &[3i32, 1, 3, 2]).unwrap();
    let dict = dictionary::encode(&base.view().unwrap(), TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    let indices_view = dictionary::indices(&dict_view).unwrap();
    let keys_view = dictionary::keys(&dict_view).unwrap();

    let export = dict.try_prepare_device_column_export().unwrap();
    let (indices, keys) = match export.column().layout() {
        ColumnLayout::Dictionary { indices, keys } => (*indices, *keys),
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };

    assert_eq!(export.column().data_type().id(), InteropTypeId::Dictionary32);
    assert_eq!(export.column().len(), dict_view.len().unwrap());
    assert_eq!(export.column().offset(), usize::try_from(dict_view.offset().unwrap()).unwrap());

    let keys_column = export.semantic_columns().iter().find(|column| column.id() == keys).unwrap();
    assert_eq!(keys_column.data_type().id(), InteropTypeId::Int32);
    assert_eq!(keys_column.len(), keys_view.len().unwrap());
    assert_eq!(keys_column.offset(), usize::try_from(keys_view.offset().unwrap()).unwrap());
    assert!(matches!(keys_column.layout(), ColumnLayout::FixedWidth { .. }));

    let indices_binding = export.bindings().iter().find(|binding| binding.id() == indices).unwrap();
    assert_eq!(indices_binding.bytes().base_ptr(), indices_view.device_data_ptr().unwrap());
    assert_eq!(indices_binding.bytes().byte_len(), dict_view.len().unwrap() * std::mem::size_of::<i32>());
    assert!(matches!(indices_binding.placement(), Placement::CudaDevice { .. }));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn dictionary_string_column_prepare_device_export_preserves_binary_like_keys() {
    let runtime = Runtime::builder().build().unwrap();
    let base = runtime.string_series("name", &["gamma", "alpha", "gamma", "beta"]).unwrap();
    let dict = dictionary::encode(&base.as_column().view().unwrap(), TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    let keys_view = dictionary::keys(&dict_view).unwrap();
    let keys_offsets_view = keys_view.child(0).unwrap();

    let export = dict.try_prepare_device_column_export().unwrap();
    let (_, keys) = match export.column().layout() {
        ColumnLayout::Dictionary { indices, keys } => (indices, *keys),
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };

    let keys_column = export.semantic_columns().iter().find(|column| column.id() == keys).unwrap();
    let (offsets, data) = match keys_column.layout() {
        ColumnLayout::BinaryLike { offsets, data } => (*offsets, *data),
        layout => panic!("expected binary-like dictionary keys, got {layout:?}"),
    };

    assert_eq!(keys_column.data_type().id(), InteropTypeId::String);
    assert_eq!(keys_column.len(), keys_view.len().unwrap());
    assert_eq!(keys_column.offset(), usize::try_from(keys_view.offset().unwrap()).unwrap());

    let offsets_binding = export.bindings().iter().find(|binding| binding.id() == offsets).unwrap();
    assert_eq!(offsets_binding.bytes().base_ptr(), keys_offsets_view.device_data_ptr().unwrap());
    assert_eq!(offsets_binding.bytes().byte_len(), (keys_view.len().unwrap() + 1) * std::mem::size_of::<i32>());

    let data_binding = export.bindings().iter().find(|binding| binding.id() == data).unwrap();
    assert!(matches!(data_binding.placement(), Placement::CudaDevice { .. }));
}

#[test]
fn dataframe_prepare_device_export_accepts_dictionary_columns() {
    let runtime = Runtime::builder().build().unwrap();
    let base = runtime.string_series("category", &["gamma", "alpha", "gamma", "beta"]).unwrap();
    let ids = runtime.series("id", &[10i32, 20, 30, 40]).unwrap();
    let dict = dictionary::encode(&base.as_column().view().unwrap(), TypeId::Int32).unwrap();
    let category = cudf::Series::from_column(dict).unwrap().rename("category");
    let frame = DataFrame::from_series([category, ids]).unwrap();

    let export = frame.try_prepare_device_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 4);
    assert_eq!(export.table().names().unwrap(), &["category".to_string(), "id".to_string()]);
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::Dictionary { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::FixedWidth { .. }));

    let keys = match export.columns()[0].layout() {
        ColumnLayout::Dictionary { keys, .. } => *keys,
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };
    let keys_column = export.semantic_columns().iter().find(|column| column.id() == keys).unwrap();
    assert!(matches!(keys_column.layout(), ColumnLayout::BinaryLike { .. }));
    assert_eq!(keys_column.data_type().id(), InteropTypeId::String);

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn sliced_dictionary_column_prepare_device_export_preserves_root_offset_and_indices_slice() {
    let runtime = Runtime::builder().build().unwrap();
    let base = Column::from_slice_with_runtime(&runtime, &[3i32, 1, 3, 2]).unwrap();
    let dict = dictionary::encode(&base.view().unwrap(), TypeId::Int32).unwrap();
    let dict_view = dict.view().unwrap();
    let keys_view = dictionary::keys(&dict_view).unwrap();
    let indices_view = dictionary::indices(&dict_view).unwrap();
    let sliced = dict.slice_view(1, 2).unwrap();

    let export = sliced.try_prepare_device_column_export().unwrap();
    let expected_root_offset = usize::try_from(sliced.offset().unwrap()).unwrap();
    let expected_root_len = usize::try_from(sliced.len().unwrap()).unwrap();
    let (indices, keys) = match export.column().layout() {
        ColumnLayout::Dictionary { indices, keys } => (*indices, *keys),
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };

    assert_eq!(export.column().offset(), expected_root_offset);
    assert_eq!(export.column().len(), expected_root_len);

    let keys_column = export.semantic_columns().iter().find(|column| column.id() == keys).unwrap();
    assert_eq!(keys_column.len(), keys_view.len().unwrap());
    assert_eq!(keys_column.offset(), usize::try_from(keys_view.offset().unwrap()).unwrap());

    let indices_binding = export.bindings().iter().find(|binding| binding.id() == indices).unwrap();
    assert_eq!(indices_binding.bytes().base_ptr(), indices_view.device_data_ptr().unwrap());
    assert_eq!(
        indices_binding.bytes().byte_len(),
        (expected_root_offset + expected_root_len) * std::mem::size_of::<i32>(),
    );
}

#[test]
fn list_of_lists_prepare_device_export_preserves_recursive_child_projection() {
    let frame = read_list_dataframe("{\"lists\":[[1,2],[3]]}\n{\"lists\":[[],[4,5]]}\n{\"lists\":[[6]]}\n");
    let series = frame.series("lists").unwrap();
    let view = series.as_column_view().unwrap();
    let nested_list_view = view.child(1).unwrap();

    let export = series.try_prepare_device_column_export().unwrap();
    let child = match export.column().layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected list layout, got {layout:?}"),
    };

    let nested_list = export.semantic_columns().iter().find(|column| column.id() == child).unwrap();
    let leaf = match nested_list.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let leaf_column = export.semantic_columns().iter().find(|column| column.id() == leaf).unwrap();

    assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
    assert_eq!(nested_list.len(), nested_list_view.len().unwrap());
    assert_eq!(nested_list.offset(), usize::try_from(nested_list_view.offset().unwrap()).unwrap());
    assert!(matches!(leaf_column.layout(), ColumnLayout::FixedWidth { .. }));
    assert_eq!(leaf_column.data_type().id(), InteropTypeId::Int64);
}

#[test]
fn struct_with_list_child_prepare_device_export_preserves_recursive_child_projection() {
    let frame = read_struct_dataframe(
        "{\"point\":{\"coords\":[1,2],\"label\":\"one\"}}\n{\"point\":{\"coords\":[],\"label\":\"two\"}}\n{\"point\":{\"coords\":[3],\"label\":\"three\"}}\n",
    );
    let series = frame.series("point").unwrap();

    let export = series.try_prepare_device_column_export().unwrap();
    let children = match export.column().layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };

    let list_child = children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::List)
        })
        .expect("expected struct child with list layout");
    let leaf = match list_child.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let leaf_column = export.semantic_columns().iter().find(|column| column.id() == leaf).unwrap();

    assert_eq!(list_child.data_type().id(), InteropTypeId::List);
    assert!(matches!(leaf_column.layout(), ColumnLayout::FixedWidth { .. }));
    assert_eq!(leaf_column.data_type().id(), InteropTypeId::Int64);
}

#[test]
fn dataframe_prepare_device_export_accepts_dictionary_and_recursive_nested_columns() {
    let frame = read_json_dataframe(
        "dictionary_nested.jsonl",
        "{\"category\":\"beta\",\"nested\":[[1,2],[3]],\"point\":{\"coords\":[1,2],\"label\":\"one\"},\"id\":10}\n\
         {\"category\":\"alpha\",\"nested\":[[],[4,5]],\"point\":{\"coords\":[],\"label\":\"two\"},\"id\":20}\n\
         {\"category\":\"beta\",\"nested\":[[6]],\"point\":{\"coords\":[3],\"label\":\"three\"},\"id\":30}\n",
    );

    let category = frame.series("category").unwrap();
    let nested = frame.series("nested").unwrap();
    let point = frame.series("point").unwrap();
    let id = frame.series("id").unwrap();
    let encoded = dictionary::encode(&category.as_column_view().unwrap(), TypeId::Int32).unwrap();
    let category = cudf::Series::from_column(encoded).unwrap().rename("category");
    let frame = DataFrame::from_series([category, nested, point, id]).unwrap();

    let export = frame.try_prepare_device_table_export().unwrap();

    assert_eq!(export.table().num_rows(), 3);
    assert_eq!(
        export.table().names().unwrap(),
        &["category".to_string(), "nested".to_string(), "point".to_string(), "id".to_string()]
    );
    assert!(matches!(export.columns()[0].layout(), ColumnLayout::Dictionary { .. }));
    assert!(matches!(export.columns()[1].layout(), ColumnLayout::List { .. }));
    assert!(matches!(export.columns()[2].layout(), ColumnLayout::Struct { .. }));
    assert!(matches!(export.columns()[3].layout(), ColumnLayout::FixedWidth { .. }));

    let nested_child = match export.columns()[1].layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected list layout, got {layout:?}"),
    };
    let nested_list = export.semantic_columns().iter().find(|column| column.id() == nested_child).unwrap();
    let nested_leaf = match nested_list.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let nested_leaf_column = export.semantic_columns().iter().find(|column| column.id() == nested_leaf).unwrap();
    assert_eq!(nested_list.data_type().id(), InteropTypeId::List);
    assert_eq!(nested_leaf_column.data_type().id(), InteropTypeId::Int64);

    let struct_children = match export.columns()[2].layout() {
        ColumnLayout::Struct { children } => children.clone(),
        layout => panic!("expected struct layout, got {layout:?}"),
    };
    let struct_list_child = struct_children
        .iter()
        .find_map(|id| {
            export
                .semantic_columns()
                .iter()
                .find(|column| column.id() == *id && column.data_type().id() == InteropTypeId::List)
        })
        .expect("expected struct child with list layout");
    let struct_list_leaf = match struct_list_child.layout() {
        ColumnLayout::List { child, .. } => *child,
        layout => panic!("expected nested list layout, got {layout:?}"),
    };
    let struct_leaf_column = export.semantic_columns().iter().find(|column| column.id() == struct_list_leaf).unwrap();
    assert_eq!(struct_list_child.data_type().id(), InteropTypeId::List);
    assert_eq!(struct_leaf_column.data_type().id(), InteropTypeId::Int64);

    let dictionary_keys = match export.columns()[0].layout() {
        ColumnLayout::Dictionary { keys, .. } => *keys,
        layout => panic!("expected dictionary layout, got {layout:?}"),
    };
    let dictionary_keys_column =
        export.semantic_columns().iter().find(|column| column.id() == dictionary_keys).unwrap();
    assert_eq!(dictionary_keys_column.data_type().id(), InteropTypeId::String);
    assert!(matches!(dictionary_keys_column.layout(), ColumnLayout::BinaryLike { .. }));

    let lease = export
        .try_activate_read_lease(ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)
        .unwrap();
    assert_eq!(lease.complete().unwrap(), LeaseCompletionResult::FallbackSynchronized);
}

#[test]
fn mixed_device_table_prepare_device_export_is_rejected() {
    if !should_run_multi_gpu_test() {
        return;
    }

    let (device0, device1) = first_two_test_devices();
    let rt0 = Runtime::builder().with_device(device0).build().unwrap();
    let rt1 = Runtime::builder().with_device(device1).build().unwrap();

    let left = Column::from_slice_with_runtime(&rt0, &[1i32, 2, 3])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(9001)
        .unwrap();
    let right = Column::from_slice_with_runtime(&rt1, &[4i32, 5, 6])
        .unwrap()
        .__debug_rebind_execution_domain_explicit(9001)
        .unwrap();
    let frame = crate::common::dataframe_from_columns(vec!["left".into(), "right".into()], vec![left, right]).unwrap();

    let err = frame.try_prepare_device_table_export().unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::MixedDeviceUnsupported);
}
