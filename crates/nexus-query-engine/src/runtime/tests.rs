use std::{sync::Arc, time::Duration};

use ::object_store::ObjectStoreExt;
use serde_json::Value;

use super::{
    cache_events::LakehouseCacheFillEventSink,
    cache_fill::default_lakehouse_cache_fill_budget_bytes_from_physical_memory,
    object_store::default_object_store_control_runtime,
    service_stream::{
        default_service_stream_pool_size_from_device_facts, default_service_stream_pool_size_from_sm_count,
        nonzero_sm_count,
    },
    *,
};
use crate::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
};

fn p6_snapshot_json_v1_samples() -> [(&'static str, Value); 9] {
    let backend_metrics = super::backend_metrics::BackendMetricsStore::new(7, &[3])
        .expect("fixed backend metrics fixture initializes")
        .snapshot();
    let ledger_report = DeviceLedgerReport {
        device_ordinal: 3,
        sequence: 17,
        managed_capacity_bytes: 10_000,
        configured_floor_bytes: 100,
        max_active_attempts: 4,
        active_attempts: 1,
        live_grants_bytes: 900,
        live_reserved_bytes: 640,
        root_streams_available: 2,
        auxiliary_streams_available: 3,
        root_worker_slots_available: 1,
        auxiliary_worker_slots_available: 2,
        worker_depth: 3,
        available_bytes: 9_100,
        free_bytes: 8_900,
        evictable_bytes: 200,
        protected_slot: true,
        protected_reserve_bytes: 100,
        cache_evictable_resident_bytes: 128,
        cache_pinned_in_use_bytes: 64,
        health: "open".to_owned(),
        grants: vec![DeviceLedgerGrantRow {
            grant_id: 23,
            admission_request_id: crate::observability::AdmissionRequestId::new(29),
            grant_bytes: 900,
            live_reserved_bytes: 640,
        }],
    };
    let device_memory_failure = DeviceMemoryFailureReport {
        device_ordinal: 3,
        callback_installed: true,
        callback_install_unavailable: false,
        allocation_failure_count: 2,
        total_failed_allocation_bytes: 8_192,
        largest_failed_allocation_bytes: 4_096,
        trim_retry_count: 1,
        pool_usage_observation_count: 9,
        last_failed_allocation_bytes: 4_096,
        last_pool_reserved_current_bytes: 16_384,
        last_pool_used_current_bytes: 12_288,
    };
    let device_telemetry = DeviceTelemetrySnapshot {
        sampled_at_unix_ms: 1_700_000_000_123,
        device_ordinal: 3,
        cuda_status: DeviceTelemetryStatus::Available,
        cuda_total_bytes: Some(46_068 * 1024 * 1024),
        cuda_free_bytes: Some(40_000 * 1024 * 1024),
        nvml_status: DeviceTelemetryStatus::Available,
        nvml_total_bytes: Some(46_068 * 1024 * 1024),
        nvml_reserved_bytes: Some(2_048),
        nvml_used_bytes: Some(6_000 * 1024 * 1024),
        nvml_free_bytes: Some(40_000 * 1024 * 1024),
        gpu_utilization_pct: Some(37),
        memory_utilization_pct: Some(21),
    };
    let execution_pool = GpuExecutionPoolSnapshot {
        pool_id: 31,
        device_ordinal: 3,
        worker_count: 4,
        active_jobs: 2,
        peak_active_jobs: 3,
        queued_jobs: 1,
        peak_queued_jobs: 5,
        active_query_jobs: 1,
        peak_active_query_jobs: 2,
        root_job_capacity: 1,
        aux_job_capacity: 3,
        root_jobs_available: 0,
        aux_jobs_available: 2,
    };

    [
        (
            "backend_metrics_snapshot",
            serde_json::to_value(backend_metrics).expect("backend metrics snapshot serializes"),
        ),
        (
            "query_service_admission_snapshot",
            serde_json::to_value(QueryServiceAdmissionSnapshot {
                active_queries: 2,
                queued_queries: 1,
                queue_high_watermark: 4,
                protected_devices: vec![1, 3],
            })
            .expect("admission snapshot serializes"),
        ),
        (
            "query_service_device_admission_snapshot",
            serde_json::to_value(QueryServiceDeviceAdmissionSnapshot {
                device_ordinal: 3,
                process_active_query_slots: 2,
                has_protection: true,
            })
            .expect("device admission snapshot serializes"),
        ),
        ("device_ledger_report", serde_json::to_value(ledger_report).expect("ledger report serializes")),
        (
            "device_memory_failure_report",
            serde_json::to_value(device_memory_failure).expect("memory failure report serializes"),
        ),
        ("device_telemetry_snapshot", serde_json::to_value(device_telemetry).expect("telemetry snapshot serializes")),
        ("gpu_execution_pool_snapshot", serde_json::to_value(execution_pool).expect("pool snapshot serializes")),
        (
            "gpu_execution_query_snapshot",
            serde_json::to_value(GpuExecutionQuerySnapshot { active_query_jobs: 1, peak_active_query_jobs: 3 })
                .expect("query snapshot serializes"),
        ),
        (
            "gpu_backend_close_snapshot",
            serde_json::to_value(GpuBackendCloseSnapshot {
                closing: true,
                gpu_resource_quiescent: true,
                observation_quiescent: false,
                active_gpu_permits: 0,
                active_observation_permits: 2,
                cleanup_failed: false,
                cleanup_deadline_elapsed_count: 1,
                cleanup_late_success_count: 2,
            })
            .expect("close snapshot serializes"),
        ),
    ]
}

fn snapshot_json_shape(value: &Value) -> Value {
    match value {
        Value::Null => Value::String("null".to_owned()),
        Value::Bool(_) => Value::String("boolean".to_owned()),
        Value::Number(_) => Value::String("number".to_owned()),
        Value::String(_) => Value::String("string".to_owned()),
        Value::Array(values) => {
            let mut item_shapes = Vec::new();
            for value in values {
                let shape = snapshot_json_shape(value);
                if !item_shapes.contains(&shape) {
                    item_shapes.push(shape);
                }
            }
            serde_json::json!({ "type": "array", "length": values.len(), "items": item_shapes })
        },
        Value::Object(values) => {
            let fields = values
                .iter()
                .map(|(name, value)| (name.clone(), snapshot_json_shape(value)))
                .collect::<serde_json::Map<_, _>>();
            serde_json::json!({ "type": "object", "fields": fields })
        },
    }
}

fn p6_snapshot_json_v1_golden(name: &str) -> Value {
    let fixture = match name {
        "backend_metrics_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/backend_metrics_snapshot.json"
        )),
        "query_service_admission_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/query_service_admission_snapshot.json"
        )),
        "query_service_device_admission_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/query_service_device_admission_snapshot.json"
        )),
        "device_ledger_report" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/device_ledger_report.json"
        )),
        "device_memory_failure_report" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/device_memory_failure_report.json"
        )),
        "device_telemetry_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/device_telemetry_snapshot.json"
        )),
        "gpu_execution_pool_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/gpu_execution_pool_snapshot.json"
        )),
        "gpu_execution_query_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/gpu_execution_query_snapshot.json"
        )),
        "gpu_backend_close_snapshot" => include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/backend_snapshot_json_v1/gpu_backend_close_snapshot.json"
        )),
        _ => panic!("missing P6 backend snapshot JSON v1 golden: {name}"),
    };
    serde_json::from_str(fixture).expect("versioned backend snapshot JSON golden is valid")
}

#[test]
fn p6_snapshot_json_v1_shapes_match_versioned_goldens() {
    for (name, value) in p6_snapshot_json_v1_samples() {
        assert_eq!(
            snapshot_json_shape(&value),
            p6_snapshot_json_v1_golden(name),
            "P6 backend snapshot JSON v1 shape changed for {name}"
        );
    }
}

#[test]
fn object_store_fetch_returns_bytes_from_full_get() {
    let runtime = default_object_store_control_runtime().unwrap();
    let store = Arc::new(::object_store::memory::InMemory::new());
    let location = ::object_store::path::Path::from("part-0.parquet");
    runtime.block_on(store.put(&location, b"abcdef".as_ref().into())).unwrap();

    let bytes = ExecutionContext::fetch_object_store_bytes_on_runtime(runtime.as_ref(), store, location).unwrap();

    assert_eq!(bytes.as_ref(), b"abcdef");
}

#[test]
fn service_stream_pool_default_size_is_sm_aware_with_cap_and_fallback() {
    assert_eq!(default_service_stream_pool_size_from_sm_count(Some(1)), 2);
    assert_eq!(default_service_stream_pool_size_from_sm_count(Some(16)), 32);
    assert_eq!(default_service_stream_pool_size_from_sm_count(Some(32)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
    assert_eq!(default_service_stream_pool_size_from_sm_count(Some(128)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
    assert_eq!(default_service_stream_pool_size_from_sm_count(Some(0)), DEFAULT_SERVICE_STREAM_POOL_SIZE);
    assert_eq!(default_service_stream_pool_size_from_sm_count(None), DEFAULT_SERVICE_STREAM_POOL_SIZE);
}

#[derive(Debug)]
struct FakeGpuInventory {
    facts: Vec<DeviceFacts>,
}

impl GpuInventory for FakeGpuInventory {
    fn device_count(&self) -> Result<usize> {
        Ok(self.facts.len())
    }

    fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
        self.facts.get(device_ordinal).cloned().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.test.gpu_inventory",
                "fake device ordinal out of range",
            )
            .with_fact("device_ordinal", device_ordinal)
        })
    }
}

fn fake_device_facts(device: usize, total_bytes: u64, sm_count: Option<u32>) -> DeviceFacts {
    DeviceFacts {
        device,
        model_name: None,
        uuid: None,
        physical_total_bytes: None,
        total_bytes,
        free_bytes: total_bytes / 2,
        sm_count,
        utilization_pct: None,
    }
}

#[test]
fn gpu_inventory_fake_drives_pool_size_and_budget_math_without_gpu() {
    let inventory = FakeGpuInventory { facts: vec![fake_device_facts(0, 16 * 1024 * 1024 * 1024, Some(12))] };
    let facts = inventory.facts(0).unwrap();

    assert_eq!(default_service_stream_pool_size_from_device_facts(&facts), 24);
    assert_eq!(device_budget_bytes_from_facts(&facts, 512 * 1024 * 1024, 256 * 1024 * 1024), 16_642_998_272);
}

#[test]
fn gpu_inventory_all_facts_uses_fake_ordinals_without_cuda_device_resolution() {
    let inventory = FakeGpuInventory {
        facts: vec![
            fake_device_facts(0, 8 * 1024 * 1024 * 1024, Some(10)),
            fake_device_facts(1, 12 * 1024 * 1024 * 1024, Some(20)),
        ],
    };

    let facts = inventory.all_facts().unwrap();

    assert_eq!(facts.iter().map(|facts| facts.device).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(facts.iter().filter_map(|facts| facts.sm_count).collect::<Vec<_>>(), vec![10, 20]);
}

#[test]
fn service_stream_pool_default_size_uses_sm_times_two_for_small_sm_device_fact() {
    let facts = fake_device_facts(0, 4 * 1024 * 1024 * 1024, Some(31));

    assert_eq!(default_service_stream_pool_size_from_device_facts(&facts), 62);
}

#[test]
fn service_stream_pool_sm_count_filters_zero_counts() {
    assert_eq!(nonzero_sm_count(142), Some(142));
    assert_eq!(nonzero_sm_count(1), Some(1));
    assert_eq!(nonzero_sm_count(0), None);
}

#[test]
fn default_lakehouse_cache_fill_budget_is_capped() {
    assert_eq!(
        default_lakehouse_cache_fill_budget_bytes_from_physical_memory(Some(128 * 1024 * 1024 * 1024)),
        LakehouseCacheFillConfig::DEFAULT_BUDGET_CAP_BYTES
    );
    assert_eq!(
        default_lakehouse_cache_fill_budget_bytes_from_physical_memory(Some(4 * 1024 * 1024 * 1024)),
        1024 * 1024 * 1024
    );
    assert_eq!(
        default_lakehouse_cache_fill_budget_bytes_from_physical_memory(None),
        LakehouseCacheFillConfig::DEFAULT_BUDGET_BYTES
    );
}

#[test]
fn cache_fill_byte_budget_tracks_acquire_grow_and_release() {
    let budget = CacheFillByteBudget::new(1024).unwrap();
    let cancellation = ExecutionCancellation::new();
    let mut first = budget
        .acquire_with_timeout(&cancellation, 256, Duration::ZERO, "nexus_query_engine.test.cache_fill")
        .unwrap()
        .unwrap();
    assert_eq!(first.bytes(), 256);
    first.grow_to(512).unwrap();
    assert_eq!(first.bytes(), 512);
    let second = budget.try_acquire(512).unwrap().unwrap();
    drop(first);
    let third = budget.try_acquire(512).unwrap().unwrap();
    drop(second);
    drop(third);
}

#[test]
fn cache_fill_byte_budget_rejects_oversized_fetches() {
    let budget = CacheFillByteBudget::new(1024).unwrap();
    let error = budget.try_acquire(2048).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("requested_bytes"), Some("2048"));
    assert_eq!(error.fact_value("budget_bytes"), Some("1024"));
}

#[test]
fn execution_context_records_lakehouse_cache_fill_resource_limits() {
    let context = ExecutionContext::builder()
        .with_lakehouse_cache_fill_budget_bytes(2048)
        .with_lakehouse_cache_fill_acquire_timeout(Duration::from_secs(11))
        .with_lakehouse_cache_fill_max_concurrent_ops(3)
        .build()
        .unwrap();

    assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
    assert_eq!(context.lakehouse_cache_fill_acquire_timeout(), Duration::from_secs(11));
    assert_eq!(context.lakehouse_cache_fill_max_concurrent_ops(), 3);
}

#[test]
fn lakehouse_cache_fill_event_sink_records_structured_events() {
    let sink = LakehouseCacheFillEventSink::default();

    sink.record(LakehouseCacheFillEvent {
        outcome: LakehouseCacheFillOutcome::Completed,
        bytes: Some(4096),
        elapsed: Duration::from_millis(7),
        cache_path_hash: "cache-path-hash".to_string(),
        stable_id_hash: "stable-id-hash".to_string(),
        file_index: Some(2),
        chunk_index: None,
        error: None,
    });
    sink.record(LakehouseCacheFillEvent {
        outcome: LakehouseCacheFillOutcome::Failed,
        bytes: None,
        elapsed: Duration::from_millis(3),
        cache_path_hash: "failed-path-hash".to_string(),
        stable_id_hash: "failed-stable-hash".to_string(),
        file_index: Some(3),
        chunk_index: Some(4),
        error: Some(LakehouseCacheFillErrorEvent {
            code: "cudf_dependency_failure".to_string(),
            kind: "external".to_string(),
            status: "temporary".to_string(),
            operation: "nexus_query_engine.lakehouse_cache.background_fill.fetch".to_string(),
        }),
    });

    let events = sink.events();

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].outcome, LakehouseCacheFillOutcome::Completed);
    assert_eq!(events[0].bytes, Some(4096));
    assert_eq!(events[1].outcome, LakehouseCacheFillOutcome::Failed);
    assert_eq!(
        events[1].error.as_ref().map(|error| error.operation.as_str()),
        Some("nexus_query_engine.lakehouse_cache.background_fill.fetch")
    );
}

#[test]
fn stream_lease_context_shares_lakehouse_cache_fill_event_sink() {
    let context = ExecutionContext::builder().build().unwrap();
    let stream_lease = context.service_stream_pool().acquire().unwrap();
    let leased_context = context.with_stream_lease(&stream_lease).unwrap();

    leased_context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
        outcome: LakehouseCacheFillOutcome::Completed,
        bytes: Some(8192),
        elapsed: Duration::from_millis(5),
        cache_path_hash: "leased-cache-path-hash".to_string(),
        stable_id_hash: "leased-stable-id-hash".to_string(),
        file_index: Some(1),
        chunk_index: Some(2),
        error: None,
    });

    let snapshots = context.lakehouse_cache_fill_event_snapshots();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].outcome, "completed");
    assert_eq!(snapshots[0].bytes, Some(8192));
    assert_eq!(snapshots[0].cache_path_hash, "leased-cache-path-hash");
}

#[test]
fn lakehouse_cache_fill_event_snapshots_can_be_drained_once() {
    let context = ExecutionContext::builder().build().unwrap();
    context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
        outcome: LakehouseCacheFillOutcome::Completed,
        bytes: Some(1024),
        elapsed: Duration::from_millis(2),
        cache_path_hash: "cache-path-hash".to_string(),
        stable_id_hash: "stable-id-hash".to_string(),
        file_index: None,
        chunk_index: None,
        error: None,
    });

    let first = context.drain_lakehouse_cache_fill_event_snapshots();
    let second = context.drain_lakehouse_cache_fill_event_snapshots();

    assert_eq!(first.len(), 1);
    assert_eq!(first[0].outcome, "completed");
    assert!(second.is_empty());
    assert!(context.lakehouse_cache_fill_event_snapshots().is_empty());
}

#[test]
fn unbounded_fragment_context_preserves_query_state() {
    let cancellation = ExecutionCancellation::new();
    let context = ExecutionContext::builder()
        .with_cancellation(cancellation.clone())
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 64 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_lakehouse_cache_fill_max_concurrent_ops(7)
        .build()
        .unwrap();
    context.lakehouse_cache_fill_events().record(LakehouseCacheFillEvent {
        outcome: LakehouseCacheFillOutcome::Completed,
        bytes: Some(2048),
        elapsed: Duration::from_millis(3),
        cache_path_hash: "fragment-cache-path-hash".to_string(),
        stable_id_hash: "fragment-stable-id-hash".to_string(),
        file_index: Some(1),
        chunk_index: Some(2),
        error: None,
    });
    cancellation.cancel();

    let fragment_context = context.for_unbounded_fragment();

    assert_eq!(fragment_context.memory_control_policy(), MemoryControlPolicy::Unbounded);
    assert!(fragment_context.cancellation().is_cancelled());
    assert_eq!(fragment_context.lakehouse_cache_fill_max_concurrent_ops(), 7);
    assert_eq!(fragment_context.lakehouse_cache_fill_event_snapshots().len(), 1);
    assert_eq!(context.lakehouse_cache_fill_event_snapshots().len(), 1);
}

#[test]
fn execution_context_records_source_chunk_planning_mode() {
    let options = SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(2048);
    let context = ExecutionContext::builder()
        .with_source_chunk_planning_mode(SourceChunkPlanningMode::LocalParquetThroughput)
        .with_source_chunk_planning_options(options)
        .build()
        .unwrap();

    assert_eq!(context.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(context.source_chunk_planning_options(), options);
}

#[test]
fn execution_context_rejects_invalid_source_chunk_planning_options() {
    let error = ExecutionContext::builder()
        .with_source_chunk_planning_options(
            SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(0),
        )
        .build()
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_max_row_groups_per_chunk"));
}

#[test]
fn lakehouse_cache_fill_config_defaults_match_contract() {
    let config = LakehouseCacheFillConfig::default();
    assert_eq!(config.budget_bytes, None);
    assert_eq!(config.acquire_timeout, Duration::from_secs(30));
    assert_eq!(config.max_concurrent_ops, 4);
    config.validate().unwrap();
}

#[test]
fn lakehouse_cache_fill_config_rejects_zero_controls() {
    for (config, field) in [
        (LakehouseCacheFillConfig { budget_bytes: Some(0), ..Default::default() }, "lakehouse_cache_fill_budget_bytes"),
        (
            LakehouseCacheFillConfig { acquire_timeout: Duration::ZERO, ..Default::default() },
            "lakehouse_cache_fill_acquire_timeout",
        ),
        (
            LakehouseCacheFillConfig { max_concurrent_ops: 0, ..Default::default() },
            "lakehouse_cache_fill_max_concurrent_ops",
        ),
    ] {
        let error = config.validate().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some(field));
    }
}
