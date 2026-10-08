use std::{
    fs,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use ::object_store::{ObjectStoreExt, memory::InMemory, path::Path};
use query_planning::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::SourceChunkPlanningMode,
};
use serde_json::Value;

use super::{
    cache_fill::default_lakehouse_cache_fill_budget_bytes_from_physical_memory,
    object_store::default_object_store_control_runtime,
    service_stream::{
        default_service_stream_pool_size_from_device_facts, default_service_stream_pool_size_from_sm_count,
        nonzero_sm_count,
    },
    *,
};
use crate::observability::AdmissionRequestId;

fn p6_snapshot_json_samples() -> [(&'static str, Value); 8] {
    let backend_metrics = super::backend_metrics::BackendMetricsStore::new(7, &[3])
        .expect("fixed backend metrics fixture initializes")
        .snapshot();
    let ledger_report = DeviceLedgerReport {
        device_ordinal: 3,
        sequence: 17,
        managed_capacity_bytes: 10_000,
        default_query_limit_bytes: 100,
        max_active_attempts: 4,
        active_attempts: 1,
        live_grants_bytes: 900,
        frozen_resident_bytes: 0,
        resident_owner_bytes: 640,
        root_streams_available: 2,
        auxiliary_streams_available: 3,
        root_worker_slots_available: 1,
        auxiliary_worker_slots_available: 2,
        worker_depth: 3,
        available_bytes: 9_100,
        free_bytes: 8_900,
        evictable_bytes: 200,
        cache_evictable_resident_bytes: 128,
        cache_pinned_in_use_bytes: 64,
        health: "open".to_owned(),
        grants: vec![DeviceLedgerGrantRow {
            grant_id: 23,
            admission_request_id: AdmissionRequestId::new(29),
            grant_bytes: 900,
            resident_owner_bytes: 640,
            frozen_resident_bytes: None,
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
            })
            .expect("admission snapshot serializes"),
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

fn p6_snapshot_json_golden(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/backend_snapshot_json/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let fixture = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing backend snapshot JSON golden {path}: {error}"));
    serde_json::from_str(&fixture).expect("backend snapshot JSON golden is valid")
}

#[test]
fn p6_snapshot_json_shapes_match_goldens() {
    for (name, value) in p6_snapshot_json_samples() {
        assert_eq!(
            snapshot_json_shape(&value),
            p6_snapshot_json_golden(name),
            "backend snapshot JSON shape changed for {name}"
        );
    }
}

#[test]
fn object_store_fetch_returns_bytes_from_full_get() {
    let runtime = default_object_store_control_runtime().unwrap();
    let store = Arc::new(InMemory::new());
    let location = Path::from("part-0.parquet");
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
    fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
        self.facts.get(device_ordinal).cloned().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.test.gpu_inventory",
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
        .acquire_with_timeout(&cancellation, 256, Duration::ZERO, "query_engine.test.cache_fill")
        .unwrap()
        .unwrap();
    assert_eq!(first.bytes(), 256);
    assert!(first.try_grow(512).unwrap());
    assert_eq!(first.bytes(), 512);
    let second = budget.try_acquire(512).unwrap().unwrap();
    drop(first);
    let third = budget.try_acquire(512).unwrap().unwrap();
    drop(second);
    drop(third);
}

#[test]
fn execution_cancellation_sleep_ends_at_cancellation_or_deadline() {
    let cancellation = ExecutionCancellation::new();
    let canceller = cancellation.clone();
    let started = Instant::now();
    let cancel = thread::spawn(move || canceller.cancel());
    let error = cancellation.sleep(Duration::from_secs(600), "query_engine.test.sleep").unwrap_err();
    cancel.join().unwrap();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(60));

    let expiring = ExecutionCancellation::with_deadline(Some(Instant::now() + Duration::from_millis(10)));
    let error = expiring.sleep(Duration::from_secs(600), "query_engine.test.sleep").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(60));
}

#[test]
fn cache_fill_byte_budget_grow_reports_saturation_without_waiting() {
    let budget = CacheFillByteBudget::new(1024).unwrap();
    let mut first = budget.try_acquire(768).unwrap().unwrap();
    let _second = budget.try_acquire(256).unwrap().unwrap();

    // The budget is fully held: growth fails immediately instead of waiting
    // for the other holder, and the permit keeps its original charge.
    assert!(!first.try_grow(1024).unwrap());
    assert_eq!(first.bytes(), 768);
    // Shrinking or re-stating the current charge always succeeds.
    assert!(first.try_grow(768).unwrap());
    assert!(first.try_grow(100).unwrap());
    assert_eq!(first.bytes(), 768);
}

#[test]
fn cache_fill_byte_budget_two_holders_growing_past_the_remainder_never_deadlock() {
    // Two fills each hold half the budget and both ask for more than the
    // remainder covers. Neither waits, so neither can strand the other while
    // holding its share; both observe saturation and abandon.
    let budget = CacheFillByteBudget::new(1024).unwrap();
    let mut first = budget.try_acquire(512).unwrap().unwrap();
    let mut second = budget.try_acquire(512).unwrap().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let first_gate = Arc::clone(&barrier);
    let second_gate = Arc::clone(&barrier);
    let (first_grew, second_grew) = thread::scope(|scope| {
        let first_handle = scope.spawn(move || {
            first_gate.wait();
            // Return the permit so its release cannot land before the other fill tries to grow.
            (first.try_grow(1024), first)
        });
        second_gate.wait();
        let second_grew = second.try_grow(1024);
        let (first_grew, _first) = first_handle.join().unwrap();
        (first_grew, second_grew)
    });
    assert_eq!((first_grew.unwrap(), second_grew.unwrap()), (false, false));
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
    let context = ExecutionContext::standalone(StandaloneConfig {
        lakehouse_cache_fill: LakehouseCacheFillConfig {
            budget_bytes: Some(2048),
            acquire_timeout: Duration::from_secs(11),
            max_concurrent_ops: 3,
        },
        ..StandaloneConfig::default()
    })
    .unwrap();

    assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
    assert_eq!(context.lakehouse_cache_fill_acquire_timeout(), Duration::from_secs(11));
    assert_eq!(context.lakehouse_cache_fill_max_concurrent_ops(), 3);
}

#[test]
fn execution_context_records_source_chunk_planning_mode() {
    let options = SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(2048);
    let context = ExecutionContext::standalone(StandaloneConfig {
        source_chunk_planning_mode: SourceChunkPlanningMode::LocalParquetThroughput,
        source_chunk_planning_options: options,
        ..StandaloneConfig::default()
    })
    .unwrap();

    assert_eq!(context.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(context.source_chunk_planning_options(), options);
}

#[test]
fn execution_context_rejects_invalid_source_chunk_planning_options() {
    let error = ExecutionContext::standalone(StandaloneConfig {
        source_chunk_planning_options: SourceChunkPlanningOptions::default()
            .with_local_parquet_throughput_max_row_groups_per_chunk(0),
        ..StandaloneConfig::default()
    })
    .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
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
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(error.fact_value("field"), Some(field));
    }
}
