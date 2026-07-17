use std::{sync::Arc, time::Duration};

use ::object_store::ObjectStoreExt;

use super::{
    cache_events::LakehouseCacheFillEventSink,
    object_store::default_source_fetch_runtime,
    service_stream::{
        default_service_stream_pool_size_from_device_facts, default_service_stream_pool_size_from_sm_count,
        nonzero_sm_count,
    },
    source_fetch::{GdsConcurrencyGate, default_host_fetch_budget_bytes_from_physical_memory},
    *,
};
use crate::{
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
};

#[test]
fn object_store_fetch_returns_bytes_from_full_get() {
    let runtime = default_source_fetch_runtime().unwrap();
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
    DeviceFacts { device, total_bytes, free_bytes: total_bytes / 2, sm_count, utilization_pct: None }
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
fn default_host_fetch_budget_is_capped() {
    assert_eq!(
        default_host_fetch_budget_bytes_from_physical_memory(Some(128 * 1024 * 1024 * 1024)),
        SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES
    );
    assert_eq!(default_host_fetch_budget_bytes_from_physical_memory(Some(4 * 1024 * 1024 * 1024)), 1024 * 1024 * 1024);
    assert_eq!(
        default_host_fetch_budget_bytes_from_physical_memory(None),
        SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_BYTES
    );
}

#[test]
fn host_fetch_budget_tracks_acquire_grow_and_release() {
    let budget = HostFetchBudget::new(1024).unwrap();
    let mut first = budget.acquire(256).unwrap();
    assert_eq!(first.bytes(), 256);
    first.grow_to(512).unwrap();
    assert_eq!(first.bytes(), 512);
    let second = budget.acquire(512).unwrap();
    drop(first);
    let third = budget.acquire(512).unwrap();
    drop(second);
    drop(third);
}

#[test]
fn host_fetch_budget_rejects_oversized_fetches() {
    let budget = HostFetchBudget::new(1024).unwrap();
    let error = budget.acquire(2048).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("requested_bytes"), Some("2048"));
    assert_eq!(error.fact_value("budget_bytes"), Some("1024"));
}

#[test]
fn execution_context_records_host_fetch_concurrency_limit() {
    let context = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(7).build().unwrap();
    assert_eq!(context.host_fetch_max_concurrent_ops(), 7);
}

#[test]
fn execution_context_records_host_fetch_acquire_timeout() {
    let context = ExecutionContext::builder().with_host_fetch_acquire_timeout(Duration::from_secs(11)).build().unwrap();
    assert_eq!(context.host_fetch_acquire_timeout(), Duration::from_secs(11));
}

#[test]
fn execution_context_records_lakehouse_cache_fill_resource_limits() {
    let context = ExecutionContext::builder()
        .with_lakehouse_cache_fill_budget_bytes(2048)
        .with_lakehouse_cache_fill_max_concurrent_ops(3)
        .build()
        .unwrap();

    assert_eq!(context.lakehouse_cache_fill_budget().budget_bytes(), 2048);
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
        .with_host_fetch_max_concurrent_ops(7)
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
    assert_eq!(fragment_context.host_fetch_max_concurrent_ops(), 7);
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
fn execution_context_rejects_zero_host_fetch_concurrency_limit() {
    let error = ExecutionContext::builder().with_host_fetch_max_concurrent_ops(0).build().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("host_fetch_max_concurrent_ops"));
}

#[test]
fn execution_context_rejects_zero_host_fetch_acquire_timeout() {
    let error = ExecutionContext::builder().with_host_fetch_acquire_timeout(Duration::ZERO).build().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("host_fetch_acquire_timeout"));
}

fn available_gds_capability() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::Present { path: std::path::PathBuf::from(NVIDIA_FS_PROC_PATH) },
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::Available,
    )
}

fn gds_source_fetch_config() -> SourceFetchConfig {
    SourceFetchConfig {
        gds_capable_mount_roots: vec![std::path::PathBuf::from("/mnt/nexus-gds")],
        ..SourceFetchConfig::default()
    }
}

#[test]
fn source_fetch_config_defaults_match_source_fetch_contract() {
    let config = SourceFetchConfig::default();
    assert_eq!(config.substrate, SourceFetchSubstrate::Auto);
    assert_eq!(config.host_fetch_budget_bytes, None);
    assert_eq!(config.host_fetch_acquire_timeout, SourceFetchConfig::DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT);
    assert_eq!(config.host_fetch_max_concurrent_ops, SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS);
    assert_eq!(config.lakehouse_cache_fill_budget_bytes, None);
    assert_eq!(
        config.lakehouse_cache_fill_max_concurrent_ops,
        SourceFetchConfig::DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS
    );
    assert_eq!(config.gds_max_concurrent_ops, SourceFetchConfig::DEFAULT_GDS_MAX_CONCURRENT_OPS);
    assert_eq!(config.gds_min_chunk_bytes, SourceFetchConfig::DEFAULT_GDS_MIN_CHUNK_BYTES);
    assert_eq!(SourceFetchConfig::DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT, Duration::from_secs(30));
    assert_eq!(DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT, SourceFetchConfig::DEFAULT_HOST_FETCH_ACQUIRE_TIMEOUT);
    assert_eq!(DEFAULT_HOST_FETCH_BUDGET_BYTES, SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_BYTES);
    assert_eq!(DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES, SourceFetchConfig::DEFAULT_HOST_FETCH_BUDGET_CAP_BYTES);
    assert_eq!(DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS, SourceFetchConfig::DEFAULT_HOST_FETCH_MAX_CONCURRENT_OPS);
    assert_eq!(
        DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS,
        SourceFetchConfig::DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS
    );
    assert_eq!(DEFAULT_GDS_MAX_CONCURRENT_OPS, SourceFetchConfig::DEFAULT_GDS_MAX_CONCURRENT_OPS);
    assert_eq!(DEFAULT_GDS_MIN_CHUNK_BYTES, SourceFetchConfig::DEFAULT_GDS_MIN_CHUNK_BYTES);
}

#[test]
fn source_fetch_strategy_auto_rejects_remote_object_store_without_host_buffer_fallback() {
    let config = SourceFetchConfig::default();
    let decision = config.decide_source_fetch_strategy(
        &available_gds_capability(),
        "s3://bucket/path/file.parquet",
        Some(64 * 1024 * 1024),
    );

    assert_eq!(decision.strategy, SourceFetchStrategy::Rejected);
    assert_eq!(decision.reason, SourceFetchStrategyReason::RemoteObjectStore);
    assert_eq!(decision.gds_eligibility, SourceGdsEligibility::RemoteObjectStore);
    assert_eq!(
        decision.diagnostic_facts(),
        vec![
            ("source_fetch_strategy", "rejected".to_string()),
            ("source_fetch_strategy_reason", "remote_object_store".to_string()),
            ("source_fetch_substrate", "auto".to_string()),
            ("source_gds_eligibility", "remote_object_store".to_string()),
        ]
    );
}

#[test]
fn source_fetch_strategy_auto_selects_direct_device_only_for_gds_eligible_local_chunks() {
    let config = gds_source_fetch_config();
    let decision = config.decide_source_fetch_strategy(
        &available_gds_capability(),
        "/mnt/nexus-gds/tpch/sf10/lineitem.parquet",
        Some(64 * 1024 * 1024),
    );

    assert_eq!(decision.strategy, SourceFetchStrategy::DirectDevice);
    assert_eq!(decision.reason, SourceFetchStrategyReason::GdsEligible);
    assert_eq!(decision.gds_eligibility, SourceGdsEligibility::Eligible);
}

#[test]
fn source_fetch_config_rejects_host_buffer_substrate() {
    let config = SourceFetchConfig { substrate: SourceFetchSubstrate::HostBuffer, ..gds_source_fetch_config() };
    let error = config.validate_for_deployment(&available_gds_capability()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("source_fetch_substrate"), Some("host_buffer"));
    assert_eq!(error.fact_value("reason"), Some("host_buffer_disabled"));

    let decision = config.decide_source_fetch_strategy(
        &available_gds_capability(),
        "/mnt/nexus-gds/tpch/sf10/lineitem.parquet",
        Some(64 * 1024 * 1024),
    );

    assert_eq!(decision.strategy, SourceFetchStrategy::Rejected);
    assert_eq!(decision.reason, SourceFetchStrategyReason::HostBufferDisabled);
    assert_eq!(decision.gds_eligibility, SourceGdsEligibility::Eligible);
}

#[test]
fn source_fetch_strategy_direct_device_rejects_ineligible_sources() {
    let config = SourceFetchConfig { substrate: SourceFetchSubstrate::DirectDevice, ..gds_source_fetch_config() };

    let remote = config.decide_source_fetch_strategy(
        &available_gds_capability(),
        "https://example.test/file.parquet",
        Some(64 * 1024 * 1024),
    );
    assert_eq!(remote.strategy, SourceFetchStrategy::Rejected);
    assert_eq!(remote.reason, SourceFetchStrategyReason::RemoteObjectStore);

    let small = config.decide_source_fetch_strategy(
        &available_gds_capability(),
        "/mnt/nexus-gds/tpch/sf10/lineitem.parquet",
        Some(1024),
    );
    assert_eq!(small.strategy, SourceFetchStrategy::Rejected);
    assert_eq!(small.reason, SourceFetchStrategyReason::BelowMinChunkBytes);

    let eligible = config.decide_source_fetch_strategy(
        &available_gds_capability(),
        "/mnt/nexus-gds/tpch/sf10/lineitem.parquet",
        Some(64 * 1024 * 1024),
    );
    assert_eq!(eligible.strategy, SourceFetchStrategy::DirectDevice);
    assert_eq!(eligible.reason, SourceFetchStrategyReason::GdsEligible);
}

#[test]
fn execution_context_exposes_effective_source_fetch_strategy() {
    let context = ExecutionContext::builder()
        .with_deployment_io_capability(available_gds_capability())
        .with_source_fetch_config(gds_source_fetch_config())
        .build()
        .unwrap();

    let decision =
        context.source_fetch_strategy_for_location("/mnt/nexus-gds/tpch/sf10/lineitem.parquet", Some(64 * 1024 * 1024));

    assert_eq!(decision.strategy, SourceFetchStrategy::DirectDevice);
    assert_eq!(decision.substrate, SourceFetchSubstrate::Auto);
}

#[test]
fn source_fetch_config_rejects_zero_gds_controls() {
    let deployment = available_gds_capability();
    let zero_ops = SourceFetchConfig { gds_max_concurrent_ops: 0, ..SourceFetchConfig::default() };
    let error = zero_ops.validate_for_deployment(&deployment).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("gds_max_concurrent_ops"));

    let zero_cache_fill_ops =
        SourceFetchConfig { lakehouse_cache_fill_max_concurrent_ops: 0, ..SourceFetchConfig::default() };
    let error = zero_cache_fill_ops.validate_for_deployment(&deployment).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("lakehouse_cache_fill_max_concurrent_ops"));

    let zero_cache_fill_budget =
        SourceFetchConfig { lakehouse_cache_fill_budget_bytes: Some(0), ..SourceFetchConfig::default() };
    let error = zero_cache_fill_budget.validate_for_deployment(&deployment).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("lakehouse_cache_fill_budget_bytes"));

    let zero_host_fetch_acquire_timeout =
        SourceFetchConfig { host_fetch_acquire_timeout: Duration::ZERO, ..SourceFetchConfig::default() };
    let error = zero_host_fetch_acquire_timeout.validate_for_deployment(&deployment).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("host_fetch_acquire_timeout"));

    let zero_min_chunk = SourceFetchConfig { gds_min_chunk_bytes: 0, ..SourceFetchConfig::default() };
    let error = zero_min_chunk.validate_for_deployment(&deployment).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("gds_min_chunk_bytes"));
}

#[test]
fn direct_device_source_fetch_config_requires_gds_available() {
    let unavailable = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::Available,
    );
    let config = SourceFetchConfig { substrate: SourceFetchSubstrate::DirectDevice, ..gds_source_fetch_config() };
    let error = config.validate_for_deployment(&unavailable).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("source_fetch_substrate"), Some("direct_device"));
    assert_eq!(error.fact_value("gds_available"), Some("false"));
    assert_eq!(error.fact_value("gds_status"), Some("driver_missing"));

    config.validate_for_deployment(&available_gds_capability()).unwrap();
}

#[test]
fn gds_concurrency_gate_limits_in_flight_ops() {
    let gate = GdsConcurrencyGate::new(1).unwrap();
    let permit = gate.acquire().unwrap();
    let gate_for_thread = gate.clone();
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (acquired_sender, acquired_receiver) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        started_sender.send(()).unwrap();
        let permit = gate_for_thread.acquire().unwrap();
        acquired_sender.send(()).unwrap();
        drop(permit);
    });
    started_receiver.recv().unwrap();
    assert!(acquired_receiver.try_recv().is_err());
    drop(permit);
    acquired_receiver.recv().unwrap();
    handle.join().unwrap();
}
