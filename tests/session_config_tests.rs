use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, config::ConfigOptions},
    execution::SessionStateBuilder,
    prelude::SessionContext,
};
use datafusion_nexus::{
    error::{Error, ErrorCode},
    native_exec::NexusNativeExecutionConfig,
    planner::NexusNativeOptimizerConfig,
    session::NexusSessionStateBuilderExt,
    session_config::{DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS, DataFusionNexusSessionConfig, NativeMemoryPolicyConfig},
};
use nexus_query_engine::{
    mode::{NativePipelineExecutionMode, StatisticalAggregateExecutionMode},
    runtime::{
        MemoryControlPolicy, QueryService, QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee,
        SourceChunkPlanningMode, SourceSchedulerPolicy,
    },
};

#[test]
fn nexus_session_builder_registers_datafusion_nexus_config_extension() {
    let state = SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();

    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("with_nexus_native must register datafusion_nexus config options");

    assert_eq!(extension.native_memory_policy, NativeMemoryPolicyConfig::Bounded);
}

#[test]
fn datafusion_nexus_session_config_applies_native_bounded_options() {
    let base = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1));
    let mut options = options_with_nexus_extension(&base);

    options.set("datafusion_nexus.native.device_budget_bytes", "2048").unwrap();
    options.set("datafusion_nexus.native.max_source_chunk_bytes", "1024").unwrap();
    options.set("datafusion_nexus.native.max_row_groups_per_chunk", "8").unwrap();
    options.set("datafusion_nexus.native.max_retry_attempts", "5").unwrap();
    options.set("datafusion_nexus.native.min_retry_chunk_row_groups", "2").unwrap();
    options.set("datafusion_nexus.native.max_source_reads_in_flight", "4").unwrap();
    options.set("datafusion_nexus.native.source_chunk_planning_mode", "local_parquet_throughput").unwrap();
    options.set("datafusion_nexus.native.pipeline_execution_mode", "adaptive").unwrap();
    options.set("datafusion_nexus.native.statistical_aggregate_execution_mode", "gpu_native_tolerant").unwrap();
    options.set("datafusion_nexus.request.requested_grant_min_bytes", "1024").unwrap();
    options.set("datafusion_nexus.request.requested_grant_bytes", "1536").unwrap();
    options.set("datafusion_nexus.request.graph_raw_input_edges", "45564149").unwrap();

    let effective = base.with_datafusion_config_options(&options).unwrap();
    let native = effective.native_execution_config();

    assert_eq!(
        native.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 2048,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 5,
            min_retry_chunk_row_groups: 2,
        }
    );
    assert_eq!(native.source_scheduler_policy().max_source_reads_in_flight(), 4);
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(native.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
    assert_eq!(native.requested_grant_min_bytes(), Some(1024));
    assert_eq!(native.requested_grant_bytes(), Some(1536));
    let graph_calibration = native.graph_memory_calibration_override();
    assert_eq!(graph_calibration.raw_input_edges, Some(45_564_149));
}

#[test]
fn datafusion_nexus_session_config_unbounded_disables_bounded_options() {
    let base_execution = NexusNativeExecutionConfig::bounded(1024, 512, 8, 5, 2)
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(4))
        .with_source_chunk_planning_mode(SourceChunkPlanningMode::LocalParquetThroughput)
        .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive);
    let base = NexusNativeOptimizerConfig::default().with_native_execution_config(base_execution);
    let mut options = options_with_nexus_extension(&base);

    options.set("datafusion_nexus.native.memory_policy", "unbounded").unwrap();
    options.set("datafusion_nexus.native.statistical_aggregate_execution_mode", "gpu_native_tolerant").unwrap();

    let effective = base.with_datafusion_config_options(&options).unwrap();
    let native = effective.native_execution_config();

    assert_eq!(native.memory_control_policy(), MemoryControlPolicy::Unbounded);
    assert_eq!(
        native.source_scheduler_policy().max_source_reads_in_flight(),
        SourceSchedulerPolicy::default().max_source_reads_in_flight()
    );
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::Conservative);
    assert_eq!(native.native_pipeline_execution_mode(), NativePipelineExecutionMode::StrictBounded);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

#[test]
fn datafusion_nexus_session_config_rejects_invalid_values() {
    let base = NexusNativeOptimizerConfig::default();
    let mut options = options_with_nexus_extension(&base);

    let error = options.set("datafusion_nexus.native.max_retry_attempts", "0").unwrap_err();
    assert!(error.to_string().contains("native.max_retry_attempts"));

    let error = options.set("datafusion_nexus.native.pipeline_execution_mode", "strict").unwrap_err();
    assert!(error.to_string().contains("strict_bounded"));

    let error = options.set("datafusion_nexus.request.requested_grant_min_bytes", "0").unwrap_err();
    assert!(error.to_string().contains("request.requested_grant_min_bytes"));

    let error = options.set("datafusion_nexus.request.requested_grant_bytes", "0").unwrap_err();
    assert!(error.to_string().contains("request.requested_grant_bytes"));

    let error = options.set("datafusion_nexus.request.graph_raw_input_edges", "0").unwrap_err();
    assert!(error.to_string().contains("request.graph_raw_input_edges"));

    let removed_key = ["datafusion_nexus.request.graph_topology_", "family"].concat();
    let error = options.set(&removed_key, "citation_like_sparse").unwrap_err();
    assert!(error.to_string().contains(&removed_key["datafusion_nexus.".len()..]));
    assert!(error.to_string().contains("not found"));
}

#[test]
fn datafusion_nexus_session_config_rejects_bounded_settings_while_unbounded() {
    let base = NexusNativeOptimizerConfig::default();
    let mut options = options_with_nexus_extension(&base);

    options.set("datafusion_nexus.native.memory_policy", "unbounded").unwrap();

    let error = options.set("datafusion_nexus.native.max_source_reads_in_flight", "4").unwrap_err();
    assert!(error.to_string().contains("native.max_source_reads_in_flight"));
    assert!(error.to_string().contains("unbounded"));
}

#[test]
fn datafusion_nexus_request_grant_rejects_pre_admitted_query_handle() {
    let service = QueryService::new(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            stream_pool_size: Some(3),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 3 * 1024 * 1024)],
    )
    .unwrap();
    let handle = Arc::new(service.try_admit_query("tenant").unwrap().expect("query should admit"));
    let base = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let mut options = options_with_nexus_extension(&base);

    options.set("datafusion_nexus.request.requested_grant_bytes", "1048576").unwrap();
    let error = base.with_datafusion_config_options(&options).unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::InvalidRequest);
    assert_eq!(adapter_error.fact_value("reason"), Some("request_grant_pre_admitted_query_handle"));
    assert_eq!(adapter_error.fact_value("requested_grant_bytes"), Some("1048576"));
}

#[test]
fn datafusion_nexus_request_grant_min_rejects_pre_admitted_query_handle() {
    let service = QueryService::new(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            stream_pool_size: Some(3),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 3 * 1024 * 1024)],
    )
    .unwrap();
    let handle = Arc::new(service.try_admit_query("tenant").unwrap().expect("query should admit"));
    let base = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let mut options = options_with_nexus_extension(&base);

    options.set("datafusion_nexus.request.requested_grant_min_bytes", "1048576").unwrap();
    let error = base.with_datafusion_config_options(&options).unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::InvalidRequest);
    assert_eq!(adapter_error.fact_value("reason"), Some("request_grant_pre_admitted_query_handle"));
    assert_eq!(adapter_error.fact_value("requested_grant_min_bytes"), Some("1048576"));
}

#[test]
fn datafusion_nexus_pre_admitted_query_handle_without_request_grant_is_unchanged() {
    let service = QueryService::new(
        QueryServiceConfig {
            max_concurrent_queries: 1,
            stream_pool_size: Some(3),
            slot_guarantee: SlotGuarantee::Optimistic,
            ..QueryServiceConfig::default()
        },
        vec![QueryServiceTenantConfig::new("tenant", 3 * 1024 * 1024)],
    )
    .unwrap();
    let handle = Arc::new(service.try_admit_query("tenant").unwrap().expect("query should admit"));
    let base = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded_auto_sized().with_query_service_handle(handle),
    );
    let options = options_with_nexus_extension(&base);

    let effective = base.with_datafusion_config_options(&options).unwrap();

    assert_eq!(effective.native_execution_config().requested_grant_min_bytes(), None);
    assert_eq!(effective.native_execution_config().requested_grant_bytes(), None);
    assert_eq!(
        effective.native_execution_config().memory_control_policy(),
        base.native_execution_config().memory_control_policy()
    );
}

#[test]
fn datafusion_nexus_session_config_ignores_direct_bounded_fields_while_unbounded() {
    let base = NexusNativeOptimizerConfig::default();
    let mut session_config = DataFusionNexusSessionConfig::from_optimizer_config(&base);
    session_config.native_memory_policy = NativeMemoryPolicyConfig::Unbounded;
    session_config.native_device_budget_bytes = 0;
    session_config.native_max_source_chunk_bytes = 0;
    session_config.native_max_row_groups_per_chunk = 0;
    session_config.native_max_retry_attempts = 0;
    session_config.native_min_retry_chunk_row_groups = 0;
    session_config.native_max_source_reads_in_flight = 0;

    let effective = session_config.apply_to_optimizer_config(base.clone()).unwrap();
    assert_eq!(effective.native_execution_config().memory_control_policy(), MemoryControlPolicy::Unbounded);

    session_config.native_memory_policy = NativeMemoryPolicyConfig::Bounded;
    let error = session_config.apply_to_optimizer_config(base).unwrap_err();
    assert!(error.to_string().contains("native.device_budget_bytes"));
}

#[test]
fn datafusion_nexus_session_config_entries_are_discoverable() {
    let base = NexusNativeOptimizerConfig::default();
    let options = options_with_nexus_extension(&base);
    let entries = options.entries();
    let nexus_entries = entries.iter().filter(|entry| entry.key.starts_with("datafusion_nexus.")).collect::<Vec<_>>();

    assert!(entries.iter().any(|entry| entry.key == "datafusion_nexus.native.memory_policy"));
    assert!(entries.iter().any(|entry| entry.key == "datafusion_nexus.native.statistical_aggregate_execution_mode"));
    assert_eq!(nexus_entries.len(), DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS.len());
    for option in DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS {
        let entry = nexus_entries
            .iter()
            .find(|entry| entry.key == option.full_key())
            .unwrap_or_else(|| panic!("missing entry for {}", option.full_key()));
        assert_eq!(entry.description, option.description);
        assert!(entry.value.is_some(), "{} should report current value", option.full_key());
    }
}

#[test]
fn datafusion_nexus_session_config_option_descriptors_document_scope_and_types() {
    let bounded_keys = DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
        .iter()
        .filter(|option| option.bounded_only)
        .map(|option| option.key)
        .collect::<Vec<_>>();

    assert!(bounded_keys.contains(&"native.max_source_reads_in_flight"));
    assert!(!bounded_keys.contains(&"native.memory_policy"));
    assert!(!bounded_keys.contains(&"native.statistical_aggregate_execution_mode"));
    assert!(
        DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
            .iter()
            .any(|option| option.full_key() == "datafusion_nexus.native.pipeline_execution_mode"
                && option.value_type == "enum: strict_bounded|adaptive")
    );
    assert!(
        DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
            .iter()
            .any(|option| option.full_key() == "datafusion_nexus.request.requested_grant_min_bytes"
                && option.value_type == "positive integer bytes or none")
    );
    assert!(
        DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
            .iter()
            .any(|option| option.full_key() == "datafusion_nexus.request.requested_grant_bytes"
                && option.value_type == "positive integer bytes or none")
    );
    assert!(
        DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
            .iter()
            .any(|option| option.full_key() == "datafusion_nexus.request.graph_raw_input_edges"
                && option.value_type == "positive integer edges or none")
    );
    assert!(
        DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
            .iter()
            .all(|option| option.full_key() != ["datafusion_nexus.request.graph_topology_", "family"].concat())
    );
}

#[tokio::test]
async fn datafusion_nexus_session_config_can_be_set_through_sql() {
    let state = SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    ctx.sql("SET datafusion_nexus.native.memory_policy = 'unbounded'").await.unwrap().collect().await.unwrap();

    let state = ctx.state();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("extension remains registered after SQL SET");
    assert_eq!(extension.native_memory_policy, NativeMemoryPolicyConfig::Unbounded);
}

#[tokio::test]
async fn datafusion_nexus_request_config_can_be_set_and_reset_through_sql() {
    let state = SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);

    ctx.sql("SET datafusion_nexus.request.requested_grant_min_bytes = '1024'").await.unwrap().collect().await.unwrap();
    ctx.sql("SET datafusion_nexus.request.requested_grant_bytes = '1536'").await.unwrap().collect().await.unwrap();
    ctx.sql("SET datafusion_nexus.request.graph_raw_input_edges = '45564149'").await.unwrap().collect().await.unwrap();
    let state = ctx.state();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("extension remains registered after SQL SET");
    assert_eq!(extension.request_requested_grant_min_bytes, Some(1024));
    assert_eq!(extension.request_requested_grant_bytes, Some(1536));
    assert_eq!(extension.request_graph_raw_input_edges(), Some(45_564_149));

    ctx.sql("SET datafusion_nexus.request.requested_grant_min_bytes = 'none'").await.unwrap().collect().await.unwrap();
    ctx.sql("SET datafusion_nexus.request.requested_grant_bytes = 'none'").await.unwrap().collect().await.unwrap();
    ctx.sql("SET datafusion_nexus.request.graph_raw_input_edges = 'none'").await.unwrap().collect().await.unwrap();
    let state = ctx.state();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("extension remains registered after SQL reset");
    assert_eq!(extension.request_requested_grant_min_bytes, None);
    assert_eq!(extension.request_requested_grant_bytes, None);
    assert_eq!(extension.request_graph_raw_input_edges(), None);
}

#[tokio::test]
async fn datafusion_nexus_removed_graph_topology_config_key_is_rejected_through_sql() {
    let state = SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();
    let ctx = SessionContext::new_with_state(state);
    let removed_key = ["datafusion_nexus.request.graph_topology_", "family"].concat();

    let error = ctx.sql(&format!("SET {removed_key} = 'citation_like_sparse'")).await.unwrap_err();

    assert!(error.to_string().contains(&removed_key["datafusion_nexus.".len()..]));
    assert!(error.to_string().contains("not found"));
}

fn options_with_nexus_extension(config: &NexusNativeOptimizerConfig) -> ConfigOptions {
    let mut options = ConfigOptions::default();
    options.extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(config));
    options
}

fn adapter_error(error: &DataFusionError) -> &Error {
    match error {
        DataFusionError::External(source) => source.downcast_ref::<Error>().expect("expected adapter error"),
        other => panic!("expected adapter error, got {other:?}"),
    }
}
