use datafusion::{
    common::config::ConfigOptions,
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_exec::NexusNativeExecutionConfig,
    planner::NexusNativeOptimizerConfig,
    session::NexusSessionStateBuilderExt,
    session_config::{DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS, DataFusionNexusSessionConfig},
};
use nexus_query_engine::{
    mode::StatisticalAggregateExecutionMode,
    runtime::{MemoryControlPolicy, SourceChunkPlanningMode},
};

#[test]
fn nexus_session_builder_registers_datafusion_nexus_config_extension() {
    let state = SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();

    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("with_nexus_native must register datafusion_nexus config options");

    assert!(extension.native_max_source_chunk_bytes() > 0);
}

#[test]
fn datafusion_nexus_session_config_applies_native_bounded_options() {
    let base = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1));
    let mut options = options_with_nexus_extension(&base);

    options.set("datafusion_nexus.native.max_source_chunk_bytes", "1024").unwrap();
    options.set("datafusion_nexus.native.max_row_groups_per_chunk", "8").unwrap();
    options.set("datafusion_nexus.native.max_retry_attempts", "5").unwrap();
    options.set("datafusion_nexus.native.max_source_reads_in_flight", "4").unwrap();
    options.set("datafusion_nexus.native.source_chunk_planning_mode", "local_parquet_throughput").unwrap();
    options.set("datafusion_nexus.native.statistical_aggregate_execution_mode", "gpu_native_tolerant").unwrap();
    options.set("datafusion_nexus.request.graph_raw_input_edges", "45564149").unwrap();

    let effective = base.with_datafusion_config_options(&options).unwrap();
    let native = effective.native_execution_config();
    let projection = DataFusionNexusSessionConfig::from_optimizer_config(&effective);

    assert_eq!(projection.native_max_source_chunk_bytes(), 1024);
    assert_eq!(projection.native_max_row_groups_per_chunk(), 8);
    assert_eq!(projection.native_max_retry_attempts(), 5);
    assert_eq!(
        native.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            // Engine-owned values from the base config survive session overrides.
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 5,
            min_retry_chunk_row_groups: 1,
        }
    );
    assert_eq!(native.source_scheduler_policy().max_source_reads_in_flight(), 4);
    assert_eq!(native.source_chunk_planning_mode(), SourceChunkPlanningMode::LocalParquetThroughput);
    assert_eq!(native.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let graph_calibration = native.graph_memory_calibration_override();
    assert_eq!(graph_calibration.raw_input_edges, Some(45_564_149));
}

#[test]
fn datafusion_nexus_session_config_rejects_invalid_values() {
    let base = NexusNativeOptimizerConfig::default();
    let mut options = options_with_nexus_extension(&base);

    let error = options.set("datafusion_nexus.native.max_retry_attempts", "0").unwrap_err();
    assert!(error.to_string().contains("native.max_retry_attempts"));

    // The removed pipeline-mode key remains an unknown datafusion-nexus key.
    let error = options.set("datafusion_nexus.native.pipeline_execution_mode", "strict").unwrap_err();
    assert!(error.to_string().contains("pipeline_execution_mode"));

    let error = options.set("datafusion_nexus.native.memory_policy", "unbounded").unwrap_err();
    assert!(error.to_string().contains("memory_policy"));

    let error = options.set("datafusion_nexus.request.requested_grant_bytes", "1536").unwrap_err();
    assert!(error.to_string().contains("request.requested_grant_bytes"));

    let error = options.set("datafusion_nexus.request.graph_raw_input_edges", "0").unwrap_err();
    assert!(error.to_string().contains("request.graph_raw_input_edges"));

    let removed_key = ["datafusion_nexus.request.graph_topology_", "family"].concat();
    let error = options.set(&removed_key, "citation_like_sparse").unwrap_err();
    assert!(error.to_string().contains(&removed_key["datafusion_nexus.".len()..]));
    assert!(error.to_string().contains("not found"));
}

#[test]
fn datafusion_nexus_session_config_entries_are_discoverable() {
    let base = NexusNativeOptimizerConfig::default();
    let options = options_with_nexus_extension(&base);
    let entries = options.entries();
    let nexus_entries = entries.iter().filter(|entry| entry.key.starts_with("datafusion_nexus.")).collect::<Vec<_>>();

    assert!(!entries.iter().any(|entry| entry.key == "datafusion_nexus.native.memory_policy"));
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
    assert!(!bounded_keys.contains(&"native.statistical_aggregate_execution_mode"));
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
    let ctx = config_only_session_context(&NexusNativeOptimizerConfig::default());

    ctx.sql("SET datafusion_nexus.native.max_source_chunk_bytes = '2048'").await.unwrap().collect().await.unwrap();

    let state = ctx.state();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("extension remains registered after SQL SET");
    assert_eq!(extension.native_max_source_chunk_bytes(), 2048);
}

#[tokio::test]
async fn datafusion_nexus_graph_request_config_can_be_set_and_reset_through_sql() {
    let ctx = config_only_session_context(&NexusNativeOptimizerConfig::default());

    ctx.sql("SET datafusion_nexus.request.graph_raw_input_edges = '45564149'").await.unwrap().collect().await.unwrap();
    let state = ctx.state();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("extension remains registered after SQL SET");
    assert_eq!(extension.request_graph_raw_input_edges(), Some(45_564_149));

    ctx.sql("SET datafusion_nexus.request.graph_raw_input_edges = 'none'").await.unwrap().collect().await.unwrap();
    let state = ctx.state();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("extension remains registered after SQL reset");
    assert_eq!(extension.request_graph_raw_input_edges(), None);
}

#[tokio::test]
async fn datafusion_nexus_removed_graph_topology_config_key_is_rejected_through_sql() {
    let ctx = config_only_session_context(&NexusNativeOptimizerConfig::default());
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

fn config_only_session_context(config: &NexusNativeOptimizerConfig) -> SessionContext {
    let mut session_config = SessionConfig::new();
    session_config.options_mut().extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(config));
    SessionContext::new_with_config(session_config)
}
