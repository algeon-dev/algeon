use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use datafusion_nexus::{
    backend::{GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceConfig, NexusGpuDeviceProfile},
    native_exec::NexusNativeExecutionConfig,
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};

#[test]
fn nexus_session_builder_does_not_register_session_config_extension() {
    let state = SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("datafusion_nexus.")));
}

#[test]
fn backend_session_does_not_register_session_config_extension() {
    let device = NexusGpuDeviceConfig::new(NexusGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = NexusGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("datafusion_nexus.")));
}

#[tokio::test]
async fn sql_set_datafusion_nexus_keys_rejected_as_unrecognized() {
    let ctx = SessionContext::new();

    let error = ctx.sql("SET datafusion_nexus.native.max_source_chunk_bytes = '2048'").await.unwrap_err();
    assert!(error.to_string().contains("not found") || error.to_string().contains("datafusion_nexus"));

    let error = ctx.sql("SET datafusion_nexus.request.graph_raw_input_edges = '45564149'").await.unwrap_err();
    assert!(error.to_string().contains("not found") || error.to_string().contains("datafusion_nexus"));
}

#[test]
fn optimizer_rule_retains_configured_native_optimizer_config() {
    let config = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1));
    let state = SessionStateBuilder::new().with_nexus_native(config.clone()).build();

    let rule = state
        .physical_optimizers()
        .iter()
        .find_map(|rule| {
            let rule = rule.as_ref() as &dyn std::any::Any;
            rule.downcast_ref::<NexusNativePhysicalOptimizerRule>()
        })
        .expect("nexus native optimizer rule must be installed");
    assert_eq!(
        rule.config().native_execution_config().max_source_chunk_bytes(),
        config.native_execution_config().max_source_chunk_bytes()
    );
    assert_eq!(
        rule.config().native_execution_config().max_row_groups_per_chunk(),
        config.native_execution_config().max_row_groups_per_chunk()
    );
    assert_eq!(
        rule.config().native_execution_config().max_retry_attempts(),
        config.native_execution_config().max_retry_attempts()
    );
    assert_eq!(
        rule.config().native_execution_config().max_source_reads_in_flight(),
        config.native_execution_config().max_source_reads_in_flight()
    );
}

#[test]
fn execution_policy_configured_via_native_execution_config() {
    use nexus_query_engine::runtime::SourceSchedulerPolicy;

    let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 8, 5, 1)
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(4));
    assert_eq!(execution_config.max_source_chunk_bytes(), 512);
    assert_eq!(execution_config.max_row_groups_per_chunk(), 8);
    assert_eq!(execution_config.max_retry_attempts(), 5);
    assert_eq!(execution_config.max_source_reads_in_flight(), 4);

    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config);
    assert_eq!(optimizer_config.native_execution_config().max_source_chunk_bytes(), 512);
    assert_eq!(optimizer_config.native_execution_config().max_row_groups_per_chunk(), 8);
    assert_eq!(optimizer_config.native_execution_config().max_retry_attempts(), 5);
    assert_eq!(optimizer_config.native_execution_config().max_source_reads_in_flight(), 4);
}
