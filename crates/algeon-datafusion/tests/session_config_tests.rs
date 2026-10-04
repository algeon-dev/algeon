use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
    planner::{AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule},
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};

#[test]
fn algeon_session_builder_does_not_register_session_config_extension() {
    let state = SessionStateBuilder::new().with_algeon_native(AlgeonNativeOptimizerConfig::default()).build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("algeon_datafusion.")));
}

#[test]
fn backend_session_does_not_register_session_config_extension() {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder().devices([device]).build().unwrap();
    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("algeon_datafusion.")));
}

#[tokio::test]
async fn sql_set_algeon_datafusion_keys_rejected_as_unrecognized() {
    let ctx = SessionContext::new();

    let error = ctx.sql("SET algeon_datafusion.native.max_source_chunk_bytes = '2048'").await.unwrap_err();
    assert!(error.to_string().contains("not found") || error.to_string().contains("algeon_datafusion"));

    let error = ctx.sql("SET algeon_datafusion.request.graph_raw_input_edges = '45564149'").await.unwrap_err();
    assert!(error.to_string().contains("not found") || error.to_string().contains("algeon_datafusion"));
}

#[test]
fn optimizer_rule_retains_configured_native_optimizer_config() {
    let config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::new(
        SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"),
    ));
    let state = SessionStateBuilder::new().with_algeon_native(config.clone()).build();

    let rule = state
        .physical_optimizers()
        .iter()
        .find_map(|rule| {
            let rule = rule.as_ref() as &dyn std::any::Any;
            rule.downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
        })
        .expect("algeon native optimizer rule must be installed");
    assert_eq!(
        rule.config().native_execution_config().source_chunking().max_source_chunk_bytes.get(),
        config.native_execution_config().source_chunking().max_source_chunk_bytes.get()
    );
    assert_eq!(
        rule.config().native_execution_config().source_chunking().max_row_groups_per_chunk.get(),
        config.native_execution_config().source_chunking().max_row_groups_per_chunk.get()
    );
    assert_eq!(
        rule.config().native_execution_config().source_chunking().max_retry_attempts.get(),
        config.native_execution_config().source_chunking().max_retry_attempts.get()
    );
    assert_eq!(
        rule.config().native_execution_config().max_source_reads_in_flight(),
        config.native_execution_config().max_source_reads_in_flight()
    );
}

#[test]
fn execution_policy_configured_via_native_execution_config() {
    use query_runtime::runtime::SourceSchedulerPolicy;

    let execution_config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 8, 5, 1).expect("valid source chunking"))
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(4));
    assert_eq!(execution_config.source_chunking().max_source_chunk_bytes.get(), 512);
    assert_eq!(execution_config.source_chunking().max_row_groups_per_chunk.get(), 8);
    assert_eq!(execution_config.source_chunking().max_retry_attempts.get(), 5);
    assert_eq!(execution_config.max_source_reads_in_flight(), 4);

    let optimizer_config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(execution_config);
    assert_eq!(optimizer_config.native_execution_config().source_chunking().max_source_chunk_bytes.get(), 512);
    assert_eq!(optimizer_config.native_execution_config().source_chunking().max_row_groups_per_chunk.get(), 8);
    assert_eq!(optimizer_config.native_execution_config().source_chunking().max_retry_attempts.get(), 5);
    assert_eq!(optimizer_config.native_execution_config().max_source_reads_in_flight(), 4);
}
