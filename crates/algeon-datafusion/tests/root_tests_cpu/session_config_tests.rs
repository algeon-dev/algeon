use algeon_datafusion::{
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
    planner::{AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule},
    session::AlgeonSessionStateBuilderExt,
};
use datafusion::execution::SessionStateBuilder;
#[test]
fn algeon_session_builder_does_not_register_session_config_extension() {
    let state = SessionStateBuilder::new().with_algeon_native(AlgeonNativeOptimizerConfig::default()).build();

    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("algeon_datafusion.")));
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
