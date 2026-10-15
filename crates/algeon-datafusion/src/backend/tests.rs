use std::sync::Arc;

use cudf::testing;
use datafusion::{
    catalog::Session,
    common::Result as DataFusionResult,
    execution::{SessionStateBuilder, context::QueryPlanner},
    logical_expr::LogicalPlan,
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
};
use query_planning::{
    error::{QueryError, QueryErrorCode},
    optimizer::NativeOptimizerOptions,
};
use tokio::runtime::Builder as TestRuntimeBuilder;

use super::*;
use crate::{
    error::{ErrorCode, ErrorKind},
    planner::AlgeonGpuExecutionMode,
    session::{AlgeonSessionStateBuilderExt, installed_algeon_native_optimizer_config},
};

fn backend_builder(device_ordinals: &[usize]) -> AlgeonGpuBackendBuilder {
    let devices = device_ordinals.iter().copied().map(|ordinal| {
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )
    });
    AlgeonGpuBackend::builder().devices(devices)
}

fn bounded_execution_config() -> AlgeonNativeExecutionConfig {
    AlgeonNativeExecutionConfig::new(SourceChunking::try_new(1024, 1, 3, 1).expect("valid source chunking"))
}

fn bounded_backend_builder(device_ordinals: &[usize]) -> AlgeonGpuBackendBuilder {
    backend_builder(device_ordinals).with_native_execution_config(bounded_execution_config())
}

fn validated(builder: AlgeonGpuBackendBuilder) -> Result<super::ValidatedBackendConstruction> {
    let AlgeonGpuBackendBuilder { optimizer_config, devices, queue_policy, attempt_completion_queue_capacity, .. } =
        builder;
    super::validated_backend_construction(optimizer_config, devices, queue_policy, attempt_completion_queue_capacity)
}

fn validated_devices(backend: &super::ValidatedBackendConstruction) -> Vec<AlgeonGpuDeviceConfig> {
    backend.core_config.devices().cloned().collect()
}

#[test]
fn backend_validation_projection_keeps_device_configs_independent() {
    let devices = vec![
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
                .with_default_query_limit_bytes(512)
                .with_source_work_per_attempt(1)
                .with_max_active_attempts(2)
                .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(4, 3, 2))
                .with_telemetry(false),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(1)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 })
                .with_backend_reserve_bytes(256)
                .with_default_query_limit_bytes(1024)
                .with_source_work_per_attempt(3)
                .with_max_active_attempts(1)
                .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(4, 2, 1)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
    ];
    let policy = AlgeonGpuQueuePolicy::default().with_max_queued_attempts(8);
    let backend = validated(
        AlgeonGpuBackend::builder()
            .with_native_execution_config(bounded_execution_config())
            .devices(devices.clone())
            .queue_policy(policy),
    )
    .unwrap();

    assert_eq!(backend.queue_policy.max_queued_attempts(), 8);
    assert_eq!(validated_devices(&backend), devices);
    // Validation preserves the configured envelopes but does not resolve a
    // runtime query limit or manufacture a grant.
    assert_eq!(validated_devices(&backend)[0].profile().default_query_limit_bytes(), Some(512));
    assert_eq!(
        backend.optimizer_config.native_execution_config().source_chunking(),
        SourceChunking::try_new(1024, 1, 3, 1).expect("valid source chunking")
    );
}

#[test]
fn backend_rejects_invalid_queue_policy_bounds() {
    for policy in [
        AlgeonGpuQueuePolicy::default().with_max_queued_attempts(0),
        AlgeonGpuQueuePolicy::default().with_max_queue_wait(std::time::Duration::ZERO),
    ] {
        let error = validated(bounded_backend_builder(&[0]).queue_policy(policy)).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    }
}

#[test]
fn backend_install_installs_optimizer_rule_matching_backend_config() {
    let native_execution_config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(1024, 2, 4, 2).expect("valid source chunking"));
    let optimizer_options = NativeOptimizerOptions::disabled();
    let optimizer_config = AlgeonNativeOptimizerConfig::default()
        .with_native_execution_config(native_execution_config.clone())
        .with_native_optimizer_options(optimizer_options.clone())
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired);

    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend =
        validated(AlgeonGpuBackend::builder().with_optimizer_config(optimizer_config).devices([device])).unwrap();

    let state =
        super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config).unwrap().build();
    let installed = installed_algeon_native_optimizer_config(&state).unwrap();
    assert_eq!(installed.execution_mode(), AlgeonGpuExecutionMode::NativeRequired);
    assert_eq!(installed.native_optimizer_options(), &optimizer_options);
    assert_eq!(installed.native_execution_config().source_chunking(), native_execution_config.source_chunking());
    assert_eq!(installed.native_optimizer_options(), backend.optimizer_config.native_optimizer_options());
    assert_eq!(
        installed.native_execution_config().source_chunking().max_source_chunk_bytes.get(),
        backend.optimizer_config.native_execution_config().source_chunking().max_source_chunk_bytes.get()
    );
    assert_eq!(
        installed.native_execution_config().source_chunking().max_row_groups_per_chunk.get(),
        backend.optimizer_config.native_execution_config().source_chunking().max_row_groups_per_chunk.get()
    );
    assert_eq!(
        installed.native_execution_config().source_chunking().max_retry_attempts.get(),
        backend.optimizer_config.native_execution_config().source_chunking().max_retry_attempts.get()
    );
    assert_eq!(
        installed.native_execution_config().max_source_reads_in_flight(),
        backend.optimizer_config.native_execution_config().max_source_reads_in_flight()
    );
    assert!(!state.config_options().entries().iter().any(|entry| entry.key.starts_with("algeon_datafusion.")));
}

#[test]
fn tests_cases_1() {
    testing::run_named_cases(&[
        (
            "backend_session_binding_predicate_tracks_installed_backend_identity",
            backend_session_binding_predicate_tracks_installed_backend_identity,
        ),
        (
            "backend_can_install_independent_custom_planner_and_cuvs_sessions",
            backend_can_install_independent_custom_planner_and_cuvs_sessions,
        ),
        (
            "backend_session_binding_predicate_rejects_sessions_from_another_backend",
            backend_session_binding_predicate_rejects_sessions_from_another_backend,
        ),
        ("backend_async_close_returns_the_sync_close_snapshot_contract", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(backend_async_close_returns_the_sync_close_snapshot_contract())
        }),
        (
            "backend_probe_device_capacity_reports_live_cuda_totals",
            backend_probe_device_capacity_reports_live_cuda_totals,
        ),
    ]);
}

fn backend_session_binding_predicate_tracks_installed_backend_identity() {
    let backend = InstalledTestGpuBackend::new();

    let bound_state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
    assert!(
        backend.backend().is_session_state_bound(&bound_state),
        "a session installed by this backend is bound to it"
    );

    let uninstalled_state = SessionStateBuilder::new().build();
    assert!(
        !backend.backend().is_session_state_bound(&uninstalled_state),
        "a session without the native optimizer is not bound"
    );

    let planning_only_state =
        SessionStateBuilder::new().with_algeon_native(AlgeonNativeOptimizerConfig::default()).build();
    assert!(
        !backend.backend().is_session_state_bound(&planning_only_state),
        "an installed optimizer without admission authority is not bound"
    );
}

#[derive(Debug)]
struct SentinelQueryPlanner;

#[async_trait::async_trait]
impl QueryPlanner for SentinelQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::default().create_physical_plan(logical_plan, session).await
    }
}

fn backend_can_install_independent_custom_planner_and_cuvs_sessions() {
    let backend = InstalledTestGpuBackend::new();
    let core_builder = SessionStateBuilder::new().with_query_planner(Arc::new(SentinelQueryPlanner));
    let core_state = backend.install_on(core_builder).expect("core install succeeds").build();
    let any = core_state.query_planner().as_ref() as &dyn std::any::Any;
    assert!(any.is::<SentinelQueryPlanner>());
    assert!(backend.backend().is_session_state_bound(&core_state));

    let cuvs_builder = SessionStateBuilder::new_with_default_features();
    let cuvs_builder = backend.install_on(cuvs_builder).expect("backend install succeeds");
    let cuvs_state = cuvs_builder.try_with_cuvs_sql().expect("cuVS install succeeds").build();
    assert!(backend.backend().is_session_state_bound(&cuvs_state));
}

fn backend_session_binding_predicate_rejects_sessions_from_another_backend() {
    let first = InstalledTestGpuBackend::new();
    let first_state = first.install_on(SessionStateBuilder::new()).unwrap().build();
    drop(first);

    let second = InstalledTestGpuBackend::new();
    assert!(
        !second.backend().is_session_state_bound(&first_state),
        "a session installed by another backend is not bound"
    );
}

#[test]
fn backend_install_with_cugraph_sql_preserves_conflicting_reinstall_error() {
    use crate::{
        cugraph_sql::{CugraphGraphOptions, CugraphSqlConfig},
        error::DataFusionErrorExt,
    };

    let backend = validated(bounded_backend_builder(&[0])).unwrap();
    let mut installed = super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config)
        .unwrap()
        .try_with_cugraph_sql(CugraphSqlConfig::default())
        .unwrap();
    // Carry the installed catalog (including the cugraph family) into a fresh
    // builder whose table functions are empty, so the conflicting family
    // projection is the failure.
    let builder = SessionStateBuilder::new().with_config(installed.config().as_ref().unwrap().clone());
    let conflicting = CugraphSqlConfig {
        default_graph_options: CugraphGraphOptions { directed: false, ..Default::default() },
        ..Default::default()
    };
    let error = builder.try_with_cugraph_sql(conflicting).unwrap_err();
    let typed = error.find_algeon_error().expect("structured error");
    assert_eq!(typed.code(), ErrorCode::InvalidRequest);
    assert_eq!(typed.operation(), "algeon_datafusion.gpu_functions.install");
    assert!(typed.message().contains("conflicting GPU function family installation"));
}

async fn backend_async_close_returns_the_sync_close_snapshot_contract() {
    let backend = InstalledTestGpuBackend::new();

    let snapshot = backend
        .backend()
        .close_and_wait_async(std::time::Duration::from_secs(10))
        .await
        .expect("async close waits for backend quiescence");
    assert!(snapshot.gpu_resource_quiescent);
    assert!(snapshot.observation_quiescent);
}

#[cfg(feature = "qualification")]
#[test]
fn backend_qualification_quarantine_rejects_unknown_ordinal_with_adapter_operation() {
    let backend = InstalledTestGpuBackend::new();

    let error = backend
        .backend()
        .quarantine_device_for_qualification(usize::MAX)
        .expect_err("quarantine must reject an ordinal with no installed ledger");
    assert_eq!(error.operation(), "algeon_datafusion.backend.quarantine_device_for_qualification");
    assert!(error.fact_value("native_code").is_some());
}

#[test]
fn backend_builder_validate_accepts_consistent_declaration_without_gpu_resources() {
    bounded_backend_builder(&[0]).validate().expect("consistent declaration passes dry-run validation");
}

#[test]
fn backend_builder_validate_rejects_empty_devices() {
    let error = AlgeonGpuBackend::builder().devices([]).validate().expect_err("dry-run must reject empty devices");
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    assert_eq!(error.fact_value("field"), Some("device_ordinals"));
    assert_eq!(error.fact_value("reason"), Some("empty"));
}

fn backend_probe_device_capacity_reports_live_cuda_totals() {
    let probe = AlgeonGpuBackend::probe_device_capacity(0, GpuMemoryOwnership::WholeDeviceExclusive, 0)
        .expect("GPU lane provides device 0 for capacity probing");
    assert!(probe.cuda_total_bytes > 0);
    assert!(probe.cuda_free_bytes <= probe.cuda_total_bytes);
    assert_eq!(probe.owned_capacity_bytes, probe.cuda_total_bytes);
    assert_eq!(probe.max_declarable_limit_bytes, probe.cuda_total_bytes);
    assert_eq!(probe.max_managed_capacity_bytes, probe.cuda_total_bytes);
}

#[test]
fn backend_core_build_error_surfaces_capacity_cause_in_display() {
    let native = QueryError::new(
        QueryErrorCode::InvalidConfiguration,
        "query_engine.gpu.device_registry.capacity",
        "explicit managed capacity cannot exceed owned device memory",
    )
    .with_fact("device_ordinal", 0_usize)
    .with_fact("configured_limit_bytes", 118111600640_u64)
    .with_fact("owned_capacity_bytes", 101973819392_u64)
    .with_fact("reason", "explicit_capacity_exceeds_owned_capacity");
    let error = super::backend_core_build_error("failed to initialize ownership-enforcing GPU backend core", native);
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    let display = error.to_string();
    assert!(display.contains("explicit_capacity_exceeds_owned_capacity"), "{display}");
    assert!(display.contains("owned_capacity_bytes=101973819392"), "{display}");
    assert!(display.contains("configured_limit_bytes=118111600640"), "{display}");
}

#[test]
fn backend_builder_validate_states_the_setting_value_limit_and_largest_accepted_value() {
    const GIB: u64 = 1 << 30;
    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 100 * GIB }),
        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 95 * GIB },
    );

    let error = AlgeonGpuBackend::builder()
        .with_native_execution_config(bounded_execution_config())
        .devices([device])
        .validate()
        .expect_err("a limit above the static partition is a configuration error");

    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(
        error.message().contains(
            "device 0 memory_limit_bytes is 107374182400 bytes (100.00 GiB), above the static partition of 102005473280 bytes (95.00 GiB); the largest accepted value is 102005473280 bytes (95.00 GiB)"
        ),
        "{}",
        error.message()
    );
}
