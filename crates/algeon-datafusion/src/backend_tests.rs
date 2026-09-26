use std::sync::Arc;

use datafusion::{
    catalog::Session,
    common::Result as DataFusionResult,
    execution::{SessionStateBuilder, context::QueryPlanner},
    logical_expr::LogicalPlan,
    physical_plan::ExecutionPlan,
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
};
use query_engine::{
    error::{QueryError, QueryErrorCode},
    optimizer::NativeOptimizerOptions,
    plan::NativePlanCostOptions,
    runtime::DeviceCapability,
};

use super::*;
use crate::{planner::DataFusionCpuExecutionPolicy, session::installed_algeon_native_optimizer_config};

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
    let AlgeonGpuBackendBuilder { optimizer_config, devices, queue_policy, .. } = builder;
    super::validated_backend_construction(optimizer_config, devices, queue_policy)
}

#[test]
fn backend_rejects_empty_devices_declaration() {
    let missing = validated(AlgeonGpuBackend::builder().devices([])).unwrap_err();
    assert_eq!(missing.code(), ErrorCode::InvalidRequest);
    assert_eq!(missing.fact_value("field"), Some("devices"));
    assert_eq!(missing.fact_value("reason"), Some("missing_devices"));
}

#[test]
fn backend_rejects_duplicate_devices_declaration() {
    let duplicate = validated(AlgeonGpuBackend::builder().devices([
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
            GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 1 },
        ),
    ]))
    .unwrap_err();
    assert_eq!(duplicate.code(), ErrorCode::InvalidRequest);
    assert_eq!(duplicate.fact_value("field"), Some("devices"));
    assert_eq!(duplicate.fact_value("reason"), Some("duplicate_device"));
    assert_eq!(duplicate.fact_value("device_ordinal"), Some("0"));
}

#[test]
fn backend_device_config_preserves_static_partitions_and_installation_bounds() {
    let devices = [
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
                .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(4, 3, 2)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
        AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(2)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 })
                .with_max_active_attempts(1)
                .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(2, 2, 1)),
            GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 },
        ),
    ];
    let backend = validated(AlgeonGpuBackend::builder().devices(devices.clone())).unwrap();
    assert_eq!(backend.devices, devices);
    assert_eq!(backend.devices[0].memory_ownership(), GpuMemoryOwnership::WholeDeviceExclusive);
    assert_eq!(
        backend.devices[1].memory_ownership(),
        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 }
    );
    assert_eq!(
        backend.devices[0].profile().installation_bounds(),
        Some(AlgeonGpuDeviceInstallationBounds::new(4, 3, 2))
    );
    assert_eq!(
        backend.devices[1].profile().installation_bounds(),
        Some(AlgeonGpuDeviceInstallationBounds::new(2, 2, 1))
    );
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
    assert_eq!(backend.devices, devices);
    // Validation preserves the configured envelopes but does not resolve a
    // runtime query limit or manufacture a grant.
    assert_eq!(backend.devices[0].profile().default_query_limit_bytes(), Some(512));
    assert_eq!(
        backend.optimizer_config.native_execution_config().source_chunking(),
        SourceChunking::try_new(1024, 1, 3, 1).expect("valid source chunking")
    );
}

#[test]
fn backend_queue_policy_defaults_to_canonical_bounds() {
    let backend = validated(bounded_backend_builder(&[0, 1])).unwrap();
    assert_eq!(backend.queue_policy, AlgeonGpuQueuePolicy::default());
    assert!(backend.devices.iter().all(|device| device.profile().max_active_attempts() == 2));
}

#[test]
fn backend_default_device_profile_needs_no_ownership_for_defaults() {
    let profile = AlgeonGpuDeviceProfile::new(3);

    assert_eq!(profile.device_ordinal(), 3);
    assert_eq!(profile.default_query_limit_bytes(), None);
    assert_eq!(profile.source_work_per_attempt(), 1);
    assert_eq!(profile.max_active_attempts(), 2);
    assert_eq!(profile.installation_bounds(), None);
}

#[test]
fn backend_profile_binding_preserves_each_ordinal_local_envelope() {
    let capabilities = DeviceCapabilities::only(Vec::<DeviceCapability>::new());
    let first = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
            .with_backend_reserve_bytes(128)
            .with_default_query_limit_bytes(512)
            .with_source_work_per_attempt(2)
            .with_max_active_attempts(2)
            .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(6, 3, 2))
            .with_device_capabilities(capabilities.clone())
            .with_telemetry(false),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let second = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(3)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 16384 })
            .with_backend_reserve_bytes(256)
            .with_default_query_limit_bytes(1024)
            .with_source_work_per_attempt(1)
            .with_max_active_attempts(1)
            .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(2, 2, 1))
            .with_telemetry(true),
        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 },
    );

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(first.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 });
    assert_eq!(first.profile().backend_reserve_bytes(), 128);
    assert_eq!(first.default_query_limit_bytes(), Some(512));
    assert_eq!(first.source_work_per_attempt(), 2);
    assert_eq!(first.max_active_attempts(), 2);
    assert_eq!((first.stream_pool_size(), first.worker_count(), first.lane_count()), (6, 3, 2));
    assert_eq!(first.profile().device_capabilities(), &capabilities);

    assert_eq!(second.device_ordinal(), 3);
    assert_eq!(second.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 16384 });
    assert_eq!(second.profile().backend_reserve_bytes(), 256);
    assert_eq!(second.default_query_limit_bytes(), Some(1024));
    assert_eq!(second.source_work_per_attempt(), 1);
    assert_eq!(second.max_active_attempts(), 1);
    assert_eq!((second.stream_pool_size(), second.worker_count(), second.lane_count()), (2, 2, 1));
    assert_eq!(second.memory_ownership(), GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 });
    assert_ne!(first.profile().device_capabilities(), second.profile().device_capabilities());
}

#[test]
fn backend_rejects_invalid_queue_policy_bounds() {
    for policy in [
        AlgeonGpuQueuePolicy::default().with_max_queued_attempts(0),
        AlgeonGpuQueuePolicy::default().with_max_queue_wait(std::time::Duration::ZERO),
    ] {
        let error = validated(bounded_backend_builder(&[0]).queue_policy(policy)).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidRequest);
    }
}

#[test]
fn backend_validation_preserves_explicit_query_limit_on_each_device_profile() {
    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 100 })
            .with_max_active_attempts(1)
            .with_default_query_limit_bytes(60),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend = validated(
        AlgeonGpuBackend::builder()
            .with_native_execution_config(AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64, 1, 1, 1).expect("valid source chunking"),
            ))
            .devices([device]),
    )
    .unwrap();
    assert_eq!(backend.devices[0].profile().default_query_limit_bytes(), Some(60));
}

#[test]
fn backend_builder_preserves_full_optimizer_config_and_policy() {
    let native_execution_config =
        AlgeonNativeExecutionConfig::new(SourceChunking::try_new(1024, 2, 4, 2).expect("valid source chunking"));
    let optimizer_options = NativeOptimizerOptions::disabled();
    let cost_options = NativePlanCostOptions {
        enabled: true,
        max_small_selective_row_groups: 42,
        ..NativePlanCostOptions::disabled()
    };
    let optimizer_config = AlgeonNativeOptimizerConfig::default()
        .with_native_execution_config(native_execution_config.clone())
        .with_native_optimizer_options(optimizer_options.clone())
        .with_native_plan_cost_options(cost_options)
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);

    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend =
        validated(AlgeonGpuBackend::builder().with_optimizer_config(optimizer_config).devices([device])).unwrap();

    assert_eq!(
        backend.optimizer_config.native_execution_config().source_chunking(),
        native_execution_config.source_chunking()
    );
    assert_eq!(backend.optimizer_config.datafusion_cpu_execution_policy(), DataFusionCpuExecutionPolicy::Reject);
    assert_eq!(backend.optimizer_config.native_optimizer_options(), &optimizer_options);
    assert_eq!(backend.optimizer_config.native_plan_cost_options(), &cost_options);
}

#[test]
fn backend_install_installs_optimizer_rule_matching_backend_config() {
    let backend = validated(bounded_backend_builder(&[0])).unwrap();
    let state =
        super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config).unwrap().build();
    let installed = installed_algeon_native_optimizer_config(&state).unwrap();
    assert_eq!(installed.datafusion_cpu_execution_policy(), backend.optimizer_config.datafusion_cpu_execution_policy());
    assert_eq!(installed.native_optimizer_options(), backend.optimizer_config.native_optimizer_options());
    assert_eq!(installed.native_plan_cost_options(), backend.optimizer_config.native_plan_cost_options());
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
fn backend_installs_configured_datafusion_cpu_policy() {
    let backend = validated(
        bounded_backend_builder(&[0]).with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject),
    )
    .unwrap();
    assert_eq!(backend.optimizer_config.datafusion_cpu_execution_policy(), DataFusionCpuExecutionPolicy::Reject);

    let state =
        super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config).unwrap().build();
    let installed = installed_algeon_native_optimizer_config(&state).unwrap();
    assert_eq!(installed.datafusion_cpu_execution_policy(), DataFusionCpuExecutionPolicy::Reject);
}

#[test]
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

#[test]
fn backend_install_on_preserves_caller_custom_query_planner() {
    let backend = InstalledTestGpuBackend::new();
    let builder = SessionStateBuilder::new().with_query_planner(Arc::new(SentinelQueryPlanner));
    let state = backend.install_on(builder).expect("backend install_on must succeed with custom query planner").build();
    let any = state.query_planner().as_ref() as &dyn std::any::Any;
    assert!(any.is::<SentinelQueryPlanner>(), "installed session must retain the custom query planner");
    assert!(backend.backend().is_session_state_bound(&state));
}

#[test]
fn backend_can_install_independent_custom_planner_and_cuvs_sessions() {
    use crate::session::AlgeonSessionStateBuilderExt;

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

#[test]
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
        session::AlgeonSessionStateBuilderExt,
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
    let conflicting =
        CugraphSqlConfig { default_graph_options: CugraphGraphOptions { directed: false, ..Default::default() } };
    let error = builder.try_with_cugraph_sql(conflicting).unwrap_err();
    let typed = error.find_algeon_error().expect("structured error");
    assert_eq!(typed.code(), ErrorCode::InvalidRequest);
    assert_eq!(typed.operation(), "algeon_datafusion.gpu_functions.install");
    assert!(typed.message().contains("conflicting GPU function family installation"));
}

#[tokio::test]
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
    assert_eq!(error.fact_value("reason"), Some("missing_devices"));
}

#[test]
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
        QueryErrorCode::InvalidPlan,
        "query_engine.gpu.device_registry.capacity",
        "explicit managed capacity cannot exceed owned device memory",
    )
    .with_fact("device_ordinal", 0_usize)
    .with_fact("configured_limit_bytes", 118111600640_u64)
    .with_fact("owned_capacity_bytes", 101973819392_u64)
    .with_fact("reason", "explicit_capacity_exceeds_owned_capacity");
    let error = super::backend_core_build_error("failed to initialize ownership-enforcing GPU backend core", native);
    let display = error.to_string();
    assert!(display.contains("explicit_capacity_exceeds_owned_capacity"), "{display}");
    assert!(display.contains("owned_capacity_bytes=101973819392"), "{display}");
    assert!(display.contains("configured_limit_bytes=118111600640"), "{display}");
}
