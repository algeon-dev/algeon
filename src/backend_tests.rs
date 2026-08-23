use datafusion::{common::config::ExtensionOptions, execution::SessionStateBuilder, prelude::SessionConfig};
use nexus_query_engine::{optimizer::NativeOptimizerOptions, plan::NativePlanCostOptions};

use super::*;
use crate::{
    planner::FinalPlanRequirement, session::installed_nexus_native_optimizer_config,
    session_config::DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS,
};

fn backend_builder(device_ordinals: &[usize]) -> NexusGpuBackendBuilder {
    let devices = device_ordinals.iter().copied().map(|ordinal| {
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(ordinal).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )
    });
    NexusGpuBackend::builder().devices(devices)
}

fn bounded_execution_config(device_budget_bytes: u64) -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(device_budget_bytes, 1024, 1, 3, 1)
}

fn bounded_backend_builder(device_ordinals: &[usize]) -> NexusGpuBackendBuilder {
    backend_builder(device_ordinals).with_native_execution_config(bounded_execution_config(1024))
}

fn validated(builder: NexusGpuBackendBuilder) -> Result<super::ValidatedBackendConstruction> {
    let NexusGpuBackendBuilder { optimizer_config, devices, queue_policy, .. } = builder;
    super::validated_backend_construction(optimizer_config, devices, queue_policy)
}

#[test]
fn backend_rejects_empty_devices_declaration() {
    let missing = validated(NexusGpuBackend::builder().devices([])).unwrap_err();
    assert_eq!(missing.code(), ErrorCode::InvalidRequest);
    assert_eq!(missing.fact_value("field"), Some("devices"));
    assert_eq!(missing.fact_value("reason"), Some("missing_devices"));
}

#[test]
fn backend_rejects_duplicate_devices_declaration() {
    let duplicate = validated(NexusGpuBackend::builder().devices([
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
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
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
                .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(4, 2, 2)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(2)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 })
                .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(2, 1, 1)),
            GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 },
        ),
    ];
    let backend = validated(NexusGpuBackend::builder().devices(devices.clone())).unwrap();
    assert_eq!(backend.devices, devices);
    assert_eq!(backend.devices[0].memory_ownership(), GpuMemoryOwnership::WholeDeviceExclusive);
    assert_eq!(
        backend.devices[1].memory_ownership(),
        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 }
    );
    assert_eq!(
        backend.devices[0].profile().installation_bounds(),
        Some(NexusGpuDeviceInstallationBounds::new(4, 2, 2))
    );
    assert_eq!(
        backend.devices[1].profile().installation_bounds(),
        Some(NexusGpuDeviceInstallationBounds::new(2, 1, 1))
    );
}

#[test]
fn backend_validation_projection_keeps_device_configs_independent() {
    let devices = vec![
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
                .with_attempt_floor_bytes(Some(512))
                .with_source_work_per_attempt(1)
                .with_max_active_attempts(2)
                .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(4, 2, 2))
                .with_telemetry(false),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
        NexusGpuDeviceConfig::new(
            NexusGpuDeviceProfile::new(1)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 })
                .with_backend_reserve_bytes(256)
                .with_attempt_floor_bytes(Some(1024))
                .with_source_work_per_attempt(3)
                .with_max_active_attempts(1)
                .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(4, 1, 1)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        ),
    ];
    let policy = NexusGpuQueuePolicy::default().with_max_queued_attempts(8);
    let backend = validated(
        NexusGpuBackend::builder()
            .with_native_execution_config(bounded_execution_config(1024))
            .devices(devices.clone())
            .queue_policy(policy.clone()),
    )
    .unwrap();

    assert_eq!(backend.queue_policy.max_queued_attempts(), 8);
    assert_eq!(backend.devices, devices);
    // Validation preserves the configured envelopes but does not resolve a
    // runtime floor or manufacture a grant.
    assert_eq!(backend.devices[0].profile().attempt_floor_bytes(), Some(512));
    assert_eq!(
        backend.optimizer_config.native_execution_config().memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    );
}

#[test]
fn backend_queue_policy_defaults_to_canonical_bounds() {
    let backend = validated(bounded_backend_builder(&[0, 1])).unwrap();
    assert_eq!(backend.queue_policy, NexusGpuQueuePolicy::default());
    assert!(backend.devices.iter().all(|device| device.profile().max_active_attempts() == 1));
}

#[test]
fn backend_default_device_profile_projects_engine_defaults_by_construction() {
    let ownership = GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 };

    assert_eq!(
        NexusGpuDeviceProfile::new(3).engine_config(ownership),
        nexus_query_engine::runtime::GpuBackendDeviceConfig::new(3, ownership),
        "the root profile must project the engine's automatic floor, source-work, and active-attempt defaults"
    );
}

#[test]
fn backend_profile_projection_preserves_each_ordinal_local_envelope() {
    let capabilities = DeviceCapabilities::only(Vec::<nexus_query_engine::runtime::DeviceCapability>::new());
    let first = NexusGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
        .with_backend_reserve_bytes(128)
        .with_attempt_floor_bytes(Some(512))
        .with_source_work_per_attempt(2)
        .with_max_active_attempts(2)
        .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(6, 2, 2))
        .with_device_capabilities(capabilities.clone())
        .with_telemetry(false)
        .engine_config(GpuMemoryOwnership::WholeDeviceExclusive);
    let second = NexusGpuDeviceProfile::new(3)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 16384 })
        .with_backend_reserve_bytes(256)
        .with_attempt_floor_bytes(Some(1024))
        .with_source_work_per_attempt(1)
        .with_max_active_attempts(1)
        .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(2, 1, 1))
        .with_telemetry(true)
        .engine_config(GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 });

    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(first.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 });
    assert_eq!(first.backend_reserve_bytes(), 128);
    assert_eq!(first.attempt_floor_bytes(), Some(512));
    assert_eq!(first.source_work_per_attempt(), 2);
    assert_eq!(first.max_active_attempts(), 2);
    assert_eq!((first.stream_pool_size(), first.worker_count(), first.lane_count()), (6, 2, 2));
    assert_eq!(first.device_capabilities(), &capabilities);

    assert_eq!(second.device_ordinal(), 3);
    assert_eq!(second.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 16384 });
    assert_eq!(second.backend_reserve_bytes(), 256);
    assert_eq!(second.attempt_floor_bytes(), Some(1024));
    assert_eq!(second.source_work_per_attempt(), 1);
    assert_eq!(second.max_active_attempts(), 1);
    assert_eq!((second.stream_pool_size(), second.worker_count(), second.lane_count()), (2, 1, 1));
    assert_eq!(second.memory_ownership(), GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 });
    assert_ne!(first.device_capabilities(), second.device_capabilities());
}

#[test]
fn backend_rejects_invalid_queue_policy_bounds() {
    for policy in [
        NexusGpuQueuePolicy::default().with_max_queued_attempts(0),
        NexusGpuQueuePolicy::default().with_max_overtakes_per_waiter(0),
        NexusGpuQueuePolicy::default().with_max_queue_wait(std::time::Duration::ZERO),
    ] {
        let error = validated(bounded_backend_builder(&[0]).queue_policy(policy)).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidRequest);
    }
}

#[test]
fn backend_validation_preserves_explicit_floor_on_each_device_profile() {
    let device = NexusGpuDeviceConfig::new(
        NexusGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 100 })
            .with_max_active_attempts(1)
            .with_attempt_floor_bytes(Some(60)),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend = validated(
        NexusGpuBackend::builder()
            .with_native_execution_config(NexusNativeExecutionConfig::bounded(100, 64, 1, 1, 1))
            .devices([device]),
    )
    .unwrap();
    assert_eq!(backend.devices[0].profile().attempt_floor_bytes(), Some(60));
}

#[test]
fn backend_builder_preserves_full_optimizer_config_and_requirements() {
    let native_execution_config = NexusNativeExecutionConfig::bounded(4096, 1024, 2, 4, 2);
    let optimizer_options = NativeOptimizerOptions::disabled();
    let cost_options = NativePlanCostOptions {
        enabled: true,
        max_small_selective_row_groups: 42,
        ..NativePlanCostOptions::disabled()
    };
    let optimizer_config = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(native_execution_config.clone())
        .with_native_optimizer_options(optimizer_options.clone())
        .with_native_plan_cost_options(cost_options)
        .with_final_plan_requirement(FinalPlanRequirement::NoDataFusionCpu);

    let device = NexusGpuDeviceConfig::new(
        NexusGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 }),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend =
        validated(NexusGpuBackend::builder().with_optimizer_config(optimizer_config).devices([device])).unwrap();

    assert_eq!(
        backend.optimizer_config.native_execution_config().memory_control_policy(),
        native_execution_config.memory_control_policy()
    );
    assert_eq!(backend.optimizer_config.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
    assert_eq!(backend.optimizer_config.native_optimizer_options(), &optimizer_options);
    assert_eq!(backend.optimizer_config.native_plan_cost_options(), &cost_options);
}

#[test]
fn backend_install_freezes_native_session_keys() {
    let backend = validated(bounded_backend_builder(&[0])).unwrap();
    let state =
        super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config).unwrap().build();
    let extension = state
        .config_options()
        .extensions
        .get::<DataFusionNexusSessionConfig>()
        .expect("backend install must register datafusion-nexus session config");
    assert!(extension.is_backend_admission_frozen());

    for option in DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS {
        let mut extension = extension.clone();
        if option.key.starts_with("request.") {
            extension.set(option.key, "1").unwrap();
            continue;
        }
        let error = extension.set(option.key, "1").unwrap_err();
        assert!(error.to_string().contains(&option.full_key()));
        assert!(error.to_string().contains("query-service admission"));
    }
}

#[test]
fn backend_installs_configured_final_plan_requirements() {
    let backend = validated(bounded_backend_builder(&[0]).with_no_datafusion_cpu_requirement()).unwrap();
    assert_eq!(backend.optimizer_config.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);

    let state =
        super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config).unwrap().build();
    let installed = installed_nexus_native_optimizer_config(&state).unwrap();
    assert_eq!(installed.final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);
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
        SessionStateBuilder::new().with_nexus_native(NexusNativeOptimizerConfig::default()).build();
    assert!(
        !backend.backend().is_session_state_bound(&planning_only_state),
        "an installed optimizer without admission authority is not bound"
    );
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
fn backend_install_rejects_conflicting_preexisting_session_extension() {
    let backend = validated(bounded_backend_builder(&[0])).unwrap();
    let mut config = SessionConfig::new();
    config
        .options_mut()
        .extensions
        .insert(DataFusionNexusSessionConfig::from_optimizer_config(&NexusNativeOptimizerConfig::default()));
    let error = super::install_backend_optimizer_config(
        SessionStateBuilder::new().with_config(config),
        &backend.optimizer_config,
    )
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("reason"), Some("session_config_conflicts_with_backend_admission"));
}

#[test]
fn backend_install_on_with_cugraph_sql_preserves_conflicting_reinstall_error() {
    use crate::cugraph_sql::{CugraphGraphOptions, CugraphSqlConfig};

    let backend = validated(bounded_backend_builder(&[0])).unwrap();
    let mut installed = super::install_cugraph_sql_on_installed_backend(
        super::install_backend_optimizer_config(SessionStateBuilder::new(), &backend.optimizer_config).unwrap(),
        &backend.optimizer_config,
        CugraphSqlConfig::default(),
    )
    .unwrap();
    // Carry the installed catalog (including the cugraph family) into a fresh
    // builder whose table functions are empty, so the conflicting family
    // projection — not the preexisting-UDTF admission gate — is the failure.
    let builder = SessionStateBuilder::new().with_config(installed.config().as_ref().unwrap().clone());
    let conflicting = CugraphSqlConfig {
        default_graph_options: CugraphGraphOptions { directed: false, ..Default::default() },
        ..Default::default()
    };
    let error =
        super::install_cugraph_sql_on_installed_backend(builder, &backend.optimizer_config, conflicting).unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "datafusion_nexus.gpu_functions.install");
    assert!(error.message().contains("conflicting GPU function family installation"));
}

#[cfg(feature = "qualification")]
#[test]
fn backend_qualification_quarantine_rejects_unknown_ordinal_with_adapter_operation() {
    let backend = InstalledTestGpuBackend::new();

    let error = backend
        .backend()
        .quarantine_device_for_qualification(usize::MAX)
        .expect_err("quarantine must reject an ordinal with no installed ledger");
    assert_eq!(error.operation(), "datafusion_nexus.backend.quarantine_device_for_qualification");
    assert!(error.fact_value("native_code").is_some());
}
