use datafusion::{common::config::ExtensionOptions, execution::SessionStateBuilder, prelude::SessionConfig};

use super::*;
use crate::{
    planner::FinalPlanRequirement, session::installed_nexus_native_optimizer_config,
    session_config::DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS,
};

fn backend_builder(device_ordinals: &[usize]) -> NexusGpuBackendBuilder {
    NexusGpuBackend::builder().device_memory_ownership(
        device_ordinals.iter().copied().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)),
    )
}

fn explicit_profiles(device_ordinals: &[usize], limit_bytes: u64) -> Vec<NexusGpuDeviceProfile> {
    device_ordinals
        .iter()
        .copied()
        .map(|ordinal| {
            NexusGpuDeviceProfile::new(ordinal).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes })
        })
        .collect()
}

fn bounded_execution_config(device_budget_bytes: u64) -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(device_budget_bytes, 1024, 1, 3, 1)
}

fn bounded_backend_builder(device_ordinals: &[usize]) -> NexusGpuBackendBuilder {
    backend_builder(device_ordinals)
        .with_native_execution_config(bounded_execution_config(1024))
        .device_profiles(explicit_profiles(device_ordinals, 4096))
}

#[test]
fn backend_requires_an_exact_device_ownership_declaration_before_engine_build() {
    let missing = NexusGpuBackend::builder().device_profiles(explicit_profiles(&[0], 4096)).build().unwrap_err();
    assert_eq!(missing.code(), ErrorCode::InvalidRequest);
    assert_eq!(missing.fact_value("field"), Some("device_memory_ownership"));
    assert_eq!(missing.fact_value("reason"), Some("missing_devices"));

    let extra = NexusGpuBackend::builder()
        .device_profiles(explicit_profiles(&[0], 4096))
        .device_memory_ownership([
            (0, GpuMemoryOwnership::WholeDeviceExclusive),
            (1, GpuMemoryOwnership::WholeDeviceExclusive),
        ])
        .build()
        .unwrap_err();
    assert_eq!(extra.fact_value("reason"), Some("extra_devices"));

    let duplicate = NexusGpuBackend::builder()
        .device_profiles(explicit_profiles(&[0], 4096))
        .device_memory_ownership([
            (0, GpuMemoryOwnership::WholeDeviceExclusive),
            (0, GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 1 }),
        ])
        .build()
        .unwrap_err();
    assert_eq!(duplicate.fact_value("reason"), Some("duplicate_device"));
}

#[test]
fn backend_exact_ownership_validation_preserves_static_partitions() {
    let validated = validate_device_memory_ownership(
        &[0, 2],
        &[
            (2, GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 }),
            (0, GpuMemoryOwnership::WholeDeviceExclusive),
        ],
    )
    .unwrap();
    assert_eq!(validated[&0], GpuMemoryOwnership::WholeDeviceExclusive);
    assert_eq!(validated[&2], GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8 * 1024 });
}

#[test]
fn backend_validation_projection_keeps_device_profiles_independent() {
    let profiles = vec![
        NexusGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 4096 })
            .with_attempt_floor_bytes(Some(512))
            .with_source_work_per_attempt(1)
            .with_max_active_attempts(2)
            .with_installation_bounds(4, 2, 2)
            .with_telemetry(false),
        NexusGpuDeviceProfile::new(1)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 8192 })
            .with_backend_reserve_bytes(256)
            .with_attempt_floor_bytes(Some(1024))
            .with_source_work_per_attempt(3)
            .with_max_active_attempts(1)
            .with_installation_bounds(4, 1, 1),
    ];
    let policy = NexusGpuQueuePolicy::default().with_max_queued_attempts(8);
    let backend = NexusGpuBackend::builder()
        .device_memory_ownership([
            (0, GpuMemoryOwnership::WholeDeviceExclusive),
            (1, GpuMemoryOwnership::WholeDeviceExclusive),
        ])
        .with_native_execution_config(bounded_execution_config(1024))
        .device_profiles(profiles.clone())
        .queue_policy(policy.clone())
        .build_test_validation_projection()
        .unwrap();

    assert_eq!(backend.queue_policy().max_queued_attempts(), 8);
    assert_eq!(backend.device_ordinals(), &[0, 1]);
    assert_eq!(backend.device_profiles(), profiles.as_slice());
    // Validation preserves the configured envelopes but does not resolve a
    // runtime floor or manufacture a grant.
    assert_eq!(backend.device_profiles()[0].attempt_floor_bytes(), Some(512));
    assert_eq!(
        backend.optimizer_config().native_execution_config().memory_control_policy(),
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
    let backend = bounded_backend_builder(&[0, 1]).build_test_validation_projection().unwrap();
    assert_eq!(backend.queue_policy(), &NexusGpuQueuePolicy::default());
    assert!(backend.device_profiles().iter().all(|profile| profile.max_active_attempts() == 1));
}

#[test]
fn backend_default_device_profile_projects_engine_defaults_by_construction() {
    let ownership = GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 };

    assert_eq!(
        NexusGpuDeviceProfile::new(3).engine_config_for_test(ownership),
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
        .with_installation_bounds(6, 2, 2)
        .with_device_capabilities(capabilities.clone())
        .with_telemetry(false)
        .engine_config_for_test(GpuMemoryOwnership::WholeDeviceExclusive);
    let second = NexusGpuDeviceProfile::new(3)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 16384 })
        .with_backend_reserve_bytes(256)
        .with_attempt_floor_bytes(Some(1024))
        .with_source_work_per_attempt(1)
        .with_max_active_attempts(1)
        .with_installation_bounds(2, 1, 1)
        .with_telemetry(true)
        .engine_config_for_test(GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 8192 });

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
        let error = bounded_backend_builder(&[0]).queue_policy(policy).build_test_validation_projection().unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidRequest);
    }
}

#[test]
fn backend_validation_preserves_explicit_floor_on_each_device_profile() {
    let backend = backend_builder(&[0])
        .with_native_execution_config(NexusNativeExecutionConfig::bounded(100, 64, 1, 1, 1))
        .device_profiles(vec![
            explicit_profiles(&[0], 100)[0].clone().with_max_active_attempts(1).with_attempt_floor_bytes(Some(60)),
        ])
        .build_test_validation_projection()
        .unwrap();
    assert_eq!(backend.device_profiles()[0].attempt_floor_bytes(), Some(60));
}

#[test]
fn installed_backend_resolves_explicit_floor_in_the_real_device_ledger() {
    let limit_bytes = 256 * 1024 * 1024;
    let floor_bytes = 64 * 1024 * 1024;
    let backend = InstalledTestGpuBackend::with_config(
        NexusNativeExecutionConfig::bounded(limit_bytes, 64 * 1024 * 1024, 2, 3, 1),
        NexusGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes })
            .with_attempt_floor_bytes(Some(floor_bytes)),
        NexusGpuQueuePolicy::default(),
    );

    let handle = backend.admit_query().expect("real backend should issue an immutable grant");
    assert!(handle.device_budget_bytes() >= floor_bytes);
    assert!(handle.device_budget_bytes() <= limit_bytes);
    handle.finish();
}

#[test]
fn backend_requires_explicit_device_profiles() {
    let error = NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .build_test_validation_projection()
        .unwrap_err();

    assert_eq!(error.fact_value("field"), Some("device_profiles"));
    assert_eq!(error.fact_value("reason"), Some("missing_device_profiles"));
}

#[test]
fn backend_rejects_duplicate_device_profiles() {
    let duplicate = NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .device_profiles(explicit_profiles(&[0, 0], 4096))
        .build_test_validation_projection()
        .unwrap_err();
    assert_eq!(duplicate.fact_value("field"), Some("device_profiles"));
    assert_eq!(duplicate.fact_value("reason"), Some("duplicate_device"));
    assert_eq!(duplicate.fact_value("device_ordinal"), Some("0"));
}

#[test]
fn backend_install_freezes_native_session_keys() {
    let backend = bounded_backend_builder(&[0]).build_test_validation_projection().unwrap();
    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
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
    let backend =
        bounded_backend_builder(&[0]).with_no_datafusion_cpu_requirement().build_test_validation_projection().unwrap();
    assert_eq!(backend.optimizer_config().final_plan_requirements(), &[FinalPlanRequirement::NoDataFusionCpu]);

    let state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
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
    let backend = bounded_backend_builder(&[0]).build_test_validation_projection().unwrap();
    let mut config = SessionConfig::new();
    config
        .options_mut()
        .extensions
        .insert(DataFusionNexusSessionConfig::from_optimizer_config(&NexusNativeOptimizerConfig::default()));
    let error = backend.install_on(SessionStateBuilder::new().with_config(config)).unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.fact_value("reason"), Some("session_config_conflicts_with_backend_admission"));
}

#[test]
fn backend_install_on_with_cugraph_sql_preserves_conflicting_reinstall_error() {
    use crate::cugraph_sql::{CugraphGraphOptions, CugraphSqlConfig};

    let backend = bounded_backend_builder(&[0]).build_test_validation_projection().unwrap();
    let mut installed =
        backend.install_on_with_cugraph_sql(SessionStateBuilder::new(), CugraphSqlConfig::default()).unwrap();
    // Carry the installed catalog (including the cugraph family) into a fresh
    // builder whose table functions are empty, so the conflicting family
    // projection — not the preexisting-UDTF admission gate — is the failure.
    let builder = SessionStateBuilder::new().with_config(installed.config().as_ref().unwrap().clone());
    let conflicting = CugraphSqlConfig {
        default_graph_options: CugraphGraphOptions { directed: false, ..Default::default() },
        ..Default::default()
    };
    let error = backend.install_on_with_cugraph_sql(builder, conflicting).unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "datafusion_nexus.gpu_functions.install");
    assert!(error.message().contains("conflicting GPU function family installation"));
}
