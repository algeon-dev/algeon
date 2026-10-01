use std::{
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

use datafusion::{execution::SessionStateBuilder, prelude::SessionConfig};
use query_engine::observability::RuntimeObservationConfig;
use tokio::time::timeout;

use super::*;
use crate::server::{
    adapter_error::ErrorCode,
    backend::{
        AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    },
    native_exec::AlgeonNativeExecutionConfig,
    planner::{AlgeonNativeOptimizerConfig, DataFusionCpuExecutionPolicy},
};

const TEST_DEVICE_BUDGET_BYTES: u64 = 4096;
const TEST_MAX_SOURCE_CHUNK_BYTES: u64 = 1024;
const TEST_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
const TEST_MAX_RETRY_ATTEMPTS: usize = 3;
const TEST_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

#[test]
fn server_session_configuration_preserves_datafusion_settings() {
    let mut base = SessionConfig::new().with_target_partitions(16).with_repartition_joins(false);
    base.options_mut().optimizer.enable_dynamic_filter_pushdown = false;

    let session_config = super::session::enable_session_discovery(base);
    let options = session_config.options();

    assert_eq!(options.execution.target_partitions, 16);
    assert!(!options.optimizer.repartition_joins);
    assert!(!options.optimizer.enable_dynamic_filter_pushdown);
    assert!(options.catalog.information_schema);
}

fn base_config() -> ServerConfig {
    ServerConfig {
        bind_addr: "127.0.0.1:0".parse().expect("test bind address"),
        queue_policy: AlgeonGpuQueuePolicy::default(),
        gpu_device_profiles: vec![AlgeonGpuDeviceProfile::new(0)],
        gpu_devices: vec![0],
        gpu_memory_ownership: vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        statement_ticket_ttl: Duration::from_secs(60),
        observability: ServerObservabilityConfig::default(),
        native_optimizer_config: AlgeonNativeOptimizerConfig::default(),
        auth: AuthConfig::default(),
        workspace: None,
        cugraph: None,
        models: None,
    }
}

fn bounded_config(max_active_attempts_per_device: usize) -> ServerConfig {
    let mut config = base_config();
    config.native_optimizer_config =
        AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            TEST_DEVICE_BUDGET_BYTES,
            TEST_MAX_SOURCE_CHUNK_BYTES,
            TEST_MAX_ROW_GROUPS_PER_CHUNK,
            TEST_MAX_RETRY_ATTEMPTS,
            TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));
    config.gpu_device_profiles = vec![
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
            .with_max_active_attempts(max_active_attempts_per_device),
    ];
    config
}

fn runtime_backend(max_active_attempts_per_device: usize) -> AlgeonGpuBackend {
    runtime_backend_with_policy(max_active_attempts_per_device, DataFusionCpuExecutionPolicy::Allow)
}

fn strict_runtime_backend(max_active_attempts_per_device: usize) -> AlgeonGpuBackend {
    runtime_backend_with_policy(max_active_attempts_per_device, DataFusionCpuExecutionPolicy::Reject)
}

fn runtime_backend_with_policy(
    max_active_attempts_per_device: usize,
    datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy,
) -> AlgeonGpuBackend {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
        .with_max_active_attempts(max_active_attempts_per_device);
    AlgeonGpuBackend::builder()
        .devices([AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)])
        .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            TEST_DEVICE_BUDGET_BYTES,
            TEST_MAX_SOURCE_CHUNK_BYTES,
            TEST_MAX_ROW_GROUPS_PER_CHUNK,
            TEST_MAX_RETRY_ATTEMPTS,
            TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        ))
        .with_datafusion_cpu_execution_policy(datafusion_cpu_execution_policy)
        .build()
        .expect("runtime backend should build")
}

#[test]
fn shutdown_closes_native_admission_before_coordinator_terminalization() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let admission = admission::QueryAdmission::from_backend(&backend);
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
    ));
    let planning = coordinator
        .register_with_id("shutdown-order", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap();
    let activity_factory = planning.activity_factory();

    let coordinator_live = after_closing_admission(&admission, || {
        let error = backend
            .begin_attempt(attempt::AttemptOptions::default())
            .expect_err("native QueryService must be closed before coordinator shutdown begins");
        assert_eq!(error.code(), ErrorCode::Cancelled);
        let activity = activity_factory().expect("coordinator remains live until admission is closed");
        drop(activity);
        coordinator.begin_shutdown(Instant::now())
    });

    assert_eq!(coordinator_live.live_count(), 1);
    assert!(activity_factory().is_none(), "coordinator rejects new activity after terminalization");
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_activity_race_is_fallible_and_drain_waits_for_prior_activity() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let admission = Arc::new(admission::QueryAdmission::from_backend(&backend));
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
    ));
    let maintenance = coordinator::QueryCoordinatorMaintenance::start(Arc::clone(&coordinator));
    let execution = coordinator
        .register_with_id("shutdown-race", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap()
        .into_execution()
        .unwrap();
    let activity_factory = execution.activity_factory();
    let prior_activity = activity_factory().expect("activity acquired before shutdown");
    let barrier = Arc::new(Barrier::new(2));

    let (coordinator_live, racing_activity) = thread::scope(|scope| {
        let shutdown_admission = Arc::clone(&admission);
        let shutdown_coordinator = Arc::clone(&coordinator);
        let shutdown_barrier = Arc::clone(&barrier);
        let shutdown = scope.spawn(move || {
            after_closing_admission(&shutdown_admission, || {
                shutdown_barrier.wait();
                shutdown_coordinator.begin_shutdown(Instant::now())
            })
        });

        barrier.wait();
        let error = backend
            .begin_attempt(attempt::AttemptOptions::default())
            .expect_err("native admission is closed before the coordinator race begins");
        assert_eq!(error.code(), ErrorCode::Cancelled);
        let racing_activity = activity_factory();
        (shutdown.join().unwrap(), racing_activity)
    });

    assert_eq!(coordinator_live.live_count(), 1);
    drop(execution);
    let mut finish = tokio::spawn(maintenance.finish(coordinator_live));
    assert!(
        timeout(Duration::from_millis(20), &mut finish).await.is_err(),
        "maintenance cannot drain while pre-shutdown activity is live"
    );
    drop(racing_activity);
    assert!(
        timeout(Duration::from_millis(20), &mut finish).await.is_err(),
        "racing acquisition cannot release the prior activity"
    );
    drop(prior_activity);
    assert!(!timeout(Duration::from_secs(1), finish).await.unwrap().unwrap().unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_supervisor_reports_coordinator_deadline_expiry_as_error() {
    let admission = Arc::new(admission::QueryAdmission::new_for_tests());
    let observability =
        ServerObservabilityConfig { shutdown_grace: Duration::from_millis(20), ..ServerObservabilityConfig::default() };
    let coordinator =
        coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(&observability));
    let execution = coordinator
        .register_with_id(
            "supervisor-forced-timeout",
            "select 1",
            None,
            None,
            RuntimeObservationConfig::bounded(16, 4096),
        )
        .unwrap()
        .into_execution()
        .unwrap();
    let activity = execution.activity_factory()().expect("activity lease");
    let maintenance = coordinator::QueryCoordinatorMaintenance::start(Arc::clone(&coordinator));
    let mut supervisor = ShutdownSupervisor::start(
        admission,
        Arc::new(io_lifecycle::ServerIoLifecycle::new()),
        Arc::clone(&coordinator),
        maintenance,
    );

    assert_eq!(supervisor.begin_shutdown(Instant::now()).live_count(), 1);
    let error = timeout(Duration::from_secs(1), supervisor.finish())
        .await
        .expect("supervisor must use the configured deadline")
        .expect_err("forced coordinator timeout must fail server shutdown");

    assert_eq!(error.operation(), "algeon_datafusion.server.shutdown");
    assert_eq!(error.fact_value("reason"), Some("coordinator_shutdown_timeout"));
    drop(activity);
    drop(execution);
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_supervisor_propagates_native_admission_cleanup_timeout() {
    let mut config = bounded_config(1);
    config.observability.shutdown_grace = Duration::from_millis(20);
    let backend = runtime_backend(1);
    let _live_attempt = backend.begin_attempt(attempt::AttemptOptions::default()).unwrap();
    let admission = Arc::new(admission::QueryAdmission::from_backend(&backend));
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
    ));
    let maintenance = coordinator::QueryCoordinatorMaintenance::start(Arc::clone(&coordinator));
    let mut supervisor = ShutdownSupervisor::start(
        admission,
        Arc::new(io_lifecycle::ServerIoLifecycle::new()),
        coordinator,
        maintenance,
    );

    assert_eq!(supervisor.begin_shutdown(Instant::now()).live_count(), 0);
    let error = timeout(Duration::from_secs(1), supervisor.finish())
        .await
        .expect("supervisor must not grant admission a second grace period")
        .expect_err("native admission cleanup timeout must fail server shutdown");

    assert_eq!(error.fact_value("reason"), Some("backend_close_timeout"));
    assert_eq!(error.fact_value("shutdown_component"), Some("query_admission"));
    drop(_live_attempt);
    let snapshot = backend
        .close_and_wait(Duration::from_secs(1))
        .expect("releasing the held query must allow the already-closing backend to quiesce");
    assert!(snapshot.gpu_resource_quiescent);
    assert!(snapshot.observation_quiescent);
    assert_eq!(snapshot.active_gpu_permits, 0);
    assert_eq!(snapshot.active_observation_permits, 0);
    assert_eq!(snapshot.cleanup_late_success_count, snapshot.cleanup_deadline_elapsed_count);
}

#[tokio::test(flavor = "current_thread")]
async fn supervisor_uses_captured_expired_plan_after_coordinator_finalization_clears_deadline() {
    let admission = Arc::new(admission::QueryAdmission::new_for_tests());
    let observability =
        ServerObservabilityConfig { shutdown_grace: Duration::ZERO, ..ServerObservabilityConfig::default() };
    let coordinator =
        coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(&observability));
    let execution = coordinator
        .register_with_id("fast-finalization", "select 1", None, None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap()
        .into_execution()
        .unwrap();
    let plan = coordinator.begin_shutdown(Instant::now());
    assert_eq!(plan.live_count(), 1);
    drop(execution);
    assert!(coordinator.is_drained());
    assert!(coordinator.shutdown_deadline().is_none(), "fast finalization clears mutable coordinator state");

    let maintenance = coordinator::QueryCoordinatorMaintenance::start(Arc::clone(&coordinator));
    let io_lifecycle = Arc::new(io_lifecycle::ServerIoLifecycle::new());
    let mut supervisor = ShutdownSupervisor::start_with_plan(
        Arc::clone(&admission),
        Arc::clone(&io_lifecycle),
        coordinator,
        maintenance,
        plan,
    );
    let error = supervisor.finish().await.expect_err("an already-expired captured plan must time out cleanup");

    assert_eq!(error.fact_value("reason"), Some("kvikio_drain_timeout"));
    assert_eq!(io_lifecycle.drain_attempts(), 1, "I/O cleanup is not skipped when coordinator state is gone");
    assert_eq!(
        io_lifecycle.drain_invocations(),
        1,
        "an expired idle phase still performs the mandatory one-shot KvikIO drain"
    );
    assert!(admission.try_acquire().is_err(), "admission cleanup closes the gate under the captured plan");
}

#[tokio::test]
async fn builder_rejects_session_context_without_injected_backend() {
    let config = bounded_config(1);
    let ctx = Arc::new(SessionContext::new());

    let error = builder_error(FlightSqlServer::builder().config(config).session_context(ctx)).await;

    assert_eq!(error.fact_value("reason"), Some("server_session_without_backend"));
}

#[test]
fn builder_rejects_injected_backend_admission_conflict() {
    let backend = runtime_backend(1);

    let error = validate_injected_backend_matches_config(&backend, &bounded_config(2)).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles.max_active_attempts"));
}

#[test]
fn builder_rejects_injected_backend_default_query_limit_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_default_query_limit_bytes(256);

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles.default_query_limit_bytes"));
}

#[test]
fn builder_rejects_injected_backend_per_device_limit_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_max_active_attempts(2);

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles.max_active_attempts"));
}

#[test]
fn builder_accepts_matching_explicit_per_device_cap() {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
        .with_max_active_attempts(1);
    let backend = AlgeonGpuBackend::builder()
        .devices([AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)])
        .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            TEST_DEVICE_BUDGET_BYTES,
            TEST_MAX_SOURCE_CHUNK_BYTES,
            TEST_MAX_ROW_GROUPS_PER_CHUNK,
            TEST_MAX_RETRY_ATTEMPTS,
            TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        ))
        .build()
        .expect("injected backend should build");
    let config = bounded_config(1);

    validate_injected_backend_matches_config(&backend, &config).unwrap();
}

#[test]
fn builder_rejects_injected_backend_queue_limit_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.queue_policy = config.queue_policy.with_max_queued_attempts(3);

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued_attempts"));
}

#[test]
fn builder_rejects_injected_backend_wait_timeout_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.queue_policy = config.queue_policy.with_max_queue_wait(Duration::from_secs(5));

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queue_wait_ms"));
}

#[test]
fn builder_rejects_injected_backend_gpu_device_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.gpu_devices = vec![1];
    config.gpu_device_profiles = vec![
        AlgeonGpuDeviceProfile::new(1)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES }),
    ];
    config.gpu_memory_ownership = vec![(1, GpuMemoryOwnership::WholeDeviceExclusive)];

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles"));
}

#[test]
fn injected_backend_validation_does_not_conflate_planner_template_with_profile_capacity() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.native_optimizer_config =
        AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            TEST_DEVICE_BUDGET_BYTES / 2,
            TEST_MAX_SOURCE_CHUNK_BYTES,
            TEST_MAX_ROW_GROUPS_PER_CHUNK,
            TEST_MAX_RETRY_ATTEMPTS,
            TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));

    validate_injected_backend_matches_config(&backend, &config)
        .expect("planner template bytes are not a backend capacity or floor assertion");
}

#[tokio::test]
async fn build_session_context_with_backend_installs_backend_admission_source() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);

    let ctx = build_session_context_with_backend(&config, &backend).await.unwrap();

    assert!(backend.is_session_state_bound(&ctx.state()));
}

#[tokio::test]
async fn strict_injected_backend_is_the_session_policy_authority() {
    let mut caller_config = bounded_config(1);
    caller_config.native_optimizer_config = AlgeonNativeOptimizerConfig::default();
    let backend = strict_runtime_backend(1);

    let ctx = build_session_context_with_backend(&caller_config, &backend).await.unwrap();

    assert!(backend.is_session_state_bound(&ctx.state()));
    assert_eq!(ctx.state().config_options().execution.target_partitions, 1);
}

#[tokio::test]
async fn builder_rejects_strict_backend_session_with_incompatible_target_partitions() {
    let mut caller_config = bounded_config(1);
    caller_config.native_optimizer_config = AlgeonNativeOptimizerConfig::default();
    let backend = strict_runtime_backend(1);
    let ctx = build_session_context_with_backend(&caller_config, &backend).await.unwrap();
    let incompatible_config = ctx.state().config().clone().with_target_partitions(8);
    let incompatible_state =
        SessionStateBuilder::new_from_existing(ctx.state()).with_config(incompatible_config).build();
    let incompatible_ctx = Arc::new(SessionContext::new_with_state(incompatible_state));

    let error = builder_error(
        FlightSqlServer::builder().config(caller_config).backend(backend).session_context(incompatible_ctx),
    )
    .await;

    assert_eq!(error.code(), ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "algeon_datafusion.server.builder");
    assert_eq!(error.fact_value("reason"), Some("server_session_optimizer_policy_mismatch"));
    assert_eq!(error.fact_value("config_field"), Some("datafusion.execution.target_partitions"));
    assert_eq!(error.fact_value("expected_value"), Some("1"));
    assert_eq!(error.fact_value("actual_value"), Some("8"));
}

#[tokio::test]
async fn server_base_and_flight_style_clone_retain_cuvs_query_planner() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let ctx = build_session_context_with_backend(&config, &backend).await.unwrap();

    assert!(format!("{:?}", ctx.state().query_planner()).contains("AlgeonQueryPlanner"));
    let cloned = SessionStateBuilder::new_from_existing(ctx.state()).build();
    assert!(format!("{:?}", cloned.query_planner()).contains("AlgeonQueryPlanner"));
}

#[test]
fn builder_rejects_injected_session_context_from_different_backend() {
    let error = validate_injected_session_backend_binding(false).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_session_backend_mismatch"));
}

async fn builder_error(builder: FlightSqlServerBuilder) -> ServerError {
    match builder.build().await {
        Ok(_) => panic!("builder should reject invalid configuration"),
        Err(error) => error,
    }
}

#[test]
fn distinct_librmm_mappings_dedupes_segments_and_matches_versioned_sonames() {
    let maps = "7f0a1c000000-7f0a1c2bc000 r-xp 00000000 103:07 4584938  /opt/rapids/lib/librmm.so\n\
                7f0a1c2bc000-7f0a1c2c0000 rw-p 002bb000 103:07 4584938  /opt/rapids/lib/librmm.so\n\
                7f0a1d000000-7f0a1d100000 r-xp 00000000 103:07 4584939  /opt/other/librmm.so.26.06\n\
                7f0a1e000000-7f0a1e100000 r-xp 00000000 103:07 4584940  /usr/lib/libstdc++.so.6\n";

    let mappings = distinct_librmm_mappings(maps);

    assert_eq!(mappings, vec!["/opt/rapids/lib/librmm.so".to_string(), "/opt/other/librmm.so.26.06".to_string()]);
}

#[test]
fn distinct_librmm_mappings_counts_deleted_and_ignores_lookalikes() {
    let maps = "7f0a1c000000-7f0a1c2bc000 r-xp 00000000 103:07 4584938  /opt/rapids/lib/librmm.so (deleted)\n\
                7f0a1d000000-7f0a1d100000 r-xp 00000000 103:07 4584939  /opt/other/librmm_helpers.so\n\
                7f0a1e000000-7f0a1e100000 rw-p 00000000 00:00 0\n";

    let mappings = distinct_librmm_mappings(maps);

    assert_eq!(mappings, vec!["/opt/rapids/lib/librmm.so".to_string()]);
}

#[test]
fn single_librmm_runtime_mapping_check_passes_in_this_process() {
    ensure_single_librmm_runtime_mapping()
        .expect("this test process must not map two librmm.so builds; a failure here is a real regression");
}
