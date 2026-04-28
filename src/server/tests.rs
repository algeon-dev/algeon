use std::{num::NonZeroUsize, sync::Barrier};

use datafusion::execution::SessionStateBuilder;
use nexus_query_engine::runtime::{QueryPlacementPolicy, SlotGuarantee};

use super::*;
use crate::{
    backend::{GpuMemoryOwnership, GrantBand, MemoryShare, NexusAdmissionPolicy, NexusMemoryPolicy},
    native_exec::{NativeMetricsCapture, NexusNativeExecutionConfig},
    planner::{
        HostExecutionQosConfig, NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME, NEXUS_HOST_EXECUTION_QOS_RULE_NAME,
        NexusNativeOptimizerConfig,
    },
    session::NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME,
};

const TEST_DEVICE_BUDGET_BYTES: u64 = 4096;
const TEST_MAX_SOURCE_CHUNK_BYTES: u64 = 1024;
const TEST_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
const TEST_MAX_RETRY_ATTEMPTS: usize = 3;
const TEST_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

#[test]
fn server_session_configuration_preserves_datafusion_settings() {
    let mut base = datafusion::prelude::SessionConfig::new().with_target_partitions(16).with_repartition_joins(false);
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
        log_filter: "info".to_string(),
        max_in_flight: 1,
        admission_policy: crate::backend::NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
        host_execution_qos: None,
        gpu_devices: vec![0],
        gpu_memory_ownership: vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        device_budget_fraction: None,
        statement_ticket_ttl: std::time::Duration::from_secs(60),
        observability: ServerObservabilityConfig::default(),
        native_optimizer_config: NexusNativeOptimizerConfig::default(),
        auth: auth::AuthConfig::default(),
        workspace: None,
        #[cfg(feature = "iceberg")]
        iceberg: None,
        cugraph: None,
    }
}

fn bounded_config(max_concurrent_queries: usize) -> ServerConfig {
    let mut config = base_config();
    config.max_in_flight = max_concurrent_queries;
    config.admission_policy = NexusAdmissionPolicy::default().with_max_concurrent_queries(max_concurrent_queries);
    config.native_optimizer_config =
        NexusNativeOptimizerConfig::default().with_native_execution_config(NexusNativeExecutionConfig::bounded(
            TEST_DEVICE_BUDGET_BYTES,
            TEST_MAX_SOURCE_CHUNK_BYTES,
            TEST_MAX_ROW_GROUPS_PER_CHUNK,
            TEST_MAX_RETRY_ATTEMPTS,
            TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));
    config
}

fn unbounded_config() -> ServerConfig {
    let mut config = base_config();
    config.max_in_flight = 1;
    config.admission_policy = NexusAdmissionPolicy::default().with_max_concurrent_queries(1);
    config.native_optimizer_config =
        NexusNativeOptimizerConfig::default().with_native_execution_config(NexusNativeExecutionConfig::unbounded());
    config
}

fn runtime_backend(max_concurrent_queries: usize) -> NexusGpuBackend {
    NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: TEST_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: TEST_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TEST_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: TEST_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        })
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(max_concurrent_queries))
        .build()
        .expect("runtime backend should build")
}

fn controller_projection(max_concurrent_queries: usize) -> crate::backend::TestBackendControllerProjection {
    NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: TEST_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: TEST_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TEST_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: TEST_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        })
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(max_concurrent_queries))
        .build_test_controller_projection()
        .expect("CUDA-free backend controller projection should build")
}

#[test]
fn shutdown_closes_native_admission_before_coordinator_terminalization() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let admission = admission::QueryAdmission::from_backend(Some(&backend), &config).unwrap();
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
    ));
    let planning = coordinator
        .register_with_id(
            "shutdown-order",
            "select 1",
            None,
            None,
            nexus_query_engine::observability::RuntimeObservationConfig::bounded(16, 4096),
        )
        .unwrap();
    let activity_factory = planning.activity_factory();

    let coordinator_live = after_closing_admission(&admission, || {
        let error = backend
            .try_admit_query()
            .expect_err("native QueryService must be closed before coordinator shutdown begins");
        assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("backend_closing"));
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
    let admission = Arc::new(admission::QueryAdmission::from_backend(Some(&backend), &config).unwrap());
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
    ));
    let maintenance = coordinator::QueryCoordinatorMaintenance::start(Arc::clone(&coordinator));
    let execution = coordinator
        .register_with_id(
            "shutdown-race",
            "select 1",
            None,
            None,
            nexus_query_engine::observability::RuntimeObservationConfig::bounded(16, 4096),
        )
        .unwrap()
        .into_execution()
        .unwrap();
    let observation = execution.observation_handle().unwrap();
    let capture = NativeMetricsCapture::coordinated("shutdown-race", observation)
        .with_observation_activity_factory(execution.activity_factory())
        .new_execution();
    let prior_activity = capture
        .new_observation_activity()
        .expect("coordinated capture always requires activity")
        .expect("activity acquired before shutdown");
    let barrier = Arc::new(Barrier::new(2));

    let (coordinator_live, racing_activity) = std::thread::scope(|scope| {
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
        let error =
            backend.try_admit_query().expect_err("native admission is closed before the coordinator race begins");
        assert_eq!(error.fact_value("reason"), Some("backend_closing"));
        let racing_activity = capture
            .new_observation_activity()
            .expect("coordinated acquisition cannot downgrade to no activity requirement");
        (shutdown.join().unwrap(), racing_activity)
    });

    assert_eq!(coordinator_live.live_count(), 1);
    drop(execution);
    let mut finish = tokio::spawn(maintenance.finish(coordinator_live));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut finish).await.is_err(),
        "maintenance cannot drain while pre-shutdown activity is live"
    );
    match racing_activity {
        Ok(activity) => drop(activity),
        Err(error) => {
            assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::Cancelled);
            assert_eq!(error.fact_value("reason"), Some("query_coordinator_activity_rejected"));
        },
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut finish).await.is_err(),
        "racing acquisition cannot release the prior activity"
    );
    drop(prior_activity);
    assert!(!tokio::time::timeout(std::time::Duration::from_secs(1), finish).await.unwrap().unwrap().unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_supervisor_reports_coordinator_deadline_expiry_as_error() {
    let admission = Arc::new(admission::QueryAdmission::new(1).unwrap());
    let observability = ServerObservabilityConfig {
        shutdown_grace: std::time::Duration::from_millis(20),
        ..ServerObservabilityConfig::default()
    };
    let coordinator =
        coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(&observability));
    let execution = coordinator
        .register_with_id(
            "supervisor-forced-timeout",
            "select 1",
            None,
            None,
            nexus_query_engine::observability::RuntimeObservationConfig::bounded(16, 4096),
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
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), supervisor.finish())
        .await
        .expect("supervisor must use the configured deadline")
        .expect_err("forced coordinator timeout must fail server shutdown");

    assert_eq!(error.operation(), "datafusion_nexus.server.shutdown");
    assert_eq!(error.fact_value("reason"), Some("coordinator_shutdown_timeout"));
    drop(activity);
    drop(execution);
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_supervisor_propagates_native_admission_cleanup_timeout() {
    let mut config = bounded_config(1);
    config.observability.shutdown_grace = std::time::Duration::from_millis(20);
    let backend = runtime_backend(1);
    let live_query = backend.admit_query().unwrap();
    let admission = Arc::new(admission::QueryAdmission::from_backend(Some(&backend), &config).unwrap());
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
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), supervisor.finish())
        .await
        .expect("supervisor must not grant admission a second grace period")
        .expect_err("native admission cleanup timeout must fail server shutdown");

    assert_eq!(error.fact_value("reason"), Some("backend_close_timeout"));
    assert_eq!(error.fact_value("shutdown_component"), Some("query_admission"));
    drop(live_query);
    let snapshot = backend
        .close_and_wait(std::time::Duration::from_secs(1))
        .expect("releasing the held query must allow the already-closing backend to quiesce");
    assert!(snapshot.gpu_resource_quiescent);
    assert!(snapshot.observation_quiescent);
    assert_eq!(snapshot.active_gpu_permits, 0);
    assert_eq!(snapshot.active_observation_permits, 0);
    assert_eq!(snapshot.cleanup_late_success_count, snapshot.cleanup_deadline_elapsed_count);
}

#[tokio::test(flavor = "current_thread")]
async fn supervisor_uses_captured_expired_plan_after_coordinator_finalization_clears_deadline() {
    let admission = Arc::new(admission::QueryAdmission::new(1).unwrap());
    let observability =
        ServerObservabilityConfig { shutdown_grace: std::time::Duration::ZERO, ..ServerObservabilityConfig::default() };
    let coordinator =
        coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(&observability));
    let execution = coordinator
        .register_with_id(
            "fast-finalization",
            "select 1",
            None,
            None,
            nexus_query_engine::observability::RuntimeObservationConfig::bounded(16, 4096),
        )
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
fn builder_rejects_injected_backend_with_unbounded_config() {
    let backend = controller_projection(1);

    let error = validate_test_controller_projection_matches_config(&backend, &unbounded_config()).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_unbounded_config"));
}

#[test]
fn builder_rejects_injected_backend_admission_conflict() {
    let backend = controller_projection(1);

    let error = validate_test_controller_projection_matches_config(&backend, &bounded_config(2)).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("max_in_flight"));
}

#[test]
fn builder_validation_uses_normalized_max_in_flight_for_default_queue_limit() {
    let mut config = bounded_config(1);
    config.max_in_flight = 2;
    let backend = controller_projection(2);

    validate_test_controller_projection_matches_config(&backend, &config).unwrap();
}

#[test]
fn builder_rejects_injected_backend_grant_band_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.admission_policy = config
        .admission_policy
        .with_grant_band(GrantBand { min: MemoryShare::Bytes(256), want: MemoryShare::Bytes(2048) });

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.grant_min"));
}

#[test]
fn builder_rejects_injected_backend_slot_guarantee_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_slot_guarantee(SlotGuarantee::Optimistic);

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("slot_guarantee"));
}

#[test]
fn builder_rejects_injected_backend_per_device_cap_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_max_queries_per_device(Some(1));

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queries_per_device"));
}

#[test]
fn builder_accepts_matching_explicit_per_device_cap() {
    let admission_policy =
        NexusAdmissionPolicy::default().with_max_concurrent_queries(1).with_max_queries_per_device(Some(1));
    let backend = NexusGpuBackend::builder()
        .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: TEST_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: TEST_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TEST_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: TEST_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        })
        .admission(admission_policy.clone())
        .build_test_controller_projection()
        .expect("CUDA-free backend controller projection should build");
    let mut config = bounded_config(1);
    config.admission_policy = admission_policy;

    validate_test_controller_projection_matches_config(&backend, &config).unwrap();
}

#[test]
fn builder_rejects_injected_backend_placement_policy_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_placement_policy(QueryPlacementPolicy::BestFit);

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.placement_policy"));
}

#[test]
fn builder_rejects_injected_backend_queue_limit_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_max_queued(3);

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued"));
}

#[test]
fn builder_rejects_injected_backend_wait_timeout_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_wait_timeout(std::time::Duration::from_secs(5));

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.wait_timeout"));
}

#[test]
fn builder_rejects_injected_backend_gpu_device_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.gpu_devices = vec![1];

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("gpu_devices"));
}

#[test]
fn builder_rejects_injected_backend_device_budget_fraction_conflict() {
    let backend = controller_projection(1);
    let mut config = bounded_config(1);
    config.device_budget_fraction = Some(0.5);
    config.native_optimizer_config =
        NexusNativeOptimizerConfig::default().with_native_execution_config(NexusNativeExecutionConfig::bounded(
            TEST_DEVICE_BUDGET_BYTES / 2,
            TEST_MAX_SOURCE_CHUNK_BYTES,
            TEST_MAX_ROW_GROUPS_PER_CHUNK,
            TEST_MAX_RETRY_ATTEMPTS,
            TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));

    let error = validate_test_controller_projection_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some(config::NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV));
}

#[tokio::test]
async fn injected_backend_allows_independent_host_execution_qos() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.host_execution_qos = Some(
        HostExecutionQosConfig::new(NonZeroUsize::new(1).unwrap())
            .with_wait_timeout(std::time::Duration::from_millis(5000)),
    );

    let ctx = build_session_context_with_backend(&config, &backend)
        .await
        .expect("host QoS is server-owned and must not conflict with backend GPU admission");
    let state = ctx.state();
    let rules = state.physical_optimizers();
    let native = rules
        .iter()
        .position(|rule| rule.name() == NEXUS_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME)
        .expect("backend native selection rule installed");
    let host_qos = rules
        .iter()
        .position(|rule| rule.name() == NEXUS_HOST_EXECUTION_QOS_RULE_NAME)
        .expect("server host QoS rule installed");
    let requirement = rules
        .iter()
        .position(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("final-plan requirement rule installed");

    assert!(native < host_qos && host_qos < requirement);
}

#[tokio::test]
async fn builder_requires_injected_session_host_execution_qos_to_match_config() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.host_execution_qos = Some(HostExecutionQosConfig::new(NonZeroUsize::new(2).unwrap()));

    let missing_qos_state = backend.install_on(SessionStateBuilder::new()).unwrap().build();
    let missing_qos_ctx = Arc::new(SessionContext::new_with_state(missing_qos_state));
    let error = builder_error(
        FlightSqlServer::builder().config(config.clone()).backend(backend.clone()).session_context(missing_qos_ctx),
    )
    .await;
    assert_eq!(error.fact_value("reason"), Some("server_session_host_execution_qos_mismatch"));

    let matching_qos_ctx =
        Arc::new(build_session_context_with_backend(&config, &backend).await.expect("matching host QoS installs"));
    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend)
        .session_context(matching_qos_ctx)
        .build()
        .await
        .expect("matching injected host QoS is accepted");
    assert_eq!(
        crate::session::installed_host_execution_qos_config(&server.session_context().state())
            .map(|qos| qos.max_concurrent_statements().get()),
        Some(2)
    );
}

#[tokio::test]
async fn build_session_context_with_backend_installs_backend_admission_source() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);

    let ctx = build_session_context_with_backend(&config, &backend).await.unwrap();
    let state = ctx.state();
    let optimizer_config =
        crate::session::installed_nexus_native_optimizer_config(&state).expect("native optimizer installed");

    assert!(
        optimizer_config
            .native_execution_config()
            .query_admission_source_matches(backend.session_capability(), backend.tenant_id())
    );
}

#[tokio::test]
async fn server_base_and_flight_style_clone_retain_cuvs_query_planner() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let ctx = build_session_context_with_backend(&config, &backend).await.unwrap();

    assert!(format!("{:?}", ctx.state().query_planner()).contains("CuvsQueryPlanner"));
    let cloned = SessionStateBuilder::new_from_existing(ctx.state()).build();
    assert!(format!("{:?}", cloned.query_planner()).contains("CuvsQueryPlanner"));
}

#[tokio::test]
async fn builder_rejects_unbounded_server_before_constructing_runtime_state() {
    let error = builder_error(FlightSqlServer::builder().config(unbounded_config())).await;

    assert_eq!(error.code(), crate::error::ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "datafusion_nexus.server.builder");
    assert_eq!(error.fact_value("reason"), Some("server_unbounded_backend_unsupported"));
    assert_eq!(error.fact_value("config_field"), Some(config::NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
    assert_eq!(error.fact_value("required_memory_policy"), Some("bounded"));
}

#[tokio::test]
async fn backendless_session_context_builder_is_rejected() {
    let error = match build_session_context(&bounded_config(1)).await {
        Ok(_) => panic!("backendless Flight SQL session construction must fail"),
        Err(error) => error,
    };

    assert_eq!(error.code(), crate::error::ErrorCode::InvalidRequest);
    assert_eq!(error.operation(), "datafusion_nexus.server.builder");
    assert_eq!(error.fact_value("reason"), Some("server_backend_required"));
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

#[cfg(feature = "iceberg")]
#[test]
fn iceberg_backend_snapshot_sanitizes_rest_urls_for_logs() {
    let backend = config::IcebergCatalogBackendConfig::Rest(
        crate::table_format::iceberg::catalog::RestIcebergCatalogConfig::new(
            "lake",
            "https://user:rest-secret@rest.local:8181/catalog?token=secret#frag",
            "s3://lakehouse/warehouse",
        ),
    );
    let source_access = crate::table_format::iceberg::SourceAccessPolicy::s3_custom_endpoint(
        "http://user:s3-secret@rustfs.local:9000/path?sig=secret#frag",
        None,
        Some(true),
    );

    let snapshot = iceberg_backend_snapshot(&backend, &source_access);

    assert_eq!(snapshot.rest_uri, "https://rest.local:8181/catalog");
    assert_eq!(snapshot.rest_s3_endpoint, "http://rustfs.local:9000/path");
    assert!(!snapshot.rest_uri.contains("secret"));
    assert!(!snapshot.rest_s3_endpoint.contains("secret"));
    assert!(!snapshot.rest_s3_endpoint.contains("sig="));
}
