use std::{num::NonZeroUsize, sync::Barrier};

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    common::config::ConfigOptions,
    logical_expr::Operator,
    physical_expr::expressions::{binary, col},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan,
        empty::EmptyExec,
        projection::{ProjectionExec, ProjectionExpr},
    },
};
use nexus_query_engine::runtime::{QueryPlacementPolicy, SlotGuarantee};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::{
    backend::{GrantBand, MemoryShare, NexusAdmissionPolicy, NexusMemoryPolicy},
    native_exec::{NativeMetricsCapture, NexusNativeExecutionConfig},
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    policy::FallbackReason,
    report::CandidateOutcome,
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
        cpu_fallback_max_concurrent_statements: None,
        cpu_fallback_wait_timeout: None,
        gpu_devices: vec![0],
        device_budget_fraction: None,
        statement_ticket_ttl: std::time::Duration::from_secs(60),
        observability: ServerObservabilityConfig::default(),
        native_optimizer_config: NexusNativeOptimizerConfig::default(),
        auth: auth::AuthConfig::default(),
        workspace: None,
        #[cfg(feature = "iceberg")]
        iceberg: None,
        #[cfg(feature = "cugraph")]
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

fn admission_only_backend(max_concurrent_queries: usize) -> NexusGpuBackend {
    NexusGpuBackend::builder()
        .memory_policy(NexusMemoryPolicy::Bounded {
            device_ordinals: vec![0],
            device_budget_bytes: TEST_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: TEST_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: TEST_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: TEST_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
        })
        .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(max_concurrent_queries))
        .build_admission_only()
        .expect("admission-only backend should build")
}

#[test]
fn shutdown_closes_native_admission_before_coordinator_terminalization() {
    let config = bounded_config(1);
    let backend = admission_only_backend(1);
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
            .query_service()
            .try_admit_query(backend.tenant_id().clone())
            .expect_err("native QueryService must be closed before coordinator shutdown begins");
        assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("admission_closed"));
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
    let backend = admission_only_backend(1);
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
        let error = backend
            .query_service()
            .try_admit_query(backend.tenant_id().clone())
            .expect_err("native admission is closed before the coordinator race begins");
        assert_eq!(error.fact_value("reason"), Some("admission_closed"));
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
    let backend = admission_only_backend(1);
    let live_query = backend.query_service().admit_query(backend.tenant_id().clone()).unwrap();
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

    assert_eq!(error.fact_value("reason"), Some("query_service_shutdown_timeout"));
    assert_eq!(error.fact_value("shutdown_component"), Some("query_admission"));
    drop(live_query);
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

fn empty_exec_plan() -> Arc<dyn ExecutionPlan> {
    Arc::new(EmptyExec::new(Arc::new(Schema::empty())))
}

fn unsupported_projection_plan() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let input = Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>;
    let remainder =
        binary(col("m00", &schema).unwrap(), Operator::Modulo, col("m01", &schema).unwrap(), &schema).unwrap();

    Arc::new(
        ProjectionExec::try_new([ProjectionExpr { expr: remainder, alias: "remainder".to_string() }], input).unwrap(),
    )
}

#[tokio::test]
async fn builder_rejects_session_context_without_injected_backend() {
    let config = bounded_config(1);
    let ctx = Arc::new(SessionContext::new());

    let error = builder_error(FlightSqlServer::builder().config(config).session_context(ctx)).await;

    assert_eq!(error.fact_value("reason"), Some("server_session_without_backend"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_with_unbounded_config() {
    let backend = admission_only_backend(1);

    let error = builder_error(FlightSqlServer::builder().config(unbounded_config()).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_unbounded_config"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_admission_conflict() {
    let backend = admission_only_backend(1);

    let error = builder_error(FlightSqlServer::builder().config(bounded_config(2)).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("max_in_flight"));
}

#[tokio::test]
async fn builder_validation_uses_normalized_max_in_flight_for_default_queue_limit() {
    let mut config = bounded_config(1);
    config.max_in_flight = 2;
    let backend = NexusGpuBackend::from_optimizer_config_for_admission_only(
        config.native_optimizer_config.clone(),
        config.gpu_devices.clone(),
        admission::server_admission_policy(&config),
        nexus_query_engine::runtime::TenantId::new("default"),
    )
    .unwrap()
    .expect("bounded config should build backend");

    validate_injected_backend_matches_config(&backend, &config).unwrap();
}

#[tokio::test]
async fn builder_rejects_injected_backend_grant_band_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.admission_policy = config
        .admission_policy
        .with_grant_band(GrantBand { min: MemoryShare::Bytes(256), want: MemoryShare::Bytes(2048) });

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.grant_min"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_slot_guarantee_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_slot_guarantee(SlotGuarantee::Optimistic);

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("slot_guarantee"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_per_device_cap_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_max_queries_per_device(Some(1));

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queries_per_device"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_placement_policy_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_placement_policy(QueryPlacementPolicy::BestFit);

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.placement_policy"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_queue_limit_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_max_queued(3);

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_wait_timeout_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.admission_policy = config.admission_policy.with_wait_timeout(std::time::Duration::from_secs(5));

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.wait_timeout"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_gpu_device_conflict() {
    let backend = admission_only_backend(1);
    let mut config = bounded_config(1);
    config.gpu_devices = vec![1];

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("gpu_devices"));
}

#[tokio::test]
async fn builder_rejects_injected_backend_device_budget_fraction_conflict() {
    let backend = admission_only_backend(1);
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

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some(config::NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV));
}

#[tokio::test]
async fn builder_rejects_injected_backend_fallback_wait_timeout_conflict() {
    let mut backend_config = bounded_config(1);
    backend_config.native_optimizer_config = backend_config
        .native_optimizer_config
        .with_cpu_fallback_wait_timeout(std::time::Duration::from_millis(7000))
        .with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());
    let backend = NexusGpuBackend::from_optimizer_config_for_admission_only(
        backend_config.native_optimizer_config.clone(),
        backend_config.gpu_devices.clone(),
        admission::server_admission_policy(&backend_config),
        nexus_query_engine::runtime::TenantId::new("default"),
    )
    .unwrap()
    .expect("bounded config should build backend");

    let mut config = bounded_config(1);
    config.native_optimizer_config = config
        .native_optimizer_config
        .with_cpu_fallback_wait_timeout(std::time::Duration::from_millis(5000))
        .with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());

    let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("fallback.wait_timeout_ms"));
}

#[tokio::test]
async fn build_session_context_with_backend_installs_backend_admission_source() {
    let config = bounded_config(1);
    let backend = admission_only_backend(1);

    let ctx = build_session_context_with_backend(&config, Some(&backend)).await.unwrap();
    let state = ctx.state();
    let optimizer_config =
        crate::session::installed_nexus_native_optimizer_config(&state).expect("native optimizer installed");

    assert!(
        optimizer_config
            .native_execution_config()
            .query_admission_source_matches(backend.query_service(), backend.tenant_id())
    );
}

#[test]
fn server_cpu_fallback_limit_reaches_optimizer_fallback_plan() {
    let mut config = bounded_config(1);
    config.native_optimizer_config =
        config.native_optimizer_config.with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());
    let rule = NexusNativePhysicalOptimizerRule::new(config.native_optimizer_config.clone());
    let fallback_plan = unsupported_projection_plan();

    let optimized = rule.optimize(fallback_plan, &ConfigOptions::new()).unwrap();
    let fallback_report = rule.last_report().expect("optimizer should retain a fallback planning report");

    assert_eq!(fallback_report.rewritten_count(), 0, "outcomes={:?}", fallback_report.outcomes());
    assert!(matches!(
        fallback_report.outcomes(),
        [CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }]
    ));
    assert!(
        crate::native_exec::plan_is_cpu_fallback_limited(optimized.as_ref()),
        "configured server fallback limit must wrap non-rewritten CPU fallback plans"
    );

    let optimized_empty = rule.optimize(empty_exec_plan(), &ConfigOptions::new()).unwrap();
    let empty_report = rule.last_report().expect("optimizer should retain an empty-input planning report");

    assert_eq!(empty_report.rewritten_count(), 1, "outcomes={:?}", empty_report.outcomes());
    assert_eq!(empty_report.rejected_count(), 0, "outcomes={:?}", empty_report.outcomes());
    assert!(
        !crate::native_exec::plan_is_cpu_fallback_limited(optimized_empty.as_ref()),
        "rewritten EmptyExec plans must not take the CPU fallback limiter"
    );
}

#[tokio::test]
async fn builder_installs_unbounded_native_execution_limit() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut config = unbounded_config();
    config.bind_addr = listener.local_addr().unwrap();
    let server = FlightSqlServer::builder()
        .config(config)
        .listener(listener)
        .shutdown(ServerShutdown::Token(CancellationToken::new()))
        .build()
        .await
        .unwrap();

    assert_eq!(
        server
            .service
            .native_optimizer_config()
            .native_execution_config()
            .unbounded_native_max_concurrent_queries()
            .map(|value| value.get()),
        Some(1)
    );
    let state = server.session_context().state();
    let optimizer_config =
        crate::session::installed_nexus_native_optimizer_config(&state).expect("native optimizer installed");
    assert_eq!(
        optimizer_config.native_execution_config().unbounded_native_max_concurrent_queries().map(|value| value.get()),
        Some(1)
    );
}

#[tokio::test]
async fn builder_rejects_injected_session_context_from_different_backend() {
    let config = bounded_config(1);
    let session_backend = admission_only_backend(1);
    let server_backend = admission_only_backend(1);
    let ctx = Arc::new(build_session_context_with_backend(&config, Some(&session_backend)).await.unwrap());

    let error =
        builder_error(FlightSqlServer::builder().config(config).backend(server_backend).session_context(ctx)).await;

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
        )
        .with_s3_endpoint("http://user:s3-secret@rustfs.local:9000/path?sig=secret#frag"),
    );

    let snapshot = iceberg_backend_snapshot(&backend);

    assert_eq!(snapshot.rest_uri, "https://rest.local:8181/catalog");
    assert_eq!(snapshot.rest_s3_endpoint, "http://rustfs.local:9000/path");
    assert!(!snapshot.rest_uri.contains("secret"));
    assert!(!snapshot.rest_s3_endpoint.contains("secret"));
    assert!(!snapshot.rest_s3_endpoint.contains("sig="));
}
