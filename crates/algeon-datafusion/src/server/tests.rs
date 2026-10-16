use std::{
    io::Write,
    os::unix::fs::PermissionsExt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow::{
    array::{Int64Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use cudf::testing;
use datafusion::{execution::SessionStateBuilder, prelude::SessionConfig};
use parquet::arrow::ArrowWriter;
use query_runtime::observability::RuntimeObservationConfig;
use tokio::{runtime::Builder as TestRuntimeBuilder, time::timeout};
use tracing::{Level, subscriber};
use tracing_subscriber::fmt::{MakeWriter, Subscriber as FmtSubscriber};

use super::*;
use crate::{
    closing_backend::ClosingBackend,
    server::{
        adapter_error::ErrorCode,
        backend::{
            AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
            GpuMemoryOwnership,
        },
        native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
        planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
    },
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
        gpu_memory_ownership: vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)],
        statement_ticket_ttl: Duration::from_secs(60),
        http2_keepalive: Http2Keepalive::default(),
        observability: ServerObservabilityConfig::default(),
        native_optimizer_config: AlgeonNativeOptimizerConfig::default(),
        auth: AuthConfig::default(),
        cugraph: None,
        models: None,
        data_root: None,
        tls: None,
        init_sql: None,
    }
}

fn bounded_config(max_active_attempts_per_device: usize) -> ServerConfig {
    let mut config = base_config();
    config.native_optimizer_config =
        AlgeonNativeOptimizerConfig::default().with_native_execution_config(AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(
                TEST_MAX_SOURCE_CHUNK_BYTES,
                TEST_MAX_ROW_GROUPS_PER_CHUNK,
                TEST_MAX_RETRY_ATTEMPTS,
                TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
            )
            .expect("valid source chunking"),
        ));
    config.gpu_device_profiles = vec![
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
            .with_max_active_attempts(max_active_attempts_per_device),
    ];
    config
}

fn runtime_backend(max_active_attempts_per_device: usize) -> ClosingBackend {
    runtime_backend_with_policy(max_active_attempts_per_device, AlgeonGpuExecutionMode::NativePreferred)
}

fn strict_runtime_backend(max_active_attempts_per_device: usize) -> ClosingBackend {
    runtime_backend_with_policy(max_active_attempts_per_device, AlgeonGpuExecutionMode::NativeRequired)
}

fn runtime_backend_with_policy(
    max_active_attempts_per_device: usize,
    execution_mode: AlgeonGpuExecutionMode,
) -> ClosingBackend {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
        .with_max_active_attempts(max_active_attempts_per_device);
    ClosingBackend::new(
        AlgeonGpuBackend::builder()
            .devices([AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)])
            .with_native_execution_config(AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(
                    TEST_MAX_SOURCE_CHUNK_BYTES,
                    TEST_MAX_ROW_GROUPS_PER_CHUNK,
                    TEST_MAX_RETRY_ATTEMPTS,
                    TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
                )
                .expect("valid source chunking"),
            ))
            .with_execution_mode(execution_mode)
            .build()
            .expect("runtime backend should build"),
    )
}

#[test]
fn unset_logical_capture_budget_registers_every_attempt_admission_can_hold_beside_a_finalizing_query() {
    let config = bounded_config(1);
    let live_attempts = config.queue_policy.max_queued_attempts() + 1;
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
        config.active_logical_capture_budget_bytes().expect("default config resolves a capture budget"),
    ));
    // A finished statement keeps its reservation until it finalizes, after
    // its admission slot is free again.
    let _finalizing = coordinator
        .register_new("select 0", None, config.observability.runtime_observation_config())
        .expect("the earlier statement registers");

    let leases = (1..=live_attempts)
        .map(|index| {
            coordinator
                .register_new(format!("select {index}"), None, config.observability.runtime_observation_config())
                .unwrap_or_else(|error| panic!("live query {index} of {live_attempts} was refused: {error:?}"))
        })
        .collect::<Vec<_>>();

    assert_eq!(leases.len(), live_attempts);
}

#[test]
fn shutdown_closes_native_admission_before_coordinator_terminalization() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let admission = admission::QueryAdmission::from_backend(&backend);
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &config.observability,
        config.active_logical_capture_budget_bytes().expect("test config resolves a capture budget"),
    ));
    let _planning = coordinator
        .register_with_id("shutdown-order", "select 1", None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap();
    let register_probe = |query_id: &str| {
        coordinator.register_with_id(query_id, "select 2", None, RuntimeObservationConfig::bounded(16, 4096))
    };

    let coordinator_live = after_closing_admission(&admission, || {
        let error = backend
            .begin_attempt(attempt::AttemptOptions::default())
            .expect_err("native QueryService must be closed before coordinator shutdown begins");
        assert_eq!(error.code(), ErrorCode::Cancelled);
        drop(register_probe("before-shutdown").expect("coordinator remains live until admission is closed"));
        coordinator.begin_shutdown(Instant::now())
    });

    assert_eq!(coordinator_live.live_count(), 1);
    assert!(register_probe("after-shutdown").is_err(), "coordinator rejects registration after shutdown begins");
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_supervisor_reports_coordinator_deadline_expiry_as_error() {
    let admission = Arc::new(admission::QueryAdmission::new_for_tests());
    let observability =
        ServerObservabilityConfig { shutdown_grace: Duration::from_millis(20), ..ServerObservabilityConfig::default() };
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &observability,
        ServerObservabilityConfig::DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES,
    ));
    let execution = coordinator
        .register_with_id("supervisor-forced-timeout", "select 1", None, RuntimeObservationConfig::bounded(16, 4096))
        .unwrap()
        .into_execution()
        .unwrap();
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
        config.active_logical_capture_budget_bytes().expect("test config resolves a capture budget"),
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
    let coordinator = coordinator::QueryCoordinator::new(coordinator::QueryCoordinatorConfig::from_observability(
        &observability,
        ServerObservabilityConfig::DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES,
    ));
    let execution = coordinator
        .register_with_id("fast-finalization", "select 1", None, RuntimeObservationConfig::bounded(16, 4096))
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

#[tokio::test]
async fn builder_rejects_admission_bounds_the_coordinator_cannot_register() {
    let mut queued = bounded_config(1);
    queued.queue_policy = queued.queue_policy.clone().with_max_queued_attempts(1023);
    let error = builder_error(FlightSqlServer::builder().config(queued)).await;
    assert_eq!(error.fact_value("reason"), Some("admission_exceeds_live_query_capacity"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued_attempts"));

    let active = bounded_config(1024);
    let error = builder_error(FlightSqlServer::builder().config(active)).await;
    assert_eq!(error.fact_value("reason"), Some("admission_exceeds_live_query_capacity"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles.max_active_attempts"));
}

#[test]
fn builder_rejects_injected_backend_admission_conflict() {
    let backend = runtime_backend(1);

    let error = validate_injected_backend_matches_config(&backend, &bounded_config(2)).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles.max_active_attempts"));
}

fn builder_rejects_injected_backend_default_query_limit_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_default_query_limit_bytes(256);

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles.default_query_limit_bytes"));
}

fn builder_accepts_matching_explicit_per_device_cap() {
    let profile = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
        .with_max_active_attempts(1);
    let backend = ClosingBackend::new(
        AlgeonGpuBackend::builder()
            .devices([AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive)])
            .with_native_execution_config(AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(
                    TEST_MAX_SOURCE_CHUNK_BYTES,
                    TEST_MAX_ROW_GROUPS_PER_CHUNK,
                    TEST_MAX_RETRY_ATTEMPTS,
                    TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
                )
                .expect("valid source chunking"),
            ))
            .build()
            .expect("injected backend should build"),
    );
    let config = bounded_config(1);

    validate_injected_backend_matches_config(&backend, &config).unwrap();
}

fn builder_rejects_injected_backend_queue_limit_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.queue_policy = config.queue_policy.with_max_queued_attempts(3);

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued_attempts"));
}

#[test]
fn tests_cases_1() {
    testing::run_named_cases(&[
        (
            "builder_rejects_injected_backend_wait_timeout_conflict",
            builder_rejects_injected_backend_wait_timeout_conflict,
        ),
        (
            "builder_rejects_injected_backend_default_query_limit_conflict",
            builder_rejects_injected_backend_default_query_limit_conflict,
        ),
        ("builder_accepts_matching_explicit_per_device_cap", builder_accepts_matching_explicit_per_device_cap),
        (
            "builder_rejects_injected_backend_queue_limit_conflict",
            builder_rejects_injected_backend_queue_limit_conflict,
        ),
        ("builder_rejects_injected_backend_gpu_device_conflict", builder_rejects_injected_backend_gpu_device_conflict),
        ("server_base_and_flight_style_clone_retain_cuvs_query_planner", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(server_base_and_flight_style_clone_retain_cuvs_query_planner())
        }),
        ("builder_rejects_strict_backend_session_with_incompatible_target_partitions", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(builder_rejects_strict_backend_session_with_incompatible_target_partitions())
        }),
    ]);
}

fn builder_rejects_injected_backend_wait_timeout_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.queue_policy = config.queue_policy.with_max_queue_wait(Duration::from_secs(5));

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queue_wait_ms"));
}

fn builder_rejects_injected_backend_gpu_device_conflict() {
    let backend = runtime_backend(1);
    let mut config = bounded_config(1);
    config.gpu_device_profiles = vec![
        AlgeonGpuDeviceProfile::new(1)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES }),
    ];
    config.gpu_memory_ownership = vec![(1, GpuMemoryOwnership::WholeDeviceExclusive)];

    let error = validate_injected_backend_matches_config(&backend, &config).unwrap_err();

    assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
    assert_eq!(error.fact_value("config_field"), Some("admission.device_profiles"));
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
        FlightSqlServer::builder().config(caller_config).backend(backend.clone()).session_context(incompatible_ctx),
    )
    .await;

    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    assert_eq!(error.operation(), "algeon_datafusion.server.builder");
    assert_eq!(error.fact_value("reason"), Some("server_session_optimizer_policy_mismatch"));
    assert_eq!(error.fact_value("config_field"), Some("datafusion.execution.target_partitions"));
    assert_eq!(error.fact_value("expected_value"), Some("1"));
    assert_eq!(error.fact_value("actual_value"), Some("8"));
}

async fn server_base_and_flight_style_clone_retain_cuvs_query_planner() {
    let config = bounded_config(1);
    let backend = runtime_backend(1);
    let ctx = build_session_context_with_backend(&config, &backend).await.unwrap();

    assert!(format!("{:?}", ctx.state().query_planner()).contains("AlgeonQueryPlanner"));
    let cloned = SessionStateBuilder::new_from_existing(ctx.state()).build();
    assert!(format!("{:?}", cloned.query_planner()).contains("AlgeonQueryPlanner"));
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

#[test]
fn startup_rejection_of_a_configuration_prints_the_setting_value_and_accepted_range() {
    const GIB: u64 = 1 << 30;
    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0).with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 100 * GIB }),
        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 95 * GIB },
    );
    let backend_error = AlgeonGpuBackend::builder()
        .with_native_execution_config(AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(
                TEST_MAX_SOURCE_CHUNK_BYTES,
                TEST_MAX_ROW_GROUPS_PER_CHUNK,
                TEST_MAX_RETRY_ATTEMPTS,
                TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
            )
            .expect("valid source chunking"),
        ))
        .devices([device])
        .validate()
        .expect_err("a limit above the static partition is a configuration error");

    let error = backend_build_error(backend_error);

    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    // The standalone binary logs `ServerError::message()` without a length
    // bound, unlike the cause chain.
    assert!(
        error.message().starts_with("invalid server configuration: invalid ownership-enforcing GPU backend core configuration: device 0 memory_limit_bytes is 107374182400 bytes (100.00 GiB), above the static partition of 102005473280 bytes (95.00 GiB); the largest accepted value is 102005473280 bytes (95.00 GiB)"),
        "{}",
        error.message()
    );

    let other = backend_build_error(adapter_error::Error::new(ErrorCode::Internal, "algeon_datafusion.test", "boom"));
    assert_eq!(other.message(), "failed to initialize query service admission");
}

#[tokio::test(flavor = "current_thread")]
async fn an_occupied_port_fails_serving_with_the_address_and_no_readiness_line() {
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").expect("occupy a loopback port");
    let addr = occupied.local_addr().expect("occupied address");
    let mut config = bounded_config(1);
    config.bind_addr = addr;
    let backend = runtime_backend(1);
    let logs = CapturedLogs::default();
    let _subscriber = subscriber::set_default(
        FmtSubscriber::builder().with_writer(logs.clone()).with_ansi(false).with_max_level(Level::INFO).finish(),
    );

    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend.clone())
        .shutdown(ServerShutdown::Token(CancellationToken::new()))
        .build()
        .await
        .expect("the server builds before it binds");
    let error = server.serve().await.expect_err("an occupied port fails serving");

    assert_eq!(error.code(), ErrorCode::ServerTransportFailure, "{error}");
    assert_eq!(error.message(), format!("failed to bind Flight SQL listener at {addr}"));
    assert_eq!(error.fact_value("bind_addr"), Some(addr.to_string().as_str()));
    let logs = logs.text();
    assert!(logs.contains("server config snapshot"), "{logs}");
    assert!(!logs.contains("starting Algeon Flight SQL server"), "{logs}");
}

#[tokio::test]
async fn init_sql_runs_ddl_and_names_the_first_statement_that_is_not_ddl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("init.sql");
    std::fs::write(&path, "CREATE VIEW seeded AS VALUES (1), (2);\nCREATE VIEW seeded_view AS SELECT * FROM seeded;\nSELECT * FROM seeded;\n")
        .expect("write init_sql");
    let ctx = SessionContext::new();

    let error = session::run_init_sql(&ctx, &path).await.expect_err("a query is not DDL");

    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("init_sql"));
    assert_eq!(error.fact_value("statement_index"), Some("3"));
    assert_eq!(error.fact_value("reason"), Some("init_sql_statement_not_ddl"));
    assert!(error.message().contains("SELECT * FROM seeded"), "{}", error.message());
    assert!(ctx.table_exist("seeded_view").expect("catalog lookup"), "earlier DDL statements ran");
}

#[tokio::test]
async fn init_sql_applies_the_flight_ddl_policy() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("init.sql");
    std::fs::write(&path, "CREATE TABLE seeded AS VALUES (1), (2);\n").expect("write init_sql");
    let ctx = SessionContext::new();

    let error = session::run_init_sql(&ctx, &path).await.expect_err("CTAS is refused like a Flight update");

    assert_eq!(error.fact_value("reason"), Some("init_sql_statement_rejected"), "{error}");
    assert_eq!(error.fact_value("statement_index"), Some("1"));
    assert!(error.message().contains("CREATE TABLE AS SELECT"), "{}", error.message());
    assert!(!ctx.table_exist("seeded").expect("catalog lookup"));
}

#[tokio::test]
async fn init_sql_errors_mask_statement_option_values() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("init.sql");
    // Short enough that the OPTIONS clause lies inside the 120-character excerpt.
    std::fs::write(
        &path,
        "CREATE EXTERNAL TABLE r STORED AS PARQUET LOCATION 's3://b/r.parquet' OPTIONS ('aws.secret_access_key' 'sekrit');\n",
    )
    .expect("write init_sql");
    let ctx = SessionContext::new();

    let error = session::run_init_sql(&ctx, &path).await.expect_err("no object store serves s3://");

    // Two layers keep the value out of the startup log: DataFusion's statement
    // rendering omits the OPTIONS clause, and the excerpt passes through
    // `RedactedSql` so a rendering that includes it would still be masked.
    assert!(!error.message().contains("sekrit"), "{}", error.message());
    assert!(error.message().contains("CREATE EXTERNAL TABLE r"), "{}", error.message());
}

fn write_parquet(path: &std::path::Path, values: &[i64]) {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(values.to_vec()))])
        .expect("test batch");
    let mut writer = ArrowWriter::try_new(std::fs::File::create(path).expect("create parquet"), schema, None)
        .expect("parquet writer");
    writer.write(&batch).expect("write parquet");
    writer.close().expect("close parquet");
}

async fn count_rows(ctx: &SessionContext, table: &str) -> i64 {
    let batches =
        ctx.sql(&format!("SELECT count(*) FROM {table}")).await.expect("plan count").collect().await.expect("count");
    batches[0].column(0).as_any().downcast_ref::<Int64Array>().expect("count column").value(0)
}

#[tokio::test]
async fn data_root_registers_parquet_files_and_dataset_directories_by_their_names() {
    let root = tempfile::tempdir().expect("data root");
    write_parquet(&root.path().join("Orders.PARQUET"), &[1, 2, 3]);
    let events = root.path().join("events");
    std::fs::create_dir(&events).unwrap();
    write_parquet(&events.join("part-0.parquet"), &[1, 2]);
    write_parquet(&events.join("part-1.parquet"), &[3, 4, 5]);
    std::fs::write(events.join("_SUCCESS"), b"").unwrap();
    let notes = root.path().join("notes");
    std::fs::create_dir(&notes).unwrap();
    std::fs::write(notes.join("readme.txt"), b"no data").unwrap();
    write_parquet(&root.path().join(".staging.parquet"), &[1]);
    write_parquet(&root.path().join("_tmp.parquet"), &[1]);
    std::fs::create_dir(root.path().join("_temporary")).unwrap();
    write_parquet(&root.path().join("_temporary").join("part-0.parquet"), &[1]);
    std::fs::write(root.path().join("orders.csv"), b"v\n1\n").unwrap();
    let ctx = SessionContext::new();

    session::register_data_root(&ctx, root.path()).await.expect("register the data root");

    let schema = ctx.catalog("datafusion").and_then(|catalog| catalog.schema("public")).expect("default schema");
    let mut names = schema.table_names();
    names.sort();
    assert_eq!(names, ["Orders", "events"]);
    assert_eq!(count_rows(&ctx, "\"Orders\"").await, 3);
    assert_eq!(count_rows(&ctx, "events").await, 5);
}

#[tokio::test]
async fn data_root_reads_names_with_glob_and_url_characters_literally() {
    let parent = tempfile::tempdir().expect("data root parent");
    // A glob character in the root itself splits every scheme-less location.
    let root = parent.path().join("[prod] data");
    std::fs::create_dir(&root).unwrap();
    // Each sibling matches the glob a scheme-less location would derive from
    // the entry before it.
    write_parquet(&root.join("sales[2024].parquet"), &[1, 2, 3]);
    write_parquet(&root.join("sales2.parquet"), &[1; 7]);
    write_parquet(&root.join("wild*card?.parquet"), &[1; 4]);
    write_parquet(&root.join("wild-card!.parquet"), &[1]);
    write_parquet(&root.join("space and %41.parquet"), &[1; 6]);
    for (directory, rows) in [("q[1]", 2), ("q1", 5)] {
        std::fs::create_dir(root.join(directory)).unwrap();
        write_parquet(&root.join(directory).join("part-0.parquet"), &vec![1; rows]);
    }
    let ctx = SessionContext::new();

    session::register_data_root(&ctx, &root).await.expect("register the data root");

    for (table, rows) in [
        ("sales[2024]", 3),
        ("sales2", 7),
        ("wild*card?", 4),
        ("wild-card!", 1),
        ("space and %41", 6),
        ("q[1]", 2),
        ("q1", 5),
    ] {
        assert_eq!(count_rows(&ctx, &format!("\"{table}\"")).await, rows, "{table}");
    }
}

#[tokio::test]
async fn data_root_skips_entries_it_cannot_inspect_with_a_warning() {
    let root = tempfile::tempdir().expect("data root");
    write_parquet(&root.path().join("orders.parquet"), &[1, 2]);
    // A host symlink outside a container bind mount dangles inside it.
    let dangling = root.path().join("linked.parquet");
    std::os::unix::fs::symlink(root.path().join("absent").join("linked.parquet"), &dangling).unwrap();
    let logs = CapturedLogs::default();
    let _subscriber = subscriber::set_default(
        FmtSubscriber::builder().with_writer(logs.clone()).with_ansi(false).with_max_level(Level::INFO).finish(),
    );
    let ctx = SessionContext::new();

    session::register_data_root(&ctx, root.path()).await.expect("a dangling link does not stop startup");

    let schema = ctx.catalog("datafusion").and_then(|catalog| catalog.schema("public")).expect("default schema");
    assert_eq!(schema.table_names(), ["orders"]);
    let logs = logs.text();
    assert!(logs.contains("skipping data root entry that cannot be inspected"), "{logs}");
    assert!(logs.contains(&dangling.display().to_string()), "{logs}");
}

#[tokio::test]
async fn data_root_skips_a_directory_whose_listing_fails_before_a_parquet_file() {
    let root = tempfile::tempdir().expect("data root");
    write_parquet(&root.path().join("orders.parquet"), &[1, 2]);
    let locked = root.path().join("scratch").join("private");
    std::fs::create_dir_all(&locked).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::read_dir(&locked).is_ok() {
        // A privileged user reads the directory regardless of its mode.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o700)).unwrap();
        return;
    }
    let logs = CapturedLogs::default();
    let _subscriber = subscriber::set_default(
        FmtSubscriber::builder().with_writer(logs.clone()).with_ansi(false).with_max_level(Level::INFO).finish(),
    );
    let ctx = SessionContext::new();

    let result = session::register_data_root(&ctx, root.path()).await;

    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o700)).unwrap();
    result.expect("an unlistable directory without Parquet files does not stop startup");
    let schema = ctx.catalog("datafusion").and_then(|catalog| catalog.schema("public")).expect("default schema");
    assert_eq!(schema.table_names(), ["orders"]);
    let logs = logs.text();
    assert!(logs.contains("skipping data root directory whose listing failed"), "{logs}");
    assert!(logs.contains(&root.path().join("scratch").display().to_string()), "{logs}");
}

#[tokio::test]
async fn data_root_rejects_two_entries_with_one_table_name() {
    let root = tempfile::tempdir().expect("data root");
    write_parquet(&root.path().join("sales.parquet"), &[1]);
    let directory = root.path().join("sales");
    std::fs::create_dir(&directory).unwrap();
    write_parquet(&directory.join("part-0.parquet"), &[2]);
    let ctx = SessionContext::new();

    let error = session::register_data_root(&ctx, root.path()).await.expect_err("two entries name `sales`");

    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("data.root"));
    assert_eq!(error.fact_value("reason"), Some("data_root_duplicate_table"));
    assert_eq!(error.fact_value("table"), Some("sales"));
    for path in [directory, root.path().join("sales.parquet")] {
        assert!(error.message().contains(&path.display().to_string()), "{}", error.message());
    }
}

#[tokio::test]
async fn data_root_names_a_parquet_file_whose_schema_cannot_be_read() {
    let root = tempfile::tempdir().expect("data root");
    let broken = root.path().join("broken.parquet");
    std::fs::write(&broken, b"not parquet").unwrap();
    let ctx = SessionContext::new();

    let error = session::register_data_root(&ctx, root.path()).await.expect_err("an unreadable footer");

    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("data.root"));
    assert_eq!(error.fact_value("table"), Some("broken"));
    assert!(error.message().contains(&broken.display().to_string()), "{}", error.message());
    assert!(std::error::Error::source(&error).is_some(), "the DataFusion cause is retained");
    assert!(!ctx.table_exist("broken").expect("catalog lookup"));
}

#[tokio::test]
async fn startup_sql_runs_after_the_data_root_tables_are_registered() {
    let root = tempfile::tempdir().expect("data root");
    write_parquet(&root.path().join("orders.parquet"), &[1, 5, 10]);
    let init_sql = root.path().join("_init.sql");
    std::fs::write(&init_sql, "CREATE VIEW large_orders AS SELECT v FROM orders WHERE v > 2;\n")
        .expect("write init_sql");
    let mut config = base_config();
    config.data_root = Some(root.path().to_path_buf());
    config.init_sql = Some(init_sql);
    let ctx = SessionContext::new();

    session::prepare_shared_session(&ctx, &config).await.expect("prepare the shared session");

    assert_eq!(count_rows(&ctx, "large_orders").await, 2);
}

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

impl CapturedLogs {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("log buffer")).into_owned()
    }
}

impl Write for CapturedLogs {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buffer").extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for CapturedLogs {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}
