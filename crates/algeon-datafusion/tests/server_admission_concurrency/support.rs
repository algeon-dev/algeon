#![allow(dead_code, reason = "shared test helpers used across independent integration test targets")]

use std::{
    fs::File,
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use algeon_datafusion::{
    attempt::{AlgeonAttempt, AttemptOptions, AttemptStream},
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceInstallationBounds, GpuDeviceMemoryLimit, GpuMemoryOwnership},
    native_exec::AlgeonNativeExecutionConfig,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, FlightSqlServer, Result as ServerResult, ServerConfig, ServerShutdown,
        backend_from_server_config, init_cuda_device,
    },
};
use arrow::{
    array::{ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use query_runtime::observability::QueryAttemptReport;
use tokio::{
    task::JoinHandle,
    time::{sleep, timeout},
};
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};

pub(crate) const GIB: u64 = 1024 * 1024 * 1024;
/// The worked concurrency configuration from the admission design: a 12 GiB
/// managed capacity issuing three simultaneous 4 GiB query caps.
pub(crate) const MANAGED_CAPACITY_BYTES: u64 = 12 * GIB;
pub(crate) const QUERY_CAP_BYTES: u64 = 4 * GIB;
pub(crate) const MAX_ACTIVE_ATTEMPTS: usize = 3;

pub(crate) const CONCURRENCY_SQL: &str = "select value from concurrency_small order by value desc limit 4";
/// Rows in the always-small table used by slot holders and bounded queries.
pub(crate) const SMALL_TABLE_ROWS: i64 = 4096;

fn server_config_for_concurrency(managed_capacity_bytes: u64, query_cap_bytes: u64) -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").expect("write server config");
    let path = path.to_string_lossy().into_owned();
    let mut config = ServerConfig::from_env_get(|name| (name == ALGEON_SERVER_CONFIG_FILE_ENV).then(|| path.clone()))
        .expect("explicit test device profile");
    config.gpu_memory_ownership = vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)];
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0]
        .clone()
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: managed_capacity_bytes })
        .with_backend_reserve_bytes(0)
        .with_default_query_limit_bytes(query_cap_bytes)
        .with_max_active_attempts(MAX_ACTIVE_ATTEMPTS)
        // Lanes must cover every active attempt; streams and workers are sized
        // per attempt so three admissions can hold their resources at once.
        .with_installation_bounds(AlgeonGpuDeviceInstallationBounds::new(
            4 * MAX_ACTIVE_ATTEMPTS,
            2 * MAX_ACTIVE_ATTEMPTS,
            MAX_ACTIVE_ATTEMPTS,
        ));
    config.native_optimizer_config =
        config.native_optimizer_config.clone().with_native_execution_config(AlgeonNativeExecutionConfig::default());
    config.queue_policy =
        config.queue_policy.clone().with_max_queued_attempts(32).with_max_queue_wait(Duration::from_secs(300));
    config
}

fn write_rows_parquet(path: &Path, rows: i64) {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new((0..rows).collect::<Int64Array>()) as ArrayRef])
            .expect("row fixture");
    let file = File::create(path).expect("create row fixture");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("create Parquet writer");
    writer.write(&batch).expect("write row fixture");
    writer.close().expect("finish row fixture");
}

pub(crate) struct ConcurrencyServer {
    pub(crate) backend: AlgeonGpuBackend,
    pub(crate) ctx: Arc<SessionContext>,
    pub(crate) addr: SocketAddr,
    shutdown: CancellationToken,
    server: JoinHandle<ServerResult<()>>,
    _dir: tempfile::TempDir,
}

#[derive(Clone, Copy)]
pub(crate) enum ListenMode {
    Addr,
    Prebound,
}

impl ConcurrencyServer {
    pub(crate) async fn spawn(rows: i64) -> Self {
        Self::spawn_with(rows, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES).await
    }

    pub(crate) async fn spawn_with(rows: i64, managed_capacity_bytes: u64, query_cap_bytes: u64) -> Self {
        Self::spawn_with_listener(rows, managed_capacity_bytes, query_cap_bytes, ListenMode::Prebound).await
    }

    pub(crate) async fn spawn_with_listener(
        rows: i64,
        managed_capacity_bytes: u64,
        query_cap_bytes: u64,
        listen_mode: ListenMode,
    ) -> Self {
        let _ = init_cuda_device(0);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("listener address");
        let mut config = server_config_for_concurrency(managed_capacity_bytes, query_cap_bytes);
        config.bind_addr = addr;
        let backend = backend_from_server_config(&config).expect("concurrency backend must build");

        let dir = tempfile::Builder::new().prefix("algeon-datafusion-concurrency-").tempdir().expect("tempdir");
        let parquet_path = dir.path().join("rows.parquet");
        write_rows_parquet(&parquet_path, rows);

        let shutdown = CancellationToken::new();
        let builder = FlightSqlServer::builder()
            .config(config)
            .backend(backend.clone())
            .shutdown(ServerShutdown::Token(shutdown.clone()));
        let builder = match listen_mode {
            ListenMode::Addr => {
                drop(listener);
                builder
            },
            ListenMode::Prebound => builder.listener(listener),
        };
        let server = builder.build().await.expect("concurrency server must build");
        let ctx = Arc::clone(server.session_context());
        ctx.register_parquet("concurrency_rows", parquet_path.to_str().unwrap(), ParquetReadOptions::default())
            .await
            .expect("register fixture");
        let small_path = dir.path().join("small.parquet");
        write_rows_parquet(&small_path, SMALL_TABLE_ROWS);
        ctx.register_parquet("concurrency_small", small_path.to_str().unwrap(), ParquetReadOptions::default())
            .await
            .expect("register small fixture");
        let server = tokio::spawn(async move { server.serve().await });
        Self { backend, ctx, addr, shutdown, server, _dir: dir }
    }

    pub(crate) async fn client(&self) -> FlightSqlServiceClient<Channel> {
        let uri = format!("http://{}", self.addr);
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut backoff = Duration::from_millis(10);
        loop {
            match Endpoint::from_shared(uri.clone()).expect("endpoint").connect().await {
                Ok(channel) => return FlightSqlServiceClient::new(channel),
                Err(error) => {
                    assert!(Instant::now() < deadline, "server did not accept connections: {error:?}");
                    sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_millis(250));
                },
            }
        }
    }

    /// Occupies one active slot until the returned holder is dropped.
    ///
    /// The stream is polled once so admission and GPU execution both start;
    /// leaving it suspended keeps the grant live without reaching EOF.
    pub(crate) async fn hold_one_slot(&self) -> (AlgeonAttempt, AttemptStream) {
        let attempt = self.backend.begin_attempt(AttemptOptions::default()).expect("holder attempt begins");
        let mut stream = attempt.sql(&self.ctx, CONCURRENCY_SQL).await.expect("holder plans");
        stream.try_next().await.expect("holder admits on first poll").expect("holder batch");
        (attempt, stream)
    }

    pub(crate) async fn wait_for_admission(&self, active: usize, queued: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let snapshot = self.backend.diagnostics().expect("admission snapshot").admission;
            if snapshot.active_queries == active && snapshot.queued_queries == queued {
                return;
            }
            assert!(Instant::now() < deadline, "expected active={active} queued={queued}, got {snapshot:?}");
            sleep(Duration::from_millis(20)).await;
        }
    }

    pub(crate) async fn shutdown(self) {
        self.shutdown.cancel();
        timeout(Duration::from_secs(20), self.server)
            .await
            .expect("shutdown drains")
            .expect("server task should not panic")
            .expect("server stops cleanly");
    }
}

/// Runs one attempt to completion and returns the report sealed for it.
///
/// The Flight surface projects only a presence flag for resource evidence, so
/// per-attempt allocator provenance has to be observed on the attempt API.
pub(crate) async fn sealed_report_for(server: &ConcurrencyServer, sql: &str) -> Arc<QueryAttemptReport> {
    let attempt = server.backend.begin_attempt(AttemptOptions::default()).expect("attempt begins");
    let sealed: Arc<Mutex<Option<Arc<QueryAttemptReport>>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&sealed);
    attempt.register_sealed_report_observer(move |report, _snapshot| {
        *sink.lock().expect("sealed report sink") = report;
    });
    let mut stream = attempt.sql(&server.ctx, sql).await.expect("plan");
    while stream.try_next().await.expect("batch").is_some() {}
    drop(stream);
    drop(attempt);

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(report) = sealed.lock().expect("sealed report sink").clone() {
            return report;
        }
        assert!(Instant::now() < deadline, "attempt never sealed a report");
        sleep(Duration::from_millis(20)).await;
    }
}
