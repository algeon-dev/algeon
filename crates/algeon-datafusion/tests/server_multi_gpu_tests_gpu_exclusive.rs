//! Multi-device placement on the real Flight SQL service path.
//!
//! Every case needs two physical GPUs and exclusive ownership of both, so the
//! per-device ledgers observed here are not perturbed by another process.

use std::{
    collections::BTreeMap,
    fs::File,
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    attempt::{AlgeonAttempt, AttemptOptions, AttemptStream},
    backend::{AlgeonGpuBackend, GpuDeviceMemoryLimit, GpuMemoryOwnership},
    native_exec::AlgeonNativeExecutionConfig,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_GPU_DEVICES_ENV, FlightSqlServer, Result as ServerResult,
        ServerConfig, ServerShutdown, backend_from_server_config, init_cuda_device,
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
use tokio::{
    task::JoinHandle,
    time::{sleep, timeout},
};
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};

const GIB: u64 = 1024 * 1024 * 1024;
const DEVICE_CAPACITY_BYTES: u64 = 8 * GIB;
/// Deliberately unequal per-device query caps: a device must issue its own
/// configured limit rather than a fleet-wide one.
const DEVICE_QUERY_CAPS: [u64; 2] = [8 * GIB, 2 * GIB];

const PLACEMENT_SQL: &str = "select value from placement_rows order by value desc limit 4";

fn required_devices_present() -> bool {
    match cudf::DeviceId::count() {
        Ok(available) if available >= 2 => true,
        Ok(available) => {
            eprintln!("SKIP unsupported_gpu_inventory: requires 2 GPU devices, found {available}");
            false
        },
        Err(error) => {
            eprintln!("SKIP unsupported_gpu_inventory: could not enumerate GPUs: {error}");
            false
        },
    }
}

fn two_device_server_config() -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(
        &path,
        "[[admission.device_profiles]]\ndevice_ordinal = 0\n\n[[admission.device_profiles]]\ndevice_ordinal = 1\n",
    )
    .expect("write server config");
    let path = path.to_string_lossy().into_owned();
    let env =
        BTreeMap::from([(ALGEON_SERVER_CONFIG_FILE_ENV, path), (ALGEON_SERVER_GPU_DEVICES_ENV, "0,1".to_string())]);
    let mut config = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("two-device test profile");
    config.gpu_memory_ownership =
        config.gpu_devices.iter().copied().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)).collect();
    config.gpu_device_profiles = config
        .gpu_device_profiles
        .iter()
        .cloned()
        .map(|profile| {
            let cap = DEVICE_QUERY_CAPS[profile.device_ordinal()];
            profile
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: DEVICE_CAPACITY_BYTES })
                .with_backend_reserve_bytes(0)
                .with_default_query_limit_bytes(cap)
        })
        .collect();
    config.native_optimizer_config =
        config.native_optimizer_config.clone().with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
            DEVICE_QUERY_CAPS[1],
            AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            AlgeonNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));
    config.queue_policy =
        config.queue_policy.clone().with_max_queued_attempts(4).with_max_queue_wait(Duration::from_secs(120));
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

struct TwoDeviceServer {
    backend: AlgeonGpuBackend,
    ctx: Arc<SessionContext>,
    addr: SocketAddr,
    shutdown: CancellationToken,
    server: JoinHandle<ServerResult<()>>,
    _dir: tempfile::TempDir,
}

impl TwoDeviceServer {
    async fn spawn() -> Self {
        let _ = init_cuda_device(0);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("listener address");
        let mut config = two_device_server_config();
        config.bind_addr = addr;
        let backend = backend_from_server_config(&config).expect("two-device backend must build");

        let dir = tempfile::Builder::new().prefix("algeon-datafusion-placement-").tempdir().expect("tempdir");
        let parquet_path = dir.path().join("rows.parquet");
        write_rows_parquet(&parquet_path, 4096);

        let shutdown = CancellationToken::new();
        let server = FlightSqlServer::builder()
            .config(config)
            .backend(backend.clone())
            .listener(listener)
            .shutdown(ServerShutdown::Token(shutdown.clone()))
            .build()
            .await
            .expect("two-device server must build");
        let ctx = Arc::clone(server.session_context());
        ctx.register_parquet("placement_rows", parquet_path.to_str().unwrap(), ParquetReadOptions::default())
            .await
            .expect("register fixture");
        let server = tokio::spawn(async move { server.serve().await });
        Self { backend, ctx, addr, shutdown, server, _dir: dir }
    }

    async fn client(&self) -> FlightSqlServiceClient<Channel> {
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
    async fn hold_one_slot(&self) -> (AlgeonAttempt, AttemptStream) {
        let attempt = self.backend.begin_attempt(AttemptOptions::default()).expect("holder attempt begins");
        let mut stream = attempt.sql(&self.ctx, PLACEMENT_SQL).await.expect("holder plans");
        stream.try_next().await.expect("holder admits on first poll").expect("holder batch");
        (attempt, stream)
    }

    async fn wait_for_admission(&self, active: usize, queued: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let snapshot = self.backend.admission_snapshot().expect("admission snapshot");
            if snapshot.active_queries == active && snapshot.queued_queries == queued {
                return;
            }
            assert!(Instant::now() < deadline, "expected active={active} queued={queued}, got {snapshot:?}");
            sleep(Duration::from_millis(20)).await;
        }
    }

    async fn shutdown(self) {
        self.shutdown.cancel();
        timeout(Duration::from_secs(20), self.server)
            .await
            .expect("shutdown drains")
            .expect("server task should not panic")
            .expect("server stops cleanly");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[ignore = "requires at least 2 GPUs"]
async fn each_attempt_occupies_one_gpu_and_the_second_query_uses_the_other_device() {
    if !required_devices_present() {
        return;
    }
    let server = TwoDeviceServer::spawn().await;

    let first = server.hold_one_slot().await;
    let second = server.hold_one_slot().await;
    server.wait_for_admission(2, 0).await;

    // Each device ledger owns its own placement, so per-device active counts
    // are what prove the two attempts did not share a GPU.
    let reports = server.backend.device_ledger_reports().expect("ledger reports");
    assert_eq!(reports.len(), 2);
    for report in &reports {
        assert_eq!(report.active_attempts, 1, "device {} holds exactly one attempt", report.device_ordinal);
        assert_eq!(report.grants.len(), 1);
        assert_eq!(
            report.grants[0].grant_bytes, DEVICE_QUERY_CAPS[report.device_ordinal],
            "device {} issues its own configured default cap",
            report.device_ordinal
        );
    }

    drop(first);
    drop(second);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
#[ignore = "requires at least 2 GPUs"]
async fn a_waiting_flight_query_runs_on_whichever_eligible_device_frees_first() {
    if !required_devices_present() {
        return;
    }
    let server = TwoDeviceServer::spawn().await;

    let first = server.hold_one_slot().await;
    let second = server.hold_one_slot().await;
    server.wait_for_admission(2, 0).await;

    let mut client = server.client().await;
    let flight_info = client.execute(PLACEMENT_SQL.to_string(), None).await.expect("plan waiting query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut queued_client = client.clone();
    let queued = tokio::spawn(async move { queued_client.do_get(ticket).await?.try_collect::<Vec<_>>().await });
    server.wait_for_admission(2, 1).await;

    // Freeing exactly one device leaves the waiter a single eligible choice,
    // so the device it lands on is observable rather than a race.
    drop(second);
    let batches = timeout(Duration::from_secs(120), queued)
        .await
        .expect("the waiter admits once a device frees")
        .expect("waiting query task should not panic")
        .expect("waiting query succeeds");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    // The retained holder still pins its own device, so the waiter can only
    // have run on the one that was freed.
    let reports = server.backend.device_ledger_reports().expect("ledger reports");
    let busy = reports.iter().filter(|report| report.active_attempts == 1).count();
    assert_eq!(busy, 1, "exactly the retained holder's device stays busy: {reports:?}");
    for report in &reports {
        assert!(
            report.active_attempts <= report.max_active_attempts,
            "device {} exceeded its active limit",
            report.device_ordinal
        );
    }

    drop(first);
    server.shutdown().await;
}
