//! Same-device concurrency on the real Flight SQL service path.
//!
//! These cases need the whole device: they configure an explicit managed
//! capacity and assert the ledger's exact grant accounting, which another test
//! process sharing the GPU would perturb.

use std::{
    fs::File,
    net::TcpListener,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use arrow::{
    array::{ArrayRef, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use datafusion_nexus::{
    attempt::{AttemptOptions, AttemptStream, NexusAttempt},
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceInstallationBounds},
    native_exec::NexusNativeExecutionConfig,
};
use futures::TryStreamExt;
use nexus_query_engine::observability::{AttemptEvidenceProvenance, QueryAttemptReport};
use nexus_server::{FlightSqlServer, ServerConfig, ServerShutdown};
use parquet::arrow::ArrowWriter;
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;

const GIB: u64 = 1024 * 1024 * 1024;
/// The worked concurrency configuration from the admission design: a 12 GiB
/// managed capacity issuing three simultaneous 4 GiB query caps.
const MANAGED_CAPACITY_BYTES: u64 = 12 * GIB;
const QUERY_CAP_BYTES: u64 = 4 * GIB;
const MAX_ACTIVE_ATTEMPTS: usize = 3;

const CONCURRENCY_SQL: &str = "select value from concurrency_small order by value desc limit 4";
/// Rows in the always-small table used by slot holders and bounded queries.
const SMALL_TABLE_ROWS: i64 = 4096;

fn server_config_for_concurrency(managed_capacity_bytes: u64, query_cap_bytes: u64) -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").expect("write server config");
    let path = path.to_string_lossy().into_owned();
    let mut config =
        ServerConfig::from_env_get(|name| (name == nexus_server::NEXUS_SERVER_CONFIG_FILE_ENV).then(|| path.clone()))
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
        .with_installation_bounds(NexusGpuDeviceInstallationBounds::new(
            4 * MAX_ACTIVE_ATTEMPTS,
            2 * MAX_ACTIVE_ATTEMPTS,
            MAX_ACTIVE_ATTEMPTS,
        ));
    config.native_optimizer_config =
        config.native_optimizer_config.clone().with_native_execution_config(NexusNativeExecutionConfig::bounded(
            query_cap_bytes,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
        ));
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

struct ConcurrencyServer {
    backend: NexusGpuBackend,
    ctx: Arc<SessionContext>,
    addr: std::net::SocketAddr,
    shutdown: CancellationToken,
    server: tokio::task::JoinHandle<nexus_server::Result<()>>,
    _dir: tempfile::TempDir,
}

impl ConcurrencyServer {
    async fn spawn(rows: i64) -> Self {
        Self::spawn_with(rows, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES).await
    }

    async fn spawn_with(rows: i64, managed_capacity_bytes: u64, query_cap_bytes: u64) -> Self {
        let _ = nexus_server::init_cuda_device(0);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let addr = listener.local_addr().expect("listener address");
        let mut config = server_config_for_concurrency(managed_capacity_bytes, query_cap_bytes);
        config.bind_addr = addr;
        let backend = nexus_server::backend_from_server_config(&config).expect("concurrency backend must build");

        let dir = tempfile::Builder::new().prefix("datafusion-nexus-concurrency-").tempdir().expect("tempdir");
        let parquet_path = dir.path().join("rows.parquet");
        write_rows_parquet(&parquet_path, rows);

        let shutdown = CancellationToken::new();
        let server = FlightSqlServer::builder()
            .config(config)
            .backend(backend.clone())
            .listener(listener)
            .shutdown(ServerShutdown::Token(shutdown.clone()))
            .build()
            .await
            .expect("concurrency server must build");
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

    async fn client(&self) -> FlightSqlServiceClient<tonic::transport::Channel> {
        let uri = format!("http://{}", self.addr);
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut backoff = Duration::from_millis(10);
        loop {
            match Endpoint::from_shared(uri.clone()).expect("endpoint").connect().await {
                Ok(channel) => return FlightSqlServiceClient::new(channel),
                Err(error) => {
                    assert!(Instant::now() < deadline, "server did not accept connections: {error:?}");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_millis(250));
                },
            }
        }
    }

    /// Occupies one active slot until the returned holder is dropped.
    ///
    /// The stream is polled once so admission and GPU execution both start;
    /// leaving it suspended keeps the grant live without reaching EOF.
    async fn hold_one_slot(&self) -> (NexusAttempt, AttemptStream) {
        let attempt = self.backend.begin_attempt(AttemptOptions::default()).expect("holder attempt begins");
        let mut stream = attempt.sql(&self.ctx, CONCURRENCY_SQL).await.expect("holder plans");
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
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn shutdown(self) {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(20), self.server)
            .await
            .expect("shutdown drains")
            .expect("server task should not panic")
            .expect("server stops cleanly");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn three_simultaneous_query_caps_fill_the_device_and_the_fourth_query_waits_without_a_grant() {
    let server = ConcurrencyServer::spawn(4096).await;

    let holders = vec![server.hold_one_slot().await, server.hold_one_slot().await, server.hold_one_slot().await];
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 0).await;

    let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
    assert_eq!(report.managed_capacity_bytes, MANAGED_CAPACITY_BYTES);
    assert_eq!(report.max_active_attempts, MAX_ACTIVE_ATTEMPTS);
    assert_eq!(report.active_attempts, MAX_ACTIVE_ATTEMPTS);
    assert_eq!(report.grants.len(), MAX_ACTIVE_ATTEMPTS, "each active attempt owns one grant: {:?}", report.grants);
    for grant in &report.grants {
        assert_eq!(grant.grant_bytes, QUERY_CAP_BYTES, "every attempt is issued the configured default cap");
    }
    assert_eq!(report.live_grants_bytes, MANAGED_CAPACITY_BYTES, "three 4 GiB caps exactly fill the 12 GiB capacity");
    assert_eq!(report.available_bytes, 0);

    // A fourth query has neither an active slot nor spare capacity, so it
    // waits instead of being issued a smaller cap.
    let mut client = server.client().await;
    let flight_info = client.execute(CONCURRENCY_SQL.to_string(), None).await.expect("plan fourth query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut queued_client = client.clone();
    let queued = tokio::spawn(async move { queued_client.do_get(ticket).await?.try_collect::<Vec<_>>().await });
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 1).await;

    let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
    assert_eq!(report.grants.len(), MAX_ACTIVE_ATTEMPTS, "a waiting query holds no grant: {:?}", report.grants);
    assert_eq!(report.live_grants_bytes, MANAGED_CAPACITY_BYTES);

    drop(holders);
    let batches = tokio::time::timeout(Duration::from_secs(120), queued)
        .await
        .expect("released capacity admits the waiting query")
        .expect("waiting query task should not panic")
        .expect("waiting query succeeds");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    server.shutdown().await;
}

/// A cap small enough that the device could fund more simultaneous grants
/// than `max_active_attempts` allows, isolating the active-attempt limit from
/// the capacity equation.
const UNDERSUBSCRIBED_QUERY_CAP_BYTES: u64 = 2 * GIB;

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn the_active_attempt_limit_holds_a_fourth_query_back_while_capacity_is_still_free() {
    let server =
        ConcurrencyServer::spawn_with(SMALL_TABLE_ROWS, MANAGED_CAPACITY_BYTES, UNDERSUBSCRIBED_QUERY_CAP_BYTES).await;

    let holders = vec![server.hold_one_slot().await, server.hold_one_slot().await, server.hold_one_slot().await];
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 0).await;

    let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
    let live = UNDERSUBSCRIBED_QUERY_CAP_BYTES * MAX_ACTIVE_ATTEMPTS as u64;
    assert_eq!(report.live_grants_bytes, live);
    assert!(
        report.available_bytes >= UNDERSUBSCRIBED_QUERY_CAP_BYTES,
        "the device still has room for another cap: {report:?}"
    );

    // Capacity alone would fund a fourth grant, so anything that admits here
    // would have to be ignoring the active-attempt limit.
    let mut client = server.client().await;
    let flight_info = client.execute(CONCURRENCY_SQL.to_string(), None).await.expect("plan fourth query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut queued_client = client.clone();
    let queued = tokio::spawn(async move { queued_client.do_get(ticket).await?.try_collect::<Vec<_>>().await });
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 1).await;
    assert_eq!(
        server.backend.device_ledger_reports().expect("ledger report").remove(0).grants.len(),
        MAX_ACTIVE_ATTEMPTS,
        "the fourth query waits on the active-attempt limit, not on capacity"
    );

    drop(holders);
    let batches = tokio::time::timeout(Duration::from_secs(120), queued)
        .await
        .expect("a freed slot admits the waiting query")
        .expect("waiting query task should not panic")
        .expect("waiting query succeeds");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn flight_queries_execute_concurrently_on_one_device_with_independent_grants() {
    let server = ConcurrencyServer::spawn(4_000_000).await;

    // Pin two slots so the third, real Flight query is guaranteed to execute
    // while two other query domains are live rather than racing to overlap.
    let holders = vec![server.hold_one_slot().await, server.hold_one_slot().await];
    server.wait_for_admission(2, 0).await;

    let mut client = server.client().await;
    let flight_info = client
        .execute("select value from concurrency_rows order by value desc limit 1000000".to_string(), None)
        .await
        .expect("plan overlapping query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut stream = client.do_get(ticket).await.expect("overlapping DoGet");
    let first = stream.try_next().await.expect("first Flight batch").expect("the overlapping query delivers rows");

    // The million-row result exceeds the bounded Flight channel. Holding the
    // client stream after its first batch keeps the server-side attempt live,
    // so this observes all three grants directly instead of sampling a short
    // query and hoping a timer catches it.
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 0).await;
    let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
    assert_eq!(report.grants.len(), MAX_ACTIVE_ATTEMPTS, "both holders and the Flight query own grants: {report:?}");
    assert!(report.grants.iter().all(|grant| grant.grant_bytes == QUERY_CAP_BYTES));

    let mut rows = first.num_rows();
    while let Some(batch) = stream.try_next().await.expect("remaining Flight batches") {
        rows += batch.num_rows();
    }
    assert_eq!(rows, 1_000_000);

    drop(holders);
    server.shutdown().await;
}

/// A query cap small enough that a whole-table sort cannot fit inside it,
/// while still leaving room for two simultaneous grants.
const SMALL_QUERY_CAP_BYTES: u64 = 24 * 1024 * 1024;
const SMALL_MANAGED_CAPACITY_BYTES: u64 = 3 * SMALL_QUERY_CAP_BYTES;
const OVERSIZED_SORT_ROWS: i64 = 24_000_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn one_query_exceeding_its_cap_leaves_the_other_query_and_the_ledger_intact() {
    let server =
        ConcurrencyServer::spawn_with(OVERSIZED_SORT_ROWS, SMALL_MANAGED_CAPACITY_BYTES, SMALL_QUERY_CAP_BYTES).await;

    // Hold one slot so both queries are on the device at the same time, then
    // run the oversized sort and a bounded query concurrently.
    let holder = server.hold_one_slot().await;
    server.wait_for_admission(1, 0).await;

    let mut failing_client = server.client().await;
    let failing = tokio::spawn(async move {
        let flight_info =
            failing_client.execute("select value from concurrency_rows order by value desc".to_string(), None).await?;
        let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
        failing_client.do_get(ticket).await?.try_collect::<Vec<_>>().await
    });

    let mut surviving_client = server.client().await;
    let flight_info = surviving_client.execute(CONCURRENCY_SQL.to_string(), None).await.expect("plan bounded query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let surviving = surviving_client
        .do_get(ticket)
        .await
        .expect("bounded DoGet")
        .try_collect::<Vec<_>>()
        .await
        .expect("the bounded query is unaffected by the other query's memory failure");
    assert_eq!(surviving.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    let error = tokio::time::timeout(Duration::from_secs(180), failing)
        .await
        .expect("the oversized query terminates")
        .expect("oversized query task should not panic")
        .expect_err("a sort that cannot fit the query cap must fail");
    let status = match error {
        arrow_flight::error::FlightError::Tonic(status) => *status,
        other => panic!("expected a tonic Status, got {other:?}"),
    };
    assert_eq!(
        status.metadata().get("x-datafusion-nexus-error-code").and_then(|value| value.to_str().ok()),
        Some("query_memory_limit_exceeded"),
        "the failure must be attributed to the query's own cap: {status:?}"
    );

    // The failed query's capacity returns to the ledger once its cleanup
    // completes; the holder's grant is untouched throughout.
    drop(holder);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.available_bytes, SMALL_MANAGED_CAPACITY_BYTES);
            break;
        }
        assert!(Instant::now() < deadline, "capacity was not returned after the failure: {report:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    server.shutdown().await;
}

/// Enough rows that a full sort stays on the device long enough to be
/// observed as admitted and then interrupted.
const LONG_RUNNING_SORT_ROWS: i64 = 24_000_000;
const LONG_RUNNING_SORT_SQL: &str = "select value from concurrency_rows order by value desc";

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn cancelling_a_streaming_flight_query_fails_it_whole_and_returns_its_capacity() {
    let server = ConcurrencyServer::spawn(LONG_RUNNING_SORT_ROWS).await;
    let mut client = server.client().await;
    let mut canceller = server.client().await;

    let flight_info = client.execute(LONG_RUNNING_SORT_SQL.to_string(), None).await.expect("plan long query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut stream = client.do_get(ticket).await.expect("streaming DoGet");

    // Taking one batch first makes the cancellation land on a query that is
    // provably executing and already delivering results, rather than racing
    // admission.
    let delivered = stream.try_next().await.expect("first batch").expect("the query delivers before cancellation");
    assert!(delivered.num_rows() > 0);
    server.wait_for_admission(1, 0).await;

    let action_body = {
        use prost::Message as _;
        arrow_flight::sql::ActionCancelQueryRequest { info: flight_info.encode_to_vec().into() }
    };
    let action = {
        use arrow_flight::sql::ProstMessageExt as _;
        use prost::Message as _;
        arrow_flight::Action { r#type: "CancelQuery".to_string(), body: action_body.as_any().encode_to_vec().into() }
    };
    canceller.do_action(action).await.expect("CancelQuery action").message().await.expect("cancel result");

    // Already-delivered batches do not make the query a partial success: the
    // rest of the stream terminates as a cancellation, with no replay.
    let mut error = None;
    while error.is_none() {
        match stream.try_next().await {
            Ok(Some(_)) => continue,
            Ok(None) => panic!("a cancelled stream must not reach a clean end of stream"),
            Err(failure) => error = Some(failure),
        }
    }
    let status = match error.expect("cancelled stream reports an error") {
        arrow_flight::error::FlightError::Tonic(status) => *status,
        other => panic!("expected a tonic Status, got {other:?}"),
    };
    assert_eq!(status.code(), tonic::Code::Cancelled);

    // Cancellation does not preempt launched kernels, but cleanup must still
    // drain and hand the whole cap back before the ledger reports it free.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.available_bytes, MANAGED_CAPACITY_BYTES);
            break;
        }
        assert!(Instant::now() < deadline, "cancelled query did not return its capacity: {report:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_slow_flight_consumer_keeps_the_query_cap_until_its_stream_drains() {
    let server = ConcurrencyServer::spawn(4_000_000).await;
    let mut client = server.client().await;

    let flight_info = client
        .execute("select value from concurrency_rows order by value desc limit 1000000".to_string(), None)
        .await
        .expect("plan streaming query");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut stream = client.do_get(ticket).await.expect("streaming DoGet");

    let mut rows = 0usize;
    let mut batches = 0usize;
    let mut saw_cap_held_mid_stream = false;
    while let Some(batch) = stream.try_next().await.expect("streaming batch") {
        rows += batch.num_rows();
        batches += 1;
        // Read deliberately slowly so the server's bounded output channel has
        // to apply backpressure instead of buffering the whole result.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
        if report.live_grants_bytes == QUERY_CAP_BYTES {
            saw_cap_held_mid_stream = true;
        }
    }
    assert_eq!(rows, 1_000_000);
    assert!(batches > 1, "a million rows must arrive as more than one batch to exercise a slow consumer");
    assert!(saw_cap_held_mid_stream, "the query cap stays charged while its result stream is still draining");

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "capacity was not released after the stream drained: {report:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    server.shutdown().await;
}

/// Enough back-to-back failures that cleanup for earlier attempts is still
/// draining while later ones terminate.
const CLEANUP_BACKLOG_QUERIES: usize = 6;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_backlog_of_failing_queries_never_lets_the_ledger_over_issue_capacity() {
    let server =
        ConcurrencyServer::spawn_with(OVERSIZED_SORT_ROWS, SMALL_MANAGED_CAPACITY_BYTES, SMALL_QUERY_CAP_BYTES).await;

    // Sample the ledger throughout: a grant released before its attempt's
    // report sealed, or a cleaning attempt whose cap was handed out again,
    // would show up here as an over-issue.
    let watcher_backend = server.backend.clone();
    let watcher_stop = CancellationToken::new();
    let watcher_token = watcher_stop.clone();
    let watcher = tokio::spawn(async move {
        let mut worst = (0usize, 0u64);
        while !watcher_token.is_cancelled() {
            if let Ok(mut reports) = watcher_backend.device_ledger_reports() {
                let report = reports.remove(0);
                worst.0 = worst.0.max(report.grants.len());
                worst.1 = worst.1.max(report.live_grants_bytes);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        worst
    });

    let mut failing = Vec::new();
    for _ in 0..CLEANUP_BACKLOG_QUERIES {
        let mut client = server.client().await;
        failing.push(tokio::spawn(async move {
            let flight_info =
                client.execute("select value from concurrency_rows order by value desc".to_string(), None).await?;
            let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
            client.do_get(ticket).await?.try_collect::<Vec<_>>().await
        }));
    }

    for (index, task) in failing.into_iter().enumerate() {
        let error = tokio::time::timeout(Duration::from_secs(300), task)
            .await
            .unwrap_or_else(|_| panic!("backlog query {index} did not terminate"))
            .unwrap_or_else(|error| panic!("backlog query {index} panicked: {error:?}"))
            .expect_err("every backlog query exceeds its cap");
        let status = match error {
            arrow_flight::error::FlightError::Tonic(status) => *status,
            other => panic!("expected a tonic Status, got {other:?}"),
        };
        assert_eq!(
            status.metadata().get("x-datafusion-nexus-error-code").and_then(|value| value.to_str().ok()),
            Some("query_memory_limit_exceeded"),
            "backlog query {index} must fail on its own cap: {status:?}"
        );
    }

    watcher_stop.cancel();
    let (peak_grants, peak_live_bytes) = watcher.await.expect("watcher task should not panic");
    assert!(
        peak_grants <= MAX_ACTIVE_ATTEMPTS,
        "cleanup backlog issued {peak_grants} simultaneous grants for {MAX_ACTIVE_ATTEMPTS} slots"
    );
    assert!(
        peak_live_bytes <= SMALL_MANAGED_CAPACITY_BYTES,
        "cleanup backlog over-issued capacity: {peak_live_bytes} > {SMALL_MANAGED_CAPACITY_BYTES}"
    );

    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.available_bytes, SMALL_MANAGED_CAPACITY_BYTES);
            break;
        }
        assert!(Instant::now() < deadline, "cleanup backlog did not drain: {report:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    server.shutdown().await;
}

/// Large enough that the sort cannot fit in the few hundred MiB the pressure
/// lease deliberately leaves free, while staying far inside the query cap so
/// the ledger admits it and the driver is what refuses.
const PHYSICAL_PRESSURE_SORT_ROWS: i64 = 48_000_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_physically_exhausted_device_fails_the_query_without_quarantining_itself() {
    let server =
        ConcurrencyServer::spawn_with(PHYSICAL_PRESSURE_SORT_ROWS, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES).await;

    // The lease allocates through its own CUDA resource, outside the manager
    // the ledger accounts for, so admission still succeeds and the failure has
    // to come from the driver rather than from the query's cap.
    let lease =
        cudf::testing::memory::PhysicalMemoryPressureLease::establish(cudf::DeviceId::new(0).expect("device zero"))
            .expect("bounded physical pressure must be establishable on an installed device");
    let facts = lease.facts();
    assert!(facts.bytes_held > 0, "the lease must actually hold device memory: {facts:?}");

    let mut client = server.client().await;
    let flight_info = client
        .execute("select value from concurrency_rows order by value desc".to_string(), None)
        .await
        .expect("plan sort under pressure");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let error = client
        .do_get(ticket)
        .await
        .expect("DoGet under pressure")
        .try_collect::<Vec<_>>()
        .await
        .expect_err("a sort that cannot fit in free device memory must fail");
    let status = match error {
        arrow_flight::error::FlightError::Tonic(status) => *status,
        other => panic!("expected a tonic Status, got {other:?}"),
    };
    let code = status
        .metadata()
        .get("x-datafusion-nexus-error-code")
        .and_then(|value| value.to_str().ok())
        .expect("failure carries an error code")
        .to_string();
    assert_eq!(
        code, "device_out_of_memory",
        "physical exhaustion must be reported with confirmed physical provenance, not as a cap or unknown-source \
         failure: {status:?}"
    );

    // Allocation failure alone is not a device fault: the ledger must stay
    // open and the capacity must come back.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.health, "open", "a physical OOM must not quarantine the device: {report:?}");
            break;
        }
        assert!(Instant::now() < deadline, "capacity was not returned after physical OOM: {report:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let released = lease.release_and_verify().expect("pressure lease releases cleanly");
    assert!(released.recovered_bytes.is_some(), "released pressure must be observable: {released:?}");

    // The device is reusable once the external pressure is gone.
    let flight_info = client.execute(CONCURRENCY_SQL.to_string(), None).await.expect("plan after release");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let batches = client
        .do_get(ticket)
        .await
        .expect("DoGet after release")
        .try_collect::<Vec<_>>()
        .await
        .expect("the device is reusable once external pressure is gone");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    server.shutdown().await;
}

/// Runs one attempt to completion and returns the report sealed for it.
///
/// The Flight surface projects only a presence flag for resource evidence, so
/// per-attempt allocator provenance has to be observed on the attempt API.
async fn sealed_report_for(server: &ConcurrencyServer, sql: &str) -> Arc<QueryAttemptReport> {
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
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn overlapping_attempts_report_independent_peaks_and_no_unattributed_allocations() {
    let server = ConcurrencyServer::spawn(4_000_000).await;

    // A large sort and a tiny one, run while a third slot is held, so both
    // domains are live on the device at the same time.
    let holder = server.hold_one_slot().await;
    let large = sealed_report_for(&server, "select value from concurrency_rows order by value desc limit 1000");
    let small = sealed_report_for(&server, CONCURRENCY_SQL);
    let (large, small) = tokio::join!(large, small);
    drop(holder);

    for (label, report) in [("large", &large), ("small", &small)] {
        let evidence = report
            .resource_evidence
            .as_ref()
            .unwrap_or_else(|| panic!("{label} attempt must publish resource evidence"));
        assert_eq!(
            evidence.unattributed_device_allocator_event.provenance,
            AttemptEvidenceProvenance::Unavailable,
            "{label} attempt retained a device-level allocator sample it could not attribute: {:?}",
            evidence.unattributed_device_allocator_event
        );
    }

    // Each attempt's logical peak belongs to its own domain: the domains are
    // distinct and the larger sort must account for strictly more bytes.
    let large_usage = large.final_domain_usage.expect("large attempt reports its domain usage");
    let small_usage = small.final_domain_usage.expect("small attempt reports its domain usage");
    assert_ne!(large_usage.domain_id, small_usage.domain_id, "overlapping attempts must not share a domain");
    assert!(small_usage.peak_bytes > 0, "every executed attempt has a nonzero logical peak");
    assert!(
        large_usage.peak_bytes > small_usage.peak_bytes,
        "per-domain peaks are independent: large={} small={}",
        large_usage.peak_bytes,
        small_usage.peak_bytes
    );
    assert_eq!(large_usage.current_bytes, 0, "a sealed attempt's domain is drained");
    assert_eq!(small_usage.current_bytes, 0, "a sealed attempt's domain is drained");

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn a_bounded_streaming_query_survives_device_starvation_arriving_mid_stream() {
    let server =
        ConcurrencyServer::spawn_with(PHYSICAL_PRESSURE_SORT_ROWS, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES).await;
    let mut client = server.client().await;

    let flight_info = client
        .execute("select value from concurrency_rows".to_string(), None)
        .await
        .expect("plan streaming projection");
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let mut stream = client.do_get(ticket).await.expect("streaming DoGet");

    let first = stream.try_next().await.expect("first batch").expect("the query delivers before starvation");
    let mut rows = first.num_rows();
    let mut batches = 1usize;

    // Take every free byte on the device once the query is already streaming.
    let lease =
        cudf::testing::memory::PhysicalMemoryPressureLease::establish(cudf::DeviceId::new(0).expect("device zero"))
            .expect("bounded physical pressure must be establishable");
    let free_under_pressure = lease.facts().free_under_pressure_bytes;

    // Bounded streaming releases each chunk before reading the next, so the
    // query's resident set stays flat and it never has to grow its pool into
    // the starved device -- however far the result set exceeds free memory.
    while let Some(batch) =
        stream.try_next().await.expect("a flat-working-set query must not fail when free memory disappears")
    {
        rows += batch.num_rows();
        batches += 1;
    }
    assert_eq!(rows, PHYSICAL_PRESSURE_SORT_ROWS as usize);
    assert!(batches > 1, "the projection must actually stream");
    assert!(
        free_under_pressure < (rows as u64).saturating_mul(size_of::<i64>() as u64),
        "the run only means something when the result set exceeds free device memory: free={free_under_pressure} \
         rows={rows}"
    );

    lease.release_and_verify().expect("pressure lease releases cleanly");
    server.shutdown().await;
}
