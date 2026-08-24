// Service admission semantics on the Flight SQL path: queue bounds,
// cancellation, client disconnect, and the queue-wait deadline.
//
// Every case occupies the single GPU slot with a suspended holder attempt
// rather than a timer, so queue depth is a coordinated state rather than a
// race against real device work.

/// Occupies the backend's one active slot until the returned holder is
/// dropped.
///
/// The stream is polled exactly once: the first poll admits, and leaving it
/// suspended keeps the grant live without ever reaching EOF.
async fn hold_the_only_gpu_slot(
    backend: &NexusGpuBackend,
    ctx: &Arc<SessionContext>,
    sql: &str,
) -> (datafusion_nexus::attempt::NexusAttempt, datafusion_nexus::attempt::AttemptStream) {
    let attempt = backend.begin_attempt(AttemptOptions::default()).expect("holder attempt should begin");
    let mut stream = attempt.sql(ctx, sql).await.expect("holder should plan");
    stream.try_next().await.expect("holder admits on first poll").expect("holder batch");
    assert_eq!(backend.admission_snapshot().unwrap().active_queries, 1, "holder occupies the only active slot");
    (attempt, stream)
}

async fn admission_test_server(
    max_queued_attempts: usize,
    max_queue_wait: Duration,
) -> (
    NexusGpuBackend,
    Arc<SessionContext>,
    FlightSqlServiceClient<tonic::transport::Channel>,
    CancellationToken,
    tokio::task::JoinHandle<nexus_server::Result<()>>,
    tempfile::TempDir,
) {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;
    config.gpu_device_profiles[0] = config.gpu_device_profiles[0].clone().with_max_active_attempts(1);
    config.queue_policy = config
        .queue_policy
        .clone()
        .with_max_queued_attempts(max_queued_attempts)
        .with_max_queue_wait(max_queue_wait);
    let backend = build_injected_test_backend(&mut config);

    let dir = tempfile::Builder::new().prefix("datafusion-nexus-admission-").tempdir().unwrap();
    let parquet_path = dir.path().join("edges.parquet");
    write_update_edges_parquet(&parquet_path);

    let shutdown = CancellationToken::new();
    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend.clone())
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown.clone()))
        .build()
        .await
        .unwrap();
    let ctx = Arc::clone(server.session_context());
    ctx.register_parquet("admission_edges", parquet_path.to_str().unwrap(), ParquetReadOptions::default())
        .await
        .unwrap();
    let server = tokio::spawn(async move { server.serve().await });
    let client = connect_with_retry(addr).await;
    (backend, ctx, client, shutdown, server, dir)
}

const ADMISSION_SQL: &str = "select src from admission_edges order by src limit 1";

async fn shutdown_admission_test_server(
    shutdown: CancellationToken,
    server: tokio::task::JoinHandle<nexus_server::Result<()>>,
) {
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(10), server)
        .await
        .expect("token shutdown should drain promptly")
        .expect("server task should not panic")
        .expect("token shutdown should stop the server cleanly");
}

/// Starts a `DoGet` whose admission will queue, and returns its ticket task.
async fn start_queued_do_get(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
) -> (FlightInfo, tokio::task::JoinHandle<std::result::Result<Vec<RecordBatch>, FlightError>>) {
    let flight_info = client.execute(ADMISSION_SQL.to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let mut do_get_client = client.clone();
    let task = tokio::spawn(async move { do_get_client.do_get(ticket).await?.try_collect::<Vec<_>>().await });
    (flight_info, task)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flight_do_get_rejects_a_full_admission_queue_as_resource_exhausted() {
    let (backend, ctx, mut client, shutdown, server, _dir) =
        admission_test_server(1, Duration::from_secs(60)).await;
    let holder = hold_the_only_gpu_slot(&backend, &ctx, ADMISSION_SQL).await;

    let (_first_info, first) = start_queued_do_get(&mut client).await;
    wait_for_shared_ledger_queue(&backend, 1).await;

    // The queue bound is one waiter, so the second DoGet has nowhere to wait.
    let (_second_info, second) = start_queued_do_get(&mut client).await;
    let status = unwrap_flight_status(second.await.unwrap().expect_err("a full queue must reject the second DoGet"));
    assert_eq!(status.code(), Code::ResourceExhausted);
    assert_error_metadata(&status, "service_overloaded", "resource_exhausted", "temporary");

    drop(holder);
    let batches = tokio::time::timeout(Duration::from_secs(60), first)
        .await
        .expect("the queued DoGet should finish once the holder releases")
        .expect("queued DoGet task should not panic")
        .expect("queued DoGet should succeed");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

    shutdown_admission_test_server(shutdown, server).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_flight_query_holds_no_gpu_grant_and_cancels_without_admission() {
    let (backend, ctx, mut client, shutdown, server, _dir) =
        admission_test_server(4, Duration::from_secs(60)).await;
    let holder = hold_the_only_gpu_slot(&backend, &ctx, ADMISSION_SQL).await;

    let (flight_info, queued) = start_queued_do_get(&mut client).await;
    wait_for_shared_ledger_queue(&backend, 1).await;

    // A queued request must not materialize GPU state: the only live grant is
    // the holder's.
    let reports = backend.device_ledger_reports().unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].grants.len(), 1, "a queued request must not hold a grant: {:?}", reports[0].grants);
    assert_eq!(reports[0].active_attempts, 1);

    let cancel = cancel_query_raw(&mut client, &flight_info).await;
    assert_eq!(cancel.result, CANCEL_RESULT_CANCELLING);

    let status = unwrap_flight_status(queued.await.unwrap().expect_err("a cancelled queued query must not succeed"));
    assert_eq!(status.code(), Code::Cancelled);

    // Removing the waiter frees the queue slot without ever issuing a grant.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = backend.admission_snapshot().unwrap();
        if snapshot.queued_queries == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "cancelled waiter was not removed: {snapshot:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(backend.device_ledger_reports().unwrap()[0].grants.len(), 1);

    drop(holder);
    wait_for_empty_gpu_ledger(&backend).await;
    shutdown_admission_test_server(shutdown, server).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disconnecting_a_queued_flight_client_releases_the_waiter_for_the_next_query() {
    let (backend, ctx, mut client, shutdown, server, _dir) =
        admission_test_server(4, Duration::from_secs(60)).await;
    let holder = hold_the_only_gpu_slot(&backend, &ctx, ADMISSION_SQL).await;

    let (_flight_info, queued) = start_queued_do_get(&mut client).await;
    wait_for_shared_ledger_queue(&backend, 1).await;

    // Client disconnect is the same lifecycle as an explicit cancel: aborting
    // the DoGet task drops the transport stream.
    queued.abort();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = backend.admission_snapshot().unwrap();
        if snapshot.queued_queries == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "disconnected waiter was not removed: {snapshot:?}");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    drop(holder);
    let batches = tokio::time::timeout(Duration::from_secs(60), execute_batches(&mut client, ADMISSION_SQL))
        .await
        .expect("the freed slot should admit the next query");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

    wait_for_empty_gpu_ledger(&backend).await;
    shutdown_admission_test_server(shutdown, server).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_queued_flight_query_times_out_against_the_configured_queue_wait() {
    // Long enough that resource-envelope acquisition is never the thing that
    // expires, short enough to keep the case bounded.
    let (backend, ctx, mut client, shutdown, server, _dir) =
        admission_test_server(4, Duration::from_secs(2)).await;
    let holder = hold_the_only_gpu_slot(&backend, &ctx, ADMISSION_SQL).await;

    let (_flight_info, queued) = start_queued_do_get(&mut client).await;
    let status = unwrap_flight_status(
        tokio::time::timeout(Duration::from_secs(30), queued)
            .await
            .expect("the queued DoGet should expire against the queue wait bound")
            .expect("queued DoGet task should not panic")
            .expect_err("an expired queue wait must not succeed"),
    );
    assert_eq!(status.code(), Code::ResourceExhausted);
    assert_error_metadata(&status, "service_overloaded", "resource_exhausted", "temporary");

    // The expired waiter is gone and never held a grant.
    assert_eq!(backend.admission_snapshot().unwrap().queued_queries, 0);
    assert_eq!(backend.device_ledger_reports().unwrap()[0].grants.len(), 1);

    drop(holder);
    wait_for_empty_gpu_ledger(&backend).await;
    shutdown_admission_test_server(shutdown, server).await;
}
