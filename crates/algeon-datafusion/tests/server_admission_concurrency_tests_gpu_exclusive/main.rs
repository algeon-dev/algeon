//! Same-device concurrency on the real Flight SQL service path.
//!
//! The ledger is in-process integer accounting against an explicit managed
//! capacity. These cases deliberately hold several live query domains and
//! therefore run in the exclusive GPU lane so unrelated kernels cannot stall
//! their bounded admission and shutdown deadlines.

#[path = "../support/server_admission_concurrency.rs"]
mod support;

mod transport;
use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use arrow_flight::{error::FlightError, sql::ActionCancelQueryRequest};
use futures::TryStreamExt;
use support::{
    CONCURRENCY_SQL, ConcurrencyServer, MANAGED_CAPACITY_BYTES, MAX_ACTIVE_ATTEMPTS, QUERY_CAP_BYTES, execution_value,
    query_stats,
};
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;
use tonic::Code;

/// A query cap small enough that a whole-table sort cannot fit inside it,
/// while still leaving room for two simultaneous grants.
const SMALL_QUERY_CAP_BYTES: u64 = 24 * 1024 * 1024;
const SMALL_MANAGED_CAPACITY_BYTES: u64 = 3 * SMALL_QUERY_CAP_BYTES;
const OVERSIZED_SORT_ROWS: i64 = 8_000_000;

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

    let error = timeout(Duration::from_secs(180), failing)
        .await
        .expect("the oversized query terminates")
        .expect("oversized query task should not panic")
        .expect_err("a sort that cannot fit the query cap must fail");
    let status = match error {
        FlightError::Tonic(status) => *status,
        other => panic!("expected a tonic Status, got {other:?}"),
    };
    assert_eq!(
        status.metadata().get("x-algeon-error-code").and_then(|value| value.to_str().ok()),
        Some("query_memory_limit_exceeded"),
        "the failure must be attributed to the query's own cap: {status:?}"
    );
    // The status and the stats of the failed query carry the same byte counts,
    // and the peak makes the failure add up: the request did not fit beside it.
    let message = status.message();
    assert!(message.contains("the allocation peak was"), "{message}");
    assert!(message.contains(&format!("the query grant is {SMALL_QUERY_CAP_BYTES} bytes")), "{message}");
    let metadata_bytes = |key: &str| -> u64 {
        let value = status.metadata().get(key).and_then(|value| value.to_str().ok());
        value.unwrap_or_else(|| panic!("{key} missing: {status:?}")).parse().expect("byte count metadata")
    };
    let requested = metadata_bytes("x-algeon-requested-allocation-bytes");
    let query_id = status
        .metadata()
        .get("x-algeon-query-id")
        .and_then(|value| value.to_str().ok())
        .expect("a failed statement carries its query id")
        .to_string();
    let payload = query_stats(&mut server.client().await, &query_id).await;
    let execution = |column: &str| execution_value(&payload, column);
    assert_eq!(execution("query_grant_bytes"), SMALL_QUERY_CAP_BYTES.to_string(), "{payload}");
    let peak: u64 = execution("allocator_high_water_bytes")
        .parse()
        .unwrap_or_else(|_| panic!("a failed attempt reports its allocator high-water mark:\n{payload}"));
    assert!(
        peak + requested > SMALL_QUERY_CAP_BYTES && peak <= SMALL_QUERY_CAP_BYTES,
        "peak {peak} and request {requested} must explain a failure against the {SMALL_QUERY_CAP_BYTES} byte grant"
    );
    for fact in [
        format!("fact.requested_allocation_bytes={requested}\n"),
        format!("fact.budget_bytes={SMALL_QUERY_CAP_BYTES}\n"),
    ] {
        assert!(payload.contains(&fact), "{fact} missing from the structured error:\n{payload}");
    }

    // The failed query's capacity returns to the ledger once its cleanup
    // completes; the holder's grant is untouched throughout.
    drop(holder);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.available_bytes, SMALL_MANAGED_CAPACITY_BYTES);
            break;
        }
        assert!(Instant::now() < deadline, "capacity was not returned after the failure: {report:?}");
        sleep(Duration::from_millis(50)).await;
    }

    server.shutdown().await;
}

/// Enough rows that a full sort stays on the device long enough to be
/// observed as admitted and then interrupted.
const LONG_RUNNING_SORT_ROWS: i64 = 4_000_000;
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
    // provably executing and already delivering results, so the cancellation
    // does not race admission.
    let delivered = stream.try_next().await.expect("first batch").expect("the query delivers before cancellation");
    assert!(delivered.num_rows() > 0);
    server.wait_for_admission(1, 0).await;

    let action_body = {
        use prost::Message as _;
        ActionCancelQueryRequest { info: flight_info.encode_to_vec().into() }
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
        FlightError::Tonic(status) => *status,
        other => panic!("expected a tonic Status, got {other:?}"),
    };
    assert_eq!(status.code(), Code::Cancelled);

    // Cancellation does not preempt launched kernels, but cleanup must still
    // drain and release the whole cap before the ledger reports it free.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.available_bytes, MANAGED_CAPACITY_BYTES);
            break;
        }
        assert!(Instant::now() < deadline, "cancelled query did not return its capacity: {report:?}");
        sleep(Duration::from_millis(50)).await;
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
        // to apply backpressure and does not buffer the whole result.
        sleep(Duration::from_millis(20)).await;
        let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
        if report.live_grants_bytes == QUERY_CAP_BYTES {
            saw_cap_held_mid_stream = true;
        }
    }
    assert_eq!(rows, 1_000_000);
    assert!(batches > 1, "a million rows must arrive as more than one batch to exercise a slow consumer");
    assert!(saw_cap_held_mid_stream, "the query cap stays charged while its result stream is still draining");

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "capacity was not released after the stream drained: {report:?}");
        sleep(Duration::from_millis(50)).await;
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
    // report sealed, or a cleaning attempt whose cap was issued again,
    // would show up here as an over-issue.
    let watcher_backend = server.backend.clone();
    let watcher_stop = CancellationToken::new();
    let watcher_token = watcher_stop.clone();
    let watcher = tokio::spawn(async move {
        let mut worst = (0usize, 0u64);
        while !watcher_token.is_cancelled() {
            if let Ok(mut reports) = watcher_backend.diagnostics().map(|diagnostics| diagnostics.ledgers) {
                let report = reports.remove(0);
                worst.0 = worst.0.max(report.grants.len());
                worst.1 = worst.1.max(report.live_grants_bytes);
            }
            sleep(Duration::from_millis(5)).await;
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
        let error = timeout(Duration::from_secs(300), task)
            .await
            .unwrap_or_else(|_| panic!("backlog query {index} did not terminate"))
            .unwrap_or_else(|error| panic!("backlog query {index} panicked: {error:?}"))
            .expect_err("every backlog query exceeds its cap");
        let status = match error {
            FlightError::Tonic(status) => *status,
            other => panic!("expected a tonic Status, got {other:?}"),
        };
        assert_eq!(
            status.metadata().get("x-algeon-error-code").and_then(|value| value.to_str().ok()),
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
        let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
        if report.grants.is_empty() && report.live_grants_bytes == 0 {
            assert_eq!(report.available_bytes, SMALL_MANAGED_CAPACITY_BYTES);
            break;
        }
        assert!(Instant::now() < deadline, "cleanup backlog did not drain: {report:?}");
        sleep(Duration::from_millis(50)).await;
    }

    server.shutdown().await;
}
