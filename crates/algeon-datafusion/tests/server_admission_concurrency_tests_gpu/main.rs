//! Flight SQL admission ledger concurrency in the shared GPU lane.

#[path = "../support/server_admission_concurrency.rs"]
mod support;

use std::time::Duration;

use arrow::record_batch::RecordBatch;
use futures::TryStreamExt;
use query_runtime::observability::AttemptEvidenceProvenance;
use support::{
    CONCURRENCY_SQL, ConcurrencyServer, GIB, MANAGED_CAPACITY_BYTES, MAX_ACTIVE_ATTEMPTS, QUERY_CAP_BYTES,
    SMALL_TABLE_ROWS, sealed_report_for,
};
use tokio::time::timeout;

const UNDERSUBSCRIBED_QUERY_CAP_BYTES: u64 = 2 * GIB;

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
    let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
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
async fn the_active_attempt_limit_holds_a_fourth_query_back_while_capacity_is_still_free() {
    let server =
        ConcurrencyServer::spawn_with(SMALL_TABLE_ROWS, MANAGED_CAPACITY_BYTES, UNDERSUBSCRIBED_QUERY_CAP_BYTES).await;

    let holders = vec![server.hold_one_slot().await, server.hold_one_slot().await, server.hold_one_slot().await];
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 0).await;

    let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
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
        server.backend.diagnostics().expect("ledger report").ledgers.remove(0).grants.len(),
        MAX_ACTIVE_ATTEMPTS,
        "the fourth query waits on the active-attempt limit, not on capacity"
    );

    drop(holders);
    let batches = timeout(Duration::from_secs(120), queued)
        .await
        .expect("a freed slot admits the waiting query")
        .expect("waiting query task should not panic")
        .expect("waiting query succeeds");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn three_simultaneous_query_caps_fill_the_device_and_the_fourth_query_waits_without_a_grant() {
    let server = ConcurrencyServer::spawn(4096).await;

    let holders = vec![server.hold_one_slot().await, server.hold_one_slot().await, server.hold_one_slot().await];
    server.wait_for_admission(MAX_ACTIVE_ATTEMPTS, 0).await;

    let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
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

    let report = server.backend.diagnostics().expect("ledger report").ledgers.remove(0);
    assert_eq!(report.grants.len(), MAX_ACTIVE_ATTEMPTS, "a waiting query holds no grant: {:?}", report.grants);
    assert_eq!(report.live_grants_bytes, MANAGED_CAPACITY_BYTES);

    drop(holders);
    let batches = timeout(Duration::from_secs(120), queued)
        .await
        .expect("released capacity admits the waiting query")
        .expect("waiting query task should not panic")
        .expect("waiting query succeeds");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);

    server.shutdown().await;
}
