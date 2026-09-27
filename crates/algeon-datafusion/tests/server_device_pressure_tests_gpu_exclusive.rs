//! Establishes physical device memory pressure (nearly all free memory), so
//! these tests run alone.

#[path = "server_admission_concurrency/support.rs"]
mod support;

use std::time::{Duration, Instant};

use arrow::record_batch::RecordBatch;
use arrow_flight::error::FlightError;
use cudf::{DeviceId, diagnostics::memory::PhysicalMemoryPressureLease};
use futures::TryStreamExt;
use support::{CONCURRENCY_SQL, ConcurrencyServer, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES};
use tokio::time::sleep;

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
    let lease = PhysicalMemoryPressureLease::establish(DeviceId::new(0).expect("device zero"))
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
        FlightError::Tonic(status) => *status,
        other => panic!("expected a tonic Status, got {other:?}"),
    };
    let code = status
        .metadata()
        .get("x-algeon-error-code")
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
        sleep(Duration::from_millis(50)).await;
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
    let lease = PhysicalMemoryPressureLease::establish(DeviceId::new(0).expect("device zero"))
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
