#![cfg(all(feature = "server", feature = "iceberg"))]

use std::{net::TcpListener, time::Duration};

use arrow::array::{Int32Array, Int64Array, UInt32Array, UInt64Array};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion_nexus::server::{
    ServerConfig,
    config::{DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV, DATAFUSION_NEXUS_ICEBERG_TABLES_ENV},
    serve,
};
use futures::TryStreamExt;
use tonic::transport::Endpoint;

const EXPECTED_TPCH_SF1_LINEITEM_ROWS: i64 = 6_001_215;

/// Milestone 1 backward-compatibility smoke test: bare alias via pre-declared
/// `DATAFUSION_NEXUS_ICEBERG_TABLES`.  Requires the full set of Iceberg env
/// vars including `DATAFUSION_NEXUS_ICEBERG_TABLES` (or the legacy Glue alias).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires live AWS Glue/S3 credentials and network access"]
async fn glue_tpch_sf1_lineitem_count_via_flight_sql() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env().unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let row_count = query_count(&mut client, "SELECT count(*) AS row_count FROM lineitem").await;
    assert_eq!(row_count, EXPECTED_TPCH_SF1_LINEITEM_ROWS);

    server.abort();
}

/// Milestone 2 smoke test: three-part addressing via the dynamic Glue catalog
/// provider, **without** pre-declaring any tables in `DATAFUSION_NEXUS_ICEBERG_TABLES`.
///
/// Required env vars:
/// - `DATAFUSION_NEXUS_ICEBERG_NAMESPACE=demo_db`
/// - `DATAFUSION_NEXUS_ICEBERG_WAREHOUSE=s3://example-warehouse/demo_db/`
/// - `AWS_REGION=us-east-1`
/// - AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
///
/// Optional:
/// - `DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID` (account ID if cross-account)
///
/// `DATAFUSION_NEXUS_ICEBERG_TABLES` must NOT be set; this test verifies that
/// `SELECT count(*) FROM glue.demo_db.tpch_sf1_lineitem` works purely via
/// the dynamic catalog discovery path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires live AWS Glue/S3 credentials and network access"]
async fn glue_three_part_lineitem_count_dynamic_catalog() {
    let addr = free_local_addr();
    let config = ServerConfig::from_env_get(|name| {
        // Explicitly strip the tables env vars so the dynamic catalog is the
        // only path for table resolution.
        if name == DATAFUSION_NEXUS_ICEBERG_TABLES_ENV || name == DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV {
            return None;
        }
        std::env::var(name).ok()
    })
    .expect("failed to build ServerConfig; check Iceberg + AWS env vars");

    assert!(
        config.iceberg.as_ref().map_or(true, |ic| ic.tables.is_empty()),
        "test requires no pre-declared tables; unset DATAFUSION_NEXUS_ICEBERG_TABLES"
    );

    let mut config = config;
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    // Three-part address: catalog = "glue", schema = "demo_db",
    // table = "tpch_sf1_lineitem" (real Glue table name, no alias needed).
    let sql = "SELECT count(*) AS row_count FROM glue.demo_db.tpch_sf1_lineitem";
    let row_count = query_count(&mut client, sql).await;
    assert_eq!(
        row_count, EXPECTED_TPCH_SF1_LINEITEM_ROWS,
        "three-part dynamic catalog query returned unexpected row count"
    );

    server.abort();
}

/// Milestone 2 + Milestone 1 combined: pre-declared alias AND three-part real-name
/// both work in the same server instance.
///
/// Requires `DATAFUSION_NEXUS_ICEBERG_TABLES=lineitem=tpch_sf1_lineitem` plus
/// the standard Iceberg + AWS env vars.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires live AWS Glue/S3 credentials and network access"]
async fn glue_bare_alias_and_three_part_address_both_work() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env().unwrap();
    config.bind_addr = addr;

    assert!(
        config.iceberg.as_ref().map_or(false, |ic| !ic.tables.is_empty()),
        "test requires DATAFUSION_NEXUS_ICEBERG_TABLES to be set (e.g. lineitem=tpch_sf1_lineitem)"
    );

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    // Bare alias (Milestone 1 backward compat).
    let bare_count = query_count(&mut client, "SELECT count(*) AS rc FROM lineitem").await;
    assert_eq!(bare_count, EXPECTED_TPCH_SF1_LINEITEM_ROWS, "bare alias count mismatch");

    // Three-part real table name (Milestone 2 dynamic catalog + startup seeding).
    let three_part_count =
        query_count(&mut client, "SELECT count(*) AS rc FROM glue.demo_db.tpch_sf1_lineitem").await;
    assert_eq!(three_part_count, EXPECTED_TPCH_SF1_LINEITEM_ROWS, "three-part address count mismatch");

    server.abort();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn query_count(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, sql: &str) -> i64 {
    let flight_info = client.execute(sql.to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|ep| ep.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    int_value(batches[0].column(0)).unwrap_or_else(|| panic!("could not read count from result"))
}

fn free_local_addr() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

async fn connect_with_retry(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    let mut last_error = None;
    for _ in 0..100 {
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
        }
    }
    panic!("Flight SQL server did not accept connections at {addr}: {last_error:?}");
}

fn int_value(array: &arrow::array::ArrayRef) -> Option<i64> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|a| a.value(0))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|a| i64::from(a.value(0))))
        .or_else(|| array.as_any().downcast_ref::<UInt64Array>().map(|a| a.value(0) as i64))
        .or_else(|| array.as_any().downcast_ref::<UInt32Array>().map(|a| i64::from(a.value(0))))
}
