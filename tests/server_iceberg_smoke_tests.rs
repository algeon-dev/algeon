#![cfg(all(feature = "server", feature = "iceberg"))]

use std::{net::TcpListener, time::Duration};

use arrow::array::{Int32Array, Int64Array, UInt32Array, UInt64Array};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion_cudf::server::{ServerConfig, serve};
use futures::TryStreamExt;
use tonic::transport::Endpoint;

const EXPECTED_TPCH_SF1_LINEITEM_ROWS: i64 = 6_001_215;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires live AWS Glue/S3 credentials and network access"]
async fn glue_tpch_sf1_lineitem_count_via_flight_sql() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env().unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select count(*) as row_count from lineitem".to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(EXPECTED_TPCH_SF1_LINEITEM_ROWS));
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
        .map(|array| array.value(0))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|array| i64::from(array.value(0))))
        .or_else(|| array.as_any().downcast_ref::<UInt64Array>().map(|array| array.value(0) as i64))
        .or_else(|| array.as_any().downcast_ref::<UInt32Array>().map(|array| i64::from(array.value(0))))
}
