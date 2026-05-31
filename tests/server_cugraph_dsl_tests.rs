#![cfg(all(feature = "server", feature = "cugraph"))]

use std::{
    collections::BTreeMap,
    fs,
    net::TcpListener,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{error::FlightError, sql::client::FlightSqlServiceClient};
use datafusion_nexus::server::{
    DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, ServerConfig, serve,
};
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use tonic::{Code, transport::Endpoint};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cugraph_dsl_metadata_and_dry_run_work_over_flight_sql_view() {
    let addr = free_local_addr();
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let mut config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;
    let dir = unique_temp_dir("datafusion-nexus-server-cugraph-flight");
    let parquet_path = dir.join("edges.parquet");
    write_edge_parquet(&parquet_path);

    execute_batches(
        &mut client,
        &format!("CREATE EXTERNAL TABLE raw_edges STORED AS PARQUET LOCATION '{}'", parquet_path.to_string_lossy()),
    )
    .await;
    execute_batches(&mut client, "CREATE VIEW target_edges AS SELECT src, dst, weight FROM raw_edges WHERE src >= 0")
        .await;

    let batches = execute_batches(
        &mut client,
        "SELECT function_name, enabled \
         FROM cugraph_list_algorithms() \
         WHERE function_name IN ('cugraph_bfs', 'cugraph_pagerank') \
         ORDER BY function_name",
    )
    .await;
    let batch = single_batch(&batches);
    assert_eq!(string_value(batch, 0, 0), "cugraph_bfs");
    assert!(bool_value(batch, 1, 0));
    assert_eq!(string_value(batch, 0, 1), "cugraph_pagerank");
    assert!(bool_value(batch, 1, 1));

    let batches = execute_batches(
        &mut client,
        r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, weight_dtype, would_execute_gpu
           FROM cugraph_validate_call(
             'cugraph_bfs',
             'target_edges',
             '{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":"weight","options":{"depth_limit":2}}'
           )"#,
    )
    .await;
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));

    execute_batches(
        &mut client,
        r#"CREATE VIEW graph_hits AS
           SELECT *
           FROM cugraph_bfs('target_edges', 0, 'src', 'dst', 'weight', '{"depth_limit":2}')"#,
    )
    .await;
    let schema = client
        .execute("SELECT vertex, distance FROM graph_hits LIMIT 0".to_string(), None)
        .await
        .unwrap()
        .try_decode_schema()
        .unwrap();

    server.abort();
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(0).name(), "vertex");
    assert_eq!(schema.field(1).name(), "distance");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cugraph_dsl_nullable_parquet_edge_schema_validates_over_flight_sql() {
    let addr = free_local_addr();
    let env = BTreeMap::from([(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let mut config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;
    let dir = unique_temp_dir("datafusion-nexus-server-cugraph-nullable-flight");
    let parquet_path = dir.join("edges.parquet");
    write_nullable_edge_parquet(&parquet_path);

    execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE nullable_edges STORED AS PARQUET LOCATION '{}'",
            parquet_path.to_string_lossy()
        ),
    )
    .await;

    let batches = execute_batches(
        &mut client,
        r#"SELECT valid, error_code, table_resolved, src_dtype, dst_dtype, weight_dtype, would_execute_gpu
           FROM cugraph_validate_call(
             'cugraph_bfs',
             'nullable_edges',
             '{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":"weight","options":{"depth_limit":2}}'
           )"#,
    )
    .await;
    let batch = single_batch(&batches);

    server.abort();
    let _ = fs::remove_dir_all(&dir);

    assert!(bool_value(batch, 0, 0));
    assert_eq!(string_value(batch, 1, 0), "ok");
    assert!(bool_value(batch, 2, 0));
    assert_eq!(string_value(batch, 3, 0), "Int64");
    assert_eq!(string_value(batch, 4, 0), "Int64");
    assert_eq!(string_value(batch, 5, 0), "Float64");
    assert!(bool_value(batch, 6, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cugraph_dsl_disallowed_algorithm_rejected_over_flight_sql() {
    let addr = free_local_addr();
    let env = BTreeMap::from([
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, "cugraph_pagerank".to_string()),
    ]);
    let mut config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("SELECT * FROM cugraph_louvain('g')".to_string(), None).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("not enabled in this session"));
}

fn free_local_addr() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_edge_parquet(path: &Path) {
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(Int64Array::from(vec![0_i64, 1, 0])) as ArrayRef,
            std::sync::Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(vec![1.0_f64, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_nullable_edge_parquet(path: &Path) {
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, true),
        Field::new("dst", DataType::Int64, true),
        Field::new("weight", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(Int64Array::from(vec![0_i64, 1, 0])) as ArrayRef,
            std::sync::Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(vec![1.0_f64, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn connect_with_retry(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    let mut last_error = None;
    for _ in 0..50 {
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(Duration::from_millis(50)).await;
            },
        }
    }
    panic!("Flight SQL server did not accept connections at {addr}: {last_error:?}");
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> Vec<arrow::record_batch::RecordBatch> {
    let flight_info = client.execute(sql.to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap()
}

fn single_batch(batches: &[arrow::record_batch::RecordBatch]) -> &arrow::record_batch::RecordBatch {
    assert_eq!(batches.len(), 1);
    &batches[0]
}

fn string_value(batch: &arrow::record_batch::RecordBatch, column: usize, row: usize) -> String {
    batch
        .column(column)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column must be StringArray")
        .value(row)
        .to_string()
}

fn bool_value(batch: &arrow::record_batch::RecordBatch, column: usize, row: usize) -> bool {
    batch.column(column).as_any().downcast_ref::<BooleanArray>().expect("column must be BooleanArray").value(row)
}
