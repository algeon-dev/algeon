#![cfg(all(feature = "server", feature = "cugraph"))]

use std::{
    collections::BTreeMap,
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion_nexus::{
    backend::GpuMemoryOwnership,
    server::{NEXUS_SERVER_CUGRAPH_ENABLED_ENV, Result as ServerResult, ServerConfig, serve_with_listener},
};
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use tonic::transport::Endpoint;

fn declare_test_gpu_ownership(config: &mut ServerConfig) {
    config.gpu_memory_ownership =
        config.gpu_devices.iter().copied().map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive)).collect();
}

fn test_server_config(mut env: BTreeMap<&'static str, String>) -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").expect("write server config");
    env.insert(datafusion_nexus::server::NEXUS_SERVER_CONFIG_FILE_ENV, path.to_string_lossy().into_owned());
    ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("explicit test device profile")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cugraph_dsl_metadata_and_dry_run_work_over_flight_sql_view() {
    let _guard = crate::root_gpu_fixture::lock_async().await;
    let (listener, addr) = free_local_addr();
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let mut config = test_server_config(env);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
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
        "SELECT function_name \
         FROM gpu_list_functions() \
         WHERE function_name IN ('cugraph_bfs', 'cugraph_pagerank') \
         ORDER BY function_name",
    )
    .await;
    let batch = single_batch(&batches);
    assert_eq!(string_value(batch, 0, 0), "cugraph_bfs");
    assert_eq!(string_value(batch, 0, 1), "cugraph_pagerank");

    let batches = execute_batches(
        &mut client,
        r#"SELECT valid, error_code, relations_resolved, details_json, would_execute_gpu
           FROM gpu_validate_call(
             'cugraph_bfs',
             '{"schema_version":1,"relations":{"edges":{"table":"target_edges"}},"options":{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":"weight","depth_limit":2}}'
           )"#,
    )
    .await;
    let batch = single_batch(&batches);
    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(bool_value(batch, 2, 0));
    let details: serde_json::Value = serde_json::from_str(&string_value(batch, 3, 0)).unwrap();
    assert_eq!(details["edges"]["src_dtype"], "Int64");
    assert_eq!(details["edges"]["dst_dtype"], "Int64");
    assert_eq!(details["edges"]["weight_dtype"], "Float64");
    assert!(bool_value(batch, 4, 0));

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
    let _guard = crate::root_gpu_fixture::lock_async().await;
    let (listener, addr) = free_local_addr();
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let mut config = test_server_config(env);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
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
        r#"SELECT valid, error_code, relations_resolved, details_json, would_execute_gpu
           FROM gpu_validate_call(
             'cugraph_bfs',
             '{"schema_version":1,"relations":{"edges":{"table":"nullable_edges"}},"options":{"source_vertex":0,"src_col":"src","dst_col":"dst","weight_col":"weight","depth_limit":2}}'
           )"#,
    )
    .await;
    let batch = single_batch(&batches);

    server.abort();
    let _ = fs::remove_dir_all(&dir);

    assert!(bool_value(batch, 0, 0));
    assert!(is_null_value(batch, 1, 0));
    assert!(bool_value(batch, 2, 0));
    let details: serde_json::Value = serde_json::from_str(&string_value(batch, 3, 0)).unwrap();
    assert_eq!(details["edges"]["src_dtype"], "Int64");
    assert_eq!(details["edges"]["dst_dtype"], "Int64");
    assert_eq!(details["edges"]["weight_dtype"], "Float64");
    assert!(bool_value(batch, 4, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cugraph_bfs_path_uses_gpu_resident_join_side_inputs_over_flight_sql() {
    let _guard = crate::root_gpu_fixture::lock_async().await;
    let (listener, addr) = free_local_addr();
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let mut config = test_server_config(env);
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let _server = ScopedServer::spawn(config, listener);
    let mut client = connect_with_retry(addr).await;
    let dir = ScopedTempDir::new("datafusion-nexus-server-cugraph-bfs-gpu-sides");
    let edges_path = dir.path().join("edges.parquet");
    let source_candidates_path = dir.path().join("source_candidates.parquet");
    let source_keep_path = dir.path().join("source_keep.parquet");
    let target_candidates_path = dir.path().join("target_candidates.parquet");
    let target_keep_path = dir.path().join("target_keep.parquet");
    write_disconnected_edge_parquet(&edges_path);
    write_vertex_key_parquet(&source_candidates_path, "vertex_id", "keep_id", &[(0, 1), (10, 2), (99, 9)]);
    write_key_parquet(&source_keep_path, "keep_id", &[1, 2]);
    write_vertex_key_parquet(&target_candidates_path, "vertex_id", "keep_id", &[(3, 7), (11, 8)]);
    write_key_parquet(&target_keep_path, "keep_id", &[7]);

    execute_batches(
        &mut client,
        &format!("CREATE EXTERNAL TABLE raw_edges STORED AS PARQUET LOCATION '{}'", edges_path.to_string_lossy()),
    )
    .await;
    execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE source_candidates STORED AS PARQUET LOCATION '{}'",
            source_candidates_path.to_string_lossy()
        ),
    )
    .await;
    execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE source_keep STORED AS PARQUET LOCATION '{}'",
            source_keep_path.to_string_lossy()
        ),
    )
    .await;
    execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE target_candidates STORED AS PARQUET LOCATION '{}'",
            target_candidates_path.to_string_lossy()
        ),
    )
    .await;
    execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE target_keep STORED AS PARQUET LOCATION '{}'",
            target_keep_path.to_string_lossy()
        ),
    )
    .await;
    execute_batches(&mut client, "CREATE VIEW graph_edges AS SELECT src, dst, weight FROM raw_edges WHERE src >= 0")
        .await;
    execute_batches(
        &mut client,
        "CREATE VIEW bfs_sources AS \
         SELECT c.vertex_id \
         FROM source_candidates c \
         JOIN source_keep k ON c.keep_id = k.keep_id",
    )
    .await;
    execute_batches(
        &mut client,
        "CREATE VIEW bfs_targets AS \
         SELECT c.vertex_id \
         FROM target_candidates c \
         JOIN target_keep k ON c.keep_id = k.keep_id",
    )
    .await;

    let bfs_sql = r#"SELECT path_index, source, target, vertex, distance
           FROM cugraph_bfs(
             'graph_edges',
             NULL,
             'src',
             'dst',
             'weight',
             '{"directed":true,
               "renumber":true,
               "construction_policy":"raw_libcugraph",
               "source_vertices_table":"bfs_sources",
               "source_vertex_col":"vertex_id",
               "output_mode":"path",
               "target_vertices_table":"bfs_targets",
               "target_vertex_col":"vertex_id"}'
           )
           ORDER BY path_index"#;
    let explain = execute_batches(&mut client, &format!("EXPLAIN {bfs_sql}")).await;
    let explain_text = string_values(&explain);
    assert!(
        explain_text.contains("CugraphAlgoExec"),
        "Flight SQL EXPLAIN should expose the cuGraph physical operator:\n{explain_text}",
    );
    assert!(
        explain_text.contains("predicate_source_count=2"),
        "Flight SQL EXPLAIN should show the source and target relation inputs:\n{explain_text}",
    );
    assert!(
        explain_text.matches("NexusNativeExec:").count() >= 3,
        "edge, source, and target relation inputs should all be native children before GPU-resident execution:\n{explain_text}",
    );
    assert!(
        !explain_text.contains("HashJoinExec"),
        "join-backed BFS relation inputs should not fall back to a DataFusion join child:\n{explain_text}",
    );

    let batches = execute_batches(&mut client, bfs_sql).await;

    let batch = single_batch(&batches);
    assert_eq!(int64_column(batch, 0), vec![0, 1, 2]);
    assert_eq!(int64_column(batch, 1), vec![0, 0, 0]);
    assert_eq!(int64_column(batch, 2), vec![3, 3, 3]);
    assert_eq!(int64_column(batch, 3), vec![0, 2, 3]);
    assert_eq!(int64_column(batch, 4), vec![0, 1, 2]);

    execute_batches(&mut client, "CREATE VIEW missing_bfs_sources AS SELECT CAST(12345 AS BIGINT) AS vertex_id").await;
    let empty_bfs_sql = r#"SELECT path_index, source, target, vertex, distance
           FROM cugraph_bfs(
             'graph_edges',
             NULL,
             'src',
             'dst',
             'weight',
             '{"directed":true,
               "renumber":true,
               "construction_policy":"raw_libcugraph",
               "source_vertices_table":"missing_bfs_sources",
               "source_vertex_col":"vertex_id",
               "output_mode":"path",
               "target_vertices_table":"bfs_targets",
               "target_vertex_col":"vertex_id"}'
           )
           ORDER BY path_index"#;
    let empty_flight_info = client.execute(empty_bfs_sql.to_string(), None).await.unwrap();
    let ticket = empty_flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let empty_schema = empty_flight_info.try_decode_schema().unwrap();
    let empty_batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(empty_batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 0);
    assert_eq!(empty_schema.field(0).name(), "path_index");
    assert_eq!(empty_schema.field(4).name(), "distance");

    execute_batches(
        &mut client,
        "CREATE VIEW partial_bfs_sources AS \
         SELECT CAST(0 AS BIGINT) AS vertex_id \
         UNION ALL \
         SELECT CAST(12345 AS BIGINT) AS vertex_id",
    )
    .await;
    let partial_error_sql = r#"SELECT path_index, source, target, vertex, distance
           FROM cugraph_bfs(
             'graph_edges',
             NULL,
             'src',
             'dst',
             'weight',
             '{"directed":true,
               "renumber":true,
               "construction_policy":"raw_libcugraph",
               "source_vertices_table":"partial_bfs_sources",
               "source_vertex_col":"vertex_id",
               "output_mode":"path",
               "target_vertices_table":"bfs_targets",
               "target_vertex_col":"vertex_id"}'
           )
           ORDER BY path_index"#;
    let partial_flight_info = client.execute(partial_error_sql.to_string(), None).await.unwrap();
    let ticket = partial_flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let error = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap_err().to_string();
    assert!(error.contains("partial_source_vertices_absent"), "unexpected Flight SQL error: {error}");
    assert!(!error.contains("Found invalid vertex"), "Flight SQL error should be graph-aware: {error}");

    execute_batches(&mut client, "CREATE VIEW empty_bfs_sources AS SELECT CAST(0 AS BIGINT) AS vertex_id WHERE 1 = 0")
        .await;
    execute_batches(&mut client, "CREATE VIEW bad_included_vertices AS SELECT CAST(12345 AS BIGINT) AS vertex_id")
        .await;
    let bad_predicate_sql = r#"SELECT path_index, source, target, vertex, distance
           FROM cugraph_bfs(
             'graph_edges',
             NULL,
             'src',
             'dst',
             'weight',
             '{"directed":true,
               "renumber":true,
               "construction_policy":"raw_libcugraph",
               "source_vertices_table":"empty_bfs_sources",
               "source_vertex_col":"vertex_id",
               "include_vertices_table":"bad_included_vertices",
               "include_vertex_col":"vertex_id",
               "output_mode":"path",
               "target_vertices_table":"bfs_targets",
               "target_vertex_col":"vertex_id"}'
           )
           ORDER BY path_index"#;
    let bad_predicate_flight_info = client.execute(bad_predicate_sql.to_string(), None).await.unwrap();
    let ticket = bad_predicate_flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let error = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap_err().to_string();
    assert!(error.contains("predicate_vertices_absent"), "unexpected Flight SQL error: {error}");
    assert!(error.contains("include_vertices"), "Flight SQL error should carry predicate_role: {error}");
    assert!(!error.contains("Found invalid vertex"), "Flight SQL error should be graph-aware: {error}");
}

/// Warm the process-wide CUDA context exactly once. The first CUDA call pays a
/// multi-second `cuInit()` cost; doing it here (every test reserves an address
/// before spawning a server) keeps that cost off each server's connect window,
/// and the `OnceLock` collapses concurrent callers onto a single init.
fn warm_cuda_once() {
    static CUDA_WARM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = datafusion_nexus::server::init_cuda_device(0);
    });
}

/// Reserve a loopback port and hand back the live listener so the caller can
/// pass it straight to `serve_with_listener`. Keeping the socket bound closes
/// the bind/serve gap where a parallel test could steal the port.
fn free_local_addr() -> (TcpListener, std::net::SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

struct ScopedServer {
    handle: tokio::task::JoinHandle<ServerResult<()>>,
}

impl ScopedServer {
    fn spawn(config: ServerConfig, listener: TcpListener) -> Self {
        Self { handle: tokio::spawn(async move { serve_with_listener(config, listener).await }) }
    }
}

impl Drop for ScopedServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        Self { path: unique_temp_dir(prefix) }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
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

fn write_disconnected_edge_parquet(path: &Path) {
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(Int64Array::from(vec![0_i64, 0, 2, 10, 99])) as ArrayRef,
            std::sync::Arc::new(Int64Array::from(vec![1_i64, 2, 3, 11, 3])) as ArrayRef,
            std::sync::Arc::new(Float64Array::from(vec![1.0_f64, 1.0, 1.0, 1.0, 1.0])) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_vertex_key_parquet(path: &Path, vertex_col: &str, key_col: &str, rows: &[(i64, i64)]) {
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new(vertex_col, DataType::Int64, false),
        Field::new(key_col, DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(Int64Array::from(rows.iter().map(|(vertex, _)| *vertex).collect::<Vec<_>>()))
                as ArrayRef,
            std::sync::Arc::new(Int64Array::from(rows.iter().map(|(_, key)| *key).collect::<Vec<_>>())) as ArrayRef,
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_key_parquet(path: &Path, key_col: &str, keys: &[i64]) {
    let schema = std::sync::Arc::new(Schema::new(vec![Field::new(key_col, DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![std::sync::Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef],
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
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                if std::time::Instant::now() >= deadline {
                    panic!("Flight SQL server did not accept connections at {addr} within 15s: {error:?}");
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            },
        }
    }
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

fn string_values(batches: &[arrow::record_batch::RecordBatch]) -> String {
    let mut values = Vec::new();
    for batch in batches {
        for column in batch.columns() {
            if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                values.extend((0..strings.len()).map(|row| strings.value(row).to_string()));
            } else if let Some(strings) = column.as_any().downcast_ref::<LargeStringArray>() {
                values.extend((0..strings.len()).map(|row| strings.value(row).to_string()));
            }
        }
    }
    values.join("\n")
}

fn bool_value(batch: &arrow::record_batch::RecordBatch, column: usize, row: usize) -> bool {
    batch.column(column).as_any().downcast_ref::<BooleanArray>().expect("column must be BooleanArray").value(row)
}

fn is_null_value(batch: &arrow::record_batch::RecordBatch, column: usize, row: usize) -> bool {
    batch.column(column).is_null(row)
}

fn int64_column(batch: &arrow::record_batch::RecordBatch, column: usize) -> Vec<i64> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().expect("column must be Int64Array");
    (0..array.len()).map(|row| array.value(row)).collect()
}
