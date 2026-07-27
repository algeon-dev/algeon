#![cfg(all(feature = "server", feature = "cuvs"))]

use std::{collections::BTreeMap, fs::File, net::TcpListener, path::Path, sync::OnceLock, time::Duration};

use arrow::{
    array::{Array, ArrayRef, Float32Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{error::FlightError, sql::client::FlightSqlServiceClient};
use datafusion_nexus::{
    gpu_functions::GPU_LIST_FUNCTIONS_FUNCTION,
    server::{
        ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
        NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, Result as ServerResult, ServerConfig, serve_with_listener,
    },
};
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use tonic::{Code, transport::Endpoint};

const KNN_OPTIONS: &str = r#"{
  "dataset":{"id":"item_id","vector":{"columns":["d0","d1"]}},
  "queries":{"id":"query_id","vector":{"columns":["d0","d1"]}},
  "k":1,
  "metric":"l2_expanded"
}"#;

const KMEANS_OPTIONS: &str = r#"{
  "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
  "n_clusters":2,
  "init":"kmeans++"
}"#;

const PCA_OPTIONS: &str = r#"{
  "input":{"id":"item_id","vector":{"columns":["d0","d1"]}},
  "n_components":1
}"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicitly_unbounded_flight_sql_executes_all_cuvs_operations_over_relation_subqueries() {
    let _guard = server_test_lock().lock().await;
    let (listener, addr) = free_local_addr();
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string())]);
    let mut config = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("unbounded server config");
    config.bind_addr = addr;

    let _server = ScopedServer::spawn(config, listener);
    let mut client = connect_with_retry(addr).await;
    register_vector_views(&mut client).await;

    let metadata = execute_batches(
        &mut client,
        &format!(
            "SELECT function_name FROM {GPU_LIST_FUNCTIONS_FUNCTION}() WHERE provider = 'cuvs' ORDER BY function_name"
        ),
    )
    .await;
    let metadata = single_batch(&metadata);
    let operations = metadata
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("metadata function_name must be Utf8");
    assert_eq!(
        (0..operations.len()).map(|row| operations.value(row)).collect::<Vec<_>>(),
        vec!["cuvs_brute_force_knn", "cuvs_kmeans", "cuvs_pca"],
    );

    let knn = execute_batches(
        &mut client,
        &format!(
            "SELECT query_id, neighbor_id, rank \
             FROM cuvs_brute_force_knn((SELECT * FROM dataset_vectors), (SELECT * FROM query_vectors), '{}') \
             ORDER BY query_ordinal, rank",
            sql_literal(KNN_OPTIONS)
        ),
    )
    .await;
    let knn = single_batch(&knn);
    assert_eq!(int64_values(knn, 0), vec![99]);
    assert_eq!(int64_values(knn, 1), vec![0]);
    assert_eq!(u32_values(knn, 2), vec![0]);

    let kmeans = execute_batches(
        &mut client,
        &format!(
            "SELECT id, cluster_id \
             FROM cuvs_kmeans((SELECT * FROM dataset_vectors), '{}') \
             ORDER BY row_ordinal",
            sql_literal(KMEANS_OPTIONS)
        ),
    )
    .await;
    let kmeans = single_batch(&kmeans);
    assert_eq!(int64_values(kmeans, 0), vec![0, 1, 2, 3]);
    let clusters = int32_values(kmeans, 1);
    assert_eq!(clusters.len(), 4);
    assert_eq!(clusters[0], clusters[1], "the first two points form one cluster");
    assert_eq!(clusters[2], clusters[3], "the last two points form one cluster");
    assert_ne!(clusters[0], clusters[2], "the two well-separated groups must have distinct labels");

    let pca = execute_batches(
        &mut client,
        &format!(
            "SELECT id, pc_0 \
             FROM cuvs_pca((SELECT * FROM dataset_vectors), '{}') \
             ORDER BY row_ordinal",
            sql_literal(PCA_OPTIONS)
        ),
    )
    .await;
    let pca = single_batch(&pca);
    assert_eq!(pca.schema().field(0).name(), "id");
    assert_eq!(pca.schema().field(1).name(), "pc_0");
    assert_eq!(int64_values(pca, 0), vec![0, 1, 2, 3]);
    let components = pca.column(1).as_any().downcast_ref::<arrow::array::Float32Array>().expect("pc_0 must be Float32");
    assert!((0..components.len()).all(|row| components.value(row).is_finite()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicitly_unbounded_flight_sql_executes_kmeans_and_pca_over_parquet_relation_subqueries() {
    let _guard = server_test_lock().lock().await;
    let temp_dir = tempfile::tempdir().expect("temporary Parquet directory");
    let parquet_path = temp_dir.path().join("vector_input.parquet");
    write_vector_parquet(&parquet_path);

    let (listener, addr) = free_local_addr();
    let env = BTreeMap::from([(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string())]);
    let mut config = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("unbounded server config");
    config.bind_addr = addr;

    let _server = ScopedServer::spawn(config, listener);
    let mut client = connect_with_retry(addr).await;
    execute_batches(
        &mut client,
        &format!(
            "CREATE EXTERNAL TABLE parquet_vector_input STORED AS PARQUET LOCATION '{}'",
            parquet_path.to_string_lossy()
        ),
    )
    .await;

    let kmeans = execute_batches(
        &mut client,
        &format!(
            "SELECT id, cluster_id \
             FROM cuvs_kmeans((SELECT item_id, d0, d1 FROM parquet_vector_input), '{}') \
             ORDER BY row_ordinal",
            sql_literal(KMEANS_OPTIONS)
        ),
    )
    .await;
    let kmeans = single_batch(&kmeans);
    assert_eq!(int64_values(kmeans, 0), vec![1, 2, 3, 4]);
    let clusters = int32_values(kmeans, 1);
    assert_eq!(clusters[0], clusters[1], "nearby low-valued rows must share a cluster");
    assert_eq!(clusters[2], clusters[3], "nearby high-valued rows must share a cluster");
    assert_ne!(clusters[0], clusters[2], "the separated groups must have distinct clusters");

    let pca = execute_batches(
        &mut client,
        &format!(
            "SELECT id, pc_0 \
             FROM cuvs_pca((SELECT item_id, d0, d1 FROM parquet_vector_input), '{}') \
             ORDER BY row_ordinal",
            sql_literal(PCA_OPTIONS)
        ),
    )
    .await;
    let pca = single_batch(&pca);
    assert_eq!(int64_values(pca, 0), vec![1, 2, 3, 4]);
    let components = pca.column(1).as_any().downcast_ref::<Float32Array>().expect("pc_0 must be Float32");
    assert!((0..components.len()).all(|row| components.value(row).is_finite()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn default_bounded_flight_sql_rejects_cuvs_before_child_execution_with_structured_status() {
    let _guard = server_test_lock().lock().await;
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).expect("default bounded server config");
    config.bind_addr = addr;

    let _server = ScopedServer::spawn(config, listener);
    let mut client = connect_with_retry(addr).await;
    register_vector_views(&mut client).await;
    let info = client
        .execute(
            format!(
                "SELECT * FROM cuvs_brute_force_knn((SELECT * FROM dataset_vectors), (SELECT * FROM query_vectors), '{}')",
                sql_literal(KNN_OPTIONS)
            ),
            None,
        )
        .await
        .expect("bounded execution is rejected when Flight consumes the statement ticket");
    let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("statement ticket");
    let error = client
        .do_get(ticket)
        .await
        .expect("open bounded Flight result stream")
        .try_collect::<Vec<_>>()
        .await
        .expect_err("bounded cuVS execution must be rejected");
    let status = flight_status(error);

    assert_eq!(status.code(), Code::Unimplemented);
    assert_metadata(&status, "unsupported_operator", "unsupported", "permanent");
    assert!(
        status.message().contains("bounded cuVS execution requires a lower-level peak-memory preflight contract"),
        "unexpected bounded cuVS message: {}",
        status.message()
    );
}

async fn register_vector_views(client: &mut FlightSqlServiceClient<tonic::transport::Channel>) {
    execute_batches(
        client,
        "CREATE VIEW dataset_vectors AS \
         SELECT CAST(0 AS BIGINT) AS item_id, CAST(0.0 AS REAL) AS d0, CAST(0.0 AS REAL) AS d1 \
         UNION ALL SELECT CAST(1 AS BIGINT), CAST(0.2 AS REAL), CAST(0.0 AS REAL) \
         UNION ALL SELECT CAST(2 AS BIGINT), CAST(10.0 AS REAL), CAST(10.0 AS REAL) \
         UNION ALL SELECT CAST(3 AS BIGINT), CAST(10.0 AS REAL), CAST(10.2 AS REAL)",
    )
    .await;
    execute_batches(
        client,
        "CREATE VIEW query_vectors AS \
         SELECT CAST(99 AS BIGINT) AS query_id, CAST(0.0 AS REAL) AS d0, CAST(0.0 AS REAL) AS d1",
    )
    .await;
}

fn write_vector_parquet(path: &Path) {
    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("item_id", DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let batch = RecordBatch::try_new(
        std::sync::Arc::clone(&schema),
        vec![
            std::sync::Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4])) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(vec![0.0_f32, 0.2, 10.0, 10.2])) as ArrayRef,
            std::sync::Arc::new(Float32Array::from(vec![0.0_f32, 0.1, 10.0, 10.1])) as ArrayRef,
        ],
    )
    .expect("Parquet vector fixture");
    let file = File::create(path).expect("create Parquet vector fixture");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("create Parquet writer");
    writer.write(&batch).expect("write Parquet vector fixture");
    writer.close().expect("finish Parquet vector fixture");
}

fn warm_cuda_once() {
    static CUDA_WARM: OnceLock<()> = OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        let _ = datafusion_nexus::server::init_cuda_device(0);
    });
}

fn server_test_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

fn free_local_addr() -> (TcpListener, std::net::SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
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

async fn connect_with_retry(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match Endpoint::from_shared(uri.clone()).expect("Flight endpoint").connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) if std::time::Instant::now() < deadline => {
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
                let _ = error;
            },
            Err(error) => panic!("Flight SQL server did not accept connections at {addr}: {error:?}"),
        }
    }
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> Vec<RecordBatch> {
    let info = client.execute(sql.to_string(), None).await.expect("Flight SQL statement");
    let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("statement ticket");
    client
        .do_get(ticket)
        .await
        .expect("Flight SQL result stream")
        .try_collect::<Vec<_>>()
        .await
        .expect("Flight SQL batches")
}

fn single_batch(batches: &[RecordBatch]) -> &RecordBatch {
    assert_eq!(batches.len(), 1, "expected exactly one result batch");
    &batches[0]
}

fn int64_values(batch: &RecordBatch, column: usize) -> Vec<i64> {
    let values = batch.column(column).as_any().downcast_ref::<Int64Array>().expect("Int64 result column");
    (0..values.len()).map(|row| values.value(row)).collect()
}

fn int32_values(batch: &RecordBatch, column: usize) -> Vec<i32> {
    let values = batch.column(column).as_any().downcast_ref::<Int32Array>().expect("Int32 result column");
    (0..values.len()).map(|row| values.value(row)).collect()
}

fn u32_values(batch: &RecordBatch, column: usize) -> Vec<u32> {
    let values =
        batch.column(column).as_any().downcast_ref::<arrow::array::UInt32Array>().expect("UInt32 result column");
    (0..values.len()).map(|row| values.value(row)).collect()
}

fn flight_status(error: FlightError) -> tonic::Status {
    match error {
        FlightError::Tonic(status) => *status,
        other => panic!("expected tonic Flight error, got {other:?}"),
    }
}

fn assert_metadata(status: &tonic::Status, code: &str, kind: &str, error_status: &str) {
    assert_eq!(
        status.metadata().get(ERROR_CODE_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(code),
        "error code metadata"
    );
    assert_eq!(
        status.metadata().get(ERROR_KIND_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(kind),
        "error kind metadata"
    );
    assert_eq!(
        status.metadata().get(ERROR_STATUS_METADATA_KEY).and_then(|value| value.to_str().ok()),
        Some(error_status),
        "error status metadata"
    );
}

fn sql_literal(json: &str) -> String {
    json.replace('\'', "''")
}
