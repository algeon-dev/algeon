#[path = "server_gpu_fixture.rs"]
mod server_gpu_fixture;

use std::{
    fs::File,
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::OnceLock,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    gpu_functions::GPU_LIST_FUNCTIONS_FUNCTION,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, Result as ServerResult, ServerConfig, init_cuda_device, serve_with_listener,
    },
};
use arrow::{
    array::{Array, ArrayRef, Float32Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use parquet::arrow::ArrowWriter;
use server_gpu_fixture::lock_async;
use tokio::{task::JoinHandle, time::sleep};
use tonic::transport::{Channel, Endpoint};

fn test_server_config() -> ServerConfig {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").expect("write server config");
    let path = path.to_string_lossy().into_owned();
    ServerConfig::from_env_get(|name| (name == ALGEON_SERVER_CONFIG_FILE_ENV).then(|| path.clone()))
        .expect("explicit test device profile")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_flight_sql_lists_and_executes_all_cuvs_operations() {
    let _guard = lock_async().await;
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    config.gpu_memory_ownership = vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)];
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
    let operations =
        metadata.column(0).as_any().downcast_ref::<StringArray>().expect("metadata function_name must be Utf8");
    assert_eq!(
        (0..operations.len()).map(|row| operations.value(row)).collect::<Vec<_>>(),
        vec![
            "cuvs_brute_force_knn",
            "cuvs_cagra",
            "cuvs_ivf_flat",
            "cuvs_ivf_pq",
            "cuvs_kmeans",
            "cuvs_pairwise_distance",
            "cuvs_pca",
            "cuvs_top_k"
        ],
    );

    let knn = execute_batches(
        &mut client,
        "SELECT query_id, neighbor_id, rank \
         FROM cuvs_brute_force_knn(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'l2_expanded') \
         ORDER BY query_ordinal, rank",
    )
    .await;
    assert_nonempty(&knn);

    let ivf = execute_batches(
        &mut client,
        "SELECT query_id, neighbor_id, rank FROM cuvs_ivf_flat(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'l2_expanded', n_lists => 2, n_probes => 2) \
         ORDER BY query_ordinal, rank",
    )
    .await;
    assert_nonempty(&ivf);

    let ivf_pq = execute_batches(
        &mut client,
        "SELECT query_id, neighbor_id, rank FROM cuvs_ivf_pq(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'l2_expanded', n_lists => 2, n_probes => 2, pq_dim => 2) \
         ORDER BY query_ordinal, rank",
    )
    .await;
    assert_nonempty(&ivf_pq);

    let pairwise = execute_batches(
        &mut client,
        "SELECT x_id, y_id, distance FROM cuvs_pairwise_distance(\
             x => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             y => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             metric => 'l2_expanded') ORDER BY x_ordinal, y_ordinal",
    )
    .await;
    assert_nonempty(&pairwise);

    let top_k = execute_batches(
        &mut client,
        "SELECT id, rank, position, value FROM cuvs_top_k(\
             input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             k => 1, select => 'max') ORDER BY row_ordinal, rank",
    )
    .await;
    assert_nonempty(&top_k);

    let kmeans = execute_batches(
        &mut client,
        "SELECT id, cluster_id \
         FROM cuvs_kmeans(\
             input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             n_clusters => 2, init => 'kmeans++') \
         ORDER BY row_ordinal",
    )
    .await;
    assert_nonempty(&kmeans);

    let pca = execute_batches(
        &mut client,
        "SELECT id, pc_0 \
         FROM cuvs_pca(\
             input => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             n_components => 1) \
         ORDER BY row_ordinal",
    )
    .await;
    assert_nonempty(&pca);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_flight_sql_executes_parquet_cuvs_relations() {
    let _guard = lock_async().await;
    let temp_dir = tempfile::tempdir().expect("temporary Parquet directory");
    let parquet_path = temp_dir.path().join("vector_input.parquet");
    write_vector_parquet(&parquet_path);

    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    config.gpu_memory_ownership = vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)];
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
        "SELECT id, cluster_id \
         FROM cuvs_kmeans(\
             input => (SELECT item_id AS id, d0, d1 FROM parquet_vector_input), \
             n_clusters => 2, init => 'kmeans++') \
         ORDER BY row_ordinal",
    )
    .await;
    assert_nonempty(&kmeans);

    let pca = execute_batches(
        &mut client,
        "SELECT id, pc_0 \
         FROM cuvs_pca(\
             input => (SELECT item_id AS id, d0, d1 FROM parquet_vector_input), \
             n_components => 1) \
         ORDER BY row_ordinal",
    )
    .await;
    assert_nonempty(&pca);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn default_bounded_flight_sql_executes_cuvs() {
    let _guard = lock_async().await;
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    config.gpu_memory_ownership = vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)];
    config.bind_addr = addr;

    let _server = ScopedServer::spawn(config, listener);
    let mut client = connect_with_retry(addr).await;
    register_vector_views(&mut client).await;
    let batches = execute_batches(
        &mut client,
        "SELECT * FROM cuvs_brute_force_knn(\
             dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), \
             queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), \
             k => 1, metric => 'l2_expanded')",
    )
    .await;
    assert_nonempty(&batches);
}

async fn register_vector_views(client: &mut FlightSqlServiceClient<Channel>) {
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
        let _ = init_cuda_device(0);
    });
}

fn free_local_addr() -> (TcpListener, SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
    let addr = listener.local_addr().expect("test listener address");
    (listener, addr)
}

struct ScopedServer {
    handle: JoinHandle<ServerResult<()>>,
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

async fn connect_with_retry(addr: SocketAddr) -> FlightSqlServiceClient<Channel> {
    let uri = format!("http://{addr}");
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match Endpoint::from_shared(uri.clone()).expect("Flight endpoint").connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) if Instant::now() < deadline => {
                sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
                let _ = error;
            },
            Err(error) => panic!("Flight SQL server did not accept connections at {addr}: {error:?}"),
        }
    }
}

async fn execute_batches(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> Vec<RecordBatch> {
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

fn assert_nonempty(batches: &[RecordBatch]) {
    assert!(batches.iter().map(RecordBatch::num_rows).sum::<usize>() > 0, "cuVS execution must return rows");
}
