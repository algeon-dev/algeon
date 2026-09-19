#[path = "server_gpu_fixture.rs"]
mod server_gpu_fixture;

use std::{
    fs::{self, OpenOptions},
    io::Write,
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::OnceLock,
    time::{Duration, Instant},
};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, REASON_METADATA_KEY, Result as ServerResult, ServerConfig, init_cuda_device,
        serve_with_listener,
    },
};
use arrow::{
    array::{Array, BooleanArray, StringArray},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use arrow_flight::{error::FlightError, sql::client::FlightSqlServiceClient};
use futures::TryStreamExt;
use server_gpu_fixture::lock_async;
use tokio::{task::JoinHandle, time::sleep};
use tonic::{
    Status,
    transport::{Channel, Endpoint},
};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_lists_and_executes_all_cuml_functions() {
    let _guard = lock_async().await;
    let (listener, addr) = free_local_addr();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.toml");
    std::fs::write(&path, "[[admission.device_profiles]]\ndevice_ordinal = 0\n").unwrap();
    let path = path.to_string_lossy().into_owned();
    let mut config =
        ServerConfig::from_env_get(|name| (name == ALGEON_SERVER_CONFIG_FILE_ENV).then(|| path.clone())).unwrap();
    config.gpu_memory_ownership = vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)];
    config.bind_addr = addr;
    let server = ScopedServer { handle: tokio::spawn(async move { serve_with_listener(config, listener).await }) };
    let mut client = connect_with_retry(addr).await;
    execute_batches(
        &mut client,
        "CREATE VIEW input_vectors AS \
        SELECT CAST(0 AS BIGINT) AS id, CAST(0.0 AS REAL) AS d0, CAST(0.0 AS REAL) AS d1, CAST(0.0 AS REAL) AS d2 \
        UNION ALL SELECT CAST(1 AS BIGINT), CAST(1.0 AS REAL), CAST(2.0 AS REAL), CAST(0.0 AS REAL) \
        UNION ALL SELECT CAST(2 AS BIGINT), CAST(2.0 AS REAL), CAST(4.0 AS REAL), CAST(0.0 AS REAL) \
        UNION ALL SELECT CAST(3 AS BIGINT), CAST(3.0 AS REAL), CAST(6.0 AS REAL), CAST(0.0 AS REAL)",
    )
    .await;
    let listed = execute_batches(
        &mut client,
        "SELECT function_name, available, unavailable_reason FROM gpu_list_functions() WHERE provider = 'cuml' ORDER BY function_name",
    )
    .await;
    let names = listed[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let available = listed[0].column(1).as_any().downcast_ref::<BooleanArray>().unwrap();
    let reasons = listed[0].column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(
        (0..names.len()).map(|row| names.value(row)).collect::<Vec<_>>(),
        [
            "cuml_forest_predict",
            "cuml_hdbscan",
            "cuml_logistic_regression",
            "cuml_random_forest_classifier",
            "cuml_random_forest_regressor",
            "cuml_tsvd",
            "cuml_umap"
        ]
    );
    // Without a [models] root the forest function stays listed but unavailable.
    assert!((1..names.len()).all(|row| available.value(row)));
    assert!(!available.value(0));
    assert_eq!(reasons.value(0), "model_root_unconfigured");
    for function in [
        "cuml_tsvd",
        "cuml_hdbscan",
        "cuml_umap",
        "cuml_logistic_regression",
        "cuml_random_forest_classifier",
        "cuml_random_forest_regressor",
    ] {
        let projection = match function {
            "cuml_logistic_regression" | "cuml_random_forest_classifier" | "cuml_random_forest_regressor" => {
                "id, prediction"
            },
            "cuml_hdbscan" => "id, cluster_id, probability",
            "cuml_umap" => "id, dim_0, dim_1",
            _ => "id, pc_0, pc_1",
        };
        let options = match function {
            "cuml_logistic_regression" | "cuml_random_forest_classifier" | "cuml_random_forest_regressor" => {
                "n_trees => 4"
            },
            "cuml_hdbscan" => "min_samples => 2, min_cluster_size => 2",
            "cuml_umap" => "n_neighbors => 2, n_components => 2, init => 'random'",
            _ => "n_components => 2",
        };
        let sql = if matches!(
            function,
            "cuml_logistic_regression" | "cuml_random_forest_classifier" | "cuml_random_forest_regressor"
        ) {
            let label_type = if function == "cuml_random_forest_regressor" { "REAL" } else { "INT" };
            let options = if function == "cuml_logistic_regression" { "" } else { ", n_trees => 4" };
            format!(
                "SELECT {projection} FROM {function}(training => (SELECT id, d0, d1, CAST(CASE WHEN id < 2 THEN 0 ELSE 1 END AS {label_type}) AS label FROM input_vectors), predict => (SELECT id, d0, d1 FROM input_vectors){options})"
            )
        } else {
            format!(
                "SELECT {projection} FROM {function}(input => (SELECT id, d0, d1, d2 FROM input_vectors), {options})"
            )
        };
        let batches = execute_batches(&mut client, &sql).await;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4, "{function}");
        let schema = batches[0].schema();
        assert_eq!(
            schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
            projection.split(", ").collect::<Vec<_>>()
        );
        assert_eq!(
            schema.field(1).data_type(),
            if function == "cuml_hdbscan" {
                &DataType::Int64
            } else if matches!(function, "cuml_logistic_regression" | "cuml_random_forest_classifier") {
                &DataType::Int32
            } else {
                &DataType::Float32
            }
        );
    }
    let status = execute_error_status(
        &mut client,
        "SELECT id, prediction FROM cuml_logistic_regression(\
         training => (SELECT id, d0, d1, CAST(0 AS INT) AS label FROM input_vectors WHERE id < 0), \
         predict => (SELECT id, d0, d1 FROM input_vectors))",
    )
    .await;
    for reason in ["training requires at least two rows", "reason=insufficient_training_rows", "training_rows=0"] {
        assert!(status.message().contains(reason), "{status:?}");
    }
    drop(client);
    drop(server);
}

const FOREST_SQL: &str = "SELECT id, prediction FROM cuml_forest_predict(input => (SELECT id, d0, d1, d2, d3 FROM forest_vectors), model => 'scorer.json', model_format => 'xgboost_json')";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_forest_predict_rejects_model_bytes_that_change_after_planning() {
    let _guard = lock_async().await;
    let (listener, addr) = free_local_addr();
    let models = tempfile::tempdir().unwrap();
    let model = models.path().join("scorer.json");
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../components/cuml/algeon-cuml/tests/fixtures/forest/xgb_binary_logistic.json"),
        &model,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.toml");
    fs::write(
        &path,
        format!("[models]\nroot = {:?}\n\n[[admission.device_profiles]]\ndevice_ordinal = 0\n", models.path()),
    )
    .unwrap();
    let path = path.to_string_lossy().into_owned();
    let mut config =
        ServerConfig::from_env_get(|name| (name == ALGEON_SERVER_CONFIG_FILE_ENV).then(|| path.clone())).unwrap();
    config.gpu_memory_ownership = vec![(0, GpuMemoryOwnership::WholeDeviceExclusive)];
    config.bind_addr = addr;
    let server = ScopedServer { handle: tokio::spawn(async move { serve_with_listener(config, listener).await }) };
    let mut client = connect_with_retry(addr).await;
    execute_batches(
        &mut client,
        "CREATE VIEW forest_vectors AS \
        SELECT CAST(0 AS BIGINT) AS id, CAST(0.0 AS REAL) AS d0, CAST(0.5 AS REAL) AS d1, CAST(1.0 AS REAL) AS d2, CAST(-1.0 AS REAL) AS d3 \
        UNION ALL SELECT CAST(1 AS BIGINT), CAST(2.0 AS REAL), CAST(-0.5 AS REAL), CAST(0.0 AS REAL), CAST(1.5 AS REAL)",
    )
    .await;
    let listed = execute_batches(
        &mut client,
        "SELECT available FROM gpu_list_functions() WHERE function_name = 'cuml_forest_predict'",
    )
    .await;
    assert!(listed[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap().value(0));

    let batches = execute_batches(&mut client, FOREST_SQL).await;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    assert_eq!(batches[0].schema().field(1).data_type(), &DataType::Float32);

    // A trailing newline keeps the model valid but changes its bytes, so only
    // the digest check can reject the replanned statement.
    let touch = || OpenOptions::new().append(true).open(&model).unwrap().write_all(b"\n").unwrap();
    let info = client.execute(FOREST_SQL.to_string(), None).await.unwrap();
    let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    touch();
    let error = match client.do_get(ticket).await {
        Ok(stream) => stream.try_collect::<Vec<_>>().await.unwrap_err(),
        Err(error) => error,
    };
    assert_eq!(flight_reason(&error), Some("model_changed"), "{error:?}");

    let mut prepared = client.prepare(FOREST_SQL.to_string(), None).await.unwrap();
    for _ in 0..2 {
        let info = prepared.execute().await.unwrap();
        let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
        let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    }
    touch();
    let error = prepared.execute().await.unwrap_err();
    assert_eq!(flight_reason(&error), Some("model_changed"), "{error:?}");
    drop(client);
    drop(server);
}

fn flight_reason(error: &FlightError) -> Option<&str> {
    let FlightError::Tonic(status) = error else {
        panic!("expected a tonic Flight error, got {error:?}");
    };
    status.metadata().get(REASON_METADATA_KEY).and_then(|value| value.to_str().ok())
}

fn free_local_addr() -> (TcpListener, SocketAddr) {
    static CUDA_WARM: OnceLock<()> = OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        let _ = init_cuda_device(0);
    });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

struct ScopedServer {
    handle: JoinHandle<ServerResult<()>>,
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
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(_) if Instant::now() < deadline => {
                sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            },
            Err(error) => panic!("Flight SQL server did not accept connections at {addr}: {error:?}"),
        }
    }
}

async fn execute_batches(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> Vec<RecordBatch> {
    let info = client.execute(sql.to_string(), None).await.expect("Flight SQL statement");
    let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("statement ticket");
    client.do_get(ticket).await.expect("Flight SQL stream").try_collect::<Vec<_>>().await.expect("Flight SQL batches")
}

async fn execute_error_status(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> Status {
    let error = match client.execute(sql.to_string(), None).await {
        Ok(info) => {
            let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("statement ticket");
            match client.do_get(ticket).await {
                Ok(stream) => stream.try_collect::<Vec<_>>().await.expect_err("statement must fail"),
                Err(error) => error,
            }
        },
        Err(error) => error,
    };
    let FlightError::Tonic(status) = error else {
        panic!("expected a tonic Flight error, got {error:?}");
    };
    *status
}
