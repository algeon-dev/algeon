use std::{
    env,
    error::Error,
    fs::{self, File},
    io,
    path::Path,
    process,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "cugraph")]
use algeon_datafusion::cugraph_sql::CugraphSqlConfig;
#[cfg(any(feature = "cuvs", feature = "cugraph", feature = "cuml"))]
use algeon_datafusion::session::AlgeonSessionStateBuilderExt;
use algeon_datafusion::{
    attempt::{AttemptDisposition, AttemptOptions},
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    planner::{AlgeonGpuExecutionMode, FinalPlanDisposition, classify_final_plan},
};
#[cfg(any(feature = "cuvs", feature = "cuml"))]
use datafusion::arrow::array::Float32Array;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::SessionStateBuilder,
    parquet::arrow::ArrowWriter,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::TryStreamExt;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[tokio::main]
async fn main() -> Result<()> {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .devices([device])
        .build()?;

    let result = run(&backend).await;
    let close = backend.close_and_wait_async(Duration::from_secs(30)).await;
    result?;
    let close = close?;
    require(close.gpu_resource_quiescent && close.observation_quiescent, "backend cleanup did not complete")?;
    println!("registry consumer passed");
    Ok(())
}

async fn run(backend: &AlgeonGpuBackend) -> Result<()> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = env::temp_dir().join(format!("algeon-registry-consumer-{}-{nonce}", process::id()));
    fs::create_dir(&directory)?;
    let result = run_session(backend, &directory).await;
    let cleanup = fs::remove_dir_all(&directory);
    result?;
    cleanup?;
    Ok(())
}

async fn run_session(backend: &AlgeonGpuBackend, directory: &Path) -> Result<()> {
    let builder = backend.install_on(
        SessionStateBuilder::new_with_default_features().with_config(SessionConfig::new().with_target_partitions(1)),
    )?;
    #[cfg(feature = "cuvs")]
    let builder = builder.try_with_cuvs_sql()?;
    #[cfg(feature = "cugraph")]
    let builder = builder.try_with_cugraph_sql(CugraphSqlConfig::default())?;
    #[cfg(feature = "cuml")]
    let builder = builder.try_with_cuml_sql()?;
    let ctx = SessionContext::new_with_state(builder.build());

    let orders = directory.join("orders.parquet");
    write_parquet(
        &orders,
        vec![Field::new("amount", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![10, 20, 30, 40]))],
    )?;
    register_parquet(&ctx, "orders", &orders).await?;
    check_relational(backend, &ctx).await?;

    #[cfg(any(feature = "cuvs", feature = "cuml"))]
    {
        let vectors = directory.join("vectors.parquet");
        let x = (0..8).map(|i| i as f32 - 3.0).collect::<Vec<_>>();
        write_parquet(
            &vectors,
            vec![
                Field::new("id", DataType::Int64, false),
                Field::new("d0", DataType::Float32, false),
                Field::new("d1", DataType::Float32, false),
                Field::new("d2", DataType::Float32, false),
            ],
            vec![
                Arc::new(Int64Array::from_iter_values(0..8)),
                Arc::new(Float32Array::from(x.clone())),
                Arc::new(Float32Array::from(x.iter().map(|v| 2.0 * v).collect::<Vec<_>>())),
                Arc::new(Float32Array::from(x.iter().map(|v| -v).collect::<Vec<_>>())),
            ],
        )?;
        register_parquet(&ctx, "vectors", &vectors).await?;
    }

    #[cfg(feature = "cuvs")]
    check_cuvs(backend, &ctx).await?;
    #[cfg(feature = "cugraph")]
    check_cugraph(backend, &ctx, directory).await?;
    #[cfg(feature = "cuml")]
    check_cuml(backend, &ctx).await?;
    #[cfg(feature = "server")]
    check_server(backend, &orders).await?;
    #[cfg(feature = "nvml")]
    check_nvml(backend)?;
    Ok(())
}

async fn execute_native(backend: &AlgeonGpuBackend, ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    require(classify_final_plan(plan.as_ref()) == FinalPlanDisposition::Native, "query plan contains CPU execution")?;

    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    let completion = attempt.completion();
    let result = async { attempt.sql(ctx, sql).await?.try_collect::<Vec<_>>().await }.await;
    let report = completion.await;
    let batches = result?;
    require(
        report.disposition == AttemptDisposition::Native && report.admission.is_some(),
        "query did not execute with native GPU admission",
    )?;
    println!("native query passed: {sql}");
    Ok(batches)
}

async fn check_relational(backend: &AlgeonGpuBackend, ctx: &SessionContext) -> Result<()> {
    let batches = execute_native(backend, ctx, "SELECT SUM(amount) AS total FROM orders WHERE amount >= 20").await?;
    let totals = int64_values(&batches, 0)?;
    require(totals == vec![90], "relational SUM result differs from 90")
}

#[cfg(feature = "cuvs")]
async fn check_cuvs(backend: &AlgeonGpuBackend, ctx: &SessionContext) -> Result<()> {
    let batches = execute_native(
        backend,
        ctx,
        "SELECT query_id, neighbor_id, distance FROM cuvs_brute_force_knn(\
         dataset => (SELECT id, d0, d1, d2 FROM vectors), \
         queries => (SELECT id, d0, d1, d2 FROM vectors WHERE id = 7), \
         k => 1, metric => 'l2_expanded')",
    )
    .await?;
    require(
        int64_values(&batches, 0)? == vec![7] && int64_values(&batches, 1)? == vec![7],
        "cuVS nearest neighbor differs from id 7",
    )?;
    let distances = float32_values(&batches, 2)?;
    require(distances.len() == 1 && distances[0].abs() < 1e-5, "cuVS self distance differs from zero")
}

#[cfg(feature = "cugraph")]
async fn check_cugraph(backend: &AlgeonGpuBackend, ctx: &SessionContext, directory: &Path) -> Result<()> {
    let edges = directory.join("edges.parquet");
    write_parquet(
        &edges,
        vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![0, 1, 2])), Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )?;
    register_parquet(ctx, "edges", &edges).await?;
    let batches = execute_native(
        backend,
        ctx,
        "SELECT vertex, distance FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => 0)",
    )
    .await?;
    let mut rows = int64_values(&batches, 0)?.into_iter().zip(int64_values(&batches, 1)?).collect::<Vec<_>>();
    rows.sort_unstable();
    require(rows == vec![(0, 0), (1, 1), (2, 2), (3, 3)], "cuGraph BFS distances differ from the path graph")
}

#[cfg(feature = "cuml")]
async fn check_cuml(backend: &AlgeonGpuBackend, ctx: &SessionContext) -> Result<()> {
    let batches = execute_native(
        backend,
        ctx,
        "SELECT id, pc_0, pc_1 FROM cuml_tsvd(input => (SELECT id, d0, d1, d2 FROM vectors), n_components => 2)",
    )
    .await?;
    let ids = int64_values(&batches, 0)?;
    let scores = float32_values(&batches, 1)?;
    let second = float32_values(&batches, 2)?;
    let mut seen = ids.clone();
    seen.sort_unstable();
    require(seen == (0..8).collect::<Vec<_>>(), "cuML TSVD did not retain all row ids")?;
    for ((id, score), residual) in ids.into_iter().zip(scores).zip(second) {
        let expected = (id as f32 - 3.0).abs() * 6.0_f32.sqrt();
        require(
            (score.abs() - expected).abs() < 1e-2 && residual.abs() < 1e-2,
            "cuML TSVD differs from the rank-one matrix oracle",
        )?;
    }
    Ok(())
}

#[cfg(feature = "server")]
async fn check_server(backend: &AlgeonGpuBackend, orders: &Path) -> Result<()> {
    use std::iter;

    use algeon_datafusion::server::{ServerConfig, build_session_context_with_backend};

    let config = ServerConfig::from_vars(iter::empty::<(String, String)>())?;
    let ctx = build_session_context_with_backend(&config, backend).await?;
    register_parquet(&ctx, "orders", orders).await?;
    check_relational(backend, &ctx).await?;
    println!("server session initialization passed");
    Ok(())
}

#[cfg(feature = "nvml")]
fn check_nvml(backend: &AlgeonGpuBackend) -> Result<()> {
    use algeon_datafusion::observability::DeviceTelemetryStatus;

    let snapshots = backend.device_telemetry_snapshots()?;
    require(snapshots.len() == 1 && snapshots[0].device_ordinal == 0, "telemetry did not sample CUDA device 0")?;
    let snapshot = &snapshots[0];
    require(snapshot.cuda_status == DeviceTelemetryStatus::Available, "CUDA telemetry is unavailable")?;
    require(snapshot.nvml_status != DeviceTelemetryStatus::FeatureDisabled, "NVML feature was not forwarded")?;
    // NVML availability is optional, independently of CUDA execution support.
    println!("nvml initialization status: {}", snapshot.nvml_status.as_str());
    Ok(())
}

fn write_parquet(path: &Path, fields: Vec<Field>, columns: Vec<ArrayRef>) -> Result<()> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

async fn register_parquet(ctx: &SessionContext, name: &str, path: &Path) -> Result<()> {
    let path = path.to_str().ok_or_else(|| io::Error::other("Parquet path is not UTF-8"))?;
    ctx.register_parquet(name, path, ParquetReadOptions::default()).await?;
    Ok(())
}

fn int64_values(batches: &[RecordBatch], column: usize) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| io::Error::other("expected non-null Int64 result"))?;
        require(array.null_count() == 0, "query returned a null Int64 result")?;
        values.extend(array.values().iter().copied());
    }
    Ok(values)
}

#[cfg(any(feature = "cuvs", feature = "cuml"))]
fn float32_values(batches: &[RecordBatch], column: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(column)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| io::Error::other("expected non-null Float32 result"))?;
        require(array.null_count() == 0, "query returned a null Float32 result")?;
        values.extend(array.values().iter().copied());
    }
    Ok(values)
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition { Ok(()) } else { Err(io::Error::other(message).into()) }
}
