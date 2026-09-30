//! Explicit E2E for one S3 configuration serving DataFusion and native reads.
//!
//! The runner needs an S3-compatible service and an existing bucket, so it
//! stays out of Cargo test targets. It uploads a generated Parquet object
//! through the session's `s3://` object store and runs one aggregate three
//! ways: a native-required and a functions-only session from a library backend
//! built with the S3 storage, and a native-required session built from server
//! configuration. Every run must return the generated groups, and each final
//! plan must be the disposition its mode requires. Credentials come from the
//! default AWS chain.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    native_exec::{AlgeonNativeExecutionConfig, S3SourceStorageProfile},
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig, FinalPlanDisposition, classify_final_plan},
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, ServerConfig, backend_from_server_config, build_session_context_with_backend,
    },
};
use datafusion::{
    arrow::{
        array::{Array, Int64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl},
    object_store::{ObjectStoreExt, path::Path as ObjectStorePath},
    parquet::arrow::ArrowWriter,
    physical_plan::collect,
    prelude::{ParquetReadOptions, SessionContext},
};

type E2eResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const ENDPOINT_ENV: &str = "ALGEON_S3_PARQUET_E2E_ENDPOINT";
const BUCKET_ENV: &str = "ALGEON_S3_PARQUET_E2E_BUCKET";
const REGION_ENV: &str = "ALGEON_S3_PARQUET_E2E_REGION";
const ROWS: i64 = 10_000;
const GROUPS: i64 = 7;
const QUERY: &str = "SELECT grp, COUNT(*) AS n, SUM(val) AS total FROM t GROUP BY grp ORDER BY grp";
const CLOSE_TIMEOUT: Duration = Duration::from_secs(60);

pub async fn run(args: Vec<String>) -> E2eResult<()> {
    if let Some(arg) = args.first() {
        if arg == "-h" || arg == "--help" {
            print_usage();
            return Ok(());
        }
        return Err(format!("unknown s3-parquet-e2e argument: {arg}").into());
    }
    let endpoint = required_env(ENDPOINT_ENV)?;
    let bucket = required_env(BUCKET_ENV)?;
    let region = env::var(REGION_ENV).ok().filter(|value| !value.is_empty()).unwrap_or_else(|| "us-east-1".into());
    let storage = S3SourceStorageProfile::default_chain()
        .with_endpoint(endpoint.clone())
        .with_region(region.clone())
        .with_path_style_access(true);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let key = format!("algeon-s3-parquet-e2e/{}-{nanos}.parquet", std::process::id());
    let location = format!("s3://{bucket}/{key}");

    let backend = library_backend(&storage, AlgeonGpuExecutionMode::NativeRequired)?;
    let ctx = library_session(&backend)?;
    upload(&ctx, &bucket, &key).await?;
    check(&ctx, &location, FinalPlanDisposition::Native).await?;
    close(ctx, backend)?;
    println!("library native_required ok");

    let backend = library_backend(&storage, AlgeonGpuExecutionMode::FunctionsOnly)?;
    let ctx = library_session(&backend)?;
    check(&ctx, &location, FinalPlanDisposition::DataFusion).await?;
    close(ctx, backend)?;
    println!("library functions_only ok");

    let config_dir = env::temp_dir().join(format!("algeon-s3-parquet-e2e-{}-{nanos}", std::process::id()));
    let config = server_config(&config_dir, &endpoint, &region);
    let result = async {
        let config = config?;
        let backend = backend_from_server_config(&config)?;
        let ctx = build_session_context_with_backend(&config, &backend).await?;
        check(&ctx, &location, FinalPlanDisposition::Native).await?;
        close(ctx, backend)
    }
    .await;
    fs::remove_dir_all(&config_dir)?;
    result?;
    println!("server native_required ok");
    println!("s3_parquet_e2e ok");
    Ok(())
}

fn print_usage() {
    println!(
        "Usage: s3-parquet-e2e\n\n\
         Reads {ENDPOINT_ENV} (an S3-compatible endpoint), {BUCKET_ENV} (an existing bucket), and the optional\n\
         {REGION_ENV} (default us-east-1). Credentials come from the default AWS chain, for example\n\
         AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
    );
}

fn required_env(name: &str) -> E2eResult<String> {
    env::var(name).ok().filter(|value| !value.is_empty()).ok_or_else(|| format!("{name} must be set").into())
}

fn device() -> AlgeonGpuDeviceConfig {
    AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive)
}

fn library_backend(storage: &S3SourceStorageProfile, mode: AlgeonGpuExecutionMode) -> E2eResult<AlgeonGpuBackend> {
    let optimizer_config = AlgeonNativeOptimizerConfig::default()
        .with_execution_mode(mode)
        .with_native_execution_config(AlgeonNativeExecutionConfig::default().with_s3_storage(storage.clone()));
    Ok(AlgeonGpuBackend::builder().with_optimizer_config(optimizer_config).devices([device()]).build()?)
}

fn library_session(backend: &AlgeonGpuBackend) -> E2eResult<SessionContext> {
    let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
    Ok(SessionContext::new_with_state(state))
}

fn server_config(dir: &std::path::Path, endpoint: &str, region: &str) -> E2eResult<ServerConfig> {
    fs::create_dir_all(dir)?;
    let path = dir.join("server.toml");
    fs::write(
        &path,
        format!(
            "[native]\nexecution_mode = \"native_required\"\n\n\
             [s3]\nendpoint = \"{endpoint}\"\nregion = \"{region}\"\npath_style = true\n\n\
             [[admission.device_profiles]]\ndevice_ordinal = 0\n"
        ),
    )?;
    let values = BTreeMap::from([(ALGEON_SERVER_CONFIG_FILE_ENV, path.to_string_lossy().into_owned())]);
    let config = ServerConfig::from_env_get(|name| values.get(name).cloned())?;
    Ok(config.with_gpu_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)]))
}

fn close(ctx: SessionContext, backend: AlgeonGpuBackend) -> E2eResult<()> {
    // The next backend in this process needs the owner lease this one holds.
    drop(ctx);
    backend.close_and_wait(CLOSE_TIMEOUT)?;
    Ok(())
}

fn sample_batch() -> E2eResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("grp", DataType::Int64, false),
        Field::new("val", DataType::Int64, false),
    ]));
    let ids = (0..ROWS).collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Int64Array::from(ids.iter().map(|id| id % GROUPS).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(ids.iter().map(|id| id * 2).collect::<Vec<_>>())),
        ],
    )?)
}

/// Writes the generated object through the session's object store for the
/// bucket, which the installed S3 storage resolves.
async fn upload(ctx: &SessionContext, bucket: &str, key: &str) -> E2eResult<()> {
    let batch = sample_batch()?;
    let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), None)?;
    writer.write(&batch)?;
    let bytes = writer.into_inner()?;
    let store = ctx.runtime_env().object_store(ObjectStoreUrl::parse(format!("s3://{bucket}"))?)?;
    store.put(&ObjectStorePath::from(key), bytes.into()).await?;
    Ok(())
}

async fn check(ctx: &SessionContext, location: &str, disposition: FinalPlanDisposition) -> E2eResult<()> {
    ctx.register_parquet("t", location, ParquetReadOptions::default()).await?;
    let plan = ctx.sql(QUERY).await?.create_physical_plan().await?;
    let observed = classify_final_plan(plan.as_ref());
    if observed != disposition {
        return Err(format!("final plan is {observed}, expected {disposition}").into());
    }
    let batches = collect(plan, ctx.task_ctx()).await?;
    let mut rows = Vec::new();
    for batch in &batches {
        let column = |index: usize| -> E2eResult<&Int64Array> {
            Ok(batch.column(index).as_any().downcast_ref::<Int64Array>().ok_or("result column is not Int64")?)
        };
        let (grp, count, total) = (column(0)?, column(1)?, column(2)?);
        for row in 0..batch.num_rows() {
            rows.push((grp.value(row), count.value(row), total.value(row)));
        }
    }
    let expected = (0..GROUPS)
        .map(|grp| {
            let ids = (0..ROWS).filter(|id| id % GROUPS == grp);
            let count = i64::try_from(ids.clone().count()).unwrap_or(i64::MAX);
            (grp, count, ids.map(|id| id * 2).sum::<i64>())
        })
        .collect::<Vec<_>>();
    if rows != expected {
        return Err(format!("query returned {rows:?}, expected {expected:?}").into());
    }
    ctx.deregister_table("t")?;
    Ok(())
}
