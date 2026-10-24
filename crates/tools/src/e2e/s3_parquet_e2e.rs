//! Explicit E2E for `s3://` Parquet scans through one S3 configuration.
//!
//! The runner needs real S3 or an S3-compatible service and an existing
//! bucket, so it stays out of Cargo test targets. It uploads generated Parquet
//! objects through the session's `s3://` object store, then runs one aggregate
//! over three layouts: one object, a prefix of three objects and one object
//! that DataFusion splits into byte ranges. Each layout runs in a
//! native-required and a functions-only session from a library backend built
//! with the S3 storage, and in a native-required session built from server
//! configuration. Every check plans and executes through an attempt, because
//! only an attempt's planning invocation carries the remote footers native
//! lowering reads. Each final plan must be the disposition its mode requires
//! and each result must equal the generated data.

use std::{
    env,
    error::Error,
    fmt::Write as _,
    fs,
    ops::Range,
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    attempt::{AlgeonAttempt, AttemptOptions},
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
    datasource::{physical_plan::ParquetSource, source::DataSourceExec},
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl},
    object_store::{ObjectStore, ObjectStoreExt, path::Path as ObjectStorePath},
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionContext},
};
use futures::TryStreamExt;

type E2eResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const ENDPOINT_ENV: &str = "ALGEON_S3_PARQUET_E2E_ENDPOINT";
const BUCKET_ENV: &str = "ALGEON_S3_PARQUET_E2E_BUCKET";
const REGION_ENV: &str = "ALGEON_S3_PARQUET_E2E_REGION";
const DEFAULT_REGION: &str = "us-east-1";
const GROUPS: i64 = 7;
const SMALL_OBJECT_ROWS: i64 = 10_000;
const MULTI_OBJECT_COUNT: i64 = 3;
/// Large enough that one object spans many row groups and a few megabytes, so
/// a byte-range split lands inside it.
const RANGED_OBJECT_ROWS: i64 = 400_000;
const ROW_GROUP_ROWS: usize = 25_000;
const RANGE_TARGET_PARTITIONS: usize = 4;
const QUERY: &str = "SELECT grp, COUNT(*) AS n, SUM(val) AS total FROM t GROUP BY grp ORDER BY grp";
const CLOSE_TIMEOUT: Duration = Duration::from_secs(60);

struct Settings {
    endpoint: Option<String>,
    bucket: String,
    region: String,
    storage: S3SourceStorageProfile,
}

struct Object {
    key: String,
    ids: Range<i64>,
}

struct Case {
    name: &'static str,
    location: String,
    objects: Vec<Object>,
    /// The case splits its one object into byte ranges.
    ranged: bool,
}

pub async fn run(args: Vec<String>) -> E2eResult<()> {
    if let Some(arg) = args.first() {
        if arg == "-h" || arg == "--help" {
            print_usage();
            return Ok(());
        }
        return Err(format!("unknown s3-parquet-e2e argument: {arg}").into());
    }
    let settings = Settings::from_env()?;
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run_id = format!("{}-{nanos}", std::process::id());
    let cases = cases(&settings.bucket, &run_id);

    let backend = library_backend(&settings.storage, AlgeonGpuExecutionMode::NativeRequired)?;
    let ctx = library_session(&backend)?;
    let store = ctx.runtime_env().object_store(ObjectStoreUrl::parse(format!("s3://{}", settings.bucket))?)?;
    close(ctx, backend)?;

    let mut uploaded = Vec::new();
    let result = async {
        upload(store.as_ref(), &cases, &mut uploaded).await?;
        run_scenarios(&settings, &cases, &run_id).await
    }
    .await;
    let cleanup = delete(store.as_ref(), &uploaded).await;
    result?;
    cleanup?;
    println!("s3_parquet_e2e ok");
    Ok(())
}

async fn run_scenarios(settings: &Settings, cases: &[Case], run_id: &str) -> E2eResult<()> {
    for (label, mode, disposition) in [
        ("library native_required", AlgeonGpuExecutionMode::NativeRequired, FinalPlanDisposition::Native),
        ("library functions_only", AlgeonGpuExecutionMode::FunctionsOnly, FinalPlanDisposition::DataFusion),
    ] {
        let backend = library_backend(&settings.storage, mode)?;
        let ctx = library_session(&backend)?;
        check_cases(label, &backend, &ctx, cases, disposition).await?;
        close(ctx, backend)?;
        println!("{label} ok");
    }

    let config_dir = env::temp_dir().join(format!("algeon-s3-parquet-e2e-{run_id}"));
    let result = async {
        let config = server_config(&config_dir, settings)?;
        let backend = backend_from_server_config(&config)?;
        let ctx = build_session_context_with_backend(&config, &backend).await?;
        check_cases("server native_required", &backend, &ctx, cases, FinalPlanDisposition::Native).await?;
        close(ctx, backend)
    }
    .await;
    fs::remove_dir_all(&config_dir)?;
    result?;
    println!("server native_required ok");
    Ok(())
}

impl Settings {
    fn from_env() -> E2eResult<Self> {
        let bucket = required_env(BUCKET_ENV)?;
        let endpoint = optional_env(ENDPOINT_ENV);
        let region = optional_env(REGION_ENV)
            .or_else(|| optional_env("AWS_REGION"))
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());
        let mut storage = S3SourceStorageProfile::default_chain().with_region(region.clone());
        // A custom endpoint is addressed path-style. Real S3 keeps the
        // default virtual-hosted addressing.
        if let Some(endpoint) = &endpoint {
            storage = storage.with_endpoint(endpoint.clone()).with_path_style_access(true);
        }
        Ok(Self { endpoint, bucket, region, storage })
    }
}

fn print_usage() {
    println!(
        "Usage: s3-parquet-e2e\n\n\
         Uploads generated Parquet objects under a unique key prefix, checks one object, a prefix of three\n\
         objects and one object split into byte ranges in library and server-configured sessions, and\n\
         deletes the objects. {BUCKET_ENV} names an existing bucket. {ENDPOINT_ENV} selects an\n\
         S3-compatible service addressed path-style; without it the run uses real S3 with virtual-hosted\n\
         addressing. {REGION_ENV} defaults to AWS_REGION, then {DEFAULT_REGION}. Credentials come from the\n\
         default AWS chain.\n\n\
         Local RustFS (compose file: crates/tools/fixtures/s3-local/compose.yaml):\n\
           docker compose -f crates/tools/fixtures/s3-local/compose.yaml up -d --wait\n\
           ALGEON_S3_PARQUET_E2E_ENDPOINT=http://127.0.0.1:9000 ALGEON_S3_PARQUET_E2E_BUCKET=algeon-e2e \\\n\
           AWS_ACCESS_KEY_ID=algeon-e2e AWS_SECRET_ACCESS_KEY=algeon-e2e-secret \\\n\
             flock /tmp/cudf-gpu.lock cargo run -p tools --features s3-parquet-e2e -- s3-parquet-e2e\n\
           docker compose -f crates/tools/fixtures/s3-local/compose.yaml down -v\n\n\
         Real S3 (credentials from the default chain):\n\
           ALGEON_S3_PARQUET_E2E_BUCKET=<bucket> ALGEON_S3_PARQUET_E2E_REGION=<region> \\\n\
             flock /tmp/cudf-gpu.lock cargo run -p tools --features s3-parquet-e2e -- s3-parquet-e2e"
    );
}

fn required_env(name: &str) -> E2eResult<String> {
    optional_env(name).ok_or_else(|| format!("{name} must be set").into())
}

fn optional_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
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

fn server_config(dir: &Path, settings: &Settings) -> E2eResult<ServerConfig> {
    fs::create_dir_all(dir)?;
    let path = dir.join("server.toml");
    let mut toml = String::from("[native]\nexecution_mode = \"native_required\"\n\n[s3]\n");
    writeln!(toml, "region = \"{}\"", settings.region)?;
    if let Some(endpoint) = &settings.endpoint {
        writeln!(toml, "endpoint = \"{endpoint}\"\npath_style = true")?;
    }
    toml.push_str("\n[[admission.device_profiles]]\ndevice_ordinal = 0\n");
    fs::write(&path, toml)?;
    let config = ServerConfig::from_vars([(ALGEON_SERVER_CONFIG_FILE_ENV, path.to_string_lossy().into_owned())])?;
    Ok(config.with_gpu_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)]))
}

fn close(ctx: SessionContext, backend: AlgeonGpuBackend) -> E2eResult<()> {
    // The next backend in this process needs the owner lease this one holds.
    drop(ctx);
    backend.close_and_wait(CLOSE_TIMEOUT)?;
    Ok(())
}

fn cases(bucket: &str, run_id: &str) -> Vec<Case> {
    let prefix = format!("algeon-s3-parquet-e2e/{run_id}");
    let single_key = format!("{prefix}/single/data.parquet");
    let ranged_key = format!("{prefix}/ranged/data.parquet");
    let multi_objects = (0..MULTI_OBJECT_COUNT)
        .map(|index| Object {
            key: format!("{prefix}/multi/part-{index}.parquet"),
            ids: index * SMALL_OBJECT_ROWS..(index + 1) * SMALL_OBJECT_ROWS,
        })
        .collect();
    vec![
        Case {
            name: "one object",
            location: format!("s3://{bucket}/{single_key}"),
            objects: vec![Object { key: single_key, ids: 0..SMALL_OBJECT_ROWS }],
            ranged: false,
        },
        Case {
            name: "prefix of three objects",
            location: format!("s3://{bucket}/{prefix}/multi/"),
            objects: multi_objects,
            ranged: false,
        },
        Case {
            name: "one object in byte ranges",
            location: format!("s3://{bucket}/{ranged_key}"),
            objects: vec![Object { key: ranged_key, ids: 0..RANGED_OBJECT_ROWS }],
            ranged: true,
        },
    ]
}

fn object_batch(ids: &Range<i64>) -> E2eResult<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("grp", DataType::Int64, false),
        Field::new("val", DataType::Int64, false),
    ]));
    let ids = ids.clone().collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Int64Array::from(ids.iter().map(|id| id % GROUPS).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(ids.iter().map(|id| id * 2).collect::<Vec<_>>())),
        ],
    )?)
}

/// Writes the generated objects through the object store the installed S3
/// storage resolves for the bucket. Keys are recorded before each write so a
/// failed run still deletes what reached the bucket.
async fn upload(store: &dyn ObjectStore, cases: &[Case], uploaded: &mut Vec<ObjectStorePath>) -> E2eResult<()> {
    let properties = WriterProperties::builder().set_max_row_group_row_count(Some(ROW_GROUP_ROWS)).build();
    for object in cases.iter().flat_map(|case| &case.objects) {
        let batch = object_batch(&object.ids)?;
        let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), Some(properties.clone()))?;
        writer.write(&batch)?;
        let bytes = writer.into_inner()?;
        let path = ObjectStorePath::from(object.key.as_str());
        uploaded.push(path.clone());
        store.put(&path, bytes.into()).await?;
    }
    Ok(())
}

async fn delete(store: &dyn ObjectStore, uploaded: &[ObjectStorePath]) -> E2eResult<()> {
    for path in uploaded {
        store.delete(path).await?;
    }
    Ok(())
}

async fn check_cases(
    label: &str,
    backend: &AlgeonGpuBackend,
    ctx: &SessionContext,
    cases: &[Case],
    disposition: FinalPlanDisposition,
) -> E2eResult<()> {
    for case in cases {
        if case.ranged {
            split_scans_into_ranges(ctx);
        }
        let ranged_files = check(backend, ctx, case, disposition)
            .await
            .map_err(|error| -> Box<dyn Error + Send + Sync> { format!("{label}, {}: {error}", case.name).into() })?;
        println!("{label}, {}: {disposition} plan, {ranged_files} ranged files, result ok", case.name);
    }
    Ok(())
}

/// Lets DataFusion split the one object of the ranged case across several
/// file groups. The setting stays on the session for the cases after it.
fn split_scans_into_ranges(ctx: &SessionContext) {
    let state = ctx.state_ref();
    let mut state = state.write();
    let options = state.config_mut().options_mut();
    options.optimizer.repartition_file_min_size = 1;
    options.execution.target_partitions = RANGE_TARGET_PARTITIONS;
}

/// Plans and runs the aggregate over the case's objects as one attempt and
/// returns the number of byte-ranged files in the scanned plan.
async fn check(
    backend: &AlgeonGpuBackend,
    ctx: &SessionContext,
    case: &Case,
    disposition: FinalPlanDisposition,
) -> E2eResult<usize> {
    ctx.register_parquet("t", &case.location, ParquetReadOptions::default()).await?;
    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    let plan = attempt.plan_sql(ctx, QUERY).await?;
    let observed = classify_final_plan(plan.as_ref());
    if observed != disposition {
        return Err(format!("final plan is {observed}, expected {disposition}").into());
    }
    let ranged_files = ranged_file_count(&attempt, plan.as_ref(), disposition);
    if case.ranged != (ranged_files > 1) {
        return Err(format!(
            "plan has {ranged_files} ranged files, but the case expects {}",
            if case.ranged { "several" } else { "none" }
        )
        .into());
    }
    let batches = attempt.execute_physical_plan(plan, ctx)?.try_collect::<Vec<_>>().await?;
    let rows = result_rows(&batches)?;
    let expected = expected_rows(case);
    if rows != expected {
        return Err(format!("query returned {rows:?}, expected {expected:?}").into());
    }
    ctx.deregister_table("t")?;
    Ok(ranged_files)
}

/// A native plan no longer holds the file scan, so its ranged files come from
/// the attempt's source diagnostics. A DataFusion plan still holds them.
fn ranged_file_count(attempt: &AlgeonAttempt, plan: &dyn ExecutionPlan, disposition: FinalPlanDisposition) -> usize {
    match disposition {
        FinalPlanDisposition::Native => attempt.planning_diagnostics().map_or(0, |diagnostics| {
            diagnostics.source_diagnostics.iter().map(|source| source.ranged_file_count).sum()
        }),
        _ => datafusion_ranged_files(plan),
    }
}

fn datafusion_ranged_files(plan: &dyn ExecutionPlan) -> usize {
    let own = plan
        .downcast_ref::<DataSourceExec>()
        .and_then(|scan| scan.downcast_to_file_source::<ParquetSource>())
        .map_or(0, |(file_scan, _)| {
            file_scan.file_groups.iter().flat_map(|group| group.files()).filter(|file| file.range.is_some()).count()
        });
    own + plan.children().into_iter().map(|child| datafusion_ranged_files(child.as_ref())).sum::<usize>()
}

fn result_rows(batches: &[RecordBatch]) -> E2eResult<Vec<(i64, i64, i64)>> {
    let mut rows = Vec::new();
    for batch in batches {
        let column = |index: usize| -> E2eResult<&Int64Array> {
            Ok(batch.column(index).as_any().downcast_ref::<Int64Array>().ok_or("result column is not Int64")?)
        };
        let (grp, count, total) = (column(0)?, column(1)?, column(2)?);
        for row in 0..batch.num_rows() {
            rows.push((grp.value(row), count.value(row), total.value(row)));
        }
    }
    Ok(rows)
}

fn expected_rows(case: &Case) -> Vec<(i64, i64, i64)> {
    (0..GROUPS)
        .map(|grp| {
            let ids = case.objects.iter().flat_map(|object| object.ids.clone()).filter(|id| id % GROUPS == grp);
            let count = i64::try_from(ids.clone().count()).unwrap_or(i64::MAX);
            (grp, count, ids.map(|id| id * 2).sum::<i64>())
        })
        .collect()
}
