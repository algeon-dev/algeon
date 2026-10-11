//! Sort-lane sweep for plan F: single-query width comparison and a mixed-load
//! service check for lane-parallel top-k, on a self-generated dataset.
//!
//! The sweep builds one backend per profile width (`source_work_per_attempt`
//! sets the attempt's auxiliary fork pool, hence its execution width) and
//! runs a two-branch union sort/top-k through the public SQL attempt path,
//! so every cell exercises DataFusion lowering, compiled admission, and the
//! lane dispatcher as production does. A backend cannot change its frozen
//! profile mid-process, so widths run as separate backends in the configured
//! order; the report carries per-width variance and does not claim an
//! interleaved comparison.
//!
//! The union branches read distinct files, so the fused top-k consumes a
//! two-frame stream and expands one candidate per frame onto lanes. union
//! reads finish before the sort merge begins, which keeps the width
//! comparison on the sort phase even though the profile width also sizes
//! source concurrency.
//!
//! Everything is environment-variable driven. GPU runs must be wrapped in
//! `flock /tmp/cudf-gpu.lock`.

use std::{
    error::Error,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(feature = "cugraph")]
use algeon_datafusion::cugraph_sql::CugraphSqlConfig;
use algeon_datafusion::{
    attempt::{AttemptOptions, QueryAttemptReport},
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
    session::AlgeonSessionStateBuilderExt as _,
};
use arrow::{
    array::{Float32Array, Int64Array, RecordBatch},
    datatypes::{DataType, Field, Schema, SchemaRef},
    util::display::array_value_to_string,
};
use datafusion::{
    execution::SessionStateBuilder,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::StreamExt as _;
use parquet::arrow::ArrowWriter;
use query_planning::util::StableFnv1a64;
use query_runtime::observability::{AttemptCleanupDisposition, RuntimeTerminalOutcome};
use tokio::time::{sleep, timeout};

const ENV_ROWS_PER_BRANCH: &str = "ALGEON_SORT_SWEEP_ROWS_PER_BRANCH";
const ENV_WIDTHS: &str = "ALGEON_SORT_SWEEP_WIDTHS";
const ENV_REPS: &str = "ALGEON_SORT_SWEEP_REPS";
const ENV_OUT: &str = "ALGEON_SORT_SWEEP_OUT";
const ENV_DATA_DIR: &str = "ALGEON_SORT_SWEEP_DATA_DIR";
const ENV_QUERY_CAP_BYTES: &str = "ALGEON_SORT_SWEEP_QUERY_CAP_BYTES";
const ENV_DEVICE_CAPACITY_BYTES: &str = "ALGEON_SORT_SWEEP_DEVICE_CAPACITY_BYTES";
const ENV_QUERY_TIMEOUT_SECS: &str = "ALGEON_SORT_SWEEP_QUERY_TIMEOUT_SECS";

const HEAVY_SQL: &str = "SELECT k, v FROM a UNION ALL SELECT k, v FROM b ORDER BY k DESC LIMIT 100";
const LIGHT_SQL: &str = "SELECT COUNT(*) AS n, SUM(v) AS s FROM a WHERE k > 0";
#[cfg(feature = "cugraph")]
const GRAPH_SQL: &str =
    "SELECT COUNT(*) AS n, SUM(value) AS total FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM edges))";
const VECTOR_SQL: &str = r#"SELECT COUNT(*) AS n, SUM(neighbor_id) AS total FROM cuvs_brute_force_knn(dataset => (SELECT item_id AS id, d0, d1 FROM dataset_vectors), queries => (SELECT query_id AS id, d0, d1 FROM query_vectors), k => 5, metric => 'l2_sqrt_expanded')"#;

const MIXED_ROUNDS: usize = 5;

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where T::Err: std::fmt::Display {
    match std::env::var_os(name) {
        Some(value) => {
            value.to_string_lossy().parse().map_err(|error| format!("{name} failed to parse: {error}").into())
        },
        None => Ok(default),
    }
}

/// Deterministic 64-bit linear congruential generator. The dataset needs a
/// fixed seed, and statistical quality is not a requirement. The standard
/// library carries no RNG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0.rotate_left(31)
    }
}

fn sweep_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true), Field::new("v", DataType::Int64, true)]))
}

/// Writes one branch file with distinct shuffled keys, sparse nulls, and a
/// key-derived payload. Null keys sort first under the sweep's descending
/// nulls-first order; tied keys carry identical payloads so the unstable tie
/// pick cannot drift the fingerprint.
fn write_branch(path: &Path, rows: usize, seed: u64) -> Result<String, Box<dyn Error>> {
    let schema = sweep_schema();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)?;
    let mut rng = Lcg(seed);
    let mut keys: Vec<Option<i64>> = (0..rows as i64).map(Some).collect();
    for index in (1..keys.len()).rev() {
        let swap = (rng.next() as usize) % (index + 1);
        keys.swap(index, swap);
    }
    for (position, key) in keys.iter_mut().enumerate() {
        if position % 97 == 0 {
            *key = None;
        }
    }
    let values: Vec<Option<i64>> = keys.iter().map(|key| key.map(|key| key * 3 + 1)).collect();
    for chunk in keys.chunks(250_000).zip(values.chunks(250_000)) {
        let (keys, values) = chunk;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(keys.to_vec())), Arc::new(Int64Array::from(values.to_vec()))],
        )?;
        writer.write(&batch)?;
    }
    writer.close()?;
    let bytes = std::fs::metadata(path)?.len();
    Ok(format!("rows={rows} bytes={bytes} seed={seed}"))
}

/// Writes a small deterministic directed graph: a ring plus fixed-offset
/// shortcuts, so PageRank has one weakly connected component with stable
/// aggregates.
fn write_edges(path: &Path, vertices: usize) -> Result<String, Box<dyn Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float32, false),
    ]));
    let mut src = Vec::with_capacity(vertices * 2);
    let mut dst = Vec::with_capacity(vertices * 2);
    let mut weight = Vec::with_capacity(vertices * 2);
    for vertex in 0..vertices as i64 {
        src.push(vertex);
        dst.push((vertex + 1) % vertices as i64);
        weight.push(1.0_f32);
        src.push(vertex);
        dst.push((vertex + 7) % vertices as i64);
        weight.push(0.5_f32);
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(src)), Arc::new(Int64Array::from(dst)), Arc::new(Float32Array::from(weight))],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(format!("vertices={vertices} edges={} bytes={}", batch.num_rows(), std::fs::metadata(path)?.len()))
}

/// Writes deterministic 2-D vectors on a grid plus queries near grid
/// points, so brute-force KNN neighbor sets are stable.
fn write_vectors(path: &Path, items: usize, queries: usize, is_queries: bool) -> Result<String, Box<dyn Error>> {
    let id_name = if is_queries { "query_id" } else { "item_id" };
    let count = if is_queries { queries } else { items };
    let schema = Arc::new(Schema::new(vec![
        Field::new(id_name, DataType::Int64, false),
        Field::new("d0", DataType::Float32, false),
        Field::new("d1", DataType::Float32, false),
    ]));
    let mut ids = Vec::with_capacity(count);
    let mut d0 = Vec::with_capacity(count);
    let mut d1 = Vec::with_capacity(count);
    let mut rng = Lcg(if is_queries { 0x2545F4914F6CDD1D } else { 0x9E3779B97F4A7C15 });
    for index in 0..count as i64 {
        ids.push(index);
        if is_queries {
            let anchor = (rng.next() as usize) % items;
            d0.push((anchor % 200) as f32 + 0.1);
            d1.push((anchor / 200) as f32 + 0.1);
        } else {
            d0.push((index as usize % 200) as f32);
            d1.push((index as usize / 200) as f32);
        }
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(Float32Array::from(d0)), Arc::new(Float32Array::from(d1))],
    )?;
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(format!("rows={count} bytes={}", std::fs::metadata(path)?.len()))
}

struct SweepSample {
    wall_ms: f64,
    gpu_exec_ms: Option<f64>,
    rows: usize,
    fingerprint: String,
    disposition: String,
    outcome: String,
}

fn gpu_exec_ms(report: &QueryAttemptReport) -> Option<f64> {
    let start = report.timestamps.gpu_execution_started_ns?;
    let end = report.timestamps.outcome_terminal_ns;
    Some((end.saturating_sub(start)) as f64 / 1_000_000.0)
}

/// Order-sensitive fingerprint: the sweep asserts a global order, so row
/// sequence is hashed as content. Each batch is hashed column by column, so for
/// outputs with more than one column, a different batch split can change the
/// fingerprint.
fn fingerprint_ordered(batches: &[RecordBatch]) -> Result<String, Box<dyn Error>> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("algeon-bench-sort-sweep-ordered-v1");
    let mut rows = 0_usize;
    for batch in batches {
        for column in batch.columns() {
            for row in 0..batch.num_rows() {
                hasher.update_field(
                    &array_value_to_string(column, row).map_err(|error| format!("fingerprint render: {error}"))?,
                );
            }
        }
        rows += batch.num_rows();
    }
    hasher.update_field(&format!("rows={rows}"));
    Ok(format!("{:016x}:{rows}", hasher.finish()))
}

fn build_backend(
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    query_cap_bytes: u64,
    device_capacity_bytes: u64,
) -> Result<AlgeonGpuBackend, Box<dyn Error>> {
    let execution_config = AlgeonNativeExecutionConfig::new(SourceChunking::try_new(4 * 1024 * 1024, 1, 3, 1)?);
    let template = AlgeonGpuDeviceProfile::new(0)
        .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: device_capacity_bytes })
        .with_default_query_limit_bytes(query_cap_bytes)
        .with_max_active_attempts(max_active_attempts)
        .with_source_work_per_attempt(source_work_per_attempt)
        .with_derived_installation_bounds(max_active_attempts);
    let devices = AlgeonGpuDeviceConfig::homogeneous([0], &template, GpuMemoryOwnership::WholeDeviceExclusive);
    Ok(AlgeonGpuBackend::builder()
        .with_native_execution_config(execution_config)
        .devices(devices)
        .queue_policy(AlgeonGpuQueuePolicy::default())
        .build()?)
}

fn new_session(backend: &AlgeonGpuBackend) -> Result<SessionContext, Box<dyn Error>> {
    let session_config = SessionConfig::new()
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false);
    let builder = SessionStateBuilder::new().with_config(session_config).with_default_features();
    let builder = backend.install_on(builder)?;
    let builder = builder.try_with_cuvs_sql()?;
    #[cfg(feature = "cugraph")]
    let builder = builder.try_with_cugraph_sql(CugraphSqlConfig::default())?;
    Ok(SessionContext::new_with_state(builder.build()))
}

async fn run_query(
    backend: &AlgeonGpuBackend,
    ctx: &SessionContext,
    sql: &str,
    query_cap_bytes: u64,
    query_timeout: Duration,
) -> Result<(SweepSample, Option<Arc<QueryAttemptReport>>), Box<dyn Error>> {
    let started = Instant::now();
    let attempt = backend.begin_attempt_with_query_cap_bytes(AttemptOptions::default(), query_cap_bytes)?;
    let completion = attempt.completion();
    let outcome = timeout(query_timeout, async {
        let mut stream = attempt.sql(ctx, sql).await?;
        let mut batches = Vec::new();
        let mut rows = 0_usize;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            rows += batch.num_rows();
            batches.push(batch);
        }
        let fingerprint = fingerprint_ordered(&batches)?;
        Ok::<_, Box<dyn Error>>((rows, fingerprint))
    })
    .await;
    // If planning failed or timed out, dropping `attempt` commits the report waited on below.
    drop(attempt);
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let report = completion.wait_timeout(query_timeout);
    let (rows, fingerprint, error) = match outcome {
        Ok(Ok(output)) => (output.0, output.1, None),
        Ok(Err(error)) => (0, format!("error:{error}"), Some(error.to_string())),
        Err(_) => (0, "error:harness-timeout".to_string(), Some("harness timeout".to_string())),
    };
    if let Some(error) = error {
        return Err(format!("sweep query failed [{sql}]: {error}").into());
    }
    let report = report.ok_or("sweep attempt produced no sealed report")?;
    Ok((
        SweepSample {
            wall_ms,
            gpu_exec_ms: gpu_exec_ms(&report),
            rows,
            fingerprint,
            disposition: format!("{:?}", report.disposition),
            outcome: format!("{:?}", report.outcome),
        },
        Some(report),
    ))
}

fn percentile(sorted: &[f64], rank: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = (rank * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[position.min(sorted.len() - 1)]
}

fn summarize(values: &[f64]) -> (f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let median = percentile(&sorted, 0.5);
    let p95 = percentile(&sorted, 0.95);
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len().max(1) as f64;
    (median, p95, variance.sqrt())
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let rows_per_branch: usize = env_parse(ENV_ROWS_PER_BRANCH, 2_000_000)?;
    let widths: Vec<usize> = std::env::var_os(ENV_WIDTHS)
        .map(|value| {
            value
                .to_string_lossy()
                .split(',')
                .map(|part| {
                    part.trim().parse::<usize>().map_err(|error| format!("{ENV_WIDTHS} failed to parse: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![1, 2, 4]);
    let reps: usize = env_parse(ENV_REPS, 10)?;
    let out = std::env::var_os(ENV_OUT).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("sort-lane-sweep.tsv"));
    let data_dir = std::env::var_os(ENV_DATA_DIR)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("algeon-sort-lane-sweep"));
    let query_cap_bytes: u64 = env_parse(ENV_QUERY_CAP_BYTES, 2_000_000_000)?;
    let device_capacity_bytes: u64 = env_parse(ENV_DEVICE_CAPACITY_BYTES, 8_000_000_000)?;
    let query_timeout = Duration::from_secs(env_parse(ENV_QUERY_TIMEOUT_SECS, 300)?);
    if widths.is_empty() || reps == 0 || rows_per_branch == 0 {
        return Err("widths, reps, and rows per branch must be non-empty".into());
    }

    std::fs::create_dir_all(&data_dir)?;
    let branch_a = data_dir.join("a.parquet");
    let branch_b = data_dir.join("b.parquet");
    let identity_a = write_branch(&branch_a, rows_per_branch, 0x9E3779B97F4A7C15)?;
    let identity_b = write_branch(&branch_b, rows_per_branch, 0xBF58476D1CE4E5B9)?;
    let dataset_identity = format!("a[{identity_a}] b[{identity_b}]");

    let mut tsv = String::from("section\twidth\trep\twall_ms\tgpu_exec_ms\trows\tfingerprint\tdisposition\toutcome\n");
    let mut reference: Option<String> = None;
    for width in &widths {
        let backend = build_backend(*width, 1, query_cap_bytes, device_capacity_bytes)?;
        let ctx = new_session(&backend)?;
        ctx.register_parquet("a", branch_a.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        ctx.register_parquet("b", branch_b.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        let model = backend
            .device_startup_records()
            .first()
            .and_then(|record| record.model_name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!("[sweep] width={width} gpu={model} dataset={dataset_identity}");
        let (warm, _) = run_query(&backend, &ctx, HEAVY_SQL, query_cap_bytes, query_timeout).await?;
        eprintln!(
            "[sweep] width={width} warmup wall={:.1}ms rows={} disposition={}",
            warm.wall_ms, warm.rows, warm.disposition
        );
        if warm.disposition != "Native" {
            return Err(format!("width {width}: sort sweep must run natively, saw {}", warm.disposition).into());
        }
        let current = reference.get_or_insert_with(|| warm.fingerprint.clone());
        if warm.fingerprint != *current {
            return Err(format!("width {width}: fingerprint drift across widths").into());
        }
        for rep in 0..reps {
            let (sample, _) = run_query(&backend, &ctx, HEAVY_SQL, query_cap_bytes, query_timeout).await?;
            if sample.fingerprint != *current {
                return Err(format!("width {width} rep {rep}: fingerprint drift").into());
            }
            tsv.push_str(&format!(
                "single\t{width}\t{rep}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\n",
                sample.wall_ms,
                sample.gpu_exec_ms.unwrap_or(f64::NAN),
                sample.rows,
                sample.fingerprint,
                sample.disposition,
                sample.outcome,
            ));
        }
        let walls: Vec<f64> = tsv
            .lines()
            .skip(1)
            .filter(|line| line.starts_with(&format!("single\t{width}\t")))
            .filter_map(|line| line.split('\t').nth(3)?.parse::<f64>().ok())
            .collect();
        let (median, p95, stddev) = summarize(&walls);
        eprintln!(
            "[sweep] width={width} wall median={median:.1}ms p95={p95:.1}ms stddev={stddev:.1}ms n={}",
            walls.len()
        );
        backend.close_and_wait(Duration::from_secs(120))?;
    }

    let edges_path = data_dir.join("edges.parquet");
    let dataset_vectors_path = data_dir.join("dataset_vectors.parquet");
    let query_vectors_path = data_dir.join("query_vectors.parquet");
    let edges_identity = write_edges(&edges_path, 8192)?;
    let dataset_identity_extra = write_vectors(&dataset_vectors_path, 8192, 0, false)?;
    let query_identity_extra = write_vectors(&query_vectors_path, 8192, 256, true)?;
    eprintln!(
        "[sweep] mixed data edges[{edges_identity}] vectors[{dataset_identity_extra}] queries[{query_identity_extra}]"
    );

    let backend = build_backend(2, 4, query_cap_bytes, device_capacity_bytes)?;
    let heavy_ctx = Arc::new(new_session(&backend)?);
    let light_ctx = Arc::new(new_session(&backend)?);
    let graph_ctx = Arc::new(new_session(&backend)?);
    let vector_ctx = Arc::new(new_session(&backend)?);
    for ctx in [&heavy_ctx, &light_ctx, &graph_ctx, &vector_ctx] {
        ctx.register_parquet("a", branch_a.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        ctx.register_parquet("b", branch_b.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        ctx.register_parquet("edges", edges_path.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        ctx.register_parquet(
            "dataset_vectors",
            dataset_vectors_path.to_string_lossy().as_ref(),
            ParquetReadOptions::default(),
        )
        .await?;
        ctx.register_parquet(
            "query_vectors",
            query_vectors_path.to_string_lossy().as_ref(),
            ParquetReadOptions::default(),
        )
        .await?;
    }
    let mixed_items: Vec<(&str, &str)> = vec![("heavy", HEAVY_SQL), ("light", LIGHT_SQL), ("vector", VECTOR_SQL)];
    #[cfg(feature = "cugraph")]
    let mixed_items = mixed_items.into_iter().chain([("graph", GRAPH_SQL)]).collect::<Vec<_>>();
    let mut references: std::collections::BTreeMap<&str, String> = std::collections::BTreeMap::new();
    references.insert("heavy", reference.as_ref().expect("single section sets the heavy reference").clone());
    for (label, sql) in &mixed_items {
        if *label == "heavy" {
            continue;
        }
        let ctx = match *label {
            "light" => &light_ctx,
            "graph" => &graph_ctx,
            _ => &vector_ctx,
        };
        let (warm, _) = run_query(&backend, ctx, sql, query_cap_bytes, query_timeout).await?;
        eprintln!(
            "[sweep] mixed warmup {label} wall={:.1}ms rows={} disposition={}",
            warm.wall_ms, warm.rows, warm.disposition
        );
        if warm.outcome != "Succeeded" {
            return Err(format!("mixed warmup {label}: outcome {}", warm.outcome).into());
        }
        references.insert(label, warm.fingerprint);
    }
    let mut walls: std::collections::BTreeMap<&str, Vec<f64>> = std::collections::BTreeMap::new();
    let mut queue_waits: std::collections::BTreeMap<&str, Vec<f64>> = std::collections::BTreeMap::new();
    for round in 0..MIXED_ROUNDS {
        let mut tasks = Vec::new();
        for (label, sql) in &mixed_items {
            let ctx = match *label {
                "heavy" => Arc::clone(&heavy_ctx),
                "light" => Arc::clone(&light_ctx),
                "graph" => Arc::clone(&graph_ctx),
                _ => Arc::clone(&vector_ctx),
            };
            let task_backend = backend.clone();
            let sql = sql.to_string();
            tasks.push(tokio::spawn(async move {
                run_query(&task_backend, &ctx, &sql, query_cap_bytes, query_timeout)
                    .await
                    .map_err(|error| error.to_string())
            }));
        }
        for ((label, _), task) in mixed_items.iter().zip(tasks) {
            let (sample, report) =
                task.await.map_err(|error| format!("mixed round {round} {label} join: {error}"))??;
            let report = report.expect("mixed attempts seal reports");
            if sample.outcome != "Succeeded" {
                return Err(format!("mixed round {round} {label}: outcome {}", sample.outcome).into());
            }
            if sample.fingerprint != references[*label] {
                return Err(format!("mixed round {round} {label}: fingerprint drift").into());
            }
            let queue_wait_ms = match (report.timestamps.admission_requested_ns, report.timestamps.admitted_ns) {
                (Some(requested), Some(admitted)) => Some(admitted.saturating_sub(requested) as f64 / 1_000_000.0),
                _ => None,
            }
            .unwrap_or(f64::NAN);
            walls.entry(label).or_default().push(sample.wall_ms);
            queue_waits.entry(label).or_default().push(queue_wait_ms);
            tsv.push_str(&format!(
                "mixed\t2\t{round}-{label}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\n",
                sample.wall_ms,
                sample.gpu_exec_ms.unwrap_or(f64::NAN),
                sample.rows,
                sample.fingerprint,
                sample.disposition,
                sample.outcome,
            ));
            if report.final_domain_usage.as_ref().is_some_and(|usage| usage.current_bytes != 0) {
                return Err(format!("mixed round {round} {label}: residual domain charge").into());
            }
            if !matches!(
                report.cleanup_disposition,
                AttemptCleanupDisposition::Completed | AttemptCleanupDisposition::NotRequired
            ) {
                return Err(format!("mixed round {round} {label}: attempt did not clean up").into());
            }
        }
    }
    for label in mixed_items.iter().map(|(label, _)| *label) {
        let samples = walls[label].clone();
        let (median, p95, _) = summarize(&samples);
        let waits = queue_waits[label].clone();
        let (wait_median, wait_p95, _) = summarize(&waits);
        eprintln!(
            "[sweep] mixed {label} wall median={median:.1}ms p95={p95:.1}ms queue_wait median={wait_median:.1}ms \
             p95={wait_p95:.1}ms n={}",
            samples.len()
        );
    }
    // Cancel probe: a heavy attempt cancelled mid-flight must reach a
    // terminal cancelled report with cleaned resources, without hanging or leaking.
    // The cancel races execution; a completed-before-cancel attempt records
    // no latency instead of failing.
    {
        let cancel_ctx = Arc::new(new_session(&backend)?);
        cancel_ctx.register_parquet("a", branch_a.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        cancel_ctx.register_parquet("b", branch_b.to_string_lossy().as_ref(), ParquetReadOptions::default()).await?;
        let attempt = backend.begin_attempt_with_query_cap_bytes(AttemptOptions::default(), query_cap_bytes)?;
        let cancellation = attempt.cancellation();
        let completion = attempt.completion();
        let drain = tokio::spawn(async move {
            let mut stream = attempt.sql(&cancel_ctx, HEAVY_SQL).await.map_err(|error| error.to_string())?;
            while stream.next().await.is_some() {}
            Ok::<_, String>(())
        });
        sleep(Duration::from_millis(10)).await;
        if drain.is_finished() {
            eprintln!("[sweep] cancel probe skipped: heavy attempt finished before cancel");
            drain.await.map_err(|error| format!("cancel probe join: {error}"))??;
        } else {
            cancellation.cancel();
            let cancel_at = Instant::now();
            let outcome = timeout(query_timeout, drain).await;
            let cancel_latency_ms = cancel_at.elapsed().as_secs_f64() * 1000.0;
            let report = completion.wait_timeout(query_timeout).ok_or("cancel probe sealed no report")?;
            let outcome_label = format!("{:?}", report.outcome);
            eprintln!(
                "[sweep] cancel probe outcome={outcome_label} cleanup={:?} cancel_latency={cancel_latency_ms:.1}ms",
                report.cleanup_disposition
            );
            tsv.push_str(&format!(
                "cancel\t2\tcancel\t{cancel_latency_ms:.3}\tNaN\t0\t{outcome_label}\t{:?}\t{:?}\n",
                report.disposition, report.outcome
            ));
            if !matches!(report.outcome, RuntimeTerminalOutcome::Cancelled) {
                let _ = outcome;
                return Err(format!("cancel probe: expected Cancelled, saw {outcome_label}").into());
            }
            if report.final_domain_usage.as_ref().is_some_and(|usage| usage.current_bytes != 0) {
                return Err("cancel probe: residual domain charge".into());
            }
        }
    }
    backend.close_and_wait(Duration::from_secs(120))?;
    std::fs::write(&out, &tsv)?;
    eprintln!("[sweep] wrote {}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_reports_median_p95_and_stddev() {
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        let (median, p95, stddev) = summarize(&values);
        assert_eq!(median, 6.0);
        assert_eq!(p95, 10.0);
        assert!((stddev - 2.8722813232690143).abs() < 1e-9, "stddev={stddev}");
        assert!(percentile(&[], 0.5).is_nan(), "empty samples stay NaN, never zero");
    }
}
