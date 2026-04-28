use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue, utils::memory::get_record_batch_memory_size},
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSourceExec,
    },
    execution::context::TaskContext,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties},
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect, displayable,
        filter::FilterExec,
        limit::LocalLimitExec,
        metrics::MetricsSet,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfGpuIslandExec, CudfOptimizerConfig, GpuFallbackPolicy, LargerIslandBenchmarkMetrics,
    LargerIslandBenchmarkPlanning, TimedBenchmarkRun, optimize_plan,
};

const THREE_GIB: u64 = 3_221_225_472;
const DEFAULT_BENCHMARK_DATA_DIR: &str = "/opt/dlami/nvme/data";

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let config = LocalParquetBenchmarkConfig::from_env()?;
    let dir = config.output_dir.clone().unwrap_or_else(default_benchmark_dir);
    fs::create_dir_all(&dir)?;
    let path = dir.join("wide.parquet");

    write_benchmark_parquet(&path, &config)?;
    let file_bytes = fs::metadata(&path)?.len();
    validate_file_size(file_bytes, &config)?;

    let source_diagnostic = local_parquet_scan_diagnostic(&path).await?;
    println!("local_parquet_benchmark");
    println!("parquet_path={}", path.display());
    println!("source_diagnostic={source_diagnostic}");
    println!("parquet_file_bytes={file_bytes}");
    println!("input_rows={}", config.input_rows);
    println!("measure_columns={}", config.measure_columns);
    println!("projected_measure_pairs={}", config.projected_measure_pairs());
    println!("group_count={}", config.group_count);
    println!("batch_size={}", config.batch_size);
    println!("row_group_size={}", config.row_group_size);
    println!("compression={}", config.compression.name());
    println!("limit={}", config.limit);
    println!("warmups={}", config.warmup_iterations);
    println!("iterations={}", config.iterations);

    let report = run_direct_parquet_benchmark(&path, file_bytes, source_diagnostic, &config).await?;
    println!();
    print!("{}", report.to_text());

    Ok(())
}

#[derive(Debug, Clone)]
struct LocalParquetBenchmarkConfig {
    input_rows: usize,
    measure_columns: usize,
    batch_size: usize,
    row_group_size: usize,
    group_count: usize,
    limit: usize,
    warmup_iterations: usize,
    iterations: usize,
    minimum_file_bytes: u64,
    compression: BenchmarkCompression,
    output_dir: Option<PathBuf>,
}

impl LocalParquetBenchmarkConfig {
    fn from_env() -> DataFusionResult<Self> {
        let config = Self {
            input_rows: env_usize("CUDF_PARQUET_BENCH_ROWS", 12_250_000)?,
            measure_columns: env_usize("CUDF_PARQUET_BENCH_MEASURE_COLUMNS", 32)?,
            batch_size: env_usize("CUDF_PARQUET_BENCH_BATCH_SIZE", 65_536)?,
            row_group_size: env_usize("CUDF_PARQUET_BENCH_ROW_GROUP_SIZE", 16_777_216)?,
            group_count: env_usize("CUDF_PARQUET_BENCH_GROUPS", 1_048_576)?,
            limit: env_usize("CUDF_PARQUET_BENCH_LIMIT", 100)?,
            warmup_iterations: env_usize("CUDF_PARQUET_BENCH_WARMUPS", 1)?,
            iterations: env_usize("CUDF_PARQUET_BENCH_ITERATIONS", 3)?,
            minimum_file_bytes: env_u64("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", THREE_GIB)?,
            compression: BenchmarkCompression::from_env("CUDF_PARQUET_BENCH_COMPRESSION")?,
            output_dir: std::env::var_os("CUDF_PARQUET_BENCH_DIR").map(PathBuf::from),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("CUDF_PARQUET_BENCH_ROWS", self.input_rows)?;
        validate_nonzero("CUDF_PARQUET_BENCH_MEASURE_COLUMNS", self.measure_columns)?;
        validate_nonzero("CUDF_PARQUET_BENCH_BATCH_SIZE", self.batch_size)?;
        validate_nonzero("CUDF_PARQUET_BENCH_ROW_GROUP_SIZE", self.row_group_size)?;
        validate_nonzero("CUDF_PARQUET_BENCH_GROUPS", self.group_count)?;
        validate_nonzero("CUDF_PARQUET_BENCH_LIMIT", self.limit)?;
        validate_nonzero("CUDF_PARQUET_BENCH_ITERATIONS", self.iterations)?;
        if self.measure_columns % 2 != 0 {
            return Err(plan_error(format!(
                "CUDF_PARQUET_BENCH_MEASURE_COLUMNS must be even, got {}",
                self.measure_columns
            )));
        }
        if self.measure_columns > 100 {
            return Err(plan_error(format!(
                "CUDF_PARQUET_BENCH_MEASURE_COLUMNS must be <= 100 because columns are named m00..m99, got {}",
                self.measure_columns
            )));
        }
        if self.group_count > i32::MAX as usize {
            return Err(plan_error(format!("CUDF_PARQUET_BENCH_GROUPS must fit in Int32, got {}", self.group_count)));
        }
        Ok(())
    }

    fn projected_measure_pairs(&self) -> usize {
        self.measure_columns / 2
    }

    fn expected_read_columns(&self) -> Vec<String> {
        let mut columns = Vec::with_capacity(self.measure_columns + 2);
        columns.push("key".to_string());
        columns.push("payload".to_string());
        columns.extend((0..self.measure_columns).map(measure_column_name));
        columns.sort();
        columns
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkCompression {
    Uncompressed,
    Snappy,
}

impl BenchmarkCompression {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::Uncompressed);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "uncompressed" | "none" => Ok(Self::Uncompressed),
            "snappy" => Ok(Self::Snappy),
            other => Err(plan_error(format!("invalid {name}: {other}; expected uncompressed or snappy"))),
        }
    }

    fn parquet(self) -> Compression {
        match self {
            Self::Uncompressed => Compression::UNCOMPRESSED,
            Self::Snappy => Compression::SNAPPY,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Uncompressed => "uncompressed",
            Self::Snappy => "snappy",
        }
    }
}

#[derive(Debug)]
struct DirectParquetBenchmarkReport {
    parquet_file_bytes: u64,
    input_rows: usize,
    measure_columns: usize,
    batch_size: usize,
    row_group_size: usize,
    group_count: usize,
    warmup_iterations: usize,
    iterations: usize,
    limit: usize,
    compression: BenchmarkCompression,
    measured_run_order: String,
    cpu_comparable: TimedBenchmarkRun,
    cpu_default: TimedBenchmarkRun,
    gpu: TimedBenchmarkRun,
    cpu_comparable_samples: Vec<Duration>,
    cpu_default_samples: Vec<Duration>,
    gpu_samples: Vec<Duration>,
    gpu_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    final_cpu_comparable: CollectMeasurement,
    final_cpu_default: CollectMeasurement,
    final_gpu: GpuCollectMeasurement,
    source_diagnostic: String,
}

impl DirectParquetBenchmarkReport {
    fn to_text(&self) -> String {
        let comparable_speedup = speedup(self.cpu_comparable.mean, self.gpu.mean);
        let default_speedup = speedup(self.cpu_default.mean, self.gpu.mean);
        let metrics = &self.final_gpu.metrics;
        let planning = &self.final_gpu.planning;
        let mut text = String::new();

        push_line(&mut text, format_args!("case=gpu_friendly_local_parquet"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("measure_columns={}", self.measure_columns));
        push_line(&mut text, format_args!("batch_size={}", self.batch_size));
        push_line(&mut text, format_args!("row_group_size={}", self.row_group_size));
        push_line(&mut text, format_args!("group_count={}", self.group_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("warmups={}", self.warmup_iterations));
        push_line(&mut text, format_args!("iterations={}", self.iterations));
        push_line(&mut text, format_args!("measured_run_order={}", self.measured_run_order));
        push_line(&mut text, format_args!("cpu_comparable_wall_ms={:.3}", duration_ms(self.cpu_comparable.mean)));
        push_line(&mut text, format_args!("cpu_wall_ms={:.3}", duration_ms(self.cpu_comparable.mean)));
        push_line(&mut text, format_args!("cpu_default_wall_ms={:.3}", duration_ms(self.cpu_default.mean)));
        push_line(&mut text, format_args!("gpu_wall_ms={:.3}", duration_ms(self.gpu.mean)));
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_comparable={comparable_speedup:.3}"));
        push_line(&mut text, format_args!("gpu_speedup={comparable_speedup:.3}"));
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_default={default_speedup:.3}"));
        push_line(
            &mut text,
            format_args!("cpu_comparable_wall_ms_samples={}", duration_samples_csv(&self.cpu_comparable_samples)),
        );
        push_line(
            &mut text,
            format_args!("cpu_default_wall_ms_samples={}", duration_samples_csv(&self.cpu_default_samples)),
        );
        push_line(&mut text, format_args!("gpu_wall_ms_samples={}", duration_samples_csv(&self.gpu_samples)));
        push_line(&mut text, format_args!("output_rows={}", self.final_gpu.output_rows));
        push_line(&mut text, format_args!("output_batches={}", self.final_gpu.output_batches));
        push_line(&mut text, format_args!("output_bytes={}", self.final_gpu.output_bytes));
        push_line(&mut text, format_args!("cpu_comparable_output_rows={}", self.final_cpu_comparable.output_rows));
        push_line(
            &mut text,
            format_args!("cpu_comparable_output_batches={}", self.final_cpu_comparable.output_batches),
        );
        push_line(&mut text, format_args!("cpu_comparable_output_bytes={}", self.final_cpu_comparable.output_bytes));
        push_line(&mut text, format_args!("cpu_default_output_rows={}", self.final_cpu_default.output_rows));
        push_line(&mut text, format_args!("cpu_default_output_batches={}", self.final_cpu_default.output_batches));
        push_line(&mut text, format_args!("cpu_default_output_bytes={}", self.final_cpu_default.output_bytes));
        push_line(&mut text, format_args!("limit={}", self.limit));
        push_line(&mut text, format_args!("parquet_read_ms={:.3}", duration_ms(metrics.parquet_read_time)));
        push_line(
            &mut text,
            format_args!(
                "parquet_read_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| metrics.parquet_read_time))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "parquet_read_ms_samples={}",
                metric_duration_samples_csv(&self.gpu_metric_samples, |metrics| metrics.parquet_read_time)
            ),
        );
        push_line(&mut text, format_args!("cudf_operator_ms={:.3}", duration_ms(metrics.cudf_operator_time)));
        push_line(
            &mut text,
            format_args!(
                "cudf_operator_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| metrics.cudf_operator_time))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "cudf_operator_ms_samples={}",
                metric_duration_samples_csv(&self.gpu_metric_samples, |metrics| metrics.cudf_operator_time)
            ),
        );
        push_line(
            &mut text,
            format_args!("record_batch_upload_ms={:.3}", duration_ms(metrics.record_batch_upload_time)),
        );
        push_line(
            &mut text,
            format_args!(
                "record_batch_upload_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| metrics.record_batch_upload_time))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "record_batch_materialization_ms={:.3}",
                duration_ms(metrics.record_batch_materialization_time)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "record_batch_materialization_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| {
                    metrics.record_batch_materialization_time
                }))
            ),
        );
        push_line(&mut text, format_args!("host_to_device_uploads={}", metrics.host_to_device_uploads));
        push_line(&mut text, format_args!("host_to_device_uploaded_rows={}", metrics.host_to_device_uploaded_rows));
        push_line(
            &mut text,
            format_args!("device_to_host_materializations={}", metrics.device_to_host_materializations),
        );
        push_line(
            &mut text,
            format_args!("device_to_host_materialized_rows={}", metrics.device_to_host_materialized_rows),
        );
        push_line(&mut text, format_args!("parquet_files={}", metrics.parquet_files));
        push_line(&mut text, format_args!("parquet_bytes_known={}", metrics.parquet_bytes_known));
        push_line(&mut text, format_args!("read_column_count={}", metrics.read_columns));
        push_line(&mut text, format_args!("read_columns={}", self.final_gpu.read_columns.join(",")));
        push_line(&mut text, format_args!("fallback_candidate_count={}", planning.candidate_count));
        push_line(&mut text, format_args!("fallback_rewritten_count={}", planning.rewritten_count));
        push_line(&mut text, format_args!("fallback_rejected_count={}", planning.rejected_count));
        push_line(&mut text, format_args!("gpu_plan={}", self.final_gpu.plan_one_line));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

#[derive(Debug, Clone)]
struct CollectMeasurement {
    wall_time: Duration,
    output_batches: usize,
    output_rows: usize,
    output_bytes: usize,
    batches: Vec<RecordBatch>,
}

#[derive(Debug, Clone)]
struct GpuCollectMeasurement {
    wall_time: Duration,
    output_batches: usize,
    output_rows: usize,
    output_bytes: usize,
    metrics: LargerIslandBenchmarkMetrics,
    planning: LargerIslandBenchmarkPlanning,
    plan_one_line: String,
    read_columns: Vec<String>,
    batches: Vec<RecordBatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkPath {
    CpuComparable,
    CpuDefault,
    GpuDirect,
}

impl BenchmarkPath {
    fn label(self) -> &'static str {
        match self {
            Self::CpuComparable => "cpu_comparable",
            Self::CpuDefault => "cpu_default",
            Self::GpuDirect => "gpu_direct",
        }
    }
}

async fn run_direct_parquet_benchmark(
    path: &Path,
    parquet_file_bytes: u64,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<DirectParquetBenchmarkReport> {
    for _ in 0..config.warmup_iterations {
        let _ = collect_cpu_comparable_once(path, config).await?;
        let _ = collect_cpu_default_once(path, config).await?;
        let _ = collect_gpu_direct_once(path, config).await?;
    }

    let mut cpu_comparable_samples = Vec::with_capacity(config.iterations);
    let mut cpu_default_samples = Vec::with_capacity(config.iterations);
    let mut gpu_samples = Vec::with_capacity(config.iterations);
    let mut gpu_metric_samples = Vec::with_capacity(config.iterations);
    let mut final_cpu_comparable = None;
    let mut final_cpu_default = None;
    let mut final_gpu = None;

    for iteration in 0..config.iterations {
        for path_kind in measured_iteration_order(iteration) {
            match path_kind {
                BenchmarkPath::CpuComparable => {
                    let measurement = collect_cpu_comparable_once(path, config).await?;
                    cpu_comparable_samples.push(measurement.wall_time);
                    final_cpu_comparable = Some(measurement);
                },
                BenchmarkPath::CpuDefault => {
                    let measurement = collect_cpu_default_once(path, config).await?;
                    cpu_default_samples.push(measurement.wall_time);
                    final_cpu_default = Some(measurement);
                },
                BenchmarkPath::GpuDirect => {
                    let measurement = collect_gpu_direct_once(path, config).await?;
                    gpu_samples.push(measurement.wall_time);
                    gpu_metric_samples.push(measurement.metrics.clone());
                    final_gpu = Some(measurement);
                },
            }
        }
    }

    let final_cpu_comparable =
        final_cpu_comparable.expect("validated benchmark config requires measured comparable CPU iterations");
    let final_cpu_default =
        final_cpu_default.expect("validated benchmark config requires measured default CPU iterations");
    let final_gpu = final_gpu.expect("validated benchmark config requires measured GPU iterations");
    compare_outputs(&final_cpu_comparable.batches, &final_gpu.batches)?;
    compare_outputs(&final_cpu_default.batches, &final_gpu.batches)?;
    validate_gpu_direct_measurement(&final_gpu, parquet_file_bytes, config)?;

    Ok(DirectParquetBenchmarkReport {
        parquet_file_bytes,
        input_rows: config.input_rows,
        measure_columns: config.measure_columns,
        batch_size: config.batch_size,
        row_group_size: config.row_group_size,
        group_count: config.group_count,
        warmup_iterations: config.warmup_iterations,
        iterations: config.iterations,
        limit: config.limit,
        compression: config.compression,
        measured_run_order: measured_run_order_summary(config.iterations),
        cpu_comparable: TimedBenchmarkRun::from_samples(cpu_comparable_samples.iter().copied()),
        cpu_default: TimedBenchmarkRun::from_samples(cpu_default_samples.iter().copied()),
        gpu: TimedBenchmarkRun::from_samples(gpu_samples.iter().copied()),
        cpu_comparable_samples,
        cpu_default_samples,
        gpu_samples,
        gpu_metric_samples,
        final_cpu_comparable,
        final_cpu_default,
        final_gpu,
        source_diagnostic,
    })
}

async fn collect_cpu_comparable_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<CollectMeasurement> {
    let plan = build_cpu_comparable_plan(path, config).await?;
    collect_cpu_plan(plan).await
}

async fn collect_cpu_default_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<CollectMeasurement> {
    let plan = build_cpu_default_plan(path, config).await?;
    collect_cpu_plan(plan).await
}

async fn collect_cpu_plan(plan: Arc<dyn ExecutionPlan>) -> DataFusionResult<CollectMeasurement> {
    let started = Instant::now();
    let batches = collect(plan, Arc::new(TaskContext::default())).await?;
    Ok(measure_cpu_output(started.elapsed(), batches))
}

async fn collect_gpu_direct_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let optimizer = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly).with_min_operators_per_island(2);
    let source = parquet_scan_plan(path).await?;
    let plan = build_gpu_friendly_plan(source, config)?;
    let optimized = optimize_plan(plan, &optimizer)?;
    let planning = LargerIslandBenchmarkPlanning::from_report(&optimized.report);
    if planning.rewritten_count != 1 {
        return Err(plan_error(format!(
            "direct Parquet benchmark expected exactly one rewritten GPU island, got {}",
            planning.rewritten_count
        )));
    }

    let gpu_island = optimized.plan.as_any().downcast_ref::<CudfGpuIslandExec>().ok_or_else(|| {
        plan_error(format!("optimized plan root was {}, expected CudfGpuIslandExec", optimized.plan.name()))
    })?;
    if gpu_island.input_display_name() != "local_parquet_scan"
        || gpu_island.input_boundary_name() != "cudf_parquet_reader"
    {
        return Err(plan_error(format!(
            "optimized GPU plan did not use direct local Parquet input: input={}, input_boundary={}",
            gpu_island.input_display_name(),
            gpu_island.input_boundary_name()
        )));
    }
    let read_columns = gpu_island.local_parquet_read_columns().unwrap_or_default();
    let plan_one_line = displayable(optimized.plan.as_ref()).one_line().to_string();

    let started = Instant::now();
    let batches = collect(Arc::clone(&optimized.plan), Arc::new(TaskContext::default())).await?;
    let wall_time = started.elapsed();
    let metrics = optimized
        .plan
        .metrics()
        .map(|metrics| LargerIslandBenchmarkMetrics::from_metrics(&metrics))
        .unwrap_or_else(|| LargerIslandBenchmarkMetrics::from_metrics(&MetricsSet::new()));
    let output = measure_output(&batches);

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        metrics,
        planning,
        plan_one_line,
        read_columns,
        batches,
    })
}

fn measure_cpu_output(wall_time: Duration, batches: Vec<RecordBatch>) -> CollectMeasurement {
    let output = measure_output(&batches);
    CollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        batches,
    }
}

#[derive(Debug, Clone, Copy)]
struct OutputShape {
    output_batches: usize,
    output_rows: usize,
    output_bytes: usize,
}

fn measure_output(batches: &[RecordBatch]) -> OutputShape {
    OutputShape {
        output_batches: batches.len(),
        output_rows: batches.iter().map(RecordBatch::num_rows).sum(),
        output_bytes: batches.iter().map(get_record_batch_memory_size).sum(),
    }
}

async fn build_cpu_comparable_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let source = parquet_scan_plan(path).await?;
    build_gpu_friendly_plan(source, config)
}

async fn build_cpu_default_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ctx = SessionContext::new();
    let path =
        path.to_str().ok_or_else(|| plan_error(format!("benchmark path is not valid UTF-8: {}", path.display())))?;
    ctx.register_parquet("bench", path, ParquetReadOptions::default()).await?;
    let df = ctx.sql(&benchmark_sql(config)).await?;
    df.create_physical_plan().await
}

async fn parquet_scan_plan(path: &Path) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ctx = SessionContext::new_with_config(benchmark_session_config());
    let path =
        path.to_str().ok_or_else(|| plan_error(format!("benchmark path is not valid UTF-8: {}", path.display())))?;
    let df = ctx.read_parquet(path, ParquetReadOptions::default()).await?;
    df.create_physical_plan().await
}

fn build_gpu_friendly_plan(
    source: Arc<dyn ExecutionPlan>,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let input_schema = source.schema();
    let predicate =
        binary(col("payload", &input_schema)?, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &input_schema)?;
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source)?);

    let filter_schema = filter.schema();
    let mut projection_exprs = Vec::with_capacity(config.projected_measure_pairs() + 1);
    projection_exprs.push(ProjectionExpr { expr: col("key", &filter_schema)?, alias: "key".to_string() });
    for pair in 0..config.projected_measure_pairs() {
        let left = measure_column_name(pair * 2);
        let right = measure_column_name(pair * 2 + 1);
        let expr = binary(col(&left, &filter_schema)?, Operator::Plus, col(&right, &filter_schema)?, &filter_schema)?;
        projection_exprs.push(ProjectionExpr { expr, alias: projected_column_name(pair) });
    }
    let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(projection_exprs, filter)?);

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema)?, "key".to_string())]);
    let mut aggregate_exprs = Vec::with_capacity(config.projected_measure_pairs());
    for pair in 0..config.projected_measure_pairs() {
        let input_name = projected_column_name(pair);
        let output_name = aggregate_column_name(pair);
        let aggregate = AggregateExprBuilder::new(sum_udaf(), vec![col(&input_name, &projection_schema)?])
            .schema(Arc::clone(&projection_schema))
            .alias(output_name)
            .build()?;
        aggregate_exprs.push(Arc::new(aggregate));
    }
    let aggregate_filter_exprs = vec![None; aggregate_exprs.len()];
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        group_by,
        aggregate_exprs,
        aggregate_filter_exprs,
        projection,
        projection_schema,
    )?);

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema)?, ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema)?, ArrowSortOptions::new(false, false)),
    ])
    .ok_or_else(|| plan_error("benchmark sort ordering cannot be empty"))?;
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));

    Ok(Arc::new(LocalLimitExec::new(sort, config.limit)))
}

fn benchmark_sql(config: &LocalParquetBenchmarkConfig) -> String {
    let aggregates = (0..config.projected_measure_pairs())
        .map(|pair| {
            format!(
                "SUM({} + {}) AS {}",
                measure_column_name(pair * 2),
                measure_column_name(pair * 2 + 1),
                aggregate_column_name(pair)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "SELECT key, {aggregates} FROM bench WHERE payload >= 10 GROUP BY key ORDER BY sum_x00 DESC, key ASC LIMIT {}",
        config.limit
    )
}

async fn local_parquet_scan_diagnostic(path: &Path) -> DataFusionResult<String> {
    let plan = parquet_scan_plan(path).await?;
    let Some(data_source) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return Ok(format!("root_plan={}", plan.name()));
    };
    let Some((file_scan, parquet_source)) = data_source.downcast_to_file_source::<ParquetSource>() else {
        return Ok("root_plan=DataSourceExec, source=not_parquet".to_string());
    };
    let ranged_files =
        file_scan.file_groups.iter().flat_map(|group| group.files().iter()).filter(|file| file.range.is_some()).count();
    let total_files = file_scan.file_groups.iter().map(|group| group.files().len()).sum::<usize>();

    Ok(format!(
        "object_store_url={}, partition_cols={}, expr_adapter={}, scan_limit={}, preserve_order={}, parquet_filter={}, file_groups={}, files={}, ranged_files={}",
        file_scan.object_store_url,
        file_scan.table_partition_cols().len(),
        file_scan.expr_adapter_factory.is_some(),
        file_scan.limit.is_some(),
        file_scan.preserve_order,
        parquet_source.filter().is_some(),
        file_scan.file_groups.len(),
        total_files,
        ranged_files,
    ))
}

fn benchmark_session_config() -> SessionConfig {
    SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1)
}

fn write_benchmark_parquet(path: &Path, config: &LocalParquetBenchmarkConfig) -> DataFusionResult<()> {
    let file = File::create(path)?;
    let schema = benchmark_schema(config.measure_columns);
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(false)
        .set_compression(config.compression.parquet())
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;
    let mut start = 0usize;

    while start < config.input_rows {
        let len = (config.input_rows - start).min(config.batch_size);
        writer.write(&benchmark_batch(Arc::clone(&schema), start, len, config)?)?;
        start += len;
    }

    writer.close()?;
    Ok(())
}

fn benchmark_schema(measure_columns: usize) -> SchemaRef {
    let mut fields = Vec::with_capacity(measure_columns + 2);
    fields.push(Field::new("key", DataType::Int32, false));
    fields.push(Field::new("payload", DataType::Int32, false));
    fields.extend((0..measure_columns).map(|index| Field::new(measure_column_name(index), DataType::Int64, false)));
    Arc::new(Schema::new(fields))
}

fn benchmark_batch(
    schema: SchemaRef,
    start: usize,
    len: usize,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<RecordBatch> {
    let mut keys = Vec::with_capacity(len);
    let mut payload = Vec::with_capacity(len);
    let mut measures = (0..config.measure_columns).map(|_| Vec::with_capacity(len)).collect::<Vec<_>>();

    for offset in 0..len {
        let row = start + offset;
        keys.push((row % config.group_count) as i32);
        payload.push((splitmix64(row as u64 ^ 0x9e37_79b9_7f4a_7c15) % 100) as i32);
        for (column_index, values) in measures.iter_mut().enumerate() {
            values.push(measure_value(row, column_index));
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(config.measure_columns + 2);
    columns.push(Arc::new(Int32Array::from(keys)));
    columns.push(Arc::new(Int32Array::from(payload)));
    columns.extend(measures.into_iter().map(|values| Arc::new(Int64Array::from(values)) as ArrayRef));
    Ok(RecordBatch::try_new(schema, columns)?)
}

fn measure_value(row: usize, column_index: usize) -> i64 {
    let row = row as u64;
    let column = column_index as u64;
    let mixed = splitmix64(row.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ column.wrapping_mul(0xa076_1d64_78bd_642f));
    ((mixed ^ row.rotate_left((column_index % 31) as u32)) & 0xffff_ffff) as i64
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn validate_file_size(file_bytes: u64, config: &LocalParquetBenchmarkConfig) -> DataFusionResult<()> {
    if file_bytes < config.minimum_file_bytes {
        return Err(plan_error(format!(
            "generated Parquet file is below benchmark minimum: parquet_file_bytes={file_bytes}, minimum_file_bytes={}. Increase CUDF_PARQUET_BENCH_ROWS, CUDF_PARQUET_BENCH_MEASURE_COLUMNS, or use CUDF_PARQUET_BENCH_COMPRESSION=uncompressed",
            config.minimum_file_bytes
        )));
    }
    Ok(())
}

fn validate_gpu_direct_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let metrics = &measurement.metrics;
    if !measurement.plan_one_line.contains("input=local_parquet_scan")
        || !measurement.plan_one_line.contains("input_boundary=cudf_parquet_reader")
    {
        return Err(plan_error(format!(
            "GPU plan did not use direct local Parquet input: {}",
            measurement.plan_one_line
        )));
    }
    if metrics.host_to_device_uploads != 0 || metrics.host_to_device_uploaded_rows != 0 {
        return Err(plan_error(format!(
            "direct GPU Parquet path crossed host-to-device upload boundary: uploads={}, uploaded_rows={}",
            metrics.host_to_device_uploads, metrics.host_to_device_uploaded_rows
        )));
    }
    if metrics.parquet_files != 1 {
        return Err(plan_error(format!("direct GPU Parquet path expected one file, got {}", metrics.parquet_files)));
    }
    if metrics.parquet_bytes_known < parquet_file_bytes as usize
        || metrics.parquet_bytes_known < config.minimum_file_bytes as usize
    {
        return Err(plan_error(format!(
            "direct GPU Parquet metrics did not account for the benchmark file size: parquet_bytes_known={}, parquet_file_bytes={}, minimum_file_bytes={}",
            metrics.parquet_bytes_known, parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if metrics.device_to_host_materialized_rows > config.limit {
        return Err(plan_error(format!(
            "direct GPU Parquet path materialized too many rows: materialized_rows={}, limit={}",
            metrics.device_to_host_materialized_rows, config.limit
        )));
    }

    let mut actual = measurement.read_columns.clone();
    actual.sort();
    let expected = config.expected_read_columns();
    if actual != expected {
        return Err(plan_error(format!(
            "direct GPU Parquet read columns did not match workload requirements: actual={}, expected={}",
            actual.join(","),
            expected.join(",")
        )));
    }
    Ok(())
}

fn compare_outputs(cpu: &[RecordBatch], gpu: &[RecordBatch]) -> DataFusionResult<()> {
    let cpu_rows = cpu.iter().map(RecordBatch::num_rows).sum::<usize>();
    let gpu_rows = gpu.iter().map(RecordBatch::num_rows).sum::<usize>();
    if cpu_rows != gpu_rows {
        return Err(plan_error(format!("CPU output rows ({cpu_rows}) did not match GPU output rows ({gpu_rows})")));
    }

    let cpu_values = batch_rows(cpu)?;
    let gpu_values = batch_rows(gpu)?;
    if cpu_values != gpu_values {
        return Err(plan_error("CPU and GPU benchmark outputs differed"));
    }
    Ok(())
}

fn batch_rows(batches: &[RecordBatch]) -> DataFusionResult<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());
            for column in 0..batch.num_columns() {
                values.push(array_value_to_string(batch.column(column).as_ref(), row)?);
            }
            rows.push(values);
        }
    }
    Ok(rows)
}

fn measure_column_name(index: usize) -> String {
    format!("m{index:02}")
}

fn projected_column_name(pair: usize) -> String {
    format!("x{pair:02}")
}

fn aggregate_column_name(pair: usize) -> String {
    format!("sum_x{pair:02}")
}

fn env_usize(name: &'static str, default: usize) -> DataFusionResult<usize> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<usize>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn env_u64(name: &'static str, default: u64) -> DataFusionResult<u64> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<u64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn validate_nonzero(name: &'static str, value: usize) -> DataFusionResult<()> {
    if value == 0 {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}

fn default_benchmark_dir() -> PathBuf {
    unique_benchmark_dir(Path::new(DEFAULT_BENCHMARK_DATA_DIR))
}

fn unique_benchmark_dir(base_dir: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    base_dir.join(format!("datafusion-cudf-local-parquet-bench-{}-{nonce}", std::process::id()))
}

fn speedup(cpu: Duration, gpu: Duration) -> f64 {
    if cpu.is_zero() {
        return 0.0;
    }
    if gpu.is_zero() {
        return f64::INFINITY;
    }
    cpu.as_secs_f64() / gpu.as_secs_f64()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn measured_iteration_order(iteration: usize) -> [BenchmarkPath; 3] {
    match iteration % 3 {
        0 => [BenchmarkPath::CpuComparable, BenchmarkPath::CpuDefault, BenchmarkPath::GpuDirect],
        1 => [BenchmarkPath::GpuDirect, BenchmarkPath::CpuComparable, BenchmarkPath::CpuDefault],
        _ => [BenchmarkPath::CpuDefault, BenchmarkPath::GpuDirect, BenchmarkPath::CpuComparable],
    }
}

fn measured_run_order_summary(iterations: usize) -> String {
    (0..iterations)
        .map(|iteration| {
            measured_iteration_order(iteration).into_iter().map(BenchmarkPath::label).collect::<Vec<_>>().join(">")
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn duration_samples_csv(samples: &[Duration]) -> String {
    samples.iter().map(|duration| format!("{:.3}", duration_ms(*duration))).collect::<Vec<_>>().join(",")
}

fn metric_duration_samples_csv(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Duration,
) -> String {
    samples.iter().map(|sample| format!("{:.3}", duration_ms(metric(sample)))).collect::<Vec<_>>().join(",")
}

fn mean_metric_duration(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Duration,
) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let nanos = samples.iter().map(|sample| metric(sample).as_nanos()).sum::<u128>() / samples.len() as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

fn push_line(text: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write as _;

    let _ = text.write_fmt(args);
    text.push('\n');
}

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}
