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
use cudf_query_engine::{
    ExecutionContext as NativeExecutionContext, MemoryControlPolicy as NativeMemoryControlPolicy,
    PlanNode as NativePlanNode, QueryError as NativeQueryError, QueryErrorCode as NativeQueryErrorCode,
    QueryMetrics as NativeQueryMetrics, QueryOutput as NativeQueryOutput, QueryPlan as NativeQueryPlan,
    SourceNode as NativeSourceNode, execute_query_plan as execute_native_query_plan,
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
    parquet::{
        arrow::ArrowWriter,
        basic::Compression,
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
    },
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
    CandidateOutcome, CudfGpuIslandExec, CudfOptimizerConfig, GpuFallbackPolicy, LargerIslandBenchmarkMetrics,
    LargerIslandBenchmarkPlanning, TimedBenchmarkRun, lower_physical_plan_to_native, optimize_plan,
};

const THREE_GIB: u64 = 3_221_225_472;
const DEFAULT_NATIVE_DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_ROW_GROUP_SIZE: usize = 1_048_576;
const DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
const DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS: usize = 4;
const DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;
const DEFAULT_MIN_NATIVE_GPU_SPEEDUP: f64 = 1.0;
const DEFAULT_BENCHMARK_DATA_DIR: &str = "/opt/dlami/nvme/data";
#[cfg(test)]
const SIX_GIB: u64 = 6 * 1024 * 1024 * 1024;
#[cfg(test)]
const TEN_GIB: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let config = LocalParquetBenchmarkConfig::from_env()?;
    let dir = config.output_dir.clone().unwrap_or_else(default_benchmark_dir);
    fs::create_dir_all(&dir)?;
    let path = benchmark_input_path(&dir, &config);

    write_benchmark_parquet_dataset(&path, &config)?;
    let file_bytes = parquet_dataset_bytes(&path)?;
    validate_file_size(file_bytes, &config)?;
    let dataset_layout = parquet_dataset_layout(&path)?;
    validate_dataset_layout(&dataset_layout, &config)?;

    let source_diagnostic = local_parquet_scan_diagnostic(&path).await?;
    println!("local_parquet_benchmark");
    println!("parquet_path={}", path.display());
    println!("source_diagnostic={source_diagnostic}");
    println!("parquet_file_bytes={file_bytes}");
    println!("input_rows={}", config.input_rows);
    println!("parquet_files={}", config.file_count);
    println!("measure_columns={}", config.measure_columns);
    println!("projected_measure_pairs={}", config.projected_measure_pairs());
    println!("group_count={}", config.group_count);
    println!("batch_size={}", config.batch_size);
    println!("row_group_size={}", config.row_group_size);
    println!("parquet_row_groups={}", dataset_layout.row_group_count);
    println!("minimum_native_gpu_speedup={:.3}", config.minimum_native_gpu_speedup);
    println!("native_memory_control_policy={}", config.native_memory_control_policy.as_str());
    println!("benchmark_paths={}", config.benchmark_paths.name());
    println!("compression={}", config.compression.name());
    println!("native_plan_shape={}", config.native_plan_shape.name());
    println!("expected_outcome={}", config.expected_outcome.name());
    println!("phase_c_decision_matrix_cases={}", phase_c_decision_matrix().len());
    println!("limit={}", config.limit);
    println!("warmups={}", config.warmup_iterations);
    println!("iterations={}", config.iterations);

    println!();
    match config.expected_outcome {
        BenchmarkExpectedOutcome::Success => {
            let report =
                run_direct_parquet_benchmark(&path, file_bytes, dataset_layout, source_diagnostic, &config).await?;
            print!("{}", report.to_text());
        },
        BenchmarkExpectedOutcome::ResourceFailure => {
            let report =
                run_expected_resource_failure_probe(&path, file_bytes, dataset_layout, source_diagnostic, &config)
                    .await?;
            print!("{}", report.to_text());
        },
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct LocalParquetBenchmarkConfig {
    input_rows: usize,
    file_count: usize,
    measure_columns: usize,
    batch_size: usize,
    row_group_size: usize,
    group_count: usize,
    limit: usize,
    warmup_iterations: usize,
    iterations: usize,
    minimum_file_bytes: u64,
    minimum_native_gpu_speedup: f64,
    native_memory_control_policy: NativeMemoryControlPolicy,
    benchmark_paths: BenchmarkPathSet,
    compression: BenchmarkCompression,
    native_plan_shape: BenchmarkPlanShape,
    expected_outcome: BenchmarkExpectedOutcome,
    output_dir: Option<PathBuf>,
}

impl LocalParquetBenchmarkConfig {
    fn from_env() -> DataFusionResult<Self> {
        let config = Self {
            input_rows: env_usize("CUDF_PARQUET_BENCH_ROWS", 12_250_000)?,
            file_count: env_usize("CUDF_PARQUET_BENCH_FILES", 1)?,
            measure_columns: env_usize("CUDF_PARQUET_BENCH_MEASURE_COLUMNS", 32)?,
            batch_size: env_usize("CUDF_PARQUET_BENCH_BATCH_SIZE", 65_536)?,
            row_group_size: env_usize("CUDF_PARQUET_BENCH_ROW_GROUP_SIZE", DEFAULT_ROW_GROUP_SIZE)?,
            group_count: env_usize("CUDF_PARQUET_BENCH_GROUPS", 1_048_576)?,
            limit: env_usize("CUDF_PARQUET_BENCH_LIMIT", 100)?,
            warmup_iterations: env_usize("CUDF_PARQUET_BENCH_WARMUPS", 1)?,
            iterations: env_usize("CUDF_PARQUET_BENCH_ITERATIONS", 3)?,
            minimum_file_bytes: env_u64("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", THREE_GIB)?,
            minimum_native_gpu_speedup: env_f64(
                "CUDF_PARQUET_BENCH_MIN_NATIVE_GPU_SPEEDUP",
                DEFAULT_MIN_NATIVE_GPU_SPEEDUP,
            )?,
            native_memory_control_policy: native_memory_control_policy_from_env()?,
            benchmark_paths: BenchmarkPathSet::from_env("CUDF_PARQUET_BENCH_PATHS")?,
            compression: BenchmarkCompression::from_env("CUDF_PARQUET_BENCH_COMPRESSION")?,
            native_plan_shape: BenchmarkPlanShape::from_env("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE")?,
            expected_outcome: BenchmarkExpectedOutcome::from_env("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME")?,
            output_dir: std::env::var_os("CUDF_PARQUET_BENCH_DIR").map(PathBuf::from),
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("CUDF_PARQUET_BENCH_ROWS", self.input_rows)?;
        validate_nonzero("CUDF_PARQUET_BENCH_FILES", self.file_count)?;
        validate_nonzero("CUDF_PARQUET_BENCH_MEASURE_COLUMNS", self.measure_columns)?;
        validate_nonzero("CUDF_PARQUET_BENCH_BATCH_SIZE", self.batch_size)?;
        validate_nonzero("CUDF_PARQUET_BENCH_ROW_GROUP_SIZE", self.row_group_size)?;
        validate_nonzero("CUDF_PARQUET_BENCH_GROUPS", self.group_count)?;
        validate_nonzero("CUDF_PARQUET_BENCH_LIMIT", self.limit)?;
        validate_nonzero("CUDF_PARQUET_BENCH_ITERATIONS", self.iterations)?;
        validate_positive_f64("CUDF_PARQUET_BENCH_MIN_NATIVE_GPU_SPEEDUP", self.minimum_native_gpu_speedup)?;
        if !self.measure_columns.is_multiple_of(2) {
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
        validate_native_memory_control_policy(self.native_memory_control_policy)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkPlanShape {
    DefaultSql,
    DirectFriendly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkExpectedOutcome {
    Success,
    ResourceFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkSourceLayout {
    SingleFileMultiRowGroup,
    MultiFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkScalePoint {
    ThreeGiB,
    SixGiB,
    TenGiB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PhaseCMatrixCase {
    scale: BenchmarkScalePoint,
    compression: BenchmarkCompression,
    source_layout: BenchmarkSourceLayout,
    native_plan_shape: BenchmarkPlanShape,
    expected_outcome: BenchmarkExpectedOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkPathSet {
    All,
    NativeOnly,
}

impl BenchmarkPathSet {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::All);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "native_only" | "native-only" => Ok(Self::NativeOnly),
            other => Err(plan_error(format!("invalid {name}: {other}; expected all or native_only"))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::NativeOnly => "native_only",
        }
    }

    fn includes_gpu_direct(self) -> bool {
        matches!(self, Self::All)
    }
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

impl BenchmarkPlanShape {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::DefaultSql);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "default_sql" | "default-sql" => Ok(Self::DefaultSql),
            "direct_friendly" | "direct-friendly" => Ok(Self::DirectFriendly),
            other => Err(plan_error(format!("invalid {name}: {other}; expected default_sql or direct_friendly"))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::DefaultSql => "default_sql",
            Self::DirectFriendly => "direct_friendly",
        }
    }
}

impl BenchmarkExpectedOutcome {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::Success);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "success" => Ok(Self::Success),
            "resource_failure" | "resource-failure" => Ok(Self::ResourceFailure),
            other => Err(plan_error(format!("invalid {name}: {other}; expected success or resource_failure"))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::ResourceFailure => "resource_failure",
        }
    }
}

impl BenchmarkSourceLayout {
    #[cfg(test)]
    fn name(self) -> &'static str {
        match self {
            Self::SingleFileMultiRowGroup => "single_file_multi_row_group",
            Self::MultiFile => "multi_file",
        }
    }

    #[cfg(test)]
    fn file_count(self) -> usize {
        match self {
            Self::SingleFileMultiRowGroup => 1,
            Self::MultiFile => 4,
        }
    }
}

impl BenchmarkScalePoint {
    #[cfg(test)]
    fn name(self) -> &'static str {
        match self {
            Self::ThreeGiB => "3gib",
            Self::SixGiB => "6gib",
            Self::TenGiB => "10gib",
        }
    }

    #[cfg(test)]
    fn input_rows(self) -> usize {
        match self {
            Self::ThreeGiB => 12_250_000,
            Self::SixGiB => 24_500_000,
            Self::TenGiB => 40_900_000,
        }
    }

    #[cfg(test)]
    fn minimum_file_bytes(self) -> u64 {
        match self {
            Self::ThreeGiB => THREE_GIB,
            Self::SixGiB => SIX_GIB,
            Self::TenGiB => TEN_GIB,
        }
    }
}

impl PhaseCMatrixCase {
    #[cfg(test)]
    fn name(self) -> String {
        format!(
            "{}-{}-{}-{}-{}",
            self.scale.name(),
            self.compression.name(),
            self.source_layout.name(),
            self.native_plan_shape.name(),
            self.expected_outcome.name()
        )
    }

    #[cfg(test)]
    fn env_overrides(self) -> Vec<(&'static str, String)> {
        vec![
            ("CUDF_PARQUET_BENCH_ROWS", self.scale.input_rows().to_string()),
            ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", self.scale.minimum_file_bytes().to_string()),
            ("CUDF_PARQUET_BENCH_COMPRESSION", self.compression.name().to_string()),
            ("CUDF_PARQUET_BENCH_FILES", self.source_layout.file_count().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", self.native_plan_shape.name().to_string()),
            ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", self.expected_outcome.name().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
        ]
    }
}

fn phase_c_decision_matrix() -> Vec<PhaseCMatrixCase> {
    let mut cases = Vec::new();
    let compressions = [BenchmarkCompression::Uncompressed, BenchmarkCompression::Snappy];
    let layouts = [BenchmarkSourceLayout::SingleFileMultiRowGroup, BenchmarkSourceLayout::MultiFile];
    let plan_shapes = [BenchmarkPlanShape::DefaultSql, BenchmarkPlanShape::DirectFriendly];

    for scale in [BenchmarkScalePoint::ThreeGiB, BenchmarkScalePoint::SixGiB] {
        for compression in compressions {
            for source_layout in layouts {
                for native_plan_shape in plan_shapes {
                    cases.push(PhaseCMatrixCase {
                        scale,
                        compression,
                        source_layout,
                        native_plan_shape,
                        expected_outcome: BenchmarkExpectedOutcome::Success,
                    });
                }
            }
        }
    }

    for compression in compressions {
        for source_layout in layouts {
            for native_plan_shape in plan_shapes {
                cases.push(PhaseCMatrixCase {
                    scale: BenchmarkScalePoint::TenGiB,
                    compression,
                    source_layout,
                    native_plan_shape,
                    expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
                });
            }
        }
    }

    cases
}

#[derive(Debug)]
struct DirectParquetBenchmarkReport {
    parquet_file_bytes: u64,
    parquet_row_groups: usize,
    input_rows: usize,
    file_count: usize,
    measure_columns: usize,
    batch_size: usize,
    row_group_size: usize,
    group_count: usize,
    warmup_iterations: usize,
    iterations: usize,
    limit: usize,
    minimum_native_gpu_speedup: f64,
    benchmark_paths: BenchmarkPathSet,
    compression: BenchmarkCompression,
    native_plan_shape: BenchmarkPlanShape,
    expected_outcome: BenchmarkExpectedOutcome,
    measured_run_order: String,
    cpu_comparable: TimedBenchmarkRun,
    cpu_default: TimedBenchmarkRun,
    gpu: Option<TimedBenchmarkRun>,
    native_gpu: TimedBenchmarkRun,
    cpu_comparable_samples: Vec<Duration>,
    cpu_default_samples: Vec<Duration>,
    gpu_samples: Vec<Duration>,
    native_gpu_samples: Vec<Duration>,
    gpu_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    native_gpu_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    final_cpu_comparable: CollectMeasurement,
    final_cpu_default: CollectMeasurement,
    final_gpu: Option<GpuCollectMeasurement>,
    final_native_gpu: GpuCollectMeasurement,
    source_diagnostic: String,
}

impl DirectParquetBenchmarkReport {
    fn to_text(&self) -> String {
        let comparable_speedup = self.gpu.as_ref().map(|gpu| speedup(self.cpu_comparable.mean, gpu.mean));
        let default_speedup = self.gpu.as_ref().map(|gpu| speedup(self.cpu_default.mean, gpu.mean));
        let native_comparable_speedup = speedup(self.cpu_comparable.mean, self.native_gpu.mean);
        let native_default_speedup = speedup(self.cpu_default.mean, self.native_gpu.mean);
        let native_metrics = &self.final_native_gpu.metrics;
        let native_planning = &self.final_native_gpu.planning;
        let mut text = String::new();

        push_line(&mut text, format_args!("case=gpu_friendly_local_parquet"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("parquet_row_groups={}", self.parquet_row_groups));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("configured_parquet_files={}", self.file_count));
        push_line(&mut text, format_args!("measure_columns={}", self.measure_columns));
        push_line(&mut text, format_args!("batch_size={}", self.batch_size));
        push_line(&mut text, format_args!("row_group_size={}", self.row_group_size));
        push_line(&mut text, format_args!("group_count={}", self.group_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("warmups={}", self.warmup_iterations));
        push_line(&mut text, format_args!("iterations={}", self.iterations));
        push_line(&mut text, format_args!("minimum_native_gpu_speedup={:.3}", self.minimum_native_gpu_speedup));
        push_line(&mut text, format_args!("benchmark_paths={}", self.benchmark_paths.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("phase_c_decision_matrix_cases={}", phase_c_decision_matrix().len()));
        push_line(&mut text, format_args!("phase_b_runtime_gate=pass"));
        push_line(
            &mut text,
            format_args!("phase_c_memory_control_policy={}", native_metrics.phase_c_memory_control_policy),
        );
        push_line(&mut text, format_args!("phase_c_runtime_gate={}", native_metrics.phase_c_runtime_gate));
        push_line(&mut text, format_args!("measured_run_order={}", self.measured_run_order));
        push_line(&mut text, format_args!("cpu_comparable_wall_ms={:.3}", duration_ms(self.cpu_comparable.mean)));
        push_line(&mut text, format_args!("cpu_wall_ms={:.3}", duration_ms(self.cpu_comparable.mean)));
        push_line(&mut text, format_args!("cpu_default_wall_ms={:.3}", duration_ms(self.cpu_default.mean)));
        if let Some(gpu) = &self.gpu {
            push_line(&mut text, format_args!("gpu_wall_ms={:.3}", duration_ms(gpu.mean)));
        } else {
            push_line(&mut text, format_args!("gpu_wall_ms=not_run"));
        }
        push_line(&mut text, format_args!("native_gpu_wall_ms={:.3}", duration_ms(self.native_gpu.mean)));
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_comparable={}", optional_f64_text(comparable_speedup)));
        push_line(&mut text, format_args!("gpu_speedup={}", optional_f64_text(comparable_speedup)));
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_default={}", optional_f64_text(default_speedup)));
        push_line(&mut text, format_args!("native_gpu_speedup_vs_cpu_comparable={native_comparable_speedup:.3}"));
        push_line(&mut text, format_args!("native_gpu_speedup={native_comparable_speedup:.3}"));
        push_line(&mut text, format_args!("native_gpu_speedup_vs_cpu_default={native_default_speedup:.3}"));
        push_line(
            &mut text,
            format_args!("cpu_comparable_wall_ms_samples={}", duration_samples_csv(&self.cpu_comparable_samples)),
        );
        push_line(
            &mut text,
            format_args!("cpu_default_wall_ms_samples={}", duration_samples_csv(&self.cpu_default_samples)),
        );
        push_line(&mut text, format_args!("gpu_wall_ms_samples={}", optional_duration_samples_csv(&self.gpu_samples)));
        push_line(
            &mut text,
            format_args!("native_gpu_wall_ms_samples={}", duration_samples_csv(&self.native_gpu_samples)),
        );
        if let Some(final_gpu) = &self.final_gpu {
            push_line(&mut text, format_args!("output_rows={}", final_gpu.output_rows));
            push_line(&mut text, format_args!("output_batches={}", final_gpu.output_batches));
            push_line(&mut text, format_args!("output_bytes={}", final_gpu.output_bytes));
        } else {
            push_line(&mut text, format_args!("output_rows=not_run"));
            push_line(&mut text, format_args!("output_batches=not_run"));
            push_line(&mut text, format_args!("output_bytes=not_run"));
        }
        push_line(&mut text, format_args!("native_output_rows={}", self.final_native_gpu.output_rows));
        push_line(&mut text, format_args!("native_output_batches={}", self.final_native_gpu.output_batches));
        push_line(&mut text, format_args!("native_output_bytes={}", self.final_native_gpu.output_bytes));
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
        if let Some(final_gpu) = &self.final_gpu {
            let metrics = &final_gpu.metrics;
            let planning = &final_gpu.planning;
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
                    duration_ms(mean_metric_duration(&self.gpu_metric_samples, |metrics| {
                        metrics.record_batch_upload_time
                    }))
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
            push_line(&mut text, format_args!("row_groups_selected={}", metrics.row_groups_selected));
            push_line(&mut text, format_args!("read_column_count={}", metrics.read_columns));
            push_line(&mut text, format_args!("read_columns={}", final_gpu.read_columns.join(",")));
            push_line(&mut text, format_args!("fallback_candidate_count={}", planning.candidate_count));
            push_line(&mut text, format_args!("fallback_rewritten_count={}", planning.rewritten_count));
            push_line(&mut text, format_args!("fallback_rejected_count={}", planning.rejected_count));
            push_line(&mut text, format_args!("gpu_plan={}", final_gpu.plan_one_line));
        } else {
            push_line(&mut text, format_args!("gpu_direct_status=not_run"));
        }
        push_line(
            &mut text,
            format_args!("native_parquet_read_ms={:.3}", duration_ms(native_metrics.parquet_read_time)),
        );
        push_line(
            &mut text,
            format_args!(
                "native_parquet_read_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.native_gpu_metric_samples, |metrics| metrics.parquet_read_time))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_parquet_read_ms_samples={}",
                metric_duration_samples_csv(&self.native_gpu_metric_samples, |metrics| metrics.parquet_read_time)
            ),
        );
        push_line(
            &mut text,
            format_args!("native_cudf_operator_ms={:.3}", duration_ms(native_metrics.cudf_operator_time)),
        );
        push_line(
            &mut text,
            format_args!(
                "native_cudf_operator_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.native_gpu_metric_samples, |metrics| metrics.cudf_operator_time))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_cudf_operator_ms_samples={}",
                metric_duration_samples_csv(&self.native_gpu_metric_samples, |metrics| metrics.cudf_operator_time)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_record_batch_materialization_ms={:.3}",
                duration_ms(native_metrics.record_batch_materialization_time)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_record_batch_materialization_ms_mean={:.3}",
                duration_ms(mean_metric_duration(&self.native_gpu_metric_samples, |metrics| {
                    metrics.record_batch_materialization_time
                }))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_peak_device_memory_bytes={}",
                optional_usize_text(native_metrics.peak_device_memory_bytes)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_resident_dataframe_peak_bytes={}",
                optional_usize_text(native_metrics.resident_dataframe_peak_bytes)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_allocator_high_water_bytes={}",
                optional_usize_text(native_metrics.allocator_high_water_bytes)
            ),
        );
        push_line(
            &mut text,
            format_args!("native_reservation_requested_bytes={}", native_metrics.reservation_requested_bytes),
        );
        push_line(
            &mut text,
            format_args!("native_reservation_granted_bytes={}", native_metrics.reservation_granted_bytes),
        );
        push_line(
            &mut text,
            format_args!("native_reservation_denied_count={}", native_metrics.reservation_denied_count),
        );
        push_line(&mut text, format_args!("native_source_chunks_planned={}", native_metrics.source_chunks_planned));
        push_line(&mut text, format_args!("native_source_chunks_attempted={}", native_metrics.source_chunks_attempted));
        push_line(&mut text, format_args!("native_source_chunks_committed={}", native_metrics.source_chunks_committed));
        push_line(&mut text, format_args!("native_source_chunks_retried={}", native_metrics.source_chunks_retried));
        push_line(&mut text, format_args!("native_row_groups_planned={}", native_metrics.row_groups_planned));
        push_line(&mut text, format_args!("native_row_groups_committed={}", native_metrics.row_groups_committed));
        push_line(&mut text, format_args!("native_partial_aggregate_count={}", native_metrics.partial_aggregate_count));
        push_line(&mut text, format_args!("native_partial_aggregate_rows={}", native_metrics.partial_aggregate_rows));
        push_line(&mut text, format_args!("native_retry_count={}", native_metrics.retry_count));
        push_line(
            &mut text,
            format_args!("native_retry_resource_exhausted_count={}", native_metrics.retry_resource_exhausted_count),
        );
        push_line(
            &mut text,
            format_args!("native_retry_reservation_failure_count={}", native_metrics.retry_reservation_failure_count),
        );
        push_line(&mut text, format_args!("native_final_merge_attempts={}", native_metrics.final_merge_attempts));
        push_line(&mut text, format_args!("native_final_merge_levels={}", native_metrics.final_merge_levels));
        push_line(
            &mut text,
            format_args!(
                "native_peak_device_memory_bytes_mean={}",
                optional_usize_text(mean_metric_optional_usize(&self.native_gpu_metric_samples, |metrics| {
                    metrics.peak_device_memory_bytes
                }))
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "native_peak_device_memory_bytes_samples={}",
                metric_optional_usize_samples_csv(&self.native_gpu_metric_samples, |metrics| {
                    metrics.peak_device_memory_bytes
                })
            ),
        );
        push_line(&mut text, format_args!("native_host_to_device_uploads={}", native_metrics.host_to_device_uploads));
        push_line(
            &mut text,
            format_args!("native_host_to_device_uploaded_rows={}", native_metrics.host_to_device_uploaded_rows),
        );
        push_line(
            &mut text,
            format_args!("native_device_to_host_materializations={}", native_metrics.device_to_host_materializations),
        );
        push_line(
            &mut text,
            format_args!("native_device_to_host_materialized_rows={}", native_metrics.device_to_host_materialized_rows),
        );
        push_line(&mut text, format_args!("native_parquet_files={}", native_metrics.parquet_files));
        push_line(&mut text, format_args!("native_parquet_bytes_known={}", native_metrics.parquet_bytes_known));
        push_line(&mut text, format_args!("native_row_groups_selected={}", native_metrics.row_groups_selected));
        push_line(&mut text, format_args!("native_read_column_count={}", native_metrics.read_columns));
        push_line(&mut text, format_args!("native_read_columns={}", self.final_native_gpu.read_columns.join(",")));
        push_line(&mut text, format_args!("native_fallback_candidate_count={}", native_planning.candidate_count));
        push_line(&mut text, format_args!("native_fallback_rewritten_count={}", native_planning.rewritten_count));
        push_line(&mut text, format_args!("native_fallback_rejected_count={}", native_planning.rejected_count));
        push_line(
            &mut text,
            format_args!(
                "native_default_sql_physical_plan={}",
                self.final_native_gpu.source_physical_plan_one_line.as_deref().unwrap_or("not_available")
            ),
        );
        push_line(
            &mut text,
            format_args!("native_lowered_candidate_shapes={}", self.final_native_gpu.native_candidate_shapes.join(";")),
        );
        push_line(&mut text, format_args!("native_gpu_plan={}", self.final_native_gpu.plan_one_line));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

#[derive(Debug)]
struct ExpectedResourceFailureReport {
    parquet_file_bytes: u64,
    parquet_row_groups: usize,
    input_rows: usize,
    file_count: usize,
    compression: BenchmarkCompression,
    native_plan_shape: BenchmarkPlanShape,
    expected_outcome: BenchmarkExpectedOutcome,
    native_candidate_shapes: Vec<String>,
    source_physical_plan_one_line: String,
    source_diagnostic: String,
    error: NativeQueryError,
}

impl ExpectedResourceFailureReport {
    fn to_text(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("case=phase_c_expected_resource_failure"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("parquet_row_groups={}", self.parquet_row_groups));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("configured_parquet_files={}", self.file_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("phase_c_runtime_gate=expected_resource_failure"));
        push_line(&mut text, format_args!("no_silent_cpu_fallback=true"));
        push_line(&mut text, format_args!("native_error_code={}", self.error.code()));
        push_line(&mut text, format_args!("native_error_kind={}", self.error.kind()));
        push_line(&mut text, format_args!("native_error_status={}", self.error.status()));
        push_line(&mut text, format_args!("native_error_operation={}", self.error.operation()));
        push_line(&mut text, format_args!("native_error_facts={}", native_error_facts_tsv(&self.error)));
        push_line(
            &mut text,
            format_args!("native_lowered_candidate_shapes={}", self.native_candidate_shapes.join(";")),
        );
        push_line(&mut text, format_args!("native_default_sql_physical_plan={}", self.source_physical_plan_one_line));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

async fn run_expected_resource_failure_probe(
    path: &Path,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<ExpectedResourceFailureReport> {
    if !config.native_memory_control_policy.is_bounded() {
        return Err(plan_error(
            "expected resource-failure benchmark mode requires CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY=bounded",
        ));
    }

    let plan = build_native_physical_plan(path, config).await?;
    let source_physical_plan_one_line = displayable(plan.as_ref()).one_line().to_string();
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan));
    let native_candidate_shapes = native_candidate_shapes(&lowering.report);
    if lowering.plans.len() != 1 {
        return Err(plan_error(format!(
            "expected resource-failure benchmark mode requires one lowered native plan, got native_plan_count={}, candidates={}",
            lowering.plans.len(),
            native_candidate_shapes.join(";")
        )));
    }

    let native_plan = lowering.plans.into_iter().next().expect("validated native lowering contains one plan");
    let context = NativeExecutionContext::builder()
        .with_memory_control_policy(config.native_memory_control_policy)
        .build()
        .map_err(native_query_error)?;

    match execute_native_query_plan(&context, &native_plan) {
        Ok(_) => Err(plan_error(
            "expected resource-failure benchmark mode completed successfully; tighten the bounded memory policy or run as expected_outcome=success",
        )),
        Err(error) if is_planned_resource_failure(&error) => Ok(ExpectedResourceFailureReport {
            parquet_file_bytes,
            parquet_row_groups: dataset_layout.row_group_count,
            input_rows: config.input_rows,
            file_count: config.file_count,
            compression: config.compression,
            native_plan_shape: config.native_plan_shape,
            expected_outcome: config.expected_outcome,
            native_candidate_shapes,
            source_physical_plan_one_line,
            source_diagnostic,
            error,
        }),
        Err(error) => Err(native_query_error(error)),
    }
}

fn is_planned_resource_failure(error: &NativeQueryError) -> bool {
    matches!(error.code(), NativeQueryErrorCode::MemoryReservationFailed | NativeQueryErrorCode::CudfResourceExhausted)
}

fn native_error_facts_tsv(error: &NativeQueryError) -> String {
    let facts = error
        .facts()
        .iter()
        .map(|(key, value)| format!("{key}:{}", value.replace(['\t', '\n'], " ")))
        .collect::<Vec<_>>();
    if facts.is_empty() { "none".to_string() } else { facts.join(",") }
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
    source_physical_plan_one_line: Option<String>,
    native_candidate_shapes: Vec<String>,
    read_columns: Vec<String>,
    batches: Vec<RecordBatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParquetDatasetLayout {
    file_count: usize,
    row_group_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkPath {
    CpuComparable,
    CpuDefault,
    GpuDirect,
    NativeGpuDirect,
}

impl BenchmarkPath {
    fn label(self) -> &'static str {
        match self {
            Self::CpuComparable => "cpu_comparable",
            Self::CpuDefault => "cpu_default",
            Self::GpuDirect => "gpu_direct",
            Self::NativeGpuDirect => "native_gpu_direct",
        }
    }
}

async fn run_direct_parquet_benchmark(
    path: &Path,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<DirectParquetBenchmarkReport> {
    for _ in 0..config.warmup_iterations {
        let _ = collect_cpu_comparable_once(path, config).await?;
        let _ = collect_cpu_default_once(path, config).await?;
        if config.benchmark_paths.includes_gpu_direct() {
            let _ = collect_gpu_direct_once(path, config).await?;
        }
        let _ = collect_native_gpu_direct_once(path, config).await?;
    }

    let mut cpu_comparable_samples = Vec::with_capacity(config.iterations);
    let mut cpu_default_samples = Vec::with_capacity(config.iterations);
    let mut gpu_samples = Vec::with_capacity(config.iterations);
    let mut native_gpu_samples = Vec::with_capacity(config.iterations);
    let mut gpu_metric_samples = Vec::with_capacity(config.iterations);
    let mut native_gpu_metric_samples = Vec::with_capacity(config.iterations);
    let mut final_cpu_comparable = None;
    let mut final_cpu_default = None;
    let mut final_gpu = None;
    let mut final_native_gpu = None;

    for iteration in 0..config.iterations {
        for path_kind in measured_iteration_order(iteration, config.benchmark_paths) {
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
                BenchmarkPath::NativeGpuDirect => {
                    let measurement = collect_native_gpu_direct_once(path, config).await?;
                    native_gpu_samples.push(measurement.wall_time);
                    native_gpu_metric_samples.push(measurement.metrics.clone());
                    final_native_gpu = Some(measurement);
                },
            }
        }
    }

    let final_cpu_comparable =
        final_cpu_comparable.expect("validated benchmark config requires measured comparable CPU iterations");
    let final_cpu_default =
        final_cpu_default.expect("validated benchmark config requires measured default CPU iterations");
    let mut final_native_gpu =
        final_native_gpu.expect("validated benchmark config requires measured native GPU iterations");
    if let Some(final_gpu) = &final_gpu {
        compare_outputs(&final_cpu_comparable.batches, &final_gpu.batches)?;
        compare_outputs(&final_cpu_default.batches, &final_gpu.batches)?;
        compare_outputs(&final_gpu.batches, &final_native_gpu.batches)?;
        validate_gpu_direct_measurement(final_gpu, parquet_file_bytes, config)?;
    }
    compare_outputs(&final_cpu_comparable.batches, &final_native_gpu.batches)?;
    compare_outputs(&final_cpu_default.batches, &final_native_gpu.batches)?;
    validate_native_gpu_direct_measurement(&final_native_gpu, parquet_file_bytes, config)?;
    final_native_gpu.metrics.phase_c_runtime_gate = final_native_gpu.metrics.assess_phase_c_runtime_gate(true).code();
    let cpu_comparable = TimedBenchmarkRun::from_samples(cpu_comparable_samples.iter().copied());
    let cpu_default = TimedBenchmarkRun::from_samples(cpu_default_samples.iter().copied());
    let gpu = config
        .benchmark_paths
        .includes_gpu_direct()
        .then(|| TimedBenchmarkRun::from_samples(gpu_samples.iter().copied()));
    let native_gpu = TimedBenchmarkRun::from_samples(native_gpu_samples.iter().copied());
    validate_phase_b_runtime_gate(
        &cpu_comparable,
        &native_gpu,
        &final_native_gpu,
        parquet_file_bytes,
        dataset_layout,
        config,
    )?;
    validate_phase_c_runtime_gate(&final_native_gpu, config)?;

    Ok(DirectParquetBenchmarkReport {
        parquet_file_bytes,
        parquet_row_groups: dataset_layout.row_group_count,
        input_rows: config.input_rows,
        file_count: config.file_count,
        measure_columns: config.measure_columns,
        batch_size: config.batch_size,
        row_group_size: config.row_group_size,
        group_count: config.group_count,
        warmup_iterations: config.warmup_iterations,
        iterations: config.iterations,
        limit: config.limit,
        minimum_native_gpu_speedup: config.minimum_native_gpu_speedup,
        benchmark_paths: config.benchmark_paths,
        compression: config.compression,
        native_plan_shape: config.native_plan_shape,
        expected_outcome: config.expected_outcome,
        measured_run_order: measured_run_order_summary(config.iterations, config.benchmark_paths),
        cpu_comparable,
        cpu_default,
        gpu,
        native_gpu,
        cpu_comparable_samples,
        cpu_default_samples,
        gpu_samples,
        native_gpu_samples,
        gpu_metric_samples,
        native_gpu_metric_samples,
        final_cpu_comparable,
        final_cpu_default,
        final_gpu,
        final_native_gpu,
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
        source_physical_plan_one_line: None,
        native_candidate_shapes: Vec::new(),
        read_columns,
        batches,
    })
}

async fn collect_native_gpu_direct_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let plan = build_native_physical_plan(path, config).await?;
    let source_physical_plan_one_line = displayable(plan.as_ref()).one_line().to_string();
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan));
    let planning = LargerIslandBenchmarkPlanning::from_report(&lowering.report);
    let native_candidate_shapes = native_candidate_shapes(&lowering.report);
    if planning.rewritten_count != 1 || lowering.plans.len() != 1 {
        return Err(plan_error(format!(
            "native default SQL Parquet benchmark expected exactly one lowered GPU query plan, got rewritten_count={}, native_plan_count={}, candidates={}",
            planning.rewritten_count,
            lowering.plans.len(),
            native_candidate_shapes.join(";")
        )));
    }

    let native_plan =
        lowering.plans.into_iter().next().expect("validated native lowering result contains exactly one plan");
    let read_columns = native_local_parquet_read_columns(&native_plan)?;
    let plan_one_line = native_plan_one_line(&native_plan);

    let started = Instant::now();
    let context = NativeExecutionContext::builder()
        .with_memory_control_policy(config.native_memory_control_policy)
        .build()
        .map_err(native_query_error)?;
    let result = execute_native_query_plan(&context, &native_plan).map_err(native_query_error)?;
    let wall_time = started.elapsed();
    let (output, native_metrics) = result.into_parts();
    let NativeQueryOutput::HostArrow(batch) = output else {
        return Err(plan_error("native default SQL Parquet benchmark expected a HostArrow result sink"));
    };
    let batches = vec![batch];
    let output = measure_output(&batches);

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        metrics: benchmark_metrics_from_native(&native_metrics),
        planning,
        plan_one_line,
        source_physical_plan_one_line: Some(source_physical_plan_one_line),
        native_candidate_shapes,
        read_columns,
        batches,
    })
}

async fn build_native_physical_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    match config.native_plan_shape {
        BenchmarkPlanShape::DefaultSql => build_cpu_default_plan(path, config).await,
        BenchmarkPlanShape::DirectFriendly => build_cpu_comparable_plan(path, config).await,
    }
}

fn native_candidate_shapes(report: &datafusion_cudf::PlanningReport) -> Vec<String> {
    report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => format!("rewritten:{}", candidate.shape),
            CandidateOutcome::Rejected { candidate, reason } => {
                format!("rejected:{}:{}", candidate.shape, reason.code())
            },
        })
        .collect()
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

fn benchmark_metrics_from_native(metrics: &NativeQueryMetrics) -> LargerIslandBenchmarkMetrics {
    LargerIslandBenchmarkMetrics {
        phase_c_memory_control_policy: metrics.phase_c_memory_control_policy.as_str(),
        phase_c_runtime_gate: phase_c_runtime_gate(metrics.phase_c_memory_control_policy),
        input_batches: u64_to_usize(metrics.input_batches),
        input_rows: u64_to_usize(metrics.input_rows),
        peak_device_memory_bytes: metrics.peak_device_memory_bytes.map(u64_to_usize),
        resident_dataframe_peak_bytes: metrics.resident_dataframe_peak_bytes.map(u64_to_usize),
        allocator_high_water_bytes: metrics.allocator_high_water_bytes.map(u64_to_usize),
        reservation_requested_bytes: u64_to_usize(metrics.reservation_requested_bytes),
        reservation_granted_bytes: u64_to_usize(metrics.reservation_granted_bytes),
        reservation_denied_count: u64_to_usize(metrics.reservation_denied_count),
        source_chunks_planned: u64_to_usize(metrics.source_chunks_planned),
        source_chunks_attempted: u64_to_usize(metrics.source_chunks_attempted),
        source_chunks_committed: u64_to_usize(metrics.source_chunks_committed),
        source_chunks_retried: u64_to_usize(metrics.source_chunks_retried),
        row_groups_planned: u64_to_usize(metrics.row_groups_planned),
        row_groups_committed: u64_to_usize(metrics.row_groups_committed),
        partial_aggregate_count: u64_to_usize(metrics.partial_aggregate_count),
        partial_aggregate_rows: u64_to_usize(metrics.partial_aggregate_rows),
        final_merge_attempts: u64_to_usize(metrics.final_merge_attempts),
        final_merge_levels: u64_to_usize(metrics.final_merge_levels),
        retry_count: u64_to_usize(metrics.retry_count),
        retry_resource_exhausted_count: u64_to_usize(metrics.retry_resource_exhausted_count),
        retry_reservation_failure_count: u64_to_usize(metrics.retry_reservation_failure_count),
        host_to_device_uploads: u64_to_usize(metrics.boundary.host_to_device_uploads),
        host_to_device_uploaded_rows: u64_to_usize(metrics.boundary.host_to_device_uploaded_rows),
        host_to_device_uploaded_bytes: u64_to_usize(metrics.boundary.host_to_device_uploaded_bytes),
        device_to_host_materializations: u64_to_usize(metrics.boundary.device_to_host_materializations),
        device_to_host_materialized_rows: u64_to_usize(metrics.boundary.device_to_host_materialized_rows),
        device_to_host_materialized_bytes: u64_to_usize(metrics.boundary.device_to_host_materialized_bytes),
        parquet_files: u64_to_usize(metrics.parquet_files),
        parquet_bytes_known: u64_to_usize(metrics.parquet_bytes_known),
        row_groups_selected: u64_to_usize(metrics.row_groups_selected),
        read_columns: u64_to_usize(metrics.read_columns),
        parquet_read_time: metrics.parquet_read_time,
        record_batch_upload_time: Duration::ZERO,
        cudf_operator_time: metrics.operator_time,
        record_batch_materialization_time: metrics.materialization_time,
        elapsed_compute: None,
        output_rows: Some(u64_to_usize(metrics.boundary.device_to_host_materialized_rows)),
    }
}

fn native_local_parquet_read_columns(plan: &NativeQueryPlan) -> DataFusionResult<Vec<String>> {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            NativePlanNode::Source(NativeSourceNode::LocalParquet(source)) => Some(source.read_columns.clone()),
            _ => None,
        })
        .ok_or_else(|| plan_error("native query plan did not contain a local Parquet source"))
}

fn native_plan_one_line(plan: &NativeQueryPlan) -> String {
    plan.to_string().lines().map(str::trim).filter(|line| !line.is_empty()).collect::<Vec<_>>().join(" | ")
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

fn write_benchmark_parquet_dataset(path: &Path, config: &LocalParquetBenchmarkConfig) -> DataFusionResult<()> {
    if config.file_count == 1 {
        return write_benchmark_parquet_file(path, 0, config.input_rows, config);
    }

    fs::create_dir_all(path)?;
    let mut start = 0usize;
    for file_index in 0..config.file_count {
        let remaining_rows = config.input_rows - start;
        let remaining_files = config.file_count - file_index;
        let rows = remaining_rows.div_ceil(remaining_files);
        let file_path = path.join(format!("part-{file_index:05}.parquet"));
        write_benchmark_parquet_file(&file_path, start, rows, config)?;
        start += rows;
    }
    Ok(())
}

fn write_benchmark_parquet_file(
    path: &Path,
    start_row: usize,
    rows: usize,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let file = File::create(path)?;
    let schema = benchmark_schema(config.measure_columns);
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(false)
        .set_compression(config.compression.parquet())
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;
    let mut offset = 0usize;

    while offset < rows {
        let len = (rows - offset).min(config.batch_size);
        writer.write(&benchmark_batch(Arc::clone(&schema), start_row + offset, offset, rows, len, config)?)?;
        offset += len;
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
    file_local_start: usize,
    file_rows: usize,
    len: usize,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<RecordBatch> {
    let mut keys = Vec::with_capacity(len);
    let mut payload = Vec::with_capacity(len);
    let mut measures = (0..config.measure_columns).map(|_| Vec::with_capacity(len)).collect::<Vec<_>>();

    for offset in 0..len {
        let row = start + offset;
        let file_local_row = file_local_start + offset;
        keys.push((row % config.group_count) as i32);
        payload.push(benchmark_payload_value(file_local_row, file_rows, config));
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

fn benchmark_payload_value(file_local_row: usize, file_rows: usize, config: &LocalParquetBenchmarkConfig) -> i32 {
    let total_row_groups = file_rows.div_ceil(config.row_group_size);
    let pruned_row_groups = total_row_groups / 2;
    let row_group = file_local_row / config.row_group_size;
    if row_group < pruned_row_groups {
        (file_local_row % 10) as i32
    } else {
        10 + (splitmix64(file_local_row as u64 ^ 0x9e37_79b9_7f4a_7c15) % 90) as i32
    }
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

fn validate_dataset_layout(
    dataset_layout: &ParquetDatasetLayout,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if dataset_layout.file_count != config.file_count {
        return Err(plan_error(format!(
            "generated Parquet file count did not match benchmark config: actual_files={}, configured_files={}",
            dataset_layout.file_count, config.file_count
        )));
    }
    if dataset_layout.row_group_count <= dataset_layout.file_count {
        return Err(plan_error(format!(
            "generated Parquet dataset must contain at least one multi-row-group file for the Phase B runtime gate: files={}, row_groups={}, row_group_size={}. Lower CUDF_PARQUET_BENCH_ROW_GROUP_SIZE or increase CUDF_PARQUET_BENCH_ROWS",
            dataset_layout.file_count, dataset_layout.row_group_count, config.row_group_size
        )));
    }
    Ok(())
}

fn parquet_dataset_bytes(path: &Path) -> DataFusionResult<u64> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn parquet_dataset_layout(path: &Path) -> DataFusionResult<ParquetDatasetLayout> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(ParquetDatasetLayout { file_count: 1, row_group_count: parquet_file_row_group_count(path)? });
    }

    let mut file_count = 0usize;
    let mut row_group_count = 0usize;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            file_count += 1;
            row_group_count += parquet_file_row_group_count(&entry.path())?;
        }
    }
    Ok(ParquetDatasetLayout { file_count, row_group_count })
}

fn parquet_file_row_group_count(path: &Path) -> DataFusionResult<usize> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|error| plan_error(format!("failed to read Parquet metadata for {}: {error}", path.display())))?;
    Ok(reader.metadata().num_row_groups())
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
    if metrics.parquet_files != config.file_count {
        return Err(plan_error(format!(
            "direct GPU Parquet path expected {} files, got {}",
            config.file_count, metrics.parquet_files
        )));
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

fn validate_phase_b_runtime_gate(
    cpu_comparable: &TimedBenchmarkRun,
    native_gpu: &TimedBenchmarkRun,
    native_measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if parquet_file_bytes < config.minimum_file_bytes {
        return Err(plan_error(format!(
            "Phase B runtime gate requires a 3 GiB+ Parquet dataset: parquet_file_bytes={}, minimum_file_bytes={}",
            parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if dataset_layout.row_group_count <= dataset_layout.file_count {
        return Err(plan_error(format!(
            "Phase B runtime gate requires a multi-row-group Parquet dataset: files={}, row_groups={}",
            dataset_layout.file_count, dataset_layout.row_group_count
        )));
    }
    if native_measurement.metrics.host_to_device_uploads != 0
        || native_measurement.metrics.host_to_device_uploaded_rows != 0
    {
        return Err(plan_error(format!(
            "Phase B runtime gate requires zero native host-to-device uploads: uploads={}, uploaded_rows={}",
            native_measurement.metrics.host_to_device_uploads, native_measurement.metrics.host_to_device_uploaded_rows
        )));
    }
    if native_measurement.metrics.device_to_host_materializations != 1 {
        return Err(plan_error(format!(
            "Phase B runtime gate requires one final native host materialization: materializations={}",
            native_measurement.metrics.device_to_host_materializations
        )));
    }
    if native_measurement.metrics.row_groups_selected == 0
        || native_measurement.metrics.row_groups_selected >= dataset_layout.row_group_count
    {
        return Err(plan_error(format!(
            "Phase B runtime gate requires native row-group pruning: row_groups_selected={}, total_row_groups={}",
            native_measurement.metrics.row_groups_selected, dataset_layout.row_group_count
        )));
    }

    let native_speedup = speedup(cpu_comparable.mean, native_gpu.mean);
    if native_speedup <= config.minimum_native_gpu_speedup {
        return Err(plan_error(format!(
            "Phase B runtime gate did not preserve the native GPU win: native_gpu_speedup={native_speedup:.3}, minimum_native_gpu_speedup={:.3}, cpu_comparable_wall_ms={:.3}, native_gpu_wall_ms={:.3}",
            config.minimum_native_gpu_speedup,
            duration_ms(cpu_comparable.mean),
            duration_ms(native_gpu.mean)
        )));
    }
    Ok(())
}

fn validate_phase_c_runtime_gate(
    native_measurement: &GpuCollectMeasurement,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if !config.native_memory_control_policy.is_bounded() {
        return Ok(());
    }

    if native_measurement.metrics.phase_c_runtime_gate != "pass" {
        return Err(plan_error(format!(
            "Phase C runtime gate failed: gate={}, policy={}, source_chunks_planned={}, source_chunks_committed={}, row_groups_planned={}, row_groups_committed={}, host_to_device_uploads={}, device_to_host_materializations={}, resident_dataframe_peak_bytes={}",
            native_measurement.metrics.phase_c_runtime_gate,
            native_measurement.metrics.phase_c_memory_control_policy,
            native_measurement.metrics.source_chunks_planned,
            native_measurement.metrics.source_chunks_committed,
            native_measurement.metrics.row_groups_planned,
            native_measurement.metrics.row_groups_committed,
            native_measurement.metrics.host_to_device_uploads,
            native_measurement.metrics.device_to_host_materializations,
            optional_usize_text(native_measurement.metrics.resident_dataframe_peak_bytes),
        )));
    }

    Ok(())
}

fn validate_native_gpu_direct_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let metrics = &measurement.metrics;
    if !measurement.plan_one_line.contains("LocalParquetSource")
        || !measurement.plan_one_line.contains("format=Parquet")
        || !measurement.plan_one_line.contains("direct_device_read=true")
        || !measurement.plan_one_line.contains("sink=host_arrow")
    {
        return Err(plan_error(format!(
            "native default SQL plan did not lower to the direct local Parquet host-arrow path: {}",
            measurement.plan_one_line
        )));
    }
    if metrics.host_to_device_uploads != 0 || metrics.host_to_device_uploaded_rows != 0 {
        return Err(plan_error(format!(
            "native default SQL direct Parquet path crossed host-to-device upload boundary: uploads={}, uploaded_rows={}",
            metrics.host_to_device_uploads, metrics.host_to_device_uploaded_rows
        )));
    }
    if metrics.device_to_host_materializations != 1 {
        return Err(plan_error(format!(
            "native default SQL direct Parquet path expected one final host materialization, got {}",
            metrics.device_to_host_materializations
        )));
    }
    if metrics.device_to_host_materialized_rows != measurement.output_rows {
        return Err(plan_error(format!(
            "native materialized row metrics did not match output rows: materialized_rows={}, output_rows={}",
            metrics.device_to_host_materialized_rows, measurement.output_rows
        )));
    }
    let Some(peak_device_memory_bytes) = metrics.peak_device_memory_bytes else {
        return Err(plan_error("native default SQL direct Parquet path did not record peak device memory"));
    };
    if peak_device_memory_bytes == 0 {
        return Err(plan_error("native default SQL direct Parquet path recorded zero peak device memory"));
    }
    if measurement.output_rows > config.limit {
        return Err(plan_error(format!(
            "native default SQL direct Parquet path returned too many rows: output_rows={}, limit={}",
            measurement.output_rows, config.limit
        )));
    }
    if metrics.parquet_files != config.file_count {
        return Err(plan_error(format!(
            "native default SQL direct Parquet path expected {} files, got {}",
            config.file_count, metrics.parquet_files
        )));
    }
    if metrics.parquet_bytes_known < parquet_file_bytes as usize
        || metrics.parquet_bytes_known < config.minimum_file_bytes as usize
    {
        return Err(plan_error(format!(
            "native default SQL direct Parquet metrics did not account for the benchmark file size: parquet_bytes_known={}, parquet_file_bytes={}, minimum_file_bytes={}",
            metrics.parquet_bytes_known, parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if measurement.planning.rejected_count != 0 {
        return Err(plan_error(format!(
            "native default SQL Parquet lowering rejected fallback candidates: rejected_count={}",
            measurement.planning.rejected_count
        )));
    }

    let mut actual = measurement.read_columns.clone();
    actual.sort();
    let expected = config.expected_read_columns();
    if actual != expected {
        return Err(plan_error(format!(
            "native default SQL direct Parquet read columns did not match workload requirements: actual={}, expected={}",
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

fn env_f64(name: &'static str, default: f64) -> DataFusionResult<f64> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<f64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

fn native_memory_control_policy_from_env() -> DataFusionResult<NativeMemoryControlPolicy> {
    let mode = std::env::var_os("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY")
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_else(|| "bounded".to_string());

    match mode.as_str() {
        "bounded" => Ok(NativeMemoryControlPolicy::Bounded {
            device_budget_bytes: env_u64(
                "CUDF_PARQUET_BENCH_NATIVE_DEVICE_BUDGET_BYTES",
                DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
            )?,
            max_source_chunk_bytes: env_u64(
                "CUDF_PARQUET_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES",
                DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            )?,
            max_row_groups_per_chunk: env_usize(
                "CUDF_PARQUET_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK",
                DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            )?,
            max_retry_attempts: env_usize(
                "CUDF_PARQUET_BENCH_NATIVE_MAX_RETRY_ATTEMPTS",
                DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            )?,
            min_retry_chunk_row_groups: env_usize(
                "CUDF_PARQUET_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS",
                DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            )?,
        }),
        "unbounded" => Ok(NativeMemoryControlPolicy::Unbounded),
        other => Err(plan_error(format!(
            "invalid CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY: {other}; expected bounded or unbounded"
        ))),
    }
}

fn validate_nonzero(name: &'static str, value: usize) -> DataFusionResult<()> {
    if value == 0 {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}

fn validate_native_memory_control_policy(policy: NativeMemoryControlPolicy) -> DataFusionResult<()> {
    policy.validate().map_err(native_query_error)
}

fn validate_positive_f64(name: &'static str, value: f64) -> DataFusionResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(plan_error(format!("{name} must be finite and greater than zero, got {value}")));
    }
    Ok(())
}

fn default_benchmark_dir() -> PathBuf {
    unique_benchmark_dir(Path::new(DEFAULT_BENCHMARK_DATA_DIR))
}

fn benchmark_input_path(dir: &Path, config: &LocalParquetBenchmarkConfig) -> PathBuf {
    if config.file_count == 1 { dir.join("wide.parquet") } else { dir.join("wide_parquet_dataset") }
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

fn u64_to_usize(value: u64) -> usize {
    value.min(usize::MAX as u64) as usize
}

fn measured_iteration_order(iteration: usize, path_set: BenchmarkPathSet) -> Vec<BenchmarkPath> {
    match path_set {
        BenchmarkPathSet::NativeOnly => match iteration % 2 {
            0 => vec![BenchmarkPath::CpuComparable, BenchmarkPath::CpuDefault, BenchmarkPath::NativeGpuDirect],
            _ => vec![BenchmarkPath::NativeGpuDirect, BenchmarkPath::CpuDefault, BenchmarkPath::CpuComparable],
        },
        BenchmarkPathSet::All => match iteration % 4 {
            0 => vec![
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::GpuDirect,
                BenchmarkPath::NativeGpuDirect,
            ],
            1 => vec![
                BenchmarkPath::NativeGpuDirect,
                BenchmarkPath::GpuDirect,
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
            ],
            2 => vec![
                BenchmarkPath::GpuDirect,
                BenchmarkPath::NativeGpuDirect,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
            ],
            _ => vec![
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
                BenchmarkPath::NativeGpuDirect,
                BenchmarkPath::GpuDirect,
            ],
        },
    }
}

fn measured_run_order_summary(iterations: usize, path_set: BenchmarkPathSet) -> String {
    (0..iterations)
        .map(|iteration| {
            measured_iteration_order(iteration, path_set)
                .into_iter()
                .map(BenchmarkPath::label)
                .collect::<Vec<_>>()
                .join(">")
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn duration_samples_csv(samples: &[Duration]) -> String {
    samples.iter().map(|duration| format!("{:.3}", duration_ms(*duration))).collect::<Vec<_>>().join(",")
}

fn optional_duration_samples_csv(samples: &[Duration]) -> String {
    if samples.is_empty() { "not_run".to_string() } else { duration_samples_csv(samples) }
}

fn optional_f64_text(value: Option<f64>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| format!("{value:.3}"))
}

fn metric_duration_samples_csv(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Duration,
) -> String {
    samples.iter().map(|sample| format!("{:.3}", duration_ms(metric(sample)))).collect::<Vec<_>>().join(",")
}

fn metric_optional_usize_samples_csv(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Option<usize>,
) -> String {
    samples.iter().map(|sample| optional_usize_text(metric(sample))).collect::<Vec<_>>().join(",")
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

fn mean_metric_optional_usize(
    samples: &[LargerIslandBenchmarkMetrics],
    metric: impl Fn(&LargerIslandBenchmarkMetrics) -> Option<usize>,
) -> Option<usize> {
    let mut total = 0u128;
    let mut count = 0u128;
    for sample in samples {
        let value = metric(sample)?;
        total += value as u128;
        count += 1;
    }
    if count == 0 { None } else { Some((total / count).min(usize::MAX as u128) as usize) }
}

fn optional_usize_text(value: Option<usize>) -> String {
    value.map_or_else(|| "unavailable".to_string(), |value| value.to_string())
}

fn phase_c_runtime_gate(_policy: NativeMemoryControlPolicy) -> &'static str {
    "not_evaluated"
}

fn push_line(text: &mut String, args: std::fmt::Arguments<'_>) {
    use std::fmt::Write as _;

    let _ = text.write_fmt(args);
    text.push('\n');
}

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}

fn native_query_error(error: cudf_query_engine::QueryError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_c_decision_matrix_covers_required_dimensions() {
        let cases = phase_c_decision_matrix();

        assert!(cases.iter().any(|case| case.scale == BenchmarkScalePoint::ThreeGiB));
        assert!(cases.iter().any(|case| case.scale == BenchmarkScalePoint::SixGiB));
        assert!(cases.iter().any(|case| case.scale == BenchmarkScalePoint::TenGiB));
        assert!(cases.iter().any(|case| case.compression == BenchmarkCompression::Uncompressed));
        assert!(cases.iter().any(|case| case.compression == BenchmarkCompression::Snappy));
        assert!(cases.iter().any(|case| case.source_layout == BenchmarkSourceLayout::SingleFileMultiRowGroup));
        assert!(cases.iter().any(|case| case.source_layout == BenchmarkSourceLayout::MultiFile));
        assert!(cases.iter().any(|case| case.native_plan_shape == BenchmarkPlanShape::DefaultSql));
        assert!(cases.iter().any(|case| case.native_plan_shape == BenchmarkPlanShape::DirectFriendly));
        assert!(cases.iter().any(|case| case.expected_outcome == BenchmarkExpectedOutcome::Success));
        assert!(cases.iter().any(|case| case.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure));

        let success_cases =
            cases.iter().filter(|case| case.expected_outcome == BenchmarkExpectedOutcome::Success).collect::<Vec<_>>();
        assert_eq!(success_cases.len(), 16);
        assert!(success_cases.iter().all(|case| case.scale != BenchmarkScalePoint::TenGiB));

        let expected_failure_cases = cases
            .iter()
            .filter(|case| case.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure)
            .collect::<Vec<_>>();
        assert_eq!(expected_failure_cases.len(), 8);
        assert!(expected_failure_cases.iter().all(|case| case.scale == BenchmarkScalePoint::TenGiB));
    }

    #[test]
    fn phase_c_matrix_case_names_and_env_overrides_are_stable() {
        let case = PhaseCMatrixCase {
            scale: BenchmarkScalePoint::TenGiB,
            compression: BenchmarkCompression::Snappy,
            source_layout: BenchmarkSourceLayout::MultiFile,
            native_plan_shape: BenchmarkPlanShape::DirectFriendly,
            expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
        };

        assert_eq!(case.name(), "10gib-snappy-multi_file-direct_friendly-resource_failure");
        assert_eq!(
            case.env_overrides(),
            vec![
                ("CUDF_PARQUET_BENCH_ROWS", "40900000".to_string()),
                ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", TEN_GIB.to_string()),
                ("CUDF_PARQUET_BENCH_COMPRESSION", "snappy".to_string()),
                ("CUDF_PARQUET_BENCH_FILES", "4".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", "direct_friendly".to_string()),
                ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", "resource_failure".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
            ]
        );
    }

    #[test]
    fn expected_resource_failure_report_includes_resource_error_facts() {
        let error = NativeQueryError::new(
            NativeQueryErrorCode::MemoryReservationFailed,
            "cudf_query_engine.memory.reserve",
            "memory reservation failed before native cudf execution",
        )
        .with_fact("memory_boundary", "final_sort")
        .with_fact("stage", "final_sort")
        .with_fact("attempt", 1)
        .with_fact("row_groups", "0,1")
        .with_fact("resume_boundary", "final_sort")
        .with_fact("committed_chunks", 2);
        let report = ExpectedResourceFailureReport {
            parquet_file_bytes: TEN_GIB,
            parquet_row_groups: 32,
            input_rows: 1,
            file_count: 1,
            compression: BenchmarkCompression::Uncompressed,
            native_plan_shape: BenchmarkPlanShape::DefaultSql,
            expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
            native_candidate_shapes: vec!["rewritten:LocalLimitExec(SortExec(...))".to_string()],
            source_physical_plan_one_line: "SortExec(fetch=100)".to_string(),
            source_diagnostic: "files=1".to_string(),
            error,
        };

        let text = report.to_text();

        assert!(text.contains("case=phase_c_expected_resource_failure"));
        assert!(text.contains("phase_c_runtime_gate=expected_resource_failure"));
        assert!(text.contains("no_silent_cpu_fallback=true"));
        assert!(text.contains("native_error_code=memory_reservation_failed"));
        assert!(text.contains("native_error_kind=resource_exhausted"));
        assert!(text.contains("native_error_status=temporary"));
        assert!(text.contains("native_error_facts="));
        assert!(text.contains("memory_boundary:final_sort"));
        assert!(text.contains("stage:final_sort"));
        assert!(text.contains("attempt:1"));
        assert!(text.contains("row_groups:0,1"));
        assert!(text.contains("resume_boundary:final_sort"));
        assert!(text.contains("committed_chunks:2"));
    }

    #[test]
    fn benchmark_payload_pruning_pattern_is_file_local() {
        let config = test_config();
        let file_rows = config.row_group_size * 4;

        assert!(benchmark_payload_value(0, file_rows, &config) < 10);
        assert!(benchmark_payload_value(config.row_group_size - 1, file_rows, &config) < 10);
        assert!(benchmark_payload_value(config.row_group_size, file_rows, &config) < 10);
        assert!(benchmark_payload_value((config.row_group_size * 2) - 1, file_rows, &config) < 10);
        assert!(benchmark_payload_value(config.row_group_size * 2, file_rows, &config) >= 10);
        assert!(benchmark_payload_value((config.row_group_size * 4) - 1, file_rows, &config) >= 10);
    }

    fn test_config() -> LocalParquetBenchmarkConfig {
        LocalParquetBenchmarkConfig {
            input_rows: 16,
            file_count: 4,
            measure_columns: 2,
            batch_size: 4,
            row_group_size: 4,
            group_count: 4,
            limit: 2,
            warmup_iterations: 0,
            iterations: 1,
            minimum_file_bytes: 1,
            minimum_native_gpu_speedup: 1.0,
            native_memory_control_policy: NativeMemoryControlPolicy::Bounded {
                device_budget_bytes: DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
                max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
                max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
                max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
                min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            },
            benchmark_paths: BenchmarkPathSet::NativeOnly,
            compression: BenchmarkCompression::Uncompressed,
            native_plan_shape: BenchmarkPlanShape::DefaultSql,
            expected_outcome: BenchmarkExpectedOutcome::Success,
            output_dir: None,
        }
    }
}
