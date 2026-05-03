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
    PlanNode as NativePlanNode, QueryMetrics as NativeQueryMetrics, QueryOutput as NativeQueryOutput,
    QueryPlan as NativeQueryPlan, SourceNode as NativeSourceNode, execute_query_plan as execute_native_query_plan,
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, ScalarValue, utils::memory::get_record_batch_memory_size},
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSourceExec,
    },
    execution::{SessionStateBuilder, context::TaskContext},
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
    CandidateOutcome, CudfGpuIslandExec, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
    CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt, Error as AdapterError, GpuFallbackPolicy,
    LargerIslandBenchmarkMetrics, LargerIslandBenchmarkPlanning, NativeCudfExec, NativeCudfExecutionConfig,
    NativeDataFusionPlanningReport, TimedBenchmarkRun, lower_physical_plan_to_native, optimize_plan,
};

const THREE_GIB: u64 = 3_221_225_472;
const DEFAULT_NATIVE_DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_ROW_GROUP_SIZE: usize = 1_048_576;
const DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
const DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS: usize = 4;
const DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;
const DEFAULT_MIN_NATIVE_GPU_SPEEDUP: f64 = 1.0;
const DEFAULT_BENCHMARK_DATA_DIR: &str = "test/data/local-parquet-benchmark";
const DEFAULT_PHASE_C_MATRIX_DATA_DIR: &str = "test/data/phase-c-matrix";
const SIX_GIB: u64 = 6 * 1024 * 1024 * 1024;
const TEN_GIB: u64 = 10 * 1024 * 1024 * 1024;

#[tokio::main]
async fn main() -> DataFusionResult<()> {
    let run_mode = BenchmarkRunMode::from_env("CUDF_PARQUET_BENCH_RUN")?;
    let config = LocalParquetBenchmarkConfig::from_env()?;

    match run_mode {
        BenchmarkRunMode::SingleCase => {
            let report = run_configured_benchmark_case(&config).await?;
            print!("{}", benchmark_case_prologue(&report, &config));
            println!();
            print!("{}", report.report.to_text());
        },
        BenchmarkRunMode::PhaseCMatrix => {
            let report = run_phase_c_matrix(&config).await?;
            print!("{}", report.to_text());
            if !report.decision_gate_pass() {
                return Err(plan_error(format!(
                    "Phase C DataFusion native exec matrix decision gate failed: unexpected_outcomes={}",
                    report.unexpected_outcome_count()
                )));
            }
        },
    }

    Ok(())
}

#[derive(Debug)]
struct ConfiguredBenchmarkCaseReport {
    path: PathBuf,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    report: BenchmarkCaseReport,
}

#[derive(Debug)]
enum BenchmarkCaseReport {
    Success(DirectParquetBenchmarkReport),
    PlanningRejected(ExpectedPlanningRejectedReport),
    ResourceFailure(ExpectedResourceFailureReport),
}

impl BenchmarkCaseReport {
    fn to_text(&self) -> String {
        match self {
            Self::Success(report) => report.to_text(),
            Self::PlanningRejected(report) => report.to_text(),
            Self::ResourceFailure(report) => report.to_text(),
        }
    }
}

async fn run_configured_benchmark_case(
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<ConfiguredBenchmarkCaseReport> {
    let dir = config.output_dir.clone().unwrap_or_else(default_benchmark_dir);
    fs::create_dir_all(&dir)?;
    let path = benchmark_input_path(&dir, config);

    write_benchmark_parquet_dataset(&path, config)?;
    let parquet_file_bytes = parquet_dataset_bytes(&path)?;
    validate_file_size(parquet_file_bytes, config)?;
    let dataset_layout = parquet_dataset_layout(&path)?;
    validate_dataset_layout(&dataset_layout, config)?;

    let source_diagnostic = local_parquet_scan_diagnostic(&path).await?;
    let report = match config.expected_outcome {
        BenchmarkExpectedOutcome::Success => BenchmarkCaseReport::Success(
            run_direct_parquet_benchmark(&path, parquet_file_bytes, dataset_layout, source_diagnostic.clone(), config)
                .await?,
        ),
        BenchmarkExpectedOutcome::PlanningRejected => BenchmarkCaseReport::PlanningRejected(
            run_expected_planning_rejection_probe(
                &path,
                parquet_file_bytes,
                dataset_layout,
                source_diagnostic.clone(),
                config,
            )
            .await?,
        ),
        BenchmarkExpectedOutcome::ResourceFailure => BenchmarkCaseReport::ResourceFailure(
            run_expected_resource_failure_probe(
                &path,
                parquet_file_bytes,
                dataset_layout,
                source_diagnostic.clone(),
                config,
            )
            .await?,
        ),
    };

    Ok(ConfiguredBenchmarkCaseReport { path, parquet_file_bytes, dataset_layout, source_diagnostic, report })
}

fn benchmark_case_prologue(report: &ConfiguredBenchmarkCaseReport, config: &LocalParquetBenchmarkConfig) -> String {
    let mut text = String::new();
    push_line(&mut text, format_args!("local_parquet_benchmark"));
    push_line(&mut text, format_args!("parquet_path={}", report.path.display()));
    push_line(&mut text, format_args!("source_diagnostic={}", report.source_diagnostic));
    push_line(&mut text, format_args!("parquet_file_bytes={}", report.parquet_file_bytes));
    push_line(&mut text, format_args!("input_rows={}", config.input_rows));
    push_line(&mut text, format_args!("parquet_files={}", config.file_count));
    push_line(&mut text, format_args!("measure_columns={}", config.measure_columns));
    push_line(&mut text, format_args!("projected_measure_pairs={}", config.projected_measure_pairs()));
    push_line(&mut text, format_args!("group_count={}", config.group_count));
    push_line(&mut text, format_args!("batch_size={}", config.batch_size));
    push_line(&mut text, format_args!("row_group_size={}", config.row_group_size));
    push_line(&mut text, format_args!("parquet_row_groups={}", report.dataset_layout.row_group_count));
    push_line(&mut text, format_args!("minimum_native_gpu_speedup={:.3}", config.minimum_native_gpu_speedup));
    push_line(&mut text, format_args!("native_memory_control_policy={}", config.native_memory_control_policy.as_str()));
    push_line(&mut text, format_args!("benchmark_paths={}", config.benchmark_paths.name()));
    push_line(&mut text, format_args!("compression={}", config.compression.name()));
    push_line(&mut text, format_args!("native_plan_shape={}", config.native_plan_shape.name()));
    push_line(&mut text, format_args!("expected_outcome={}", config.expected_outcome.name()));
    push_line(&mut text, format_args!("phase_c_decision_matrix_cases={}", phase_c_decision_matrix().len()));
    push_line(&mut text, format_args!("limit={}", config.limit));
    push_line(&mut text, format_args!("warmups={}", config.warmup_iterations));
    push_line(&mut text, format_args!("iterations={}", config.iterations));
    text
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
    PlanningRejected,
    ResourceFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkRunMode {
    SingleCase,
    PhaseCMatrix,
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
            "planning_rejected" | "planning-rejected" => Ok(Self::PlanningRejected),
            "resource_failure" | "resource-failure" => Ok(Self::ResourceFailure),
            other => Err(plan_error(format!(
                "invalid {name}: {other}; expected success, planning_rejected, or resource_failure"
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::PlanningRejected => "planning_rejected",
            Self::ResourceFailure => "resource_failure",
        }
    }
}

impl BenchmarkRunMode {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::SingleCase);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "single" | "single_case" | "single-case" => Ok(Self::SingleCase),
            "phase_c_matrix" | "phase-c-matrix" => Ok(Self::PhaseCMatrix),
            other => Err(plan_error(format!("invalid {name}: {other}; expected single_case or phase_c_matrix"))),
        }
    }
}

impl BenchmarkSourceLayout {
    fn name(self) -> &'static str {
        match self {
            Self::SingleFileMultiRowGroup => "single_file_multi_row_group",
            Self::MultiFile => "multi_file",
        }
    }

    fn file_count(self) -> usize {
        match self {
            Self::SingleFileMultiRowGroup => 1,
            Self::MultiFile => 4,
        }
    }
}

impl BenchmarkScalePoint {
    fn name(self) -> &'static str {
        match self {
            Self::ThreeGiB => "3gib",
            Self::SixGiB => "6gib",
            Self::TenGiB => "10gib",
        }
    }

    fn input_rows(self, compression: BenchmarkCompression) -> usize {
        match (self, compression) {
            (Self::ThreeGiB, BenchmarkCompression::Uncompressed) => 12_250_000,
            (Self::ThreeGiB, BenchmarkCompression::Snappy) => 16_000_000,
            (Self::SixGiB, BenchmarkCompression::Uncompressed) => 24_500_000,
            (Self::SixGiB, BenchmarkCompression::Snappy) => 32_000_000,
            (Self::TenGiB, BenchmarkCompression::Uncompressed) => 40_900_000,
            (Self::TenGiB, BenchmarkCompression::Snappy) => 52_000_000,
        }
    }

    fn minimum_file_bytes(self) -> u64 {
        match self {
            Self::ThreeGiB => THREE_GIB,
            Self::SixGiB => SIX_GIB,
            Self::TenGiB => TEN_GIB,
        }
    }
}

impl PhaseCMatrixCase {
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
        let mut overrides = vec![
            ("CUDF_PARQUET_BENCH_ROWS", self.scale.input_rows(self.compression).to_string()),
            ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", self.scale.minimum_file_bytes().to_string()),
            ("CUDF_PARQUET_BENCH_COMPRESSION", self.compression.name().to_string()),
            ("CUDF_PARQUET_BENCH_FILES", self.source_layout.file_count().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", self.native_plan_shape.name().to_string()),
            ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", self.expected_outcome.name().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
        ];
        if self.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure {
            overrides.extend([
                ("CUDF_PARQUET_BENCH_NATIVE_DEVICE_BUDGET_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_RETRY_ATTEMPTS", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS", "1".to_string()),
            ]);
        }
        overrides
    }

    fn configured_case(self, base: &LocalParquetBenchmarkConfig, output_dir: PathBuf) -> LocalParquetBenchmarkConfig {
        let mut config = base.clone();
        config.input_rows = self.scale.input_rows(self.compression);
        config.file_count = self.source_layout.file_count();
        config.minimum_file_bytes = self.scale.minimum_file_bytes();
        config.compression = self.compression;
        config.native_plan_shape = self.native_plan_shape;
        config.expected_outcome = self.expected_outcome;
        config.benchmark_paths = BenchmarkPathSet::NativeOnly;
        config.output_dir = Some(output_dir);
        if self.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure {
            config.native_memory_control_policy = forced_resource_failure_policy();
        }
        config
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

const PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION: u32 = 2;
const PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_HEADER: &str = concat!(
    "schema_version\tcase_id\tscale\tcompression\tsource_layout\tnative_plan_shape\texpected_outcome\toutcome\t",
    "planning_outcome\tplanning_reason_code\tplanning_source_diagnostics\tnative_cudf_exec_accepted_count\t",
    "native_runtime_started\tno_silent_cpu_fallback\tzero_host_to_device_uploads\t",
    "one_device_to_host_materialization\trow_groups_selected\trow_groups_planned\trow_groups_committed\t",
    "source_chunks_planned\tsource_chunks_committed\tsource_chunks_retried\treservation_denied_count\t",
    "retry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\tcpu_datafusion_wall_ms\t",
    "datafusion_native_exec_wall_ms\tdatafusion_native_exec_speedup_vs_cpu_datafusion\t",
    "datafusion_native_exec_speedup_vs_cpu_comparable\tparquet_file_bytes\tparquet_row_groups\t",
    "runtime_error_code\truntime_error_kind\truntime_error_status\truntime_error_operation\truntime_error_facts\tlog_path"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkMatrixOutcome {
    Pass,
    PlanningRejected,
    ResourceFailure,
    RuntimeFailure,
}

impl BenchmarkMatrixOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::PlanningRejected => "planning_rejected",
            Self::ResourceFailure => "resource_failure",
            Self::RuntimeFailure => "runtime_failure",
        }
    }
}

#[derive(Debug)]
struct PhaseCMatrixReport {
    artifact_dir: PathBuf,
    tsv_path: PathBuf,
    json_path: PathBuf,
    markdown_path: PathBuf,
    rows: Vec<PhaseCMatrixReportRow>,
}

impl PhaseCMatrixReport {
    fn decision_gate_pass(&self) -> bool {
        self.rows.iter().all(PhaseCMatrixReportRow::matches_expected_outcome)
    }

    fn unexpected_outcome_count(&self) -> usize {
        self.rows.iter().filter(|row| !row.matches_expected_outcome()).count()
    }

    fn to_text(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("phase_c_datafusion_native_exec_matrix"));
        push_line(
            &mut text,
            format_args!("schema_version={}", PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION),
        );
        push_line(&mut text, format_args!("case_count={}", self.rows.len()));
        push_line(&mut text, format_args!("decision_gate={}", if self.decision_gate_pass() { "pass" } else { "fail" }));
        push_line(&mut text, format_args!("unexpected_outcomes={}", self.unexpected_outcome_count()));
        push_line(&mut text, format_args!("artifact_dir={}", self.artifact_dir.display()));
        push_line(&mut text, format_args!("tsv_report={}", self.tsv_path.display()));
        push_line(&mut text, format_args!("json_report={}", self.json_path.display()));
        push_line(&mut text, format_args!("markdown_report={}", self.markdown_path.display()));
        text
    }

    fn to_tsv(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("{PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_HEADER}"));
        for row in &self.rows {
            push_line(&mut text, format_args!("{}", row.to_tsv_row()));
        }
        text
    }

    fn to_json(&self) -> String {
        let rows = self.rows.iter().map(PhaseCMatrixReportRow::to_json_value).collect::<Vec<_>>();
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION,
            "decision_gate": if self.decision_gate_pass() { "pass" } else { "fail" },
            "unexpected_outcomes": self.unexpected_outcome_count(),
            "rows": rows,
        }))
        .expect("phase c matrix JSON report should serialize")
    }

    fn to_markdown(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("# DataFusion Native Exec Phase C Matrix Results"));
        push_line(&mut text, format_args!(""));
        push_line(
            &mut text,
            format_args!(
                "Generated by `cargo run --example local_parquet_benchmark` with `CUDF_PARQUET_BENCH_RUN=phase_c_matrix`."
            ),
        );
        push_line(&mut text, format_args!(""));
        push_line(
            &mut text,
            format_args!(
                "Decision gate: **{}**. Unexpected outcomes: {}.",
                if self.decision_gate_pass() { "pass" } else { "fail" },
                self.unexpected_outcome_count()
            ),
        );
        push_line(&mut text, format_args!(""));
        push_line(
            &mut text,
            format_args!(
                "| case id | expected | outcome | planning | source facts | accepted | zero upload | one materialization | row groups | chunks | retries | native wall ms | speedup | error |"
            ),
        );
        push_line(
            &mut text,
            format_args!("| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"),
        );
        for row in &self.rows {
            push_line(
                &mut text,
                format_args!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} | {}/{}/{} | {}/{}/{} | {} | {} | {} |",
                    row.case_id,
                    row.expected_outcome,
                    row.outcome.as_str(),
                    row.planning_outcome,
                    markdown_table_cell(&row.planning_source_diagnostics),
                    row.native_cudf_exec_accepted_count,
                    row.zero_host_to_device_uploads,
                    row.one_device_to_host_materialization,
                    row.row_groups_selected,
                    row.row_groups_planned,
                    row.row_groups_committed,
                    row.source_chunks_planned,
                    row.source_chunks_committed,
                    row.source_chunks_retried,
                    row.retry_count,
                    row.retry_resource_exhausted_count,
                    row.retry_reservation_failure_count,
                    row.datafusion_native_exec_wall_ms,
                    row.datafusion_native_exec_speedup_vs_cpu_datafusion,
                    row.runtime_error_code
                ),
            );
        }
        text
    }
}

#[derive(Debug, Clone)]
struct PhaseCMatrixReportRow {
    case_id: String,
    scale: &'static str,
    compression: &'static str,
    source_layout: &'static str,
    native_plan_shape: &'static str,
    expected_outcome: &'static str,
    outcome: BenchmarkMatrixOutcome,
    planning_outcome: String,
    planning_reason_code: String,
    planning_source_diagnostics: String,
    native_cudf_exec_accepted_count: String,
    native_runtime_started: &'static str,
    no_silent_cpu_fallback: &'static str,
    zero_host_to_device_uploads: String,
    one_device_to_host_materialization: String,
    row_groups_selected: String,
    row_groups_planned: String,
    row_groups_committed: String,
    source_chunks_planned: String,
    source_chunks_committed: String,
    source_chunks_retried: String,
    reservation_denied_count: String,
    retry_count: String,
    retry_resource_exhausted_count: String,
    retry_reservation_failure_count: String,
    cpu_datafusion_wall_ms: String,
    datafusion_native_exec_wall_ms: String,
    datafusion_native_exec_speedup_vs_cpu_datafusion: String,
    datafusion_native_exec_speedup_vs_cpu_comparable: String,
    parquet_file_bytes: String,
    parquet_row_groups: String,
    runtime_error_code: String,
    runtime_error_kind: String,
    runtime_error_status: String,
    runtime_error_operation: String,
    runtime_error_facts: String,
    log_path: String,
}

impl PhaseCMatrixReportRow {
    fn from_case_report(case: PhaseCMatrixCase, report: &ConfiguredBenchmarkCaseReport, log_path: &Path) -> Self {
        match &report.report {
            BenchmarkCaseReport::Success(success) => Self::from_success(case, success, log_path),
            BenchmarkCaseReport::PlanningRejected(rejected) => Self::from_planning_rejected(case, rejected, log_path),
            BenchmarkCaseReport::ResourceFailure(resource_failure) => {
                Self::from_resource_failure(case, resource_failure, log_path)
            },
        }
    }

    fn runtime_failure(case: PhaseCMatrixCase, error: &DataFusionError, log_path: &Path) -> Self {
        let adapter_error = adapter_error_from_datafusion(error).ok();
        let (code, kind, status, operation, facts) = adapter_error.map_or_else(
            || {
                (
                    "unavailable".to_string(),
                    "unavailable".to_string(),
                    "unavailable".to_string(),
                    "unavailable".to_string(),
                    sanitize_report_value(&error.to_string()),
                )
            },
            |error| {
                (
                    error.code().as_str().to_string(),
                    error.kind().as_str().to_string(),
                    error.status().as_str().to_string(),
                    error.operation().to_string(),
                    adapter_error_facts_tsv(error),
                )
            },
        );
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::RuntimeFailure,
            planning_outcome: "unavailable".to_string(),
            planning_reason_code: "unavailable".to_string(),
            planning_source_diagnostics: "unavailable".to_string(),
            native_cudf_exec_accepted_count: "unavailable".to_string(),
            native_runtime_started: "unknown",
            no_silent_cpu_fallback: "unknown",
            zero_host_to_device_uploads: "unavailable".to_string(),
            one_device_to_host_materialization: "unavailable".to_string(),
            row_groups_selected: "unavailable".to_string(),
            row_groups_planned: "unavailable".to_string(),
            row_groups_committed: "unavailable".to_string(),
            source_chunks_planned: "unavailable".to_string(),
            source_chunks_committed: "unavailable".to_string(),
            source_chunks_retried: "unavailable".to_string(),
            reservation_denied_count: "unavailable".to_string(),
            retry_count: "unavailable".to_string(),
            retry_resource_exhausted_count: "unavailable".to_string(),
            retry_reservation_failure_count: "unavailable".to_string(),
            cpu_datafusion_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_datafusion: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_comparable: "unavailable".to_string(),
            parquet_file_bytes: "unavailable".to_string(),
            parquet_row_groups: "unavailable".to_string(),
            runtime_error_code: code,
            runtime_error_kind: kind,
            runtime_error_status: status,
            runtime_error_operation: operation,
            runtime_error_facts: facts,
            log_path: log_path.display().to_string(),
        }
    }

    fn from_success(case: PhaseCMatrixCase, report: &DirectParquetBenchmarkReport, log_path: &Path) -> Self {
        let metrics = &report.final_datafusion_native_exec.metrics;
        let planning_row = report
            .final_datafusion_native_exec
            .native_datafusion_planning_report
            .as_ref()
            .and_then(|report| report.rows.first());
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::Pass,
            planning_outcome: planning_row.map_or("unavailable", |row| row.outcome.as_str()).to_string(),
            planning_reason_code: planning_row.and_then(|row| row.reason_code.as_deref()).unwrap_or("none").to_string(),
            planning_source_diagnostics: planning_row
                .map(|row| native_source_diagnostics_text(&row.source_diagnostics))
                .unwrap_or_else(|| "unavailable".to_string()),
            native_cudf_exec_accepted_count: report
                .final_datafusion_native_exec
                .native_datafusion_planning_report
                .as_ref()
                .map_or_else(|| "unavailable".to_string(), |report| report.accepted_count().to_string()),
            native_runtime_started: "true",
            no_silent_cpu_fallback: "true",
            zero_host_to_device_uploads: (metrics.host_to_device_uploads == 0).to_string(),
            one_device_to_host_materialization: (metrics.device_to_host_materializations == 1).to_string(),
            row_groups_selected: metrics.row_groups_selected.to_string(),
            row_groups_planned: metrics.row_groups_planned.to_string(),
            row_groups_committed: metrics.row_groups_committed.to_string(),
            source_chunks_planned: metrics.source_chunks_planned.to_string(),
            source_chunks_committed: metrics.source_chunks_committed.to_string(),
            source_chunks_retried: metrics.source_chunks_retried.to_string(),
            reservation_denied_count: metrics.reservation_denied_count.to_string(),
            retry_count: metrics.retry_count.to_string(),
            retry_resource_exhausted_count: metrics.retry_resource_exhausted_count.to_string(),
            retry_reservation_failure_count: metrics.retry_reservation_failure_count.to_string(),
            cpu_datafusion_wall_ms: format!("{:.3}", duration_ms(report.cpu_default.mean)),
            datafusion_native_exec_wall_ms: format!("{:.3}", duration_ms(report.datafusion_native_exec.mean)),
            datafusion_native_exec_speedup_vs_cpu_datafusion: format!(
                "{:.3}",
                speedup(report.cpu_default.mean, report.datafusion_native_exec.mean)
            ),
            datafusion_native_exec_speedup_vs_cpu_comparable: format!(
                "{:.3}",
                speedup(report.cpu_comparable.mean, report.datafusion_native_exec.mean)
            ),
            parquet_file_bytes: report.parquet_file_bytes.to_string(),
            parquet_row_groups: report.parquet_row_groups.to_string(),
            runtime_error_code: "none".to_string(),
            runtime_error_kind: "none".to_string(),
            runtime_error_status: "none".to_string(),
            runtime_error_operation: "none".to_string(),
            runtime_error_facts: "none".to_string(),
            log_path: log_path.display().to_string(),
        }
    }

    fn from_planning_rejected(
        case: PhaseCMatrixCase,
        report: &ExpectedPlanningRejectedReport,
        log_path: &Path,
    ) -> Self {
        let row = report.planning_report.rows.first();
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::PlanningRejected,
            planning_outcome: row.map_or("unavailable", |row| row.outcome.as_str()).to_string(),
            planning_reason_code: row.and_then(|row| row.reason_code.as_deref()).unwrap_or("none").to_string(),
            planning_source_diagnostics: row
                .map(|row| native_source_diagnostics_text(&row.source_diagnostics))
                .unwrap_or_else(|| "unavailable".to_string()),
            native_cudf_exec_accepted_count: report.planning_report.accepted_count().to_string(),
            native_runtime_started: "false",
            no_silent_cpu_fallback: "true",
            zero_host_to_device_uploads: "unavailable".to_string(),
            one_device_to_host_materialization: "unavailable".to_string(),
            row_groups_selected: "unavailable".to_string(),
            row_groups_planned: "unavailable".to_string(),
            row_groups_committed: "unavailable".to_string(),
            source_chunks_planned: "unavailable".to_string(),
            source_chunks_committed: "unavailable".to_string(),
            source_chunks_retried: "unavailable".to_string(),
            reservation_denied_count: "unavailable".to_string(),
            retry_count: "unavailable".to_string(),
            retry_resource_exhausted_count: "unavailable".to_string(),
            retry_reservation_failure_count: "unavailable".to_string(),
            cpu_datafusion_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_datafusion: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_comparable: "unavailable".to_string(),
            parquet_file_bytes: report.parquet_file_bytes.to_string(),
            parquet_row_groups: report.parquet_row_groups.to_string(),
            runtime_error_code: "none".to_string(),
            runtime_error_kind: "none".to_string(),
            runtime_error_status: "none".to_string(),
            runtime_error_operation: "none".to_string(),
            runtime_error_facts: "none".to_string(),
            log_path: log_path.display().to_string(),
        }
    }

    fn from_resource_failure(case: PhaseCMatrixCase, report: &ExpectedResourceFailureReport, log_path: &Path) -> Self {
        let row = report.planning_report.rows.first();
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::ResourceFailure,
            planning_outcome: row.map_or("unavailable", |row| row.outcome.as_str()).to_string(),
            planning_reason_code: row.and_then(|row| row.reason_code.as_deref()).unwrap_or("none").to_string(),
            planning_source_diagnostics: row
                .map(|row| native_source_diagnostics_text(&row.source_diagnostics))
                .unwrap_or_else(|| "unavailable".to_string()),
            native_cudf_exec_accepted_count: report.planning_report.accepted_count().to_string(),
            native_runtime_started: "true",
            no_silent_cpu_fallback: "true",
            zero_host_to_device_uploads: "unavailable".to_string(),
            one_device_to_host_materialization: "unavailable".to_string(),
            row_groups_selected: "unavailable".to_string(),
            row_groups_planned: "unavailable".to_string(),
            row_groups_committed: "unavailable".to_string(),
            source_chunks_planned: "unavailable".to_string(),
            source_chunks_committed: "unavailable".to_string(),
            source_chunks_retried: "unavailable".to_string(),
            reservation_denied_count: "unavailable".to_string(),
            retry_count: "unavailable".to_string(),
            retry_resource_exhausted_count: "unavailable".to_string(),
            retry_reservation_failure_count: "unavailable".to_string(),
            cpu_datafusion_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_datafusion: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_comparable: "unavailable".to_string(),
            parquet_file_bytes: report.parquet_file_bytes.to_string(),
            parquet_row_groups: report.parquet_row_groups.to_string(),
            runtime_error_code: report.error.code.clone(),
            runtime_error_kind: report.error.kind.clone(),
            runtime_error_status: report.error.status.clone(),
            runtime_error_operation: report.error.operation.to_string(),
            runtime_error_facts: report.error.facts.clone(),
            log_path: log_path.display().to_string(),
        }
    }

    fn matches_expected_outcome(&self) -> bool {
        matches!(
            (self.expected_outcome, self.outcome),
            ("success", BenchmarkMatrixOutcome::Pass)
                | ("planning_rejected", BenchmarkMatrixOutcome::PlanningRejected)
                | ("resource_failure", BenchmarkMatrixOutcome::ResourceFailure)
        )
    }

    fn to_tsv_row(&self) -> String {
        [
            PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION.to_string(),
            self.case_id.clone(),
            self.scale.to_string(),
            self.compression.to_string(),
            self.source_layout.to_string(),
            self.native_plan_shape.to_string(),
            self.expected_outcome.to_string(),
            self.outcome.as_str().to_string(),
            self.planning_outcome.clone(),
            self.planning_reason_code.clone(),
            self.planning_source_diagnostics.clone(),
            self.native_cudf_exec_accepted_count.clone(),
            self.native_runtime_started.to_string(),
            self.no_silent_cpu_fallback.to_string(),
            self.zero_host_to_device_uploads.clone(),
            self.one_device_to_host_materialization.clone(),
            self.row_groups_selected.clone(),
            self.row_groups_planned.clone(),
            self.row_groups_committed.clone(),
            self.source_chunks_planned.clone(),
            self.source_chunks_committed.clone(),
            self.source_chunks_retried.clone(),
            self.reservation_denied_count.clone(),
            self.retry_count.clone(),
            self.retry_resource_exhausted_count.clone(),
            self.retry_reservation_failure_count.clone(),
            self.cpu_datafusion_wall_ms.clone(),
            self.datafusion_native_exec_wall_ms.clone(),
            self.datafusion_native_exec_speedup_vs_cpu_datafusion.clone(),
            self.datafusion_native_exec_speedup_vs_cpu_comparable.clone(),
            self.parquet_file_bytes.clone(),
            self.parquet_row_groups.clone(),
            self.runtime_error_code.clone(),
            self.runtime_error_kind.clone(),
            self.runtime_error_status.clone(),
            self.runtime_error_operation.clone(),
            self.runtime_error_facts.clone(),
            self.log_path.clone(),
        ]
        .into_iter()
        .map(|value| tsv_escape(&value))
        .collect::<Vec<_>>()
        .join("\t")
    }

    fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "case_id": self.case_id,
            "scale": self.scale,
            "compression": self.compression,
            "source_layout": self.source_layout,
            "native_plan_shape": self.native_plan_shape,
            "expected_outcome": self.expected_outcome,
            "outcome": self.outcome.as_str(),
            "planning_outcome": self.planning_outcome,
            "planning_reason_code": self.planning_reason_code,
            "planning_source_diagnostics": self.planning_source_diagnostics,
            "native_cudf_exec_accepted_count": self.native_cudf_exec_accepted_count,
            "native_runtime_started": self.native_runtime_started,
            "no_silent_cpu_fallback": self.no_silent_cpu_fallback,
            "zero_host_to_device_uploads": self.zero_host_to_device_uploads,
            "one_device_to_host_materialization": self.one_device_to_host_materialization,
            "row_groups_selected": self.row_groups_selected,
            "row_groups_planned": self.row_groups_planned,
            "row_groups_committed": self.row_groups_committed,
            "source_chunks_planned": self.source_chunks_planned,
            "source_chunks_committed": self.source_chunks_committed,
            "source_chunks_retried": self.source_chunks_retried,
            "reservation_denied_count": self.reservation_denied_count,
            "retry_count": self.retry_count,
            "retry_resource_exhausted_count": self.retry_resource_exhausted_count,
            "retry_reservation_failure_count": self.retry_reservation_failure_count,
            "cpu_datafusion_wall_ms": self.cpu_datafusion_wall_ms,
            "datafusion_native_exec_wall_ms": self.datafusion_native_exec_wall_ms,
            "datafusion_native_exec_speedup_vs_cpu_datafusion": self.datafusion_native_exec_speedup_vs_cpu_datafusion,
            "datafusion_native_exec_speedup_vs_cpu_comparable": self.datafusion_native_exec_speedup_vs_cpu_comparable,
            "parquet_file_bytes": self.parquet_file_bytes,
            "parquet_row_groups": self.parquet_row_groups,
            "runtime_error_code": self.runtime_error_code,
            "runtime_error_kind": self.runtime_error_kind,
            "runtime_error_status": self.runtime_error_status,
            "runtime_error_operation": self.runtime_error_operation,
            "runtime_error_facts": self.runtime_error_facts,
            "log_path": self.log_path,
        })
    }
}

async fn run_phase_c_matrix(base_config: &LocalParquetBenchmarkConfig) -> DataFusionResult<PhaseCMatrixReport> {
    let artifact_dir = phase_c_matrix_artifact_dir();
    let logs_dir = artifact_dir.join("logs");
    let data_dir = phase_c_matrix_data_dir();
    fs::create_dir_all(&logs_dir)?;
    fs::create_dir_all(&data_dir)?;

    let mut rows = Vec::new();
    for case in phase_c_decision_matrix() {
        let case_id = case.name();
        let log_path = logs_dir.join(format!("{case_id}.log"));
        let case_config = case.configured_case(base_config, data_dir.join(&case_id));
        match run_configured_benchmark_case(&case_config).await {
            Ok(case_report) => {
                let mut log = benchmark_case_prologue(&case_report, &case_config);
                log.push('\n');
                log.push_str(&case_report.report.to_text());
                fs::write(&log_path, log)?;
                rows.push(PhaseCMatrixReportRow::from_case_report(case, &case_report, &log_path));
            },
            Err(error) => {
                fs::write(&log_path, format!("case={case_id}\noutcome=runtime_failure\nerror={error}\n"))?;
                rows.push(PhaseCMatrixReportRow::runtime_failure(case, &error, &log_path));
            },
        }
    }

    let report = PhaseCMatrixReport {
        tsv_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.tsv"),
        json_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.json"),
        markdown_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.md"),
        artifact_dir,
        rows,
    };
    fs::write(&report.tsv_path, report.to_tsv())?;
    fs::write(&report.json_path, report.to_json())?;
    fs::write(&report.markdown_path, report.to_markdown())?;
    Ok(report)
}

fn phase_c_matrix_artifact_dir() -> PathBuf {
    std::env::var_os("CUDF_PARQUET_BENCH_MATRIX_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/phase-c-matrix"))
}

fn phase_c_matrix_data_dir() -> PathBuf {
    std::env::var_os("CUDF_PARQUET_BENCH_MATRIX_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PHASE_C_MATRIX_DATA_DIR))
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
    datafusion_native_exec: TimedBenchmarkRun,
    manual_native_lowering: TimedBenchmarkRun,
    cpu_comparable_samples: Vec<Duration>,
    cpu_default_samples: Vec<Duration>,
    gpu_samples: Vec<Duration>,
    datafusion_native_exec_samples: Vec<Duration>,
    manual_native_lowering_samples: Vec<Duration>,
    gpu_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    datafusion_native_exec_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    manual_native_lowering_metric_samples: Vec<LargerIslandBenchmarkMetrics>,
    final_cpu_comparable: CollectMeasurement,
    final_cpu_default: CollectMeasurement,
    final_gpu: Option<GpuCollectMeasurement>,
    final_datafusion_native_exec: GpuCollectMeasurement,
    final_manual_native_lowering: GpuCollectMeasurement,
    source_diagnostic: String,
}

impl DirectParquetBenchmarkReport {
    fn to_text(&self) -> String {
        let comparable_speedup = self.gpu.as_ref().map(|gpu| speedup(self.cpu_comparable.mean, gpu.mean));
        let default_speedup = self.gpu.as_ref().map(|gpu| speedup(self.cpu_default.mean, gpu.mean));
        let datafusion_native_comparable_speedup = speedup(self.cpu_comparable.mean, self.datafusion_native_exec.mean);
        let datafusion_native_default_speedup = speedup(self.cpu_default.mean, self.datafusion_native_exec.mean);
        let manual_native_comparable_speedup = speedup(self.cpu_comparable.mean, self.manual_native_lowering.mean);
        let manual_native_default_speedup = speedup(self.cpu_default.mean, self.manual_native_lowering.mean);
        let native_metrics = &self.final_datafusion_native_exec.metrics;
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
        push_line(&mut text, format_args!("cpu_datafusion_wall_ms={:.3}", duration_ms(self.cpu_default.mean)));
        if let Some(gpu) = &self.gpu {
            push_line(&mut text, format_args!("gpu_wall_ms={:.3}", duration_ms(gpu.mean)));
        } else {
            push_line(&mut text, format_args!("gpu_wall_ms=not_run"));
        }
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_wall_ms={:.3}", duration_ms(self.datafusion_native_exec.mean)),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_wall_ms={:.3}", duration_ms(self.manual_native_lowering.mean)),
        );
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_comparable={}", optional_f64_text(comparable_speedup)));
        push_line(&mut text, format_args!("gpu_speedup={}", optional_f64_text(comparable_speedup)));
        push_line(&mut text, format_args!("gpu_speedup_vs_cpu_default={}", optional_f64_text(default_speedup)));
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_speedup_vs_cpu_comparable={datafusion_native_comparable_speedup:.3}"),
        );
        push_line(&mut text, format_args!("datafusion_native_exec_speedup={datafusion_native_comparable_speedup:.3}"));
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_speedup_vs_cpu_datafusion={datafusion_native_default_speedup:.3}"),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_speedup_vs_cpu_comparable={manual_native_comparable_speedup:.3}"),
        );
        push_line(&mut text, format_args!("manual_native_lowering_speedup={manual_native_comparable_speedup:.3}"));
        push_line(
            &mut text,
            format_args!("manual_native_lowering_speedup_vs_cpu_datafusion={manual_native_default_speedup:.3}"),
        );
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
            format_args!(
                "datafusion_native_exec_wall_ms_samples={}",
                duration_samples_csv(&self.datafusion_native_exec_samples)
            ),
        );
        push_line(
            &mut text,
            format_args!(
                "manual_native_lowering_wall_ms_samples={}",
                duration_samples_csv(&self.manual_native_lowering_samples)
            ),
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
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_output_rows={}", self.final_datafusion_native_exec.output_rows),
        );
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_output_batches={}", self.final_datafusion_native_exec.output_batches),
        );
        push_line(
            &mut text,
            format_args!("datafusion_native_exec_output_bytes={}", self.final_datafusion_native_exec.output_bytes),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_output_rows={}", self.final_manual_native_lowering.output_rows),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_output_batches={}", self.final_manual_native_lowering.output_batches),
        );
        push_line(
            &mut text,
            format_args!("manual_native_lowering_output_bytes={}", self.final_manual_native_lowering.output_bytes),
        );
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
        push_native_measurement_lines(
            &mut text,
            "datafusion_native_exec",
            &self.final_datafusion_native_exec,
            &self.datafusion_native_exec_metric_samples,
        );
        push_native_datafusion_planning_lines(
            &mut text,
            "datafusion_native_exec",
            self.final_datafusion_native_exec.native_datafusion_planning_report.as_ref(),
        );
        push_native_measurement_lines(
            &mut text,
            "manual_native_lowering",
            &self.final_manual_native_lowering,
            &self.manual_native_lowering_metric_samples,
        );
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

fn push_native_measurement_lines(
    text: &mut String,
    prefix: &str,
    measurement: &GpuCollectMeasurement,
    metric_samples: &[LargerIslandBenchmarkMetrics],
) {
    let metrics = &measurement.metrics;
    let planning = &measurement.planning;
    push_line(text, format_args!("{prefix}_parquet_read_ms={:.3}", duration_ms(metrics.parquet_read_time)));
    push_line(
        text,
        format_args!(
            "{prefix}_parquet_read_ms_mean={:.3}",
            duration_ms(mean_metric_duration(metric_samples, |metrics| metrics.parquet_read_time))
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_parquet_read_ms_samples={}",
            metric_duration_samples_csv(metric_samples, |metrics| metrics.parquet_read_time)
        ),
    );
    push_line(text, format_args!("{prefix}_cudf_operator_ms={:.3}", duration_ms(metrics.cudf_operator_time)));
    push_line(
        text,
        format_args!(
            "{prefix}_cudf_operator_ms_mean={:.3}",
            duration_ms(mean_metric_duration(metric_samples, |metrics| metrics.cudf_operator_time))
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_cudf_operator_ms_samples={}",
            metric_duration_samples_csv(metric_samples, |metrics| metrics.cudf_operator_time)
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_record_batch_materialization_ms={:.3}",
            duration_ms(metrics.record_batch_materialization_time)
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_record_batch_materialization_ms_mean={:.3}",
            duration_ms(mean_metric_duration(metric_samples, |metrics| metrics.record_batch_materialization_time))
        ),
    );
    push_line(
        text,
        format_args!("{prefix}_peak_device_memory_bytes={}", optional_usize_text(metrics.peak_device_memory_bytes)),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_resident_dataframe_peak_bytes={}",
            optional_usize_text(metrics.resident_dataframe_peak_bytes)
        ),
    );
    push_line(
        text,
        format_args!("{prefix}_allocator_high_water_bytes={}", optional_usize_text(metrics.allocator_high_water_bytes)),
    );
    push_line(text, format_args!("{prefix}_reservation_requested_bytes={}", metrics.reservation_requested_bytes));
    push_line(text, format_args!("{prefix}_reservation_granted_bytes={}", metrics.reservation_granted_bytes));
    push_line(text, format_args!("{prefix}_reservation_denied_count={}", metrics.reservation_denied_count));
    push_line(text, format_args!("{prefix}_source_chunks_planned={}", metrics.source_chunks_planned));
    push_line(text, format_args!("{prefix}_source_chunks_attempted={}", metrics.source_chunks_attempted));
    push_line(text, format_args!("{prefix}_source_chunks_committed={}", metrics.source_chunks_committed));
    push_line(text, format_args!("{prefix}_source_chunks_retried={}", metrics.source_chunks_retried));
    push_line(text, format_args!("{prefix}_row_groups_planned={}", metrics.row_groups_planned));
    push_line(text, format_args!("{prefix}_row_groups_committed={}", metrics.row_groups_committed));
    push_line(text, format_args!("{prefix}_partial_aggregate_count={}", metrics.partial_aggregate_count));
    push_line(text, format_args!("{prefix}_partial_aggregate_rows={}", metrics.partial_aggregate_rows));
    push_line(text, format_args!("{prefix}_retry_count={}", metrics.retry_count));
    push_line(text, format_args!("{prefix}_retry_resource_exhausted_count={}", metrics.retry_resource_exhausted_count));
    push_line(
        text,
        format_args!("{prefix}_retry_reservation_failure_count={}", metrics.retry_reservation_failure_count),
    );
    push_line(text, format_args!("{prefix}_final_merge_attempts={}", metrics.final_merge_attempts));
    push_line(text, format_args!("{prefix}_final_merge_levels={}", metrics.final_merge_levels));
    push_line(
        text,
        format_args!(
            "{prefix}_peak_device_memory_bytes_mean={}",
            optional_usize_text(mean_metric_optional_usize(metric_samples, |metrics| metrics.peak_device_memory_bytes))
        ),
    );
    push_line(
        text,
        format_args!(
            "{prefix}_peak_device_memory_bytes_samples={}",
            metric_optional_usize_samples_csv(metric_samples, |metrics| metrics.peak_device_memory_bytes)
        ),
    );
    push_line(text, format_args!("{prefix}_host_to_device_uploads={}", metrics.host_to_device_uploads));
    push_line(text, format_args!("{prefix}_host_to_device_uploaded_rows={}", metrics.host_to_device_uploaded_rows));
    push_line(
        text,
        format_args!("{prefix}_device_to_host_materializations={}", metrics.device_to_host_materializations),
    );
    push_line(
        text,
        format_args!("{prefix}_device_to_host_materialized_rows={}", metrics.device_to_host_materialized_rows),
    );
    push_line(text, format_args!("{prefix}_parquet_files={}", metrics.parquet_files));
    push_line(text, format_args!("{prefix}_parquet_bytes_known={}", metrics.parquet_bytes_known));
    push_line(text, format_args!("{prefix}_row_groups_selected={}", metrics.row_groups_selected));
    push_line(text, format_args!("{prefix}_read_column_count={}", metrics.read_columns));
    push_line(text, format_args!("{prefix}_read_columns={}", measurement.read_columns.join(",")));
    push_line(text, format_args!("{prefix}_fallback_candidate_count={}", planning.candidate_count));
    push_line(text, format_args!("{prefix}_fallback_rewritten_count={}", planning.rewritten_count));
    push_line(text, format_args!("{prefix}_fallback_rejected_count={}", planning.rejected_count));
    push_line(
        text,
        format_args!(
            "{prefix}_source_physical_plan={}",
            measurement.source_physical_plan_one_line.as_deref().unwrap_or("not_available")
        ),
    );
    push_line(
        text,
        format_args!("{prefix}_lowered_candidate_shapes={}", measurement.native_candidate_shapes.join(";")),
    );
    push_line(text, format_args!("{prefix}_plan={}", measurement.plan_one_line));
}

fn push_native_datafusion_planning_lines(
    text: &mut String,
    prefix: &str,
    report: Option<&NativeDataFusionPlanningReport>,
) {
    let Some(report) = report else {
        push_line(text, format_args!("{prefix}_planning_report_status=not_available"));
        return;
    };
    push_line(text, format_args!("{prefix}_planning_report_status=available"));
    push_line(text, format_args!("{prefix}_planning_accepted_count={}", report.accepted_count()));
    push_line(text, format_args!("{prefix}_planning_rejected_count={}", report.rejected_count()));
    let Some(row) = report.rows.first() else {
        push_line(text, format_args!("{prefix}_planning_outcome=none"));
        return;
    };
    push_line(text, format_args!("{prefix}_planning_outcome={}", row.outcome.as_str()));
    push_line(text, format_args!("{prefix}_planning_reason_code={}", row.reason_code.as_deref().unwrap_or("none")));
    push_line(
        text,
        format_args!(
            "{prefix}_planning_unsupported_category={}",
            row.unsupported_category.as_deref().unwrap_or("none")
        ),
    );
    push_line(text, format_args!("{prefix}_planning_post_native_plan_root={}", row.post_native_plan_root));
    push_line(
        text,
        format_args!("{prefix}_planning_post_native_execution_path={}", row.post_native_execution_path.as_str()),
    );
    push_line(text, format_args!("{prefix}_planning_rejected_before_execution={}", row.rejected_before_execution));
    push_line(
        text,
        format_args!(
            "{prefix}_planning_source_diagnostics={}",
            native_source_diagnostics_text(&row.source_diagnostics)
        ),
    );
}

fn native_source_diagnostics_text(diagnostics: &[datafusion_cudf::NativeSourceDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    diagnostics
        .iter()
        .map(|diagnostic| {
            format!(
                "format={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},schema_mapping={},rejected_features={}",
                diagnostic.format,
                diagnostic.direct_device_read,
                diagnostic.file_count,
                diagnostic.ranged_file_count,
                diagnostic.known_bytes.map_or_else(|| "unknown".to_string(), |value| value.to_string()),
                diagnostic.selected_row_groups.map_or_else(|| "all".to_string(), |value| value.to_string()),
                diagnostic.read_column_count,
                diagnostic.projected_output_column_count,
                diagnostic.pure_filter_column_count,
                diagnostic.predicate_disposition,
                diagnostic.byte_range_disposition,
                diagnostic.partition_column_disposition,
                diagnostic.zero_column_count_disposition,
                diagnostic.schema_mapping_disposition,
                if diagnostic.rejected_features.is_empty() {
                    "none".to_string()
                } else {
                    diagnostic.rejected_features.join(",")
                },
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[derive(Debug)]
struct ExpectedPlanningRejectedReport {
    parquet_file_bytes: u64,
    parquet_row_groups: usize,
    input_rows: usize,
    file_count: usize,
    compression: BenchmarkCompression,
    native_plan_shape: BenchmarkPlanShape,
    expected_outcome: BenchmarkExpectedOutcome,
    final_plan_root: String,
    source_diagnostic: String,
    planning_report: NativeDataFusionPlanningReport,
}

impl ExpectedPlanningRejectedReport {
    fn to_text(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("case=datafusion_native_exec_planning_rejected"));
        push_line(&mut text, format_args!("parquet_file_bytes={}", self.parquet_file_bytes));
        push_line(&mut text, format_args!("parquet_row_groups={}", self.parquet_row_groups));
        push_line(&mut text, format_args!("input_rows={}", self.input_rows));
        push_line(&mut text, format_args!("configured_parquet_files={}", self.file_count));
        push_line(&mut text, format_args!("compression={}", self.compression.name()));
        push_line(&mut text, format_args!("native_plan_shape={}", self.native_plan_shape.name()));
        push_line(&mut text, format_args!("expected_outcome={}", self.expected_outcome.name()));
        push_line(&mut text, format_args!("phase_c_runtime_gate=planning_rejected"));
        push_line(&mut text, format_args!("native_runtime_executed=false"));
        push_line(&mut text, format_args!("datafusion_native_exec_final_plan_root={}", self.final_plan_root));
        push_native_datafusion_planning_lines(&mut text, "datafusion_native_exec", Some(&self.planning_report));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

async fn run_expected_planning_rejection_probe(
    path: &Path,
    parquet_file_bytes: u64,
    dataset_layout: ParquetDatasetLayout,
    source_diagnostic: String,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<ExpectedPlanningRejectedReport> {
    let planned = plan_datafusion_native_sql(path, rejected_benchmark_sql(), config).await?;
    if planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_some() || planned.planning_report.accepted_count() != 0
    {
        return Err(plan_error(format!(
            "expected planning-rejected benchmark mode produced NativeCudfExec: root={}, accepted_count={}, rejected_count={}",
            planned.plan.name(),
            planned.planning_report.accepted_count(),
            planned.planning_report.rejected_count()
        )));
    }
    if planned.planning_report.rejected_count() == 0 {
        return Err(plan_error(format!(
            "expected planning-rejected benchmark mode did not retain a rejection row: root={}, candidates={}",
            planned.plan.name(),
            planned.native_candidate_shapes.join(";")
        )));
    }

    Ok(ExpectedPlanningRejectedReport {
        parquet_file_bytes,
        parquet_row_groups: dataset_layout.row_group_count,
        input_rows: config.input_rows,
        file_count: config.file_count,
        compression: config.compression,
        native_plan_shape: config.native_plan_shape,
        expected_outcome: config.expected_outcome,
        final_plan_root: planned.plan.name().to_string(),
        source_diagnostic,
        planning_report: planned.planning_report,
    })
}

fn rejected_benchmark_sql() -> &'static str {
    "SELECT AVG(m00) AS avg_m00 FROM bench"
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
    final_plan_root: String,
    planning_report: NativeDataFusionPlanningReport,
    native_candidate_shapes: Vec<String>,
    source_physical_plan_one_line: String,
    source_diagnostic: String,
    error: AdapterRuntimeErrorReport,
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
        push_line(&mut text, format_args!("native_runtime_executed=true"));
        push_line(&mut text, format_args!("datafusion_native_exec_final_plan_root={}", self.final_plan_root));
        push_native_datafusion_planning_lines(&mut text, "datafusion_native_exec", Some(&self.planning_report));
        push_line(&mut text, format_args!("native_error_code={}", self.error.code));
        push_line(&mut text, format_args!("native_error_kind={}", self.error.kind));
        push_line(&mut text, format_args!("native_error_status={}", self.error.status));
        push_line(&mut text, format_args!("native_error_operation={}", self.error.operation));
        push_line(&mut text, format_args!("native_error_facts={}", self.error.facts));
        push_line(
            &mut text,
            format_args!("native_lowered_candidate_shapes={}", self.native_candidate_shapes.join(";")),
        );
        push_line(&mut text, format_args!("native_default_sql_physical_plan={}", self.source_physical_plan_one_line));
        push_line(&mut text, format_args!("source_diagnostic={}", self.source_diagnostic));
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdapterRuntimeErrorReport {
    code: String,
    kind: String,
    status: String,
    operation: &'static str,
    facts: String,
}

impl AdapterRuntimeErrorReport {
    fn from_adapter_error(error: &AdapterError) -> Self {
        Self {
            code: error.code().as_str().to_string(),
            kind: error.kind().as_str().to_string(),
            status: error.status().as_str().to_string(),
            operation: error.operation(),
            facts: adapter_error_facts_tsv(error),
        }
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

    let planned = plan_datafusion_native_sql(path, &benchmark_sql(config), config).await?;
    let DataFusionNativePlannedExec { plan, planning: _, planning_report, native_candidate_shapes } = planned;
    if plan.as_any().downcast_ref::<NativeCudfExec>().is_none() || planning_report.accepted_count() != 1 {
        return Err(plan_error(format!(
            "expected resource-failure benchmark mode requires one accepted NativeCudfExec, got root={}, accepted_count={}, rejected_count={}, candidates={}",
            plan.name(),
            planning_report.accepted_count(),
            planning_report.rejected_count(),
            native_candidate_shapes.join(";")
        )));
    }

    let final_plan_root = plan.name().to_string();
    let source_physical_plan_one_line = planning_report
        .rows
        .first()
        .map(|row| row.physical_shape.clone())
        .unwrap_or_else(|| displayable(plan.as_ref()).one_line().to_string());

    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(_) => Err(plan_error(
            "expected resource-failure benchmark mode completed successfully; tighten the bounded memory policy or run as expected_outcome=success",
        )),
        Err(error) => {
            let adapter_error = adapter_error_from_datafusion(&error)?;
            if !is_planned_adapter_resource_failure(adapter_error) {
                return Err(error);
            }
            if adapter_error.fact_value("native_datafusion_phase") != Some("runtime_execution") {
                return Err(plan_error(format!(
                    "expected resource-failure benchmark mode to fail after native execution started, got native_datafusion_phase={:?}",
                    adapter_error.fact_value("native_datafusion_phase")
                )));
            }
            if adapter_error.fact_value("cpu_fallback_after_native_start") != Some("false") {
                return Err(plan_error(format!(
                    "expected resource-failure benchmark mode to preserve no-silent-fallback fact, got cpu_fallback_after_native_start={:?}",
                    adapter_error.fact_value("cpu_fallback_after_native_start")
                )));
            }
            Ok(ExpectedResourceFailureReport {
                parquet_file_bytes,
                parquet_row_groups: dataset_layout.row_group_count,
                input_rows: config.input_rows,
                file_count: config.file_count,
                compression: config.compression,
                native_plan_shape: config.native_plan_shape,
                expected_outcome: config.expected_outcome,
                final_plan_root,
                planning_report,
                native_candidate_shapes,
                source_physical_plan_one_line,
                source_diagnostic,
                error: AdapterRuntimeErrorReport::from_adapter_error(adapter_error),
            })
        },
    }
}

fn is_planned_adapter_resource_failure(error: &AdapterError) -> bool {
    matches!(
        error.code(),
        datafusion_cudf::ErrorCode::MemoryReservationFailed | datafusion_cudf::ErrorCode::CudfResourceExhausted
    )
}

fn adapter_error_from_datafusion(error: &DataFusionError) -> DataFusionResult<&AdapterError> {
    let DataFusionError::External(source) = error else {
        return Err(plan_error(format!(
            "native execution errors should cross DataFusion as external adapter errors: {error}"
        )));
    };
    source.downcast_ref::<AdapterError>().ok_or_else(|| {
        plan_error(format!("DataFusion external error did not preserve datafusion_cudf::Error: {error}"))
    })
}

fn adapter_error_facts_tsv(error: &AdapterError) -> String {
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
    native_datafusion_planning_report: Option<NativeDataFusionPlanningReport>,
    plan_one_line: String,
    source_physical_plan_one_line: Option<String>,
    native_candidate_shapes: Vec<String>,
    read_columns: Vec<String>,
    batches: Vec<RecordBatch>,
}

#[derive(Debug)]
struct DataFusionNativePlannedExec {
    plan: Arc<dyn ExecutionPlan>,
    planning: LargerIslandBenchmarkPlanning,
    planning_report: NativeDataFusionPlanningReport,
    native_candidate_shapes: Vec<String>,
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
    DataFusionNativeExec,
    ManualNativeLowering,
}

impl BenchmarkPath {
    fn label(self) -> &'static str {
        match self {
            Self::CpuComparable => "cpu_comparable",
            Self::CpuDefault => "cpu_default",
            Self::GpuDirect => "gpu_direct",
            Self::DataFusionNativeExec => "datafusion_native_exec",
            Self::ManualNativeLowering => "manual_native_lowering",
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
        let _ = collect_datafusion_native_exec_once(path, config).await?;
        let _ = collect_manual_native_lowering_once(path, config).await?;
    }

    let mut cpu_comparable_samples = Vec::with_capacity(config.iterations);
    let mut cpu_default_samples = Vec::with_capacity(config.iterations);
    let mut gpu_samples = Vec::with_capacity(config.iterations);
    let mut datafusion_native_exec_samples = Vec::with_capacity(config.iterations);
    let mut manual_native_lowering_samples = Vec::with_capacity(config.iterations);
    let mut gpu_metric_samples = Vec::with_capacity(config.iterations);
    let mut datafusion_native_exec_metric_samples = Vec::with_capacity(config.iterations);
    let mut manual_native_lowering_metric_samples = Vec::with_capacity(config.iterations);
    let mut final_cpu_comparable = None;
    let mut final_cpu_default = None;
    let mut final_gpu = None;
    let mut final_datafusion_native_exec = None;
    let mut final_manual_native_lowering = None;

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
                BenchmarkPath::DataFusionNativeExec => {
                    let measurement = collect_datafusion_native_exec_once(path, config).await?;
                    datafusion_native_exec_samples.push(measurement.wall_time);
                    datafusion_native_exec_metric_samples.push(measurement.metrics.clone());
                    final_datafusion_native_exec = Some(measurement);
                },
                BenchmarkPath::ManualNativeLowering => {
                    let measurement = collect_manual_native_lowering_once(path, config).await?;
                    manual_native_lowering_samples.push(measurement.wall_time);
                    manual_native_lowering_metric_samples.push(measurement.metrics.clone());
                    final_manual_native_lowering = Some(measurement);
                },
            }
        }
    }

    let final_cpu_comparable =
        final_cpu_comparable.expect("validated benchmark config requires measured comparable CPU iterations");
    let final_cpu_default =
        final_cpu_default.expect("validated benchmark config requires measured default CPU iterations");
    let mut final_datafusion_native_exec = final_datafusion_native_exec
        .expect("validated benchmark config requires measured DataFusion native iterations");
    let final_manual_native_lowering =
        final_manual_native_lowering.expect("validated benchmark config requires measured manual native iterations");
    if let Some(final_gpu) = &final_gpu {
        compare_outputs(&final_cpu_comparable.batches, &final_gpu.batches)?;
        compare_outputs(&final_cpu_default.batches, &final_gpu.batches)?;
        compare_outputs(&final_gpu.batches, &final_datafusion_native_exec.batches)?;
        compare_outputs(&final_gpu.batches, &final_manual_native_lowering.batches)?;
        validate_gpu_direct_measurement(final_gpu, parquet_file_bytes, config)?;
    }
    compare_outputs(&final_cpu_comparable.batches, &final_datafusion_native_exec.batches)?;
    compare_outputs(&final_cpu_default.batches, &final_datafusion_native_exec.batches)?;
    compare_outputs(&final_datafusion_native_exec.batches, &final_manual_native_lowering.batches)?;
    validate_datafusion_native_exec_measurement(&final_datafusion_native_exec, parquet_file_bytes, config)?;
    validate_manual_native_lowering_measurement(&final_manual_native_lowering, parquet_file_bytes, config)?;
    final_datafusion_native_exec.metrics.phase_c_runtime_gate =
        final_datafusion_native_exec.metrics.assess_phase_c_runtime_gate(true).code();
    let cpu_comparable = TimedBenchmarkRun::from_samples(cpu_comparable_samples.iter().copied());
    let cpu_default = TimedBenchmarkRun::from_samples(cpu_default_samples.iter().copied());
    let gpu = config
        .benchmark_paths
        .includes_gpu_direct()
        .then(|| TimedBenchmarkRun::from_samples(gpu_samples.iter().copied()));
    let datafusion_native_exec = TimedBenchmarkRun::from_samples(datafusion_native_exec_samples.iter().copied());
    let manual_native_lowering = TimedBenchmarkRun::from_samples(manual_native_lowering_samples.iter().copied());
    validate_phase_b_runtime_gate(
        &cpu_comparable,
        &datafusion_native_exec,
        &final_datafusion_native_exec,
        parquet_file_bytes,
        dataset_layout,
        config,
    )?;
    validate_phase_c_runtime_gate(&final_datafusion_native_exec, config)?;

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
        datafusion_native_exec,
        manual_native_lowering,
        cpu_comparable_samples,
        cpu_default_samples,
        gpu_samples,
        datafusion_native_exec_samples,
        manual_native_lowering_samples,
        gpu_metric_samples,
        datafusion_native_exec_metric_samples,
        manual_native_lowering_metric_samples,
        final_cpu_comparable,
        final_cpu_default,
        final_gpu,
        final_datafusion_native_exec,
        final_manual_native_lowering,
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
        native_datafusion_planning_report: None,
        plan_one_line,
        source_physical_plan_one_line: None,
        native_candidate_shapes: Vec::new(),
        read_columns,
        batches,
    })
}

async fn collect_datafusion_native_exec_once(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<GpuCollectMeasurement> {
    let planned = plan_datafusion_native_sql(path, &benchmark_sql(config), config).await?;
    let DataFusionNativePlannedExec { plan, planning, planning_report, native_candidate_shapes } = planned;
    let Some(native_exec) = plan.as_any().downcast_ref::<NativeCudfExec>() else {
        return Err(plan_error(format!(
            "datafusion_native_exec planning rejected NativeCudfExec path: root={}, accepted_count={}, rejected_count={}, candidates={}",
            plan.name(),
            planning_report.accepted_count(),
            planning_report.rejected_count(),
            native_candidate_shapes.join(";")
        )));
    };
    let read_columns = native_local_parquet_read_columns(native_exec.native_plan())?;
    let source_physical_plan_one_line = planning_report
        .rows
        .first()
        .map(|row| row.physical_shape.clone())
        .unwrap_or_else(|| "not_available".to_string());
    let plan_one_line = displayable(plan.as_ref()).one_line().to_string();

    let started = Instant::now();
    let batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await?;
    let wall_time = started.elapsed();
    let metrics = plan.metrics().unwrap_or_else(MetricsSet::new);
    let output = measure_output(&batches);

    Ok(GpuCollectMeasurement {
        wall_time,
        output_batches: output.output_batches,
        output_rows: output.output_rows,
        output_bytes: output.output_bytes,
        metrics: benchmark_metrics_from_datafusion_native(&metrics, config.native_memory_control_policy),
        planning,
        native_datafusion_planning_report: Some(planning_report),
        plan_one_line,
        source_physical_plan_one_line: Some(source_physical_plan_one_line),
        native_candidate_shapes,
        read_columns,
        batches,
    })
}

async fn plan_datafusion_native_sql(
    path: &Path,
    sql: &str,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<DataFusionNativePlannedExec> {
    let native_execution_config = native_datafusion_execution_config(config)?;
    let optimizer_config =
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_execution_config);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let gpu_island_rule = Arc::new(CudfPhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_datafusion_session_config())
        .with_cudf_native_and_gpu_island_rules(Arc::clone(&native_rule), gpu_island_rule)
        .build();
    let ctx = SessionContext::new_with_state(state);
    let path_text =
        path.to_str().ok_or_else(|| plan_error(format!("benchmark path is not valid UTF-8: {}", path.display())))?;
    ctx.register_parquet("bench", path_text, ParquetReadOptions::default()).await?;
    let df = ctx.sql(sql).await?;
    let plan = df.create_physical_plan().await?;
    let planning_report = NativeDataFusionPlanningReport::from_retained_rule(
        "local_parquet_benchmark",
        Arc::clone(&plan),
        native_rule.as_ref(),
    );
    let planning = native_rule
        .last_report()
        .map(|report| LargerIslandBenchmarkPlanning::from_report(&report))
        .unwrap_or_else(|| LargerIslandBenchmarkPlanning::from_report(&datafusion_cudf::PlanningReport::new()));
    let native_candidate_shapes = native_datafusion_candidate_shapes(&planning_report);
    Ok(DataFusionNativePlannedExec { plan, planning, planning_report, native_candidate_shapes })
}

async fn collect_manual_native_lowering_once(
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
        native_datafusion_planning_report: None,
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

fn native_datafusion_execution_config(
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<NativeCudfExecutionConfig> {
    match config.native_memory_control_policy {
        NativeMemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } => Ok(NativeCudfExecutionConfig::bounded(
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        )),
        NativeMemoryControlPolicy::Unbounded => {
            Err(plan_error("datafusion_native_exec requires CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY=bounded"))
        },
    }
}

fn native_datafusion_candidate_shapes(report: &NativeDataFusionPlanningReport) -> Vec<String> {
    report
        .rows
        .iter()
        .map(|row| {
            format!(
                "{}:{}:{}",
                row.outcome.as_str(),
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.reason_code.as_deref().unwrap_or("none")
            )
        })
        .collect()
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

fn benchmark_metrics_from_datafusion_native(
    metrics: &MetricsSet,
    memory_control_policy: NativeMemoryControlPolicy,
) -> LargerIslandBenchmarkMetrics {
    LargerIslandBenchmarkMetrics {
        phase_c_memory_control_policy: memory_control_policy.as_str(),
        phase_c_runtime_gate: phase_c_runtime_gate(memory_control_policy),
        input_batches: metric_count(metrics, "input_batches"),
        input_rows: metric_count(metrics, "input_rows"),
        peak_device_memory_bytes: metric_optional_count(metrics, "peak_device_memory_bytes"),
        resident_dataframe_peak_bytes: metric_optional_count(metrics, "resident_dataframe_peak_bytes"),
        allocator_high_water_bytes: metric_optional_count(metrics, "allocator_high_water_bytes"),
        reservation_requested_bytes: metric_count(metrics, "reservation_requested_bytes"),
        reservation_granted_bytes: metric_count(metrics, "reservation_granted_bytes"),
        reservation_denied_count: metric_count(metrics, "reservation_denied_count"),
        source_chunks_planned: metric_count(metrics, "source_chunks_planned"),
        source_chunks_attempted: metric_count(metrics, "source_chunks_attempted"),
        source_chunks_committed: metric_count(metrics, "source_chunks_committed"),
        source_chunks_retried: metric_count(metrics, "source_chunks_retried"),
        row_groups_planned: metric_count(metrics, "row_groups_planned"),
        row_groups_committed: metric_count(metrics, "row_groups_committed"),
        partial_aggregate_count: metric_count(metrics, "partial_aggregate_count"),
        partial_aggregate_rows: metric_count(metrics, "partial_aggregate_rows"),
        final_merge_attempts: metric_count(metrics, "final_merge_attempts"),
        final_merge_levels: metric_count(metrics, "final_merge_levels"),
        retry_count: metric_count(metrics, "retry_count"),
        retry_resource_exhausted_count: metric_count(metrics, "retry_resource_exhausted_count"),
        retry_reservation_failure_count: metric_count(metrics, "retry_reservation_failure_count"),
        host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
        host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
        host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
        device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
        device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
        device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
        parquet_files: metric_count(metrics, "parquet_files"),
        parquet_bytes_known: metric_count(metrics, "parquet_bytes_known"),
        row_groups_selected: metric_count(metrics, "row_groups_selected"),
        read_columns: metric_count(metrics, "read_columns"),
        parquet_read_time: metric_duration(metrics, "parquet_read_time"),
        record_batch_upload_time: Duration::ZERO,
        cudf_operator_time: metric_duration(metrics, "operator_time"),
        record_batch_materialization_time: metric_duration(metrics, "materialization_time"),
        elapsed_compute: metrics.elapsed_compute().map(duration_from_nanos),
        output_rows: metrics.output_rows(),
    }
}

fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

fn metric_optional_count(metrics: &MetricsSet, name: &str) -> Option<usize> {
    metrics.sum_by_name(name).map(|value| value.as_usize())
}

fn metric_duration(metrics: &MetricsSet, name: &str) -> Duration {
    metrics.sum_by_name(name).map(|value| duration_from_nanos(value.as_usize())).unwrap_or(Duration::ZERO)
}

fn duration_from_nanos(nanos: usize) -> Duration {
    Duration::from_nanos(nanos.min(u64::MAX as usize) as u64)
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

fn native_datafusion_session_config() -> SessionConfig {
    let mut config = benchmark_session_config().with_repartition_aggregations(false).with_repartition_sorts(false);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
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
    datafusion_native_exec: &TimedBenchmarkRun,
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

    let native_speedup = speedup(cpu_comparable.mean, datafusion_native_exec.mean);
    if native_speedup <= config.minimum_native_gpu_speedup {
        return Err(plan_error(format!(
            "Phase B runtime gate did not preserve the DataFusion native exec win: datafusion_native_exec_speedup={native_speedup:.3}, minimum_native_gpu_speedup={:.3}, cpu_comparable_wall_ms={:.3}, datafusion_native_exec_wall_ms={:.3}",
            config.minimum_native_gpu_speedup,
            duration_ms(cpu_comparable.mean),
            duration_ms(datafusion_native_exec.mean)
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

fn validate_datafusion_native_exec_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if !measurement.plan_one_line.contains("NativeCudfExec") {
        return Err(plan_error(format!(
            "datafusion_native_exec benchmark did not execute through NativeCudfExec: {}",
            measurement.plan_one_line
        )));
    }
    let Some(report) = &measurement.native_datafusion_planning_report else {
        return Err(plan_error("datafusion_native_exec benchmark did not retain a native planning report"));
    };
    if report.accepted_count() != 1 || report.rejected_count() != 0 {
        return Err(plan_error(format!(
            "datafusion_native_exec planning did not accept exactly one native plan: accepted_count={}, rejected_count={}",
            report.accepted_count(),
            report.rejected_count()
        )));
    }
    let row = report.rows.first().ok_or_else(|| plan_error("datafusion_native_exec planning report was empty"))?;
    if row.post_native_plan_root != "NativeCudfExec" || row.post_native_execution_path.as_str() != "native" {
        return Err(plan_error(format!(
            "datafusion_native_exec planning did not route to native execution: root={}, path={}",
            row.post_native_plan_root,
            row.post_native_execution_path.as_str()
        )));
    }
    validate_native_direct_metrics("datafusion_native_exec", measurement, parquet_file_bytes, config)
}

fn validate_manual_native_lowering_measurement(
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if !measurement.plan_one_line.contains("LocalParquetSource")
        || !measurement.plan_one_line.contains("format=Parquet")
        || !measurement.plan_one_line.contains("direct_device_read=true")
        || !measurement.plan_one_line.contains("sink=host_arrow")
    {
        return Err(plan_error(format!(
            "manual_native_lowering plan did not lower to the direct local Parquet host-arrow path: {}",
            measurement.plan_one_line
        )));
    }
    validate_native_direct_metrics("manual_native_lowering", measurement, parquet_file_bytes, config)
}

fn validate_native_direct_metrics(
    label: &str,
    measurement: &GpuCollectMeasurement,
    parquet_file_bytes: u64,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let metrics = &measurement.metrics;
    if metrics.host_to_device_uploads != 0 || metrics.host_to_device_uploaded_rows != 0 {
        return Err(plan_error(format!(
            "{label} direct Parquet path crossed host-to-device upload boundary: uploads={}, uploaded_rows={}",
            metrics.host_to_device_uploads, metrics.host_to_device_uploaded_rows
        )));
    }
    if metrics.device_to_host_materializations != 1 {
        return Err(plan_error(format!(
            "{label} direct Parquet path expected one final host materialization, got {}",
            metrics.device_to_host_materializations
        )));
    }
    if metrics.device_to_host_materialized_rows != measurement.output_rows {
        return Err(plan_error(format!(
            "{label} materialized row metrics did not match output rows: materialized_rows={}, output_rows={}",
            metrics.device_to_host_materialized_rows, measurement.output_rows
        )));
    }
    let Some(peak_device_memory_bytes) = metrics.peak_device_memory_bytes else {
        return Err(plan_error(format!("{label} direct Parquet path did not record peak device memory")));
    };
    if peak_device_memory_bytes == 0 {
        return Err(plan_error(format!("{label} direct Parquet path recorded zero peak device memory")));
    }
    if measurement.output_rows > config.limit {
        return Err(plan_error(format!(
            "{label} direct Parquet path returned too many rows: output_rows={}, limit={}",
            measurement.output_rows, config.limit
        )));
    }
    if metrics.parquet_files != config.file_count {
        return Err(plan_error(format!(
            "{label} direct Parquet path expected {} files, got {}",
            config.file_count, metrics.parquet_files
        )));
    }
    if metrics.parquet_bytes_known < parquet_file_bytes as usize
        || metrics.parquet_bytes_known < config.minimum_file_bytes as usize
    {
        return Err(plan_error(format!(
            "{label} direct Parquet metrics did not account for the benchmark file size: parquet_bytes_known={}, parquet_file_bytes={}, minimum_file_bytes={}",
            metrics.parquet_bytes_known, parquet_file_bytes, config.minimum_file_bytes
        )));
    }
    if measurement.planning.rejected_count != 0 {
        return Err(plan_error(format!(
            "{label} lowering rejected fallback candidates: rejected_count={}",
            measurement.planning.rejected_count
        )));
    }

    let mut actual = measurement.read_columns.clone();
    actual.sort();
    let expected = config.expected_read_columns();
    if actual != expected {
        return Err(plan_error(format!(
            "{label} direct Parquet read columns did not match workload requirements: actual={}, expected={}",
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

fn forced_resource_failure_policy() -> NativeMemoryControlPolicy {
    NativeMemoryControlPolicy::Bounded {
        device_budget_bytes: 1,
        max_source_chunk_bytes: 1,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
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
            0 => vec![
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::ManualNativeLowering,
            ],
            _ => vec![
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
            ],
        },
        BenchmarkPathSet::All => match iteration % 4 {
            0 => vec![
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::GpuDirect,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::ManualNativeLowering,
            ],
            1 => vec![
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::GpuDirect,
                BenchmarkPath::CpuComparable,
                BenchmarkPath::CpuDefault,
            ],
            2 => vec![
                BenchmarkPath::GpuDirect,
                BenchmarkPath::DataFusionNativeExec,
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
            ],
            _ => vec![
                BenchmarkPath::CpuDefault,
                BenchmarkPath::CpuComparable,
                BenchmarkPath::ManualNativeLowering,
                BenchmarkPath::DataFusionNativeExec,
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

fn sanitize_report_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn tsv_escape(value: &str) -> String {
    sanitize_report_value(value)
}

fn markdown_table_cell(value: &str) -> String {
    sanitize_report_value(value).replace('|', ";")
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
                ("CUDF_PARQUET_BENCH_ROWS", "52000000".to_string()),
                ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", TEN_GIB.to_string()),
                ("CUDF_PARQUET_BENCH_COMPRESSION", "snappy".to_string()),
                ("CUDF_PARQUET_BENCH_FILES", "4".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", "direct_friendly".to_string()),
                ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", "resource_failure".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_DEVICE_BUDGET_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_RETRY_ATTEMPTS", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS", "1".to_string()),
            ]
        );

        let mut base = test_config();
        base.benchmark_paths = BenchmarkPathSet::All;
        let configured = case.configured_case(&base, PathBuf::from("target/phase-c-matrix-data"));
        assert_eq!(configured.input_rows, 52_000_000);
        assert_eq!(configured.minimum_file_bytes, TEN_GIB);
        assert_eq!(configured.file_count, 4);
        assert_eq!(configured.benchmark_paths, BenchmarkPathSet::NativeOnly);
        assert_eq!(configured.native_memory_control_policy, forced_resource_failure_policy());
    }

    #[test]
    fn expected_resource_failure_report_includes_resource_error_facts() {
        let error = AdapterError::new(
            datafusion_cudf::ErrorCode::MemoryReservationFailed,
            "datafusion_cudf.native_exec.execute",
            "native cudf query execution failed",
        )
        .with_fact("native_code", cudf_query_engine::QueryErrorCode::MemoryReservationFailed)
        .with_fact("native_kind", "resource_exhausted")
        .with_fact("native_status", "temporary")
        .with_fact("native_operation", "cudf_query_engine.memory.reserve")
        .with_fact("native_datafusion_phase", "runtime_execution")
        .with_fact("cpu_fallback_after_native_start", false)
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
            final_plan_root: "NativeCudfExec".to_string(),
            planning_report: test_native_datafusion_planning_report(),
            native_candidate_shapes: vec!["accepted_replaced:LocalLimitExec(SortExec(...)):none".to_string()],
            source_physical_plan_one_line: "SortExec(fetch=100)".to_string(),
            source_diagnostic: "files=1".to_string(),
            error: AdapterRuntimeErrorReport::from_adapter_error(&error),
        };

        let text = report.to_text();

        assert!(text.contains("case=phase_c_expected_resource_failure"));
        assert!(text.contains("phase_c_runtime_gate=expected_resource_failure"));
        assert!(text.contains("no_silent_cpu_fallback=true"));
        assert!(text.contains("native_runtime_executed=true"));
        assert!(text.contains("datafusion_native_exec_planning_outcome=accepted_replaced"));
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

    #[tokio::test]
    async fn supported_sql_benchmark_plan_reports_datafusion_native_accepted() {
        let dir = unique_benchmark_dir(&std::env::temp_dir());
        let config = test_config();
        let path = benchmark_input_path(&dir, &config);
        write_benchmark_parquet_dataset(&path, &config).unwrap();

        let planned = plan_datafusion_native_sql(&path, &benchmark_sql(&config), &config).await.unwrap();

        assert_eq!(planned.plan.name(), "NativeCudfExec");
        assert!(planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
        assert_eq!(planned.planning_report.accepted_count(), 1);
        assert_eq!(planned.planning_report.rejected_count(), 0);
        let row = &planned.planning_report.rows[0];
        assert_eq!(row.outcome.as_str(), "accepted_replaced");
        assert_eq!(row.post_native_plan_root, "NativeCudfExec");
        assert_eq!(row.post_native_execution_path.as_str(), "native");
        assert!(!row.rejected_before_execution);
        assert!(row.source_diagnostics.iter().any(|diagnostic| diagnostic.direct_device_read));

        fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn rejected_sql_benchmark_plan_reports_reason_without_native_runtime() {
        let dir = unique_benchmark_dir(&std::env::temp_dir());
        let config = test_config();
        let path = benchmark_input_path(&dir, &config);
        write_benchmark_parquet_dataset(&path, &config).unwrap();

        let planned = plan_datafusion_native_sql(&path, rejected_benchmark_sql(), &config).await.unwrap();

        assert_ne!(planned.plan.name(), "NativeCudfExec");
        assert!(planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_none());
        assert_eq!(planned.planning_report.accepted_count(), 0);
        assert_eq!(planned.planning_report.rejected_count(), 1);
        let row = &planned.planning_report.rows[0];
        assert_eq!(row.outcome.as_str(), "rejected_before_execution");
        assert!(row.reason_code.is_some());
        assert!(row.rejected_before_execution);
        assert_ne!(row.post_native_execution_path.as_str(), "native");
        assert!(planned.plan.metrics().is_none_or(|metrics| metrics.sum_by_name("host_to_device_uploads").is_none()));

        let mut rejected_config = config.clone();
        rejected_config.expected_outcome = BenchmarkExpectedOutcome::PlanningRejected;
        let report = run_expected_planning_rejection_probe(
            &path,
            parquet_dataset_bytes(&path).unwrap(),
            parquet_dataset_layout(&path).unwrap(),
            "files=4".to_string(),
            &rejected_config,
        )
        .await
        .unwrap();
        let text = report.to_text();
        assert!(text.contains("case=datafusion_native_exec_planning_rejected"));
        assert!(text.contains("native_runtime_executed=false"));
        assert!(text.contains("datafusion_native_exec_planning_outcome=rejected_before_execution"));
        assert!(text.contains("datafusion_native_exec_planning_reason_code="));
        assert!(text.contains("datafusion_native_exec_planning_rejected_before_execution=true"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn direct_parquet_benchmark_output_names_primary_datafusion_native_path() {
        let config = test_config();
        let metrics = test_native_metrics();
        let planning = LargerIslandBenchmarkPlanning::from_report(&datafusion_cudf::PlanningReport::from_outcomes([
            CandidateOutcome::Rewritten {
                candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                    "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))",
                    2,
                    4,
                ),
            },
        ]));
        let datafusion_planning_report = NativeDataFusionPlanningReport {
            rows: vec![datafusion_cudf::NativeDataFusionPlanningRow {
                query_id: "local_parquet_benchmark".to_string(),
                physical_root: "SortExec".to_string(),
                physical_shape: "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))"
                    .to_string(),
                native_candidate_shape: Some(
                    "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))".to_string(),
                ),
                outcome: datafusion_cudf::NativeDataFusionPlanningOutcome::AcceptedReplaced,
                reason_code: None,
                unsupported_category: None,
                source_diagnostics: vec![datafusion_cudf::NativeSourceDiagnosticSummary {
                    format: "Parquet".to_string(),
                    direct_device_read: true,
                    file_count: config.file_count,
                    ranged_file_count: 0,
                    known_bytes: Some(4096),
                    selected_row_groups: Some(2),
                    predicate_disposition: "pruning_only".to_string(),
                    byte_range_disposition: "full_files".to_string(),
                    partition_column_disposition: "absent".to_string(),
                    zero_column_count_disposition: "not_required".to_string(),
                    schema_mapping_disposition: "direct_by_name".to_string(),
                    read_column_count: 4,
                    projected_output_column_count: 3,
                    pure_filter_column_count: 1,
                    rejected_features: Vec::new(),
                }],
                post_native_plan_root: "NativeCudfExec".to_string(),
                post_native_execution_path: datafusion_cudf::NativeDataFusionPostNativeExecutionPath::Native,
                rejected_before_execution: false,
            }],
        };
        let report = DirectParquetBenchmarkReport {
            parquet_file_bytes: 4096,
            parquet_row_groups: 4,
            input_rows: config.input_rows,
            file_count: config.file_count,
            measure_columns: config.measure_columns,
            batch_size: config.batch_size,
            row_group_size: config.row_group_size,
            group_count: config.group_count,
            warmup_iterations: 0,
            iterations: 1,
            limit: config.limit,
            minimum_native_gpu_speedup: config.minimum_native_gpu_speedup,
            benchmark_paths: config.benchmark_paths,
            compression: config.compression,
            native_plan_shape: config.native_plan_shape,
            expected_outcome: config.expected_outcome,
            measured_run_order: measured_run_order_summary(1, config.benchmark_paths),
            cpu_comparable: TimedBenchmarkRun::from_samples([Duration::from_millis(100)]),
            cpu_default: TimedBenchmarkRun::from_samples([Duration::from_millis(110)]),
            gpu: None,
            datafusion_native_exec: TimedBenchmarkRun::from_samples([Duration::from_millis(50)]),
            manual_native_lowering: TimedBenchmarkRun::from_samples([Duration::from_millis(55)]),
            cpu_comparable_samples: vec![Duration::from_millis(100)],
            cpu_default_samples: vec![Duration::from_millis(110)],
            gpu_samples: Vec::new(),
            datafusion_native_exec_samples: vec![Duration::from_millis(50)],
            manual_native_lowering_samples: vec![Duration::from_millis(55)],
            gpu_metric_samples: Vec::new(),
            datafusion_native_exec_metric_samples: vec![metrics.clone()],
            manual_native_lowering_metric_samples: vec![metrics.clone()],
            final_cpu_comparable: test_collect_measurement(Duration::from_millis(100)),
            final_cpu_default: test_collect_measurement(Duration::from_millis(110)),
            final_gpu: None,
            final_datafusion_native_exec: test_gpu_measurement(
                Duration::from_millis(50),
                metrics.clone(),
                planning.clone(),
                Some(datafusion_planning_report),
                "NativeCudfExec: native_nodes=5",
                Some("SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))"),
                config.expected_read_columns(),
            ),
            final_manual_native_lowering: test_gpu_measurement(
                Duration::from_millis(55),
                metrics,
                planning,
                None,
                "LocalParquetSource: format=Parquet, direct_device_read=true, sink=host_arrow",
                Some("SortExec(fetch)"),
                config.expected_read_columns(),
            ),
            source_diagnostic: "files=4".to_string(),
        };

        let text = report.to_text();
        assert!(text.contains("cpu_datafusion_wall_ms=110.000"));
        assert!(text.contains("datafusion_native_exec_wall_ms=50.000"));
        assert!(text.contains("manual_native_lowering_wall_ms=55.000"));
        assert!(text.contains("datafusion_native_exec_planning_outcome=accepted_replaced"));
        assert!(text.contains("datafusion_native_exec_planning_post_native_execution_path=native"));
        assert!(text.contains("datafusion_native_exec_planning_source_diagnostics=format=Parquet"));
        assert!(text.contains("read_columns=4,projected_output_columns=3,pure_filter_columns=1"));
        assert!(text.contains("byte_range=full_files,partition_columns=absent,zero_column_count=not_required"));
        let zero_column_count_artifact_text =
            native_source_diagnostics_text(&[datafusion_cudf::NativeSourceDiagnosticSummary {
                format: "Parquet".to_string(),
                direct_device_read: true,
                file_count: 1,
                ranged_file_count: 0,
                known_bytes: Some(4096),
                selected_row_groups: Some(2),
                predicate_disposition: "absent".to_string(),
                byte_range_disposition: "full_files".to_string(),
                partition_column_disposition: "absent".to_string(),
                zero_column_count_disposition: "represented".to_string(),
                schema_mapping_disposition: "direct_by_name".to_string(),
                read_column_count: 0,
                projected_output_column_count: 0,
                pure_filter_column_count: 0,
                rejected_features: Vec::new(),
            }]);
        assert!(zero_column_count_artifact_text.contains("read_columns=0"));
        assert!(zero_column_count_artifact_text.contains("zero_column_count=represented"));
        assert_key_order(&text, "cpu_datafusion_wall_ms", "datafusion_native_exec_wall_ms");
        assert_key_order(&text, "datafusion_native_exec_wall_ms", "manual_native_lowering_wall_ms");
        assert_key_order(&text, "datafusion_native_exec_planning_accepted_count", "manual_native_lowering_plan");

        let matrix_case = PhaseCMatrixCase {
            scale: BenchmarkScalePoint::ThreeGiB,
            compression: BenchmarkCompression::Uncompressed,
            source_layout: BenchmarkSourceLayout::MultiFile,
            native_plan_shape: BenchmarkPlanShape::DefaultSql,
            expected_outcome: BenchmarkExpectedOutcome::Success,
        };
        let matrix_row =
            PhaseCMatrixReportRow::from_success(matrix_case, &report, Path::new("target/phase-c-matrix/logs/case.log"));
        assert!(matrix_row.matches_expected_outcome());
        assert_eq!(matrix_row.outcome.as_str(), "pass");
        assert_eq!(matrix_row.planning_outcome, "accepted_replaced");
        assert!(matrix_row.planning_source_diagnostics.contains("read_columns=4"));
        assert!(matrix_row.planning_source_diagnostics.contains("schema_mapping=direct_by_name"));
        assert_eq!(matrix_row.native_cudf_exec_accepted_count, "1");
        assert_eq!(matrix_row.zero_host_to_device_uploads, "true");
        assert_eq!(matrix_row.one_device_to_host_materialization, "true");

        let matrix_report = PhaseCMatrixReport {
            artifact_dir: PathBuf::from("target/phase-c-matrix"),
            tsv_path: PathBuf::from("target/phase-c-matrix/datafusion-native-exec-phase-c-matrix.tsv"),
            json_path: PathBuf::from("target/phase-c-matrix/datafusion-native-exec-phase-c-matrix.json"),
            markdown_path: PathBuf::from("target/phase-c-matrix/datafusion-native-exec-phase-c-matrix.md"),
            rows: vec![matrix_row],
        };
        let tsv = matrix_report.to_tsv();
        assert!(tsv.starts_with(PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_HEADER));
        assert!(tsv.contains("\tpass\taccepted_replaced\tnone\tformat=Parquet"));
        assert!(tsv.contains("read_columns=4,projected_output_columns=3,pure_filter_columns=1"));
        assert!(tsv.contains("\t1\ttrue\ttrue\ttrue\ttrue\t"));
        let json = matrix_report.to_json();
        assert!(json.contains("\"schema_version\": 2"));
        assert!(json.contains("\"case_id\": \"3gib-uncompressed-multi_file-default_sql-success\""));
        assert!(json.contains("\"planning_source_diagnostics\": \"format=Parquet"));
        assert!(json.contains("\"zero_host_to_device_uploads\": \"true\""));
        let markdown = matrix_report.to_markdown();
        assert!(markdown.contains("Decision gate: **pass**"));
        assert!(markdown.contains("3gib-uncompressed-multi_file-default_sql-success"));
        assert!(markdown.contains("schema_mapping=direct_by_name"));
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

    fn test_collect_measurement(wall_time: Duration) -> CollectMeasurement {
        CollectMeasurement { wall_time, output_batches: 1, output_rows: 2, output_bytes: 64, batches: Vec::new() }
    }

    fn test_gpu_measurement(
        wall_time: Duration,
        metrics: LargerIslandBenchmarkMetrics,
        planning: LargerIslandBenchmarkPlanning,
        native_datafusion_planning_report: Option<NativeDataFusionPlanningReport>,
        plan_one_line: &str,
        source_physical_plan_one_line: Option<&str>,
        read_columns: Vec<String>,
    ) -> GpuCollectMeasurement {
        GpuCollectMeasurement {
            wall_time,
            output_batches: 1,
            output_rows: 2,
            output_bytes: 64,
            metrics,
            planning,
            native_datafusion_planning_report,
            plan_one_line: plan_one_line.to_string(),
            source_physical_plan_one_line: source_physical_plan_one_line.map(str::to_string),
            native_candidate_shapes: vec![
                "accepted_replaced:SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))):none"
                    .to_string(),
            ],
            read_columns,
            batches: Vec::new(),
        }
    }

    fn test_native_datafusion_planning_report() -> NativeDataFusionPlanningReport {
        NativeDataFusionPlanningReport {
            rows: vec![datafusion_cudf::NativeDataFusionPlanningRow {
                query_id: "local_parquet_benchmark".to_string(),
                physical_root: "SortExec".to_string(),
                physical_shape: "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))"
                    .to_string(),
                native_candidate_shape: Some(
                    "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))".to_string(),
                ),
                outcome: datafusion_cudf::NativeDataFusionPlanningOutcome::AcceptedReplaced,
                reason_code: None,
                unsupported_category: None,
                source_diagnostics: vec![datafusion_cudf::NativeSourceDiagnosticSummary {
                    format: "Parquet".to_string(),
                    direct_device_read: true,
                    file_count: 1,
                    ranged_file_count: 0,
                    known_bytes: Some(4096),
                    selected_row_groups: Some(2),
                    predicate_disposition: "pruning_only".to_string(),
                    byte_range_disposition: "full_files".to_string(),
                    partition_column_disposition: "absent".to_string(),
                    zero_column_count_disposition: "not_required".to_string(),
                    schema_mapping_disposition: "direct_by_name".to_string(),
                    read_column_count: 4,
                    projected_output_column_count: 3,
                    pure_filter_column_count: 1,
                    rejected_features: Vec::new(),
                }],
                post_native_plan_root: "NativeCudfExec".to_string(),
                post_native_execution_path: datafusion_cudf::NativeDataFusionPostNativeExecutionPath::Native,
                rejected_before_execution: false,
            }],
        }
    }

    fn test_native_metrics() -> LargerIslandBenchmarkMetrics {
        LargerIslandBenchmarkMetrics {
            phase_c_memory_control_policy: "bounded",
            phase_c_runtime_gate: "pass",
            input_batches: 1,
            input_rows: 16,
            resident_dataframe_peak_bytes: Some(1024),
            allocator_high_water_bytes: Some(2048),
            reservation_requested_bytes: 1024,
            reservation_granted_bytes: 1024,
            reservation_denied_count: 0,
            source_chunks_planned: 2,
            source_chunks_attempted: 2,
            source_chunks_committed: 2,
            source_chunks_retried: 0,
            row_groups_planned: 2,
            row_groups_committed: 2,
            partial_aggregate_count: 2,
            partial_aggregate_rows: 4,
            final_merge_attempts: 1,
            final_merge_levels: 1,
            retry_count: 0,
            retry_resource_exhausted_count: 0,
            retry_reservation_failure_count: 0,
            peak_device_memory_bytes: Some(1024),
            host_to_device_uploads: 0,
            host_to_device_uploaded_rows: 0,
            host_to_device_uploaded_bytes: 0,
            device_to_host_materializations: 1,
            device_to_host_materialized_rows: 2,
            device_to_host_materialized_bytes: 64,
            parquet_files: 4,
            parquet_bytes_known: 4096,
            row_groups_selected: 2,
            read_columns: 4,
            parquet_read_time: Duration::from_millis(3),
            record_batch_upload_time: Duration::ZERO,
            cudf_operator_time: Duration::from_millis(4),
            record_batch_materialization_time: Duration::from_millis(1),
            elapsed_compute: Some(Duration::from_millis(8)),
            output_rows: Some(2),
        }
    }

    fn assert_key_order(text: &str, before: &str, after: &str) {
        let before_index = text.find(before).unwrap_or_else(|| panic!("missing key {before}"));
        let after_index = text.find(after).unwrap_or_else(|| panic!("missing key {after}"));
        assert!(before_index < after_index, "expected {before} to appear before {after}");
    }
}
