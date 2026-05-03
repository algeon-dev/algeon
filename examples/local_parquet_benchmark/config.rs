use super::*;

pub(super) const THREE_GIB: u64 = 3_221_225_472;
pub(super) const DEFAULT_NATIVE_DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub(super) const DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
pub(super) const DEFAULT_ROW_GROUP_SIZE: usize = 1_048_576;
pub(super) const DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
pub(super) const DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS: usize = 4;
pub(super) const DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;
pub(super) const DEFAULT_MIN_NATIVE_GPU_SPEEDUP: f64 = 1.0;
pub(super) const DEFAULT_BENCHMARK_DATA_DIR: &str = "tests/data/local-parquet-benchmark";
pub(super) const DEFAULT_PHASE_C_MATRIX_DATA_DIR: &str = "tests/data/phase-c-matrix";
pub(super) const SIX_GIB: u64 = 6 * 1024 * 1024 * 1024;
pub(super) const TEN_GIB: u64 = 10 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct LocalParquetBenchmarkConfig {
    pub(super) input_rows: usize,
    pub(super) file_count: usize,
    pub(super) measure_columns: usize,
    pub(super) batch_size: usize,
    pub(super) row_group_size: usize,
    pub(super) group_count: usize,
    pub(super) limit: usize,
    pub(super) warmup_iterations: usize,
    pub(super) iterations: usize,
    pub(super) minimum_file_bytes: u64,
    pub(super) minimum_native_gpu_speedup: f64,
    pub(super) native_memory_control_policy: NativeMemoryControlPolicy,
    pub(super) benchmark_paths: BenchmarkPathSet,
    pub(super) compression: BenchmarkCompression,
    pub(super) native_plan_shape: BenchmarkPlanShape,
    pub(super) expected_outcome: BenchmarkExpectedOutcome,
    pub(super) output_dir: Option<PathBuf>,
}

impl LocalParquetBenchmarkConfig {
    pub(super) fn from_env() -> DataFusionResult<Self> {
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

    pub(super) fn validate(&self) -> DataFusionResult<()> {
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

    pub(super) fn projected_measure_pairs(&self) -> usize {
        self.measure_columns / 2
    }

    pub(super) fn expected_read_columns(&self) -> Vec<String> {
        let mut columns = Vec::with_capacity(self.measure_columns + 2);
        columns.push("key".to_string());
        columns.push("payload".to_string());
        columns.extend((0..self.measure_columns).map(measure_column_name));
        columns.sort();
        columns
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkCompression {
    Uncompressed,
    Snappy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkPlanShape {
    DefaultSql,
    DirectFriendly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkExpectedOutcome {
    Success,
    PlanningRejected,
    ResourceFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkRunMode {
    SingleCase,
    PhaseCMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkPathSet {
    All,
    NativeOnly,
}

impl BenchmarkPathSet {
    pub(super) fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::All);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "all" => Ok(Self::All),
            "native_only" | "native-only" => Ok(Self::NativeOnly),
            other => Err(plan_error(format!("invalid {name}: {other}; expected all or native_only"))),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::NativeOnly => "native_only",
        }
    }

    pub(super) fn includes_gpu_direct(self) -> bool {
        matches!(self, Self::All)
    }
}

impl BenchmarkCompression {
    pub(super) fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::Uncompressed);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "uncompressed" | "none" => Ok(Self::Uncompressed),
            "snappy" => Ok(Self::Snappy),
            other => Err(plan_error(format!("invalid {name}: {other}; expected uncompressed or snappy"))),
        }
    }

    pub(super) fn parquet(self) -> Compression {
        match self {
            Self::Uncompressed => Compression::UNCOMPRESSED,
            Self::Snappy => Compression::SNAPPY,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Uncompressed => "uncompressed",
            Self::Snappy => "snappy",
        }
    }
}

impl BenchmarkPlanShape {
    pub(super) fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let Some(value) = std::env::var_os(name) else {
            return Ok(Self::DefaultSql);
        };
        match value.to_string_lossy().to_ascii_lowercase().as_str() {
            "default_sql" | "default-sql" => Ok(Self::DefaultSql),
            "direct_friendly" | "direct-friendly" => Ok(Self::DirectFriendly),
            other => Err(plan_error(format!("invalid {name}: {other}; expected default_sql or direct_friendly"))),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::DefaultSql => "default_sql",
            Self::DirectFriendly => "direct_friendly",
        }
    }
}

impl BenchmarkExpectedOutcome {
    pub(super) fn from_env(name: &'static str) -> DataFusionResult<Self> {
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

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::PlanningRejected => "planning_rejected",
            Self::ResourceFailure => "resource_failure",
        }
    }
}

impl BenchmarkRunMode {
    pub(super) fn from_env(name: &'static str) -> DataFusionResult<Self> {
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

pub(super) fn env_usize(name: &'static str, default: usize) -> DataFusionResult<usize> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<usize>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

pub(super) fn env_u64(name: &'static str, default: u64) -> DataFusionResult<u64> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<u64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

pub(super) fn env_f64(name: &'static str, default: f64) -> DataFusionResult<f64> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value.to_string_lossy().parse::<f64>().map_err(|error| plan_error(format!("invalid {name}: {error}")))
}

pub(super) fn native_memory_control_policy_from_env() -> DataFusionResult<NativeMemoryControlPolicy> {
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

pub(super) fn forced_resource_failure_policy() -> NativeMemoryControlPolicy {
    NativeMemoryControlPolicy::Bounded {
        device_budget_bytes: 1,
        max_source_chunk_bytes: 1,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    }
}

pub(super) fn validate_nonzero(name: &'static str, value: usize) -> DataFusionResult<()> {
    if value == 0 {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}

pub(super) fn validate_native_memory_control_policy(policy: NativeMemoryControlPolicy) -> DataFusionResult<()> {
    policy.validate().map_err(native_query_error)
}

pub(super) fn validate_positive_f64(name: &'static str, value: f64) -> DataFusionResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(plan_error(format!("{name} must be finite and greater than zero, got {value}")));
    }
    Ok(())
}
