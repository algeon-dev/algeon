use super::*;

pub(super) const DEFAULT_NATIVE_DEVICE_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub(super) const DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
pub(super) const DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
pub(super) const DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS: usize = 4;
pub(super) const DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NativeTpchBenchmarkConfig {
    pub(super) query_ids: Vec<String>,
    pub(super) warmup_iterations: usize,
    pub(super) iterations: usize,
    pub(super) dataset_mode: NativeTpchDatasetMode,
    pub(super) native_memory_policy: NativeTpchMemoryPolicy,
}

impl NativeTpchBenchmarkConfig {
    pub(super) fn from_env() -> DataFusionResult<Self> {
        let config = Self {
            query_ids: query_ids_from_env("CUDF_TPCH_BENCH_QUERY_IDS")?,
            warmup_iterations: env_usize("CUDF_TPCH_BENCH_WARMUPS", 0)?,
            iterations: env_usize("CUDF_TPCH_BENCH_ITERATIONS", 1)?,
            dataset_mode: NativeTpchDatasetMode::from_env("CUDF_TPCH_BENCH_DATASET_MODE")?,
            native_memory_policy: NativeTpchMemoryPolicy::from_env("CUDF_TPCH_BENCH_NATIVE_MEMORY_POLICY")?,
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate(&self) -> DataFusionResult<()> {
        validate_nonzero("CUDF_TPCH_BENCH_ITERATIONS", self.iterations)?;
        validate_query_ids(&self.query_ids)?;
        self.native_memory_policy.validate()
    }

    pub(super) fn selected_queries(&self) -> Vec<&'static TpchQuery> {
        self.query_ids
            .iter()
            .map(|query_id| {
                TPCH_QUERIES
                    .iter()
                    .find(|query| query.id == query_id)
                    .expect("validated benchmark config references only known TPC-H query ids")
            })
            .collect()
    }

    pub(super) fn native_execution_config(&self) -> NativeCudfExecutionConfig {
        match self.native_memory_policy {
            NativeTpchMemoryPolicy::Unbounded => NativeCudfExecutionConfig::unbounded(),
            NativeTpchMemoryPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            } => NativeCudfExecutionConfig::bounded(
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            ),
        }
    }
}

impl Default for NativeTpchBenchmarkConfig {
    fn default() -> Self {
        Self {
            query_ids: TPCH_QUERIES.iter().map(|query| query.id.to_string()).collect(),
            warmup_iterations: 0,
            iterations: 1,
            dataset_mode: NativeTpchDatasetMode::Sf1External,
            native_memory_policy: NativeTpchMemoryPolicy::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchDatasetMode {
    Sf1External,
    AllowTiny,
}

impl NativeTpchDatasetMode {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let mode = match std::env::var_os(name) {
            None => "sf1_external".to_string(),
            Some(value) => value.to_string_lossy().to_ascii_lowercase(),
        };
        match mode.as_str() {
            "sf1_external" | "external" | "sf1" => Ok(Self::Sf1External),
            "allow_tiny" | "dev_tiny" | "tiny" => Ok(Self::AllowTiny),
            other => Err(plan_error(format!("invalid {name}: {other}; expected sf1_external or allow_tiny"))),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Sf1External => "sf1_external",
            Self::AllowTiny => "allow_tiny",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeTpchMemoryPolicy {
    Unbounded,
    Bounded {
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    },
}

impl NativeTpchMemoryPolicy {
    fn from_env(name: &'static str) -> DataFusionResult<Self> {
        let policy = match std::env::var_os(name) {
            None => "bounded".to_string(),
            Some(value) => value.to_string_lossy().to_ascii_lowercase(),
        };
        match policy.as_str() {
            "unbounded" => Ok(Self::Unbounded),
            "bounded" => Ok(Self::Bounded {
                device_budget_bytes: env_u64(
                    "CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES",
                    DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
                )?,
                max_source_chunk_bytes: env_u64(
                    "CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES",
                    DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
                )?,
                max_row_groups_per_chunk: env_usize(
                    "CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK",
                    DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
                )?,
                max_retry_attempts: env_usize(
                    "CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS",
                    DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
                )?,
                min_retry_chunk_row_groups: env_usize(
                    "CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS",
                    DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
                )?,
            }),
            other => Err(plan_error(format!("invalid {name}: {other}; expected bounded or unbounded"))),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::Bounded { .. } => "bounded",
        }
    }

    fn validate(self) -> DataFusionResult<()> {
        let Self::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = self
        else {
            return Ok(());
        };
        validate_nonzero("CUDF_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES", device_budget_bytes)?;
        validate_nonzero("CUDF_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", max_source_chunk_bytes)?;
        validate_nonzero("CUDF_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", max_row_groups_per_chunk)?;
        validate_nonzero("CUDF_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS", max_retry_attempts)?;
        validate_nonzero("CUDF_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS", min_retry_chunk_row_groups)?;
        Ok(())
    }
}

impl Default for NativeTpchMemoryPolicy {
    fn default() -> Self {
        Self::Bounded {
            device_budget_bytes: DEFAULT_NATIVE_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
        }
    }
}

fn query_ids_from_env(name: &'static str) -> DataFusionResult<Vec<String>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(TPCH_QUERIES.iter().map(|query| query.id.to_string()).collect());
    };
    let query_ids = value
        .to_string_lossy()
        .split(',')
        .map(str::trim)
        .filter(|query_id| !query_id.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if query_ids.is_empty() {
        return Err(plan_error(format!("{name} must name at least one query id")));
    }
    Ok(query_ids)
}

fn validate_query_ids(query_ids: &[String]) -> DataFusionResult<()> {
    if query_ids.is_empty() {
        return Err(plan_error("CUDF_TPCH_BENCH_QUERY_IDS must name at least one query id"));
    }

    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    for query_id in query_ids {
        if !known.contains(query_id.as_str()) {
            return Err(plan_error(format!("unknown TPC-H query id {query_id}")));
        }
        if !seen.insert(query_id.as_str()) {
            return Err(plan_error(format!("duplicate TPC-H query id {query_id}")));
        }
    }
    Ok(())
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

fn validate_nonzero<T>(name: &'static str, value: T) -> DataFusionResult<()>
where T: PartialEq + From<u8> + std::fmt::Display {
    if value == T::from(0) {
        return Err(plan_error(format!("{name} must be greater than zero")));
    }
    Ok(())
}
