//! DataFusion `ConfigExtension` surface for native query options.
//!
//! The Flight SQL server owns durable process config, but embedded DataFusion
//! sessions need a DataFusion-native way to discover and override native query
//! policy. This module provides the `datafusion_nexus.native.*` namespace used
//! by `SessionConfig::set_str` and SQL `SET`.

use std::{any::Any, fmt};

use datafusion::common::{
    DataFusionError, Result as DataFusionResult,
    config::{ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions},
};
use nexus_query_engine::{
    MemoryControlPolicy, NativePipelineExecutionMode, SourceChunkPlanningMode, SourceSchedulerPolicy,
    StatisticalAggregateExecutionMode,
};

use crate::{CudfOptimizerConfig, NativeCudfExecutionConfig};

/// DataFusion config-extension namespace prefix for datafusion-nexus options.
pub const DATAFUSION_NEXUS_CONFIG_PREFIX: &str = "datafusion_nexus";
const NATIVE_MEMORY_POLICY_KEY: &str = "native.memory_policy";
const NATIVE_DEVICE_BUDGET_BYTES_KEY: &str = "native.device_budget_bytes";
const NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY: &str = "native.max_source_chunk_bytes";
const NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY: &str = "native.max_row_groups_per_chunk";
const NATIVE_MAX_RETRY_ATTEMPTS_KEY: &str = "native.max_retry_attempts";
const NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY: &str = "native.min_retry_chunk_row_groups";
const NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY: &str = "native.max_source_reads_in_flight";
const NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY: &str = "native.source_chunk_planning_mode";
const NATIVE_PIPELINE_EXECUTION_MODE_KEY: &str = "native.pipeline_execution_mode";
const NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY: &str = "native.statistical_aggregate_execution_mode";

/// Discoverable metadata for one `datafusion_nexus.native.*` session option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DataFusionNexusSessionConfigOption {
    /// Key below [`DATAFUSION_NEXUS_CONFIG_PREFIX`], for example
    /// `native.memory_policy`.
    pub key: &'static str,
    /// Human-readable value type used by docs and operator surfaces.
    pub value_type: &'static str,
    /// Whether this option is active only when native memory policy is bounded.
    pub bounded_only: bool,
    /// Description shown through DataFusion config entries.
    pub description: &'static str,
}

impl DataFusionNexusSessionConfigOption {
    /// Fully-qualified DataFusion config key.
    pub fn full_key(&self) -> String {
        format!("{DATAFUSION_NEXUS_CONFIG_PREFIX}.{}", self.key)
    }
}

/// DataFusion session options exposed by datafusion-nexus.
///
/// This table is the programmatic discovery surface for embedded DataFusion
/// users and documentation generators. `ConfigOptions::entries` reports the
/// same keys and descriptions with current values.
pub const DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS: &[DataFusionNexusSessionConfigOption] = &[
    DataFusionNexusSessionConfigOption {
        key: NATIVE_MEMORY_POLICY_KEY,
        value_type: "enum: bounded|unbounded",
        bounded_only: false,
        description: "Native memory-control mode: bounded or unbounded.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_DEVICE_BUDGET_BYTES_KEY,
        value_type: "positive integer bytes",
        bounded_only: true,
        description: "Bounded native device-memory budget in bytes.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY,
        value_type: "positive integer bytes",
        bounded_only: true,
        description: "Maximum source bytes planned into one native bounded source chunk.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY,
        value_type: "positive integer",
        bounded_only: true,
        description: "Maximum source row groups planned into one native bounded source chunk.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_MAX_RETRY_ATTEMPTS_KEY,
        value_type: "positive integer",
        bounded_only: true,
        description: "Maximum bounded retry attempts.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY,
        value_type: "positive integer",
        bounded_only: true,
        description: "Row-group count at or below which bounded source retry splitting stops.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY,
        value_type: "positive integer",
        bounded_only: true,
        description: "Maximum active native source reads allowed per bounded query.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY,
        value_type: "enum: conservative|local_parquet_throughput",
        bounded_only: true,
        description: "Bounded source chunk planning mode.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_PIPELINE_EXECUTION_MODE_KEY,
        value_type: "enum: strict_bounded|adaptive",
        bounded_only: true,
        description: "Native bounded-pipeline execution mode.",
    },
    DataFusionNexusSessionConfigOption {
        key: NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY,
        value_type: "enum: datafusion_exact_host_welford|gpu_native_tolerant",
        bounded_only: false,
        description: "Floating statistical aggregate execution contract.",
    },
];

/// Native memory-control policy label used by the DataFusion config extension.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum NativeMemoryPolicyConfig {
    /// Memory-controlled native execution.
    #[default]
    Bounded,
    /// Disable native bounded-memory controls for the session.
    Unbounded,
}

impl NativeMemoryPolicyConfig {
    /// Stable snake-case label used in DataFusion config values.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bounded => "bounded",
            Self::Unbounded => "unbounded",
        }
    }
}

impl fmt::Display for NativeMemoryPolicyConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// DataFusion session options under the `datafusion_nexus.native.*` namespace.
///
/// `with_cudf_native` registers this extension automatically, initialized from
/// the optimizer config passed by the caller. Applications may also insert or
/// update it directly through `SessionConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DataFusionNexusSessionConfig {
    /// Native memory-control mode.
    pub native_memory_policy: NativeMemoryPolicyConfig,
    /// Bounded native device-memory budget in bytes.
    pub native_device_budget_bytes: u64,
    /// Maximum source bytes planned into one native bounded source chunk.
    pub native_max_source_chunk_bytes: u64,
    /// Maximum source row groups planned into one native bounded source chunk.
    pub native_max_row_groups_per_chunk: usize,
    /// Maximum bounded retry attempts.
    pub native_max_retry_attempts: usize,
    /// Row-group count at or below which bounded source retry splitting stops.
    pub native_min_retry_chunk_row_groups: usize,
    /// Maximum active native source reads allowed per bounded query.
    pub native_max_source_reads_in_flight: usize,
    /// Bounded source chunk planning mode.
    pub native_source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Native bounded-pipeline execution mode.
    pub native_pipeline_execution_mode: NativePipelineExecutionMode,
    /// Floating statistical aggregate execution contract.
    pub native_statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
}

impl Default for DataFusionNexusSessionConfig {
    fn default() -> Self {
        Self::from_optimizer_config(&CudfOptimizerConfig::default())
    }
}

impl DataFusionNexusSessionConfig {
    /// Build config-extension values from an optimizer config.
    pub fn from_optimizer_config(config: &CudfOptimizerConfig) -> Self {
        let execution = config.native_execution_config();
        let (
            native_memory_policy,
            native_device_budget_bytes,
            native_max_source_chunk_bytes,
            native_max_row_groups_per_chunk,
            native_max_retry_attempts,
            native_min_retry_chunk_row_groups,
        ) = match execution.memory_control_policy() {
            MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            } => (
                NativeMemoryPolicyConfig::Bounded,
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            ),
            MemoryControlPolicy::Unbounded => (
                NativeMemoryPolicyConfig::Unbounded,
                NativeCudfExecutionConfig::auto_sized_device_budget_bytes(),
                NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
                NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            ),
        };

        Self {
            native_memory_policy,
            native_device_budget_bytes,
            native_max_source_chunk_bytes,
            native_max_row_groups_per_chunk,
            native_max_retry_attempts,
            native_min_retry_chunk_row_groups,
            native_max_source_reads_in_flight: execution.source_scheduler_policy().max_source_reads_in_flight(),
            native_source_chunk_planning_mode: execution.source_chunk_planning_mode(),
            native_pipeline_execution_mode: execution.native_pipeline_execution_mode(),
            native_statistical_aggregate_execution_mode: execution.statistical_aggregate_execution_mode(),
        }
    }

    /// Apply these session options to an optimizer config, preserving non-session
    /// execution resources already attached to the base config.
    pub fn apply_to_optimizer_config(&self, base: CudfOptimizerConfig) -> DataFusionResult<CudfOptimizerConfig> {
        self.validate()?;
        let mut execution = base.native_execution_config().clone();
        execution = match self.native_memory_policy {
            NativeMemoryPolicyConfig::Bounded => execution
                .with_memory_control_policy(MemoryControlPolicy::Bounded {
                    device_budget_bytes: self.native_device_budget_bytes,
                    max_source_chunk_bytes: self.native_max_source_chunk_bytes,
                    max_row_groups_per_chunk: self.native_max_row_groups_per_chunk,
                    max_retry_attempts: self.native_max_retry_attempts,
                    min_retry_chunk_row_groups: self.native_min_retry_chunk_row_groups,
                })
                .with_source_scheduler_policy(SourceSchedulerPolicy::new(self.native_max_source_reads_in_flight))
                .with_source_chunk_planning_mode(self.native_source_chunk_planning_mode)
                .with_native_pipeline_execution_mode(self.native_pipeline_execution_mode),
            NativeMemoryPolicyConfig::Unbounded => execution
                .with_memory_control_policy(MemoryControlPolicy::Unbounded)
                .with_source_scheduler_policy(SourceSchedulerPolicy::default())
                .with_source_chunk_planning_mode(SourceChunkPlanningMode::default())
                .with_native_pipeline_execution_mode(NativePipelineExecutionMode::default()),
        };
        execution =
            execution.with_statistical_aggregate_execution_mode(self.native_statistical_aggregate_execution_mode);
        Ok(base.with_native_execution_config(execution))
    }

    fn validate(&self) -> DataFusionResult<()> {
        if self.native_memory_policy == NativeMemoryPolicyConfig::Bounded {
            require_positive_u64(NATIVE_DEVICE_BUDGET_BYTES_KEY, self.native_device_budget_bytes)?;
            require_positive_u64(NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY, self.native_max_source_chunk_bytes)?;
            require_positive_usize(NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY, self.native_max_row_groups_per_chunk)?;
            require_positive_usize(NATIVE_MAX_RETRY_ATTEMPTS_KEY, self.native_max_retry_attempts)?;
            require_positive_usize(NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY, self.native_min_retry_chunk_row_groups)?;
            require_positive_usize(NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY, self.native_max_source_reads_in_flight)?;
        }
        Ok(())
    }

    fn reject_bounded_key_when_unbounded(&self, key: &str) -> DataFusionResult<()> {
        if self.native_memory_policy == NativeMemoryPolicyConfig::Unbounded {
            return Err(config_error(format!("{key} cannot be set while {NATIVE_MEMORY_POLICY_KEY} is `unbounded`")));
        }
        Ok(())
    }

    fn value_for_key(&self, key: &str) -> String {
        match key {
            NATIVE_MEMORY_POLICY_KEY => self.native_memory_policy.to_string(),
            NATIVE_DEVICE_BUDGET_BYTES_KEY => self.native_device_budget_bytes.to_string(),
            NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY => self.native_max_source_chunk_bytes.to_string(),
            NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY => self.native_max_row_groups_per_chunk.to_string(),
            NATIVE_MAX_RETRY_ATTEMPTS_KEY => self.native_max_retry_attempts.to_string(),
            NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY => self.native_min_retry_chunk_row_groups.to_string(),
            NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY => self.native_max_source_reads_in_flight.to_string(),
            NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY => self.native_source_chunk_planning_mode.to_string(),
            NATIVE_PIPELINE_EXECUTION_MODE_KEY => self.native_pipeline_execution_mode.to_string(),
            NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY => {
                self.native_statistical_aggregate_execution_mode.to_string()
            },
            _ => unreachable!("unknown datafusion-nexus session config key: {key}"),
        }
    }
}

impl ConfigExtension for DataFusionNexusSessionConfig {
    const PREFIX: &'static str = DATAFUSION_NEXUS_CONFIG_PREFIX;
}

impl ExtensionOptions for DataFusionNexusSessionConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> DataFusionResult<()> {
        match key {
            NATIVE_MEMORY_POLICY_KEY => self.native_memory_policy = parse_native_memory_policy(value)?,
            NATIVE_DEVICE_BUDGET_BYTES_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_device_budget_bytes = parse_positive_u64(key, value)?;
            },
            NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_max_source_chunk_bytes = parse_positive_u64(key, value)?;
            },
            NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_max_row_groups_per_chunk = parse_positive_usize(key, value)?;
            },
            NATIVE_MAX_RETRY_ATTEMPTS_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_max_retry_attempts = parse_positive_usize(key, value)?;
            },
            NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_min_retry_chunk_row_groups = parse_positive_usize(key, value)?;
            },
            NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_max_source_reads_in_flight = parse_positive_usize(key, value)?;
            },
            NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_source_chunk_planning_mode = parse_source_chunk_planning_mode(value)?;
            },
            NATIVE_PIPELINE_EXECUTION_MODE_KEY => {
                self.reject_bounded_key_when_unbounded(key)?;
                self.native_pipeline_execution_mode = parse_native_pipeline_execution_mode(value)?;
            },
            NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY => {
                self.native_statistical_aggregate_execution_mode = parse_statistical_aggregate_execution_mode(value)?
            },
            _ => return Err(config_error(format!("Config value \"{key}\" not found on DataFusionNexusSessionConfig"))),
        }
        self.validate()
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS
            .iter()
            .map(|option| ConfigEntry {
                key: option.full_key(),
                value: Some(self.value_for_key(option.key)),
                description: option.description,
            })
            .collect()
    }
}

/// Return the native optimizer config after applying any registered
/// `datafusion_nexus.native.*` DataFusion session options.
pub fn optimizer_config_with_datafusion_options(
    config: &CudfOptimizerConfig,
    options: &ConfigOptions,
) -> DataFusionResult<CudfOptimizerConfig> {
    match options.extensions.get::<DataFusionNexusSessionConfig>() {
        Some(session_config) => session_config.apply_to_optimizer_config(config.clone()),
        None => Ok(config.clone()),
    }
}

fn parse_native_memory_policy(value: &str) -> DataFusionResult<NativeMemoryPolicyConfig> {
    match value.trim().to_ascii_lowercase().as_str() {
        "bounded" => Ok(NativeMemoryPolicyConfig::Bounded),
        "unbounded" => Ok(NativeMemoryPolicyConfig::Unbounded),
        other => Err(config_error(format!("must be `bounded` or `unbounded`, got `{other}`"))),
    }
}

fn parse_source_chunk_planning_mode(value: &str) -> DataFusionResult<SourceChunkPlanningMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => Ok(SourceChunkPlanningMode::Conservative),
        "local_parquet_throughput" => Ok(SourceChunkPlanningMode::LocalParquetThroughput),
        other => Err(config_error(format!("must be `conservative` or `local_parquet_throughput`, got `{other}`"))),
    }
}

fn parse_native_pipeline_execution_mode(value: &str) -> DataFusionResult<NativePipelineExecutionMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "strict_bounded" => Ok(NativePipelineExecutionMode::StrictBounded),
        "adaptive" => Ok(NativePipelineExecutionMode::Adaptive),
        other => Err(config_error(format!("must be `strict_bounded` or `adaptive`, got `{other}`"))),
    }
}

fn parse_statistical_aggregate_execution_mode(value: &str) -> DataFusionResult<StatisticalAggregateExecutionMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "datafusion_exact_host_welford" => Ok(StatisticalAggregateExecutionMode::DataFusionExactHostWelford),
        "gpu_native_tolerant" => Ok(StatisticalAggregateExecutionMode::GpuNativeTolerant),
        other => Err(config_error(format!(
            "must be `datafusion_exact_host_welford` or `gpu_native_tolerant`, got `{other}`"
        ))),
    }
}

fn parse_positive_u64(key: &str, value: &str) -> DataFusionResult<u64> {
    let parsed =
        value.trim().parse::<u64>().map_err(|error| config_error(format!("{key} must be an integer: {error}")))?;
    require_positive_u64(key, parsed)?;
    Ok(parsed)
}

fn parse_positive_usize(key: &str, value: &str) -> DataFusionResult<usize> {
    let parsed =
        value.trim().parse::<usize>().map_err(|error| config_error(format!("{key} must be an integer: {error}")))?;
    require_positive_usize(key, parsed)?;
    Ok(parsed)
}

fn require_positive_u64(key: &str, value: u64) -> DataFusionResult<()> {
    if value == 0 {
        return Err(config_error(format!("{key} must be greater than zero")));
    }
    Ok(())
}

fn require_positive_usize(key: &str, value: usize) -> DataFusionResult<()> {
    if value == 0 {
        return Err(config_error(format!("{key} must be greater than zero")));
    }
    Ok(())
}

fn config_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Configuration(message.into())
}
