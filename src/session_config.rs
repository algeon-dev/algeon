//! DataFusion `ConfigExtension` surface for native query options.
//!
//! The Flight SQL server owns durable process config, but embedded DataFusion
//! sessions need a DataFusion-native way to discover and override native query
//! policy. This module provides the `datafusion_nexus.native.*` namespace used
//! by `SessionConfig::set_str` and SQL `SET`, plus the request-scoped
//! `datafusion_nexus.request.*` namespace read by the optimizer when it mints a
//! plan ticket.

use std::any::Any;

use datafusion::common::{
    DataFusionError, Result as DataFusionResult,
    config::{ConfigEntry, ConfigExtension, ConfigOptions, ExtensionOptions},
};
use nexus_query_engine::{
    graph_memory::GraphMemoryCalibrationOverride,
    mode::StatisticalAggregateExecutionMode,
    runtime::{MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy},
};

use crate::{native_exec::NexusNativeExecutionConfig, planner::NexusNativeOptimizerConfig};

/// DataFusion config-extension namespace prefix for datafusion-nexus options.
pub const DATAFUSION_NEXUS_CONFIG_PREFIX: &str = "datafusion_nexus";
const NATIVE_DEVICE_BUDGET_BYTES_KEY: &str = "native.device_budget_bytes";
const NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY: &str = "native.max_source_chunk_bytes";
const NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY: &str = "native.max_row_groups_per_chunk";
const NATIVE_MAX_RETRY_ATTEMPTS_KEY: &str = "native.max_retry_attempts";
const NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY: &str = "native.min_retry_chunk_row_groups";
const NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY: &str = "native.max_source_reads_in_flight";
const NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY: &str = "native.source_chunk_planning_mode";
const NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY: &str = "native.statistical_aggregate_execution_mode";
const REQUEST_GRAPH_RAW_INPUT_EDGES_KEY: &str = "request.graph_raw_input_edges";

/// Discoverable metadata for one `datafusion_nexus.native.*` session option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DataFusionNexusSessionConfigOption {
    /// Key below [`DATAFUSION_NEXUS_CONFIG_PREFIX`], for example
    /// `native.device_budget_bytes`.
    pub key: &'static str,
    /// Human-readable value type used by docs and operator surfaces.
    pub value_type: &'static str,
    /// Whether this option controls bounded native execution.
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
        key: NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY,
        value_type: "enum: datafusion_exact_host_welford|gpu_native_tolerant",
        bounded_only: false,
        description: "Floating statistical aggregate execution contract.",
    },
    DataFusionNexusSessionConfigOption {
        key: REQUEST_GRAPH_RAW_INPUT_EDGES_KEY,
        value_type: "positive integer edges or none",
        bounded_only: false,
        description: "Session-scoped fallback raw graph edge count used for graph admission when exact plan evidence is unavailable.",
    },
];

/// DataFusion session options under the `datafusion_nexus.native.*` namespace.
///
/// `with_nexus_native` registers this extension automatically, initialized from
/// the optimizer config passed by the caller. Applications may also insert or
/// update it directly through `SessionConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DataFusionNexusSessionConfig {
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
    /// Floating statistical aggregate execution contract.
    pub native_statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    /// Request-scoped raw edge rows before graph normalization.
    request_graph_raw_input_edges: Option<u64>,
    request_graph_raw_input_edges_configured: bool,
    authority: DataFusionNexusSessionConfigAuthority,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum DataFusionNexusSessionConfigAuthority {
    #[default]
    SessionMutable,
    BackendAdmission,
}

impl Default for DataFusionNexusSessionConfig {
    fn default() -> Self {
        Self::from_optimizer_config(&NexusNativeOptimizerConfig::default())
    }
}

impl DataFusionNexusSessionConfig {
    /// Build config-extension values from an optimizer config.
    pub fn from_optimizer_config(config: &NexusNativeOptimizerConfig) -> Self {
        let execution = config.native_execution_config();
        let MemoryControlPolicy::Bounded {
            device_budget_bytes: native_device_budget_bytes,
            max_source_chunk_bytes: native_max_source_chunk_bytes,
            max_row_groups_per_chunk: native_max_row_groups_per_chunk,
            max_retry_attempts: native_max_retry_attempts,
            min_retry_chunk_row_groups: native_min_retry_chunk_row_groups,
        } = execution.memory_control_policy()
        else {
            unreachable!("datafusion-nexus exposes only bounded native execution")
        };

        let graph_memory_calibration = execution.graph_memory_calibration_override();
        Self {
            native_device_budget_bytes,
            native_max_source_chunk_bytes,
            native_max_row_groups_per_chunk,
            native_max_retry_attempts,
            native_min_retry_chunk_row_groups,
            native_max_source_reads_in_flight: execution.source_scheduler_policy().max_source_reads_in_flight(),
            native_source_chunk_planning_mode: execution.source_chunk_planning_mode(),
            native_statistical_aggregate_execution_mode: execution.statistical_aggregate_execution_mode(),
            request_graph_raw_input_edges: graph_memory_calibration.raw_input_edges,
            request_graph_raw_input_edges_configured: false,
            authority: DataFusionNexusSessionConfigAuthority::SessionMutable,
        }
    }

    pub(crate) fn frozen_for_backend_admission(mut self) -> Self {
        self.authority = DataFusionNexusSessionConfigAuthority::BackendAdmission;
        self
    }

    pub(crate) fn is_backend_admission_frozen(&self) -> bool {
        self.authority == DataFusionNexusSessionConfigAuthority::BackendAdmission
    }

    pub(crate) fn validate_backend_admission_matches(
        &self,
        config: &NexusNativeOptimizerConfig,
    ) -> DataFusionResult<()> {
        if !self.is_backend_admission_frozen() {
            return Ok(());
        }
        self.validate_matches_optimizer_config(config)
    }

    pub(crate) fn validate_matches_query_service_policy(
        &self,
        config: &NexusNativeOptimizerConfig,
    ) -> DataFusionResult<()> {
        self.validate_matches_optimizer_config_inner(config, NativeDeviceBudgetValidation::QueryServiceGrant)
    }

    pub(crate) fn validate_matches_optimizer_config(
        &self,
        config: &NexusNativeOptimizerConfig,
    ) -> DataFusionResult<()> {
        self.validate_matches_optimizer_config_inner(config, NativeDeviceBudgetValidation::Exact)
    }

    fn validate_matches_optimizer_config_inner(
        &self,
        config: &NexusNativeOptimizerConfig,
        device_budget_validation: NativeDeviceBudgetValidation,
    ) -> DataFusionResult<()> {
        let expected = Self::from_optimizer_config(config);
        if device_budget_validation == NativeDeviceBudgetValidation::Exact
            && self.native_device_budget_bytes != expected.native_device_budget_bytes
        {
            return Err(frozen_backend_config_error(NATIVE_DEVICE_BUDGET_BYTES_KEY));
        }
        if self.native_max_source_chunk_bytes != expected.native_max_source_chunk_bytes {
            return Err(frozen_backend_config_error(NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY));
        }
        if self.native_max_row_groups_per_chunk != expected.native_max_row_groups_per_chunk {
            return Err(frozen_backend_config_error(NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY));
        }
        if self.native_max_retry_attempts != expected.native_max_retry_attempts {
            return Err(frozen_backend_config_error(NATIVE_MAX_RETRY_ATTEMPTS_KEY));
        }
        if self.native_min_retry_chunk_row_groups != expected.native_min_retry_chunk_row_groups {
            return Err(frozen_backend_config_error(NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY));
        }
        if self.native_max_source_reads_in_flight != expected.native_max_source_reads_in_flight {
            return Err(frozen_backend_config_error(NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY));
        }
        if self.native_source_chunk_planning_mode != expected.native_source_chunk_planning_mode {
            return Err(frozen_backend_config_error(NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY));
        }
        if self.native_statistical_aggregate_execution_mode != expected.native_statistical_aggregate_execution_mode {
            return Err(frozen_backend_config_error(NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY));
        }
        Ok(())
    }

    /// Apply these session options to an optimizer config, preserving non-session
    /// execution resources already attached to the base config.
    pub fn apply_to_optimizer_config(
        &self,
        base: NexusNativeOptimizerConfig,
    ) -> DataFusionResult<NexusNativeOptimizerConfig> {
        self.validate()?;
        let mut execution = base.native_execution_config().clone();
        execution = execution
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: self.native_device_budget_bytes,
                max_source_chunk_bytes: self.native_max_source_chunk_bytes,
                max_row_groups_per_chunk: self.native_max_row_groups_per_chunk,
                max_retry_attempts: self.native_max_retry_attempts,
                min_retry_chunk_row_groups: self.native_min_retry_chunk_row_groups,
            })
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(self.native_max_source_reads_in_flight))
            .with_source_chunk_planning_mode(self.native_source_chunk_planning_mode);
        execution =
            execution.with_statistical_aggregate_execution_mode(self.native_statistical_aggregate_execution_mode);
        execution = self.apply_graph_memory_calibration_request_options(execution);
        Ok(base.with_native_execution_config(execution))
    }

    pub(crate) fn graph_memory_calibration_override(&self) -> GraphMemoryCalibrationOverride {
        GraphMemoryCalibrationOverride { raw_input_edges: self.request_graph_raw_input_edges }
    }

    /// Request-scoped raw edge rows before graph normalization.
    pub fn request_graph_raw_input_edges(&self) -> Option<u64> {
        self.request_graph_raw_input_edges
    }

    /// Set request-scoped raw edge rows before graph normalization.
    ///
    /// `None` is an explicit request reset and will override service-scoped
    /// calibration when this session config is applied to an acquired context.
    pub fn set_request_graph_raw_input_edges(&mut self, raw_input_edges: Option<u64>) {
        self.request_graph_raw_input_edges = raw_input_edges;
        self.request_graph_raw_input_edges_configured = true;
    }

    /// Return this config with request-scoped raw edge rows set.
    pub fn with_request_graph_raw_input_edges(mut self, raw_input_edges: Option<u64>) -> Self {
        self.set_request_graph_raw_input_edges(raw_input_edges);
        self
    }

    pub(crate) fn apply_graph_memory_calibration_request_options(
        &self,
        mut execution: NexusNativeExecutionConfig,
    ) -> NexusNativeExecutionConfig {
        let base = execution.graph_memory_calibration_override();
        let request = self.graph_memory_calibration_override();
        if self.request_graph_raw_input_edges_configured || request.raw_input_edges != base.raw_input_edges {
            execution = execution.with_graph_raw_input_edges_override(request.raw_input_edges);
        }
        execution
    }

    fn validate(&self) -> DataFusionResult<()> {
        require_positive_u64(NATIVE_DEVICE_BUDGET_BYTES_KEY, self.native_device_budget_bytes)?;
        require_positive_u64(NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY, self.native_max_source_chunk_bytes)?;
        require_positive_usize(NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY, self.native_max_row_groups_per_chunk)?;
        require_positive_usize(NATIVE_MAX_RETRY_ATTEMPTS_KEY, self.native_max_retry_attempts)?;
        require_positive_usize(NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY, self.native_min_retry_chunk_row_groups)?;
        require_positive_usize(NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY, self.native_max_source_reads_in_flight)?;
        Ok(())
    }

    fn reject_backend_admission_key(&self, key: &str) -> DataFusionResult<()> {
        if self.is_backend_admission_frozen() {
            return Err(frozen_backend_config_error(key));
        }
        Ok(())
    }

    fn value_for_key(&self, key: &str) -> String {
        match key {
            NATIVE_DEVICE_BUDGET_BYTES_KEY => self.native_device_budget_bytes.to_string(),
            NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY => self.native_max_source_chunk_bytes.to_string(),
            NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY => self.native_max_row_groups_per_chunk.to_string(),
            NATIVE_MAX_RETRY_ATTEMPTS_KEY => self.native_max_retry_attempts.to_string(),
            NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY => self.native_min_retry_chunk_row_groups.to_string(),
            NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY => self.native_max_source_reads_in_flight.to_string(),
            NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY => self.native_source_chunk_planning_mode.to_string(),
            NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY => {
                self.native_statistical_aggregate_execution_mode.to_string()
            },
            REQUEST_GRAPH_RAW_INPUT_EDGES_KEY => {
                self.request_graph_raw_input_edges.map_or_else(|| "none".to_string(), |value| value.to_string())
            },
            _ => unreachable!("unknown datafusion-nexus session config key: {key}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeDeviceBudgetValidation {
    Exact,
    QueryServiceGrant,
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
            NATIVE_DEVICE_BUDGET_BYTES_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_device_budget_bytes = parse_positive_u64(key, value)?;
            },
            NATIVE_MAX_SOURCE_CHUNK_BYTES_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_max_source_chunk_bytes = parse_positive_u64(key, value)?;
            },
            NATIVE_MAX_ROW_GROUPS_PER_CHUNK_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_max_row_groups_per_chunk = parse_positive_usize(key, value)?;
            },
            NATIVE_MAX_RETRY_ATTEMPTS_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_max_retry_attempts = parse_positive_usize(key, value)?;
            },
            NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_min_retry_chunk_row_groups = parse_positive_usize(key, value)?;
            },
            NATIVE_MAX_SOURCE_READS_IN_FLIGHT_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_max_source_reads_in_flight = parse_positive_usize(key, value)?;
            },
            NATIVE_SOURCE_CHUNK_PLANNING_MODE_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_source_chunk_planning_mode = parse_source_chunk_planning_mode(value)?;
            },
            NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_KEY => {
                self.reject_backend_admission_key(key)?;
                self.native_statistical_aggregate_execution_mode = parse_statistical_aggregate_execution_mode(value)?
            },
            REQUEST_GRAPH_RAW_INPUT_EDGES_KEY => {
                self.set_request_graph_raw_input_edges(parse_optional_positive_u64(key, value)?);
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
    config: &NexusNativeOptimizerConfig,
    options: &ConfigOptions,
) -> DataFusionResult<NexusNativeOptimizerConfig> {
    if config.native_execution_config().has_query_service_handle() {
        // QueryService admission is the authority for bounded server/query execution.
        if let Some(session_config) = options.extensions.get::<DataFusionNexusSessionConfig>() {
            session_config.validate_matches_optimizer_config(config)?;
            return Ok(apply_request_options(config.clone(), session_config));
        }
        return Ok(config.clone());
    }
    match options.extensions.get::<DataFusionNexusSessionConfig>() {
        Some(session_config) if session_config.is_backend_admission_frozen() => {
            session_config.validate_backend_admission_matches(config)?;
            Ok(apply_request_options(config.clone(), session_config))
        },
        Some(session_config) => {
            if config.native_execution_config().has_query_admission_authority() {
                if !session_config.is_backend_admission_frozen() {
                    return Err(backend_admission_session_config_not_frozen_error());
                }
                session_config.validate_backend_admission_matches(config)?;
                Ok(apply_request_options(config.clone(), session_config))
            } else {
                session_config.apply_to_optimizer_config(config.clone())
            }
        },
        None => Ok(config.clone()),
    }
}

fn apply_request_options(
    config: NexusNativeOptimizerConfig,
    session_config: &DataFusionNexusSessionConfig,
) -> NexusNativeOptimizerConfig {
    let execution_config =
        session_config.apply_graph_memory_calibration_request_options(config.native_execution_config().clone());
    config.with_native_execution_config(execution_config)
}

fn parse_source_chunk_planning_mode(value: &str) -> DataFusionResult<SourceChunkPlanningMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => Ok(SourceChunkPlanningMode::Conservative),
        "local_parquet_throughput" => Ok(SourceChunkPlanningMode::LocalParquetThroughput),
        other => Err(config_error(format!("must be `conservative` or `local_parquet_throughput`, got `{other}`"))),
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

fn parse_optional_positive_u64(key: &str, value: &str) -> DataFusionResult<Option<u64>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "null" | "default" => Ok(None),
        _ => parse_positive_u64(key, value).map(Some),
    }
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

fn frozen_backend_config_error(key: &str) -> DataFusionError {
    config_error(format!(
        "{DATAFUSION_NEXUS_CONFIG_PREFIX}.{key} is managed by query-service admission and cannot be overridden"
    ))
}

fn backend_admission_session_config_not_frozen_error() -> DataFusionError {
    config_error("datafusion_nexus native session config must be frozen when query-service admission is authoritative")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_exec::admission::{TestAdmissionController, TestAdmissionDeviceProfile};

    #[test]
    fn graph_request_key_is_mutable_when_backend_admission_config_is_frozen() {
        let base = NexusNativeOptimizerConfig::default();
        let mut session_config =
            DataFusionNexusSessionConfig::from_optimizer_config(&base).frozen_for_backend_admission();

        let native_error = session_config.set(NATIVE_DEVICE_BUDGET_BYTES_KEY, "2048").unwrap_err();
        assert!(native_error.to_string().contains("native.device_budget_bytes"));

        session_config.set(REQUEST_GRAPH_RAW_INPUT_EDGES_KEY, "45564149").unwrap();
        assert_eq!(session_config.request_graph_raw_input_edges, Some(45_564_149));

        session_config.set(REQUEST_GRAPH_RAW_INPUT_EDGES_KEY, "default").unwrap();
        assert_eq!(session_config.request_graph_raw_input_edges, None);
    }

    #[test]
    fn graph_request_key_applies_to_backend_frozen_optimizer_config() {
        let controller =
            TestAdmissionController::new(TestAdmissionDeviceProfile::new(0, 2048, 1, 512), Some(2048)).unwrap();
        let execution_config =
            NexusNativeExecutionConfig::bounded(512, 512, 1, 3, 1).with_query_admission_source(controller.source());
        let base = NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config);
        let mut session_config =
            DataFusionNexusSessionConfig::from_optimizer_config(&base).frozen_for_backend_admission();
        session_config.set(REQUEST_GRAPH_RAW_INPUT_EDGES_KEY, "45564149").unwrap();
        let mut options = ConfigOptions::default();
        options.extensions.insert(session_config);

        let effective = optimizer_config_with_datafusion_options(&base, &options).unwrap();

        let graph_calibration = effective.native_execution_config().graph_memory_calibration_override();
        assert_eq!(graph_calibration.raw_input_edges, Some(45_564_149));
        assert_eq!(
            effective.native_execution_config().memory_control_policy(),
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 512,
                max_source_chunk_bytes: 512,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            }
        );
    }

    #[test]
    fn default_request_graph_options_do_not_override_service_acquired_calibration() {
        let service_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);
        let base = NexusNativeOptimizerConfig::default()
            .with_native_execution_config(NexusNativeExecutionConfig::bounded_default());
        let mut options = ConfigOptions::default();
        options
            .extensions
            .insert(DataFusionNexusSessionConfig::from_optimizer_config(&base).frozen_for_backend_admission());

        let effective = optimizer_config_with_datafusion_options(&base, &options).unwrap();

        assert_eq!(
            effective
                .native_execution_config()
                .graph_memory_calibration_override_for_acquired_service_context(service_override),
            service_override
        );
    }

    #[test]
    fn explicit_request_graph_reset_overrides_service_acquired_calibration() {
        let service_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);
        let base = NexusNativeOptimizerConfig::default()
            .with_native_execution_config(NexusNativeExecutionConfig::bounded_default());
        let mut session_config =
            DataFusionNexusSessionConfig::from_optimizer_config(&base).frozen_for_backend_admission();
        session_config.set(REQUEST_GRAPH_RAW_INPUT_EDGES_KEY, "none").unwrap();
        let mut options = ConfigOptions::default();
        options.extensions.insert(session_config);

        let effective = optimizer_config_with_datafusion_options(&base, &options).unwrap();

        assert_eq!(
            effective
                .native_execution_config()
                .graph_memory_calibration_override_for_acquired_service_context(service_override),
            GraphMemoryCalibrationOverride::none()
        );
    }

    #[test]
    fn public_request_graph_reset_setters_mark_request_options_configured() {
        let service_override = GraphMemoryCalibrationOverride::default().with_raw_input_edges(45_564_149);
        let base = NexusNativeOptimizerConfig::default()
            .with_native_execution_config(NexusNativeExecutionConfig::bounded_default());
        let mut session_config =
            DataFusionNexusSessionConfig::from_optimizer_config(&base).frozen_for_backend_admission();
        session_config.set_request_graph_raw_input_edges(None);
        let mut options = ConfigOptions::default();
        options.extensions.insert(session_config);

        let effective = optimizer_config_with_datafusion_options(&base, &options).unwrap();

        assert_eq!(
            effective
                .native_execution_config()
                .graph_memory_calibration_override_for_acquired_service_context(service_override),
            GraphMemoryCalibrationOverride::none()
        );
    }
}
