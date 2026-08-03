use super::{
    env::{DurationZeroPolicy, EnvValues, NativeConfigValue},
    file::ServerNativeFileConfig,
    *,
};

/// Native bounded-execution device-memory budget in bytes. When one bounded
/// budget value is set, the source chunk byte and row-group limits must also be set.
pub const NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES";
/// Maximum source row groups planned into one native bounded source chunk.
/// Required with the other bounded budget values when set explicitly.
pub const NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum source bytes planned into one native bounded source chunk. Required
/// with the other bounded budget values when set explicitly.
pub const NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Maximum bounded retry attempts for source splitting, final merge, and final
/// sort retry paths. Defaults to the native library bounded default.
pub const NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS";
/// Row-group count at or below which bounded source retry splitting stops.
/// Defaults to the native library bounded default.
pub const NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
/// Maximum active native source reads allowed per bounded query. Values above
/// `1` allow source prefetch overlap when the plan shape supports it.
pub const NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT";
/// Bounded source chunk planning mode. Accepts `conservative` or
/// `local_parquet_throughput`.
pub const NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str = "NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE";
/// Enables per-query CUDA allocator domains and sets the immutable private-pool
/// ceiling as a positive multiple of the initial admission grant.
/// Floating statistical aggregate execution contract. Accepts
/// `datafusion_exact_host_welford` or `gpu_native_tolerant`.
pub const NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
/// Scope for a server `[native]` configuration option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServerNativeConfigScope {
    /// Native execution setting that applies to every bounded execution.
    NativeExecution,
    /// Bounded execution setting.
    BoundedExecution,
    /// Service-level source IO resource setting, not session-scoped.
    SourceIoResource,
}

impl ServerNativeConfigScope {
    /// Whether this option controls bounded execution.
    pub const fn is_bounded_only(self) -> bool {
        matches!(self, Self::BoundedExecution)
    }
}

/// Discoverable metadata for one server `[native]` TOML/env option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NexusServerNativeConfigOption {
    /// TOML field including the `[native]` section prefix, for example
    /// `native.memory_policy`.
    pub toml_field: &'static str,
    /// Environment variable that overrides the TOML field.
    pub env_var: &'static str,
    /// Human-readable value type used by docs and operator surfaces.
    pub value_type: &'static str,
    /// Server-side scope for this option.
    pub scope: ServerNativeConfigScope,
    /// Matching DataFusion session key when this is also session-scoped.
    pub session_key: Option<&'static str>,
    /// Human-readable operator description.
    pub description: &'static str,
}

/// Server `[native]` options exposed through TOML and environment variables.
///
/// This table is the programmatic discovery surface for operator docs and
/// config introspection. Parser code remains explicit so validation stays
/// local to `native_optimizer_config`.
pub const NEXUS_SERVER_NATIVE_CONFIG_OPTIONS: &[NexusServerNativeConfigOption] = &[
    NexusServerNativeConfigOption {
        toml_field: "native.device_budget_bytes",
        env_var: NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
        value_type: "positive integer bytes",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.device_budget_bytes"),
        description: "Bounded native device-memory budget in bytes.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_source_chunk_bytes",
        env_var: NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        value_type: "positive integer bytes",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_source_chunk_bytes"),
        description: "Maximum source bytes planned into one native bounded source chunk.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_row_groups_per_chunk",
        env_var: NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_row_groups_per_chunk"),
        description: "Maximum source row groups planned into one native bounded source chunk.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_retry_attempts",
        env_var: NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_retry_attempts"),
        description: "Maximum bounded retry attempts.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.min_retry_chunk_row_groups",
        env_var: NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.min_retry_chunk_row_groups"),
        description: "Row-group count at or below which bounded source retry splitting stops.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_source_reads_in_flight",
        env_var: NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_source_reads_in_flight"),
        description: "Maximum active native source reads allowed per bounded query.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.source_chunk_planning_mode",
        env_var: NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        value_type: "enum: conservative|local_parquet_throughput",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.source_chunk_planning_mode"),
        description: "Bounded source chunk planning mode.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.statistical_aggregate_execution_mode",
        env_var: NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        value_type: "enum: datafusion_exact_host_welford|gpu_native_tolerant",
        scope: ServerNativeConfigScope::NativeExecution,
        session_key: Some("datafusion_nexus.native.statistical_aggregate_execution_mode"),
        description: "Floating statistical aggregate execution contract.",
    },
];
pub(super) fn native_optimizer_config(
    values: &mut EnvValues<'_>,
    file: &ServerNativeFileConfig,
) -> Result<NexusNativeOptimizerConfig> {
    let device_budget = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
        "native.device_budget_bytes",
        file.device_budget_bytes,
    )?;
    let max_source_chunk = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        "native.max_source_chunk_bytes",
        file.max_source_chunk_bytes,
    )?;
    let max_row_groups = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        "native.max_row_groups_per_chunk",
        file.max_row_groups_per_chunk,
    )?;
    let max_retry_attempts = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        "native.max_retry_attempts",
        file.max_retry_attempts,
    )?;
    let min_retry_chunk_row_groups = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
        "native.min_retry_chunk_row_groups",
        file.min_retry_chunk_row_groups,
    )?;
    let max_source_reads_in_flight = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        "native.max_source_reads_in_flight",
        file.max_source_reads_in_flight,
    )?;
    let source_chunk_planning_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        "native.source_chunk_planning_mode",
        file.source_chunk_planning_mode.as_deref(),
    )
    .map(|raw| {
        parse_source_chunk_planning_mode(raw.field, &raw.value).map(|mode| NativeConfigValue::new(mode, raw.field))
    })
    .transpose()?;
    let statistical_aggregate_execution_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        "native.statistical_aggregate_execution_mode",
        file.statistical_aggregate_execution_mode.as_deref(),
    )
    .map(|raw| {
        parse_statistical_aggregate_execution_mode(raw.field, &raw.value)
            .map(|mode| NativeConfigValue::new(mode, raw.field))
    })
    .transpose()?;
    let retry_attempts = max_retry_attempts
        .as_ref()
        .map_or(NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value);
    let retry_floor = min_retry_chunk_row_groups
        .as_ref()
        .map_or(NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| value.value);
    let mut execution_config = match (&device_budget, &max_source_chunk, &max_row_groups) {
        (None, None, None) => NexusNativeExecutionConfig::bounded_default(),
        (None, Some(max_source_chunk), Some(max_row_groups)) => {
            if max_source_chunk.value == 0 {
                return Err(ServerError::invalid_config(max_source_chunk.field, "must be greater than zero"));
            }
            if max_row_groups.value == 0 {
                return Err(ServerError::invalid_config(max_row_groups.field, "must be greater than zero"));
            }
            NexusNativeExecutionConfig::bounded(
                NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
                max_source_chunk.value,
                max_row_groups.value,
                retry_attempts,
                retry_floor,
            )
        },
        (Some(device_budget), Some(max_source_chunk), Some(max_row_groups)) => {
            if device_budget.value == 0 {
                return Err(ServerError::invalid_config(device_budget.field, "must be greater than zero"));
            }
            if max_source_chunk.value == 0 {
                return Err(ServerError::invalid_config(max_source_chunk.field, "must be greater than zero"));
            }
            if max_row_groups.value == 0 {
                return Err(ServerError::invalid_config(max_row_groups.field, "must be greater than zero"));
            }
            NexusNativeExecutionConfig::bounded(
                device_budget.value,
                max_source_chunk.value,
                max_row_groups.value,
                retry_attempts,
                retry_floor,
            )
        },
        _ => {
            return Err(ServerError::invalid_config(
                first_present_bounded_budget_field(&device_budget, &max_source_chunk, &max_row_groups),
                "native execution requires device budget, max source chunk bytes, and max row groups per chunk when any explicit budget field is set",
            ));
        },
    };
    if let Some(max_retry_attempts) = max_retry_attempts {
        if max_retry_attempts.value == 0 {
            return Err(ServerError::invalid_config(max_retry_attempts.field, "must be greater than zero"));
        }
        if let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            min_retry_chunk_row_groups,
            ..
        } = execution_config.memory_control_policy()
        {
            execution_config = execution_config.with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts: max_retry_attempts.value,
                min_retry_chunk_row_groups,
            });
        }
    }
    if let Some(min_retry_chunk_row_groups) = min_retry_chunk_row_groups {
        if min_retry_chunk_row_groups.value == 0 {
            return Err(ServerError::invalid_config(min_retry_chunk_row_groups.field, "must be greater than zero"));
        }
        if let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            ..
        } = execution_config.memory_control_policy()
        {
            execution_config = execution_config.with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups: min_retry_chunk_row_groups.value,
            });
        }
    }
    if let Some(max_source_reads_in_flight) = max_source_reads_in_flight {
        if max_source_reads_in_flight.value == 0 {
            return Err(ServerError::invalid_config(max_source_reads_in_flight.field, "must be greater than zero"));
        }
        execution_config =
            execution_config.with_source_scheduler_policy(SourceSchedulerPolicy::new(max_source_reads_in_flight.value));
    }
    if let Some(mode) = source_chunk_planning_mode {
        execution_config = execution_config.with_source_chunk_planning_mode(mode.value);
    }
    if let Some(mode) = statistical_aggregate_execution_mode {
        execution_config = execution_config.with_statistical_aggregate_execution_mode(mode.value);
    }

    Ok(NexusNativeOptimizerConfig::default().with_native_execution_config(execution_config))
}

pub(super) fn optional_native_string(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<&str>,
) -> Option<NativeConfigValue<String>> {
    values
        .string(env_name)
        .map(|value| NativeConfigValue::new(value, env_name))
        .or_else(|| file_value.map(|value| NativeConfigValue::new(value.to_string(), file_field)))
}

pub(super) fn optional_native_u64(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<u64>,
) -> Result<Option<NativeConfigValue<u64>>> {
    Ok(match values.optional_u64(env_name)? {
        Some(value) => Some(NativeConfigValue::new(value, env_name)),
        None => file_value.map(|value| NativeConfigValue::new(value, file_field)),
    })
}

pub(super) fn optional_native_usize(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<usize>,
) -> Result<Option<NativeConfigValue<usize>>> {
    Ok(match values.optional_usize(env_name)? {
        Some(value) => Some(NativeConfigValue::new(value, env_name)),
        None => file_value.map(|value| NativeConfigValue::new(value, file_field)),
    })
}

pub(super) fn optional_duration_millis_value(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<u64>,
) -> Result<Option<NativeConfigValue<u64>>> {
    optional_native_u64(values, env_name, file_field, file_value)
}

pub(super) fn duration_millis_from_value(
    selected: NativeConfigValue<u64>,
    zero_policy: DurationZeroPolicy,
) -> Result<Option<Duration>> {
    if selected.value == 0 {
        return match zero_policy {
            DurationZeroPolicy::Reject => Err(ServerError::invalid_config(selected.field, "must be greater than zero")),
            DurationZeroPolicy::Indefinite => Ok(None),
        };
    }
    Ok(Some(Duration::from_millis(selected.value)))
}

pub(super) fn first_present_bounded_budget_field(
    device_budget: &Option<NativeConfigValue<u64>>,
    max_source_chunk: &Option<NativeConfigValue<u64>>,
    max_row_groups: &Option<NativeConfigValue<usize>>,
) -> &'static str {
    device_budget
        .as_ref()
        .map(|value| value.field)
        .or_else(|| max_source_chunk.as_ref().map(|value| value.field))
        .or_else(|| max_row_groups.as_ref().map(|value| value.field))
        .unwrap_or(NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV)
}

pub(super) fn parse_source_chunk_planning_mode(field: &'static str, raw: &str) -> Result<SourceChunkPlanningMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "conservative" => Ok(SourceChunkPlanningMode::Conservative),
        "local_parquet_throughput" => Ok(SourceChunkPlanningMode::LocalParquetThroughput),
        other => Err(ServerError::invalid_config(
            field,
            format!("must be `conservative` or `local_parquet_throughput`, got `{other}`"),
        )),
    }
}

pub(super) fn parse_statistical_aggregate_execution_mode(
    field: &'static str,
    raw: &str,
) -> Result<StatisticalAggregateExecutionMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "datafusion_exact_host_welford" => Ok(StatisticalAggregateExecutionMode::DataFusionExactHostWelford),
        "gpu_native_tolerant" => Ok(StatisticalAggregateExecutionMode::GpuNativeTolerant),
        other => Err(ServerError::invalid_config(
            field,
            format!("must be `datafusion_exact_host_welford` or `gpu_native_tolerant`, got `{other}`"),
        )),
    }
}
