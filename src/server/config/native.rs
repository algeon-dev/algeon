use super::{
    env::{DurationZeroPolicy, EnvValues, NativeConfigValue},
    file::ServerNativeFileConfig,
    *,
};

/// Native bounded-execution device-memory budget in bytes. When one bounded
/// budget value is set, the source chunk byte and row-group limits must also be set.
pub const NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES";
/// Native memory-control mode. Use `bounded` for memory-controlled execution
/// or `unbounded` to disable native bounded-memory controls; defaults to
/// `bounded`.
pub const NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV: &str = "NEXUS_SERVER_NATIVE_MEMORY_POLICY";
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
/// Seconds a native object-store source read may wait for host-buffer budget
/// or source-fetch concurrency before returning a resource timeout. This
/// service-level source IO setting applies in both bounded and unbounded mode.
pub const NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV: &str =
    "NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS";
/// Per-device retained-byte cap for the native query-service decoded local
/// Parquet source cache. Unset disables the service cache.
pub const NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV: &str =
    "NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES";
/// Per-device retained-byte cap for the native query-service resident cuGraph
/// cache. Unset disables the service cache. Requires a `cugraph` build.
pub const NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV: &str =
    "NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES";
/// Enables per-query CUDA allocator domains and sets the immutable private-pool
/// ceiling as a positive multiple of the initial admission grant.
pub const NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV: &str =
    "NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER";
/// Native bounded-pipeline execution mode. Accepts `strict_bounded` or
/// `adaptive`.
pub const NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE";
/// Floating statistical aggregate execution contract. Accepts
/// `datafusion_exact_host_welford` or `gpu_native_tolerant`.
pub const NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
/// Scope for a server `[native]` configuration option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServerNativeConfigScope {
    /// Native execution setting that applies in both bounded and unbounded mode.
    NativeExecution,
    /// Bounded execution setting rejected when native memory policy is unbounded.
    BoundedExecution,
    /// Service-level source IO resource setting, not session-scoped.
    SourceIoResource,
}

impl ServerNativeConfigScope {
    /// Whether this option is active only under bounded native memory policy.
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
/// config introspection. Parser code remains explicit so validation and
/// bounded/unbounded interactions stay local to `native_optimizer_config`.
pub const NEXUS_SERVER_NATIVE_CONFIG_OPTIONS: &[NexusServerNativeConfigOption] = &[
    NexusServerNativeConfigOption {
        toml_field: "native.memory_policy",
        env_var: NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
        value_type: "enum: bounded|unbounded",
        scope: ServerNativeConfigScope::NativeExecution,
        session_key: Some("datafusion_nexus.native.memory_policy"),
        description: "Native memory-control mode: bounded or unbounded.",
    },
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
        toml_field: "native.host_fetch_acquire_timeout_secs",
        env_var: NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV,
        value_type: "positive integer seconds",
        scope: ServerNativeConfigScope::SourceIoResource,
        session_key: None,
        description: "Service-level host-buffer budget and source-fetch concurrency wait timeout.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.source_cache_max_retained_bytes",
        env_var: NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV,
        value_type: "positive integer bytes",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: None,
        description: "Per-device retained-byte cap for the query-service decoded local-Parquet source cache; runtime-affine entries may duplicate per warmed service runtime.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.graph_cache_max_retained_bytes",
        env_var: NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
        value_type: "positive integer bytes",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: None,
        description: "Per-device retained-byte cap for the query-service parked cuGraph cache; requires the cugraph feature.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.query_memory_domain_ceiling_multiplier",
        env_var: NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV,
        value_type: "positive integer multiplier",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: None,
        description: "Enable per-query CUDA allocator domains with an immutable private-pool ceiling derived from the initial grant.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.pipeline_execution_mode",
        env_var: NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV,
        value_type: "enum: strict_bounded|adaptive",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.pipeline_execution_mode"),
        description: "Native bounded-pipeline execution mode.",
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
    gpu_devices: &[usize],
    device_budget_fraction: Option<f64>,
) -> Result<CudfOptimizerConfig> {
    let memory_policy = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
        "native.memory_policy",
        file.memory_policy.as_deref(),
    )
    .map(|raw| {
        parse_native_memory_policy(raw.field, &raw.value)
            .map(|memory_policy| NativeConfigValue::new(memory_policy, raw.field))
    })
    .transpose()?
    .unwrap_or(NativeConfigValue::new(ServerNativeMemoryPolicy::Bounded, NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
    let env_forces_unbounded = matches!(memory_policy.value, ServerNativeMemoryPolicy::Unbounded)
        && memory_policy.field == NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV;
    let device_budget = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
        "native.device_budget_bytes",
        if env_forces_unbounded { None } else { file.device_budget_bytes },
    )?;
    if device_budget_fraction.is_some() && device_budget.is_some() {
        return Err(ServerError::invalid_config(
            NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV,
            format!("cannot be combined with {NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV}"),
        ));
    }
    let max_source_chunk = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        "native.max_source_chunk_bytes",
        if env_forces_unbounded { None } else { file.max_source_chunk_bytes },
    )?;
    let max_row_groups = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        "native.max_row_groups_per_chunk",
        if env_forces_unbounded { None } else { file.max_row_groups_per_chunk },
    )?;
    let max_retry_attempts = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        "native.max_retry_attempts",
        if env_forces_unbounded { None } else { file.max_retry_attempts },
    )?;
    let min_retry_chunk_row_groups = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
        "native.min_retry_chunk_row_groups",
        if env_forces_unbounded { None } else { file.min_retry_chunk_row_groups },
    )?;
    let max_source_reads_in_flight = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        "native.max_source_reads_in_flight",
        if env_forces_unbounded { None } else { file.max_source_reads_in_flight },
    )?;
    let source_chunk_planning_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        "native.source_chunk_planning_mode",
        if env_forces_unbounded { None } else { file.source_chunk_planning_mode.as_deref() },
    )
    .map(|raw| {
        parse_source_chunk_planning_mode(raw.field, &raw.value).map(|mode| NativeConfigValue::new(mode, raw.field))
    })
    .transpose()?;
    let host_fetch_acquire_timeout_secs = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV,
        "native.host_fetch_acquire_timeout_secs",
        file.host_fetch_acquire_timeout_secs,
    )?;
    let source_cache_max_retained_bytes = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV,
        "native.source_cache_max_retained_bytes",
        if env_forces_unbounded { None } else { file.source_cache_max_retained_bytes },
    )?;
    let graph_cache_max_retained_bytes = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
        "native.graph_cache_max_retained_bytes",
        if env_forces_unbounded { None } else { file.graph_cache_max_retained_bytes },
    )?;
    let query_memory_domain_ceiling_multiplier = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV,
        "native.query_memory_domain_ceiling_multiplier",
        if env_forces_unbounded { None } else { file.query_memory_domain_ceiling_multiplier },
    )?;
    let native_pipeline_execution_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV,
        "native.pipeline_execution_mode",
        if env_forces_unbounded { None } else { file.pipeline_execution_mode.as_deref() },
    )
    .map(|raw| {
        parse_native_pipeline_execution_mode(raw.field, &raw.value).map(|mode| NativeConfigValue::new(mode, raw.field))
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
    if matches!(memory_policy.value, ServerNativeMemoryPolicy::Unbounded)
        && (device_budget.is_some()
            || max_source_chunk.is_some()
            || max_row_groups.is_some()
            || max_retry_attempts.is_some()
            || min_retry_chunk_row_groups.is_some()
            || max_source_reads_in_flight.is_some()
            || source_chunk_planning_mode.is_some()
            || source_cache_max_retained_bytes.is_some()
            || graph_cache_max_retained_bytes.is_some()
            || query_memory_domain_ceiling_multiplier.is_some()
            || native_pipeline_execution_mode.is_some()
            || device_budget_fraction.is_some())
    {
        return Err(ServerError::invalid_config(
            memory_policy.field,
            "unbounded native execution cannot be combined with bounded native execution settings",
        ));
    }

    let device_ordinal = *gpu_devices.first().unwrap_or(&0);
    let mut execution_config = match memory_policy.value {
        ServerNativeMemoryPolicy::Unbounded => NativeCudfExecutionConfig::unbounded(),
        ServerNativeMemoryPolicy::Bounded => match (&device_budget, &max_source_chunk, &max_row_groups) {
            (None, None, None) => {
                let device_budget_bytes = match device_budget_fraction {
                    Some(fraction) => NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device_with_fraction(
                        device_ordinal,
                        fraction,
                    ),
                    None => NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device(device_ordinal),
                };
                NativeCudfExecutionConfig::bounded(
                    device_budget_bytes,
                    NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                    NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                    max_retry_attempts
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value),
                    min_retry_chunk_row_groups
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| {
                            value.value
                        }),
                )
            },
            (None, Some(max_source_chunk), Some(max_row_groups)) if device_budget_fraction.is_some() => {
                if max_source_chunk.value == 0 {
                    return Err(ServerError::invalid_config(max_source_chunk.field, "must be greater than zero"));
                }
                if max_row_groups.value == 0 {
                    return Err(ServerError::invalid_config(max_row_groups.field, "must be greater than zero"));
                }
                NativeCudfExecutionConfig::bounded(
                    NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device_with_fraction(
                        device_ordinal,
                        device_budget_fraction.expect("checked above"),
                    ),
                    max_source_chunk.value,
                    max_row_groups.value,
                    max_retry_attempts
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value),
                    min_retry_chunk_row_groups
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| {
                            value.value
                        }),
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
                NativeCudfExecutionConfig::bounded(
                    device_budget.value,
                    max_source_chunk.value,
                    max_row_groups.value,
                    max_retry_attempts
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value),
                    min_retry_chunk_row_groups
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| {
                            value.value
                        }),
                )
            },
            _ => {
                return Err(ServerError::invalid_config(
                    first_present_bounded_budget_field(&device_budget, &max_source_chunk, &max_row_groups),
                    "bounded native execution requires device budget, max source chunk bytes, and max row groups per chunk",
                ));
            },
        },
    }
    .with_device_ordinal(device_ordinal);
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
    if let Some(timeout_secs) = host_fetch_acquire_timeout_secs {
        if timeout_secs.value == 0 {
            return Err(ServerError::invalid_config(timeout_secs.field, "must be greater than zero"));
        }
        execution_config = execution_config.with_host_fetch_acquire_timeout(Duration::from_secs(timeout_secs.value));
    }
    if let Some(max_retained_bytes) = source_cache_max_retained_bytes {
        if max_retained_bytes.value == 0 {
            return Err(ServerError::invalid_config(max_retained_bytes.field, "must be greater than zero"));
        }
        execution_config = execution_config
            .with_query_service_local_parquet_source_cache_max_retained_bytes(Some(max_retained_bytes.value));
    }
    if let Some(max_retained_bytes) = graph_cache_max_retained_bytes {
        if max_retained_bytes.value == 0 {
            return Err(ServerError::invalid_config(max_retained_bytes.field, "must be greater than zero"));
        }
        #[cfg(not(feature = "cugraph"))]
        {
            return Err(ServerError::invalid_config(
                max_retained_bytes.field,
                "resident graph cache requires building with --features server,cugraph or --features server-cugraph",
            ));
        }
        #[cfg(feature = "cugraph")]
        {
            if values.optional_bool(NEXUS_SERVER_CUGRAPH_ENABLED_ENV)? != Some(true) {
                return Err(ServerError::invalid_config(
                    max_retained_bytes.field,
                    format!("requires {NEXUS_SERVER_CUGRAPH_ENABLED_ENV}=true"),
                ));
            }
            execution_config = execution_config
                .with_query_service_resident_graph_cache_max_retained_bytes(Some(max_retained_bytes.value));
        }
    }
    if let Some(multiplier) = query_memory_domain_ceiling_multiplier {
        if multiplier.value == 0 {
            return Err(ServerError::invalid_config(multiplier.field, "must be greater than zero"));
        }
        execution_config = execution_config.with_query_service_query_memory_domain(Some(
            QueryMemoryDomainPolicy::new().with_ceiling_multiplier(multiplier.value),
        ));
    }
    if let Some(mode) = native_pipeline_execution_mode {
        execution_config = execution_config.with_native_pipeline_execution_mode(mode.value);
    }
    if let Some(mode) = statistical_aggregate_execution_mode {
        execution_config = execution_config.with_statistical_aggregate_execution_mode(mode.value);
    }

    Ok(CudfOptimizerConfig::new(GpuFallbackPolicy::default()).with_native_execution_config(execution_config))
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

pub(super) fn optional_duration_millis(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<u64>,
    zero_policy: DurationZeroPolicy,
) -> Result<Option<Duration>> {
    optional_duration_millis_value(values, env_name, file_field, file_value)?
        .map(|selected| duration_millis_from_value(selected, zero_policy))
        .transpose()
        .map(Option::flatten)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ServerNativeMemoryPolicy {
    Bounded,
    Unbounded,
}

pub(super) fn parse_native_memory_policy(field: &'static str, raw: &str) -> Result<ServerNativeMemoryPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bounded" => Ok(ServerNativeMemoryPolicy::Bounded),
        "unbounded" => Ok(ServerNativeMemoryPolicy::Unbounded),
        other => Err(ServerError::invalid_config(field, format!("must be `bounded` or `unbounded`, got `{other}`"))),
    }
}

pub(super) fn parse_memory_share(field: &'static str, raw: &str) -> Result<MemoryShare> {
    let raw = raw.trim();
    let lower = raw.to_ascii_lowercase();
    if raw.contains('.') || lower.contains('e') {
        let fraction = raw.parse::<f64>().map_err(|error| {
            ServerError::invalid_config_with_source(field, "must be a fraction or byte integer", error)
        })?;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || fraction == 0.0 {
            return Err(ServerError::invalid_config(field, "fraction must be greater than 0 and at most 1"));
        }
        return Ok(MemoryShare::Fraction(fraction));
    }
    let bytes = raw
        .parse::<u64>()
        .map_err(|error| ServerError::invalid_config_with_source(field, "must be a fraction or byte integer", error))?;
    if bytes == 0 {
        return Err(ServerError::invalid_config(field, "must be greater than zero"));
    }
    Ok(MemoryShare::Bytes(bytes))
}

pub(super) fn parse_slot_guarantee(field: &'static str, raw: &str) -> Result<SlotGuarantee> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "guaranteed_minimum" => Ok(SlotGuarantee::GuaranteedMinimum),
        "optimistic" => Ok(SlotGuarantee::Optimistic),
        other => Err(ServerError::invalid_config(
            field,
            format!("must be `guaranteed_minimum` or `optimistic`, got `{other}`"),
        )),
    }
}

pub(super) fn parse_query_placement_policy(
    field: &'static str,
    raw: &str,
    pinned_device_ordinal: Option<NativeConfigValue<usize>>,
) -> Result<QueryPlacementPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "least_loaded" | "least-loaded" => Ok(QueryPlacementPolicy::LeastLoaded),
        "best_fit" | "best-fit" => Ok(QueryPlacementPolicy::BestFit),
        "pinned" => {
            let Some(device_ordinal) = pinned_device_ordinal else {
                let required_field = pinned_device_field_for_placement_source(field);
                return Err(ServerError::invalid_config(field, format!("requires {required_field}"))
                    .with_fact("reason", "pinned_placement_missing_device_ordinal")
                    .with_fact("required_config_field", required_field));
            };
            Ok(QueryPlacementPolicy::Pinned { device_ordinal: device_ordinal.value })
        },
        other => Err(ServerError::invalid_config(
            field,
            format!("must be `least_loaded`, `best_fit`, or `pinned`, got `{other}`"),
        )),
    }
}

pub(super) fn pinned_device_field_for_placement_source(placement_field: &'static str) -> &'static str {
    if placement_field == NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV {
        NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV
    } else {
        "admission.pinned_device_ordinal"
    }
}

pub(super) fn placement_policy_field_for_pinned_source(pinned_field: &'static str) -> &'static str {
    if pinned_field == NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV {
        NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV
    } else {
        "admission.placement_policy"
    }
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

pub(super) fn parse_native_pipeline_execution_mode(
    field: &'static str,
    raw: &str,
) -> Result<NativePipelineExecutionMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "strict_bounded" => Ok(NativePipelineExecutionMode::StrictBounded),
        "adaptive" => Ok(NativePipelineExecutionMode::Adaptive),
        other => {
            Err(ServerError::invalid_config(field, format!("must be `strict_bounded` or `adaptive`, got `{other}`")))
        },
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
