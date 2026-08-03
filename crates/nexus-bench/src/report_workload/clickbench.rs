use std::{path::Path, sync::Arc};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_exec::NexusNativeExecutionConfig,
    native_report::{
        NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
        NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
    },
    planner::{NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};
use nexus_workloads::clickbench::{
    CLICKBENCH_QUERIES, ClickbenchQuery, ClickbenchWorkloadResult, discover_clickbench_layout,
    register_clickbench_table,
};

/// Device-byte budget for bounded native ClickBench execution.
///
/// When unset, the selected native profile supplies the budget. The
/// conservative profile uses the bounded planning fallback. Deployments that
/// need per-device capacity install a backend; this profile never derives all
/// devices' budgets from GPU 0. Invalid values panic during profile construction.
pub const NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES";
/// Named native memory/source profile used by ClickBench benchmark/report
/// surfaces. Accepts `conservative` (default) or `clickbench_gpu_throughput`;
/// individual `NEXUS_CLICKBENCH_NATIVE_*` knobs may still override the profile.
pub const NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV: &str = "NEXUS_CLICKBENCH_NATIVE_PROFILE";
/// Planning target, in bytes, for native Parquet source chunks.
pub const NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Planning target for row groups per native Parquet source chunk.
pub const NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str =
    "NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum bounded retry attempts after resource or reservation pressure.
pub const NEXUS_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "NEXUS_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS";
/// Minimum row-group granularity used when retrying with smaller chunks.
pub const NEXUS_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str =
    "NEXUS_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
/// Source chunk planning mode used by bounded native ClickBench execution.
/// Accepts `conservative` or `local_parquet_throughput`; defaults to the
/// selected profile.
pub const NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str =
    "NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE";

/// Conservative bounded profile: production-safe ClickBench defaults.
pub const CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE: &str = "conservative";
/// GPU throughput profile for local ClickBench Parquet on high-memory cards.
pub const CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT: &str = "clickbench_gpu_throughput";
/// Profile name used when explicit knobs do not match a named profile.
pub const CLICKBENCH_NATIVE_PROFILE_CUSTOM: &str = "custom";
/// Throughput profile device budget, matching the TPC SF100 95 GiB probe.
pub const CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES: u64 = 102_005_473_280;
/// Throughput profile source byte target: 32 GiB.
pub const CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES: u64 = 34_359_738_368;
/// Throughput profile row-group batch target.
pub const CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK: usize = 1024;

/// Resolved bounded native ClickBench memory/source profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClickbenchNativeProfileSettings {
    pub profile_name: &'static str,
    pub profile_source: &'static str,
    pub device_budget_bytes: u64,
    pub max_source_chunk_bytes: u64,
    pub max_row_groups_per_chunk: usize,
    pub max_retry_attempts: usize,
    pub min_retry_chunk_row_groups: usize,
    pub source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode,
}

impl ClickbenchNativeProfileSettings {
    fn from_base(base: NativeProfileBase) -> Self {
        Self {
            profile_name: base.name,
            profile_source: base.source,
            device_budget_bytes: base.device_budget_bytes,
            max_source_chunk_bytes: base.max_source_chunk_bytes,
            max_row_groups_per_chunk: base.max_row_groups_per_chunk,
            max_retry_attempts: base.max_retry_attempts,
            min_retry_chunk_row_groups: base.min_retry_chunk_row_groups,
            source_chunk_planning_mode: base.source_chunk_planning_mode,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NativeProfileBase {
    name: &'static str,
    source: &'static str,
    device_budget_bytes: u64,
    max_source_chunk_bytes: u64,
    max_row_groups_per_chunk: usize,
    max_retry_attempts: usize,
    min_retry_chunk_row_groups: usize,
    source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode,
}

/// Decision-grade session config for ClickBench reports.
///
/// Uses one target partition, disables file/aggregate/join/sort repartitioning
/// and dynamic filter pushdown, leaves Parquet pushdown filters enabled, and
/// sets `binary_as_string=true` for upstream `BYTE_ARRAY` string columns.
pub fn clickbench_report_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config.options_mut().execution.parquet.pushdown_filters = true;
    config.options_mut().execution.parquet.binary_as_string = true;
    config
}

/// Tuned CPU session config for ClickBench parity baselines.
///
/// Uses available CPU parallelism, batch size 8192, repartitioning enabled for
/// scans/aggregates/joins/sorts, dynamic filters enabled, Parquet pushdown
/// filters disabled, and `binary_as_string=true`.
pub fn clickbench_tuned_cpu_session_config() -> SessionConfig {
    let target_partitions = std::thread::available_parallelism().map(usize::from).unwrap_or(1);
    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(8192)
        .with_repartition_file_scans(true)
        .with_repartition_aggregations(true)
        .with_repartition_joins(true)
        .with_repartition_sorts(true);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_join_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = true;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().execution.parquet.binary_as_string = true;
    config
}

/// Native optimizer config used by ClickBench report and benchmark surfaces.
///
/// Resolves [`NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV`] plus optional
/// `NEXUS_CLICKBENCH_NATIVE_*` overrides. Defaults remain conservative (64 MiB /
/// 1 row group); callers select normal or strict native execution when they
/// install the resulting rule.
pub fn clickbench_native_optimizer_config() -> NexusNativeOptimizerConfig {
    clickbench_native_optimizer_config_with_profile_settings(&clickbench_native_profile_settings_from_env())
}

/// Native optimizer config using an already-resolved ClickBench native profile.
pub fn clickbench_native_optimizer_config_with_profile_settings(
    profile: &ClickbenchNativeProfileSettings,
) -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded(
            profile.device_budget_bytes,
            profile.max_source_chunk_bytes,
            profile.max_row_groups_per_chunk,
            profile.max_retry_attempts,
            profile.min_retry_chunk_row_groups,
        )
        .with_source_chunk_planning_mode(profile.source_chunk_planning_mode),
    )
}

/// Resolves the ClickBench native bounded memory/source profile from env.
///
/// `NEXUS_CLICKBENCH_NATIVE_PROFILE` chooses a named base profile:
/// `conservative` keeps the existing bounded defaults, while
/// `clickbench_gpu_throughput` applies the TPC SF100-like local Parquet
/// throughput knobs. Individual `NEXUS_CLICKBENCH_NATIVE_*` knobs override
/// either base. If those final values exactly match a named profile, the
/// resolved `profile_name` is that profile; otherwise it is `custom`.
pub fn clickbench_native_profile_settings_from_env() -> ClickbenchNativeProfileSettings {
    clickbench_native_profile_settings_with_env(|name| std::env::var(name).ok())
}

/// Resolves ClickBench native profile settings using an injected env reader.
pub fn clickbench_native_profile_settings_with_env(
    mut get: impl FnMut(&'static str) -> Option<String>,
) -> ClickbenchNativeProfileSettings {
    let requested_profile = get(NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV).map(|value| value.trim().to_ascii_lowercase());
    let base = native_profile_base(requested_profile.as_deref());
    let mut resolved = ClickbenchNativeProfileSettings::from_base(base);
    let device_budget_bytes = get(NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV);
    let max_source_chunk_bytes = get(NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV);
    let max_row_groups_per_chunk = get(NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV);
    let max_retry_attempts = get(NEXUS_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV);
    let min_retry_chunk_row_groups = get(NEXUS_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV);
    let source_chunk_planning_mode = get(NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV);
    let had_override = device_budget_bytes.is_some()
        || max_source_chunk_bytes.is_some()
        || max_row_groups_per_chunk.is_some()
        || max_retry_attempts.is_some()
        || min_retry_chunk_row_groups.is_some()
        || source_chunk_planning_mode.is_some();

    if let Some(value) = device_budget_bytes {
        resolved.device_budget_bytes = parse_env_u64(NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, value);
    }
    if let Some(value) = max_source_chunk_bytes {
        resolved.max_source_chunk_bytes = parse_env_u64(NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, value);
    }
    if let Some(value) = max_row_groups_per_chunk {
        resolved.max_row_groups_per_chunk =
            parse_env_usize(NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, value);
    }
    if let Some(value) = max_retry_attempts {
        resolved.max_retry_attempts = parse_env_usize(NEXUS_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, value);
    }
    if let Some(value) = min_retry_chunk_row_groups {
        resolved.min_retry_chunk_row_groups =
            parse_env_usize(NEXUS_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, value);
    }
    if let Some(value) = source_chunk_planning_mode {
        resolved.source_chunk_planning_mode = parse_env_source_chunk_planning_mode(value);
    }
    validate_clickbench_native_profile_settings(&resolved);

    let conservative = native_profile_base(Some(CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE));
    let throughput = native_profile_base(Some(CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT));
    let requested_without_override = requested_profile.is_some() && !had_override;
    if same_profile_values(&resolved, conservative) {
        resolved.profile_name = CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE;
        resolved.profile_source = if requested_without_override {
            "env_profile"
        } else if had_override {
            "resolved_values"
        } else {
            "default"
        };
    } else if same_profile_values(&resolved, throughput) {
        resolved.profile_name = CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT;
        resolved.profile_source = if requested_without_override {
            "env_profile"
        } else if had_override {
            "resolved_values"
        } else {
            "default"
        };
    } else if had_override {
        resolved.profile_name = CLICKBENCH_NATIVE_PROFILE_CUSTOM;
        resolved.profile_source = "custom_overrides";
    }
    resolved
}

fn validate_clickbench_native_profile_settings(profile: &ClickbenchNativeProfileSettings) {
    validate_positive(NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, profile.device_budget_bytes);
    validate_positive(NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, profile.max_source_chunk_bytes);
    validate_positive(NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, profile.max_row_groups_per_chunk);
    validate_positive(NEXUS_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, profile.max_retry_attempts);
    validate_positive(NEXUS_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, profile.min_retry_chunk_row_groups);
}

fn validate_positive<T>(name: &'static str, value: T)
where T: PartialEq + From<u8> {
    assert!(value != T::from(0), "unsupported {name}=0; expected positive unsigned integer");
}

fn native_profile_base(profile: Option<&str>) -> NativeProfileBase {
    let profile = profile.unwrap_or(CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE);
    match profile {
        CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE => NativeProfileBase {
            name: CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE,
            source: "default",
            device_budget_bytes: NexusNativeExecutionConfig::AUTO_SIZED_DEVICE_BUDGET_FALLBACK_BYTES,
            max_source_chunk_bytes: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative,
        },
        CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT => NativeProfileBase {
            name: CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT,
            source: "env_profile",
            device_budget_bytes: CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES,
            max_source_chunk_bytes: CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES,
            max_row_groups_per_chunk: CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            max_retry_attempts: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS,
            source_chunk_planning_mode: nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput,
        },
        other => panic!(
            "unsupported {NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV}={other}; expected {CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE} or {CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT}"
        ),
    }
}

fn same_profile_values(profile: &ClickbenchNativeProfileSettings, base: NativeProfileBase) -> bool {
    profile.device_budget_bytes == base.device_budget_bytes
        && profile.max_source_chunk_bytes == base.max_source_chunk_bytes
        && profile.max_row_groups_per_chunk == base.max_row_groups_per_chunk
        && profile.max_retry_attempts == base.max_retry_attempts
        && profile.min_retry_chunk_row_groups == base.min_retry_chunk_row_groups
        && profile.source_chunk_planning_mode == base.source_chunk_planning_mode
}

fn parse_env_u64(name: &'static str, value: String) -> u64 {
    value
        .parse::<u64>()
        .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
}

fn parse_env_usize(name: &'static str, value: String) -> usize {
    value
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
}

fn parse_env_source_chunk_planning_mode(value: String) -> nexus_query_engine::runtime::SourceChunkPlanningMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "conservative" => nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative,
        "local_parquet_throughput" | "local_throughput" => {
            nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
        },
        other => panic!(
            "unsupported {NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV}={other}; expected conservative or local_parquet_throughput"
        ),
    }
}

/// Builds fallback-pressure planning items for all ClickBench queries.
///
/// SQL and physical-planning failures are encoded as per-query rows so report
/// consumers can distinguish planning failures from native shape rejection.
/// Output follows canonical ClickBench query order.
pub async fn clickbench_fallback_pressure_workload_items(
    root: &Path,
    config: SessionConfig,
) -> ClickbenchWorkloadResult<Vec<NativeFallbackPressureWorkloadItem>> {
    let ctx = SessionContext::new_with_config(config);
    let layout = discover_clickbench_layout(root)?;
    register_clickbench_table(&ctx, &layout).await?;

    let mut items = Vec::with_capacity(CLICKBENCH_QUERIES.len());
    for query in CLICKBENCH_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => NativeFallbackPressureWorkloadItem::new(query.id, plan),
                Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeFallbackPressureWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        items.push(item);
    }
    Ok(items)
}

/// Builds native-planning report items using the default ClickBench native config.
///
/// Output follows canonical ClickBench query order.
pub async fn clickbench_datafusion_planning_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    clickbench_datafusion_planning_workload_items_with_optimizer_config(
        root,
        session_config,
        clickbench_native_optimizer_config(),
    )
    .await
}

/// Builds native-planning report items using an explicit optimizer config.
///
/// The native rule retains diagnostics for the most recently planned query;
/// missing diagnostics are captured as per-query planning failures. Output
/// follows canonical ClickBench query order.
pub async fn clickbench_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: NexusNativeOptimizerConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let layout = discover_clickbench_layout(root)?;
    register_clickbench_table(&ctx, &layout).await?;

    let mut items = Vec::with_capacity(CLICKBENCH_QUERIES.len());
    for query in CLICKBENCH_QUERIES {
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => match native_rule.last_planning_diagnostics() {
                    Some(diagnostics) => NativeDataFusionPlanningWorkloadItem::new(query.id, plan, diagnostics),
                    None => NativeDataFusionPlanningWorkloadItem::planning_failure(
                        query.id,
                        "native_diagnostics",
                        "missing_native_diagnostics",
                        format!("missing retained native planning diagnostics for {}", query.id),
                    ),
                },
                Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeDataFusionPlanningWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        items.push(item);
    }
    Ok(items)
}

/// Builds native execution report items using the default ClickBench native config.
///
/// Native runtime starts only for queries where the optimizer produced at
/// least one native rewrite; other rows keep `NotStarted` runtime status.
pub async fn clickbench_datafusion_execution_workload_items(
    root: &Path,
    session_config: SessionConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    clickbench_datafusion_execution_workload_items_with_optimizer_config(
        root,
        session_config,
        clickbench_native_optimizer_config(),
    )
    .await
}

/// Builds native execution report items using an explicit optimizer config.
///
/// This may touch CUDA because accepted native plans are executed.
pub async fn clickbench_datafusion_execution_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: NexusNativeOptimizerConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    clickbench_datafusion_execution_workload_items_filtered(root, session_config, optimizer_config, None, None).await
}

/// Per-query lifecycle events emitted by the filtered ClickBench execution
/// loop. `query_index` is 1-based and counts only the queries selected for
/// this run.
pub enum ClickbenchExecutionProgressEvent<'a> {
    /// A selected query is about to be planned and possibly executed.
    QueryStarted {
        /// One-based index among selected queries.
        query_index: usize,
        /// Total selected query count.
        total: usize,
        /// Stable ClickBench query id.
        query_id: &'a str,
    },
    /// A selected query finished. `elapsed_ms` covers SQL planning, physical
    /// planning, and any native execution.
    QueryFinished {
        /// One-based index among selected queries.
        query_index: usize,
        /// Total selected query count.
        total: usize,
        /// Stable ClickBench query id.
        query_id: &'a str,
        /// Elapsed wall-clock milliseconds for planning and execution.
        elapsed_ms: u128,
        /// Stable runtime outcome label.
        runtime_outcome: &'a str,
    },
}

/// Builds execution report items for an optional exact query-id subset.
///
/// Unknown query ids are silently ignored. Progress callbacks are emitted in
/// selected-query order: one start event, then one finish event per query.
/// Duplicate requested ids do not duplicate output. Runtime outcome strings
/// are `planning_failure`, `not_started`, `succeeded`, or `failed`.
pub async fn clickbench_datafusion_execution_workload_items_filtered(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: NexusNativeOptimizerConfig,
    query_ids: Option<&[&str]>,
    mut progress: Option<&mut dyn FnMut(ClickbenchExecutionProgressEvent<'_>)>,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let layout = discover_clickbench_layout(root)?;
    register_clickbench_table(&ctx, &layout).await?;

    let selected: Vec<&ClickbenchQuery> = match query_ids {
        Some(ids) => CLICKBENCH_QUERIES.iter().filter(|q| ids.iter().any(|id| id == &q.id)).collect(),
        None => CLICKBENCH_QUERIES.iter().collect(),
    };
    let total = selected.len();
    let mut items = Vec::with_capacity(total);
    for (index, query) in selected.into_iter().enumerate() {
        let query_index = index + 1;
        if let Some(cb) = progress.as_deref_mut() {
            cb(ClickbenchExecutionProgressEvent::QueryStarted { query_index, total, query_id: query.id });
        }
        let started = std::time::Instant::now();
        let item = match ctx.sql(query.sql).await {
            Ok(df) => match df.create_physical_plan().await {
                Ok(plan) => match native_rule.last_planning_diagnostics() {
                    Some(diagnostics) => {
                        let runtime = if diagnostics.report.selected_count() > 0 {
                            execute_clickbench_native_plan(Arc::clone(&plan)).await
                        } else {
                            NativeDataFusionRuntime::NotStarted
                        };
                        NativeDataFusionExecutionWorkloadItem::new(query.id, plan, diagnostics, runtime)
                    },
                    None => NativeDataFusionExecutionWorkloadItem::planning_failure(
                        query.id,
                        "native_diagnostics",
                        "missing_native_diagnostics",
                        format!("missing retained native planning diagnostics for {}", query.id),
                    ),
                },
                Err(error) => NativeDataFusionExecutionWorkloadItem::planning_failure(
                    query.id,
                    "physical_planning",
                    "physical_planning_failed",
                    error.to_string(),
                ),
            },
            Err(error) => NativeDataFusionExecutionWorkloadItem::planning_failure(
                query.id,
                "sql_to_logical_plan",
                "sql_planning_failed",
                error.to_string(),
            ),
        };
        if let Some(cb) = progress.as_deref_mut() {
            let runtime_outcome = match &item.input {
                NativeDataFusionExecutionWorkloadInput::PlanningFailure(_) => "planning_failure",
                NativeDataFusionExecutionWorkloadInput::Planned { runtime, .. } => match runtime {
                    NativeDataFusionRuntime::NotStarted => "not_started",
                    NativeDataFusionRuntime::Succeeded { .. } => "succeeded",
                    NativeDataFusionRuntime::Failed { .. } => "failed",
                    NativeDataFusionRuntime::Interrupted { outcome, .. } => outcome.as_str(),
                },
            };
            cb(ClickbenchExecutionProgressEvent::QueryFinished {
                query_index,
                total,
                query_id: query.id,
                elapsed_ms: started.elapsed().as_millis(),
                runtime_outcome,
            });
        }
        items.push(item);
    }
    Ok(items)
}

async fn execute_clickbench_native_plan(plan: Arc<dyn ExecutionPlan>) -> NativeDataFusionRuntime {
    match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
        Ok(batches) => {
            let output_batches = batches.len();
            let output_rows = batches.iter().map(RecordBatch::num_rows).sum();
            let output_bytes = batches.iter().map(get_record_batch_memory_size).sum();
            let metrics = plan.metrics().unwrap_or_default();
            NativeDataFusionRuntime::Succeeded { output_batches, output_rows, output_bytes, metrics }
        },
        Err(error) => NativeDataFusionRuntime::Failed { message: error.to_string(), metrics: plan.metrics() },
    }
}
