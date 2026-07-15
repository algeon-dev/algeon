use std::{path::Path, sync::Arc};

use arrow::record_batch::RecordBatch;
use datafusion::{
    common::utils::memory::get_record_batch_memory_size,
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_exec::NativeCudfExecutionConfig,
    native_report::{
        NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
        NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime, NativeFallbackPressureWorkloadItem,
    },
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
    policy::GpuFallbackPolicy,
    session::CudfSessionStateBuilderExt,
};
use datafusion_nexus_workloads::clickbench::{
    CLICKBENCH_QUERIES, ClickbenchQuery, ClickbenchWorkloadResult, discover_clickbench_layout,
    register_clickbench_table,
};

/// Device-byte budget for bounded native ClickBench execution.
///
/// When unset, the workload uses CUDA/cudf-rs inventory total memory minus
/// 500 MiB, with a 256 MiB fallback if GPU memory cannot be detected. Invalid
/// values panic during config construction.
pub const CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES";
/// Planning target, in bytes, for native Parquet source chunks. Default: 64 MiB.
pub const CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Planning target for row groups per native Parquet source chunk. Default: 1.
pub const CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum bounded retry attempts after resource or reservation pressure. Default: 3.
pub const CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS";
/// Minimum row-group granularity used when retrying with smaller chunks. Default: 1.
pub const CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str =
    "CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";

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
/// All env vars parse unsigned integers. Source chunk bytes and row-group
/// limits are planning targets; positive-value validation happens when the
/// native execution context is built. The config uses `GpuPreferred` fallback
/// policy with bounded native execution.
pub fn clickbench_native_optimizer_config() -> CudfOptimizerConfig {
    let device_budget_bytes = optional_env_u64(CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV)
        .unwrap_or_else(NativeCudfExecutionConfig::auto_sized_device_budget_bytes);
    let max_source_chunk_bytes = optional_env_u64(CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES);
    let max_row_groups_per_chunk = optional_env_usize(CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK);
    let max_retry_attempts = optional_env_usize(CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS);
    let min_retry_chunk_row_groups = optional_env_usize(CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV)
        .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS);
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        ),
    )
}

fn optional_env_u64(name: &'static str) -> Option<u64> {
    std::env::var(name).ok().map(|value| {
        value
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
    })
}

fn optional_env_usize(name: &'static str) -> Option<usize> {
    std::env::var(name).ok().map(|value| {
        value
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("unsupported {name}={value}; expected unsigned integer: {error}"))
    })
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
    optimizer_config: CudfOptimizerConfig,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
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
    optimizer_config: CudfOptimizerConfig,
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
    optimizer_config: CudfOptimizerConfig,
    query_ids: Option<&[&str]>,
    mut progress: Option<&mut dyn FnMut(ClickbenchExecutionProgressEvent<'_>)>,
) -> ClickbenchWorkloadResult<Vec<NativeDataFusionExecutionWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
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
                        let runtime = if diagnostics.report.rewritten_count() > 0 {
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
