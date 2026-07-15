use std::{
    any::Any,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::Arc,
};

use datafusion::{
    execution::{SessionStateBuilder, context::TaskContext},
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native_exec::NativeCudfExecutionConfig,
    native_report::{
        NativeDataFusionPlanningWorkloadItem, NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome,
        NativePlanningFailure, NativeTpcdsParityCpuOutcome, NativeTpcdsParityNativePlanningOutcome,
        NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome, NativeTpcdsParityRow,
        NativeTpcdsParityWorkloadItem, ParityComparisonMode, ParityStageStatus, ParityStatus,
    },
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
    policy::GpuFallbackPolicy,
    session::CudfSessionStateBuilderExt,
    source_cardinality::register_source_unique_keys_for_path,
};
use datafusion_nexus_workloads::tpcds::{
    CUDF_TPCDS_PARQUET_DIR_ENV, TPCDS_QUERIES, TPCDS_TABLES, TpcdsWorkloadResult, default_tpcds_sf1_parquet_root,
    discover_tpcds_table_layout, register_tpcds_tables,
};
use futures::FutureExt;

static TPCDS_NATIVE_PARITY_WORKLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub(super) fn sf1_fixture_root_or_skip() -> Option<PathBuf> {
    if let Ok(root) = std::env::var(CUDF_TPCDS_PARQUET_DIR_ENV) {
        return Some(PathBuf::from(root));
    }

    let root = PathBuf::from(default_tpcds_sf1_parquet_root());
    let complete = TPCDS_TABLES.iter().all(|table| discover_tpcds_table_layout(&root, table).is_ok());
    if complete {
        Some(root)
    } else {
        eprintln!("skipping external SF1 TPC-DS parity: no complete default SF1 fixture at {}", root.display());
        None
    }
}

pub(super) fn assert_whole_native_parity_row(
    row: &NativeTpcdsParityRow,
    query_id: &str,
    comparison_mode: ParityComparisonMode,
    output_rows: usize,
) {
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
    assert_eq!(row.native_reason_code, None, "{query_id}");
    assert_eq!(row.native_unsupported_category, None, "{query_id}");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
    assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
    assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
    assert_successful_native_parity_row(row, query_id, comparison_mode, output_rows);
}

pub(super) fn assert_partial_native_parity_row(
    row: &NativeTpcdsParityRow,
    query_id: &str,
    comparison_mode: ParityComparisonMode,
    output_rows: usize,
    native_plan_fragment_count: usize,
    host_datafusion_boundary_count: usize,
) {
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
    assert_eq!(row.native_reason_code, None, "{query_id}");
    assert_eq!(row.native_unsupported_category, None, "{query_id}");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::PartialNative, "{query_id}");
    assert_eq!(row.native_plan_fragment_count, native_plan_fragment_count, "{query_id}");
    assert_eq!(row.host_datafusion_boundary_count, host_datafusion_boundary_count, "{query_id}");
    assert_successful_native_parity_row(row, query_id, comparison_mode, output_rows);
}

fn assert_successful_native_parity_row(
    row: &NativeTpcdsParityRow,
    query_id: &str,
    comparison_mode: ParityComparisonMode,
    output_rows: usize,
) {
    assert_eq!(row.query_id, query_id);
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
    assert!(row.native_runtime_started, "{query_id} must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
    assert_eq!(
        row.parity_status,
        ParityStatus::Matched,
        "{query_id}: parity_detail={:?} cpu_fingerprint={:?} native_fingerprint={:?}",
        row.parity_detail,
        row.cpu_output_fingerprint,
        row.native_output_fingerprint
    );
    assert_eq!(row.comparison_mode, comparison_mode, "{query_id}");
    assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
    assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
    assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
    assert_eq!(row.native_retry_count, Some(0), "{query_id}");
    assert_eq!(row.native_runtime_failure, None, "{query_id}");
}

pub(super) async fn tpcds_parity_workload_items(
    root: &Path,
    cpu_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    // The parity workload executes native CUDA/libcudf plans. Those bindings use
    // process-global CUDA/RMM state, so parallel libtest workers in this one test
    // binary can crash the process even when each individual query is valid.
    let _guard = TPCDS_NATIVE_PARITY_WORKLOAD_LOCK.lock().await;
    raw_tpcds_parity_workload_items(
        root,
        cpu_session_config,
        tpcds_native_report_session_config(),
        optimizer_config,
        query_filter,
    )
    .await
}

pub(super) async fn plain_tpcds_physical_plan(root: &Path, query_id: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(tpcds_native_report_session_config());
    register_tpcds_test_tables(&ctx, root).await.unwrap();
    let query = TPCDS_QUERIES.iter().find(|query| query.id == query_id).unwrap();
    let df = ctx.sql(query.sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) fn sf1_tpcds_test_session_config() -> SessionConfig {
    tpcds_report_session_config().with_target_partitions(tpcds_test_target_partitions())
}

pub(super) fn tpcds_report_session_config() -> SessionConfig {
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
    config
}

pub(super) fn tpcds_native_report_session_config() -> SessionConfig {
    native_tpcds_session_config(tpcds_report_session_config())
}

fn native_tpcds_session_config(mut config: SessionConfig) -> SessionConfig {
    config.options_mut().execution.collect_statistics = false;
    config.options_mut().optimizer.enable_physical_uncorrelated_scalar_subquery = false;
    config
}

pub(super) async fn tpcds_datafusion_planning_workload_items_with_optimizer_config(
    root: &Path,
    session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
) -> TpcdsWorkloadResult<Vec<NativeDataFusionPlanningWorkloadItem>> {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_tpcds_session_config(session_config))
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_test_tables(&ctx, root).await?;

    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
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

async fn raw_tpcds_parity_workload_items(
    root: &Path,
    cpu_session_config: SessionConfig,
    native_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    let runner = TpcdsParityRunner::new(root, cpu_session_config, native_session_config, optimizer_config).await?;
    let mut items = Vec::with_capacity(TPCDS_QUERIES.len());
    for query in TPCDS_QUERIES {
        if query_filter.is_some_and(|allowed| !allowed.iter().any(|query_id| query_id == query.id)) {
            continue;
        }
        items.push(runner.run_query(query.id, query.sql).await);
    }
    Ok(items)
}

struct TpcdsParityRunner {
    cpu_ctx: SessionContext,
    cpu_stats_fallback_ctx: Option<SessionContext>,
    native_ctx: SessionContext,
    native_rule: Arc<CudfNativePhysicalOptimizerRule>,
}

impl TpcdsParityRunner {
    async fn new(
        root: &Path,
        cpu_session_config: SessionConfig,
        native_session_config: SessionConfig,
        optimizer_config: CudfOptimizerConfig,
    ) -> TpcdsWorkloadResult<Self> {
        let cpu_stats_fallback_ctx = if cpu_session_config.options().execution.collect_statistics {
            let mut fallback_config = cpu_session_config.clone();
            fallback_config.options_mut().execution.collect_statistics = false;
            let fallback_ctx = SessionContext::new_with_config(fallback_config);
            register_tpcds_test_tables(&fallback_ctx, root).await?;
            Some(fallback_ctx)
        } else {
            None
        };
        let cpu_ctx = SessionContext::new_with_config(cpu_session_config);
        register_tpcds_test_tables(&cpu_ctx, root).await?;

        let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
        let native_state = SessionStateBuilder::new_with_default_features()
            .with_config(native_tpcds_session_config(native_session_config))
            .with_cudf_native_rule(Arc::clone(&native_rule))
            .build();
        let native_ctx = SessionContext::new_with_state(native_state);
        register_tpcds_test_tables(&native_ctx, root).await?;

        Ok(Self { cpu_ctx, cpu_stats_fallback_ctx, native_ctx, native_rule })
    }

    async fn run_query(&self, query_id: &str, sql: &str) -> NativeTpcdsParityWorkloadItem {
        let cpu_outcome = run_cpu_query(&self.cpu_ctx, self.cpu_stats_fallback_ctx.as_ref(), sql).await;
        let (native_planning, native_runtime) = run_native_query(&self.native_ctx, &self.native_rule, sql).await;
        NativeTpcdsParityWorkloadItem {
            query_id: query_id.to_string(),
            sql: sql.to_string(),
            cpu_outcome,
            native_planning,
            native_runtime,
            duckdb_output: datafusion_nexus::native_report::NativeTpcdsParityExternalOutput::NotConfigured,
        }
    }
}

async fn run_cpu_query(
    ctx: &SessionContext,
    stats_fallback_ctx: Option<&SessionContext>,
    sql: &str,
) -> NativeTpcdsParityCpuOutcome {
    let primary = run_cpu_query_once(ctx, sql).await;
    let fallback = match (&primary, stats_fallback_ctx) {
        (CpuQueryAttempt::PhysicalPlanningPanicked(_), Some(fallback_ctx)) => {
            Some(run_cpu_query_once(fallback_ctx, sql).await)
        },
        _ => None,
    };
    match primary {
        CpuQueryAttempt::Outcome(outcome) => outcome,
        CpuQueryAttempt::PhysicalPlanningPanicked(message) => match fallback {
            Some(CpuQueryAttempt::Outcome(outcome)) => outcome,
            Some(CpuQueryAttempt::PhysicalPlanningPanicked(fallback_message)) => {
                NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "physical_planning".to_string(),
                    reason_code: "physical_planning_panicked".to_string(),
                    message: format!(
                        "initial physical planning panicked: {message}; stats-off retry panicked: {fallback_message}"
                    ),
                })
            },
            None => NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_panicked".to_string(),
                message,
            }),
        },
    }
}

enum CpuQueryAttempt {
    Outcome(NativeTpcdsParityCpuOutcome),
    PhysicalPlanningPanicked(String),
}

async fn run_cpu_query_once(ctx: &SessionContext, sql: &str) -> CpuQueryAttempt {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: error.to_string(),
            }));
        },
    };
    let plan = match AssertUnwindSafe(df.create_physical_plan()).catch_unwind().await {
        Ok(Ok(plan)) => plan,
        Ok(Err(error)) => {
            return CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::PlanningFailed(NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_failed".to_string(),
                message: error.to_string(),
            }));
        },
        Err(payload) => return CpuQueryAttempt::PhysicalPlanningPanicked(panic_payload_to_string(payload.as_ref())),
    };
    let schema = plan.schema();
    match collect(plan, Arc::new(TaskContext::default())).await {
        Ok(batches) => CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::Succeeded { schema, batches }),
        Err(error) => {
            CpuQueryAttempt::Outcome(NativeTpcdsParityCpuOutcome::ExecutionFailed { message: error.to_string() })
        },
    }
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "unknown panic payload".to_string()
    }
}

async fn run_native_query(
    ctx: &SessionContext,
    native_rule: &Arc<CudfNativePhysicalOptimizerRule>,
    sql: &str,
) -> (NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome) {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(error) => {
            return (
                NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "sql_to_logical_plan".to_string(),
                    reason_code: "sql_planning_failed".to_string(),
                    message: error.to_string(),
                }),
                NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
            );
        },
    };
    let plan = match df.create_physical_plan().await {
        Ok(plan) => plan,
        Err(error) => {
            return (
                NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                    stage: "physical_planning".to_string(),
                    reason_code: "physical_planning_failed".to_string(),
                    message: error.to_string(),
                }),
                NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
            );
        },
    };
    let Some(diagnostics) = native_rule.last_planning_diagnostics() else {
        return (
            NativeTpcdsParityNativePlanningOutcome::PlanningFailed(NativePlanningFailure {
                stage: "native_diagnostics".to_string(),
                reason_code: "missing_native_diagnostics".to_string(),
                message: "native optimizer rule did not retain planning diagnostics".to_string(),
            }),
            NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        );
    };
    let accepted = diagnostics.report.rewritten_count() > 0;
    let planning_outcome = NativeTpcdsParityNativePlanningOutcome::Planned {
        accepted,
        planned_plan: Arc::clone(&plan),
        diagnostics: Box::new(diagnostics),
    };
    let runtime_outcome = if accepted {
        let schema = plan.schema();
        match collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await {
            Ok(batches) => NativeTpcdsParityNativeRuntimeOutcome::Succeeded {
                schema,
                batches,
                metrics: plan.metrics().unwrap_or_default(),
            },
            Err(error) => NativeTpcdsParityNativeRuntimeOutcome::Failed { message: error.to_string() },
        }
    } else {
        NativeTpcdsParityNativeRuntimeOutcome::NotStarted
    };
    (planning_outcome, runtime_outcome)
}

async fn register_tpcds_test_tables(ctx: &SessionContext, root: &Path) -> TpcdsWorkloadResult<()> {
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table)?;
        if let Some(key) = tpcds_source_unique_key(table) {
            register_source_unique_keys_for_path(&layout.anchor, &[key]);
            for file in &layout.files {
                register_source_unique_keys_for_path(file, &[key]);
            }
        }
    }
    register_tpcds_tables(ctx, root).await
}

fn tpcds_source_unique_key(table: &str) -> Option<&'static [&'static str]> {
    match table {
        "call_center" => Some(&["cc_call_center_sk"]),
        "catalog_page" => Some(&["cp_catalog_page_sk"]),
        "customer" => Some(&["c_customer_sk"]),
        "customer_address" => Some(&["ca_address_sk"]),
        "customer_demographics" => Some(&["cd_demo_sk"]),
        "date_dim" => Some(&["d_date_sk"]),
        "household_demographics" => Some(&["hd_demo_sk"]),
        "income_band" => Some(&["ib_income_band_sk"]),
        "item" => Some(&["i_item_sk"]),
        "promotion" => Some(&["p_promo_sk"]),
        "reason" => Some(&["r_reason_sk"]),
        "ship_mode" => Some(&["sm_ship_mode_sk"]),
        "store" => Some(&["s_store_sk"]),
        "time_dim" => Some(&["t_time_sk"]),
        "warehouse" => Some(&["w_warehouse_sk"]),
        "web_page" => Some(&["wp_web_page_sk"]),
        "web_site" => Some(&["web_site_sk"]),
        _ => None,
    }
}

// This controls only the CPU DataFusion oracle. The native side keeps the
// report session's single DataFusion partition; GPU chunking is owned by the
// native/libcudf execution config below.
fn tpcds_test_target_partitions() -> usize {
    test_threads_from_args()
        .or_else(test_threads_from_env)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1))
}

fn test_threads_from_args() -> Option<usize> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--test-threads=") {
            return parse_positive_usize(value);
        }
        if arg == "--test-threads" {
            return args.next().and_then(|value| parse_positive_usize(&value));
        }
    }
    None
}

fn test_threads_from_env() -> Option<usize> {
    std::env::var("RUST_TEST_THREADS").ok().and_then(|value| parse_positive_usize(&value))
}

fn parse_positive_usize(value: &str) -> Option<usize> {
    value.parse::<usize>().ok().filter(|value| *value > 0)
}

pub(super) fn sf1_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(
            NativeCudfExecutionConfig::auto_sized_device_budget_bytes(),
            4 * 1024 * 1024 * 1024,
            1024,
            3,
            1,
        ),
    )
}

pub(super) fn q72_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(
            NativeCudfExecutionConfig::auto_sized_device_budget_bytes(),
            4 * 1024 * 1024 * 1024,
            1024,
            3,
            1,
        ),
    )
}

pub(super) fn plan_contains_aggregate_function(
    plan: &nexus_query_engine::plan::QueryPlan,
    function: nexus_query_engine::expr::AggregateFunction,
) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        nexus_query_engine::plan::PlanNode::Aggregate(aggregate) => {
            aggregate.spec.aggr_exprs.iter().any(|expr| expr.function == function)
        },
        _ => false,
    })
}
