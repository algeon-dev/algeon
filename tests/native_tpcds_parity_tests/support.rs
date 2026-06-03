use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use datafusion::{
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CudfOptimizerConfig, GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionPlanningOutcome,
    NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome,
    NativeTpcdsParityPlanningOutcome, NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode,
    ParityStageStatus, ParityStatus,
};
use datafusion_nexus_bench::{
    CUDF_TPCDS_PARQUET_DIR_ENV, TPCDS_QUERIES, TPCDS_TABLES, TpcdsWorkloadResult, default_tpcds_sf1_parquet_root,
    discover_tpcds_table_layout, tpcds_parity_workload_items_with_session_configs as raw_tpcds_parity_workload_items,
    tpcds_report_session_config,
};

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
) {
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
    assert_eq!(row.native_reason_code, None, "{query_id}");
    assert_eq!(row.native_unsupported_category, None, "{query_id}");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::PartialNative, "{query_id}");
    assert!(row.native_plan_fragment_count > 0, "{query_id} should report native fragments");
    assert!(row.host_datafusion_boundary_count > 0, "{query_id} should report host DataFusion boundary operators");
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
    assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
    assert_eq!(row.comparison_mode, comparison_mode, "{query_id}");
    assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
    assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
    assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
    assert_eq!(row.native_retry_count, Some(0), "{query_id}");
    assert_eq!(row.native_runtime_failure, None, "{query_id}");
}

pub(super) fn assert_rejected_before_native_execution_parity_row(
    row: &NativeTpcdsParityRow,
    query_id: &str,
    unsupported_category: &str,
    cpu_output_rows: usize,
) {
    assert_eq!(row.query_id, query_id);
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::RejectedBeforeExecution, "{query_id}");
    assert_eq!(row.native_reason_code.as_deref(), Some("native_bounded_join_unsupported"), "{query_id}");
    assert_eq!(row.native_unsupported_category.as_deref(), Some(unsupported_category), "{query_id}");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Cpu, "{query_id}");
    assert_eq!(row.native_plan_fragment_count, 0, "{query_id}");
    assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
    assert!(!row.native_runtime_started, "{query_id} must not start native runtime after strict admission rejection");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::NotStarted, "{query_id}");
    assert_eq!(row.parity_status, ParityStatus::NativeRejected, "{query_id}");
    assert_eq!(row.comparison_mode, ParityComparisonMode::NotApplicable, "{query_id}");
    assert_eq!(row.cpu_output_rows, Some(cpu_output_rows), "{query_id}");
    assert_eq!(row.native_output_rows, None, "{query_id}");
    assert_eq!(row.native_reservation_denied_count, None, "{query_id}");
    assert_eq!(row.native_retry_count, None, "{query_id}");
    assert_eq!(row.native_runtime_failure, None, "{query_id}");
}

pub(super) fn assert_strict_bounded_rejected_before_execution_planning_row(
    row: &NativeDataFusionPlanningRow,
    query_id: &str,
    unsupported_category: &str,
) {
    assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution, "{query_id}");
    assert_eq!(row.reason_code.as_deref(), Some("native_bounded_join_unsupported"), "{query_id}");
    assert_eq!(row.unsupported_category.as_deref(), Some(unsupported_category), "{query_id}");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Cpu, "{query_id}");
    assert_eq!(row.native_plan_fragment_count, 0, "{query_id}");
    assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
    assert!(row.rejected_before_execution, "{query_id}");

    let gate = row
        .strict_bounded_contract_rows
        .first()
        .unwrap_or_else(|| panic!("{query_id} should report the first strict bounded blocking gate"));
    assert_eq!(gate.execution_mode.as_str(), "unsupported_strict_bounded", "{query_id}");
    assert_eq!(gate.memory_boundary, "admission_capability", "{query_id}");
    assert_eq!(gate.required_contract, "root_join_admission_requires_sink_root_join", "{query_id}");
    assert_eq!(gate.contract_status.as_str(), "rejected_by_capability", "{query_id}");
    let stable_gate = gate.to_stable_kv_row();
    assert!(stable_gate.contains("owner=admission:root_join_admission"), "{query_id}: {stable_gate}");
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
        tpcds_report_session_config(),
        optimizer_config,
        query_filter,
    )
    .await
}

pub(super) async fn plain_tpcds_physical_plan(root: &Path, query_id: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(tpcds_report_session_config());
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table).unwrap();
        ctx.register_parquet(table, layout.anchor.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
    let query = TPCDS_QUERIES.iter().find(|query| query.id == query_id).unwrap();
    let df = ctx.sql(query.sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) fn sf1_tpcds_test_session_config() -> SessionConfig {
    tpcds_report_session_config().with_target_partitions(tpcds_test_target_partitions())
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
    plan: &nexus_query_engine::QueryPlan,
    function: nexus_query_engine::AggregateFunction,
) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        nexus_query_engine::PlanNode::Aggregate(aggregate) => {
            aggregate.spec.aggr_exprs.iter().any(|expr| expr.function == function)
        },
        _ => false,
    })
}
