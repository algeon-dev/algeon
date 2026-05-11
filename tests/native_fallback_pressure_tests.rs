use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Float64Array, Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    GpuResidentSource, LocalParquetSource, QueryPlanBuilder, SourceNode, StrictBoundedContractGateId,
    StrictBoundedContractGateRow, StrictBoundedContractOwner, StrictBoundedContractReport, StrictBoundedExecutionMode,
    capability::{ContractQuality, ContractStatus, ReservationRequirementScope},
    pipeline::{ReservationGateId, ReservationGateOwner},
    runtime::ReservationKind,
};
use datafusion::{
    common::ScalarValue,
    execution::SessionStateBuilder,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{TryCastExpr, binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        limit::LocalLimitExec,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricType},
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuCandidate, GpuFallbackPolicy,
    NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION,
    NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeCudfComposedExec,
    NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec, NativeDataFusionExecutionReport,
    NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport,
    NativeDataFusionPlanningWorkloadItem, NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntime,
    NativeDataFusionRuntimeOutcome, NativeFallbackPressureOutcome, NativeFallbackPressureReport,
    NativeFallbackPressureWorkloadItem, NativePlanningDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
    NativeStrictBoundedContractDiagnostics, PlanningReport, TPCH_QUERY_COUNT, TPCH_TABLES,
    describe_tpch_dataset_with_content_fingerprint, tiny_tpch_dataset_metadata,
    tpch_datafusion_planning_workload_items, tpch_fallback_pressure_workload_items, tpch_query_ids,
    tpch_report_session_config, tpch_scalar_fallback_pressure_workload_items, write_tiny_tpch_parquet_dataset,
};

const EXPECTED_BOUNDED_TPCH_ACCEPTED: &[&str] = &[
    "q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10", "q11", "q12", "q13", "q14", "q15", "q16",
    "q19", "q21", "q17", "q18", "q20", "q22",
];
const EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED: &[&str] = &[
    "q02", "q03", "q04", "q05", "q07", "q08", "q09", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q19",
    "q21", "q18", "q20", "q22",
];

#[path = "native_fallback_pressure_tests/execution_reports.rs"]
mod execution_reports;
#[path = "native_fallback_pressure_tests/sql.rs"]
mod sql;
#[path = "native_fallback_pressure_tests/support.rs"]
mod support;
#[path = "native_fallback_pressure_tests/tpch_reports.rs"]
mod tpch_reports;

use sql::*;
use support::*;

#[tokio::test]
async fn native_fallback_pressure_report_summarizes_workload_outcomes() {
    let dir = unique_temp_dir("datafusion-cudf-native-fallback-pressure");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let workload = [
        ("sum_rewrite", SQL_SUM_QUERY),
        ("not_rewrite", SQL_NOT_QUERY),
        ("cast_rewrite", SQL_CAST_QUERY),
        ("avg_rewrite", SQL_AVG_QUERY),
        ("filtered_reject", SQL_FILTERED_QUERY),
        ("modulo_rewrite", SQL_MODULO_QUERY),
    ];
    let mut items = Vec::with_capacity(workload.len());
    for (query_id, sql) in workload {
        items.push(NativeFallbackPressureWorkloadItem::new(
            query_id,
            sql_plan(&path, direct_friendly_sql_session_config(), sql).await,
        ));
    }
    for (query_id, sql) in [
        ("distributed_avg_rewrite", SQL_AVG_QUERY),
        ("distributed_filtered_avg_reject", SQL_FILTERED_AVG_QUERY),
        ("distributed_distinct_avg_rewrite", SQL_DISTINCT_AVG_QUERY),
    ] {
        items.push(NativeFallbackPressureWorkloadItem::new(query_id, sql_plan(&path, SessionConfig::new(), sql).await));
    }
    items.push(NativeFallbackPressureWorkloadItem::new(
        "count_only_rewrite",
        sql_plan(&path, count_only_sql_session_config(), SQL_COUNT_ONLY_QUERY).await,
    ));
    items.push(NativeFallbackPressureWorkloadItem::planning_failure(
        "q00_planning_failure",
        "sql_to_logical_plan",
        "sql_planning_failed",
        "No field named missing_col",
    ));

    let report = NativeFallbackPressureReport::collect(items);

    assert_eq!(report.rows.len(), 11);
    assert_eq!(report.rewritten_count(), 8);
    assert_eq!(report.rejected_count(), 3);

    let rewritten = report_row(&report, "sum_rewrite");
    assert_eq!(rewritten.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(rewritten.reason_code, None);
    assert_eq!(rewritten.unsupported_category, None);
    assert!(!rewritten.rejected_before_execution);
    assert_eq!(rewritten.source_diagnostics.len(), 1);
    assert!(rewritten.source_diagnostics[0].direct_device_read);
    assert_eq!(rewritten.source_diagnostics[0].predicate_disposition, "pruning_only");
    assert_eq!(rewritten.source_diagnostics[0].byte_range_disposition, "full_files");
    assert_eq!(rewritten.source_diagnostics[0].partition_column_disposition, "absent");
    assert_eq!(rewritten.source_diagnostics[0].zero_column_count_disposition, "not_required");
    assert_eq!(rewritten.source_diagnostics[0].schema_mapping_disposition, "direct_by_name");
    assert_eq!(rewritten.source_diagnostics[0].read_column_count, 4);
    assert_eq!(rewritten.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(rewritten.source_diagnostics[0].pure_filter_column_count, 1);

    let not = report_row(&report, "not_rewrite");
    assert_eq!(not.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(not.reason_code, None);
    assert_eq!(not.unsupported_category, None);
    assert!(!not.rejected_before_execution);

    let cast = report_row(&report, "cast_rewrite");
    assert_eq!(cast.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(cast.reason_code, None);
    assert_eq!(cast.unsupported_category, None);
    assert!(!cast.rejected_before_execution);

    let avg = report_row(&report, "avg_rewrite");
    assert_eq!(avg.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(avg.reason_code, None);
    assert_eq!(avg.unsupported_category, None);
    assert!(!avg.rejected_before_execution);

    let distributed_avg = report_row(&report, "distributed_avg_rewrite");
    assert_eq!(distributed_avg.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(
        distributed_avg.native_candidate_shape.as_deref(),
        Some(
            "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec(RepartitionExec(PartialAggregateExec(FilterExec(RepartitionExec(LocalParquetScan))))))))",
        )
    );
    assert_eq!(
        distributed_avg.native_candidate_shape_family.as_deref(),
        Some("sort_preserving_merge_fetch_local_parquet_distributed_aggregate")
    );
    assert_eq!(distributed_avg.reason_code, None);
    assert_eq!(distributed_avg.unsupported_category, None);
    assert!(!distributed_avg.rejected_before_execution);

    let count_only = report_row(&report, "count_only_rewrite");
    assert_eq!(count_only.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(count_only.native_candidate_shape.as_deref(), Some("ProjectionExec(AggregateExec(LocalParquetScan))"));
    assert_eq!(count_only.native_candidate_shape_family.as_deref(), Some("projected_global_local_parquet_aggregate"));
    assert_eq!(count_only.reason_code, None);
    assert_eq!(count_only.unsupported_category, None);
    assert!(!count_only.rejected_before_execution);
    assert_eq!(count_only.source_diagnostics.len(), 1);
    assert_eq!(count_only.source_diagnostics[0].read_column_count, 0);
    assert_eq!(count_only.source_diagnostics[0].projected_output_column_count, 0);
    assert_eq!(count_only.source_diagnostics[0].pure_filter_column_count, 0);
    assert_eq!(count_only.source_diagnostics[0].zero_column_count_disposition, "represented");

    let filtered = report_row(&report, "filtered_reject");
    assert_eq!(filtered.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(filtered.unsupported_category.as_deref(), Some("aggregate_filter"));
    assert!(filtered.rejected_before_execution);

    let distributed_filtered_avg = report_row(&report, "distributed_filtered_avg_reject");
    assert_eq!(distributed_filtered_avg.reason_code.as_deref(), Some("native_distributed_aggregate_unsupported"));
    assert_eq!(distributed_filtered_avg.unsupported_category.as_deref(), Some("aggregate_filter"));
    assert!(distributed_filtered_avg.rejected_before_execution);

    let distributed_distinct_avg = report_row(&report, "distributed_distinct_avg_rewrite");
    assert_eq!(distributed_distinct_avg.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(distributed_distinct_avg.reason_code, None);
    assert_eq!(distributed_distinct_avg.unsupported_category, None);
    assert!(!distributed_distinct_avg.rejected_before_execution);

    let modulo = report_row(&report, "modulo_rewrite");
    assert_eq!(modulo.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(modulo.reason_code, None);
    assert_eq!(modulo.unsupported_category, None);
    assert!(!modulo.rejected_before_execution);

    let planning_failure = report_row(&report, "q00_planning_failure");
    assert_eq!(planning_failure.physical_root, "PlanningFailure");
    assert_eq!(planning_failure.physical_shape, "sql_to_logical_plan");
    assert_eq!(planning_failure.native_candidate_shape, None);
    assert_eq!(planning_failure.native_candidate_shape_family, None);
    assert_eq!(planning_failure.outcome, NativeFallbackPressureOutcome::Rejected);
    assert_eq!(planning_failure.reason_code.as_deref(), Some("sql_planning_failed"));
    assert_eq!(planning_failure.unsupported_category.as_deref(), Some("planning"));
    assert!(planning_failure.source_diagnostics.is_empty());
    assert!(planning_failure.rejected_before_execution);

    let text = report.to_tsv();
    assert_eq!(NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, 5);
    assert_eq!(
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tjoin_diagnostics\tsource_diagnostics\trejected_before_execution"
    );
    assert!(text.starts_with("# native_fallback_pressure_tsv_schema_version=5\n"));
    assert!(text.contains(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains("sum_rewrite\tSortExec"));
    assert!(text.contains("not_rewrite\tSortExec"));
    assert!(text.contains("cast_rewrite\tSortExec"));
    assert!(text.contains("avg_rewrite\tSortExec"));
    assert!(text.contains("distributed_avg_rewrite\tSortPreservingMergeExec"));
    assert!(text.contains("count_only_rewrite\t"));
    assert!(text.contains("zero_column_count=represented"));
    assert!(text.contains("\trewritten\tnone\tnone\t"));
    assert!(text.contains("\tnative_distributed_aggregate_unsupported\taggregate_filter\t"));
    assert!(text.contains("\tunsupported_expression\taggregate_filter\t"));
    assert!(text.contains("q00_planning_failure\tPlanningFailure\tsql_to_logical_plan\tnone\tnone\trejected\tsql_planning_failed\tplanning\tnone\tnone\ttrue"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_fallback_pressure_report_keeps_supported_typed_null_literals_out_of_unsupported_category() {
    let dir = unique_temp_dir("datafusion-cudf-native-fallback-pressure-typed-null-literal");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path).await;
    let plan = typed_null_and_try_cast_reject_plan(scan);

    let report = NativeFallbackPressureReport::collect([NativeFallbackPressureWorkloadItem::new(
        "typed_null_try_cast_reject",
        plan,
    )]);

    assert_eq!(report.rows.len(), 1);
    let row = report_row(&report, "typed_null_try_cast_reject");
    assert_eq!(row.outcome, NativeFallbackPressureOutcome::Rejected);
    assert_eq!(row.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(row.unsupported_category.as_deref(), Some("expression_try_cast"));
    assert_ne!(row.unsupported_category.as_deref(), Some("expression_literal"));
    assert!(row.rejected_before_execution);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_planning_report_summarizes_session_optimizer_outcomes() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-planning-report");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let accepted = native_datafusion_planning_report("sum_session", &path, SQL_SUM_QUERY).await;
    assert_eq!(accepted.rows.len(), 1);
    assert_eq!(accepted.accepted_count(), 1);
    assert_eq!(accepted.rejected_count(), 0);
    let accepted_row = &accepted.rows[0];
    assert_eq!(accepted_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(accepted_row.reason_code, None);
    assert_eq!(accepted_row.unsupported_category, None);
    assert_eq!(accepted_row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(accepted_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert!(!accepted_row.rejected_before_execution);
    assert_eq!(accepted_row.source_diagnostics.len(), 1);
    assert!(accepted_row.source_diagnostics[0].direct_device_read);
    assert_eq!(accepted_row.source_diagnostics[0].read_column_count, 4);
    assert_eq!(accepted_row.source_diagnostics[0].pure_filter_column_count, 1);

    let count_only = native_datafusion_planning_report("count_only_session", &path, SQL_COUNT_ONLY_QUERY).await;
    assert_eq!(count_only.rows.len(), 1);
    assert_eq!(count_only.accepted_count(), 1);
    assert_eq!(count_only.rejected_count(), 0);
    let count_only_row = &count_only.rows[0];
    assert_eq!(count_only_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(
        count_only_row.native_candidate_shape.as_deref(),
        Some("ProjectionExec(FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(LocalParquetScan))))")
    );
    assert_eq!(
        count_only_row.native_candidate_shape_family.as_deref(),
        Some("projected_global_local_parquet_distributed_aggregate")
    );
    assert_eq!(count_only_row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(count_only_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(count_only_row.source_diagnostics.len(), 1);
    assert_eq!(count_only_row.source_diagnostics[0].read_column_count, 0);
    assert_eq!(count_only_row.source_diagnostics[0].zero_column_count_disposition, "represented");

    let distributed_avg = native_datafusion_planning_report_with_session(
        "distributed_avg_session",
        &path,
        SQL_AVG_QUERY,
        SessionConfig::new(),
    )
    .await;
    assert_eq!(distributed_avg.rows.len(), 1);
    assert_eq!(distributed_avg.accepted_count(), 1);
    assert_eq!(distributed_avg.rejected_count(), 0);
    let distributed_avg_row = &distributed_avg.rows[0];
    assert_eq!(distributed_avg_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(distributed_avg_row.reason_code, None);
    assert_eq!(distributed_avg_row.unsupported_category, None);
    assert_eq!(distributed_avg_row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(distributed_avg_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert!(!distributed_avg_row.rejected_before_execution);

    let rejected = native_datafusion_planning_report("filtered_session", &path, SQL_FILTERED_QUERY).await;
    assert_eq!(rejected.rows.len(), 1);
    assert_eq!(rejected.accepted_count(), 0);
    assert_eq!(rejected.rejected_count(), 1);
    let rejected_row = &rejected.rows[0];
    assert_eq!(rejected_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
    assert_eq!(rejected_row.reason_code.as_deref(), Some("unsupported_expression"));
    assert_ne!(rejected_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert!(rejected_row.rejected_before_execution);
    assert_eq!(rejected_row.source_diagnostics.len(), 1);
    assert!(rejected_row.source_diagnostics[0].direct_device_read);

    let planning_failure =
        NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::planning_failure(
            "broken_session",
            "sql_to_logical_plan",
            "sql_planning_failed",
            "synthetic planning failure",
        )]);
    assert_eq!(planning_failure.rows.len(), 1);
    assert_eq!(planning_failure.accepted_count(), 0);
    assert_eq!(planning_failure.rejected_count(), 1);
    let planning_failure_row = &planning_failure.rows[0];
    assert_eq!(planning_failure_row.physical_root, "PlanningFailure");
    assert_eq!(planning_failure_row.physical_shape, "sql_to_logical_plan");
    assert_eq!(planning_failure_row.native_candidate_shape, None);
    assert_eq!(planning_failure_row.native_candidate_shape_family, None);
    assert_eq!(planning_failure_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
    assert_eq!(planning_failure_row.reason_code.as_deref(), Some("sql_planning_failed"));
    assert_eq!(planning_failure_row.unsupported_category.as_deref(), Some("planning"));
    assert!(planning_failure_row.source_diagnostics.is_empty());
    assert_eq!(planning_failure_row.post_native_plan_root, "PlanningFailure");
    assert_eq!(planning_failure_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Cpu);
    assert!(planning_failure_row.rejected_before_execution);

    let text = accepted.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION, 7);
    assert_eq!(
        NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tstrict_bounded_contract_rows\tstrict_bounded_compiled_graphs\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution"
    );
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=7\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("sum_session\t"));
    assert!(text.contains("\taccepted_replaced\tnone\tnone\t"));
    assert!(text.contains("\tNativeCudfExec\tnative\tfalse"));

    fs::remove_dir_all(dir).unwrap();
}
