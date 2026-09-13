use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    capability::CapabilityReason,
    native_exec::AlgeonNativeExecutionConfig,
    native_report::{
        CandidateOutcomeKind, FinalPlanDisposition, NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NativeDataFusionPlanningReport, NativeDataFusionPlanningRow,
        NativeDataFusionPlanningWorkloadItem, NativeFallbackPressureReport, NativeFallbackPressureRow,
        NativeFallbackPressureWorkloadItem,
    },
    planner::{AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule, NativePlanningDiagnostics},
    report::{GpuCandidate, PlanningReport},
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{Float64Array, Int32Array, Int64Array, StringArray},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{ScalarValue, config::ConfigOptions},
    execution::SessionStateBuilder,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_expr::{
        LexOrdering, PhysicalExpr, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{TryCastExpr, binary, col, lit},
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use query_engine::mode::StatisticalAggregateExecutionMode;
use workloads::tpch::{TPCH_QUERY_COUNT, tiny_tpch_dataset_metadata, tpch_query_ids, write_tiny_tpch_parquet_dataset};

#[path = "../support/analytical_olap_planning.rs"]
mod analytical_olap_planning;
#[path = "../support/tpch_report.rs"]
mod tpch_report;

use analytical_olap_planning::analytical_olap_datafusion_planning_workload_items;
use tpch_report::{
    TestReportDatasetMetadataExt, tpch_datafusion_planning_workload_items, tpch_scalar_fallback_pressure_workload_items,
};

use crate::{
    analytical_olap_workload::*, tpch_fallback_workload::tpch_fallback_pressure_workload_items,
    tpch_session::tpch_report_session_config,
};

const EXPECTED_BOUNDED_TPCH_ACCEPTED: &[&str] = &[
    "q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10", "q11", "q12", "q13", "q14", "q15", "q16",
    "q19", "q21", "q17", "q18", "q20", "q22",
];
const EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED: &[&str] = &[
    "q02", "q03", "q04", "q05", "q07", "q08", "q09", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q19",
    "q21", "q18", "q20", "q22",
];

#[path = "../native_fallback_pressure_tests/analytical_olap.rs"]
mod analytical_olap;
#[path = "../native_fallback_pressure_tests/sql.rs"]
mod sql;
#[path = "../native_fallback_pressure_tests/support.rs"]
mod support;
#[path = "../native_fallback_pressure_tests/tpch_reports.rs"]
mod tpch_reports;

use sql::*;
use support::*;

#[tokio::test]
async fn native_fallback_pressure_report_summarizes_workload_outcomes() {
    let dir = unique_temp_dir("algeon-datafusion-native-fallback-pressure");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let workload = [
        ("sum_rewrite", SQL_SUM_QUERY),
        ("not_rewrite", SQL_NOT_QUERY),
        ("cast_rewrite", SQL_CAST_QUERY),
        ("avg_rewrite", SQL_AVG_QUERY),
        ("filtered_rewrite", SQL_FILTERED_QUERY),
        ("integer_modulo_fallback", SQL_MODULO_QUERY),
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
        ("distributed_filtered_avg_rewrite", SQL_FILTERED_AVG_QUERY),
        ("distributed_distinct_avg_rewrite", SQL_DISTINCT_AVG_QUERY),
    ] {
        items.push(NativeFallbackPressureWorkloadItem::new(query_id, sql_plan(&path, SessionConfig::new(), sql).await));
    }
    items.push(NativeFallbackPressureWorkloadItem::new(
        "count_only_rewrite",
        sql_plan(&path, count_only_sql_session_config(), SQL_COUNT_ONLY_QUERY).await,
    ));
    let try_cast_reject = typed_null_and_try_cast_reject_plan(parquet_scan(&path).await);
    items.push(NativeFallbackPressureWorkloadItem::new("try_cast_reject", try_cast_reject));
    items.push(NativeFallbackPressureWorkloadItem::planning_failure(
        "q00_planning_failure",
        "sql_to_logical_plan",
        "sql_planning_failed",
        "No field named missing_col",
    ));

    let report = NativeFallbackPressureReport::collect(items);

    assert_eq!(report.rows.len(), 12);
    assert_eq!(report.selected_count(), 9);
    assert_eq!(report.not_selected_count(), 2);

    let rewritten = report_row(&report, "sum_rewrite");
    assert_eq!(rewritten.outcome, Some(CandidateOutcomeKind::Selected));
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
    assert_eq!(not.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(not.reason_code, None);
    assert_eq!(not.unsupported_category, None);
    assert!(!not.rejected_before_execution);

    let cast = report_row(&report, "cast_rewrite");
    assert_eq!(cast.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(cast.reason_code, None);
    assert_eq!(cast.unsupported_category, None);
    assert!(!cast.rejected_before_execution);

    let avg = report_row(&report, "avg_rewrite");
    assert_eq!(avg.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(avg.reason_code, None);
    assert_eq!(avg.unsupported_category, None);
    assert!(!avg.rejected_before_execution);

    let distributed_avg = report_row(&report, "distributed_avg_rewrite");
    assert_eq!(distributed_avg.outcome, Some(CandidateOutcomeKind::Selected));
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
    assert_eq!(count_only.outcome, Some(CandidateOutcomeKind::Selected));
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

    let filtered = report_row(&report, "filtered_rewrite");
    assert_eq!(filtered.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(filtered.reason_code, None);
    assert_eq!(filtered.unsupported_category, None);
    assert!(!filtered.rejected_before_execution);

    let distributed_filtered_avg = report_row(&report, "distributed_filtered_avg_rewrite");
    assert_eq!(distributed_filtered_avg.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(distributed_filtered_avg.reason_code, None);
    assert_eq!(distributed_filtered_avg.unsupported_category, None);
    assert!(!distributed_filtered_avg.rejected_before_execution);

    let try_cast = report_row(&report, "try_cast_reject");
    assert_eq!(try_cast.outcome, Some(CandidateOutcomeKind::NotSupported));
    assert_eq!(try_cast.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(try_cast.unsupported_category.as_deref(), Some("expression_try_cast_int64_to_int32"));
    assert!(!try_cast.rejected_before_execution);

    let distributed_distinct_avg = report_row(&report, "distributed_distinct_avg_rewrite");
    assert_eq!(distributed_distinct_avg.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(distributed_distinct_avg.reason_code, None);
    assert_eq!(distributed_distinct_avg.unsupported_category, None);
    assert!(!distributed_distinct_avg.rejected_before_execution);

    // Integer modulo cannot match DataFusion by-zero errors on GPU, so the
    // native candidate is left on the executable DataFusion baseline.
    let modulo = report_row(&report, "integer_modulo_fallback");
    assert_eq!(modulo.outcome, Some(CandidateOutcomeKind::NotSupported));
    assert_eq!(modulo.reason_code.as_deref(), Some("unsupported_expression"));
    assert!(!modulo.rejected_before_execution);

    let planning_failure = report_row(&report, "q00_planning_failure");
    assert_eq!(planning_failure.physical_root, "PlanningFailure");
    assert_eq!(planning_failure.physical_shape, "sql_to_logical_plan");
    assert_eq!(planning_failure.native_candidate_shape, None);
    assert_eq!(planning_failure.native_candidate_shape_family, None);
    assert_eq!(planning_failure.outcome, None);
    assert_eq!(planning_failure.reason_code.as_deref(), Some("sql_planning_failed"));
    assert_eq!(planning_failure.unsupported_category.as_deref(), Some("planning"));
    assert!(planning_failure.source_diagnostics.is_empty());
    assert!(planning_failure.rejected_before_execution);

    let text = report.to_tsv();
    assert_eq!(
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tjoin_diagnostics\tsource_diagnostics\trejected_before_execution"
    );
    assert!(text.starts_with(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains("sum_rewrite\tProjectionExec"));
    assert!(text.contains("not_rewrite\tProjectionExec"));
    assert!(text.contains("cast_rewrite\tProjectionExec"));
    assert!(text.contains("avg_rewrite\tProjectionExec"));
    assert!(text.contains("distributed_avg_rewrite\tSortPreservingMergeExec"));
    assert!(text.contains("count_only_rewrite\t"));
    assert!(text.contains("zero_column_count=represented"));
    assert!(text.contains("\tselected\tnone\tnone\t"));
    assert!(text.contains("\tunsupported_expression\texpression_try_cast_int64_to_int32\t"));
    assert!(!text.contains("\taggregate_filter\t"));
    assert!(text.contains(
        "q00_planning_failure\tPlanningFailure\tsql_to_logical_plan\tnone\tnone\tnone\tsql_planning_failed\tplanning\tnone\tnone\ttrue"
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_planning_report_does_not_blame_supported_topk_wrapper() {
    let dir = unique_temp_dir("algeon-datafusion-native-planning-topk-category");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT a.key FROM bench AS a CROSS JOIN bench AS b ORDER BY a.key LIMIT 1",
    )
    .await;
    assert_eq!(plan.name(), "SortExec(TopK)");

    let mut planning_report = PlanningReport::new();
    planning_report.push_not_supported(
        GpuCandidate::new("SortExec(fetch)(ProjectionExec(CrossJoinExec(LocalParquetScan,LocalParquetScan)))", 1),
        CapabilityReason::UnsupportedOperator,
    );
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: plan.name().to_string(),
        input_plan_shape: "SortExec(fetch)(ProjectionExec(CrossJoinExec(DataSourceExec,DataSourceExec)))".to_string(),
        input_join_diagnostics: Vec::new(),
        report: planning_report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: Default::default(),
    };

    let report = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "topk_cross_join",
        plan,
        diagnostics,
    )]);
    let row = planning_replacement_row(&report, "topk_cross_join");

    assert_eq!(row.reason_code.as_deref(), Some("unsupported_operator"));
    assert_eq!(row.unsupported_category.as_deref(), Some("operator_crossjoin"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_fallback_pressure_report_keeps_supported_typed_null_literals_out_of_unsupported_category() {
    let dir = unique_temp_dir("algeon-datafusion-native-fallback-pressure-typed-null-literal");
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
    assert_eq!(row.outcome, Some(CandidateOutcomeKind::NotSupported));
    assert_eq!(row.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(row.unsupported_category.as_deref(), Some("expression_try_cast_int64_to_int32"));
    assert_ne!(row.unsupported_category.as_deref(), Some("expression_literal"));
    assert!(!row.rejected_before_execution);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_fallback_pressure_reports_narrow_decimal_literals_as_unsupported_expressions() {
    let dir = unique_temp_dir("algeon-datafusion-native-fallback-pressure-narrow-decimal-literal");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path).await;
    let projection = |value, alias| {
        Arc::new(ProjectionExec::try_new(vec![ProjectionExpr::new(lit(value), alias)], Arc::clone(&scan)).unwrap())
            as Arc<dyn ExecutionPlan>
    };
    let report = NativeFallbackPressureReport::collect([
        NativeFallbackPressureWorkloadItem::new(
            "decimal32_literal",
            projection(ScalarValue::Decimal32(Some(123), 9, 2), "amount32"),
        ),
        NativeFallbackPressureWorkloadItem::new(
            "decimal64_literal",
            projection(ScalarValue::Decimal64(Some(-456), 18, 2), "amount64"),
        ),
    ]);

    for query_id in ["decimal32_literal", "decimal64_literal"] {
        let row = report_row(&report, query_id);
        assert_eq!(row.outcome, Some(CandidateOutcomeKind::NotSupported));
        assert_eq!(row.reason_code.as_deref(), Some("unsupported_expression"));
        assert_eq!(row.unsupported_category.as_deref(), Some("expression"));
        assert!(!row.rejected_before_execution);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_planning_report_summarizes_session_optimizer_outcomes() {
    let dir = unique_temp_dir("algeon-datafusion-native-datafusion-planning-report");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let accepted = native_datafusion_planning_report("sum_session", &path, SQL_SUM_QUERY).await;
    assert_eq!(accepted.rows.len(), 1);
    assert_eq!(accepted.selected_count(), 1);
    assert_eq!(accepted.not_selected_count(), 0);
    let accepted_row = &accepted.rows[0];
    assert_eq!(accepted_row.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(accepted_row.reason_code, None);
    assert_eq!(accepted_row.unsupported_category, None);
    assert_eq!(accepted_row.post_native_plan_root, "AlgeonNativeExec");
    assert_eq!(accepted_row.final_disposition, Some(FinalPlanDisposition::Native));
    assert!(!accepted_row.rejected_before_execution);
    assert_eq!(accepted_row.source_diagnostics.len(), 1);
    assert!(accepted_row.source_diagnostics[0].direct_device_read);
    assert_eq!(accepted_row.source_diagnostics[0].read_column_count, 4);
    assert_eq!(accepted_row.source_diagnostics[0].pure_filter_column_count, 1);

    let count_only = native_datafusion_planning_report("count_only_session", &path, SQL_COUNT_ONLY_QUERY).await;
    assert_eq!(count_only.rows.len(), 1);
    assert_eq!(count_only.selected_count(), 1);
    assert_eq!(count_only.not_selected_count(), 0);
    let count_only_row = &count_only.rows[0];
    assert_eq!(count_only_row.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(
        count_only_row.native_candidate_shape.as_deref(),
        Some("ProjectionExec(FinalAggregateExec(CoalescePartitionsExec(PartialAggregateExec(LocalParquetScan))))")
    );
    assert_eq!(
        count_only_row.native_candidate_shape_family.as_deref(),
        Some("projected_global_local_parquet_distributed_aggregate")
    );
    assert_eq!(count_only_row.post_native_plan_root, "AlgeonNativeExec");
    assert_eq!(count_only_row.final_disposition, Some(FinalPlanDisposition::Native));
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
    assert_eq!(distributed_avg.selected_count(), 1);
    assert_eq!(distributed_avg.not_selected_count(), 0);
    let distributed_avg_row = &distributed_avg.rows[0];
    assert_eq!(distributed_avg_row.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(distributed_avg_row.reason_code, None);
    assert_eq!(distributed_avg_row.unsupported_category, None);
    assert_eq!(distributed_avg_row.post_native_plan_root, "AlgeonNativeExec");
    assert_eq!(distributed_avg_row.final_disposition, Some(FinalPlanDisposition::Native));
    assert!(!distributed_avg_row.rejected_before_execution);

    let rejected_plan = typed_null_and_try_cast_reject_plan(parquet_scan(&path).await);
    let optimizer_config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(
        AlgeonNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1),
    );
    let native_rule = AlgeonNativePhysicalOptimizerRule::new(optimizer_config);
    let rejected_plan = native_rule.optimize(rejected_plan, &ConfigOptions::default()).unwrap();
    let rejected = NativeDataFusionPlanningReport::from_retained_rule("try_cast_session", rejected_plan, &native_rule);
    assert_eq!(rejected.rows.len(), 1);
    assert_eq!(rejected.selected_count(), 0);
    assert_eq!(rejected.not_selected_count(), 1);
    let rejected_row = &rejected.rows[0];
    assert_eq!(rejected_row.outcome, Some(CandidateOutcomeKind::NotSupported));
    assert_eq!(rejected_row.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(rejected_row.unsupported_category.as_deref(), Some("expression_try_cast_int64_to_int32"));
    assert_eq!(rejected_row.final_disposition, Some(FinalPlanDisposition::DataFusion));
    assert!(!rejected_row.rejected_before_execution);
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
    assert_eq!(planning_failure.selected_count(), 0);
    assert_eq!(planning_failure.not_selected_count(), 0);
    let planning_failure_row = &planning_failure.rows[0];
    assert_eq!(planning_failure_row.physical_root, "PlanningFailure");
    assert_eq!(planning_failure_row.physical_shape, "sql_to_logical_plan");
    assert_eq!(planning_failure_row.native_candidate_shape, None);
    assert_eq!(planning_failure_row.native_candidate_shape_family, None);
    assert_eq!(planning_failure_row.outcome, None);
    assert_eq!(planning_failure_row.reason_code.as_deref(), Some("sql_planning_failed"));
    assert_eq!(planning_failure_row.unsupported_category.as_deref(), Some("planning"));
    assert!(planning_failure_row.source_diagnostics.is_empty());
    assert_eq!(planning_failure_row.post_native_plan_root, "PlanningFailure");
    assert_eq!(planning_failure_row.final_disposition, None);
    assert_eq!(planning_failure_row.native_plan_fragment_count, None);
    assert_eq!(planning_failure_row.host_datafusion_boundary_count, None);
    assert!(planning_failure_row.rejected_before_execution);

    let text = accepted.to_tsv();
    assert_eq!(
        NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tcapability_decision_rows\tcapability_compiled_graphs\tnative_optimizer\tedge_normalize_inserted\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\trejected_before_execution"
    );
    assert!(text.starts_with(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("sum_session\t"));
    assert!(text.contains("\tselected\tnone\tnone\t"));
    assert!(text.contains("\tfalse\tnone\t"));
    assert!(text.contains("\tAlgeonNativeExec\tnative\t1\t0\tfalse"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn exact_statistical_aggregate_mode_keeps_aggregates_outside_the_host_kernel_on_datafusion() {
    let dir = unique_temp_dir("algeon-datafusion-exact-statistical-aggregate-planning");
    let path = dir.join("input.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("label", DataType::Utf8, false),
        Field::new("m00", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 5.0])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let sql = "SELECT key, MIN(label) AS first_label, STDDEV_SAMP(m00) AS m00_stddev FROM bench GROUP BY key";
    let planning_report = |query_id: &'static str, mode: StatisticalAggregateExecutionMode| {
        let optimizer_config = AlgeonNativeOptimizerConfig::default()
            .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
                256 * 1024 * 1024,
                64 * 1024 * 1024,
                1,
                3,
                1,
            ))
            .with_statistical_aggregate_execution_mode(mode);
        native_datafusion_planning_report_with_optimizer_config(
            query_id,
            &path,
            sql,
            direct_friendly_sql_session_config(),
            optimizer_config,
        )
    };

    let tolerant = planning_report("tolerant", StatisticalAggregateExecutionMode::GpuNativeTolerant).await;
    assert_eq!(tolerant.rows.len(), 1);
    assert_eq!(tolerant.rows[0].final_disposition, Some(FinalPlanDisposition::Native));

    let exact = planning_report("exact", StatisticalAggregateExecutionMode::DataFusionExactHostWelford).await;
    assert_eq!(exact.rows.len(), 1);
    let exact_row = &exact.rows[0];
    assert_eq!(exact_row.outcome, Some(CandidateOutcomeKind::NotSupported));
    assert_eq!(exact_row.reason_code.as_deref(), Some("unsupported_type"));
    assert_eq!(exact_row.final_disposition, Some(FinalPlanDisposition::DataFusion));

    fs::remove_dir_all(dir).unwrap();
}
