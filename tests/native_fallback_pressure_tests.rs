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
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt,
    GpuFallbackPolicy, NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeCudfExecutionConfig,
    NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPostNativeExecutionPath,
    NativeFallbackPressureOutcome, NativeFallbackPressureReport, NativeFallbackPressureWorkloadItem, TPCH_QUERY_COUNT,
    TPCH_TABLES, tiny_tpch_dataset_metadata, tpch_datafusion_planning_workload_items,
    tpch_fallback_pressure_workload_items, tpch_query_ids, tpch_report_session_config, write_tiny_tpch_parquet_dataset,
};

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
        ("distributed_distinct_avg_reject", SQL_DISTINCT_AVG_QUERY),
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
    assert_eq!(report.rewritten_count(), 7);
    assert_eq!(report.rejected_count(), 4);

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
            "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec(RepartitionExec(PartialAggregateExec(FilterExec(RepartitionExec(LocalParquetScan)))))))",
        )
    );
    assert_eq!(distributed_avg.reason_code, None);
    assert_eq!(distributed_avg.unsupported_category, None);
    assert!(!distributed_avg.rejected_before_execution);

    let count_only = report_row(&report, "count_only_rewrite");
    assert_eq!(count_only.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(count_only.native_candidate_shape.as_deref(), Some("ProjectionExec(AggregateExec(LocalParquetScan))"));
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

    let distributed_distinct_avg = report_row(&report, "distributed_distinct_avg_reject");
    assert_eq!(distributed_distinct_avg.reason_code.as_deref(), Some("unsupported_operator"));
    assert_eq!(distributed_distinct_avg.unsupported_category.as_deref(), Some("operator_sortpreservingmerge"));
    assert!(distributed_distinct_avg.rejected_before_execution);

    let modulo = report_row(&report, "modulo_rewrite");
    assert_eq!(modulo.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(modulo.reason_code, None);
    assert_eq!(modulo.unsupported_category, None);
    assert!(!modulo.rejected_before_execution);

    let planning_failure = report_row(&report, "q00_planning_failure");
    assert_eq!(planning_failure.physical_root, "PlanningFailure");
    assert_eq!(planning_failure.physical_shape, "sql_to_logical_plan");
    assert_eq!(planning_failure.native_candidate_shape, None);
    assert_eq!(planning_failure.outcome, NativeFallbackPressureOutcome::Rejected);
    assert_eq!(planning_failure.reason_code.as_deref(), Some("sql_planning_failed"));
    assert_eq!(planning_failure.unsupported_category.as_deref(), Some("planning"));
    assert!(planning_failure.source_diagnostics.is_empty());
    assert!(planning_failure.rejected_before_execution);

    let text = report.to_tsv();
    assert_eq!(NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, 3);
    assert_eq!(
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\toutcome\treason_code\tunsupported_category\tsource_diagnostics\trejected_before_execution"
    );
    assert!(text.starts_with("# native_fallback_pressure_tsv_schema_version=3\n"));
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
    assert!(text.contains("\tunsupported_operator\toperator_sortpreservingmerge\t"));
    assert!(text.contains("\tunsupported_expression\taggregate_filter\t"));
    assert!(text.contains("q00_planning_failure\tPlanningFailure\tsql_to_logical_plan\tnone\trejected\tsql_planning_failed\tplanning\tnone\ttrue"));

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

    let text = accepted.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION, 3);
    assert_eq!(
        NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\toutcome\treason_code\tunsupported_category\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution"
    );
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=3\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("sum_session\t"));
    assert!(text.contains("\taccepted_replaced\tnone\tnone\t"));
    assert!(text.contains("\tNativeCudfExec\tnative\tfalse"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_workload_builder_generates_stable_query_items() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-workload-builder");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    for table in TPCH_TABLES {
        assert!(dir.join(format!("{table}.parquet")).exists());
    }
    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    assert_eq!(metadata.scale_factor, "tiny");
    assert_eq!(metadata.provenance, "generated_tiny");
    assert_eq!(metadata.generation_command, "datafusion-cudf::write_tiny_tpch_parquet_dataset");
    assert_eq!(metadata.table_count, TPCH_TABLES.len());
    assert_eq!(metadata.file_count, TPCH_TABLES.len());
    assert_eq!(metadata.total_rows, 26);
    assert_eq!(
        metadata.table_row_counts(),
        "customer=3,lineitem=3,nation=5,orders=3,part=3,partsupp=3,region=3,supplier=3"
    );
    assert_eq!(
        metadata.table_file_counts(),
        "customer=1,lineitem=1,nation=1,orders=1,part=1,partsupp=1,region=1,supplier=1"
    );
    assert!(metadata.fingerprint.starts_with("metadata-fnv1a64:"));

    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let actual_ids = items.iter().map(|item| item.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(TPCH_QUERY_COUNT, 22);
    assert_eq!(items.len(), TPCH_QUERY_COUNT);
    assert_eq!(actual_ids, expected_ids);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_fallback_pressure_report_emits_one_pre_execution_reject_per_query() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-fallback-pressure");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items).with_tpch_dataset_metadata(metadata.clone());

    assert_eq!(report.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), TPCH_QUERY_COUNT);

    let actual_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);
    assert_eq!(TPCH_FALLBACK_PRESSURE_BASELINE.len(), TPCH_QUERY_COUNT);

    for (row, expected) in report.rows.iter().zip(TPCH_FALLBACK_PRESSURE_BASELINE) {
        assert_eq!(row.query_id, expected.query_id);
        assert_eq!(row.outcome, expected.outcome, "unexpected outcome for {}", row.query_id);
        assert_eq!(row.reason_code.as_deref(), Some(expected.reason_code), "unexpected reason for {}", row.query_id);
        assert_eq!(
            row.unsupported_category.as_deref(),
            Some(expected.unsupported_category),
            "unexpected unsupported category for {}",
            row.query_id
        );
        assert!(row.rejected_before_execution, "{} should reject before native execution", row.query_id);
    }

    let text = report.to_tsv();
    assert!(text.starts_with("# native_fallback_pressure_tsv_schema_version=3\n"));
    assert!(text.contains("# tpch_dataset_scale_factor=tiny\n"));
    assert!(text.contains("# tpch_dataset_provenance=generated_tiny\n"));
    assert!(text.contains("# tpch_dataset_generation_command=datafusion-cudf::write_tiny_tpch_parquet_dataset\n"));
    assert!(text.contains("# tpch_dataset_table_count=8\n"));
    assert!(text.contains("# tpch_dataset_file_count=8\n"));
    assert!(text.contains("# tpch_dataset_total_rows=26\n"));
    assert!(text.contains(&format!("# tpch_dataset_total_bytes={}\n", metadata.total_bytes)));
    assert!(text.contains(&format!("# tpch_dataset_fingerprint={}\n", metadata.fingerprint)));
    assert!(text.contains("# tpch_dataset_table_file_counts=customer=1,lineitem=1,nation=1,orders=1,part=1,partsupp=1,region=1,supplier=1\n"));
    assert!(text.contains("# tpch_dataset_table_row_counts=customer=3,lineitem=3,nation=5,orders=3,part=3,partsupp=3,region=3,supplier=3\n"));
    assert!(text.contains(&format!("# tpch_dataset_table_byte_sizes={}\n", metadata.table_byte_sizes())));
    assert!(text.contains(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q22\t"));
    assert!(!text.contains("\trewritten\t"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_datafusion_planning_report_emits_one_stable_row_per_query() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-datafusion-planning");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeDataFusionPlanningReport::collect(items).with_tpch_dataset_metadata(metadata.clone());

    assert_eq!(report.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(report.accepted_count(), 0);
    assert_eq!(report.rejected_count(), TPCH_QUERY_COUNT);

    let actual_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);

    for row in &report.rows {
        assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
        assert_eq!(row.reason_code.as_deref(), Some("unsupported_operator"));
        assert!(row.unsupported_category.is_some(), "missing unsupported category for {}", row.query_id);
        assert_ne!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
        assert!(row.rejected_before_execution, "{} should reject before native execution", row.query_id);
    }

    let text = report.to_tsv();
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=3\n"));
    assert!(text.contains("# tpch_dataset_scale_factor=tiny\n"));
    assert!(text.contains(&format!("# tpch_dataset_fingerprint={}\n", metadata.fingerprint)));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q22\t"));
    assert!(!text.contains("\taccepted_replaced\t"));

    fs::remove_dir_all(dir).unwrap();
}

#[derive(Debug)]
struct TpchFallbackPressureBaseline {
    query_id: &'static str,
    outcome: NativeFallbackPressureOutcome,
    reason_code: &'static str,
    unsupported_category: &'static str,
}

const TPCH_FALLBACK_PRESSURE_BASELINE: &[TpchFallbackPressureBaseline] = &[
    TpchFallbackPressureBaseline {
        query_id: "q01",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_shape",
    },
    TpchFallbackPressureBaseline {
        query_id: "q02",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_sort_topk",
    },
    TpchFallbackPressureBaseline {
        query_id: "q03",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_sort_topk",
    },
    TpchFallbackPressureBaseline {
        query_id: "q04",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q05",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q06",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "aggregate_global",
    },
    TpchFallbackPressureBaseline {
        query_id: "q07",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q08",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q09",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q10",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_sort_topk",
    },
    TpchFallbackPressureBaseline {
        query_id: "q11",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_nestedloopjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q12",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q13",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q14",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q15",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q16",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q17",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q18",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_sort_topk",
    },
    TpchFallbackPressureBaseline {
        query_id: "q19",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q20",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_hashjoin",
    },
    TpchFallbackPressureBaseline {
        query_id: "q21",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_sort_topk",
    },
    TpchFallbackPressureBaseline {
        query_id: "q22",
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: "unsupported_operator",
        unsupported_category: "operator_nestedloopjoin",
    },
];

fn report_row<'a>(
    report: &'a NativeFallbackPressureReport,
    query_id: &str,
) -> &'a datafusion_cudf::NativeFallbackPressureRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(direct_friendly_sql_session_config());
    let df = ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

fn typed_null_and_try_cast_reject_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let typed_null_comparison =
        binary(lit(ScalarValue::Int64(None)), Operator::Eq, lit(ScalarValue::Int64(None)), &scan_schema).unwrap();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int32));
    let try_cast_comparison =
        binary(try_cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let predicate = binary(typed_null_comparison, Operator::And, try_cast_comparison, &scan_schema).unwrap();
    aggregate_sort_limit_plan_with_filter(scan, predicate)
}

fn aggregate_sort_limit_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn datafusion::physical_expr::PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("m00", &filter_schema).unwrap(), alias: "m00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_m00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(2)));
    Arc::new(LocalLimitExec::new(sort, 2))
}

async fn native_datafusion_planning_report(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
) -> NativeDataFusionPlanningReport {
    native_datafusion_planning_report_with_session(query_id, path, sql, direct_friendly_sql_session_config()).await
}

async fn native_datafusion_planning_report_with_session(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
    session_config: SessionConfig,
) -> NativeDataFusionPlanningReport {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1));
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let gpu_island_rule = Arc::new(CudfPhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_and_gpu_island_rules(Arc::clone(&native_rule), gpu_island_rule)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    NativeDataFusionPlanningReport::from_retained_rule(query_id, plan, native_rule.as_ref())
}

fn direct_friendly_sql_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn count_only_sql_session_config() -> SessionConfig {
    let mut config = direct_friendly_sql_session_config().with_collect_statistics(false);
    config.options_mut().execution.collect_statistics = false;
    config
}

fn write_numeric_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Int64Array::from(vec![5, 10, 15, 20, 25, 30])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

const SQL_SUM_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_NOT_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE NOT (payload < 10)
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_CAST_QUERY: &str = "
SELECT key, SUM(CAST(key AS BIGINT)) AS sum_cast
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_cast DESC, key ASC
LIMIT 2
";

const SQL_AVG_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_COUNT_ONLY_QUERY: &str = "
SELECT COUNT(*) AS count_rows
FROM bench
";

const SQL_FILTERED_QUERY: &str = "
SELECT key, SUM(m00) FILTER (WHERE payload >= 10) AS sum_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_m00 DESC, key ASC
LIMIT 2
";

const SQL_FILTERED_AVG_QUERY: &str = "
SELECT key, AVG(m00) FILTER (WHERE payload >= 10) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_DISTINCT_AVG_QUERY: &str = "
SELECT key, AVG(DISTINCT m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_MODULO_QUERY: &str = "
SELECT key, SUM(payload % 2) AS modulo_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY modulo_payload DESC, key ASC
LIMIT 2
";
