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
    BoundedJoinReservationContract, ExecutionBoundaryPolicy, GpuResidentSource, LocalParquetSource, QueryPlanBuilder,
    SourceNode,
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
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt,
    GpuCandidate, GpuFallbackPolicy, NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
    NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
    NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION, NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
    NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig,
    NativeCudfGpuProducerExec, NativeDataFusionExecutionReport, NativeDataFusionExecutionWorkloadItem,
    NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntime, NativeDataFusionRuntimeOutcome,
    NativeFallbackPressureOutcome, NativeFallbackPressureReport, NativeFallbackPressureWorkloadItem,
    NativePlanningDiagnostics, PlanningReport, TPCH_QUERY_COUNT, TPCH_TABLES,
    describe_tpch_dataset_with_content_fingerprint, tiny_tpch_dataset_metadata,
    tpch_datafusion_planning_workload_items, tpch_fallback_pressure_workload_items, tpch_query_ids,
    tpch_report_session_config, tpch_scalar_fallback_pressure_workload_items, write_tiny_tpch_parquet_dataset,
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
    assert_eq!(NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION, 6);
    assert_eq!(
        NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tbounded_join_reservation_facts\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution"
    );
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=6\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("sum_session\t"));
    assert!(text.contains("\taccepted_replaced\tnone\tnone\t"));
    assert!(text.contains("\tNativeCudfExec\tnative\tfalse"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_datafusion_execution_report_records_runtime_boundaries() {
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "q01",
        native_exec_leaf_for_report(),
        accepted_native_diagnostics(),
        NativeDataFusionRuntime::Succeeded {
            output_batches: 1,
            output_rows: 4,
            output_bytes: 256,
            metrics: native_execution_metrics_set(),
        },
    )]);

    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.runtime_started_count(), 1);
    assert_eq!(report.rejected_before_execution_count(), 0);
    let row = &report.rows[0];
    assert_eq!(row.planning_outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert!(!row.rejected_before_execution);
    assert!(row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.output_batches, Some(1));
    assert_eq!(row.output_rows, Some(4));
    assert_eq!(row.output_bytes, Some(256));
    assert_eq!(row.metrics.gpu_resident_input_handoffs, 1);
    assert_eq!(row.metrics.host_to_device_uploads, 0);
    assert_eq!(row.metrics.host_to_device_uploaded_rows, 0);
    assert_eq!(row.metrics.host_to_device_uploaded_bytes, 0);
    assert_eq!(row.metrics.device_to_host_materializations, 1);
    assert_eq!(row.metrics.device_to_host_materialized_rows, 4);
    assert_eq!(row.metrics.device_to_host_materialized_bytes, 256);
    assert_eq!(row.metrics.device_to_device_copies, 2);
    assert_eq!(row.metrics.device_to_device_copied_bytes, 512);
    assert_eq!(row.metrics.parquet_files, 1);
    assert_eq!(row.metrics.parquet_bytes_known, 1024);
    assert_eq!(row.metrics.row_groups_selected, 2);
    assert_eq!(row.metrics.read_columns, 4);
    assert_eq!(row.metrics.resident_dataframe_peak_bytes, Some(4096));
    assert_eq!(row.metrics.allocator_high_water_bytes, Some(8192));
    assert_eq!(row.metrics.source_chunks_planned, 6);
    assert_eq!(row.metrics.source_chunks_attempted, 7);
    assert_eq!(row.metrics.source_chunks_committed, 5);
    assert_eq!(row.metrics.source_chunks_retried, 2);
    assert_eq!(row.metrics.row_groups_planned, 8);
    assert_eq!(row.metrics.row_groups_committed, 4);
    assert_eq!(row.metrics.partial_aggregate_count, 5);
    assert_eq!(row.metrics.partial_aggregate_rows, 9);
    assert_eq!(row.metrics.final_merge_attempts, 2);
    assert_eq!(row.metrics.final_merge_levels, 1);
    assert_eq!(row.metrics.retry_count, 3);
    assert_eq!(row.metrics.retry_resource_exhausted_count, 1);
    assert_eq!(row.metrics.retry_reservation_failure_count, 2);
    assert_eq!(row.metrics.hash_join_build_count, 1);
    assert_eq!(row.metrics.filtered_join_build_count, 6);
    assert_eq!(row.metrics.hash_join_cache_hits, 2);
    assert_eq!(row.metrics.hash_join_cache_misses, 1);
    assert_eq!(row.metrics.hash_join_durable_cache_hits, 3);
    assert_eq!(row.metrics.hash_join_durable_cache_misses, 4);
    assert_eq!(row.metrics.hash_join_durable_cache_inserts, 5);
    assert_eq!(row.metrics.hash_join_hash_table_bytes, 16384);
    assert_eq!(row.metrics.filtered_join_hash_table_bytes, 8192);
    assert_eq!(row.metrics.filtered_join_probe_state_bytes, 1024);
    assert_eq!(row.metrics.join_gather_map_bytes, 2048);

    let text = report.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, 7);
    assert_eq!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\tplanning_outcome\treason_code\tunsupported_category\tbounded_join_reservation_facts\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution\tnative_runtime_started\truntime_outcome\truntime_failure\toutput_batches\toutput_rows\toutput_bytes\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\treservation_denied_count\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\trow_groups_planned\trow_groups_committed\tpartial_aggregate_count\tpartial_aggregate_rows\tfinal_merge_attempts\tfinal_merge_levels\tretry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\thash_join_build_count\tfiltered_join_build_count\thash_join_cache_hits\thash_join_cache_misses\thash_join_durable_cache_hits\thash_join_durable_cache_misses\thash_join_durable_cache_inserts\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tjoin_gather_map_bytes\tsource_diagnostics"
    );
    assert!(text.starts_with("# native_datafusion_execution_tsv_schema_version=7\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_EXECUTION_TSV_HEADER}\n")));
    assert!(text.contains("\taccepted_replaced\tnone\tnone\tnone\tNativeCudfExec\tnative\tfalse\ttrue\tsucceeded\tnone\t1\t4\t256\t1\t0\t0\t0\t1\t4\t256\t2\t512\t1\t1024\t2\t4\t4096\t8192\t128\t128\t0\t6\t7\t5\t2\t8\t4\t5\t9\t2\t1\t3\t1\t2\t1\t6\t2\t1\t3\t4\t5\t16384\t8192\t1024\t2048\t"));
}

#[test]
fn native_datafusion_execution_report_keeps_rejected_queries_out_of_runtime() {
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "q03",
        native_exec_leaf_for_report(),
        rejected_native_diagnostics(),
        NativeDataFusionRuntime::NotStarted,
    )]);

    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.runtime_started_count(), 0);
    assert_eq!(report.rejected_before_execution_count(), 1);
    let row = &report.rows[0];
    assert_eq!(row.planning_outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
    assert_eq!(row.reason_code.as_deref(), Some("native_bounded_join_unsupported"));
    assert_eq!(row.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(
        row.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert!(row.rejected_before_execution);
    assert!(!row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::NotStarted);
    assert_eq!(row.output_batches, None);
    assert_eq!(row.metrics.device_to_host_materializations, 0);
}

#[test]
fn native_datafusion_reports_classify_composed_native_roots_as_native() {
    let composed = composed_native_exec_for_report();

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "composed",
        Arc::clone(&composed),
        accepted_native_diagnostics(),
    )]);
    assert_eq!(planning.rows[0].post_native_plan_root, "NativeCudfComposedExec");
    assert_eq!(planning.rows[0].post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);

    let execution = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "composed",
        composed,
        accepted_native_diagnostics(),
        NativeDataFusionRuntime::NotStarted,
    )]);
    assert_eq!(execution.rows[0].post_native_plan_root, "NativeCudfComposedExec");
    assert_eq!(execution.rows[0].post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
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
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert_eq!(metadata.content_fingerprint, None);

    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let actual_ids = items.iter().map(|item| item.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(TPCH_QUERY_COUNT, 22);
    assert_eq!(items.len(), TPCH_QUERY_COUNT);
    assert_eq!(actual_ids, expected_ids);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_metadata_accepts_directory_layouts_and_pins_fingerprint_semantics() {
    let source = unique_temp_dir("datafusion-cudf-tpch-directory-layout-source");
    let layout_a = unique_temp_dir("datafusion-cudf-tpch-directory-layout-a");
    let layout_b = unique_temp_dir("datafusion-cudf-tpch-directory-layout-b");
    write_tiny_tpch_parquet_dataset(&source).unwrap();
    copy_tiny_tpch_as_directory_layout(&source, &layout_a, &["part-000.parquet"]);
    copy_tiny_tpch_as_directory_layout(&source, &layout_b, &["renamed-000.parquet"]);

    let metadata =
        describe_tpch_dataset_with_content_fingerprint(&layout_a, "tiny", "directory_layout", "copy_tiny", true)
            .unwrap();
    assert_eq!(metadata.table_count, TPCH_TABLES.len());
    assert_eq!(metadata.file_count, TPCH_TABLES.len());
    assert_eq!(metadata.total_rows, 26);
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert!(metadata.content_fingerprint.as_deref().unwrap().starts_with("content-fnv1a64:"));

    let lineitem = metadata.tables.iter().find(|table| table.table == "lineitem").unwrap();
    assert_eq!(lineitem.path, layout_a.join("lineitem"));
    assert_eq!(lineitem.file_count, 1);
    assert_eq!(lineitem.files[0].relative_path, "lineitem/part-000.parquet");
    assert!(lineitem.schema.contains("l_orderkey"));

    let same_content_different_paths =
        describe_tpch_dataset_with_content_fingerprint(&layout_b, "tiny", "directory_layout", "copy_tiny", true)
            .unwrap();
    assert_ne!(
        metadata.metadata_fingerprint, same_content_different_paths.metadata_fingerprint,
        "metadata fingerprint must include table-relative file paths"
    );
    assert_ne!(
        metadata.content_fingerprint, same_content_different_paths.content_fingerprint,
        "content fingerprint includes relative paths as part of dataset identity"
    );

    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(layout_a).unwrap();
    fs::remove_dir_all(layout_b).unwrap();
}

#[tokio::test]
async fn tpch_workload_builder_registers_multi_file_directory_layout() {
    let source = unique_temp_dir("datafusion-cudf-tpch-multi-file-source");
    let layout = unique_temp_dir("datafusion-cudf-tpch-multi-file-layout");
    write_tiny_tpch_parquet_dataset(&source).unwrap();
    copy_tiny_tpch_as_directory_layout(&source, &layout, &["part-000.parquet", "part-001.parquet"]);

    let metadata =
        describe_tpch_dataset_with_content_fingerprint(&layout, "tiny", "directory_layout", "copy_tiny", false)
            .unwrap();
    assert_eq!(metadata.file_count, TPCH_TABLES.len() * 2);
    assert_eq!(metadata.total_rows, 52);
    assert_eq!(
        metadata.table_file_counts(),
        "customer=2,lineitem=2,nation=2,orders=2,part=2,partsupp=2,region=2,supplier=2"
    );

    let items = tpch_fallback_pressure_workload_items(&layout, tpch_report_session_config()).await.unwrap();
    assert_eq!(items.len(), TPCH_QUERY_COUNT);

    fs::remove_dir_all(source).unwrap();
    fs::remove_dir_all(layout).unwrap();
}

#[tokio::test]
async fn tpch_fallback_pressure_report_emits_one_pre_execution_reject_per_query() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-fallback-pressure");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items).with_tpch_dataset_metadata(metadata.clone());

    assert_eq!(report.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(report.rewritten_count(), TPCH_QUERY_COUNT);
    assert_eq!(report.rejected_count(), 0);

    let actual_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);
    assert_eq!(TPCH_FALLBACK_PRESSURE_BASELINE.len(), TPCH_QUERY_COUNT);

    for (row, expected) in report.rows.iter().zip(TPCH_FALLBACK_PRESSURE_BASELINE) {
        assert_eq!(row.query_id, expected.query_id);
        assert_eq!(row.outcome, expected.outcome, "unexpected outcome for {}", row.query_id);
        let expected_reason = (expected.reason_code != "none").then_some(expected.reason_code);
        let expected_category = (expected.unsupported_category != "none").then_some(expected.unsupported_category);
        assert_eq!(row.reason_code.as_deref(), expected_reason, "unexpected reason for {}", row.query_id);
        assert_eq!(
            row.unsupported_category.as_deref(),
            expected_category,
            "unexpected unsupported category for {}",
            row.query_id
        );
        assert_eq!(row.physical_root, expected_tpch_physical_root(row.query_id.as_str()));
        assert_eq!(
            row.native_candidate_shape.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape(row.query_id.as_str()),
            "unexpected native candidate shape for {}",
            row.query_id
        );
        assert_eq!(
            row.native_candidate_shape_family.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape_family(row.query_id.as_str()),
            "unexpected native candidate shape family for {}",
            row.query_id
        );
        assert!(
            row.physical_shape.contains(expected_tpch_physical_shape_marker(row.query_id.as_str())),
            "unexpected physical shape for {}: {}",
            row.query_id,
            row.physical_shape
        );
        assert_tpch_join_diagnostics(row.query_id.as_str(), &row.join_diagnostics);
        assert_eq!(
            row.rejected_before_execution,
            expected.outcome == NativeFallbackPressureOutcome::Rejected,
            "unexpected pre-execution rejection flag for {}",
            row.query_id
        );
    }
    let q06 = report_row(&report, "q06");
    assert_eq!(
        q06.native_candidate_shape.as_deref(),
        Some("ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))")
    );
    assert_eq!(q06.native_candidate_shape_family.as_deref(), Some("projected_global_local_parquet_aggregate"));

    let text = report.to_tsv();
    assert!(text.starts_with("# native_fallback_pressure_tsv_schema_version=5\n"));
    assert!(text.contains("# tpch_dataset_scale_factor=tiny\n"));
    assert!(text.contains("# tpch_dataset_provenance=generated_tiny\n"));
    assert!(text.contains("# tpch_dataset_generation_command=datafusion-cudf::write_tiny_tpch_parquet_dataset\n"));
    assert!(text.contains("# tpch_dataset_table_count=8\n"));
    assert!(text.contains("# tpch_dataset_file_count=8\n"));
    assert!(text.contains("# tpch_dataset_total_rows=26\n"));
    assert!(text.contains(&format!("# tpch_dataset_total_bytes={}\n", metadata.total_bytes)));
    assert!(text.contains(&format!("# tpch_dataset_metadata_fingerprint={}\n", metadata.metadata_fingerprint)));
    assert!(text.contains("# tpch_dataset_content_fingerprint=not_computed\n"));
    assert!(text.contains("# tpch_dataset_table_file_counts=customer=1,lineitem=1,nation=1,orders=1,part=1,partsupp=1,region=1,supplier=1\n"));
    assert!(text.contains("# tpch_dataset_table_row_counts=customer=3,lineitem=3,nation=5,orders=3,part=3,partsupp=3,region=3,supplier=3\n"));
    assert!(text.contains(&format!("# tpch_dataset_table_byte_sizes={}\n", metadata.table_byte_sizes())));
    assert!(text.contains(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q06\t"));
    assert!(text.contains("q22\t"));
    assert!(text.contains("\trewritten\t"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_scalar_fallback_pressure_report_pins_expression_categories() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-scalar-fallback-pressure");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_scalar_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items).with_tpch_dataset_metadata(metadata);

    let expected = [
        ("q12_scalar_in_list_null", "expression_in_list"),
        ("q22_scalar_substring_dynamic_start", "expression_substring_argument"),
        ("q07_scalar_extract_hour", "expression_extract_field"),
        ("q01_scalar_interval_minus_date", "expression_interval_arithmetic"),
    ];
    assert_eq!(report.rows.len(), expected.len());
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), expected.len());

    for (query_id, expected_category) in expected {
        let row = report_row(&report, query_id);
        assert_eq!(row.outcome, NativeFallbackPressureOutcome::Rejected, "unexpected outcome for {query_id}");
        assert_eq!(row.reason_code.as_deref(), Some("unsupported_expression"), "unexpected reason for {query_id}");
        assert_eq!(row.unsupported_category.as_deref(), Some(expected_category), "unexpected category for {query_id}");
        assert!(row.rejected_before_execution, "{query_id} should reject before native execution");
    }

    let text = report.to_tsv();
    assert!(text.contains("\tunsupported_expression\texpression_in_list\t"));
    assert!(text.contains("\tunsupported_expression\texpression_substring_argument\t"));
    assert!(text.contains("\tunsupported_expression\texpression_extract_field\t"));
    assert!(text.contains("\tunsupported_expression\texpression_interval_arithmetic\t"));

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
    assert_eq!(report.accepted_count(), 10);
    assert_eq!(report.rejected_count(), TPCH_QUERY_COUNT - 10);

    let actual_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);

    for row in &report.rows {
        if matches!(
            row.query_id.as_str(),
            "q01" | "q03" | "q04" | "q05" | "q06" | "q08" | "q09" | "q10" | "q12" | "q14"
        ) {
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
            assert_eq!(row.reason_code.as_deref(), None);
            assert_eq!(row.unsupported_category.as_deref(), None);
            assert!(row.bounded_join_reservation_facts.is_empty());
            assert_eq!(row.post_native_plan_root, "NativeCudfExec");
            assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            assert!(!row.rejected_before_execution, "{} should be replaced by native execution", row.query_id);
        } else {
            let (expected_reason, expected_category) =
                expected_tpch_datafusion_planning_rejection(row.query_id.as_str());
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
            assert_eq!(row.reason_code.as_deref(), Some(expected_reason), "unexpected reason for {}", row.query_id);
            assert_eq!(
                row.unsupported_category.as_deref(),
                Some(expected_category),
                "unexpected unsupported category for {}",
                row.query_id
            );
            if expected_reason == "native_bounded_join_unsupported" {
                assert_eq!(
                    row.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str),
                    Some(""),
                    "missing bounded join blocker facts for {}",
                    row.query_id
                );
                assert_eq!(
                    row.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
                    Some(""),
                    "missing lower-level blocker facts for {}",
                    row.query_id
                );
            } else {
                assert!(row.bounded_join_reservation_facts.is_empty());
            }
            assert_ne!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            assert!(row.rejected_before_execution, "{} should reject before native execution", row.query_id);
        }
        assert_eq!(row.physical_root, expected_tpch_physical_root(row.query_id.as_str()));
        assert_eq!(
            row.native_candidate_shape.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape(row.query_id.as_str()),
            "unexpected native candidate shape for {}",
            row.query_id
        );
        assert_eq!(
            row.native_candidate_shape_family.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape_family(row.query_id.as_str()),
            "unexpected native candidate shape family for {}",
            row.query_id
        );
        assert!(
            row.physical_shape.contains(expected_tpch_physical_shape_marker(row.query_id.as_str())),
            "unexpected physical shape for {}: {}",
            row.query_id,
            row.physical_shape
        );
        if matches!(row.query_id.as_str(), "q03" | "q04" | "q05" | "q08" | "q09" | "q10" | "q12" | "q14")
            && row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        {
            assert!(
                row.join_diagnostics.is_empty(),
                "{} accepted replacement should not expose join diagnostics",
                row.query_id
            );
        } else {
            assert_tpch_join_diagnostics(row.query_id.as_str(), &row.join_diagnostics);
        }
    }

    let text = report.to_tsv();
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=6\n"));
    assert!(text.contains("# tpch_dataset_scale_factor=tiny\n"));
    assert!(text.contains(&format!("# tpch_dataset_metadata_fingerprint={}\n", metadata.metadata_fingerprint)));
    assert!(text.contains("# tpch_dataset_content_fingerprint=not_computed\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q06\t"));
    assert!(text.contains("q22\t"));
    assert!(text.contains("\taccepted_replaced\t"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_phase5_progression_contract_pairs_fallback_pressure_and_bounded_planning() {
    let dir = unique_temp_dir("datafusion-cudf-tpch-phase5-progression");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let fallback = NativeFallbackPressureReport::collect(
        tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap(),
    );
    let planning = NativeDataFusionPlanningReport::collect(
        tpch_datafusion_planning_workload_items(&dir, tpch_report_session_config()).await.unwrap(),
    );

    assert_eq!(fallback.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(planning.rows.len(), TPCH_QUERY_COUNT);
    for query_id in tpch_query_ids() {
        let fallback_row = report_row(&fallback, query_id);
        let planning_row = planning_row(&planning, query_id);
        assert_eq!(
            fallback_row.native_candidate_shape_family, planning_row.native_candidate_shape_family,
            "{query_id} should preserve the same native candidate family across fallback-pressure and bounded planning"
        );

        match fallback_row.outcome {
            NativeFallbackPressureOutcome::Rewritten if matches!(query_id, "q01" | "q06") => {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
                assert_eq!(planning_row.reason_code, None);
                assert_eq!(planning_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            },
            NativeFallbackPressureOutcome::Rewritten
                if matches!(query_id, "q03" | "q04" | "q05" | "q08" | "q09" | "q10" | "q12" | "q14") =>
            {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
                assert_eq!(planning_row.reason_code, None);
                assert_eq!(planning_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            },
            NativeFallbackPressureOutcome::Rewritten => {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
                assert_eq!(planning_row.reason_code.as_deref(), Some("native_bounded_join_unsupported"));
                assert_ne!(planning_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            },
            NativeFallbackPressureOutcome::Rejected => {
                assert_eq!(planning_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
                assert_eq!(
                    planning_row.reason_code, fallback_row.reason_code,
                    "{query_id} should keep the same precise pre-execution rejection reason"
                );
                assert_eq!(
                    planning_row.unsupported_category, fallback_row.unsupported_category,
                    "{query_id} should keep the same unsupported category"
                );
            },
        }
    }

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
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q02",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q03",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q04",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q05",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q06",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q07",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q08",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q09",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q10",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q11",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q12",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q13",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q14",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q15",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q16",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q17",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q18",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q19",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q20",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q21",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
    TpchFallbackPressureBaseline {
        query_id: "q22",
        outcome: NativeFallbackPressureOutcome::Rewritten,
        reason_code: "none",
        unsupported_category: "none",
    },
];

fn expected_tpch_physical_root(query_id: &str) -> &'static str {
    match query_id {
        "q02" | "q03" | "q10" | "q18" | "q21" => "SortExec(TopK)",
        "q06" | "q14" | "q17" | "q19" => "ProjectionExec",
        "q01" | "q04" | "q05" | "q07" | "q08" | "q09" | "q11" | "q12" | "q13" | "q15" | "q16" | "q20" | "q22" => {
            "SortExec"
        },
        other => panic!("missing TPC-H physical root expectation for {other}"),
    }
}

fn expected_tpch_native_candidate_shape(query_id: &str) -> &'static str {
    match query_id {
        "q01" => "SortExec(ProjectionExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))",
        "q02" => {
            "SortExec(fetch)(ProjectionExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),LocalParquetScan),LocalParquetScan),FilterExec(LocalParquetScan)),ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))))))))"
        },
        "q03" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan)),FilterExec(LocalParquetScan)))))"
        },
        "q04" => {
            "SortExec(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan)))))"
        },
        "q05" => {
            "SortExec(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),LocalParquetScan),LocalParquetScan),LocalParquetScan),FilterExec(LocalParquetScan)))))"
        },
        "q06" => "ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))",
        "q07" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)))))),FilterExec(LocalParquetScan)),FilterExec(LocalParquetScan))))))"
        },
        "q08" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),HashJoinExec(ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),LocalParquetScan))))),LocalParquetScan)),LocalParquetScan)))))"
        },
        "q09" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(LocalParquetScan,HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),LocalParquetScan))))),LocalParquetScan)))))"
        },
        "q10" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan)),LocalParquetScan))))"
        },
        "q12" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)))))"
        },
        "q14" => {
            "ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan))))"
        },
        "q13" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)))))))"
        },
        "q16" => {
            "SortExec(ProjectionExec(AggregateExec(AggregateExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan))))))"
        },
        "q17" => {
            "ProjectionExec(AggregateExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),ProjectionExec(AggregateExec(LocalParquetScan)))))"
        },
        "q18" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(AggregateExec(LocalParquetScan)),HashJoinExec(HashJoinExec(LocalParquetScan,LocalParquetScan),LocalParquetScan)))))"
        },
        "q19" => {
            "ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan))))"
        },
        "q11" => {
            "SortExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))),ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))))))"
        },
        "q15" => {
            "SortExec(HashJoinExec(ProjectionExec(HashJoinExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))),LocalParquetScan)),AggregateExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))))"
        },
        "q20" => {
            "SortExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),HashJoinExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))),HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan))))"
        },
        "q21" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan)),FilterExec(LocalParquetScan)),LocalParquetScan),FilterExec(LocalParquetScan)))))"
        },
        "q22" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),AggregateExec(FilterExec(LocalParquetScan)))))))"
        },
        other => panic!("missing TPC-H native candidate shape expectation for {other}"),
    }
}

fn expected_tpch_native_candidate_shape_family(query_id: &str) -> &'static str {
    match query_id {
        "q01" => "sort_local_parquet_aggregate",
        "q02" => "sort_fetch_hash_join_input",
        "q03" | "q10" | "q18" | "q21" => "sort_fetch_hash_join_aggregate",
        "q04" | "q05" | "q07" | "q08" | "q09" | "q12" | "q13" | "q16" => "sort_hash_join_aggregate",
        "q14" | "q17" | "q19" => "projected_global_hash_join_aggregate",
        "q06" => "projected_global_local_parquet_aggregate",
        "q11" => "sort_mixed_hash_nested_loop_join_input",
        "q15" | "q20" => "sort_hash_join_input",
        "q22" => "sort_mixed_hash_nested_loop_join_aggregate",
        other => panic!("missing TPC-H native candidate shape family expectation for {other}"),
    }
}

fn expected_tpch_datafusion_planning_rejection(query_id: &str) -> (&'static str, &'static str) {
    match query_id {
        "q02" | "q07" | "q13" | "q15" | "q18" | "q20" | "q21" => {
            ("native_bounded_join_unsupported", "operator_hashjoin")
        },
        "q11" => ("native_bounded_join_unsupported", "operator_nestedloopjoin"),
        "q16" => ("native_bounded_join_unsupported", "operator_hashjoin"),
        "q22" => ("native_bounded_join_unsupported", "operator_nestedloopjoin"),
        "q17" => ("native_bounded_join_unsupported", "operator_hashjoin"),
        "q19" => ("native_bounded_join_unsupported", "operator_hashjoin"),
        _ => ("unsupported_operator", expected_tpch_operator_unsupported_category(query_id)),
    }
}

fn expected_tpch_operator_unsupported_category(query_id: &str) -> &'static str {
    match query_id {
        "q02" | "q10" | "q18" | "q21" => "operator_sort_topk",
        "q11" | "q22" => "operator_nestedloopjoin",
        "q04" | "q05" | "q07" | "q08" | "q09" | "q12" | "q13" | "q15" | "q16" | "q17" | "q20" => "operator_hashjoin",
        "q01" => "operator_shape",
        other => panic!("missing TPC-H operator unsupported category expectation for {other}"),
    }
}

fn expected_tpch_physical_shape_marker(query_id: &str) -> &'static str {
    match query_id {
        "q01" => "AggregateExec(ProjectionExec(FilterExec(DataSourceExec)))",
        "q02" | "q03" | "q04" | "q05" | "q07" | "q08" | "q09" | "q10" | "q12" | "q13" | "q14" | "q15" | "q16"
        | "q17" | "q18" | "q19" | "q20" | "q21" => "HashJoinExec",
        "q06" => "ProjectionExec(AggregateExec(FilterExec(DataSourceExec)))",
        "q11" | "q22" => "NestedLoopJoinExec",
        other => panic!("missing TPC-H physical shape marker expectation for {other}"),
    }
}

fn assert_tpch_join_diagnostics(query_id: &str, diagnostics: &[datafusion_cudf::NativeJoinDiagnosticSummary]) {
    match query_id {
        "q02" | "q03" | "q04" | "q05" | "q07" | "q08" | "q09" | "q10" | "q12" | "q13" | "q14" | "q15" | "q16"
        | "q17" | "q18" | "q19" | "q20" | "q21" => {
            assert!(!diagnostics.is_empty(), "{query_id} should expose join diagnostics");
            let first = &diagnostics[0];
            assert_eq!(first.operator, "HashJoinExec");
            assert!(first.key_count > 0, "{query_id} should expose hash join key count");
            assert_ne!(first.key_shape, "", "{query_id} should expose hash join key shape");
            assert_eq!(first.build_side, "left");
            assert_eq!(first.probe_side, "right");
        },
        "q11" | "q22" => {
            assert!(!diagnostics.is_empty(), "{query_id} should expose join diagnostics");
            assert_eq!(diagnostics[0].operator, "NestedLoopJoinExec");
            assert_eq!(diagnostics[0].key_count, 0);
            assert_eq!(diagnostics[0].key_shape, "none");
        },
        "q01" | "q06" => assert!(diagnostics.is_empty(), "{query_id} should not expose join diagnostics"),
        other => panic!("missing TPC-H join diagnostics expectation for {other}"),
    }
}

fn native_exec_leaf_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-execution-report-source-not-read.parquet",
        schema,
        vec!["key".to_string()],
    )));
    Arc::new(NativeCudfExec::try_new(builder.finish_with_host_sink(source), native_execution_config()).unwrap())
}

fn composed_native_exec_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut producer_builder = QueryPlanBuilder::new();
    let producer_source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-composed-report-source-not-read.parquet",
        Arc::clone(&schema),
        vec!["key".to_string()],
    )));
    let producer = Arc::new(
        NativeCudfGpuProducerExec::try_new("fragment_input", producer_builder.finish_gpu_resident(producer_source))
            .unwrap(),
    );

    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_source = consumer_builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("fragment_input", Arc::clone(&schema))));
    Arc::new(
        NativeCudfComposedExec::try_new(
            consumer_builder.finish_with_host_sink(consumer_source),
            vec![producer],
            native_execution_config(),
        )
        .unwrap(),
    )
}

fn native_execution_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
}

fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("SortExec(fetch)(AggregateExec(LocalParquetScan))", 1));
    NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(LocalParquetScan))".to_string(),
        report,
        source_diagnostics: Vec::new(),
        bounded_join_reservation_facts: Default::default(),
    }
}

fn rejected_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_rejected(
        GpuCandidate::new("SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))", 1),
        datafusion_cudf::FallbackReason::NativeBoundedJoinUnsupported,
    );
    NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        report,
        source_diagnostics: Vec::new(),
        bounded_join_reservation_facts: BoundedJoinReservationContract
            .rejection_facts(ExecutionBoundaryPolicy::default())
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    }
}

fn native_execution_metrics_set() -> datafusion::physical_plan::metrics::MetricsSet {
    let metrics = ExecutionPlanMetricsSet::new();
    for (name, value) in [
        ("gpu_resident_input_handoffs", 1),
        ("host_to_device_uploads", 0),
        ("host_to_device_uploaded_rows", 0),
        ("host_to_device_uploaded_bytes", 0),
        ("device_to_host_materializations", 1),
        ("device_to_host_materialized_rows", 4),
        ("device_to_host_materialized_bytes", 256),
        ("device_to_device_copies", 2),
        ("device_to_device_copied_bytes", 512),
        ("parquet_files", 1),
        ("parquet_bytes_known", 1024),
        ("row_groups_selected", 2),
        ("read_columns", 4),
        ("resident_dataframe_peak_bytes", 4096),
        ("allocator_high_water_bytes", 8192),
        ("reservation_requested_bytes", 128),
        ("reservation_granted_bytes", 128),
        ("reservation_denied_count", 0),
        ("source_chunks_planned", 6),
        ("source_chunks_attempted", 7),
        ("source_chunks_committed", 5),
        ("source_chunks_retried", 2),
        ("row_groups_planned", 8),
        ("row_groups_committed", 4),
        ("partial_aggregate_count", 5),
        ("partial_aggregate_rows", 9),
        ("final_merge_attempts", 2),
        ("final_merge_levels", 1),
        ("retry_count", 3),
        ("retry_resource_exhausted_count", 1),
        ("retry_reservation_failure_count", 2),
        ("hash_join_build_count", 1),
        ("filtered_join_build_count", 6),
        ("hash_join_cache_hits", 2),
        ("hash_join_cache_misses", 1),
        ("hash_join_durable_cache_hits", 3),
        ("hash_join_durable_cache_misses", 4),
        ("hash_join_durable_cache_inserts", 5),
        ("hash_join_hash_table_bytes", 16384),
        ("filtered_join_hash_table_bytes", 8192),
        ("filtered_join_probe_state_bytes", 1024),
        ("join_gather_map_bytes", 2048),
    ] {
        MetricBuilder::new(&metrics).with_type(MetricType::SUMMARY).counter(name, 0).add(value);
    }
    metrics.clone_inner()
}

fn report_row<'a>(
    report: &'a NativeFallbackPressureReport,
    query_id: &str,
) -> &'a datafusion_cudf::NativeFallbackPressureRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

fn planning_row<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> &'a datafusion_cudf::NativeDataFusionPlanningRow {
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

fn copy_tiny_tpch_as_directory_layout(source: &std::path::Path, destination: &std::path::Path, file_names: &[&str]) {
    for table in TPCH_TABLES {
        let table_dir = destination.join(table);
        fs::create_dir_all(&table_dir).unwrap();
        let source_file = source.join(format!("{table}.parquet"));
        for file_name in file_names {
            fs::copy(&source_file, table_dir.join(file_name)).unwrap();
        }
    }
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
