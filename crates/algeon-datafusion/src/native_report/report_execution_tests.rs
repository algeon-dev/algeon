use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    physical_expr::expressions::col,
    physical_plan::{
        ExecutionPlan,
        empty::EmptyExec,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet},
        projection::{ProjectionExec, ProjectionExpr},
    },
};
use query_engine::{
    contract_report::CapabilityContractReport,
    dimension_pruning::local_dimension_pruning_resolver,
    expr::{AggregateSpec, ColumnRef, Expr},
    mode::CapabilityEnvelopeStatus,
    optimizer::{AggregateJoinPushdownDiagnostics, NativeOptimizerDiagnostics, optimize_query_plan},
    plan::{
        JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NativePlanCostReason, NodeId, ProjectionOutput,
        QueryPlan, QueryPlanBuilder,
    },
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

use super::{
    CandidateOutcomeKind, FinalPlanDisposition, NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
    NativeAggregateStrategyMetricRow, NativeDataFusionExecutionMetrics, NativeDataFusionExecutionReport,
    NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionRuntime, NativeDataFusionRuntimeOutcome, NativeDynamicFilterRuntimeMetricRow,
};
use crate::{
    capability::CapabilityReason,
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig, SERVER_CAPTURE_LOSS_METRIC_NAME},
    planner::{
        NativeCapabilityAcceptedCandidateDiagnostics, NativeCapabilityCompiledGraphDiagnostics,
        NativeCapabilityContractDiagnostics, NativePlanningDiagnostics,
    },
    report::{GpuCandidate, NativeCandidateCostEvidence, PlanningReport},
};

#[test]
fn native_datafusion_execution_report_records_runtime_boundaries() {
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "q01",
        native_exec_leaf_for_report(),
        accepted_bounded_join_native_diagnostics(),
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
    assert_eq!(row.planning_outcome, Some(CandidateOutcomeKind::Selected));
    assert!(row.capability_decision_rows.is_empty());
    assert_eq!(row.capability_compiled_graphs.len(), 1);
    assert_eq!(row.post_native_plan_root, "AlgeonNativeExec");
    assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Native));
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
    assert_eq!(row.metrics.max_task_observed_peak_bytes, Some(72));
    assert_eq!(row.metrics.max_task_retained_bytes, Some(48));
    assert_eq!(row.metrics.graph_runtime_rows.len(), 1);
    assert_eq!(row.metrics.graph_runtime_rows[0].algorithm, "pagerank");
    assert_eq!(row.metrics.graph_runtime_rows[0].runtime_outcome, "succeeded");
    assert_eq!(row.metrics.graph_runtime_rows[0].construction_policy, "python_cugraph");
    assert_eq!(row.metrics.graph_runtime_rows[0].source_column, "src");
    assert_eq!(row.metrics.graph_runtime_rows[0].destination_column, "dst");
    assert_eq!(row.metrics.graph_runtime_rows[0].weight_column.as_deref(), Some("weight"));
    assert!(row.metrics.graph_runtime_rows[0].directed);
    assert!(row.metrics.graph_runtime_rows[0].renumber);
    assert_eq!(row.metrics.graph_runtime_rows[0].algorithm_metadata, "converged=true,iterations=12");
    assert_eq!(row.metrics.graph_runtime_rows[0].output_device_bytes, Some(320));
    assert_eq!(row.metrics.graph_runtime_rows[0].graph_peak_bytes, Some(2048));
    assert_eq!(row.metrics.task_operator_rows.len(), 1);
    assert_eq!(row.metrics.task_operator_rows[0].operator_kind, "source_read");
    assert_eq!(row.metrics.task_operator_rows[0].task_count, 3);
    assert_eq!(row.metrics.task_operator_rows[0].max_observed_peak_bytes, Some(72));
    assert_eq!(row.metrics.task_operator_rows[0].max_retained_bytes, Some(48));
    assert_eq!(row.metrics.task_runtime_rows.len(), 1);
    assert_eq!(row.metrics.task_runtime_rows[0].compiled_task_id, "t7");
    assert_eq!(row.metrics.task_runtime_rows[0].operator_kind, "source_read");
    assert_eq!(row.metrics.task_runtime_rows[0].max_observed_peak_bytes, Some(72));
    assert_eq!(row.metrics.task_runtime_rows[0].max_retained_bytes, Some(48));
    assert_eq!(row.metrics.source_chunks_planned, 6);
    assert_eq!(row.metrics.source_chunks_attempted, 7);
    assert_eq!(row.metrics.source_chunks_committed, 5);
    assert_eq!(row.metrics.source_chunks_retried, 2);
    assert_eq!(row.metrics.source_read_prefetches_started, 3);
    assert_eq!(row.metrics.source_read_prefetches_completed, 2);
    assert_eq!(row.metrics.source_read_prefetches_skipped, 1);
    assert_eq!(row.metrics.source_read_prefetches_abandoned, 1);
    assert_eq!(row.metrics.row_groups_planned, 8);
    assert_eq!(row.metrics.row_groups_committed, 4);
    assert_eq!(row.metrics.partial_aggregate_count, 5);
    assert_eq!(row.metrics.partial_aggregate_input_rows, 11);
    assert_eq!(row.metrics.partial_aggregate_rows, 9);
    assert_eq!(row.metrics.partial_aggregate_retained_bytes, 4096);
    assert_eq!(row.metrics.final_merge_attempts, 2);
    assert_eq!(row.metrics.final_merge_levels, 1);
    assert_eq!(row.metrics.final_merge_max_fan_in, 3);
    assert_eq!(row.metrics.retry_count, 3);
    assert_eq!(row.metrics.retry_resource_exhausted_count, 1);
    assert_eq!(row.metrics.dynamic_filter_produced_count, 1);
    assert_eq!(row.metrics.dynamic_filter_skipped_count, 1);
    assert_eq!(row.metrics.dynamic_filter_applied_count, 1);
    assert_eq!(row.metrics.dynamic_filter_row_groups_pruned, 3);
    assert_eq!(row.metrics.dynamic_filter_source_bytes_pruned, 4096);
    assert_eq!(row.metrics.dynamic_filter_reader_predicate_rows_before, 1000);
    assert_eq!(row.metrics.dynamic_filter_reader_predicate_rows_after, 250);
    assert_eq!(row.metrics.dynamic_filter_reader_predicate_rows_filtered, 750);
    assert_eq!(row.metrics.dynamic_filter_post_decode_rows_before, 80);
    assert_eq!(row.metrics.dynamic_filter_post_decode_rows_after, 25);
    assert_eq!(row.metrics.dynamic_filter_post_decode_rows_filtered, 55);
    assert_eq!(row.metrics.dynamic_filter_source_post_decode_rows_filtered, 55);
    assert_eq!(row.metrics.dynamic_filter_filter_post_decode_rows_filtered, 0);
    assert_eq!(row.metrics.dynamic_filter_join_probe_post_decode_rows_filtered, 0);
    assert_eq!(row.metrics.dynamic_filter_measured_reduction_status().as_str(), "measured_source_pruned");
    assert_eq!(row.metrics.dynamic_filter_post_decode_reduction_stage().as_str(), "source_post_decode");
    assert_eq!(row.metrics.dynamic_filter_work_avoidance_status().as_str(), "source_read_pruned");
    assert_eq!(row.metrics.dynamic_filter_runtime_rows.len(), 2);
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[0].dynamic_filter_id, "df0");
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[0].event, "applied");
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[0].filter_kind, "small_in_list");
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[0].row_groups_pruned, 3);
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[1].event, "applied_post_decode");
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[1].consumer_reduction_stage, "source_post_decode");
    assert_eq!(row.metrics.dynamic_filter_runtime_rows[1].post_decode_rows_filtered, 55);
    assert_eq!(row.metrics.hash_join_build_count, 1);
    assert_eq!(row.metrics.filtered_join_build_count, 6);
    assert_eq!(row.metrics.one_shot_join_count, 7);
    assert_eq!(row.metrics.hash_join_hash_table_bytes, 16384);
    assert_eq!(row.metrics.filtered_join_hash_table_bytes, 8192);
    assert_eq!(row.metrics.filtered_join_probe_state_bytes, 1024);
    assert_eq!(row.metrics.mixed_join_size_data_bytes, 512);
    assert_eq!(row.metrics.one_shot_join_index_bytes, 4096);
    assert_eq!(row.metrics.one_shot_join_materialization_bytes, 8192);
    assert_eq!(row.metrics.one_shot_join_output_bytes, 16384);
    assert_eq!(row.metrics.join_gather_map_bytes, 2048);
    assert_eq!(row.metrics.aggregate_strategy_rows.len(), 1);
    assert_eq!(row.metrics.aggregate_strategy_rows[0].node_id, "n4");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].selected_strategy, "libcudf_groupby");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].execution_stage, "partial");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].group_key_dtypes, "Utf8");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].aggregate_functions, "sum,count");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].input_rows, 6001215);
    assert_eq!(row.metrics.aggregate_strategy_rows[0].output_group_count, 4);
    assert_eq!(row.metrics.aggregate_strategy_rows[0].source_adjacent_partial, "true");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].source_chunk_row_group_count, "1");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].source_chunk_known_row_count, "6001215");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].groupby_elapsed_ns, 123456789);
    assert_eq!(row.metrics.aggregate_strategy_rows[0].aggregate_structural_key, "agg-v1-fnv64-0123456789abcdef");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].prior_cardinality_evidence_scope, "absent");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].encoded_key_partial_cost_model, "prior_partial_reduction_ratio");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].encoded_key_partial_cost_model_status, "rejected_by_cost_model");
    assert_eq!(
        row.metrics.aggregate_strategy_rows[0].encoded_key_partial_selection_reason,
        "no_prior_cardinality_evidence"
    );
    assert_eq!(row.metrics.aggregate_strategy_rows[0].prepared_sort_groupby_rejected_reason, "none");
    assert_eq!(row.metrics.aggregate_strategy_rows[0].prepared_sort_groupby_contract_status, "available");
    assert_eq!(
        row.metrics.aggregate_strategy_rows[0].prepared_sort_groupby_cost_model_status,
        "selected_libcudf_runtime_probe"
    );
    assert_eq!(
        row.metrics.aggregate_strategy_rows[0].prepared_sort_groupby_selection_reason,
        "runtime_cardinality_probe_required"
    );
    assert_eq!(row.metrics.aggregate_strategy_rows[0].prepared_sort_groupby_num_groups, "unknown");
    assert_eq!(row.metrics.aggregate_libcudf_runtime_probe_count(), 1);
    assert_eq!(row.metrics.aggregate_libcudf_runtime_probe_elapsed_ns(), 123456789);
    assert_eq!(row.metrics.aggregate_libcudf_runtime_probe_input_rows(), 6001215);
    assert_eq!(row.metrics.aggregate_exact_cardinality_probe_count(), 0);
    assert_eq!(row.metrics.aggregate_exact_cardinality_probe_elapsed_ns(), 0);
    assert_eq!(row.metrics.aggregate_exact_cardinality_probe_input_rows(), 0);
    assert_eq!(row.metrics.aggregate_prepared_sort_groupby_elapsed_ns(), 0);
    assert_eq!(row.metrics.aggregate_encoded_key_partial_elapsed_ns(), 0);
    assert_eq!(row.metrics.lakehouse_cache_fill_event_snapshot_rows.len(), 1);
    assert_eq!(row.metrics.lakehouse_cache_fill_event_snapshot_rows[0].outcome, "completed");
    assert_eq!(row.metrics.lakehouse_cache_fill_event_snapshot_rows[0].bytes, Some(4096));
    assert_eq!(row.metrics.join_strategy_rows.len(), 1);
    assert_eq!(row.metrics.join_strategy_rows[0].node_id, "n2");
    assert_eq!(row.metrics.join_strategy_rows[0].join_strategy, "one_shot_equi");
    assert_eq!(
        row.metrics.join_strategy_rows[0].reusable_hash_join_rejected_reason.as_deref(),
        Some("build_side_too_large")
    );
    assert_eq!(row.metrics.join_strategy_rows[0].estimated_output_rows, Some(102));
    assert_eq!(row.metrics.join_strategy_rows[0].estimated_output_bytes, Some(4096));
    assert_eq!(row.metrics.join_strategy_rows[0].build_key_unique.as_deref(), Some("true"));
    assert_eq!(row.metrics.join_strategy_rows[0].build_key_uniqueness_probe_status.as_deref(), Some("measured"));
    assert_eq!(row.metrics.join_strategy_rows[0].build_key_uniqueness_probe_error_code, None);
    let text = report.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.split('\t').count(), 106);
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.starts_with("query_id\t"));
    assert!(text.contains("\nq01\t"));
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains(
        "\trejected_before_execution\tadmission_sequence\tadmission_wait_ms\tdevice_ordinal\tquery_grant_bytes\tnative_runtime_started\t"
    ));
    assert!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER
            .contains("\tcapability_decision_rows\tcapability_compiled_graphs\tpost_native_plan_root\t")
    );
    assert!(text.starts_with(&format!("{NATIVE_DATAFUSION_EXECUTION_TSV_HEADER}\n")));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_EXECUTION_TSV_HEADER}\n")));
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tgraph_runtime_rows\ttask_operator_rows\t"));
    assert!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER
            .contains("\tdynamic_filter_produced_count\tdynamic_filter_skipped_count\tdynamic_filter_applied_count\t")
    );
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tdynamic_filter_measured_reduction_status\t"));
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tdynamic_filter_work_avoidance_status\t"));
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tdynamic_filter_post_decode_reduction_stage\t"));
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains(
        "\tdynamic_filter_post_decode_rows_before\tdynamic_filter_post_decode_rows_after\tdynamic_filter_post_decode_rows_filtered\tdynamic_filter_source_post_decode_rows_filtered\tdynamic_filter_filter_post_decode_rows_filtered\tdynamic_filter_join_probe_post_decode_rows_filtered\tdynamic_filter_post_decode_reduction_stage\t"
    ));
    assert!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER
            .contains("\taggregate_libcudf_runtime_probe_count\taggregate_libcudf_runtime_probe_elapsed_ns\t")
    );
    assert!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER
            .contains("\taggregate_exact_cardinality_probe_count\taggregate_exact_cardinality_probe_elapsed_ns\t")
    );
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\taggregate_strategy_rows\tjoin_strategy_rows\t"));
    for removed in [
        "source_cache_hits",
        "source_cache_misses",
        "source_cache_inserts",
        "source_cache_insert_skipped",
        "source_cache_retained_bytes",
        "source_cache_lookup_time_ms",
        "hash_join_durable_cache_hits",
        "hash_join_durable_cache_misses",
        "hash_join_durable_cache_inserts",
        "hash_join_cache_hits",
        "hash_join_cache_misses",
        concat!("source_", "fetch_bytes"),
        concat!("source_", "fetch_count"),
        concat!("host_", "fetch_budget_wait_time_ms"),
        concat!("source_", "fetch_concurrency_wait_time_ms"),
        concat!("source_", "fetch_time_ms"),
        concat!("source_", "decode_time_ms"),
    ] {
        assert!(!NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.split('\t').any(|column| column == removed));
    }
    assert!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER
            .ends_with("\tsource_diagnostics\tlakehouse_cache_fill_event_snapshot_rows\tplan_task_mapping_rows")
    );
    assert!(text.contains("selected_strategy=libcudf_groupby,execution_stage=partial"));
    assert!(text.contains("algorithm=pagerank,runtime_outcome=succeeded"));
    assert!(text.contains("graph_peak_bytes=2048"));
    assert!(
        text.contains(
            "construction_policy=python_cugraph,source_column=src,destination_column=dst,weight_column=weight"
        )
    );
    assert!(text.contains("algorithm_metadata=converged=true,iterations=12"));
    assert!(text.contains("dynamic_filter_id=df0,event=applied,producer_kind=hash_join_build"));
    assert!(
        text.contains(
            "\tsource_post_decode\tmeasured_source_pruned\tsource_read_pruned\tdynamic_filter_id=df0,event=applied,producer_kind=hash_join_build"
        )
    );
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("execution report should contain a TSV header");
    let data = lines.next().expect("execution report should contain a data row");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let data_fields = data.split('\t').collect::<Vec<_>>();
    assert_eq!(header_fields.len(), data_fields.len());
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_libcudf_runtime_probe_count")], "1");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_libcudf_runtime_probe_elapsed_ns")], "123456789");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_libcudf_runtime_probe_input_rows")], "6001215");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_exact_cardinality_probe_count")], "0");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_exact_cardinality_probe_elapsed_ns")], "0");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_exact_cardinality_probe_input_rows")], "0");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_prepared_sort_groupby_elapsed_ns")], "0");
    assert_eq!(data_fields[header_index(&header_fields, "aggregate_encoded_key_partial_elapsed_ns")], "0");
    let fill_rows = data_fields[header_index(&header_fields, "lakehouse_cache_fill_event_snapshot_rows")];
    assert!(fill_rows.contains("outcome=completed,bytes=4096,elapsed_ns=9000000"));
    assert!(fill_rows.contains("cache_path_hash=cache123,stable_id_hash=stable456"));
    assert!(text.contains(
        "operator_kind=source_read,task_count=3,elapsed_ns=0,max_observed_peak_bytes=72,max_retained_bytes=48\t"
    ));
    assert!(text.contains(
        "\tcompiled_task_id=t7,operator_kind=source_read,elapsed_ns=0,max_observed_peak_bytes=72,max_retained_bytes=48\t"
    ));
    assert!(!text.contains("bounded_join_execution_status"));
    assert!(!text.contains("bounded_join_reservation_facts"));
    assert!(text.contains("node_id=n2,join_strategy=one_shot_equi,execution_strategy=bounded_one_shot_equi_join"));
    assert!(text.contains(
        "\tAlgeonNativeExec\tnative\t1\t0\tfalse\tnone\tnone\tnone\tnone\ttrue\tsucceeded\tnone\t1\t4\t256\t"
    ));
}

#[test]
fn aggregate_runtime_probe_summary_uses_selection_evidence_not_strategy_name_alone() {
    let metrics = NativeDataFusionExecutionMetrics {
        aggregate_strategy_rows: vec![
            aggregate_strategy_row(
                "libcudf_groupby",
                "partial",
                "selected_libcudf_runtime_probe",
                "runtime_cardinality_probe_required",
                100,
                11,
            ),
            aggregate_strategy_row(
                "libcudf_groupby",
                "partial",
                "selected_libcudf_runtime_probe",
                "runtime_cardinality_probe_required",
                200,
                22,
            ),
            aggregate_strategy_row(
                "libcudf_groupby",
                "single",
                "selected_libcudf_runtime_probe",
                "runtime_cardinality_probe_required",
                300,
                33,
            ),
            aggregate_strategy_row(
                "libcudf_groupby",
                "partial",
                "rejected_by_cost_model",
                "no_prior_cardinality_evidence",
                400,
                44,
            ),
            aggregate_strategy_row("prepared_sort_groupby", "partial", "selected", "selected", 500, 55),
            aggregate_strategy_row("encoded_key_partial", "partial", "selected", "selected", 600, 66),
            NativeAggregateStrategyMetricRow {
                selected_strategy: "prepared_sort_groupby".to_string(),
                execution_stage: "partial".to_string(),
                exact_cardinality_probe_status: "measured".to_string(),
                exact_cardinality_probe_input_rows: 700,
                exact_cardinality_probe_elapsed_ns: 77,
                strategy_operator_elapsed_ns: 88,
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    assert_eq!(metrics.aggregate_libcudf_runtime_probe_count(), 2);
    assert_eq!(metrics.aggregate_libcudf_runtime_probe_elapsed_ns(), 33);
    assert_eq!(metrics.aggregate_libcudf_runtime_probe_input_rows(), 300);
    assert_eq!(metrics.aggregate_exact_cardinality_probe_count(), 1);
    assert_eq!(metrics.aggregate_exact_cardinality_probe_elapsed_ns(), 77);
    assert_eq!(metrics.aggregate_exact_cardinality_probe_input_rows(), 700);
    assert_eq!(metrics.aggregate_prepared_sort_groupby_elapsed_ns(), 143);
    assert_eq!(metrics.aggregate_encoded_key_partial_elapsed_ns(), 66);
}

#[test]
fn dynamic_filter_measured_reduction_status_uses_measured_counters_not_applied_count() {
    assert_eq!(
        NativeDataFusionExecutionMetrics::default().dynamic_filter_measured_reduction_status().as_str(),
        "unknown"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics { dynamic_filter_applied_count: 1, ..Default::default() }
            .dynamic_filter_measured_reduction_status()
            .as_str(),
        "measured_no_reduction"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_reader_predicate_rows_filtered: 7,
            dynamic_filter_applied_count: 1,
            ..Default::default()
        }
        .dynamic_filter_measured_reduction_status()
        .as_str(),
        "measured_reader_rows_reduced"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics { dynamic_filter_post_decode_rows_filtered: 3, ..Default::default() }
            .dynamic_filter_measured_reduction_status()
            .as_str(),
        "measured_post_decode_reduced"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics::default().dynamic_filter_post_decode_reduction_stage().as_str(),
        "none"
    );
    assert_eq!(NativeDataFusionExecutionMetrics::default().dynamic_filter_work_avoidance_status().as_str(), "unknown");
    assert_eq!(
        NativeDataFusionExecutionMetrics { dynamic_filter_applied_count: 1, ..Default::default() }
            .dynamic_filter_work_avoidance_status()
            .as_str(),
        "measured_no_reduction"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_reader_predicate_rows_filtered: 7,
            dynamic_filter_applied_count: 1,
            ..Default::default()
        }
        .dynamic_filter_work_avoidance_status()
        .as_str(),
        "source_reader_rows_reduced"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics { dynamic_filter_post_decode_rows_filtered: 3, ..Default::default() }
            .dynamic_filter_post_decode_reduction_stage()
            .as_str(),
        "unknown"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_post_decode_rows_filtered: 3,
            dynamic_filter_filter_post_decode_rows_filtered: 3,
            dynamic_filter_runtime_rows: vec![NativeDynamicFilterRuntimeMetricRow {
                consumer_reduction_stage: "filter_post_decode".to_string(),
                post_decode_rows_filtered: 3,
                ..Default::default()
            }],
            ..Default::default()
        }
        .dynamic_filter_post_decode_reduction_stage()
        .as_str(),
        "filter_post_decode"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_post_decode_rows_filtered: 3,
            dynamic_filter_filter_post_decode_rows_filtered: 3,
            ..Default::default()
        }
        .dynamic_filter_work_avoidance_status()
        .as_str(),
        "filter_post_decode_rows_reduced"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_post_decode_rows_filtered: 5,
            dynamic_filter_filter_post_decode_rows_filtered: 3,
            dynamic_filter_join_probe_post_decode_rows_filtered: 2,
            dynamic_filter_runtime_rows: vec![
                NativeDynamicFilterRuntimeMetricRow {
                    consumer_reduction_stage: "filter_post_decode".to_string(),
                    post_decode_rows_filtered: 3,
                    ..Default::default()
                },
                NativeDynamicFilterRuntimeMetricRow {
                    consumer_reduction_stage: "join_probe_post_decode".to_string(),
                    post_decode_rows_filtered: 2,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
        .dynamic_filter_post_decode_reduction_stage()
        .as_str(),
        "mixed_post_decode"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_post_decode_rows_filtered: 5,
            dynamic_filter_filter_post_decode_rows_filtered: 3,
            dynamic_filter_join_probe_post_decode_rows_filtered: 2,
            ..Default::default()
        }
        .dynamic_filter_work_avoidance_status()
        .as_str(),
        "mixed_post_decode_rows_reduced"
    );
    assert_eq!(
        NativeDataFusionExecutionMetrics {
            dynamic_filter_source_bytes_pruned: 1,
            dynamic_filter_reader_predicate_rows_filtered: 7,
            dynamic_filter_post_decode_rows_filtered: 3,
            ..Default::default()
        }
        .dynamic_filter_measured_reduction_status()
        .as_str(),
        "measured_source_pruned"
    );
}

fn header_index(header_fields: &[&str], name: &str) -> usize {
    header_fields.iter().position(|field| *field == name).unwrap_or_else(|| panic!("missing TSV column {name}"))
}

fn aggregate_strategy_row(
    selected_strategy: &str,
    execution_stage: &str,
    prepared_sort_groupby_cost_model_status: &str,
    prepared_sort_groupby_selection_reason: &str,
    input_rows: usize,
    strategy_operator_elapsed_ns: usize,
) -> NativeAggregateStrategyMetricRow {
    NativeAggregateStrategyMetricRow {
        selected_strategy: selected_strategy.to_string(),
        execution_stage: execution_stage.to_string(),
        prepared_sort_groupby_cost_model_status: prepared_sort_groupby_cost_model_status.to_string(),
        prepared_sort_groupby_selection_reason: prepared_sort_groupby_selection_reason.to_string(),
        input_rows,
        strategy_operator_elapsed_ns,
        ..Default::default()
    }
}

#[test]
fn native_datafusion_execution_report_keeps_unobserved_optional_memory_metrics_empty() {
    let metrics = ExecutionPlanMetricsSet::new();
    MetricBuilder::new(&metrics).with_type(MetricType::Summary).counter("resident_dataframe_peak_bytes", 0);
    MetricBuilder::new(&metrics).with_type(MetricType::Summary).counter("allocator_high_water_bytes", 0);
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "q01",
        native_exec_leaf_for_report(),
        accepted_native_diagnostics(),
        NativeDataFusionRuntime::Succeeded {
            output_batches: 1,
            output_rows: 4,
            output_bytes: 256,
            metrics: metrics.clone_inner(),
        },
    )]);

    let row = &report.rows[0];
    assert_eq!(row.metrics.resident_dataframe_peak_bytes, None);
    assert_eq!(row.metrics.allocator_high_water_bytes, None);
}

#[test]
fn native_datafusion_execution_report_marks_lost_metric_detail_without_changing_the_header() {
    let report_for = |capture_loss: usize| {
        let metrics = ExecutionPlanMetricsSet::new();
        MetricBuilder::new(&metrics)
            .with_type(MetricType::Summary)
            .counter(SERVER_CAPTURE_LOSS_METRIC_NAME, 0)
            .add(capture_loss);
        NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
            "q01",
            native_exec_leaf_for_report(),
            accepted_native_diagnostics(),
            NativeDataFusionRuntime::Succeeded {
                output_batches: 1,
                output_rows: 4,
                output_bytes: 256,
                metrics: metrics.clone_inner(),
            },
        )])
        .to_tsv()
    };

    let complete = report_for(0);
    let incomplete = report_for(1);

    assert!(!complete.contains("# metric_detail_capture"));
    assert!(
        incomplete
            .lines()
            .any(|line| line == "# metric_detail_capture query_id=q01 status=incomplete lost_records=unknown"),
        "{incomplete}"
    );
    let header = |text: &str| text.lines().find(|line| !line.starts_with('#')).map(str::to_string);
    assert_eq!(header(&incomplete).as_deref(), Some(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER));
    assert_eq!(header(&complete), header(&incomplete));
}

#[test]
fn native_datafusion_execution_report_projects_runtime_for_mixed_plan_despite_candidate_declines() {
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "mixed_gpu_island",
        mixed_native_exec_for_runtime_projection(),
        capability_and_cost_declined_diagnostics(),
        NativeDataFusionRuntime::Succeeded {
            output_batches: 1,
            output_rows: 4,
            output_bytes: 256,
            metrics: native_execution_metrics_set(),
        },
    )]);

    assert_eq!(report.rows.len(), 2);
    assert!(report.rows.iter().any(|row| {
        row.planning_outcome == Some(CandidateOutcomeKind::NotSupported)
            && row.reason_code.as_deref() == Some("unsupported_expression")
    }));
    assert!(report.rows.iter().any(|row| {
        row.planning_outcome == Some(CandidateOutcomeKind::NotSelectedByCost)
            && row.reason_code.as_deref() == Some("native_small_selective_topk_not_selected_by_cost")
    }));

    for row in &report.rows {
        assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Mixed));
        assert!(row.native_runtime_started);
        assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
        assert_eq!(row.output_batches, Some(1));
        assert_eq!(row.output_rows, Some(4));
        assert_eq!(row.output_bytes, Some(256));
        assert_eq!(row.metrics.graph_runtime_rows.len(), 1);
        assert_eq!(row.metrics.graph_runtime_rows[0].algorithm, "pagerank");
    }
}

#[test]
fn native_datafusion_execution_report_keeps_runtime_unstarted_for_pure_datafusion_candidate_baseline() {
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "cpu_baseline",
        Arc::new(EmptyExec::new(Arc::new(Schema::empty()))),
        capability_and_cost_declined_diagnostics(),
        NativeDataFusionRuntime::Succeeded {
            output_batches: 1,
            output_rows: 4,
            output_bytes: 256,
            metrics: native_execution_metrics_set(),
        },
    )]);

    assert_eq!(report.rows.len(), 2);
    for row in &report.rows {
        assert_eq!(row.final_disposition, Some(FinalPlanDisposition::DataFusion));
        assert!(!row.native_runtime_started);
        assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::NotStarted);
        assert_eq!(row.runtime_failure, None);
        assert_eq!(row.output_batches, None);
        assert_eq!(row.output_rows, None);
        assert_eq!(row.output_bytes, None);
        assert!(row.metrics.graph_runtime_rows.is_empty());
    }
}

fn mixed_native_exec_for_runtime_projection() -> Arc<dyn ExecutionPlan> {
    let native = native_exec_leaf_for_report();
    let schema = native.schema();
    Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &schema).unwrap(), alias: "key".to_string() }],
            native,
        )
        .unwrap(),
    )
}

fn capability_and_cost_declined_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_not_supported(GpuCandidate::new("UnsupportedCandidate", 1), CapabilityReason::UnsupportedExpression);
    report.push_not_selected_by_cost(
        GpuCandidate::new("CostDeclinedCandidate", 1),
        NativePlanCostReason::SmallSelectiveTopK,
        NativeCandidateCostEvidence {
            node_count: 1,
            source_count: 1,
            local_parquet_source_count: 1,
            selected_row_groups_known: true,
            selected_row_groups: Some(1),
            estimated_source_rows: Some(4),
            estimated_source_bytes: Some(256),
            selected_source_bytes_known: true,
            join_count: 0,
            residual_join_count: 0,
            aggregate_count: 0,
            grouped_aggregate_count: 0,
            sort_count: 0,
            limit_count: 1,
            min_topk_fetch: Some(1),
        },
    );
    NativePlanningDiagnostics {
        input_plan_root: "ProjectionExec".to_string(),
        input_plan_shape: "ProjectionExec(GpuIsland)".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: Default::default(),
    }
}

#[test]
fn native_datafusion_reports_scope_admission_contract_rows_to_join_candidates() {
    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new(
        "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))",
        1,
    ));
    report.push_not_supported(
        GpuCandidate::new("SortExec(fetch)(AggregateExec(LocalParquetScan))", 1),
        CapabilityReason::UnsupportedExpression,
    );
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: accepted_capability_contract_diagnostics(),
    };
    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "mixed",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);

    let join_row = planning
        .rows
        .iter()
        .find(|row| row.native_candidate_shape.as_deref().is_some_and(|shape| shape.contains("HashJoinExec")))
        .expect("join candidate row should exist");
    let non_join_row = planning
        .rows
        .iter()
        .find(|row| {
            row.native_candidate_shape
                .as_deref()
                .is_some_and(|shape| shape == "SortExec(fetch)(AggregateExec(LocalParquetScan))")
        })
        .expect("non-join candidate row should exist");

    assert!(join_row.capability_decision_rows.is_empty());
    assert_eq!(join_row.capability_compiled_graphs.len(), 1);
    assert!(
        non_join_row.capability_decision_rows.is_empty(),
        "bounded contract rows must not bleed into unrelated candidate rows"
    );
    assert!(
        non_join_row.capability_compiled_graphs.is_empty(),
        "bounded graph diagnostics must not bleed into unrelated candidate rows"
    );
}

#[test]
fn native_datafusion_reports_keep_multiple_accepted_join_candidates_separated() {
    let first_candidate = GpuCandidate::new("HashJoinExec(LocalParquetScan,LocalParquetScan)", 1);
    let second_candidate = GpuCandidate::new("NestedLoopJoinExec(LocalParquetScan,LocalParquetScan)", 1);
    let mut report = PlanningReport::new();
    report.push_selected(first_candidate.clone());
    report.push_selected(second_candidate.clone());

    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "UnionExec".to_string(),
        input_plan_shape: "UnionExec(HashJoinExec(...),NestedLoopJoinExec(...))".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: NativeCapabilityContractDiagnostics {
            accepted_candidates: vec![
                accepted_candidate_contract_entry(first_candidate.clone(), 11),
                accepted_candidate_contract_entry(second_candidate.clone(), 22),
            ],
            rejected_candidates: Vec::new(),
        },
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "two-joins",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);

    let first_row = planning
        .rows
        .iter()
        .find(|row| row.native_candidate_shape.as_deref() == Some(first_candidate.shape.as_str()))
        .expect("first join row should exist");
    let second_row = planning
        .rows
        .iter()
        .find(|row| row.native_candidate_shape.as_deref() == Some(second_candidate.shape.as_str()))
        .expect("second join row should exist");

    assert!(first_row.capability_decision_rows.is_empty());
    assert_eq!(first_row.capability_compiled_graphs[0].task_count, 11);
    assert!(second_row.capability_decision_rows.is_empty());
    assert_eq!(second_row.capability_compiled_graphs[0].task_count, 22);
}

#[test]
fn native_datafusion_planning_report_projects_native_optimizer_diagnostics() {
    let optimizer_result = optimize_query_plan(
        semijoin_reduction_projection_plan(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_eq!(optimizer_result.diagnostics.semijoin_reduction.applied_count, 1);

    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new("HashJoinExec(HashJoinExec,GpuKeyset)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "HashJoinExec".to_string(),
        input_plan_shape: "HashJoinExec(HashJoinExec,GpuKeyset)".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: optimizer_result.diagnostics,
        source_diagnostics: Vec::new(),
        capability_contract: NativeCapabilityContractDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "generic_semijoin",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);
    let row = &planning.rows[0];
    assert_eq!(row.native_optimizer.semijoin_reduction.considered_count, 1);
    assert_eq!(row.native_optimizer.semijoin_reduction.applied_count, 1);
    assert_eq!(row.native_optimizer.shared_resident_extraction.considered_count, 6);
    assert_eq!(row.native_optimizer.shared_resident_extraction.extracted_count, 0);

    let text = planning.to_tsv();
    let optimizer_cell = planning_native_optimizer_cell(&planning);
    assert!(optimizer_cell.starts_with("rule=semijoin_reduction,considered_count=1,applied_count=1"));
    assert!(optimizer_cell.contains(";rule=join_build_side_selection,considered_count=1,applied_count=0"));
    assert!(optimizer_cell.contains(";rule=dimension_derived_row_group_pruning,considered_count=2,applied_count=0"));
    assert!(optimizer_cell.contains(
        ";rule=shared_resident_extraction,considered_count=6,extracted_count=0,rejected_count=0,candidates="
    ));
    assert!(optimizer_cell.contains("decision=Skipped,reason=NotRepeated,resident_input_name=none"));
    assert!(optimizer_cell.contains(";rule=dynamic_filter_route_discovery,considered_count=2,route_count=1"));
    assert!(optimizer_cell.contains(";rule=aggregate_join_pushdown,considered_count=1,applied_count=0"));
    assert!(text.contains("\tnative_optimizer\t"));
    assert!(text.contains("rule=semijoin_reduction,considered_count=1,applied_count=1"));
    assert!(text.contains("crossed_join_node_ids="));
    assert!(text.contains("cost_gate="));
    assert!(!text.contains("q18"));
}

#[test]
fn native_datafusion_planning_report_projects_projection_simplification_diagnostics() {
    let optimizer_result = optimize_query_plan(
        projection_simplification_identity_plan(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_eq!(optimizer_result.diagnostics.projection_simplification.considered_count, 1);
    assert_eq!(optimizer_result.diagnostics.projection_simplification.applied_count, 1);

    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new("ProjectionExec(GpuResidentSource)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "ProjectionExec".to_string(),
        input_plan_shape: "ProjectionExec(GpuResidentSource)".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: optimizer_result.diagnostics,
        source_diagnostics: Vec::new(),
        capability_contract: NativeCapabilityContractDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "projection_simplification",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);
    let row = &planning.rows[0];
    assert_eq!(row.native_optimizer.projection_simplification.considered_count, 1);
    assert_eq!(row.native_optimizer.projection_simplification.applied_count, 1);
    assert_eq!(row.native_optimizer.shared_resident_extraction.considered_count, 1);

    let text = planning.to_tsv();
    let optimizer_cell = planning_native_optimizer_cell(&planning);
    assert!(optimizer_cell.starts_with(
        "rule=projection_simplification,considered_count=1,applied_count=1,candidates=rule=projection_simplification,candidate_node_id=n1,replacement_node_id=n0,applied=true,rejected_reason=none"
    ));
    assert!(optimizer_cell.contains(
        ";rule=shared_resident_extraction,considered_count=1,extracted_count=0,rejected_count=0,candidates="
    ));
    assert!(optimizer_cell.contains("decision=Skipped,reason=NotRepeated,resident_input_name=none"));
    assert!(text.contains("rule=projection_simplification,considered_count=1,applied_count=1"));
    assert!(text.contains("replacement_node_id="));
    assert!(text.contains("applied=true"));
    assert!(!text.contains("q13"));
}

#[test]
fn native_datafusion_planning_report_projects_dynamic_filter_route_diagnostics() {
    let optimizer_result = optimize_query_plan(
        dynamic_filter_route_plan(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_eq!(optimizer_result.diagnostics.dynamic_filter_route_discovery.route_count, 1);

    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new("HashJoinExec(LocalParquetScan,LocalParquetScan)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "HashJoinExec".to_string(),
        input_plan_shape: "HashJoinExec(LocalParquetScan,LocalParquetScan)".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: optimizer_result.diagnostics,
        source_diagnostics: Vec::new(),
        capability_contract: NativeCapabilityContractDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "dynamic_filter_route",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);

    let text = planning.to_tsv();
    assert_eq!(
        planning_native_optimizer_cell(&planning),
        "rule=join_build_side_selection,considered_count=1,applied_count=0,candidates=rule=join_build_side_selection,candidate_node_id=n2,before_build_side=right,after_build_side=none,left_estimated_rows=10,right_estimated_rows=2,left_estimated_bytes=1024,right_estimated_bytes=64,left_advisory_rows=none,right_advisory_rows=none,left_advisory_bytes=none,right_advisory_bytes=none,small_build_side_bytes=67108864,small_build_side_rows=1000000,selection_basis=conservative_bytes,applied=false,rejected_reason=NoBetterBuildSide;rule=dimension_derived_row_group_pruning,considered_count=2,applied_count=0,candidates=rule=dimension_derived_row_group_pruning,candidate_node_id=n2,dimension_source_node_id=n0,target_source_node_id=n1,dimension_key_column=left_key,target_key_column=right_key,derived_min_key=none,derived_max_key=none,selected_row_groups_before=none,selected_row_groups_after=none,max_dimension_rows=1000000,applied=false,rejected_reason=DimensionPredicateAbsent|rule=dimension_derived_row_group_pruning,candidate_node_id=n2,dimension_source_node_id=n1,target_source_node_id=n0,dimension_key_column=right_key,target_key_column=left_key,derived_min_key=none,derived_max_key=none,selected_row_groups_before=none,selected_row_groups_after=none,max_dimension_rows=1000000,applied=false,rejected_reason=DimensionPredicateAbsent;rule=dynamic_filter_route_discovery,considered_count=1,route_count=1,candidates=rule=dynamic_filter_route_discovery,candidate_node_id=n2,dynamic_filter_id=df0,producer_build_side=right,producer_input_node_id=n1,consumer_input_node_id=n0,consumer_kind=local_parquet_source,consumer_node_id=n0,consumer_source_node_id=n0,producer_key_column=right_key,consumer_key_column=left_key,filter_kinds=range+small_in_list+hash_set_contains,consumer_lowering=reader_predicate,consumer_reduction_stage=source_reader,selected_source_bytes=1024,total_source_bytes=1024,routed=true,rejected_reason=none"
    );
    assert!(text.contains("rule=dynamic_filter_route_discovery,considered_count=1,route_count=1"));
    assert!(text.contains("dynamic_filter_id=df0"));
    assert!(text.contains("filter_kinds=range+small_in_list+hash_set_contains"));
    assert!(text.contains("consumer_kind=local_parquet_source"));
    assert!(text.contains("consumer_node_id="));
    assert!(text.contains("consumer_lowering=reader_predicate"));
    assert!(text.contains("consumer_reduction_stage=source_reader"));
    assert!(!text.contains("q08"));
}

#[test]
fn native_datafusion_planning_report_uses_native_optimizer_stable_rule_rows() {
    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new("AggregateExec(HashJoinExec)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "AggregateExec".to_string(),
        input_plan_shape: "AggregateExec(HashJoinExec)".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: NativeOptimizerDiagnostics {
            aggregate_join_pushdown: AggregateJoinPushdownDiagnostics {
                considered_count: 1,
                applied_count: 1,
                candidates: Vec::new(),
            },
            ..Default::default()
        },
        source_diagnostics: Vec::new(),
        capability_contract: NativeCapabilityContractDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "aggregate_join_pushdown",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);

    assert_eq!(
        planning_native_optimizer_cell(&planning),
        "rule=aggregate_join_pushdown,considered_count=1,applied_count=1,candidates=none"
    );
}

fn planning_native_optimizer_cell(planning: &NativeDataFusionPlanningReport) -> String {
    let text = planning.to_tsv();
    let mut table_lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = table_lines.next().expect("planning TSV header");
    let row = table_lines.next().expect("planning TSV data row");
    let native_optimizer_index =
        header.split('\t').position(|column| column == "native_optimizer").expect("native_optimizer TSV column");
    row.split('\t').nth(native_optimizer_index).expect("native_optimizer TSV cell").to_string()
}

fn semijoin_reduction_projection_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(semijoin_gpu_source("x", &["x_key", "x_payload"]));
    let y = builder.add_source(semijoin_gpu_source("y", &["y_key", "y_payload"]));
    let keyset = semijoin_aggregate_keyset_source(&mut builder, "keyset", "y_key");

    let xy_schema = semijoin_schema(&["x_key", "x_payload", "y_key", "y_payload"]);
    let xy = builder
        .add_join(
            x,
            y,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "x_key"),
                JoinOutputColumn::left(ColumnRef::input(0, "x_payload", "x_payload"), "x_payload"),
                JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "y_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "y_payload", "y_payload"), "y_payload"),
            ]),
            Arc::clone(&xy_schema),
        )
        .unwrap();
    let semi = builder
        .add_join(
            xy,
            keyset,
            JoinSpec::new(
                JoinType::LeftSemi,
                vec![JoinKey::new(ColumnRef::input(0, "y_key", "y_key"), ColumnRef::input(1, "y_key", "y_key"))],
            )
            .with_output_projection(
                ["x_key", "x_payload", "y_key", "y_payload"]
                    .into_iter()
                    .map(|name| JoinOutputColumn::left(ColumnRef::input(0, name, name), name))
                    .collect(),
            ),
            xy_schema,
        )
        .unwrap();
    builder.finish_gpu_resident(semi).unwrap()
}

fn projection_simplification_identity_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "identity_projection_source",
        semijoin_schema(&["key"]),
    )));
    let projection = builder
        .add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::Int32, false))],
        )
        .unwrap();
    builder.finish_gpu_resident(projection).unwrap()
}

fn dynamic_filter_route_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("left_key", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("right_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Int32, false),
        Field::new("right_key", DataType::Int32, false),
    ]));
    let left_file = LocalParquetFile::new("/tmp/dynamic-filter-left.parquet")
        .with_stable_id("dynamic-filter-left")
        .with_projected_row_group_bytes(vec![1024], vec![1024])
        .with_row_group_rows(vec![10]);
    let right_file = LocalParquetFile::new("/tmp/dynamic-filter-right.parquet")
        .with_stable_id("dynamic-filter-right")
        .with_projected_row_group_bytes(vec![64], vec![64])
        .with_row_group_rows(vec![2]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![left_file],
        left_schema,
        vec!["left_key".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![right_file],
        right_schema,
        vec!["right_key".to_string()],
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "left_key", "left_key"),
                    ColumnRef::input(1, "right_key", "right_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
            ])
            .with_build_side(JoinSide::Right),
            output_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(join).unwrap()
}

fn semijoin_gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, semijoin_schema(fields)))
}

fn semijoin_aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> NodeId {
    let source = builder.add_source(semijoin_gpu_source(name, &[key_name]));
    builder
        .add_aggregate(
            source,
            AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()),
            semijoin_schema(&[key_name]),
        )
        .unwrap()
}

fn semijoin_schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

fn accepted_candidate_contract_entry(
    candidate: GpuCandidate,
    task_count: usize,
) -> NativeCapabilityAcceptedCandidateDiagnostics {
    NativeCapabilityAcceptedCandidateDiagnostics {
        candidate,
        report: CapabilityContractReport::from_decision_rows(CapabilityEnvelopeStatus::Admitted, Vec::new()).unwrap(),
        compiled_graph: NativeCapabilityCompiledGraphDiagnostics {
            task_count,
            edge_count: 1,
            barrier_count: 0,
            resource_count: 1,
            sink_kind: "host_arrow_materialization".to_string(),
        },
    }
}

#[test]
fn native_datafusion_execution_report_marks_terminal_planning_failure_before_runtime() {
    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::planning_failure(
        "q03",
        "native_planning",
        "native_planning_failed",
        "native planning did not produce an executable plan",
    )]);

    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.runtime_started_count(), 0);
    assert_eq!(report.rejected_before_execution_count(), 1);
    let row = &report.rows[0];
    assert_eq!(row.planning_outcome, None);
    assert_eq!(row.reason_code.as_deref(), Some("native_planning_failed"));
    assert_eq!(row.final_disposition, None);
    assert_eq!(row.native_plan_fragment_count, None);
    assert_eq!(row.host_datafusion_boundary_count, None);
    assert!(row.capability_decision_rows.is_empty());
    assert!(row.capability_compiled_graphs.is_empty());
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
    assert_eq!(planning.rows[0].post_native_plan_root, "AlgeonNativeComposedExec");
    assert_eq!(planning.rows[0].final_disposition, Some(FinalPlanDisposition::Native));

    let execution = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "composed",
        composed,
        accepted_native_diagnostics(),
        NativeDataFusionRuntime::NotStarted,
    )]);
    assert_eq!(execution.rows[0].post_native_plan_root, "AlgeonNativeComposedExec");
    assert_eq!(execution.rows[0].final_disposition, Some(FinalPlanDisposition::Native));
}

fn native_exec_leaf_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-execution-report-source-not-read.parquet",
        schema,
        vec!["key".to_string()],
    )));
    Arc::new(
        AlgeonNativeExec::try_new_host_for_test(
            builder.finish_with_host_sink(source).unwrap(),
            native_execution_config(),
        )
        .unwrap(),
    )
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
        AlgeonNativeExec::try_new_producer_for_test(
            "fragment_input",
            producer_builder.finish_gpu_resident(producer_source).unwrap(),
        )
        .unwrap(),
    );

    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_source = consumer_builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("fragment_input", Arc::clone(&schema))));
    Arc::new(
        AlgeonNativeExec::try_new_composed_for_test(
            consumer_builder.finish_with_host_sink(consumer_source).unwrap(),
            vec![producer],
            native_execution_config(),
        )
        .unwrap(),
    )
}

fn native_execution_config() -> AlgeonNativeExecutionConfig {
    AlgeonNativeExecutionConfig::default()
}

fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new("SortExec(fetch)(AggregateExec(LocalParquetScan))", 1));
    NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(LocalParquetScan))".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: Default::default(),
    }
}

fn accepted_bounded_join_native_diagnostics() -> NativePlanningDiagnostics {
    let mut report = PlanningReport::new();
    report.push_selected(GpuCandidate::new(
        "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))",
        1,
    ));
    NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        input_join_diagnostics: Vec::new(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        capability_contract: accepted_capability_contract_diagnostics(),
    }
}

fn accepted_capability_contract_diagnostics() -> NativeCapabilityContractDiagnostics {
    let candidate =
        GpuCandidate::new("SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))", 1);
    NativeCapabilityContractDiagnostics {
        accepted_candidates: vec![NativeCapabilityAcceptedCandidateDiagnostics {
            candidate,
            report: CapabilityContractReport::from_decision_rows(CapabilityEnvelopeStatus::Admitted, Vec::new())
                .unwrap(),
            compiled_graph: NativeCapabilityCompiledGraphDiagnostics {
                task_count: 2,
                edge_count: 1,
                barrier_count: 0,
                resource_count: 2,
                sink_kind: "host_arrow_materialization".to_string(),
            },
        }],
        rejected_candidates: Vec::new(),
    }
}

fn native_execution_metrics_set() -> MetricsSet {
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
        ("resident_dataframe_peak_bytes_observed", 1),
        ("allocator_high_water_bytes", 8192),
        ("allocator_high_water_bytes_observed", 1),
        ("max_task_observed_peak_bytes", 72),
        ("max_task_observed_peak_bytes_observed", 1),
        ("max_task_retained_bytes", 48),
        ("max_task_retained_bytes_observed", 1),
        ("source_chunks_planned", 6),
        ("source_chunks_attempted", 7),
        ("source_chunks_committed", 5),
        ("source_chunks_retried", 2),
        ("source_read_prefetches_started", 3),
        ("source_read_prefetches_completed", 2),
        ("source_read_prefetches_skipped", 1),
        ("source_read_prefetches_abandoned", 1),
        ("row_groups_planned", 8),
        ("row_groups_committed", 4),
        ("partial_aggregate_count", 5),
        ("partial_aggregate_input_rows", 11),
        ("partial_aggregate_rows", 9),
        ("partial_aggregate_retained_bytes", 4096),
        ("final_merge_attempts", 2),
        ("final_merge_levels", 1),
        ("final_merge_max_fan_in", 3),
        ("retry_count", 3),
        ("retry_resource_exhausted_count", 1),
        ("dynamic_filter_produced_count", 1),
        ("dynamic_filter_skipped_count", 1),
        ("dynamic_filter_applied_count", 1),
        ("dynamic_filter_row_groups_pruned", 3),
        ("dynamic_filter_source_bytes_pruned", 4096),
        ("dynamic_filter_reader_predicate_rows_before", 1000),
        ("dynamic_filter_reader_predicate_rows_after", 250),
        ("dynamic_filter_reader_predicate_rows_filtered", 750),
        ("dynamic_filter_post_decode_rows_before", 80),
        ("dynamic_filter_post_decode_rows_after", 25),
        ("dynamic_filter_post_decode_rows_filtered", 55),
        ("dynamic_filter_source_post_decode_rows_filtered", 55),
        ("dynamic_filter_filter_post_decode_rows_filtered", 0),
        ("dynamic_filter_join_probe_post_decode_rows_filtered", 0),
        ("hash_join_build_count", 1),
        ("filtered_join_build_count", 6),
        ("one_shot_join_count", 7),
        ("hash_join_hash_table_bytes", 16384),
        ("filtered_join_hash_table_bytes", 8192),
        ("filtered_join_probe_state_bytes", 1024),
        ("mixed_join_size_data_bytes", 512),
        ("one_shot_join_index_bytes", 4096),
        ("one_shot_join_materialization_bytes", 8192),
        ("one_shot_join_output_bytes", 16384),
        ("join_gather_map_bytes", 2048),
    ] {
        MetricBuilder::new(&metrics).with_type(MetricType::Summary).counter(name, 0).add(value);
    }
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("node_id", "n3")
        .with_new_label("algorithm", "pagerank")
        .with_new_label("runtime_outcome", "succeeded")
        .with_new_label("input_boundary", "gpu_resident")
        .with_new_label("output_boundary", "gpu_resident")
        .with_new_label("construction_policy", "python_cugraph")
        .with_new_label("source_column", "src")
        .with_new_label("destination_column", "dst")
        .with_new_label("weight_column", "weight")
        .with_new_label("directed", "true")
        .with_new_label("renumber", "true")
        .with_new_label("algorithm_parameters", "alpha=0.85,epsilon=0.000001,max_iterations=100")
        .with_new_label("algorithm_metadata", "converged=true,iterations=12")
        .with_new_label("input_rows", "6")
        .with_new_label("input_columns", "3")
        .with_new_label("output_rows", "5")
        .with_new_label("output_columns", "2")
        .with_new_label("output_device_bytes", "320")
        .with_new_label("graph_peak_bytes", "2048")
        .with_new_label("elapsed_ns", "123456")
        .with_new_label("gpu_resident_input_handoffs", "0")
        .with_new_label("host_to_device_uploads", "0")
        .with_new_label("device_to_host_materializations", "0")
        .global_counter("native_graph_runtime_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("operator_kind", "source_read")
        .with_new_label("task_count", "3")
        .with_new_label("max_observed_peak_bytes", "72")
        .with_new_label("max_retained_bytes", "48")
        .global_counter("native_task_operator_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("compiled_task_id", "t7")
        .with_new_label("operator_kind", "source_read")
        .with_new_label("max_observed_peak_bytes", "72")
        .with_new_label("max_retained_bytes", "48")
        .global_counter("native_task_runtime_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("dynamic_filter_id", "df0")
        .with_new_label("event", "applied")
        .with_new_label("producer_kind", "hash_join_build")
        .with_new_label("producer_node_id", "n2")
        .with_new_label("consumer_kind", "local_parquet_source")
        .with_new_label("consumer_node_id", "n1")
        .with_new_label("consumer_lowering", "merged_reader_predicate")
        .with_new_label("consumer_reduction_stage", "source_reader")
        .with_new_label("filter_kind", "small_in_list")
        .with_new_label("payload_value_count", "3")
        .with_new_label("payload_min_i64", "10")
        .with_new_label("payload_max_i64", "30")
        .with_new_label("producer_rows", "none")
        .with_new_label("skipped_reason", "none")
        .with_new_label("selected_row_groups_before", "5")
        .with_new_label("selected_row_groups_after", "2")
        .with_new_label("selected_source_bytes_before", "8192")
        .with_new_label("selected_source_bytes_after", "4096")
        .with_new_label("row_groups_pruned", "3")
        .with_new_label("source_bytes_pruned", "4096")
        .global_counter("native_dynamic_filter_runtime_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("dynamic_filter_id", "df0")
        .with_new_label("event", "applied_post_decode")
        .with_new_label("producer_kind", "hash_join_build")
        .with_new_label("producer_node_id", "n2")
        .with_new_label("consumer_kind", "local_parquet_source")
        .with_new_label("consumer_node_id", "n1")
        .with_new_label("consumer_lowering", "post_decode_only")
        .with_new_label("consumer_reduction_stage", "source_post_decode")
        .with_new_label("filter_kind", "hash_set_contains")
        .with_new_label("payload_value_count", "3")
        .with_new_label("payload_min_i64", "10")
        .with_new_label("payload_max_i64", "30")
        .with_new_label("producer_rows", "none")
        .with_new_label("skipped_reason", "none")
        .with_new_label("post_decode_rows_before", "80")
        .with_new_label("post_decode_rows_after", "25")
        .with_new_label("post_decode_rows_filtered", "55")
        .with_new_label("row_groups_pruned", "0")
        .with_new_label("source_bytes_pruned", "0")
        .global_counter("native_dynamic_filter_runtime_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("node_id", "n4")
        .with_new_label("selected_strategy", "libcudf_groupby")
        .with_new_label("execution_stage", "partial")
        .with_new_label("group_key_count", "1")
        .with_new_label("group_key_exprs", "l_returnflag")
        .with_new_label("group_key_dtypes", "Utf8")
        .with_new_label("aggregate_functions", "sum,count")
        .with_new_label("aggregate_filter_count", "0")
        .with_new_label("distinct_aggregate_count", "0")
        .with_new_label("input_rows", "6001215")
        .with_new_label("output_group_count", "4")
        .with_new_label("reduction_ratio_ppm", "0")
        .with_new_label("chunk_row_count", "6001215")
        .with_new_label("chunk_count", "1")
        .with_new_label("partial_aggregate_count", "0")
        .with_new_label("source_adjacent_partial", "true")
        .with_new_label("source_chunk_row_group_count", "1")
        .with_new_label("source_chunk_reads_all_row_groups", "false")
        .with_new_label("source_chunk_known_row_count", "6001215")
        .with_new_label("source_chunk_estimated_source_read_bytes", "123456")
        .with_new_label("source_chunk_size_estimate_based", "false")
        .with_new_label("selection_elapsed_ns", "0")
        .with_new_label("encode_elapsed_ns", "0")
        .with_new_label("groupby_elapsed_ns", "123456789")
        .with_new_label("decode_elapsed_ns", "0")
        .with_new_label("strategy_operator_elapsed_ns", "123456789")
        .with_new_label("aggregate_structural_key", "agg-v1-fnv64-0123456789abcdef")
        .with_new_label("prior_cardinality_evidence_scope", "absent")
        .with_new_label("encoded_key_partial_cost_model", "prior_partial_reduction_ratio")
        .with_new_label("encoded_key_partial_cost_model_status", "rejected_by_cost_model")
        .with_new_label("encoded_key_partial_selection_reason", "no_prior_cardinality_evidence")
        .with_new_label("encoded_key_partial_reduction_ratio_threshold_ppm", "100000")
        .with_new_label("encoded_key_partial_prior_reduction_ratio_ppm", "0")
        .with_new_label("encoded_key_partial_prior_strategy", "")
        .with_new_label("prepared_sort_groupby_rejected_reason", "none")
        .with_new_label("prepared_sort_groupby_contract_status", "available")
        .with_new_label("prepared_sort_groupby_required_contract", "prepared_sort_groupby_execution_contract")
        .with_new_label("prepared_sort_groupby_cost_model", "exact_grouped_key_facts_required")
        .with_new_label("prepared_sort_groupby_cost_model_status", "selected_libcudf_runtime_probe")
        .with_new_label("prepared_sort_groupby_selection_reason", "runtime_cardinality_probe_required")
        .with_new_label("prepared_sort_groupby_num_groups", "unknown")
        .with_new_label("prepared_sort_groupby_grouped_key_output_bytes", "unknown")
        .with_new_label("prepared_sort_groupby_max_group_size", "unknown")
        .with_new_label(
            "rejected_candidate_strategies",
            "encoded_key_partial:encoded_key_partial_cost_model_no_prior_cardinality_evidence;partitioned_merge:partitioned_merge_requires_final_merge_stage",
        )
        .global_counter("native_aggregate_strategy_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("node_id", "n2")
        .with_new_label("join_strategy", "one_shot_equi")
        .with_new_label("execution_strategy", "bounded_one_shot_equi_join")
        .with_new_label("join_type", "inner")
        .with_new_label("build_side", "right")
        .with_new_label("key_count", "1")
        .with_new_label("estimated_build_rows", "100")
        .with_new_label("observed_build_rows", "101")
        .with_new_label("build_side_bytes", "1024")
        .with_new_label("probe_side_bytes", "2048")
        .with_new_label("estimated_output_rows", "102")
        .with_new_label("estimated_output_bytes", "4096")
        .with_new_label("build_key_unique", "true")
        .with_new_label("build_key_uniqueness_probe_status", "measured")
        .with_new_label("build_key_uniqueness_probe_error_code", "none")
        .with_new_label("probe_reuse_count", "1")
        .with_new_label("reusable_hash_join_rejected_reason", "build_side_too_large")
        .with_new_label("one_shot_equi_rejected_reason", "none")
        .with_new_label("hash_join_build_count", "0")
        .with_new_label("hash_join_hash_table_bytes", "0")
        .with_new_label("filtered_join_build_count", "0")
        .with_new_label("filtered_join_hash_table_bytes", "0")
        .with_new_label("filtered_join_probe_state_bytes", "0")
        .with_new_label("mixed_join_size_data_bytes", "0")
        .with_new_label("one_shot_join_count", "1")
        .with_new_label("one_shot_join_index_bytes", "2048")
        .with_new_label("one_shot_join_materialization_bytes", "4096")
        .with_new_label("one_shot_join_output_bytes", "8192")
        .with_new_label("join_gather_map_bytes", "2048")
        .global_counter("native_join_strategy_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("outcome", "completed")
        .with_new_label("bytes", "4096")
        .with_new_label("elapsed_ns", "9000000")
        .with_new_label("cache_path_hash", "cache123")
        .with_new_label("stable_id_hash", "stable456")
        .with_new_label("file_index", "0")
        .with_new_label("chunk_index", "2")
        .with_new_label("error_code", "none")
        .with_new_label("error_kind", "none")
        .with_new_label("error_status", "none")
        .with_new_label("error_operation", "none")
        .global_counter("native_lakehouse_cache_fill_event_snapshot_rows")
        .add(1);
    metrics.clone_inner()
}

#[test]
fn unexecuted_native_wrappers_mint_no_handoff_metrics() {
    let leaf = native_exec_leaf_for_report();
    let composed = composed_native_exec_for_report();
    let mut frontier = vec![Arc::clone(&leaf), Arc::clone(&composed)];
    let mut visited = 0;
    while let Some(plan) = frontier.pop() {
        visited += 1;
        let metrics = plan.metrics().expect("fresh native wrapper must expose a metrics set");
        assert!(metrics.sum_by_name("gpu_resident_input_handoffs").is_none());
        frontier.extend(plan.children().into_iter().map(Arc::clone));
    }
    assert!(visited >= 3);

    let report = NativeDataFusionExecutionReport::collect([NativeDataFusionExecutionWorkloadItem::new(
        "unexecuted",
        composed,
        accepted_native_diagnostics(),
        NativeDataFusionRuntime::NotStarted,
    )]);
    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.rows[0].metrics.gpu_resident_input_handoffs, 0);
}
