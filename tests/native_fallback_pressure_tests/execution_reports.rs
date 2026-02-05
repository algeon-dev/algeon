use super::*;

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
    assert_eq!(row.planning_outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.strict_bounded_contract_rows.len(), 2);
    assert_eq!(row.strict_bounded_compiled_graphs.len(), 1);
    assert!(
        row.strict_bounded_contract_rows.iter().any(|gate| gate.reservation_kind == ReservationKind::Materialization
            && gate.contract_quality == ContractQuality::QueryEngineOwnedBoundary)
    );
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
    assert_eq!(row.metrics.source_read_prefetches_started, 3);
    assert_eq!(row.metrics.source_read_prefetches_completed, 2);
    assert_eq!(row.metrics.source_read_prefetches_skipped, 1);
    assert_eq!(row.metrics.source_read_prefetches_abandoned, 1);
    assert_eq!(row.metrics.source_cache_hits, 4);
    assert_eq!(row.metrics.source_cache_misses, 5);
    assert_eq!(row.metrics.source_cache_inserts, 3);
    assert_eq!(row.metrics.source_cache_insert_skipped, 2);
    assert_eq!(row.metrics.source_cache_retained_bytes, 65536);
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
    assert_eq!(row.metrics.mixed_join_size_data_bytes, 512);
    assert_eq!(row.metrics.join_gather_map_bytes, 2048);

    let text = report.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, 11);
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.split('\t').count(), 72);
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts\tsource_cache_insert_skipped\tsource_cache_retained_bytes\t"));
    assert!(text.starts_with("# native_datafusion_execution_tsv_schema_version=11\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_EXECUTION_TSV_HEADER}\n")));
    assert!(text.contains("execution_mode=strict_bounded"));
    assert!(text.contains("contract_status=modeled"));
    assert!(text.contains("contract_quality=query_engine_owned_boundary"));
    assert!(!text.contains("bounded_join_execution_status"));
    assert!(!text.contains("bounded_join_reservation_facts"));
    assert!(text.contains("\tNativeCudfExec\tnative\tfalse\ttrue\tsucceeded\tnone\t1\t4\t256\t"));
}

#[test]
fn native_datafusion_execution_report_keeps_unobserved_optional_memory_metrics_empty() {
    let metrics = ExecutionPlanMetricsSet::new();
    MetricBuilder::new(&metrics).with_type(MetricType::SUMMARY).counter("resident_dataframe_peak_bytes", 0);
    MetricBuilder::new(&metrics).with_type(MetricType::SUMMARY).counter("allocator_high_water_bytes", 0);
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
fn native_datafusion_reports_scope_strict_contract_rows_to_join_candidates() {
    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new(
        "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))",
        1,
    ));
    report.push_rejected(
        GpuCandidate::new("SortExec(fetch)(AggregateExec(LocalParquetScan))", 1),
        datafusion_cudf::FallbackReason::UnsupportedExpression,
    );
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        report,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: accepted_strict_bounded_contract_diagnostics(),
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

    assert_eq!(join_row.strict_bounded_contract_rows.len(), 2);
    assert_eq!(join_row.strict_bounded_compiled_graphs.len(), 1);
    assert!(
        non_join_row.strict_bounded_contract_rows.is_empty(),
        "strict bounded contract rows must not bleed into unrelated candidate rows"
    );
    assert!(
        non_join_row.strict_bounded_compiled_graphs.is_empty(),
        "strict bounded graph diagnostics must not bleed into unrelated candidate rows"
    );
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
    assert_eq!(row.strict_bounded_contract_rows.len(), 1);
    assert_eq!(
        row.strict_bounded_contract_rows[0].execution_mode,
        StrictBoundedExecutionMode::UnsupportedStrictBounded
    );
    assert_eq!(row.strict_bounded_contract_rows[0].contract_status, ContractStatus::BlockedByLowerLevel);
    assert!(row.strict_bounded_compiled_graphs.is_empty());
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
