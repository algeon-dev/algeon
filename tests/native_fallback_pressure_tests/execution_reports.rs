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
    assert_eq!(
        row.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(row.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
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
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, 9);
    assert_eq!(
        NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\tplanning_outcome\treason_code\tunsupported_category\tbounded_join_reservation_facts\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution\tnative_runtime_started\truntime_outcome\truntime_failure\toutput_batches\toutput_rows\toutput_bytes\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\treservation_denied_count\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\tsource_read_prefetches_started\tsource_read_prefetches_completed\tsource_read_prefetches_skipped\tsource_read_prefetches_abandoned\trow_groups_planned\trow_groups_committed\tpartial_aggregate_count\tpartial_aggregate_rows\tfinal_merge_attempts\tfinal_merge_levels\tretry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\thash_join_build_count\tfiltered_join_build_count\thash_join_cache_hits\thash_join_cache_misses\thash_join_durable_cache_hits\thash_join_durable_cache_misses\thash_join_durable_cache_inserts\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tmixed_join_size_data_bytes\tjoin_gather_map_bytes\tsource_diagnostics"
    );
    assert!(text.starts_with("# native_datafusion_execution_tsv_schema_version=9\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_EXECUTION_TSV_HEADER}\n")));
    assert!(text.contains("bounded_join_execution_status=enabled"));
    assert!(text.contains("\tNativeCudfExec\tnative\tfalse\ttrue\tsucceeded\tnone\t1\t4\t256\t1\t0\t0\t0\t1\t4\t256\t2\t512\t1\t1024\t2\t4\t4096\t8192\t128\t128\t0\t6\t7\t5\t2\t3\t2\t1\t1\t8\t4\t5\t9\t2\t1\t3\t1\t2\t1\t6\t2\t1\t3\t4\t5\t16384\t8192\t1024\t512\t2048\t"));
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
fn native_datafusion_reports_scope_bounded_join_facts_to_join_candidates() {
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
        bounded_join_reservation_facts: BoundedJoinReservationContract
            .facts(ExecutionBoundaryPolicy::default())
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
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

    assert_eq!(
        join_row.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(
        join_row
            .bounded_join_reservation_facts
            .get("bounded_join_execution_family_alias_compatibility")
            .map(String::as_str),
        None
    );
    assert_eq!(
        join_row
            .bounded_join_reservation_facts
            .get("bounded_join_mixed_keyed_residual_capability_detail")
            .map(String::as_str),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );
    assert!(
        non_join_row.bounded_join_reservation_facts.is_empty(),
        "bounded join facts must not bleed into unrelated candidate rows"
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
    assert_eq!(row.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(
        row.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(
        row.bounded_join_reservation_facts.get("cudf_rs_mixed_join_size_data_alias_compatibility").map(String::as_str),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
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
