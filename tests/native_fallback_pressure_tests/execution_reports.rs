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
    assert_eq!(row.metrics.reservation_live_peak_bytes, 96);
    assert_eq!(row.metrics.max_task_reservation_bytes, 64);
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
    assert_eq!(row.metrics.task_operator_rows.len(), 1);
    assert_eq!(row.metrics.task_operator_rows[0].operator_kind, "source_read");
    assert_eq!(row.metrics.task_operator_rows[0].task_count, 3);
    assert_eq!(row.metrics.task_operator_rows[0].reservation_requested_bytes, 120);
    assert_eq!(row.metrics.task_operator_rows[0].reservation_granted_bytes, 96);
    assert_eq!(row.metrics.task_operator_rows[0].reservation_released_bytes, 32);
    assert_eq!(row.metrics.task_operator_rows[0].reservation_live_peak_bytes, 64);
    assert_eq!(row.metrics.task_operator_rows[0].max_observed_peak_bytes, Some(72));
    assert_eq!(row.metrics.task_operator_rows[0].max_retained_bytes, Some(48));
    assert_eq!(row.metrics.task_runtime_rows.len(), 1);
    assert_eq!(row.metrics.task_runtime_rows[0].compiled_task_id, "t7");
    assert_eq!(row.metrics.task_runtime_rows[0].operator_kind, "source_read");
    assert_eq!(row.metrics.task_runtime_rows[0].reservation_requested_bytes, 80);
    assert_eq!(row.metrics.task_runtime_rows[0].reservation_granted_bytes, 64);
    assert_eq!(row.metrics.task_runtime_rows[0].reservation_released_bytes, 48);
    assert_eq!(row.metrics.task_runtime_rows[0].reservation_live_peak_bytes, 64);
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
    assert_eq!(row.metrics.source_cache_hits, 4);
    assert_eq!(row.metrics.source_cache_misses, 5);
    assert_eq!(row.metrics.source_cache_inserts, 3);
    assert_eq!(row.metrics.source_cache_insert_skipped, 2);
    assert_eq!(row.metrics.source_cache_retained_bytes, 65536);
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
    assert_eq!(row.metrics.retry_reservation_failure_count, 2);
    assert_eq!(row.metrics.adaptive_retry_count, 10);
    assert_eq!(row.metrics.adaptive_spill_count, 11);
    assert_eq!(row.metrics.adaptive_downgrade_count, 12);
    assert_eq!(row.metrics.adaptive_memory_history_lookup_count, 13);
    assert_eq!(row.metrics.adaptive_memory_history_hit_count, 8);
    assert_eq!(row.metrics.adaptive_memory_history_miss_count, 5);
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
    assert_eq!(row.metrics.hash_join_cache_hits, 2);
    assert_eq!(row.metrics.hash_join_cache_misses, 1);
    assert_eq!(row.metrics.hash_join_durable_cache_hits, 3);
    assert_eq!(row.metrics.hash_join_durable_cache_misses, 4);
    assert_eq!(row.metrics.hash_join_durable_cache_inserts, 5);
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
    assert_eq!(row.metrics.join_strategy_rows.len(), 1);
    assert_eq!(row.metrics.join_strategy_rows[0].node_id, "n2");
    assert_eq!(row.metrics.join_strategy_rows[0].join_strategy, "one_shot_equi");
    assert_eq!(
        row.metrics.join_strategy_rows[0].reusable_hash_join_rejected_reason.as_deref(),
        Some("build_side_too_large")
    );

    let text = report.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, 37);
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.split('\t').count(), 129);
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains(
        "\treservation_requested_bytes\treservation_granted_bytes\treservation_live_peak_bytes\treservation_denied_count\t"
    ));
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts\tsource_cache_insert_skipped\tsource_cache_retained_bytes\t"));
    assert!(text.starts_with(&format!(
        "# native_datafusion_execution_tsv_schema_version={NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION}\n"
    )));
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
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains(
        "\tsource_fetch_bytes\tsource_fetch_count\thost_fetch_budget_wait_time_ms\tsource_fetch_concurrency_wait_time_ms\tsource_fetch_time_ms\tsource_decode_time_ms\t"
    ));
    assert!(text.contains("selected_strategy=libcudf_groupby,execution_stage=partial"));
    assert!(text.contains("algorithm=pagerank,runtime_outcome=succeeded"));
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
    assert!(text.contains("operator_kind=source_read,task_count=3,elapsed_ns=0,reservation_requested_bytes=120,reservation_granted_bytes=96,reservation_released_bytes=32,reservation_live_peak_bytes=64,max_observed_peak_bytes=72,max_retained_bytes=48\t"));
    assert!(text.contains("\tcompiled_task_id=t7,operator_kind=source_read,elapsed_ns=0,reservation_requested_bytes=80,reservation_granted_bytes=64,reservation_released_bytes=48,reservation_live_peak_bytes=64,max_observed_peak_bytes=72,max_retained_bytes=48\t"));
    assert!(text.contains("execution_mode=strict_bounded"));
    assert!(text.contains("contract_status=modeled"));
    assert!(text.contains("contract_quality=query_engine_owned_boundary"));
    assert!(!text.contains("bounded_join_execution_status"));
    assert!(!text.contains("bounded_join_reservation_facts"));
    assert!(text.contains("node_id=n2,join_strategy=one_shot_equi,execution_strategy=bounded_one_shot_equi_join"));
    assert!(text.contains("\tNativeCudfExec\tnative\t1\t0\tfalse\ttrue\tsucceeded\tnone\t1\t4\t256\t"));
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
            datafusion_nexus::NativeAggregateStrategyMetricRow {
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
) -> datafusion_nexus::NativeAggregateStrategyMetricRow {
    datafusion_nexus::NativeAggregateStrategyMetricRow {
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
        datafusion_nexus::FallbackReason::UnsupportedExpression,
    );
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "SortExec".to_string(),
        input_plan_shape: "SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))".to_string(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        strict_bounded_contract: accepted_strict_bounded_contract_diagnostics(),
        adaptive_pipeline: Default::default(),
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
fn native_datafusion_reports_keep_multiple_accepted_join_candidates_separated() {
    let first_candidate = GpuCandidate::new("HashJoinExec(LocalParquetScan,LocalParquetScan)", 1);
    let second_candidate = GpuCandidate::new("NestedLoopJoinExec(LocalParquetScan,LocalParquetScan)", 1);
    let mut report = PlanningReport::new();
    report.push_rewritten(first_candidate.clone());
    report.push_rewritten(second_candidate.clone());

    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "UnionExec".to_string(),
        input_plan_shape: "UnionExec(HashJoinExec(...),NestedLoopJoinExec(...))".to_string(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        strict_bounded_contract: NativeStrictBoundedContractDiagnostics {
            accepted_candidates: vec![
                accepted_candidate_contract_entry(
                    first_candidate.clone(),
                    ReservationGateId::new(11),
                    ReservationKind::BuildSide,
                    "join_build_input",
                    "query_engine_build_side_source_reservation",
                    11,
                ),
                accepted_candidate_contract_entry(
                    second_candidate.clone(),
                    ReservationGateId::new(22),
                    ReservationKind::Materialization,
                    "host_arrow_materialization",
                    "query_engine_explicit_sink_materialization_reservation",
                    22,
                ),
            ],
            rejected_candidates: Vec::new(),
            non_join_bounded_plan_count: 0,
            non_join_bounded_plan_reason: None,
        },
        adaptive_pipeline: Default::default(),
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

    assert_eq!(first_row.strict_bounded_contract_rows.len(), 1);
    assert_eq!(
        first_row.strict_bounded_contract_rows[0].gate_id,
        StrictBoundedContractGateId::Compiled(ReservationGateId::new(11))
    );
    assert_eq!(first_row.strict_bounded_compiled_graphs[0].task_count, 11);
    assert_eq!(second_row.strict_bounded_contract_rows.len(), 1);
    assert_eq!(
        second_row.strict_bounded_contract_rows[0].gate_id,
        StrictBoundedContractGateId::Compiled(ReservationGateId::new(22))
    );
    assert_eq!(second_row.strict_bounded_compiled_graphs[0].task_count, 22);
}

#[test]
fn native_datafusion_reports_project_adaptive_rows_without_strict_contract_rows() {
    let candidate = GpuCandidate::new("HashJoinExec(LocalParquetScan,LocalParquetScan)", 1);
    let mut report = PlanningReport::new();
    report.push_rewritten(candidate.clone());
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "HashJoinExec".to_string(),
        input_plan_shape: "HashJoinExec(LocalParquetScan,LocalParquetScan)".to_string(),
        report,
        native_optimizer: Default::default(),
        source_diagnostics: Vec::new(),
        strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
        adaptive_pipeline: NativeAdaptivePipelineDiagnostics {
            candidates: vec![NativeAdaptivePipelineCandidateDiagnostics {
                candidate,
                report: adaptive_report_for_projection_test(),
            }],
        },
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "adaptive",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);
    let row = &planning.rows[0];
    assert!(row.strict_bounded_contract_rows.is_empty());
    assert!(row.strict_bounded_compiled_graphs.is_empty());
    assert_eq!(row.adaptive_pipeline_rows.len(), 1);

    let rendered = row.adaptive_pipeline_rows[0].to_stable_kv_row();
    assert!(rendered.contains("execution_mode=adaptive"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));

    let text = planning.to_tsv();
    assert!(text.contains("adaptive_pipeline_rows"));
    assert!(text.contains("execution_mode=adaptive"));
    assert!(!text.contains("execution_mode=strict_bounded"));
}

#[test]
fn native_datafusion_planning_report_projects_native_optimizer_diagnostics() {
    let optimizer_result =
        nexus_query_engine::optimize_query_plan(semijoin_reduction_projection_plan(), &Default::default());
    assert_eq!(optimizer_result.diagnostics.semijoin_reduction.applied_count, 1);

    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("HashJoinExec(HashJoinExec,GpuKeyset)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "HashJoinExec".to_string(),
        input_plan_shape: "HashJoinExec(HashJoinExec,GpuKeyset)".to_string(),
        report,
        native_optimizer: optimizer_result.diagnostics,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
        adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "generic_semijoin",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);
    let row = &planning.rows[0];
    assert_eq!(row.native_optimizer.semijoin_reduction.considered_count, 1);
    assert_eq!(row.native_optimizer.semijoin_reduction.applied_count, 1);

    let text = planning.to_tsv();
    assert!(text.contains("\tnative_optimizer\t"));
    assert!(text.contains("rule=semijoin_reduction,considered_count=1,applied_count=1"));
    assert!(text.contains("crossed_join_node_ids="));
    assert!(text.contains("cost_gate="));
    assert!(!text.contains("q18"));
}

#[test]
fn native_datafusion_planning_report_projects_projection_simplification_diagnostics() {
    let optimizer_result =
        nexus_query_engine::optimize_query_plan(projection_simplification_identity_plan(), &Default::default());
    assert_eq!(optimizer_result.diagnostics.projection_simplification.considered_count, 1);
    assert_eq!(optimizer_result.diagnostics.projection_simplification.applied_count, 1);

    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("ProjectionExec(GpuResidentSource)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "ProjectionExec".to_string(),
        input_plan_shape: "ProjectionExec(GpuResidentSource)".to_string(),
        report,
        native_optimizer: optimizer_result.diagnostics,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
        adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "projection_simplification",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);
    let row = &planning.rows[0];
    assert_eq!(row.native_optimizer.projection_simplification.considered_count, 1);
    assert_eq!(row.native_optimizer.projection_simplification.applied_count, 1);

    let text = planning.to_tsv();
    assert!(text.contains("rule=projection_simplification,considered_count=1,applied_count=1"));
    assert!(text.contains("replacement_node_id="));
    assert!(text.contains("applied=true"));
    assert!(!text.contains("q13"));
}

#[test]
fn native_datafusion_planning_report_projects_dynamic_filter_route_diagnostics() {
    let optimizer_result = nexus_query_engine::optimize_query_plan(dynamic_filter_route_plan(), &Default::default());
    assert_eq!(optimizer_result.diagnostics.dynamic_filter_route_discovery.route_count, 1);

    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("HashJoinExec(LocalParquetScan,LocalParquetScan)", 1));
    let diagnostics = NativePlanningDiagnostics {
        input_plan_root: "HashJoinExec".to_string(),
        input_plan_shape: "HashJoinExec(LocalParquetScan,LocalParquetScan)".to_string(),
        report,
        native_optimizer: optimizer_result.diagnostics,
        source_diagnostics: Vec::new(),
        strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
        adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
    };

    let planning = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "dynamic_filter_route",
        native_exec_leaf_for_report(),
        diagnostics,
    )]);

    let text = planning.to_tsv();
    assert!(text.contains("rule=dynamic_filter_route_discovery,considered_count=1,route_count=1"));
    assert!(text.contains("dynamic_filter_id=df0"));
    assert!(text.contains("filter_kinds=range+small_in_list+hash_set_contains"));
    assert!(text.contains("consumer_kind=local_parquet_source"));
    assert!(text.contains("consumer_node_id="));
    assert!(text.contains("consumer_lowering=reader_predicate"));
    assert!(text.contains("consumer_reduction_stage=source_reader"));
    assert!(!text.contains("q08"));
}

fn semijoin_reduction_projection_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(semijoin_gpu_source("x", &["x_key", "x_payload"]));
    let y = builder.add_source(semijoin_gpu_source("y", &["y_key", "y_payload"]));
    let keyset = semijoin_aggregate_keyset_source(&mut builder, "keyset", "y_key");

    let xy_schema = semijoin_schema(&["x_key", "x_payload", "y_key", "y_payload"]);
    let xy = builder.add_join(
        x,
        y,
        nexus_query_engine::JoinSpec::new(
            nexus_query_engine::JoinType::Inner,
            vec![nexus_query_engine::JoinKey::new(
                nexus_query_engine::ColumnRef::input(0, "x_key", "x_key"),
                nexus_query_engine::ColumnRef::input(1, "y_key", "y_key"),
            )],
        )
        .with_output_projection(vec![
            nexus_query_engine::JoinOutputColumn::left(
                nexus_query_engine::ColumnRef::input(0, "x_key", "x_key"),
                "x_key",
            ),
            nexus_query_engine::JoinOutputColumn::left(
                nexus_query_engine::ColumnRef::input(0, "x_payload", "x_payload"),
                "x_payload",
            ),
            nexus_query_engine::JoinOutputColumn::right(
                nexus_query_engine::ColumnRef::input(1, "y_key", "y_key"),
                "y_key",
            ),
            nexus_query_engine::JoinOutputColumn::right(
                nexus_query_engine::ColumnRef::input(1, "y_payload", "y_payload"),
                "y_payload",
            ),
        ]),
        Arc::clone(&xy_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        nexus_query_engine::JoinSpec::new(
            nexus_query_engine::JoinType::LeftSemi,
            vec![nexus_query_engine::JoinKey::new(
                nexus_query_engine::ColumnRef::input(0, "y_key", "y_key"),
                nexus_query_engine::ColumnRef::input(1, "y_key", "y_key"),
            )],
        )
        .with_output_projection(
            ["x_key", "x_payload", "y_key", "y_payload"]
                .into_iter()
                .map(|name| {
                    nexus_query_engine::JoinOutputColumn::left(
                        nexus_query_engine::ColumnRef::input(0, name, name),
                        name,
                    )
                })
                .collect(),
        ),
        xy_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn projection_simplification_identity_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "identity_projection_source",
        semijoin_schema(&["key"]),
    )));
    let projection = builder.add_projection(
        source,
        vec![nexus_query_engine::ProjectionOutput::new(
            nexus_query_engine::Expr::column("key"),
            Field::new("key", DataType::Int32, false),
        )],
    );
    builder.finish_gpu_resident(projection)
}

fn dynamic_filter_route_plan() -> nexus_query_engine::QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![Field::new("left_key", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("right_key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Int32, false),
        Field::new("right_key", DataType::Int32, false),
    ]));
    let left_file = nexus_query_engine::LocalParquetFile::new("/tmp/dynamic-filter-left.parquet")
        .with_stable_id("dynamic-filter-left")
        .with_projected_row_group_bytes(vec![1024], vec![1024])
        .with_row_group_rows(vec![10]);
    let right_file = nexus_query_engine::LocalParquetFile::new("/tmp/dynamic-filter-right.parquet")
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
    let join = builder.add_join(
        left,
        right,
        nexus_query_engine::JoinSpec::new(
            nexus_query_engine::JoinType::Inner,
            vec![nexus_query_engine::JoinKey::new(
                nexus_query_engine::ColumnRef::input(0, "left_key", "left_key"),
                nexus_query_engine::ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![
            nexus_query_engine::JoinOutputColumn::left(
                nexus_query_engine::ColumnRef::input(0, "left_key", "left_key"),
                "left_key",
            ),
            nexus_query_engine::JoinOutputColumn::right(
                nexus_query_engine::ColumnRef::input(1, "right_key", "right_key"),
                "right_key",
            ),
        ])
        .with_build_side(nexus_query_engine::JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn semijoin_gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, semijoin_schema(fields)))
}

fn semijoin_aggregate_keyset_source(
    builder: &mut QueryPlanBuilder,
    name: &str,
    key_name: &str,
) -> nexus_query_engine::NodeId {
    let source = builder.add_source(semijoin_gpu_source(name, &[key_name]));
    builder.add_aggregate(
        source,
        nexus_query_engine::AggregateSpec::new(vec![nexus_query_engine::Expr::column(key_name)], Vec::new()),
        semijoin_schema(&[key_name]),
    )
}

fn semijoin_schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

fn adaptive_report_for_projection_test() -> AdaptivePipelineReport {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-adaptive-report-source-not-read.parquet",
        schema,
        vec!["key".to_string()],
    )));
    let plan = builder.finish_with_host_sink(source);
    let analysis = plan.capability_analysis();
    AdaptivePipelineReport::from_admission_parts(
        &analysis,
        &plan,
        AdaptiveCandidateStatus::Admitted,
        AdaptiveStrategy::ConservativeCompiledPipeline,
        nexus_query_engine::AdaptiveAdmissionRuntimeEvidence::default(),
        None,
    )
    .expect("adaptive projection report should be valid")
}

fn accepted_candidate_contract_entry(
    candidate: GpuCandidate,
    gate_id: ReservationGateId,
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    required_contract: &'static str,
    task_count: usize,
) -> NativeStrictBoundedAcceptedCandidateDiagnostics {
    let contract_scope = ReservationRequirementScope::Edge;
    let owner = if reservation_kind == ReservationKind::Materialization {
        StrictBoundedContractOwner::Compiled(ReservationGateOwner::SinkBoundary)
    } else {
        StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(
            nexus_query_engine::pipeline::PipelineEdgeId::new(gate_id.index()),
        ))
    };
    NativeStrictBoundedAcceptedCandidateDiagnostics {
        candidate,
        report: StrictBoundedContractReport::from_gate_rows(
            StrictBoundedExecutionMode::StrictBounded,
            vec![StrictBoundedContractGateRow {
                execution_mode: StrictBoundedExecutionMode::StrictBounded,
                contract_scope,
                gate_id: StrictBoundedContractGateId::Compiled(gate_id),
                owner,
                reservation_kind,
                memory_boundary,
                required_contract,
                contract_status: ContractStatus::Modeled,
                contract_quality: if reservation_kind == ReservationKind::Materialization {
                    ContractQuality::QueryEngineOwnedBoundary
                } else {
                    ContractQuality::ExactPreflight
                },
                unresolved_preflight_gap: None,
            }],
        )
        .unwrap(),
        compiled_graph: NativeStrictBoundedCompiledGraphDiagnostics {
            task_count,
            edge_count: 1,
            barrier_count: 0,
            resource_count: 1,
            gate_count: 1,
            sink_kind: "host_arrow_materialization".to_string(),
            sink_gate_count: usize::from(reservation_kind == ReservationKind::Materialization),
        },
    }
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
