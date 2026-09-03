use super::*;

pub(super) fn native_exec_leaf_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-execution-report-source-not-read.parquet",
        schema,
        vec!["key".to_string()],
    )));
    Arc::new(
        NexusNativeExec::try_new_for_test(builder.finish_with_host_sink(source), native_execution_config()).unwrap(),
    )
}

pub(super) fn composed_native_exec_for_report() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut producer_builder = QueryPlanBuilder::new();
    let producer_source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/native-composed-report-source-not-read.parquet",
        Arc::clone(&schema),
        vec!["key".to_string()],
    )));
    let producer = Arc::new(
        NexusNativeGpuProducerExec::try_new_for_test(
            "fragment_input",
            producer_builder.finish_gpu_resident(producer_source),
        )
        .unwrap(),
    );

    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_source = consumer_builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("fragment_input", Arc::clone(&schema))));
    Arc::new(
        NexusNativeComposedExec::try_new_for_test(
            consumer_builder.finish_with_host_sink(consumer_source),
            vec![producer],
            native_execution_config(),
        )
        .unwrap(),
    )
}

pub(super) fn native_execution_config() -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1)
}

pub(super) fn accepted_native_diagnostics() -> NativePlanningDiagnostics {
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

pub(super) fn accepted_bounded_join_native_diagnostics() -> NativePlanningDiagnostics {
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

pub(super) fn accepted_capability_contract_diagnostics() -> NativeCapabilityContractDiagnostics {
    let candidate =
        GpuCandidate::new("SortExec(fetch)(AggregateExec(HashJoinExec(LocalParquetScan,LocalParquetScan)))", 1);
    let rows = vec![
        bounded_gate_row(
            CapabilityContractGateId::Compiled(ReservationGateId::new(0)),
            CapabilityContractOwner::Compiled(ReservationGateOwner::Edge(
                nexus_query_engine::pipeline::PipelineEdgeId::new(0),
            )),
            ReservationRequirementScope::Edge,
            ReservationKind::BuildSide,
            "join_build_input",
            "query_engine_build_side_source_reservation",
            ContractQuality::ExactPreflight,
            None,
        ),
        bounded_gate_row(
            CapabilityContractGateId::Compiled(ReservationGateId::new(1)),
            CapabilityContractOwner::Compiled(ReservationGateOwner::SinkBoundary),
            ReservationRequirementScope::Edge,
            ReservationKind::Materialization,
            "host_arrow_materialization",
            "query_engine_explicit_sink_materialization_reservation",
            ContractQuality::QueryEngineOwnedBoundary,
            None,
        ),
    ];
    NativeCapabilityContractDiagnostics {
        accepted_candidates: vec![NativeCapabilityAcceptedCandidateDiagnostics {
            candidate,
            report: CapabilityContractReport::from_gate_rows(CapabilityEnvelopeStatus::Admitted, rows).unwrap(),
            compiled_graph: NativeCapabilityCompiledGraphDiagnostics {
                task_count: 2,
                edge_count: 1,
                barrier_count: 0,
                resource_count: 2,
                gate_count: 2,
                sink_kind: "host_arrow_materialization".to_string(),
                sink_gate_count: 1,
            },
        }],
        deferred_candidates: Vec::new(),
        rejected_candidates: Vec::new(),
        non_join_bounded_plan_count: 0,
        non_join_bounded_plan_reason: None,
    }
}

#[expect(clippy::too_many_arguments, reason = "inputs are distinct domain values that share no owner")]
fn bounded_gate_row(
    gate_id: CapabilityContractGateId,
    owner: CapabilityContractOwner,
    contract_scope: ReservationRequirementScope,
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    required_contract: &'static str,
    contract_quality: ContractQuality,
    unresolved_preflight_gap: Option<&'static str>,
) -> CapabilityContractGateRow {
    CapabilityContractGateRow {
        execution_mode: CapabilityEnvelopeStatus::Admitted,
        contract_scope,
        gate_id,
        owner,
        reservation_kind,
        memory_boundary,
        required_contract,
        contract_status: ContractStatus::Modeled,
        contract_quality,
        unresolved_preflight_gap,
    }
}

pub(super) fn native_execution_metrics_set() -> datafusion::physical_plan::metrics::MetricsSet {
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
        ("reservation_requested_bytes", 128),
        ("reservation_granted_bytes", 128),
        ("reservation_live_peak_bytes", 96),
        ("reservation_denied_count", 0),
        ("max_task_reservation_bytes", 64),
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
        ("retry_reservation_failure_count", 2),
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
        ("hash_join_cache_hits", 2),
        ("hash_join_cache_misses", 1),
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
        .with_new_label("class", "output_sized")
        .with_new_label("outcome", "admitted")
        .with_new_label("raw_input_edges", "none")
        .with_new_label("input_edges", "6")
        .with_new_label("vertex_upper_bound", "12")
        .with_new_label("vertex_estimate", "12")
        .with_new_label("vertex_estimate_source", "edge_endpoint_upper_bound")
        .with_new_label("vertex_width_bytes", "4")
        .with_new_label("weight_width_bytes", "8")
        .with_new_label("pair_rows", "none")
        .with_new_label("pair_vertex_width_bytes", "none")
        .with_new_label("pair_cardinality_source", "none")
        .with_new_label("pair_cardinality_admission_mode", "none")
        .with_new_label("has_edge_id", "false")
        .with_new_label("directed", "true")
        .with_new_label("renumber", "true")
        .with_new_label("construction_policy", "python_cugraph")
        .with_new_label("edge_normalize_inserted", "false")
        .with_new_label("calibration_source", "none")
        .with_new_label("seed_count", "none")
        .with_new_label("seed_mode", "none")
        .with_new_label("data_dependent", "false")
        .with_new_label("model_requested_bytes", "4096")
        .with_new_label("requested_bytes", "4096")
        .with_new_label("budget_bytes", "1048576")
        .with_new_label("graph_model_error_ratio", "2.000")
        .with_new_label("measured_peak_bytes", "2048")
        .with_new_label("reason", "none")
        .global_counter("native_graph_reservation_rows")
        .add(1);
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
        .with_new_label("reservation_requested_bytes", "120")
        .with_new_label("reservation_granted_bytes", "96")
        .with_new_label("reservation_released_bytes", "32")
        .with_new_label("reservation_live_peak_bytes", "64")
        .with_new_label("max_observed_peak_bytes", "72")
        .with_new_label("max_retained_bytes", "48")
        .global_counter("native_task_operator_rows")
        .add(1);
    MetricBuilder::new(&metrics)
        .with_type(MetricType::Summary)
        .with_new_label("compiled_task_id", "t7")
        .with_new_label("operator_kind", "source_read")
        .with_new_label("reservation_requested_bytes", "80")
        .with_new_label("reservation_granted_bytes", "64")
        .with_new_label("reservation_released_bytes", "48")
        .with_new_label("reservation_live_peak_bytes", "64")
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
        .with_new_label("memory_mode", "bounded")
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

pub(super) fn report_row<'a>(
    report: &'a NativeFallbackPressureReport,
    query_id: &str,
) -> &'a datafusion_nexus::native_report::NativeFallbackPressureRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

pub(super) fn planning_rows<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> Vec<&'a datafusion_nexus::native_report::NativeDataFusionPlanningRow> {
    report.rows.iter().filter(|row| row.query_id == query_id).collect()
}

pub(super) fn whole_plan_planning_row<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> &'a datafusion_nexus::native_report::NativeDataFusionPlanningRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

pub(super) fn planning_replacement_row<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> &'a datafusion_nexus::native_report::NativeDataFusionPlanningRow {
    let rows = report.rows.iter().filter(|row| row.query_id == query_id);
    // Replacement assertions use the last accepted fragment row when bounded
    // whole-plan admission fails closed and child native fragments still
    // replace part of the DataFusion plan. Whole-plan assertions must use
    // `whole_plan_planning_row` so those rejections remain visible.
    rows.clone()
        .rev()
        .find(|row| row.outcome == Some(CandidateOutcomeKind::Selected))
        .or_else(|| rows.into_iter().next())
        .expect("report row should exist")
}

pub(super) fn planning_query_count(report: &NativeDataFusionPlanningReport) -> usize {
    report.rows.iter().map(|row| row.query_id.as_str()).collect::<std::collections::BTreeSet<_>>().len()
}

pub(super) async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(direct_friendly_sql_session_config());
    let df = ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) fn typed_null_and_try_cast_reject_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let typed_null_comparison =
        binary(lit(ScalarValue::Int64(None)), Operator::Eq, lit(ScalarValue::Int64(None)), &scan_schema).unwrap();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int32));
    let try_cast_comparison =
        binary(try_cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let predicate = binary(typed_null_comparison, Operator::And, try_cast_comparison, &scan_schema).unwrap();
    aggregate_sort_limit_plan_with_filter(scan, predicate)
}

pub(super) fn aggregate_sort_limit_plan_with_filter(
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

pub(super) async fn native_datafusion_planning_report(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
) -> NativeDataFusionPlanningReport {
    native_datafusion_planning_report_with_session(query_id, path, sql, direct_friendly_sql_session_config()).await
}

pub(super) async fn native_datafusion_planning_report_with_session(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
    session_config: SessionConfig,
) -> NativeDataFusionPlanningReport {
    let optimizer_config = NexusNativeOptimizerConfig::default().with_native_execution_config(
        NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1),
    );
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    NativeDataFusionPlanningReport::from_retained_rule(query_id, plan, native_rule.as_ref())
}

pub(super) fn direct_friendly_sql_session_config() -> SessionConfig {
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

pub(super) fn count_only_sql_session_config() -> SessionConfig {
    let mut config = direct_friendly_sql_session_config().with_collect_statistics(false);
    config.options_mut().execution.collect_statistics = false;
    config
}

pub(super) fn write_numeric_parquet_file(path: &std::path::Path) {
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

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
