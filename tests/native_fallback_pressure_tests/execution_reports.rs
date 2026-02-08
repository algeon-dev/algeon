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
    assert_eq!(row.metrics.adaptive_retry_count, 10);
    assert_eq!(row.metrics.adaptive_spill_count, 11);
    assert_eq!(row.metrics.adaptive_downgrade_count, 12);
    assert_eq!(row.metrics.adaptive_memory_history_lookup_count, 13);
    assert_eq!(row.metrics.adaptive_memory_history_hit_count, 8);
    assert_eq!(row.metrics.adaptive_memory_history_miss_count, 5);
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

    let text = report.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, 14);
    assert_eq!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.split('\t').count(), 83);
    assert!(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER.contains("\tsource_cache_hits\tsource_cache_misses\tsource_cache_inserts\tsource_cache_insert_skipped\tsource_cache_retained_bytes\t"));
    assert!(text.starts_with("# native_datafusion_execution_tsv_schema_version=14\n"));
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
        cudf_query_engine::AdaptiveAdmissionRuntimeEvidence::default(),
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
            cudf_query_engine::pipeline::PipelineEdgeId::new(gate_id.index()),
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
