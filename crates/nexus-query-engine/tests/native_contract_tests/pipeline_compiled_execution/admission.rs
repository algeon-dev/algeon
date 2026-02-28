use super::*;

#[test]
fn compiled_bounded_support_accepts_root_join_graph_without_legacy_routing_terms() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("root_join"));

    let source_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("root join should retain source-read gates");
    assert_eq!(source_gate.status(), ContractStatus::Modeled);
    assert_eq!(source_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
}

#[test]
fn compiled_bounded_support_rejects_non_d1_task_envelope_before_execution() {
    let plan = aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate sort limit plan should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_accepts_single_join_post_root_d2_tasks_without_legacy_routing_terms() {
    let plan = single_join_post_root_aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert!(pipeline.barriers().iter().any(|barrier| barrier.kind() == PipelineBarrierKind::AggregateBoundary));
    assert!(pipeline.barriers().iter().any(|barrier| barrier.kind() == PipelineBarrierKind::SortBoundary));
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Aggregate)
        .expect("single-join post-root plan should have an aggregate task");
    let aggregate_gate = aggregate_task
        .gates()
        .iter()
        .find_map(|gate_id| pipeline.gates().iter().find(|gate| gate.id() == *gate_id))
        .expect("compiled aggregate task should retain its output reservation gate");
    assert_eq!(aggregate_gate.reservation_kind(), ReservationKind::OperatorOutput);
    assert_eq!(aggregate_gate.status(), ContractStatus::Modeled);
    assert_eq!(aggregate_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let final_sort_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::FinalSort)
        .expect("single-join post-root plan should retain final-sort gates");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_accepts_simple_multi_join_tree_without_legacy_routing_terms() {
    let plan = simple_multi_join_plan(
        Path::new("/tmp/compiled-d26-a.parquet"),
        Path::new("/tmp/compiled-d26-b.parquet"),
        Path::new("/tmp/compiled-d26-c.parquet"),
    );
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_closed_multi_join_subgraph());
    let pipeline = compile_pipeline(&plan, &analysis).expect("simple multi-join plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::SourceRead, 3);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 2);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_rejects_post_root_aggregate_without_output_gate_before_execution() {
    let plan = single_join_post_root_aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");
    let malformed = pipeline_without_reservation_gates(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_rejects_post_root_final_sort_without_gate_before_execution() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d24-final-sort-left.parquet"),
        Path::new("/tmp/compiled-d24-final-sort-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join final-sort plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let malformed = pipeline_without_reservation_gates(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_accepts_grouped_aggregate_join_build_input_without_legacy_routing_terms() {
    let plan = aggregate_join_input_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 1);

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Aggregate)
        .expect("aggregate join input plan should have an aggregate task");
    let join_build = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::JoinBuild)
        .expect("aggregate join input plan should have a join build task");
    assert!(pipeline.edges().iter().any(|edge| {
        edge.from() == aggregate_task.id()
            && edge.to() == join_build.id()
            && edge.kind() == PipelineEdgeKind::JoinBuildInput
    }));
    let aggregate_gate =
        runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::Aggregate, ReservationKind::OperatorOutput)
            .expect("compiled aggregate input task should retain its modeled output gate");
    assert_eq!(aggregate_gate.status(), ContractStatus::Modeled);
    assert_eq!(aggregate_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
    let aggregate_barrier = pipeline
        .barriers()
        .iter()
        .find(|barrier| {
            barrier.kind() == PipelineBarrierKind::AggregateBoundary
                && barrier.upstream_tasks() == [aggregate_task.id()]
        })
        .expect("aggregate input task should have an explicit aggregate boundary");
    assert!(aggregate_barrier.downstream_tasks().contains(&join_build.id()));

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_accepts_root_topk_join_aggregate_graph() {
    let plan = root_topk_join_aggregate_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("root TopK join/aggregate plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let final_sort_gate =
        runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::FinalSort, ReservationKind::FinalSort)
            .expect("root TopK plan should retain a final-sort gate");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn strict_admission_accepts_projection_root_global_limit_join_aggregate_with_modeled_gates() {
    let plan = projection_root_global_limit_join_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_projection_global_limit_pipeline());
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!(
            "expected projection/global-limit selected envelope, got {:?}",
            analysis.strict_bounded_envelope_admission()
        );
    };
    assert_eq!(envelope, StrictBoundedEnvelope::ProjectionGlobalLimit);
    assert_eq!(envelope.as_str(), "projection_global_limit");

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("projection/global-limit join aggregate should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ProjectionGlobalLimit);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::Unary, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled projection/global-limit report should be valid");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert_modeled_gate_row("projection/global-limit", &report, ReservationKind::OperatorOutput, "aggregate_contract");
    assert_modeled_gate_row(
        "projection/global-limit",
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_limit_bounded_projection_output_boundary",
    );
    assert_modeled_gate_row(
        "projection/global-limit",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn strict_admission_accepts_limit_root_projected_global_hash_join_aggregate_with_sibling_input() {
    let plan = limit_root_projected_global_hash_join_aggregate_with_sibling_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_projection_global_limit_pipeline(),
        "limit-root projection/global-limit sibling shape rejected with {:?}",
        analysis.projection_global_limit_admission
    );
    assert!(!analysis.admits_root_join(), "limit-root sibling shape must not admit through generic root-join");
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!(
            "expected projection/global-limit selected envelope, got {:?}",
            analysis.strict_bounded_envelope_admission()
        );
    };
    assert_eq!(envelope, StrictBoundedEnvelope::ProjectionGlobalLimit);

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("limit-root projection/global-limit sibling shape should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ProjectionGlobalLimit);
    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 4);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled limit-root projection/global-limit report should be valid");
    assert_modeled_gate_row(
        "limit-root projection/global-limit sibling",
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_global_limit_output_boundary",
    );
    assert_modeled_gate_row(
        "limit-root projection/global-limit sibling",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "limit-root projection/global-limit sibling",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn projection_scalar_setop_scalar_fanout_without_cardinality_proof_is_parked() {
    let plan = projection_root_scalar_fanout_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_projection_scalar_setop_pipeline());
    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_contract_not_implemented")
    );
    assert_projection_scalar_setop_contract_not_implemented_projection(&plan, &analysis, "scalar fanout");
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_scalar_fanout_with_one_row_driver() {
    assert_projection_scalar_setop_scalar_fanout_admission(
        projection_root_scalar_fanout_one_row_driver_plan(),
        "literal one-row driver",
        1,
        1,
        0,
        2,
    );
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_scalar_fanout_with_scan_limit_driver() {
    assert_projection_scalar_setop_scalar_fanout_admission(
        projection_root_scalar_fanout_scan_limit_driver_plan(),
        "scan-limit driver",
        0,
        2,
        0,
        2,
    );
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_scalar_fanout_with_max_rows_driver() {
    assert_projection_scalar_setop_scalar_fanout_admission(
        projection_root_scalar_fanout_max_rows_driver_plan(),
        "max-rows driver",
        0,
        2,
        0,
        2,
    );
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_scalar_fanout_with_limit_driver() {
    assert_projection_scalar_setop_scalar_fanout_admission(
        projection_root_scalar_fanout_limit_driver_plan(),
        "limit driver",
        0,
        2,
        1,
        2,
    );
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_repeated_scalar_fanout() {
    assert_projection_scalar_setop_scalar_fanout_admission(
        projection_root_repeated_scalar_fanout_plan(),
        "repeated scalar fanout",
        0,
        1,
        0,
        1,
    );
}

#[test]
fn projection_scalar_setop_repeated_scalar_fanout_with_unbounded_input_is_parked() {
    let plan = projection_root_repeated_scalar_fanout_unbounded_input_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_projection_scalar_setop_pipeline());
    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_contract_not_implemented")
    );
    assert_projection_scalar_setop_contract_not_implemented_projection(&plan, &analysis, "unbounded repeated fanout");
}

fn assert_projection_scalar_setop_scalar_fanout_admission(
    plan: QueryPlan,
    label: &str,
    expected_literal_sources: usize,
    expected_source_reads: usize,
    expected_limits: usize,
    expected_aggregates: usize,
) {
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_projection_scalar_setop_pipeline(),
        "{label}: scalar fanout should admit through projection-scalar/setop, got {:?}",
        analysis.strict_bounded_envelope_admission()
    );
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!(
            "{label}: expected projection-scalar/setop selected envelope, got {:?}",
            analysis.strict_bounded_envelope_admission()
        );
    };
    assert_eq!(envelope, StrictBoundedEnvelope::ProjectionScalarSetop);
    assert_eq!(envelope.as_str(), "projection_scalar_setop");

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .unwrap_or_else(|| panic!("{label}: scalar fanout should be admitted"));
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ProjectionScalarSetop);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::LiteralBatchSource, expected_literal_sources);
    assert_task_kind_count(pipeline, PipelineTaskKind::SourceRead, expected_source_reads);
    assert_task_kind_count(pipeline, PipelineTaskKind::Limit, expected_limits);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, expected_aggregates);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert!(
        pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinMaterialize).count() >= 1,
        "{label}: scalar fanout should retain join materialization tasks"
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .unwrap_or_else(|error| panic!("{label}: compiled projection-scalar/setop report should be valid: {error}"));
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert_modeled_gate_row(label, &report, ReservationKind::SourceRead, "source_contract");
    assert_modeled_gate_row(label, &report, ReservationKind::OperatorOutput, "aggregate_contract");
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_projection_scalar_setop_final_projection_output_boundary",
    );
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::JoinOutput,
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
    );
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_dedup_scalar_with_modeled_gates() {
    let plan = projection_root_setop_dedup_scalar_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_projection_scalar_setop_pipeline(),
        "setop/dedup scalar should admit through projection-scalar/setop, got {:?}",
        analysis.strict_bounded_envelope_admission()
    );
    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("setop/dedup scalar should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ProjectionScalarSetop);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::SourceRead, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 3);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled setop/dedup scalar report should be valid");
    assert_modeled_gate_row(
        "setop/dedup scalar",
        &report,
        ReservationKind::PartialState,
        "distinct_dedup_aggregate_state_materialization_contract",
    );
    assert_modeled_gate_row("setop/dedup scalar", &report, ReservationKind::OperatorOutput, "aggregate_contract");
    assert_modeled_gate_row(
        "setop/dedup scalar",
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_projection_scalar_setop_final_projection_output_boundary",
    );
    assert_modeled_gate_row(
        "setop/dedup scalar",
        &report,
        ReservationKind::JoinOutput,
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
    );
    assert_modeled_gate_row(
        "setop/dedup scalar",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn projection_scalar_setop_dedup_scalar_rejects_non_dedup_left_anti_input() {
    let plan = projection_root_setop_dedup_scalar_non_dedup_left_anti_input_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_projection_scalar_setop_pipeline());
    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_contract_not_implemented")
    );
    let projection = assert_projection_scalar_setop_contract_not_implemented_projection(
        &plan,
        &analysis,
        "non-dedup left anti input",
    );
    assert_eq!(
        projection.facts.get("shape_kind").map(String::as_str),
        Some("setop_dedup_scalar"),
        "setop/dedup scalar should require each left-anti input to be a group-only dedup aggregate"
    );
}

#[test]
fn projection_scalar_setop_distinct_admission_is_parked_until_cardinality_contract_lands() {
    let plan = projection_root_setop_distinct_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_projection_scalar_setop_pipeline());
    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_contract_not_implemented")
    );
    assert_projection_scalar_setop_contract_not_implemented_projection(&plan, &analysis, "distinct aggregate");
}

#[test]
fn projection_scalar_setop_setop_input_candidate_is_owned_before_root_join() {
    let plan = projection_root_setop_dedup_input_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_projection_scalar_setop_pipeline());
    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_contract_not_implemented")
    );
    let projection =
        assert_projection_scalar_setop_contract_not_implemented_projection(&plan, &analysis, "setop dedup input");
    assert_eq!(
        projection.facts.get("shape_kind").map(String::as_str),
        Some("setop_dedup_input"),
        "setop/dedup input should be owned locally before generic root-join admission"
    );
}

#[test]
fn projection_scalar_setop_keyless_input_candidate_is_owned_before_root_join() {
    let plan = projection_root_keyless_grouped_aggregate_input_plan();
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_projection_scalar_setop_pipeline());
    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_contract_not_implemented")
    );
    let projection =
        assert_projection_scalar_setop_contract_not_implemented_projection(&plan, &analysis, "keyless grouped input");
    assert_eq!(
        projection.facts.get("shape_kind").map(String::as_str),
        Some("keyless_or_cross_join_input"),
        "keyless/cross input should be owned locally before generic root-join admission"
    );
}

#[test]
fn strict_admission_accepts_projection_scalar_setop_scalar_join_aggregate_input_with_modeled_gates() {
    let plan = projection_root_scalar_join_aggregate_input_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_projection_scalar_setop_pipeline(),
        "scalar join aggregate should admit through projection-scalar/setop, got {:?}",
        analysis.strict_bounded_envelope_admission()
    );
    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("scalar join aggregate should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ProjectionScalarSetop);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::SourceRead, 3);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled scalar join aggregate report should be valid");
    assert_modeled_gate_row("scalar join aggregate", &report, ReservationKind::SourceRead, "source_contract");
    assert_modeled_gate_row("scalar join aggregate", &report, ReservationKind::OperatorOutput, "aggregate_contract");
    assert_modeled_gate_row(
        "scalar join aggregate",
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_projection_scalar_setop_final_projection_output_boundary",
    );
    assert_modeled_gate_row(
        "scalar join aggregate",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn projection_scalar_setop_does_not_claim_plain_keyed_grouped_aggregate_projection() {
    let plan = projection_root_keyed_grouped_aggregate_input_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        analysis.projection_scalar_setop_admission.rejection_reason(),
        Some("projection_scalar_setop_requires_scalar_or_setop_contract")
    );
    assert_ne!(
        analysis.first_rejection_projection().map(|projection| projection.owner),
        Some(nexus_query_engine::capability::CapabilityRejectionOwner::Admission("projection_scalar_setop_admission")),
        "plain keyed grouped aggregate projections must keep their existing non-PSS ownership"
    );
}

fn assert_projection_scalar_setop_contract_not_implemented_projection(
    plan: &QueryPlan,
    analysis: &nexus_query_engine::capability::CapabilityAnalysis,
    label: &str,
) -> nexus_query_engine::capability::CapabilityRejectionProjection {
    let projection = analysis.first_rejection_projection().unwrap_or_else(|| {
        panic!(
            "{label}: scalar/setop parked contract should project locally; admission={:?}",
            analysis.strict_bounded_envelope_admission()
        )
    });
    assert!(
        matches!(
            projection.owner,
            nexus_query_engine::capability::CapabilityRejectionOwner::Admission("projection_scalar_setop_admission")
        ),
        "{label}: projection={projection:?}"
    );
    assert_eq!(projection.memory_boundary, "admission_capability", "{label}");
    assert_eq!(projection.category.as_str(), "missing_subgraph_invariant", "{label}");
    assert_eq!(projection.reason, "projection_scalar_setop_contract_not_implemented", "{label}");
    assert_eq!(
        projection.facts.get("required_remediation").map(String::as_str),
        Some("projection_scalar_setop_cardinality_contract"),
        "{label}"
    );
    assert_eq!(
        analysis.projection_global_limit_admission.rejection_reason(),
        Some("projection_global_limit_requires_projection_input_limit"),
        "{label}: global-limit should recognize this is not its direct limit-projection shape"
    );
    assert!(
        matches!(
            analysis.strict_bounded_envelope_admission(),
            CapabilityEnvelopeAdmission::Rejected {
                projection: nexus_query_engine::capability::CapabilityRejectionProjection {
                    owner: nexus_query_engine::capability::CapabilityRejectionOwner::Admission(
                        "projection_scalar_setop_admission"
                    ),
                    ..
                }
            }
        ),
        "{label}: scalar/setop parked contract must be selected as a strict rejection"
    );
    assert!(
        strict_bounded_pipeline_admission(plan).expect("strict admission should not fail").is_none(),
        "scalar/setop placeholder must not compile through generic strict admission"
    );
    assert_ne!(
        projection.reason, "root_join_admission_requires_sink_root_join",
        "{label}: scalar/setop candidates must not fall through to generic root-join rejection"
    );
    projection
}

#[test]
fn strict_admission_accepts_projection_root_global_limit_distinct_and_dedup_aggregate_state() {
    assert_projection_global_limit_distinct_or_dedup_admission(
        projection_root_global_limit_distinct_join_aggregate_plan(),
        "nunique aggregate",
    );
    assert_projection_global_limit_distinct_or_dedup_admission(
        projection_root_global_limit_count_distinct_join_aggregate_plan(),
        "count distinct aggregate",
    );
    assert_projection_global_limit_distinct_or_dedup_admission(
        projection_root_global_limit_dedup_join_aggregate_plan(),
        "dedup aggregate",
    );
}

fn assert_projection_global_limit_distinct_or_dedup_admission(plan: QueryPlan, label: &str) {
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_projection_global_limit_pipeline(),
        "{label}: projection/global-limit admission rejected with {:?}",
        analysis.projection_global_limit_admission
    );
    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .unwrap_or_else(|| panic!("{label}: distinct/dedup aggregate state should be admitted"));
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ProjectionGlobalLimit, "{label}");

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .unwrap_or_else(|error| panic!("{label}: compiled report should be valid: {error}"));
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::PartialState,
        "distinct_dedup_aggregate_state_materialization_contract",
    );
    assert_modeled_gate_row(label, &report, ReservationKind::OperatorOutput, "aggregate_contract");
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_limit_bounded_projection_output_boundary",
    );
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn strict_admission_rejects_projection_root_global_limit_unsafe_distinct_variants() {
    let filtered = projection_root_global_limit_filtered_distinct_join_aggregate_plan();
    let filtered_analysis = filtered.capability_analysis();
    assert!(!filtered_analysis.admits_projection_global_limit_pipeline());
    assert_eq!(
        filtered_analysis.projection_global_limit_admission.rejection_reason(),
        Some("projection_global_limit_rejects_unsupported_distinct_or_dedup_aggregate")
    );
    let filtered_projection = filtered_analysis
        .first_rejection_projection()
        .expect("filtered distinct should project as projection/global-limit rejection");
    assert_eq!(filtered_projection.reason, "projection_global_limit_rejects_unsupported_distinct_or_dedup_aggregate");
    assert_eq!(
        filtered_projection.facts.get("aggregate_state_contract_rejection").map(String::as_str),
        Some("filtered_distinct")
    );
    assert!(
        strict_bounded_pipeline_admission(&filtered).expect("strict admission should not fail").is_none(),
        "filtered distinct must not compile through generic strict admission"
    );

    let non_count = projection_root_global_limit_non_count_distinct_join_aggregate_plan();
    let non_count_analysis = non_count.capability_analysis();
    assert!(!non_count_analysis.admits_projection_global_limit_pipeline());
    assert_eq!(
        non_count_analysis
            .nodes
            .iter()
            .find_map(|node| node.executable.rejection_facts())
            .and_then(|facts| facts.get("reason"))
            .map(String::as_str),
        Some("distinct_aggregate_requires_count")
    );
    assert!(
        strict_bounded_pipeline_admission(&non_count).expect("strict admission should not fail").is_none(),
        "non-count distinct must stay rejected by local aggregate expression capability"
    );
}

#[test]
fn projection_global_limit_recognizer_rejects_boundary_variants() {
    for (label, build_plan, expected_reason) in [
        (
            "offset-only limit",
            projection_root_no_fetch_limit_join_aggregate_plan as fn() -> QueryPlan,
            "projection_global_limit_requires_fetch_limit",
        ),
        (
            "no pre-limit aggregate",
            projection_root_global_limit_join_without_aggregate_plan as fn() -> QueryPlan,
            "projection_global_limit_requires_pre_limit_aggregate",
        ),
        (
            "no join fragment",
            projection_root_global_limit_aggregate_without_join_plan as fn() -> QueryPlan,
            "projection_global_limit_requires_join_fragment",
        ),
        (
            "sort/top-k overlap",
            projection_root_global_limit_over_sort_join_aggregate_plan as fn() -> QueryPlan,
            "projection_global_limit_requires_non_sort_limit_input",
        ),
        (
            "hidden sort/top-k overlap",
            projection_root_global_limit_over_filter_sort_join_aggregate_plan as fn() -> QueryPlan,
            "projection_global_limit_requires_non_sort_limit_input",
        ),
    ] {
        let analysis = build_plan().capability_analysis();

        assert!(!analysis.admits_projection_global_limit_pipeline(), "{label}");
        assert_eq!(analysis.projection_global_limit_admission.rejection_reason(), Some(expected_reason), "{label}");
        assert!(
            matches!(
                analysis.strict_bounded_envelope_admission(),
                CapabilityEnvelopeAdmission::Rejected {
                    projection: nexus_query_engine::capability::CapabilityRejectionProjection {
                        owner: nexus_query_engine::capability::CapabilityRejectionOwner::Admission(
                            "projection_global_limit_admission"
                        ),
                        ..
                    }
                }
            ),
            "{label}: projection/global-limit boundary miss must not fall through to generic strict admission"
        );
        assert!(
            strict_bounded_pipeline_admission(&build_plan()).expect("strict admission should not fail").is_none(),
            "{label}: projection/global-limit boundary miss must not compile through generic strict admission"
        );
    }
}

#[test]
fn compiled_support_rejects_post_limit_filter_on_projection_sink_path() {
    let plan = projection_root_filter_after_global_limit_join_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        analysis.projection_global_limit_admission.rejection_reason(),
        Some("projection_global_limit_requires_projection_input_limit")
    );

    let pipeline = compile_pipeline(&plan, &analysis).expect("post-limit filter pipeline should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph-only support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported,
        "compiled strict execution must not accept arbitrary post-limit unary nodes on the sink path"
    );
}

#[test]
fn strict_admission_rejects_root_full_sort_join_aggregate_with_local_envelope_reason() {
    for (label, plan, expected_subcontract, generic_single_join_overlap) in [(
        "unproven keyless/cross full-sort",
        root_full_sort_keyless_join_grouped_aggregate_plan(),
        "keyless_or_cross_join",
        false,
    )] {
        let analysis = plan.capability_analysis();

        assert!(!analysis.admits_root_join(), "{label}");
        assert!(!analysis.admits_root_topk_join_aggregate_pipeline(), "{label}");
        assert!(!analysis.admits_root_full_sort_join_aggregate_pipeline(), "{label}");
        assert_eq!(
            analysis.root_full_sort_join_aggregate_admission.rejection_reason(),
            Some("root_full_sort_join_aggregate_rejects_unsupported_subcontract"),
            "{label}"
        );
        if generic_single_join_overlap {
            assert_eq!(
                analysis.single_join_post_root_pipeline_admission,
                CapabilityDecision::Satisfied,
                "{label}: fixture must pin the overlap with generic single-join post-root admission"
            );
        }

        let projection = analysis.first_rejection_projection().unwrap_or_else(|| {
            panic!(
                "{label}: full-sort envelope should project locally; admission={:?}",
                analysis.strict_bounded_envelope_admission()
            )
        });
        assert!(
            matches!(
                projection.owner,
                nexus_query_engine::capability::CapabilityRejectionOwner::Admission(
                    "root_full_sort_join_aggregate_admission"
                )
            ),
            "{label}"
        );
        assert_eq!(projection.memory_boundary, "admission_capability", "{label}");
        assert_eq!(projection.category.as_str(), "missing_subgraph_invariant", "{label}");
        assert_eq!(projection.reason, "root_full_sort_join_aggregate_rejects_unsupported_subcontract", "{label}");
        assert_eq!(projection.facts.get("subcontract").map(String::as_str), Some(expected_subcontract), "{label}");
        assert_ne!(
            projection.reason, "root_join_admission_requires_sink_root_join",
            "{label}: root full-sort candidates must not fall through to the generic root-join rejection"
        );

        assert!(
            strict_bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
            "{label}: Phase 1 owns and rejects the full-sort shape before whole-native execution"
        );

        let report = StrictBoundedContractReport::from_capability_analysis(&analysis)
            .expect("rejected full-sort envelope should still produce a contract report");
        let blocking_row = report
            .gate_rows()
            .iter()
            .find(|row| row.contract_status == ContractStatus::RejectedByCapability)
            .expect("strict report should preserve the full-sort envelope rejection");
        assert_eq!(
            blocking_row.required_contract, "root_full_sort_join_aggregate_rejects_unsupported_subcontract",
            "{label}"
        );
        assert_eq!(blocking_row.memory_boundary, "admission_capability", "{label}");
    }
}

#[test]
fn strict_admission_rejects_statistical_root_full_sort_when_subcontract_is_unproven() {
    let plan = root_full_sort_keyless_join_statistical_grouped_aggregate_plan();
    let rendered = plan.to_string();
    assert!(rendered.contains("StddevSamp"), "fixture must exercise statistical aggregate:\n{rendered}");

    let analysis = plan.capability_analysis();
    assert_eq!(
        analysis.root_full_sort_join_aggregate_admission.rejection_reason(),
        Some("root_full_sort_join_aggregate_rejects_unsupported_subcontract")
    );
    assert_eq!(
        analysis.exact_statistical_join_final_full_sort_admission.rejection_reason(),
        Some("root_full_sort_join_aggregate_rejects_unsupported_subcontract")
    );

    let projection = analysis
        .first_rejection_projection()
        .expect("unproven statistical full-sort shape should project a local rejection");
    assert!(
        matches!(
            projection.owner,
            nexus_query_engine::capability::CapabilityRejectionOwner::Admission(
                "root_full_sort_join_aggregate_admission"
            )
        ),
        "root-full-sort subcontract rejection must own the failure before exact statistical admission: {projection:?}"
    );
    assert_eq!(projection.reason, "root_full_sort_join_aggregate_rejects_unsupported_subcontract");
    assert_ne!(
        analysis.strict_bounded_envelope_admission(),
        CapabilityEnvelopeAdmission::Admitted { envelope: StrictBoundedEnvelope::ExactStatisticalJoinFinalFullSort }
    );
    assert!(
        strict_bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
        "unproven root full-sort subcontract must not be admitted through the exact statistical envelope"
    );
}

#[test]
fn strict_admission_accepts_root_full_sort_shared_aggregate_fanout_with_modeled_gates() {
    let plan = root_full_sort_shared_aggregate_fanout_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_root_full_sort_join_aggregate_pipeline());
    assert!(analysis.invariants.iter().any(|invariant| {
        invariant.kind == SubgraphInvariantKind::SharedResidentSubplan
            && invariant.nodes.first().is_some_and(|producer| producer.to_string() == "n3")
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.strict_bounded_envelope_admission());
    };
    assert_eq!(envelope, StrictBoundedEnvelope::RootFullSortJoinAggregate);

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("shared aggregate fanout root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::RootFullSortJoinAggregate);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::SourceRead, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert_modeled_gate_row(
        "root full-sort shared aggregate fanout",
        &report,
        ReservationKind::SourceRead,
        "source_contract",
    );
    assert_modeled_gate_row(
        "root full-sort shared aggregate fanout",
        &report,
        ReservationKind::CapabilityContract,
        "root_full_sort_shared_aggregate_fanout_subcontract",
    );
    assert_modeled_gate_row(
        "root full-sort shared aggregate fanout",
        &report,
        ReservationKind::JoinOutput,
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
    );
    assert_modeled_gate_row(
        "root full-sort shared aggregate fanout",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "root full-sort shared aggregate fanout",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
    assert_modeled_gate_row(
        "root full-sort shared aggregate fanout",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn strict_admission_accepts_root_full_sort_keyed_join_aggregate_with_modeled_gates() {
    let plan = root_full_sort_single_keyed_join_grouped_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert_eq!(
        analysis.single_join_post_root_pipeline_admission,
        CapabilityDecision::Satisfied,
        "fixture must pin the overlap with generic single-join post-root admission"
    );
    assert!(analysis.admits_root_full_sort_join_aggregate_pipeline());
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.strict_bounded_envelope_admission());
    };
    assert_eq!(envelope, StrictBoundedEnvelope::RootFullSortJoinAggregate);
    assert_eq!(envelope.as_str(), "root_full_sort_join_aggregate");

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("keyed root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::RootFullSortJoinAggregate);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert_modeled_gate_row(
        "root full-sort keyed join",
        &report,
        ReservationKind::JoinOutput,
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
    );
    assert_modeled_gate_row(
        "root full-sort keyed join",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row("root full-sort keyed join", &report, ReservationKind::FinalSort, "final_sort_contract");
    assert_modeled_gate_row(
        "root full-sort keyed join",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn strict_admission_accepts_root_full_sort_keyless_scalar_projection_fanout_with_modeled_gates() {
    let plan = root_full_sort_keyless_scalar_projection_fanout_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_root_full_sort_join_aggregate_pipeline());
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.strict_bounded_envelope_admission());
    };
    assert_eq!(envelope, StrictBoundedEnvelope::RootFullSortJoinAggregate);

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("keyless scalar root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::RootFullSortJoinAggregate);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, admission.pipeline())
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert_modeled_gate_row(
        "root full-sort keyless scalar",
        &report,
        ReservationKind::CapabilityContract,
        "root_full_sort_keyless_or_cross_join_subcontract",
    );
    assert_modeled_gate_row(
        "root full-sort keyless scalar",
        &report,
        ReservationKind::JoinOutput,
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
    );
    assert_modeled_gate_row(
        "root full-sort keyless scalar",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "root full-sort keyless scalar",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
    assert_modeled_gate_row(
        "root full-sort keyless scalar",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn strict_admission_accepts_root_full_sort_repeated_projection_fanout_with_modeled_gates() {
    let plan = root_full_sort_repeated_projection_fanout_grouped_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_root_full_sort_join_aggregate_pipeline());
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.strict_bounded_envelope_admission());
    };
    assert_eq!(envelope, StrictBoundedEnvelope::RootFullSortJoinAggregate);

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("repeated projection fanout root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::RootFullSortJoinAggregate);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, admission.pipeline())
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert_modeled_gate_row(
        "root full-sort repeated projection fanout",
        &report,
        ReservationKind::CapabilityContract,
        "root_full_sort_repeated_projection_fanout_subcontract",
    );
    assert_modeled_gate_row(
        "root full-sort repeated projection fanout",
        &report,
        ReservationKind::JoinOutput,
        "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
    );
    assert_modeled_gate_row(
        "root full-sort repeated projection fanout",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "root full-sort repeated projection fanout",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
    assert_modeled_gate_row(
        "root full-sort repeated projection fanout",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn root_full_sort_recognizer_does_not_claim_topk_or_projection_roots() {
    let topk = root_topk_join_aggregate_plan();
    let topk_analysis = topk.capability_analysis();
    assert!(topk_analysis.admits_root_topk_join_aggregate_pipeline());
    assert_eq!(
        topk_analysis.root_full_sort_join_aggregate_admission.rejection_reason(),
        Some("root_full_sort_join_aggregate_requires_sink_full_sort")
    );
    assert!(
        topk_analysis
            .first_rejection_projection()
            .is_none_or(|projection| projection.reason != "root_full_sort_join_aggregate_requires_sink_full_sort"),
        "TopK roots must keep root TopK ownership"
    );

    let projection_root = projection_root_over_full_sort_join_aggregate_plan();
    let projection_root_analysis = projection_root.capability_analysis();
    assert_eq!(
        projection_root_analysis.root_full_sort_join_aggregate_admission.rejection_reason(),
        Some("root_full_sort_join_aggregate_requires_sink_full_sort")
    );
    assert!(
        projection_root_analysis.first_rejection_projection().is_none_or(|projection| {
            projection.reason != "root_full_sort_join_aggregate_requires_sink_full_sort"
                && projection.reason != "root_full_sort_join_aggregate_rejects_unsupported_subcontract"
        }),
        "projection roots must not be classified by the root full-sort recognizer"
    );
}

#[test]
fn compiled_bounded_support_accepts_root_topk_shared_resident_fanout_graph() {
    for variant in [
        RootTopKJoinAggregateVariant::SharedComputedFanout,
        RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout,
        RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        assert!(analysis.invariants.iter().any(|invariant| {
            matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
                && invariant.decision == CapabilityDecision::Satisfied
        }));
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK shared-fanout plan should compile: {error}"));

        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        let shared_edges = pipeline
            .edges()
            .iter()
            .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
            .collect::<Vec<_>>();
        assert!(!shared_edges.is_empty(), "{variant:?}: shared producer must remain explicit in the compiled graph");
        let shared_producer = shared_edges[0].from();
        assert_eq!(shared_edges[0].resources().len(), 1, "{variant:?}: shared edge should carry one resource");
        let shared_resource = shared_edges[0].resources()[0];
        assert!(
            shared_edges.iter().all(|edge| edge.from() == shared_producer && edge.resources() == [shared_resource]),
            "{variant:?}: shared edges must preserve one producer/resource identity"
        );
        assert_eq!(
            compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );

        let rendered = pipeline.debug_render();
        assert_no_legacy_execution_selection_terms("root TopK shared resident fanout", &rendered);
    }
}

#[test]
fn compiled_bounded_support_accepts_root_topk_distinct_and_keyless_subcontract_graphs() {
    for (variant, expected_contract) in [
        (RootTopKJoinAggregateVariant::DistinctAggregate, "root_topk_distinct_aggregate_subcontract"),
        (
            RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin,
            "root_topk_distinct_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin,
            "root_topk_distinct_aggregate_subcontract",
        ),
        (RootTopKJoinAggregateVariant::KeylessJoin, "root_topk_keyless_or_cross_join_subcontract"),
        (RootTopKJoinAggregateVariant::ProjectedCrossJoin, "root_topk_keyless_or_cross_join_subcontract"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK RTK-3 plan should compile: {error}"));

        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::CapabilityContract,
            expected_contract,
        );
        if matches!(variant, RootTopKJoinAggregateVariant::KeylessJoin) {
            assert_modeled_gate_row(
                "keyless root TopK",
                &report,
                ReservationKind::MixedSizeData,
                "mixed_keyed_residual_or_conditional_full_retained_size_data",
            );
            assert!(
                pipeline.gates().iter().any(|gate| {
                    gate.reservation_kind() == ReservationKind::GatherMap && gate.status() == ContractStatus::Modeled
                }),
                "keyless root TopK should retain a modeled gather-map gate"
            );
        }
        if matches!(variant, RootTopKJoinAggregateVariant::ProjectedCrossJoin) {
            assert!(
                pipeline.gates().iter().any(|gate| {
                    gate.reservation_kind() == ReservationKind::JoinOutput
                        && gate.required_contract() == "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                        && gate.status() == ContractStatus::Modeled
                        && gate.quality() == ContractQuality::ExactPreflight
                }),
                "projected-cross root TopK should retain an exact join-output bound"
            );
        }
    }
}

#[test]
fn compiled_bounded_support_accepts_root_topk_sort_fetch_hash_join_input_subcontract_graph() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchHashJoinInput);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline =
        compile_pipeline(&plan, &analysis).expect("sort-fetch hash-join-input root TopK plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(sink_input_edge_count(&pipeline), 1, "expected one explicit host sink path");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported,
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .expect("sort-fetch hash-join-input compiled report should be valid");
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::CapabilityContract,
        "root_topk_sort_fetch_hash_join_input_subcontract",
    );
    assert_modeled_gate_row("sort-fetch hash-join-input", &report, ReservationKind::SourceRead, "source_contract");
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_final_topk_output_boundary",
    );
    assert_modeled_gate_row("sort-fetch hash-join-input", &report, ReservationKind::FinalSort, "final_sort_contract");
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn compiled_bounded_support_accepts_root_topk_rtk9_subcontract_graphs() {
    for (variant, expected_contract) in [
        (
            RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput, "root_topk_keyed_filtered_semi_join_subcontract"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK plan should compile: {error}"));

        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}",
        );

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::CapabilityContract,
            expected_contract,
        );
        assert!(
            report.gate_rows().iter().any(|row| {
                row.reservation_kind == ReservationKind::SourceRead
                    && (row.required_contract == "source_contract"
                        || row.required_contract == "parquet_source_read_exact_preflight")
                    && row.contract_status == ContractStatus::Modeled
            }),
            "{variant:?}: expected modeled source-read gate row; rows={:?}",
            report.gate_rows()
        );
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::OperatorOutput,
            "aggregate_contract",
        );
        if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::HashTable,
                "hash_join_pre_build_reservation_size_bytes",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::FilteredProbeState,
                "filtered_join_retained_probe_state",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::GatherMap,
                "join_gather_map_reservation_size_bytes",
            );
        }
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::OperatorOutput,
            "query_engine_final_topk_output_boundary",
        );
        assert_modeled_gate_row(&format!("{variant:?}"), &report, ReservationKind::FinalSort, "final_sort_contract");
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::Materialization,
            "query_engine_explicit_sink_materialization_reservation",
        );
    }
}

#[test]
fn compiled_bounded_support_accepts_root_topk_rtk10_null_filtered_aggregate_fan_in_graph() {
    let plan =
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline =
        compile_pipeline(&plan, &analysis).expect("RTK-10 null-filtered aggregate fan-in plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 8);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 3);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(sink_input_edge_count(&pipeline), 1, "expected one explicit host sink path");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported,
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .expect("RTK-10 compiled report should be valid");
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::CapabilityContract,
        "root_topk_null_filtered_left_join_fan_in_subcontract",
    );
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn compiled_bounded_support_accepts_root_topk_window_subcontract_graphs() {
    for variant in [
        RootTopKJoinAggregateVariant::Window,
        RootTopKJoinAggregateVariant::BranchLocalWindow,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindow,
        RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin,
        RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK window plan should compile: {error}"));

        let expected_window_count = match variant {
            RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin => 2,
            RootTopKJoinAggregateVariant::Window
            | RootTopKJoinAggregateVariant::BranchLocalWindow
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection => 1,
            _ => unreachable!("window subcontract variants are enumerated above"),
        };
        let expected_sort_count = match variant {
            RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin => 3,
            RootTopKJoinAggregateVariant::Window
            | RootTopKJoinAggregateVariant::BranchLocalWindow
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection => 2,
            _ => unreachable!("window subcontract variants are enumerated above"),
        };
        assert_task_kind_count(&pipeline, PipelineTaskKind::Window, expected_window_count);
        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, expected_sort_count);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::CapabilityContract,
            "root_topk_window_subcontract",
        );
        if matches!(variant, RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::CapabilityContract,
                "root_topk_window_full_outer_join_subcontract",
            );
        }
        if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindow) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::CapabilityContract,
                "root_topk_keyless_outer_cross_window_subcontract",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::GatherMap,
                "join_gather_map_reservation_size_bytes",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::JoinOutput,
                "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
            );
        }
        assert_modeled_gate_row(&format!("{variant:?}"), &report, ReservationKind::OperatorOutput, "window_contract");
        if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::OperatorOutput,
                "query_engine_rank_filter_output_boundary",
            );
        }
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::OperatorOutput,
            "query_engine_final_topk_output_boundary",
        );
    }
}

#[test]
fn compiled_bounded_support_rejects_unmodeled_reservation_gates_before_execution() {
    let pipeline = root_join_shaped_pipeline_with_blocked_gate();

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_query_plan_support_accepts_keyless_root_join_strategy_without_legacy_routing_terms() {
    let plan = keyless_root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph shape should still match the structural envelope"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_no_legacy_execution_selection_terms("keyless root join strategy", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_query_plan_support_accepts_keyless_outer_cross_root_join_strategy() {
    let plan = keyless_outer_cross_root_join_plan();
    let analysis = plan.capability_analysis();
    assert!(
        analysis.admits_root_join(),
        "keyless outer-cross root join should be admitted; first_rejection={:?}",
        analysis.first_rejection_projection()
    );
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless outer-cross root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph shape should still match the structural envelope"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::GatherMap
                && gate.status() == ContractStatus::Modeled
                && gate.quality() == ContractQuality::ExactPreflight
        }),
        "keyless outer-cross support must retain a modeled gather-map gate for empty-side null extension"
    );
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::JoinOutput
                && gate.status() == ContractStatus::Modeled
                && gate.quality() == ContractQuality::ExactPreflight
        }),
        "keyless outer-cross support must retain a modeled join-output gate"
    );
    assert_no_legacy_execution_selection_terms("keyless outer-cross root join strategy", &pipeline.debug_render());
}

#[test]
fn strict_admission_accepts_rank_window_branches_feeding_native_union() {
    let plan = union_of_ranked_join_branches_plan();
    let analysis = plan.capability_analysis();
    assert!(
        analysis.union_pipeline_admission == CapabilityDecision::Satisfied,
        "union pipeline admission should accept branch-local windows; first_rejection={:?}",
        analysis.first_rejection_projection()
    );
    assert!(analysis.admits_strict_bounded_join());

    let pipeline = compile_pipeline(&plan, &analysis).expect("rank-window union plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Window, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Union, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 3);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("rank-window union plan should be strictly admitted");
    let rendered = admission.pipeline().debug_render();
    assert!(rendered.contains("kind=window"), "compiled graph must keep window tasks:\n{rendered}");
    assert!(rendered.contains("kind=union"), "compiled graph must keep native union:\n{rendered}");
    assert_no_legacy_execution_selection_terms("rank-window union plan", &rendered);
}

#[test]
fn strict_admission_rejects_large_full_scan_rollup_rank_topk_before_execution() {
    let plan = large_full_scan_rollup_rank_topk_plan();
    let rendered_plan = plan.to_string();
    assert!(
        rendered_plan.contains("grouping_set_count=4"),
        "native IR display must expose rollup/grouping-set metadata:\n{rendered_plan}"
    );
    assert!(
        rendered_plan.contains("grouping_id_alias=__grouping_id"),
        "native IR display must expose grouping id output metadata:\n{rendered_plan}"
    );
    assert!(
        rendered_plan.contains("grouping_sets=000|001|011|111"),
        "native IR display must expose grouping-set masks:\n{rendered_plan}"
    );

    let analysis = plan.capability_analysis();
    assert!(analysis.has_rejections(), "large full-scan rollup/rank/top-k should be rejected before runtime");
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("grouping_set_count").map(String::as_str), Some("4"));
    assert_eq!(rejection.facts.get("final_topk_fetch").map(String::as_str), Some("100"));
    assert_ne!(rejection.facts.get("rank_sort_id").map(String::as_str), Some("absent"));
    assert_ne!(rejection.facts.get("rank_filter_id").map(String::as_str), Some("absent"));
    assert_eq!(rejection.facts.get("large_source_selected_bytes").map(String::as_str), Some("2147483648"));
    assert_eq!(rejection.facts.get("large_source_selected_row_groups").map(String::as_str), Some("unknown"));
    assert_eq!(
        rejection.facts.get("required_remediation").map(String::as_str),
        Some("source_reduction_or_exact_blocking_stage_preflight")
    );

    assert!(
        strict_bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
        "rejected shape must not produce a strict pipeline for execution"
    );

    let report = StrictBoundedContractReport::from_capability_analysis(&analysis)
        .expect("rejected capability analysis should still produce a contract report");
    let blocking_row = report
        .gate_rows()
        .iter()
        .find(|row| row.required_contract == "blocking_rollup_rank_full_scan_requires_exact_contracts")
        .expect("contract report should preserve the structured rejection code");
    assert_eq!(blocking_row.execution_mode, StrictBoundedExecutionMode::UnsupportedStrictBounded);
    assert_eq!(blocking_row.contract_status, ContractStatus::RejectedByCapability);
    assert_eq!(blocking_row.contract_quality, ContractQuality::CapabilityProof);
}

#[test]
fn strict_admission_rejects_large_rollup_rank_native_fragment_without_outer_topk() {
    let plan = large_full_scan_rollup_rank_native_fragment_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.has_rejections(),
        "large rollup/rank native fragment should be rejected even when the outer TopK remains outside NativeCudfExec"
    );
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("final_topk_fetch").map(String::as_str), Some("unknown"));
    assert_ne!(rejection.facts.get("rank_sort_id").map(String::as_str), Some("absent"));
    assert_ne!(rejection.facts.get("rank_filter_id").map(String::as_str), Some("absent"));
    assert!(
        strict_bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
        "native fragment rejected by the guard must not produce a strict pipeline"
    );
}

#[test]
fn strict_admission_rejects_rollup_rank_topk_when_selected_bytes_are_still_large() {
    let plan = selected_large_rollup_rank_topk_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.has_rejections(), "selected row groups are not enough when selected bytes remain large");
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("large_source_selected_bytes").map(String::as_str), Some("1610612736"));
    assert_eq!(rejection.facts.get("large_source_selected_row_groups").map(String::as_str), Some("2"));
    assert_eq!(rejection.facts.get("large_source_total_row_groups").map(String::as_str), Some("4"));
}

#[test]
fn strict_admission_rejects_q86_class_rollup_rank_topk_selected_scan() {
    let plan = selected_q86_class_rollup_rank_topk_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.has_rejections(),
        "q86-class rollup/rank/top-k selected scans are fact-sized even below the old 1 GiB cutoff"
    );
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("large_source_selected_bytes").map(String::as_str), Some("939524096"));
    assert_eq!(rejection.facts.get("large_source_selected_row_groups").map(String::as_str), Some("448"));
}

#[test]
fn strict_admission_selects_q86_class_rollup_rank_topk_envelope_after_aggregate_join_pushdown() {
    let plan = selected_q86_class_rollup_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the q86-class shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized plan should contain a detail-level pre-aggregate"
    );
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized plan should retain the final grouping-set aggregate"
    );

    let analysis = optimized.plan.capability_analysis();
    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(rejection.is_none(), "source reduced by pre-aggregate should not trip the blocking-shape guard");
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "q86-class",
        ExpectedCompiledGraph {
            tasks: 13,
            edges: 13,
            barriers: 10,
            resources: 10,
            gates: 15,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: None,
        },
    );
}

#[test]
fn strict_admission_selects_q70_class_left_semi_rollup_rank_topk_envelope_after_aggregate_join_pushdown() {
    let plan = selected_q70_class_left_semi_rollup_rank_topk_plan();
    let raw_analysis = plan.capability_analysis();
    assert!(
        raw_analysis.has_rejections(),
        "raw q70-class left-semi rollup/rank/top-k selected scan should still hit the blocking guard"
    );
    assert_eq!(
        raw_analysis.first_rejection_projection().expect("shape should project a rejection").reason,
        "blocking_rollup_rank_full_scan_requires_exact_contracts"
    );

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the q70-class left-semi shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized q70-class plan should contain a detail-level pre-aggregate"
    );
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized q70-class plan should retain the final grouping-set aggregate"
    );

    let analysis = optimized.plan.capability_analysis();
    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(rejection.is_none(), "left-semi source reduced by pre-aggregate should not trip the guard");
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "q70-class left-semi",
        ExpectedCompiledGraph {
            tasks: 13,
            edges: 12,
            barriers: 10,
            resources: 10,
            gates: 16,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: None,
        },
    );
}

#[test]
fn strict_admission_selects_q67_class_rollup_rank_topk_projection_chain_envelope_after_aggregate_join_pushdown() {
    let plan = selected_q67_class_rollup_rank_topk_projection_chain_plan();

    // Sanity: before the rewrite the q67-class shape (projection chain over a
    // dimension-fact join with mixed group sides) trips the blocking guard.
    let raw_analysis = plan.capability_analysis();
    assert!(
        raw_analysis.has_rejections(),
        "raw q67-class projection-chain rollup/rank/top-k should still hit the blocking guard"
    );
    assert_eq!(
        raw_analysis.first_rejection_projection().expect("shape should project a rejection").reason,
        "blocking_rollup_rank_full_scan_requires_exact_contracts"
    );

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the q67-class projection-chain shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized q67-class plan should contain a detail-level pre-aggregate"
    );
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized q67-class plan should retain the final grouping-set aggregate"
    );

    // The pre-aggregate must read a plain column reference; the inlined
    // measure expression (the original projection-chain `COALESCE(amount *
    // quantity, 0)`) is materialized in a projection above the pre-aggregate's
    // input so the executor's plain-column aggregate-input contract holds.
    let preagg = optimized
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none() => Some(aggregate),
            _ => None,
        })
        .expect("optimized plan should contain a pre-aggregate");
    assert_eq!(preagg.spec.aggr_exprs.len(), 1);
    let preagg_aggr = &preagg.spec.aggr_exprs[0];
    assert_eq!(preagg_aggr.function, AggregateFunction::Sum);
    assert!(
        matches!(&preagg_aggr.expr, Expr::Column(_)),
        "pre-aggregate measure must be a plain column reference, got: {}",
        preagg_aggr.expr
    );

    let measure_projection = optimized.plan.node(preagg.input).expect("pre-aggregate input must exist");
    let PlanNode::Projection(measure_projection) = &measure_projection.kind else {
        panic!("pre-aggregate input should be the materializing projection, got: {:?}", measure_projection.kind);
    };
    let measure_output = measure_projection
        .outputs
        .iter()
        .find(|output| output.expr.to_string().contains("CASE WHEN"))
        .expect("materializing projection should carry the inlined CASE expression for the measure");
    assert!(
        measure_output.expr.to_string().contains("CASE WHEN"),
        "materializing projection should retain the inlined CASE expression, got: {}",
        measure_output.expr
    );

    let analysis = optimized.plan.capability_analysis();
    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(
        rejection.is_none(),
        "q67-class projection-chain source reduced by pre-aggregate should not trip the blocking-shape guard"
    );
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "q67-class projection-chain",
        ExpectedCompiledGraph {
            tasks: 14,
            edges: 14,
            barriers: 10,
            resources: 11,
            gates: 15,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: None,
        },
    );
}

#[test]
fn strict_admission_selects_wide_rollup_rank_topk_envelope_after_aggregate_join_pushdown() {
    let plan = selected_wide_q67_class_rollup_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the wide q67-class shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let final_rollup = optimized
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some() => Some(aggregate),
            _ => None,
        })
        .expect("optimized wide rollup plan should retain the final grouping-set aggregate");
    assert_eq!(final_rollup.spec.group_exprs.len(), 8);
    assert_eq!(final_rollup.spec.grouping_sets.as_ref().expect("grouping sets").groups.len(), 9);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized wide rollup plan should contain a detail-level pre-aggregate"
    );

    let analysis = optimized.plan.capability_analysis();
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "wide rollup/rank/top-k",
        ExpectedCompiledGraph {
            tasks: 13,
            edges: 13,
            barriers: 10,
            resources: 10,
            gates: 16,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: Some("rollup_rank_topk_wide_final_rollup_subcontract"),
        },
    );
}

#[test]
fn strict_admission_rejects_too_wide_rollup_rank_topk_without_explicit_width_contract() {
    let plan = selected_too_wide_rollup_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the too-wide rollup/rank/top-k shape");
    let analysis = optimized.plan.capability_analysis();
    assert_eq!(
        analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission.rejection_reason(),
        Some("rollup_rank_topk_after_pushdown_requires_bounded_final_rollup_width"),
        "a 9-key/10-set rollup must stay rejected until a separate width contract is modeled"
    );
    assert!(
        !analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "too-wide rollup must not be admitted through the q67-wide subcontract"
    );
}

#[test]
fn strict_admission_rejects_wide_non_rollup_grouping_sets_without_explicit_width_contract() {
    let plan = selected_wide_non_rollup_grouping_sets_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the wide non-rollup grouping-set shape");
    let analysis = optimized.plan.capability_analysis();
    assert_eq!(
        analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission.rejection_reason(),
        Some("rollup_rank_topk_after_pushdown_requires_bounded_final_rollup_width"),
        "an 8-key/9-set grouping aggregate must still prove the modeled suffix-rollup mask contract"
    );
    assert!(
        !analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "same-width non-rollup grouping sets must not be admitted through the q67-wide subcontract"
    );
}

#[test]
fn strict_admission_accepts_exact_statistical_join_final_full_sort_envelope() {
    let plan = statistical_join_final_full_sort_plan();
    let rendered = plan.to_string();
    assert!(rendered.contains("StddevSamp"), "fixture must exercise q39-class statistical aggregate:\n{rendered}");
    assert!(rendered.contains("Sort"), "fixture must retain the final full sort:\n{rendered}");

    let analysis = plan.capability_analysis();
    let statistical_invariant = analysis
        .invariants
        .iter()
        .find(|invariant| invariant.kind == SubgraphInvariantKind::StatisticalJoinFinalSortExactness)
        .expect("q39-class shape should carry a statistical/final-sort exactness invariant");
    assert_eq!(statistical_invariant.decision, CapabilityDecision::Satisfied);
    let exact_contracts = statistical_invariant
        .reservation_requirements
        .iter()
        .map(|requirement| {
            assert_eq!(requirement.kind, ReservationKind::CapabilityContract);
            assert_eq!(requirement.status, ContractStatus::Modeled);
            requirement.required_contract
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        exact_contracts,
        BTreeSet::from([
            "host_welford_input_materialization_preflight",
            "host_welford_output_upload_reservation",
            "final_full_sort_output_cardinality_bound",
        ])
    );
    assert!(analysis.first_rejection_projection().is_none(), "{:?}", analysis.first_rejection_projection());

    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("q39-class exact statistical final-sort shape should be admitted");
    assert_eq!(admission.envelope(), StrictBoundedEnvelope::ExactStatisticalJoinFinalFullSort);
    assert_eq!(
        compiled_bounded_pipeline_support(admission.pipeline()).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("accepted statistical final-sort capability analysis should produce a strict contract report");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    for required_contract in [
        "host_welford_input_materialization_preflight",
        "host_welford_output_upload_reservation",
        "final_full_sort_output_cardinality_bound",
        "final_sort_contract",
        "query_engine_explicit_sink_materialization_reservation",
    ] {
        assert!(
            report.gate_rows().iter().any(|row| {
                row.required_contract == required_contract && row.contract_status == ContractStatus::Modeled
            }),
            "missing compiled strict gate {required_contract}: {:?}",
            report.gate_rows()
        );
    }
}

#[test]
fn strict_admission_mode_boundary_accepts_tolerant_statistical_envelope_without_exact_host_gates() {
    let plan = statistical_join_final_full_sort_plan();
    let shape_admission = strict_bounded_pipeline_admission(&plan)
        .expect("shape-only strict admission should not fail")
        .expect("shape-only exact statistical envelope should be admitted");
    assert_eq!(shape_admission.envelope(), StrictBoundedEnvelope::ExactStatisticalJoinFinalFullSort);

    let tolerant_admission = strict_bounded_pipeline_admission_with_statistical_aggregate_mode(
        &plan,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
    .expect("mode-aware strict admission should not fail")
    .expect("tolerant statistical envelope should be admitted");
    assert_eq!(tolerant_admission.envelope(), StrictBoundedEnvelope::TolerantStatisticalJoinFinalFullSort);

    let projection = strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
        &plan,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
    .expect("mode-aware rejection projection should not fail");
    assert_eq!(projection, None);

    let report = StrictBoundedContractReport::from_compiled_pipeline(
        tolerant_admission.analysis(),
        tolerant_admission.pipeline(),
    )
    .expect("accepted tolerant statistical final-sort admission should produce a strict contract report");
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert!(report.gate_rows().iter().any(|row| {
        row.required_contract == "gpu_native_tolerant_statistical_aggregate_contract"
            && row.memory_boundary == "tolerant_statistical_join_final_full_sort"
            && row.contract_status == ContractStatus::Modeled
    }));
    assert!(report.gate_rows().iter().any(|row| {
        row.required_contract == "final_full_sort_output_cardinality_bound"
            && row.contract_status == ContractStatus::Modeled
    }));
    assert_eq!(
        report
            .gate_rows()
            .iter()
            .filter(|row| matches!(
                row.required_contract,
                "host_welford_input_materialization_preflight" | "host_welford_output_upload_reservation"
            ))
            .count(),
        0,
        "tolerant mode must not publish exact host-Welford gates"
    );
}

#[derive(Debug, Clone, Copy)]
struct ExpectedCompiledGraph {
    tasks: usize,
    edges: usize,
    barriers: usize,
    resources: usize,
    gates: usize,
    sink: SinkBoundaryKind,
    capability_contract: Option<&'static str>,
}

fn assert_rollup_rank_topk_after_pushdown_envelope(
    plan: &QueryPlan,
    analysis: &nexus_query_engine::capability::CapabilityAnalysis,
    label: &str,
    expected_graph: ExpectedCompiledGraph,
) {
    assert!(
        !analysis.has_rejections(),
        "{label}: optimized rollup/rank/top-k shape should have satisfied local capability facts; first_rejection={:?}",
        analysis.first_rejection_projection()
    );
    assert!(
        analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "{label}: expected rollup/rank/top-k post-pushdown envelope admission; rollup_decision={:?}; first_rejection={:?}",
        analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission,
        analysis.first_rejection_projection()
    );
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.strict_bounded_envelope_admission() else {
        panic!("{label}: expected selected strict envelope, got {:?}", analysis.strict_bounded_envelope_admission());
    };
    assert_eq!(envelope, StrictBoundedEnvelope::RollupRankTopKAfterAggregateJoinPushdown);
    assert_eq!(envelope.as_str(), "rollup_rank_topk_after_aggregate_join_pushdown");

    let report = StrictBoundedContractReport::from_capability_analysis(analysis)
        .unwrap_or_else(|error| panic!("{label}: capability contract report failed: {error}"));
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
    assert!(report.is_strict_bounded_ready(), "{label}: capability gate rows should be strict-ready");

    let aggregate_rows = report
        .gate_rows()
        .iter()
        .filter(|row| {
            row.reservation_kind == ReservationKind::OperatorOutput && row.required_contract == "aggregate_contract"
        })
        .count();
    assert!(
        aggregate_rows == 2,
        "{label}: detail and final grouping aggregates should both contribute modeled aggregate gate rows"
    );
    assert_modeled_gate_row(label, &report, ReservationKind::SourceRead, "source_contract");
    assert_modeled_gate_row(label, &report, ReservationKind::OperatorOutput, "window_contract");
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_rank_filter_output_boundary",
    );
    assert_modeled_gate_row(label, &report, ReservationKind::FinalSort, "final_sort_contract");
    assert_modeled_gate_row(label, &report, ReservationKind::OperatorOutput, "query_engine_final_topk_output_boundary");
    if let Some(required_contract) = expected_graph.capability_contract {
        assert_modeled_gate_row(label, &report, ReservationKind::CapabilityContract, required_contract);
    }
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );

    let pipeline = compile_pipeline(plan, analysis)
        .unwrap_or_else(|error| panic!("{label}: rollup/rank/top-k pipeline should compile: {error}"));
    pipeline
        .validate()
        .unwrap_or_else(|error| panic!("{label}: rollup/rank/top-k compiled pipeline should validate: {error}"));
    assert_compiled_graph_counts(label, &pipeline, expected_graph);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline)
            .unwrap_or_else(|error| panic!("{label}: graph support check failed: {error}")),
        CompiledBoundedPipelineSupport::Supported,
        "{label}: compiled graph should be inside the strict runtime envelope"
    );
    assert_eq!(
        compiled_bounded_query_plan_support(plan, &pipeline)
            .unwrap_or_else(|error| panic!("{label}: plan-aware support check failed: {error}")),
        CompiledBoundedPipelineSupport::Supported,
        "{label}: compiled plan should be inside the strict runtime envelope"
    );
    let runtime_admission = strict_bounded_pipeline_admission(plan)
        .unwrap_or_else(|error| panic!("{label}: strict bounded runtime admission failed: {error}"))
        .unwrap_or_else(|| panic!("{label}: strict bounded runtime admission should be selected"));
    assert_eq!(runtime_admission.envelope(), StrictBoundedEnvelope::RollupRankTopKAfterAggregateJoinPushdown);
    assert_compiled_graph_counts(label, runtime_admission.pipeline(), expected_graph);
}

fn assert_compiled_graph_counts(label: &str, pipeline: &CompiledPipeline, expected: ExpectedCompiledGraph) {
    assert_eq!(pipeline.tasks().len(), expected.tasks, "{label}: compiled task count changed");
    assert_eq!(pipeline.edges().len(), expected.edges, "{label}: compiled edge count changed");
    assert_eq!(pipeline.barriers().len(), expected.barriers, "{label}: compiled barrier count changed");
    assert_eq!(pipeline.resources().len(), expected.resources, "{label}: compiled resource count changed");
    assert_eq!(pipeline.gates().len(), expected.gates, "{label}: compiled gate count changed");
    assert_eq!(pipeline.sink().kind(), expected.sink, "{label}: compiled sink kind changed");
}

fn assert_modeled_gate_row(
    label: &str,
    report: &StrictBoundedContractReport,
    kind: ReservationKind,
    required_contract: &str,
) {
    assert!(
        report.gate_rows().iter().any(|row| {
            row.reservation_kind == kind
                && row.required_contract == required_contract
                && row.contract_status == ContractStatus::Modeled
        }),
        "{label}: expected modeled {kind:?} gate row for {required_contract}; rows={:?}",
        report.gate_rows()
    );
}

#[test]
fn strict_admission_does_not_apply_rollup_rank_topk_guard_when_selected_bytes_are_small() {
    let plan = selected_small_rollup_rank_topk_plan();
    let analysis = plan.capability_analysis();

    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(rejection.is_none(), "small selected source should not trip the large blocking-shape guard");
}

#[test]
fn compiled_bounded_support_records_d3_inventory_migration_boundaries_before_scheduler_execution() {
    for case in d3_1_inventory_cases() {
        let analysis = case.plan.capability_analysis();
        assert!(
            !analysis.has_rejections(),
            "{}: compiler should still receive locally satisfied capability facts",
            case.label
        );
        assert!((case.admits)(&analysis), "{}: expected D.3 target admission to be satisfied", case.label);
        let pipeline = compile_pipeline(&case.plan, &analysis)
            .unwrap_or_else(|error| panic!("{}: compile failed: {error}", case.label));
        pipeline
            .validate()
            .unwrap_or_else(|error| panic!("{}: compiled pipeline should validate: {error}", case.label));

        assert_eq!(
            compiled_bounded_pipeline_support(&pipeline).expect("graph-only support check should not fail"),
            case.graph_support,
            "{}: graph-only support status should match the current D.3 migration boundary",
            case.label
        );
        let plan_support = compiled_bounded_query_plan_support(&case.plan, &pipeline)
            .expect("plan-aware support check should not fail");
        assert_eq!(plan_support, case.plan_support, "{}: plan-aware compiled support status changed", case.label);

        assert_compiled_graph_inventory(case.label, &pipeline);
        let rendered = pipeline.debug_render();
        for expected in case.render_fragments {
            assert!(rendered.contains(expected), "{}: compiled graph rendering should contain {expected}", case.label);
        }
        assert_no_legacy_execution_selection_terms(case.label, &rendered);

        if plan_support == CompiledBoundedPipelineSupport::Unsupported {
            let executor = MockExecutor::new();
            assert!(
                executor.executed.is_empty(),
                "{}: unsupported compiled graph should stop at support checking and execute zero compiled tasks",
                case.label
            );
        }
    }
}

#[test]
fn d5_contract_gate_rows_are_inventory_backed_and_family_path_free() {
    let mut hard_budgeted_unresolved_rows = 0;
    let mut source_unresolved_rows = 0;
    let mut aggregate_unresolved_rows = 0;
    let mut final_sort_unresolved_rows = 0;
    let mut one_shot_output_unresolved_rows = 0;
    let mut filtered_retained_state_rows = 0;
    let mut mixed_retained_state_rows = 0;
    let mut observed_retained_state_rows = 0;

    for case in d3_1_inventory_cases() {
        let analysis = case.plan.capability_analysis();
        assert!((case.admits)(&analysis), "{}: expected D.5 inventory case to remain admitted", case.label);
        let pipeline = compile_pipeline(&case.plan, &analysis)
            .unwrap_or_else(|error| panic!("{}: compile failed: {error}", case.label));

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{}: strict bounded contract report failed: {error}", case.label));
        assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
        assert!(report.is_strict_bounded_ready(), "{}: compiled gate report should be strict-ready", case.label);
        assert_eq!(
            report.gate_rows().len(),
            pipeline.gates().len(),
            "{}: report should preserve every compiled reservation gate",
            case.label
        );

        let capability_report = StrictBoundedContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{}: capability contract report failed: {error}", case.label));
        assert_eq!(capability_report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
        assert!(capability_report.is_strict_bounded_ready());

        for row in report.gate_rows() {
            let rendered = row.to_stable_kv_row();
            assert_d5_contract_gate_row_is_family_path_free(case.label, &rendered);
            assert_d5_contract_gate_row_excludes_adaptive_terms(case.label, &rendered);
            assert!(rendered.contains("contract_status="), "{}: row must serialize contract status", case.label);
            assert!(rendered.contains("contract_quality="), "{}: row must serialize contract quality", case.label);
            assert!(
                !rendered.contains("contract_modes"),
                "{}: row must not serialize legacy contract modes",
                case.label
            );

            if row.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
                hard_budgeted_unresolved_rows += 1;
                match row.reservation_kind {
                    ReservationKind::SourceRead => source_unresolved_rows += 1,
                    ReservationKind::OperatorOutput => aggregate_unresolved_rows += 1,
                    ReservationKind::FinalSort => final_sort_unresolved_rows += 1,
                    ReservationKind::JoinOutput => one_shot_output_unresolved_rows += 1,
                    _ => {},
                }
                assert_eq!(
                    row.contract_status,
                    ContractStatus::Modeled,
                    "{}: hard-budgeted unresolved preflight rows stay modeled",
                    case.label
                );
                assert!(
                    row.unresolved_preflight_gap.is_some_and(|gap| gap.starts_with("libcudf_")),
                    "{}: hard-budgeted row should name its lower-level preflight gap",
                    case.label
                );
            }

            if matches!(row.reservation_kind, ReservationKind::FilteredProbeState | ReservationKind::MixedSizeData) {
                observed_retained_state_rows += 1;
                if row.reservation_kind == ReservationKind::FilteredProbeState {
                    filtered_retained_state_rows += 1;
                }
                if row.reservation_kind == ReservationKind::MixedSizeData {
                    mixed_retained_state_rows += 1;
                }
                assert_eq!(
                    row.contract_quality,
                    ContractQuality::ObservedRetainedState,
                    "{}: retained join state must not collapse into exact preflight",
                    case.label
                );
            }

            match row.reservation_kind {
                ReservationKind::HashTable if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert_eq!(
                        row.required_contract, "hash_join_pre_build_reservation_size_bytes",
                        "{}: hash-table exact preflight must name its cudf-rs pre-build sizing owner",
                        case.label
                    );
                },
                ReservationKind::GatherMap if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert_eq!(
                        row.required_contract, "join_gather_map_reservation_size_bytes",
                        "{}: gather-map exact preflight must name its cudf-rs gather-map sizing owner",
                        case.label
                    );
                },
                ReservationKind::JoinOutput if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert!(
                        matches!(
                            row.required_contract,
                            "dataframe_gather_output_reservation_size_bytes"
                                | "projected_cross_join_reservation_size_bytes"
                                | "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                        ),
                        "{}: join-output exact preflight must name an owned lower-level output sizing contract",
                        case.label
                    );
                },
                _ => {},
            }
        }
    }

    assert!(hard_budgeted_unresolved_rows > 0, "D.5 inventory should cover unresolved hard-budgeted gates");
    assert!(source_unresolved_rows > 0, "D.5 inventory should cover source unresolved preflight gates");
    assert!(aggregate_unresolved_rows > 0, "D.5 inventory should cover aggregate unresolved preflight gates");
    assert!(final_sort_unresolved_rows > 0, "D.5 inventory should cover final-sort unresolved preflight gates");
    assert!(
        one_shot_output_unresolved_rows > 0,
        "D.5 inventory should cover one-shot equality join output-size preflight gates"
    );
    assert!(filtered_retained_state_rows > 0, "D.5 inventory should cover filtered semi/anti retained state");
    assert!(mixed_retained_state_rows > 0, "D.5 inventory should cover mixed residual retained state");
    assert!(observed_retained_state_rows > 0, "D.5 inventory should cover observed retained-state gates");
}

#[test]
fn strict_preflight_gap_strict_preflight_gap_registry_tracks_current_retirement_protocol() {
    assert_eq!(
        STRICT_UNRESOLVED_PREFLIGHT_GAPS.len(),
        7,
        "strict unresolved preflight gap retirement must remove entries from this registry"
    );

    let mut markers = BTreeSet::new();
    let mut requirements = BTreeSet::new();
    for gap in STRICT_UNRESOLVED_PREFLIGHT_GAPS {
        assert!(
            markers.insert(gap.unresolved_gap),
            "unresolved preflight markers should be unique: {}",
            gap.unresolved_gap
        );
        assert!(
            requirements.insert(format!("{}:{}", gap.kind.as_str(), gap.required_contract)),
            "reservation-kind/contract pairs should be unique: {}:{}",
            gap.kind.as_str(),
            gap.required_contract
        );
        assert!(
            gap.retirement_requirement.contains("cudf-rs/libcudf"),
            "retirement must wait on an owning lower-level API: {}",
            gap.retirement_requirement
        );
        assert_eq!(lower_level_unresolved_preflight_gap(gap.kind, gap.required_contract), Some(gap.unresolved_gap));
    }

    assert!(markers.contains("libcudf_parquet_decoded_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"));
    assert!(markers.contains("libcudf_window_output_temporary_workspace_preflight_unavailable"));
    assert!(markers.contains("libcudf_regexp_replace_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_edge_normalize_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_final_sort_order_preflight_unavailable"));
    assert!(markers.contains("libcudf_equality_join_output_size_preflight_unavailable"));
}

#[test]
fn d5_contract_report_rejects_unresolved_preflight_gate_labeled_exact() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let mut gates = pipeline.gates().to_vec();
    let source_gate_index = gates
        .iter()
        .position(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("root join should have source-read gates");
    let source_gate = &gates[source_gate_index];
    gates[source_gate_index] = ReservationGate::new(
        source_gate.id(),
        source_gate.owner(),
        ReservationKind::SourceRead,
        ReservationRequirementScope::Edge,
        "source_read",
        "source_contract",
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );
    let malformed = CompiledPipeline::new(
        pipeline.tasks().to_vec(),
        pipeline.edges().to_vec(),
        pipeline.barriers().to_vec(),
        pipeline.resources().to_vec(),
        gates,
        pipeline.sink().clone(),
    );

    let error = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &malformed)
        .expect_err("D.5 report construction should reject mislabeled unresolved preflight");

    assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("contract_quality"), Some("exact_preflight"));
}

#[test]
fn strict_preflight_gap_contract_report_rejects_hard_budgeted_quality_without_tracked_gap() {
    let invalid = d5_test_gate_row(
        ReservationKind::SourceRead,
        "source_read",
        "libcudf_decoded_output_preflight_contract_after_api_lands",
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        None,
    );

    let error = StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
        .expect_err("F.6 should reject unresolved quality after the placeholder contract is replaced");

    assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(
        error.fact_value("required_contract"),
        Some("libcudf_decoded_output_preflight_contract_after_api_lands")
    );
}

#[test]
fn d5_contract_report_rejects_public_gate_rows_with_invalid_quality_accounting() {
    for invalid in [
        d5_test_gate_row(
            ReservationKind::SourceRead,
            "source_read",
            "source_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            "filtered_join_retained_probe_state",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::MixedSizeData,
            "join_mixed_size_data",
            "mixed_keyed_residual_or_conditional_full_retained_size_data",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "adapter_guessed_hash_table_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::GatherMap,
            "join_gather_map",
            "adapter_guessed_gather_map_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::JoinOutput,
            "join_output",
            "adapter_guessed_join_output_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
    ] {
        let kind = invalid.reservation_kind;
        let error =
            StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
                .unwrap_err();
        assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reservation_kind"), Some(kind.as_str()));
    }
}

#[test]
fn d5_contract_report_rejects_public_unresolved_preflight_rows_with_non_hard_budgeted_quality() {
    let unresolved_contracts = [
        (ReservationKind::SourceRead, "source_read", "source_contract"),
        (ReservationKind::OperatorOutput, "aggregate_output", "aggregate_contract"),
        (ReservationKind::FinalSort, "final_sort", "final_sort_contract"),
        (ReservationKind::JoinOutput, "join_output", "keyed_equality_join_output_size_preflight"),
    ];
    let non_hard_budgeted_qualities = [
        ContractQuality::ExactPreflight,
        ContractQuality::MetadataBound,
        ContractQuality::ObservedRetainedState,
        ContractQuality::QueryEngineOwnedOrdering,
        ContractQuality::QueryEngineOwnedBoundary,
        ContractQuality::CapabilityProof,
    ];

    for (reservation_kind, memory_boundary, required_contract) in unresolved_contracts {
        for contract_quality in non_hard_budgeted_qualities {
            let invalid =
                d5_test_gate_row(reservation_kind, memory_boundary, required_contract, contract_quality, None);
            let error =
                StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
                    .unwrap_err();
            assert_eq!(error.code(), nexus_query_engine::QueryErrorCode::InvalidPlan);
            assert_eq!(error.fact_value("reservation_kind"), Some(reservation_kind.as_str()));
            assert_eq!(error.fact_value("required_contract"), Some(required_contract));
            assert_eq!(error.fact_value("contract_quality"), Some(contract_quality.as_str()));
        }
    }
}

#[test]
fn d5_contract_report_rejects_public_rows_with_inconsistent_execution_modes_and_scope_owners() {
    let strict_with_blocked = StrictBoundedContractGateRow {
        contract_status: ContractStatus::BlockedByLowerLevel,
        ..d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
            None,
        )
    };
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::StrictBounded,
        vec![strict_with_blocked],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("report_execution_mode"), Some("strict_bounded"));

    let unsupported_with_modeled = d5_test_gate_row(
        ReservationKind::HashTable,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        ContractQuality::ExactPreflight,
        None,
    );
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::UnsupportedStrictBounded,
        vec![unsupported_with_modeled],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("report_execution_mode"), Some("unsupported_strict_bounded"));

    let edge_owned_node_scope = StrictBoundedContractGateRow {
        owner: StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(PipelineEdgeId::new(0))),
        ..d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
            None,
        )
    };
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::StrictBounded,
        vec![edge_owned_node_scope],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("contract_scope"), Some("node"));
    assert_eq!(error.fact_value("owner"), Some("edge:e0"));
}

#[test]
fn compiled_bounded_support_accepts_keyed_hash_resident_subplan_shared_producer_without_role_routing_terms() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_keyed_hash_join_resident_subplan_pipeline());
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyed hash resident subplan should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
        .collect::<Vec<_>>();
    assert_eq!(shared_edges.len(), 2, "shared producer should fan out to both downstream branches");
    let shared_resource = shared_edges[0].resources()[0];
    assert!(shared_edges.iter().all(|edge| edge.resources() == [shared_resource]));
    let shared_producer = pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == shared_edges[0].from())
        .expect("shared producer task should be present");
    assert_eq!(shared_producer.kind(), PipelineTaskKind::Aggregate);
    assert_no_legacy_execution_selection_terms("keyed hash resident subplan", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_support_accepts_shared_grouped_aggregate_fanout_without_scalar_derivation() {
    let plan = shared_grouped_aggregate_fanout_to_join_and_grouped_aggregate_plan();
    let analysis = plan.capability_analysis();
    assert!(!analysis.has_rejections(), "expected satisfied local capability facts");
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(!analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared grouped aggregate fanout should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
        .collect::<Vec<_>>();
    assert!(!shared_edges.is_empty(), "shared producer should feed a downstream compiled branch");
    let shared_producer = pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == shared_edges[0].from())
        .expect("shared producer task should be present");
    assert!(matches!(shared_producer.kind(), PipelineTaskKind::Unary | PipelineTaskKind::Aggregate));
    assert_no_legacy_execution_selection_terms("shared grouped aggregate fanout", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_support_accepts_d3_6_nonlinear_topk_and_subquery_filter_paths() {
    for (label, plan, admits, expected_aggregate_barriers) in [
        (
            "aggregate-over-join-subplan TopK",
            aggregate_over_join_subplan_topk_plan(),
            nexus_query_engine::capability::CapabilityAnalysis::admits_multi_join_aggregate_over_join_subplan_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            1,
        ),
        (
            "resident grouped-aggregate TopK",
            resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            nexus_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate subquery filter",
            resident_grouped_aggregate_subquery_filter_plan(),
            nexus_query_engine::capability::CapabilityAnalysis::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            1,
        ),
    ] {
        let analysis = plan.capability_analysis();
        assert!(!analysis.has_rejections(), "{label}: expected satisfied local capability facts");
        assert!(admits(&analysis), "{label}: expected bounded admission to be satisfied");
        let pipeline = compile_pipeline(&plan, &analysis).unwrap_or_else(|error| panic!("{label}: compile failed: {error}"));

        assert_eq!(
            compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: graph-only compiled support should admit the D.3.6 path"
        );
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: plan-aware compiled support should admit the D.3.6 path"
        );
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{label}: expected one explicit host sink path");
        assert!(
            task_kind_count(&pipeline, PipelineTaskKind::JoinBuild) > 1
                || task_kind_count(&pipeline, PipelineTaskKind::Aggregate) > 1,
            "{label}: expected a non-linear join or aggregate DAG"
        );
        assert!(
            barrier_kind_count(&pipeline, PipelineBarrierKind::AggregateBoundary) >= expected_aggregate_barriers,
            "{label}: expected aggregate barriers to stay explicit"
        );
        let final_sort_gate =
            runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::FinalSort, ReservationKind::FinalSort)
                .expect("D.3.6 path should retain its final-sort reservation gate");
        assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
        assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
        assert_no_legacy_execution_selection_terms(label, &pipeline.debug_render());
    }
}

#[test]
fn compiled_bounded_support_rejects_multiple_explicit_sink_candidates() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d36-multiple-sink-left.parquet"),
        Path::new("/tmp/compiled-d36-multiple-sink-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join TopK plan should compile");
    let malformed = pipeline_with_extra_sink_candidate(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_rejects_post_root_data_fanout() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d36-post-root-fanout-left.parquet"),
        Path::new("/tmp/compiled-d36-post-root-fanout-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join TopK plan should compile");
    let malformed = pipeline_with_post_root_fanout(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}
