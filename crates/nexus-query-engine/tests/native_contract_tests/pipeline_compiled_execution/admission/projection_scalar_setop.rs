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
