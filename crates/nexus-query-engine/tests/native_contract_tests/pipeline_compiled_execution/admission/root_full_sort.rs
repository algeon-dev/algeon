#[test]
fn strict_admission_rejects_root_full_sort_join_aggregate_with_local_envelope_reason() {
    let label = "unproven keyless/cross full-sort";
    let plan = root_full_sort_keyless_join_grouped_aggregate_plan();
    let expected_subcontract = "keyless_or_cross_join";
    let analysis = plan.capability_analysis();

    assert!(!analysis.admits_root_join(), "{label}");
    assert!(!analysis.admits_root_topk_join_aggregate_pipeline(), "{label}");
    assert!(!analysis.admits_root_full_sort_join_aggregate_pipeline(), "{label}");
    assert_eq!(
        analysis.root_full_sort_join_aggregate_admission.rejection_reason(),
        Some("root_full_sort_join_aggregate_rejects_unsupported_subcontract"),
        "{label}"
    );

    let projection = analysis.first_rejection_projection().unwrap_or_else(|| {
        panic!(
            "{label}: full-sort envelope should project locally; admission={:?}",
            analysis.capability_envelope_admission()
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
        bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
        "{label}: Phase 1 owns and rejects the full-sort shape before whole-native execution"
    );

    let report = CapabilityContractReport::from_capability_analysis(&analysis)
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
        analysis.capability_envelope_admission(),
        CapabilityEnvelopeAdmission::Admitted { envelope: CapabilityEnvelope::ExactStatisticalJoinFinalFullSort }
    );
    assert!(
        bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
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
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.capability_envelope_admission());
    };
    assert_eq!(envelope, CapabilityEnvelope::RootFullSortJoinAggregate);

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("shared aggregate fanout root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootFullSortJoinAggregate);

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

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
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
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.capability_envelope_admission());
    };
    assert_eq!(envelope, CapabilityEnvelope::RootFullSortJoinAggregate);
    assert_eq!(envelope.as_str(), "root_full_sort_join_aggregate");

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("keyed root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootFullSortJoinAggregate);

    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
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
fn strict_admission_accepts_root_full_sort_keyed_grouped_aggregate_below_orderless_window() {
    let plan = root_full_sort_keyed_join_grouped_aggregate_with_orderless_window_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_root_full_sort_join_aggregate_pipeline(),
        "sink-path grouped aggregate below an orderless whole-partition window should be admitted: {:?}",
        analysis.root_full_sort_join_aggregate_admission
    );
    assert_eq!(
        analysis.capability_envelope_admission(),
        CapabilityEnvelopeAdmission::Admitted { envelope: CapabilityEnvelope::RootFullSortJoinAggregate }
    );

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("q98-shaped root full-sort join/aggregate should be admitted");
    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::FinalSort, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::Window, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled q98-shaped root full-sort report should be valid");
    assert_modeled_gate_row(
        "root full-sort keyed grouped aggregate below window",
        &report,
        ReservationKind::OperatorOutput,
        "window_contract",
    );
    assert_modeled_gate_row(
        "root full-sort keyed grouped aggregate below window",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "root full-sort keyed grouped aggregate below window",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
}

#[test]
fn strict_admission_accepts_root_full_sort_keyless_scalar_projection_fanout_with_modeled_gates() {
    let plan = root_full_sort_keyless_scalar_projection_fanout_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_root_full_sort_join_aggregate_pipeline());
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.capability_envelope_admission());
    };
    assert_eq!(envelope, CapabilityEnvelope::RootFullSortJoinAggregate);

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("keyless scalar root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootFullSortJoinAggregate);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, admission.pipeline())
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
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
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!("expected root full-sort selected envelope, got {:?}", analysis.capability_envelope_admission());
    };
    assert_eq!(envelope, CapabilityEnvelope::RootFullSortJoinAggregate);

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("repeated projection fanout root full-sort join/aggregate should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootFullSortJoinAggregate);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, admission.pipeline())
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("compiled root full-sort report should be valid");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
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
