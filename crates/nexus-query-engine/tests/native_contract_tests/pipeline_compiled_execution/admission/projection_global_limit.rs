#[test]
fn strict_admission_accepts_projection_root_global_limit_join_aggregate_with_modeled_gates() {
    let plan = projection_root_global_limit_join_aggregate_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.admits_projection_global_limit_pipeline());
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!(
            "expected projection/global-limit selected envelope, got {:?}",
            analysis.capability_envelope_admission()
        );
    };
    assert_eq!(envelope, CapabilityEnvelope::ProjectionGlobalLimit);
    assert_eq!(envelope.as_str(), "projection_global_limit");

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("projection/global-limit join aggregate should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::ProjectionGlobalLimit);

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

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
        .expect("compiled projection/global-limit report should be valid");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
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
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!(
            "expected projection/global-limit selected envelope, got {:?}",
            analysis.capability_envelope_admission()
        );
    };
    assert_eq!(envelope, CapabilityEnvelope::ProjectionGlobalLimit);

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("limit-root projection/global-limit sibling shape should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::ProjectionGlobalLimit);
    let pipeline = admission.pipeline();
    assert_task_kind_count(pipeline, PipelineTaskKind::JoinMaterialize, 4);
    assert_task_kind_count(pipeline, PipelineTaskKind::Aggregate, 2);
    assert_task_kind_count(pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), pipeline)
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
    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .unwrap_or_else(|| panic!("{label}: distinct/dedup aggregate state should be admitted"));
    assert_eq!(admission.envelope(), CapabilityEnvelope::ProjectionGlobalLimit, "{label}");

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
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
fn strict_admission_accepts_projection_root_global_limit_filtered_distinct_count() {
    let filtered = projection_root_global_limit_filtered_distinct_join_aggregate_plan();
    let filtered_analysis = filtered.capability_analysis();
    assert!(
        filtered_analysis.admits_projection_global_limit_pipeline(),
        "filtered count-distinct is modeled by masked inputs before exact NUnique/count-distinct state"
    );
    assert!(
        bounded_pipeline_admission(&filtered).expect("strict admission should not fail").is_some(),
        "filtered count-distinct should compile through projection/global-limit admission"
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
        bounded_pipeline_admission(&non_count).expect("strict admission should not fail").is_none(),
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
                analysis.capability_envelope_admission(),
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
            bounded_pipeline_admission(&build_plan()).expect("strict admission should not fail").is_none(),
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
