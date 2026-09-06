fn assert_rejection_reason_is_family_free(reason: &str, label: &str) {
    for token in ["envelope", "_shape", "family", "topology", "recognizer", "catalogue", "whitelist"] {
        assert!(
            !reason.contains(token),
            "{label}: rejection reason {reason:?} must name the operator or edge contract, not a plan family"
        );
    }
}

#[test]
fn bounded_preflight_rejection_reports_caller_visible_sink() {
    // A rejected join plan reports the caller-visible sink as a diagnostic
    // fact only; the fact never selects a second decision.
    let (plan, _) = compositional_edge_normalize_plan();
    let error = nexus_query_engine::exec::preflight_bounded_query_plan(
        &plan,
        nexus_query_engine::mode::StatisticalAggregateExecutionMode::default(),
    )
    .expect_err("edge normalization must not preflight");
    assert_eq!(error.fact_value("caller_visible_sink"), Some("host_arrow"));
}

fn admitted_artifact_for(plan: &nexus_query_engine::plan::QueryPlan) -> nexus_query_engine::admission::BoundedPipelineAdmission {
    bounded_pipeline_admission(plan)
        .expect("artifact fixture should not fail admission")
        .expect("artifact fixture should be admitted")
}

#[test]
fn bounded_artifact_binding_matches_decided_plan_and_mode() {
    // The artifact binds the exact plan, mode, and sink it was decided for;
    // execution and reporting read this same binding.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let plan = builder.finish_with_host_sink(join);
    let mode = nexus_query_engine::mode::StatisticalAggregateExecutionMode::default();

    let admission = admitted_artifact_for(&plan);
    assert!(admission.matches(&plan, mode));
    assert_eq!(admission.sink_kind(), nexus_query_engine::pipeline::SinkBoundaryKind::HostArrowMaterialization);
    assert_eq!(admission.statistical_aggregate_execution_mode(), mode);
    assert!(admission.scope().matches(&plan));
}

#[test]
fn bounded_artifact_binding_rejects_mismatched_plan_and_mode() {
    // A plan, mode, or sink the artifact does not bind to is a caller error
    // with a typed reason, never a silent re-decision.
    let mut builder = QueryPlanBuilder::new();
    let (join, _) = compositional_inner_join_base(&mut builder);
    let plan = builder.finish_with_host_sink(join);
    let mode = nexus_query_engine::mode::StatisticalAggregateExecutionMode::default();
    let admission = admitted_artifact_for(&plan);

    let aliased = compositional_alias_join_plan();
    assert!(!admission.matches(&aliased, mode));
    let projection =
        admission.binding_rejection(&aliased, mode).expect("mismatched plan must name its binding clause");
    assert_eq!(projection.reason, "compiled_pipeline_sink_output_plan_node_mismatch");
    assert_rejection_reason_is_family_free(projection.reason, "artifact binding rejection");

    let other_mode = nexus_query_engine::mode::StatisticalAggregateExecutionMode::DataFusionExactHostWelford;
    assert!(mode != other_mode, "binding test needs two distinct execution modes");
    assert!(!admission.matches(&plan, other_mode));
    let projection =
        admission.binding_rejection(&plan, other_mode).expect("mismatched mode must name its binding clause");
    assert_eq!(projection.reason, "compiled_pipeline_artifact_mode_mismatch");
}

fn assert_rejection_facts_are_family_free(facts: &std::collections::BTreeMap<&'static str, String>, label: &str) {
    for retired_key in [
        "bounded_join_execution_family",
        "bounded_join_blocked_join_family",
        "bounded_join_specific_blocker",
        "bounded_join_shape_blocker",
        "bounded_join_lower_level_blockers",
        "bounded_join_required_shape_contract",
        "bounded_join_required_lower_level_contract",
    ] {
        assert_eq!(facts.get(retired_key), None, "{label}: retired rejection fact {retired_key} must not be emitted");
    }
    for (key, value) in facts {
        assert!(
            !value.contains("topology_unsupported")
                && !value.contains("BoundedJoinExecutionFamily")
                && !value.contains("BoundedJoinExecutionPath")
                && !key.ends_with("_pipeline_capability"),
            "{label}: {key}={value:?} must not carry retired family/path vocabulary"
        );
    }
}
