#[test]
fn bounded_keyless_conditional_join_input_subplan_pipeline_admits_only_keyless_input_subplan_bounded_shapes() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);

    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none());

    for variant in [
        KeylessInputSubplanVariant::GroupedAggregate,
        KeylessInputSubplanVariant::SortInput,
        KeylessInputSubplanVariant::LimitInput,
        KeylessInputSubplanVariant::BareLeftAntiInput,
        KeylessInputSubplanVariant::InnerJoinInput,
        KeylessInputSubplanVariant::NestedJoinInput,
    ] {
        let plan = keyless_conditional_join_input_subplan_plan(variant);

        assert!(!plan.capability_analysis().admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph(), "{variant:?}");
        let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
            .expect("unsupported keyless input-subplan subplan shape should produce facts");
        assert_eq!(
            facts.get("bounded_join_keyless_conditional_join_input_subplan_pipeline_capability").map(String::as_str),
            Some("keyless_conditional_join_input_subplan_pipeline")
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_admission_uses_capability_proof() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline(),
        analysis.admits_keyless_conditional_join_input_subplan_pipeline()
    );
    assert_eq!(analysis.keyless_conditional_join_input_subplan_admission, CapabilityDecision::Satisfied);
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_rejects_non_input_subplan_shape() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::InnerJoinInput);
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyless_conditional_join_input_subplan_admission
    else {
        panic!("inner join input should reject keyless input-subplan admission");
    };
    assert_eq!(*reason, "keyless_conditional_input_subplan_requires_projection_wrapped_left_anti_join");
    assert!(facts.contains_key("node_id"));
}
