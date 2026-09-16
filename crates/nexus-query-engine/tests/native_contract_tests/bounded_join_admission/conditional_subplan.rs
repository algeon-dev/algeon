#[test]
fn bounded_keyless_conditional_join_input_subplan_pipeline_admits_variants_through_composed_path() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);

    assert!(!plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_root_join());
    assert!(plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_keyless_conditional_join_input_subplan_pipeline());
    assert!(!plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(unsupported_facts(&plan, nexus_query_engine::rapids_backend::rapids_analysis_context()).is_none());

    // S3 admits these variants through the compiled task contracts once they
    // compile; what stays narrow is the keyless input-subplan envelope itself.
    for variant in [
        KeylessInputSubplanVariant::GroupedAggregate,
        KeylessInputSubplanVariant::SortInput,
        KeylessInputSubplanVariant::LimitInput,
        KeylessInputSubplanVariant::BareLeftAntiInput,
        KeylessInputSubplanVariant::InnerJoinInput,
        KeylessInputSubplanVariant::NestedJoinInput,
    ] {
        let plan = keyless_conditional_join_input_subplan_plan(variant);

        assert!(!plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(unsupported_facts(&plan, nexus_query_engine::rapids_backend::rapids_analysis_context()).is_none(), "{variant:?}");
    }
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_admission_uses_capability_evidence() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);
    let analysis = plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context());

    assert_eq!(
        plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_keyless_conditional_join_input_subplan_pipeline(),
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
    let analysis = plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context());

    assert!(!plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).admits_keyless_conditional_join_input_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyless_conditional_join_input_subplan_admission
    else {
        panic!("inner join input should reject keyless input-subplan admission");
    };
    assert_eq!(*reason, "keyless_conditional_input_subplan_requires_projection_wrapped_left_anti_join");
    assert!(facts.contains_key("node_id"));
}
