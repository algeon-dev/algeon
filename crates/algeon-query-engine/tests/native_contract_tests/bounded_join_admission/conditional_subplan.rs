#[test]
fn bounded_keyless_conditional_join_input_subplan_pipeline_admits_variants_through_composed_path() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);

    assert!(unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context(PlanningPolicy::default())).is_none());

    // S3 admits these variants through the compiled task contracts once they compile.
    for variant in [
        KeylessInputSubplanVariant::GroupedAggregate,
        KeylessInputSubplanVariant::SortInput,
        KeylessInputSubplanVariant::LimitInput,
        KeylessInputSubplanVariant::BareLeftAntiInput,
        KeylessInputSubplanVariant::InnerJoinInput,
        KeylessInputSubplanVariant::NestedJoinInput,
    ] {
        let plan = keyless_conditional_join_input_subplan_plan(variant);

        assert!(unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context(PlanningPolicy::default())).is_none(), "{variant:?}");
    }
}

#[test]
fn bounded_keyless_conditional_join_input_subplan_admission_uses_capability_evidence() {
    let plan = keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}
