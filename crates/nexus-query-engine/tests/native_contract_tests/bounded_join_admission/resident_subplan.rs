#[test]
fn bounded_keyed_hash_join_resident_subplan_admission_uses_capability_evidence() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline(),
        analysis.admits_keyed_hash_join_resident_subplan_pipeline()
    );
    assert_eq!(analysis.keyed_hash_join_resident_subplan_admission, CapabilityDecision::Satisfied);
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds
            .contains(&EdgeContractKind::SharedProducer { identity: SharedProducerIdentity::SameNodeIdAfterCse })
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_rejects_structural_equivalence_without_same_node_id() {
    let plan = keyed_hash_join_resident_subplan_pipeline_plan(
        KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyed_hash_join_resident_subplan_admission else {
        panic!("structurally equivalent grouped aggregates must not satisfy resident-subplan admission");
    };
    assert_eq!(*reason, "resident_subplan_requires_same_node_id_after_cse");
    assert_eq!(facts.get("identity").map(String::as_str), Some("same_node_id_after_cse"));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_preserves_node_failure_locality() {
    let plan = keyed_hash_join_resident_subplan_pipeline_plan(
        KeyedHashJoinResidentSubplanPipelineVariant::UnsupportedRootResidualExpression,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline());
    let CapabilityDecision::Rejected { reason, facts } = &analysis.keyed_hash_join_resident_subplan_admission else {
        panic!("unsupported residual expression should reject keyed hash resident-subplan admission");
    };
    assert_eq!(*reason, "unsupported_join_residual_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("like"));
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_non_matching_variants_use_full_sort_envelope() {
    for variant in [
        KeyedHashJoinResidentSubplanPipelineVariant::SortBelowRoot,
        KeyedHashJoinResidentSubplanPipelineVariant::ScalarAggregateNotMax,
        KeyedHashJoinResidentSubplanPipelineVariant::GroupedAggregateNotSum,
    ] {
        let plan = keyed_hash_join_resident_subplan_pipeline_plan(variant);
        let analysis = plan.capability_analysis();

        assert!(!analysis.admits_keyed_hash_join_resident_subplan_pipeline(), "{variant:?}");
        assert!(analysis.admits_root_full_sort_join_aggregate_pipeline(), "{variant:?}");
        assert!(unsupported_facts(&plan).is_none(), "{variant:?}");
        assert!(modeled_facts(&plan).is_some(), "{variant:?}");
    }
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_pipeline_admission_stays_narrow() {
    for variant in [
        KeyedHashJoinResidentSubplanPipelineVariant::ResidualRootJoin,
        KeyedHashJoinResidentSubplanPipelineVariant::UnsupportedRootResidualExpression,
        KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate,
    ] {
        let plan = keyed_hash_join_resident_subplan_pipeline_plan(variant);

        assert!(!plan.capability_analysis().admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_root_full_sort_join_aggregate_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_keyed_hash_join_resident_subplan_pipeline(), "{variant:?}");
        let facts = unsupported_facts(&plan)
            .expect("unsupported keyed hash join resident subplan variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_keyed_hash_resident_subplan");
        assert!(facts.contains_key("bounded_join_rejection_reason"), "{variant:?}");
    }
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_accepts_single_output_post_root_filtering_join() {
    let plan =
        aggregate_over_join_subplan_topk_variant_plan(AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin);
    let analysis = plan.capability_analysis();

    assert!(
        analysis.admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
        "single-output keyed post-root filtering joins should not hide the aggregate-over-join TopK root:\n{plan}"
    );
    assert!(
        analysis
            .invariants
            .iter()
            .any(|invariant| matches!(invariant.kind, SubgraphInvariantKind::PostRootStageOrder { .. })
                && invariant.nodes.len() == 5),
        "post-root filtering join should be part of the post-root stage order evidence"
    );

    let pipeline = nexus_query_engine::pipeline::compile_pipeline(&plan, &analysis)
        .expect("post-root filtering join TopK plan should compile");
    assert_eq!(
        nexus_query_engine::admission::compiled_bounded_pipeline_support(&pipeline)
            .expect("graph support check should not fail"),
        nexus_query_engine::admission::CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        nexus_query_engine::admission::compiled_bounded_query_plan_support(&plan, &pipeline)
            .expect("plan-aware support check should not fail"),
        nexus_query_engine::admission::CompiledBoundedPipelineSupport::Supported
    );
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_preserves_node_failure_locality() {
    let plan = aggregate_over_join_subplan_topk_variant_plan(
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_multi_join_aggregate_over_join_subplan_topk_pipeline());
    let CapabilityDecision::Rejected { reason, facts } =
        &analysis.multi_join_aggregate_over_join_subplan_topk_admission
    else {
        panic!("unsupported projection expression should reject aggregate-over-join TopK admission");
    };
    assert_eq!(*reason, "unsupported_projection_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("case"));
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_capability_evidence_stays_narrow() {
    for variant in [
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit,
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan,
        AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide,
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    ] {
        let plan = aggregate_over_join_subplan_topk_variant_plan(variant);

        assert!(
            !plan.capability_analysis().admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
            "{variant:?} must not satisfy the aggregate-over-join TopK capability evidence"
        );
        assert!(
            unsupported_facts(&plan).is_some(),
            "{variant:?} should remain rejected with bounded unsupported-shape facts"
        );
        let facts = unsupported_facts(&plan).expect("rejected aggregate-over-join TopK variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_aggregate_over_join_topk");
        assert!(facts.contains_key("bounded_join_rejection_reason"), "{variant:?}");
    }
}

#[test]
fn bounded_multi_join_resident_grouped_aggregate_subquery_filter_admission_uses_capability_evidence() {
    let plan = resident_grouped_aggregate_subquery_filter_plan(false);
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(),
        analysis.admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline()
    );
    assert_eq!(analysis.multi_join_resident_grouped_aggregate_subquery_filter_admission, CapabilityDecision::Satisfied);
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_topk_preserves_node_failure_locality() {
    let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter,
    );
    let analysis = plan.capability_analysis();

    assert!(
        !plan
            .capability_analysis()
            .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline()
    );
    let CapabilityDecision::Rejected { reason, facts } =
        &analysis.hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_admission
    else {
        panic!("unsupported filter should reject resident grouped-aggregate TopK admission");
    };
    assert_eq!(*reason, "unsupported_filter_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("in_list"));
    assert_eq!(facts.get("reason").map(String::as_str), Some("in_list_must_not_be_empty"));
}

#[test]
fn bounded_hash_join_resident_grouped_aggregate_topk_admission_stays_narrow() {
    for variant in [
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::MissingFetchLimit,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::ScalarPostJoinAggregate,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter,
    ] {
        let plan = resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(variant);

        assert!(
            !plan
                .capability_analysis()
                .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(),
            "{variant:?} must not be admitted by the resident grouped-aggregate TopK family"
        );
    }
}
