#[test]
fn bounded_keyed_hash_join_resident_subplan_admission_uses_capability_evidence() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert_eq!(
        plan.capability_analysis(rapids_analysis_context()).admits_keyed_hash_join_resident_subplan_pipeline(),
        analysis.admits_keyed_hash_join_resident_subplan_pipeline()
    );
    assert_eq!(analysis.keyed_hash_join_resident_subplan_admission, CapabilityDecision::Satisfied);
    assert!(analysis.edges.iter().any(|edge| edge.contract_kinds.contains(&EdgeContractKind::SharedProducer)));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    // Sort acyclicity is no longer a plan invariant: the compiled schedule's
    // barrier dominance check owns it, so no FinalBarrierOrder evidence is
    // produced here.
    assert!(
        !analysis
            .invariants
            .iter()
            .any(|invariant| { matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder) })
    );
}

#[test]
fn bounded_keyed_hash_join_resident_subplan_rejects_structural_equivalence_without_same_node_id() {
    let plan = keyed_hash_join_resident_subplan_pipeline_plan(
        KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate,
    );
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_keyed_hash_join_resident_subplan_pipeline());
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
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_keyed_hash_join_resident_subplan_pipeline());
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
        let analysis = plan.capability_analysis(rapids_analysis_context());

        assert!(!analysis.admits_keyed_hash_join_resident_subplan_pipeline(), "{variant:?}");
        assert!(analysis.admits_root_full_sort_join_aggregate_pipeline(), "{variant:?}");
        assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none(), "{variant:?}");
        assert!(modeled_facts(&plan, rapids_analysis_context()).is_some(), "{variant:?}");
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

        assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_full_sort_join_aggregate_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis(rapids_analysis_context()).admits_keyed_hash_join_resident_subplan_pipeline(), "{variant:?}");
        // S3 admits the structurally equivalent aggregate through the compiled
        // contracts: without shared identity it simply compiles to two
        // separate aggregate tasks, so only the residual variants below stay
        // rejected.
        if matches!(
            variant,
            KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate
        ) {
            assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none(), "{variant:?}");
            continue;
        }
        let facts = unsupported_facts(&plan, rapids_analysis_context())
            .expect("unsupported keyed hash join resident subplan variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_keyed_hash_resident_subplan");
        assert!(facts.contains_key("bounded_join_rejection_reason"), "{variant:?}");
    }
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_accepts_single_output_post_root_filtering_join() {
    let plan =
        aggregate_over_join_subplan_topk_variant_plan(AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin);
    let analysis = plan.capability_analysis(rapids_analysis_context());

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

    let pipeline = compile_pipeline(&plan, &analysis)
        .expect("post-root filtering join TopK plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline)
            .expect("graph support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_preserves_node_failure_locality() {
    let plan = aggregate_over_join_subplan_topk_variant_plan(
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    );
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_multi_join_aggregate_over_join_subplan_topk_pipeline());
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
    // S3 admits these variants through the compiled task contracts once they
    // compile; what stays narrow is the aggregate-over-join TopK envelope
    // itself.
    for variant in [
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit,
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan,
    ] {
        let plan = aggregate_over_join_subplan_topk_variant_plan(variant);

        assert!(
            !plan.capability_analysis(rapids_analysis_context()).admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
            "{variant:?} must not satisfy the aggregate-over-join TopK capability evidence"
        );
        assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none(), "{variant:?}");
    }

    let plan = aggregate_over_join_subplan_topk_variant_plan(
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    );

    assert!(
        !plan.capability_analysis(rapids_analysis_context()).admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
        "unsupported projection expression must not satisfy the aggregate-over-join TopK capability evidence"
    );
    assert!(
        unsupported_facts(&plan, rapids_analysis_context()).is_some(),
        "unsupported projection expression should remain rejected with bounded unsupported-shape facts"
    );
    let facts = unsupported_facts(&plan, rapids_analysis_context()).expect("rejected aggregate-over-join TopK variant should produce facts");
    assert_rejection_facts_are_family_free(&facts, "unsupported_aggregate_over_join_topk");
    assert!(facts.contains_key("bounded_join_rejection_reason"));

    // A side too shallow for the multi-join proof stays outside this envelope,
    // but the shape itself (keyed joins, a grouped aggregate over a join subplan,
    // projection, sort, fetch) is executable and is admitted by the root TopK
    // join/aggregate envelope; capacity is the allocator's, enforced as a typed
    // cap failure at runtime, not an admission gate.
    let plan = aggregate_over_join_subplan_topk_variant_plan(AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide);
    assert!(
        !plan.capability_analysis(rapids_analysis_context()).admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
        "ShallowMultiJoinSide must not satisfy the aggregate-over-join TopK capability evidence"
    );
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("ShallowMultiJoinSide should be admitted through the root TopK join/aggregate envelope");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootTopKJoinAggregate);
    assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none(), "ShallowMultiJoinSide is admitted, not a bounded support gap");
}

#[test]
fn bounded_multi_join_resident_grouped_aggregate_subquery_filter_admission_uses_capability_evidence() {
    let plan = resident_grouped_aggregate_subquery_filter_plan(false);
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert_eq!(
        plan.capability_analysis(rapids_analysis_context()).admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(),
        analysis.admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline()
    );
    assert_eq!(analysis.multi_join_resident_grouped_aggregate_subquery_filter_admission, CapabilityDecision::Satisfied);
    // Sort acyclicity is no longer a plan invariant: the compiled schedule's
    // barrier dominance check owns it, so no FinalBarrierOrder evidence is
    // produced here.
    assert!(
        !analysis
            .invariants
            .iter()
            .any(|invariant| { matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder) })
    );
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
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert!(
        !plan
            .capability_analysis(rapids_analysis_context())
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
                .capability_analysis(rapids_analysis_context())
                .admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(),
            "{variant:?} must not be admitted by the resident grouped-aggregate TopK family"
        );
    }
}

fn root_full_sort_shared_keyed_join_fanout_plan(
    shared_join_type: JoinType,
    interpose_shared_sort: bool,
) -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d1", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("d1", DataType::Int32, false)]));
    let shared_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let branch_dim_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let branch_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
    ]));
    let root_join_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("amount_sum", DataType::Int64, true),
        Field::new("amount", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-join-fact.parquet",
        fact_schema,
        vec!["k".into(), "d1".into(), "amount".into()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-join-dim.parquet",
        dim_schema,
        vec!["d1".into()],
    )));
    // The deduplicated join fragment: two downstream keyed joins below consume this
    // single node, making it the shared producer under test.
    let shared_join_schema_for_sort = Arc::clone(&shared_join_schema);
    let shared_join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            shared_join_type,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        shared_join_schema,
    );
    // With the sort interposed it becomes the two-consumer frontier node, moving the
    // shared producer out of every admitted class while leaving the joins untouched.
    let shared_producer = if interpose_shared_sort {
        builder.add_sort(
            shared_join,
            SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
            Arc::clone(&shared_join_schema_for_sort),
        )
    } else {
        shared_join
    };
    let dim_a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-join-dim-a.parquet",
        Arc::clone(&branch_dim_schema),
        vec!["k".into()],
    )));
    let join_a = builder.add_join(
        shared_producer,
        dim_a,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_join_schema),
    );
    let dim_b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-join-dim-b.parquet",
        branch_dim_schema,
        vec!["k".into()],
    )));
    let join_b = builder.add_join(
        shared_producer,
        dim_b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        branch_join_schema,
    );
    let aggregate_a = builder.add_aggregate(
        join_a,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
        ),
        aggregate_schema,
    );
    let root_join = builder.add_join(
        aggregate_a,
        join_b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount_sum", "amount_sum"), "amount_sum"),
            JoinOutputColumn::right(ColumnRef::input(1, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&root_join_schema),
    );
    let sort = builder.add_sort(
        root_join,
        SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]),
        root_join_schema,
    );
    builder.finish_with_host_sink(sort)
}

#[test]
fn bounded_root_full_sort_join_aggregate_admits_shared_keyed_join_fanout() {
    // Pins that a shared keyed-hash join producer consumed by two downstream
    // keyed joins is admitted through the SharedResidentFanout subcontract
    // instead of being rejected as an unsupported shared producer.
    let plan = root_full_sort_shared_keyed_join_fanout_plan(JoinType::Inner, false);
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert_eq!(analysis.root_full_sort_join_aggregate_admission, CapabilityDecision::Satisfied);
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("shared keyed-join fanout should be admitted through the root full-sort envelope");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootFullSortJoinAggregate);
}

#[test]
fn bounded_root_full_sort_join_aggregate_admits_shared_keyed_semi_join_fanout() {
    // A keyed filtered semi join is a member of the admitted fragment and its output is
    // an ordinary resident frame, so sharing it is admitted on the same contract as a
    // keyed-hash producer. TPC-DS q64's shared producer is exactly this shape.
    let plan = root_full_sort_shared_keyed_join_fanout_plan(JoinType::LeftSemi, false);
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert_eq!(analysis.root_full_sort_join_aggregate_admission, CapabilityDecision::Satisfied);
}

#[test]
fn bounded_root_full_sort_join_aggregate_rejects_shared_producer_outside_every_admitted_class() {
    // Pins that widening the shared-producer classes did not blanket-allow fanout: a
    // shared producer whose node kind has no admitted class (here a Sort) must still be
    // rejected as an unsupported shared producer.
    let plan = root_full_sort_shared_keyed_join_fanout_plan(JoinType::Inner, true);
    let analysis = plan.capability_analysis(rapids_analysis_context());

    let CapabilityDecision::Rejected { reason, facts } = &analysis.root_full_sort_join_aggregate_admission else {
        panic!("a shared sort producer must reject root full-sort join/aggregate admission");
    };
    assert_eq!(*reason, "root_full_sort_join_aggregate_rejects_unsupported_subcontract");
    assert_eq!(facts.get("subcontract").map(String::as_str), Some("unsupported_shared_or_fanout"));
}
