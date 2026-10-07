use super::*;

#[test]
fn bounded_keyed_hash_join_resident_subplan_admission_uses_capability_evidence() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

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
fn bounded_keyed_hash_join_resident_subplan_non_matching_variants_are_admitted() {
    for variant in [
        KeyedHashJoinResidentSubplanPipelineVariant::SortBelowRoot,
        KeyedHashJoinResidentSubplanPipelineVariant::ScalarAggregateNotMax,
        KeyedHashJoinResidentSubplanPipelineVariant::GroupedAggregateNotSum,
    ] {
        let plan = keyed_hash_join_resident_subplan_pipeline_plan(variant);

        assert!(
            unsupported_facts(
                &plan,
                StatisticalAggregateExecutionMode::default(),
                rapids_analysis_context(PlanningPolicy::default())
            )
            .is_none(),
            "{variant:?}"
        );
        assert!(
            modeled_facts(
                &plan,
                StatisticalAggregateExecutionMode::default(),
                rapids_analysis_context(PlanningPolicy::default())
            )
            .is_some(),
            "{variant:?}"
        );
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

        // S3 admits the structurally equivalent aggregate through the compiled
        // contracts: without shared identity it simply compiles to two
        // separate aggregate tasks, so only the residual variants below stay
        // rejected.
        if matches!(variant, KeyedHashJoinResidentSubplanPipelineVariant::StructurallyEquivalentGroupedAggregate) {
            assert!(
                unsupported_facts(
                    &plan,
                    StatisticalAggregateExecutionMode::default(),
                    rapids_analysis_context(PlanningPolicy::default())
                )
                .is_none(),
                "{variant:?}"
            );
            continue;
        }
        let facts = unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default()),
        )
        .expect("unsupported keyed hash join resident subplan variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_keyed_hash_resident_subplan");
        assert!(facts.contains_key("bounded_join_rejection_reason"), "{variant:?}");
    }
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_accepts_single_output_post_root_filtering_join() {
    let plan =
        aggregate_over_join_subplan_topk_variant_plan(AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "single-output keyed post-root filtering joins should not hide the aggregate-over-join TopK root:\n{plan}"
    );

    let pipeline = compile_pipeline(&plan, &analysis).expect("post-root filtering join TopK plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
}

#[test]
fn bounded_multi_join_aggregate_over_join_subplan_topk_variants_follow_compiled_contract_admission() {
    // S3 admits these variants through the compiled task contracts once they
    // compile.
    for variant in [
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit,
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan,
    ] {
        let plan = aggregate_over_join_subplan_topk_variant_plan(variant);

        assert!(
            unsupported_facts(
                &plan,
                StatisticalAggregateExecutionMode::default(),
                rapids_analysis_context(PlanningPolicy::default())
            )
            .is_none(),
            "{variant:?}"
        );
    }

    let plan = aggregate_over_join_subplan_topk_variant_plan(
        AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression,
    );

    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_some(),
        "unsupported projection expression should remain rejected with bounded unsupported-shape facts"
    );
    let facts = unsupported_facts(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("rejected aggregate-over-join TopK variant should produce facts");
    assert_rejection_facts_are_family_free(&facts, "unsupported_aggregate_over_join_topk");
    assert!(facts.contains_key("bounded_join_rejection_reason"));

    // A side too shallow for the multi-join proof is still executable: the shape
    // (keyed joins, a grouped aggregate over a join subplan, projection, sort,
    // fetch) is admitted; capacity is the allocator's, enforced as a typed cap
    // failure at runtime, not an admission gate.
    let plan = aggregate_over_join_subplan_topk_variant_plan(AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide);
    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "ShallowMultiJoinSide should be admitted"
    );
    assert!(
        unsupported_facts(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none(),
        "ShallowMultiJoinSide is admitted, not a bounded support gap"
    );
}

#[test]
fn bounded_multi_join_resident_grouped_aggregate_subquery_filter_admission_uses_capability_evidence() {
    let plan = resident_grouped_aggregate_subquery_filter_plan(false);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));

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

fn root_full_sort_shared_keyed_join_fanout_plan(shared_join_type: JoinType, interpose_shared_sort: bool) -> QueryPlan {
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
    let shared_join = builder
        .add_join(
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
        )
        .unwrap();
    // With the sort interposed it becomes the two-consumer frontier node, moving the
    // shared producer out of every admitted class while leaving the joins untouched.
    let shared_producer = if interpose_shared_sort {
        builder.add_sort(shared_join, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap()
    } else {
        shared_join
    };
    let dim_a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-join-dim-a.parquet",
        Arc::clone(&branch_dim_schema),
        vec!["k".into()],
    )));
    let join_a = builder
        .add_join(
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
        )
        .unwrap();
    let dim_b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/root-full-sort-shared-join-dim-b.parquet",
        branch_dim_schema,
        vec!["k".into()],
    )));
    let join_b = builder
        .add_join(
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
        )
        .unwrap();
    let aggregate_a = builder
        .add_aggregate(
            join_a,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "amount_sum")],
            ),
            aggregate_schema,
        )
        .unwrap();
    let root_join = builder
        .add_join(
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
        )
        .unwrap();
    let sort = builder
        .add_sort(root_join, SortSpec::new(vec![SortExpr::new(Expr::column("amount_sum"), true, false)]))
        .unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

#[test]
fn bounded_root_full_sort_join_aggregate_admits_shared_keyed_join_fanout() {
    // Pins that a shared keyed-hash join producer consumed by two downstream
    // keyed joins is admitted through the SharedResidentFanout subcontract
    // instead of being rejected as an unsupported shared producer.
    let plan = root_full_sort_shared_keyed_join_fanout_plan(JoinType::Inner, false);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "shared keyed-join fanout should be admitted"
    );
}

#[test]
fn bounded_root_full_sort_join_aggregate_admits_shared_keyed_semi_join_fanout() {
    // A keyed filtered semi join is a member of the admitted fragment and its output is
    // an ordinary resident frame, so sharing it is admitted on the same contract as a
    // keyed-hash producer. TPC-DS q64's shared producer is exactly this shape.
    let plan = root_full_sort_shared_keyed_join_fanout_plan(JoinType::LeftSemi, false);

    assert!(
        bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_some(),
        "shared keyed semi-join fanout should be admitted"
    );
}
