#[test]
fn native_optimizer_rejects_dynamic_filter_route_when_consumer_source_feeds_producer_build_subtree() {
    let plan = build_subtree_source_reuse_dynamic_filter_cycle_plan();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    let diagnostics = &result.diagnostics.dynamic_filter_route_discovery;
    assert!(diagnostics.candidates.iter().any(|candidate| {
        candidate.candidate_node_id.index() == 3
            && !candidate.routed
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerUpstreamOfProducer)
    }));
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| route.producer.node_id().index() != 3));
}

#[test]
fn native_optimizer_discovers_source_dynamic_filter_for_independent_repeated_consumer_file_set() {
    let plan = repeated_consumer_file_set_dynamic_filter_plan();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    let diagnostics = &result.diagnostics.dynamic_filter_route_discovery;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.route_count, 1);
    assert!(diagnostics.candidates[0].routed);
    assert_eq!(diagnostics.candidates[0].rejected_reason, None);
    assert_eq!(result.plan.dynamic_filter_routes().len(), 1);
}

#[test]
fn native_optimizer_rejects_probe_dynamic_filter_when_consumer_lineage_crosses_shared_local_source() {
    let (plan, first_branch_join, second_branch_join) = shared_prefix_branch_local_dynamic_filter_plan();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());
    let diagnostics = &result.diagnostics.dynamic_filter_route_discovery;

    for branch_join in [first_branch_join, second_branch_join] {
        assert!(diagnostics.candidates.iter().any(|candidate| {
            candidate.candidate_node_id == branch_join
                && !candidate.routed
                && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
        }));
        assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
            !matches!(
                route.producer,
                DynamicFilterProducer::HashJoinBuild {
                    join_node_id,
                    build_side: JoinSide::Right,
                    ..
                } if join_node_id == branch_join
            )
        }));
    }
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
        !matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
            || (route.producer.node_id() != first_branch_join && route.producer.node_id() != second_branch_join)
    }));
}

#[test]
fn native_optimizer_does_not_attach_branch_dynamic_filter_to_pre_split_shared_filter() {
    let (plan, shared_filter, first_branch_join, second_branch_join) =
        shared_filtered_prefix_branch_local_dynamic_filter_plan();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());
    let diagnostics = &result.diagnostics.dynamic_filter_route_discovery;

    for branch_join in [first_branch_join, second_branch_join] {
        assert!(diagnostics.candidates.iter().any(|candidate| {
            candidate.candidate_node_id == branch_join
                && !candidate.routed
                && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
        }));
    }
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
        !matches!(
            route.consumer,
            DynamicFilterConsumer::Filter {
                filter_node_id
            } if filter_node_id == shared_filter
        )
    }));
}

#[test]
fn native_optimizer_discovers_hash_join_build_to_probe_dynamic_filter_route_for_resident_probe() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["fact_key"]);
    let fact = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&fact_schema))));
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("hash_join_probe"));
    assert_eq!(candidate.consumer_node_id, Some(join));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode));
    assert_eq!(candidate.selected_source_bytes, None);
    assert_eq!(candidate.total_source_bytes, None);

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["dim_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["fact_key".to_string()]);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }));
    assert!(matches!(routes[0].consumer, DynamicFilterConsumer::HashJoinProbe { probe_side: JoinSide::Left, .. }));
}

#[test]
fn native_optimizer_discovers_hash_join_build_to_filter_dynamic_filter_route_for_resident_probe_filter() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["fact_key"]);
    let fact = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&fact_schema))));
    let filtered_fact = builder.add_filter(
        fact,
        Expr::binary(Expr::column("fact_key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&fact_schema),
    );
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        filtered_fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("filter"));
    assert_eq!(candidate.consumer_node_id, Some(filtered_fact));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::FilterPostDecode));
    assert_eq!(candidate.selected_source_bytes, None);
    assert_eq!(candidate.total_source_bytes, None);

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["dim_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["fact_key".to_string()]);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }));
    assert!(matches!(
        routes[0].consumer,
        DynamicFilterConsumer::Filter {
            filter_node_id
        } if filter_node_id == filtered_fact
    ));
}

#[test]
fn native_optimizer_rejects_join_probe_dynamic_filter_for_reused_local_source() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["fact_key"]);
    let fact = builder.add_source(estimated_local_source("fact", "fact_key", 256 * 1024 * 1024, 10_000_000));
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&fact_schema),
    );
    let passthrough = builder.add_projection(
        fact,
        vec![ProjectionOutput::new(Expr::column("fact_key"), Field::new("fact_key", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![join, passthrough], fact_schema);
    let plan = builder.finish_gpu_resident(union);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, Some(fact));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.consumer_kind, None);
    assert_eq!(candidate.consumer_node_id, None);
    assert_eq!(candidate.consumer_lowering, None);
    assert_eq!(candidate.consumer_reduction_stage, None);
    assert_eq!(candidate.selected_source_bytes, Some(256 * 1024 * 1024));
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));

    let routes = result.plan.dynamic_filter_routes();
    assert!(routes.is_empty());
}

#[test]
fn native_optimizer_routes_q02_like_shared_partsupp_source_when_all_key_consumers_are_constrained() {
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions::default());
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, Some(fixture.partsupp));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("ps_partkey"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("local_parquet_source"));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::SourcePostDecode));
    assert_eq!(candidate.rejected_reason, None);

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["p_partkey".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["ps_partkey".to_string()]);
    assert_eq!(routes[0].consumer_lowering, DynamicFilterConsumerLowering::PostDecodeOnly);
    assert!(matches!(
        routes[0].producer,
        DynamicFilterProducer::HashJoinBuild {
            join_node_id,
            build_side: JoinSide::Left,
            ..
        } if join_node_id == fixture.part_partsupp_join
    ));
    assert!(
        matches!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == fixture.partsupp)
    );
}

#[test]
fn native_optimizer_rejects_q02_like_shared_source_filter_with_unconstrained_extra_consumer() {
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions {
        extra_unconstrained_partsupp_consumer: true,
        ..SharedPartsuppDynamicFilterPlanOptions::default()
    });
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, Some(fixture.partsupp));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("ps_partkey"));
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_rejects_q02_like_shared_source_filter_when_aggregate_key_is_outer_joined() {
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions {
        final_join_type: JoinType::Left,
        ..SharedPartsuppDynamicFilterPlanOptions::default()
    });
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_rejects_q02_like_shared_source_filter_when_aggregate_key_uses_null_equal_join() {
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions {
        final_join_null_equality: JoinNullEquality::NullsMatch,
        ..SharedPartsuppDynamicFilterPlanOptions::default()
    });
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_rejects_q02_like_shared_source_filter_when_aggregate_group_key_is_derived() {
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions {
        derived_aggregate_group_key: true,
        ..SharedPartsuppDynamicFilterPlanOptions::default()
    });
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_rejects_q02_like_shared_source_filter_when_constraint_uses_same_source_but_not_keyset() {
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions {
        filter_part_build_keyset: true,
        final_join_uses_unfiltered_part_keyset: true,
        ..SharedPartsuppDynamicFilterPlanOptions::default()
    });
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
        !(route.consumer_columns == ["ps_partkey".to_string()]
            && matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == fixture.partsupp))
    }));
}

#[test]
fn native_optimizer_rejects_q02_like_shared_source_filter_when_constrained_consumer_is_below_limit() {
    // A shared-partsupp consumer constrained by the same `part` keyset, but only after an
    // intervening `Limit`. Applying the keyset filter in place at the source would change which
    // rows the `Limit` selects, so the shared source must NOT be lowered to an in-place source
    // post-decode filter. (Without the `Limit` arm being rejected, this route would be wrongly
    // admitted as a `LocalParquetSource` consumer and silently drop rows the limited branch needs.)
    let fixture = q02_like_shared_partsupp_dynamic_filter_plan(SharedPartsuppDynamicFilterPlanOptions {
        limited_constrained_partsupp_consumer: true,
        ..SharedPartsuppDynamicFilterPlanOptions::default()
    });
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(fixture.plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == fixture.part_partsupp_join)
        .expect("part/partsupp join should be considered");
    assert_eq!(candidate.consumer_source_node_id, Some(fixture.partsupp));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("ps_partkey"));
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert_ne!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::SourcePostDecode));
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
        !matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == fixture.partsupp)
    }));
}

#[test]
fn native_optimizer_skips_join_probe_dynamic_filter_when_it_feeds_scalar_decimal_sum() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(18, 2), true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("sum_amount", DataType::Decimal128(38, 2), true)]));
    let join_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(18, 2), true)]));
    let fact_file = LocalParquetFile::new("/tmp/fact.parquet")
        .with_stable_id("fact")
        .with_projected_row_group_bytes(vec![256 * 1024 * 1024], vec![256 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![fact_file],
        Arc::clone(&fact_schema),
        vec!["fact_key".to_string(), "amount".to_string()],
    )));
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let passthrough = builder.add_projection(
        fact,
        vec![ProjectionOutput::new(
            Expr::column("amount"),
            Field::new("sum_amount", DataType::Decimal128(38, 2), true),
        )],
    );
    let union = builder.add_union(vec![aggregate, passthrough], aggregate_schema);
    let plan = builder.finish_gpu_resident(union);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_kind, None);
    assert_eq!(candidate.consumer_node_id, None);
    assert_eq!(candidate.consumer_lowering, None);
    assert_eq!(candidate.consumer_reduction_stage, None);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_skips_join_probe_dynamic_filter_when_filtered_join_feeds_scalar_decimal_sum() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(18, 2), true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("sum_amount", DataType::Decimal128(38, 2), true)]));
    let join_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(18, 2), true)]));
    let fact = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&fact_schema))));
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let filtered = builder.add_filter(
        join,
        Expr::binary(
            Expr::column("amount"),
            BinaryOp::GtEq,
            Expr::literal(LiteralValue::Decimal128 { raw: 0, precision: 18, scale: 2 }),
        ),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        filtered,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_gpu_resident(aggregate);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_kind.as_deref(), Some("hash_join_probe"));
    assert_eq!(candidate.consumer_node_id, Some(join));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode));
    assert_eq!(
        candidate.rejected_reason,
        Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum)
    );
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_keeps_join_probe_dynamic_filter_when_scalar_sum_is_not_decimal() {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_schema = schema(&["sum_amount"]);
    let fact = builder.add_source(estimated_local_source_with_stable_id(
        "fact",
        "fact",
        &["fact_key", "amount"],
        256 * 1024 * 1024,
        10_000_000,
    ));
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount")])
        .with_build_side(JoinSide::Right),
        schema(&["amount"]),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let passthrough = builder.add_projection(
        fact,
        vec![ProjectionOutput::new(Expr::column("amount"), Field::new("sum_amount", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![aggregate, passthrough], aggregate_schema);
    let plan = builder.finish_gpu_resident(union);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_kind, None);
    assert_eq!(candidate.consumer_node_id, None);
    assert_eq!(candidate.consumer_lowering, None);
    assert_eq!(candidate.consumer_reduction_stage, None);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_skips_resident_join_probe_dynamic_filter_when_it_feeds_scalar_decimal_sum() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(18, 2), true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("sum_amount", DataType::Decimal128(38, 2), true)]));
    let fact = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&fact_schema))));
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount")])
        .with_build_side(JoinSide::Right),
        Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(18, 2), true)])),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_gpu_resident(aggregate);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_kind.as_deref(), Some("hash_join_probe"));
    assert_eq!(candidate.consumer_node_id, Some(join));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode));
    assert_eq!(
        candidate.rejected_reason,
        Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum)
    );
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_falls_back_to_branch_filter_dynamic_filter_for_reused_local_source() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["fact_key"]);
    let fact = builder.add_source(estimated_local_source("fact", "fact_key", 256 * 1024 * 1024, 10_000_000));
    let filtered_fact = builder.add_filter(
        fact,
        Expr::binary(Expr::column("fact_key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&fact_schema),
    );
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        filtered_fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&fact_schema),
    );
    let passthrough = builder.add_projection(
        fact,
        vec![ProjectionOutput::new(Expr::column("fact_key"), Field::new("fact_key", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![join, passthrough], fact_schema);
    let plan = builder.finish_gpu_resident(union);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, Some(fact));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("filter"));
    assert_eq!(candidate.consumer_node_id, Some(filtered_fact));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::FilterPostDecode));
    assert_eq!(candidate.selected_source_bytes, Some(256 * 1024 * 1024));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["dim_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["fact_key".to_string()]);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }));
    assert!(matches!(
        routes[0].consumer,
        DynamicFilterConsumer::Filter {
            filter_node_id
        } if filter_node_id == filtered_fact
    ));
}

#[test]
fn native_optimizer_discovers_filter_dynamic_filter_route_through_direct_projection_alias() {
    let mut builder = QueryPlanBuilder::new();
    let raw_fact_schema = schema(&["raw_fact_key"]);
    let fact =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&raw_fact_schema))));
    let filtered_fact = builder.add_filter(
        fact,
        Expr::binary(Expr::column("raw_fact_key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&raw_fact_schema),
    );
    let projected_fact = builder.add_projection(
        filtered_fact,
        vec![ProjectionOutput::new(Expr::column("raw_fact_key"), Field::new("fact_key", DataType::Int32, false))],
    );
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        projected_fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("raw_fact_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("filter"));
    assert_eq!(candidate.consumer_node_id, Some(filtered_fact));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::FilterPostDecode));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["dim_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["raw_fact_key".to_string()]);
    assert!(matches!(
        routes[0].consumer,
        DynamicFilterConsumer::Filter {
            filter_node_id
        } if filter_node_id == filtered_fact
    ));
}

#[test]
fn native_optimizer_does_not_push_filter_dynamic_filter_below_limit() {
    let mut builder = QueryPlanBuilder::new();
    let raw_fact_schema = schema(&["raw_fact_key"]);
    let fact =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&raw_fact_schema))));
    let filtered_fact = builder.add_filter(
        fact,
        Expr::binary(Expr::column("raw_fact_key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&raw_fact_schema),
    );
    let limited_fact = builder.add_limit(filtered_fact, LimitSpec::fetch(10), Arc::clone(&raw_fact_schema));
    let projected_fact = builder.add_projection(
        limited_fact,
        vec![ProjectionOutput::new(Expr::column("raw_fact_key"), Field::new("fact_key", DataType::Int32, false))],
    );
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        projected_fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("hash_join_probe"));
    assert_eq!(candidate.consumer_node_id, Some(join));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].consumer_columns, vec!["fact_key".to_string()]);
    assert!(matches!(routes[0].consumer, DynamicFilterConsumer::HashJoinProbe { probe_side: JoinSide::Left, .. }));
}

#[test]
fn native_optimizer_does_not_attach_source_dynamic_filter_below_limit() {
    let mut builder = QueryPlanBuilder::new();
    let raw_fact_schema = schema(&["raw_fact_key"]);
    let fact = builder.add_source(estimated_local_source("fact", "raw_fact_key", 512 * 1024 * 1024, 10_000_000));
    let limited_fact = builder.add_limit(fact, LimitSpec::fetch(10), Arc::clone(&raw_fact_schema));
    let projected_fact = builder.add_projection(
        limited_fact,
        vec![ProjectionOutput::new(Expr::column("raw_fact_key"), Field::new("fact_key", DataType::Int32, false))],
    );
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        projected_fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, None);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("hash_join_probe"));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].consumer_columns, vec!["fact_key".to_string()]);
    assert!(matches!(routes[0].consumer, DynamicFilterConsumer::HashJoinProbe { probe_side: JoinSide::Left, .. }));
}

#[test]
fn native_optimizer_does_not_attach_post_decode_dynamic_filter_to_shared_filter_below_limit() {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["fact_key"]);
    let fact =
        builder.add_source(estimated_local_source("shared_limited_fact", "fact_key", 256 * 1024 * 1024, 10_000_000));
    let limited_fact = builder.add_limit(fact, LimitSpec::fetch(10), Arc::clone(&fact_schema));
    let shared_filter = builder.add_filter(
        limited_fact,
        Expr::binary(Expr::column("fact_key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&fact_schema),
    );
    let dim = builder.add_source(estimated_local_source("shared_limited_dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        shared_filter,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&fact_schema),
    );
    let root = builder.add_union(vec![join, shared_filter], Arc::clone(&fact_schema));
    let plan = builder.finish_gpu_resident(root);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_kind.as_deref(), Some("hash_join_probe"));
    assert_eq!(candidate.consumer_node_id, Some(join));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::PostDecodeOnly));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert!(
        matches!(routes[0].consumer, DynamicFilterConsumer::HashJoinProbe { join_node_id, probe_side: JoinSide::Left } if join_node_id == join)
    );
    assert!(
        !matches!(routes[0].consumer, DynamicFilterConsumer::Filter { filter_node_id } if filter_node_id == shared_filter)
    );
}
