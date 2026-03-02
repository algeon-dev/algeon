#[test]
fn native_optimizer_discovers_hash_join_build_to_parquet_source_dynamic_filter_route() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 256 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("left_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("local_parquet_source"));
    assert!(candidate.consumer_node_id.is_some());
    assert_eq!(
        candidate.filter_kinds,
        vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains]
    );
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::ReaderPredicate));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::SourceReader));
    assert_eq!(candidate.selected_source_bytes, Some(256 * 1024 * 1024));
    assert_eq!(candidate.total_source_bytes, Some(256 * 1024 * 1024));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].id.index(), 0);
    assert_eq!(routes[0].producer_columns, vec!["right_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["left_key".to_string()]);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }));
    assert!(matches!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { .. }));
}

#[test]
fn native_optimizer_discovers_dynamic_filter_route_for_inner_keyed_residual_join() {
    let plan =
        estimated_inner_residual_join_plan(JoinSide::Right, 256 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("left_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("local_parquet_source"));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::ReaderPredicate));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }));
    assert_eq!(routes[0].producer_columns, vec!["right_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["left_key".to_string()]);
}

#[test]
fn native_optimizer_rejects_dynamic_filter_route_when_consumer_source_feeds_producer_build_subtree() {
    let plan = build_subtree_source_reuse_dynamic_filter_cycle_plan();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);
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

    let result = optimize_query_plan(plan, &options);
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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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

    let result = optimize_query_plan(plan, &options);

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
fn native_optimizer_discovers_hash_join_dynamic_filter_route_through_aggregate_group_key() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source("fact", "fact_key", 256 * 1024 * 1024, 10_000_000));
    let dim = builder.add_source(estimated_local_source("dim", "raw_dim_key", 16 * 1024 * 1024, 100_000));
    let keyset = builder.add_aggregate(
        dim,
        AggregateSpec::new(vec![Expr::column("raw_dim_key").alias("dim_key")], Vec::new()),
        schema(&["dim_key"]),
    );
    let join = builder.add_join(
        fact,
        keyset,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.producer_key_column.as_deref(), Some("dim_key"));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("fact_key"));
    assert_eq!(candidate.selected_source_bytes, Some(256 * 1024 * 1024));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["dim_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["fact_key".to_string()]);
    assert!(matches!(
        routes[0].producer,
        DynamicFilterProducer::AggregateGroupKeys {
            aggregate_node_id
        } if aggregate_node_id == keyset
    ));
    assert!(matches!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { .. }));
}

#[test]
fn native_optimizer_discovers_dynamic_filter_consumer_below_aggregate_group_key() {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let fact = builder.add_source(estimated_local_source("fact", "raw_fact_key", 256 * 1024 * 1024, 10_000_000));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(vec![Expr::column("raw_fact_key").alias("fact_key")], Vec::new()),
        schema(&["fact_key"]),
    );
    let join = builder.add_join(
        dim,
        grouped_fact,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_key", "dim_key"), ColumnRef::input(1, "fact_key", "fact_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "dim_key", "dim_key"), "dim_key")])
        .with_build_side(JoinSide::Left),
        schema(&["dim_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, Some(fact));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("raw_fact_key"));
    assert_eq!(candidate.selected_source_bytes, Some(256 * 1024 * 1024));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["dim_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["raw_fact_key".to_string()]);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Left, .. }));
    assert!(
        matches!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == fact)
    );
}

#[test]
fn native_optimizer_discovers_dynamic_filter_routes_for_union_probe_branches() {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("date_dim", "d_date_sk", 16 * 1024 * 1024, 100_000));
    let fact_a = builder.add_source(estimated_local_source("fact_a", "a_date_sk", 256 * 1024 * 1024, 10_000_000));
    let fact_b = builder.add_source(estimated_local_source("fact_b", "b_date_sk", 128 * 1024 * 1024, 5_000_000));
    let fact_schema = schema(&["date_sk"]);
    let fact_a_projected = builder.add_projection(
        fact_a,
        vec![ProjectionOutput::new(Expr::column("a_date_sk"), Field::new("date_sk", DataType::Int32, false))],
    );
    let fact_b_projected = builder.add_projection(
        fact_b,
        vec![ProjectionOutput::new(Expr::column("b_date_sk"), Field::new("date_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![fact_a_projected, fact_b_projected], Arc::clone(&fact_schema));
    let join = builder.add_join(
        dim,
        union,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "d_date_sk", "d_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(ColumnRef::input(1, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Left),
        fact_schema,
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 2);
    assert_eq!(
        result
            .diagnostics
            .dynamic_filter_route_discovery
            .candidates
            .iter()
            .filter(|candidate| candidate.routed)
            .count(),
        2
    );
    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 2);
    assert_eq!(
        routes.iter().map(|route| route.producer_columns.clone()).collect::<Vec<_>>(),
        vec![vec!["d_date_sk".to_string()], vec!["d_date_sk".to_string()],]
    );
    assert_eq!(
        routes.iter().map(|route| route.consumer_columns.clone()).collect::<Vec<_>>(),
        vec![vec!["a_date_sk".to_string()], vec!["b_date_sk".to_string()],]
    );
    assert!(routes.iter().all(|route| matches!(
        route.producer,
        DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Left, .. }
    )));
    let consumer_sources = routes
        .iter()
        .map(|route| match route.consumer {
            DynamicFilterConsumer::LocalParquetSource { source_node_id } => source_node_id,
            _ => panic!("union branch routes should target local Parquet sources"),
        })
        .collect::<Vec<_>>();
    assert_eq!(consumer_sources, vec![fact_a, fact_b]);
}

#[test]
fn dynamic_filter_runtime_metrics_record_produced_skipped_and_applied_events() {
    let plan = optimize_query_plan(
        estimated_inner_join_plan(JoinSide::Right, 256 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000),
        &NativeOptimizerOptions::default(),
    )
    .plan;
    let route = &plan.dynamic_filter_routes()[0];
    let filter = DynamicFilter::SmallInListI64 { values: vec![10, 20, 30], min: 10, max: 30 };
    let mut metrics = QueryMetrics::default();

    metrics.record_dynamic_filter_produced(route, &filter, 3);
    metrics.record_dynamic_filter_applied(
        route,
        &filter,
        DynamicFilterSourceSnapshot { row_groups: 5, source_bytes: Some(8192) },
        DynamicFilterSourceSnapshot { row_groups: 2, source_bytes: Some(4096) },
    );
    metrics.record_dynamic_filter_reader_predicate_source_reduction(Some(1000), Some(250));
    metrics.record_dynamic_filter_post_decode_rows(route, &filter, Some(80), Some(25));
    metrics.record_dynamic_filter_skipped(route, "unsupported_producer_key_type");

    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert_eq!(metrics.dynamic_filter_skipped_count, 1);
    assert_eq!(metrics.dynamic_filter_row_groups_pruned, 3);
    assert_eq!(metrics.dynamic_filter_source_bytes_pruned, 4096);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_before, 1000);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_after, 250);
    assert_eq!(metrics.dynamic_filter_reader_predicate_rows_filtered, 750);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_before, 80);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_after, 25);
    assert_eq!(metrics.dynamic_filter_post_decode_rows_filtered, 55);
    assert_eq!(metrics.dynamic_filter_source_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_filter_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_join_probe_post_decode_rows_filtered, 0);
    assert_eq!(metrics.dynamic_filter_runtime_rows.len(), 4);
    assert!(metrics.dynamic_filter_runtime_rows[0].to_stable_kv_row().contains("event=produced"));
    assert!(metrics.dynamic_filter_runtime_rows[1].to_stable_kv_row().contains("event=applied"));
    assert!(metrics.dynamic_filter_runtime_rows[2].to_stable_kv_row().contains("event=post_decode_rows"));
    assert!(
        metrics.dynamic_filter_runtime_rows[0].to_stable_kv_row().contains("consumer_reduction_stage=source_reader")
    );
    assert!(
        metrics.dynamic_filter_runtime_rows[2]
            .to_stable_kv_row()
            .contains("post_decode_rows_before=80,post_decode_rows_after=25,post_decode_rows_filtered=55")
    );
    assert!(
        metrics.dynamic_filter_runtime_rows[3]
            .to_stable_kv_row()
            .contains("skipped_reason=unsupported_producer_key_type")
    );
}

#[test]
fn dynamic_filter_keeps_large_integer_keysets_as_hash_set_contains_payloads() {
    let filter = DynamicFilter::from_i64_values([Some(5), Some(1), None, Some(5), Some(3)], 2);

    assert_eq!(filter, DynamicFilter::HashSetContainsI64 { values: vec![1, 3, 5], min: 1, max: 5 });
    assert_eq!(filter.kind(), DynamicFilterKind::HashSetContains);
    assert_eq!(filter.value_count(), Some(3));
    assert_eq!(filter.i64_range(), Some((1, 5)));
}

#[test]
fn dynamic_filter_compacts_dense_integer_keysets_to_range_payloads() {
    let filter = DynamicFilter::from_i64_values([Some(3), Some(1), Some(2), Some(2), None], 2);

    assert_eq!(filter, DynamicFilter::RangeI64 { min: 1, max: 3 });
    assert_eq!(filter.kind(), DynamicFilterKind::Range);
    assert_eq!(filter.value_count(), None);
    assert_eq!(filter.i64_range(), Some((1, 3)));
}

#[test]
fn aggregate_group_key_dynamic_filter_uses_aggregate_output_column_before_projection_alias() {
    let mut builder = QueryPlanBuilder::new();
    let aggregate = aggregate_keyset_source(&mut builder, "returns", "raw_customer_sk");
    let projected = builder.add_projection(
        aggregate,
        vec![ProjectionOutput::new(
            Expr::column("raw_customer_sk"),
            Field::new("ctr_customer_sk", DataType::Int32, false),
        )],
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let join = builder.add_join(
        projected,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            "c_customer_sk",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["c_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let route = &result.plan.dynamic_filter_routes()[0];
    assert!(
        matches!(route.producer, DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } if aggregate_node_id == aggregate)
    );
    assert_eq!(route.producer_columns, vec!["raw_customer_sk".to_string()]);
    assert_eq!(route.consumer_columns, vec!["c_customer_sk".to_string()]);
}

#[test]
fn dynamic_filter_uses_hash_join_build_when_join_input_narrows_aggregate_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let returns = builder.add_source(gpu_source("returns", &["raw_customer_sk", "raw_store_sk"]));
    let aggregate = builder.add_aggregate(
        returns,
        AggregateSpec::new(vec![Expr::column("raw_customer_sk"), Expr::column("raw_store_sk")], Vec::new()),
        schema(&["raw_customer_sk", "raw_store_sk"]),
    );
    let projected = builder.add_projection(
        aggregate,
        vec![
            ProjectionOutput::new(
                Expr::column("raw_customer_sk"),
                Field::new("ctr_customer_sk", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("raw_store_sk"), Field::new("ctr_store_sk", DataType::Int32, false)),
        ],
    );
    let store = builder.add_source(estimated_local_source("store", "s_store_sk", 1024, 10));
    let narrowed = builder.add_join(
        store,
        projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "s_store_sk", "s_store_sk"),
                ColumnRef::input(1, "ctr_store_sk", "ctr_store_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "ctr_customer_sk", "ctr_customer_sk"),
            "ctr_customer_sk",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["ctr_customer_sk"]),
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let final_join = builder.add_join(
        narrowed,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            "c_customer_sk",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["c_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(final_join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    let route = result
        .plan
        .dynamic_filter_routes()
        .iter()
        .find(|route| route.consumer_columns == ["c_customer_sk".to_string()])
        .expect("final join should route a customer key filter");
    assert!(matches!(
        route.producer,
        DynamicFilterProducer::HashJoinBuild {
            join_node_id,
            build_side: JoinSide::Left,
            ..
        } if join_node_id == final_join
    ));
    assert_eq!(route.producer_columns, vec!["ctr_customer_sk".to_string()]);
}

fn build_subtree_source_reuse_dynamic_filter_cycle_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let shared_probe =
        builder.add_source(estimated_local_source("shared_probe", "shared_key", 8 * 1024 * 1024, 10_000));
    let build_seed = builder.add_source(estimated_local_source("build_seed", "build_key", 8 * 1024 * 1024, 10_000));
    let build_subtree_schema = schema(&["build_key"]);
    let build_subtree = builder.add_join(
        build_seed,
        shared_probe,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "build_key", "build_key"),
                ColumnRef::input(1, "shared_key", "shared_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "build_key", "build_key"),
            "build_key",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&build_subtree_schema),
    );
    let root = builder.add_join(
        shared_probe,
        build_subtree,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "shared_key", "shared_key"),
                ColumnRef::input(1, "build_key", "build_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "shared_key", "shared_key"),
            "shared_key",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["shared_key"]),
    );
    builder.finish_gpu_resident(root)
}

fn repeated_consumer_file_set_dynamic_filter_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let output_schema = schema(&["left_key"]);
    let fact = builder.add_source(estimated_local_source_with_stable_id(
        "fact_probe",
        "shared_fact",
        &["left_key"],
        256 * 1024 * 1024,
        10_000_000,
    ));
    let repeated_fact = builder.add_source(estimated_local_source_with_stable_id(
        "fact_repeated",
        "shared_fact",
        &["left_key"],
        256 * 1024 * 1024,
        10_000_000,
    ));
    let dim = builder.add_source(estimated_local_source("dim", "right_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let union = builder.add_union(vec![join, repeated_fact], output_schema);
    builder.finish_gpu_resident(union)
}

fn shared_prefix_branch_local_dynamic_filter_plan() -> (nexus_query_engine::QueryPlan, NodeId, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "store_sales",
        &["sold_date_sk", "item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let shared_prefix = builder.add_projection(
        fact,
        vec![
            ProjectionOutput::new(Expr::column("sold_date_sk"), Field::new("sold_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("item_sk"), Field::new("item_sk", DataType::Int32, false)),
        ],
    );
    let first_date = builder.add_source(estimated_local_source("date_1999", "date_sk", 16 * 1024 * 1024, 366));
    let second_date = builder.add_source(estimated_local_source("date_2000", "date_sk", 16 * 1024 * 1024, 366));
    let branch_schema = schema(&["item_sk"]);
    let first_branch_join = builder.add_join(
        shared_prefix,
        first_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let second_branch_join = builder.add_join(
        shared_prefix,
        second_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let final_join = builder.add_join(
        first_branch_join,
        second_branch_join,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        branch_schema,
    );
    (builder.finish_gpu_resident(final_join), first_branch_join, second_branch_join)
}

fn shared_filtered_prefix_branch_local_dynamic_filter_plan() -> (nexus_query_engine::QueryPlan, NodeId, NodeId, NodeId)
{
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["sold_date_sk", "item_sk"]);
    let fact = builder.add_source(estimated_local_source_with_fields(
        "store_sales",
        &["sold_date_sk", "item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let shared_filter = builder.add_filter(
        fact,
        Expr::binary(Expr::column("item_sk"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&fact_schema),
    );
    let shared_prefix = builder.add_projection(
        shared_filter,
        vec![
            ProjectionOutput::new(Expr::column("sold_date_sk"), Field::new("sold_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("item_sk"), Field::new("item_sk", DataType::Int32, false)),
        ],
    );
    let first_date = builder.add_source(estimated_local_source("date_1999", "date_sk", 16 * 1024 * 1024, 366));
    let second_date = builder.add_source(estimated_local_source("date_2000", "date_sk", 16 * 1024 * 1024, 366));
    let branch_schema = schema(&["item_sk"]);
    let first_branch_join = builder.add_join(
        shared_prefix,
        first_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let second_branch_join = builder.add_join(
        shared_prefix,
        second_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let final_join = builder.add_join(
        first_branch_join,
        second_branch_join,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        branch_schema,
    );
    (builder.finish_gpu_resident(final_join), shared_filter, first_branch_join, second_branch_join)
}

fn aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> nexus_query_engine::NodeId {
    let source = builder.add_source(gpu_source(name, &[key_name]));
    builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()), schema(&[key_name]))
}

fn expr_contains_int32_equality(expr: &Expr, column: &str, value: i32) -> bool {
    match expr {
        Expr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            (expr_is_column(left, column) && expr_is_int32_literal(right, value))
                || (expr_is_column(right, column) && expr_is_int32_literal(left, value))
                || expr_contains_int32_equality(left, column, value)
                || expr_contains_int32_equality(right, column, value)
        },
        Expr::Binary { left, right, .. } => {
            expr_contains_int32_equality(left, column, value) || expr_contains_int32_equality(right, column, value)
        },
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            expr_contains_int32_equality(inner, column, value)
        },
        Expr::Cast { expr, .. } | Expr::TryCast { expr, .. } => expr_contains_int32_equality(expr, column, value),
        Expr::Alias { expr, .. } => expr_contains_int32_equality(expr, column, value),
        _ => false,
    }
}

fn expr_is_column(expr: &Expr, column: &str) -> bool {
    matches!(expr, Expr::Column(expr_column) if expr_column.name() == column)
}

fn expr_is_int32_literal(expr: &Expr, value: i32) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::Int32(candidate)) if *candidate == value)
}

fn max_direct_join_probe_consumer_count(plan: &nexus_query_engine::QueryPlan) -> usize {
    let mut counts = std::collections::BTreeMap::<nexus_query_engine::NodeId, usize>::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let probe_input = match join.spec.build_side {
            JoinSide::Left => join.inputs[1],
            JoinSide::Right => join.inputs[0],
        };
        *counts.entry(probe_input).or_default() += 1;
    }
    counts.into_values().max().unwrap_or(0)
}
