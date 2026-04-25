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
fn native_optimizer_keeps_join_backed_aggregate_keyset_route_without_large_keyset_evidence() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "fact",
        &["fact_key", "date_key"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let date = builder.add_source(estimated_local_source("date_dim", "date_key", 1024 * 1024, 365));
    let date_join = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_key", "date_key"),
                ColumnRef::input(1, "date_key", "date_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "fact_key", "fact_key"),
            "fact_key",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["fact_key"]),
    );
    let keyset = builder.add_aggregate(
        date_join,
        AggregateSpec::new(vec![Expr::column("fact_key")], Vec::new()),
        schema(&["fact_key"]),
    );
    let item = builder.add_source(estimated_local_source("item", "item_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        keyset,
        item,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "item_key", "item_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "fact_key", "fact_key"),
            "fact_key",
        )])
        .with_build_side(JoinSide::Left),
        schema(&["fact_key"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 2);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("final join candidate should be reported");
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_source_node_id, Some(item));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("item_key"));
    assert_eq!(candidate.selected_source_bytes, Some(16 * 1024 * 1024));
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(result.plan.dynamic_filter_routes().len(), 1);
}

#[test]
fn native_optimizer_discovers_dynamic_filter_consumer_below_aggregate_group_key() {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("dim", "dim_key", 16 * 1024 * 1024, 100_000));
    let fact = builder.add_source(estimated_local_source("fact", "raw_fact_key", 512 * 1024 * 1024, 10_000_000));
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
    assert_eq!(candidate.selected_source_bytes, Some(512 * 1024 * 1024));

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
fn native_optimizer_keeps_small_dimension_build_for_source_filter_below_aggregate_group_key() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "sales_grouped_before_date",
        &["raw_date_sk", "sales_price"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            vec![Expr::column("raw_date_sk").alias("date_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales_price"), "sales_total")],
        ),
        schema(&["date_sk", "sales_total"]),
    );
    let date = builder.add_source(estimated_local_source("small_date_dim_for_grouped_fact", "d_date_sk", 1024 * 1024, 365));
    let join = builder.add_join(
        grouped_fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales_total", "sales_total"), "sales_total"),
        ])
        .with_build_side(JoinSide::Right),
        schema(&["date_sk", "sales_total"]),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the date join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.routed)
        .expect("small date build should route into grouped fact source");
    assert_eq!(candidate.producer_build_side, Some(JoinSide::Right));
    assert_eq!(candidate.consumer_source_node_id, Some(fact));
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("raw_date_sk"));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::SourceReader));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].producer_columns, vec!["d_date_sk".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["raw_date_sk".to_string()]);
    assert!(matches!(
        routes[0].producer,
        DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }
    ));
    assert!(
        matches!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == fact)
    );
}
