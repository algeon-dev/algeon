#[test]
fn native_optimizer_discovers_dynamic_filter_routes_for_union_probe_branches() {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("date_dim", "d_date_sk", 16 * 1024 * 1024, 100_000));
    let fact_a = builder.add_source(estimated_local_source("fact_a", "a_date_sk", 512 * 1024 * 1024, 10_000_000));
    let fact_b = builder.add_source(estimated_local_source("fact_b", "b_date_sk", 384 * 1024 * 1024, 5_000_000));
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

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

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
fn native_optimizer_adds_post_decode_probe_route_for_mixed_union_branch_reuse() {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("date_dim", "d_date_sk", 16 * 1024 * 1024, 100_000));
    let sales = builder.add_source(estimated_local_source_with_fields(
        "web_sales",
        &["ws_sold_date_sk", "ws_item_sk", "ws_order_number"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let returns = builder.add_source(estimated_local_source_with_fields(
        "web_returns",
        &["wr_returned_date_sk", "wr_item_sk", "wr_order_number"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let branch_schema = schema(&["date_sk"]);
    let sales_branch = builder.add_projection(
        sales,
        vec![ProjectionOutput::new(Expr::column("ws_sold_date_sk"), Field::new("date_sk", DataType::Int32, false))],
    );
    let returns_projected = builder.add_projection(
        returns,
        vec![
            ProjectionOutput::new(
                Expr::column("wr_returned_date_sk"),
                Field::new("wr_returned_date_sk", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("wr_item_sk"), Field::new("wr_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::column("wr_order_number"),
                Field::new("wr_order_number", DataType::Int32, false),
            ),
        ],
    );
    let sales_join_build = builder.add_projection(
        sales,
        vec![
            ProjectionOutput::new(Expr::column("ws_item_sk"), Field::new("ws_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::column("ws_order_number"),
                Field::new("ws_order_number", DataType::Int32, false),
            ),
        ],
    );
    let returns_with_sales = builder.add_join(
        returns_projected,
        sales_join_build,
        JoinSpec::new(
            JoinType::Left,
            vec![
                JoinKey::new(
                    ColumnRef::input(0, "wr_item_sk", "wr_item_sk"),
                    ColumnRef::input(1, "ws_item_sk", "ws_item_sk"),
                ),
                JoinKey::new(
                    ColumnRef::input(0, "wr_order_number", "wr_order_number"),
                    ColumnRef::input(1, "ws_order_number", "ws_order_number"),
                ),
            ],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "wr_returned_date_sk", "wr_returned_date_sk"),
            "date_sk",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let union = builder.add_union(vec![sales_branch, returns_with_sales], Arc::clone(&branch_schema));
    let final_join = builder.add_join(
        union,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Right),
        branch_schema,
    );
    let plan = builder.finish_gpu_resident(final_join);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 2);
    let final_join_candidates = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .filter(|candidate| candidate.candidate_node_id == final_join)
        .collect::<Vec<_>>();
    assert!(final_join_candidates.iter().any(|candidate| {
        !candidate.routed
            && candidate.consumer_source_node_id == Some(sales)
            && candidate.consumer_key_column.as_deref() == Some("ws_sold_date_sk")
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
    }));
    assert!(final_join_candidates.iter().any(|candidate| {
        candidate.routed
            && candidate.consumer_source_node_id == Some(returns)
            && candidate.consumer_key_column.as_deref() == Some("wr_returned_date_sk")
            && candidate.consumer_kind.as_deref() == Some("local_parquet_source")
            && candidate.consumer_reduction_stage == Some(DynamicFilterConsumerReductionStage::SourceReader)
    }));
    assert!(final_join_candidates.iter().any(|candidate| {
        candidate.routed
            && candidate.consumer_source_node_id.is_none()
            && candidate.consumer_key_column.as_deref() == Some("date_sk")
            && candidate.consumer_kind.as_deref() == Some("hash_join_probe")
            && candidate.consumer_node_id == Some(final_join)
            && candidate.consumer_lowering == Some(DynamicFilterConsumerLowering::PostDecodeOnly)
            && candidate.consumer_reduction_stage == Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode)
    }));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 2);
    assert!(routes.iter().any(|route| {
        route.consumer_columns == ["wr_returned_date_sk".to_string()]
            && matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == returns)
    }));
    assert!(routes.iter().any(|route| {
        route.consumer_columns == ["date_sk".to_string()]
            && matches!(route.consumer, DynamicFilterConsumer::HashJoinProbe { join_node_id, probe_side: JoinSide::Left } if join_node_id == final_join)
    }));
    assert!(routes.iter().all(|route| route.producer_columns == ["d_date_sk".to_string()]));
}

#[test]
fn native_optimizer_adds_post_decode_probe_route_when_all_union_branch_lineages_are_reused() {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("date_dim_all_reused", "d_date_sk", 16 * 1024 * 1024, 100_000));
    let sales = builder.add_source(estimated_local_source_with_fields(
        "web_sales_all_reused",
        &["ws_sold_date_sk", "ws_item_sk", "ws_order_number"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let returns = builder.add_source(estimated_local_source_with_fields(
        "web_returns_all_reused",
        &["wr_returned_date_sk", "wr_item_sk", "wr_order_number"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let branch_schema = schema(&["date_sk"]);
    let sales_branch = builder.add_projection(
        sales,
        vec![ProjectionOutput::new(Expr::column("ws_sold_date_sk"), Field::new("date_sk", DataType::Int32, false))],
    );
    let sales_join_build = builder.add_projection(
        sales,
        vec![
            ProjectionOutput::new(Expr::column("ws_item_sk"), Field::new("ws_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::column("ws_order_number"),
                Field::new("ws_order_number", DataType::Int32, false),
            ),
        ],
    );
    let returns_projected = builder.add_projection(
        returns,
        vec![
            ProjectionOutput::new(
                Expr::column("wr_returned_date_sk"),
                Field::new("wr_returned_date_sk", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("wr_item_sk"), Field::new("wr_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::column("wr_order_number"),
                Field::new("wr_order_number", DataType::Int32, false),
            ),
        ],
    );
    let returns_with_sales = builder.add_join(
        returns_projected,
        sales_join_build,
        JoinSpec::new(
            JoinType::Left,
            vec![
                JoinKey::new(
                    ColumnRef::input(0, "wr_item_sk", "wr_item_sk"),
                    ColumnRef::input(1, "ws_item_sk", "ws_item_sk"),
                ),
                JoinKey::new(
                    ColumnRef::input(0, "wr_order_number", "wr_order_number"),
                    ColumnRef::input(1, "ws_order_number", "ws_order_number"),
                ),
            ],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "wr_returned_date_sk", "wr_returned_date_sk"),
            "date_sk",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    );
    let returns_passthrough = builder.add_projection(
        returns,
        vec![ProjectionOutput::new(Expr::column("wr_returned_date_sk"), Field::new("date_sk", DataType::Int32, false))],
    );
    let union =
        builder.add_union(vec![sales_branch, returns_with_sales, returns_passthrough], Arc::clone(&branch_schema));
    let final_join = builder.add_join(
        union,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Right),
        branch_schema,
    );
    let plan = builder.finish_gpu_resident(final_join);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let final_join_candidates = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .filter(|candidate| candidate.candidate_node_id == final_join)
        .collect::<Vec<_>>();
    assert_eq!(final_join_candidates.iter().filter(|candidate| candidate.routed).count(), 1);
    assert!(final_join_candidates.iter().any(|candidate| {
        !candidate.routed
            && candidate.consumer_source_node_id == Some(sales)
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
    }));
    assert!(final_join_candidates.iter().any(|candidate| {
        !candidate.routed
            && candidate.consumer_source_node_id == Some(returns)
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
    }));
    assert!(final_join_candidates.iter().any(|candidate| {
        candidate.routed
            && candidate.consumer_source_node_id.is_none()
            && candidate.consumer_kind.as_deref() == Some("hash_join_probe")
            && candidate.consumer_node_id == Some(final_join)
            && candidate.consumer_reduction_stage == Some(DynamicFilterConsumerReductionStage::JoinProbePostDecode)
    }));
    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].consumer_columns, vec!["date_sk".to_string()]);
    assert!(matches!(
        routes[0].consumer,
        DynamicFilterConsumer::HashJoinProbe {
            join_node_id,
            probe_side: JoinSide::Left,
        } if join_node_id == final_join
    ));
}
