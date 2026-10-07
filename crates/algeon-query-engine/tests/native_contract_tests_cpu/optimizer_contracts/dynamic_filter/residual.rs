use super::*;

#[test]
fn native_optimizer_discovers_residual_equality_route_between_filtered_dimensions() {
    let mut builder = QueryPlanBuilder::new();
    let catalog_sales = builder.add_source(estimated_local_source_with_fields(
        "catalog_sales",
        &["cs_sold_date_sk", "cs_item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let filtered_d1 = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "date_dim_d1",
        "date_dim_d1",
        &["d_date_sk", "d_week_seq", "d_year"],
        1024 * 1024,
        365,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1999))),
    ));
    let inventory = builder.add_source(estimated_local_source_with_fields(
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        2 * 1024 * 1024 * 1024,
        400_000_000,
    ));
    let d2 = builder.add_source(estimated_local_source_with_fields(
        "date_dim_d2",
        &["d_date_sk", "d_week_seq"],
        16 * 1024 * 1024,
        73_049,
    ));
    let sales_with_d1_schema = schema(&["cs_item_sk", "d_week_seq"]);
    let sales_with_d1 = builder
        .add_join(
            catalog_sales,
            filtered_d1,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "cs_sold_date_sk", "cs_sold_date_sk"),
                    ColumnRef::input(1, "d_date_sk", "d_date_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "cs_item_sk", "cs_item_sk"), "cs_item_sk"),
                JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&sales_with_d1_schema),
        )
        .unwrap();
    let inventory_with_d2_schema = schema(&["inv_item_sk", "d_week_seq"]);
    let inventory_with_d2 = builder
        .add_join(
            inventory,
            d2,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "inv_date_sk", "inv_date_sk"),
                    ColumnRef::input(1, "d_date_sk", "d_date_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "inv_item_sk", "inv_item_sk"), "inv_item_sk"),
                JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&inventory_with_d2_schema),
        )
        .unwrap();
    let root = builder
        .add_join(
            sales_with_d1,
            inventory_with_d2,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "cs_item_sk", "cs_item_sk"),
                    ColumnRef::input(1, "inv_item_sk", "inv_item_sk"),
                )],
            )
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "d_week_seq", "d_week_seq"),
                BinaryOp::Eq,
                Expr::input_column(1, "d_week_seq", "d_week_seq"),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "cs_item_sk", "cs_item_sk"),
                "cs_item_sk",
            )])
            .with_build_side(JoinSide::Right),
            schema(&["cs_item_sk"]),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(root).unwrap();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    let route = result
        .plan
        .dynamic_filter_routes()
        .iter()
        .find(|route| {
            matches!(
                route.producer,
                DynamicFilterProducer::HashJoinBuild {
                    join_node_id,
                    build_side: JoinSide::Right,
                    ..
                } if join_node_id == sales_with_d1
            ) && matches!(
                route.consumer,
                DynamicFilterConsumer::LocalParquetSource {
                    source_node_id
                } if source_node_id == d2
            ) && route.producer_columns == ["d_week_seq".to_string()]
                && route.consumer_columns == ["d_week_seq".to_string()]
        })
        .expect("filtered d1 week sequence should dynamically filter the d2 source");
    assert_eq!(route.consumer_reduction_stage(), DynamicFilterConsumerReductionStage::SourceReader);
    assert!(result.diagnostics.dynamic_filter_route_discovery.candidates.iter().any(|candidate| {
        candidate.candidate_node_id == root
            && candidate.routed
            && candidate.producer_input_node_id == Some(filtered_d1)
            && candidate.consumer_source_node_id == Some(d2)
            && candidate.producer_key_column.as_deref() == Some("d_week_seq")
            && candidate.consumer_key_column.as_deref() == Some("d_week_seq")
    }));
}

#[test]
fn native_optimizer_rejects_residual_equality_route_when_producer_lineage_crosses_shared_path() {
    let mut builder = QueryPlanBuilder::new();
    let catalog_sales = builder.add_source(estimated_local_source_with_fields(
        "shared_catalog_sales",
        &["cs_sold_date_sk", "cs_item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let filtered_d1 = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "shared_date_dim_d1",
        "shared_date_dim_d1",
        &["d_date_sk", "d_week_seq", "d_year"],
        1024 * 1024,
        365,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1999))),
    ));
    let inventory = builder.add_source(estimated_local_source_with_fields(
        "shared_inventory",
        &["inv_item_sk", "inv_date_sk"],
        2 * 1024 * 1024 * 1024,
        400_000_000,
    ));
    let d2 = builder.add_source(estimated_local_source_with_fields(
        "shared_date_dim_d2",
        &["d_date_sk", "d_week_seq"],
        16 * 1024 * 1024,
        73_049,
    ));
    let left_schema = schema(&["cs_item_sk", "d_week_seq"]);
    let sales_with_d1 = builder
        .add_join(
            catalog_sales,
            filtered_d1,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "cs_sold_date_sk", "cs_sold_date_sk"),
                    ColumnRef::input(1, "d_date_sk", "d_date_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "cs_item_sk", "cs_item_sk"), "cs_item_sk"),
                JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&left_schema),
        )
        .unwrap();
    let d1_passthrough = builder
        .add_projection(
            filtered_d1,
            vec![
                ProjectionOutput::new(Expr::column("d_date_sk"), Field::new("cs_item_sk", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false)),
            ],
        )
        .unwrap();
    let left_union = builder.add_union(vec![sales_with_d1, d1_passthrough], Arc::clone(&left_schema)).unwrap();
    let right_schema = schema(&["inv_item_sk", "d_week_seq"]);
    let inventory_with_d2 = builder
        .add_join(
            inventory,
            d2,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "inv_date_sk", "inv_date_sk"),
                    ColumnRef::input(1, "d_date_sk", "d_date_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "inv_item_sk", "inv_item_sk"), "inv_item_sk"),
                JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&right_schema),
        )
        .unwrap();
    let root = builder
        .add_join(
            left_union,
            inventory_with_d2,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "cs_item_sk", "cs_item_sk"),
                    ColumnRef::input(1, "inv_item_sk", "inv_item_sk"),
                )],
            )
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "d_week_seq", "d_week_seq"),
                BinaryOp::Eq,
                Expr::input_column(1, "d_week_seq", "d_week_seq"),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "cs_item_sk", "cs_item_sk"),
                "cs_item_sk",
            )])
            .with_build_side(JoinSide::Right),
            schema(&["cs_item_sk"]),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(root).unwrap();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert!(result.diagnostics.dynamic_filter_route_discovery.candidates.iter().any(|candidate| {
        candidate.candidate_node_id == root
            && !candidate.routed
            && candidate.producer_input_node_id == Some(filtered_d1)
            && candidate.consumer_source_node_id == Some(d2)
            && candidate.producer_key_column.as_deref() == Some("d_week_seq")
            && candidate.consumer_key_column.as_deref() == Some("d_week_seq")
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ProducerLineageSharedPath)
    }));
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
        !matches!(
            route.producer,
            DynamicFilterProducer::HashJoinBuild {
                join_node_id,
                build_side: JoinSide::Right,
                ..
            } if join_node_id == sales_with_d1
        ) || !matches!(
            route.consumer,
            DynamicFilterConsumer::LocalParquetSource {
                source_node_id
            } if source_node_id == d2
        ) || route.producer_columns != ["d_week_seq".to_string()]
            || route.consumer_columns != ["d_week_seq".to_string()]
    }));
}

#[test]
fn native_optimizer_avoids_mutual_residual_equality_dynamic_filter_cycle() {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(estimated_local_source_with_fields(
        "cycle_catalog_sales",
        &["sold_date_sk", "item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let d1 = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "cycle_date_dim_d1",
        "cycle_date_dim_d1",
        &["d_date_sk", "d_week_seq", "d_year"],
        1024 * 1024,
        365,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1999))),
    ));
    let inventory = builder.add_source(estimated_local_source_with_fields(
        "cycle_inventory",
        &["inv_date_sk", "item_sk"],
        2 * 1024 * 1024 * 1024,
        400_000_000,
    ));
    let d2 = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "cycle_date_dim_d2",
        "cycle_date_dim_d2",
        &["d_date_sk", "d_week_seq", "d_year"],
        1024 * 1024,
        365,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
    ));
    let branch_schema = schema(&["item_sk", "d_week_seq"]);
    let sales_with_d1 = builder
        .add_join(
            sales,
            d1,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                    ColumnRef::input(1, "d_date_sk", "d_date_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
                JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&branch_schema),
        )
        .unwrap();
    let inventory_with_d2 = builder
        .add_join(
            inventory,
            d2,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "inv_date_sk", "inv_date_sk"),
                    ColumnRef::input(1, "d_date_sk", "d_date_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk"),
                JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&branch_schema),
        )
        .unwrap();
    let root = builder
        .add_join(
            sales_with_d1,
            inventory_with_d2,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "item_sk", "item_sk"),
                    ColumnRef::input(1, "item_sk", "item_sk"),
                )],
            )
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "d_week_seq", "d_week_seq"),
                BinaryOp::Eq,
                Expr::input_column(1, "d_week_seq", "d_week_seq"),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
            .with_build_side(JoinSide::Right),
            schema(&["item_sk"]),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(root).unwrap();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    let residual_week_routes = result
        .plan
        .dynamic_filter_routes()
        .iter()
        .filter(|route| {
            route.producer_columns == ["d_week_seq".to_string()]
                && route.consumer_columns == ["d_week_seq".to_string()]
                && matches!(
                    route.producer,
                    DynamicFilterProducer::HashJoinBuild {
                        join_node_id,
                        build_side: JoinSide::Right,
                        ..
                    } if join_node_id == sales_with_d1 || join_node_id == inventory_with_d2
                )
                && matches!(
                    route.consumer,
                    DynamicFilterConsumer::LocalParquetSource {
                        source_node_id
                    } if source_node_id == d1 || source_node_id == d2
                )
        })
        .collect::<Vec<_>>();
    assert_eq!(residual_week_routes.len(), 1);
    assert!(matches!(
        residual_week_routes[0].producer,
        DynamicFilterProducer::HashJoinBuild {
            join_node_id,
            build_side: JoinSide::Right,
            ..
        } if join_node_id == sales_with_d1
    ));
    assert!(matches!(
        residual_week_routes[0].consumer,
        DynamicFilterConsumer::LocalParquetSource {
            source_node_id
        } if source_node_id == d2
    ));
    assert!(result.diagnostics.dynamic_filter_route_discovery.candidates.iter().any(|candidate| {
        candidate.candidate_node_id == root
            && !candidate.routed
            && candidate.producer_input_node_id == Some(d2)
            && candidate.consumer_source_node_id == Some(d1)
            && candidate.producer_key_column.as_deref() == Some("d_week_seq")
            && candidate.consumer_key_column.as_deref() == Some("d_week_seq")
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::DynamicFilterControlCycle)
    }));
}
