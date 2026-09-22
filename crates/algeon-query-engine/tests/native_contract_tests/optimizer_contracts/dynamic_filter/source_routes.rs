#[test]
fn native_optimizer_discovers_hash_join_build_to_parquet_source_dynamic_filter_route() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 512 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000);

    let result = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

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
    assert_eq!(candidate.selected_source_bytes, Some(512 * 1024 * 1024));
    assert_eq!(candidate.total_source_bytes, Some(512 * 1024 * 1024));

    let routes = result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1);
    assert_eq!(routes[0].id.index(), 0);
    assert_eq!(routes[0].producer_columns, vec!["right_key".to_string()]);
    assert_eq!(routes[0].consumer_columns, vec!["left_key".to_string()]);
    assert!(matches!(routes[0].producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. }));
    assert!(matches!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { .. }));
    assert_eq!(result.plan.dynamic_filter_execution_options(), DynamicFilterExecutionOptions::default());
}

#[test]
fn dynamic_filter_execution_options_survive_gpu_resident_sink_clone() {
    let execution = DynamicFilterExecutionOptions::default().with_max_build_rows(17);
    let keyed_hash_join_strategy =
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions { enabled: true, execution },
        keyed_hash_join_strategy,
        ..NativeOptimizerOptions::disabled()
    };
    let plan = estimated_inner_join_plan(JoinSide::Right, 512 * 1024 * 1024, 10, 16 * 1024 * 1024, 10);

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );
    let gpu_plan = result.plan.clone_with_gpu_resident_sink();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(gpu_plan.dynamic_filter_routes(), result.plan.dynamic_filter_routes());
    assert_eq!(gpu_plan.dynamic_filter_execution_options(), execution);
    assert_eq!(gpu_plan.keyed_hash_join_strategy_options(), keyed_hash_join_strategy);
}

#[test]
fn native_optimizer_rejects_large_keyset_route_to_dimension_sized_local_parquet_consumer() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 16 * 1024 * 1024, 1_000_000, 1024 * 1024, 100_000);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("left_key"));
    assert!(candidate.consumer_source_node_id.is_some());
    assert_eq!(candidate.consumer_kind, None);
    assert_eq!(candidate.consumer_node_id, None);
    assert_eq!(candidate.consumer_lowering, None);
    assert_eq!(candidate.selected_source_bytes, Some(16 * 1024 * 1024));
    assert_eq!(
        candidate.rejected_reason,
        Some(DynamicFilterRouteRejectedReason::LargeKeysetDimensionSizedLocalParquetConsumer)
    );
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_keeps_large_keyset_source_reader_route_when_downstream_work_fans_out() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "skinny_fact_with_shared_aggregate",
        &["fact_key", "store_key"],
        16 * 1024 * 1024,
        1_000_000,
    ));
    let dim = builder.add_source(estimated_local_source("large_dim_keyset", "dim_key", 1024 * 1024, 100_000));
    let joined = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "store_key", "store_key"),
            "store_key",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["store_key"]),
    );
    let grouped = builder.add_aggregate(
        joined,
        AggregateSpec::new(vec![Expr::column("store_key")], Vec::new()),
        schema(&["store_key"]),
    );
    let left = builder.add_projection(
        grouped,
        vec![ProjectionOutput::new(Expr::column("store_key"), Field::new("left_store_key", DataType::Int32, false))],
    );
    let right = builder.add_projection(
        grouped,
        vec![ProjectionOutput::new(Expr::column("store_key"), Field::new("right_store_key", DataType::Int32, false))],
    );
    let root = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_store_key", "left_store_key"),
                ColumnRef::input(1, "right_store_key", "right_store_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "left_store_key", "left_store_key"),
            "left_store_key",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["left_store_key"]),
    );
    let plan = builder.finish_gpu_resident(root);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    let candidate = result
        .diagnostics
        .dynamic_filter_route_discovery
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == joined && candidate.consumer_source_node_id == Some(fact))
        .expect("initial join should keep a source-reader dynamic-filter candidate");
    assert!(candidate.routed);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::ReaderPredicate));
    assert_eq!(candidate.consumer_reduction_stage, Some(DynamicFilterConsumerReductionStage::SourceReader));
    assert!(result.plan.dynamic_filter_routes().iter().any(|route| {
        route.consumer_columns == ["fact_key".to_string()]
            && matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == fact)
    }));
}

#[test]
fn native_optimizer_rejects_dynamic_filter_route_when_producer_keyset_exceeds_runtime_limit() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 512 * 1024 * 1024, 1_000_000, 64 * 1024 * 1024, 1_000_001);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ProducerKeysetExceedsRuntimeLimit));
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_uses_custom_dynamic_filter_runtime_limit_for_route_discovery() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 512 * 1024 * 1024, 1_000_000, 64 * 1024 * 1024, 12);
    let execution = DynamicFilterExecutionOptions::default().with_max_build_rows(11);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions { execution, ..Default::default() },
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::ProducerKeysetExceedsRuntimeLimit));
    assert_eq!(result.plan.dynamic_filter_execution_options(), execution);
}

#[test]
fn native_optimizer_keeps_filtered_producer_when_unfiltered_row_estimate_exceeds_runtime_limit() {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", 512 * 1024 * 1024, 10_000_000));
    let right = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "right_filtered",
        "right_filtered",
        &["right_key"],
        64 * 1024 * 1024,
        1_000_001,
        Expr::binary(Expr::column("right_key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
    ));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(candidate.consumer_source_node_id, Some(left));
    assert_eq!(result.plan.dynamic_filter_routes().len(), 1);
}

#[test]
fn native_optimizer_rejects_large_keyset_dimension_route_when_producer_is_larger_than_many_row_groups_consumer() {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_row_group_count(
        "left_many_row_groups",
        &["left_key"],
        64,
        2 * 1024 * 1024,
        16_000,
    ));
    let right =
        builder.add_source(estimated_local_source("right_large_keyset", "right_key", 512 * 1024 * 1024, 1_000_000));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert_eq!(candidate.selected_source_bytes, Some(128 * 1024 * 1024));
    assert_eq!(
        candidate.rejected_reason,
        Some(DynamicFilterRouteRejectedReason::LargeKeysetDimensionSizedLocalParquetConsumer)
    );
    assert!(result.plan.dynamic_filter_routes().is_empty());
}

#[test]
fn native_optimizer_keeps_large_keyset_dimension_route_when_many_row_groups_consumer_is_larger_than_producer() {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_row_group_count(
        "left_many_row_groups_small_producer",
        &["left_key"],
        64,
        2 * 1024 * 1024,
        16_000,
    ));
    let right =
        builder.add_source(estimated_local_source("right_smaller_keyset", "right_key", 64 * 1024 * 1024, 1_000_000));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.selected_source_bytes, Some(128 * 1024 * 1024));
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(result.plan.dynamic_filter_routes().len(), 1);
}

#[test]
fn native_optimizer_keeps_dimension_sized_consumer_route_when_producer_keyset_is_small() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 16 * 1024 * 1024, 1_000_000, 64 * 1024 * 1024, 4 * 1024);
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(candidate.routed);
    assert_eq!(candidate.consumer_key_column.as_deref(), Some("left_key"));
    assert_eq!(candidate.consumer_kind.as_deref(), Some("local_parquet_source"));
    assert_eq!(candidate.consumer_lowering, Some(DynamicFilterConsumerLowering::ReaderPredicate));
    assert_eq!(candidate.selected_source_bytes, Some(16 * 1024 * 1024));
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(result.plan.dynamic_filter_routes().len(), 1);
}

#[test]
fn native_optimizer_discovers_dynamic_filter_route_for_inner_keyed_residual_join() {
    let plan =
        estimated_inner_residual_join_plan(JoinSide::Right, 512 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000);

    let result = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

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
