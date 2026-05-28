use cudf_query_engine::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownOptions,
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
    DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DimensionDerivedRowGroupPruningOptions,
    DimensionDerivedRowGroupPruningRejectedReason, DynamicFilter, DynamicFilterConsumer, DynamicFilterConsumerLowering,
    DynamicFilterConsumerReductionStage, DynamicFilterKind, DynamicFilterProducer, DynamicFilterRouteDiscoveryOptions,
    DynamicFilterRouteRejectedReason, DynamicFilterSourceSnapshot, JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
    JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinBuildSideSelectionOptions, LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
    LocalParquetSourceReuseOptions, NativeOptimizerChange, NativeOptimizerOptions, NodeId,
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionRejectedReason, PROJECTION_SIMPLIFICATION_RULE_NAME,
    ProjectionSimplificationOptions, SEMIJOIN_REDUCTION_RULE_NAME, SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
    STREAM_FANOUT_REPLAY_RULE_NAME, SemijoinReductionAdmissionMode, SemijoinReductionCostSignal,
    SemijoinReductionOptions, SemijoinReductionRejectedReason, SharedSubplanMaterializationOptions,
    StreamFanoutReplayOptions, optimize_query_plan,
};

use super::*;

#[test]
fn native_optimizer_phase1_semijoin_entry_point_returns_unchanged_plan_and_empty_diagnostics() {
    let plan = single_source_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(!result.changed());
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert!(result.diagnostics.semijoin_reduction.candidates.is_empty());
}

#[test]
fn native_optimizer_options_can_disable_semijoin_reduction_pass() {
    let plan = single_source_plan();
    let options = NativeOptimizerOptions::disabled();

    let result = optimize_query_plan(plan.clone(), &options);

    assert!(!options.projection_simplification.enabled);
    assert!(!options.semijoin_reduction.enabled);
    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn native_optimizer_flips_inner_equi_join_to_build_smaller_estimated_left_side() {
    let plan = estimated_inner_equi_join_plan(16 * 1024 * 1024, 256 * 1024 * 1024);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner join");
    };
    assert_eq!(join.spec.join_type, JoinType::Inner);
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn native_optimizer_flips_inner_keyed_residual_join_to_build_smaller_estimated_left_side() {
    let plan =
        estimated_inner_residual_join_plan(JoinSide::Right, 16 * 1024 * 1024, 100_000, 256 * 1024 * 1024, 10_000_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner residual join");
    };
    assert_eq!(join.spec.join_type, JoinType::Inner);
    assert!(join.spec.residual_predicate.is_some());
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn native_optimizer_disabled_keeps_default_inner_equi_join_build_side() {
    let plan = estimated_inner_equi_join_plan(16 * 1024 * 1024, 256 * 1024 * 1024);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::disabled());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn projection_simplification_options_and_rule_name_are_stable() {
    assert_eq!(PROJECTION_SIMPLIFICATION_RULE_NAME, "projection_simplification");
    assert!(ProjectionSimplificationOptions::default().enabled);
    assert!(!ProjectionSimplificationOptions::disabled().enabled);
}

#[test]
fn semijoin_reduction_phase1_options_and_rule_name_are_stable() {
    assert_eq!(SEMIJOIN_REDUCTION_RULE_NAME, "semijoin_reduction");
    assert!(SemijoinReductionOptions::default().enabled);
    assert_eq!(
        SemijoinReductionOptions::default().cost_threshold_multiplier,
        DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER
    );
    assert_eq!(SemijoinReductionOptions::default().admission_mode, SemijoinReductionAdmissionMode::NotEvaluated);
    assert!(!SemijoinReductionOptions::disabled().enabled);
}

#[test]
fn join_build_side_selection_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_BUILD_SIDE_SELECTION_RULE_NAME, "join_build_side_selection");
    assert!(JoinBuildSideSelectionOptions::default().enabled);
    assert!(!JoinBuildSideSelectionOptions::disabled().enabled);
    assert_eq!(JoinBuildSideSelectionOptions::default().small_build_side_bytes, 64 * 1024 * 1024);
    assert_eq!(JoinBuildSideSelectionOptions::default().small_build_side_rows, 1_000_000);
}

#[test]
fn aggregate_join_pushdown_options_and_rule_name_are_stable() {
    assert_eq!(AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, "aggregate_join_pushdown");
    assert!(AggregateJoinPushdownOptions::default().enabled);
    assert!(!AggregateJoinPushdownOptions::disabled().enabled);
}

#[test]
fn outer_join_null_rejection_options_and_rule_name_are_stable() {
    assert_eq!(OUTER_JOIN_NULL_REJECTION_RULE_NAME, "outer_join_null_rejection");
    assert!(cudf_query_engine::OuterJoinNullRejectionOptions::default().enabled);
    assert!(!cudf_query_engine::OuterJoinNullRejectionOptions::disabled().enabled);
}

#[test]
fn join_filter_reassociation_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_FILTER_REASSOCIATION_RULE_NAME, "join_filter_reassociation");
    assert!(cudf_query_engine::JoinFilterReassociationOptions::default().enabled);
    assert!(cudf_query_engine::JoinFilterReassociationOptions::default().residual_filter_pushdown_enabled);
    assert!(!cudf_query_engine::JoinFilterReassociationOptions::disabled().enabled);
    assert!(!cudf_query_engine::JoinFilterReassociationOptions::disabled().residual_filter_pushdown_enabled);
}

#[test]
fn dimension_derived_row_group_pruning_options_and_rule_name_are_stable() {
    assert_eq!(DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME, "dimension_derived_row_group_pruning");
    assert!(DimensionDerivedRowGroupPruningOptions::default().enabled);
    assert_eq!(DimensionDerivedRowGroupPruningOptions::default().max_dimension_rows, 1_000_000);
    assert!(!DimensionDerivedRowGroupPruningOptions::disabled().enabled);
}

#[test]
fn stream_fanout_replay_options_and_rule_name_are_stable() {
    assert_eq!(STREAM_FANOUT_REPLAY_RULE_NAME, "stream_fanout_replay");
    assert!(StreamFanoutReplayOptions::default().enabled);
    assert!(!StreamFanoutReplayOptions::disabled().enabled);
}

#[test]
fn local_parquet_source_reuse_options_and_rule_name_are_stable() {
    assert_eq!(LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, "local_parquet_source_reuse");
    assert!(LocalParquetSourceReuseOptions::default().enabled);
    assert!(!LocalParquetSourceReuseOptions::disabled().enabled);
}

#[test]
fn shared_subplan_materialization_options_and_rule_name_are_stable() {
    assert_eq!(SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, "shared_subplan_materialization");
    assert!(SharedSubplanMaterializationOptions::default().enabled);
    assert!(!SharedSubplanMaterializationOptions::disabled().enabled);
}

#[test]
fn dynamic_filter_route_discovery_options_and_rule_name_are_stable() {
    assert_eq!(DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, "dynamic_filter_route_discovery");
    assert!(DynamicFilterRouteDiscoveryOptions::default().enabled);
    assert!(!DynamicFilterRouteDiscoveryOptions::disabled().enabled);
}

#[test]
fn native_optimizer_discovers_repeated_grouped_aggregate_materialization_family() {
    let plan = repeated_grouped_aggregate_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 3);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 3);
    assert_eq!(candidate.aggregate_node_ids.len(), 3);
    assert_eq!(candidate.group_exprs, vec!["county".to_string(), "quarter".to_string(), "year".to_string()]);
    assert_eq!(candidate.exact_source_variant_count, 3);
    assert_eq!(candidate.repeated_selected_source_bytes, Some(3 * 512 * 1024 * 1024));
    assert_eq!(candidate.duplicate_selected_source_bytes, Some(2 * 512 * 1024 * 1024));
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).count(), 3);
    assert_eq!(result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).count(), 1);
    assert_eq!(result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Filter(_))).count(), 3);
    let source_predicates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.reader_predicate.as_ref().map(ToString::to_string),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(source_predicates.len(), 1);
    assert!(source_predicates[0].contains(" OR "));
    assert!(source_predicates[0].contains("quarter = 1"));
    assert!(source_predicates[0].contains("quarter = 2"));
    assert!(source_predicates[0].contains("quarter = 3"));
}

#[test]
fn native_optimizer_combines_repeated_scalar_count_branches_with_filtered_aggregate() {
    let plan = repeated_scalar_count_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert!(candidate.group_exprs.is_empty());
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 2);
    assert!(aggregates[0].spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    let plan_text = result.plan.to_string();
    assert!(plan_text.contains("t_hour = 8"));
    assert!(plan_text.contains("t_hour = 9"));
    let source_predicates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.reader_predicate.as_ref().map(ToString::to_string),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(source_predicates.iter().any(|predicate| predicate.contains(" OR ")));
    assert!(result.plan.to_string().contains("__cudf_shared_scalar_count"));
}

#[test]
fn disabled_shared_subplan_materialization_emits_no_diagnostics() {
    let plan = repeated_grouped_aggregate_materialization_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.shared_subplan_materialization.is_empty());
}

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
    assert!(result.plan.to_string().contains("DynamicFilterRoute"));
}

#[test]
fn native_optimizer_rejects_dynamic_filter_route_for_inner_keyed_residual_join() {
    let plan =
        estimated_inner_residual_join_plan(JoinSide::Right, 256 * 1024 * 1024, 10_000_000, 16 * 1024 * 1024, 100_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.considered_count, 1);
    assert_eq!(result.diagnostics.dynamic_filter_route_discovery.route_count, 0);
    let candidate = &result.diagnostics.dynamic_filter_route_discovery.candidates[0];
    assert!(!candidate.routed);
    assert_eq!(candidate.rejected_reason, Some(DynamicFilterRouteRejectedReason::UnsupportedJoinShape));
    assert!(result.plan.dynamic_filter_routes().is_empty());
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
fn native_optimizer_rejects_source_dynamic_filter_when_consumer_lineage_crosses_shared_subplan() {
    let (plan, first_branch_join, second_branch_join) = shared_prefix_branch_local_dynamic_filter_plan();
    let options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);
    let diagnostics = &result.diagnostics.dynamic_filter_route_discovery;

    assert!(diagnostics.candidates.iter().any(|candidate| {
        candidate.candidate_node_id == first_branch_join
            && !candidate.routed
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
    }));
    assert!(diagnostics.candidates.iter().any(|candidate| {
        candidate.candidate_node_id == second_branch_join
            && !candidate.routed
            && candidate.rejected_reason == Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan)
    }));
    assert!(result.plan.dynamic_filter_routes().iter().all(|route| {
        route.producer.node_id() != first_branch_join && route.producer.node_id() != second_branch_join
    }));
}

#[test]
fn dimension_pruning_rejects_shared_target_lineage_before_mutating_source_row_groups() {
    let plan = shared_fact_target_dimension_pruning_plan();
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    let diagnostics = &result.diagnostics.dimension_derived_row_group_pruning;
    assert_eq!(diagnostics.considered_count, 4);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.target_key_column.as_deref() == Some("fact_key")
                    && candidate.rejected_reason
                        == Some(DimensionDerivedRowGroupPruningRejectedReason::TargetLineageReusedInPlan)
            })
            .count(),
        2
    );
    assert!(diagnostics.candidates.iter().all(|candidate| !candidate.applied));
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
fn native_optimizer_replays_shared_projected_probe_input_for_join_fanout() {
    let plan = shared_projected_probe_fanout_plan();
    let original_node_count = plan.node_count();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.stream_fanout_replay.considered_count, 1);
    assert_eq!(result.diagnostics.stream_fanout_replay.applied_count, 1);
    let candidate = &result.diagnostics.stream_fanout_replay.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.consumer_count, 2);
    assert_eq!(candidate.rejected_reason, None);
    assert!(result.plan.node_count() > original_node_count);
    assert_eq!(max_direct_join_probe_consumer_count(&result.plan), 1);
}

#[test]
fn native_optimizer_reuses_compatible_small_local_parquet_source_variants() {
    let plan = compatible_small_local_parquet_source_variants_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    let candidate = &result.diagnostics.local_parquet_source_reuse.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.reuse_count, 2);
    assert_eq!(candidate.branch_filter_count, 1);
    assert_eq!(candidate.superset_read_columns, vec!["id", "state", "zip"]);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address"), 1);
    assert!(
        result.plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Filter(_))),
        "reader predicate should be replayed as a branch-local native filter"
    );
}

#[test]
fn native_optimizer_keeps_large_local_parquet_source_variants_independent() {
    let plan = large_local_parquet_source_variants_plan();
    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(cudf_query_engine::LocalParquetSourceReuseRejectedReason::VariantTooLarge)
    );
}

#[test]
fn disabled_native_optimizer_keeps_stream_fanout_replay_off() {
    let plan = shared_projected_probe_fanout_plan();
    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.stream_fanout_replay.is_empty());
}

#[test]
fn outer_join_null_rejection_converts_left_join_when_inner_join_key_rejects_right_nulls() {
    let plan = inner_join_over_projected_left_join_right_key_plan();
    let options = NativeOptimizerOptions {
        join_filter_reassociation: cudf_query_engine::JoinFilterReassociationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);
    assert!(result.diagnostics.outer_join_null_rejection.candidates[0].applied);
    assert_eq!(result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason, None);

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("root should remain an inner join");
    };
    let PlanNode::Projection(projection) = &result.plan.node(root_join.inputs[0]).expect("projection").kind else {
        panic!("root left input should remain a projection");
    };
    let PlanNode::Join(rewritten_join) = &result.plan.node(projection.input).expect("left join").kind else {
        panic!("projection input should be rewritten join");
    };
    assert_eq!(rewritten_join.spec.join_type, JoinType::Inner);
}

#[test]
fn outer_join_null_rejection_keeps_left_join_when_key_comes_from_preserved_side() {
    let plan = inner_join_over_projected_left_join_left_key_plan();

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

#[test]
fn outer_join_null_rejection_rejects_shared_outer_join_output() {
    let plan = shared_left_join_output_with_one_null_rejecting_consumer_plan();

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 0);
    assert_eq!(
        result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason,
        Some(OuterJoinNullRejectionRejectedReason::SharedOuterJoinOutput)
    );
}

#[test]
fn join_filter_reassociation_pushes_dimension_filter_below_fact_join_without_losing_multiplicity() {
    let plan = inner_join_filter_over_projected_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_filter_reassociation.applied_count, 1);
    let candidate = &result.diagnostics.join_filter_reassociation.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let PlanNode::Projection(parent_projection) = &result.plan.root_node().kind else {
        panic!("rewritten parent join should become a projection preserving the original output schema");
    };
    let PlanNode::Projection(case_projection) =
        &result.plan.node(parent_projection.input).expect("case projection").kind
    else {
        panic!("parent projection should consume rebuilt case projection");
    };
    let PlanNode::Join(fact_join) = &result.plan.node(case_projection.input).expect("rebuilt fact join").kind else {
        panic!("case projection should consume rebuilt fact join");
    };
    assert_eq!(fact_join.spec.join_type, JoinType::Inner);
    let PlanNode::Join(filter_join) = &result.plan.node(fact_join.inputs[1]).expect("filter join").kind else {
        panic!("fact join right input should be prefiltered by dimension");
    };
    assert_eq!(filter_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, filter_join.inputs[0], "returns");
    assert_source_input(&result.plan, filter_join.inputs[1], "reason");
}

#[test]
fn join_filter_reassociation_repeats_until_filter_reaches_base_fact_input() {
    let plan = nested_inner_join_filter_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_filter_reassociation.applied_count, 2);
    assert!(
        result
            .diagnostics
            .join_filter_reassociation
            .candidates
            .iter()
            .filter(|candidate| candidate.applied)
            .all(|candidate| candidate.rejected_reason.is_none())
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            let PlanNode::Join(join) = &node.kind else { return false };
            join.spec.join_type == JoinType::Inner
                && join.spec.keys.len() == 1
                && source_input_name(&result.plan, join.inputs[0]).as_deref() == Some("sales")
                && source_input_name(&result.plan, join.inputs[1]).as_deref() == Some("date")
        }),
        "fixed-point reassociation should push the selective date join down to the sales source"
    );
}

#[test]
fn join_build_side_selection_switches_inner_equi_join_to_small_left_input() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 1_024, 100, 9 * 1024 * 1024 * 1024, 100_000_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert!(result.diagnostics.semijoin_reduction.is_empty());
    assert_eq!(result.diagnostics.join_build_side_selection.considered_count, 1);
    assert_eq!(result.diagnostics.join_build_side_selection.applied_count, 1);
    let candidate = &result.diagnostics.join_build_side_selection.candidates[0];
    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert!(candidate.applied);
    assert_eq!(candidate.left_estimated_bytes, Some(1_024));
    assert_eq!(candidate.right_estimated_bytes, Some(9 * 1024 * 1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain an inner join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn join_build_side_selection_prefers_grouped_aggregate_keyset_side_when_size_is_unknown() {
    let mut builder = QueryPlanBuilder::new();
    let raw_keyset =
        builder.add_source(estimated_local_source("returns", "raw_customer_sk", 512 * 1024 * 1024, 10_000_000));
    let aggregate = builder.add_aggregate(
        raw_keyset,
        AggregateSpec::new(vec![Expr::column("raw_customer_sk")], Vec::new()),
        schema(&["raw_customer_sk"]),
    );
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
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
            "ctr_customer_sk",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["ctr_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("join candidate should be considered");
    assert!(candidate.applied);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn join_build_side_selection_does_not_prefer_grouped_aggregate_side_when_join_key_is_measure() {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(estimated_sales_source_with_quarter_predicate(1, 512 * 1024 * 1024));
    let aggregate = builder.add_aggregate(
        sales,
        AggregateSpec::new(
            vec![Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        schema(&["county", "store_sales"]),
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let join = builder.add_join(
        aggregate,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "store_sales", "store_sales"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            "c_customer_sk",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["c_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("join candidate should be considered");
    assert!(!candidate.applied);
    assert_eq!(candidate.after_build_side, None);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn disabled_native_optimizer_keeps_join_build_side_selection_off() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 1_024, 100, 9 * 1024 * 1024 * 1024, 100_000_000);

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.is_empty());
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

#[test]
fn semijoin_reduction_pushes_left_semi_into_left_inner_join_child() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Left);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.crossed_join_node_ids.len(), 1);
    assert_eq!(
        candidate.cost_gate.as_ref().map(|gate| &gate.signal),
        Some(&SemijoinReductionCostSignal::AggregateKeysetShape)
    );
    assert_eq!(candidate.key_mappings[0].before_column, "x_key");
    assert_eq!(candidate.key_mappings[0].after_column.as_deref(), Some("x_key"));

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the crossed inner join");
    };
    assert_eq!(root_join.spec.join_type, JoinType::Inner);
    assert_left_semi_input(&result.plan, root_join.inputs[0], "x_key");
    assert_source_input(&result.plan, root_join.inputs[1], "y");
}

#[test]
fn semijoin_reduction_pushes_left_semi_into_right_inner_join_child() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the crossed inner join");
    };
    assert_eq!(root_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, root_join.inputs[0], "x");
    assert_left_semi_input(&result.plan, root_join.inputs[1], "y_key");
}

#[test]
fn semijoin_reduction_pushes_to_deepest_safe_inner_join_subtree() {
    let plan = root_semijoin_over_nested_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].crossed_join_node_ids.len(), 2);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the outer crossed inner join");
    };
    assert_eq!(root_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, root_join.inputs[1], "c");

    let PlanNode::Join(inner_join) = &result.plan.node(root_join.inputs[0]).expect("inner join").kind else {
        panic!("outer left input should be the rebuilt inner join");
    };
    assert_eq!(inner_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, inner_join.inputs[0], "a");
    assert_left_semi_input(&result.plan, inner_join.inputs[1], "b_key");
}

#[test]
fn semijoin_reduction_rejects_unestimated_non_keyset_shape() {
    let plan = root_semijoin_over_inner_join_plan_with_raw_keyset(SemijoinTarget::Right);

    let result = optimize_query_plan(plan, &semijoin_reduction_only_options(SemijoinReductionOptions::default()));

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.rejected_reason, Some(SemijoinReductionRejectedReason::CostGateRejected));
    assert_eq!(candidate.cost_gate.as_ref().map(|gate| &gate.signal), Some(&SemijoinReductionCostSignal::Unavailable));
}

#[test]
fn semijoin_reduction_strict_admission_gate_keeps_original_when_rewrite_is_not_admitted() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Right);
    let options = semijoin_reduction_only_options(
        SemijoinReductionOptions::default().with_admission_mode(SemijoinReductionAdmissionMode::StrictBounded),
    );

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.rejected_reason, Some(SemijoinReductionRejectedReason::AdmissionRejectedRewrite));
    assert_eq!(candidate.original_admitted, Some(false));
    assert_eq!(candidate.rewritten_admitted, Some(false));
}

#[test]
fn semijoin_reduction_rejects_outer_crossed_joins() {
    for join_type in [JoinType::Left, JoinType::Full] {
        assert_semijoin_rejection(
            root_semijoin_over_crossed_join_type_plan(join_type),
            SemijoinReductionRejectedReason::UnsupportedCrossedJoinType,
        );
    }
}

#[test]
fn semijoin_reduction_rejects_ambiguous_lineage_candidate() {
    assert_semijoin_rejection(
        root_semijoin_over_duplicate_output_name_plan(),
        SemijoinReductionRejectedReason::AmbiguousKeyLineage,
    );
}

#[test]
fn semijoin_reduction_rejects_expression_derived_key_candidate() {
    assert_semijoin_rejection(
        root_semijoin_over_expression_key_projection_plan(),
        SemijoinReductionRejectedReason::ExpressionDerivedKey,
    );
}

#[test]
fn semijoin_reduction_rejects_dropped_key_lineage_candidate() {
    assert_semijoin_rejection(
        root_semijoin_over_dropped_key_projection_plan(),
        SemijoinReductionRejectedReason::DroppedKeyLineage,
    );
}

#[test]
fn semijoin_reduction_rejects_unsupported_null_semantics() {
    assert_semijoin_rejection(
        root_semijoin_over_inner_join_plan_with_null_equality(
            JoinNullEquality::NullsMatch,
            JoinNullEquality::NullsDoNotMatch,
        ),
        SemijoinReductionRejectedReason::UnsupportedNullEquality,
    );
    assert_semijoin_rejection(
        root_semijoin_over_inner_join_plan_with_null_equality(
            JoinNullEquality::NullsDoNotMatch,
            JoinNullEquality::NullsMatch,
        ),
        SemijoinReductionRejectedReason::UnsupportedNullEquality,
    );
}

fn single_source_plan() -> cudf_query_engine::QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source)
}

fn estimated_inner_equi_join_plan(left_bytes: u64, right_bytes: u64) -> cudf_query_engine::QueryPlan {
    estimated_inner_join_plan(JoinSide::Right, left_bytes, 1, right_bytes, 1)
}

fn estimated_inner_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", left_bytes, left_rows));
    let right = builder.add_source(estimated_local_source("right", "right_key", right_bytes, right_rows));
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
        .with_build_side(build_side),
        output_schema,
    );
    builder.finish_gpu_resident(join)
}

fn estimated_inner_residual_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_fields(
        "left_residual",
        &["left_key", "left_payload"],
        left_bytes,
        left_rows,
    ));
    let right = builder.add_source(estimated_local_source_with_fields(
        "right_residual",
        &["right_key", "right_payload"],
        right_bytes,
        right_rows,
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
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_payload", "left_payload"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_payload", "right_payload"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(build_side),
        output_schema,
    );
    builder.finish_gpu_resident(join)
}

fn build_subtree_source_reuse_dynamic_filter_cycle_plan() -> cudf_query_engine::QueryPlan {
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

fn repeated_consumer_file_set_dynamic_filter_plan() -> cudf_query_engine::QueryPlan {
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

fn shared_prefix_branch_local_dynamic_filter_plan() -> (cudf_query_engine::QueryPlan, NodeId, NodeId) {
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

fn repeated_grouped_aggregate_materialization_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_schema = schema(&["county", "quarter", "year", "store_sales"]);
    let mut aggregates = Vec::new();
    for quarter in 1..=3 {
        let source = builder.add_source(estimated_sales_source_with_quarter_predicate(quarter, 512 * 1024 * 1024));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_scalar_count_materialization_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for hour in [8, 9] {
        let fact =
            builder.add_source(estimated_local_source("store_sales", "ss_sold_time_sk", 512 * 1024 * 1024, 10_000_000));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let time_key = builder.add_projection(
            time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let joined = builder.add_join(
            fact,
            time_key,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                    ColumnRef::input(1, "t_time_sk", "t_time_sk"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                "ss_sold_time_sk",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&fact_schema),
        );
        let count = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count")],
            ),
            Arc::clone(&count_schema),
        );
        counts.push(count);
    }
    let union = builder.add_union(counts, count_schema);
    builder.finish_gpu_resident(union)
}

fn shared_fact_target_dimension_pruning_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source("store_sales", "fact_key", 512 * 1024 * 1024, 10_000_000));
    let output_schema = schema(&["dim_key", "fact_key"]);
    let mut joins = Vec::new();

    for (name, day) in [("date_dim_a", 1), ("date_dim_b", 2)] {
        let dim = builder.add_source(estimated_date_source_with_day_predicate(name, day));
        let join = builder.add_join(
            dim,
            fact,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "dim_key", "dim_key"),
                    ColumnRef::input(1, "fact_key", "fact_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "dim_key", "dim_key"), "dim_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "fact_key", "fact_key"), "fact_key"),
            ]),
            Arc::clone(&output_schema),
        );
        joins.push(join);
    }

    let union = builder.add_union(joins, output_schema);
    builder.finish_gpu_resident(union)
}

fn estimated_sales_source_with_quarter_predicate(quarter: i32, bytes: u64) -> SourceNode {
    let source_schema = schema(&["county", "quarter", "year", "sales"]);
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![10_000_000]);
    let predicate = Expr::binary(Expr::column("quarter"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(quarter)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["county".to_string(), "quarter".to_string(), "year".to_string(), "sales".to_string()],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_time_source_with_hour_predicate(hour: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("t_time_sk", DataType::Int32, false),
        Field::new("t_hour", DataType::Int32, false),
    ]));
    let file = LocalParquetFile::new("/tmp/time_dim.parquet")
        .with_stable_id("time_dim")
        .with_projected_row_group_bytes(vec![1024 * 1024], vec![1024 * 1024])
        .with_row_group_rows(vec![86_400]);
    let predicate = Expr::binary(Expr::column("t_hour"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(hour)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["t_time_sk".to_string(), "t_hour".to_string()])
            .with_column_roles(vec!["t_time_sk".to_string()], vec!["t_hour".to_string()])
            .with_reader_predicate(predicate),
    )
}

fn estimated_date_source_with_day_predicate(name: &str, day: i32) -> SourceNode {
    let source_schema = schema(&["dim_key", "day"]);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(name)
        .with_projected_row_group_bytes(vec![1024 * 1024], vec![1024 * 1024])
        .with_row_group_rows(vec![365]);
    let predicate = Expr::binary(Expr::column("day"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(day)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["dim_key".to_string(), "day".to_string()])
            .with_reader_predicate(predicate),
    )
}

fn estimated_local_source(name: &str, key_name: &str, bytes: u64, rows: u64) -> SourceNode {
    estimated_local_source_with_fields(name, &[key_name], bytes, rows)
}

fn estimated_local_source_with_fields(name: &str, field_names: &[&str], bytes: u64, rows: u64) -> SourceNode {
    estimated_local_source_with_stable_id(name, name, field_names, bytes, rows)
}

fn estimated_local_source_with_stable_id(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(stable_id)
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![rows]);
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        source_schema,
        field_names.iter().map(|name| (*name).to_string()).collect(),
    ))
}

#[derive(Debug, Clone, Copy)]
enum SemijoinTarget {
    Left,
    Right,
}

fn root_semijoin_over_inner_join_plan(target: SemijoinTarget) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, true)
}

fn root_semijoin_over_inner_join_plan_with_raw_keyset(target: SemijoinTarget) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, false)
}

fn root_semijoin_over_inner_join_plan_impl(
    target: SemijoinTarget,
    aggregate_keyset: bool,
) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        target,
        aggregate_keyset,
        JoinType::Inner,
        JoinNullEquality::NullsDoNotMatch,
        JoinNullEquality::NullsDoNotMatch,
    )
}

fn root_semijoin_over_inner_join_plan_with_null_equality(
    semijoin_null_equality: JoinNullEquality,
    crossed_join_null_equality: JoinNullEquality,
) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        SemijoinTarget::Right,
        true,
        JoinType::Inner,
        semijoin_null_equality,
        crossed_join_null_equality,
    )
}

fn root_semijoin_over_crossed_join_type_plan(join_type: JoinType) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        SemijoinTarget::Right,
        true,
        join_type,
        JoinNullEquality::NullsDoNotMatch,
        JoinNullEquality::NullsDoNotMatch,
    )
}

fn root_semijoin_over_inner_join_plan_with_options(
    target: SemijoinTarget,
    aggregate_keyset: bool,
    crossed_join_type: JoinType,
    semijoin_null_equality: JoinNullEquality,
    crossed_join_null_equality: JoinNullEquality,
) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key", "x_payload"]));
    let y = builder.add_source(gpu_source("y", &["y_key", "y_payload"]));
    let key_name = match target {
        SemijoinTarget::Left => "x_key",
        SemijoinTarget::Right => "y_key",
    };
    let keyset = if aggregate_keyset {
        aggregate_keyset_source(&mut builder, "keyset", key_name)
    } else {
        builder.add_source(gpu_source("keyset", &[key_name]))
    };
    let xy_schema = schema(&["x_key", "x_payload", "y_key", "y_payload"]);
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            crossed_join_type,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "x_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "x_payload", "x_payload"), "x_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "y_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_payload", "y_payload"), "y_payload"),
        ])
        .with_null_equality(crossed_join_null_equality),
        Arc::clone(&xy_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, key_name, key_name), ColumnRef::input(1, key_name, key_name))],
        )
        .with_output_projection(left_projection(&["x_key", "x_payload", "y_key", "y_payload"]))
        .with_null_equality(semijoin_null_equality),
        xy_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_duplicate_output_name_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key"]));
    let y = builder.add_source(gpu_source("y", &["y_key"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "dup");
    let duplicate_schema = schema(&["dup", "dup"]);
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "dup"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "dup"),
        ]),
        Arc::clone(&duplicate_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "dup", "dup"), ColumnRef::input(1, "dup", "dup"))],
        )
        .with_output_projection(left_projection(&["dup", "dup"])),
        duplicate_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_expression_key_projection_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key"]));
    let y = builder.add_source(gpu_source("y", &["y_key"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "y_key");
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "y_key")]),
        schema(&["y_key"]),
    );
    let projected = builder.add_projection(
        xy,
        vec![ProjectionOutput::new(
            Expr::binary(Expr::column("y_key"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0))),
            Field::new("y_key", DataType::Int32, false),
        )],
    );
    let semi = builder.add_join(
        projected,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "y_key", "y_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(left_projection(&["y_key"])),
        schema(&["y_key"]),
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_dropped_key_projection_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key"]));
    let y = builder.add_source(gpu_source("y", &["y_key"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "y_key");
    let xy_schema = schema(&["x_key", "y_key"]);
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "x_key")]),
        Arc::clone(&xy_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "y_key", "y_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(left_projection(&["x_key", "y_key"])),
        xy_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_nested_inner_join_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(gpu_source("a", &["a_key", "a_payload"]));
    let b = builder.add_source(gpu_source("b", &["b_key", "b_payload"]));
    let c = builder.add_source(gpu_source("c", &["c_key", "c_payload"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "b_key");

    let ab_schema = schema(&["a_key", "a_payload", "b_key", "b_payload"]);
    let ab = builder.add_join(
        a,
        b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_key", "a_key"), ColumnRef::input(1, "b_key", "b_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_payload", "a_payload"), "a_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_payload", "b_payload"), "b_payload"),
        ]),
        ab_schema,
    );

    let abc_schema = schema(&["a_key", "a_payload", "b_key", "b_payload", "c_key", "c_payload"]);
    let abc = builder.add_join(
        ab,
        c,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "c_key", "c_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_payload", "a_payload"), "a_payload"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_payload", "b_payload"), "b_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_key", "c_key"), "c_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_payload", "c_payload"), "c_payload"),
        ]),
        Arc::clone(&abc_schema),
    );
    let semi = builder.add_join(
        abc,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "b_key", "b_key"))],
        )
        .with_output_projection(left_projection(&[
            "a_key",
            "a_payload",
            "b_key",
            "b_payload",
            "c_key",
            "c_payload",
        ])),
        abc_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn inner_join_over_projected_left_join_right_key_plan() -> cudf_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("r_key")
}

fn inner_join_over_projected_left_join_left_key_plan() -> cudf_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("l_key")
}

fn inner_join_over_projected_left_join_key_plan(rejecting_key: &str) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key", "payload"]));
    let right = builder.add_source(gpu_source("right", &["r_key", "r_reason"]));
    let dim = builder.add_source(gpu_source("dim", &[rejecting_key]));
    let left_join_schema = schema(&["l_key", "payload", "r_key", "r_reason"]);
    let left_join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_reason", "r_reason"), "r_reason"),
        ]),
        Arc::clone(&left_join_schema),
    );
    let projected = builder.add_projection(
        left_join,
        vec![
            ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
            ProjectionOutput::new(Expr::column("r_reason"), Field::new("r_reason", DataType::Int32, true)),
        ],
    );
    let root_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, true),
        Field::new("r_reason", DataType::Int32, true),
    ]));
    let root = builder.add_join(
        projected,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, rejecting_key, rejecting_key),
                ColumnRef::input(1, rejecting_key, rejecting_key),
            )],
        )
        .with_output_projection(left_projection(&["l_key", "payload", "r_key", "r_reason"])),
        root_schema,
    );
    builder.finish_gpu_resident(root)
}

fn shared_left_join_output_with_one_null_rejecting_consumer_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key"]));
    let right = builder.add_source(gpu_source("right", &["r_key"]));
    let dim = builder.add_source(gpu_source("dim", &["r_key"]));
    let left_join_schema = schema(&["l_key", "r_key"]);
    let left_join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
        ]),
        Arc::clone(&left_join_schema),
    );
    let rejecting = builder.add_join(
        left_join,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "r_key", "r_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(left_projection(&["l_key", "r_key"])),
        Arc::clone(&left_join_schema),
    );
    let root = builder.add_join(
        left_join,
        rejecting,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "l_key", "l_key"))],
        )
        .with_output_projection(left_projection(&["l_key", "r_key"])),
        left_join_schema,
    );
    builder.finish_gpu_resident(root)
}

fn inner_join_filter_over_projected_inner_join_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["s_item", "s_ticket", "customer", "amount"]));
    let returns = builder.add_source(gpu_source("returns", &["r_item", "r_ticket", "reason_key", "returned"]));
    let reason = builder.add_source(gpu_source("reason", &["reason_key"]));
    let sales_returns_schema =
        schema(&["s_item", "s_ticket", "customer", "amount", "r_item", "r_ticket", "reason_key", "returned"]);
    let sales_returns = builder.add_join(
        sales,
        returns,
        JoinSpec::new(
            JoinType::Inner,
            vec![
                JoinKey::new(ColumnRef::input(0, "s_item", "s_item"), ColumnRef::input(1, "r_item", "r_item")),
                JoinKey::new(ColumnRef::input(0, "s_ticket", "s_ticket"), ColumnRef::input(1, "r_ticket", "r_ticket")),
            ],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "s_item", "s_item"), "s_item"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_ticket", "s_ticket"), "s_ticket"),
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_item", "r_item"), "r_item"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_ticket", "r_ticket"), "r_ticket"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_key", "reason_key"), "reason_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "returned", "returned"), "returned"),
        ]),
        sales_returns_schema,
    );
    let projected_schema = schema(&["customer", "net_amount", "reason_key"]);
    let projected = builder.add_projection(
        sales_returns,
        vec![
            ProjectionOutput::new(Expr::column("customer"), Field::new("customer", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::binary(Expr::column("amount"), BinaryOp::Subtract, Expr::column("returned")),
                Field::new("net_amount", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("reason_key"), Field::new("reason_key", DataType::Int32, false)),
        ],
    );
    let filtered = builder.add_join(
        projected,
        reason,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "reason_key", "reason_key"),
                ColumnRef::input(1, "reason_key", "reason_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "net_amount", "net_amount"), "net_amount"),
        ]),
        schema(&["customer", "net_amount"]),
    );
    assert_eq!(projected_schema.fields().len(), 3);
    builder.finish_gpu_resident(filtered)
}

fn nested_inner_join_filter_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["sold_date_sk", "customer_sk", "amount"]));
    let customer = builder.add_source(gpu_source("customer", &["customer_sk", "addr_sk"]));
    let address = builder.add_source(gpu_source("address", &["addr_sk", "zip"]));
    let date = builder.add_source(gpu_source("date", &["date_sk"]));

    let sales_customer = builder.add_join(
        sales,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "customer_sk", "customer_sk"),
                ColumnRef::input(1, "customer_sk", "customer_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "sold_date_sk", "sold_date_sk"), "sold_date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "addr_sk", "addr_sk"), "addr_sk"),
        ]),
        schema(&["sold_date_sk", "amount", "addr_sk"]),
    );
    let projected_sales_customer = builder.add_projection(
        sales_customer,
        vec![
            ProjectionOutput::new(Expr::column("sold_date_sk"), Field::new("sold_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("amount"), Field::new("amount", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("addr_sk"), Field::new("addr_sk", DataType::Int32, false)),
        ],
    );
    let address_sales = builder.add_join(
        address,
        projected_sales_customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "addr_sk", "addr_sk"), ColumnRef::input(1, "addr_sk", "addr_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "sold_date_sk", "sold_date_sk"), "sold_date_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "amount", "amount"), "amount"),
            JoinOutputColumn::left(ColumnRef::input(0, "zip", "zip"), "zip"),
        ]),
        schema(&["sold_date_sk", "amount", "zip"]),
    );
    let date_filtered = builder.add_join(
        date,
        address_sales,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "sold_date_sk", "sold_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "zip", "zip"), "zip"),
        ]),
        schema(&["amount", "zip"]),
    );
    builder.finish_gpu_resident(date_filtered)
}

fn shared_projected_probe_fanout_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(gpu_source("fact", &["key", "qty"]));
    let prefix_dim = builder.add_source(gpu_source("prefix_dim", &["key"]));
    let prefix = builder.add_join(
        fact,
        prefix_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        schema(&["key", "qty"]),
    );
    let projected = builder.add_projection(
        prefix,
        vec![
            ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("qty"), Field::new("qty", DataType::Int32, false)),
        ],
    );
    let first_dim = builder.add_source(gpu_source("first_dim", &["key"]));
    let second_dim = builder.add_source(gpu_source("second_dim", &["key"]));
    let first = builder.add_join(
        projected,
        first_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
        )
        .with_output_projection(left_projection(&["key", "qty"]))
        .with_build_side(JoinSide::Right),
        schema(&["key", "qty"]),
    );
    let second = builder.add_join(
        projected,
        second_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
        )
        .with_output_projection(left_projection(&["key", "qty"]))
        .with_build_side(JoinSide::Right),
        schema(&["key", "qty"]),
    );
    let union = builder.add_union(vec![first, second], schema(&["key", "qty"]));
    builder.finish_gpu_resident(union)
}

fn compatible_small_local_parquet_source_variants_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let full = builder.add_source(customer_address_source(&["id", "state"], &[0, 1], None));
    let filtered = builder.add_source(customer_address_source(
        &["id", "zip"],
        &[0, 2],
        Some(Expr::binary(Expr::column("zip"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(30301)))),
    ));
    let full_id = builder.add_projection(
        full,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let filtered_id = builder.add_projection(
        filtered,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![full_id, filtered_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn large_local_parquet_source_variants_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let right = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_warehouse_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let left_item = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let right_item = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left_item, right_item], schema(&["inv_item_sk"]));
    builder.finish_gpu_resident(union)
}

fn customer_address_source(columns: &[&str], file_indices: &[usize], predicate: Option<Expr>) -> SourceNode {
    let fields = columns.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let file = LocalParquetFile::new("/tmp/customer_address.parquet")
        .with_stable_id("customer_address")
        .with_row_group_bytes(vec![2 * 1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    let read_columns = columns.iter().map(|column| (*column).to_string()).collect::<Vec<_>>();
    let mut source = LocalParquetSource::from_files(vec![file], schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    );
    if let Some(predicate) = predicate {
        source =
            source.with_column_roles(vec!["id".to_string()], vec!["zip".to_string()]).with_reader_predicate(predicate);
    }
    SourceNode::LocalParquet(source)
}

fn gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
}

fn aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> cudf_query_engine::NodeId {
    let source = builder.add_source(gpu_source(name, &[key_name]));
    builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()), schema(&[key_name]))
}

fn schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

fn left_projection(fields: &[&str]) -> Vec<JoinOutputColumn> {
    fields.iter().map(|name| JoinOutputColumn::left(ColumnRef::input(0, *name, *name), *name)).collect()
}

fn max_direct_join_probe_consumer_count(plan: &cudf_query_engine::QueryPlan) -> usize {
    let mut counts = std::collections::BTreeMap::<cudf_query_engine::NodeId, usize>::new();
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

fn local_parquet_source_count_for_stable_id(plan: &cudf_query_engine::QueryPlan, stable_id: &str) -> usize {
    plan.nodes()
        .iter()
        .filter(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.files.iter().any(|file| file.stable_id == stable_id),
                _ => false,
            },
            _ => false,
        })
        .count()
}

fn assert_source_input(plan: &cudf_query_engine::QueryPlan, node_id: cudf_query_engine::NodeId, name: &str) {
    let node = plan.node(node_id).expect("node should exist");
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(source) => assert_eq!(source.input_name, name),
            other => panic!("expected GPU-resident source {name}, got {other:?}"),
        },
        other => panic!("expected source {name}, got {other:?}"),
    }
}

fn source_input_name(plan: &cudf_query_engine::QueryPlan, node_id: cudf_query_engine::NodeId) -> Option<String> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else { return None };
    match source.as_ref() {
        SourceNode::GpuResident(source) => Some(source.input_name.clone()),
        _ => None,
    }
}

fn assert_left_semi_input(plan: &cudf_query_engine::QueryPlan, node_id: cudf_query_engine::NodeId, key_name: &str) {
    let node = plan.node(node_id).expect("node should exist");
    let PlanNode::Join(join) = &node.kind else {
        panic!("expected pushed left-semi join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftSemi);
    assert_eq!(join.spec.keys.len(), 1);
    assert_eq!(join.spec.keys[0].left.name(), key_name);
    assert_eq!(join.spec.keys[0].right.name(), key_name);
    assert!(
        join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left),
        "pushed left-semi must expose only target-side columns"
    );
}

fn assert_semijoin_rejection(plan: cudf_query_engine::QueryPlan, reason: SemijoinReductionRejectedReason) {
    let result = optimize_query_plan(plan, &semijoin_reduction_only_options(SemijoinReductionOptions::default()));
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].rejected_reason, Some(reason));
}

fn outer_join_null_rejection_only_options() -> NativeOptimizerOptions {
    NativeOptimizerOptions {
        outer_join_null_rejection: cudf_query_engine::OuterJoinNullRejectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    }
}

fn semijoin_reduction_only_options(semijoin_reduction: SemijoinReductionOptions) -> NativeOptimizerOptions {
    NativeOptimizerOptions { semijoin_reduction, ..NativeOptimizerOptions::disabled() }
}
