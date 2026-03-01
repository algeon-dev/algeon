use nexus_query_engine::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
    DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DimensionDerivedRowGroupPruningDiagnostics,
    DimensionDerivedRowGroupPruningOptions, DimensionDerivedRowGroupPruningRejectedReason, DynamicFilter,
    DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage, DynamicFilterKind,
    DynamicFilterProducer, DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions,
    DynamicFilterRouteRejectedReason, DynamicFilterSourceSnapshot, GraphAlgorithm, GraphAlgorithmParams,
    GraphAlgorithmSpec, GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions, GraphEdgeColumnBindings,
    GraphPredicateInput, JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JOIN_FILTER_REASSOCIATION_RULE_NAME,
    JoinBuildSideSelectionDiagnostics, JoinBuildSideSelectionOptions, JoinFilterReassociationDiagnostics,
    LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetSource, LocalParquetSourceReuseDiagnostics,
    LocalParquetSourceReuseOptions, NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, NodeId,
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionDiagnostics, OuterJoinNullRejectionRejectedReason,
    PROJECTION_SIMPLIFICATION_RULE_NAME, ProjectionSimplificationDiagnostics, ProjectionSimplificationOptions,
    SEMIJOIN_REDUCTION_RULE_NAME, SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME,
    SemijoinReductionAdmissionMode, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
    SemijoinReductionOptions, SemijoinReductionRejectedReason, SharedSubplanMaterializationDiagnostics,
    SharedSubplanMaterializationOptions, StreamFanoutReplayDiagnostics, StreamFanoutReplayOptions,
    apply_dimension_derived_pruning_with_stats, discover_dimension_derived_pruning_requests, optimize_query_plan,
    seed_dimension_derived_reader_predicates,
};

use super::*;

#[test]
fn native_optimizer_semijoin_entry_point_returns_unchanged_plan_and_empty_diagnostics() {
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
fn semijoin_reduction_options_and_rule_name_are_stable() {
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
fn native_optimizer_stable_rule_rows_cover_every_diagnostic_field() {
    let diagnostics = NativeOptimizerDiagnostics {
        aggregate_join_pushdown: AggregateJoinPushdownDiagnostics { considered_count: 1, ..Default::default() },
        projection_simplification: ProjectionSimplificationDiagnostics { considered_count: 1, ..Default::default() },
        semijoin_reduction: SemijoinReductionDiagnostics { considered_count: 1, ..Default::default() },
        outer_join_null_rejection: OuterJoinNullRejectionDiagnostics { considered_count: 1, ..Default::default() },
        join_filter_reassociation: JoinFilterReassociationDiagnostics { considered_count: 1, ..Default::default() },
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningDiagnostics {
            considered_count: 1,
            ..Default::default()
        },
        join_build_side_selection: JoinBuildSideSelectionDiagnostics { considered_count: 1, ..Default::default() },
        stream_fanout_replay: StreamFanoutReplayDiagnostics { considered_count: 1, ..Default::default() },
        local_parquet_source_reuse: LocalParquetSourceReuseDiagnostics { considered_count: 1, ..Default::default() },
        shared_subplan_materialization: SharedSubplanMaterializationDiagnostics {
            considered_count: 1,
            ..Default::default()
        },
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryDiagnostics {
            considered_count: 1,
            ..Default::default()
        },
    };

    let rows = diagnostics.stable_rule_rows();
    assert_eq!(rows.len(), 11);
    let rendered = rows.join(";");
    for rule_name in [
        PROJECTION_SIMPLIFICATION_RULE_NAME,
        SEMIJOIN_REDUCTION_RULE_NAME,
        JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
        OUTER_JOIN_NULL_REJECTION_RULE_NAME,
        JOIN_FILTER_REASSOCIATION_RULE_NAME,
        DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
        STREAM_FANOUT_REPLAY_RULE_NAME,
        LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
        SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
        DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
        AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
    ] {
        assert!(
            rendered.contains(&format!("rule={rule_name},")),
            "missing stable optimizer diagnostic row for {rule_name}: {rendered}"
        );
    }
}

#[test]
fn native_optimizer_stable_rule_rows_project_aggregate_join_pushdown_candidates() {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", 1_024, 10));
    let right = builder.add_source(estimated_local_source("right", "right_key", 1_024, 10));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        ),
        schema(&["left_key", "right_key"]),
    );

    let diagnostics = NativeOptimizerDiagnostics {
        aggregate_join_pushdown: AggregateJoinPushdownDiagnostics {
            considered_count: 1,
            applied_count: 0,
            candidates: vec![AggregateJoinPushdownCandidateDiagnostics {
                rule_name: AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
                aggregate_node_id: join,
                join_node_id: Some(join),
                applied: false,
                rejected_reason: Some(AggregateJoinPushdownRejectedReason::GroupingSetsRequired),
            }],
        },
        ..Default::default()
    };

    assert_eq!(
        diagnostics.stable_rule_rows(),
        vec![
            "rule=aggregate_join_pushdown,considered_count=1,applied_count=0,candidates=rule=aggregate_join_pushdown,aggregate_node_id=n2,join_node_id=n2,applied=false,rejected_reason=GroupingSetsRequired".to_string()
        ]
    );
}

#[test]
fn outer_join_null_rejection_options_and_rule_name_are_stable() {
    assert_eq!(OUTER_JOIN_NULL_REJECTION_RULE_NAME, "outer_join_null_rejection");
    assert!(nexus_query_engine::OuterJoinNullRejectionOptions::default().enabled);
    assert!(!nexus_query_engine::OuterJoinNullRejectionOptions::disabled().enabled);
}

#[test]
fn join_filter_reassociation_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_FILTER_REASSOCIATION_RULE_NAME, "join_filter_reassociation");
    assert!(nexus_query_engine::JoinFilterReassociationOptions::default().enabled);
    assert!(nexus_query_engine::JoinFilterReassociationOptions::default().residual_filter_pushdown_enabled);
    assert!(!nexus_query_engine::JoinFilterReassociationOptions::disabled().enabled);
    assert!(!nexus_query_engine::JoinFilterReassociationOptions::disabled().residual_filter_pushdown_enabled);
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
fn native_optimizer_materializes_aliased_grouped_aggregates_when_branch_replay_is_unneeded() {
    let plan = repeated_aliased_grouped_aggregate_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 2);
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
    assert!(
        aggregates[0]
            .spec
            .aggr_exprs
            .iter()
            .filter_map(|expr| expr.filter.as_ref())
            .any(|filter| { expr_contains_int32_equality(filter, "t_hour", 8) })
    );
    assert!(
        aggregates[0]
            .spec
            .aggr_exprs
            .iter()
            .filter_map(|expr| expr.filter.as_ref())
            .any(|filter| { expr_contains_int32_equality(filter, "t_hour", 9) })
    );
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
    assert!(plan_has_output_name_prefix(&result.plan, "__cudf_shared_scalar_"));
}

#[test]
fn native_optimizer_keeps_shared_subplan_materialization_contract_isolated() {
    let plan = repeated_scalar_count_materialization_with_mismatched_source_contracts_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_subplan_materialization.applied_count, 0);
}

#[test]
fn native_optimizer_keeps_shared_subplan_materialization_decimal_contract_isolated() {
    let plan = repeated_scalar_count_materialization_with_mismatched_decimal_contracts_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_subplan_materialization.applied_count, 0);
}

#[test]
fn native_optimizer_ignores_identity_projection_depth_for_scalar_count_materialization() {
    let plan = repeated_scalar_count_materialization_with_redundant_projection_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 2);
}

#[test]
fn native_optimizer_lifts_reader_predicate_filters_for_scalar_count_materialization() {
    let plan = repeated_scalar_count_materialization_with_reader_predicate_filters_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 2);

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
    assert_eq!(
        result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Filter(_))).count(),
        0,
        "source-adjacent filters represented by reader predicates should move into aggregate filters"
    );
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
}

#[test]
fn native_optimizer_rejects_repeated_scalar_mixed_distinct_branches_without_streaming_support() {
    let plan = repeated_scalar_mixed_aggregate_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 3);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 3);
    assert!(candidate.group_exprs.is_empty());
    assert!(candidate.aggregate_exprs.iter().any(|expr| expr.contains("fn=avg")));
    assert!(candidate.aggregate_exprs.iter().any(|expr| expr.contains("distinct=true")));
    assert!(!candidate.applied);
    assert_eq!(candidate.filtered_aggregate_count, 9);
    assert_eq!(candidate.filtered_aggregate_memory_shape, "streaming_filtered_state_not_applied");
    assert_eq!(candidate.filtered_aggregate_estimation_input_rows, Some(10_000_000));
    assert!(candidate.filtered_aggregate_estimated_materialization_bytes.unwrap_or(0) > 0);
    assert_eq!(candidate.filtered_aggregate_memory_threshold_policy, "runtime_budget/2");
    assert_eq!(
        candidate.rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)
    );

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 3);
    assert!(!plan_has_output_name_prefix(&result.plan, "__cudf_shared_scalar_"));
}

#[test]
fn native_optimizer_combines_q09_like_scalar_branches_with_different_read_columns() {
    let plan = repeated_q09_like_scalar_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 6);
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let sources = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 1);
    assert_eq!(
        sources[0].read_columns,
        vec!["ss_quantity".to_string(), "ss_ext_discount_amt".to_string(), "ss_net_paid".to_string(),]
    );

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
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 6);
    assert!(aggregates[0].spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    assert_eq!(aggregates[0].spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg).count(), 4);
}

#[test]
fn native_optimizer_combines_q09_like_scalar_branches_when_projected_count_is_not_representative() {
    let plan = repeated_q09_like_scalar_materialization_plan_with_order(&[
        Q09ScalarBranch::AvgDiscount,
        Q09ScalarBranch::Count,
        Q09ScalarBranch::AvgNetPaid,
    ]);

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 6);

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
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 6);
    assert!(
        aggregates[0]
            .spec
            .aggr_exprs
            .iter()
            .filter(|expr| expr.function == AggregateFunction::Count)
            .all(|expr| matches!(&expr.expr, Expr::Literal(value) if !matches!(value, LiteralValue::Null(_)))),
        "projected literal count-star branches should be canonicalized to COUNT(non-null-literal):\n{}",
        result.plan
    );
}

#[test]
fn native_optimizer_combines_projected_literal_count_star_scalar_branches() {
    let plan = repeated_projected_literal_count_star_scalar_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(candidate.filtered_aggregate_count, 2);
    assert_eq!(candidate.filtered_aggregate_memory_shape, "runtime_budgeted_segmented_state_available");
    assert_eq!(candidate.filtered_aggregate_estimation_input_rows, Some(10_000_000));
    assert!(candidate.filtered_aggregate_estimated_materialization_bytes.unwrap_or(0) > 0);
    assert_eq!(candidate.filtered_aggregate_memory_budget_bytes, None);
    assert_eq!(candidate.filtered_aggregate_memory_threshold_bytes, None);
    assert_eq!(candidate.filtered_aggregate_memory_threshold_policy, "runtime_budget/2");

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
    assert!(plan_has_output_name_prefix(&result.plan, "__cudf_shared_scalar_"));
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_different_projection_local_filter_semantics() {
    let plan = repeated_scalar_count_with_different_projected_filter_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_different_local_filters() {
    let plan = repeated_scalar_count_with_different_local_filter_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_rejects_grouped_materialization_when_source_predicate_collides_with_aggregate_alias() {
    let plan = repeated_grouped_aggregate_predicate_alias_collision_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)
    );
}

#[test]
fn native_optimizer_rejects_grouped_materialization_when_predicate_column_has_different_group_lineage() {
    let plan = repeated_grouped_aggregate_predicate_group_name_lineage_collision_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)
    );
}

#[test]
fn native_optimizer_rejects_grouped_materialization_with_duplicate_source_occurrences() {
    let plan = repeated_grouped_aggregate_self_join_duplicate_source_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_rejects_grouped_materialization_with_reused_source_node_occurrences() {
    let plan = repeated_grouped_aggregate_reused_source_node_self_join_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_rejects_unreachable_grouped_materialization_without_claiming_applied_rewrite() {
    let plan = unreachable_grouped_aggregate_materialization_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::NotReachableFromSink)
    );
}

#[test]
fn native_optimizer_rejects_scalar_materialization_with_conflicting_source_refs() {
    let plan = repeated_scalar_avg_with_conflicting_source_refs_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(
        candidate.rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_limit_below_aggregate() {
    let plan = repeated_scalar_count_with_limit_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_subplan_materialization.considered_count, 0);
}

#[test]
fn native_optimizer_rejects_scalar_materialization_with_duplicate_join_input_column_names() {
    let plan = repeated_scalar_join_count_with_duplicate_input_names_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_duplicate_rowset_join_inputs() {
    let plan = repeated_scalar_self_join_count_with_duplicate_rowset_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_rejects_scalar_materialization_with_reused_source_node_join_inputs() {
    let plan = repeated_scalar_self_join_count_with_reused_source_node_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_derives_scalar_sum_from_grouped_sum_with_permuted_source_files() {
    let plan = grouped_and_scalar_sum_over_permuted_source_files_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert_eq!(candidate.aggregate_node_ids.len(), 2);
    assert!(candidate.applied);

    let scalar_aggregate_node = result
        .plan
        .nodes()
        .iter()
        .find(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty()))
        .expect("optimized plan should retain a scalar aggregate");
    let PlanNode::Aggregate(scalar_aggregate) = &scalar_aggregate_node.kind else {
        unreachable!("matched above");
    };
    let grouped_aggregate_node = result.plan.node(scalar_aggregate.input).expect("scalar aggregate input should exist");
    assert!(
        matches!(&grouped_aggregate_node.kind, PlanNode::Aggregate(aggregate) if !aggregate.spec.group_exprs.is_empty()),
        "scalar aggregate should derive from grouped aggregate output:\n{}",
        result.plan
    );
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
fn dimension_pruning_seeds_date_reader_predicate_from_aggregate_group_key_filter() {
    let dir = unique_contract_temp_dir("dimension-group-filter");
    let plan = aggregate_group_filter_dimension_pruning_plan(&dir);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.dimension_derived_row_group_pruning;
    assert_eq!(diagnostics.applied_count, 1);
    let applied = diagnostics.candidates.iter().find(|candidate| candidate.applied).expect("one candidate prunes");
    assert_eq!(applied.dimension_key_column.as_deref(), Some("d_date_sk"));
    assert_eq!(applied.target_key_column.as_deref(), Some("ss_sold_date_sk"));
    assert_eq!(applied.derived_min_key, Some(2));
    assert_eq!(applied.derived_max_key, Some(3));
    assert_eq!(applied.selected_row_groups_before, Some(4));
    assert_eq!(applied.selected_row_groups_after, Some(2));

    let sources = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    let date_source = sources
        .iter()
        .find(|source| source.read_columns.iter().any(|column| column == "d_year"))
        .expect("date source should remain present");
    assert_eq!(date_source.reader_predicate.as_ref().map(ToString::to_string).as_deref(), Some("(d_year = 2000)"));
    let fact_source = sources
        .iter()
        .find(|source| source.read_columns.iter().any(|column| column == "ss_sold_date_sk"))
        .expect("fact source should remain present");
    assert_eq!(fact_source.files[0].row_groups.as_deref(), Some(&[1, 2][..]));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_seeded_plan_exposes_group_key_filter_discovery_request() {
    let dir = unique_contract_temp_dir("dimension-group-filter-discovery");
    let plan = aggregate_group_filter_dimension_pruning_plan(&dir);

    assert!(discover_dimension_derived_pruning_requests(&plan).is_empty());

    let seeded = seed_dimension_derived_reader_predicates(plan);
    let requests = discover_dimension_derived_pruning_requests(&seeded);

    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].dimension_key_column, "d_date_sk");
    assert_eq!(requests[0].target_key_column, "ss_sold_date_sk");

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_does_not_seed_unsupported_reader_predicate() {
    let dir = unique_contract_temp_dir("dimension-group-filter-unsupported");
    let plan = aggregate_group_filter_dimension_pruning_plan_with_predicate(
        &dir,
        Expr::binary(
            Expr::try_cast(Expr::column("ss_sold_year"), DataType::Int64),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Int64(2000)),
        ),
    );

    let seeded = seed_dimension_derived_reader_predicates(plan.clone());

    assert_eq!(seeded, plan);
    assert!(discover_dimension_derived_pruning_requests(&seeded).is_empty());

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_external_stats_narrow_existing_row_group_selection() {
    let mut builder = QueryPlanBuilder::new();
    let schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/external-stats-row-groups.parquet").with_row_groups(vec![1, 3]);
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["fact_key".to_string()],
    )));
    let plan = builder.finish_gpu_resident(source);
    let result = apply_dimension_derived_pruning_with_stats(
        plan,
        source,
        &[vec![
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 10, max: 19 }),
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 30, max: 39 }),
        ]],
        &nexus_query_engine::DimensionKeyFilter::from_bounds(nexus_query_engine::DimensionDerivedKeyRange {
            min: 35,
            max: 35,
        }),
    );

    assert!(result.applied);
    assert_eq!(result.before, 2);
    assert_eq!(result.after, 1);
    let PlanNode::Source(source) = &result.plan.node(source).expect("source should remain present").kind else {
        panic!("node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("source should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[3][..]));
}

#[test]
fn dimension_pruning_scattered_keyset_prunes_gaps_a_single_range_would_keep() {
    // The qualifying keyset {5, 35} bounds to [5, 35]. A single-range filter
    // overlaps BOTH row groups ([10,19] and [30,39]); the set-aware filter keeps
    // only the row group containing 35, proving runs prune the inner gap.
    let mut builder = QueryPlanBuilder::new();
    let schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/scattered-keyset-row-groups.parquet").with_row_groups(vec![1, 3]);
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["fact_key".to_string()],
    )));
    let plan = builder.finish_gpu_resident(source);

    let scattered =
        nexus_query_engine::DimensionKeyFilter::from_keys(&[5, 35]).expect("non-empty keyset builds a filter");
    assert_eq!(scattered.run_count(), 2);

    let result = apply_dimension_derived_pruning_with_stats(
        plan,
        source,
        &[vec![
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 10, max: 19 }),
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 30, max: 39 }),
        ]],
        &scattered,
    );

    assert!(result.applied);
    assert_eq!(result.before, 2);
    assert_eq!(result.after, 1);
    let PlanNode::Source(source) = &result.plan.node(source).expect("source should remain present").kind else {
        panic!("node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("source should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[3][..]));
}

#[test]
fn dimension_pruning_does_not_seed_right_side_left_join_group_key_filter() {
    let dir = unique_contract_temp_dir("dimension-left-right-filter");
    let plan = aggregate_right_group_filter_left_join_plan(&dir);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dimension_derived_row_group_pruning.applied_count, 0);
    assert!(result.plan.nodes().iter().all(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source.reader_predicate.is_none(),
            _ => true,
        },
        _ => true,
    }));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_does_not_seed_reader_predicate_through_shared_aggregate_subtree() {
    let dir = unique_contract_temp_dir("dimension-shared-group-filter");
    let plan = shared_aggregate_group_filter_dimension_pruning_plan(&dir);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dimension_derived_row_group_pruning.applied_count, 0);
    assert!(result.plan.nodes().iter().all(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source.reader_predicate.is_none(),
            _ => true,
        },
        _ => true,
    }));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
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
fn gpu_local_parquet_source_cache_key_isolates_resolved_source_contracts() {
    let direct_source = contract_sensitive_customer_address_source(remote_direct_customer_address_contract());
    let host_buffer_source = contract_sensitive_customer_address_source(host_buffer_customer_address_contract());
    let decimal_source = contract_sensitive_customer_address_source(remote_direct_customer_address_contract())
        .with_reader_decimal_contract("id", ParquetReaderDecimalTypeId::Decimal64, -2);

    let direct_source_key = GpuLocalParquetSourceCacheKey::from_source(&direct_source);
    let host_buffer_source_key = GpuLocalParquetSourceCacheKey::from_source(&host_buffer_source);
    let decimal_source_key = GpuLocalParquetSourceCacheKey::from_source(&decimal_source);
    assert_ne!(
        direct_source_key, host_buffer_source_key,
        "whole-source GPU cache keys must include resolved read/cache/credential contract fields"
    );
    assert_ne!(
        direct_source_key, decimal_source_key,
        "whole-source GPU cache keys must include per-file Parquet reader decimal contracts"
    );
    assert!(direct_source_key.as_str().contains("file.0.resolved.read_strategy=remote_direct"));
    assert!(direct_source_key.as_str().contains("file.0.resolved.remote_read_mode=require_direct"));
    assert!(direct_source_key.as_str().contains("file.0.resolved.credential_mode=shared_process_env"));
    assert!(direct_source_key.as_str().contains("file.0.resolved.storage_profile_id=remote-profile-a"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.read_strategy=remote_host_buffer"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.decode_input=host_buffer"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.lakehouse_cache_status=miss"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.lakehouse_cache_read_mode=enabled"));
    assert!(host_buffer_source_key.as_str().contains("file.0.resolved.lakehouse_cache_fill_mode=background"));
    assert!(decimal_source_key.as_str().contains("file.0.reader_decimal_types.0.type_id=decimal64"));
    assert!(!direct_source_key.as_str().contains("query_variant=a"));
    assert!(!host_buffer_source_key.as_str().contains("/tmp/nexus-lakehouse-cache"));

    let mut builder = QueryPlanBuilder::new();
    let direct_source_id = builder.add_source(SourceNode::LocalParquet(direct_source.clone()));
    let host_buffer_source_id = builder.add_source(SourceNode::LocalParquet(host_buffer_source.clone()));
    let decimal_source_id = builder.add_source(SourceNode::LocalParquet(decimal_source.clone()));
    let direct_chunk_plan =
        plan_local_parquet_source_chunks(direct_source_id, &direct_source, SourceChunkPlanningPolicy::new(None, 1))
            .unwrap();
    let host_buffer_chunk_plan = plan_local_parquet_source_chunks(
        host_buffer_source_id,
        &host_buffer_source,
        SourceChunkPlanningPolicy::new(None, 1),
    )
    .unwrap();
    let decimal_chunk_plan =
        plan_local_parquet_source_chunks(decimal_source_id, &decimal_source, SourceChunkPlanningPolicy::new(None, 1))
            .unwrap();
    let direct_chunk_key =
        GpuLocalParquetSourceCacheKey::from_source_chunk(&direct_source, &direct_chunk_plan.chunks[0]);
    let host_buffer_chunk_key =
        GpuLocalParquetSourceCacheKey::from_source_chunk(&host_buffer_source, &host_buffer_chunk_plan.chunks[0]);
    let decimal_chunk_key =
        GpuLocalParquetSourceCacheKey::from_source_chunk(&decimal_source, &decimal_chunk_plan.chunks[0]);

    assert_ne!(
        direct_chunk_key, host_buffer_chunk_key,
        "bounded chunk GPU cache keys must include resolved read/cache/credential contract fields"
    );
    assert_ne!(
        direct_chunk_key, decimal_chunk_key,
        "bounded chunk GPU cache keys must include per-chunk Parquet reader decimal contracts"
    );
    assert!(direct_chunk_key.as_str().contains("chunk.resolved.read_strategy=remote_direct"));
    assert!(host_buffer_chunk_key.as_str().contains("chunk.resolved.read_strategy=remote_host_buffer"));
    assert!(decimal_chunk_key.as_str().contains("chunk.reader_decimal_types.0.type_id=decimal64"));
    assert!(decimal_chunk_key.as_str().contains("chunk.reader_decimal_types_footer_verified=false"));
}

#[test]
fn native_optimizer_keeps_local_parquet_source_reuse_contract_isolated() {
    let plan = contract_mismatched_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(
        result.diagnostics.local_parquet_source_reuse.considered_count, 0,
        "sources with different resolved read/cache/credential contracts must not enter the same reuse group"
    );
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 0);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address"), 2);
}

#[test]
fn native_optimizer_local_parquet_source_reuse_diagnostics_redact_stable_file_ids() {
    let plan = compatible_sensitive_stable_id_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    let stable_file_ids = &result.diagnostics.local_parquet_source_reuse.candidates[0].stable_file_ids;
    assert_eq!(stable_file_ids.len(), 1);
    assert!(stable_file_ids[0].starts_with("object_store:s3:fnv64:"));
    for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
        assert!(
            !stable_file_ids[0].contains(sensitive),
            "{sensitive} leaked in source reuse diagnostic: {:?}",
            stable_file_ids
        );
    }
}

#[test]
fn native_optimizer_keeps_decimal_contract_source_reuse_isolated() {
    let plan = decimal_contract_mismatched_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(
        result.diagnostics.local_parquet_source_reuse.considered_count, 0,
        "sources with different per-file decimal reader contracts must not enter the same reuse group"
    );
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 0);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address"), 2);
}

#[test]
fn native_optimizer_rejects_local_parquet_source_reuse_with_different_file_order() {
    let plan = compatible_local_parquet_source_variants_with_different_file_order_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };
    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 0);
    assert_eq!(
        result.diagnostics.local_parquet_source_reuse.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "customer_address_a"), 2);
}

#[test]
fn native_optimizer_rejects_source_reuse_with_conflicting_file_index_column_names() {
    let plan = conflicting_file_index_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
}

#[test]
fn native_optimizer_rejects_source_reuse_with_conflicting_column_name_file_indices() {
    let plan = conflicting_column_name_local_parquet_source_variants_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::UnsupportedSourceContract)
    );
}

#[test]
fn native_optimizer_does_not_lift_variant_max_rows_to_shared_local_parquet_source() {
    let plan = compatible_small_local_parquet_source_variants_with_max_rows_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    let sources = local_parquet_sources_for_stable_id(&result.plan, "customer_address");
    let [shared_source] = sources.as_slice() else {
        panic!("source reuse should produce exactly one shared customer_address source, got {}", sources.len());
    };
    assert_eq!(
        shared_source.max_rows, None,
        "branch-local max_rows is valid only after the branch predicate and must not be lifted to the shared source"
    );
}

#[test]
fn native_optimizer_rewrites_graph_algorithm_bfs_side_input_references_with_source_reuse() {
    let plan = graph_bfs_local_source_reuse_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.considered_count, 1);
    assert_eq!(result.diagnostics.local_parquet_source_reuse.applied_count, 1);
    assert_all_node_inputs_exist(&result.plan);

    let PlanNode::GraphAlgorithm(graph) = &result.plan.root_node().kind else {
        panic!("optimized root should remain a graph algorithm");
    };
    let GraphAlgorithmParams::Bfs(params) = &graph.spec.params else {
        panic!("expected BFS graph params");
    };
    let target_vertices = params.predicates.target_vertices.as_ref().expect("BFS target side input should exist");

    assert_eq!(graph.inputs.first().copied(), Some(graph.input));
    assert_eq!(graph.inputs.get(1).copied(), Some(target_vertices.input));
    assert!(
        result.plan.node(target_vertices.input).is_some(),
        "BFS target side-input binding must be remapped to an existing rewritten node"
    );
    assert_eq!(result.plan.root_node().input_ids(), graph.inputs.as_slice());
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
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::VariantTooLarge)
    );
}

#[test]
fn native_optimizer_reuses_large_existing_superset_with_smaller_local_parquet_source_variant() {
    let plan = large_superset_with_smaller_local_parquet_source_variant_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.superset_read_columns, vec!["inv_item_sk", "inv_date_sk"]);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
}

#[test]
fn native_optimizer_rejects_large_superset_reuse_when_variant_feeds_grouped_aggregate() {
    let plan = large_superset_with_grouped_aggregate_source_variant_plan();
    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::LocalParquetSourceReuseRejectedReason::VariantTooLarge)
    );
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 2);
}

#[test]
fn native_optimizer_reuses_large_superset_when_grouped_aggregate_variant_is_not_widened() {
    let plan = large_grouped_aggregate_superset_with_smaller_source_variant_plan();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert!(diagnostics.candidates[0].applied);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
}

#[test]
fn native_optimizer_reuses_large_filtered_existing_superset_outside_source_adjacent_grouped_aggregate() {
    let plan = large_filtered_superset_with_smaller_local_parquet_source_variant_plan();
    let options = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };
    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert!(diagnostics.candidates[0].applied);
    assert_eq!(diagnostics.candidates[0].branch_filter_count, 1);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "inventory"), 1);
    let branch_filters = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Filter(filter) => Some(&filter.predicate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(branch_filters.len(), 1);
    assert!(
        branch_filters[0].to_string().contains(" OR "),
        "test predicate should cover recursive replay normalization"
    );
    assert!(
        branch_filters[0].to_string().contains(" AND "),
        "test predicate should cover recursive replay normalization"
    );
    assert_no_non_decimal_binary_output_types(
        branch_filters[0],
        "replayed reader predicate should strip declared boolean output type before native filter validation",
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
        join_filter_reassociation: nexus_query_engine::JoinFilterReassociationOptions::disabled(),
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
fn outer_join_null_rejection_converts_marker_null_filter_to_left_anti() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::Eligible);

    let result = optimize_query_plan(plan, &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);

    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("root should remain the final projection");
    };
    let PlanNode::Join(join) = &result.plan.node(projection.input).expect("anti join").kind else {
        panic!("projection should consume rewritten anti join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftAnti);
    assert_eq!(join.spec.build_side, JoinSide::Right);
    assert!(join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left));
    assert_eq!(join.spec.output_projection.len(), 2);
}

#[test]
fn outer_join_null_rejection_converts_marker_null_filter_when_key_schema_is_nullable() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::NullableMarker);

    let result = optimize_query_plan(plan, &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("root should remain the final projection");
    };
    let PlanNode::Join(join) = &result.plan.node(projection.input).expect("anti join").kind else {
        panic!("projection should consume rewritten anti join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftAnti);
}

#[test]
fn outer_join_null_rejection_keeps_marker_null_filter_with_nulls_match_join() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::NullsMatch);

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

#[test]
fn outer_join_null_rejection_keeps_marker_null_filter_when_marker_escapes_projection() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::MarkerEscapes);

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
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
fn semijoin_reduction_accepts_filtered_dimension_keyset_for_transitive_fact_filtering() {
    let plan = root_semijoin_over_fact_date_dimension_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.crossed_join_node_ids.len(), 1);
    assert_eq!(candidate.key_mappings[0].before_column, "d_date");
    assert_eq!(candidate.key_mappings[0].after_column.as_deref(), Some("d_date"));
    assert!(matches!(
        candidate.cost_gate.as_ref().map(|gate| &gate.signal),
        Some(SemijoinReductionCostSignal::FilteredSourceKeysetShape {
            filtered_source_count: 1,
            target_rows: Some(73_049),
            crossed_join_count: 1,
            ..
        })
    ));

    assert!(
        result.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(&route.producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. })
                && route.producer_columns == vec!["d_date_sk".to_string()]
                && route.consumer_columns == vec!["ss_sold_date_sk".to_string()]
                && matches!(&route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
                && route.consumer_lowering == DynamicFilterConsumerLowering::ReaderPredicate
        }),
        "optimized dimension-side semijoin should expose a reusable date_sk dynamic-filter route to the fact source"
    );
}

#[test]
fn semijoin_reduction_strict_gate_allows_later_passes_to_route_dimension_keyset_filter() {
    let plan = root_semijoin_over_fact_date_dimension_plan();
    let options = NativeOptimizerOptions {
        semijoin_reduction: SemijoinReductionOptions::default()
            .with_admission_mode(SemijoinReductionAdmissionMode::StrictBounded),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(
        candidate.admission_decision,
        Some(nexus_query_engine::SemijoinReductionAdmissionDecision::StrictBoundedKept)
    );
    assert_eq!(candidate.original_admitted, Some(false));
    assert_eq!(candidate.rewritten_admitted, Some(false));
    assert!(
        result.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(&route.producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. })
                && route.producer_columns == vec!["d_date_sk".to_string()]
                && route.consumer_columns == vec!["ss_sold_date_sk".to_string()]
                && matches!(&route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
                && route.consumer_lowering == DynamicFilterConsumerLowering::ReaderPredicate
        }),
        "strict-gated semijoin rewrite should remain available to downstream dynamic-filter routing"
    );
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
fn semijoin_reduction_strict_admission_gate_keeps_rewrite_when_original_is_not_yet_admitted() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Right);
    let options = semijoin_reduction_only_options(
        SemijoinReductionOptions::default().with_admission_mode(SemijoinReductionAdmissionMode::StrictBounded),
    );

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(
        candidate.admission_decision,
        Some(nexus_query_engine::SemijoinReductionAdmissionDecision::StrictBoundedKept)
    );
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

fn single_source_plan() -> nexus_query_engine::QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source)
}

fn estimated_inner_equi_join_plan(left_bytes: u64, right_bytes: u64) -> nexus_query_engine::QueryPlan {
    estimated_inner_join_plan(JoinSide::Right, left_bytes, 1, right_bytes, 1)
}

fn estimated_inner_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> nexus_query_engine::QueryPlan {
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
) -> nexus_query_engine::QueryPlan {
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

fn repeated_grouped_aggregate_materialization_plan() -> nexus_query_engine::QueryPlan {
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

fn repeated_aliased_grouped_aggregate_materialization_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source_schema = schema(&["county", "sales"]);
    let aggregate_schema = schema(&["county_name", "store_sales"]);
    let source = builder.add_source(estimated_local_source_with_stable_id(
        "store_sales_alias",
        "store_sales_alias",
        &["county", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("county").alias("county_name")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        source,
        SortSpec::new(vec![SortExpr::new(Expr::column("county"), false, false)]),
        source_schema,
    );
    let sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("county").alias("county_name")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let aggregates = vec![aggregate, sorted_aggregate];
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_predicate_alias_collision_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_schema = schema(&["county", "quarter", "year", "store_sales"]);
    let mut aggregates = Vec::new();
    for value in 1..=2 {
        let source = builder.add_source(estimated_sales_source_with_store_sales_predicate(value));
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

fn repeated_grouped_aggregate_predicate_group_name_lineage_collision_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let join_schema = schema(&["customer_id", "sales"]);
    let aggregate_schema = schema(&["customer_id", "store_sales"]);
    let mut aggregates = Vec::new();
    for customer_id in 1..=2 {
        let fact = builder.add_source(estimated_local_source_with_stable_id(
            "fact_sales",
            "fact_sales",
            &["customer_id", "dim_key", "sales"],
            512 * 1024 * 1024,
            10_000_000,
        ));
        let dim = builder.add_source(estimated_customer_dimension_source_with_customer_predicate(customer_id));
        let joined = builder.add_join(
            fact,
            dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "dim_key", "dim_key"),
                    ColumnRef::input(1, "dim_key", "dim_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
            ]),
            Arc::clone(&join_schema),
        );
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                vec![Expr::column("customer_id")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_self_join_duplicate_source_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let join_schema = schema(&["customer_id", "sales"]);
    let aggregate_schema = schema(&["customer_id", "store_sales"]);
    let left = builder.add_source(estimated_customer_fact_source_with_customer_predicate(1));
    let right = builder.add_source(estimated_customer_fact_source_with_customer_predicate(2));
    let joined = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_key", "dim_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        joined,
        SortSpec::new(vec![SortExpr::new(Expr::column("customer_id"), false, false)]),
        join_schema,
    );
    let sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let union = builder.add_union(vec![aggregate, sorted_aggregate], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_reused_source_node_self_join_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let join_schema = schema(&["customer_id", "sales"]);
    let aggregate_schema = schema(&["customer_id", "store_sales"]);
    let source = builder.add_source(estimated_customer_fact_source_with_customer_predicate(1));
    let joined = builder.add_join(
        source,
        source,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_key", "dim_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        joined,
        SortSpec::new(vec![SortExpr::new(Expr::column("customer_id"), false, false)]),
        join_schema,
    );
    let sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let union = builder.add_union(vec![aggregate, sorted_aggregate], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn unreachable_grouped_aggregate_materialization_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let root = builder.add_source(estimated_local_source("root", "root_key", 1024, 10));
    let aggregate_schema = schema(&["county", "store_sales"]);
    let source = builder.add_source(estimated_local_source_with_stable_id(
        "unreachable_sales",
        "unreachable_sales",
        &["county", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        source,
        SortSpec::new(vec![SortExpr::new(Expr::column("county"), false, false)]),
        schema(&["county", "sales"]),
    );
    let _sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        aggregate_schema,
    );
    assert_ne!(aggregate, root, "unreachable aggregate should not be the plan root");
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_materialization_plan() -> nexus_query_engine::QueryPlan {
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

fn repeated_scalar_count_materialization_with_mismatched_source_contracts_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for (hour, fact_contract) in
        [8, 9].into_iter().zip([remote_direct_customer_address_contract(), host_buffer_customer_address_contract()])
    {
        let fact = builder.add_source(estimated_source_with_resolved_contract(
            fact_contract,
            "store_sales",
            &["ss_sold_time_sk"],
            512 * 1024 * 1024,
            10_000_000,
        ));
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

fn repeated_scalar_count_materialization_with_mismatched_decimal_contracts_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for (hour, decimal_contract) in [8, 9].into_iter().zip([false, true]) {
        let mut fact_source = estimated_source_with_resolved_contract(
            remote_direct_customer_address_contract(),
            "store_sales",
            &["ss_sold_time_sk"],
            512 * 1024 * 1024,
            10_000_000,
        );
        if decimal_contract {
            let SourceNode::LocalParquet(source) = &mut fact_source else {
                unreachable!("estimated source helper always builds local Parquet")
            };
            source.files[0].reader_decimal_types =
                vec![ParquetReaderDecimalType::new("ss_sold_time_sk", ParquetReaderDecimalTypeId::Decimal64, -2)];
            source.refresh_diagnostics();
        }
        let fact = builder.add_source(fact_source);
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

fn repeated_scalar_count_materialization_with_redundant_projection_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for (hour, projection_depth) in [(8, 1), (9, 3)] {
        let fact =
            builder.add_source(estimated_local_source("store_sales", "ss_sold_time_sk", 512 * 1024 * 1024, 10_000_000));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let time_key = builder.add_projection(
            time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let mut input = builder.add_join(
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
        for _ in 0..projection_depth {
            input = builder.add_projection(
                input,
                vec![ProjectionOutput::new(
                    Expr::column("ss_sold_time_sk"),
                    Field::new("ss_sold_time_sk", DataType::Int32, false),
                )],
            );
        }
        let count = builder.add_aggregate(
            input,
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

fn repeated_scalar_count_materialization_with_reader_predicate_filters_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let time_schema = Arc::new(Schema::new(vec![
        Field::new("t_time_sk", DataType::Int32, false),
        Field::new("t_hour", DataType::Int32, false),
    ]));
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for hour in [8, 9] {
        let fact =
            builder.add_source(estimated_local_source("store_sales", "ss_sold_time_sk", 512 * 1024 * 1024, 10_000_000));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let predicate = Expr::binary(Expr::column("t_hour"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(hour)));
        let filtered_time = builder.add_filter(time, predicate, Arc::clone(&time_schema));
        let time_key = builder.add_projection(
            filtered_time,
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

fn repeated_scalar_mixed_aggregate_materialization_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..3 {
        let source = builder.add_source(q28_like_store_sales_source(bucket));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new(format!("B{bucket}_LP"), DataType::Float64, true),
            Field::new(format!("B{bucket}_CNT"), DataType::Int64, false),
            Field::new(format!("B{bucket}_CNTD"), DataType::Int64, false),
        ]));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("ss_list_price"), format!("B{bucket}_LP")),
                    AggregateExpr::new(
                        AggregateFunction::Count,
                        Expr::column("ss_list_price"),
                        format!("B{bucket}_CNT"),
                    ),
                    AggregateExpr::new(
                        AggregateFunction::Count,
                        Expr::column("ss_list_price"),
                        format!("B{bucket}_CNTD"),
                    )
                    .with_distinct(true),
                ],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    let output_schema = Arc::new(Schema::new(
        branch_schemas
            .iter()
            .flat_map(|schema| schema.fields().iter().map(|field| field.as_ref().clone()))
            .collect::<Vec<_>>(),
    ));
    let root = builder.add_limit(root, LimitSpec::fetch(100), output_schema);
    builder.finish_gpu_resident(root)
}

#[derive(Debug, Clone, Copy)]
enum Q09ScalarBranch {
    Count,
    AvgDiscount,
    AvgNetPaid,
}

fn repeated_q09_like_scalar_materialization_plan() -> nexus_query_engine::QueryPlan {
    repeated_q09_like_scalar_materialization_plan_with_order(&[
        Q09ScalarBranch::Count,
        Q09ScalarBranch::AvgDiscount,
        Q09ScalarBranch::AvgNetPaid,
    ])
}

fn repeated_q09_like_scalar_materialization_plan_with_order(
    branch_order: &[Q09ScalarBranch],
) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        for &branch in branch_order {
            let (aggregate_input, aggregate_expr, field_name, data_type) = match branch {
                Q09ScalarBranch::Count => {
                    let source = builder.add_source(q09_filter_only_count_source(bucket));
                    let count_arg = builder.add_projection(
                        source,
                        vec![
                            ProjectionOutput::new(
                                Expr::literal(LiteralValue::Int64(1)),
                                Field::new("__cudf_aggregate_arg_0", DataType::Int64, false),
                            ),
                            ProjectionOutput::new(
                                Expr::column("ss_quantity"),
                                Field::new("ss_quantity", DataType::Int32, false),
                            ),
                        ],
                    );
                    (
                        count_arg,
                        AggregateExpr::new(
                            AggregateFunction::Count,
                            Expr::column("__cudf_aggregate_arg_0"),
                            format!("B{bucket}_CNT"),
                        ),
                        format!("B{bucket}_CNT"),
                        DataType::Int64,
                    )
                },
                Q09ScalarBranch::AvgDiscount => {
                    let source = builder.add_source(q09_like_store_sales_source(bucket, branch));
                    (
                        source,
                        AggregateExpr::new(
                            AggregateFunction::Avg,
                            Expr::column("ss_ext_discount_amt"),
                            format!("B{bucket}_AVG_DISCOUNT"),
                        ),
                        format!("B{bucket}_AVG_DISCOUNT"),
                        DataType::Float64,
                    )
                },
                Q09ScalarBranch::AvgNetPaid => {
                    let source = builder.add_source(q09_like_store_sales_source(bucket, branch));
                    (
                        source,
                        AggregateExpr::new(
                            AggregateFunction::Avg,
                            Expr::column("ss_net_paid"),
                            format!("B{bucket}_AVG_NET"),
                        ),
                        format!("B{bucket}_AVG_NET"),
                        DataType::Float64,
                    )
                },
            };
            let output_schema = Arc::new(Schema::new(vec![Field::new(field_name, data_type, true)]));
            let aggregate = builder.add_aggregate(
                aggregate_input,
                AggregateSpec::new(Vec::new(), vec![aggregate_expr]),
                Arc::clone(&output_schema),
            );
            branches.push(aggregate);
            branch_schemas.push(output_schema);
        }
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_projected_literal_count_star_scalar_materialization_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(q09_filter_only_count_source(bucket));
        let empty = builder.add_projection(source, Vec::new());
        let count_arg = builder.add_projection(
            empty,
            vec![ProjectionOutput::new(
                Expr::literal(LiteralValue::Int64(1)),
                Field::new("__cudf_aggregate_arg_0", DataType::Int64, false),
            )],
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            count_arg,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::column("__cudf_aggregate_arg_0"),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_with_different_projected_filter_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for projected_column in ["a", "b"] {
        let source = builder.add_source(two_value_projection_filter_source());
        let projection_schema = Arc::new(Schema::new(vec![Field::new("p", DataType::Int32, false)]));
        let projected = builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column(projected_column), Field::new("p", DataType::Int32, false))],
        );
        let filtered = builder.add_filter(
            projected,
            Expr::binary(Expr::column("p"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
            projection_schema,
        );
        let output_schema =
            Arc::new(Schema::new(vec![Field::new(format!("CNT_{projected_column}"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            filtered,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("CNT_{projected_column}"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_with_different_local_filter_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(q09_like_store_sales_source(bucket, Q09ScalarBranch::Count));
        let filter_schema = Arc::new(Schema::new(vec![Field::new("ss_quantity", DataType::Int32, false)]));
        let filtered = builder.add_filter(
            source,
            Expr::binary(
                Expr::column("ss_quantity"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Int32(bucket * 20 + 10)),
            ),
            filter_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            filtered,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_avg_with_conflicting_source_refs_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for file_index in [0, 1] {
        let source = builder.add_source(conflicting_ref_value_source(file_index));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new(format!("AVG_V_{file_index}"), DataType::Float64, true)]));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), format!("AVG_V_{file_index}"))],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_with_limit_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(q09_like_store_sales_source(bucket, Q09ScalarBranch::Count));
        let limit_schema = Arc::new(Schema::new(vec![Field::new("ss_quantity", DataType::Int32, false)]));
        let limited = builder.add_limit(source, LimitSpec::fetch(10), limit_schema);
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            limited,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_join_count_with_duplicate_input_names_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let left = builder.add_source(duplicate_named_source("left", bucket));
        let right = builder.add_source(duplicate_named_source("right", 0));
        let join_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let joined = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "v", "v"), ColumnRef::input(1, "v", "v"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v")]),
            join_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_self_join_count_with_duplicate_rowset_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let left = builder.add_source(duplicate_named_source("same-rowset", bucket));
        let right = builder.add_source(duplicate_named_source("same-rowset", 0));
        let join_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let joined = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "v", "v"), ColumnRef::input(1, "v", "v"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v")]),
            join_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_self_join_count_with_reused_source_node_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(duplicate_named_source("same-reused-rowset", bucket));
        let join_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let joined = builder.add_join(
            source,
            source,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "v", "v"), ColumnRef::input(1, "v", "v"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v")]),
            join_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn grouped_and_scalar_sum_over_permuted_source_files_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let grouped_input = builder.add_source(two_file_source(["a", "b"]));
    let scalar_input = builder.add_source(two_file_source(["b", "a"]));
    let grouped = builder.add_aggregate(
        grouped_input,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "sum_v")],
        ),
        schema(&["k", "sum_v"]),
    );
    let scalar = builder.add_aggregate(
        scalar_input,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total_v")]),
        schema(&["total_v"]),
    );
    let root = builder.add_join(
        grouped,
        scalar,
        JoinSpec::new(JoinType::Inner, Vec::new()).with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_v", "sum_v"), "sum_v"),
            JoinOutputColumn::right(ColumnRef::input(1, "total_v", "total_v"), "total_v"),
        ]),
        schema(&["k", "sum_v", "total_v"]),
    );
    builder.finish_gpu_resident(root)
}

fn two_file_source(order: [&str; 2]) -> SourceNode {
    let source_schema = schema(&["k", "v"]);
    let files = order
        .into_iter()
        .map(|name| {
            LocalParquetFile::new(format!("/tmp/{name}.parquet"))
                .with_stable_id(name)
                .with_projected_row_group_bytes(vec![1024], vec![1024])
                .with_row_group_rows(vec![10])
        })
        .collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        files,
        source_schema,
        vec!["k".to_string(), "v".to_string()],
    ))
}

fn shared_fact_target_dimension_pruning_plan() -> nexus_query_engine::QueryPlan {
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

fn aggregate_group_filter_dimension_pruning_plan(dir: &Path) -> nexus_query_engine::QueryPlan {
    aggregate_group_filter_dimension_pruning_plan_with_predicate(
        dir,
        Expr::binary(Expr::column("ss_sold_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
    )
}

fn aggregate_group_filter_dimension_pruning_plan_with_predicate(
    dir: &Path,
    filter_predicate: Expr,
) -> nexus_query_engine::QueryPlan {
    let fact_path = dir.join("store_sales.parquet");
    let date_path = dir.join("date_dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
    ]));
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        vec![
            Int32Array::from(vec![1, 2, 3, 4]),
            Int32Array::from(vec![10, 20, 30, 40]),
            Int32Array::from(vec![1, 1, 1, 1]),
        ],
        1,
    );
    write_i32_record_batch_parquet(
        &date_path,
        Arc::clone(&date_schema),
        vec![Int32Array::from(vec![1, 2, 3, 4]), Int32Array::from(vec![1999, 2000, 2000, 2001])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["ss_sold_date_sk".to_string(), "ss_item_sk".to_string(), "ss_quantity".to_string()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&date_path)],
        Arc::clone(&date_schema),
        vec!["d_date_sk".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let joined = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"), "ss_sold_date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_item_sk", "ss_item_sk"), "ss_item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_quantity", "ss_quantity"), "ss_quantity"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_year", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_qty", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year").alias("ss_sold_year"), Expr::column("ss_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ss_quantity"), "ss_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(aggregate, filter_predicate, aggregate_schema);
    builder.finish_gpu_resident(filtered)
}

fn shared_aggregate_group_filter_dimension_pruning_plan(dir: &Path) -> nexus_query_engine::QueryPlan {
    let fact_path = dir.join("store_sales.parquet");
    let date_path = dir.join("date_dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
    ]));
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        vec![
            Int32Array::from(vec![1, 2, 3, 4]),
            Int32Array::from(vec![10, 20, 30, 40]),
            Int32Array::from(vec![1, 1, 1, 1]),
        ],
        1,
    );
    write_i32_record_batch_parquet(
        &date_path,
        Arc::clone(&date_schema),
        vec![Int32Array::from(vec![1, 2, 3, 4]), Int32Array::from(vec![1999, 2000, 2000, 2001])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["ss_sold_date_sk".to_string(), "ss_item_sk".to_string(), "ss_quantity".to_string()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&date_path)],
        Arc::clone(&date_schema),
        vec!["d_date_sk".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let joined = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"), "ss_sold_date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_item_sk", "ss_item_sk"), "ss_item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_quantity", "ss_quantity"), "ss_quantity"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_year", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_qty", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year").alias("ss_sold_year"), Expr::column("ss_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ss_quantity"), "ss_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(
        aggregate,
        Expr::binary(Expr::column("ss_sold_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
        Arc::clone(&aggregate_schema),
    );
    let union = builder.add_union(vec![filtered, aggregate], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn aggregate_right_group_filter_left_join_plan(dir: &Path) -> nexus_query_engine::QueryPlan {
    let fact_path = dir.join("left_fact.parquet");
    let dim_path = dir.join("right_dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(&fact_path, Arc::clone(&fact_schema), vec![Int32Array::from(vec![1, 2, 3])], 1);
    write_i32_record_batch_parquet(
        &dim_path,
        Arc::clone(&dim_schema),
        vec![Int32Array::from(vec![1, 2]), Int32Array::from(vec![1999, 2000])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["fact_key".to_string()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&dim_path)],
        Arc::clone(&dim_schema),
        vec!["dim_key".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, true),
    ]));
    let joined = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        join_schema,
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("d_year", DataType::Int32, true),
        Field::new("cnt", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "cnt")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(
        aggregate,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
        aggregate_schema,
    );
    builder.finish_gpu_resident(filtered)
}

fn write_i32_record_batch_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<Int32Array>, row_group_rows: usize) {
    let arrays = columns.into_iter().map(|column| Arc::new(column) as Arc<dyn arrow::array::Array>).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).expect("test batch should be valid");
    let file = File::create(path).expect("test parquet file should be creatable");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("test parquet writer should be creatable");
    writer.write(&batch).expect("test parquet batch should be writable");
    writer.close().expect("test parquet writer should close");
}

fn unique_contract_temp_dir(label: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock should be monotonic").as_nanos();
    let dir = std::env::temp_dir().join(format!("datafusion-nexus-{label}-{id}"));
    fs::create_dir_all(&dir).expect("temporary test directory should be creatable");
    dir
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

fn estimated_sales_source_with_store_sales_predicate(value: i32) -> SourceNode {
    let source_schema = schema(&["county", "quarter", "year", "sales", "store_sales"]);
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    let predicate = Expr::binary(Expr::column("store_sales"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(value)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec![
                "county".to_string(),
                "quarter".to_string(),
                "year".to_string(),
                "sales".to_string(),
                "store_sales".to_string(),
            ],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_customer_dimension_source_with_customer_predicate(customer_id: i32) -> SourceNode {
    let source_schema = schema(&["dim_key", "customer_id"]);
    let file = LocalParquetFile::new("/tmp/customer_dim.parquet")
        .with_stable_id("customer_dim")
        .with_projected_row_group_bytes(vec![1024 * 1024], vec![1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    let predicate =
        Expr::binary(Expr::column("customer_id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(customer_id)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["dim_key".to_string(), "customer_id".to_string()],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_customer_fact_source_with_customer_predicate(customer_id: i32) -> SourceNode {
    let source_schema = schema(&["dim_key", "customer_id", "sales"]);
    let file = LocalParquetFile::new("/tmp/customer_fact.parquet")
        .with_stable_id("customer_fact")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    let predicate =
        Expr::binary(Expr::column("customer_id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(customer_id)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["dim_key".to_string(), "customer_id".to_string(), "sales".to_string()],
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

fn q28_like_store_sales_source(bucket: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_list_price", DataType::Int32, true),
        Field::new("ss_coupon_amt", DataType::Int32, true),
        Field::new("ss_wholesale_cost", DataType::Int32, true),
    ]));
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec![
                "ss_quantity".to_string(),
                "ss_list_price".to_string(),
                "ss_coupon_amt".to_string(),
                "ss_wholesale_cost".to_string(),
            ],
        )
        .with_reader_predicate(q28_like_bucket_predicate(bucket)),
    )
}

fn q09_like_store_sales_source(bucket: i32, branch: Q09ScalarBranch) -> SourceNode {
    let all_fields = [
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_ext_discount_amt", DataType::Float64, true),
        Field::new("ss_net_paid", DataType::Float64, true),
    ];
    let read_columns = match branch {
        Q09ScalarBranch::Count => vec!["ss_quantity".to_string()],
        Q09ScalarBranch::AvgDiscount => vec!["ss_quantity".to_string(), "ss_ext_discount_amt".to_string()],
        Q09ScalarBranch::AvgNetPaid => vec!["ss_quantity".to_string(), "ss_net_paid".to_string()],
    };
    let fields = read_columns
        .iter()
        .map(|column| {
            all_fields.iter().find(|field| field.name() == column).expect("read column should have a field").clone()
        })
        .collect::<Vec<_>>();
    let source_schema = Arc::new(Schema::new(fields));
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, read_columns).with_reader_predicate(between_i32(
            "ss_quantity",
            bucket * 20 + 1,
            bucket * 20 + 20,
        )),
    )
}

fn q09_filter_only_count_source(bucket: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![Field::new("ss_quantity", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["ss_quantity".to_string()])
            .with_column_roles(Vec::new(), vec!["ss_quantity".to_string()])
            .with_reader_predicate(between_i32("ss_quantity", bucket * 20 + 1, bucket * 20 + 20)),
    )
}

fn two_value_projection_filter_source() -> SourceNode {
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false), Field::new("b", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/two-value-projection-filter.parquet")
        .with_stable_id("two-value-projection-filter")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        source_schema,
        vec!["a".to_string(), "b".to_string()],
    ))
}

fn conflicting_ref_value_source(file_index: usize) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
    let file = LocalParquetFile::new("/tmp/conflicting-value-source.parquet")
        .with_stable_id("conflicting-value-source")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["v".to_string()])
            .with_column_file_indices(vec![file_index], vec![file_index], Vec::new())
            .with_reader_predicate(Expr::binary(
                Expr::column("v"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Float64(0.0)),
            )),
    )
}

fn duplicate_named_source(name: &str, bucket: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
    let file = LocalParquetFile::new(format!("/tmp/{name}-duplicate-v.parquet"))
        .with_stable_id(name)
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["v".to_string()]).with_reader_predicate(
            Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(bucket))),
        ),
    )
}

fn q28_like_bucket_predicate(bucket: i32) -> Expr {
    let quantity_min = bucket * 5;
    let quantity_max = quantity_min + 5;
    let list_min = 10 + bucket * 20;
    let coupon_min = 1000 + bucket * 100;
    let cost_min = 30 + bucket * 10;
    Expr::binary(
        between_i32("ss_quantity", quantity_min, quantity_max),
        BinaryOp::And,
        Expr::binary(
            between_i32("ss_list_price", list_min, list_min + 10),
            BinaryOp::Or,
            Expr::binary(
                between_i32("ss_coupon_amt", coupon_min, coupon_min + 1000),
                BinaryOp::Or,
                between_i32("ss_wholesale_cost", cost_min, cost_min + 20),
            ),
        ),
    )
}

fn between_i32(column: &str, min: i32, max: i32) -> Expr {
    Expr::binary(
        Expr::binary(Expr::column(column), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(min))),
        BinaryOp::And,
        Expr::binary(Expr::column(column), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(max))),
    )
}

fn cross_join_scalar_branches(
    builder: &mut QueryPlanBuilder,
    branches: &[NodeId],
    branch_schemas: &[Arc<Schema>],
) -> NodeId {
    let mut current = branches[0];
    let mut current_fields = branch_schemas[0].fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    for (branch, schema) in branches.iter().copied().zip(branch_schemas).skip(1) {
        let mut output_projection = Vec::new();
        for field in &current_fields {
            output_projection.push(JoinOutputColumn::left(
                ColumnRef::input(0, field.name().clone(), field.name().clone()),
                field.name().clone(),
            ));
        }
        for field in schema.fields() {
            output_projection.push(JoinOutputColumn::right(
                ColumnRef::input(1, field.name().clone(), field.name().clone()),
                field.name().clone(),
            ));
            current_fields.push(field.as_ref().clone());
        }
        let output_schema = Arc::new(Schema::new(current_fields.clone()));
        current = builder.add_join(
            current,
            branch,
            JoinSpec::new(JoinType::Inner, Vec::new()).with_output_projection(output_projection),
            output_schema,
        );
    }
    current
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
    let file_indices = (0..field_names.len()).collect::<Vec<_>>();
    estimated_local_source_with_stable_id_and_file_indices(name, stable_id, field_names, &file_indices, bytes, rows)
}

fn estimated_local_source_with_stable_id_and_file_indices(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    file_indices: &[usize],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(stable_id)
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![rows]);
    let read_columns = field_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(vec![file], source_schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    );
    SourceNode::LocalParquet(source)
}

fn estimated_source_with_resolved_contract(
    resolved_source: ResolvedParquetSource,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
        .unwrap()
        .with_stable_id(stable_id)
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![rows]);
    let read_columns = field_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(vec![file], source_schema, read_columns))
}

fn estimated_local_source_with_stable_id_and_predicate(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
    predicate: Expr,
) -> SourceNode {
    let SourceNode::LocalParquet(source) =
        estimated_local_source_with_stable_id(name, stable_id, field_names, bytes, rows)
    else {
        unreachable!("estimated source helper always builds local Parquet")
    };
    SourceNode::LocalParquet(source.with_reader_predicate(predicate))
}

#[derive(Debug, Clone, Copy)]
enum SemijoinTarget {
    Left,
    Right,
}

fn root_semijoin_over_inner_join_plan(target: SemijoinTarget) -> nexus_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, true)
}

fn root_semijoin_over_inner_join_plan_with_raw_keyset(target: SemijoinTarget) -> nexus_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, false)
}

fn root_semijoin_over_inner_join_plan_impl(
    target: SemijoinTarget,
    aggregate_keyset: bool,
) -> nexus_query_engine::QueryPlan {
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
) -> nexus_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        SemijoinTarget::Right,
        true,
        JoinType::Inner,
        semijoin_null_equality,
        crossed_join_null_equality,
    )
}

fn root_semijoin_over_crossed_join_type_plan(join_type: JoinType) -> nexus_query_engine::QueryPlan {
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
) -> nexus_query_engine::QueryPlan {
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

fn root_semijoin_over_duplicate_output_name_plan() -> nexus_query_engine::QueryPlan {
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

fn root_semijoin_over_expression_key_projection_plan() -> nexus_query_engine::QueryPlan {
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

fn root_semijoin_over_dropped_key_projection_plan() -> nexus_query_engine::QueryPlan {
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

fn root_semijoin_over_nested_inner_join_plan() -> nexus_query_engine::QueryPlan {
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

fn root_semijoin_over_fact_date_dimension_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "store_sales",
        &["ss_sold_date_sk", "ss_ext_sales_price"],
        256 * 1024 * 1024,
        10_000_000,
    ));
    let date_dim = builder.add_source(estimated_local_source_with_fields(
        "date_dim",
        &["d_date_sk", "d_date", "d_week_seq"],
        1024 * 1024,
        73_049,
    ));

    let sales_with_date_schema = schema(&["ss_sold_date_sk", "ss_ext_sales_price", "d_date_sk", "d_date"]);
    let sales_with_date = builder.add_join(
        fact,
        date_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"), "ss_sold_date_sk"),
            JoinOutputColumn::left(
                ColumnRef::input(0, "ss_ext_sales_price", "ss_ext_sales_price"),
                "ss_ext_sales_price",
            ),
            JoinOutputColumn::right(ColumnRef::input(1, "d_date_sk", "d_date_sk"), "d_date_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_date", "d_date"), "d_date"),
        ]),
        Arc::clone(&sales_with_date_schema),
    );

    let week_dates = week_dates_from_filtered_date_dimension(&mut builder, date_dim);
    let semi = builder.add_join(
        sales_with_date,
        week_dates,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "d_date", "d_date"), ColumnRef::input(1, "d_date", "d_date"))],
        )
        .with_output_projection(left_projection(&[
            "ss_sold_date_sk",
            "ss_ext_sales_price",
            "d_date_sk",
            "d_date",
        ])),
        sales_with_date_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn week_dates_from_filtered_date_dimension(
    builder: &mut QueryPlanBuilder,
    date_dim: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let all_dates = builder.add_projection(
        date_dim,
        vec![
            ProjectionOutput::new(Expr::column("d_date"), Field::new("d_date", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false)),
        ],
    );
    let filtered_date = builder.add_filter(
        date_dim,
        Expr::binary(Expr::column("d_date"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(10_959))),
        schema(&["d_date_sk", "d_date", "d_week_seq"]),
    );
    let filtered_week = builder.add_projection(
        filtered_date,
        vec![ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false))],
    );
    builder.add_join(
        all_dates,
        filtered_week,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "d_week_seq", "d_week_seq"),
                ColumnRef::input(1, "d_week_seq", "d_week_seq"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "d_date", "d_date"), "d_date")]),
        schema(&["d_date"]),
    )
}

fn inner_join_over_projected_left_join_right_key_plan() -> nexus_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("r_key")
}

fn inner_join_over_projected_left_join_left_key_plan() -> nexus_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("l_key")
}

fn inner_join_over_projected_left_join_key_plan(rejecting_key: &str) -> nexus_query_engine::QueryPlan {
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

fn shared_left_join_output_with_one_null_rejecting_consumer_plan() -> nexus_query_engine::QueryPlan {
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

#[derive(Debug, Clone, Copy)]
enum MarkerNullFilterPlanVariant {
    Eligible,
    NullableMarker,
    NullsMatch,
    MarkerEscapes,
}

fn marker_null_filter_left_join_plan(variant: MarkerNullFilterPlanVariant) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key", "payload"]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("r_key", DataType::Int32, matches!(variant, MarkerNullFilterPlanVariant::NullableMarker)),
        Field::new("r_payload", DataType::Int32, false),
    ]));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", right_schema)));
    let marker_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, true),
    ]));
    let mut spec = JoinSpec::new(
        JoinType::Left,
        vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
    )
    .with_output_projection(vec![
        JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
        JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
        JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
    ])
    .with_build_side(JoinSide::Right);
    if matches!(variant, MarkerNullFilterPlanVariant::NullsMatch) {
        spec = spec.with_null_equality(JoinNullEquality::NullsMatch);
    }
    let left_join = builder.add_join(left, right, spec, Arc::clone(&marker_schema));
    let filtered = builder.add_filter(left_join, Expr::is_null(Expr::column("r_key")), marker_schema);
    let projected = if matches!(variant, MarkerNullFilterPlanVariant::MarkerEscapes) {
        builder.add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
            ],
        )
    } else {
        builder.add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
            ],
        )
    };
    builder.finish_gpu_resident(projected)
}

fn inner_join_filter_over_projected_inner_join_plan() -> nexus_query_engine::QueryPlan {
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

fn nested_inner_join_filter_plan() -> nexus_query_engine::QueryPlan {
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

fn shared_projected_probe_fanout_plan() -> nexus_query_engine::QueryPlan {
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

fn compatible_small_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
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

fn contract_mismatched_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let direct = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        remote_direct_customer_address_contract(),
    )));
    let host_buffer = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        host_buffer_customer_address_contract(),
    )));
    let direct_id = builder.add_projection(
        direct,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let host_buffer_id = builder.add_projection(
        host_buffer,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![direct_id, host_buffer_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_sensitive_stable_id_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let full = builder.add_source(sensitive_stable_id_customer_address_source(&["id", "state"], &[0, 1], None));
    let filtered = builder.add_source(sensitive_stable_id_customer_address_source(
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

fn decimal_contract_mismatched_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let plain = builder.add_source(SourceNode::LocalParquet(contract_sensitive_customer_address_source(
        remote_direct_customer_address_contract(),
    )));
    let decimal = builder.add_source(SourceNode::LocalParquet(
        contract_sensitive_customer_address_source(remote_direct_customer_address_contract())
            .with_reader_decimal_contract("id", ParquetReaderDecimalTypeId::Decimal64, -2),
    ));
    let plain_id = builder.add_projection(
        plain,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let decimal_id = builder.add_projection(
        decimal,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![plain_id, decimal_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_local_parquet_source_variants_with_different_file_order_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let first = builder.add_source(customer_address_two_file_source(&["id", "state"], &[0, 1], false));
    let second = builder.add_source(customer_address_two_file_source(&["id", "zip"], &[0, 2], true));
    let first_id = builder.add_projection(
        first,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let second_id = builder.add_projection(
        second,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![first_id, second_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn compatible_small_local_parquet_source_variants_with_max_rows_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let full = builder.add_source(customer_address_source(&["id", "state"], &[0, 1], None));
    let filtered = builder.add_source(customer_address_source_with_max_rows(
        &["id", "zip"],
        &[0, 2],
        Some(Expr::binary(Expr::column("zip"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(30301)))),
        Some(1),
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

fn conflicting_file_index_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let id_source = builder.add_source(customer_address_source(&["id"], &[0], None));
    let zip_source = builder.add_source(customer_address_source(&["zip"], &[0], None));
    let id = builder.add_projection(
        id_source,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let zip_as_id = builder.add_projection(
        zip_source,
        vec![ProjectionOutput::new(Expr::column("zip").alias("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![id, zip_as_id], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn conflicting_column_name_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let first = builder.add_source(customer_address_source(&["id"], &[0], None));
    let second = builder.add_source(customer_address_source(&["id"], &[1], None));
    let first = builder.add_projection(
        first,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let second = builder.add_projection(
        second,
        vec![ProjectionOutput::new(Expr::column("id"), Field::new("id", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![first, second], schema(&["id"]));
    builder.finish_gpu_resident(union)
}

fn graph_bfs_local_source_reuse_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(graph_reuse_source(&["src", "dst", "weight"], &[0, 1, 2]));
    let target_vertices = builder.add_source(graph_reuse_source(&["vertex_id"], &[3]));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            target_vertices: Some(GraphPredicateInput::new(target_vertices, "vertex_id")),
            ..Default::default()
        },
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    builder.finish_gpu_resident(graph)
}

fn graph_reuse_source(columns: &[&str], file_indices: &[usize]) -> SourceNode {
    let fields = columns.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let file = LocalParquetFile::new("/tmp/graph_inputs.parquet")
        .with_stable_id("graph_inputs")
        .with_row_group_bytes(vec![1024 * 1024])
        .with_row_group_rows(vec![4096]);
    let read_columns = columns.iter().map(|column| (*column).to_string()).collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(vec![file], schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    ))
}

fn large_local_parquet_source_variants_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let right = builder.add_source(estimated_local_source_with_stable_id_and_file_indices(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_store_sk"],
        &[0, 2],
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

fn large_superset_with_smaller_local_parquet_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        600 * 1024 * 1024,
        10_000_000,
    ));
    let subset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let superset_item = builder.add_projection(
        superset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let subset_item = builder.add_projection(
        subset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![superset_item, subset_item], schema(&["inv_item_sk"]));
    builder.finish_gpu_resident(union)
}

fn large_superset_with_grouped_aggregate_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity", "inv_price"],
        600 * 1024 * 1024,
        10_000_000,
    ));
    let aggregate_input = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let aggregate_schema = schema(&["inv_item_sk", "sum_qty"]);
    let aggregate = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            vec![Expr::column("inv_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("inv_quantity"), "sum_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let superset_item = builder.add_projection(
        superset,
        vec![
            ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("inv_quantity"), Field::new("sum_qty", DataType::Int32, false)),
        ],
    );
    let union = builder.add_union(vec![aggregate, superset_item], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn large_grouped_aggregate_superset_with_smaller_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_input = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_quantity", "inv_price"],
        600 * 1024 * 1024,
        10_000_000,
    ));
    let subset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let aggregate_schema = schema(&["inv_item_sk", "sum_qty"]);
    let aggregate = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            vec![Expr::column("inv_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("inv_quantity"), "sum_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let subset_item = builder.add_projection(
        subset,
        vec![
            ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::literal(LiteralValue::Int32(0)), Field::new("sum_qty", DataType::Int32, false)),
        ],
    );
    let union = builder.add_union(vec![aggregate, subset_item], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn large_filtered_superset_with_smaller_local_parquet_source_variant_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let superset = builder.add_source(estimated_local_source_with_stable_id_and_predicate(
        "inventory",
        "inventory",
        &["inv_item_sk", "inv_date_sk"],
        600 * 1024 * 1024,
        10_000_000,
        nested_reader_predicate_with_declared_boolean_output_types(),
    ));
    let subset = builder.add_source(estimated_local_source_with_stable_id(
        "inventory",
        "inventory",
        &["inv_item_sk"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let superset_item = builder.add_projection(
        superset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let subset_item = builder.add_projection(
        subset,
        vec![ProjectionOutput::new(Expr::column("inv_item_sk"), Field::new("inv_item_sk", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![superset_item, subset_item], schema(&["inv_item_sk"]));
    builder.finish_gpu_resident(union)
}

fn nested_reader_predicate_with_declared_boolean_output_types() -> Expr {
    let date_lower = Expr::binary_with_output_type(
        Expr::column("inv_date_sk"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Int32(2_452_001)),
        DataType::Boolean,
    );
    let date_upper = Expr::binary_with_output_type(
        Expr::column("inv_date_sk"),
        BinaryOp::Lt,
        Expr::literal(LiteralValue::Int32(2_452_010)),
        DataType::Boolean,
    );
    let date_range = Expr::binary_with_output_type(date_lower, BinaryOp::And, date_upper, DataType::Boolean);
    let item_range = Expr::binary_with_output_type(
        Expr::column("inv_item_sk"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Int32(10)),
        DataType::Boolean,
    );
    Expr::binary_with_output_type(date_range, BinaryOp::Or, item_range, DataType::Boolean)
}

fn assert_no_non_decimal_binary_output_types(expr: &Expr, message: &str) {
    match expr {
        Expr::Binary { left, op, right, output_type } => {
            assert!(
                output_type.is_none()
                    || (*op == BinaryOp::Divide
                        && output_type
                            .as_ref()
                            .is_some_and(|data_type| matches!(data_type, DataType::Decimal128(_, _)))),
                "{message}: {expr}"
            );
            assert_no_non_decimal_binary_output_types(left, message);
            assert_no_non_decimal_binary_output_types(right, message);
        },
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::Like { expr: inner, .. }
        | Expr::Substring { expr: inner, .. }
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::StringLength { expr: inner, .. }
        | Expr::RegexpReplace { expr: inner, .. }
        | Expr::Abs(inner)
        | Expr::Sqrt { expr: inner, .. }
        | Expr::Round { expr: inner, .. }
        | Expr::Extract { expr: inner, .. }
        | Expr::DateTrunc { expr: inner, .. }
        | Expr::Alias { expr: inner, .. } => assert_no_non_decimal_binary_output_types(inner, message),
        Expr::Concat(exprs) => {
            for expr in exprs {
                assert_no_non_decimal_binary_output_types(expr, message);
            }
        },
        Expr::InList { expr, list, .. } => {
            assert_no_non_decimal_binary_output_types(expr, message);
            for expr in list {
                assert_no_non_decimal_binary_output_types(expr, message);
            }
        },
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            assert_no_non_decimal_binary_output_types(left, message);
            assert_no_non_decimal_binary_output_types(right, message);
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                assert_no_non_decimal_binary_output_types(&branch.when, message);
                assert_no_non_decimal_binary_output_types(&branch.then, message);
            }
            assert_no_non_decimal_binary_output_types(else_expr, message);
        },
        Expr::Column(_) | Expr::Literal(_) => {},
    }
}

fn customer_address_source(columns: &[&str], file_indices: &[usize], predicate: Option<Expr>) -> SourceNode {
    customer_address_source_with_max_rows(columns, file_indices, predicate, None)
}

fn sensitive_stable_id_customer_address_source(
    columns: &[&str],
    file_indices: &[usize],
    predicate: Option<Expr>,
) -> SourceNode {
    let SourceNode::LocalParquet(mut source) = customer_address_source(columns, file_indices, predicate) else {
        unreachable!("customer address source helper always builds local Parquet")
    };
    source.files[0].stable_id = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc".to_string();
    source.refresh_diagnostics();
    SourceNode::LocalParquet(source)
}

fn contract_sensitive_customer_address_source(resolved_source: ResolvedParquetSource) -> LocalParquetSource {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
        .unwrap()
        .with_stable_id("customer_address")
        .with_row_group_bytes(vec![2 * 1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    LocalParquetSource::from_files(vec![file], schema, vec!["id".to_string()])
}

trait LocalParquetSourceTestExt {
    fn with_reader_decimal_contract(
        self,
        column_name: &str,
        type_id: ParquetReaderDecimalTypeId,
        cudf_scale: i32,
    ) -> Self;
}

impl LocalParquetSourceTestExt for LocalParquetSource {
    fn with_reader_decimal_contract(
        mut self,
        column_name: &str,
        type_id: ParquetReaderDecimalTypeId,
        cudf_scale: i32,
    ) -> Self {
        self.files[0].reader_decimal_types =
            vec![ParquetReaderDecimalType::new(column_name.to_string(), type_id, cudf_scale)];
        self.refresh_diagnostics();
        self
    }
}

fn remote_direct_customer_address_contract() -> ResolvedParquetSource {
    ResolvedParquetSource::remote_direct(
        "s3://warehouse/customer_address.parquet?query_variant=a",
        CredentialMode::SharedProcessEnv,
    )
    .with_remote_read_mode(RemoteParquetReadMode::RequireDirect)
    .with_storage_profile_id("remote-profile-a")
    .with_storage_domain("warehouse@s3")
    .with_content_identity("snapshot=42,file=customer_address,etag=abc")
}

fn host_buffer_customer_address_contract() -> ResolvedParquetSource {
    ResolvedParquetSource::remote_host_buffer(
        "s3://warehouse/customer_address.parquet?query_variant=a",
        "/tmp/nexus-lakehouse-cache/customer_address.parquet",
        CredentialMode::SharedProcessEnv,
    )
    .with_storage_profile_id("remote-profile-a")
    .with_storage_domain("warehouse@s3")
    .with_content_identity("snapshot=42,file=customer_address,etag=abc")
    .with_lakehouse_cache_config("/tmp/nexus-lakehouse-cache", Some(128 * 1024 * 1024))
    .with_lakehouse_cache_path("/tmp/nexus-lakehouse-cache/customer_address.parquet")
    .with_lakehouse_cache_read_mode(LakehouseCacheReadMode::Enabled)
    .with_lakehouse_cache_fill_mode(LakehouseCacheFillMode::Background)
    .with_cache_status(LakehouseCacheStatus::Miss)
}

fn customer_address_two_file_source(columns: &[&str], file_indices: &[usize], reversed: bool) -> SourceNode {
    let fields = columns.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(fields));
    let mut files = vec![
        LocalParquetFile::new("/tmp/customer_address_a.parquet")
            .with_stable_id("customer_address_a")
            .with_row_group_bytes(vec![2 * 1024 * 1024])
            .with_row_group_rows(vec![100_000]),
        LocalParquetFile::new("/tmp/customer_address_b.parquet")
            .with_stable_id("customer_address_b")
            .with_row_group_bytes(vec![2 * 1024 * 1024])
            .with_row_group_rows(vec![100_000]),
    ];
    if reversed {
        files.reverse();
    }
    let read_columns = columns.iter().map(|column| (*column).to_string()).collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(files, schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    );
    SourceNode::LocalParquet(source)
}

fn customer_address_source_with_max_rows(
    columns: &[&str],
    file_indices: &[usize],
    predicate: Option<Expr>,
    max_rows: Option<usize>,
) -> SourceNode {
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
    source = source.with_max_rows(max_rows);
    SourceNode::LocalParquet(source)
}

fn gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
}

fn aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> nexus_query_engine::NodeId {
    let source = builder.add_source(gpu_source(name, &[key_name]));
    builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()), schema(&[key_name]))
}

fn schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

fn plan_has_output_name_prefix(plan: &nexus_query_engine::QueryPlan, prefix: &str) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Aggregate(aggregate) => aggregate.spec.aggr_exprs.iter().any(|expr| expr.alias.starts_with(prefix)),
        PlanNode::Projection(projection) => {
            projection.outputs.iter().any(|output| output.field.name().starts_with(prefix))
        },
        _ => false,
    })
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

fn left_projection(fields: &[&str]) -> Vec<JoinOutputColumn> {
    fields.iter().map(|name| JoinOutputColumn::left(ColumnRef::input(0, *name, *name), *name)).collect()
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

fn assert_all_node_inputs_exist(plan: &nexus_query_engine::QueryPlan) {
    for node in plan.nodes() {
        for input in node.input_ids() {
            assert!(
                plan.node(*input).is_some(),
                "node {} has dangling input reference {} after optimizer rewrite",
                node.id,
                input
            );
        }
    }
}

fn local_parquet_source_count_for_stable_id(plan: &nexus_query_engine::QueryPlan, stable_id: &str) -> usize {
    local_parquet_sources_for_stable_id(plan, stable_id).len()
}

fn local_parquet_sources_for_stable_id<'a>(
    plan: &'a nexus_query_engine::QueryPlan,
    stable_id: &str,
) -> Vec<&'a LocalParquetSource> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) if source.files.iter().any(|file| file.stable_id == stable_id) => {
                    Some(source)
                },
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn assert_source_input(plan: &nexus_query_engine::QueryPlan, node_id: nexus_query_engine::NodeId, name: &str) {
    let node = plan.node(node_id).expect("node should exist");
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(source) => assert_eq!(source.input_name, name),
            other => panic!("expected GPU-resident source {name}, got {other:?}"),
        },
        other => panic!("expected source {name}, got {other:?}"),
    }
}

fn source_input_name(plan: &nexus_query_engine::QueryPlan, node_id: nexus_query_engine::NodeId) -> Option<String> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else { return None };
    match source.as_ref() {
        SourceNode::GpuResident(source) => Some(source.input_name.clone()),
        _ => None,
    }
}

fn assert_left_semi_input(plan: &nexus_query_engine::QueryPlan, node_id: nexus_query_engine::NodeId, key_name: &str) {
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

fn assert_semijoin_rejection(plan: nexus_query_engine::QueryPlan, reason: SemijoinReductionRejectedReason) {
    let result = optimize_query_plan(plan, &semijoin_reduction_only_options(SemijoinReductionOptions::default()));
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].rejected_reason, Some(reason));
}

fn outer_join_null_rejection_only_options() -> NativeOptimizerOptions {
    NativeOptimizerOptions {
        outer_join_null_rejection: nexus_query_engine::OuterJoinNullRejectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    }
}

fn semijoin_reduction_only_options(semijoin_reduction: SemijoinReductionOptions) -> NativeOptimizerOptions {
    NativeOptimizerOptions { semijoin_reduction, ..NativeOptimizerOptions::disabled() }
}
