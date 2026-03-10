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
    assert!(!options.count_non_null_column_pruning.enabled);
    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.is_empty());
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
fn count_non_null_column_pruning_options_and_rule_name_are_stable() {
    assert_eq!(COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME, "count_non_null_column_pruning");
    assert!(CountNonNullColumnPruningOptions::default().enabled);
    assert!(!CountNonNullColumnPruningOptions::disabled().enabled);
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
