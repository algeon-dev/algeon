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
fn native_optimizer_captures_keyed_hash_join_strategy_options_in_plan() {
    let keyed_hash_join_strategy =
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11);
    let options = NativeOptimizerOptions { keyed_hash_join_strategy, ..NativeOptimizerOptions::disabled() };

    let result = optimize_query_plan(single_source_plan(), &options);

    assert_eq!(result.plan.keyed_hash_join_strategy_options(), keyed_hash_join_strategy);
}

#[test]
fn native_optimizer_preserves_keyed_hash_join_strategy_options_through_projection_rewrite() {
    let plan_schema = schema(&["key"]);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&plan_schema))));
    let projection = builder.add_projection(source, projection_outputs(vec![Expr::column("key")], Arc::clone(&plan_schema)));
    let plan = builder.finish_gpu_resident(projection);
    let keyed_hash_join_strategy =
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11);
    let options = NativeOptimizerOptions {
        projection_simplification: ProjectionSimplificationOptions::default(),
        keyed_hash_join_strategy,
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.projection_simplification.applied_count, 1);
    assert_eq!(result.plan.keyed_hash_join_strategy_options(), keyed_hash_join_strategy);
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
        SemijoinReductionOptions::DEFAULT_COST_THRESHOLD_MULTIPLIER
    );
    assert_eq!(
        DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER,
        SemijoinReductionOptions::DEFAULT_COST_THRESHOLD_MULTIPLIER
    );
    assert_eq!(SemijoinReductionOptions::default().admission_mode, SemijoinReductionAdmissionMode::NotEvaluated);
    assert!(!SemijoinReductionOptions::disabled().enabled);
    SemijoinReductionOptions::default().validate().unwrap();
}

#[test]
fn join_build_side_selection_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_BUILD_SIDE_SELECTION_RULE_NAME, "join_build_side_selection");
    assert!(JoinBuildSideSelectionOptions::default().enabled);
    assert!(!JoinBuildSideSelectionOptions::disabled().enabled);
    assert_eq!(
        JoinBuildSideSelectionOptions::default().small_build_side_bytes,
        JoinBuildSideSelectionOptions::DEFAULT_SMALL_BUILD_SIDE_BYTES
    );
    assert_eq!(
        JoinBuildSideSelectionOptions::default().small_build_side_rows,
        JoinBuildSideSelectionOptions::DEFAULT_SMALL_BUILD_SIDE_ROWS
    );
    JoinBuildSideSelectionOptions::default().validate().unwrap();
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
    assert!(nexus_query_engine::optimizer::OuterJoinNullRejectionOptions::default().enabled);
    assert!(!nexus_query_engine::optimizer::OuterJoinNullRejectionOptions::disabled().enabled);
}

#[test]
fn join_filter_reassociation_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_FILTER_REASSOCIATION_RULE_NAME, "join_filter_reassociation");
    assert!(nexus_query_engine::optimizer::JoinFilterReassociationOptions::default().enabled);
    assert!(nexus_query_engine::optimizer::JoinFilterReassociationOptions::default().residual_filter_pushdown_enabled);
    assert!(!nexus_query_engine::optimizer::JoinFilterReassociationOptions::disabled().enabled);
    assert!(!nexus_query_engine::optimizer::JoinFilterReassociationOptions::disabled().residual_filter_pushdown_enabled);
}

#[test]
fn dimension_derived_row_group_pruning_options_and_rule_name_are_stable() {
    assert_eq!(DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME, "dimension_derived_row_group_pruning");
    assert!(DimensionDerivedRowGroupPruningOptions::default().enabled);
    assert_eq!(
        DimensionDerivedRowGroupPruningOptions::default().max_dimension_rows,
        DimensionDerivedRowGroupPruningOptions::DEFAULT_MAX_DIMENSION_ROWS
    );
    assert!(!DimensionDerivedRowGroupPruningOptions::disabled().enabled);
    assert_eq!(nexus_query_engine::optimizer::DIMENSION_KEY_FILTER_MAX_RUNS, nexus_query_engine::optimizer::DimensionKeyFilter::MAX_RUNS);
    DimensionDerivedRowGroupPruningOptions::default().validate().unwrap();
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
    assert_eq!(
        LocalParquetSourceReuseOptions::default().max_variant_selected_bytes,
        LocalParquetSourceReuseOptions::DEFAULT_MAX_VARIANT_SELECTED_BYTES
    );
    LocalParquetSourceReuseOptions::default().validate().unwrap();
}

#[test]
fn native_optimizer_options_validate_expert_optimizer_thresholds() {
    NativeOptimizerOptions::default().validate().unwrap();
    NativeOptimizerOptions::disabled().validate().unwrap();

    let error = NativeOptimizerOptions {
        semijoin_reduction: SemijoinReductionOptions { cost_threshold_multiplier: 0, ..Default::default() },
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("cost_threshold_multiplier"));
    assert_eq!(error.fact_value("reason"), Some("zero"));

    let error = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions {
            small_build_side_bytes: 0,
            ..Default::default()
        },
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("small_build_side_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));

    let error = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions {
            small_build_side_rows: 0,
            ..Default::default()
        },
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("small_build_side_rows"));
    assert_eq!(error.fact_value("reason"), Some("zero"));

    let error = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions {
            max_dimension_rows: 0,
            ..Default::default()
        },
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("max_dimension_rows"));
    assert_eq!(error.fact_value("reason"), Some("zero"));

    let error = NativeOptimizerOptions {
        local_parquet_source_reuse: LocalParquetSourceReuseOptions {
            max_variant_selected_bytes: 0,
            ..Default::default()
        },
        ..Default::default()
    }
    .validate()
    .unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("field"), Some("max_variant_selected_bytes"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
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
    assert_eq!(DynamicFilterRouteDiscoveryOptions::default().execution, DynamicFilterExecutionOptions::default());
    assert!(!DynamicFilterRouteDiscoveryOptions::disabled().enabled);
    assert_eq!(DynamicFilterRouteDiscoveryOptions::disabled().execution, DynamicFilterExecutionOptions::default());
}

#[test]
fn dynamic_filter_execution_options_defaults_are_stable() {
    let options = DynamicFilterExecutionOptions::default();

    assert_eq!(options.small_in_list_limit(), 1024);
    assert_eq!(options.max_build_rows(), 1_000_000);
    assert_eq!(options.large_local_parquet_range_producer_rows(), 4 * 1024);
    assert_eq!(options.dimension_sized_consumer_bytes(), 256 * 1024 * 1024);
    assert_eq!(options.tiny_local_parquet_consumer_row_groups(), 32);
    assert_eq!(options.max_range_reader_predicate_selectivity_ppm(), 950_000);
    assert_eq!(options.max_deferred_small_in_list_reader_values(), 128);
    assert_eq!(options.max_unpruned_hash_set_post_decode_values(), 4096);
    assert_eq!(options.min_unpruned_hash_set_post_decode_source_rows(), 1_000_000);
}

#[test]
fn keyed_hash_join_strategy_options_defaults_are_stable() {
    let options = KeyedHashJoinStrategyOptions::default();

    assert_eq!(options.large_keyed_hash_build_row_threshold(), 64_000_000);
    assert_eq!(options.large_keyed_hash_build_bytes_threshold(), 8 * 1024 * 1024 * 1024);
    assert_eq!(options.large_keyed_hash_probe_bytes_threshold(), 2 * 1024 * 1024 * 1024);
    assert_eq!(options.small_reusable_hash_build_row_threshold(), 1_000_000);
    assert_eq!(options.small_reusable_hash_build_bytes_threshold(), 64 * 1024 * 1024);
    assert_eq!(options.small_one_shot_probe_bytes_threshold(), 256 * 1024 * 1024);
    assert_eq!(options.max_small_probe_left_one_shot_build_row_threshold(), 128_000_000);
    assert_eq!(
        options.max_small_probe_left_one_shot_build_bytes_threshold(),
        2 * 1024 * 1024 * 1024
    );
}
