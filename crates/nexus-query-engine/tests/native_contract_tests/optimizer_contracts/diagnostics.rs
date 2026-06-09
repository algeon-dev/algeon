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
