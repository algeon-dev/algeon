#[test]
fn native_optimizer_stable_rule_rows_cover_every_diagnostic_field() {
    let diagnostics = NativeOptimizerDiagnostics {
        aggregate_join_pushdown: AggregateJoinPushdownDiagnostics { considered_count: 1, ..Default::default() },
        count_non_null_column_pruning: CountNonNullColumnPruningDiagnostics {
            considered_count: 1,
            ..Default::default()
        },
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
        shared_resident_extraction: SharedResidentExtractionDiagnostics {
            considered_count: 1,
            ..Default::default()
        },
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryDiagnostics {
            considered_count: 1,
            ..Default::default()
        },
    };

    let rows = diagnostics.stable_rule_rows();
    assert_eq!(rows.len(), 13);
    let rendered = rows.join(";");
    for rule_name in [
        PROJECTION_SIMPLIFICATION_RULE_NAME,
        SEMIJOIN_REDUCTION_RULE_NAME,
        JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
        COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME,
        OUTER_JOIN_NULL_REJECTION_RULE_NAME,
        JOIN_FILTER_REASSOCIATION_RULE_NAME,
        DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
        STREAM_FANOUT_REPLAY_RULE_NAME,
        LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
        SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
        SHARED_RESIDENT_EXTRACTION_RULE_NAME,
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
fn native_optimizer_diagnostics_preserve_shared_resident_stage_evidence() {
    let mut diagnostics = NativeOptimizerDiagnostics::default();
    diagnostics.merge(NativeOptimizerDiagnostics {
        shared_resident_extraction: SharedResidentExtractionDiagnostics {
            considered_count: 2,
            extracted_count: 1,
            ..Default::default()
        },
        ..Default::default()
    });

    assert!(!diagnostics.is_empty());
    assert_eq!(diagnostics.shared_resident_extraction.considered_count, 2);
    assert_eq!(diagnostics.shared_resident_extraction.extracted_count, 1);
}

#[test]
fn native_optimizer_stable_rule_rows_project_count_non_null_pruning_candidates() {
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(estimated_local_source_with_fields("orders", &["o_custkey", "o_orderkey"], 1_024, 10));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("o_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        Arc::new(Schema::new(vec![
            Field::new("o_custkey", DataType::Int32, false),
            Field::new("c_count", DataType::Int64, false),
        ])),
    );

    let diagnostics = NativeOptimizerDiagnostics {
        count_non_null_column_pruning: CountNonNullColumnPruningDiagnostics {
            considered_count: 1,
            applied_count: 0,
            candidates: vec![CountNonNullColumnPruningCandidateDiagnostics {
                rule_name: COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME,
                aggregate_node_id: aggregate,
                column: "o_orderkey".to_string(),
                source_node_id: Some(source),
                file_column_index: Some(1),
                applied: false,
                rejected_reason: Some(CountNonNullColumnPruningRejectedReason::NullCountProofUnavailable),
            }],
        },
        ..Default::default()
    };

    assert_eq!(
        diagnostics.stable_rule_rows(),
        vec![
            "rule=count_non_null_column_pruning,considered_count=1,applied_count=0,candidates=rule=count_non_null_column_pruning,aggregate_node_id=n1,column=o_orderkey,source_node_id=n0,file_column_index=1,applied=false,rejected_reason=NullCountProofUnavailable".to_string()
        ]
    );
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
                rejected_reason: Some(AggregateJoinPushdownRejectedReason::UnsupportedJoinType),
            }],
        },
        ..Default::default()
    };

    assert_eq!(
        diagnostics.stable_rule_rows(),
        vec![
            "rule=aggregate_join_pushdown,considered_count=1,applied_count=0,candidates=rule=aggregate_join_pushdown,aggregate_node_id=n2,join_node_id=n2,applied=false,rejected_reason=UnsupportedJoinType".to_string()
        ]
    );
}

#[test]
fn native_optimizer_stable_rule_rows_project_local_source_reuse_options() {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", 1_024, 10));
    let right = builder.add_source(estimated_local_source("right", "right_key", 1_024, 10));

    let diagnostics = NativeOptimizerDiagnostics {
        local_parquet_source_reuse: LocalParquetSourceReuseDiagnostics {
            considered_count: 1,
            applied_count: 0,
            candidates: vec![LocalParquetSourceReuseCandidateDiagnostics {
                rule_name: LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
                source_node_ids: vec![left, right],
                reuse_count: 2,
                stable_file_ids: vec!["orders".to_string()],
                superset_read_columns: vec!["left_key".to_string(), "right_key".to_string()],
                branch_filter_count: 1,
                max_variant_selected_bytes: 500 * 1024 * 1024,
                applied: false,
                rejected_reason: Some(LocalParquetSourceReuseRejectedReason::VariantTooLarge),
            }],
        },
        ..Default::default()
    };

    assert_eq!(
        diagnostics.stable_rule_rows(),
        vec![
            "rule=local_parquet_source_reuse,considered_count=1,applied_count=0,candidates=rule=local_parquet_source_reuse,source_node_ids=n0+n1,reuse_count=2,stable_file_ids=orders,superset_read_columns=left_key+right_key,branch_filter_count=1,max_variant_selected_bytes=524288000,applied=false,rejected_reason=VariantTooLarge".to_string()
        ]
    );
}
