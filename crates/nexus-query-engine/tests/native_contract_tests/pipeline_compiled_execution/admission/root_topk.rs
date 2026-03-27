#[test]
fn compiled_bounded_support_accepts_root_topk_shared_resident_fanout_graph() {
    for variant in [
        RootTopKJoinAggregateVariant::SharedComputedFanout,
        RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout,
        RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout,
        RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        assert!(analysis.invariants.iter().any(|invariant| {
            matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
                && invariant.decision == CapabilityDecision::Satisfied
        }));
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK shared-fanout plan should compile: {error}"));

        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        let shared_edges = pipeline
            .edges()
            .iter()
            .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
            .collect::<Vec<_>>();
        assert!(!shared_edges.is_empty(), "{variant:?}: shared producer must remain explicit in the compiled graph");
        let shared_producer = shared_edges[0].from();
        assert_eq!(shared_edges[0].resources().len(), 1, "{variant:?}: shared edge should carry one resource");
        let shared_resource = shared_edges[0].resources()[0];
        assert!(
            shared_edges.iter().all(|edge| edge.from() == shared_producer && edge.resources() == [shared_resource]),
            "{variant:?}: shared edges must preserve one producer/resource identity"
        );
        assert_eq!(
            compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );

        let rendered = pipeline.debug_render();
        assert_no_legacy_execution_selection_terms("root TopK shared resident fanout", &rendered);
    }
}

#[test]
fn compiled_bounded_support_accepts_root_topk_distinct_and_keyless_subcontract_graphs() {
    for (variant, expected_contract) in [
        (RootTopKJoinAggregateVariant::DistinctAggregate, "root_topk_distinct_aggregate_subcontract"),
        (
            RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin,
            "root_topk_distinct_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin,
            "root_topk_distinct_aggregate_subcontract",
        ),
        (RootTopKJoinAggregateVariant::KeylessJoin, "root_topk_keyless_or_cross_join_subcontract"),
        (RootTopKJoinAggregateVariant::ProjectedCrossJoin, "root_topk_keyless_or_cross_join_subcontract"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} distinct/keyless root TopK plan should compile: {error}"));

        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::CapabilityContract,
            expected_contract,
        );
        if matches!(variant, RootTopKJoinAggregateVariant::KeylessJoin) {
            assert_modeled_gate_row(
                "keyless root TopK",
                &report,
                ReservationKind::MixedSizeData,
                "mixed_keyed_residual_or_conditional_full_retained_size_data",
            );
            assert!(
                pipeline.gates().iter().any(|gate| {
                    gate.reservation_kind() == ReservationKind::GatherMap && gate.status() == ContractStatus::Modeled
                }),
                "keyless root TopK should retain a modeled gather-map gate"
            );
        }
        if matches!(variant, RootTopKJoinAggregateVariant::ProjectedCrossJoin) {
            assert!(
                pipeline.gates().iter().any(|gate| {
                    gate.reservation_kind() == ReservationKind::JoinOutput
                        && gate.required_contract() == "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                        && gate.status() == ContractStatus::Modeled
                        && gate.quality() == ContractQuality::ExactPreflight
                }),
                "projected-cross root TopK should retain an exact join-output bound"
            );
        }
    }
}

#[test]
fn compiled_bounded_support_accepts_root_topk_sort_fetch_hash_join_input_subcontract_graph() {
    let plan = root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchHashJoinInput);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline =
        compile_pipeline(&plan, &analysis).expect("sort-fetch hash-join-input root TopK plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(sink_input_edge_count(&pipeline), 1, "expected one explicit host sink path");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported,
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .expect("sort-fetch hash-join-input compiled report should be valid");
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::CapabilityContract,
        "root_topk_sort_fetch_hash_join_input_subcontract",
    );
    assert_modeled_gate_row("sort-fetch hash-join-input", &report, ReservationKind::SourceRead, "source_contract");
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_final_topk_output_boundary",
    );
    assert_modeled_gate_row("sort-fetch hash-join-input", &report, ReservationKind::FinalSort, "final_sort_contract");
    assert_modeled_gate_row(
        "sort-fetch hash-join-input",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn compiled_bounded_support_accepts_root_topk_sort_fetch_join_aggregate_subcontract_graphs() {
    for (variant, expected_contract) in [
        (
            RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (
            RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
            "root_topk_sort_fetch_hash_join_aggregate_subcontract",
        ),
        (RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput, "root_topk_keyed_filtered_semi_join_subcontract"),
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK plan should compile: {error}"));

        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}",
        );

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::CapabilityContract,
            expected_contract,
        );
        assert!(
            report.gate_rows().iter().any(|row| {
                row.reservation_kind == ReservationKind::SourceRead
                    && (row.required_contract == "source_contract"
                        || row.required_contract == "parquet_source_read_exact_preflight")
                    && row.contract_status == ContractStatus::Modeled
            }),
            "{variant:?}: expected modeled source-read gate row; rows={:?}",
            report.gate_rows()
        );
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::OperatorOutput,
            "aggregate_contract",
        );
        if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::HashTable,
                "hash_join_pre_build_reservation_size_bytes",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::FilteredProbeState,
                "filtered_join_retained_probe_state",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::GatherMap,
                "join_gather_map_reservation_size_bytes",
            );
        }
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::OperatorOutput,
            "query_engine_final_topk_output_boundary",
        );
        assert_modeled_gate_row(&format!("{variant:?}"), &report, ReservationKind::FinalSort, "final_sort_contract");
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::Materialization,
            "query_engine_explicit_sink_materialization_reservation",
        );
    }
}

#[test]
fn compiled_bounded_support_accepts_rollup_topk_without_rank_after_aggregate_join_pushdown() {
    let plan = selected_q22_class_rollup_topk_after_pushdown_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "q22-class rollup TopK should be source-reduced by aggregate-join pushdown");
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized q22-class plan should retain the final grouping-set aggregate"
    );
    assert!(
        !optimized.plan.nodes().iter().any(|node| matches!(&node.kind, PlanNode::Window(_))),
        "q22-class rollup TopK has no rank/window subcontract"
    );

    let analysis = optimized.plan.capability_analysis();
    assert!(
        analysis.admits_root_topk_join_aggregate_pipeline(),
        "q22-class rollup TopK should use the root TopK envelope; root_decision={:?}; first_rejection={:?}",
        analysis.root_topk_join_aggregate_admission,
        analysis.first_rejection_projection()
    );
    assert!(
        !analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "rankless rollup TopK must not be mislabeled as the rollup/rank envelope"
    );
    let pipeline = compile_pipeline(&optimized.plan, &analysis).expect("q22-class rollup TopK plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&optimized.plan, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported,
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .expect("q22-class rollup TopK compiled report should be valid");
    assert_modeled_gate_row("q22-class rollup TopK", &report, ReservationKind::OperatorOutput, "aggregate_contract");
    assert_modeled_gate_row("q22-class rollup TopK", &report, ReservationKind::FinalSort, "final_sort_contract");
}

#[test]
fn compiled_bounded_support_rejects_rankless_non_rollup_grouping_sets_root_topk() {
    let plan = rankless_non_rollup_grouping_sets_root_topk_plan();
    let analysis = plan.capability_analysis();

    let CapabilityDecision::Rejected { reason, facts } = &analysis.root_topk_join_aggregate_admission else {
        panic!(
            "rankless non-rollup grouping sets must not enter the generic root TopK envelope: {:?}",
            analysis.root_topk_join_aggregate_admission
        );
    };
    assert_eq!(*reason, "root_topk_join_aggregate_requires_rankless_rollup_grouping_sets_contract");
    assert_eq!(facts.get("required_grouping_mask_contract").map(String::as_str), Some("suffix_rollup"));
}

#[test]
fn compiled_bounded_support_accepts_root_topk_null_filtered_aggregate_fan_in_graph() {
    let optimized = optimize_query_plan(
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn),
        &NativeOptimizerOptions::default(),
    );
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 3);
    let plan = optimized.plan;
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis)
        .expect("anti-rewritten null-filtered aggregate fan-in plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 8);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 6);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(sink_input_edge_count(&pipeline), 1, "expected one explicit host sink path");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported,
    );

    let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
        .expect("null-filtered aggregate fan-in compiled report should be valid");
    assert_modeled_gate_row(
        "null-filtered aggregate fan-in",
        &report,
        ReservationKind::CapabilityContract,
        "root_topk_keyed_filtered_semi_join_subcontract",
    );
    assert_modeled_gate_row(
        "null-filtered aggregate fan-in",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "null-filtered aggregate fan-in",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
    assert_modeled_gate_row(
        "null-filtered aggregate fan-in",
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );
}

#[test]
fn compiled_bounded_support_accepts_root_topk_window_subcontract_graphs() {
    for variant in [
        RootTopKJoinAggregateVariant::Window,
        RootTopKJoinAggregateVariant::BranchLocalWindow,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindow,
        RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin,
        RootTopKJoinAggregateVariant::SortedAggregateWindow,
        RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection,
    ] {
        let plan = root_topk_join_aggregate_variant_plan(variant);
        let analysis = plan.capability_analysis();
        assert!(analysis.admits_root_topk_join_aggregate_pipeline(), "{variant:?}");
        let pipeline = compile_pipeline(&plan, &analysis)
            .unwrap_or_else(|error| panic!("{variant:?} root TopK window plan should compile: {error}"));

        let expected_window_count = match variant {
            RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin => 2,
            RootTopKJoinAggregateVariant::Window
            | RootTopKJoinAggregateVariant::BranchLocalWindow
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::SortedAggregateWindow
            | RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection => 1,
            _ => unreachable!("window subcontract variants are enumerated above"),
        };
        let expected_sort_count = match variant {
            RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin => 3,
            RootTopKJoinAggregateVariant::Window
            | RootTopKJoinAggregateVariant::BranchLocalWindow
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::SortedAggregateWindow
            | RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection => 2,
            _ => unreachable!("window subcontract variants are enumerated above"),
        };
        assert_task_kind_count(&pipeline, PipelineTaskKind::Window, expected_window_count);
        assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, expected_sort_count);
        assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
        assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{variant:?}: expected one explicit host sink path");
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{variant:?}"
        );

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{variant:?} compiled report should be valid: {error}"));
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::CapabilityContract,
            "root_topk_window_subcontract",
        );
        if matches!(variant, RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::CapabilityContract,
                "root_topk_window_full_outer_join_subcontract",
            );
        }
        if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindow) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::CapabilityContract,
                "root_topk_keyless_outer_cross_window_subcontract",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::GatherMap,
                "join_gather_map_reservation_size_bytes",
            );
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::JoinOutput,
                "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
            );
        }
        assert_modeled_gate_row(&format!("{variant:?}"), &report, ReservationKind::OperatorOutput, "window_contract");
        if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
            assert_modeled_gate_row(
                &format!("{variant:?}"),
                &report,
                ReservationKind::OperatorOutput,
                "query_engine_rank_filter_output_boundary",
            );
        }
        assert_modeled_gate_row(
            &format!("{variant:?}"),
            &report,
            ReservationKind::OperatorOutput,
            "query_engine_final_topk_output_boundary",
        );
    }
}

fn rankless_non_rollup_grouping_sets_root_topk_plan() -> QueryPlan {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("d1", DataType::Int32, false),
        Field::new("d2", DataType::Int32, false),
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Int32, true),
        Field::new("k3", DataType::Int32, true),
        Field::new("m", DataType::Int64, true),
    ]));
    let dim1_schema = Arc::new(Schema::new(vec![Field::new("d1", DataType::Int32, false)]));
    let dim2_schema = Arc::new(Schema::new(vec![Field::new("d2", DataType::Int32, false)]));
    let join1_schema = Arc::new(Schema::new(vec![
        Field::new("d2", DataType::Int32, false),
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Int32, true),
        Field::new("k3", DataType::Int32, true),
        Field::new("m", DataType::Int64, true),
    ]));
    let join2_schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Int32, true),
        Field::new("k3", DataType::Int32, true),
        Field::new("m", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Int32, true),
        Field::new("k3", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("total", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int32, true),
        Field::new("k2", DataType::Int32, true),
        Field::new("k3", DataType::Int32, true),
        Field::new("total", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rankless-non-rollup-root-topk-fact.parquet",
        fact_schema,
        vec!["d1".into(), "d2".into(), "k1".into(), "k2".into(), "k3".into(), "m".into()],
    )));
    let dim1 = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rankless-non-rollup-root-topk-dim1.parquet",
        dim1_schema,
        vec!["d1".into()],
    )));
    let dim2 = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/rankless-non-rollup-root-topk-dim2.parquet",
        dim2_schema,
        vec!["d2".into()],
    )));
    let join1 = builder.add_join(
        fact,
        dim1,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d1", "d1"), ColumnRef::input(1, "d1", "d1"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "d2", "d2"), "d2"),
            JoinOutputColumn::left(ColumnRef::input(0, "k1", "k1"), "k1"),
            JoinOutputColumn::left(ColumnRef::input(0, "k2", "k2"), "k2"),
            JoinOutputColumn::left(ColumnRef::input(0, "k3", "k3"), "k3"),
            JoinOutputColumn::left(ColumnRef::input(0, "m", "m"), "m"),
        ])
        .with_build_side(JoinSide::Right),
        join1_schema,
    );
    let join2 = builder.add_join(
        join1,
        dim2,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d2", "d2"), ColumnRef::input(1, "d2", "d2"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k1", "k1"), "k1"),
            JoinOutputColumn::left(ColumnRef::input(0, "k2", "k2"), "k2"),
            JoinOutputColumn::left(ColumnRef::input(0, "k3", "k3"), "k3"),
            JoinOutputColumn::left(ColumnRef::input(0, "m", "m"), "m"),
        ])
        .with_build_side(JoinSide::Right),
        join2_schema,
    );
    let aggregate = builder.add_aggregate(
        join2,
        AggregateSpec::new(
            vec![Expr::column("k1"), Expr::column("k2"), Expr::column("k3")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m"), "total")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(
            vec![vec![true, true, true], vec![true, false, true], vec![false, false, false]],
            "__grouping_id",
        )),
        Arc::clone(&aggregate_schema),
    );
    let projection = builder.add_projection(
        aggregate,
        vec![
            ProjectionOutput::new(Expr::column("k1"), output_schema.field(0).as_ref().clone()),
            ProjectionOutput::new(Expr::column("k2"), output_schema.field(1).as_ref().clone()),
            ProjectionOutput::new(Expr::column("k3"), output_schema.field(2).as_ref().clone()),
            ProjectionOutput::new(Expr::column("total"), output_schema.field(3).as_ref().clone()),
        ],
    );
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("total"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(100), output_schema);
    builder.finish_with_host_sink(limit)
}
