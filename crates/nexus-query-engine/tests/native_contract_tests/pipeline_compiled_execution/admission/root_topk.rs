#[test]
fn compiled_bounded_support_accepts_root_topk_shared_resident_fanout_graph() {
    for variant in [
        RootTopKJoinAggregateVariant::SharedComputedFanout,
        RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout,
        RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout,
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
            .unwrap_or_else(|error| panic!("{variant:?} root TopK RTK-3 plan should compile: {error}"));

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
fn compiled_bounded_support_accepts_root_topk_rtk9_subcontract_graphs() {
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
fn compiled_bounded_support_accepts_root_topk_rtk10_null_filtered_aggregate_fan_in_graph() {
    let optimized = optimize_query_plan(
        root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn),
        &NativeOptimizerOptions::default(),
    );
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 3);
    let plan = optimized.plan;
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline =
        compile_pipeline(&plan, &analysis).expect("RTK-10 anti-rewritten aggregate fan-in plan should compile");

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
        .expect("RTK-10 compiled report should be valid");
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::CapabilityContract,
        "root_topk_keyed_filtered_semi_join_subcontract",
    );
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::OperatorOutput,
        "aggregate_contract",
    );
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
        &report,
        ReservationKind::FinalSort,
        "final_sort_contract",
    );
    assert_modeled_gate_row(
        "rtk10 null-filtered aggregate fan-in",
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
