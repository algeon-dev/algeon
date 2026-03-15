#[test]
fn compiled_bounded_support_accepts_root_join_graph_without_legacy_routing_terms() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("root_join"));

    let source_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("root join should retain source-read gates");
    assert_eq!(source_gate.status(), ContractStatus::Modeled);
    assert_eq!(source_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
}

#[test]
fn compiled_bounded_support_rejects_non_d1_task_envelope_before_execution() {
    let plan = aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate sort limit plan should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_accepts_single_join_post_root_d2_tasks_without_legacy_routing_terms() {
    let plan = single_join_post_root_aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert!(pipeline.barriers().iter().any(|barrier| barrier.kind() == PipelineBarrierKind::AggregateBoundary));
    assert!(pipeline.barriers().iter().any(|barrier| barrier.kind() == PipelineBarrierKind::SortBoundary));
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Aggregate)
        .expect("single-join post-root plan should have an aggregate task");
    let aggregate_gate = aggregate_task
        .gates()
        .iter()
        .find_map(|gate_id| pipeline.gates().iter().find(|gate| gate.id() == *gate_id))
        .expect("compiled aggregate task should retain its output reservation gate");
    assert_eq!(aggregate_gate.reservation_kind(), ReservationKind::OperatorOutput);
    assert_eq!(aggregate_gate.status(), ContractStatus::Modeled);
    assert_eq!(aggregate_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let final_sort_gate = pipeline
        .gates()
        .iter()
        .find(|gate| gate.reservation_kind() == ReservationKind::FinalSort)
        .expect("single-join post-root plan should retain final-sort gates");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_accepts_simple_multi_join_tree_without_legacy_routing_terms() {
    let plan = simple_multi_join_plan(
        Path::new("/tmp/compiled-d26-a.parquet"),
        Path::new("/tmp/compiled-d26-b.parquet"),
        Path::new("/tmp/compiled-d26-c.parquet"),
    );
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_closed_multi_join_subgraph());
    let pipeline = compile_pipeline(&plan, &analysis).expect("simple multi-join plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::SourceRead, 3);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 2);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_rejects_post_root_aggregate_without_output_gate_before_execution() {
    let plan = single_join_post_root_aggregate_sort_limit_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");
    let malformed = pipeline_without_reservation_gates(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_rejects_post_root_final_sort_without_gate_before_execution() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d24-final-sort-left.parquet"),
        Path::new("/tmp/compiled-d24-final-sort-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join final-sort plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let malformed = pipeline_without_reservation_gates(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_accepts_grouped_aggregate_join_build_input_without_legacy_routing_terms() {
    let plan = aggregate_join_input_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinBuild, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinProbe, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::JoinMaterialize, 1);

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let aggregate_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Aggregate)
        .expect("aggregate join input plan should have an aggregate task");
    let join_build = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::JoinBuild)
        .expect("aggregate join input plan should have a join build task");
    assert!(pipeline.edges().iter().any(|edge| {
        edge.from() == aggregate_task.id()
            && edge.to() == join_build.id()
            && edge.kind() == PipelineEdgeKind::JoinBuildInput
    }));
    let aggregate_gate =
        runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::Aggregate, ReservationKind::OperatorOutput)
            .expect("compiled aggregate input task should retain its modeled output gate");
    assert_eq!(aggregate_gate.status(), ContractStatus::Modeled);
    assert_eq!(aggregate_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
    let aggregate_barrier = pipeline
        .barriers()
        .iter()
        .find(|barrier| {
            barrier.kind() == PipelineBarrierKind::AggregateBoundary
                && barrier.upstream_tasks() == [aggregate_task.id()]
        })
        .expect("aggregate input task should have an explicit aggregate boundary");
    assert!(aggregate_barrier.downstream_tasks().contains(&join_build.id()));

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_accepts_root_topk_join_aggregate_graph() {
    let plan = root_topk_join_aggregate_plan();
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_topk_join_aggregate_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("root TopK join/aggregate plan should compile");

    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::SinkMaterialize, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let final_sort_gate =
        runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::FinalSort, ReservationKind::FinalSort)
            .expect("root TopK plan should retain a final-sort gate");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);

    let rendered = pipeline.debug_render();
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
}

#[test]
fn compiled_bounded_support_rejects_unmodeled_reservation_gates_before_execution() {
    let pipeline = root_join_shaped_pipeline_with_blocked_gate();

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_query_plan_support_accepts_keyless_root_join_strategy_without_legacy_routing_terms() {
    let plan = keyless_root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph shape should still match the structural envelope"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_no_legacy_execution_selection_terms("keyless root join strategy", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_query_plan_support_accepts_keyless_outer_cross_root_join_strategy() {
    let plan = keyless_outer_cross_root_join_plan();
    let analysis = plan.capability_analysis();
    assert!(
        analysis.admits_root_join(),
        "keyless outer-cross root join should be admitted; first_rejection={:?}",
        analysis.first_rejection_projection()
    );
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless outer-cross root join should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph shape should still match the structural envelope"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::GatherMap
                && gate.status() == ContractStatus::Modeled
                && gate.quality() == ContractQuality::ExactPreflight
        }),
        "keyless outer-cross support must retain a modeled gather-map gate for empty-side null extension"
    );
    assert!(
        pipeline.gates().iter().any(|gate| {
            gate.reservation_kind() == ReservationKind::JoinOutput
                && gate.status() == ContractStatus::Modeled
                && gate.quality() == ContractQuality::ExactPreflight
        }),
        "keyless outer-cross support must retain a modeled join-output gate"
    );
    assert_no_legacy_execution_selection_terms("keyless outer-cross root join strategy", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_support_records_d3_inventory_migration_boundaries_before_scheduler_execution() {
    for case in d3_1_inventory_cases() {
        let analysis = case.plan.capability_analysis();
        assert!(
            !analysis.has_rejections(),
            "{}: compiler should still receive locally satisfied capability facts",
            case.label
        );
        assert!((case.admits)(&analysis), "{}: expected D.3 target admission to be satisfied", case.label);
        let pipeline = compile_pipeline(&case.plan, &analysis)
            .unwrap_or_else(|error| panic!("{}: compile failed: {error}", case.label));
        pipeline
            .validate()
            .unwrap_or_else(|error| panic!("{}: compiled pipeline should validate: {error}", case.label));

        assert_eq!(
            compiled_bounded_pipeline_support(&pipeline).expect("graph-only support check should not fail"),
            case.graph_support,
            "{}: graph-only support status should match the current D.3 migration boundary",
            case.label
        );
        let plan_support = compiled_bounded_query_plan_support(&case.plan, &pipeline)
            .expect("plan-aware support check should not fail");
        assert_eq!(plan_support, case.plan_support, "{}: plan-aware compiled support status changed", case.label);

        assert_compiled_graph_inventory(case.label, &pipeline);
        let rendered = pipeline.debug_render();
        for expected in case.render_fragments {
            assert!(rendered.contains(expected), "{}: compiled graph rendering should contain {expected}", case.label);
        }
        assert_no_legacy_execution_selection_terms(case.label, &rendered);

        if plan_support == CompiledBoundedPipelineSupport::Unsupported {
            let executor = MockExecutor::new();
            assert!(
                executor.executed.is_empty(),
                "{}: unsupported compiled graph should stop at support checking and execute zero compiled tasks",
                case.label
            );
        }
    }
}

#[test]
fn compiled_bounded_support_accepts_keyed_hash_resident_subplan_shared_producer_without_role_routing_terms() {
    let plan =
        keyed_hash_join_resident_subplan_pipeline_plan(KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_keyed_hash_join_resident_subplan_pipeline());
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyed hash resident subplan should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
        .collect::<Vec<_>>();
    assert_eq!(shared_edges.len(), 2, "shared producer should fan out to both downstream branches");
    let shared_resource = shared_edges[0].resources()[0];
    assert!(shared_edges.iter().all(|edge| edge.resources() == [shared_resource]));
    let shared_producer = pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == shared_edges[0].from())
        .expect("shared producer task should be present");
    assert_eq!(shared_producer.kind(), PipelineTaskKind::Aggregate);
    assert_no_legacy_execution_selection_terms("keyed hash resident subplan", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_support_accepts_shared_grouped_aggregate_fanout_without_scalar_derivation() {
    let plan = shared_grouped_aggregate_fanout_to_join_and_grouped_aggregate_plan();
    let analysis = plan.capability_analysis();
    assert!(!analysis.has_rejections(), "expected satisfied local capability facts");
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    assert!(!analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared grouped aggregate fanout should compile");

    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer)
        .collect::<Vec<_>>();
    assert!(!shared_edges.is_empty(), "shared producer should feed a downstream compiled branch");
    let shared_producer = pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == shared_edges[0].from())
        .expect("shared producer task should be present");
    assert!(matches!(shared_producer.kind(), PipelineTaskKind::Unary | PipelineTaskKind::Aggregate));
    assert_no_legacy_execution_selection_terms("shared grouped aggregate fanout", &pipeline.debug_render());
}

#[test]
fn compiled_bounded_support_accepts_d3_6_nonlinear_topk_and_subquery_filter_paths() {
    for (label, plan, admits, expected_aggregate_barriers) in [
        (
            "aggregate-over-join-subplan TopK",
            aggregate_over_join_subplan_topk_plan(),
            nexus_query_engine::capability::CapabilityAnalysis::admits_multi_join_aggregate_over_join_subplan_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            1,
        ),
        (
            "resident grouped-aggregate TopK",
            resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            nexus_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate TopK direct aggregate input",
            resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::DirectPostJoinAggregateInput,
            ),
            nexus_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate TopK filter-wrapped direct aggregate input",
            resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::FilterWrappedGroupedInputDirectPostJoinAggregateInput,
            ),
            nexus_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate TopK preaggregated filter-wrapped direct aggregate input",
            resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput,
            ),
            nexus_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            3,
        ),
        (
            "resident grouped-aggregate subquery filter",
            resident_grouped_aggregate_subquery_filter_plan(),
            nexus_query_engine::capability::CapabilityAnalysis::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline
                as fn(&nexus_query_engine::capability::CapabilityAnalysis) -> bool,
            1,
        ),
    ] {
        let analysis = plan.capability_analysis();
        assert!(!analysis.has_rejections(), "{label}: expected satisfied local capability facts");
        assert!(admits(&analysis), "{label}: expected bounded admission to be satisfied");
        let pipeline = compile_pipeline(&plan, &analysis).unwrap_or_else(|error| panic!("{label}: compile failed: {error}"));

        assert_eq!(
            compiled_bounded_pipeline_support(&pipeline).expect("graph support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: graph-only compiled support should admit the D.3.6 path"
        );
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: plan-aware compiled support should admit the D.3.6 path"
        );
        assert_eq!(sink_input_edge_count(&pipeline), 1, "{label}: expected one explicit host sink path");
        assert!(
            task_kind_count(&pipeline, PipelineTaskKind::JoinBuild) > 1
                || task_kind_count(&pipeline, PipelineTaskKind::Aggregate) > 1,
            "{label}: expected a non-linear join or aggregate DAG"
        );
        assert!(
            barrier_kind_count(&pipeline, PipelineBarrierKind::AggregateBoundary) >= expected_aggregate_barriers,
            "{label}: expected aggregate barriers to stay explicit"
        );
        let final_sort_gate =
            runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::FinalSort, ReservationKind::FinalSort)
                .expect("D.3.6 path should retain its final-sort reservation gate");
        assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
        assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
        assert_no_legacy_execution_selection_terms(label, &pipeline.debug_render());
    }
}
