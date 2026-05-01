#[test]
fn strict_admission_accepts_rank_window_branches_feeding_native_union() {
    let plan = union_of_ranked_join_branches_plan();
    let analysis = plan.capability_analysis();
    assert!(
        analysis.union_pipeline_admission == CapabilityDecision::Satisfied,
        "union pipeline admission should accept branch-local windows; first_rejection={:?}",
        analysis.first_rejection_projection()
    );
    assert!(analysis.admits_bounded_join());

    let pipeline = compile_pipeline(&plan, &analysis).expect("rank-window union plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Window, 2);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Union, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 3);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("rank-window union plan should be strictly admitted");
    let rendered = admission.pipeline().debug_render();
    assert!(rendered.contains("kind=window"), "compiled graph must keep window tasks:\n{rendered}");
    assert!(rendered.contains("kind=union"), "compiled graph must keep native union:\n{rendered}");
    assert_no_legacy_execution_selection_terms("rank-window union plan", &rendered);
}

#[test]
fn strict_admission_rejects_large_full_scan_rollup_rank_topk_before_execution() {
    let plan = large_full_scan_rollup_rank_topk_plan();
    let rendered_plan = plan.to_string();
    assert!(
        rendered_plan.contains("grouping_set_count=4"),
        "native IR display must expose rollup/grouping-set metadata:\n{rendered_plan}"
    );
    assert!(
        rendered_plan.contains("grouping_id_alias=__grouping_id"),
        "native IR display must expose grouping id output metadata:\n{rendered_plan}"
    );
    assert!(
        rendered_plan.contains("grouping_sets=000|001|011|111"),
        "native IR display must expose grouping-set masks:\n{rendered_plan}"
    );

    let analysis = plan.capability_analysis();
    assert!(analysis.has_rejections(), "large full-scan rollup/rank/top-k should be rejected before runtime");
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("grouping_set_count").map(String::as_str), Some("4"));
    assert_eq!(rejection.facts.get("final_topk_fetch").map(String::as_str), Some("100"));
    assert_ne!(rejection.facts.get("rank_sort_id").map(String::as_str), Some("absent"));
    assert_ne!(rejection.facts.get("rank_filter_id").map(String::as_str), Some("absent"));
    assert_eq!(rejection.facts.get("large_source_selected_bytes").map(String::as_str), Some("2147483648"));
    assert_eq!(rejection.facts.get("large_source_threshold_bytes").map(String::as_str), Some("536870912"));
    assert_eq!(rejection.facts.get("large_source_selected_row_groups").map(String::as_str), Some("unknown"));
    assert_eq!(
        rejection.facts.get("required_remediation").map(String::as_str),
        Some("source_reduction_or_exact_blocking_stage_preflight")
    );

    assert!(
        bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
        "rejected shape must not produce a strict pipeline for execution"
    );

    let report = CapabilityContractReport::from_capability_analysis(&analysis)
        .expect("rejected capability analysis should still produce a contract report");
    let blocking_row = report
        .gate_rows()
        .iter()
        .find(|row| row.required_contract == "blocking_rollup_rank_full_scan_requires_exact_contracts")
        .expect("contract report should preserve the structured rejection code");
    assert_eq!(blocking_row.execution_mode, CapabilityEnvelopeStatus::Rejected);
    assert_eq!(blocking_row.contract_status, ContractStatus::RejectedByCapability);
    assert_eq!(blocking_row.contract_quality, ContractQuality::CapabilityProof);
}

#[test]
fn strict_admission_rejects_large_rollup_rank_native_fragment_without_outer_topk() {
    let plan = large_full_scan_rollup_rank_native_fragment_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.has_rejections(),
        "large rollup/rank native fragment should be rejected even when the outer TopK remains outside NexusNativeExec"
    );
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("final_topk_fetch").map(String::as_str), Some("unknown"));
    assert_ne!(rejection.facts.get("rank_sort_id").map(String::as_str), Some("absent"));
    assert_ne!(rejection.facts.get("rank_filter_id").map(String::as_str), Some("absent"));
    assert!(
        bounded_pipeline_admission(&plan).expect("strict admission should not fail").is_none(),
        "native fragment rejected by the guard must not produce a strict pipeline"
    );
}

#[test]
fn strict_admission_rejects_rollup_rank_topk_when_selected_bytes_are_still_large() {
    let plan = selected_large_rollup_rank_topk_plan();
    let analysis = plan.capability_analysis();

    assert!(analysis.has_rejections(), "selected row groups are not enough when selected bytes remain large");
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("large_source_selected_bytes").map(String::as_str), Some("1610612736"));
    assert_eq!(rejection.facts.get("large_source_selected_row_groups").map(String::as_str), Some("2"));
    assert_eq!(rejection.facts.get("large_source_total_row_groups").map(String::as_str), Some("4"));
}

#[test]
fn strict_admission_rejects_q86_class_rollup_rank_topk_selected_scan() {
    let plan = selected_q86_class_rollup_rank_topk_plan();
    let analysis = plan.capability_analysis();

    assert!(
        analysis.has_rejections(),
        "q86-class rollup/rank/top-k selected scans are fact-sized even below the old 1 GiB cutoff"
    );
    let rejection = analysis.first_rejection_projection().expect("shape should project a rejection");
    assert_eq!(rejection.reason, "blocking_rollup_rank_full_scan_requires_exact_contracts");
    assert_eq!(rejection.facts.get("large_source_selected_bytes").map(String::as_str), Some("939524096"));
    assert_eq!(rejection.facts.get("large_source_selected_row_groups").map(String::as_str), Some("448"));
}

#[test]
fn strict_admission_selects_q86_class_rollup_rank_topk_envelope_after_aggregate_join_pushdown() {
    let plan = selected_q86_class_rollup_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the q86-class shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized plan should contain a detail-level pre-aggregate"
    );
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized plan should retain the final grouping-set aggregate"
    );

    let analysis = optimized.plan.capability_analysis();
    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(rejection.is_none(), "source reduced by pre-aggregate should not trip the blocking-shape guard");
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "q86-class",
        ExpectedCompiledGraph {
            tasks: 13,
            edges: 13,
            barriers: 10,
            resources: 10,
            gates: 15,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: None,
        },
    );
}

#[test]
fn strict_admission_selects_q70_class_left_semi_rollup_rank_topk_envelope_after_aggregate_join_pushdown() {
    let plan = selected_q70_class_left_semi_rollup_rank_topk_plan();
    let raw_analysis = plan.capability_analysis();
    assert!(
        raw_analysis.has_rejections(),
        "raw q70-class left-semi rollup/rank/top-k selected scan should still hit the blocking guard"
    );
    assert_eq!(
        raw_analysis.first_rejection_projection().expect("shape should project a rejection").reason,
        "blocking_rollup_rank_full_scan_requires_exact_contracts"
    );

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the q70-class left-semi shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized q70-class plan should contain a detail-level pre-aggregate"
    );
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized q70-class plan should retain the final grouping-set aggregate"
    );

    let analysis = optimized.plan.capability_analysis();
    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(rejection.is_none(), "left-semi source reduced by pre-aggregate should not trip the guard");
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "q70-class left-semi",
        ExpectedCompiledGraph {
            tasks: 13,
            edges: 12,
            barriers: 10,
            resources: 10,
            gates: 17,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: None,
        },
    );
}

#[test]
fn strict_admission_selects_q67_class_rollup_rank_topk_projection_chain_envelope_after_aggregate_join_pushdown() {
    let plan = selected_q67_class_rollup_rank_topk_projection_chain_plan();

    // Sanity: before the rewrite the q67-class shape (projection chain over a
    // dimension-fact join with mixed group sides) trips the blocking guard.
    let raw_analysis = plan.capability_analysis();
    assert!(
        raw_analysis.has_rejections(),
        "raw q67-class projection-chain rollup/rank/top-k should still hit the blocking guard"
    );
    assert_eq!(
        raw_analysis.first_rejection_projection().expect("shape should project a rejection").reason,
        "blocking_rollup_rank_full_scan_requires_exact_contracts"
    );

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the q67-class projection-chain shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized q67-class plan should contain a detail-level pre-aggregate"
    );
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some())),
        "optimized q67-class plan should retain the final grouping-set aggregate"
    );

    // The pre-aggregate must read a plain column reference; the inlined
    // measure expression (the original projection-chain `COALESCE(amount *
    // quantity, 0)`) is materialized in a projection above the pre-aggregate's
    // input so the executor's plain-column aggregate-input contract holds.
    let preagg = optimized
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none() => Some(aggregate),
            _ => None,
        })
        .expect("optimized plan should contain a pre-aggregate");
    assert_eq!(preagg.spec.aggr_exprs.len(), 1);
    let preagg_aggr = &preagg.spec.aggr_exprs[0];
    assert_eq!(preagg_aggr.function, AggregateFunction::Sum);
    assert!(
        matches!(&preagg_aggr.expr, Expr::Column(_)),
        "pre-aggregate measure must be a plain column reference, got: {}",
        preagg_aggr.expr
    );

    let measure_projection = optimized.plan.node(preagg.input).expect("pre-aggregate input must exist");
    let PlanNode::Projection(measure_projection) = &measure_projection.kind else {
        panic!("pre-aggregate input should be the materializing projection, got: {:?}", measure_projection.kind);
    };
    let measure_output = measure_projection
        .outputs
        .iter()
        .find(|output| output.expr.to_string().contains("CASE WHEN"))
        .expect("materializing projection should carry the inlined CASE expression for the measure");
    assert!(
        measure_output.expr.to_string().contains("CASE WHEN"),
        "materializing projection should retain the inlined CASE expression, got: {}",
        measure_output.expr
    );

    let analysis = optimized.plan.capability_analysis();
    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(
        rejection.is_none(),
        "q67-class projection-chain source reduced by pre-aggregate should not trip the blocking-shape guard"
    );
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "q67-class projection-chain",
        ExpectedCompiledGraph {
            tasks: 14,
            edges: 14,
            barriers: 10,
            resources: 11,
            gates: 15,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: None,
        },
    );
}

#[test]
fn strict_admission_selects_wide_rollup_rank_topk_envelope_after_aggregate_join_pushdown() {
    let plan = selected_wide_q67_class_rollup_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the wide q67-class shape");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let final_rollup = optimized
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_some() => Some(aggregate),
            _ => None,
        })
        .expect("optimized wide rollup plan should retain the final grouping-set aggregate");
    assert_eq!(final_rollup.spec.group_exprs.len(), 8);
    assert_eq!(final_rollup.spec.grouping_sets.as_ref().expect("grouping sets").groups.len(), 9);
    assert!(
        optimized
            .plan
            .nodes()
            .iter()
            .any(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.grouping_sets.is_none())),
        "optimized wide rollup plan should contain a detail-level pre-aggregate"
    );

    let analysis = optimized.plan.capability_analysis();
    assert_rollup_rank_topk_after_pushdown_envelope(
        &optimized.plan,
        &analysis,
        "wide rollup/rank/top-k",
        ExpectedCompiledGraph {
            tasks: 13,
            edges: 13,
            barriers: 10,
            resources: 10,
            gates: 16,
            sink: SinkBoundaryKind::HostArrowMaterialization,
            capability_contract: Some("rollup_rank_topk_wide_final_rollup_subcontract"),
        },
    );
}

#[test]
fn strict_admission_rejects_too_wide_rollup_rank_topk_without_explicit_width_contract() {
    let plan = selected_too_wide_rollup_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the too-wide rollup/rank/top-k shape");
    let analysis = optimized.plan.capability_analysis();
    assert_eq!(
        analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission.rejection_reason(),
        Some("rollup_rank_topk_after_pushdown_requires_bounded_final_rollup_width"),
        "a 9-key/10-set rollup must stay rejected until a separate width contract is modeled"
    );
    assert!(
        !analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "too-wide rollup must not be admitted through the q67-wide subcontract"
    );
}

#[test]
fn strict_admission_rejects_wide_non_rollup_grouping_sets_without_explicit_width_contract() {
    let plan = selected_wide_non_rollup_grouping_sets_rank_topk_plan();
    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert!(optimized.changed(), "aggregate-join pushdown should rewrite the wide non-rollup grouping-set shape");
    let analysis = optimized.plan.capability_analysis();
    assert_eq!(
        analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission.rejection_reason(),
        Some("rollup_rank_topk_after_pushdown_requires_bounded_final_rollup_width"),
        "an 8-key/9-set grouping aggregate must still prove the modeled suffix-rollup mask contract"
    );
    assert!(
        !analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "same-width non-rollup grouping sets must not be admitted through the q67-wide subcontract"
    );
}

#[test]
fn strict_admission_accepts_exact_statistical_join_final_full_sort_envelope() {
    let plan = statistical_join_final_full_sort_plan();
    let rendered = plan.to_string();
    assert!(rendered.contains("StddevSamp"), "fixture must exercise q39-class statistical aggregate:\n{rendered}");
    assert!(rendered.contains("Sort"), "fixture must retain the final full sort:\n{rendered}");

    let analysis = plan.capability_analysis();
    let statistical_invariant = analysis
        .invariants
        .iter()
        .find(|invariant| invariant.kind == SubgraphInvariantKind::StatisticalJoinFinalSortExactness)
        .expect("q39-class shape should carry a statistical/final-sort exactness invariant");
    assert_eq!(statistical_invariant.decision, CapabilityDecision::Satisfied);
    let exact_contracts = statistical_invariant
        .reservation_requirements
        .iter()
        .map(|requirement| {
            assert_eq!(requirement.kind, ReservationKind::CapabilityContract);
            assert_eq!(requirement.status, ContractStatus::Modeled);
            requirement.required_contract
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        exact_contracts,
        BTreeSet::from([
            "host_welford_input_materialization_preflight",
            "host_welford_output_upload_reservation",
            "final_full_sort_output_cardinality_bound",
        ])
    );
    assert!(analysis.first_rejection_projection().is_none(), "{:?}", analysis.first_rejection_projection());

    let admission = bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("q39-class exact statistical final-sort shape should be admitted");
    assert_eq!(admission.envelope(), CapabilityEnvelope::ExactStatisticalJoinFinalFullSort);
    assert_eq!(
        compiled_bounded_pipeline_support(admission.pipeline()).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("accepted statistical final-sort capability analysis should produce a strict contract report");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
    for required_contract in [
        "host_welford_input_materialization_preflight",
        "host_welford_output_upload_reservation",
        "final_full_sort_output_cardinality_bound",
        "final_sort_contract",
        "query_engine_explicit_sink_materialization_reservation",
    ] {
        assert!(
            report.gate_rows().iter().any(|row| {
                row.required_contract == required_contract && row.contract_status == ContractStatus::Modeled
            }),
            "missing compiled strict gate {required_contract}: {:?}",
            report.gate_rows()
        );
    }
}

#[test]
fn strict_admission_mode_boundary_accepts_tolerant_statistical_envelope_without_exact_host_gates() {
    let plan = statistical_join_final_full_sort_plan();
    let shape_admission = bounded_pipeline_admission(&plan)
        .expect("shape-only strict admission should not fail")
        .expect("shape-only exact statistical envelope should be admitted");
    assert_eq!(shape_admission.envelope(), CapabilityEnvelope::ExactStatisticalJoinFinalFullSort);

    let tolerant_admission = bounded_pipeline_admission_with_statistical_aggregate_mode(
        &plan,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
    .expect("mode-aware strict admission should not fail")
    .expect("tolerant statistical envelope should be admitted");
    assert_eq!(tolerant_admission.envelope(), CapabilityEnvelope::TolerantStatisticalJoinFinalFullSort);

    let projection = bounded_pipeline_rejection_projection_with_statistical_aggregate_mode(
        &plan,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    )
    .expect("mode-aware rejection projection should not fail");
    assert_eq!(projection, None);

    let report = CapabilityContractReport::from_compiled_pipeline(
        tolerant_admission.analysis(),
        tolerant_admission.pipeline(),
    )
    .expect("accepted tolerant statistical final-sort admission should produce a strict contract report");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
    assert!(report.gate_rows().iter().any(|row| {
        row.required_contract == "gpu_native_tolerant_statistical_aggregate_contract"
            && row.memory_boundary == "tolerant_statistical_join_final_full_sort"
            && row.contract_status == ContractStatus::Modeled
    }));
    assert!(report.gate_rows().iter().any(|row| {
        row.required_contract == "final_full_sort_output_cardinality_bound"
            && row.contract_status == ContractStatus::Modeled
    }));
    assert_eq!(
        report
            .gate_rows()
            .iter()
            .filter(|row| matches!(
                row.required_contract,
                "host_welford_input_materialization_preflight" | "host_welford_output_upload_reservation"
            ))
            .count(),
        0,
        "tolerant mode must not publish exact host-Welford gates"
    );
}

#[derive(Debug, Clone, Copy)]
struct ExpectedCompiledGraph {
    tasks: usize,
    edges: usize,
    barriers: usize,
    resources: usize,
    gates: usize,
    sink: SinkBoundaryKind,
    capability_contract: Option<&'static str>,
}

fn assert_rollup_rank_topk_after_pushdown_envelope(
    plan: &QueryPlan,
    analysis: &nexus_query_engine::capability::CapabilityAnalysis,
    label: &str,
    expected_graph: ExpectedCompiledGraph,
) {
    assert!(
        !analysis.has_rejections(),
        "{label}: optimized rollup/rank/top-k shape should have satisfied local capability facts; first_rejection={:?}",
        analysis.first_rejection_projection()
    );
    assert!(
        analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "{label}: expected rollup/rank/top-k post-pushdown envelope admission; rollup_decision={:?}; first_rejection={:?}",
        analysis.rollup_rank_topk_after_aggregate_join_pushdown_admission,
        analysis.first_rejection_projection()
    );
    let CapabilityEnvelopeAdmission::Admitted { envelope } = analysis.capability_envelope_admission() else {
        panic!("{label}: expected selected strict envelope, got {:?}", analysis.capability_envelope_admission());
    };
    assert_eq!(envelope, CapabilityEnvelope::RollupRankTopKAfterAggregateJoinPushdown);
    assert_eq!(envelope.as_str(), "rollup_rank_topk_after_aggregate_join_pushdown");

    let report = CapabilityContractReport::from_capability_analysis(analysis)
        .unwrap_or_else(|error| panic!("{label}: capability contract report failed: {error}"));
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
    assert!(report.is_ready(), "{label}: capability gate rows should be strict-ready");

    let aggregate_rows = report
        .gate_rows()
        .iter()
        .filter(|row| {
            row.reservation_kind == ReservationKind::OperatorOutput && row.required_contract == "aggregate_contract"
        })
        .count();
    assert!(
        aggregate_rows == 2,
        "{label}: detail and final grouping aggregates should both contribute modeled aggregate gate rows"
    );
    assert_modeled_gate_row(label, &report, ReservationKind::SourceRead, "source_contract");
    assert_modeled_gate_row(label, &report, ReservationKind::OperatorOutput, "window_contract");
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::OperatorOutput,
        "query_engine_rank_filter_output_boundary",
    );
    assert_modeled_gate_row(label, &report, ReservationKind::FinalSort, "final_sort_contract");
    assert_modeled_gate_row(label, &report, ReservationKind::OperatorOutput, "query_engine_final_topk_output_boundary");
    if let Some(required_contract) = expected_graph.capability_contract {
        assert_modeled_gate_row(label, &report, ReservationKind::CapabilityContract, required_contract);
    }
    assert_modeled_gate_row(
        label,
        &report,
        ReservationKind::Materialization,
        "query_engine_explicit_sink_materialization_reservation",
    );

    let pipeline = compile_pipeline(plan, analysis)
        .unwrap_or_else(|error| panic!("{label}: rollup/rank/top-k pipeline should compile: {error}"));
    pipeline
        .validate()
        .unwrap_or_else(|error| panic!("{label}: rollup/rank/top-k compiled pipeline should validate: {error}"));
    assert_compiled_graph_counts(label, &pipeline, expected_graph);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline)
            .unwrap_or_else(|error| panic!("{label}: graph support check failed: {error}")),
        CompiledBoundedPipelineSupport::Supported,
        "{label}: compiled graph should be inside the strict runtime envelope"
    );
    assert_eq!(
        compiled_bounded_query_plan_support(plan, &pipeline)
            .unwrap_or_else(|error| panic!("{label}: plan-aware support check failed: {error}")),
        CompiledBoundedPipelineSupport::Supported,
        "{label}: compiled plan should be inside the strict runtime envelope"
    );
    let runtime_admission = bounded_pipeline_admission(plan)
        .unwrap_or_else(|error| panic!("{label}: strict bounded runtime admission failed: {error}"))
        .unwrap_or_else(|| panic!("{label}: strict bounded runtime admission should be selected"));
    assert_eq!(runtime_admission.envelope(), CapabilityEnvelope::RollupRankTopKAfterAggregateJoinPushdown);
    assert_compiled_graph_counts(label, runtime_admission.pipeline(), expected_graph);
}

fn assert_compiled_graph_counts(label: &str, pipeline: &CompiledPipeline, expected: ExpectedCompiledGraph) {
    assert_eq!(pipeline.tasks().len(), expected.tasks, "{label}: compiled task count changed");
    assert_eq!(pipeline.edges().len(), expected.edges, "{label}: compiled edge count changed");
    assert_eq!(pipeline.barriers().len(), expected.barriers, "{label}: compiled barrier count changed");
    assert_eq!(pipeline.resources().len(), expected.resources, "{label}: compiled resource count changed");
    assert_eq!(pipeline.gates().len(), expected.gates, "{label}: compiled gate count changed");
    assert_eq!(pipeline.sink().kind(), expected.sink, "{label}: compiled sink kind changed");
}

fn assert_modeled_gate_row(
    label: &str,
    report: &CapabilityContractReport,
    kind: ReservationKind,
    required_contract: &str,
) {
    assert!(
        report.gate_rows().iter().any(|row| {
            row.reservation_kind == kind
                && row.required_contract == required_contract
                && row.contract_status == ContractStatus::Modeled
        }),
        "{label}: expected modeled {kind:?} gate row for {required_contract}; rows={:?}",
        report.gate_rows()
    );
}

#[test]
fn strict_admission_does_not_apply_rollup_rank_topk_guard_when_selected_bytes_are_small() {
    let plan = selected_small_rollup_rank_topk_plan();
    let analysis = plan.capability_analysis();

    let rejection = analysis.invariants.iter().find(|invariant| {
        invariant.decision.rejection_reason() == Some("blocking_rollup_rank_full_scan_requires_exact_contracts")
    });
    assert!(rejection.is_none(), "small selected source should not trip the large blocking-shape guard");
}
