use std::{
    collections::BTreeSet,
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptiveCandidateStatus, AdaptiveExecutionMode,
    AdaptiveMemoryHistory, AdaptiveMemoryHistoryEvidence, AdaptiveMemoryHistoryKey, AdaptiveMemoryObservation,
    AdaptivePipelineReport, AdaptiveStrategy, AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnRef,
    CompiledPipeline, ExecutionContext, Expr, JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitSpec,
    LiteralValue, LocalParquetSource, MemoryControlPolicy, NativePipelineExecutionMode, PipelineBarrier,
    PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId, PipelineEdgeKind, PipelineTask,
    PipelineTaskId, PipelineTaskKind, ProjectionOutput, QueryMetrics, QueryOutputMode, QueryPlan, QueryPlanBuilder,
    ReservationGate, ReservationGateId, ReservationGateOwner, ReservationKind, ResidentLifetime, ResidentResource,
    ResidentResourceId, SinkBoundary, SinkBoundaryKind, SortExpr, SortSpec, SourceNode, StrictBoundedContractGateId,
    StrictBoundedContractGateRow, StrictBoundedContractOwner, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history,
    capability::{
        CapabilityDecision, ContractQuality, ContractStatus, ReservationRequirementScope,
        STRICT_UNRESOLVED_PREFLIGHT_GAPS, SubgraphInvariantKind, lower_level_unresolved_preflight_gap,
    },
    compile_pipeline,
    contract_report::{StrictBoundedContractReport, StrictBoundedExecutionMode},
    exec::{
        CompiledBoundedPipelineSupport, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
        CompiledTaskExecutor, CompiledTaskOutput, compiled_bounded_pipeline_support,
        compiled_bounded_query_plan_support,
    },
    execute_query_plan, strict_bounded_pipeline_admission,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use crate::bounded_join_admission::{
    AggregateOverJoinSubplanInputsVariant, ResidualResidentGroupedAggregateInputVariant,
    admitted_mixed_residual_multi_join_plan, keyless_conditional_aggregate_over_join_subplan_inputs_plan,
    large_keyed_join_strategy_plan,
    plans::{
        KeyedHashJoinResidentSubplanPipelineVariant, KeylessInputSubplanVariant, aggregate_over_join_subplan_topk_plan,
        keyed_hash_join_resident_subplan_pipeline_plan, keyless_conditional_join_input_subplan_plan,
        resident_grouped_aggregate_input_post_join_aggregate_topk_plan,
        resident_grouped_aggregate_subquery_filter_plan, single_join_post_root_aggregate_sort_limit_plan,
    },
    residual_resident_grouped_aggregate_input_plan, unsupported_cross_key_join_plan,
};

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
fn phase_e2_native_pipeline_modes_are_explicit_and_report_construction_stays_strict_only() {
    assert_eq!(NativePipelineExecutionMode::StrictBounded.as_str(), "strict_bounded");
    assert_eq!(NativePipelineExecutionMode::Adaptive.as_str(), "adaptive");
    assert_eq!(
        StrictBoundedExecutionMode::UnsupportedStrictBounded.native_mode(),
        NativePipelineExecutionMode::StrictBounded
    );
    assert_eq!(AdaptiveExecutionMode::Adaptive.native_mode(), NativePipelineExecutionMode::Adaptive);
    assert_ne!(AdaptiveExecutionMode::Adaptive.as_str(), StrictBoundedExecutionMode::StrictBounded.as_str());
    assert_ne!(
        AdaptiveExecutionMode::UnsupportedAdaptive.as_str(),
        StrictBoundedExecutionMode::UnsupportedStrictBounded.as_str()
    );

    let _: fn(
        StrictBoundedExecutionMode,
        Vec<StrictBoundedContractGateRow>,
    ) -> cudf_query_engine::Result<StrictBoundedContractReport> = StrictBoundedContractReport::from_gate_rows;
}

#[test]
fn phase_e2_strict_execution_uses_strict_admission_evidence() {
    let plan = root_join_plan();
    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("root join should produce strict admission evidence");

    assert_eq!(admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    assert!(admission.analysis().admits_strict_bounded_join());

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("strict admission evidence should build strict report rows");
    assert_eq!(report.execution_mode().native_mode(), NativePipelineExecutionMode::StrictBounded);
    assert!(report.gate_rows().iter().all(|row| {
        row.execution_mode.native_mode() == NativePipelineExecutionMode::StrictBounded
            && row.execution_mode.as_str() != AdaptiveExecutionMode::Adaptive.as_str()
            && row.execution_mode.as_str() != AdaptiveExecutionMode::UnsupportedAdaptive.as_str()
    }));
}

#[test]
fn phase_e3_adaptive_admission_reports_use_adaptive_rows_not_strict_contract_rows() {
    let plan = root_join_plan();
    let admission = adaptive_bounded_pipeline_admission(&plan)
        .expect("adaptive admission should not fail")
        .expect("root join should produce adaptive admission evidence");

    assert_eq!(admission.native_mode(), NativePipelineExecutionMode::Adaptive);
    assert!(admission.pipeline().is_some());
    let report = admission.report();
    assert_eq!(report.execution_mode(), AdaptiveExecutionMode::Adaptive);
    let row = report.row();
    assert_eq!(row.execution_mode, AdaptiveExecutionMode::Adaptive);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Admitted);
    assert_eq!(row.adaptive_strategy, AdaptiveStrategy::ConservativeCompiledPipeline);
    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "none");
    assert_eq!(row.runtime_action_counts.retry_count, 0);
    assert_eq!(row.runtime_action_counts.spill_count, 0);
    assert_eq!(row.runtime_action_counts.downgrade_count, 0);
    assert_eq!(row.final_sink_boundary.materialization_mode, QueryOutputMode::MaterializeToHost);
    assert!(
        row.missing_strict_contracts
            .iter()
            .all(|missing| missing.contract_status != ContractStatus::RejectedByCapability)
    );

    let rendered = report.stable_kv_row();
    assert!(rendered.contains("execution_mode=adaptive"));
    assert!(rendered.contains("adaptive_strategy=conservative_compiled_pipeline"));
    assert!(rendered.contains("retry_count=0"));
    assert!(rendered.contains("spill_count=0"));
    assert!(rendered.contains("downgrade_count=0"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("unsupported_strict_bounded"));
}

#[test]
fn phase_e3_unsupported_adaptive_rows_reference_strict_gaps_without_strict_evidence() {
    let plan = unsupported_cross_key_join_plan();
    let admission = adaptive_bounded_pipeline_admission(&plan)
        .expect("adaptive admission should not fail")
        .expect("join plan should produce adaptive report evidence");

    assert_eq!(admission.native_mode(), NativePipelineExecutionMode::Adaptive);
    assert!(admission.pipeline().is_none());
    let row = admission.report().row();
    assert_eq!(row.execution_mode, AdaptiveExecutionMode::UnsupportedAdaptive);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Rejected);
    assert_eq!(row.adaptive_strategy, AdaptiveStrategy::Unsupported);
    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "none");
    assert_eq!(row.runtime_action_counts.retry_count, 0);
    assert_eq!(row.runtime_action_counts.spill_count, 0);
    assert_eq!(row.runtime_action_counts.downgrade_count, 0);
    assert_eq!(row.rejection_reason, Some("missing_strict_contracts"));
    assert!(row.missing_strict_contracts.iter().any(|missing| {
        missing.required_contract == "unsupported_join_node_capability"
            && missing.contract_status == ContractStatus::RejectedByCapability
    }));

    let rendered = admission.report().stable_kv_row();
    assert!(rendered.contains("execution_mode=unsupported_adaptive"));
    assert!(rendered.contains("required_contract=unsupported_join_node_capability"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("execution_mode=unsupported_strict_bounded"));
}

#[test]
fn phase_e6_adaptive_memory_history_lookup_miss_is_adaptive_evidence_only() {
    let plan = root_join_plan();
    let memory_history = AdaptiveMemoryHistory::new();
    let admission = adaptive_bounded_pipeline_admission_with_memory_history(&plan, Some(&memory_history))
        .expect("adaptive admission should not fail")
        .expect("root join should produce adaptive admission evidence");
    let row = admission.report().row();

    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "memory_history_lookup");
    assert_eq!(row.memory_history_evidence, AdaptiveMemoryHistoryEvidence::LookupMiss);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Admitted);
    let strict_admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("strict admission should ignore adaptive memory history");
    assert_eq!(strict_admission.native_mode(), NativePipelineExecutionMode::StrictBounded);

    let strict_report =
        StrictBoundedContractReport::from_compiled_pipeline(strict_admission.analysis(), strict_admission.pipeline())
            .expect("strict report should not read adaptive history");
    assert!(strict_report.gate_rows().iter().all(|row| {
        let rendered = row.to_stable_kv_row();
        !rendered.contains("memory_history") && !rendered.contains("lookup_miss")
    }));
}

#[test]
fn phase_e6_adaptive_memory_history_hit_reports_observed_summary_without_changing_strict_admission() {
    let plan = root_join_plan();
    let memory_history = AdaptiveMemoryHistory::new();
    let key = AdaptiveMemoryHistoryKey::for_query_plan(&plan);
    assert!(!key.as_str().contains("compiled-d1-left"));
    assert!(!key.as_str().contains("left_id"));
    memory_history.record_observation(
        key,
        AdaptiveMemoryObservation {
            resident_dataframe_peak_bytes: Some(4096),
            reservation_granted_bytes: 8192,
            source_chunks_committed: 3,
        },
    );

    let adaptive_admission = adaptive_bounded_pipeline_admission_with_memory_history(&plan, Some(&memory_history))
        .expect("adaptive admission should not fail")
        .expect("root join should produce adaptive admission evidence");
    let row = adaptive_admission.report().row();

    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "memory_history_lookup");
    let AdaptiveMemoryHistoryEvidence::Observed(summary) = &row.memory_history_evidence else {
        panic!("memory history should be reported as observed adaptive evidence");
    };
    assert_eq!(summary.observation_count, 1);
    assert_eq!(summary.max_resident_dataframe_peak_bytes, Some(4096));
    assert_eq!(summary.max_reservation_granted_bytes, 8192);
    assert_eq!(summary.max_source_chunks_committed, 3);
    assert_eq!(row.runtime_action_counts.retry_count, 0);
    assert_eq!(row.runtime_action_counts.spill_count, 0);
    assert_eq!(row.runtime_action_counts.downgrade_count, 0);

    let rendered = adaptive_admission.report().stable_kv_row();
    assert!(rendered.contains("memory_history_evidence=observed:observation_count=1"));
    assert!(rendered.contains("resident_dataframe_peak_bytes=4096"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));

    let strict_admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("strict admission must ignore adaptive memory history");
    assert_eq!(strict_admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    assert!(strict_admission.analysis().admits_strict_bounded_join());
}

#[test]
fn phase_e8_adaptive_diagnostic_failure_rows_are_visible_without_strict_labels() {
    let plan = root_join_plan();
    let report = AdaptivePipelineReport::diagnostic_failure(&plan, ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON);
    let row = report.row();

    assert_eq!(row.execution_mode, AdaptiveExecutionMode::UnsupportedAdaptive);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Rejected);
    assert_eq!(row.adaptive_strategy, AdaptiveStrategy::Unsupported);
    assert_eq!(row.rejection_reason, Some(ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON));

    let rendered = report.stable_kv_row();
    assert!(rendered.contains("execution_mode=unsupported_adaptive"));
    assert!(rendered.contains("rejection_reason=adaptive_diagnostic_construction_failed"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("unsupported_strict_bounded"));
}

#[test]
fn phase_e8_adaptive_memory_history_key_distinguishes_memory_shape_factors() {
    let int_sum_gt = adaptive_memory_history_key_probe_plan(DataType::Int32, AggregateFunction::Sum, BinaryOp::Gt);
    let decimal_sum_gt =
        adaptive_memory_history_key_probe_plan(DataType::Decimal128(20, 2), AggregateFunction::Sum, BinaryOp::Gt);
    let int_count_gt = adaptive_memory_history_key_probe_plan(DataType::Int32, AggregateFunction::Count, BinaryOp::Gt);
    let int_sum_lt = adaptive_memory_history_key_probe_plan(DataType::Int32, AggregateFunction::Sum, BinaryOp::Lt);

    let int_sum_gt_key = AdaptiveMemoryHistoryKey::for_query_plan(&int_sum_gt);
    assert_ne!(int_sum_gt_key, AdaptiveMemoryHistoryKey::for_query_plan(&decimal_sum_gt));
    assert_ne!(int_sum_gt_key, AdaptiveMemoryHistoryKey::for_query_plan(&int_count_gt));
    assert_ne!(int_sum_gt_key, AdaptiveMemoryHistoryKey::for_query_plan(&int_sum_lt));

    assert!(!int_sum_gt_key.as_str().contains("/tmp/"));
    assert!(!int_sum_gt_key.as_str().contains("adaptive_key"));
    assert!(!int_sum_gt_key.as_str().contains("secret_literal"));
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
fn d5_contract_gate_rows_are_inventory_backed_and_family_path_free() {
    let mut hard_budgeted_unresolved_rows = 0;
    let mut source_unresolved_rows = 0;
    let mut aggregate_unresolved_rows = 0;
    let mut final_sort_unresolved_rows = 0;
    let mut one_shot_output_unresolved_rows = 0;
    let mut filtered_retained_state_rows = 0;
    let mut mixed_retained_state_rows = 0;
    let mut observed_retained_state_rows = 0;

    for case in d3_1_inventory_cases() {
        let analysis = case.plan.capability_analysis();
        assert!((case.admits)(&analysis), "{}: expected D.5 inventory case to remain admitted", case.label);
        let pipeline = compile_pipeline(&case.plan, &analysis)
            .unwrap_or_else(|error| panic!("{}: compile failed: {error}", case.label));

        let report = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &pipeline)
            .unwrap_or_else(|error| panic!("{}: strict bounded contract report failed: {error}", case.label));
        assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
        assert!(report.is_strict_bounded_ready(), "{}: compiled gate report should be strict-ready", case.label);
        assert_eq!(
            report.gate_rows().len(),
            pipeline.gates().len(),
            "{}: report should preserve every compiled reservation gate",
            case.label
        );

        let capability_report = StrictBoundedContractReport::from_capability_analysis(&analysis)
            .unwrap_or_else(|error| panic!("{}: capability contract report failed: {error}", case.label));
        assert_eq!(capability_report.execution_mode(), StrictBoundedExecutionMode::StrictBounded);
        assert!(capability_report.is_strict_bounded_ready());

        for row in report.gate_rows() {
            let rendered = row.to_stable_kv_row();
            assert_d5_contract_gate_row_is_family_path_free(case.label, &rendered);
            assert_d5_contract_gate_row_excludes_adaptive_terms(case.label, &rendered);
            assert!(rendered.contains("contract_status="), "{}: row must serialize contract status", case.label);
            assert!(rendered.contains("contract_quality="), "{}: row must serialize contract quality", case.label);
            assert!(
                !rendered.contains("contract_modes"),
                "{}: row must not serialize legacy contract modes",
                case.label
            );

            if row.contract_quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight {
                hard_budgeted_unresolved_rows += 1;
                match row.reservation_kind {
                    ReservationKind::SourceRead => source_unresolved_rows += 1,
                    ReservationKind::OperatorOutput => aggregate_unresolved_rows += 1,
                    ReservationKind::FinalSort => final_sort_unresolved_rows += 1,
                    ReservationKind::JoinOutput => one_shot_output_unresolved_rows += 1,
                    _ => {},
                }
                assert_eq!(
                    row.contract_status,
                    ContractStatus::Modeled,
                    "{}: hard-budgeted unresolved preflight rows stay modeled",
                    case.label
                );
                assert!(
                    row.unresolved_preflight_gap.is_some_and(|gap| gap.starts_with("libcudf_")),
                    "{}: hard-budgeted row should name its lower-level preflight gap",
                    case.label
                );
            }

            if matches!(row.reservation_kind, ReservationKind::FilteredProbeState | ReservationKind::MixedSizeData) {
                observed_retained_state_rows += 1;
                if row.reservation_kind == ReservationKind::FilteredProbeState {
                    filtered_retained_state_rows += 1;
                }
                if row.reservation_kind == ReservationKind::MixedSizeData {
                    mixed_retained_state_rows += 1;
                }
                assert_eq!(
                    row.contract_quality,
                    ContractQuality::ObservedRetainedState,
                    "{}: retained join state must not collapse into exact preflight",
                    case.label
                );
            }

            match row.reservation_kind {
                ReservationKind::HashTable if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert_eq!(
                        row.required_contract, "hash_join_pre_build_reservation_size_bytes",
                        "{}: hash-table exact preflight must name its cudf-rs pre-build sizing owner",
                        case.label
                    );
                },
                ReservationKind::GatherMap if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert_eq!(
                        row.required_contract, "join_gather_map_reservation_size_bytes",
                        "{}: gather-map exact preflight must name its cudf-rs gather-map sizing owner",
                        case.label
                    );
                },
                ReservationKind::JoinOutput if row.contract_quality == ContractQuality::ExactPreflight => {
                    assert!(
                        matches!(
                            row.required_contract,
                            "dataframe_gather_output_reservation_size_bytes"
                                | "projected_cross_join_reservation_size_bytes"
                                | "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes"
                        ),
                        "{}: join-output exact preflight must name an owned lower-level output sizing contract",
                        case.label
                    );
                },
                _ => {},
            }
        }
    }

    assert!(hard_budgeted_unresolved_rows > 0, "D.5 inventory should cover unresolved hard-budgeted gates");
    assert!(source_unresolved_rows > 0, "D.5 inventory should cover source unresolved preflight gates");
    assert!(aggregate_unresolved_rows > 0, "D.5 inventory should cover aggregate unresolved preflight gates");
    assert!(final_sort_unresolved_rows > 0, "D.5 inventory should cover final-sort unresolved preflight gates");
    assert!(
        one_shot_output_unresolved_rows > 0,
        "D.5 inventory should cover one-shot equality join output-size preflight gates"
    );
    assert!(filtered_retained_state_rows > 0, "D.5 inventory should cover filtered semi/anti retained state");
    assert!(mixed_retained_state_rows > 0, "D.5 inventory should cover mixed residual retained state");
    assert!(observed_retained_state_rows > 0, "D.5 inventory should cover observed retained-state gates");
}

#[test]
fn phase_f6_strict_preflight_gap_registry_tracks_current_retirement_protocol() {
    assert_eq!(
        STRICT_UNRESOLVED_PREFLIGHT_GAPS.len(),
        4,
        "F.6 starts with the four lower-level strict preflight gaps; retirement must remove one from this registry"
    );

    let mut markers = BTreeSet::new();
    let mut requirements = BTreeSet::new();
    for gap in STRICT_UNRESOLVED_PREFLIGHT_GAPS {
        assert!(
            markers.insert(gap.unresolved_gap),
            "unresolved preflight markers should be unique: {}",
            gap.unresolved_gap
        );
        assert!(
            requirements.insert(format!("{}:{}", gap.kind.as_str(), gap.required_contract)),
            "reservation-kind/contract pairs should be unique: {}:{}",
            gap.kind.as_str(),
            gap.required_contract
        );
        assert!(
            gap.retirement_requirement.contains("cudf-rs/libcudf"),
            "retirement must wait on an owning lower-level API: {}",
            gap.retirement_requirement
        );
        assert_eq!(lower_level_unresolved_preflight_gap(gap.kind, gap.required_contract), Some(gap.unresolved_gap));
    }

    assert!(markers.contains("libcudf_parquet_decoded_output_preflight_unavailable"));
    assert!(markers.contains("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"));
    assert!(markers.contains("libcudf_final_sort_order_preflight_unavailable"));
    assert!(markers.contains("libcudf_equality_join_output_size_preflight_unavailable"));
}

#[test]
fn d5_contract_report_rejects_unresolved_preflight_gate_labeled_exact() {
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let mut gates = pipeline.gates().to_vec();
    let source_gate_index = gates
        .iter()
        .position(|gate| gate.reservation_kind() == ReservationKind::SourceRead)
        .expect("root join should have source-read gates");
    let source_gate = &gates[source_gate_index];
    gates[source_gate_index] = ReservationGate::new(
        source_gate.id(),
        source_gate.owner(),
        ReservationKind::SourceRead,
        ReservationRequirementScope::Edge,
        "source_read",
        "source_contract",
        ContractStatus::Modeled,
        ContractQuality::ExactPreflight,
    );
    let malformed = CompiledPipeline::new(
        pipeline.tasks().to_vec(),
        pipeline.edges().to_vec(),
        pipeline.barriers().to_vec(),
        pipeline.resources().to_vec(),
        gates,
        pipeline.sink().clone(),
    );

    let error = StrictBoundedContractReport::from_compiled_pipeline(&analysis, &malformed)
        .expect_err("D.5 report construction should reject mislabeled unresolved preflight");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("contract_quality"), Some("exact_preflight"));
}

#[test]
fn phase_f6_contract_report_rejects_hard_budgeted_quality_without_tracked_gap() {
    let invalid = d5_test_gate_row(
        ReservationKind::SourceRead,
        "source_read",
        "libcudf_decoded_output_preflight_contract_after_api_lands",
        ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        None,
    );

    let error = StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
        .expect_err("F.6 should reject unresolved quality after the placeholder contract is replaced");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(
        error.fact_value("required_contract"),
        Some("libcudf_decoded_output_preflight_contract_after_api_lands")
    );
}

#[test]
fn d5_contract_report_rejects_public_gate_rows_with_invalid_quality_accounting() {
    for invalid in [
        d5_test_gate_row(
            ReservationKind::SourceRead,
            "source_read",
            "source_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            "filtered_join_retained_probe_state",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::MixedSizeData,
            "join_mixed_size_data",
            "mixed_keyed_residual_or_conditional_full_retained_size_data",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "adapter_guessed_hash_table_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::GatherMap,
            "join_gather_map",
            "adapter_guessed_gather_map_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
        d5_test_gate_row(
            ReservationKind::JoinOutput,
            "join_output",
            "adapter_guessed_join_output_bytes",
            ContractQuality::ExactPreflight,
            None,
        ),
    ] {
        let kind = invalid.reservation_kind;
        let error =
            StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
                .unwrap_err();
        assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("reservation_kind"), Some(kind.as_str()));
    }
}

#[test]
fn d5_contract_report_rejects_public_unresolved_preflight_rows_with_non_hard_budgeted_quality() {
    let unresolved_contracts = [
        (ReservationKind::SourceRead, "source_read", "source_contract"),
        (ReservationKind::OperatorOutput, "aggregate_output", "aggregate_contract"),
        (ReservationKind::FinalSort, "final_sort", "final_sort_contract"),
        (ReservationKind::JoinOutput, "join_output", "one_shot_equality_join_output_size_preflight"),
    ];
    let non_hard_budgeted_qualities = [
        ContractQuality::ExactPreflight,
        ContractQuality::MetadataBound,
        ContractQuality::ObservedRetainedState,
        ContractQuality::QueryEngineOwnedOrdering,
        ContractQuality::QueryEngineOwnedBoundary,
        ContractQuality::CapabilityProof,
    ];

    for (reservation_kind, memory_boundary, required_contract) in unresolved_contracts {
        for contract_quality in non_hard_budgeted_qualities {
            let invalid =
                d5_test_gate_row(reservation_kind, memory_boundary, required_contract, contract_quality, None);
            let error =
                StrictBoundedContractReport::from_gate_rows(StrictBoundedExecutionMode::StrictBounded, vec![invalid])
                    .unwrap_err();
            assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
            assert_eq!(error.fact_value("reservation_kind"), Some(reservation_kind.as_str()));
            assert_eq!(error.fact_value("required_contract"), Some(required_contract));
            assert_eq!(error.fact_value("contract_quality"), Some(contract_quality.as_str()));
        }
    }
}

#[test]
fn d5_contract_report_rejects_public_rows_with_inconsistent_execution_modes_and_scope_owners() {
    let strict_with_blocked = StrictBoundedContractGateRow {
        contract_status: ContractStatus::BlockedByLowerLevel,
        ..d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
            None,
        )
    };
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::StrictBounded,
        vec![strict_with_blocked],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("report_execution_mode"), Some("strict_bounded"));

    let unsupported_with_modeled = d5_test_gate_row(
        ReservationKind::HashTable,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        ContractQuality::ExactPreflight,
        None,
    );
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::UnsupportedStrictBounded,
        vec![unsupported_with_modeled],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("report_execution_mode"), Some("unsupported_strict_bounded"));

    let edge_owned_node_scope = StrictBoundedContractGateRow {
        owner: StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(PipelineEdgeId::new(0))),
        ..d5_test_gate_row(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
            None,
        )
    };
    let error = StrictBoundedContractReport::from_gate_rows(
        StrictBoundedExecutionMode::StrictBounded,
        vec![edge_owned_node_scope],
    )
    .unwrap_err();
    assert_eq!(error.fact_value("contract_scope"), Some("node"));
    assert_eq!(error.fact_value("owner"), Some("edge:e0"));
}

fn d5_test_gate_row(
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    required_contract: &'static str,
    contract_quality: ContractQuality,
    unresolved_preflight_gap: Option<&'static str>,
) -> StrictBoundedContractGateRow {
    let contract_scope = d5_test_scope_for_reservation_kind(reservation_kind);
    StrictBoundedContractGateRow {
        execution_mode: StrictBoundedExecutionMode::StrictBounded,
        contract_scope,
        gate_id: StrictBoundedContractGateId::Compiled(ReservationGateId::new(reservation_kind as usize)),
        owner: d5_test_owner_for_scope(contract_scope),
        reservation_kind,
        memory_boundary,
        required_contract,
        contract_status: ContractStatus::Modeled,
        contract_quality,
        unresolved_preflight_gap,
    }
}

fn d5_test_scope_for_reservation_kind(reservation_kind: ReservationKind) -> ReservationRequirementScope {
    match reservation_kind {
        ReservationKind::OperatorOutput
        | ReservationKind::HashTable
        | ReservationKind::FilteredProbeState
        | ReservationKind::MixedSizeData
        | ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge => ReservationRequirementScope::Node,
        ReservationKind::SourceRead
        | ReservationKind::BuildSide
        | ReservationKind::ProbeSide
        | ReservationKind::GatherMap
        | ReservationKind::JoinOutput
        | ReservationKind::FinalSort
        | ReservationKind::Materialization => ReservationRequirementScope::Edge,
        ReservationKind::CapabilityContract => ReservationRequirementScope::Subgraph,
    }
}

fn d5_test_owner_for_scope(scope: ReservationRequirementScope) -> StrictBoundedContractOwner {
    match scope {
        ReservationRequirementScope::Node => {
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Task(PipelineTaskId::new(0)))
        },
        ReservationRequirementScope::Edge => {
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Edge(PipelineEdgeId::new(0)))
        },
        ReservationRequirementScope::Subgraph => {
            StrictBoundedContractOwner::Compiled(ReservationGateOwner::Barrier(PipelineBarrierId::new(0)))
        },
    }
}

#[test]
fn compiled_scheduler_runs_tasks_from_dependencies_and_releases_after_all_consumers() {
    let pipeline = out_of_order_join_barrier_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new();

    let state = scheduler.run(&mut executor).expect("mock pipeline should execute");

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(2),
            PipelineTaskId::new(1),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
        ]
    );
    assert_eq!(executor.executed, state.execution_order());
    assert_eq!(executor.released, vec![ResidentResourceId::new(0), ResidentResourceId::new(1)]);
    assert_eq!(state.live_resource_count(), 0);
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
    assert!(state.is_resource_released(ResidentResourceId::new(1)));
}

#[test]
fn compiled_scheduler_runs_shared_resident_fanout_and_releases_after_all_consumers() {
    let pipeline = shared_resident_fanout_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("shared fan-out pipeline should be schedulable");
    let mut executor = MockExecutor::new();

    let state = scheduler.run(&mut executor).expect("shared fan-out pipeline should execute");

    assert_eq!(state.execution_order(), &[PipelineTaskId::new(0), PipelineTaskId::new(1), PipelineTaskId::new(2)]);
    assert_eq!(executor.executed, state.execution_order());
    assert_eq!(executor.released, vec![ResidentResourceId::new(0)]);
    assert_eq!(state.live_resource_count(), 0);
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
}

#[test]
fn compiled_state_keeps_shared_resource_live_until_every_consumer_completes() {
    let pipeline = shared_resident_fanout_pipeline();
    let mut state = CompiledPipelineState::<String>::new(&pipeline).expect("state should initialize");

    let released = state
        .complete_task(&pipeline.tasks()[0], vec![(ResidentResourceId::new(0), "frame".to_string())])
        .expect("source should produce shared resource");
    assert!(released.is_empty());
    assert!(state.is_resource_live(ResidentResourceId::new(0)));

    let released =
        state.complete_task(&pipeline.tasks()[1], vec![]).expect("first consumer should not release shared resource");
    assert!(released.is_empty());
    assert!(state.is_resource_live(ResidentResourceId::new(0)));

    let released = state
        .complete_task(&pipeline.tasks()[2], vec![])
        .expect("last completed consumer should release shared resource");
    assert_eq!(released.iter().map(|resource| resource.id()).collect::<Vec<_>>(), vec![ResidentResourceId::new(0)]);
    assert!(state.is_resource_released(ResidentResourceId::new(0)));
}

#[test]
fn compiled_scheduler_errors_when_task_omits_declared_output_resource() {
    let pipeline = simple_unary_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new().omit_output_for(PipelineTaskId::new(0));

    let error = scheduler.run(&mut executor).expect_err("missing output should fail");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t0"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
}

#[test]
fn compiled_scheduler_state_errors_when_input_resource_is_missing() {
    let pipeline = simple_unary_pipeline();
    let state = CompiledPipelineState::<String>::new(&pipeline).expect("state should initialize");

    let error = state.prepare_task(&pipeline.tasks()[1]).expect_err("downstream task should require produced input");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t1"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
}

#[test]
fn compiled_scheduler_errors_when_task_reports_duplicate_produced_resource() {
    let pipeline = simple_unary_pipeline();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let mut executor = MockExecutor::new().duplicate_output_for(PipelineTaskId::new(0));

    let error = scheduler.run(&mut executor).expect_err("duplicate output should fail");

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t0"));
    assert_eq!(error.facts().get("resident_resource_id"), Some("r0"));
}

#[test]
fn compiled_gpu_executor_runs_root_hash_join_build_probe_materialize() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d14");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let compiled_plan = root_join_data_plan(&left_path, &right_path);
    let legacy_plan = root_join_data_plan(&left_path, &right_path);
    let analysis = compiled_plan.capability_analysis();
    let pipeline = compile_pipeline(&compiled_plan, &analysis).expect("root join should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut compiled_metrics = QueryMetrics::default();
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor =
            CompiledGpuTaskExecutor::new(&context, &compiled_plan, &pipeline, &gpu_inputs, &mut compiled_metrics)
                .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled root hash join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };
    let production_result = execute_query_plan(&context, &legacy_plan).expect("bounded root join should execute");
    let production_batch = production_result.output().as_host_arrow().expect("production output should be host Arrow");

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(2),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
            PipelineTaskId::new(5),
        ]
    );
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(compiled_batch.num_rows(), production_batch.num_rows());
    assert_eq!(int32_values(&compiled_batch, 0), int32_values(production_batch, 0));
    assert_eq!(int32_values(&compiled_batch, 1), int32_values(production_batch, 1));
    assert_eq!(int32_values(&compiled_batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(&compiled_batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(compiled_metrics.hash_join_build_count, 1);
    assert!(compiled_metrics.hash_join_hash_table_bytes > 0);
}

#[test]
fn compiled_gpu_executor_runs_shared_source_hash_join_without_rereading() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d32-shared-source");
    let path = dir.join("shared.parquet");
    write_self_join_parquet(&path);

    let plan = shared_source_self_join_data_plan(&path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared-source self join should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::SourceRead, 1);
    let shared_resource = pipeline
        .resources()
        .iter()
        .find(|resource| resource.consumers().len() == 2)
        .expect("one source resource should feed both join inputs")
        .id();
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut metrics = QueryMetrics::default();
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();
        let state = scheduler.run(&mut executor).expect("compiled shared-source join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(2),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
        ]
    );
    assert!(state.is_resource_released(shared_resource));
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(compiled_batch.num_rows(), 3);
    assert_eq!(int32_values(&compiled_batch, 0), vec![Some(1), Some(2), Some(4)]);
    assert_eq!(int32_values(&compiled_batch, 1), vec![Some(10), Some(20), Some(40)]);
}

#[test]
fn execute_query_plan_routes_grouped_aggregate_join_build_input_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d33-aggregate-join-input");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_aggregate_join_input_parquets(&left_path, &right_path);

    let plan = aggregate_join_build_input_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan)
        .expect("D.3.3 grouped aggregate join-build input should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_int64_pairs_sorted(batch, 0, 1), vec![(Some(1), Some(15)), (Some(2), Some(20))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
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
fn compiled_bounded_support_accepts_d3_6_nonlinear_topk_and_subquery_filter_paths() {
    for (label, plan, admits, expected_aggregate_barriers) in [
        (
            "aggregate-over-join-subplan TopK",
            aggregate_over_join_subplan_topk_plan(),
            cudf_query_engine::capability::CapabilityAnalysis::admits_multi_join_aggregate_over_join_subplan_topk_pipeline
                as fn(&cudf_query_engine::capability::CapabilityAnalysis) -> bool,
            1,
        ),
        (
            "resident grouped-aggregate TopK",
            resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            cudf_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline
                as fn(&cudf_query_engine::capability::CapabilityAnalysis) -> bool,
            2,
        ),
        (
            "resident grouped-aggregate subquery filter",
            resident_grouped_aggregate_subquery_filter_plan(),
            cudf_query_engine::capability::CapabilityAnalysis::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline
                as fn(&cudf_query_engine::capability::CapabilityAnalysis) -> bool,
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

#[test]
fn compiled_bounded_support_rejects_multiple_explicit_sink_candidates() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d36-multiple-sink-left.parquet"),
        Path::new("/tmp/compiled-d36-multiple-sink-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join TopK plan should compile");
    let malformed = pipeline_with_extra_sink_candidate(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn compiled_bounded_support_rejects_post_root_data_fanout() {
    let plan = root_join_sort_limit_data_plan(
        Path::new("/tmp/compiled-d36-post-root-fanout-left.parquet"),
        Path::new("/tmp/compiled-d36-post-root-fanout-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join TopK plan should compile");
    let malformed = pipeline_with_post_root_fanout(&pipeline);

    assert_eq!(
        compiled_bounded_pipeline_support(&malformed).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported
    );
}

#[test]
fn execute_query_plan_routes_d3_6_nonlinear_topk_and_subquery_filter_paths_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let runtime = Arc::new(runtime);
    write_d3_6_contract_parquets();

    for (label, plan) in [
        ("aggregate-over-join-subplan TopK", aggregate_over_join_subplan_topk_plan()),
        ("resident grouped-aggregate TopK", resident_grouped_aggregate_input_post_join_aggregate_topk_plan()),
        ("resident grouped-aggregate subquery filter", resident_grouped_aggregate_subquery_filter_plan()),
    ] {
        let analysis = plan.capability_analysis();
        let pipeline =
            compile_pipeline(&plan, &analysis).unwrap_or_else(|error| panic!("{label}: compile failed: {error}"));
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: D.3.6 plan should be admitted by compiled support"
        );
        let context = bounded_context(Arc::clone(&runtime));

        let result = execute_query_plan(&context, &plan)
            .unwrap_or_else(|error| panic!("{label}: D.3.6 compiled execution failed: {error}"));
        let batch = result.output().as_host_arrow().expect("D.3.6 output should be host Arrow");
        let metrics = result.metrics();

        assert!(batch.num_rows() > 0, "{label}: compiled execution should produce rows");
        assert_eq!(metrics.compiled_pipeline_attempts, 1, "{label}: expected one compiled attempt");
        assert_eq!(
            metrics.compiled_pipeline_tasks_executed,
            pipeline.tasks().len() as u64,
            "{label}: expected all compiled tasks to execute"
        );
        assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1, "{label}: expected one host Arrow sink");
    }

    remove_d3_6_contract_parquets();
}

#[test]
fn execute_query_plan_routes_root_hash_join_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-root");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("root join should execute through compiled scheduler");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_selected_adaptive_root_hash_join_through_adaptive_shell() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-adaptive-e5-root-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_data_plan(&left_path, &right_path);
    let memory_history = Arc::new(AdaptiveMemoryHistory::new());
    memory_history.record_observation(
        AdaptiveMemoryHistoryKey::for_query_plan(&plan),
        AdaptiveMemoryObservation {
            resident_dataframe_peak_bytes: Some(2048),
            reservation_granted_bytes: 4096,
            source_chunks_committed: 2,
        },
    );
    let context = bounded_context_with_mode_and_memory_history(
        Arc::new(runtime),
        NativePipelineExecutionMode::Adaptive,
        Arc::clone(&memory_history),
    );

    assert_eq!(context.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
    let result = execute_query_plan(&context, &plan).expect("adaptive selected root join should execute");
    let batch = result.output().as_host_arrow().expect("adaptive output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.adaptive_retry_count, 0);
    assert_eq!(metrics.adaptive_spill_count, 0);
    assert_eq!(metrics.adaptive_downgrade_count, 0);
    assert_eq!(metrics.adaptive_memory_history_lookup_count, 1);
    assert_eq!(metrics.adaptive_memory_history_hit_count, 1);
    assert_eq!(metrics.adaptive_memory_history_miss_count, 0);
}

#[test]
fn execute_query_plan_rejects_selected_adaptive_unsupported_join_with_adaptive_labels() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-adaptive-e5-unsupported");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let context = bounded_context_with_mode(Arc::new(runtime), NativePipelineExecutionMode::Adaptive);

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("unsupported adaptive graph should be rejected"),
        Err(error) => error,
    };
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("execution_mode"), Some("unsupported_adaptive"));
    assert_eq!(error.fact_value("candidate_status"), Some("rejected"));
    assert_eq!(error.fact_value("adaptive_strategy"), Some("unsupported"));
    assert_eq!(error.fact_value("runtime_actions_allowed"), Some("none"));
    assert_eq!(error.fact_value("retry_count"), Some("0"));
    assert_eq!(error.fact_value("spill_count"), Some("0"));
    assert_eq!(error.fact_value("downgrade_count"), Some("0"));
    assert_eq!(error.fact_value("rejection_reason"), Some("unsupported_sink_boundary"));
    assert_eq!(error.fact_value("strict_bounded_contract_report"), Some("not_emitted"));
    assert!(rendered.contains("execution_mode=unsupported_adaptive"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("execution_mode=unsupported_strict_bounded"));
}

#[test]
fn execute_query_plan_routes_keyless_root_join_strategy_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d34-keyless-root");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = keyless_root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_join());
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless root join should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.3.4 keyless root join should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 8);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_keyless_conditional_join")
    );
}

#[test]
fn execute_query_plan_routes_root_hash_join_with_unary_inputs_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-unary-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_with_unary_inputs_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join with unary inputs should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("root join with unary inputs should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_leaves_unsupported_compiled_graphs_on_legacy_fallback() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-fallback");
    let path = dir.join("input.parquet");
    let source_schema = write_source_unary_parquet(&path);
    let plan = aggregate_sort_limit_data_plan(&path, source_schema);
    let context = bounded_context(Arc::new(runtime));

    let result =
        execute_query_plan(&context, &plan).expect("unsupported D.1 compiled graph should use legacy fallback");
    let batch = result.output().as_host_arrow().expect("fallback output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 4);
    assert_eq!(metrics.compiled_pipeline_attempts, 0);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 0);
}

#[test]
fn execute_query_plan_rejects_admitted_join_when_compiled_runtime_cannot_own_sink() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d37-required");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_join());
    let context = bounded_context(Arc::new(runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("capability-admitted bounded joins must not fall back to legacy path dispatch"),
        Err(error) => error,
    };
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("compiled_pipeline_required"), Some("true"));
    assert_eq!(error.fact_value("compiled_pipeline_rejection_reason"), Some("unsupported_sink_boundary"));
    assert_eq!(error.fact_value("legacy_bounded_join_path_bridge"), Some("disabled"));
    assert_eq!(error.fact_value("result_sink_kind"), Some("gpu_resident"));
    assert_no_legacy_execution_selection_terms("D.3.7 compiled-required rejection", &rendered);
}

#[test]
fn execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d23-aggregate");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join aggregate plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 0);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result =
        execute_query_plan(&context, &plan).expect("D.2.3 aggregate graph should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_int64_pairs_sorted(batch, 0, 1), vec![(Some(1), Some(2)), (Some(2), Some(1)), (Some(3), Some(0))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_root_hash_join_final_sort_limit_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d24-sort-limit");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_sort_limit_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join final-sort/limit plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let final_sort_gate =
        runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::FinalSort, ReservationKind::FinalSort)
            .expect("compiled final-sort task should retain its modeled reservation gate");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2.4 final-sort/limit graph should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_values(batch, 0), vec![Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_single_join_post_root_d2_graph_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d22-single-join");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_sort_limit_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2 graph should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(batch, 0), vec![Some(3), Some(2)]);
    assert_eq!(int64_values(batch, 1), vec![Some(0), Some(1)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_simple_multi_join_tree_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d26-multi-join");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_multi_join_parquets(&a_path, &b_path, &c_path);
    let plan = simple_multi_join_plan(&a_path, &b_path, &c_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_closed_multi_join_subgraph());
    let pipeline = compile_pipeline(&plan, &analysis).expect("simple multi-join plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2.6 multi-join graph should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(
        int32_quads_sorted(batch, 0, 1, 2, 3),
        vec![(Some(2), Some(20), Some(200), Some(2000)), (Some(3), Some(30), Some(300), Some(3000)),]
    );
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 2);
}

#[test]
fn compiled_host_sink_diagnostics_are_structured_without_legacy_routing_terms() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(Arc::new(runtime));
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();
    let mut metrics = QueryMetrics::default();
    let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();

    let error = executor.take_host_output().expect_err("empty host sink should be a structured invariant error");
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t5"));
    assert_eq!(error.facts().get("sink_boundary_kind"), Some("host_arrow_materialization"));
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("bounded_join_execution_path"));
}

struct D31InventoryCase {
    label: &'static str,
    plan: QueryPlan,
    admits: fn(&cudf_query_engine::capability::CapabilityAnalysis) -> bool,
    graph_support: CompiledBoundedPipelineSupport,
    plan_support: CompiledBoundedPipelineSupport,
    render_fragments: &'static [&'static str],
}

fn d3_1_inventory_cases() -> Vec<D31InventoryCase> {
    vec![
        D31InventoryCase {
            label: "shared resident producer fan-out",
            plan: keyed_hash_join_resident_subplan_pipeline_plan(
                KeyedHashJoinResidentSubplanPipelineVariant::BlockedFamily,
            ),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_keyed_hash_join_resident_subplan_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=shared_resident_producer",
                "kind=aggregate_boundary",
                "kind=join_build_before_probe",
                "consumers=t",
                "gates=g",
            ],
        },
        D31InventoryCase {
            label: "keyless conditional join input-subplan pipeline",
            plan: keyless_conditional_join_input_subplan_plan(KeylessInputSubplanVariant::Admitted),
            admits:
                cudf_query_engine::capability::CapabilityAnalysis::admits_keyless_conditional_join_input_subplan_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=unary",
                "kind=join_build_before_probe",
                "kind=aggregate_boundary",
                "kind=sink_boundary",
            ],
        },
        D31InventoryCase {
            label: "keyless aggregate-over-join subplan inputs",
            plan: keyless_conditional_aggregate_over_join_subplan_inputs_plan(
                AggregateOverJoinSubplanInputsVariant::Admitted,
            ),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=join_materialization",
                "kind=aggregate_boundary",
                "kind=join_build_before_probe",
            ],
        },
        D31InventoryCase {
            label: "resident grouped aggregate input",
            plan: resident_grouped_aggregate_input_post_join_aggregate_topk_plan(),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=final_sort",
                "kind=limit",
                "kind=aggregate_boundary",
                "kind=sort_boundary",
            ],
        },
        D31InventoryCase {
            label: "mixed keyed-residual grouped aggregate input",
            plan: residual_resident_grouped_aggregate_input_plan(
                ResidualResidentGroupedAggregateInputVariant::BlockedFamily,
            ),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=unary",
                "kind=join_build_input",
                "kind=join_probe_input",
                "kind=aggregate_boundary",
            ],
        },
        D31InventoryCase {
            label: "filtered left-semi join strategy",
            plan: root_filtered_left_only_join_plan(JoinType::LeftSemi),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_root_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "kind=join_build_before_probe",
                "kind=join_materialization",
            ],
        },
        D31InventoryCase {
            label: "filtered left-anti join strategy",
            plan: root_filtered_left_only_join_plan(JoinType::LeftAnti),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_root_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "kind=join_build_before_probe",
                "kind=join_materialization",
            ],
        },
        D31InventoryCase {
            label: "mixed residual multi-join strategy",
            plan: admitted_mixed_residual_multi_join_plan(),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_closed_multi_join_subgraph,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "kind=join_build_before_probe",
                "kind=join_materialization",
            ],
        },
        D31InventoryCase {
            label: "large keyed one-shot equality join strategy",
            plan: large_keyed_join_strategy_plan(JoinType::Inner),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_root_join,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=join_build",
                "kind=join_probe",
                "kind=join_materialize",
                "contract=one_shot_equality_join_output_size_preflight",
                "quality=hard_budgeted_report_with_unresolved_preflight",
            ],
        },
        D31InventoryCase {
            label: "broad multi-join TopK with aggregate-over-join input",
            plan: aggregate_over_join_subplan_topk_plan(),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_multi_join_aggregate_over_join_subplan_topk_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=final_sort",
                "kind=limit",
                "kind=aggregate_boundary",
                "kind=sort_boundary",
            ],
        },
        D31InventoryCase {
            label: "resident grouped aggregate subquery-filter TopK path",
            plan: resident_grouped_aggregate_subquery_filter_plan(),
            admits: cudf_query_engine::capability::CapabilityAnalysis::admits_multi_join_resident_grouped_aggregate_subquery_filter_pipeline,
            graph_support: CompiledBoundedPipelineSupport::Supported,
            plan_support: CompiledBoundedPipelineSupport::Supported,
            render_fragments: &[
                "kind=aggregate",
                "kind=final_sort",
                "kind=join_materialization",
                "kind=aggregate_boundary",
                "kind=sort_boundary",
            ],
        },
    ]
}

fn assert_compiled_graph_inventory(label: &str, pipeline: &CompiledPipeline) {
    assert!(!pipeline.tasks().is_empty(), "{label}: compiled graph should carry explicit tasks");
    assert!(!pipeline.edges().is_empty(), "{label}: compiled graph should carry explicit task edges");
    assert!(!pipeline.barriers().is_empty(), "{label}: compiled graph should carry explicit barriers");
    assert!(!pipeline.resources().is_empty(), "{label}: compiled graph should carry explicit resident resources");
    assert!(!pipeline.gates().is_empty(), "{label}: compiled graph should carry explicit reservation gates");
    assert_eq!(pipeline.sink().kind(), SinkBoundaryKind::HostArrowMaterialization, "{label}: expected host sink");

    let rendered = pipeline.debug_render();
    for expected_section in ["tasks:", "edges:", "barriers:", "resources:", "gates:", "sink:"] {
        assert!(rendered.contains(expected_section), "{label}: graph rendering should expose {expected_section}");
    }
}

fn assert_no_legacy_execution_selection_terms(label: &str, rendered: &str) {
    for forbidden in D5_FORBIDDEN_FAMILY_PATH_REPORT_TERMS {
        assert!(
            !rendered.contains(forbidden),
            "{label}: compiled graph diagnostics must not contain legacy execution selection term {forbidden}"
        );
    }
}

fn assert_d5_contract_gate_row_is_family_path_free(label: &str, row: &str) {
    for forbidden in D5_FORBIDDEN_FAMILY_PATH_REPORT_TERMS {
        assert!(
            !row.contains(forbidden),
            "{label}: D.5 strict bounded contract gate rows must not contain legacy family/path report term {forbidden}"
        );
    }
}

fn assert_d5_contract_gate_row_excludes_adaptive_terms(label: &str, row: &str) {
    for forbidden in D5_FORBIDDEN_ADAPTIVE_REPORT_TERMS {
        assert!(
            !row.contains(forbidden),
            "{label}: D.5 strict bounded contract gate rows must not contain adaptive-only term {forbidden}"
        );
    }
}

const D5_FORBIDDEN_FAMILY_PATH_REPORT_TERMS: &[&str] = &[
    "BoundedJoinExecutionFamily",
    "BoundedJoinExecutionPath",
    "bounded_join_execution_family",
    "bounded_join_execution_families",
    "bounded_join_execution_family_contracts",
    "bounded_join_execution_path",
    "bounded_join_blocked_join_family",
    "bounded_join_required_shape_contract",
    "bounded_join_contract_modes",
    "bounded_join_unresolved_preflight_contracts",
    "root_reusable_hash_join",
    "root_filtered_semi_anti_join",
    "keyless_conditional_join",
    "keyless_conditional_join_input_subplan_pipeline",
    "keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline",
    "root_projected_cross_join",
    "root_mixed_keyed_residual_join",
    "multi_join_reusable_hash_pipeline",
    "multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline",
    "multi_join_with_aggregate_over_join_subplan_topk",
    "mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline",
    "hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk",
    "multi_join_with_resident_grouped_aggregate_subquery_filter",
    "keyed_hash_join_resident_subplan_pipeline",
];

const D5_FORBIDDEN_ADAPTIVE_REPORT_TERMS: &[&str] =
    &["retry", "spill", "downgrade", "memory_history", "memory-history", "memory history"];

struct MockExecutor {
    executed: Vec<PipelineTaskId>,
    released: Vec<ResidentResourceId>,
    omit_outputs: Vec<PipelineTaskId>,
    duplicate_outputs: Vec<PipelineTaskId>,
}

impl MockExecutor {
    fn new() -> Self {
        Self { executed: Vec::new(), released: Vec::new(), omit_outputs: Vec::new(), duplicate_outputs: Vec::new() }
    }

    fn omit_output_for(mut self, task: PipelineTaskId) -> Self {
        self.omit_outputs.push(task);
        self
    }

    fn duplicate_output_for(mut self, task: PipelineTaskId) -> Self {
        self.duplicate_outputs.push(task);
        self
    }
}

impl CompiledTaskExecutor for MockExecutor {
    type Resource = String;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        _state: &CompiledPipelineState<Self::Resource>,
    ) -> cudf_query_engine::Result<CompiledTaskOutput<Self::Resource>> {
        self.executed.push(task.id());
        if self.omit_outputs.contains(&task.id()) {
            return Ok(CompiledTaskOutput::empty());
        }
        let mut outputs = task
            .output_resources()
            .iter()
            .map(|resource| (*resource, format!("{}:{resource}", task.id())))
            .collect::<Vec<_>>();
        if self.duplicate_outputs.contains(&task.id())
            && let Some((resource, payload)) = outputs.first().cloned()
        {
            outputs.push((resource, payload));
        }
        Ok(CompiledTaskOutput::new(outputs))
    }

    fn release_resource(
        &mut self,
        resource: ResidentResourceId,
        _payload: Self::Resource,
    ) -> cudf_query_engine::Result<()> {
        self.released.push(resource);
        Ok(())
    }
}

fn root_join_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn keyless_root_join_plan() -> QueryPlan {
    keyless_root_join_data_plan(
        Path::new("/tmp/compiled-d1-keyless-left.parquet"),
        Path::new("/tmp/compiled-d1-keyless-right.parquet"),
    )
}

fn keyless_root_join_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "left_id", "left_id"),
                BinaryOp::Lt,
                Expr::input_column(1, "right_id", "right_id"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
            ])
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn aggregate_join_input_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d24-aggregate-join-input-left.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d24-aggregate-join-input-right.parquet",
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let aggregate = builder.add_aggregate(
        left,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let join = builder.add_join(
        aggregate,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_amount", "sum_amount"), "sum_amount"),
        ])
        .with_build_side(JoinSide::Left),
        join_schema,
    );
    builder.finish_with_host_sink(join)
}

fn aggregate_join_build_input_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let aggregate = builder.add_aggregate(
        left,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let join = builder.add_join(
        aggregate,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_amount", "sum_amount"), "sum_amount"),
        ])
        .with_build_side(JoinSide::Left),
        join_schema,
    );
    builder.finish_with_host_sink(join)
}

fn root_filtered_left_only_join_plan(join_type: JoinType) -> QueryPlan {
    assert!(matches!(join_type, JoinType::LeftSemi | JoinType::LeftAnti));
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d31-filtered-left.parquet",
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d31-filtered-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            join_type,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_value", "left_value"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_value", "right_value"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "left_value", "left_value"), "left_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn root_join_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    root_join_data_plan_with_sink(left_path, right_path, false)
}

fn root_join_gpu_sink_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    root_join_data_plan_with_sink(left_path, right_path, true)
}

fn root_join_data_plan_with_sink(left_path: &Path, right_path: &Path, gpu_resident_sink: bool) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    if gpu_resident_sink { builder.finish_gpu_resident(join) } else { builder.finish_with_host_sink(join) }
}

fn shared_source_self_join_data_plan(path: &Path) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        source_schema,
        vec!["id".into(), "value".into()],
    )));
    let join = builder.add_join(
        source,
        source,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::right(ColumnRef::input(1, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn root_join_sort_limit_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("right_value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), output_schema);
    builder.finish_with_host_sink(limit)
}

fn simple_multi_join_plan(a_path: &Path, b_path: &Path, c_path: &Path) -> QueryPlan {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
    ]));
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("b_id", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, false),
    ]));
    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, true),
        Field::new("c_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        a_path,
        a_schema,
        vec!["a_id".into(), "a_value".into()],
    )));
    let b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        b_path,
        b_schema,
        vec!["b_id".into(), "b_value".into()],
    )));
    let c = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        c_path,
        c_schema,
        vec!["c_id".into(), "c_value".into()],
    )));
    let ab = builder.add_join(
        a,
        b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_id", "a_id"), ColumnRef::input(1, "b_id", "b_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_id", "a_id"), "a_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_value", "a_value"), "a_value"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_value", "b_value"), "b_value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&first_join_schema),
    );
    let abc = builder.add_join(
        ab,
        c,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_id", "a_id"), ColumnRef::input(1, "c_id", "c_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_id", "a_id"), "a_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_value", "a_value"), "a_value"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_value", "b_value"), "b_value"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_value", "c_value"), "c_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(abc)
}

fn pipeline_without_reservation_gates(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let tasks = pipeline
        .tasks()
        .iter()
        .map(|task| {
            let base = match task.plan_node() {
                Some(node_id) => PipelineTask::for_plan_node(task.id(), task.kind(), node_id),
                None => PipelineTask::new(task.id(), task.kind()),
            };
            base.with_input_resources(task.input_resources().to_vec())
                .with_output_resources(task.output_resources().to_vec())
                .with_gates(vec![])
        })
        .collect::<Vec<_>>();
    let edges = pipeline
        .edges()
        .iter()
        .map(|edge| {
            PipelineEdge::new(edge.id(), edge.from(), edge.to(), edge.kind())
                .with_resources(edge.resources().to_vec())
                .with_gates(vec![])
        })
        .collect::<Vec<_>>();
    let barriers = pipeline
        .barriers()
        .iter()
        .map(|barrier| {
            PipelineBarrier::new(barrier.id(), barrier.kind())
                .with_upstream_tasks(barrier.upstream_tasks().to_vec())
                .with_downstream_tasks(barrier.downstream_tasks().to_vec())
                .with_gates(vec![])
        })
        .collect::<Vec<_>>();
    let resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            ResidentResource::new(
                resource.id(),
                resource.producer(),
                resource.consumers().to_vec(),
                resource.lifetime(),
            )
        })
        .collect::<Vec<_>>();
    let sink = SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(vec![]);
    CompiledPipeline::new(tasks, edges, barriers, resources, vec![], sink)
}

fn pipeline_with_extra_sink_candidate(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let mut tasks = pipeline.tasks().to_vec();
    let mut edges = pipeline.edges().to_vec();
    let sink_edge = pipeline
        .edges()
        .iter()
        .find(|edge| edge.kind() == PipelineEdgeKind::SinkInput)
        .expect("fixture should have a sink input edge");
    let [sink_resource] = sink_edge.resources() else {
        panic!("sink input edge should carry one resource");
    };
    let extra_sink = PipelineTaskId::new(tasks.len());
    tasks.push(
        PipelineTask::new(extra_sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![*sink_resource]),
    );
    edges.push(
        PipelineEdge::new(PipelineEdgeId::new(edges.len()), sink_edge.from(), extra_sink, PipelineEdgeKind::SinkInput)
            .with_resources(vec![*sink_resource]),
    );

    let resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let mut consumers = resource.consumers().to_vec();
            if resource.id() == *sink_resource {
                consumers.push(extra_sink);
                consumers.sort_unstable();
                consumers.dedup();
            }
            ResidentResource::new(resource.id(), resource.producer(), consumers, resource.lifetime())
        })
        .collect::<Vec<_>>();
    let sink =
        SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(pipeline.sink().gates().to_vec());
    CompiledPipeline::new(tasks, edges, pipeline.barriers().to_vec(), resources, pipeline.gates().to_vec(), sink)
}

fn pipeline_with_post_root_fanout(pipeline: &CompiledPipeline) -> CompiledPipeline {
    let mut tasks = pipeline.tasks().to_vec();
    let mut edges = pipeline.edges().to_vec();
    let fanout_edge = pipeline
        .edges()
        .iter()
        .find(|edge| {
            edge.kind() == PipelineEdgeKind::ResidentFrame
                && pipeline
                    .tasks()
                    .iter()
                    .find(|task| task.id() == edge.from())
                    .is_some_and(|task| task.kind() == PipelineTaskKind::JoinMaterialize)
        })
        .expect("fixture should have a post-root resident-frame edge");
    let [fanout_resource] = fanout_edge.resources() else {
        panic!("post-root resident-frame edge should carry one resource");
    };
    let extra_task = PipelineTaskId::new(tasks.len());
    let extra_resource = ResidentResourceId::new(pipeline.resources().len());
    tasks.push(
        PipelineTask::new(extra_task, PipelineTaskKind::Unary)
            .with_input_resources(vec![*fanout_resource])
            .with_output_resources(vec![extra_resource]),
    );
    edges.push(
        PipelineEdge::new(
            PipelineEdgeId::new(edges.len()),
            fanout_edge.from(),
            extra_task,
            PipelineEdgeKind::ResidentFrame,
        )
        .with_resources(vec![*fanout_resource]),
    );

    let mut resources = pipeline
        .resources()
        .iter()
        .map(|resource| {
            let mut consumers = resource.consumers().to_vec();
            if resource.id() == *fanout_resource {
                consumers.push(extra_task);
                consumers.sort_unstable();
                consumers.dedup();
            }
            ResidentResource::new(resource.id(), resource.producer(), consumers, resource.lifetime())
        })
        .collect::<Vec<_>>();
    resources.push(ResidentResource::new(
        extra_resource,
        extra_task,
        vec![],
        ResidentLifetime::TaskLocal { task: extra_task },
    ));
    let sink =
        SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(pipeline.sink().gates().to_vec());
    CompiledPipeline::new(tasks, edges, pipeline.barriers().to_vec(), resources, pipeline.gates().to_vec(), sink)
}

fn runtime_visible_gate_for_task_kind(
    pipeline: &CompiledPipeline,
    task_kind: PipelineTaskKind,
    reservation_kind: ReservationKind,
) -> Option<&ReservationGate> {
    let task = pipeline.tasks().iter().find(|task| task.kind() == task_kind)?;
    task.gates()
        .iter()
        .chain(
            pipeline
                .edges()
                .iter()
                .filter(|edge| edge.from() == task.id() || edge.to() == task.id())
                .flat_map(|edge| edge.gates()),
        )
        .chain(
            pipeline
                .barriers()
                .iter()
                .filter(|barrier| {
                    barrier.upstream_tasks().contains(&task.id()) || barrier.downstream_tasks().contains(&task.id())
                })
                .flat_map(|barrier| barrier.gates()),
        )
        .chain((pipeline.sink().input() == task.id()).then_some(pipeline.sink().gates()).into_iter().flatten())
        .find_map(|gate_id| {
            pipeline.gates().iter().find(|gate| gate.id() == *gate_id && gate.reservation_kind() == reservation_kind)
        })
}

fn root_join_with_unary_inputs_data_plan(left_path: &Path, right_path: &Path) -> QueryPlan {
    let left_source_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_source_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        left_path,
        Arc::clone(&left_source_schema),
        vec!["left_id".into(), "left_value".into()],
    )));
    let left_filter = builder.add_filter(
        left,
        Expr::binary(Expr::column("left_id"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(1))),
        Arc::clone(&left_source_schema),
    );
    let left_projected = builder.add_projection(
        left_filter,
        vec![
            ProjectionOutput::new(Expr::column("left_id"), Field::new("left_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("left_value"), Field::new("left_value", DataType::Int32, false)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        right_path,
        Arc::clone(&right_source_schema),
        vec!["right_id".into(), "right_value".into()],
    )));
    let right_filter = builder.add_filter(
        right,
        Expr::binary(Expr::column("right_id"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(5))),
        Arc::clone(&right_source_schema),
    );
    let right_projected = builder.add_projection(
        right_filter,
        vec![
            ProjectionOutput::new(Expr::column("right_id"), Field::new("right_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("right_value"), Field::new("right_value", DataType::Int32, false)),
        ],
    );
    let join = builder.add_join(
        left_projected,
        right_projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn single_join_post_root_aggregate_data_plan(customers_path: &Path, orders_path: &Path) -> QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        customers_path,
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        orders_path,
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

fn single_join_post_root_aggregate_sort_limit_data_plan(customers_path: &Path, orders_path: &Path) -> QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        customers_path,
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        orders_path,
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        join,
        vec![
            ProjectionOutput::new(Expr::column("c_custkey"), Field::new("c_custkey", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("o_orderkey"), Field::new("o_orderkey", DataType::Int32, true)),
        ],
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("c_count"), false, false),
            SortExpr::new(Expr::column("c_custkey"), true, false),
        ]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

fn simple_unary_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let unary = PipelineTaskId::new(1);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(unary, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, unary, PipelineEdgeKind::ResidentFrame)
                .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![unary]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![unary]),
        ],
        vec![ResidentResource::new(frame, source, vec![unary], ResidentLifetime::BetweenTasks { producer: source })],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, unary),
    )
}

fn out_of_order_join_barrier_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let probe = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let materialize = PipelineTaskId::new(3);
    let sink = PipelineTaskId::new(4);
    let source_frame = ResidentResourceId::new(0);
    let join_output = ResidentResourceId::new(1);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![source_frame]),
            PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![source_frame]),
            PipelineTask::new(build, PipelineTaskKind::JoinBuild).with_input_resources(vec![source_frame]),
            PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
            PipelineTask::new(sink, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, probe, PipelineEdgeKind::JoinProbeInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(1), source, build, PipelineEdgeKind::JoinBuildInput)
                .with_resources(vec![source_frame]),
            PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
            PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink, PipelineEdgeKind::SinkInput)
                .with_resources(vec![join_output]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![build, probe]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::JoinBuildBeforeProbe)
                .with_upstream_tasks(vec![build])
                .with_downstream_tasks(vec![probe]),
            PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinMaterialization)
                .with_upstream_tasks(vec![probe])
                .with_downstream_tasks(vec![materialize]),
            PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![sink]),
        ],
        vec![
            ResidentResource::new(
                source_frame,
                source,
                vec![build, probe],
                ResidentLifetime::BetweenTasks { producer: source },
            ),
            ResidentResource::new(
                join_output,
                materialize,
                vec![sink],
                ResidentLifetime::BetweenTasks { producer: materialize },
            ),
        ],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink),
    )
}

fn shared_resident_fanout_pipeline() -> CompiledPipeline {
    let source = PipelineTaskId::new(0);
    let first_consumer = PipelineTaskId::new(1);
    let second_consumer = PipelineTaskId::new(2);
    let frame = ResidentResourceId::new(0);

    CompiledPipeline::new(
        vec![
            PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
            PipelineTask::new(first_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
            PipelineTask::new(second_consumer, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
        ],
        vec![
            PipelineEdge::new(PipelineEdgeId::new(0), source, first_consumer, PipelineEdgeKind::SharedResidentProducer)
                .with_resources(vec![frame]),
            PipelineEdge::new(
                PipelineEdgeId::new(1),
                source,
                second_consumer,
                PipelineEdgeKind::SharedResidentProducer,
            )
            .with_resources(vec![frame]),
        ],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                .with_upstream_tasks(vec![source])
                .with_downstream_tasks(vec![first_consumer, second_consumer]),
            PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(vec![second_consumer]),
        ],
        vec![ResidentResource::new(
            frame,
            source,
            vec![first_consumer, second_consumer],
            ResidentLifetime::BetweenTasks { producer: source },
        )],
        vec![],
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, second_consumer),
    )
}

fn aggregate_sort_limit_plan() -> QueryPlan {
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-d1-aggregate.parquet",
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("v_count"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

fn adaptive_memory_history_key_probe_plan(
    value_type: DataType,
    aggregate_function: AggregateFunction,
    predicate_op: BinaryOp,
) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("adaptive_key", DataType::Int32, false),
        Field::new("adaptive_value", value_type.clone(), true),
    ]));
    let aggregate_value_type = match aggregate_function {
        AggregateFunction::Count => DataType::Int64,
        AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max | AggregateFunction::Avg => {
            value_type.clone()
        },
    };
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("adaptive_key", DataType::Int32, false),
        Field::new("adaptive_metric", aggregate_value_type, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/adaptive-key-secret-literal.parquet",
        source_schema,
        vec!["adaptive_key".into(), "adaptive_value".into()],
    )));
    let filter = builder.add_filter(
        source,
        Expr::binary(Expr::column("adaptive_value"), predicate_op, literal_expr_for_type(&value_type)),
        Arc::new(Schema::new(vec![
            Field::new("adaptive_key", DataType::Int32, false),
            Field::new("adaptive_value", value_type.clone(), true),
        ])),
    );
    let aggregate = builder.add_aggregate(
        filter,
        AggregateSpec::new(
            vec![Expr::column("adaptive_key")],
            vec![AggregateExpr::new(aggregate_function, Expr::column("adaptive_value"), "adaptive_metric")],
        ),
        Arc::clone(&aggregate_schema),
    );
    builder.finish_with_host_sink(aggregate)
}

fn literal_expr_for_type(data_type: &DataType) -> Expr {
    let literal = match data_type {
        DataType::Decimal128(precision, scale) => {
            LiteralValue::Decimal128 { raw: 12_345, precision: *precision, scale: *scale }
        },
        _ => LiteralValue::Int32(12_345),
    };
    Expr::literal(literal).alias("secret_literal")
}

fn aggregate_sort_limit_data_plan(path: &Path, source_schema: Arc<Schema>) -> QueryPlan {
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        path,
        source_schema,
        vec!["k".into(), "v".into()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("k"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    builder.finish_with_host_sink(limit)
}

fn root_join_shaped_pipeline_with_blocked_gate() -> CompiledPipeline {
    let left_source = PipelineTaskId::new(0);
    let right_source = PipelineTaskId::new(1);
    let build = PipelineTaskId::new(2);
    let probe = PipelineTaskId::new(3);
    let materialize = PipelineTaskId::new(4);
    let sink_task = PipelineTaskId::new(5);

    let left_frame = ResidentResourceId::new(0);
    let right_frame = ResidentResourceId::new(1);
    let join_output = ResidentResourceId::new(2);
    let blocked_gate = ReservationGateId::new(0);

    let tasks = vec![
        PipelineTask::new(left_source, PipelineTaskKind::SourceRead).with_output_resources(vec![left_frame]),
        PipelineTask::new(right_source, PipelineTaskKind::SourceRead).with_output_resources(vec![right_frame]),
        PipelineTask::new(build, PipelineTaskKind::JoinBuild)
            .with_input_resources(vec![right_frame])
            .with_gates(vec![blocked_gate]),
        PipelineTask::new(probe, PipelineTaskKind::JoinProbe).with_input_resources(vec![left_frame]),
        PipelineTask::new(materialize, PipelineTaskKind::JoinMaterialize).with_output_resources(vec![join_output]),
        PipelineTask::new(sink_task, PipelineTaskKind::SinkMaterialize).with_input_resources(vec![join_output]),
    ];

    let edges = vec![
        PipelineEdge::new(PipelineEdgeId::new(0), right_source, build, PipelineEdgeKind::JoinBuildInput)
            .with_resources(vec![right_frame]),
        PipelineEdge::new(PipelineEdgeId::new(1), left_source, probe, PipelineEdgeKind::JoinProbeInput)
            .with_resources(vec![left_frame]),
        PipelineEdge::new(PipelineEdgeId::new(2), build, probe, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(3), probe, materialize, PipelineEdgeKind::Control),
        PipelineEdge::new(PipelineEdgeId::new(4), materialize, sink_task, PipelineEdgeKind::SinkInput)
            .with_resources(vec![join_output]),
    ];

    let barriers = vec![
        PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
            .with_upstream_tasks(vec![left_source])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
            .with_upstream_tasks(vec![right_source])
            .with_downstream_tasks(vec![build]),
        PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::JoinBuildBeforeProbe)
            .with_upstream_tasks(vec![build])
            .with_downstream_tasks(vec![probe]),
        PipelineBarrier::new(PipelineBarrierId::new(3), PipelineBarrierKind::JoinMaterialization)
            .with_upstream_tasks(vec![probe])
            .with_downstream_tasks(vec![materialize]),
        PipelineBarrier::new(PipelineBarrierId::new(4), PipelineBarrierKind::SinkBoundary)
            .with_upstream_tasks(vec![sink_task]),
    ];

    let resources = vec![
        ResidentResource::new(
            left_frame,
            left_source,
            vec![probe],
            ResidentLifetime::BetweenTasks { producer: left_source },
        ),
        ResidentResource::new(
            right_frame,
            right_source,
            vec![build],
            ResidentLifetime::BetweenTasks { producer: right_source },
        ),
        ResidentResource::new(
            join_output,
            materialize,
            vec![sink_task],
            ResidentLifetime::BetweenTasks { producer: materialize },
        ),
    ];

    let gates = vec![ReservationGate::new(
        blocked_gate,
        ReservationGateOwner::Task(build),
        ReservationKind::HashTable,
        ReservationRequirementScope::Node,
        "join_hash_table",
        "hash_join_pre_build_reservation_size_bytes",
        ContractStatus::BlockedByLowerLevel,
        ContractQuality::ExactPreflight,
    )];

    CompiledPipeline::new(
        tasks,
        edges,
        barriers,
        resources,
        gates,
        SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, sink_task),
    )
}

fn write_source_unary_parquet(path: &Path) -> Arc<Schema> {
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_join_parquets(left_path: &Path, right_path: &Path) {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    write_parquet(left_path, Arc::clone(&left_schema), &left_batch);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![2, 4, 5])), Arc::new(Int32Array::from(vec![200, 400, 500]))],
    )
    .unwrap();
    write_parquet(right_path, Arc::clone(&right_schema), &right_batch);
}

fn write_aggregate_join_input_parquets(left_path: &Path, right_path: &Path) {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 3])), Arc::new(Int64Array::from(vec![10, 5, 20, 30]))],
    )
    .unwrap();
    write_parquet(left_path, Arc::clone(&left_schema), &left_batch);

    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 4]))]).unwrap();
    write_parquet(right_path, Arc::clone(&right_schema), &right_batch);
}

fn write_self_join_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 4])), Arc::new(Int32Array::from(vec![10, 20, 40]))],
    )
    .unwrap();
    write_parquet(path, Arc::clone(&schema), &batch);
}

fn write_multi_join_parquets(a_path: &Path, b_path: &Path, c_path: &Path) {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
    ]));
    let a_batch = RecordBatch::try_new(
        Arc::clone(&a_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    write_parquet(a_path, Arc::clone(&a_schema), &a_batch);

    let b_schema = Arc::new(Schema::new(vec![
        Field::new("b_id", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, false),
    ]));
    let b_batch = RecordBatch::try_new(
        Arc::clone(&b_schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(Int32Array::from(vec![200, 300, 400]))],
    )
    .unwrap();
    write_parquet(b_path, Arc::clone(&b_schema), &b_batch);

    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let c_batch = RecordBatch::try_new(
        Arc::clone(&c_schema),
        vec![Arc::new(Int32Array::from(vec![3, 2, 5])), Arc::new(Int32Array::from(vec![3000, 2000, 5000]))],
    )
    .unwrap();
    write_parquet(c_path, Arc::clone(&c_schema), &c_batch);
}

fn write_single_join_post_root_parquets(customers_path: &Path, orders_path: &Path) {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let customer_batch =
        RecordBatch::try_new(Arc::clone(&customer_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    write_parquet(customers_path, Arc::clone(&customer_schema), &customer_batch);

    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let orders_batch = RecordBatch::try_new(
        Arc::clone(&orders_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30])), Arc::new(Int32Array::from(vec![1, 1, 2]))],
    )
    .unwrap();
    write_parquet(orders_path, Arc::clone(&orders_schema), &orders_batch);
}

fn write_d3_6_contract_parquets() {
    remove_d3_6_contract_parquets();
    for label in [
        "aggregate-over-join-topk-a",
        "aggregate-over-join-topk-b",
        "aggregate-over-join-topk-c",
        "aggregate-over-join-topk-d",
        "aggregate-over-join-topk-e",
        "aggregate-over-join-topk-f",
        "resident-aggregate-post-aggregate-topk-a",
        "resident-aggregate-post-aggregate-topk-b",
        "resident-aggregate-post-aggregate-topk-c",
        "resident-aggregate-subquery-filter-a",
        "resident-aggregate-subquery-filter-b",
        "resident-aggregate-subquery-filter-c",
        "resident-aggregate-subquery-filter-d",
    ] {
        write_key_value_parquet(Path::new(&format!("/tmp/{label}.parquet")), &[1, 2, 3], &[10, 20, 30]);
    }
    for label in ["resident-aggregate-post-aggregate-topk-lineitem", "resident-aggregate-subquery-filter-lineitem"] {
        write_key_value_parquet(Path::new(&format!("/tmp/{label}.parquet")), &[1, 2, 2, 3], &[1, 2, 3, 4]);
    }
}

fn remove_d3_6_contract_parquets() {
    for label in [
        "aggregate-over-join-topk-a",
        "aggregate-over-join-topk-b",
        "aggregate-over-join-topk-c",
        "aggregate-over-join-topk-d",
        "aggregate-over-join-topk-e",
        "aggregate-over-join-topk-f",
        "resident-aggregate-post-aggregate-topk-a",
        "resident-aggregate-post-aggregate-topk-b",
        "resident-aggregate-post-aggregate-topk-c",
        "resident-aggregate-post-aggregate-topk-lineitem",
        "resident-aggregate-subquery-filter-a",
        "resident-aggregate-subquery-filter-b",
        "resident-aggregate-subquery-filter-c",
        "resident-aggregate-subquery-filter-d",
        "resident-aggregate-subquery-filter-lineitem",
    ] {
        let _ = fs::remove_file(format!("/tmp/{label}.parquet"));
    }
}

fn write_key_value_parquet(path: &Path, keys: &[i32], values: &[i64]) {
    assert_eq!(keys.len(), values.len());
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys.to_vec())), Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap();
    write_parquet(path, schema, &batch);
}

fn write_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn int32_values(batch: &RecordBatch, column: usize) -> Vec<Option<i32>> {
    let array = batch.column(column).as_any().downcast_ref::<Int32Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int64_values(batch: &RecordBatch, column: usize) -> Vec<Option<i64>> {
    let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int32_int64_pairs_sorted(
    batch: &RecordBatch,
    left_column: usize,
    right_column: usize,
) -> Vec<(Option<i32>, Option<i64>)> {
    let left = int32_values(batch, left_column);
    let right = int64_values(batch, right_column);
    let mut pairs = left.into_iter().zip(right).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

type Int32Quad = (Option<i32>, Option<i32>, Option<i32>, Option<i32>);

fn int32_quads_sorted(
    batch: &RecordBatch,
    first_column: usize,
    second_column: usize,
    third_column: usize,
    fourth_column: usize,
) -> Vec<Int32Quad> {
    let first = int32_values(batch, first_column);
    let second = int32_values(batch, second_column);
    let third = int32_values(batch, third_column);
    let fourth = int32_values(batch, fourth_column);
    let mut quads = first
        .into_iter()
        .zip(second)
        .zip(third)
        .zip(fourth)
        .map(|(((first, second), third), fourth)| (first, second, third, fourth))
        .collect::<Vec<_>>();
    quads.sort();
    quads
}

fn assert_task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind, expected: usize) {
    assert_eq!(task_kind_count(pipeline, kind), expected, "unexpected count for {kind:?} tasks");
}

fn task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> usize {
    pipeline.tasks().iter().filter(|task| task.kind() == kind).count()
}

fn barrier_kind_count(pipeline: &CompiledPipeline, kind: PipelineBarrierKind) -> usize {
    pipeline.barriers().iter().filter(|barrier| barrier.kind() == kind).count()
}

fn sink_input_edge_count(pipeline: &CompiledPipeline) -> usize {
    pipeline.edges().iter().filter(|edge| edge.kind() == PipelineEdgeKind::SinkInput).count()
}

fn bounded_context(runtime: Arc<cudf::Runtime>) -> ExecutionContext {
    bounded_context_with_mode(runtime, NativePipelineExecutionMode::StrictBounded)
}

fn bounded_context_with_mode(runtime: Arc<cudf::Runtime>, mode: NativePipelineExecutionMode) -> ExecutionContext {
    bounded_context_builder(runtime, mode).build().unwrap()
}

fn bounded_context_with_mode_and_memory_history(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
    memory_history: Arc<AdaptiveMemoryHistory>,
) -> ExecutionContext {
    bounded_context_builder(runtime, mode).with_adaptive_memory_history(memory_history).build().unwrap()
}

fn bounded_context_builder(
    runtime: Arc<cudf::Runtime>,
    mode: NativePipelineExecutionMode,
) -> cudf_query_engine::ExecutionContextBuilder {
    ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_native_pipeline_execution_mode(mode)
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
