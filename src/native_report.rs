//! Public report and TSV contracts for native planning/execution diagnostics.
//!
//! The constants and row structs re-exported from this module are consumed by
//! benchmark binaries and external comparison scripts. TSV header order is
//! stable within a schema version; adding, removing, renaming, or reordering a
//! column requires bumping the corresponding `*_TSV_SCHEMA_VERSION`.
//! Sentinel strings such as `none`, `unknown`, `not_run`, and `not_computed`
//! are part of the report contract when emitted by the TSV helpers.

use std::{fmt::Write, sync::Arc};

use datafusion::{
    common::ScalarValue,
    datasource::{
        memory::MemorySourceConfig,
        physical_plan::{FileSource, ParquetSource},
        source::DataSourceExec,
    },
    logical_expr::{Operator, WindowFrame, WindowFrameBound, WindowFrameUnits},
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal,
            NegativeExpr, NotExpr, TryCastExpr,
        },
        window::{PlainAggregateWindowExpr, SlidingAggregateWindowExpr, StandardWindowExpr, WindowExpr},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        empty::EmptyExec,
        filter::FilterExec,
        joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
        placeholder_row::PlaceholderRowExec,
        projection::ProjectionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        windows::{BoundedWindowAggExec, WindowAggExec, WindowUDFExpr},
    },
};
use nexus_query_engine::{
    ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptivePipelineReport, AdaptivePipelineReportRow,
    SourceDiagnostics as NativeSourceDiagnostics, StrictBoundedContractGateRow, is_supported_arrow_type,
};

#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphAlgoExec;
use crate::{
    CandidateOutcome, FallbackReason, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig,
    NativeCudfGpuProducerExec, lower_physical_plan_to_native, planner::CudfNativePhysicalOptimizerRule,
    report::GpuCandidate,
};

mod dataset;
mod diagnostics;
mod metrics;
mod reports;
mod rows;
mod runtime_summary;
mod schema;
mod tpcds_parity;
mod tsv;
mod workload;

pub use dataset::WorkloadDatasetMetadata;
pub(crate) use diagnostics::join_diagnostics as physical_join_diagnostics;
use diagnostics::{
    candidate_shape_family, physical_plan_shape, post_native_plan_coverage, summarize_source_diagnostics,
    unsupported_category,
};
pub use metrics::{
    NativeAggregateStrategyMetricRow, NativeDataFusionExecutionMetrics, NativeDynamicFilterMeasuredReductionStatus,
    NativeDynamicFilterPostDecodeReductionStage, NativeDynamicFilterRuntimeMetricRow,
    NativeDynamicFilterWorkAvoidanceStatus, NativeGraphReservationMetricRow, NativeGraphRuntimeMetricRow,
    NativeJoinStrategyMetricRow, NativeLakehouseCacheFillEventMetricRow, NativeSourceReadRuntimeMetricRow,
    NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow,
};
pub use reports::{NativeDataFusionExecutionReport, NativeDataFusionPlanningReport, NativeFallbackPressureReport};
pub use rows::{
    NativeDataFusionExecutionRow, NativeDataFusionPlanCoverage, NativeDataFusionPlanningOutcome,
    NativeDataFusionPlanningRow, NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome,
    NativeFallbackPressureOutcome, NativeFallbackPressureRow, NativeJoinDiagnosticSummary,
    NativeSourceDiagnosticSummary,
};
use runtime_summary::NativeDataFusionRuntimeSummary;
pub use schema::{
    NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_HASH_JOIN_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_INPUT,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_LOCAL_PARQUET_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_AGGREGATE, NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_INPUT,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_LOCAL_PARQUET_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_INPUT,
    NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_PRESERVING_MERGE_FETCH_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE,
    NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN, NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
    NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION, NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
    NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION, NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
    NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION,
};
pub use tpcds_parity::{
    NATIVE_TPCDS_PARITY_TSV_HEADER, NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, NativeTpcdsParityCpuOutcome,
    NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus, NativeTpcdsParityExternalOutputSummary,
    NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome,
    NativeTpcdsParityReport, NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode,
    ParityComparisonResult, ParityFloatTolerance, ParityStageStatus, ParityStatus, compare_record_batches,
    sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
};
use tsv::{TsvRow, stable_tsv_field, write_workload_dataset_metadata};
pub use workload::{
    NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
    NativeDataFusionPlanningWorkloadInput, NativeDataFusionPlanningWorkloadItem, NativeDataFusionRuntime,
    NativeFallbackPressureWorkloadInput, NativeFallbackPressureWorkloadItem, NativePlanningFailure,
};

/// Format nested graph-runtime metric rows for TSV report cells.
pub fn format_native_graph_runtime_metric_rows(rows: &[NativeGraphRuntimeMetricRow]) -> String {
    tsv::format_graph_runtime_rows(rows)
}

/// Format nested graph-reservation metric rows for TSV report cells.
pub fn format_native_graph_reservation_metric_rows(rows: &[NativeGraphReservationMetricRow]) -> String {
    tsv::format_graph_reservation_rows(rows)
}

/// Format nested task-operator metric rows for TSV report cells.
pub fn format_native_task_operator_metric_rows(rows: &[NativeTaskOperatorMetricRow]) -> String {
    tsv::format_task_operator_rows(rows)
}

/// Format nested task-runtime metric rows for TSV report cells.
pub fn format_native_task_runtime_metric_rows(rows: &[NativeTaskRuntimeMetricRow]) -> String {
    tsv::format_task_runtime_rows(rows)
}

/// Format nested dynamic-filter runtime metric rows for TSV report cells.
pub fn format_native_dynamic_filter_runtime_metric_rows(rows: &[NativeDynamicFilterRuntimeMetricRow]) -> String {
    tsv::format_dynamic_filter_runtime_rows(rows)
}

/// Format nested source-read runtime metric rows for TSV report cells.
pub fn format_native_source_read_runtime_metric_rows(rows: &[NativeSourceReadRuntimeMetricRow]) -> String {
    tsv::format_source_read_runtime_rows(rows)
}

/// Format nested lakehouse-cache fill event rows for TSV report cells.
pub fn format_native_lakehouse_cache_fill_event_metric_rows(rows: &[NativeLakehouseCacheFillEventMetricRow]) -> String {
    tsv::format_lakehouse_cache_fill_event_snapshot_rows(rows)
}

/// Format nested aggregate-strategy metric rows for TSV report cells.
pub fn format_native_aggregate_strategy_metric_rows(rows: &[NativeAggregateStrategyMetricRow]) -> String {
    tsv::format_aggregate_strategy_rows(rows)
}

/// Format nested join-strategy metric rows for TSV report cells.
pub fn format_native_join_strategy_metric_rows(rows: &[NativeJoinStrategyMetricRow]) -> String {
    tsv::format_join_strategy_rows(rows)
}

fn report_rows_for_item(
    item: NativeFallbackPressureWorkloadItem,
    execution_config: &NativeCudfExecutionConfig,
) -> Vec<NativeFallbackPressureRow> {
    match item.input {
        NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) => {
            report_rows_for_plan(item.query_id, plan, execution_config)
        },
        NativeFallbackPressureWorkloadInput::PlanningFailure(failure) => {
            vec![report_row_for_planning_failure(item.query_id, failure)]
        },
    }
}

fn report_rows_for_plan(
    query_id: String,
    plan: Arc<dyn ExecutionPlan>,
    execution_config: &NativeCudfExecutionConfig,
) -> Vec<NativeFallbackPressureRow> {
    let physical_root = plan.name().to_string();
    let physical_shape = physical_plan_shape(plan.as_ref());
    let join_diagnostics = physical_join_diagnostics(plan.as_ref());
    let source_diagnostics_plan = Arc::clone(&plan);
    let result = lower_physical_plan_to_native(plan);
    let source_diagnostics = result.source_diagnostics.iter().map(summarize_source_diagnostics).collect::<Vec<_>>();
    let adaptive_pipeline = fallback_pressure_adaptive_pipeline_rows(&result, execution_config);

    result
        .report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => {
                let adaptive_pipeline_rows = adaptive_pipeline_rows_for_candidate(&adaptive_pipeline, candidate);
                let adaptive_rejected = adaptive_pipeline_rows
                    .iter()
                    .any(|row| row.candidate_status == nexus_query_engine::AdaptiveCandidateStatus::Rejected);
                NativeFallbackPressureRow {
                    query_id: query_id.clone(),
                    physical_root: physical_root.clone(),
                    physical_shape: physical_shape.clone(),
                    native_candidate_shape: Some(candidate.shape.to_string()),
                    native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                    outcome: if adaptive_rejected {
                        NativeFallbackPressureOutcome::Rejected
                    } else {
                        NativeFallbackPressureOutcome::Rewritten
                    },
                    reason_code: adaptive_rejected
                        .then(|| FallbackReason::NativeAdaptivePipelineUnsupported.code().to_string()),
                    unsupported_category: adaptive_rejected
                        .then(|| {
                            unsupported_category(
                                source_diagnostics_plan.as_ref(),
                                candidate,
                                FallbackReason::NativeAdaptivePipelineUnsupported,
                            )
                        })
                        .flatten(),
                    adaptive_pipeline_rows,
                    join_diagnostics: join_diagnostics.clone(),
                    source_diagnostics: source_diagnostics.clone(),
                    rejected_before_execution: adaptive_rejected,
                }
            },
            CandidateOutcome::Rejected { candidate, reason } => NativeFallbackPressureRow {
                query_id: query_id.clone(),
                physical_root: physical_root.clone(),
                physical_shape: physical_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                outcome: NativeFallbackPressureOutcome::Rejected,
                reason_code: Some(reason.code().to_string()),
                unsupported_category: unsupported_category(source_diagnostics_plan.as_ref(), candidate, *reason),
                adaptive_pipeline_rows: Vec::new(),
                join_diagnostics: join_diagnostics.clone(),
                source_diagnostics: source_diagnostics.clone(),
                rejected_before_execution: true,
            },
        })
        .collect()
}

fn report_row_for_planning_failure(query_id: String, failure: NativePlanningFailure) -> NativeFallbackPressureRow {
    NativeFallbackPressureRow {
        query_id,
        physical_root: "PlanningFailure".to_string(),
        physical_shape: failure.stage,
        native_candidate_shape: None,
        native_candidate_shape_family: None,
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: Some(failure.reason_code),
        unsupported_category: Some("planning".to_string()),
        adaptive_pipeline_rows: Vec::new(),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        rejected_before_execution: true,
    }
}

fn fallback_pressure_adaptive_pipeline_rows(
    lowering: &crate::native::NativeLoweringResult,
    execution_config: &NativeCudfExecutionConfig,
) -> crate::planner::NativeAdaptivePipelineDiagnostics {
    if execution_config.native_pipeline_execution_mode() != nexus_query_engine::NativePipelineExecutionMode::Adaptive {
        return crate::planner::NativeAdaptivePipelineDiagnostics::default();
    }
    let Some(plan) = lowering.plans.first() else {
        return crate::planner::NativeAdaptivePipelineDiagnostics::default();
    };
    let Some(candidate) = lowering.report.single_rewritten_candidate().cloned() else {
        return crate::planner::NativeAdaptivePipelineDiagnostics::default();
    };
    let memory_history = execution_config.adaptive_memory_history();
    match nexus_query_engine::adaptive_bounded_pipeline_admission_with_memory_history(plan, memory_history.as_deref()) {
        Ok(Some(admission)) => {
            crate::planner::NativeAdaptivePipelineDiagnostics::single(candidate, admission.report().clone())
        },
        Ok(None) => crate::planner::NativeAdaptivePipelineDiagnostics::default(),
        Err(_) => crate::planner::NativeAdaptivePipelineDiagnostics::single(
            candidate,
            AdaptivePipelineReport::diagnostic_failure(plan, ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON),
        ),
    }
}

fn post_native_plan_edge_normalize_inserted(plan: &dyn ExecutionPlan) -> bool {
    if let Some(exec) = plan.downcast_ref::<NativeCudfExec>()
        && exec.native_plan().contains_edge_normalize()
    {
        return true;
    }
    if let Some(exec) = plan.downcast_ref::<NativeCudfComposedExec>()
        && exec.native_plan().contains_edge_normalize()
    {
        return true;
    }
    if let Some(exec) = plan.downcast_ref::<NativeCudfGpuProducerExec>()
        && exec.native_plan().contains_edge_normalize()
    {
        return true;
    }
    #[cfg(feature = "cugraph")]
    if let Some(exec) = plan.downcast_ref::<CugraphAlgoExec>()
        && exec.edge_normalize_inserted()
    {
        return true;
    }

    plan.children().iter().any(|child| post_native_plan_edge_normalize_inserted(child.as_ref()))
}

fn datafusion_planning_rows_for_item(item: NativeDataFusionPlanningWorkloadItem) -> Vec<NativeDataFusionPlanningRow> {
    let (planned_plan, native_diagnostics) = match item.input {
        NativeDataFusionPlanningWorkloadInput::Planned { planned_plan, native_diagnostics } => {
            (planned_plan, native_diagnostics)
        },
        NativeDataFusionPlanningWorkloadInput::PlanningFailure(failure) => {
            return vec![datafusion_planning_row_for_planning_failure(item.query_id, failure)];
        },
    };
    let post_native_plan_root = planned_plan.name().to_string();
    let post_native_coverage = post_native_plan_coverage(planned_plan.as_ref());
    let edge_normalize_inserted = post_native_plan_edge_normalize_inserted(planned_plan.as_ref());
    let post_native_join_diagnostics = physical_join_diagnostics(planned_plan.as_ref());
    let input_join_diagnostics = if native_diagnostics.input_join_diagnostics.is_empty() {
        post_native_join_diagnostics.clone()
    } else {
        native_diagnostics.input_join_diagnostics.clone()
    };
    let source_diagnostics =
        native_diagnostics.source_diagnostics.iter().map(summarize_source_diagnostics).collect::<Vec<_>>();

    native_diagnostics
        .report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => NativeDataFusionPlanningRow {
                query_id: item.query_id.clone(),
                physical_root: native_diagnostics.input_plan_root.clone(),
                physical_shape: native_diagnostics.input_plan_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
                reason_code: None,
                unsupported_category: None,
                strict_bounded_contract_rows: strict_bounded_contract_rows_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    None,
                ),
                strict_bounded_compiled_graphs: strict_bounded_compiled_graphs_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    None,
                ),
                adaptive_pipeline_rows: adaptive_pipeline_rows_for_candidate(
                    &native_diagnostics.adaptive_pipeline,
                    candidate,
                ),
                native_optimizer: native_diagnostics.native_optimizer.clone(),
                edge_normalize_inserted,
                join_diagnostics: if post_native_coverage.execution_path
                    == NativeDataFusionPostNativeExecutionPath::Native
                {
                    post_native_join_diagnostics.clone()
                } else {
                    input_join_diagnostics.clone()
                },
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path: post_native_coverage.execution_path,
                native_plan_fragment_count: post_native_coverage.native_fragment_count,
                host_datafusion_boundary_count: post_native_coverage.host_datafusion_boundary_count,
                rejected_before_execution: false,
            },
            CandidateOutcome::Rejected { candidate, reason } => NativeDataFusionPlanningRow {
                query_id: item.query_id.clone(),
                physical_root: native_diagnostics.input_plan_root.clone(),
                physical_shape: native_diagnostics.input_plan_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
                reason_code: Some(reason.code().to_string()),
                unsupported_category: unsupported_category(planned_plan.as_ref(), candidate, *reason),
                strict_bounded_contract_rows: strict_bounded_contract_rows_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    Some(*reason),
                ),
                strict_bounded_compiled_graphs: strict_bounded_compiled_graphs_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    Some(*reason),
                ),
                adaptive_pipeline_rows: adaptive_pipeline_rows_for_candidate(
                    &native_diagnostics.adaptive_pipeline,
                    candidate,
                ),
                native_optimizer: native_diagnostics.native_optimizer.clone(),
                edge_normalize_inserted,
                join_diagnostics: input_join_diagnostics.clone(),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path: post_native_coverage.execution_path,
                native_plan_fragment_count: post_native_coverage.native_fragment_count,
                host_datafusion_boundary_count: post_native_coverage.host_datafusion_boundary_count,
                rejected_before_execution: true,
            },
        })
        .collect()
}

fn datafusion_execution_rows_for_item(
    item: NativeDataFusionExecutionWorkloadItem,
) -> Vec<NativeDataFusionExecutionRow> {
    let (planned_plan, native_diagnostics, runtime) = match item.input {
        NativeDataFusionExecutionWorkloadInput::Planned { planned_plan, native_diagnostics, runtime } => {
            (planned_plan, native_diagnostics, runtime)
        },
        NativeDataFusionExecutionWorkloadInput::PlanningFailure(failure) => {
            return vec![datafusion_execution_row_for_planning_failure(item.query_id, failure)];
        },
    };
    let post_native_plan_root = planned_plan.name().to_string();
    let post_native_coverage = post_native_plan_coverage(planned_plan.as_ref());
    let source_diagnostics =
        native_diagnostics.source_diagnostics.iter().map(summarize_source_diagnostics).collect::<Vec<_>>();
    let runtime_summary = NativeDataFusionRuntimeSummary::from_runtime(runtime);

    native_diagnostics
        .report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => NativeDataFusionExecutionRow {
                query_id: item.query_id.clone(),
                physical_root: native_diagnostics.input_plan_root.clone(),
                physical_shape: native_diagnostics.input_plan_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                planning_outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
                reason_code: None,
                unsupported_category: None,
                strict_bounded_contract_rows: strict_bounded_contract_rows_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    None,
                ),
                strict_bounded_compiled_graphs: strict_bounded_compiled_graphs_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    None,
                ),
                adaptive_pipeline_rows: adaptive_pipeline_rows_for_candidate(
                    &native_diagnostics.adaptive_pipeline,
                    candidate,
                ),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path: post_native_coverage.execution_path,
                native_plan_fragment_count: post_native_coverage.native_fragment_count,
                host_datafusion_boundary_count: post_native_coverage.host_datafusion_boundary_count,
                rejected_before_execution: false,
                native_runtime_started: runtime_summary.started,
                runtime_outcome: runtime_summary.outcome,
                runtime_failure: runtime_summary.failure.clone(),
                output_batches: runtime_summary.output_batches,
                output_rows: runtime_summary.output_rows,
                output_bytes: runtime_summary.output_bytes,
                metrics: runtime_summary.metrics.clone(),
            },
            CandidateOutcome::Rejected { candidate, reason } => NativeDataFusionExecutionRow {
                query_id: item.query_id.clone(),
                physical_root: native_diagnostics.input_plan_root.clone(),
                physical_shape: native_diagnostics.input_plan_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                planning_outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
                reason_code: Some(reason.code().to_string()),
                unsupported_category: unsupported_category(planned_plan.as_ref(), candidate, *reason),
                strict_bounded_contract_rows: strict_bounded_contract_rows_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    Some(*reason),
                ),
                strict_bounded_compiled_graphs: strict_bounded_compiled_graphs_for_candidate(
                    &native_diagnostics.strict_bounded_contract,
                    candidate,
                    Some(*reason),
                ),
                adaptive_pipeline_rows: adaptive_pipeline_rows_for_candidate(
                    &native_diagnostics.adaptive_pipeline,
                    candidate,
                ),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path: post_native_coverage.execution_path,
                native_plan_fragment_count: post_native_coverage.native_fragment_count,
                host_datafusion_boundary_count: post_native_coverage.host_datafusion_boundary_count,
                rejected_before_execution: true,
                native_runtime_started: false,
                runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
                runtime_failure: None,
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                metrics: NativeDataFusionExecutionMetrics::default(),
            },
        })
        .collect()
}

fn datafusion_planning_row_for_planning_failure(
    query_id: String,
    failure: NativePlanningFailure,
) -> NativeDataFusionPlanningRow {
    NativeDataFusionPlanningRow {
        query_id,
        physical_root: "PlanningFailure".to_string(),
        physical_shape: failure.stage,
        native_candidate_shape: None,
        native_candidate_shape_family: None,
        outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        reason_code: Some(failure.reason_code),
        unsupported_category: Some("planning".to_string()),
        strict_bounded_contract_rows: Vec::new(),
        strict_bounded_compiled_graphs: Vec::new(),
        adaptive_pipeline_rows: Vec::new(),
        native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics::default(),
        edge_normalize_inserted: false,
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "PlanningFailure".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        native_plan_fragment_count: 0,
        host_datafusion_boundary_count: 0,
        rejected_before_execution: true,
    }
}

fn datafusion_execution_row_for_planning_failure(
    query_id: String,
    failure: NativePlanningFailure,
) -> NativeDataFusionExecutionRow {
    NativeDataFusionExecutionRow {
        query_id,
        physical_root: "PlanningFailure".to_string(),
        physical_shape: failure.stage,
        native_candidate_shape: None,
        native_candidate_shape_family: None,
        planning_outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        reason_code: Some(failure.reason_code),
        unsupported_category: Some("planning".to_string()),
        strict_bounded_contract_rows: Vec::new(),
        strict_bounded_compiled_graphs: Vec::new(),
        adaptive_pipeline_rows: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "PlanningFailure".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        native_plan_fragment_count: 0,
        host_datafusion_boundary_count: 0,
        rejected_before_execution: true,
        native_runtime_started: false,
        runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
        runtime_failure: Some(stable_tsv_field(&failure.message)),
        output_batches: None,
        output_rows: None,
        output_bytes: None,
        metrics: NativeDataFusionExecutionMetrics::default(),
    }
}

fn strict_bounded_contract_rows_for_candidate(
    diagnostics: &crate::planner::NativeStrictBoundedContractDiagnostics,
    candidate: &GpuCandidate,
    reason: Option<FallbackReason>,
) -> Vec<StrictBoundedContractGateRow> {
    if !strict_bounded_contract_is_candidate_scoped(candidate, reason) {
        return Vec::new();
    }
    if reason == Some(FallbackReason::NativeBoundedJoinUnsupported) {
        return diagnostics
            .rejected_candidates
            .iter()
            .filter(|entry| entry.candidate == *candidate)
            .map(|entry| entry.blocking_contract.clone())
            .collect();
    }

    diagnostics
        .accepted_candidates
        .iter()
        .filter(|entry| entry.candidate == *candidate)
        .flat_map(|entry| entry.report.gate_rows().iter().cloned())
        .collect()
}

fn strict_bounded_compiled_graphs_for_candidate(
    diagnostics: &crate::planner::NativeStrictBoundedContractDiagnostics,
    candidate: &GpuCandidate,
    reason: Option<FallbackReason>,
) -> Vec<crate::planner::NativeStrictBoundedCompiledGraphDiagnostics> {
    if reason.is_some() || !strict_bounded_contract_is_candidate_scoped(candidate, reason) {
        return Vec::new();
    }
    diagnostics
        .accepted_candidates
        .iter()
        .filter(|entry| entry.candidate == *candidate)
        .map(|entry| entry.compiled_graph.clone())
        .collect()
}

fn adaptive_pipeline_rows_for_candidate(
    diagnostics: &crate::planner::NativeAdaptivePipelineDiagnostics,
    candidate: &GpuCandidate,
) -> Vec<AdaptivePipelineReportRow> {
    diagnostics.rows_for_candidate(candidate)
}

fn strict_bounded_contract_is_candidate_scoped(candidate: &GpuCandidate, reason: Option<FallbackReason>) -> bool {
    reason == Some(FallbackReason::NativeBoundedJoinUnsupported) || candidate.shape.contains("JoinExec")
}
