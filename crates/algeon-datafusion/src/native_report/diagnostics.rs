//! Planner-side plan inspection helpers for native report rows.
//!
//! Walks DataFusion physical plans to derive native fragment coverage, join
//! and source diagnostics summaries, and stable unsupported-category labels
//! used by the planning/execution/fallback-pressure surfaces.
use super::{
    CandidateShape, CapabilityReason, CrossJoinExec, ExecutionPlan, GpuCandidate, HashJoinExec,
    NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN, NativeJoinDiagnosticSummary,
    NativeSourceDiagnosticSummary, NativeSourceDiagnostics, NestedLoopJoinExec, ScalarValue, WindowFrame,
    WindowFrameBound,
};
use crate::{
    final_plan_observation::observe_final_plan,
    planner::FinalPlanCoverage,
    report::{AggregateDefect, NotSupportedDetail, PlanBlocker, WindowDefect},
};

mod expressions;
mod shape_family;

use expressions::expression_rejection_category;
pub(crate) use shape_family::candidate_shape_family;

pub(super) use crate::physical_plan::physical_plan_shape;

pub(crate) fn post_native_plan_coverage(plan: &dyn ExecutionPlan) -> FinalPlanCoverage {
    observe_final_plan(plan)
}

pub(crate) fn join_diagnostics(plan: &dyn ExecutionPlan) -> Vec<NativeJoinDiagnosticSummary> {
    let mut diagnostics = Vec::new();
    collect_join_diagnostics(plan, &mut diagnostics);
    diagnostics
}

fn collect_join_diagnostics(plan: &dyn ExecutionPlan, diagnostics: &mut Vec<NativeJoinDiagnosticSummary>) {
    if let Some(hash_join) = plan.downcast_ref::<HashJoinExec>() {
        diagnostics.push(NativeJoinDiagnosticSummary {
            operator: "HashJoinExec".to_string(),
            join_type: stable_debug_value(hash_join.join_type()),
            key_count: hash_join.on().len(),
            key_shape: hash_join
                .on()
                .iter()
                .map(|(left, right)| format!("{left}={right}"))
                .collect::<Vec<_>>()
                .join(","),
            residual_predicate: hash_join.filter().is_some(),
            null_equality: stable_debug_value(&hash_join.null_equality()),
            build_side: "left".to_string(),
            probe_side: "right".to_string(),
        });
    } else if let Some(nested_loop_join) = plan.downcast_ref::<NestedLoopJoinExec>() {
        diagnostics.push(NativeJoinDiagnosticSummary {
            operator: "NestedLoopJoinExec".to_string(),
            join_type: stable_debug_value(nested_loop_join.join_type()),
            key_count: 0,
            key_shape: "none".to_string(),
            residual_predicate: nested_loop_join.filter().is_some(),
            null_equality: "not_applicable".to_string(),
            build_side: "left".to_string(),
            probe_side: "right".to_string(),
        });
    } else if plan.downcast_ref::<CrossJoinExec>().is_some() {
        diagnostics.push(NativeJoinDiagnosticSummary {
            operator: "CrossJoinExec".to_string(),
            join_type: "cross".to_string(),
            key_count: 0,
            key_shape: "none".to_string(),
            residual_predicate: false,
            null_equality: "not_applicable".to_string(),
            build_side: "left".to_string(),
            probe_side: "right".to_string(),
        });
    }

    for child in plan.children() {
        collect_join_diagnostics(child.as_ref(), diagnostics);
    }
}

pub(super) fn summarize_source_diagnostics(diagnostics: &NativeSourceDiagnostics) -> NativeSourceDiagnosticSummary {
    let stats = diagnostics.selected_row_group_stats.as_ref();
    NativeSourceDiagnosticSummary {
        format: "Parquet".to_string(),
        source_read_strategy: diagnostics.source_read_strategy.clone(),
        credential_mode: diagnostics.credential_mode.clone(),
        lakehouse_cache_status: diagnostics.lakehouse_cache_status.clone(),
        lakehouse_cache_read_mode: diagnostics.lakehouse_cache_read_mode.clone(),
        lakehouse_cache_fill_mode: diagnostics.lakehouse_cache_fill_mode.clone(),
        direct_device_read: diagnostics.direct_device_read,
        file_count: diagnostics.file_count,
        ranged_file_count: diagnostics.ranged_file_count,
        known_bytes: diagnostics.known_bytes,
        selected_row_groups: diagnostics.selected_row_groups,
        total_row_groups: stats.map(|stats| stats.total_row_groups),
        selected_source_bytes: stats.and_then(|stats| stats.selected_source_bytes()),
        total_source_bytes: stats.and_then(|stats| stats.total_source_bytes()),
        selected_rows: stats.and_then(|stats| stats.selected_rows),
        total_rows: stats.and_then(|stats| stats.total_rows),
        predicate_disposition: diagnostics.predicate_disposition.as_str().to_string(),
        byte_range_disposition: diagnostics.byte_range_disposition.as_str().to_string(),
        partition_column_disposition: diagnostics.partition_column_disposition.as_str().to_string(),
        zero_column_count_disposition: diagnostics.zero_column_count_disposition.as_str().to_string(),
        scan_limit_disposition: diagnostics.scan_limit_disposition.as_str().to_string(),
        schema_mapping_disposition: diagnostics.schema_mapping_disposition.as_str().to_string(),
        read_column_count: diagnostics.read_column_count,
        projected_output_column_count: diagnostics.projected_output_column_count,
        pure_filter_column_count: diagnostics.pure_filter_column_count,
        rejected_features: diagnostics.rejected_features.iter().map(|feature| feature.as_str().to_string()).collect(),
    }
}

fn stable_debug_value(value: &impl std::fmt::Debug) -> String {
    format!("{value:?}").to_ascii_lowercase()
}

/// The unsupported category of a not-supported candidate, read from the
/// subject its producing site carried. A candidate whose site named nothing
/// that refines its reason reports the category its reason's table row carries.
pub(super) fn unsupported_category(reason: CapabilityReason, detail: &NotSupportedDetail) -> String {
    if let Some(rejection) = &detail.expression {
        return expression_rejection_category(rejection);
    }
    let refined = detail.blocker.as_deref().and_then(|blocker| match (reason, blocker) {
        (CapabilityReason::UnsupportedOperator | CapabilityReason::NativeBoundedJoinUnsupported, _)
        | (
            CapabilityReason::UnsupportedExpression | CapabilityReason::NativeDistributedAggregateUnsupported,
            PlanBlocker::Aggregate { .. },
        ) => plan_blocker_category(blocker),
        _ => None,
    });
    refined.unwrap_or_else(|| reason.facts().tsv_category.to_string())
}

fn plan_blocker_category(blocker: &PlanBlocker) -> Option<String> {
    match blocker {
        PlanBlocker::Operator(name) => Some(format!("operator_{}", stable_operator_identifier(name))),
        PlanBlocker::Aggregate { function, defect } => Some(match defect {
            AggregateDefect::Distinct => "aggregate_distinct".to_string(),
            AggregateDefect::OrderBy => "aggregate_order_by".to_string(),
            AggregateDefect::Filter => "aggregate_filter".to_string(),
            AggregateDefect::ArgumentCount(_) => "aggregate_argument_count".to_string(),
            AggregateDefect::PartitionedInput => "aggregate_partitioned_input".to_string(),
            AggregateDefect::UnsupportedFunction
            | AggregateDefect::PartialStage
            | AggregateDefect::MedianWithStatistical
            | AggregateDefect::GroupingSets => format!("aggregate_{}", stable_identifier(function)),
        }),
        PlanBlocker::Window { function, defect } => match defect {
            WindowDefect::UnsupportedFunction => Some(format!("window_function_{}", stable_identifier(function))),
            WindowDefect::MissingOrderBy => {
                Some(format!("window_function_{}_without_order", stable_identifier(function)))
            },
            WindowDefect::UnsupportedFrame(frame) => Some(format!("window_frame_{}", window_frame_identifier(frame))),
            WindowDefect::Arguments => None,
        },
        PlanBlocker::ParentOperator(_) | PlanBlocker::DistributedAggregate(_) | PlanBlocker::Ordering { .. } => None,
    }
}

fn stable_operator_identifier(name: &str) -> String {
    let stable = stable_identifier(name);
    if stable.starts_with("sortexec_topk") {
        return "sort_topk".to_string();
    }
    stable.strip_suffix("exec").map_or(stable.clone(), ToString::to_string)
}

fn window_frame_identifier(frame: &WindowFrame) -> String {
    format!(
        "{}_{}_to_{}",
        stable_identifier(format!("{:?}", frame.units).as_str()),
        window_frame_bound_identifier(&frame.start_bound),
        window_frame_bound_identifier(&frame.end_bound)
    )
}

fn window_frame_bound_identifier(bound: &WindowFrameBound) -> String {
    match bound {
        WindowFrameBound::CurrentRow => "current_row".to_string(),
        WindowFrameBound::Preceding(value) if value.is_null() => "unbounded_preceding".to_string(),
        WindowFrameBound::Following(value) if value.is_null() => "unbounded_following".to_string(),
        WindowFrameBound::Preceding(value) => format!("{}_preceding", scalar_window_bound_identifier(value)),
        WindowFrameBound::Following(value) => format!("{}_following", scalar_window_bound_identifier(value)),
    }
}

fn scalar_window_bound_identifier(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Int8(Some(value)) => value.to_string(),
        ScalarValue::Int16(Some(value)) => value.to_string(),
        ScalarValue::Int32(Some(value)) => value.to_string(),
        ScalarValue::Int64(Some(value)) => value.to_string(),
        ScalarValue::UInt8(Some(value)) => value.to_string(),
        ScalarValue::UInt16(Some(value)) => value.to_string(),
        ScalarValue::UInt32(Some(value)) => value.to_string(),
        ScalarValue::UInt64(Some(value)) => value.to_string(),
        _ => stable_identifier(format!("{value:?}").as_str()).trim_matches('_').to_string(),
    }
}

fn stable_identifier(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { '_' }).collect()
}
