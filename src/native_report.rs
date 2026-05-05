use std::{fmt::Write, sync::Arc};

use cudf_query_engine::{SourceDiagnostics as NativeSourceDiagnostics, is_supported_arrow_type};
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr, ScalarFunctionExpr,
        expressions::{
            BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
            TryCastExpr,
        },
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        filter::FilterExec,
        joins::{HashJoinExec, NestedLoopJoinExec},
        projection::ProjectionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use crate::{
    CandidateOutcome, CudfGpuIslandExec, FallbackReason, NativeCudfExec, lower_physical_plan_to_native,
    planner::{CudfNativePhysicalOptimizerRule, NativePlanningDiagnostics},
    report::GpuCandidate,
    tpch::TpchDatasetMetadata,
};

pub const NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION: u32 = 4;
pub const NATIVE_FALLBACK_PRESSURE_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\toutcome\treason_code\tunsupported_category\tjoin_diagnostics\tsource_diagnostics\trejected_before_execution";
pub const NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION: u32 = 4;
pub const NATIVE_DATAFUSION_PLANNING_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\toutcome\treason_code\tunsupported_category\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution";

#[derive(Debug, Clone)]
pub struct NativeFallbackPressureWorkloadItem {
    pub query_id: String,
    pub input: NativeFallbackPressureWorkloadInput,
}

impl NativeFallbackPressureWorkloadItem {
    pub fn new(query_id: impl Into<String>, plan: Arc<dyn ExecutionPlan>) -> Self {
        Self { query_id: query_id.into(), input: NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) }
    }

    pub fn planning_failure(
        query_id: impl Into<String>,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeFallbackPressureWorkloadInput::PlanningFailure(NativePlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NativeFallbackPressureWorkloadInput {
    PhysicalPlan(Arc<dyn ExecutionPlan>),
    PlanningFailure(NativePlanningFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanningFailure {
    pub stage: String,
    pub reason_code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFallbackPressureReport {
    pub rows: Vec<NativeFallbackPressureRow>,
    pub tpch_dataset: Option<TpchDatasetMetadata>,
}

impl NativeFallbackPressureReport {
    pub fn collect(items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(report_rows_for_item).collect();
        Self { rows, tpch_dataset: None }
    }

    pub fn with_tpch_dataset_metadata(mut self, metadata: TpchDatasetMetadata) -> Self {
        self.tpch_dataset = Some(metadata);
        self
    }

    pub fn rewritten_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rewritten).count()
    }

    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rejected).count()
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_fallback_pressure_tsv_schema_version={NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION}"
        );
        write_tpch_dataset_metadata(&mut output, self.tpch_dataset.as_ref());
        output.push_str(NATIVE_FALLBACK_PRESSURE_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.outcome.as_str(),
                row.reason_code.as_deref().unwrap_or("none"),
                row.unsupported_category.as_deref().unwrap_or("none"),
                format_join_diagnostics(&row.join_diagnostics),
                format_source_diagnostics(&row.source_diagnostics),
                row.rejected_before_execution,
            );
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningReport {
    pub rows: Vec<NativeDataFusionPlanningRow>,
    pub tpch_dataset: Option<TpchDatasetMetadata>,
}

impl NativeDataFusionPlanningReport {
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionPlanningWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_planning_rows_for_item).collect();
        Self { rows, tpch_dataset: None }
    }

    pub fn from_retained_rule(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_rule: &CudfNativePhysicalOptimizerRule,
    ) -> Self {
        let Some(diagnostics) = native_rule.last_planning_diagnostics() else {
            return Self::collect([NativeDataFusionPlanningWorkloadItem::planning_failure(
                query_id,
                "native_diagnostics",
                "missing_native_diagnostics",
                "native optimizer rule did not retain planning diagnostics",
            )]);
        };
        Self::collect([NativeDataFusionPlanningWorkloadItem::new(query_id, planned_plan, diagnostics)])
    }

    pub fn with_tpch_dataset_metadata(mut self, metadata: TpchDatasetMetadata) -> Self {
        self.tpch_dataset = Some(metadata);
        self
    }

    pub fn accepted_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced).count()
    }

    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::RejectedBeforeExecution).count()
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_datafusion_planning_tsv_schema_version={NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION}"
        );
        write_tpch_dataset_metadata(&mut output, self.tpch_dataset.as_ref());
        output.push_str(NATIVE_DATAFUSION_PLANNING_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.outcome.as_str(),
                row.reason_code.as_deref().unwrap_or("none"),
                row.unsupported_category.as_deref().unwrap_or("none"),
                format_join_diagnostics(&row.join_diagnostics),
                format_source_diagnostics(&row.source_diagnostics),
                row.post_native_plan_root,
                row.post_native_execution_path.as_str(),
                row.rejected_before_execution,
            );
        }
        output
    }
}

#[derive(Debug, Clone)]
pub struct NativeDataFusionPlanningWorkloadItem {
    pub query_id: String,
    pub input: NativeDataFusionPlanningWorkloadInput,
}

impl NativeDataFusionPlanningWorkloadItem {
    pub fn new(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: NativePlanningDiagnostics,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionPlanningWorkloadInput::Planned { planned_plan, native_diagnostics },
        }
    }

    pub fn planning_failure(
        query_id: impl Into<String>,
        stage: impl Into<String>,
        reason_code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionPlanningWorkloadInput::PlanningFailure(NativePlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NativeDataFusionPlanningWorkloadInput {
    Planned { planned_plan: Arc<dyn ExecutionPlan>, native_diagnostics: NativePlanningDiagnostics },
    PlanningFailure(NativePlanningFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningRow {
    pub query_id: String,
    pub physical_root: String,
    pub physical_shape: String,
    pub native_candidate_shape: Option<String>,
    pub outcome: NativeDataFusionPlanningOutcome,
    pub reason_code: Option<String>,
    pub unsupported_category: Option<String>,
    pub join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    pub post_native_plan_root: String,
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub rejected_before_execution: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionPlanningOutcome {
    AcceptedReplaced,
    RejectedBeforeExecution,
}

impl NativeDataFusionPlanningOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AcceptedReplaced => "accepted_replaced",
            Self::RejectedBeforeExecution => "rejected_before_execution",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionPostNativeExecutionPath {
    Native,
    GpuIsland,
    Cpu,
}

impl NativeDataFusionPostNativeExecutionPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::GpuIsland => "gpu_island",
            Self::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFallbackPressureRow {
    pub query_id: String,
    pub physical_root: String,
    pub physical_shape: String,
    pub native_candidate_shape: Option<String>,
    pub outcome: NativeFallbackPressureOutcome,
    pub reason_code: Option<String>,
    pub unsupported_category: Option<String>,
    pub join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    pub rejected_before_execution: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeFallbackPressureOutcome {
    Rewritten,
    Rejected,
}

impl NativeFallbackPressureOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rewritten => "rewritten",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeJoinDiagnosticSummary {
    pub operator: String,
    pub join_type: String,
    pub key_count: usize,
    pub key_shape: String,
    pub residual_predicate: bool,
    pub null_equality: String,
    pub build_side: String,
    pub probe_side: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSourceDiagnosticSummary {
    pub format: String,
    pub direct_device_read: bool,
    pub file_count: usize,
    pub ranged_file_count: usize,
    pub known_bytes: Option<u64>,
    pub selected_row_groups: Option<usize>,
    pub predicate_disposition: String,
    pub byte_range_disposition: String,
    pub partition_column_disposition: String,
    pub zero_column_count_disposition: String,
    pub schema_mapping_disposition: String,
    pub read_column_count: usize,
    pub projected_output_column_count: usize,
    pub pure_filter_column_count: usize,
    pub rejected_features: Vec<String>,
}

fn report_rows_for_item(item: NativeFallbackPressureWorkloadItem) -> Vec<NativeFallbackPressureRow> {
    match item.input {
        NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) => report_rows_for_plan(item.query_id, plan),
        NativeFallbackPressureWorkloadInput::PlanningFailure(failure) => {
            vec![report_row_for_planning_failure(item.query_id, failure)]
        },
    }
}

fn report_rows_for_plan(query_id: String, plan: Arc<dyn ExecutionPlan>) -> Vec<NativeFallbackPressureRow> {
    let physical_root = plan.name().to_string();
    let physical_shape = physical_plan_shape(plan.as_ref());
    let join_diagnostics = join_diagnostics(plan.as_ref());
    let source_diagnostics_plan = Arc::clone(&plan);
    let result = lower_physical_plan_to_native(plan);
    let source_diagnostics = result.source_diagnostics.iter().map(summarize_source_diagnostics).collect::<Vec<_>>();

    result
        .report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Rewritten { candidate } => NativeFallbackPressureRow {
                query_id: query_id.clone(),
                physical_root: physical_root.clone(),
                physical_shape: physical_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                outcome: NativeFallbackPressureOutcome::Rewritten,
                reason_code: None,
                unsupported_category: None,
                join_diagnostics: join_diagnostics.clone(),
                source_diagnostics: source_diagnostics.clone(),
                rejected_before_execution: false,
            },
            CandidateOutcome::Rejected { candidate, reason } => NativeFallbackPressureRow {
                query_id: query_id.clone(),
                physical_root: physical_root.clone(),
                physical_shape: physical_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                outcome: NativeFallbackPressureOutcome::Rejected,
                reason_code: Some(reason.code().to_string()),
                unsupported_category: unsupported_category(source_diagnostics_plan.as_ref(), candidate, *reason),
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
        outcome: NativeFallbackPressureOutcome::Rejected,
        reason_code: Some(failure.reason_code),
        unsupported_category: Some("planning".to_string()),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        rejected_before_execution: true,
    }
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
    let post_native_execution_path = post_native_execution_path(planned_plan.as_ref());
    let join_diagnostics = join_diagnostics(planned_plan.as_ref());
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
                outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
                reason_code: None,
                unsupported_category: None,
                join_diagnostics: join_diagnostics.clone(),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path,
                rejected_before_execution: false,
            },
            CandidateOutcome::Rejected { candidate, reason } => NativeDataFusionPlanningRow {
                query_id: item.query_id.clone(),
                physical_root: native_diagnostics.input_plan_root.clone(),
                physical_shape: native_diagnostics.input_plan_shape.clone(),
                native_candidate_shape: Some(candidate.shape.to_string()),
                outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
                reason_code: Some(reason.code().to_string()),
                unsupported_category: unsupported_category(planned_plan.as_ref(), candidate, *reason),
                join_diagnostics: join_diagnostics.clone(),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path,
                rejected_before_execution: true,
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
        outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        reason_code: Some(failure.reason_code),
        unsupported_category: Some("planning".to_string()),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "PlanningFailure".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        rejected_before_execution: true,
    }
}

fn post_native_execution_path(plan: &dyn ExecutionPlan) -> NativeDataFusionPostNativeExecutionPath {
    if plan.as_any().downcast_ref::<NativeCudfExec>().is_some() {
        return NativeDataFusionPostNativeExecutionPath::Native;
    }
    if contains_gpu_island(plan) {
        return NativeDataFusionPostNativeExecutionPath::GpuIsland;
    }
    NativeDataFusionPostNativeExecutionPath::Cpu
}

fn contains_gpu_island(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<CudfGpuIslandExec>().is_some()
        || plan.children().iter().any(|child| contains_gpu_island(child.as_ref()))
}

fn join_diagnostics(plan: &dyn ExecutionPlan) -> Vec<NativeJoinDiagnosticSummary> {
    let mut diagnostics = Vec::new();
    collect_join_diagnostics(plan, &mut diagnostics);
    diagnostics
}

fn collect_join_diagnostics(plan: &dyn ExecutionPlan, diagnostics: &mut Vec<NativeJoinDiagnosticSummary>) {
    if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
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
    } else if let Some(nested_loop_join) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() {
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
    }

    for child in plan.children() {
        collect_join_diagnostics(child.as_ref(), diagnostics);
    }
}

fn summarize_source_diagnostics(diagnostics: &NativeSourceDiagnostics) -> NativeSourceDiagnosticSummary {
    NativeSourceDiagnosticSummary {
        format: format!("{:?}", diagnostics.format),
        direct_device_read: diagnostics.direct_device_read,
        file_count: diagnostics.file_count,
        ranged_file_count: diagnostics.ranged_file_count,
        known_bytes: diagnostics.known_bytes,
        selected_row_groups: diagnostics.selected_row_groups,
        predicate_disposition: diagnostics.predicate_disposition.as_str().to_string(),
        byte_range_disposition: diagnostics.byte_range_disposition.as_str().to_string(),
        partition_column_disposition: diagnostics.partition_column_disposition.as_str().to_string(),
        zero_column_count_disposition: diagnostics.zero_column_count_disposition.as_str().to_string(),
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

fn physical_plan_shape(plan: &dyn ExecutionPlan) -> String {
    let children = plan.children();
    if children.is_empty() {
        return plan.name().to_string();
    }

    let mut shape = String::new();
    shape.push_str(plan.name());
    shape.push('(');
    for (index, child) in children.iter().enumerate() {
        if index > 0 {
            shape.push(',');
        }
        shape.push_str(&physical_plan_shape(child.as_ref()));
    }
    shape.push(')');
    shape
}

fn unsupported_category(plan: &dyn ExecutionPlan, candidate: &GpuCandidate, reason: FallbackReason) -> Option<String> {
    match reason {
        FallbackReason::UnsupportedExpression => first_unsupported_expression_category(plan)
            .or_else(|| {
                first_aggregate_category(plan).filter(|category| {
                    matches!(
                        category.as_str(),
                        "aggregate_filter" | "aggregate_distinct" | "aggregate_order_by" | "aggregate_argument_count"
                    )
                })
            })
            .or_else(|| Some("expression".to_string())),
        FallbackReason::UnsupportedOperator => first_aggregate_category(plan)
            .or_else(|| first_unsupported_operator_category(plan, candidate.shape.as_str()))
            .or_else(|| first_global_aggregate_category(plan))
            .or_else(|| Some("operator_shape".to_string())),
        FallbackReason::UnsupportedType => {
            first_unsupported_expression_category(plan).or_else(|| Some("type".to_string()))
        },
        FallbackReason::NativeDistributedAggregateUnsupported => {
            first_aggregate_category(plan).or_else(|| Some("aggregate_distributed".to_string()))
        },
        FallbackReason::NativeSourceNonLocalObjectStore
        | FallbackReason::NativeSourcePartitionColumns
        | FallbackReason::NativeSourceExprAdapter
        | FallbackReason::NativeSourceScanLimit
        | FallbackReason::NativeSourcePreserveOrder
        | FallbackReason::NativeSourceOutputOrdering
        | FallbackReason::NativeSourcePushedDownParquetPredicate
        | FallbackReason::NativeSourceEmpty
        | FallbackReason::NativeSourcePartitionValues
        | FallbackReason::NativeSourceByteRange
        | FallbackReason::NativeSourceFileOrdering
        | FallbackReason::NativeSourceRowSelection => Some("source".to_string()),
        FallbackReason::NativeSortPreservingMergeUnsupported
        | FallbackReason::NativePartitionPreservingSortUnsupported => Some("operator_sort".to_string()),
        FallbackReason::InsufficientIslandSize
        | FallbackReason::PlanPropertyMismatch
        | FallbackReason::ExplicitCpuPolicy => None,
    }
}

fn first_unsupported_operator_category(plan: &dyn ExecutionPlan, candidate_shape: &str) -> Option<String> {
    let name = plan.name();
    if !native_report_operator_is_supported(name, candidate_shape) {
        return Some(format!("operator_{}", stable_operator_identifier(name)));
    }
    for child in plan.children() {
        if let Some(category) = first_unsupported_operator_category(child.as_ref(), candidate_shape) {
            return Some(category);
        }
    }
    None
}

fn native_report_operator_is_supported(name: &str, candidate_shape: &str) -> bool {
    matches!(
        name,
        "DataSourceExec"
            | "FilterExec"
            | "ProjectionExec"
            | "AggregateExec"
            | "SortExec"
            | "LocalLimitExec"
            | "GlobalLimitExec"
            | "CoalescePartitionsExec"
    ) || (name == "RepartitionExec" && candidate_shape.contains("RepartitionExec"))
}

fn stable_operator_identifier(name: &str) -> String {
    let stable = stable_identifier(name);
    if stable.starts_with("sortexec_topk") {
        return "sort_topk".to_string();
    }
    stable.strip_suffix("exec").map_or(stable.clone(), ToString::to_string)
}

fn first_aggregate_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>()
        && let Some(category) = aggregate_semantic_modifier_category(aggregate)
    {
        return Some(category);
    }

    for child in plan.children() {
        if let Some(category) = first_aggregate_category(child.as_ref())
            && matches!(category.as_str(), "aggregate_filter" | "aggregate_distinct" | "aggregate_order_by")
        {
            return Some(category);
        }
    }

    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>() {
        for expr in aggregate.aggr_expr() {
            let function = stable_identifier(expr.fun().name());
            if !matches!(function.as_str(), "sum" | "min" | "max" | "count" | "avg") {
                return Some(format!("aggregate_{function}"));
            }
            let expressions = expr.expressions();
            let [arg] = expressions.as_slice() else {
                return Some("aggregate_argument_count".to_string());
            };
            if let Some(category) = expression_category(arg) {
                return Some(category);
            }
        }
    }

    for child in plan.children() {
        if let Some(category) = first_aggregate_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn first_global_aggregate_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(aggregate) = plan.as_any().downcast_ref::<AggregateExec>()
        && aggregate.group_expr().expr().is_empty()
    {
        return Some("aggregate_global".to_string());
    }
    for child in plan.children() {
        if let Some(category) = first_global_aggregate_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn aggregate_semantic_modifier_category(aggregate: &AggregateExec) -> Option<String> {
    for filter in aggregate.filter_expr() {
        if filter.is_some() {
            return Some("aggregate_filter".to_string());
        }
    }
    for expr in aggregate.aggr_expr() {
        if expr.is_distinct() {
            return Some("aggregate_distinct".to_string());
        }
        if !expr.order_bys().is_empty() {
            return Some("aggregate_order_by".to_string());
        }
    }
    None
}

fn first_unsupported_expression_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>()
        && let Some(category) = expression_category(filter.predicate())
    {
        return Some(category);
    }

    if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
        for expr in projection.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(sort) = plan.as_any().downcast_ref::<SortExec>() {
        for expr in sort.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(merge) = plan.as_any().downcast_ref::<SortPreservingMergeExec>() {
        for expr in merge.expr() {
            if let Some(category) = expression_category(&expr.expr) {
                return Some(category);
            }
        }
    }

    if let Some(category) = first_aggregate_category(plan) {
        return Some(category);
    }

    for child in plan.children() {
        if let Some(category) = first_unsupported_expression_category(child.as_ref()) {
            return Some(category);
        }
    }
    None
}

fn expression_category(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    if expr.as_any().downcast_ref::<Column>().is_some() {
        return None;
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return (!literal_is_supported(literal.value())).then(|| "expression_literal".to_string());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if expression_contains_interval_literal(binary.left()) || expression_contains_interval_literal(binary.right()) {
            return interval_arithmetic_category(binary);
        }
        if !binary_operator_is_supported(binary.op()) {
            return Some(format!("expression_binary_{}", binary_operator_identifier(binary.op())));
        }
        return expression_category(binary.left()).or_else(|| expression_category(binary.right()));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return expression_category(not.arg());
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return expression_category(is_null.arg());
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return expression_category(is_not_null.arg());
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        return expression_category(cast.expr());
    }
    if let Some(try_cast) = expr.as_any().downcast_ref::<TryCastExpr>() {
        return expression_category(try_cast.expr()).or_else(|| Some("expression_try_cast".to_string()));
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        if like.case_insensitive() {
            return Some("expression_like_case_insensitive".to_string());
        }
        return expression_category(like.expr()).or_else(|| {
            expression_category(like.pattern())
                .or_else(|| literal_string(like.pattern()).is_none().then(|| "expression_like_pattern".to_string()))
        });
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Some("expression_in_list".to_string());
        }
        if in_list
            .list()
            .iter()
            .any(|expr| expr.as_any().downcast_ref::<Literal>().is_none_or(|literal| literal.value().is_null()))
        {
            return Some("expression_in_list".to_string());
        }
        return expression_category(in_list.expr()).or_else(|| in_list.list().iter().find_map(expression_category));
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr()
            && let Some(category) = expression_category(expr)
        {
            return Some(category);
        }
        for (when, then) in case.when_then_expr() {
            if let Some(category) = expression_category(when).or_else(|| expression_category(then)) {
                return Some(category);
            }
        }
        return case.else_expr().and_then(expression_category);
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        let function_name = stable_identifier(function.name());
        match function_name.as_str() {
            "substr" | "substring" => {
                return function.args().iter().find_map(expression_category).or_else(|| {
                    (!substring_arguments_are_supported(function)).then(|| "expression_substring_argument".to_string())
                });
            },
            "date_part" | "extract" => {
                return function.args().iter().find_map(expression_category).or_else(|| {
                    (!extract_field_is_supported(function)).then(|| "expression_extract_field".to_string())
                });
            },
            _ => return Some(format!("expression_scalar_function_{function_name}")),
        }
    }
    Some("expression".to_string())
}

fn literal_is_supported(value: &ScalarValue) -> bool {
    is_supported_arrow_type(&value.data_type())
}

fn expression_contains_interval_literal(expr: &Arc<dyn PhysicalExpr>) -> bool {
    if expr
        .as_any()
        .downcast_ref::<Literal>()
        .is_some_and(|literal| matches!(literal.value(), ScalarValue::IntervalMonthDayNano(Some(_))))
    {
        return true;
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        return expression_contains_interval_literal(binary.left())
            || expression_contains_interval_literal(binary.right());
    }
    false
}

fn interval_arithmetic_category(binary: &BinaryExpr) -> Option<String> {
    let left_interval = is_interval_literal_expr(binary.left());
    let right_interval = is_interval_literal_expr(binary.right());
    let supported_shape = matches!(
        (binary.op(), left_interval, right_interval),
        (Operator::Plus, true, false) | (Operator::Plus | Operator::Minus, false, true)
    );
    if !supported_shape {
        return Some("expression_interval_arithmetic".to_string());
    }
    if left_interval { expression_category(binary.right()) } else { expression_category(binary.left()) }
}

fn is_interval_literal_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.as_any()
        .downcast_ref::<Literal>()
        .is_some_and(|literal| matches!(literal.value(), ScalarValue::IntervalMonthDayNano(Some(_))))
}

fn literal_string(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Some(value.clone())
        },
        _ => None,
    }
}

fn substring_arguments_are_supported(function: &ScalarFunctionExpr) -> bool {
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return false;
    }
    let Some(start) = literal_i32(&args[1]) else {
        return false;
    };
    if start < 1 {
        return false;
    }
    if args.len() == 3 {
        let Some(length) = literal_i32(&args[2]) else {
            return false;
        };
        return length >= 0 && substring_stop_position(start, length).is_some();
    }
    true
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

fn extract_field_is_supported(function: &ScalarFunctionExpr) -> bool {
    let [field, _expr] = function.args() else {
        return false;
    };
    literal_string(field)
        .map(|field| matches!(field.to_ascii_lowercase().as_str(), "year" | "month" | "day"))
        .unwrap_or(false)
}

fn literal_i32(expr: &Arc<dyn PhysicalExpr>) -> Option<i32> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::Int8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int32(Some(value)) => Some(*value),
        ScalarValue::Int64(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt64(Some(value)) => i32::try_from(*value).ok(),
        _ => None,
    }
}

fn binary_operator_is_supported(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::And
            | Operator::Or
    )
}

fn binary_operator_identifier(op: &Operator) -> &'static str {
    match op {
        Operator::Plus => "add",
        Operator::Minus => "subtract",
        Operator::Multiply => "multiply",
        Operator::Divide => "divide",
        Operator::Modulo => "modulo",
        Operator::Eq => "eq",
        Operator::NotEq => "not_eq",
        Operator::Lt => "lt",
        Operator::LtEq => "lt_eq",
        Operator::Gt => "gt",
        Operator::GtEq => "gt_eq",
        Operator::And => "and",
        Operator::Or => "or",
        _ => "other",
    }
}

fn stable_identifier(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { '_' }).collect()
}

fn format_join_diagnostics(diagnostics: &[NativeJoinDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, diagnostics) in diagnostics.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let _ = write!(
            output,
            "operator={},join_type={},key_count={},key_shape={},residual_predicate={},null_equality={},build_side={},probe_side={}",
            diagnostics.operator,
            diagnostics.join_type,
            diagnostics.key_count,
            diagnostics.key_shape.replace(['\t', '\n', '\r', '|'], " "),
            diagnostics.residual_predicate,
            diagnostics.null_equality,
            diagnostics.build_side,
            diagnostics.probe_side
        );
    }
    output
}

fn format_source_diagnostics(diagnostics: &[NativeSourceDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, diagnostics) in diagnostics.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let rejected_features = if diagnostics.rejected_features.is_empty() {
            "none".to_string()
        } else {
            diagnostics.rejected_features.join(",")
        };
        let known_bytes = diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        let selected_row_groups =
            diagnostics.selected_row_groups.map_or_else(|| "all".to_string(), |row_groups| row_groups.to_string());
        let _ = write!(
            output,
            "format={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},schema_mapping={},rejected_features={}",
            diagnostics.format,
            diagnostics.direct_device_read,
            diagnostics.file_count,
            diagnostics.ranged_file_count,
            known_bytes,
            selected_row_groups,
            diagnostics.read_column_count,
            diagnostics.projected_output_column_count,
            diagnostics.pure_filter_column_count,
            diagnostics.predicate_disposition,
            diagnostics.byte_range_disposition,
            diagnostics.partition_column_disposition,
            diagnostics.zero_column_count_disposition,
            diagnostics.schema_mapping_disposition,
            rejected_features,
        );
    }
    output
}

fn write_tpch_dataset_metadata(output: &mut String, metadata: Option<&TpchDatasetMetadata>) {
    let Some(metadata) = metadata else {
        return;
    };

    let _ = writeln!(output, "# tpch_dataset_root={}", metadata.root.display());
    let _ = writeln!(output, "# tpch_dataset_scale_factor={}", metadata_value(&metadata.scale_factor));
    let _ = writeln!(output, "# tpch_dataset_provenance={}", metadata_value(&metadata.provenance));
    let _ = writeln!(output, "# tpch_dataset_generation_command={}", metadata_value(&metadata.generation_command));
    let _ = writeln!(output, "# tpch_dataset_table_count={}", metadata.table_count);
    let _ = writeln!(output, "# tpch_dataset_file_count={}", metadata.file_count);
    let _ = writeln!(output, "# tpch_dataset_total_rows={}", metadata.total_rows);
    let _ = writeln!(output, "# tpch_dataset_total_bytes={}", metadata.total_bytes);
    let _ = writeln!(output, "# tpch_dataset_metadata_fingerprint={}", metadata.metadata_fingerprint);
    let _ = writeln!(
        output,
        "# tpch_dataset_content_fingerprint={}",
        metadata.content_fingerprint.as_deref().unwrap_or("not_computed")
    );
    let _ = writeln!(output, "# tpch_dataset_table_file_counts={}", metadata.table_file_counts());
    let _ = writeln!(output, "# tpch_dataset_table_row_counts={}", metadata.table_row_counts());
    let _ = writeln!(output, "# tpch_dataset_table_byte_sizes={}", metadata.table_byte_sizes());
    let _ = writeln!(output, "# tpch_dataset_table_row_group_counts={}", metadata.table_row_group_counts());
}

fn metadata_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
