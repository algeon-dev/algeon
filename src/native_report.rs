use std::{collections::BTreeMap, fmt::Write, sync::Arc};

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
        metrics::MetricsSet,
        projection::ProjectionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};

use crate::{
    CandidateOutcome, CudfGpuIslandExec, FallbackReason, NativeCudfComposedExec, NativeCudfExec,
    NativeCudfGpuProducerExec, lower_physical_plan_to_native,
    planner::{CudfNativePhysicalOptimizerRule, NativePlanningDiagnostics},
    report::GpuCandidate,
    tpch::TpchDatasetMetadata,
};

pub const NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION: u32 = 5;
pub const NATIVE_FALLBACK_PRESSURE_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tjoin_diagnostics\tsource_diagnostics\trejected_before_execution";
pub const NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION: u32 = 6;
pub const NATIVE_DATAFUSION_PLANNING_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\toutcome\treason_code\tunsupported_category\tbounded_join_reservation_facts\tjoin_diagnostics\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution";
pub const NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION: u32 = 7;
pub const NATIVE_DATAFUSION_EXECUTION_TSV_HEADER: &str = "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\tnative_candidate_shape_family\tplanning_outcome\treason_code\tunsupported_category\tbounded_join_reservation_facts\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution\tnative_runtime_started\truntime_outcome\truntime_failure\toutput_batches\toutput_rows\toutput_bytes\tgpu_resident_input_handoffs\thost_to_device_uploads\thost_to_device_uploaded_rows\thost_to_device_uploaded_bytes\tdevice_to_host_materializations\tdevice_to_host_materialized_rows\tdevice_to_host_materialized_bytes\tdevice_to_device_copies\tdevice_to_device_copied_bytes\tparquet_files\tparquet_bytes_known\trow_groups_selected\tread_columns\tresident_dataframe_peak_bytes\tallocator_high_water_bytes\treservation_requested_bytes\treservation_granted_bytes\treservation_denied_count\tsource_chunks_planned\tsource_chunks_attempted\tsource_chunks_committed\tsource_chunks_retried\trow_groups_planned\trow_groups_committed\tpartial_aggregate_count\tpartial_aggregate_rows\tfinal_merge_attempts\tfinal_merge_levels\tretry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\thash_join_build_count\tfiltered_join_build_count\thash_join_cache_hits\thash_join_cache_misses\thash_join_durable_cache_hits\thash_join_durable_cache_misses\thash_join_durable_cache_inserts\thash_join_hash_table_bytes\tfiltered_join_hash_table_bytes\tfiltered_join_probe_state_bytes\tjoin_gather_map_bytes\tsource_diagnostics";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN: &str = "unsupported_native_physical_plan";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_LOCAL_PARQUET_AGGREGATE: &str = "sort_local_parquet_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_LOCAL_PARQUET_AGGREGATE: &str = "sort_fetch_local_parquet_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_PRESERVING_MERGE_FETCH_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE: &str =
    "sort_preserving_merge_fetch_local_parquet_distributed_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_AGGREGATE: &str =
    "projected_global_local_parquet_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE: &str =
    "projected_global_local_parquet_distributed_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_AGGREGATE: &str = "sort_fetch_hash_join_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_AGGREGATE: &str = "sort_hash_join_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_HASH_JOIN_AGGREGATE: &str =
    "projected_global_hash_join_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_AGGREGATE: &str =
    "sort_mixed_hash_nested_loop_join_aggregate";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_INPUT: &str = "sort_fetch_hash_join_input";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_INPUT: &str = "sort_hash_join_input";
pub const NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_INPUT: &str =
    "sort_mixed_hash_nested_loop_join_input";

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
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.native_candidate_shape_family.as_deref().unwrap_or("none"),
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
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.physical_root,
                row.physical_shape,
                row.native_candidate_shape.as_deref().unwrap_or("none"),
                row.native_candidate_shape_family.as_deref().unwrap_or("none"),
                row.outcome.as_str(),
                row.reason_code.as_deref().unwrap_or("none"),
                row.unsupported_category.as_deref().unwrap_or("none"),
                format_bounded_join_reservation_facts(&row.bounded_join_reservation_facts),
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

#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionWorkloadItem {
    pub query_id: String,
    pub input: NativeDataFusionExecutionWorkloadInput,
}

impl NativeDataFusionExecutionWorkloadItem {
    pub fn new(
        query_id: impl Into<String>,
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: NativePlanningDiagnostics,
        runtime: NativeDataFusionRuntime,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            input: NativeDataFusionExecutionWorkloadInput::Planned { planned_plan, native_diagnostics, runtime },
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
            input: NativeDataFusionExecutionWorkloadInput::PlanningFailure(NativePlanningFailure {
                stage: stage.into(),
                reason_code: reason_code.into(),
                message: message.into(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NativeDataFusionExecutionWorkloadInput {
    Planned {
        planned_plan: Arc<dyn ExecutionPlan>,
        native_diagnostics: NativePlanningDiagnostics,
        runtime: NativeDataFusionRuntime,
    },
    PlanningFailure(NativePlanningFailure),
}

#[derive(Debug, Clone)]
pub enum NativeDataFusionRuntime {
    NotStarted,
    Succeeded { output_batches: usize, output_rows: usize, output_bytes: usize, metrics: MetricsSet },
    Failed { message: String },
}

#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionReport {
    pub rows: Vec<NativeDataFusionExecutionRow>,
    pub tpch_dataset: Option<TpchDatasetMetadata>,
}

impl NativeDataFusionExecutionReport {
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionExecutionWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_execution_rows_for_item).collect();
        Self { rows, tpch_dataset: None }
    }

    pub fn with_tpch_dataset_metadata(mut self, metadata: TpchDatasetMetadata) -> Self {
        self.tpch_dataset = Some(metadata);
        self
    }

    pub fn runtime_started_count(&self) -> usize {
        self.rows.iter().filter(|row| row.native_runtime_started).count()
    }

    pub fn rejected_before_execution_count(&self) -> usize {
        self.rows.iter().filter(|row| row.rejected_before_execution).count()
    }

    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "# native_datafusion_execution_tsv_schema_version={NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION}"
        );
        write_tpch_dataset_metadata(&mut output, self.tpch_dataset.as_ref());
        output.push_str(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let fields = [
                row.query_id.clone(),
                row.physical_root.clone(),
                row.physical_shape.clone(),
                row.native_candidate_shape.clone().unwrap_or_else(|| "none".to_string()),
                row.native_candidate_shape_family.clone().unwrap_or_else(|| "none".to_string()),
                row.planning_outcome.as_str().to_string(),
                row.reason_code.clone().unwrap_or_else(|| "none".to_string()),
                row.unsupported_category.clone().unwrap_or_else(|| "none".to_string()),
                format_bounded_join_reservation_facts(&row.bounded_join_reservation_facts),
                row.post_native_plan_root.clone(),
                row.post_native_execution_path.as_str().to_string(),
                row.rejected_before_execution.to_string(),
                row.native_runtime_started.to_string(),
                row.runtime_outcome.as_str().to_string(),
                row.runtime_failure.clone().unwrap_or_else(|| "none".to_string()),
                row.output_batches.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
                row.output_rows.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
                row.output_bytes.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
                row.metrics.gpu_resident_input_handoffs.to_string(),
                row.metrics.host_to_device_uploads.to_string(),
                row.metrics.host_to_device_uploaded_rows.to_string(),
                row.metrics.host_to_device_uploaded_bytes.to_string(),
                row.metrics.device_to_host_materializations.to_string(),
                row.metrics.device_to_host_materialized_rows.to_string(),
                row.metrics.device_to_host_materialized_bytes.to_string(),
                row.metrics.device_to_device_copies.to_string(),
                row.metrics.device_to_device_copied_bytes.to_string(),
                row.metrics.parquet_files.to_string(),
                row.metrics.parquet_bytes_known.to_string(),
                row.metrics.row_groups_selected.to_string(),
                row.metrics.read_columns.to_string(),
                row.metrics.resident_dataframe_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                row.metrics.allocator_high_water_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
                row.metrics.reservation_requested_bytes.to_string(),
                row.metrics.reservation_granted_bytes.to_string(),
                row.metrics.reservation_denied_count.to_string(),
                row.metrics.source_chunks_planned.to_string(),
                row.metrics.source_chunks_attempted.to_string(),
                row.metrics.source_chunks_committed.to_string(),
                row.metrics.source_chunks_retried.to_string(),
                row.metrics.row_groups_planned.to_string(),
                row.metrics.row_groups_committed.to_string(),
                row.metrics.partial_aggregate_count.to_string(),
                row.metrics.partial_aggregate_rows.to_string(),
                row.metrics.final_merge_attempts.to_string(),
                row.metrics.final_merge_levels.to_string(),
                row.metrics.retry_count.to_string(),
                row.metrics.retry_resource_exhausted_count.to_string(),
                row.metrics.retry_reservation_failure_count.to_string(),
                row.metrics.hash_join_build_count.to_string(),
                row.metrics.filtered_join_build_count.to_string(),
                row.metrics.hash_join_cache_hits.to_string(),
                row.metrics.hash_join_cache_misses.to_string(),
                row.metrics.hash_join_durable_cache_hits.to_string(),
                row.metrics.hash_join_durable_cache_misses.to_string(),
                row.metrics.hash_join_durable_cache_inserts.to_string(),
                row.metrics.hash_join_hash_table_bytes.to_string(),
                row.metrics.filtered_join_hash_table_bytes.to_string(),
                row.metrics.filtered_join_probe_state_bytes.to_string(),
                row.metrics.join_gather_map_bytes.to_string(),
                format_source_diagnostics(&row.source_diagnostics),
            ];
            let _ = writeln!(output, "{}", fields.join("\t"));
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningRow {
    pub query_id: String,
    pub physical_root: String,
    pub physical_shape: String,
    pub native_candidate_shape: Option<String>,
    pub native_candidate_shape_family: Option<String>,
    pub outcome: NativeDataFusionPlanningOutcome,
    pub reason_code: Option<String>,
    pub unsupported_category: Option<String>,
    pub bounded_join_reservation_facts: BTreeMap<String, String>,
    pub join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    pub post_native_plan_root: String,
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub rejected_before_execution: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionExecutionRow {
    pub query_id: String,
    pub physical_root: String,
    pub physical_shape: String,
    pub native_candidate_shape: Option<String>,
    pub native_candidate_shape_family: Option<String>,
    pub planning_outcome: NativeDataFusionPlanningOutcome,
    pub reason_code: Option<String>,
    pub unsupported_category: Option<String>,
    pub bounded_join_reservation_facts: BTreeMap<String, String>,
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    pub post_native_plan_root: String,
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub rejected_before_execution: bool,
    pub native_runtime_started: bool,
    pub runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub runtime_failure: Option<String>,
    pub output_batches: Option<usize>,
    pub output_rows: Option<usize>,
    pub output_bytes: Option<usize>,
    pub metrics: NativeDataFusionExecutionMetrics,
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
pub enum NativeDataFusionRuntimeOutcome {
    NotStarted,
    Succeeded,
    Failed,
}

impl NativeDataFusionRuntimeOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeDataFusionExecutionMetrics {
    pub gpu_resident_input_handoffs: usize,
    pub host_to_device_uploads: usize,
    pub host_to_device_uploaded_rows: usize,
    pub host_to_device_uploaded_bytes: usize,
    pub device_to_host_materializations: usize,
    pub device_to_host_materialized_rows: usize,
    pub device_to_host_materialized_bytes: usize,
    pub device_to_device_copies: usize,
    pub device_to_device_copied_bytes: usize,
    pub parquet_files: usize,
    pub parquet_bytes_known: usize,
    pub row_groups_selected: usize,
    pub read_columns: usize,
    pub resident_dataframe_peak_bytes: Option<usize>,
    pub allocator_high_water_bytes: Option<usize>,
    pub reservation_requested_bytes: usize,
    pub reservation_granted_bytes: usize,
    pub reservation_denied_count: usize,
    pub source_chunks_planned: usize,
    pub source_chunks_attempted: usize,
    pub source_chunks_committed: usize,
    pub source_chunks_retried: usize,
    pub row_groups_planned: usize,
    pub row_groups_committed: usize,
    pub partial_aggregate_count: usize,
    pub partial_aggregate_rows: usize,
    pub final_merge_attempts: usize,
    pub final_merge_levels: usize,
    pub retry_count: usize,
    pub retry_resource_exhausted_count: usize,
    pub retry_reservation_failure_count: usize,
    pub hash_join_build_count: usize,
    pub filtered_join_build_count: usize,
    pub hash_join_cache_hits: usize,
    pub hash_join_cache_misses: usize,
    pub hash_join_durable_cache_hits: usize,
    pub hash_join_durable_cache_misses: usize,
    pub hash_join_durable_cache_inserts: usize,
    pub hash_join_hash_table_bytes: usize,
    pub filtered_join_hash_table_bytes: usize,
    pub filtered_join_probe_state_bytes: usize,
    pub join_gather_map_bytes: usize,
}

impl NativeDataFusionExecutionMetrics {
    pub fn from_metrics(metrics: &MetricsSet) -> Self {
        Self {
            gpu_resident_input_handoffs: metric_count(metrics, "gpu_resident_input_handoffs"),
            host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
            host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
            host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
            device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
            device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
            device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
            device_to_device_copies: metric_count(metrics, "device_to_device_copies"),
            device_to_device_copied_bytes: metric_count(metrics, "device_to_device_copied_bytes"),
            parquet_files: metric_count(metrics, "parquet_files"),
            parquet_bytes_known: metric_count(metrics, "parquet_bytes_known"),
            row_groups_selected: metric_count(metrics, "row_groups_selected"),
            read_columns: metric_count(metrics, "read_columns"),
            resident_dataframe_peak_bytes: optional_observed_metric_count(metrics, "resident_dataframe_peak_bytes"),
            allocator_high_water_bytes: optional_observed_metric_count(metrics, "allocator_high_water_bytes"),
            reservation_requested_bytes: metric_count(metrics, "reservation_requested_bytes"),
            reservation_granted_bytes: metric_count(metrics, "reservation_granted_bytes"),
            reservation_denied_count: metric_count(metrics, "reservation_denied_count"),
            source_chunks_planned: metric_count(metrics, "source_chunks_planned"),
            source_chunks_attempted: metric_count(metrics, "source_chunks_attempted"),
            source_chunks_committed: metric_count(metrics, "source_chunks_committed"),
            source_chunks_retried: metric_count(metrics, "source_chunks_retried"),
            row_groups_planned: metric_count(metrics, "row_groups_planned"),
            row_groups_committed: metric_count(metrics, "row_groups_committed"),
            partial_aggregate_count: metric_count(metrics, "partial_aggregate_count"),
            partial_aggregate_rows: metric_count(metrics, "partial_aggregate_rows"),
            final_merge_attempts: metric_count(metrics, "final_merge_attempts"),
            final_merge_levels: metric_count(metrics, "final_merge_levels"),
            retry_count: metric_count(metrics, "retry_count"),
            retry_resource_exhausted_count: metric_count(metrics, "retry_resource_exhausted_count"),
            retry_reservation_failure_count: metric_count(metrics, "retry_reservation_failure_count"),
            hash_join_build_count: metric_count(metrics, "hash_join_build_count"),
            filtered_join_build_count: metric_count(metrics, "filtered_join_build_count"),
            hash_join_cache_hits: metric_count(metrics, "hash_join_cache_hits"),
            hash_join_cache_misses: metric_count(metrics, "hash_join_cache_misses"),
            hash_join_durable_cache_hits: metric_count(metrics, "hash_join_durable_cache_hits"),
            hash_join_durable_cache_misses: metric_count(metrics, "hash_join_durable_cache_misses"),
            hash_join_durable_cache_inserts: metric_count(metrics, "hash_join_durable_cache_inserts"),
            hash_join_hash_table_bytes: metric_count(metrics, "hash_join_hash_table_bytes"),
            filtered_join_hash_table_bytes: metric_count(metrics, "filtered_join_hash_table_bytes"),
            filtered_join_probe_state_bytes: metric_count(metrics, "filtered_join_probe_state_bytes"),
            join_gather_map_bytes: metric_count(metrics, "join_gather_map_bytes"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionPostNativeExecutionPath {
    Native,
    PartialNative,
    GpuIsland,
    Cpu,
}

impl NativeDataFusionPostNativeExecutionPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::PartialNative => "partial_native",
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
    pub native_candidate_shape_family: Option<String>,
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
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
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
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
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
        native_candidate_shape_family: None,
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
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
                reason_code: None,
                unsupported_category: None,
                bounded_join_reservation_facts: bounded_join_facts_for_candidate(
                    &native_diagnostics.bounded_join_reservation_facts,
                    candidate,
                    None,
                ),
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
                native_candidate_shape_family: Some(candidate_shape_family(candidate)),
                outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
                reason_code: Some(reason.code().to_string()),
                unsupported_category: unsupported_category(planned_plan.as_ref(), candidate, *reason),
                bounded_join_reservation_facts: bounded_join_facts_for_candidate(
                    &native_diagnostics.bounded_join_reservation_facts,
                    candidate,
                    Some(*reason),
                ),
                join_diagnostics: join_diagnostics.clone(),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path,
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
    let post_native_execution_path = post_native_execution_path(planned_plan.as_ref());
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
                bounded_join_reservation_facts: bounded_join_facts_for_candidate(
                    &native_diagnostics.bounded_join_reservation_facts,
                    candidate,
                    None,
                ),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path,
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
                bounded_join_reservation_facts: bounded_join_facts_for_candidate(
                    &native_diagnostics.bounded_join_reservation_facts,
                    candidate,
                    Some(*reason),
                ),
                source_diagnostics: source_diagnostics.clone(),
                post_native_plan_root: post_native_plan_root.clone(),
                post_native_execution_path,
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
        bounded_join_reservation_facts: BTreeMap::new(),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "PlanningFailure".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
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
        bounded_join_reservation_facts: BTreeMap::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "PlanningFailure".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
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

fn bounded_join_facts_for_candidate(
    facts: &BTreeMap<String, String>,
    candidate: &GpuCandidate,
    reason: Option<FallbackReason>,
) -> BTreeMap<String, String> {
    if facts.is_empty() {
        return BTreeMap::new();
    }
    if reason == Some(FallbackReason::NativeBoundedJoinUnsupported) || candidate.shape.contains("JoinExec") {
        facts.clone()
    } else {
        BTreeMap::new()
    }
}

#[derive(Debug, Clone)]
struct NativeDataFusionRuntimeSummary {
    started: bool,
    outcome: NativeDataFusionRuntimeOutcome,
    failure: Option<String>,
    output_batches: Option<usize>,
    output_rows: Option<usize>,
    output_bytes: Option<usize>,
    metrics: NativeDataFusionExecutionMetrics,
}

impl NativeDataFusionRuntimeSummary {
    fn from_runtime(runtime: NativeDataFusionRuntime) -> Self {
        match runtime {
            NativeDataFusionRuntime::NotStarted => Self {
                started: false,
                outcome: NativeDataFusionRuntimeOutcome::NotStarted,
                failure: None,
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                metrics: NativeDataFusionExecutionMetrics::default(),
            },
            NativeDataFusionRuntime::Succeeded { output_batches, output_rows, output_bytes, metrics } => Self {
                started: true,
                outcome: NativeDataFusionRuntimeOutcome::Succeeded,
                failure: None,
                output_batches: Some(output_batches),
                output_rows: Some(output_rows),
                output_bytes: Some(output_bytes),
                metrics: NativeDataFusionExecutionMetrics::from_metrics(&metrics),
            },
            NativeDataFusionRuntime::Failed { message } => Self {
                started: true,
                outcome: NativeDataFusionRuntimeOutcome::Failed,
                failure: Some(stable_tsv_field(&message)),
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                metrics: NativeDataFusionExecutionMetrics::default(),
            },
        }
    }
}

fn post_native_execution_path(plan: &dyn ExecutionPlan) -> NativeDataFusionPostNativeExecutionPath {
    if plan.as_any().downcast_ref::<NativeCudfExec>().is_some()
        || plan.as_any().downcast_ref::<NativeCudfComposedExec>().is_some()
    {
        return NativeDataFusionPostNativeExecutionPath::Native;
    }
    if contains_native_exec(plan) {
        return NativeDataFusionPostNativeExecutionPath::PartialNative;
    }
    if contains_gpu_island(plan) {
        return NativeDataFusionPostNativeExecutionPath::GpuIsland;
    }
    NativeDataFusionPostNativeExecutionPath::Cpu
}

fn contains_native_exec(plan: &dyn ExecutionPlan) -> bool {
    plan.as_any().downcast_ref::<NativeCudfExec>().is_some()
        || plan.as_any().downcast_ref::<NativeCudfComposedExec>().is_some()
        || plan.as_any().downcast_ref::<NativeCudfGpuProducerExec>().is_some()
        || plan.children().iter().any(|child| contains_native_exec(child.as_ref()))
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

fn candidate_shape_family(candidate: &GpuCandidate) -> String {
    let shape = candidate.shape.as_str();
    if shape == "UnsupportedNativePhysicalPlan" {
        return NATIVE_CANDIDATE_SHAPE_FAMILY_UNSUPPORTED_NATIVE_PHYSICAL_PLAN.to_string();
    }

    let root = if shape.starts_with("SortPreservingMergeExec(fetch)") {
        "sort_preserving_merge_fetch"
    } else if shape.starts_with("LocalLimitExec") {
        "local_limit_sort"
    } else if shape.starts_with("SortExec(fetch)") {
        "sort_fetch"
    } else if shape.starts_with("SortExec") {
        "sort"
    } else if shape.starts_with("ProjectionExec(AggregateExec")
        || shape.starts_with("ProjectionExec(FinalAggregateExec")
    {
        "projected_global"
    } else if shape.starts_with("AggregateExec") {
        "global"
    } else {
        "other"
    };
    let input = match (shape.contains("HashJoinExec"), shape.contains("NestedLoopJoinExec")) {
        (true, true) => "mixed_hash_nested_loop_join",
        (true, false) => "hash_join",
        (false, true) => "nested_loop_join",
        (false, false) => "local_parquet",
    };
    let distribution = if root_is_aggregate_shape(shape)
        && (shape.contains("FinalAggregateExec") || shape.contains("PartialAggregateExec"))
    {
        "distributed_aggregate"
    } else if root_is_aggregate_shape(shape) {
        "aggregate"
    } else {
        "input"
    };

    match (root, input, distribution) {
        ("sort", "local_parquet", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_LOCAL_PARQUET_AGGREGATE.to_string()
        },
        ("sort_fetch", "local_parquet", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_LOCAL_PARQUET_AGGREGATE.to_string()
        },
        ("sort_preserving_merge_fetch", "local_parquet", "distributed_aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_PRESERVING_MERGE_FETCH_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE.to_string()
        },
        ("projected_global", "local_parquet", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_AGGREGATE.to_string()
        },
        ("projected_global", "local_parquet", "distributed_aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_LOCAL_PARQUET_DISTRIBUTED_AGGREGATE.to_string()
        },
        ("sort_fetch", "hash_join", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_AGGREGATE.to_string()
        },
        ("sort", "hash_join", "aggregate") => NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_AGGREGATE.to_string(),
        ("projected_global", "hash_join", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_PROJECTED_GLOBAL_HASH_JOIN_AGGREGATE.to_string()
        },
        ("sort", "mixed_hash_nested_loop_join", "aggregate") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_AGGREGATE.to_string()
        },
        ("sort_fetch", "hash_join", "input") => NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_FETCH_HASH_JOIN_INPUT.to_string(),
        ("sort", "hash_join", "input") => NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_HASH_JOIN_INPUT.to_string(),
        ("sort", "mixed_hash_nested_loop_join", "input") => {
            NATIVE_CANDIDATE_SHAPE_FAMILY_SORT_MIXED_HASH_NESTED_LOOP_JOIN_INPUT.to_string()
        },
        _ => format!("{root}_{input}_{distribution}"),
    }
}

fn root_is_aggregate_shape(shape: &str) -> bool {
    shape.starts_with("AggregateExec")
        || shape.starts_with("FinalAggregateExec")
        || shape.starts_with("ProjectionExec(AggregateExec")
        || shape.starts_with("ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortExec(AggregateExec")
        || shape.starts_with("SortExec(FinalAggregateExec")
        || shape.starts_with("SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("SortExec(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortExec(fetch)(AggregateExec")
        || shape.starts_with("SortExec(fetch)(FinalAggregateExec")
        || shape.starts_with("SortExec(fetch)(ProjectionExec(AggregateExec")
        || shape.starts_with("SortExec(fetch)(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(AggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(FinalAggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("LocalLimitExec(SortExec(ProjectionExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(FinalAggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(AggregateExec")
        || shape.starts_with("SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec")
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
        FallbackReason::NativeBoundedJoinUnsupported => {
            first_join_operator_category(plan).or_else(|| Some("operator_join".to_string()))
        },
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

fn first_join_operator_category(plan: &dyn ExecutionPlan) -> Option<String> {
    if plan.as_any().downcast_ref::<HashJoinExec>().is_some() {
        return Some("operator_hashjoin".to_string());
    }
    if plan.as_any().downcast_ref::<NestedLoopJoinExec>().is_some() {
        return Some("operator_nestedloopjoin".to_string());
    }
    for child in plan.children() {
        if let Some(category) = first_join_operator_category(child.as_ref()) {
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

fn format_bounded_join_reservation_facts(facts: &BTreeMap<String, String>) -> String {
    if facts.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, (key, value)) in facts.iter().enumerate() {
        if index > 0 {
            output.push(';');
        }
        let _ = write!(output, "{}={}", stable_tsv_field(key), stable_tsv_field(value));
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

fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

fn optional_observed_metric_count(metrics: &MetricsSet, name: &str) -> Option<usize> {
    let observed_name = format!("{name}_observed");
    let observed = metrics.sum_by_name(&observed_name).map(|value| value.as_usize()).unwrap_or(0);
    (observed > 0).then(|| metric_count(metrics, name))
}
