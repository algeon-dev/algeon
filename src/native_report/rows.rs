//! Row structs for the native planning/execution/fallback-pressure TSV surfaces.
//!
//! Top-level optional fields serialize as `none`. Nested source-diagnostics
//! values use their own sentinels (e.g. `unknown`, `all`) documented on each
//! field. Reservation requested/granted byte fields are cumulative admission
//! accounting over the query lifetime; live peak and retained-byte fields
//! describe instantaneous resident device memory.

use nexus_query_engine::{adaptive_report::AdaptivePipelineReportRow, contract_report::StrictBoundedContractGateRow};

use super::{
    metrics::NativeDataFusionExecutionMetrics,
    schema::{
        NATIVE_DATAFUSION_EXECUTION_TSV_COLUMNS, NATIVE_DATAFUSION_PLANNING_TSV_COLUMNS,
        NATIVE_FALLBACK_PRESSURE_TSV_COLUMNS,
    },
    tsv::{
        TsvRow, format_adaptive_pipeline_rows, format_aggregate_strategy_rows, format_dynamic_filter_runtime_rows,
        format_graph_reservation_rows, format_graph_runtime_rows, format_join_diagnostics, format_join_strategy_rows,
        format_lakehouse_cache_fill_event_snapshot_rows, format_plan_task_mapping_rows, format_source_diagnostics,
        format_strict_bounded_compiled_graphs, format_strict_bounded_contract_rows, format_task_operator_rows,
        format_task_runtime_rows,
    },
};
use crate::planner::NativeStrictBoundedCompiledGraphDiagnostics;

/// One row in the native planning TSV.
///
/// `None` optional fields are serialized as `none`. Nested row vectors are
/// serialized through stable key/value helpers so consumers can parse them
/// independently of display formatting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningRow {
    /// Workload query id or caller-supplied label.
    pub query_id: String,
    /// Display name of the physical plan root before native rewriting.
    pub physical_root: String,
    /// Stable physical shape summary before native rewriting.
    pub physical_shape: String,
    /// Native candidate shape label, or `none` when no candidate was found.
    pub native_candidate_shape: Option<String>,
    /// Broader native candidate shape family, or `none` when unavailable.
    pub native_candidate_shape_family: Option<String>,
    /// Planning outcome for this row.
    pub outcome: NativeDataFusionPlanningOutcome,
    /// Stable fallback reason code, or `none` for accepted candidates.
    pub reason_code: Option<String>,
    /// Stable unsupported-category label, or `none` when not rejected as unsupported.
    pub unsupported_category: Option<String>,
    /// Strict-only gate rows; adaptive evidence must use a separate surface.
    pub strict_bounded_contract_rows: Vec<StrictBoundedContractGateRow>,
    /// Compiled graph summaries for strict bounded candidates.
    pub strict_bounded_compiled_graphs: Vec<NativeStrictBoundedCompiledGraphDiagnostics>,
    /// Adaptive-only rows; strict evidence must not be projected here.
    pub adaptive_pipeline_rows: Vec<AdaptivePipelineReportRow>,
    /// Native optimizer diagnostics emitted while preparing the candidate.
    pub native_optimizer: nexus_query_engine::optimizer::NativeOptimizerDiagnostics,
    /// Whether the optimized DataFusion plan contains an explicit native edge-list normalization step.
    pub edge_normalize_inserted: bool,
    /// Nested join-shape diagnostics serialized as stable key/value rows.
    pub join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    /// Nested source-scan diagnostics serialized as stable key/value rows.
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    /// Display name of the post-native-optimizer physical root.
    pub post_native_plan_root: String,
    /// Summary of CPU/native coverage after the native optimizer rule.
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    /// Number of native fragments in the post-optimizer plan.
    pub native_plan_fragment_count: usize,
    /// Number of host DataFusion boundaries around native fragments.
    pub host_datafusion_boundary_count: usize,
    /// True when planning rejected the candidate before native execution could start.
    pub rejected_before_execution: bool,
}

/// One row in the native execution TSV.
///
/// Output counters are host Arrow counts when native execution materializes a
/// result. Reservation requested/granted values are cumulative admission
/// accounting; live peak and retained-byte fields are instantaneous resident
/// device-byte observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionExecutionRow {
    /// Workload query id or caller-supplied label.
    pub query_id: String,
    /// Display name of the physical plan root before native rewriting.
    pub physical_root: String,
    /// Stable physical shape summary before native rewriting.
    pub physical_shape: String,
    /// Native candidate shape label, or `none` when no candidate was found.
    pub native_candidate_shape: Option<String>,
    /// Broader native candidate shape family, or `none` when unavailable.
    pub native_candidate_shape_family: Option<String>,
    /// Planning outcome for this row.
    pub planning_outcome: NativeDataFusionPlanningOutcome,
    /// Stable fallback reason code, or `none` for accepted candidates.
    pub reason_code: Option<String>,
    /// Stable unsupported-category label, or `none` when not rejected as unsupported.
    pub unsupported_category: Option<String>,
    /// Strict-only gate rows; adaptive evidence must use a separate surface.
    pub strict_bounded_contract_rows: Vec<StrictBoundedContractGateRow>,
    /// Compiled graph summaries for strict bounded candidates.
    pub strict_bounded_compiled_graphs: Vec<NativeStrictBoundedCompiledGraphDiagnostics>,
    /// Adaptive-only rows; strict evidence must not be projected here.
    pub adaptive_pipeline_rows: Vec<AdaptivePipelineReportRow>,
    /// Nested source-scan diagnostics serialized as stable key/value rows.
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    /// Display name of the post-native-optimizer physical root.
    pub post_native_plan_root: String,
    /// Summary of CPU/native coverage after the native optimizer rule.
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    /// Number of native fragments in the post-optimizer plan.
    pub native_plan_fragment_count: usize,
    /// Number of host DataFusion boundaries around native fragments.
    pub host_datafusion_boundary_count: usize,
    /// True when planning rejected the candidate before native execution could start.
    pub rejected_before_execution: bool,
    /// True when execution reached a native execution wrapper.
    pub native_runtime_started: bool,
    /// Runtime outcome after planning.
    pub runtime_outcome: NativeDataFusionRuntimeOutcome,
    /// Runtime failure text, or `none` when not failed.
    pub runtime_failure: Option<String>,
    /// Output batch count, or `none` when not produced.
    pub output_batches: Option<usize>,
    /// Output row count, or `none` when not produced.
    pub output_rows: Option<usize>,
    /// Output byte count, or `none` when not produced.
    pub output_bytes: Option<usize>,
    /// Native execution metrics projected from DataFusion metrics.
    pub metrics: NativeDataFusionExecutionMetrics,
}

/// Stable planning outcomes emitted by native planning reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionPlanningOutcome {
    /// Candidate was accepted and replaced by native execution.
    AcceptedReplaced,
    /// Candidate was rejected before native execution.
    RejectedBeforeExecution,
}

impl NativeDataFusionPlanningOutcome {
    /// Stable TSV label for this planning outcome.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AcceptedReplaced => "accepted_replaced",
            Self::RejectedBeforeExecution => "rejected_before_execution",
        }
    }
}

/// Stable native runtime outcomes emitted by execution reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionRuntimeOutcome {
    /// Native runtime did not start.
    NotStarted,
    /// Native runtime completed successfully.
    Succeeded,
    /// Native runtime failed after starting.
    Failed,
}

impl NativeDataFusionRuntimeOutcome {
    /// Stable TSV label for this runtime outcome.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

/// Report vocabulary for how much of the optimized DataFusion plan runs through
/// native GPU execution after native planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionPostNativeExecutionPath {
    /// The optimized plan is fully native after the native optimizer rule.
    Native,
    /// The optimized plan keeps at least one native fragment but still contains
    /// CPU DataFusion execution outside the native fragment.
    PartialNative,
    /// The optimized plan contains no native execution after planning fallback.
    Cpu,
}

impl NativeDataFusionPostNativeExecutionPath {
    /// Stable TSV label for this post-native execution path.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::PartialNative => "partial_native",
            Self::Cpu => "cpu",
        }
    }
}

/// Native coverage summary for a post-optimizer DataFusion plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDataFusionPlanCoverage {
    /// CPU/native coverage class after native planning.
    pub execution_path: NativeDataFusionPostNativeExecutionPath,
    /// Number of native fragments in the plan.
    pub native_fragment_count: usize,
    /// Number of host DataFusion boundaries around native fragments.
    pub host_datafusion_boundary_count: usize,
}

/// One row in the fallback-pressure TSV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFallbackPressureRow {
    /// Workload query id or caller-supplied label.
    pub query_id: String,
    /// Display name of the physical plan root.
    pub physical_root: String,
    /// Stable physical shape summary.
    pub physical_shape: String,
    /// Native candidate shape label, or `none` when no candidate was found.
    pub native_candidate_shape: Option<String>,
    /// Broader native candidate shape family, or `none` when unavailable.
    pub native_candidate_shape_family: Option<String>,
    /// Fallback-pressure outcome for this candidate.
    pub outcome: NativeFallbackPressureOutcome,
    /// Stable fallback reason code, or `none` for rewritten candidates.
    pub reason_code: Option<String>,
    /// Stable unsupported-category label, or `none` when not rejected as unsupported.
    pub unsupported_category: Option<String>,
    /// Adaptive-only rows; strict fallback-pressure reports leave this empty.
    pub adaptive_pipeline_rows: Vec<AdaptivePipelineReportRow>,
    /// Nested join-shape diagnostics serialized as stable key/value rows.
    pub join_diagnostics: Vec<NativeJoinDiagnosticSummary>,
    /// Nested source-scan diagnostics serialized as stable key/value rows.
    pub source_diagnostics: Vec<NativeSourceDiagnosticSummary>,
    /// True when planning rejected the candidate before native execution could start.
    pub rejected_before_execution: bool,
}

/// Stable fallback-pressure outcome strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeFallbackPressureOutcome {
    /// Candidate was rewritten to native execution.
    Rewritten,
    /// Candidate was rejected and contributes fallback pressure.
    Rejected,
}

impl NativeFallbackPressureOutcome {
    /// Stable TSV label for this fallback-pressure outcome.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rewritten => "rewritten",
            Self::Rejected => "rejected",
        }
    }
}

/// Compact join-shape summary used in TSV nested diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeJoinDiagnosticSummary {
    /// Native operator label.
    pub operator: String,
    /// Join type label.
    pub join_type: String,
    /// Number of equality key pairs.
    pub key_count: usize,
    /// Stable key-shape label.
    pub key_shape: String,
    /// Whether a residual predicate is present.
    pub residual_predicate: bool,
    /// Null-equality policy label.
    pub null_equality: String,
    /// Build-side orientation label.
    pub build_side: String,
    /// Probe-side orientation label.
    pub probe_side: String,
}

/// Compact source-scan summary used in TSV nested diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSourceDiagnosticSummary {
    /// Source format label.
    pub format: String,
    /// High-level source read strategy summary.
    pub source_read_strategy: String,
    /// cuDF decode input summary.
    pub source_decode_input: String,
    /// Credential mode summary.
    pub credential_mode: String,
    /// Lakehouse byte-cache status summary.
    pub lakehouse_cache_status: String,
    /// Foreground lakehouse cache-read mode summary.
    pub lakehouse_cache_read_mode: String,
    /// Lakehouse complete-object cache fill mode summary.
    pub lakehouse_cache_fill_mode: String,
    /// Stable source fallback reason summary.
    pub source_fallback_reason: String,
    /// Whether direct device read is enabled for this source.
    pub direct_device_read: bool,
    /// Number of source files.
    pub file_count: usize,
    /// Number of files with explicit byte ranges.
    pub ranged_file_count: usize,
    /// Known source bytes; `None` serializes as `unknown` in nested TSV cells.
    pub known_bytes: Option<u64>,
    /// Selected row-group count; `None` means "all row groups" and serializes
    /// as `all` in nested TSV cells.
    pub selected_row_groups: Option<usize>,
    /// Total row groups represented by selection stats, when known.
    pub total_row_groups: Option<usize>,
    /// Selected source-read bytes from row-group stats, when known.
    pub selected_source_bytes: Option<u64>,
    /// Total source-read bytes from row-group stats, when known.
    pub total_source_bytes: Option<u64>,
    /// Selected source rows from row-group stats, when known.
    pub selected_rows: Option<u64>,
    /// Total source rows from row-group stats, when known.
    pub total_rows: Option<u64>,
    /// Predicate support/disposition label.
    pub predicate_disposition: String,
    /// Byte-range support/disposition label.
    pub byte_range_disposition: String,
    /// Partition-column support/disposition label.
    pub partition_column_disposition: String,
    /// Zero-column count support/disposition label.
    pub zero_column_count_disposition: String,
    /// Scan-limit support/disposition label.
    pub scan_limit_disposition: String,
    /// Schema-mapping support/disposition label.
    pub schema_mapping_disposition: String,
    /// Number of physical read columns.
    pub read_column_count: usize,
    /// Number of projected output columns.
    pub projected_output_column_count: usize,
    /// Number of pure-filter columns.
    pub pure_filter_column_count: usize,
    /// Stable feature labels rejected for this source.
    pub rejected_features: Vec<String>,
}

impl TsvRow for NativeFallbackPressureRow {
    const COLUMNS: &'static [&'static str] = NATIVE_FALLBACK_PRESSURE_TSV_COLUMNS;

    fn fields(&self) -> Vec<String> {
        vec![
            self.query_id.clone(),
            self.physical_root.clone(),
            self.physical_shape.clone(),
            self.native_candidate_shape.clone().unwrap_or_else(|| "none".to_string()),
            self.native_candidate_shape_family.clone().unwrap_or_else(|| "none".to_string()),
            self.outcome.as_str().to_string(),
            self.reason_code.clone().unwrap_or_else(|| "none".to_string()),
            self.unsupported_category.clone().unwrap_or_else(|| "none".to_string()),
            format_adaptive_pipeline_rows(&self.adaptive_pipeline_rows),
            format_join_diagnostics(&self.join_diagnostics),
            format_source_diagnostics(&self.source_diagnostics),
            self.rejected_before_execution.to_string(),
        ]
    }
}

impl TsvRow for NativeDataFusionPlanningRow {
    const COLUMNS: &'static [&'static str] = NATIVE_DATAFUSION_PLANNING_TSV_COLUMNS;

    fn fields(&self) -> Vec<String> {
        vec![
            self.query_id.clone(),
            self.physical_root.clone(),
            self.physical_shape.clone(),
            self.native_candidate_shape.clone().unwrap_or_else(|| "none".to_string()),
            self.native_candidate_shape_family.clone().unwrap_or_else(|| "none".to_string()),
            self.outcome.as_str().to_string(),
            self.reason_code.clone().unwrap_or_else(|| "none".to_string()),
            self.unsupported_category.clone().unwrap_or_else(|| "none".to_string()),
            format_strict_bounded_contract_rows(&self.strict_bounded_contract_rows),
            format_strict_bounded_compiled_graphs(&self.strict_bounded_compiled_graphs),
            format_adaptive_pipeline_rows(&self.adaptive_pipeline_rows),
            super::tsv::format_native_optimizer_diagnostics(&self.native_optimizer),
            self.edge_normalize_inserted.to_string(),
            format_join_diagnostics(&self.join_diagnostics),
            format_source_diagnostics(&self.source_diagnostics),
            self.post_native_plan_root.clone(),
            self.post_native_execution_path.as_str().to_string(),
            self.native_plan_fragment_count.to_string(),
            self.host_datafusion_boundary_count.to_string(),
            self.rejected_before_execution.to_string(),
        ]
    }
}

impl TsvRow for NativeDataFusionExecutionRow {
    const COLUMNS: &'static [&'static str] = NATIVE_DATAFUSION_EXECUTION_TSV_COLUMNS;

    fn fields(&self) -> Vec<String> {
        let mut plan_task_mapping_rows = self.metrics.plan_task_mapping_rows.clone();
        for row in &mut plan_task_mapping_rows {
            row.query_id.clone_from(&self.query_id);
        }
        vec![
            self.query_id.clone(),
            self.physical_root.clone(),
            self.physical_shape.clone(),
            self.native_candidate_shape.clone().unwrap_or_else(|| "none".to_string()),
            self.native_candidate_shape_family.clone().unwrap_or_else(|| "none".to_string()),
            self.planning_outcome.as_str().to_string(),
            self.reason_code.clone().unwrap_or_else(|| "none".to_string()),
            self.unsupported_category.clone().unwrap_or_else(|| "none".to_string()),
            format_strict_bounded_contract_rows(&self.strict_bounded_contract_rows),
            format_strict_bounded_compiled_graphs(&self.strict_bounded_compiled_graphs),
            format_adaptive_pipeline_rows(&self.adaptive_pipeline_rows),
            self.post_native_plan_root.clone(),
            self.post_native_execution_path.as_str().to_string(),
            self.native_plan_fragment_count.to_string(),
            self.host_datafusion_boundary_count.to_string(),
            self.rejected_before_execution.to_string(),
            self.metrics.admission_sequence.map_or_else(|| "none".to_string(), |value| value.to_string()),
            optional_duration_ms_text(self.metrics.admission_wait_time_ns),
            self.metrics.device_ordinal.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.metrics.query_grant_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.native_runtime_started.to_string(),
            self.runtime_outcome.as_str().to_string(),
            self.runtime_failure.clone().unwrap_or_else(|| "none".to_string()),
            self.output_batches.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
            self.output_rows.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
            self.output_bytes.map_or_else(|| "not_run".to_string(), |value| value.to_string()),
            self.metrics.gpu_resident_input_handoffs.to_string(),
            self.metrics.host_to_device_uploads.to_string(),
            self.metrics.host_to_device_uploaded_rows.to_string(),
            self.metrics.host_to_device_uploaded_bytes.to_string(),
            self.metrics.device_to_host_materializations.to_string(),
            self.metrics.device_to_host_materialized_rows.to_string(),
            self.metrics.device_to_host_materialized_bytes.to_string(),
            self.metrics.device_to_device_copies.to_string(),
            self.metrics.device_to_device_copied_bytes.to_string(),
            self.metrics.parquet_files.to_string(),
            self.metrics.parquet_bytes_known.to_string(),
            self.metrics.row_groups_selected.to_string(),
            self.metrics.read_columns.to_string(),
            self.metrics.resident_dataframe_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.metrics.allocator_high_water_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.metrics.reservation_requested_bytes.to_string(),
            self.metrics.reservation_granted_bytes.to_string(),
            self.metrics.reservation_live_peak_bytes.to_string(),
            self.metrics.reservation_denied_count.to_string(),
            self.metrics.max_task_reservation_bytes.to_string(),
            self.metrics.max_task_observed_peak_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
            self.metrics.max_task_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
            format_graph_reservation_rows(&self.metrics.graph_reservation_rows),
            format_graph_runtime_rows(&self.metrics.graph_runtime_rows),
            format_task_operator_rows(&self.metrics.task_operator_rows),
            format_task_runtime_rows(&self.metrics.task_runtime_rows),
            self.metrics.dynamic_filter_produced_count.to_string(),
            self.metrics.dynamic_filter_skipped_count.to_string(),
            self.metrics.dynamic_filter_applied_count.to_string(),
            self.metrics.dynamic_filter_row_groups_pruned.to_string(),
            self.metrics.dynamic_filter_source_bytes_pruned.to_string(),
            self.metrics.dynamic_filter_reader_predicate_rows_before.to_string(),
            self.metrics.dynamic_filter_reader_predicate_rows_after.to_string(),
            self.metrics.dynamic_filter_reader_predicate_rows_filtered.to_string(),
            self.metrics.dynamic_filter_post_decode_rows_before.to_string(),
            self.metrics.dynamic_filter_post_decode_rows_after.to_string(),
            self.metrics.dynamic_filter_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_source_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_filter_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_join_probe_post_decode_rows_filtered.to_string(),
            self.metrics.dynamic_filter_post_decode_reduction_stage().as_str().to_string(),
            self.metrics.dynamic_filter_measured_reduction_status().as_str().to_string(),
            self.metrics.dynamic_filter_work_avoidance_status().as_str().to_string(),
            format_dynamic_filter_runtime_rows(&self.metrics.dynamic_filter_runtime_rows),
            self.metrics.source_chunks_planned.to_string(),
            self.metrics.source_chunks_attempted.to_string(),
            self.metrics.source_chunks_committed.to_string(),
            self.metrics.source_chunks_retried.to_string(),
            self.metrics.source_read_prefetches_started.to_string(),
            self.metrics.source_read_prefetches_completed.to_string(),
            self.metrics.source_read_prefetches_skipped.to_string(),
            self.metrics.source_read_prefetches_abandoned.to_string(),
            self.metrics.source_cache_hits.to_string(),
            self.metrics.source_cache_misses.to_string(),
            self.metrics.source_cache_inserts.to_string(),
            self.metrics.source_cache_insert_skipped.to_string(),
            self.metrics.source_cache_retained_bytes.to_string(),
            self.metrics.row_groups_planned.to_string(),
            self.metrics.row_groups_committed.to_string(),
            self.metrics.partial_aggregate_count.to_string(),
            self.metrics.partial_aggregate_input_rows.to_string(),
            self.metrics.partial_aggregate_rows.to_string(),
            self.metrics.partial_aggregate_retained_bytes.to_string(),
            self.metrics.final_merge_attempts.to_string(),
            self.metrics.final_merge_levels.to_string(),
            self.metrics.final_merge_max_fan_in.to_string(),
            self.metrics.aggregate_libcudf_runtime_probe_count().to_string(),
            self.metrics.aggregate_libcudf_runtime_probe_elapsed_ns().to_string(),
            self.metrics.aggregate_libcudf_runtime_probe_input_rows().to_string(),
            self.metrics.aggregate_exact_cardinality_probe_count().to_string(),
            self.metrics.aggregate_exact_cardinality_probe_elapsed_ns().to_string(),
            self.metrics.aggregate_exact_cardinality_probe_input_rows().to_string(),
            self.metrics.aggregate_prepared_sort_groupby_elapsed_ns().to_string(),
            self.metrics.aggregate_encoded_key_partial_elapsed_ns().to_string(),
            self.metrics.retry_count.to_string(),
            self.metrics.retry_resource_exhausted_count.to_string(),
            self.metrics.retry_reservation_failure_count.to_string(),
            self.metrics.adaptive_retry_count.to_string(),
            self.metrics.adaptive_spill_count.to_string(),
            self.metrics.adaptive_downgrade_count.to_string(),
            self.metrics.adaptive_memory_history_lookup_count.to_string(),
            self.metrics.adaptive_memory_history_hit_count.to_string(),
            self.metrics.adaptive_memory_history_miss_count.to_string(),
            self.metrics.hash_join_build_count.to_string(),
            self.metrics.filtered_join_build_count.to_string(),
            self.metrics.one_shot_join_count.to_string(),
            self.metrics.hash_join_cache_hits.to_string(),
            self.metrics.hash_join_cache_misses.to_string(),
            self.metrics.hash_join_durable_cache_hits.to_string(),
            self.metrics.hash_join_durable_cache_misses.to_string(),
            self.metrics.hash_join_durable_cache_inserts.to_string(),
            self.metrics.hash_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_hash_table_bytes.to_string(),
            self.metrics.filtered_join_probe_state_bytes.to_string(),
            self.metrics.mixed_join_size_data_bytes.to_string(),
            self.metrics.one_shot_join_index_bytes.to_string(),
            self.metrics.one_shot_join_materialization_bytes.to_string(),
            self.metrics.one_shot_join_output_bytes.to_string(),
            self.metrics.join_gather_map_bytes.to_string(),
            format_aggregate_strategy_rows(&self.metrics.aggregate_strategy_rows),
            format_join_strategy_rows(&self.metrics.join_strategy_rows),
            self.metrics.source_fetch_bytes.to_string(),
            self.metrics.source_fetch_count.to_string(),
            optional_duration_ms_text(self.metrics.host_fetch_budget_wait_time_ns),
            optional_duration_ms_text(self.metrics.source_fetch_concurrency_wait_time_ns),
            optional_duration_ms_text(self.metrics.source_fetch_time_ns),
            optional_duration_ms_text(self.metrics.source_decode_time_ns),
            format_source_diagnostics(&self.source_diagnostics),
            format_lakehouse_cache_fill_event_snapshot_rows(&self.metrics.lakehouse_cache_fill_event_snapshot_rows),
            format_plan_task_mapping_rows(&plan_task_mapping_rows),
        ]
    }
}

fn optional_duration_ms_text(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("{:.3}", value as f64 / 1_000_000.0))
}

#[cfg(test)]
mod tests {
    use super::{
        super::schema::{
            NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
            NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        },
        *,
    };

    #[test]
    fn native_report_row_contract_headers_match_public_constants() {
        assert_eq!(NativeFallbackPressureRow::COLUMNS.join("\t"), NATIVE_FALLBACK_PRESSURE_TSV_HEADER);
        assert_eq!(NativeDataFusionPlanningRow::COLUMNS.join("\t"), NATIVE_DATAFUSION_PLANNING_TSV_HEADER);
        assert_eq!(NativeDataFusionExecutionRow::COLUMNS.join("\t"), NATIVE_DATAFUSION_EXECUTION_TSV_HEADER);
    }

    #[test]
    fn native_report_row_field_counts_match_columns() {
        let fallback = NativeFallbackPressureRow {
            query_id: "q".to_string(),
            physical_root: "Root".to_string(),
            physical_shape: "Root(Source)".to_string(),
            native_candidate_shape: None,
            native_candidate_shape_family: None,
            outcome: NativeFallbackPressureOutcome::Rejected,
            reason_code: Some("unsupported_operator".to_string()),
            unsupported_category: Some("operator_sort".to_string()),
            adaptive_pipeline_rows: Vec::new(),
            join_diagnostics: Vec::new(),
            source_diagnostics: Vec::new(),
            rejected_before_execution: true,
        };
        assert_eq!(fallback.fields().len(), NativeFallbackPressureRow::COLUMNS.len());
        assert_eq!(
            fallback.to_tsv_line(),
            "q\tRoot\tRoot(Source)\tnone\tnone\trejected\tunsupported_operator\toperator_sort\tnone\tnone\tnone\ttrue"
        );

        let planning = NativeDataFusionPlanningRow {
            query_id: "q".to_string(),
            physical_root: "Root".to_string(),
            physical_shape: "Root(Source)".to_string(),
            native_candidate_shape: None,
            native_candidate_shape_family: None,
            outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
            reason_code: Some("unsupported_operator".to_string()),
            unsupported_category: Some("operator_sort".to_string()),
            strict_bounded_contract_rows: Vec::new(),
            strict_bounded_compiled_graphs: Vec::new(),
            adaptive_pipeline_rows: Vec::new(),
            native_optimizer: nexus_query_engine::optimizer::NativeOptimizerDiagnostics::default(),
            edge_normalize_inserted: false,
            join_diagnostics: Vec::new(),
            source_diagnostics: Vec::new(),
            post_native_plan_root: "Root".to_string(),
            post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
            native_plan_fragment_count: 0,
            host_datafusion_boundary_count: 0,
            rejected_before_execution: true,
        };
        assert_eq!(planning.fields().len(), NativeDataFusionPlanningRow::COLUMNS.len());
        assert_eq!(
            planning.to_tsv_line(),
            "q\tRoot\tRoot(Source)\tnone\tnone\trejected_before_execution\tunsupported_operator\toperator_sort\tnone\tnone\tnone\tnone\tfalse\tnone\tnone\tRoot\tcpu\t0\t0\ttrue"
        );

        let execution = NativeDataFusionExecutionRow {
            query_id: "q".to_string(),
            physical_root: "Root".to_string(),
            physical_shape: "Root(Source)".to_string(),
            native_candidate_shape: None,
            native_candidate_shape_family: None,
            planning_outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
            reason_code: Some("unsupported_operator".to_string()),
            unsupported_category: Some("operator_sort".to_string()),
            strict_bounded_contract_rows: Vec::new(),
            strict_bounded_compiled_graphs: Vec::new(),
            adaptive_pipeline_rows: Vec::new(),
            source_diagnostics: Vec::new(),
            post_native_plan_root: "Root".to_string(),
            post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
            native_plan_fragment_count: 0,
            host_datafusion_boundary_count: 0,
            rejected_before_execution: true,
            native_runtime_started: false,
            runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            runtime_failure: None,
            output_batches: None,
            output_rows: None,
            output_bytes: None,
            metrics: NativeDataFusionExecutionMetrics::default(),
        };
        assert_eq!(execution.fields().len(), NativeDataFusionExecutionRow::COLUMNS.len());
        let expected_execution_fields = NativeDataFusionExecutionRow::COLUMNS
            .iter()
            .map(|column| match *column {
                "query_id" => "q",
                "physical_root" | "post_native_plan_root" => "Root",
                "physical_shape" => "Root(Source)",
                "planning_outcome" => "rejected_before_execution",
                "reason_code" => "unsupported_operator",
                "unsupported_category" => "operator_sort",
                "post_native_execution_path" => "cpu",
                "rejected_before_execution" => "true",
                "native_runtime_started" => "false",
                "runtime_outcome" => "not_started",
                "output_batches" | "output_rows" | "output_bytes" => "not_run",
                "dynamic_filter_measured_reduction_status" | "dynamic_filter_work_avoidance_status" => "unknown",
                "native_candidate_shape"
                | "native_candidate_shape_family"
                | "strict_bounded_contract_rows"
                | "strict_bounded_compiled_graphs"
                | "adaptive_pipeline_rows"
                | "admission_sequence"
                | "admission_wait_ms"
                | "device_ordinal"
                | "query_grant_bytes"
                | "runtime_failure"
                | "resident_dataframe_peak_bytes"
                | "allocator_high_water_bytes"
                | "max_task_observed_peak_bytes"
                | "max_task_retained_bytes"
                | "graph_reservation_rows"
                | "graph_runtime_rows"
                | "task_operator_rows"
                | "task_runtime_rows"
                | "plan_task_mapping_rows"
                | "dynamic_filter_post_decode_reduction_stage"
                | "dynamic_filter_runtime_rows"
                | "aggregate_strategy_rows"
                | "join_strategy_rows"
                | "host_fetch_budget_wait_time_ms"
                | "source_fetch_concurrency_wait_time_ms"
                | "source_fetch_time_ms"
                | "source_decode_time_ms"
                | "source_diagnostics"
                | "lakehouse_cache_fill_event_snapshot_rows" => "none",
                _ => "0",
            })
            .collect::<Vec<_>>();
        assert_eq!(execution.to_tsv_line(), expected_execution_fields.join("\t"));
    }

    #[test]
    fn native_execution_row_serializes_literal_metric_order() {
        let execution = NativeDataFusionExecutionRow {
            query_id: "q_exec".to_string(),
            physical_root: "RootExec".to_string(),
            physical_shape: "RootExec(Source)".to_string(),
            native_candidate_shape: Some("candidate_shape".to_string()),
            native_candidate_shape_family: Some("candidate_family".to_string()),
            planning_outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
            reason_code: None,
            unsupported_category: None,
            strict_bounded_contract_rows: Vec::new(),
            strict_bounded_compiled_graphs: Vec::new(),
            adaptive_pipeline_rows: Vec::new(),
            source_diagnostics: Vec::new(),
            post_native_plan_root: "NativeCudfExec".to_string(),
            post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Native,
            native_plan_fragment_count: 7,
            host_datafusion_boundary_count: 2,
            rejected_before_execution: false,
            native_runtime_started: true,
            runtime_outcome: NativeDataFusionRuntimeOutcome::Succeeded,
            runtime_failure: None,
            output_batches: Some(3),
            output_rows: Some(111),
            output_bytes: Some(2222),
            metrics: NativeDataFusionExecutionMetrics {
                admission_sequence: Some(100),
                admission_wait_time_ns: Some(1_500_000),
                device_ordinal: Some(2),
                query_grant_bytes: Some(4096),
                gpu_resident_input_handoffs: 10,
                host_to_device_uploads: 11,
                host_to_device_uploaded_rows: 12,
                host_to_device_uploaded_bytes: 13,
                device_to_host_materializations: 14,
                device_to_host_materialized_rows: 15,
                device_to_host_materialized_bytes: 16,
                device_to_device_copies: 17,
                device_to_device_copied_bytes: 18,
                parquet_files: 19,
                parquet_bytes_known: 20,
                row_groups_selected: 21,
                read_columns: 22,
                resident_dataframe_peak_bytes: Some(23),
                allocator_high_water_bytes: Some(24),
                reservation_requested_bytes: 25,
                reservation_granted_bytes: 26,
                reservation_live_peak_bytes: 27,
                reservation_denied_count: 28,
                max_task_reservation_bytes: 29,
                max_task_observed_peak_bytes: Some(30),
                max_task_retained_bytes: Some(31),
                dynamic_filter_produced_count: 32,
                dynamic_filter_skipped_count: 33,
                dynamic_filter_applied_count: 34,
                dynamic_filter_row_groups_pruned: 35,
                dynamic_filter_source_bytes_pruned: 36,
                dynamic_filter_reader_predicate_rows_before: 37,
                dynamic_filter_reader_predicate_rows_after: 38,
                dynamic_filter_reader_predicate_rows_filtered: 39,
                dynamic_filter_post_decode_rows_before: 40,
                dynamic_filter_post_decode_rows_after: 41,
                dynamic_filter_post_decode_rows_filtered: 42,
                dynamic_filter_source_post_decode_rows_filtered: 43,
                dynamic_filter_filter_post_decode_rows_filtered: 44,
                dynamic_filter_join_probe_post_decode_rows_filtered: 45,
                source_chunks_planned: 46,
                source_chunks_attempted: 47,
                source_chunks_committed: 48,
                source_chunks_retried: 49,
                source_read_prefetches_started: 50,
                source_read_prefetches_completed: 51,
                source_read_prefetches_skipped: 52,
                source_read_prefetches_abandoned: 53,
                source_cache_hits: 54,
                source_cache_misses: 55,
                source_cache_inserts: 56,
                source_cache_insert_skipped: 57,
                source_cache_retained_bytes: 58,
                row_groups_planned: 59,
                row_groups_committed: 60,
                partial_aggregate_count: 61,
                partial_aggregate_input_rows: 62,
                partial_aggregate_rows: 63,
                partial_aggregate_retained_bytes: 64,
                final_merge_attempts: 65,
                final_merge_levels: 66,
                final_merge_max_fan_in: 67,
                retry_count: 68,
                retry_resource_exhausted_count: 69,
                retry_reservation_failure_count: 70,
                adaptive_retry_count: 71,
                adaptive_spill_count: 72,
                adaptive_downgrade_count: 73,
                adaptive_memory_history_lookup_count: 74,
                adaptive_memory_history_hit_count: 75,
                adaptive_memory_history_miss_count: 76,
                hash_join_build_count: 77,
                filtered_join_build_count: 78,
                one_shot_join_count: 79,
                hash_join_cache_hits: 80,
                hash_join_cache_misses: 81,
                hash_join_durable_cache_hits: 82,
                hash_join_durable_cache_misses: 83,
                hash_join_durable_cache_inserts: 84,
                hash_join_hash_table_bytes: 85,
                filtered_join_hash_table_bytes: 86,
                filtered_join_probe_state_bytes: 87,
                mixed_join_size_data_bytes: 88,
                one_shot_join_index_bytes: 89,
                one_shot_join_materialization_bytes: 90,
                one_shot_join_output_bytes: 91,
                join_gather_map_bytes: 92,
                source_fetch_bytes: 93,
                source_fetch_count: 94,
                host_fetch_budget_wait_time_ns: Some(1_000_000),
                source_fetch_concurrency_wait_time_ns: Some(2_500_000),
                source_fetch_time_ns: Some(3_750_000),
                source_decode_time_ns: Some(4_125_000),
                lakehouse_cache_fill_event_snapshot_rows: vec![
                    super::super::metrics::NativeLakehouseCacheFillEventMetricRow {
                        outcome: "completed".to_string(),
                        bytes: Some(4096),
                        elapsed_ns: 9_000_000,
                        cache_path_hash: "cache123".to_string(),
                        stable_id_hash: "stable456".to_string(),
                        file_index: Some(0),
                        chunk_index: Some(2),
                        error_code: "none".to_string(),
                        error_kind: "none".to_string(),
                        error_status: "none".to_string(),
                        error_operation: "none".to_string(),
                    },
                ],
                ..NativeDataFusionExecutionMetrics::default()
            },
        };

        assert_eq!(execution.fields().len(), NativeDataFusionExecutionRow::COLUMNS.len());
        assert_eq!(
            execution.to_tsv_line(),
            concat!(
                "q_exec\tRootExec\tRootExec(Source)\tcandidate_shape\tcandidate_family\taccepted_replaced\tnone\tnone\t",
                "none\tnone\tnone\tNativeCudfExec\tnative\t7\t2\tfalse\t100\t1.500\t2\t4096\t",
                "true\tsucceeded\tnone\t3\t111\t2222\t",
                "10\t11\t12\t13\t14\t15\t16\t17\t18\t19\t20\t21\t22\t23\t24\t25\t26\t27\t28\t29\t30\t31\t",
                "none\tnone\tnone\tnone\t32\t33\t34\t35\t36\t37\t38\t39\t40\t41\t42\t43\t44\t45\t",
                "mixed_post_decode\tmeasured_source_pruned\tsource_read_pruned\tnone\t",
                "46\t47\t48\t49\t50\t51\t52\t53\t54\t55\t56\t57\t58\t59\t60\t61\t62\t63\t64\t65\t66\t67\t",
                "0\t0\t0\t0\t0\t0\t0\t0\t68\t69\t70\t71\t72\t73\t74\t75\t76\t77\t78\t79\t80\t81\t82\t83\t",
                "84\t85\t86\t87\t88\t89\t90\t91\t92\tnone\tnone\t93\t94\t1.000\t2.500\t3.750\t4.125\tnone\t",
                "outcome=completed,bytes=4096,elapsed_ns=9000000,cache_path_hash=cache123,stable_id_hash=stable456,",
                "file_index=0,chunk_index=2,error_code=none,error_kind=none,error_status=none,error_operation=none",
                "\tnone"
            )
        );
    }
}
