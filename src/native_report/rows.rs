//! Row structs for the native planning/execution/fallback-pressure TSV surfaces.
//!
//! Top-level optional fields serialize as `none`. Nested source-diagnostics
//! values use their own sentinels (e.g. `unknown`, `all`) documented on each
//! field. Reservation requested/granted byte fields are cumulative admission
//! accounting over the query lifetime; live peak and retained-byte fields
//! describe instantaneous resident device memory.

#![deny(missing_docs)]

use nexus_query_engine::{AdaptivePipelineReportRow, StrictBoundedContractGateRow};

use super::metrics::NativeDataFusionExecutionMetrics;
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
    pub native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics,
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
