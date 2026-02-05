use cudf_query_engine::StrictBoundedContractGateRow;

use super::metrics::NativeDataFusionExecutionMetrics;
use crate::planner::NativeStrictBoundedCompiledGraphDiagnostics;

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
    pub strict_bounded_contract_rows: Vec<StrictBoundedContractGateRow>,
    pub strict_bounded_compiled_graphs: Vec<NativeStrictBoundedCompiledGraphDiagnostics>,
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
    pub strict_bounded_contract_rows: Vec<StrictBoundedContractGateRow>,
    pub strict_bounded_compiled_graphs: Vec<NativeStrictBoundedCompiledGraphDiagnostics>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDataFusionPostNativeExecutionPath {
    Native,
    PartialNative,
    Cpu,
}

impl NativeDataFusionPostNativeExecutionPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::PartialNative => "partial_native",
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
    pub scan_limit_disposition: String,
    pub schema_mapping_disposition: String,
    pub read_column_count: usize,
    pub projected_output_column_count: usize,
    pub pure_filter_column_count: usize,
    pub rejected_features: Vec<String>,
}
