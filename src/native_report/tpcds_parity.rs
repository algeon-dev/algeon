//! TPC-DS parity report contracts.
//!
//! This module compares CPU DataFusion output, native GPU output, and optional
//! external-oracle output for the same TPC-DS query. The TSV surface is stable
//! within [`NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION`]: column order comes from
//! [`NATIVE_TPCDS_PARITY_TSV_HEADER`], missing optional values serialize as
//! `none`, and status enums serialize through their `as_str` methods.
//!
//! Output fingerprints use a stable FNV-1a hash over schema metadata and
//! canonicalized row values. They are diagnostic identifiers, not
//! cryptographic hashes. Unordered comparisons sort canonical rows before
//! comparing values; ordered comparisons preserve batch row order.

use std::{collections::HashMap, fmt::Write, sync::Arc};

use arrow::{
    array::{
        Array, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array, Decimal128Array, Decimal256Array,
        Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        StringArray, StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Schema, SchemaRef, TimeUnit, i256},
    record_batch::RecordBatch,
};
use datafusion::physical_plan::{ExecutionPlan, metrics::MetricsSet};

use super::{
    NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome, NativeDataFusionPostNativeExecutionPath,
    NativeDataFusionRuntimeOutcome, NativePlanningFailure, WorkloadDatasetMetadata,
    diagnostics::{post_native_plan_coverage, unsupported_category},
    tsv::{stable_tsv_field, write_workload_dataset_metadata},
};
use crate::planner::NativePlanningDiagnostics;

/// TPC-DS parity TSV schema version.
///
/// Bump this when a column is added, removed, renamed, reordered, or given a
/// new meaning that would change downstream parsing.
pub const NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION: u32 = 3;
/// TPC-DS parity TSV header. Column order is stable for schema version 3.
pub const NATIVE_TPCDS_PARITY_TSV_HEADER: &str = "query_id\tcpu_planning_status\tcpu_runtime_status\tnative_planning_outcome\tnative_reason_code\tnative_unsupported_category\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\tnative_runtime_started\tnative_runtime_outcome\tparity_status\tparity_detail\tcpu_output_rows\tnative_output_rows\tcpu_output_batches\tnative_output_batches\tcpu_output_fingerprint\tnative_output_fingerprint\tcomparison_mode\tduckdb_output_status\tduckdb_cpu_parity_status\tduckdb_cpu_parity_detail\tduckdb_output_rows\tduckdb_output_batches\tduckdb_output_fingerprint\texternal-engine_output_status\texternal-engine_cpu_parity_status\texternal-engine_cpu_parity_detail\texternal-engine_output_rows\texternal-engine_output_batches\texternal-engine_output_fingerprint\tnative_reservation_live_peak_bytes\tnative_reservation_denied_count\tnative_retry_count\tnative_runtime_failure";

/// Stable top-level parity outcome for one query row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityStatus {
    /// CPU and native or oracle outputs matched under the selected comparison mode.
    Matched,
    /// Both sides ran, but schema, row count, or row values differed.
    Mismatched,
    /// The CPU DataFusion baseline did not produce comparable output.
    CpuFailed,
    /// Native planning rejected the query before native runtime started.
    NativeRejected,
    /// Native runtime started but failed before producing comparable output.
    NativeFailed,
    /// Comparison was intentionally skipped because required inputs were missing.
    NotCompared,
}

impl ParityStatus {
    /// Stable TSV value for this status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::Mismatched => "mismatched",
            Self::CpuFailed => "cpu_failed",
            Self::NativeRejected => "native_rejected",
            Self::NativeFailed => "native_failed",
            Self::NotCompared => "not_compared",
        }
    }
}

/// Row-order and float-tolerance mode used for output comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityComparisonMode {
    /// Compare canonical rows in emitted order with exact value equality.
    OrderedExact,
    /// Sort canonical rows before exact value comparison.
    UnorderedExact,
    /// Compare emitted rows in order, allowing configured float tolerance.
    OrderedFloatTolerant,
    /// Sort canonical rows, then allow configured float tolerance.
    UnorderedFloatTolerant,
    /// No record-batch comparison was attempted for this row.
    NotApplicable,
}

impl ParityComparisonMode {
    /// Stable TSV value for this comparison mode.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OrderedExact => "ordered_exact",
            Self::UnorderedExact => "unordered_exact",
            Self::OrderedFloatTolerant => "ordered_float_tolerant",
            Self::UnorderedFloatTolerant => "unordered_float_tolerant",
            Self::NotApplicable => "not_applicable",
        }
    }

    /// Selects the report mode from SQL ordering and float-tolerance evidence.
    pub const fn select(ordered: bool, float_tolerant: bool) -> Self {
        match (ordered, float_tolerant) {
            (true, false) => Self::OrderedExact,
            (false, false) => Self::UnorderedExact,
            (true, true) => Self::OrderedFloatTolerant,
            (false, true) => Self::UnorderedFloatTolerant,
        }
    }
}

/// Absolute and relative tolerance for floating-point parity checks.
///
/// A zero/zero tolerance keeps comparisons exact. Non-zero tolerances apply
/// only when the compared schema contains `Float32` or `Float64` columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityFloatTolerance {
    /// Absolute error bound accepted before relative tolerance is considered.
    pub absolute: f64,
    /// Relative error bound multiplied by the larger absolute input value.
    pub relative: f64,
}

impl ParityFloatTolerance {
    /// Exact comparison with no float tolerance.
    pub const fn exact() -> Self {
        Self { absolute: 0.0, relative: 0.0 }
    }

    /// Builds a tolerance from absolute and relative error bounds.
    pub const fn new(absolute: f64, relative: f64) -> Self {
        Self { absolute, relative }
    }

    /// Returns true when either tolerance component is non-zero.
    pub fn is_enabled(self) -> bool {
        self.absolute > 0.0 || self.relative > 0.0
    }
}

impl Default for ParityFloatTolerance {
    fn default() -> Self {
        Self::exact()
    }
}

/// One row in the TPC-DS parity TSV.
///
/// Optional string and count fields serialize as `none`. Count fields are rows
/// or batches unless the field name says `bytes`; native reservation fields
/// are device-byte runtime metrics projected from the native execution plan.
#[derive(Debug, Clone)]
pub struct NativeTpcdsParityRow {
    /// Stable TPC-DS query id such as `q01`.
    pub query_id: String,
    /// CPU DataFusion SQL/physical planning status.
    pub cpu_planning_status: ParityStageStatus,
    /// CPU DataFusion runtime status after physical planning succeeds.
    pub cpu_runtime_status: ParityStageStatus,
    /// Native planning outcome, including explicit planning failures.
    pub native_planning_outcome: NativeTpcdsParityPlanningOutcome,
    /// Native fallback or planning reason code; `None` serializes as `none`.
    pub native_reason_code: Option<String>,
    /// Coarse unsupported category for native rejection; `None` serializes as `none`.
    pub native_unsupported_category: Option<String>,
    /// Native coverage of the post-optimizer DataFusion plan.
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    /// Number of native plan fragments visible after native planning.
    pub native_plan_fragment_count: usize,
    /// Number of host DataFusion boundaries around native fragments.
    pub host_datafusion_boundary_count: usize,
    /// Whether native runtime was invoked for this query.
    pub native_runtime_started: bool,
    /// Native runtime status after planning.
    pub native_runtime_outcome: NativeDataFusionRuntimeOutcome,
    /// CPU-vs-native parity result.
    pub parity_status: ParityStatus,
    /// Human-readable mismatch or skip detail; escaped for TSV output.
    pub parity_detail: Option<String>,
    /// CPU output row count, or `none` when CPU output was unavailable.
    pub cpu_output_rows: Option<usize>,
    /// Native output row count, or `none` when native output was unavailable.
    pub native_output_rows: Option<usize>,
    /// CPU output batch count, or `none` when CPU output was unavailable.
    pub cpu_output_batches: Option<usize>,
    /// Native output batch count, or `none` when native output was unavailable.
    pub native_output_batches: Option<usize>,
    /// Stable diagnostic hash of canonical CPU output.
    pub cpu_output_fingerprint: Option<String>,
    /// Stable diagnostic hash of canonical native output.
    pub native_output_fingerprint: Option<String>,
    /// Comparison mode chosen from SQL ordering and float tolerance.
    pub comparison_mode: ParityComparisonMode,
    /// DuckDB oracle output and CPU parity summary.
    pub duckdb_output: NativeTpcdsParityExternalOutputSummary,
    /// external-engine oracle output and CPU parity summary.
    pub external-engine_output: NativeTpcdsParityExternalOutputSummary,
    /// Native reservation live peak in device bytes.
    pub native_reservation_live_peak_bytes: Option<usize>,
    /// Number of native reservation denials recorded during runtime.
    pub native_reservation_denied_count: Option<usize>,
    /// Number of native retry attempts recorded during runtime.
    pub native_retry_count: Option<usize>,
    /// CPU planning/runtime or native planning/runtime failure detail.
    pub native_runtime_failure: Option<String>,
}

/// Planning/runtime stage status used by the parity row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityStageStatus {
    /// Stage completed successfully.
    Succeeded,
    /// Stage failed.
    Failed,
    /// Stage did not start.
    NotStarted,
}

impl ParityStageStatus {
    /// Stable TSV value for this stage status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::NotStarted => "not_started",
        }
    }
}

/// Native planning outcome vocabulary for the TPC-DS parity report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTpcdsParityPlanningOutcome {
    /// Native planning accepted and replaced at least one candidate.
    AcceptedReplaced,
    /// Native planning rejected the candidate before execution.
    RejectedBeforeExecution,
    /// Native planning itself failed.
    PlanningFailed,
}

impl NativeTpcdsParityPlanningOutcome {
    /// Stable TSV value for this planning outcome.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AcceptedReplaced => "accepted_replaced",
            Self::RejectedBeforeExecution => "rejected_before_execution",
            Self::PlanningFailed => "planning_failed",
        }
    }
}

/// Execution status for an optional external TPC-DS oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTpcdsParityExternalOutputStatus {
    /// External oracle was not configured.
    NotConfigured,
    /// External oracle completed successfully.
    Succeeded,
    /// External oracle failed.
    Failed,
}

impl NativeTpcdsParityExternalOutputStatus {
    /// Stable TSV value for this external-output status.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotConfigured => "not_configured",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

/// External oracle output summary projected into one parity TSV row.
///
/// `cpu_parity_*` fields compare the oracle output against CPU DataFusion, not
/// against native GPU output. Optional counts and fingerprints serialize as
/// `none` when the oracle is not configured or did not produce output.
#[derive(Debug, Clone)]
pub struct NativeTpcdsParityExternalOutputSummary {
    /// Whether the external oracle was configured and produced output.
    pub status: NativeTpcdsParityExternalOutputStatus,
    /// Oracle-vs-CPU parity result.
    pub cpu_parity_status: ParityStatus,
    /// Oracle-vs-CPU mismatch or execution detail.
    pub cpu_parity_detail: Option<String>,
    /// Oracle output row count.
    pub output_rows: Option<usize>,
    /// Oracle output batch count.
    pub output_batches: Option<usize>,
    /// Stable diagnostic hash of canonical oracle output.
    pub output_fingerprint: Option<String>,
}

impl NativeTpcdsParityExternalOutputSummary {
    /// Summary used when no external oracle is configured.
    pub fn not_configured() -> Self {
        Self {
            status: NativeTpcdsParityExternalOutputStatus::NotConfigured,
            cpu_parity_status: ParityStatus::NotCompared,
            cpu_parity_detail: None,
            output_rows: None,
            output_batches: None,
            output_fingerprint: None,
        }
    }
}

impl From<NativeDataFusionPlanningOutcome> for NativeTpcdsParityPlanningOutcome {
    fn from(value: NativeDataFusionPlanningOutcome) -> Self {
        match value {
            NativeDataFusionPlanningOutcome::AcceptedReplaced => Self::AcceptedReplaced,
            NativeDataFusionPlanningOutcome::RejectedBeforeExecution => Self::RejectedBeforeExecution,
        }
    }
}

/// In-memory TPC-DS parity report.
///
/// `to_tsv` writes schema version, float tolerance, optional dataset metadata,
/// and then rows using [`NATIVE_TPCDS_PARITY_TSV_HEADER`].
#[derive(Debug, Clone)]
pub struct NativeTpcdsParityReport {
    /// Per-query rows in caller-provided workload order.
    pub rows: Vec<NativeTpcdsParityRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
    /// Float tolerance recorded in the TSV header and used for comparisons.
    pub float_tolerance: ParityFloatTolerance,
}

impl NativeTpcdsParityReport {
    /// Builds a report from already-derived rows.
    pub fn from_rows(rows: Vec<NativeTpcdsParityRow>, float_tolerance: ParityFloatTolerance) -> Self {
        Self { rows, workload_dataset: None, float_tolerance }
    }

    /// Builds report rows from captured CPU/native/oracle workload outcomes.
    pub fn collect(
        items: impl IntoIterator<Item = NativeTpcdsParityWorkloadItem>,
        float_tolerance: ParityFloatTolerance,
    ) -> Self {
        let rows = items.into_iter().map(|item| parity_row_for_item(item, float_tolerance)).collect();
        Self { rows, workload_dataset: None, float_tolerance }
    }

    /// Attaches fixture metadata to the TSV header.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of rows whose CPU and native outputs matched.
    pub fn matched_count(&self) -> usize {
        self.rows.iter().filter(|row| row.parity_status == ParityStatus::Matched).count()
    }

    /// Number of rows whose CPU and native outputs were compared and differed.
    pub fn mismatched_count(&self) -> usize {
        self.rows.iter().filter(|row| row.parity_status == ParityStatus::Mismatched).count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "# native_tpcds_parity_tsv_schema_version={NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION}");
        let _ = writeln!(output, "# parity_float_tolerance_absolute={}", self.float_tolerance.absolute);
        let _ = writeln!(output, "# parity_float_tolerance_relative={}", self.float_tolerance.relative);
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        output.push_str(NATIVE_TPCDS_PARITY_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                row.query_id,
                row.cpu_planning_status.as_str(),
                row.cpu_runtime_status.as_str(),
                row.native_planning_outcome.as_str(),
                row.native_reason_code.as_deref().unwrap_or("none"),
                row.native_unsupported_category.as_deref().unwrap_or("none"),
                row.post_native_execution_path.as_str(),
                row.native_plan_fragment_count,
                row.host_datafusion_boundary_count,
                row.native_runtime_started,
                row.native_runtime_outcome.as_str(),
                row.parity_status.as_str(),
                row.parity_detail.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
                format_optional_usize(row.cpu_output_rows),
                format_optional_usize(row.native_output_rows),
                format_optional_usize(row.cpu_output_batches),
                format_optional_usize(row.native_output_batches),
                row.cpu_output_fingerprint.as_deref().unwrap_or("none"),
                row.native_output_fingerprint.as_deref().unwrap_or("none"),
                row.comparison_mode.as_str(),
                row.duckdb_output.status.as_str(),
                row.duckdb_output.cpu_parity_status.as_str(),
                row.duckdb_output
                    .cpu_parity_detail
                    .as_deref()
                    .map(stable_tsv_field)
                    .unwrap_or_else(|| "none".to_string()),
                format_optional_usize(row.duckdb_output.output_rows),
                format_optional_usize(row.duckdb_output.output_batches),
                row.duckdb_output.output_fingerprint.as_deref().unwrap_or("none"),
                row.external-engine_output.status.as_str(),
                row.external-engine_output.cpu_parity_status.as_str(),
                row.external-engine_output
                    .cpu_parity_detail
                    .as_deref()
                    .map(stable_tsv_field)
                    .unwrap_or_else(|| "none".to_string()),
                format_optional_usize(row.external-engine_output.output_rows),
                format_optional_usize(row.external-engine_output.output_batches),
                row.external-engine_output.output_fingerprint.as_deref().unwrap_or("none"),
                format_optional_usize(row.native_reservation_live_peak_bytes),
                format_optional_usize(row.native_reservation_denied_count),
                format_optional_usize(row.native_retry_count),
                row.native_runtime_failure.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
            );
        }
        output
    }
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

/// Workload-item input for the parity report: paired CPU + native execution
/// outcomes for one TPC-DS query.
#[derive(Debug, Clone)]
pub struct NativeTpcdsParityWorkloadItem {
    /// Stable TPC-DS query id such as `q01`.
    pub query_id: String,
    /// SQL text used to choose ordered versus unordered comparison.
    pub sql: String,
    /// CPU DataFusion baseline outcome.
    pub cpu_outcome: NativeTpcdsParityCpuOutcome,
    /// Native optimizer outcome and retained diagnostics.
    pub native_planning: NativeTpcdsParityNativePlanningOutcome,
    /// Native runtime outcome when planning accepted a native fragment.
    pub native_runtime: NativeTpcdsParityNativeRuntimeOutcome,
    /// Optional DuckDB oracle output.
    pub duckdb_output: NativeTpcdsParityExternalOutput,
    /// Optional external-engine oracle output.
    pub external-engine_output: NativeTpcdsParityExternalOutput,
}

/// Captured CPU DataFusion baseline outcome for one parity query.
#[derive(Debug, Clone)]
pub enum NativeTpcdsParityCpuOutcome {
    /// CPU baseline produced Arrow batches under the supplied schema.
    Succeeded {
        /// Output schema produced by CPU DataFusion.
        schema: SchemaRef,
        /// Output batches produced by CPU DataFusion.
        batches: Vec<RecordBatch>,
    },
    /// CPU planning failed before execution started.
    PlanningFailed(NativePlanningFailure),
    /// CPU execution failed after physical planning succeeded.
    ExecutionFailed {
        /// Human-facing execution failure message.
        message: String,
    },
}

/// Captured native planning outcome for one parity query.
#[derive(Debug, Clone)]
pub enum NativeTpcdsParityNativePlanningOutcome {
    /// Native planning completed; `accepted` records whether a native fragment
    /// was present in the optimized plan.
    Planned {
        /// Whether the optimized plan contains a native fragment.
        accepted: bool,
        /// DataFusion physical plan after native planning.
        planned_plan: Arc<dyn ExecutionPlan>,
        /// Native planning diagnostics retained for report rows.
        diagnostics: Box<NativePlanningDiagnostics>,
    },
    /// Native SQL, physical, or diagnostic planning failed.
    PlanningFailed(NativePlanningFailure),
}

/// Captured native runtime outcome for one parity query.
#[derive(Debug, Clone)]
pub enum NativeTpcdsParityNativeRuntimeOutcome {
    /// Runtime was not invoked, usually because native planning rejected the query.
    NotStarted,
    /// Runtime produced Arrow batches and DataFusion metrics.
    Succeeded {
        /// Output schema produced by native execution.
        schema: SchemaRef,
        /// Output batches produced by native execution.
        batches: Vec<RecordBatch>,
        /// DataFusion metrics collected from the native execution plan.
        metrics: MetricsSet,
    },
    /// Runtime started but failed before producing comparable output.
    Failed {
        /// Human-facing runtime failure message.
        message: String,
    },
}

/// Captured output from an optional external TPC-DS oracle.
#[derive(Debug, Clone)]
pub enum NativeTpcdsParityExternalOutput {
    /// No binary/config was provided for this oracle.
    NotConfigured,
    /// Oracle produced Arrow batches after reading its Parquet result.
    Succeeded {
        /// Output schema produced by the external oracle.
        schema: SchemaRef,
        /// Output batches produced by the external oracle.
        batches: Vec<RecordBatch>,
    },
    /// Oracle execution or output reading failed.
    Failed {
        /// Human-facing oracle failure message.
        message: String,
    },
}

fn parity_row_for_item(
    item: NativeTpcdsParityWorkloadItem,
    float_tolerance: ParityFloatTolerance,
) -> NativeTpcdsParityRow {
    let ordered = sql_has_top_level_order_by(&item.sql);

    let (cpu_planning_status, cpu_runtime_status, cpu_batches, cpu_schema, cpu_runtime_failure) = match item.cpu_outcome
    {
        NativeTpcdsParityCpuOutcome::Succeeded { schema, batches } => {
            (ParityStageStatus::Succeeded, ParityStageStatus::Succeeded, Some(batches), Some(schema), None)
        },
        NativeTpcdsParityCpuOutcome::PlanningFailed(failure) => {
            (ParityStageStatus::Failed, ParityStageStatus::NotStarted, None, None, Some(failure.message))
        },
        NativeTpcdsParityCpuOutcome::ExecutionFailed { message } => {
            (ParityStageStatus::Succeeded, ParityStageStatus::Failed, None, None, Some(message))
        },
    };

    let (
        native_planning_outcome,
        native_reason_code,
        native_unsupported_category,
        post_native_execution_path,
        native_plan_fragment_count,
        host_datafusion_boundary_count,
        native_planning_failure_message,
    ) = match &item.native_planning {
        NativeTpcdsParityNativePlanningOutcome::Planned { accepted, planned_plan, diagnostics } => {
            let (reason_code, unsupported_category) =
                if *accepted { (None, None) } else { rejected_diagnostics(planned_plan.as_ref(), diagnostics) };
            let outcome = if *accepted {
                NativeTpcdsParityPlanningOutcome::AcceptedReplaced
            } else {
                NativeTpcdsParityPlanningOutcome::RejectedBeforeExecution
            };
            let coverage = post_native_plan_coverage(planned_plan.as_ref());
            (
                outcome,
                reason_code,
                unsupported_category,
                coverage.execution_path,
                coverage.native_fragment_count,
                coverage.host_datafusion_boundary_count,
                None,
            )
        },
        NativeTpcdsParityNativePlanningOutcome::PlanningFailed(failure) => (
            NativeTpcdsParityPlanningOutcome::PlanningFailed,
            Some(failure.reason_code.clone()),
            Some("planning".to_string()),
            NativeDataFusionPostNativeExecutionPath::Cpu,
            0,
            0,
            Some(failure.message.clone()),
        ),
    };

    let (native_runtime_started, native_runtime_outcome, native_batches, native_schema, native_metrics, native_failure) =
        match item.native_runtime {
            NativeTpcdsParityNativeRuntimeOutcome::NotStarted => (
                false,
                NativeDataFusionRuntimeOutcome::NotStarted,
                None,
                None,
                None,
                native_planning_failure_message.clone(),
            ),
            NativeTpcdsParityNativeRuntimeOutcome::Succeeded { schema, batches, metrics } => {
                (true, NativeDataFusionRuntimeOutcome::Succeeded, Some(batches), Some(schema), Some(metrics), None)
            },
            NativeTpcdsParityNativeRuntimeOutcome::Failed { message } => {
                (true, NativeDataFusionRuntimeOutcome::Failed, None, None, None, Some(message))
            },
        };

    let native_metrics_summary = native_metrics.as_ref().map(NativeDataFusionExecutionMetrics::from_metrics);

    let (parity_status, comparison_mode, parity_detail, cpu_fingerprint, native_fingerprint) =
        derive_parity_outcome(ParityInputs {
            cpu_planning_status,
            cpu_runtime_status,
            native_planning_outcome,
            native_runtime_started,
            native_runtime_outcome,
            cpu_schema: cpu_schema.as_ref(),
            cpu_batches: cpu_batches.as_deref(),
            native_schema: native_schema.as_ref(),
            native_batches: native_batches.as_deref(),
            ordered,
            float_tolerance,
        });
    let duckdb_output = summarize_external_output(
        item.duckdb_output,
        cpu_schema.as_ref(),
        cpu_batches.as_deref(),
        ordered,
        float_tolerance,
    );
    let external-engine_output = summarize_external_output(
        item.external-engine_output,
        cpu_schema.as_ref(),
        cpu_batches.as_deref(),
        ordered,
        float_tolerance,
    );

    NativeTpcdsParityRow {
        query_id: item.query_id,
        cpu_planning_status,
        cpu_runtime_status,
        native_planning_outcome,
        native_reason_code,
        native_unsupported_category,
        post_native_execution_path,
        native_plan_fragment_count,
        host_datafusion_boundary_count,
        native_runtime_started,
        native_runtime_outcome,
        parity_status,
        parity_detail,
        cpu_output_rows: cpu_batches.as_ref().map(|batches| total_rows(batches)),
        native_output_rows: native_batches.as_ref().map(|batches| total_rows(batches)),
        cpu_output_batches: cpu_batches.as_ref().map(Vec::len),
        native_output_batches: native_batches.as_ref().map(Vec::len),
        cpu_output_fingerprint: cpu_fingerprint,
        native_output_fingerprint: native_fingerprint,
        comparison_mode,
        duckdb_output,
        external-engine_output,
        native_reservation_live_peak_bytes: native_metrics_summary.as_ref().map(|m| m.reservation_live_peak_bytes),
        native_reservation_denied_count: native_metrics_summary.as_ref().map(|m| m.reservation_denied_count),
        native_retry_count: native_metrics_summary.as_ref().map(|m| m.retry_count),
        native_runtime_failure: cpu_runtime_failure.or(native_failure),
    }
}

fn summarize_external_output(
    output: NativeTpcdsParityExternalOutput,
    cpu_schema: Option<&SchemaRef>,
    cpu_batches: Option<&[RecordBatch]>,
    ordered: bool,
    float_tolerance: ParityFloatTolerance,
) -> NativeTpcdsParityExternalOutputSummary {
    match output {
        NativeTpcdsParityExternalOutput::NotConfigured => NativeTpcdsParityExternalOutputSummary::not_configured(),
        NativeTpcdsParityExternalOutput::Failed { message } => NativeTpcdsParityExternalOutputSummary {
            status: NativeTpcdsParityExternalOutputStatus::Failed,
            cpu_parity_status: ParityStatus::NotCompared,
            cpu_parity_detail: Some(message),
            output_rows: None,
            output_batches: None,
            output_fingerprint: None,
        },
        NativeTpcdsParityExternalOutput::Succeeded { schema, batches } => {
            let output_rows = Some(total_rows(&batches));
            let output_batches = Some(batches.len());
            let output_fingerprint = Some(fingerprint_batches(schema.as_ref(), &batches));
            let Some(cpu_schema) = cpu_schema else {
                return NativeTpcdsParityExternalOutputSummary {
                    status: NativeTpcdsParityExternalOutputStatus::Succeeded,
                    cpu_parity_status: ParityStatus::NotCompared,
                    cpu_parity_detail: Some("missing CPU schema".to_string()),
                    output_rows,
                    output_batches,
                    output_fingerprint,
                };
            };
            let Some(cpu_batches) = cpu_batches else {
                return NativeTpcdsParityExternalOutputSummary {
                    status: NativeTpcdsParityExternalOutputStatus::Succeeded,
                    cpu_parity_status: ParityStatus::NotCompared,
                    cpu_parity_detail: Some("missing CPU batches".to_string()),
                    output_rows,
                    output_batches,
                    output_fingerprint,
                };
            };
            let comparison = compare_record_batches_with_options(
                cpu_schema.as_ref(),
                cpu_batches,
                schema.as_ref(),
                &batches,
                ordered,
                float_tolerance,
                ParityCompareOptions::cross_engine_oracle(),
            );
            NativeTpcdsParityExternalOutputSummary {
                status: NativeTpcdsParityExternalOutputStatus::Succeeded,
                cpu_parity_status: comparison.status,
                cpu_parity_detail: comparison.detail,
                output_rows,
                output_batches,
                output_fingerprint: Some(comparison.native_fingerprint),
            }
        },
    }
}

fn rejected_diagnostics(
    plan: &dyn ExecutionPlan,
    diagnostics: &NativePlanningDiagnostics,
) -> (Option<String>, Option<String>) {
    for outcome in diagnostics.report.outcomes() {
        if let crate::report::CandidateOutcome::Rejected { candidate, reason } = outcome {
            return (Some(reason.code().to_string()), unsupported_category(plan, candidate, *reason));
        }
    }
    (None, None)
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

struct ParityInputs<'a> {
    cpu_planning_status: ParityStageStatus,
    cpu_runtime_status: ParityStageStatus,
    native_planning_outcome: NativeTpcdsParityPlanningOutcome,
    native_runtime_started: bool,
    native_runtime_outcome: NativeDataFusionRuntimeOutcome,
    cpu_schema: Option<&'a SchemaRef>,
    cpu_batches: Option<&'a [RecordBatch]>,
    native_schema: Option<&'a SchemaRef>,
    native_batches: Option<&'a [RecordBatch]>,
    ordered: bool,
    float_tolerance: ParityFloatTolerance,
}

fn derive_parity_outcome(
    inputs: ParityInputs<'_>,
) -> (ParityStatus, ParityComparisonMode, Option<String>, Option<String>, Option<String>) {
    if inputs.cpu_planning_status != ParityStageStatus::Succeeded
        || inputs.cpu_runtime_status != ParityStageStatus::Succeeded
    {
        return (ParityStatus::CpuFailed, ParityComparisonMode::NotApplicable, None, None, None);
    }
    match inputs.native_planning_outcome {
        NativeTpcdsParityPlanningOutcome::RejectedBeforeExecution
        | NativeTpcdsParityPlanningOutcome::PlanningFailed => {
            return (ParityStatus::NativeRejected, ParityComparisonMode::NotApplicable, None, None, None);
        },
        NativeTpcdsParityPlanningOutcome::AcceptedReplaced => {},
    }
    if !inputs.native_runtime_started {
        return (ParityStatus::NativeRejected, ParityComparisonMode::NotApplicable, None, None, None);
    }
    if inputs.native_runtime_outcome != NativeDataFusionRuntimeOutcome::Succeeded {
        return (ParityStatus::NativeFailed, ParityComparisonMode::NotApplicable, None, None, None);
    }
    let (Some(cpu_schema), Some(cpu_batches), Some(native_schema), Some(native_batches)) =
        (inputs.cpu_schema, inputs.cpu_batches, inputs.native_schema, inputs.native_batches)
    else {
        return (
            ParityStatus::NotCompared,
            ParityComparisonMode::NotApplicable,
            Some("missing comparison inputs".to_string()),
            None,
            None,
        );
    };
    let result = compare_record_batches(
        cpu_schema.as_ref(),
        cpu_batches,
        native_schema.as_ref(),
        native_batches,
        inputs.ordered,
        inputs.float_tolerance,
    );
    (result.status, result.mode, result.detail, Some(result.cpu_fingerprint), Some(result.native_fingerprint))
}

/// Result of comparing CPU and native Arrow outputs.
#[derive(Debug, Clone)]
pub struct ParityComparisonResult {
    /// Match, mismatch, or skip status.
    pub status: ParityStatus,
    /// Row-order and float-tolerance mode used for comparison.
    pub mode: ParityComparisonMode,
    /// Mismatch or skip detail when available.
    pub detail: Option<String>,
    /// Stable diagnostic hash of canonical CPU output.
    pub cpu_fingerprint: String,
    /// Stable diagnostic hash of canonical native output.
    pub native_fingerprint: String,
}

/// Compares CPU and native Arrow output using TPC-DS parity rules.
///
/// Schema differences are reported before row comparison. When `ordered` is
/// false, canonical rows are sorted before comparison. Float tolerance applies
/// only to float columns and is recorded in the returned comparison mode.
pub fn compare_record_batches(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordered: bool,
    tolerance: ParityFloatTolerance,
) -> ParityComparisonResult {
    compare_record_batches_with_options(
        cpu_schema,
        cpu_batches,
        native_schema,
        native_batches,
        ordered,
        tolerance,
        ParityCompareOptions::strict_native(),
    )
}

#[derive(Debug, Clone, Copy)]
struct ParityCompareOptions {
    allow_decimal_precision_mismatch: bool,
}

impl ParityCompareOptions {
    const fn strict_native() -> Self {
        Self { allow_decimal_precision_mismatch: false }
    }

    const fn cross_engine_oracle() -> Self {
        Self { allow_decimal_precision_mismatch: true }
    }
}

fn compare_record_batches_with_options(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordered: bool,
    tolerance: ParityFloatTolerance,
    options: ParityCompareOptions,
) -> ParityComparisonResult {
    let schema_diff = compare_schemas(cpu_schema, native_schema, options);
    let float_tolerant = tolerance.is_enabled() && schema_has_float_column(cpu_schema);
    let mode = ParityComparisonMode::select(ordered, float_tolerant);

    if let Some(detail) = schema_diff {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(detail),
            cpu_fingerprint: fingerprint_schema_only(cpu_schema),
            native_fingerprint: fingerprint_schema_only(native_schema),
        };
    }

    let cpu_rows = match collect_canonical_rows(cpu_schema, cpu_batches) {
        Ok(rows) => rows,
        Err(detail) => {
            return ParityComparisonResult {
                status: ParityStatus::NotCompared,
                mode,
                detail: Some(format!("cpu canonicalization failed: {detail}")),
                cpu_fingerprint: fingerprint_schema_only(cpu_schema),
                native_fingerprint: fingerprint_schema_only(native_schema),
            };
        },
    };
    let native_rows = match collect_canonical_rows(native_schema, native_batches) {
        Ok(rows) => rows,
        Err(detail) => {
            return ParityComparisonResult {
                status: ParityStatus::NotCompared,
                mode,
                detail: Some(format!("native canonicalization failed: {detail}")),
                cpu_fingerprint: fingerprint_rows(cpu_schema, &cpu_rows),
                native_fingerprint: fingerprint_schema_only(native_schema),
            };
        },
    };

    let cpu_fingerprint = fingerprint_rows(cpu_schema, &cpu_rows);
    let native_fingerprint = fingerprint_rows(native_schema, &native_rows);

    if cpu_rows.len() != native_rows.len() {
        return ParityComparisonResult {
            status: ParityStatus::Mismatched,
            mode,
            detail: Some(format!("row count differs: cpu={} native={}", cpu_rows.len(), native_rows.len())),
            cpu_fingerprint,
            native_fingerprint,
        };
    }

    let (cpu_aligned, native_aligned) = if ordered {
        (cpu_rows, native_rows)
    } else {
        let mut cpu = cpu_rows;
        let mut native = native_rows;
        cpu.sort();
        native.sort();
        (cpu, native)
    };

    for (index, (cpu_row, native_row)) in cpu_aligned.iter().zip(native_aligned.iter()).enumerate() {
        if let Some(detail) =
            compare_canonical_rows(cpu_schema, cpu_row, native_row, tolerance, float_tolerant, options)
        {
            return ParityComparisonResult {
                status: ParityStatus::Mismatched,
                mode,
                detail: Some(format!("row {index}: {detail}")),
                cpu_fingerprint,
                native_fingerprint,
            };
        }
    }

    ParityComparisonResult { status: ParityStatus::Matched, mode, detail: None, cpu_fingerprint, native_fingerprint }
}

fn compare_schemas(cpu: &Schema, native: &Schema, options: ParityCompareOptions) -> Option<String> {
    if cpu.fields().len() != native.fields().len() {
        return Some(format!(
            "schema field count differs: cpu={} native={}",
            cpu.fields().len(),
            native.fields().len()
        ));
    }
    for (index, (cpu_field, native_field)) in cpu.fields().iter().zip(native.fields().iter()).enumerate() {
        if !logical_data_types_compatible(cpu_field.data_type(), native_field.data_type(), options) {
            return Some(format!(
                "column {} ({}/{}) type differs: cpu={:?} native={:?}",
                index,
                cpu_field.name(),
                native_field.name(),
                cpu_field.data_type(),
                native_field.data_type(),
            ));
        }
    }
    None
}

fn logical_data_types_compatible(cpu: &DataType, native: &DataType, options: ParityCompareOptions) -> bool {
    if cpu == native {
        return true;
    }
    if options.allow_decimal_precision_mismatch {
        match (cpu, native) {
            (DataType::Decimal128(_, cpu_scale), DataType::Decimal128(_, native_scale))
            | (DataType::Decimal256(_, cpu_scale), DataType::Decimal256(_, native_scale))
                if cpu_scale == native_scale =>
            {
                return true;
            },
            _ => {},
        }
    }
    matches!(
        (cpu, native),
        (DataType::Utf8, DataType::Utf8View)
            | (DataType::Utf8View, DataType::Utf8)
            | (DataType::LargeUtf8, DataType::Utf8)
            | (DataType::Utf8, DataType::LargeUtf8)
            | (DataType::LargeUtf8, DataType::Utf8View)
            | (DataType::Utf8View, DataType::LargeUtf8)
            | (DataType::Binary, DataType::BinaryView)
            | (DataType::BinaryView, DataType::Binary)
            | (DataType::LargeBinary, DataType::Binary)
            | (DataType::Binary, DataType::LargeBinary)
            | (DataType::LargeBinary, DataType::BinaryView)
            | (DataType::BinaryView, DataType::LargeBinary)
    )
}

fn schema_has_float_column(schema: &Schema) -> bool {
    schema.fields().iter().any(|field| matches!(field.data_type(), DataType::Float32 | DataType::Float64))
}

fn collect_canonical_rows(schema: &Schema, batches: &[RecordBatch]) -> Result<Vec<CanonicalRow>, String> {
    let mut rows = Vec::new();
    for batch in batches {
        if batch.schema().fields().len() != schema.fields().len() {
            return Err(format!(
                "batch has {} columns but schema has {}",
                batch.schema().fields().len(),
                schema.fields().len()
            ));
        }
        for row_index in 0..batch.num_rows() {
            let mut fields = Vec::with_capacity(schema.fields().len());
            for column_index in 0..schema.fields().len() {
                let array = batch.column(column_index);
                let data_type = schema.field(column_index).data_type();
                let field = canonicalize_value(array.as_ref(), data_type, row_index)
                    .map_err(|err| format!("column {column_index} row {row_index}: {err}"))?;
                fields.push(field);
            }
            rows.push(CanonicalRow { fields });
        }
    }
    Ok(rows)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalRow {
    fields: Vec<CanonicalField>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CanonicalField {
    Null,
    Bool(bool),
    Int { width: u8, value: i128 },
    UInt { width: u8, value: u128 },
    Float { width: u8, bits: u64 },
    Decimal128 { precision: u8, scale: i8, value: i128 },
    Decimal256 { precision: u8, scale: i8, value: [u8; 32] },
    Date32(i32),
    Date64(i64),
    Timestamp { unit: TimeUnitTag, tz: Option<String>, value: i64 },
    Utf8(String),
    Binary(Vec<u8>),
    Unsupported(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TimeUnitTag {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

impl From<TimeUnit> for TimeUnitTag {
    fn from(value: TimeUnit) -> Self {
        match value {
            TimeUnit::Second => Self::Second,
            TimeUnit::Millisecond => Self::Millisecond,
            TimeUnit::Microsecond => Self::Microsecond,
            TimeUnit::Nanosecond => Self::Nanosecond,
        }
    }
}

fn canonicalize_value(array: &dyn Array, data_type: &DataType, row: usize) -> Result<CanonicalField, String> {
    if array.is_null(row) {
        return Ok(CanonicalField::Null);
    }
    match data_type {
        DataType::Boolean => {
            let array = downcast::<BooleanArray>(array, "BooleanArray")?;
            Ok(CanonicalField::Bool(array.value(row)))
        },
        DataType::Int8 => {
            let array = downcast::<Int8Array>(array, "Int8Array")?;
            Ok(CanonicalField::Int { width: 8, value: array.value(row).into() })
        },
        DataType::Int16 => {
            let array = downcast::<Int16Array>(array, "Int16Array")?;
            Ok(CanonicalField::Int { width: 16, value: array.value(row).into() })
        },
        DataType::Int32 => {
            let array = downcast::<Int32Array>(array, "Int32Array")?;
            Ok(CanonicalField::Int { width: 32, value: array.value(row).into() })
        },
        DataType::Int64 => {
            let array = downcast::<Int64Array>(array, "Int64Array")?;
            Ok(CanonicalField::Int { width: 64, value: array.value(row).into() })
        },
        DataType::UInt8 => {
            let array = downcast::<UInt8Array>(array, "UInt8Array")?;
            Ok(CanonicalField::UInt { width: 8, value: array.value(row).into() })
        },
        DataType::UInt16 => {
            let array = downcast::<UInt16Array>(array, "UInt16Array")?;
            Ok(CanonicalField::UInt { width: 16, value: array.value(row).into() })
        },
        DataType::UInt32 => {
            let array = downcast::<UInt32Array>(array, "UInt32Array")?;
            Ok(CanonicalField::UInt { width: 32, value: array.value(row).into() })
        },
        DataType::UInt64 => {
            let array = downcast::<UInt64Array>(array, "UInt64Array")?;
            Ok(CanonicalField::UInt { width: 64, value: array.value(row).into() })
        },
        DataType::Float32 => {
            let array = downcast::<Float32Array>(array, "Float32Array")?;
            Ok(CanonicalField::Float { width: 32, bits: u64::from(array.value(row).to_bits()) })
        },
        DataType::Float64 => {
            let array = downcast::<Float64Array>(array, "Float64Array")?;
            Ok(CanonicalField::Float { width: 64, bits: array.value(row).to_bits() })
        },
        DataType::Decimal128(precision, scale) => {
            let array = downcast::<Decimal128Array>(array, "Decimal128Array")?;
            Ok(CanonicalField::Decimal128 { precision: *precision, scale: *scale, value: array.value(row) })
        },
        DataType::Decimal256(precision, scale) => {
            let array = downcast::<Decimal256Array>(array, "Decimal256Array")?;
            let value: i256 = array.value(row);
            Ok(CanonicalField::Decimal256 { precision: *precision, scale: *scale, value: value.to_le_bytes() })
        },
        DataType::Date32 => {
            let array = downcast::<Date32Array>(array, "Date32Array")?;
            Ok(CanonicalField::Date32(array.value(row)))
        },
        DataType::Date64 => {
            let array = downcast::<Date64Array>(array, "Date64Array")?;
            Ok(CanonicalField::Date64(array.value(row)))
        },
        DataType::Timestamp(unit, tz) => {
            let value = match unit {
                TimeUnit::Second => downcast::<TimestampSecondArray>(array, "TimestampSecondArray")?.value(row),
                TimeUnit::Millisecond => {
                    downcast::<TimestampMillisecondArray>(array, "TimestampMillisecondArray")?.value(row)
                },
                TimeUnit::Microsecond => {
                    downcast::<TimestampMicrosecondArray>(array, "TimestampMicrosecondArray")?.value(row)
                },
                TimeUnit::Nanosecond => {
                    downcast::<TimestampNanosecondArray>(array, "TimestampNanosecondArray")?.value(row)
                },
            };
            Ok(CanonicalField::Timestamp {
                unit: TimeUnitTag::from(*unit),
                tz: tz.as_ref().map(|s| s.to_string()),
                value,
            })
        },
        DataType::Utf8 => {
            let array = downcast::<StringArray>(array, "StringArray")?;
            Ok(CanonicalField::Utf8(array.value(row).to_string()))
        },
        DataType::LargeUtf8 => {
            let array = downcast::<LargeStringArray>(array, "LargeStringArray")?;
            Ok(CanonicalField::Utf8(array.value(row).to_string()))
        },
        DataType::Utf8View => {
            let array = downcast::<StringViewArray>(array, "StringViewArray")?;
            Ok(CanonicalField::Utf8(array.value(row).to_string()))
        },
        DataType::Binary => {
            let array = downcast::<BinaryArray>(array, "BinaryArray")?;
            Ok(CanonicalField::Binary(array.value(row).to_vec()))
        },
        DataType::LargeBinary => {
            let array = downcast::<LargeBinaryArray>(array, "LargeBinaryArray")?;
            Ok(CanonicalField::Binary(array.value(row).to_vec()))
        },
        DataType::BinaryView => {
            let array = downcast::<BinaryViewArray>(array, "BinaryViewArray")?;
            Ok(CanonicalField::Binary(array.value(row).to_vec()))
        },
        other => Ok(CanonicalField::Unsupported(format!("{other:?}"))),
    }
}

fn downcast<'a, T: Array + 'static>(array: &'a dyn Array, name: &str) -> Result<&'a T, String> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| format!("expected {name}"))
}

fn compare_canonical_rows(
    schema: &Schema,
    cpu: &CanonicalRow,
    native: &CanonicalRow,
    tolerance: ParityFloatTolerance,
    float_tolerant: bool,
    options: ParityCompareOptions,
) -> Option<String> {
    if cpu.fields.len() != native.fields.len() {
        return Some(format!("column count differs: cpu={} native={}", cpu.fields.len(), native.fields.len()));
    }
    for (index, (cpu_field, native_field)) in cpu.fields.iter().zip(native.fields.iter()).enumerate() {
        let name = schema.field(index).name();
        if !canonical_field_equal(cpu_field, native_field, tolerance, float_tolerant, options) {
            return Some(format!(
                "column {index} ({name}): cpu={} native={}",
                format_field_debug(cpu_field),
                format_field_debug(native_field),
            ));
        }
    }
    None
}

fn canonical_field_equal(
    cpu: &CanonicalField,
    native: &CanonicalField,
    tolerance: ParityFloatTolerance,
    float_tolerant: bool,
    options: ParityCompareOptions,
) -> bool {
    if !float_tolerant {
        return canonical_field_exact_or_allowed(cpu, native, options);
    }
    match (cpu, native) {
        (CanonicalField::Float { width: w1, bits: b1 }, CanonicalField::Float { width: w2, bits: b2 }) => {
            if w1 != w2 {
                return false;
            }
            let cpu_value = f64::from_bits(*b1);
            let native_value = f64::from_bits(*b2);
            floats_close_enough(cpu_value, native_value, tolerance)
        },
        _ => canonical_field_exact_or_allowed(cpu, native, options),
    }
}

fn canonical_field_exact_or_allowed(
    cpu: &CanonicalField,
    native: &CanonicalField,
    options: ParityCompareOptions,
) -> bool {
    if cpu == native {
        return true;
    }
    if options.allow_decimal_precision_mismatch {
        match (cpu, native) {
            (
                CanonicalField::Decimal128 { scale: cpu_scale, value: cpu_value, .. },
                CanonicalField::Decimal128 { scale: native_scale, value: native_value, .. },
            ) => return cpu_scale == native_scale && cpu_value == native_value,
            (
                CanonicalField::Decimal256 { scale: cpu_scale, value: cpu_value, .. },
                CanonicalField::Decimal256 { scale: native_scale, value: native_value, .. },
            ) => return cpu_scale == native_scale && cpu_value == native_value,
            _ => {},
        }
    }
    false
}

fn floats_close_enough(cpu: f64, native: f64, tolerance: ParityFloatTolerance) -> bool {
    if cpu.is_nan() && native.is_nan() {
        return true;
    }
    if cpu == native {
        return true;
    }
    let diff = (cpu - native).abs();
    if diff <= tolerance.absolute {
        return true;
    }
    if tolerance.relative > 0.0 {
        let scale = cpu.abs().max(native.abs());
        return diff <= tolerance.relative * scale;
    }
    false
}

fn format_field_debug(field: &CanonicalField) -> String {
    match field {
        CanonicalField::Null => "null".to_string(),
        CanonicalField::Bool(value) => value.to_string(),
        CanonicalField::Int { value, .. } => value.to_string(),
        CanonicalField::UInt { value, .. } => value.to_string(),
        CanonicalField::Float { width, bits } => match width {
            32 => format!("{}", f32::from_bits(*bits as u32)),
            _ => format!("{}", f64::from_bits(*bits)),
        },
        CanonicalField::Decimal128 { value, scale, precision } => {
            format!("decimal128({precision},{scale}):{value}")
        },
        CanonicalField::Decimal256 { value, scale, precision } => {
            let mut hex = String::with_capacity(64);
            for byte in value.iter().rev() {
                let _ = write!(hex, "{byte:02x}");
            }
            format!("decimal256({precision},{scale}):0x{hex}")
        },
        CanonicalField::Date32(value) => format!("date32:{value}"),
        CanonicalField::Date64(value) => format!("date64:{value}"),
        CanonicalField::Timestamp { unit, tz, value } => format!("timestamp({unit:?},{tz:?}):{value}"),
        CanonicalField::Utf8(value) => format!("\"{}\"", escape_string(value)),
        CanonicalField::Binary(bytes) => format!("bytes(len={})", bytes.len()),
        CanonicalField::Unsupported(text) => format!("unsupported({text})"),
    }
}

fn escape_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t").replace('\r', "\\r")
}

fn fingerprint_schema_only(schema: &Schema) -> String {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("native-tpcds-parity-v1");
    hash_schema(&mut hasher, schema);
    hasher.update_field("rows=0");
    format!("output-fnv1a64:{:016x}", hasher.finish())
}

fn fingerprint_batches(schema: &Schema, batches: &[RecordBatch]) -> String {
    match collect_canonical_rows(schema, batches) {
        Ok(rows) => fingerprint_rows(schema, &rows),
        Err(_) => fingerprint_schema_only(schema),
    }
}

fn fingerprint_rows(schema: &Schema, rows: &[CanonicalRow]) -> String {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("native-tpcds-parity-v1");
    hash_schema(&mut hasher, schema);
    hasher.update_field(&format!("rows={}", rows.len()));
    for row in rows {
        for field in &row.fields {
            hash_field(&mut hasher, field);
        }
    }
    format!("output-fnv1a64:{:016x}", hasher.finish())
}

fn hash_schema(hasher: &mut StableFnv1a64, schema: &Schema) {
    hasher.update_field("schema");
    hasher.update_field(&schema.fields().len().to_string());
    hash_metadata(hasher, schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&format!("{:?}", field.data_type()));
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
        hash_metadata(hasher, field.metadata());
    }
}

fn hash_metadata(hasher: &mut StableFnv1a64, metadata: &HashMap<String, String>) {
    hasher.update_field(&metadata.len().to_string());
    let mut keys = metadata.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        hasher.update_field(key);
        if let Some(value) = metadata.get(key) {
            hasher.update_field(value);
        }
    }
}

fn hash_field(hasher: &mut StableFnv1a64, field: &CanonicalField) {
    match field {
        CanonicalField::Null => hasher.update_field("null"),
        CanonicalField::Bool(value) => {
            hasher.update_field("bool");
            hasher.update_field(if *value { "1" } else { "0" });
        },
        CanonicalField::Int { width, value } => {
            hasher.update_field("int");
            hasher.update_field(&width.to_string());
            hasher.update_field(&value.to_string());
        },
        CanonicalField::UInt { width, value } => {
            hasher.update_field("uint");
            hasher.update_field(&width.to_string());
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Float { width, bits } => {
            hasher.update_field("float");
            hasher.update_field(&width.to_string());
            hasher.update_field(&format!("{bits:016x}"));
        },
        CanonicalField::Decimal128 { precision, scale, value } => {
            hasher.update_field("decimal128");
            hasher.update_field(&format!("{precision}:{scale}"));
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Decimal256 { precision, scale, value } => {
            hasher.update_field("decimal256");
            hasher.update_field(&format!("{precision}:{scale}"));
            hasher.update_bytes(value);
        },
        CanonicalField::Date32(value) => {
            hasher.update_field("date32");
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Date64(value) => {
            hasher.update_field("date64");
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Timestamp { unit, tz, value } => {
            hasher.update_field("timestamp");
            hasher.update_field(&format!("{unit:?}"));
            hasher.update_field(tz.as_deref().unwrap_or(""));
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Utf8(value) => {
            hasher.update_field("utf8");
            hasher.update_field(value);
        },
        CanonicalField::Binary(bytes) => {
            hasher.update_field("binary");
            hasher.update_bytes(bytes);
        },
        CanonicalField::Unsupported(text) => {
            hasher.update_field("unsupported");
            hasher.update_field(text);
        },
    }
}

struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    fn new() -> Self {
        Self { state: 0xcbf29ce484222325 }
    }

    fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0xff]);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

/// Detect whether the given SQL has a top-level `ORDER BY` clause that
/// determines the row order of the result. Strings, comments, and
/// parenthesized subqueries are ignored.
pub fn sql_has_top_level_order_by(sql: &str) -> bool {
    let depth_zero = strip_to_depth_zero(sql);
    let lowered = depth_zero.to_ascii_lowercase();
    let bytes = lowered.as_bytes();
    let mut i = 0;
    while i + 8 <= bytes.len() {
        if matches_word_at(bytes, i, b"order") && {
            let after = skip_whitespace(bytes, i + 5);
            matches_word_at(bytes, after, b"by")
        } {
            return true;
        }
        i += 1;
    }
    false
}

fn strip_to_depth_zero(sql: &str) -> String {
    let mut output = String::with_capacity(sql.len());
    let chars: Vec<char> = sql.chars().collect();
    let mut i = 0;
    let mut depth: i32 = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '-' && i + 1 < chars.len() && chars[i + 1] == '-' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            output.push(' ');
            continue;
        }
        if c == '/' && i + 1 < chars.len() && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            output.push(' ');
            continue;
        }
        if c == '\'' {
            i += 1;
            while i < chars.len() {
                if chars[i] == '\'' && i + 1 < chars.len() && chars[i + 1] == '\'' {
                    i += 2;
                    continue;
                }
                if chars[i] == '\'' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            output.push(' ');
            continue;
        }
        if c == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            output.push(' ');
            continue;
        }
        if c == '(' {
            depth += 1;
            output.push(' ');
            i += 1;
            continue;
        }
        if c == ')' {
            depth = (depth - 1).max(0);
            output.push(' ');
            i += 1;
            continue;
        }
        if depth == 0 {
            output.push(c);
        } else {
            output.push(' ');
        }
        i += 1;
    }
    output
}

fn matches_word_at(bytes: &[u8], pos: usize, word: &[u8]) -> bool {
    if pos + word.len() > bytes.len() {
        return false;
    }
    if !word.iter().zip(bytes[pos..pos + word.len()].iter()).all(|(a, b)| a == b) {
        return false;
    }
    if pos > 0 && is_word_byte(bytes[pos - 1]) {
        return false;
    }
    if pos + word.len() < bytes.len() && is_word_byte(bytes[pos + word.len()]) {
        return false;
    }
    true
}

fn skip_whitespace(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
