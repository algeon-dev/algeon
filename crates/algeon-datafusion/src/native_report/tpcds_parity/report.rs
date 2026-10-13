use std::{fmt::Write, sync::Arc};

use datafusion::prelude::SessionConfig;

use super::{
    CandidateOutcomeKind, ExecutionPlan, FinalPlanDisposition, MetricsSet, NATIVE_TPCDS_PARITY_TSV_COLUMNS,
    NATIVE_TPCDS_PARITY_TSV_HEADER, NativeDataFusionExecutionMetrics, NativeDataFusionRuntimeOutcome,
    NativePlanningDiagnostics, NativePlanningFailure, ParityCompareOptions, ParityComparisonResult, RecordBatch,
    Schema, SchemaRef, StatisticalAggregateExecutionMode, TsvRow, WorkloadDatasetMetadata,
    compare_record_batches_with_options, fingerprint_batches, post_native_plan_coverage, sql_has_top_level_order_by,
    sql_top_level_order_by_covers_output_columns, stable_tsv_field, unsupported_category,
    write_workload_dataset_metadata,
};
use crate::report::CandidateOutcome;

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

    /// Constructs absolute and relative float error bounds without validating them.
    ///
    /// A zero/zero pair keeps exact comparisons; tolerance is enabled when either bound is positive.
    pub const fn new(absolute: f64, relative: f64) -> Self {
        Self { absolute, relative }
    }

    /// Returns whether either float tolerance bound is positive.
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
    /// Candidate-selection outcome, or `None` for a terminal planning failure.
    pub native_planning_outcome: Option<CandidateOutcomeKind>,
    /// Native fallback or planning reason code; `None` serializes as `none`.
    pub native_reason_code: Option<String>,
    /// Coarse unsupported category for native rejection; `None` serializes as `none`.
    pub native_unsupported_category: Option<String>,
    /// Native coverage of the post-optimizer DataFusion plan, or `None` when rejected.
    pub final_disposition: Option<FinalPlanDisposition>,
    /// Number of native plan fragments visible after native planning, or `None` when rejected.
    pub native_plan_fragment_count: Option<usize>,
    /// Number of host DataFusion boundaries around native fragments, or `None` when rejected.
    pub host_datafusion_boundary_count: Option<usize>,
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
    /// Number of native retry attempts recorded during runtime.
    pub native_retry_count: Option<usize>,
    /// CPU planning/runtime or native planning/runtime failure detail.
    pub native_runtime_failure: Option<String>,
}

impl TsvRow for NativeTpcdsParityRow {
    const COLUMNS: &'static [&'static str] = NATIVE_TPCDS_PARITY_TSV_COLUMNS;

    fn fields(&self) -> Vec<String> {
        vec![
            self.query_id.clone(),
            self.cpu_planning_status.as_str().to_string(),
            self.cpu_runtime_status.as_str().to_string(),
            self.native_planning_outcome.map_or_else(|| "none".to_string(), |outcome| outcome.as_str().to_string()),
            self.native_reason_code.as_deref().unwrap_or("none").to_string(),
            self.native_unsupported_category.as_deref().unwrap_or("none").to_string(),
            self.final_disposition.map_or_else(|| "none".to_string(), |d| d.as_str().to_string()),
            self.native_plan_fragment_count.map_or_else(|| "none".to_string(), |c| c.to_string()),
            self.host_datafusion_boundary_count.map_or_else(|| "none".to_string(), |c| c.to_string()),
            self.native_runtime_started.to_string(),
            self.native_runtime_outcome.as_str().to_string(),
            self.parity_status.as_str().to_string(),
            self.parity_detail.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
            format_optional_usize(self.cpu_output_rows),
            format_optional_usize(self.native_output_rows),
            format_optional_usize(self.cpu_output_batches),
            format_optional_usize(self.native_output_batches),
            self.cpu_output_fingerprint.as_deref().unwrap_or("none").to_string(),
            self.native_output_fingerprint.as_deref().unwrap_or("none").to_string(),
            self.comparison_mode.as_str().to_string(),
            self.duckdb_output.status.as_str().to_string(),
            self.duckdb_output.cpu_parity_status.as_str().to_string(),
            self.duckdb_output.cpu_parity_detail.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
            format_optional_usize(self.duckdb_output.output_rows),
            format_optional_usize(self.duckdb_output.output_batches),
            self.duckdb_output.output_fingerprint.as_deref().unwrap_or("none").to_string(),
            format_optional_usize(self.native_retry_count),
            self.native_runtime_failure.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
        ]
    }
}

impl NativeTpcdsParityRow {
    /// Converts one parity workload item into a stable report row, releasing any
    /// retained Arrow batches once row-level fingerprints and counts are derived.
    pub fn from_workload_item(item: NativeTpcdsParityWorkloadItem, float_tolerance: ParityFloatTolerance) -> Self {
        parity_row_for_item(item, float_tolerance)
    }

    /// Serializes this row using the stable TPC-DS parity TSV schema.
    pub fn to_tsv_line(&self) -> String {
        <Self as TsvRow>::to_tsv_line(self)
    }
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
/// `cpu_parity_*` fields compare the oracle output against CPU DataFusion.
/// They do not compare it against native GPU output. Optional counts and
/// fingerprints serialize as `none` when the oracle is not configured or did
/// not produce output.
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

/// In-memory TPC-DS parity report.
///
/// `to_tsv` writes float tolerance, optional statistical-aggregate mode,
/// optional CPU oracle session settings, optional dataset metadata, and then
/// rows using [`NATIVE_TPCDS_PARITY_TSV_HEADER`].
#[derive(Debug, Clone)]
pub struct NativeTpcdsParityReport {
    /// Per-query rows in caller-provided workload order.
    pub rows: Vec<NativeTpcdsParityRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
    /// Float tolerance recorded in the TSV header and used for comparisons.
    pub float_tolerance: ParityFloatTolerance,
    /// Floating statistical aggregate contract selected for native execution.
    pub statistical_aggregate_execution_mode: Option<StatisticalAggregateExecutionMode>,
    /// Target partitions of the CPU DataFusion oracle session.
    pub cpu_target_partitions: Option<usize>,
    /// Batch size, in rows, of the CPU DataFusion oracle session.
    pub cpu_batch_size: Option<usize>,
}

impl NativeTpcdsParityReport {
    /// Keeps precomputed parity rows and their float tolerance in the supplied order.
    ///
    /// This constructor does not compare outputs; dataset, statistical-mode and CPU session metadata are unset.
    pub fn from_rows(rows: Vec<NativeTpcdsParityRow>, float_tolerance: ParityFloatTolerance) -> Self {
        Self {
            rows,
            workload_dataset: None,
            float_tolerance,
            statistical_aggregate_execution_mode: None,
            cpu_target_partitions: None,
            cpu_batch_size: None,
        }
    }

    /// Compares supplied workload outputs using the configured float tolerance and collects rows in iterator order.
    ///
    /// It consumes recorded outputs and failures without planning or executing queries.
    pub fn collect(
        items: impl IntoIterator<Item = NativeTpcdsParityWorkloadItem>,
        float_tolerance: ParityFloatTolerance,
    ) -> Self {
        let rows =
            items.into_iter().map(|item| NativeTpcdsParityRow::from_workload_item(item, float_tolerance)).collect();
        Self {
            rows,
            workload_dataset: None,
            float_tolerance,
            statistical_aggregate_execution_mode: None,
            cpu_target_partitions: None,
            cpu_batch_size: None,
        }
    }

    /// Attaches fixture metadata to the TSV header.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Attaches the native floating statistical aggregate contract to TSV metadata.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.statistical_aggregate_execution_mode = Some(mode);
        self
    }

    /// Records the CPU oracle session's target partitions and batch size in TSV metadata.
    pub fn with_cpu_session_config(mut self, config: &SessionConfig) -> Self {
        self.cpu_target_partitions = Some(config.target_partitions());
        self.cpu_batch_size = Some(config.batch_size());
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
        let _ = writeln!(output, "# parity_float_tolerance_absolute={}", self.float_tolerance.absolute);
        let _ = writeln!(output, "# parity_float_tolerance_relative={}", self.float_tolerance.relative);
        if let Some(mode) = self.statistical_aggregate_execution_mode {
            let _ = writeln!(output, "# statistical_aggregate_execution_mode={}", mode.as_str());
            let _ =
                writeln!(output, "# statistical_aggregate_tolerance_contract={}", mode.tolerance_contract().as_str());
        }
        if let Some(target_partitions) = self.cpu_target_partitions {
            let _ = writeln!(output, "# cpu_target_partitions={target_partitions}");
        }
        if let Some(batch_size) = self.cpu_batch_size {
            let _ = writeln!(output, "# cpu_batch_size={batch_size}");
        }
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        debug_assert_eq!(NativeTpcdsParityRow::COLUMNS.join("\t"), NATIVE_TPCDS_PARITY_TSV_HEADER);
        output.push_str(NATIVE_TPCDS_PARITY_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(output, "{}", row.to_tsv_line());
        }
        output
    }
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tpcds_parity_row_contract_header_matches_public_constant() {
        assert_eq!(NativeTpcdsParityRow::COLUMNS.join("\t"), NATIVE_TPCDS_PARITY_TSV_HEADER);
    }

    #[test]
    fn tpcds_parity_row_fields_match_columns() {
        let row = NativeTpcdsParityRow {
            query_id: "q01".to_string(),
            cpu_planning_status: ParityStageStatus::Succeeded,
            cpu_runtime_status: ParityStageStatus::Succeeded,
            native_planning_outcome: Some(CandidateOutcomeKind::NotSupported),
            native_reason_code: Some("unsupported_operator".to_string()),
            native_unsupported_category: Some("operator_join".to_string()),
            final_disposition: Some(FinalPlanDisposition::DataFusion),
            native_plan_fragment_count: Some(0),
            host_datafusion_boundary_count: Some(0),
            native_runtime_started: false,
            native_runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            parity_status: ParityStatus::NativeRejected,
            parity_detail: None,
            cpu_output_rows: Some(0),
            native_output_rows: None,
            cpu_output_batches: Some(0),
            native_output_batches: None,
            cpu_output_fingerprint: None,
            native_output_fingerprint: None,
            comparison_mode: ParityComparisonMode::NotApplicable,
            duckdb_output: NativeTpcdsParityExternalOutputSummary::not_configured(),
            native_retry_count: None,
            native_runtime_failure: None,
        };
        assert_eq!(row.fields().len(), NativeTpcdsParityRow::COLUMNS.len());
        assert_eq!(
            row.to_tsv_line(),
            "q01\tsucceeded\tsucceeded\tnot_supported\tunsupported_operator\toperator_join\tdatafusion\t0\t0\tfalse\tnot_started\tnative_rejected\tnone\t0\tnone\t0\tnone\tnone\tnone\tnot_applicable\tnot_configured\tnot_compared\tnone\tnone\tnone\tnone\tnone\tnone"
        );
    }
}

/// Workload-item input for the parity report, pairing the CPU and native
/// execution outcomes for one TPC-DS query.
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
    let ordered_total = ordered
        && cpu_schema
            .as_ref()
            .is_some_and(|schema| sql_top_level_order_by_covers_output_columns(&item.sql, schema.as_ref()));

    let (
        native_planning_outcome,
        native_reason_code,
        native_unsupported_category,
        final_disposition,
        native_plan_fragment_count,
        host_datafusion_boundary_count,
        native_planning_failure_message,
    ) = match &item.native_planning {
        NativeTpcdsParityNativePlanningOutcome::Planned { accepted, planned_plan, diagnostics } => {
            let (outcome, reason_code, unsupported_category) = planning_outcome_diagnostics(diagnostics, *accepted);
            let coverage = post_native_plan_coverage(planned_plan.as_ref());
            (
                outcome,
                reason_code,
                unsupported_category,
                Some(coverage.disposition()),
                Some(coverage.native_fragment_count()),
                Some(coverage.datafusion_boundary_count()),
                None,
            )
        },
        NativeTpcdsParityNativePlanningOutcome::PlanningFailed(failure) => (
            None,
            Some(failure.reason_code.clone()),
            Some("planning".to_string()),
            None,
            None,
            None,
            Some(failure.message.clone()),
        ),
    };

    let (native_runtime_started, native_runtime_outcome, native_batches, native_schema, native_metrics, native_failure) =
        match item.native_runtime {
            NativeTpcdsParityNativeRuntimeOutcome::NotStarted => {
                (false, NativeDataFusionRuntimeOutcome::NotStarted, None, None, None, native_planning_failure_message)
            },
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
            ordered_total,
            float_tolerance,
        });
    let duckdb_output = summarize_external_output(
        item.duckdb_output,
        cpu_schema.as_ref(),
        cpu_batches.as_deref(),
        ordered,
        ordered_total,
        float_tolerance,
    );

    NativeTpcdsParityRow {
        query_id: item.query_id,
        cpu_planning_status,
        cpu_runtime_status,
        native_planning_outcome,
        native_reason_code,
        native_unsupported_category,
        final_disposition,
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
        native_retry_count: native_metrics_summary.as_ref().map(|m| m.retry_count),
        native_runtime_failure: cpu_runtime_failure.or(native_failure),
    }
}

fn summarize_external_output(
    output: NativeTpcdsParityExternalOutput,
    cpu_schema: Option<&SchemaRef>,
    cpu_batches: Option<&[RecordBatch]>,
    ordered: bool,
    ordered_total: bool,
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
            let comparison = compare_record_batches_with_sql_ordering(
                cpu_schema.as_ref(),
                cpu_batches,
                schema.as_ref(),
                &batches,
                ordered,
                ordered_total,
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

fn planning_outcome_diagnostics(
    diagnostics: &NativePlanningDiagnostics,
    accepted: bool,
) -> (Option<CandidateOutcomeKind>, Option<String>, Option<String>) {
    if accepted {
        return (Some(CandidateOutcomeKind::Selected), None, None);
    }
    for outcome in diagnostics.report.outcomes() {
        match outcome {
            CandidateOutcome::Selected { .. } => {},
            CandidateOutcome::NotSupported { capability_reason, detail, .. } => {
                return (
                    Some(CandidateOutcomeKind::NotSupported),
                    Some(capability_reason.code().to_string()),
                    Some(unsupported_category(*capability_reason, detail)),
                );
            },
            CandidateOutcome::NotSelectedByCost { cost_reason, .. } => {
                return (Some(CandidateOutcomeKind::NotSelectedByCost), Some(cost_reason.code().to_string()), None);
            },
        }
    }
    (None, None, None)
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

struct ParityInputs<'a> {
    cpu_planning_status: ParityStageStatus,
    cpu_runtime_status: ParityStageStatus,
    native_planning_outcome: Option<CandidateOutcomeKind>,
    native_runtime_started: bool,
    native_runtime_outcome: NativeDataFusionRuntimeOutcome,
    cpu_schema: Option<&'a SchemaRef>,
    cpu_batches: Option<&'a [RecordBatch]>,
    native_schema: Option<&'a SchemaRef>,
    native_batches: Option<&'a [RecordBatch]>,
    ordered: bool,
    ordered_total: bool,
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
        Some(CandidateOutcomeKind::NotSupported | CandidateOutcomeKind::NotSelectedByCost) | None => {
            return (ParityStatus::NativeRejected, ParityComparisonMode::NotApplicable, None, None, None);
        },
        Some(CandidateOutcomeKind::Selected) => {},
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
    let result = compare_record_batches_with_sql_ordering(
        cpu_schema.as_ref(),
        cpu_batches,
        native_schema.as_ref(),
        native_batches,
        inputs.ordered,
        inputs.ordered_total,
        inputs.float_tolerance,
        ParityCompareOptions::strict_native(),
    );
    (result.status, result.mode, result.detail, Some(result.cpu_fingerprint), Some(result.native_fingerprint))
}

#[expect(clippy::too_many_arguments, reason = "inputs are distinct domain values that share no owner")]
fn compare_record_batches_with_sql_ordering(
    cpu_schema: &Schema,
    cpu_batches: &[RecordBatch],
    native_schema: &Schema,
    native_batches: &[RecordBatch],
    ordered: bool,
    ordered_total: bool,
    float_tolerance: ParityFloatTolerance,
    options: ParityCompareOptions,
) -> ParityComparisonResult {
    let result = compare_record_batches_with_options(
        cpu_schema,
        cpu_batches,
        native_schema,
        native_batches,
        ordered,
        float_tolerance,
        options,
    );
    if !ordered || ordered_total || result.status != ParityStatus::Mismatched {
        return result;
    }
    if !result.detail.as_deref().is_some_and(|detail| detail.starts_with("row ")) {
        return result;
    }
    compare_record_batches_with_options(
        cpu_schema,
        cpu_batches,
        native_schema,
        native_batches,
        false,
        float_tolerance,
        options,
    )
}
