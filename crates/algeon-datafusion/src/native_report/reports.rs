use std::fmt::Write;

use query_engine::mode::StatisticalAggregateExecutionMode;

#[cfg(feature = "server")]
use super::NativeDataFusionExecutionMetrics;
use super::{
    AlgeonNativeExecutionConfig, CandidateOutcomeKind, FinalPlanDisposition, NATIVE_DATAFUSION_EXECUTION_TSV_HEADER,
    NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NativeDataFusionExecutionRow,
    NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningRow, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionRuntime, NativeDataFusionRuntimeSummary, NativeFallbackPressureRow,
    NativeFallbackPressureWorkloadItem, TsvRow, WorkloadDatasetMetadata, datafusion_execution_rows_for_item,
    datafusion_planning_rows_for_item, report_rows_for_item, write_metric_detail_completeness,
    write_workload_dataset_metadata,
};

/// In-memory fallback-pressure report.
///
/// Rows describe native candidate acceptance or rejection before execution.
/// This collector runs candidate-only lowering without cost selection, so it
/// observes the producer subset (`Selected`, `NotSupported`);
/// the shared [`CandidateOutcomeKind`] projection still covers cost outcomes
/// without a narrowed enum.
/// TSV serialization writes optional workload metadata, then
/// [`NATIVE_FALLBACK_PRESSURE_TSV_HEADER`].
/// Optional planning fields serialize as `none`; missing dataset content
/// fingerprints serialize as `not_computed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFallbackPressureReport {
    /// Candidate rows in workload order. One query can produce multiple rows.
    pub rows: Vec<NativeFallbackPressureRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
}

impl NativeFallbackPressureReport {
    /// Collects diagnostic fallback-pressure rows using the canonical bounded
    /// native execution config.
    ///
    /// This convenience path uses the canonical bounded execution config.
    /// Call [`Self::collect_with_native_execution_config`] with an explicit
    /// config when its precise planning settings matter to the report.
    pub fn collect(items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>) -> Self {
        let execution_config = AlgeonNativeExecutionConfig::bounded_default();
        Self::collect_with_native_execution_config(items, &execution_config)
    }

    /// Collects fallback-pressure rows using an explicit native execution config.
    ///
    /// Capability-contract evidence is emitted by the canonical planning
    /// report surfaces.
    pub fn collect_with_native_execution_config(
        items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>,
        execution_config: &AlgeonNativeExecutionConfig,
    ) -> Self {
        let rows = items.into_iter().flat_map(|item| report_rows_for_item(item, execution_config)).collect();
        Self { rows, workload_dataset: None }
    }

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of accepted candidate rows, not distinct query ids.
    pub fn selected_count(&self) -> usize {
        self.rows.iter().filter(|row| matches!(row.outcome, Some(CandidateOutcomeKind::Selected))).count()
    }

    /// Number of rejected candidate rows, not distinct query ids.
    pub fn not_selected_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| {
                matches!(
                    row.outcome,
                    Some(CandidateOutcomeKind::NotSupported | CandidateOutcomeKind::NotSelectedByCost)
                )
            })
            .count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        debug_assert_eq!(NativeFallbackPressureRow::COLUMNS.join("\t"), NATIVE_FALLBACK_PRESSURE_TSV_HEADER);
        output.push_str(NATIVE_FALLBACK_PRESSURE_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(output, "{}", row.to_tsv_line());
        }
        output
    }
}

/// In-memory DataFusion planning report after the native optimizer rule.
///
/// Rows describe post-optimizer native coverage and structured fallback
/// reasons. TSV serialization preserves header order.
/// Optional planning fields serialize as `none`; missing dataset content
/// fingerprints serialize as `not_computed`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDataFusionPlanningReport {
    /// Per-query planning rows in workload order.
    pub rows: Vec<NativeDataFusionPlanningRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
}

impl NativeDataFusionPlanningReport {
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionPlanningWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_planning_rows_for_item).collect();
        Self { rows, workload_dataset: None }
    }

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of planning rows where native planning replaced at least one fragment.
    pub fn selected_count(&self) -> usize {
        self.rows.iter().filter(|row| matches!(row.outcome, Some(CandidateOutcomeKind::Selected))).count()
    }

    /// Number of planning rows rejected before native execution.
    pub fn not_selected_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.outcome.is_some_and(|outcome| outcome != CandidateOutcomeKind::Selected))
            .count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        debug_assert_eq!(NativeDataFusionPlanningRow::COLUMNS.join("\t"), NATIVE_DATAFUSION_PLANNING_TSV_HEADER);
        output.push_str(NATIVE_DATAFUSION_PLANNING_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(output, "{}", row.to_tsv_line());
        }
        output
    }
}

/// In-memory DataFusion execution report.
///
/// Rows combine native planning diagnostics, runtime outcome, output counters,
/// and projected native metrics. `not_run` marks output counters for rows
/// where native runtime did not start; optional metric observations use `none`.
/// Missing dataset content fingerprints serialize as `not_computed`.
#[derive(Debug, Clone)]
pub struct NativeDataFusionExecutionReport {
    /// Per-query execution rows in workload order.
    pub rows: Vec<NativeDataFusionExecutionRow>,
    /// Optional fixture metadata written as `# workload_*` TSV header lines.
    pub workload_dataset: Option<WorkloadDatasetMetadata>,
    /// Floating statistical aggregate contract selected for native execution.
    pub statistical_aggregate_execution_mode: Option<StatisticalAggregateExecutionMode>,
}

impl NativeDataFusionExecutionReport {
    pub fn collect(items: impl IntoIterator<Item = NativeDataFusionExecutionWorkloadItem>) -> Self {
        let rows = items.into_iter().flat_map(datafusion_execution_rows_for_item).collect();
        Self { rows, workload_dataset: None, statistical_aggregate_execution_mode: None }
    }

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Attaches the native floating statistical aggregate contract to TSV metadata.
    pub fn with_statistical_aggregate_execution_mode(mut self, mode: StatisticalAggregateExecutionMode) -> Self {
        self.statistical_aggregate_execution_mode = Some(mode);
        self
    }

    /// Number of rows where native runtime was invoked.
    pub fn runtime_started_count(&self) -> usize {
        self.rows.iter().filter(|row| row.native_runtime_started).count()
    }

    /// Number of rows rejected before native runtime started.
    pub fn rejected_before_execution_count(&self) -> usize {
        self.rows.iter().filter(|row| row.rejected_before_execution).count()
    }

    /// Serializes the report as TSV with metadata comments followed by rows.
    pub fn to_tsv(&self) -> String {
        let mut output = String::new();
        if let Some(mode) = self.statistical_aggregate_execution_mode {
            let _ = writeln!(output, "# statistical_aggregate_execution_mode={}", mode.as_str());
            let _ =
                writeln!(output, "# statistical_aggregate_tolerance_contract={}", mode.tolerance_contract().as_str());
        }
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
        write_metric_detail_completeness(&mut output, &self.rows);
        debug_assert_eq!(NativeDataFusionExecutionRow::COLUMNS.join("\t"), NATIVE_DATAFUSION_EXECUTION_TSV_HEADER);
        output.push_str(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            let _ = writeln!(output, "{}", row.to_tsv_line());
        }
        output
    }
}

/// Compact execution-report template retained by the server after planning.
///
/// Planning-only fields are flattened into exact-size strings so the original
/// physical plan and diagnostics can be dropped before execution. Runtime
/// fields are filled from the runtime outcome supplied when the artifact is
/// rendered.
#[derive(Debug)]
pub struct NativeDataFusionExecutionTsvTemplate {
    rows: Box<[NativeDataFusionExecutionTsvTemplateRow]>,
    retained_capacity_bytes: u64,
}

#[derive(Debug)]
struct NativeDataFusionExecutionTsvTemplateRow {
    prefix: Box<[Box<str>]>,
    rejected_before_execution: bool,
}

impl NativeDataFusionExecutionTsvTemplate {
    const RUNTIME_FIELD_START: usize = 15;

    pub fn from_report(report: NativeDataFusionExecutionReport) -> Self {
        let rows = report
            .rows
            .into_iter()
            .map(|row| {
                let rejected_before_execution = row.rejected_before_execution;
                let fields = row.fields();
                debug_assert_eq!(fields.len(), NativeDataFusionExecutionRow::COLUMNS.len());
                let prefix = fields
                    .into_iter()
                    .take(Self::RUNTIME_FIELD_START)
                    .map(String::into_boxed_str)
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                NativeDataFusionExecutionTsvTemplateRow { prefix, rejected_before_execution }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let retained_capacity_bytes = rows.iter().fold(0u64, |total, row| {
            let strings = row.prefix.iter().fold(0u64, |bytes, field| bytes.saturating_add(field.len() as u64));
            total.saturating_add(strings).saturating_add((row.prefix.len() * std::mem::size_of::<Box<str>>()) as u64)
        });
        Self { rows, retained_capacity_bytes }
    }

    pub const fn retained_capacity_bytes(&self) -> u64 {
        self.retained_capacity_bytes
    }

    pub fn to_tsv(&self, runtime: NativeDataFusionRuntime) -> String {
        self.render(NativeDataFusionRuntimeSummary::from_runtime(runtime))
    }

    /// Renders `runtime` with native metrics already projected from a sealed
    /// attempt. They replace any metrics carried by `runtime`, except that a
    /// runtime that never started keeps its empty metrics.
    #[cfg(feature = "server")]
    pub(crate) fn to_tsv_with_projected_metrics(
        &self,
        runtime: NativeDataFusionRuntime,
        metrics: NativeDataFusionExecutionMetrics,
    ) -> String {
        let mut runtime_summary = NativeDataFusionRuntimeSummary::from_runtime(runtime);
        if runtime_summary.started {
            runtime_summary.metrics = metrics;
        }
        self.render(runtime_summary)
    }

    fn render(&self, runtime_summary: NativeDataFusionRuntimeSummary) -> String {
        let mut output = String::new();
        output.push_str(NATIVE_DATAFUSION_EXECUTION_TSV_HEADER);
        output.push('\n');
        for row in &self.rows {
            for (index, field) in row.prefix.iter().enumerate() {
                if index > 0 {
                    output.push('\t');
                }
                output.push_str(field);
            }
            let summary = if row.rejected_before_execution {
                NativeDataFusionRuntimeSummary::from_runtime(NativeDataFusionRuntime::NotStarted)
            } else {
                runtime_summary.clone()
            };
            let runtime_fields = runtime_fields(row.prefix[0].as_ref(), summary);
            for field in runtime_fields {
                output.push('\t');
                output.push_str(&field);
            }
            output.push('\n');
        }
        output
    }
}

fn runtime_fields(query_id: &str, summary: NativeDataFusionRuntimeSummary) -> Vec<String> {
    let row = NativeDataFusionExecutionRow {
        query_id: query_id.to_string(),
        physical_root: String::new(),
        physical_shape: String::new(),
        native_candidate_shape: None,
        native_candidate_shape_family: None,
        planning_outcome: Some(CandidateOutcomeKind::Selected),
        reason_code: None,
        unsupported_category: None,
        capability_decision_rows: Vec::new(),
        capability_compiled_graphs: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: String::new(),
        final_disposition: Some(FinalPlanDisposition::DataFusion),
        native_plan_fragment_count: Some(0),
        host_datafusion_boundary_count: Some(0),
        rejected_before_execution: false,
        native_runtime_started: summary.started,
        runtime_outcome: summary.outcome,
        runtime_failure: summary.failure,
        output_batches: summary.output_batches,
        output_rows: summary.output_rows,
        output_bytes: summary.output_bytes,
        metrics: summary.metrics,
    };
    row.fields().into_iter().skip(NativeDataFusionExecutionTsvTemplate::RUNTIME_FIELD_START).collect()
}

#[cfg(test)]
mod tests {
    use datafusion::physical_plan::metrics::MetricsSet;

    use super::*;
    use crate::native_report::{NativeDataFusionExecutionMetrics, NativeDataFusionRuntimeOutcome};

    #[test]
    fn server_execution_template_preserves_runtime_column_alignment() {
        let report = NativeDataFusionExecutionReport {
            rows: vec![NativeDataFusionExecutionRow {
                query_id: "server-template".to_string(),
                physical_root: "Root".to_string(),
                physical_shape: "Root(Source)".to_string(),
                native_candidate_shape: None,
                native_candidate_shape_family: None,
                planning_outcome: Some(CandidateOutcomeKind::Selected),
                reason_code: None,
                unsupported_category: None,
                capability_decision_rows: Vec::new(),
                capability_compiled_graphs: Vec::new(),
                source_diagnostics: Vec::new(),
                post_native_plan_root: "AlgeonNativeExec".to_string(),
                final_disposition: Some(FinalPlanDisposition::Native),
                native_plan_fragment_count: Some(1),
                host_datafusion_boundary_count: Some(0),
                rejected_before_execution: false,
                native_runtime_started: false,
                runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
                runtime_failure: None,
                output_batches: None,
                output_rows: None,
                output_bytes: None,
                metrics: NativeDataFusionExecutionMetrics::default(),
            }],
            workload_dataset: None,
            statistical_aggregate_execution_mode: None,
        };

        let template = NativeDataFusionExecutionTsvTemplate::from_report(report);
        let tsv = template.to_tsv(NativeDataFusionRuntime::Succeeded {
            output_batches: 1,
            output_rows: 2,
            output_bytes: 3,
            metrics: MetricsSet::new(),
        });
        let mut lines = tsv.lines();
        let header = lines.next().expect("TSV header").split('\t').collect::<Vec<_>>();
        let row = lines.next().expect("TSV row").split('\t').collect::<Vec<_>>();

        assert_eq!(header.len(), row.len(), "server template row must preserve the execution TSV width");
        assert_eq!(
            NativeDataFusionExecutionTsvTemplate::RUNTIME_FIELD_START,
            header.iter().position(|column| *column == "admission_sequence").expect("runtime segment start column")
        );
        let native_runtime_started =
            header.iter().position(|column| *column == "native_runtime_started").expect("native runtime start column");
        assert_eq!(native_runtime_started, NativeDataFusionExecutionTsvTemplate::RUNTIME_FIELD_START + 4);
        assert_eq!(row[native_runtime_started], "true");
    }
}
