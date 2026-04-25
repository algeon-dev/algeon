use nexus_query_engine::mode::StatisticalAggregateExecutionMode;

use super::*;

/// In-memory fallback-pressure report.
///
/// Rows describe native candidate acceptance or rejection before execution.
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
    /// Collects diagnostic fallback-pressure rows using the unbounded native
    /// execution config.
    ///
    /// This convenience path is for legacy shape-pressure diagnostics only.
    /// Production report surfaces should use
    /// [`Self::collect_with_native_execution_config`] with an explicit bounded
    /// or adaptive config so bounded diagnostics affect fallback-pressure rows.
    pub fn collect(items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>) -> Self {
        let execution_config = NativeCudfExecutionConfig::unbounded();
        Self::collect_with_native_execution_config(items, &execution_config)
    }

    /// Collects fallback-pressure rows using an explicit native execution config.
    ///
    /// Adaptive mode can add adaptive pipeline diagnostics to otherwise
    /// rewritten candidates; strict-only evidence remains on strict report
    /// surfaces.
    pub fn collect_with_native_execution_config(
        items: impl IntoIterator<Item = NativeFallbackPressureWorkloadItem>,
        execution_config: &NativeCudfExecutionConfig,
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
    pub fn rewritten_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rewritten).count()
    }

    /// Number of rejected candidate rows, not distinct query ids.
    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeFallbackPressureOutcome::Rejected).count()
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

    /// Builds a one-query report from a native rule that retained diagnostics.
    ///
    /// Missing retained diagnostics are represented as a planning-failure row.
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

    /// Attaches fixture metadata to comment lines before the TSV header.
    ///
    /// Free-form metadata text is sanitized for TSV comments by replacing tabs
    /// and line breaks with spaces.
    pub fn with_workload_dataset_metadata(mut self, metadata: WorkloadDatasetMetadata) -> Self {
        self.workload_dataset = Some(metadata);
        self
    }

    /// Number of planning rows where native planning replaced at least one fragment.
    pub fn accepted_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced).count()
    }

    /// Number of planning rows rejected before native execution.
    pub fn rejected_count(&self) -> usize {
        self.rows.iter().filter(|row| row.outcome == NativeDataFusionPlanningOutcome::RejectedBeforeExecution).count()
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
            let _ = writeln!(output, "# stddev_samp_execution_mode={}", mode.as_str());
        }
        write_workload_dataset_metadata(&mut output, self.workload_dataset.as_ref());
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
/// fields are projected from bounded metric handles when the artifact is read.
#[derive(Debug)]
pub(crate) struct NativeDataFusionExecutionTsvTemplate {
    rows: Box<[NativeDataFusionExecutionTsvTemplateRow]>,
    retained_capacity_bytes: u64,
}

#[derive(Debug)]
struct NativeDataFusionExecutionTsvTemplateRow {
    prefix: Box<[Box<str>]>,
    rejected_before_execution: bool,
}

impl NativeDataFusionExecutionTsvTemplate {
    const RUNTIME_FIELD_START: usize = 16;

    pub(crate) fn from_report(report: NativeDataFusionExecutionReport) -> Self {
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
            let strings = row
                .prefix
                .iter()
                .fold(0u64, |bytes, field| bytes.saturating_add(u64::try_from(field.len()).unwrap_or(u64::MAX)));
            total.saturating_add(strings).saturating_add(
                u64::try_from(row.prefix.len().saturating_mul(std::mem::size_of::<Box<str>>())).unwrap_or(u64::MAX),
            )
        });
        Self { rows, retained_capacity_bytes }
    }

    pub(crate) const fn retained_capacity_bytes(&self) -> u64 {
        self.retained_capacity_bytes
    }

    pub(crate) fn to_tsv(&self, runtime: NativeDataFusionRuntime) -> String {
        let runtime_summary = NativeDataFusionRuntimeSummary::from_runtime(runtime);
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
            let runtime_fields = runtime_fields(row.prefix.first().map_or("", |value| value.as_ref()), summary);
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
        planning_outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
        reason_code: None,
        unsupported_category: None,
        strict_bounded_contract_rows: Vec::new(),
        strict_bounded_compiled_graphs: Vec::new(),
        adaptive_pipeline_rows: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: String::new(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        native_plan_fragment_count: 0,
        host_datafusion_boundary_count: 0,
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
