use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkSourceLayout {
    SingleFileMultiRowGroup,
    MultiFile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkScalePoint {
    ThreeGiB,
    SixGiB,
    TenGiB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PhaseCMatrixCase {
    pub(super) scale: BenchmarkScalePoint,
    pub(super) compression: BenchmarkCompression,
    pub(super) source_layout: BenchmarkSourceLayout,
    pub(super) native_plan_shape: BenchmarkPlanShape,
    pub(super) expected_outcome: BenchmarkExpectedOutcome,
}

impl BenchmarkSourceLayout {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::SingleFileMultiRowGroup => "single_file_multi_row_group",
            Self::MultiFile => "multi_file",
        }
    }

    pub(super) fn file_count(self) -> usize {
        match self {
            Self::SingleFileMultiRowGroup => 1,
            Self::MultiFile => 4,
        }
    }
}

impl BenchmarkScalePoint {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::ThreeGiB => "3gib",
            Self::SixGiB => "6gib",
            Self::TenGiB => "10gib",
        }
    }

    pub(super) fn input_rows(self, compression: BenchmarkCompression) -> usize {
        match (self, compression) {
            (Self::ThreeGiB, BenchmarkCompression::Uncompressed) => 12_250_000,
            (Self::ThreeGiB, BenchmarkCompression::Snappy) => 16_000_000,
            (Self::SixGiB, BenchmarkCompression::Uncompressed) => 24_500_000,
            (Self::SixGiB, BenchmarkCompression::Snappy) => 32_000_000,
            (Self::TenGiB, BenchmarkCompression::Uncompressed) => 40_900_000,
            (Self::TenGiB, BenchmarkCompression::Snappy) => 52_000_000,
        }
    }

    pub(super) fn minimum_file_bytes(self) -> u64 {
        match self {
            Self::ThreeGiB => THREE_GIB,
            Self::SixGiB => SIX_GIB,
            Self::TenGiB => TEN_GIB,
        }
    }
}

impl PhaseCMatrixCase {
    pub(super) fn name(self) -> String {
        format!(
            "{}-{}-{}-{}-{}",
            self.scale.name(),
            self.compression.name(),
            self.source_layout.name(),
            self.native_plan_shape.name(),
            self.expected_outcome.name()
        )
    }

    #[cfg(test)]
    pub(super) fn env_overrides(self) -> Vec<(&'static str, String)> {
        let mut overrides = vec![
            ("CUDF_PARQUET_BENCH_ROWS", self.scale.input_rows(self.compression).to_string()),
            ("CUDF_PARQUET_BENCH_MIN_FILE_BYTES", self.scale.minimum_file_bytes().to_string()),
            ("CUDF_PARQUET_BENCH_COMPRESSION", self.compression.name().to_string()),
            ("CUDF_PARQUET_BENCH_FILES", self.source_layout.file_count().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_PLAN_SHAPE", self.native_plan_shape.name().to_string()),
            ("CUDF_PARQUET_BENCH_EXPECTED_OUTCOME", self.expected_outcome.name().to_string()),
            ("CUDF_PARQUET_BENCH_NATIVE_MEMORY_POLICY", "bounded".to_string()),
        ];
        if self.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure {
            overrides.extend([
                ("CUDF_PARQUET_BENCH_NATIVE_DEVICE_BUDGET_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MAX_RETRY_ATTEMPTS", "1".to_string()),
                ("CUDF_PARQUET_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS", "1".to_string()),
            ]);
        }
        overrides
    }

    pub(super) fn configured_case(
        self,
        base: &LocalParquetBenchmarkConfig,
        output_dir: PathBuf,
    ) -> LocalParquetBenchmarkConfig {
        let mut config = base.clone();
        config.input_rows = self.scale.input_rows(self.compression);
        config.file_count = self.source_layout.file_count();
        config.minimum_file_bytes = self.scale.minimum_file_bytes();
        config.compression = self.compression;
        config.native_plan_shape = self.native_plan_shape;
        config.expected_outcome = self.expected_outcome;
        config.benchmark_paths = BenchmarkPathSet::NativeOnly;
        config.output_dir = Some(output_dir);
        if self.expected_outcome == BenchmarkExpectedOutcome::ResourceFailure {
            config.native_memory_control_policy = forced_resource_failure_policy();
        }
        config
    }
}

pub(super) fn phase_c_decision_matrix() -> Vec<PhaseCMatrixCase> {
    let mut cases = Vec::new();
    let compressions = [BenchmarkCompression::Uncompressed, BenchmarkCompression::Snappy];
    let layouts = [BenchmarkSourceLayout::SingleFileMultiRowGroup, BenchmarkSourceLayout::MultiFile];
    let plan_shapes = [BenchmarkPlanShape::DefaultSql, BenchmarkPlanShape::DirectFriendly];

    for scale in [BenchmarkScalePoint::ThreeGiB, BenchmarkScalePoint::SixGiB] {
        for compression in compressions {
            for source_layout in layouts {
                for native_plan_shape in plan_shapes {
                    cases.push(PhaseCMatrixCase {
                        scale,
                        compression,
                        source_layout,
                        native_plan_shape,
                        expected_outcome: BenchmarkExpectedOutcome::Success,
                    });
                }
            }
        }
    }

    for compression in compressions {
        for source_layout in layouts {
            for native_plan_shape in plan_shapes {
                cases.push(PhaseCMatrixCase {
                    scale: BenchmarkScalePoint::TenGiB,
                    compression,
                    source_layout,
                    native_plan_shape,
                    expected_outcome: BenchmarkExpectedOutcome::ResourceFailure,
                });
            }
        }
    }

    cases
}

pub(super) const PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION: u32 = 2;
pub(super) const PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_HEADER: &str = concat!(
    "schema_version\tcase_id\tscale\tcompression\tsource_layout\tnative_plan_shape\texpected_outcome\toutcome\t",
    "planning_outcome\tplanning_reason_code\tplanning_source_diagnostics\tnative_cudf_exec_accepted_count\t",
    "native_runtime_started\tno_silent_cpu_fallback\tzero_host_to_device_uploads\t",
    "one_device_to_host_materialization\trow_groups_selected\trow_groups_planned\trow_groups_committed\t",
    "source_chunks_planned\tsource_chunks_committed\tsource_chunks_retried\treservation_denied_count\t",
    "retry_count\tretry_resource_exhausted_count\tretry_reservation_failure_count\tcpu_datafusion_wall_ms\t",
    "datafusion_native_exec_wall_ms\tdatafusion_native_exec_speedup_vs_cpu_datafusion\t",
    "datafusion_native_exec_speedup_vs_cpu_comparable\tparquet_file_bytes\tparquet_row_groups\t",
    "runtime_error_code\truntime_error_kind\truntime_error_status\truntime_error_operation\truntime_error_facts\tlog_path"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BenchmarkMatrixOutcome {
    Pass,
    PlanningRejected,
    ResourceFailure,
    RuntimeFailure,
}

impl BenchmarkMatrixOutcome {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::PlanningRejected => "planning_rejected",
            Self::ResourceFailure => "resource_failure",
            Self::RuntimeFailure => "runtime_failure",
        }
    }
}

#[derive(Debug)]
pub(super) struct PhaseCMatrixReport {
    pub(super) artifact_dir: PathBuf,
    pub(super) tsv_path: PathBuf,
    pub(super) json_path: PathBuf,
    pub(super) markdown_path: PathBuf,
    pub(super) rows: Vec<PhaseCMatrixReportRow>,
}

impl PhaseCMatrixReport {
    pub(super) fn decision_gate_pass(&self) -> bool {
        self.rows.iter().all(PhaseCMatrixReportRow::matches_expected_outcome)
    }

    pub(super) fn unexpected_outcome_count(&self) -> usize {
        self.rows.iter().filter(|row| !row.matches_expected_outcome()).count()
    }

    pub(super) fn to_text(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("phase_c_datafusion_native_exec_matrix"));
        push_line(
            &mut text,
            format_args!("schema_version={}", PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION),
        );
        push_line(&mut text, format_args!("case_count={}", self.rows.len()));
        push_line(&mut text, format_args!("decision_gate={}", if self.decision_gate_pass() { "pass" } else { "fail" }));
        push_line(&mut text, format_args!("unexpected_outcomes={}", self.unexpected_outcome_count()));
        push_line(&mut text, format_args!("artifact_dir={}", self.artifact_dir.display()));
        push_line(&mut text, format_args!("tsv_report={}", self.tsv_path.display()));
        push_line(&mut text, format_args!("json_report={}", self.json_path.display()));
        push_line(&mut text, format_args!("markdown_report={}", self.markdown_path.display()));
        text
    }

    pub(super) fn to_tsv(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("{PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_HEADER}"));
        for row in &self.rows {
            push_line(&mut text, format_args!("{}", row.to_tsv_row()));
        }
        text
    }

    pub(super) fn to_json(&self) -> String {
        let rows = self.rows.iter().map(PhaseCMatrixReportRow::to_json_value).collect::<Vec<_>>();
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION,
            "decision_gate": if self.decision_gate_pass() { "pass" } else { "fail" },
            "unexpected_outcomes": self.unexpected_outcome_count(),
            "rows": rows,
        }))
        .expect("phase c matrix JSON report should serialize")
    }

    pub(super) fn to_markdown(&self) -> String {
        let mut text = String::new();
        push_line(&mut text, format_args!("# DataFusion Native Exec Phase C Matrix Results"));
        push_line(&mut text, format_args!(""));
        push_line(
            &mut text,
            format_args!(
                "Generated by `cargo run --example local_parquet_benchmark` with `CUDF_PARQUET_BENCH_RUN=phase_c_matrix`."
            ),
        );
        push_line(&mut text, format_args!(""));
        push_line(
            &mut text,
            format_args!(
                "Decision gate: **{}**. Unexpected outcomes: {}.",
                if self.decision_gate_pass() { "pass" } else { "fail" },
                self.unexpected_outcome_count()
            ),
        );
        push_line(&mut text, format_args!(""));
        push_line(
            &mut text,
            format_args!(
                "| case id | expected | outcome | planning | source facts | accepted | zero upload | one materialization | row groups | chunks | retries | native wall ms | speedup | error |"
            ),
        );
        push_line(
            &mut text,
            format_args!("| --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"),
        );
        for row in &self.rows {
            push_line(
                &mut text,
                format_args!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} | {}/{}/{} | {}/{}/{} | {} | {} | {} |",
                    row.case_id,
                    row.expected_outcome,
                    row.outcome.as_str(),
                    row.planning_outcome,
                    markdown_table_cell(&row.planning_source_diagnostics),
                    row.native_cudf_exec_accepted_count,
                    row.zero_host_to_device_uploads,
                    row.one_device_to_host_materialization,
                    row.row_groups_selected,
                    row.row_groups_planned,
                    row.row_groups_committed,
                    row.source_chunks_planned,
                    row.source_chunks_committed,
                    row.source_chunks_retried,
                    row.retry_count,
                    row.retry_resource_exhausted_count,
                    row.retry_reservation_failure_count,
                    row.datafusion_native_exec_wall_ms,
                    row.datafusion_native_exec_speedup_vs_cpu_datafusion,
                    row.runtime_error_code
                ),
            );
        }
        text
    }
}

#[derive(Debug, Clone)]
pub(super) struct PhaseCMatrixReportRow {
    pub(super) case_id: String,
    pub(super) scale: &'static str,
    pub(super) compression: &'static str,
    pub(super) source_layout: &'static str,
    pub(super) native_plan_shape: &'static str,
    pub(super) expected_outcome: &'static str,
    pub(super) outcome: BenchmarkMatrixOutcome,
    pub(super) planning_outcome: String,
    pub(super) planning_reason_code: String,
    pub(super) planning_source_diagnostics: String,
    pub(super) native_cudf_exec_accepted_count: String,
    pub(super) native_runtime_started: &'static str,
    pub(super) no_silent_cpu_fallback: &'static str,
    pub(super) zero_host_to_device_uploads: String,
    pub(super) one_device_to_host_materialization: String,
    pub(super) row_groups_selected: String,
    pub(super) row_groups_planned: String,
    pub(super) row_groups_committed: String,
    pub(super) source_chunks_planned: String,
    pub(super) source_chunks_committed: String,
    pub(super) source_chunks_retried: String,
    pub(super) reservation_denied_count: String,
    pub(super) retry_count: String,
    pub(super) retry_resource_exhausted_count: String,
    pub(super) retry_reservation_failure_count: String,
    pub(super) cpu_datafusion_wall_ms: String,
    pub(super) datafusion_native_exec_wall_ms: String,
    pub(super) datafusion_native_exec_speedup_vs_cpu_datafusion: String,
    pub(super) datafusion_native_exec_speedup_vs_cpu_comparable: String,
    pub(super) parquet_file_bytes: String,
    pub(super) parquet_row_groups: String,
    pub(super) runtime_error_code: String,
    pub(super) runtime_error_kind: String,
    pub(super) runtime_error_status: String,
    pub(super) runtime_error_operation: String,
    pub(super) runtime_error_facts: String,
    pub(super) log_path: String,
}

impl PhaseCMatrixReportRow {
    pub(super) fn from_case_report(
        case: PhaseCMatrixCase,
        report: &ConfiguredBenchmarkCaseReport,
        log_path: &Path,
    ) -> Self {
        match &report.report {
            BenchmarkCaseReport::Success(success) => Self::from_success(case, success, log_path),
            BenchmarkCaseReport::PlanningRejected(rejected) => Self::from_planning_rejected(case, rejected, log_path),
            BenchmarkCaseReport::ResourceFailure(resource_failure) => {
                Self::from_resource_failure(case, resource_failure, log_path)
            },
        }
    }

    pub(super) fn runtime_failure(case: PhaseCMatrixCase, error: &DataFusionError, log_path: &Path) -> Self {
        let adapter_error = adapter_error_from_datafusion(error).ok();
        let (code, kind, status, operation, facts) = adapter_error.map_or_else(
            || {
                (
                    "unavailable".to_string(),
                    "unavailable".to_string(),
                    "unavailable".to_string(),
                    "unavailable".to_string(),
                    sanitize_report_value(&error.to_string()),
                )
            },
            |error| {
                (
                    error.code().as_str().to_string(),
                    error.kind().as_str().to_string(),
                    error.status().as_str().to_string(),
                    error.operation().to_string(),
                    adapter_error_facts_tsv(error),
                )
            },
        );
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::RuntimeFailure,
            planning_outcome: "unavailable".to_string(),
            planning_reason_code: "unavailable".to_string(),
            planning_source_diagnostics: "unavailable".to_string(),
            native_cudf_exec_accepted_count: "unavailable".to_string(),
            native_runtime_started: "unknown",
            no_silent_cpu_fallback: "unknown",
            zero_host_to_device_uploads: "unavailable".to_string(),
            one_device_to_host_materialization: "unavailable".to_string(),
            row_groups_selected: "unavailable".to_string(),
            row_groups_planned: "unavailable".to_string(),
            row_groups_committed: "unavailable".to_string(),
            source_chunks_planned: "unavailable".to_string(),
            source_chunks_committed: "unavailable".to_string(),
            source_chunks_retried: "unavailable".to_string(),
            reservation_denied_count: "unavailable".to_string(),
            retry_count: "unavailable".to_string(),
            retry_resource_exhausted_count: "unavailable".to_string(),
            retry_reservation_failure_count: "unavailable".to_string(),
            cpu_datafusion_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_datafusion: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_comparable: "unavailable".to_string(),
            parquet_file_bytes: "unavailable".to_string(),
            parquet_row_groups: "unavailable".to_string(),
            runtime_error_code: code,
            runtime_error_kind: kind,
            runtime_error_status: status,
            runtime_error_operation: operation,
            runtime_error_facts: facts,
            log_path: log_path.display().to_string(),
        }
    }

    pub(super) fn from_success(case: PhaseCMatrixCase, report: &DirectParquetBenchmarkReport, log_path: &Path) -> Self {
        let metrics = &report.final_datafusion_native_exec.metrics;
        let planning_row = report
            .final_datafusion_native_exec
            .native_datafusion_planning_report
            .as_ref()
            .and_then(|report| report.rows.first());
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::Pass,
            planning_outcome: planning_row.map_or("unavailable", |row| row.outcome.as_str()).to_string(),
            planning_reason_code: planning_row.and_then(|row| row.reason_code.as_deref()).unwrap_or("none").to_string(),
            planning_source_diagnostics: planning_row
                .map(|row| native_source_diagnostics_text(&row.source_diagnostics))
                .unwrap_or_else(|| "unavailable".to_string()),
            native_cudf_exec_accepted_count: report
                .final_datafusion_native_exec
                .native_datafusion_planning_report
                .as_ref()
                .map_or_else(|| "unavailable".to_string(), |report| report.accepted_count().to_string()),
            native_runtime_started: "true",
            no_silent_cpu_fallback: "true",
            zero_host_to_device_uploads: (metrics.host_to_device_uploads == 0).to_string(),
            one_device_to_host_materialization: (metrics.device_to_host_materializations == 1).to_string(),
            row_groups_selected: metrics.row_groups_selected.to_string(),
            row_groups_planned: metrics.row_groups_planned.to_string(),
            row_groups_committed: metrics.row_groups_committed.to_string(),
            source_chunks_planned: metrics.source_chunks_planned.to_string(),
            source_chunks_committed: metrics.source_chunks_committed.to_string(),
            source_chunks_retried: metrics.source_chunks_retried.to_string(),
            reservation_denied_count: metrics.reservation_denied_count.to_string(),
            retry_count: metrics.retry_count.to_string(),
            retry_resource_exhausted_count: metrics.retry_resource_exhausted_count.to_string(),
            retry_reservation_failure_count: metrics.retry_reservation_failure_count.to_string(),
            cpu_datafusion_wall_ms: format!("{:.3}", duration_ms(report.cpu_default.mean)),
            datafusion_native_exec_wall_ms: format!("{:.3}", duration_ms(report.datafusion_native_exec.mean)),
            datafusion_native_exec_speedup_vs_cpu_datafusion: format!(
                "{:.3}",
                speedup(report.cpu_default.mean, report.datafusion_native_exec.mean)
            ),
            datafusion_native_exec_speedup_vs_cpu_comparable: format!(
                "{:.3}",
                speedup(report.cpu_comparable.mean, report.datafusion_native_exec.mean)
            ),
            parquet_file_bytes: report.parquet_file_bytes.to_string(),
            parquet_row_groups: report.parquet_row_groups.to_string(),
            runtime_error_code: "none".to_string(),
            runtime_error_kind: "none".to_string(),
            runtime_error_status: "none".to_string(),
            runtime_error_operation: "none".to_string(),
            runtime_error_facts: "none".to_string(),
            log_path: log_path.display().to_string(),
        }
    }

    pub(super) fn from_planning_rejected(
        case: PhaseCMatrixCase,
        report: &ExpectedPlanningRejectedReport,
        log_path: &Path,
    ) -> Self {
        let row = report.planning_report.rows.first();
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::PlanningRejected,
            planning_outcome: row.map_or("unavailable", |row| row.outcome.as_str()).to_string(),
            planning_reason_code: row.and_then(|row| row.reason_code.as_deref()).unwrap_or("none").to_string(),
            planning_source_diagnostics: row
                .map(|row| native_source_diagnostics_text(&row.source_diagnostics))
                .unwrap_or_else(|| "unavailable".to_string()),
            native_cudf_exec_accepted_count: report.planning_report.accepted_count().to_string(),
            native_runtime_started: "false",
            no_silent_cpu_fallback: "true",
            zero_host_to_device_uploads: "unavailable".to_string(),
            one_device_to_host_materialization: "unavailable".to_string(),
            row_groups_selected: "unavailable".to_string(),
            row_groups_planned: "unavailable".to_string(),
            row_groups_committed: "unavailable".to_string(),
            source_chunks_planned: "unavailable".to_string(),
            source_chunks_committed: "unavailable".to_string(),
            source_chunks_retried: "unavailable".to_string(),
            reservation_denied_count: "unavailable".to_string(),
            retry_count: "unavailable".to_string(),
            retry_resource_exhausted_count: "unavailable".to_string(),
            retry_reservation_failure_count: "unavailable".to_string(),
            cpu_datafusion_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_datafusion: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_comparable: "unavailable".to_string(),
            parquet_file_bytes: report.parquet_file_bytes.to_string(),
            parquet_row_groups: report.parquet_row_groups.to_string(),
            runtime_error_code: "none".to_string(),
            runtime_error_kind: "none".to_string(),
            runtime_error_status: "none".to_string(),
            runtime_error_operation: "none".to_string(),
            runtime_error_facts: "none".to_string(),
            log_path: log_path.display().to_string(),
        }
    }

    pub(super) fn from_resource_failure(
        case: PhaseCMatrixCase,
        report: &ExpectedResourceFailureReport,
        log_path: &Path,
    ) -> Self {
        let row = report.planning_report.rows.first();
        Self {
            case_id: case.name(),
            scale: case.scale.name(),
            compression: case.compression.name(),
            source_layout: case.source_layout.name(),
            native_plan_shape: case.native_plan_shape.name(),
            expected_outcome: case.expected_outcome.name(),
            outcome: BenchmarkMatrixOutcome::ResourceFailure,
            planning_outcome: row.map_or("unavailable", |row| row.outcome.as_str()).to_string(),
            planning_reason_code: row.and_then(|row| row.reason_code.as_deref()).unwrap_or("none").to_string(),
            planning_source_diagnostics: row
                .map(|row| native_source_diagnostics_text(&row.source_diagnostics))
                .unwrap_or_else(|| "unavailable".to_string()),
            native_cudf_exec_accepted_count: report.planning_report.accepted_count().to_string(),
            native_runtime_started: "true",
            no_silent_cpu_fallback: "true",
            zero_host_to_device_uploads: "unavailable".to_string(),
            one_device_to_host_materialization: "unavailable".to_string(),
            row_groups_selected: "unavailable".to_string(),
            row_groups_planned: "unavailable".to_string(),
            row_groups_committed: "unavailable".to_string(),
            source_chunks_planned: "unavailable".to_string(),
            source_chunks_committed: "unavailable".to_string(),
            source_chunks_retried: "unavailable".to_string(),
            reservation_denied_count: "unavailable".to_string(),
            retry_count: "unavailable".to_string(),
            retry_resource_exhausted_count: "unavailable".to_string(),
            retry_reservation_failure_count: "unavailable".to_string(),
            cpu_datafusion_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_wall_ms: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_datafusion: "unavailable".to_string(),
            datafusion_native_exec_speedup_vs_cpu_comparable: "unavailable".to_string(),
            parquet_file_bytes: report.parquet_file_bytes.to_string(),
            parquet_row_groups: report.parquet_row_groups.to_string(),
            runtime_error_code: report.error.code.clone(),
            runtime_error_kind: report.error.kind.clone(),
            runtime_error_status: report.error.status.clone(),
            runtime_error_operation: report.error.operation.to_string(),
            runtime_error_facts: report.error.facts.clone(),
            log_path: log_path.display().to_string(),
        }
    }

    pub(super) fn matches_expected_outcome(&self) -> bool {
        matches!(
            (self.expected_outcome, self.outcome),
            ("success", BenchmarkMatrixOutcome::Pass)
                | ("planning_rejected", BenchmarkMatrixOutcome::PlanningRejected)
                | ("resource_failure", BenchmarkMatrixOutcome::ResourceFailure)
        )
    }

    pub(super) fn to_tsv_row(&self) -> String {
        [
            PHASE_C_DATAFUSION_NATIVE_EXEC_MATRIX_TSV_SCHEMA_VERSION.to_string(),
            self.case_id.clone(),
            self.scale.to_string(),
            self.compression.to_string(),
            self.source_layout.to_string(),
            self.native_plan_shape.to_string(),
            self.expected_outcome.to_string(),
            self.outcome.as_str().to_string(),
            self.planning_outcome.clone(),
            self.planning_reason_code.clone(),
            self.planning_source_diagnostics.clone(),
            self.native_cudf_exec_accepted_count.clone(),
            self.native_runtime_started.to_string(),
            self.no_silent_cpu_fallback.to_string(),
            self.zero_host_to_device_uploads.clone(),
            self.one_device_to_host_materialization.clone(),
            self.row_groups_selected.clone(),
            self.row_groups_planned.clone(),
            self.row_groups_committed.clone(),
            self.source_chunks_planned.clone(),
            self.source_chunks_committed.clone(),
            self.source_chunks_retried.clone(),
            self.reservation_denied_count.clone(),
            self.retry_count.clone(),
            self.retry_resource_exhausted_count.clone(),
            self.retry_reservation_failure_count.clone(),
            self.cpu_datafusion_wall_ms.clone(),
            self.datafusion_native_exec_wall_ms.clone(),
            self.datafusion_native_exec_speedup_vs_cpu_datafusion.clone(),
            self.datafusion_native_exec_speedup_vs_cpu_comparable.clone(),
            self.parquet_file_bytes.clone(),
            self.parquet_row_groups.clone(),
            self.runtime_error_code.clone(),
            self.runtime_error_kind.clone(),
            self.runtime_error_status.clone(),
            self.runtime_error_operation.clone(),
            self.runtime_error_facts.clone(),
            self.log_path.clone(),
        ]
        .into_iter()
        .map(|value| tsv_escape(&value))
        .collect::<Vec<_>>()
        .join("\t")
    }

    pub(super) fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "case_id": self.case_id,
            "scale": self.scale,
            "compression": self.compression,
            "source_layout": self.source_layout,
            "native_plan_shape": self.native_plan_shape,
            "expected_outcome": self.expected_outcome,
            "outcome": self.outcome.as_str(),
            "planning_outcome": self.planning_outcome,
            "planning_reason_code": self.planning_reason_code,
            "planning_source_diagnostics": self.planning_source_diagnostics,
            "native_cudf_exec_accepted_count": self.native_cudf_exec_accepted_count,
            "native_runtime_started": self.native_runtime_started,
            "no_silent_cpu_fallback": self.no_silent_cpu_fallback,
            "zero_host_to_device_uploads": self.zero_host_to_device_uploads,
            "one_device_to_host_materialization": self.one_device_to_host_materialization,
            "row_groups_selected": self.row_groups_selected,
            "row_groups_planned": self.row_groups_planned,
            "row_groups_committed": self.row_groups_committed,
            "source_chunks_planned": self.source_chunks_planned,
            "source_chunks_committed": self.source_chunks_committed,
            "source_chunks_retried": self.source_chunks_retried,
            "reservation_denied_count": self.reservation_denied_count,
            "retry_count": self.retry_count,
            "retry_resource_exhausted_count": self.retry_resource_exhausted_count,
            "retry_reservation_failure_count": self.retry_reservation_failure_count,
            "cpu_datafusion_wall_ms": self.cpu_datafusion_wall_ms,
            "datafusion_native_exec_wall_ms": self.datafusion_native_exec_wall_ms,
            "datafusion_native_exec_speedup_vs_cpu_datafusion": self.datafusion_native_exec_speedup_vs_cpu_datafusion,
            "datafusion_native_exec_speedup_vs_cpu_comparable": self.datafusion_native_exec_speedup_vs_cpu_comparable,
            "parquet_file_bytes": self.parquet_file_bytes,
            "parquet_row_groups": self.parquet_row_groups,
            "runtime_error_code": self.runtime_error_code,
            "runtime_error_kind": self.runtime_error_kind,
            "runtime_error_status": self.runtime_error_status,
            "runtime_error_operation": self.runtime_error_operation,
            "runtime_error_facts": self.runtime_error_facts,
            "log_path": self.log_path,
        })
    }
}

pub(super) async fn run_phase_c_matrix(
    base_config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<PhaseCMatrixReport> {
    let artifact_dir = phase_c_matrix_artifact_dir();
    let logs_dir = artifact_dir.join("logs");
    let data_dir = phase_c_matrix_data_dir();
    fs::create_dir_all(&logs_dir)?;
    fs::create_dir_all(&data_dir)?;

    let mut rows = Vec::new();
    for case in phase_c_decision_matrix() {
        let case_id = case.name();
        let log_path = logs_dir.join(format!("{case_id}.log"));
        let case_config = case.configured_case(base_config, data_dir.join(&case_id));
        match run_configured_benchmark_case(&case_config).await {
            Ok(case_report) => {
                let mut log = benchmark_case_prologue(&case_report, &case_config);
                log.push('\n');
                log.push_str(&case_report.report.to_text());
                fs::write(&log_path, log)?;
                rows.push(PhaseCMatrixReportRow::from_case_report(case, &case_report, &log_path));
            },
            Err(error) => {
                fs::write(&log_path, format!("case={case_id}\noutcome=runtime_failure\nerror={error}\n"))?;
                rows.push(PhaseCMatrixReportRow::runtime_failure(case, &error, &log_path));
            },
        }
    }

    let report = PhaseCMatrixReport {
        tsv_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.tsv"),
        json_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.json"),
        markdown_path: artifact_dir.join("datafusion-native-exec-phase-c-matrix.md"),
        artifact_dir,
        rows,
    };
    fs::write(&report.tsv_path, report.to_tsv())?;
    fs::write(&report.json_path, report.to_json())?;
    fs::write(&report.markdown_path, report.to_markdown())?;
    Ok(report)
}

pub(super) fn phase_c_matrix_artifact_dir() -> PathBuf {
    std::env::var_os("CUDF_PARQUET_BENCH_MATRIX_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/phase-c-matrix"))
}

pub(super) fn phase_c_matrix_data_dir() -> PathBuf {
    std::env::var_os("CUDF_PARQUET_BENCH_MATRIX_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PHASE_C_MATRIX_DATA_DIR))
}
