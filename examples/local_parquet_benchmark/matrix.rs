use super::*;

#[path = "matrix/cases.rs"]
mod cases;
#[path = "matrix/report.rs"]
mod report;
#[path = "matrix/runner.rs"]
mod runner;

#[cfg(test)]
pub(super) use self::cases::{BenchmarkScalePoint, BenchmarkSourceLayout};
pub(super) use self::{
    cases::{PhaseCMatrixCase, phase_c_decision_matrix},
    report::PhaseCMatrixReport,
    runner::run_phase_c_matrix,
};

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
