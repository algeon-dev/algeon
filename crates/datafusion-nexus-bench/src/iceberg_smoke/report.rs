use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergNativeSmokeReport {
    pub rows: Vec<IcebergNativeSmokeReportRow>,
}

impl IcebergNativeSmokeReport {
    pub fn to_tsv(&self) -> String {
        let rows = self.rows.iter().map(IcebergNativeSmokeReportRow::to_tsv_row).collect::<Vec<_>>().join("\n");
        format!("# native_iceberg_smoke_report_tsv_schema_version={REPORT_SCHEMA_VERSION}\n{REPORT_HEADER}\n{rows}\n",)
    }
}

pub fn write_smoke_tsv_header(writer: &mut dyn Write) -> SmokeResult<()> {
    writeln!(writer, "# native_iceberg_smoke_report_tsv_schema_version={REPORT_SCHEMA_VERSION}")?;
    writeln!(writer, "{REPORT_HEADER}")?;
    writer.flush()?;
    Ok(())
}

pub fn write_smoke_tsv_row(writer: &mut dyn Write, row: &IcebergNativeSmokeReportRow) -> SmokeResult<()> {
    writeln!(writer, "{}", row.to_tsv_row())?;
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergNativeSmokeReportRow {
    pub query_id: String,
    pub fixture_mode: String,
    pub catalog_type: String,
    pub table_ident: String,
    pub sql: String,
    pub diagnostics: TableFormatScanDiagnostics,
    pub native_planning_outcome: NativeDataFusionPlanningOutcome,
    pub native_reason_code: Option<String>,
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub staging_summary: ObjectStoreStagingSummary,
    pub cpu_arrow_reader_fallback: bool,
    pub native_runtime_started: bool,
    pub native_runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub native_runtime_elapsed_ms: Option<u128>,
    pub native_output_rows: Option<usize>,
    pub native_output_batches: Option<usize>,
    pub source_fetch_bytes: usize,
    pub source_fetch_count: usize,
    pub host_fetch_budget_wait_time_ns: Option<usize>,
    pub source_fetch_concurrency_wait_time_ns: Option<usize>,
    pub source_fetch_time_ns: Option<usize>,
    pub source_decode_time_ns: Option<usize>,
    pub native_source_read_runtime_rows: String,
    pub lakehouse_cache_fill_event_snapshot_rows: String,
    pub expected_count: Option<u64>,
    pub native_count_value: Option<u64>,
    pub cpu_runtime_started: bool,
    pub cpu_runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub cpu_runtime_elapsed_ms: Option<u128>,
    pub cpu_output_rows: Option<usize>,
    pub cpu_output_batches: Option<usize>,
    pub cpu_output_fingerprint: Option<String>,
    pub native_output_fingerprint: Option<String>,
    pub comparison_mode: ParityComparisonMode,
    pub parity_status: ParityStatus,
    pub parity_detail: Option<String>,
    pub credential_source: String,
}

impl IcebergNativeSmokeReportRow {
    pub fn new(
        query_id: String,
        fixture_mode: String,
        catalog_type: String,
        table_ident: String,
        sql: String,
        diagnostics: TableFormatScanDiagnostics,
        native_planning_outcome: NativeDataFusionPlanningOutcome,
        native_reason_code: Option<String>,
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
        expected_count: Option<u64>,
        staging_summary: ObjectStoreStagingSummary,
        credential_source: String,
    ) -> Self {
        Self {
            query_id,
            fixture_mode,
            catalog_type,
            table_ident,
            sql,
            diagnostics,
            native_planning_outcome,
            native_reason_code,
            post_native_execution_path,
            staging_summary,
            cpu_arrow_reader_fallback: false,
            native_runtime_started: false,
            native_runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            native_runtime_elapsed_ms: None,
            native_output_rows: None,
            native_output_batches: None,
            source_fetch_bytes: 0,
            source_fetch_count: 0,
            host_fetch_budget_wait_time_ns: None,
            source_fetch_concurrency_wait_time_ns: None,
            source_fetch_time_ns: None,
            source_decode_time_ns: None,
            native_source_read_runtime_rows: "none".to_string(),
            lakehouse_cache_fill_event_snapshot_rows: "none".to_string(),
            expected_count,
            native_count_value: None,
            cpu_runtime_started: false,
            cpu_runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            cpu_runtime_elapsed_ms: None,
            cpu_output_rows: None,
            cpu_output_batches: None,
            cpu_output_fingerprint: None,
            native_output_fingerprint: None,
            comparison_mode: ParityComparisonMode::NotApplicable,
            parity_status: ParityStatus::NotCompared,
            parity_detail: None,
            credential_source,
        }
    }

    pub fn record_native_metrics(&mut self, metrics: NativeDataFusionExecutionMetrics) {
        self.native_source_read_runtime_rows = format_source_read_runtime_rows(&metrics);
        self.lakehouse_cache_fill_event_snapshot_rows = format_lakehouse_cache_fill_event_snapshot_rows(&metrics);
        self.source_fetch_bytes = metrics.source_fetch_bytes;
        self.source_fetch_count = metrics.source_fetch_count;
        self.host_fetch_budget_wait_time_ns = metrics.host_fetch_budget_wait_time_ns;
        self.source_fetch_concurrency_wait_time_ns = metrics.source_fetch_concurrency_wait_time_ns;
        self.source_fetch_time_ns = metrics.source_fetch_time_ns;
        self.source_decode_time_ns = metrics.source_decode_time_ns;
    }

    pub fn finalize_count_parity_if_needed(&mut self) {
        if self.cpu_runtime_started {
            return;
        }
        if self.native_planning_outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
            self.parity_status = ParityStatus::NativeRejected;
            self.parity_detail.get_or_insert_with(|| "native planning rejected before execution".to_string());
            return;
        }
        if self.native_runtime_started && self.native_runtime_outcome == NativeDataFusionRuntimeOutcome::Failed {
            self.parity_status = ParityStatus::NativeFailed;
            return;
        }
        match (self.expected_count, self.native_count_value) {
            (Some(expected), Some(actual)) if expected == actual => {
                self.parity_status = ParityStatus::Matched;
                self.parity_detail = None;
            },
            (Some(expected), Some(actual)) => {
                self.parity_status = ParityStatus::Mismatched;
                self.parity_detail = Some(format!("expected_count={expected},native_count_value={actual}"));
            },
            _ => {
                self.parity_status = ParityStatus::NotCompared;
                self.parity_detail.get_or_insert_with(|| "count parity not available".to_string());
            },
        }
    }

    pub(crate) fn to_tsv_row(&self) -> String {
        [
            self.query_id.clone(),
            self.fixture_mode.clone(),
            self.catalog_type.clone(),
            self.table_ident.clone(),
            self.sql.clone(),
            self.diagnostics.table_format.clone(),
            self.diagnostics.snapshot_id.clone().unwrap_or_else(|| "none".to_string()),
            self.diagnostics.planned_data_file_count.to_string(),
            self.diagnostics.planned_delete_file_count.to_string(),
            optional_u64(self.diagnostics.planned_bytes),
            self.diagnostics.admitted_native_source_file_count.to_string(),
            list_or_none(&self.diagnostics.rejected_feature_list),
            empty_as_none(&self.diagnostics.native_source_read_capability),
            empty_as_none(&self.diagnostics.source_read_strategy),
            empty_as_none(&self.diagnostics.source_decode_input),
            empty_as_none(&self.diagnostics.credential_mode),
            empty_as_none(&self.diagnostics.lakehouse_cache_status),
            empty_as_none(&self.diagnostics.lakehouse_cache_read_mode),
            empty_as_none(&self.diagnostics.lakehouse_cache_fill_mode),
            empty_as_none(&self.diagnostics.source_fallback_reason),
            self.diagnostics.direct_device_read_disposition.code().to_string(),
            self.staging_summary.mode.code().to_string(),
            self.staging_summary.status.code().to_string(),
            self.staging_summary.staged_data_file_count.to_string(),
            self.staging_summary.staged_bytes.to_string(),
            self.native_planning_outcome.as_str().to_string(),
            self.native_reason_code.clone().unwrap_or_else(|| "none".to_string()),
            self.post_native_execution_path.as_str().to_string(),
            self.cpu_arrow_reader_fallback.to_string(),
            self.native_runtime_started.to_string(),
            self.native_runtime_outcome.as_str().to_string(),
            optional_u128(self.native_runtime_elapsed_ms),
            optional_usize(self.native_output_rows),
            optional_usize(self.native_output_batches),
            self.source_fetch_bytes.to_string(),
            self.source_fetch_count.to_string(),
            optional_duration_ms_text(self.host_fetch_budget_wait_time_ns),
            optional_duration_ms_text(self.source_fetch_concurrency_wait_time_ns),
            optional_duration_ms_text(self.source_fetch_time_ns),
            optional_duration_ms_text(self.source_decode_time_ns),
            self.native_source_read_runtime_rows.clone(),
            self.lakehouse_cache_fill_event_snapshot_rows.clone(),
            optional_u64(self.expected_count),
            optional_u64(self.native_count_value),
            self.cpu_runtime_started.to_string(),
            self.cpu_runtime_outcome.as_str().to_string(),
            optional_u128(self.cpu_runtime_elapsed_ms),
            optional_usize(self.cpu_output_rows),
            optional_usize(self.cpu_output_batches),
            self.cpu_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.native_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.comparison_mode.as_str().to_string(),
            self.parity_status.as_str().to_string(),
            self.parity_detail.clone().unwrap_or_else(|| "none".to_string()),
            self.credential_source.clone(),
        ]
        .into_iter()
        .map(|field| stable_tsv_field(&field))
        .collect::<Vec<_>>()
        .join("\t")
    }
}
