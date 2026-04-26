use std::fs;

use super::{
    config::{native_pipeline_execution_mode_name, slot_guarantee_name},
    *,
};

const TSV_COLUMNS: &[&str] = &[
    "row_type",
    "request_index",
    "tenant_id",
    "workload_class",
    "query_id",
    "iteration",
    "admission_sequence",
    "admission_wait_ms",
    "device_ordinal",
    "query_grant_bytes",
    "service_capacity_bytes",
    "service_outstanding_bytes_snapshot",
    "service_available_bytes_snapshot",
    "total_wall_ms",
    "native_wall_ms",
    "planning_outcome",
    "final_disposition",
    "native_runtime_outcome",
    "parity_status",
    "cpu_output_rows",
    "native_output_rows",
    "cpu_output_fingerprint",
    "native_output_fingerprint",
    "error_code",
    "error_kind",
    "error_status",
    "error_operation",
    "error_facts",
    "error_message",
];
impl StructuredFailure {
    pub(super) fn from_query_error(error: &QueryError) -> Self {
        Self {
            code: Some(error.code().as_str().to_string()),
            kind: Some(error.kind().as_str().to_string()),
            status: Some(error.status().as_str().to_string()),
            operation: Some(error.operation().to_string()),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            message: Some(error.to_string()),
        }
    }

    pub(super) fn from_datafusion_error(error: &DataFusionError) -> Self {
        error.find_nexus_error().map(Self::from_adapter_error).unwrap_or_else(|| Self {
            code: Some("datafusion_error".to_string()),
            message: Some(error.to_string()),
            ..Self::default()
        })
    }

    fn from_adapter_error(error: &AdapterError) -> Self {
        Self {
            code: Some(error.code().as_str().to_string()),
            kind: Some(error.kind().as_str().to_string()),
            status: Some(error.status().as_str().to_string()),
            operation: Some(error.operation().to_string()),
            facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
            message: Some(error.to_string()),
        }
    }
}

impl StressReport {
    pub(super) fn to_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("# dataset_root={}\n", self.config.dataset_root.display()));
        output.push_str(&format!("# dataset_metadata_fingerprint={}\n", self.dataset.metadata_fingerprint));
        output.push_str(&format!("# device_capacity_bytes={}\n", self.config.device_capacity_bytes));
        output.push_str(&format!("# tenant_min_grant_bytes={}\n", self.config.tenant_min_grant_bytes));
        output.push_str(&format!("# tenant_want_grant_bytes={}\n", self.config.tenant_want_grant_bytes));
        output.push_str(&format!("# slot_guarantee={}\n", slot_guarantee_name(self.config.slot_guarantee)));
        output.push_str(&format!("# max_concurrent_queries={}\n", self.config.max_concurrent_queries));
        output.push_str(&format!("# effective_active_max={}\n", self.effective_active_max));
        output.push_str(&format!("# max_admission_queue_len={}\n", self.config.max_admission_queue_len));
        output.push_str(&format!("# max_source_chunk_bytes={}\n", self.config.max_source_chunk_bytes));
        output.push_str(&format!("# max_row_groups_per_chunk={}\n", self.config.max_row_groups_per_chunk));
        output.push_str(&format!(
            "# source_cache_max_retained_bytes={}\n",
            self.config.source_cache_max_retained_bytes.map_or_else(|| "none".to_string(), |value| value.to_string())
        ));
        output.push_str(&format!(
            "# memory_manager_pool_initial_bytes={}\n",
            self.config.memory_manager_pool_initial_bytes.map_or_else(|| "none".to_string(), |value| value.to_string())
        ));
        output.push_str(&format!(
            "# native_pipeline_execution_mode={}\n",
            native_pipeline_execution_mode_name(self.config.native_pipeline_execution_mode)
        ));
        output.push_str(&format!(
            "# query_memory_domain_ceiling_multiplier={}\n",
            self.config
                .query_memory_domain_ceiling_multiplier
                .map_or_else(|| "none".to_string(), |value| value.to_string())
        ));
        output.push_str(&format!("# warmup_rounds={}\n", self.config.warmup_rounds));
        output.push_str(&format!("# stream_idle_timeout_ms={}\n", self.config.stream_idle_timeout.as_millis()));
        output.push_str(&format!("# cuda_profiler_capture={}\n", self.config.cuda_profiler_capture));
        if let Some(profile) = &self.config.profile {
            output.push_str(&format!("# profile_query_id={}\n", profile.query_id));
            output.push_str(&format!("# profile_lanes={}\n", profile.lanes));
            output.push_str(&format!("# profile_warmup_iterations={}\n", profile.warmup_iterations));
            output.push_str(&format!("# profile_measured_iterations={}\n", profile.measured_iterations));
            output.push_str(&format!("# profile_evidence_path={}\n", profile.evidence_path.display()));
            output.push_str(&format!(
                "# profile_measured_execution_ms={}\n",
                self.profile_measured_execution_ms.map_or_else(|| "none".to_string(), format_ms)
            ));
        }
        for summary in tenant_summaries(&self.rows) {
            output.push_str(&format!(
                "# tenant_summary tenant_id={} succeeded={} overloaded={} failed={} max_total_wall_ms={} p95_total_wall_ms={}\n",
                summary.tenant_id,
                summary.succeeded,
                summary.overloaded,
                summary.failed,
                format_ms(summary.max_total_wall_ms),
                format_ms(summary.p95_total_wall_ms),
            ));
        }
        output.push_str(&TSV_COLUMNS.join("\t"));
        output.push('\n');
        for row in &self.rows {
            output.push_str(&row.to_tsv_line());
            output.push('\n');
        }
        output
    }
}

impl ProfileEvidenceReport {
    const COLUMNS: [&'static str; 15] = [
        "phase",
        "lane",
        "iteration",
        "request_index",
        "admission_sequence",
        "device_ordinal",
        "query_id",
        "native_runtime_outcome",
        "parity_status",
        "source_cache_hits",
        "source_cache_misses",
        "source_cache_inserts",
        "source_cache_insert_skipped",
        "source_cache_retained_bytes",
        "native_wall_ms",
    ];

    pub(super) fn write_to(&self, path: &Path) -> DataFusionResult<()> {
        fs::write(path, self.to_tsv()).map_err(|error| {
            plan_error(format!("failed to write profile evidence sidecar {}: {error}", path.display()))
        })
    }

    fn to_tsv(&self) -> String {
        let mut output = Self::COLUMNS.join("\t");
        output.push('\n');
        for row in &self.rows {
            output.push_str(&row.to_tsv_line());
            output.push('\n');
        }
        output
    }
}

impl ProfileEvidenceRow {
    fn to_tsv_line(&self) -> String {
        let fields = [
            self.phase.to_string(),
            self.lane.to_string(),
            self.iteration.to_string(),
            self.request_index.to_string(),
            self.admission_sequence.to_string(),
            self.device_ordinal.to_string(),
            self.query_id.clone(),
            self.native_runtime_outcome.to_string(),
            self.parity_status.to_string(),
            self.source_cache_hits.to_string(),
            self.source_cache_misses.to_string(),
            self.source_cache_inserts.to_string(),
            self.source_cache_insert_skipped.to_string(),
            self.source_cache_retained_bytes.to_string(),
            self.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), ProfileEvidenceReport::COLUMNS.len());
        fields.into_iter().map(|field| tsv_escape(&field)).collect::<Vec<_>>().join("\t")
    }
}

impl StressRow {
    fn to_tsv_line(&self) -> String {
        let fields = [
            self.row_type.to_string(),
            self.request_index.to_string(),
            self.tenant_id.clone(),
            self.workload_class.to_string(),
            self.query_id.clone(),
            self.iteration.to_string(),
            self.admission_sequence.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_wait_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            self.admission_facts.device_ordinal.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_facts.query_grant_bytes.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.admission_facts
                .service_capacity_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.admission_facts
                .service_outstanding_bytes_snapshot
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.admission_facts
                .service_available_bytes_snapshot
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            format_ms(self.total_wall_ms),
            self.native_wall_ms.map(format_ms).unwrap_or_else(|| "none".to_string()),
            self.planning_outcome.clone(),
            self.final_disposition.clone(),
            self.native_runtime_outcome.to_string(),
            self.parity_status.to_string(),
            self.cpu_output_rows.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.native_output_rows.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string()),
            self.cpu_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.native_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.error.code.clone().unwrap_or_else(|| "none".to_string()),
            self.error.kind.clone().unwrap_or_else(|| "none".to_string()),
            self.error.status.clone().unwrap_or_else(|| "none".to_string()),
            self.error.operation.clone().unwrap_or_else(|| "none".to_string()),
            format_facts(&self.error.facts),
            self.error.message.clone().unwrap_or_else(|| "none".to_string()),
        ];
        debug_assert_eq!(fields.len(), TSV_COLUMNS.len());
        fields.into_iter().map(|field| tsv_escape(&field)).collect::<Vec<_>>().join("\t")
    }
}

#[derive(Debug, Clone)]
struct TenantSummary {
    tenant_id: String,
    succeeded: usize,
    overloaded: usize,
    failed: usize,
    max_total_wall_ms: f64,
    p95_total_wall_ms: f64,
}

fn tenant_summaries(rows: &[StressRow]) -> Vec<TenantSummary> {
    let mut by_tenant: BTreeMap<String, Vec<&StressRow>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.row_type == "query") {
        by_tenant.entry(row.tenant_id.clone()).or_default().push(row);
    }
    by_tenant
        .into_iter()
        .map(|(tenant_id, rows)| {
            let mut latencies = rows.iter().map(|row| row.total_wall_ms).collect::<Vec<_>>();
            latencies.sort_by(f64::total_cmp);
            let succeeded = rows.iter().filter(|row| row.native_runtime_outcome == "succeeded").count();
            let overloaded = rows.iter().filter(|row| row.error.code.as_deref() == Some("service_overloaded")).count();
            let failed = rows.len().saturating_sub(succeeded + overloaded);
            TenantSummary {
                tenant_id,
                succeeded,
                overloaded,
                failed,
                max_total_wall_ms: latencies.last().copied().unwrap_or(0.0),
                p95_total_wall_ms: percentile(&latencies, 0.95),
            }
        })
        .collect()
}

fn percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let index = ((sorted_values.len() - 1) as f64 * quantile).ceil() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::{
        super::config::{MAX_ADMISSION_QUEUE_LEN_ENV, SLOT_GUARANTEE_ENV},
        *,
    };

    fn queue_test_config() -> StressConfig {
        StressConfig {
            dataset_root: PathBuf::new(),
            bulk_queries: vec!["q01".to_string()],
            interactive_queries: vec!["q06".to_string()],
            bulk_iterations: 1,
            interactive_iterations: 1,
            device_capacity_bytes: 100,
            tenant_min_grant_bytes: 25,
            tenant_want_grant_bytes: 50,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_concurrent_queries: 2,
            max_admission_queue_len: 2,
            admission_wait_timeout: Duration::from_secs(1),
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            source_cache_max_retained_bytes: None,
            memory_manager_pool_initial_bytes: None,
            native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
            query_memory_domain_ceiling_multiplier: None,
            warmup_rounds: 0,
            stream_idle_timeout: Duration::from_secs(1),
            cuda_profiler_capture: false,
            profile: None,
            run_overload_probe: false,
            device_ordinals: vec![0],
        }
    }

    #[test]
    fn percentile_uses_ceil_index_for_tail_latency() {
        assert_eq!(percentile(&[], 0.95), 0.0);
        assert_eq!(percentile(&[10.0], 0.95), 10.0);
        assert_eq!(percentile(&[10.0, 20.0, 30.0], 0.95), 30.0);
    }

    #[test]
    fn tsv_escape_removes_row_and_field_breaks() {
        assert_eq!(tsv_escape("a\tb\nc\rd"), "a\\tb\\nc\\rd");
    }

    #[test]
    fn stress_tsv_row_contract_matches_columns() {
        assert_eq!(TSV_COLUMNS.first(), Some(&"row_type"));
        assert_eq!(TSV_COLUMNS.last(), Some(&"error_message"));
        let row = StressRow {
            row_type: "query",
            request_index: 1,
            tenant_id: "interactive".to_string(),
            workload_class: "interactive",
            query_id: "q06".to_string(),
            iteration: 1,
            admission_sequence: Some(1),
            admission_wait_ms: Some(0.0),
            admission_facts: AdmissionFacts::default(),
            total_wall_ms: 10.0,
            native_wall_ms: Some(5.0),
            planning_outcome: "selected".to_string(),
            final_disposition: "native".to_string(),
            native_runtime_outcome: "succeeded",
            parity_status: "matched",
            cpu_output_rows: Some(1),
            native_output_rows: Some(1),
            cpu_output_fingerprint: Some("cpu".to_string()),
            native_output_fingerprint: Some("native".to_string()),
            error: StructuredFailure::default(),
        };
        assert_eq!(row.to_tsv_line().split('\t').count(), TSV_COLUMNS.len());
    }

    #[test]
    fn stress_config_rejects_queue_smaller_than_synchronized_wave() {
        let config = queue_test_config();

        let error = config.validate().unwrap_err();

        assert!(error.to_string().contains(MAX_ADMISSION_QUEUE_LEN_ENV));
        assert!(error.to_string().contains("synchronized request count (3)"));
    }

    #[test]
    fn profile_mode_does_not_apply_barrier_queue_preflight() {
        let mut config = queue_test_config();
        config.max_admission_queue_len = 1;
        config.profile = Some(StressProfileConfig {
            query_id: "q06".to_string(),
            lanes: 1,
            warmup_iterations: 1,
            measured_iterations: 1,
            evidence_path: PathBuf::from("profile.tsv"),
        });

        config.validate().unwrap();
    }

    #[test]
    fn stress_config_rejects_overcommitted_guaranteed_minimum_band() {
        let config = StressConfig {
            dataset_root: PathBuf::new(),
            bulk_queries: vec!["q01".to_string()],
            interactive_queries: vec!["q06".to_string()],
            bulk_iterations: 1,
            interactive_iterations: 1,
            device_capacity_bytes: 100,
            tenant_min_grant_bytes: 60,
            tenant_want_grant_bytes: 60,
            slot_guarantee: SlotGuarantee::GuaranteedMinimum,
            max_concurrent_queries: 2,
            max_admission_queue_len: 1,
            admission_wait_timeout: Duration::from_secs(1),
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            source_cache_max_retained_bytes: None,
            memory_manager_pool_initial_bytes: None,
            native_pipeline_execution_mode: NativePipelineExecutionMode::StrictBounded,
            query_memory_domain_ceiling_multiplier: None,
            warmup_rounds: 0,
            stream_idle_timeout: Duration::from_secs(1),
            cuda_profiler_capture: false,
            profile: None,
            run_overload_probe: false,
            device_ordinals: vec![0],
        };

        let error = config.validate().unwrap_err();

        assert!(error.to_string().contains(SLOT_GUARANTEE_ENV));
    }

    #[test]
    fn tenant_summary_counts_success_overload_and_failures() {
        let rows = vec![
            StressRow {
                row_type: "query",
                request_index: 1,
                tenant_id: "interactive".to_string(),
                workload_class: "interactive",
                query_id: "q06".to_string(),
                iteration: 1,
                admission_sequence: Some(1),
                admission_wait_ms: Some(0.0),
                admission_facts: AdmissionFacts::default(),
                total_wall_ms: 10.0,
                native_wall_ms: Some(5.0),
                planning_outcome: "selected".to_string(),
                final_disposition: "native".to_string(),
                native_runtime_outcome: "succeeded",
                parity_status: "matched",
                cpu_output_rows: Some(1),
                native_output_rows: Some(1),
                cpu_output_fingerprint: Some("a".to_string()),
                native_output_fingerprint: Some("a".to_string()),
                error: StructuredFailure::default(),
            },
            StressRow {
                row_type: "query",
                request_index: 2,
                tenant_id: "interactive".to_string(),
                workload_class: "interactive",
                query_id: "q06".to_string(),
                iteration: 2,
                admission_sequence: None,
                admission_wait_ms: None,
                admission_facts: AdmissionFacts::default(),
                total_wall_ms: 20.0,
                native_wall_ms: None,
                planning_outcome: "not_planned".to_string(),
                final_disposition: "not_planned".to_string(),
                native_runtime_outcome: "not_started",
                parity_status: "not_run",
                cpu_output_rows: None,
                native_output_rows: None,
                cpu_output_fingerprint: None,
                native_output_fingerprint: None,
                error: StructuredFailure { code: Some("service_overloaded".to_string()), ..Default::default() },
            },
        ];

        let summaries = tenant_summaries(&rows);

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].succeeded, 1);
        assert_eq!(summaries[0].overloaded, 1);
        assert_eq!(summaries[0].failed, 0);
        assert_eq!(summaries[0].max_total_wall_ms, 20.0);
    }
}
