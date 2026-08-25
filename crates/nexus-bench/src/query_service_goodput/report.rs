use std::{error::Error, fmt::Write as _, fs, path::Path};

use datafusion_nexus::attempt::AttemptBackendCountersSnapshot;

use super::{
    calibration::CalibrationIdentity,
    config::GoodputConfig,
    record::{AttemptRecord, OutcomeClass, WindowSummary, percentile, sorted_samples},
    sampling::DeviceSample,
};

const ATTEMPT_COLUMNS: &[&str] = &[
    "sequence",
    "phase",
    "item_id",
    "family",
    "work_units",
    "scheduled_offset_ms",
    "submit_offset_ms",
    "submit_delay_ms",
    "end_to_end_ms",
    "queue_wait_ms",
    "planning_ms",
    "execution_ms",
    "cleanup_ms",
    "attempt_total_ms",
    "device_ordinal",
    "grant_bytes",
    "disposition",
    "terminal_outcome",
    "outcome_class",
    "attempt_peak_aligned_bytes",
    "attempt_terminal_usage_provenance",
    "attempt_pool_reserved_high_bytes",
    "unattributed_allocator_provenance",
    "cleanup_disposition",
    "capacity_disposition",
    "failure_code",
    "failure_status",
    "failure_operation",
    "failure_reason",
    "output_rows",
    "output_fingerprint",
    "fingerprint_match",
    "slo_deadline_ms",
    "slo_met",
    "harness_timeout",
    "error_message",
];

const SAMPLE_COLUMNS: &[&str] = &[
    "offset_ms",
    "device_ordinal",
    "active_queries",
    "queued_queries",
    "managed_capacity_bytes",
    "live_grants_bytes",
    "available_bytes",
    "active_grants",
    "pool_reserved_current_bytes",
    "pool_reserved_high_bytes",
    "pool_used_current_bytes",
    "pool_used_high_bytes",
    "allocation_failure_count",
    "trim_retry_count",
    "host_rss_bytes",
    "host_rss_peak_bytes",
];

pub(super) const SUMMARY_COLUMNS: &[&str] = &[
    "run_id",
    "group",
    "qualification",
    "arrival",
    "rate_qps",
    "rate_multiplier",
    "seed",
    "warmup_seconds",
    "window_seconds",
    "mix",
    "pool_state",
    "device_ordinals",
    "device_capacity_bytes",
    "query_cap_bytes",
    "max_active_attempts",
    "lane_count",
    "max_queued_attempts",
    "max_queue_wait_ms",
    "max_source_chunk_bytes",
    "max_row_groups_per_chunk",
    "slo_multiplier",
    "gpu_model",
    "tpch_scale",
    "tpcds_scale",
    "graph_dataset",
    "tpch_fingerprint",
    "tpcds_fingerprint",
    "graph_fingerprint",
    "source_identity",
    "offered",
    "completed",
    "succeeded",
    "slo_met",
    "slo_missed",
    "goodput_qps",
    "goodput_work_units_per_second",
    "work_units_offered",
    "work_units_slo_met",
    "memory_failure_ratio",
    "output_mismatch",
    "rejected_queue_full",
    "rejected_queue_timeout",
    "failed_query_cap",
    "failed_physical_oom",
    "failed_unknown_allocation",
    "cancelled",
    "failed_unsupported",
    "failed_other",
    "non_native_dispositions",
    "unattributed_allocations",
    "harness_timeouts",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "queue_wait_p50_ms",
    "queue_wait_p95_ms",
    "queue_wait_p99_ms",
    "execution_p50_ms",
    "execution_p95_ms",
    "execution_p99_ms",
    "cleanup_p50_ms",
    "cleanup_p95_ms",
    "cleanup_p99_ms",
    "attempt_peak_aligned_bytes_p50",
    "attempt_peak_aligned_bytes_max",
    "attempt_peak_unavailable",
    "device_pool_reserved_high_bytes",
    "device_pool_used_high_bytes",
    "device_live_grants_peak_bytes",
    "device_active_queries_peak",
    "host_rss_peak_bytes",
    "device_health",
    "attempts_begun",
    "reports_ready",
    "reports_unpublished",
];

const SENSITIVITY_COLUMNS: &[&str] = &["run_id", "group", "slo_multiplier", "slo_met", "goodput_qps"];

/// Device-scope aggregates for one run.
///
/// These are device facts and are labelled as such in every column name; they
/// are never used as an attempt or operator peak.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DeviceAggregates {
    pub(super) pool_reserved_high_bytes: u64,
    pub(super) pool_used_high_bytes: u64,
    pub(super) live_grants_peak_bytes: u64,
    pub(super) active_queries_peak: usize,
    pub(super) host_rss_peak_bytes: u64,
}

impl DeviceAggregates {
    pub(super) fn from_samples(samples: &[DeviceSample]) -> Self {
        let mut aggregates = Self::default();
        for sample in samples {
            aggregates.pool_reserved_high_bytes =
                aggregates.pool_reserved_high_bytes.max(sample.pool_reserved_high_bytes);
            aggregates.pool_used_high_bytes = aggregates.pool_used_high_bytes.max(sample.pool_used_high_bytes);
            aggregates.live_grants_peak_bytes = aggregates.live_grants_peak_bytes.max(sample.live_grants_bytes);
            aggregates.active_queries_peak = aggregates.active_queries_peak.max(sample.active_queries);
            aggregates.host_rss_peak_bytes = aggregates.host_rss_peak_bytes.max(sample.host_rss_peak_bytes);
        }
        aggregates
    }
}

pub(super) struct RunReport<'a> {
    pub(super) config: &'a GoodputConfig,
    pub(super) identity: &'a CalibrationIdentity,
    pub(super) qualification: &'a str,
    pub(super) rate_qps: f64,
    pub(super) records: &'a [AttemptRecord],
    pub(super) window: &'a WindowSummary,
    pub(super) window_records: &'a [AttemptRecord],
    pub(super) samples: &'a [DeviceSample],
    pub(super) sensitivity: &'a [(f64, usize)],
    /// Per-device admission health after the run. A group that ends with a
    /// device closed did not measure the configuration it claims to measure.
    pub(super) device_health: &'a str,
    /// Exact backend attempt counters read after every attempt's report wait
    /// returned. `attempts_begun - reports_ready` is then the number of
    /// attempts whose report never published, so an incomplete observation
    /// is a run fact rather than a line in stderr.
    pub(super) attempt_counters: AttemptBackendCountersSnapshot,
}

impl RunReport<'_> {
    pub(super) fn write_all(&self) -> Result<(), Box<dyn Error>> {
        let run_dir = self.config.run_dir();
        fs::create_dir_all(&run_dir)?;
        write_file(&run_dir.join("attempts.tsv"), &self.attempts_tsv())?;
        write_file(&run_dir.join("device_samples.tsv"), &self.samples_tsv())?;
        write_file(&run_dir.join("summary.tsv"), &self.summary_tsv())?;
        write_file(&run_dir.join("slo_sensitivity.tsv"), &self.sensitivity_tsv())?;
        Ok(())
    }

    fn attempts_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&ATTEMPT_COLUMNS.join("\t"));
        output.push('\n');
        for record in self.records {
            let fields = [
                record.sequence.to_string(),
                record.phase.as_str().to_string(),
                record.item_id.clone(),
                record.family.as_str().to_string(),
                record.work_units.to_string(),
                format_ms(record.scheduled_offset_ms),
                format_ms(record.submit_offset_ms),
                format_ms(record.submit_delay_ms),
                format_ms(record.end_to_end_ms),
                optional_ms(record.queue_wait_ms),
                optional_ms(record.planning_ms),
                optional_ms(record.execution_ms),
                optional_ms(record.cleanup_ms),
                optional_ms(record.attempt_total_ms),
                optional(record.device_ordinal),
                optional(record.grant_bytes),
                record.disposition.clone(),
                record.terminal_outcome.clone(),
                record.outcome.as_str().to_string(),
                optional(record.attempt_peak_aligned_bytes),
                record.attempt_terminal_usage_provenance.clone(),
                optional(record.attempt_pool_reserved_high_bytes),
                record.unattributed_allocator_provenance.clone(),
                record.cleanup_disposition.clone(),
                record.capacity_disposition.clone(),
                optional_text(record.failure_code.as_deref()),
                optional_text(record.failure_status.as_deref()),
                optional_text(record.failure_operation.as_deref()),
                optional_text(record.failure_reason.as_deref()),
                optional(record.output_rows),
                optional_text(record.output_fingerprint.as_deref()),
                optional(record.fingerprint_match),
                optional_ms(record.slo_deadline_ms),
                record.slo_met.to_string(),
                record.harness_timeout.to_string(),
                optional_text(record.error_message.as_deref()),
            ];
            debug_assert_eq!(fields.len(), ATTEMPT_COLUMNS.len());
            output.push_str(&fields.join("\t"));
            output.push('\n');
        }
        output
    }

    fn samples_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&SAMPLE_COLUMNS.join("\t"));
        output.push('\n');
        for sample in self.samples {
            let fields = [
                format_ms(sample.offset_ms),
                sample.device_ordinal.to_string(),
                sample.active_queries.to_string(),
                sample.queued_queries.to_string(),
                sample.managed_capacity_bytes.to_string(),
                sample.live_grants_bytes.to_string(),
                sample.available_bytes.to_string(),
                sample.active_grants.to_string(),
                sample.pool_reserved_current_bytes.to_string(),
                sample.pool_reserved_high_bytes.to_string(),
                sample.pool_used_current_bytes.to_string(),
                sample.pool_used_high_bytes.to_string(),
                sample.allocation_failure_count.to_string(),
                sample.trim_retry_count.to_string(),
                sample.host_rss_bytes.to_string(),
                sample.host_rss_peak_bytes.to_string(),
            ];
            debug_assert_eq!(fields.len(), SAMPLE_COLUMNS.len());
            output.push_str(&fields.join("\t"));
            output.push('\n');
        }
        output
    }

    pub(super) fn summary_fields(&self) -> Vec<String> {
        let config = self.config;
        let window = self.window;
        let aggregates = DeviceAggregates::from_samples(self.samples);
        let e2e = sorted_samples(self.window_records.iter().map(|record| record.end_to_end_ms));
        let queue = sorted_samples(self.window_records.iter().filter_map(|record| record.queue_wait_ms));
        let execution = sorted_samples(self.window_records.iter().filter_map(|record| record.execution_ms));
        let cleanup = sorted_samples(self.window_records.iter().filter_map(|record| record.cleanup_ms));
        let peaks = sorted_samples(
            self.window_records.iter().filter_map(|record| record.attempt_peak_aligned_bytes.map(|value| value as f64)),
        );
        let peak_unavailable =
            self.window_records.iter().filter(|record| record.attempt_peak_aligned_bytes.is_none()).count();
        let harness_timeouts = self.window_records.iter().filter(|record| record.harness_timeout).count();
        let fields = vec![
            config.run_id.clone(),
            config.group.clone(),
            self.qualification.to_string(),
            config.arrival.as_str().to_string(),
            format!("{:.6}", self.rate_qps),
            format!("{:.6}", config.rate_multiplier),
            config.seed.to_string(),
            format!("{:.3}", config.warmup_seconds),
            format!("{:.3}", config.window_seconds),
            config.mix.iter().map(|entry| format!("{}@{}", entry.item_id, entry.weight)).collect::<Vec<_>>().join("+"),
            config.pool_state.as_str().to_string(),
            config.device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join("+"),
            config.device_capacity_bytes.to_string(),
            config.query_cap_bytes.to_string(),
            config.max_active_attempts.to_string(),
            config.lane_count().to_string(),
            config.max_queued_attempts.to_string(),
            config.max_queue_wait.as_millis().to_string(),
            config.max_source_chunk_bytes.to_string(),
            config.max_row_groups_per_chunk.to_string(),
            format!("{:.3}", config.slo_multiplier),
            self.identity.gpu_model.clone(),
            config.tpch_scale.clone(),
            config.tpcds_scale.clone(),
            config.graph_dataset.clone(),
            empty_as_none(&self.identity.tpch_fingerprint),
            empty_as_none(&self.identity.tpcds_fingerprint),
            empty_as_none(&self.identity.graph_fingerprint),
            config.source_identity.clone(),
            window.offered.to_string(),
            window.completed.to_string(),
            window.succeeded.to_string(),
            window.slo_met.to_string(),
            window.slo_missed.to_string(),
            format!("{:.6}", window.goodput_qps(config.window_seconds)),
            format!("{:.6}", window.goodput_work_units_per_second(config.window_seconds)),
            window.work_units_offered.to_string(),
            window.work_units_slo_met.to_string(),
            format!("{:.6}", window.memory_failure_ratio()),
            window.count(OutcomeClass::OutputMismatch).to_string(),
            window.count(OutcomeClass::RejectedQueueFull).to_string(),
            window.count(OutcomeClass::RejectedQueueTimeout).to_string(),
            window.count(OutcomeClass::FailedQueryCap).to_string(),
            window.count(OutcomeClass::FailedPhysicalOom).to_string(),
            window.count(OutcomeClass::FailedUnknownAllocation).to_string(),
            window.count(OutcomeClass::Cancelled).to_string(),
            window.count(OutcomeClass::FailedUnsupported).to_string(),
            window.count(OutcomeClass::FailedOther).to_string(),
            window.non_native_dispositions.to_string(),
            window.unattributed_allocations.to_string(),
            harness_timeouts.to_string(),
            optional_ms(percentile(&e2e, 0.50)),
            optional_ms(percentile(&e2e, 0.95)),
            optional_ms(percentile(&e2e, 0.99)),
            optional_ms(percentile(&queue, 0.50)),
            optional_ms(percentile(&queue, 0.95)),
            optional_ms(percentile(&queue, 0.99)),
            optional_ms(percentile(&execution, 0.50)),
            optional_ms(percentile(&execution, 0.95)),
            optional_ms(percentile(&execution, 0.99)),
            optional_ms(percentile(&cleanup, 0.50)),
            optional_ms(percentile(&cleanup, 0.95)),
            optional_ms(percentile(&cleanup, 0.99)),
            optional(percentile(&peaks, 0.50).map(|value| value as u64)),
            optional(peaks.last().map(|value| *value as u64)),
            peak_unavailable.to_string(),
            aggregates.pool_reserved_high_bytes.to_string(),
            aggregates.pool_used_high_bytes.to_string(),
            aggregates.live_grants_peak_bytes.to_string(),
            aggregates.active_queries_peak.to_string(),
            aggregates.host_rss_peak_bytes.to_string(),
            self.device_health.to_string(),
            self.attempt_counters.attempts_begun.to_string(),
            self.attempt_counters.reports_ready.to_string(),
            self.attempt_counters.attempts_begun.saturating_sub(self.attempt_counters.reports_ready).to_string(),
        ];
        debug_assert_eq!(fields.len(), SUMMARY_COLUMNS.len());
        fields
    }

    fn summary_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&SUMMARY_COLUMNS.join("\t"));
        output.push('\n');
        output.push_str(&self.summary_fields().join("\t"));
        output.push('\n');
        output
    }

    fn sensitivity_tsv(&self) -> String {
        let mut output = String::new();
        output.push_str(&SENSITIVITY_COLUMNS.join("\t"));
        output.push('\n');
        for (multiplier, slo_met) in self.sensitivity {
            let _ = writeln!(
                output,
                "{}\t{}\t{multiplier:.3}\t{slo_met}\t{:.6}",
                self.config.run_id,
                self.config.group,
                *slo_met as f64 / self.config.window_seconds
            );
        }
        output
    }
}

fn write_file(path: &Path, contents: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, contents).map_err(|error| format!("failed to write {}: {error}", path.display()).into())
}

pub(super) fn format_ms(value: f64) -> String {
    format!("{value:.3}")
}

fn optional_ms(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_string(), format_ms)
}

fn optional<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "unavailable".to_string(), |value| value.to_string())
}

fn optional_text(value: Option<&str>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.replace(['\t', '\n', '\r'], " "))
}

fn empty_as_none(value: &str) -> String {
    if value.is_empty() { "none".to_string() } else { value.to_string() }
}
