use std::collections::BTreeMap;

use super::{arrival::Phase, workload::WorkloadFamily};

/// Closed outcome taxonomy for one arrival.
///
/// Every arrival inside the observation window lands in exactly one class, so
/// the goodput numerator and every failure ratio share one denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum OutcomeClass {
    /// Correct result delivered. SLO compliance is tracked separately.
    Succeeded,
    /// Terminal success whose output disagreed with the frozen calibration.
    OutputMismatch,
    /// Shed on arrival because the admission queue was full.
    RejectedQueueFull,
    /// Admitted to the queue, then expired against the queue wait bound.
    RejectedQueueTimeout,
    /// Exceeded the query's own cap.
    FailedQueryCap,
    /// The device was physically exhausted.
    FailedPhysicalOom,
    /// Allocation failed without an attributable source.
    FailedUnknownAllocation,
    /// Cancelled, including the harness cancelling after its own timeout.
    Cancelled,
    /// Planning refused the plan; this is a support fact, not a memory failure.
    FailedUnsupported,
    /// Any other terminal failure.
    FailedOther,
}

impl OutcomeClass {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::OutputMismatch => "output_mismatch",
            Self::RejectedQueueFull => "rejected_queue_full",
            Self::RejectedQueueTimeout => "rejected_queue_timeout",
            Self::FailedQueryCap => "failed_query_cap",
            Self::FailedPhysicalOom => "failed_physical_oom",
            Self::FailedUnknownAllocation => "failed_unknown_allocation",
            Self::Cancelled => "cancelled",
            Self::FailedUnsupported => "failed_unsupported",
            Self::FailedOther => "failed_other",
        }
    }

    /// Whether this outcome counts against the memory-failure tolerance.
    ///
    /// Queue rejections and unsupported plans are admission and capability
    /// facts; they are reported separately and never inflate the memory
    /// failure ratio the comparison groups are held to.
    pub(super) fn is_memory_failure(self) -> bool {
        matches!(self, Self::FailedQueryCap | Self::FailedPhysicalOom | Self::FailedUnknownAllocation)
    }
}

/// Classifies one arrival from its stable labels.
///
/// Inputs are labels rather than engine types so the taxonomy is verifiable
/// without a device. `fingerprint_match` is `None` when no frozen output was
/// available to compare against.
pub(super) fn classify(
    terminal_outcome: Option<&str>,
    failure_code: Option<&str>,
    failure_reason: Option<&str>,
    adapter_error_code: Option<&str>,
    fingerprint_match: Option<bool>,
) -> OutcomeClass {
    let code = failure_code.or(adapter_error_code);
    if let Some(code) = code {
        return match code {
            "query_memory_limit_exceeded" => OutcomeClass::FailedQueryCap,
            "device_out_of_memory" => OutcomeClass::FailedPhysicalOom,
            "device_allocation_failed" | "memory_reservation_failed" | "cudf_resource_exhausted" => {
                OutcomeClass::FailedUnknownAllocation
            },
            "cancelled" => OutcomeClass::Cancelled,
            "service_overloaded" | "query_budget_unavailable" => match failure_reason {
                Some("admission_queue_full") => OutcomeClass::RejectedQueueFull,
                _ => OutcomeClass::RejectedQueueTimeout,
            },
            "unsupported_plan_shape"
            | "unsupported_operator"
            | "unsupported_expression"
            | "unsupported_type"
            | "invalid_plan"
            | "required_gpu_operation_unsupported"
            | "final_plan_requirement_unsatisfied"
            | "invalid_execution_plan" => OutcomeClass::FailedUnsupported,
            _ => OutcomeClass::FailedOther,
        };
    }
    match terminal_outcome {
        Some("succeeded") => match fingerprint_match {
            Some(false) => OutcomeClass::OutputMismatch,
            Some(true) | None => OutcomeClass::Succeeded,
        },
        Some("cancelled") => OutcomeClass::Cancelled,
        Some(_) => OutcomeClass::FailedOther,
        None => OutcomeClass::FailedOther,
    }
}

/// One measured arrival.
#[derive(Debug, Clone)]
pub(super) struct AttemptRecord {
    pub(super) sequence: u64,
    pub(super) phase: Phase,
    pub(super) item_id: String,
    pub(super) family: WorkloadFamily,
    pub(super) work_units: u64,
    /// Cap assigned by the arrival trace before query-service admission.
    pub(super) requested_query_cap_bytes: u64,
    pub(super) scheduled_offset_ms: f64,
    pub(super) submit_offset_ms: f64,
    pub(super) submit_delay_ms: f64,
    pub(super) end_to_end_ms: f64,
    pub(super) queue_wait_ms: Option<f64>,
    pub(super) planning_ms: Option<f64>,
    pub(super) execution_ms: Option<f64>,
    pub(super) cleanup_ms: Option<f64>,
    pub(super) attempt_total_ms: Option<f64>,
    pub(super) device_ordinal: Option<usize>,
    pub(super) grant_bytes: Option<u64>,
    pub(super) disposition: String,
    pub(super) terminal_outcome: String,
    pub(super) outcome: OutcomeClass,
    pub(super) attempt_peak_aligned_bytes: Option<u64>,
    pub(super) attempt_terminal_usage_provenance: String,
    pub(super) attempt_pool_reserved_high_bytes: Option<u64>,
    pub(super) unattributed_allocator_provenance: String,
    pub(super) cleanup_disposition: String,
    pub(super) capacity_disposition: String,
    pub(super) failure_code: Option<String>,
    pub(super) failure_status: Option<String>,
    pub(super) failure_operation: Option<String>,
    pub(super) failure_reason: Option<String>,
    pub(super) output_rows: Option<usize>,
    pub(super) output_fingerprint: Option<String>,
    pub(super) fingerprint_match: Option<bool>,
    pub(super) slo_deadline_ms: Option<f64>,
    pub(super) slo_met: bool,
    pub(super) harness_timeout: bool,
    pub(super) error_message: Option<String>,
}

impl AttemptRecord {
    pub(super) fn meets(&self, deadline_ms: Option<f64>) -> bool {
        self.outcome == OutcomeClass::Succeeded && deadline_ms.is_some_and(|deadline| self.end_to_end_ms <= deadline)
    }
}

/// Aggregate outcome of one comparison group's observation window.
#[derive(Debug, Clone, Default)]
pub(super) struct WindowSummary {
    pub(super) offered: usize,
    pub(super) completed: usize,
    pub(super) succeeded: usize,
    pub(super) slo_met: usize,
    pub(super) slo_missed: usize,
    pub(super) work_units_offered: u64,
    pub(super) work_units_slo_met: u64,
    pub(super) outcome_counts: BTreeMap<&'static str, usize>,
    pub(super) memory_failures: usize,
    pub(super) unsupported: usize,
    pub(super) non_native_dispositions: usize,
    pub(super) unattributed_allocations: usize,
}

impl WindowSummary {
    /// Builds the summary from the window's arrivals.
    ///
    /// `offered` counts every arrival scheduled inside the window regardless
    /// of outcome, so shed and failed arrivals stay in the denominator.
    pub(super) fn from_records(records: &[AttemptRecord]) -> Self {
        let mut summary = Self::default();
        for record in records {
            summary.offered += 1;
            summary.work_units_offered += record.work_units;
            *summary.outcome_counts.entry(record.outcome.as_str()).or_default() += 1;
            if record.outcome.is_memory_failure() {
                summary.memory_failures += 1;
            }
            if record.outcome == OutcomeClass::FailedUnsupported {
                summary.unsupported += 1;
            }
            if !matches!(record.outcome, OutcomeClass::RejectedQueueFull | OutcomeClass::RejectedQueueTimeout) {
                summary.completed += 1;
            }
            if record.outcome == OutcomeClass::Succeeded {
                summary.succeeded += 1;
                if record.slo_met {
                    summary.slo_met += 1;
                    summary.work_units_slo_met += record.work_units;
                } else {
                    summary.slo_missed += 1;
                }
            }
            if record.disposition != "native" && record.terminal_outcome == "succeeded" {
                summary.non_native_dispositions += 1;
            }
            if record.unattributed_allocator_provenance != "unavailable" {
                summary.unattributed_allocations += 1;
            }
        }
        summary
    }

    pub(super) fn goodput_qps(&self, window_seconds: f64) -> f64 {
        if window_seconds <= 0.0 { 0.0 } else { self.slo_met as f64 / window_seconds }
    }

    pub(super) fn goodput_work_units_per_second(&self, window_seconds: f64) -> f64 {
        if window_seconds <= 0.0 { 0.0 } else { self.work_units_slo_met as f64 / window_seconds }
    }

    pub(super) fn memory_failure_ratio(&self) -> f64 {
        if self.offered == 0 { 0.0 } else { self.memory_failures as f64 / self.offered as f64 }
    }

    pub(super) fn count(&self, outcome: OutcomeClass) -> usize {
        self.outcome_counts.get(outcome.as_str()).copied().unwrap_or(0)
    }
}

/// Nearest-rank percentile over an already sorted sample.
pub(super) fn percentile(sorted: &[f64], quantile: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (quantile * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted.get(rank.min(sorted.len()) - 1).copied()
}

pub(super) fn sorted_samples(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut samples = values.collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples
}
