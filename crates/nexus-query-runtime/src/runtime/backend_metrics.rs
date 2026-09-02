//! Fixed-cardinality backend metrics state.
//!
//! This module owns one bounded, backend-generation-scoped projection of
//! ledger, finalization, and telemetry transitions. It deliberately stores no
//! query correlations, report payloads, or application-provided labels.

use std::sync::{Arc, Mutex};

use nexus_query_planning::error::{QueryError, QueryErrorCode, Result};
use serde::Serialize;

pub use crate::metrics::BACKEND_ATTEMPT_DURATION_BUCKETS_NS;
#[cfg(test)]
use crate::runtime::gpu::LedgerTransitionKind;
use crate::{
    observability::{AttemptDisposition, RuntimeTerminalOutcome, SealedAttemptProjection},
    runtime::gpu::{DeviceAdmissionHealth, DeviceMemoryFailureReport, DeviceTelemetryStatus, LedgerTransition},
};

const OPERATION: &str = "nexus_query_engine.backend_metrics";

/// Maximum number of ordinal-addressable devices in one backend metrics store.
///
/// This is the same closed device-label cardinality frozen by the backend
/// descriptor registry. A device ordinal indexes its fixed slot directly, so
/// update paths need neither a map lookup nor an allocation.
pub const BACKEND_METRIC_DEVICE_CAPACITY: usize = 64;

/// Number of cumulative histogram buckets, including the final `+Inf`
/// bucket.
pub const BACKEND_ATTEMPT_DURATION_BUCKET_COUNT: usize = BACKEND_ATTEMPT_DURATION_BUCKETS_NS.len() + 1;

/// Closed disposition dimension of terminal backend-attempt metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(usize)]
pub enum BackendAttemptDisposition {
    Unplanned = 0,
    CpuFallback = 1,
    PartialNative = 2,
    Native = 3,
}

impl BackendAttemptDisposition {
    /// Number of closed disposition values in the backend metric store.
    pub const COUNT: usize = 4;

    /// Every closed disposition value in descriptor order.
    pub const ALL: [Self; Self::COUNT] = [Self::Unplanned, Self::CpuFallback, Self::PartialNative, Self::Native];

    /// Stable closed label value for the backend descriptor table.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unplanned => "unplanned",
            Self::CpuFallback => "cpu_fallback",
            Self::PartialNative => "partial_native",
            Self::Native => "native",
        }
    }

    const fn from_projection(disposition: AttemptDisposition) -> Self {
        match disposition {
            AttemptDisposition::Unplanned => Self::Unplanned,
            AttemptDisposition::CpuFallback => Self::CpuFallback,
            AttemptDisposition::PartialNative => Self::PartialNative,
            AttemptDisposition::Native => Self::Native,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Closed outcome dimension of terminal backend-attempt metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(usize)]
pub enum BackendAttemptOutcome {
    Succeeded = 0,
    Failed = 1,
    Cancelled = 2,
    Dropped = 3,
    Panicked = 4,
}

impl BackendAttemptOutcome {
    /// Number of closed terminal outcome values in the backend metric store.
    pub const COUNT: usize = 5;

    /// Every closed terminal outcome value in descriptor order.
    pub const ALL: [Self; Self::COUNT] =
        [Self::Succeeded, Self::Failed, Self::Cancelled, Self::Dropped, Self::Panicked];

    /// Stable closed label value for the backend descriptor table.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
            Self::Panicked => "panicked",
        }
    }

    const fn from_projection(outcome: &RuntimeTerminalOutcome) -> Self {
        match outcome {
            RuntimeTerminalOutcome::Succeeded => Self::Succeeded,
            RuntimeTerminalOutcome::Failed { .. } => Self::Failed,
            RuntimeTerminalOutcome::Cancelled => Self::Cancelled,
            RuntimeTerminalOutcome::Dropped => Self::Dropped,
            RuntimeTerminalOutcome::Panicked => Self::Panicked,
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Closed duration dimension for the attempt-duration histogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(usize)]
pub enum BackendAttemptDuration {
    Total = 0,
    Planning = 1,
    QueueWait = 2,
    PhysicalExecution = 3,
}

impl BackendAttemptDuration {
    /// Number of closed attempt-duration values in the backend metric store.
    pub const COUNT: usize = 4;

    /// Every closed attempt-duration value in descriptor order.
    pub const ALL: [Self; Self::COUNT] = [Self::Total, Self::Planning, Self::QueueWait, Self::PhysicalExecution];

    /// Stable closed label value for the backend descriptor table.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Total => "total",
            Self::Planning => "planning",
            Self::QueueWait => "queue_wait",
            Self::PhysicalExecution => "physical_execution",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Explicit duration values already derived from the sealed attempt report.
///
/// The store does not inspect report timestamps or rebuild durations. A
/// missing duration remains absent from its histogram rather than becoming a
/// fabricated zero-duration sample.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendAttemptDurations {
    pub total_ns: Option<u64>,
    pub planning_ns: Option<u64>,
    pub queue_wait_ns: Option<u64>,
    pub physical_execution_ns: Option<u64>,
}

impl BackendAttemptDurations {
    const fn value(self, duration: BackendAttemptDuration) -> Option<u64> {
        match duration {
            BackendAttemptDuration::Total => self.total_ns,
            BackendAttemptDuration::Planning => self.planning_ns,
            BackendAttemptDuration::QueueWait => self.queue_wait_ns,
            BackendAttemptDuration::PhysicalExecution => self.physical_execution_ns,
        }
    }
}

/// Closed source of an explicit backend telemetry refresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BackendTelemetrySource {
    Cuda,
    Nvml,
    DeviceMemoryFailureTracker,
}

/// Closed availability state recorded with every explicit telemetry refresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BackendTelemetryAvailability {
    Available,
    FeatureDisabled,
    Unavailable,
}

impl From<DeviceTelemetryStatus> for BackendTelemetryAvailability {
    fn from(status: DeviceTelemetryStatus) -> Self {
        match status {
            DeviceTelemetryStatus::Available => Self::Available,
            DeviceTelemetryStatus::FeatureDisabled => Self::FeatureDisabled,
            DeviceTelemetryStatus::Unavailable => Self::Unavailable,
        }
    }
}

/// Cumulative device-memory failure facts sampled by the backend control
/// plane.
///
/// `source_generation` identifies the installation generation that owns the
/// callback counters. The core never guesses that association: wiring must
/// provide the generation that owns the sampled report, so stale generations
/// cannot be double-counted into a newer backend generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendMemoryFailureTelemetry {
    pub device_ordinal: usize,
    pub source_generation: u64,
    pub allocation_failure_count: u64,
    pub allocation_failure_bytes: u64,
    pub largest_allocation_failure_bytes: u64,
}

impl BackendMemoryFailureTelemetry {
    /// Converts one callback report after the owner has resolved its
    /// installation generation.
    pub fn from_report(source_generation: u64, report: &DeviceMemoryFailureReport) -> Self {
        Self {
            device_ordinal: report.device_ordinal,
            source_generation,
            allocation_failure_count: report.allocation_failure_count,
            allocation_failure_bytes: report.total_failed_allocation_bytes,
            largest_allocation_failure_bytes: report.largest_failed_allocation_bytes,
        }
    }
}

/// Explicit telemetry refresh input for one configured device.
///
/// `age_ms` is supplied by the control plane and stored verbatim. Before the
/// first refresh, snapshots expose no refresh record rather than zero-filling
/// a sample time or age.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendTelemetryRefresh {
    pub device_ordinal: usize,
    pub source: BackendTelemetrySource,
    pub sampled_at_unix_ms: u64,
    pub age_ms: u64,
    pub availability: BackendTelemetryAvailability,
    pub memory_failures: Option<BackendMemoryFailureTelemetry>,
}

/// Closed admission health projected from the owning ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BackendAdmissionHealth {
    Open,
    Closed,
}

/// A fixed cumulative histogram snapshot. `bucket_counts` follows
/// [`BACKEND_ATTEMPT_DURATION_BUCKETS_NS`] and ends with its `+Inf` bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendHistogramSnapshot {
    pub bucket_counts: [u64; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT],
    pub count: u64,
    pub sum: u64,
}

impl BackendHistogramSnapshot {
    const EMPTY: Self = Self { bucket_counts: [0; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT], count: 0, sum: 0 };
}

/// Terminal counters and duration histograms for the complete backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendAttemptMetricsSnapshot {
    completed: [[u64; BackendAttemptOutcome::COUNT]; BackendAttemptDisposition::COUNT],
    durations: [[[BackendHistogramSnapshot; BackendAttemptDuration::COUNT]; BackendAttemptOutcome::COUNT];
        BackendAttemptDisposition::COUNT],
}

impl BackendAttemptMetricsSnapshot {
    /// Returns the exact terminal counter for one closed disposition/outcome
    /// pair.
    pub const fn completed(self, disposition: BackendAttemptDisposition, outcome: BackendAttemptOutcome) -> u64 {
        self.completed[disposition.index()][outcome.index()]
    }

    /// Returns the fixed cumulative histogram for one closed terminal series
    /// and duration kind.
    pub const fn duration(
        self,
        disposition: BackendAttemptDisposition,
        outcome: BackendAttemptOutcome,
        duration: BackendAttemptDuration,
    ) -> BackendHistogramSnapshot {
        self.durations[disposition.index()][outcome.index()][duration.index()]
    }
}

/// Last refresh metadata for one telemetry source. It contains no source text
/// or sampled device payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendTelemetryRefreshSnapshot {
    pub source: BackendTelemetrySource,
    pub sampled_at_unix_ms: u64,
    pub age_ms: u64,
    pub availability: BackendTelemetryAvailability,
}

/// Telemetry-class counters and refresh status for one configured device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendTelemetryMetricsSnapshot {
    pub last_refresh: Option<BackendTelemetryRefreshSnapshot>,
    pub allocation_failure_count: u64,
    pub allocation_failure_bytes: u64,
    pub largest_allocation_failure_bytes: u64,
}

impl BackendTelemetryMetricsSnapshot {
    const EMPTY: Self = Self {
        last_refresh: None,
        allocation_failure_count: 0,
        allocation_failure_bytes: 0,
        largest_allocation_failure_bytes: 0,
    };
}

/// Ledger-class backend metrics for one fixed device slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendAdmissionMetricsSnapshot {
    pub active_attempts: u64,
    pub live_grant_bytes: u64,
    pub live_reserved_bytes: u64,
    pub max_live_reserved_bytes: u64,
    /// Evictable cache residency from the device ledger's current snapshot.
    pub cache_evictable_resident_bytes: u64,
    /// Pinned cache residency from the device ledger's current snapshot.
    pub cache_pinned_in_use_bytes: u64,
    pub health: BackendAdmissionHealth,
}

impl BackendAdmissionMetricsSnapshot {
    const EMPTY: Self = Self {
        active_attempts: 0,
        live_grant_bytes: 0,
        live_reserved_bytes: 0,
        max_live_reserved_bytes: 0,
        cache_evictable_resident_bytes: 0,
        cache_pinned_in_use_bytes: 0,
        health: BackendAdmissionHealth::Open,
    };
}

/// Metrics for one configured device. The ordinal is the bounded device-label
/// value fixed during backend construction, never an attempt or query id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendDeviceMetricsSnapshot {
    pub device_ordinal: usize,
    pub ledger_sequence: u64,
    pub admission: BackendAdmissionMetricsSnapshot,
    pub telemetry: BackendTelemetryMetricsSnapshot,
}

/// Coherent fixed-cardinality snapshot of one backend metrics generation.
///
/// `generation` advances after each accepted store update. Consumers that
/// combine this snapshot with another read can compare the two generations;
/// unequal backend or store generations make a torn combination detectable.
/// All device slots are fixed. `None` means that ordinal is not part of this
/// backend generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendMetricsSnapshot {
    pub backend_generation: u64,
    pub generation: u64,
    #[serde(serialize_with = "serialize_backend_metric_devices")]
    pub devices: [Option<BackendDeviceMetricsSnapshot>; BACKEND_METRIC_DEVICE_CAPACITY],
    pub attempts: BackendAttemptMetricsSnapshot,
}

fn serialize_backend_metric_devices<S>(
    devices: &[Option<BackendDeviceMetricsSnapshot>; BACKEND_METRIC_DEVICE_CAPACITY],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    devices.as_slice().serialize(serializer)
}

impl BackendMetricsSnapshot {
    /// Returns whether two snapshots cannot be combined as one coherent read.
    pub const fn is_torn_against(&self, other: &Self) -> bool {
        self.backend_generation != other.backend_generation || self.generation != other.generation
    }

    /// Returns the configured device record for one bounded device ordinal.
    pub const fn device(&self, device_ordinal: usize) -> Option<BackendDeviceMetricsSnapshot> {
        if device_ordinal >= BACKEND_METRIC_DEVICE_CAPACITY { None } else { self.devices[device_ordinal] }
    }

    /// Projects one canonical descriptor from this coherent snapshot.
    ///
    /// This is the only control-plane boundary that derives backend metric
    /// samples, labels, duration units, or histogram bucket semantics. An
    /// exporter therefore only iterates the immutable descriptor registry and
    /// selects its ingress by [`crate::metrics::BackendMetricKind`].
    pub fn project_backend_metric(
        &self,
        descriptor: &'static crate::metrics::BackendMetricDescriptor,
    ) -> std::result::Result<Vec<BackendMetricSample>, BackendMetricProjectionError> {
        if descriptor != descriptor.metric.descriptor() {
            return Err(BackendMetricProjectionError::NonCanonicalDescriptor { metric_name: descriptor.name });
        }

        use crate::metrics::BackendMetric;

        let mut samples = Vec::new();
        match descriptor.metric {
            BackendMetric::admission_active_attempts => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.admission.active_attempts,
                    );
                }
            },
            BackendMetric::admission_live_grant_bytes => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.admission.live_grant_bytes,
                    );
                }
            },
            BackendMetric::admission_live_reserved_bytes => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.admission.live_reserved_bytes,
                    );
                }
            },
            BackendMetric::admission_max_live_reserved_bytes => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.admission.max_live_reserved_bytes,
                    );
                }
            },
            BackendMetric::attempts_completed => {
                for disposition in BackendAttemptDisposition::ALL {
                    for outcome in BackendAttemptOutcome::ALL {
                        push_scalar_sample(
                            &mut samples,
                            descriptor,
                            attempt_attributes(descriptor, disposition, outcome),
                            self.attempts.completed(disposition, outcome),
                        );
                    }
                }
            },
            BackendMetric::attempts_duration => {
                for disposition in BackendAttemptDisposition::ALL {
                    for outcome in BackendAttemptOutcome::ALL {
                        for duration in BackendAttemptDuration::ALL {
                            let histogram = project_backend_histogram(
                                descriptor,
                                self.attempts.duration(disposition, outcome, duration),
                            )?;
                            samples.push(BackendMetricSample {
                                attributes: duration_attributes(descriptor, disposition, outcome, duration),
                                value: BackendMetricSampleValue::Histogram(histogram),
                            });
                        }
                    }
                }
            },
            BackendMetric::memory_allocation_failures => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.telemetry.allocation_failure_count,
                    );
                }
            },
            BackendMetric::memory_allocation_failure_bytes => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.telemetry.allocation_failure_bytes,
                    );
                }
            },
            BackendMetric::cache_evictable_resident_bytes => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.admission.cache_evictable_resident_bytes,
                    );
                }
            },
            BackendMetric::cache_pinned_in_use_bytes => {
                for device in self.devices.iter().flatten() {
                    push_scalar_sample(
                        &mut samples,
                        descriptor,
                        device_attributes(descriptor, device.device_ordinal),
                        device.admission.cache_pinned_in_use_bytes,
                    );
                }
            },
        }
        Ok(samples)
    }
}

/// A typed, closed backend-metric attribute value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMetricAttributeValue {
    I64(i64),
    Str(&'static str),
}

/// One descriptor-ordered attribute of a projected backend metric sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendMetricAttribute {
    pub key: &'static str,
    pub value: BackendMetricAttributeValue,
}

/// An OTLP-compatible histogram projection. Bounds and `sum` use the
/// descriptor-declared unit, while bucket counts are disjoint (including the
/// final implicit `+Inf` bucket).
#[derive(Debug, Clone, PartialEq)]
pub struct BackendHistogramProjection {
    pub explicit_bounds: Vec<f64>,
    pub bucket_counts: [u64; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT],
    pub count: u64,
    pub sum: f64,
}

/// One value produced by a backend descriptor projection.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendMetricSampleValue {
    Scalar(u64),
    Histogram(BackendHistogramProjection),
}

/// One closed-label sample for a canonical backend descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendMetricSample {
    pub attributes: Vec<BackendMetricAttribute>,
    pub value: BackendMetricSampleValue,
}

/// Projection failures that prevent an exporter from emitting a valid
/// descriptor-owned backend metric.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendMetricProjectionError {
    NonCanonicalDescriptor { metric_name: &'static str },
    Duration(crate::metrics::BackendDurationProjectionError),
    CumulativeHistogramDecreases { metric_name: &'static str, bucket_index: usize },
    CumulativeHistogramCountMismatch { metric_name: &'static str, cumulative_count: u64, count: u64 },
    DisjointHistogramCountMismatch { metric_name: &'static str, bucket_count_sum: u64, count: u64 },
}

impl std::fmt::Display for BackendMetricProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonCanonicalDescriptor { metric_name } => {
                write!(formatter, "{metric_name} is not the canonical backend descriptor")
            },
            Self::Duration(error) => error.fmt(formatter),
            Self::CumulativeHistogramDecreases { metric_name, bucket_index } => {
                write!(formatter, "{metric_name} cumulative histogram decreases at bucket {bucket_index}")
            },
            Self::CumulativeHistogramCountMismatch { metric_name, cumulative_count, count } => write!(
                formatter,
                "{metric_name} cumulative +Inf bucket {cumulative_count} does not match count {count}"
            ),
            Self::DisjointHistogramCountMismatch { metric_name, bucket_count_sum, count } => {
                write!(formatter, "{metric_name} disjoint bucket sum {bucket_count_sum} does not match count {count}")
            },
        }
    }
}

impl std::error::Error for BackendMetricProjectionError {}

impl From<crate::metrics::BackendDurationProjectionError> for BackendMetricProjectionError {
    fn from(error: crate::metrics::BackendDurationProjectionError) -> Self {
        Self::Duration(error)
    }
}

fn push_scalar_sample(
    samples: &mut Vec<BackendMetricSample>,
    descriptor: &crate::metrics::BackendMetricDescriptor,
    attributes: Vec<BackendMetricAttribute>,
    value: u64,
) {
    debug_assert_eq!(descriptor.labels.len(), attributes.len());
    samples.push(BackendMetricSample { attributes, value: BackendMetricSampleValue::Scalar(value) });
}

fn device_attributes(
    descriptor: &crate::metrics::BackendMetricDescriptor,
    device_ordinal: usize,
) -> Vec<BackendMetricAttribute> {
    descriptor_attributes(descriptor, &[BackendMetricAttributeValue::I64(device_ordinal as i64)])
}

fn attempt_attributes(
    descriptor: &crate::metrics::BackendMetricDescriptor,
    disposition: BackendAttemptDisposition,
    outcome: BackendAttemptOutcome,
) -> Vec<BackendMetricAttribute> {
    descriptor_attributes(
        descriptor,
        &[BackendMetricAttributeValue::Str(disposition.as_str()), BackendMetricAttributeValue::Str(outcome.as_str())],
    )
}

fn duration_attributes(
    descriptor: &crate::metrics::BackendMetricDescriptor,
    disposition: BackendAttemptDisposition,
    outcome: BackendAttemptOutcome,
    duration: BackendAttemptDuration,
) -> Vec<BackendMetricAttribute> {
    descriptor_attributes(
        descriptor,
        &[
            BackendMetricAttributeValue::Str(disposition.as_str()),
            BackendMetricAttributeValue::Str(outcome.as_str()),
            BackendMetricAttributeValue::Str(duration.as_str()),
        ],
    )
}

fn descriptor_attributes(
    descriptor: &crate::metrics::BackendMetricDescriptor,
    values: &[BackendMetricAttributeValue],
) -> Vec<BackendMetricAttribute> {
    assert_eq!(
        descriptor.labels.len(),
        values.len(),
        "{} values must match the descriptor's closed label table",
        descriptor.name
    );
    descriptor
        .labels
        .iter()
        .zip(values)
        .map(|(label, value)| BackendMetricAttribute { key: label.name, value: *value })
        .collect()
}

/// Converts the fixed backend cumulative-by-upper-bound histogram snapshot
/// into the descriptor-declared, OTLP-compatible disjoint representation.
pub fn project_backend_histogram(
    descriptor: &crate::metrics::BackendMetricDescriptor,
    snapshot: BackendHistogramSnapshot,
) -> std::result::Result<BackendHistogramProjection, BackendMetricProjectionError> {
    let duration = crate::metrics::project_backend_duration_histogram(descriptor, snapshot.sum)?;
    let mut bucket_counts = [0; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT];
    let mut previous = 0;
    for (bucket_index, cumulative) in snapshot.bucket_counts.iter().copied().enumerate() {
        if cumulative < previous {
            return Err(BackendMetricProjectionError::CumulativeHistogramDecreases {
                metric_name: descriptor.name,
                bucket_index,
            });
        }
        bucket_counts[bucket_index] = cumulative - previous;
        previous = cumulative;
    }
    if previous != snapshot.count {
        return Err(BackendMetricProjectionError::CumulativeHistogramCountMismatch {
            metric_name: descriptor.name,
            cumulative_count: previous,
            count: snapshot.count,
        });
    }
    let bucket_count_sum =
        bucket_counts.iter().try_fold(0_u64, |total, bucket| total.checked_add(*bucket)).unwrap_or(u64::MAX);
    if bucket_count_sum != snapshot.count {
        return Err(BackendMetricProjectionError::DisjointHistogramCountMismatch {
            metric_name: descriptor.name,
            bucket_count_sum,
            count: snapshot.count,
        });
    }
    Ok(BackendHistogramProjection {
        explicit_bounds: duration.bucket_bounds,
        bucket_counts,
        count: snapshot.count,
        sum: duration.sum,
    })
}

#[derive(Debug, Clone, Copy)]
struct HistogramState {
    snapshot: BackendHistogramSnapshot,
}

impl HistogramState {
    const EMPTY: Self = Self { snapshot: BackendHistogramSnapshot::EMPTY };

    fn record(&mut self, value: u64) {
        for (index, upper_bound) in BACKEND_ATTEMPT_DURATION_BUCKETS_NS.iter().enumerate() {
            if value <= *upper_bound {
                self.snapshot.bucket_counts[index] = self.snapshot.bucket_counts[index].saturating_add(1);
            }
        }
        let infinity_bucket = BACKEND_ATTEMPT_DURATION_BUCKET_COUNT - 1;
        self.snapshot.bucket_counts[infinity_bucket] = self.snapshot.bucket_counts[infinity_bucket].saturating_add(1);
        self.snapshot.count = self.snapshot.count.saturating_add(1);
        self.snapshot.sum = self.snapshot.sum.saturating_add(value);
    }
}

#[derive(Debug, Clone, Copy)]
struct AttemptMetricsState {
    completed: [[u64; BackendAttemptOutcome::COUNT]; BackendAttemptDisposition::COUNT],
    durations: [[[HistogramState; BackendAttemptDuration::COUNT]; BackendAttemptOutcome::COUNT];
        BackendAttemptDisposition::COUNT],
}

impl AttemptMetricsState {
    const EMPTY: Self = Self {
        completed: [[0; BackendAttemptOutcome::COUNT]; BackendAttemptDisposition::COUNT],
        durations: [[[HistogramState::EMPTY; BackendAttemptDuration::COUNT]; BackendAttemptOutcome::COUNT];
            BackendAttemptDisposition::COUNT],
    };

    fn record(&mut self, projection: &SealedAttemptProjection, durations: BackendAttemptDurations) {
        let disposition = BackendAttemptDisposition::from_projection(projection.disposition);
        let outcome = BackendAttemptOutcome::from_projection(&projection.outcome);
        self.completed[disposition.index()][outcome.index()] =
            self.completed[disposition.index()][outcome.index()].saturating_add(1);
        for duration in [
            BackendAttemptDuration::Total,
            BackendAttemptDuration::Planning,
            BackendAttemptDuration::QueueWait,
            BackendAttemptDuration::PhysicalExecution,
        ] {
            if let Some(value) = durations.value(duration) {
                self.durations[disposition.index()][outcome.index()][duration.index()].record(value);
            }
        }
    }

    const fn snapshot(self) -> BackendAttemptMetricsSnapshot {
        let mut durations = [[[BackendHistogramSnapshot::EMPTY; BackendAttemptDuration::COUNT];
            BackendAttemptOutcome::COUNT]; BackendAttemptDisposition::COUNT];
        let mut disposition = 0;
        while disposition < BackendAttemptDisposition::COUNT {
            let mut outcome = 0;
            while outcome < BackendAttemptOutcome::COUNT {
                let mut duration = 0;
                while duration < BackendAttemptDuration::COUNT {
                    durations[disposition][outcome][duration] = self.durations[disposition][outcome][duration].snapshot;
                    duration += 1;
                }
                outcome += 1;
            }
            disposition += 1;
        }
        BackendAttemptMetricsSnapshot { completed: self.completed, durations }
    }
}

#[derive(Debug, Clone, Copy)]
struct DeviceMetricsState {
    configured: bool,
    device_ordinal: usize,
    ledger_sequence: u64,
    admission: BackendAdmissionMetricsSnapshot,
    telemetry: BackendTelemetryMetricsSnapshot,
    memory_failure_source_generation: u64,
    observed_failure_count: u64,
    observed_failure_bytes: u64,
}

impl DeviceMetricsState {
    const EMPTY: Self = Self {
        configured: false,
        device_ordinal: 0,
        ledger_sequence: 0,
        admission: BackendAdmissionMetricsSnapshot::EMPTY,
        telemetry: BackendTelemetryMetricsSnapshot::EMPTY,
        memory_failure_source_generation: 0,
        observed_failure_count: 0,
        observed_failure_bytes: 0,
    };

    const fn configured(device_ordinal: usize) -> Self {
        Self { configured: true, device_ordinal, ..Self::EMPTY }
    }

    fn record_ledger_transition(&mut self, transition: &LedgerTransition) {
        let snapshot = &transition.snapshot;
        self.ledger_sequence = snapshot.sequence;
        self.admission.active_attempts = u64::try_from(snapshot.active_attempts).unwrap_or(u64::MAX);
        self.admission.live_grant_bytes = snapshot.live_grants_bytes;
        self.admission.live_reserved_bytes = snapshot.live_reserved_bytes;
        self.admission.max_live_reserved_bytes =
            self.admission.max_live_reserved_bytes.max(snapshot.live_reserved_bytes);
        // Sealed reports, admission TSV, and backend descriptors all project
        // this exact ledger transition; resident caches do not own a second
        // gauge.
        self.admission.cache_evictable_resident_bytes = snapshot.cache_evictable_resident_bytes;
        self.admission.cache_pinned_in_use_bytes = snapshot.cache_pinned_in_use_bytes;
        self.admission.health = match snapshot.health {
            DeviceAdmissionHealth::Open => BackendAdmissionHealth::Open,
            DeviceAdmissionHealth::Closed { .. } => BackendAdmissionHealth::Closed,
        };
    }

    fn record_telemetry_refresh(&mut self, refresh: BackendTelemetryRefresh) -> BackendTelemetryRefreshSnapshot {
        let metadata = BackendTelemetryRefreshSnapshot {
            source: refresh.source,
            sampled_at_unix_ms: refresh.sampled_at_unix_ms,
            age_ms: refresh.age_ms,
            availability: refresh.availability,
        };
        self.telemetry.last_refresh = Some(metadata);
        let Some(memory_failures) = refresh.memory_failures else {
            return metadata;
        };
        if memory_failures.source_generation < self.memory_failure_source_generation {
            return metadata;
        }
        if memory_failures.source_generation > self.memory_failure_source_generation {
            self.memory_failure_source_generation = memory_failures.source_generation;
            self.observed_failure_count = 0;
            self.observed_failure_bytes = 0;
        }
        let count_delta = memory_failures.allocation_failure_count.saturating_sub(self.observed_failure_count);
        let byte_delta = memory_failures.allocation_failure_bytes.saturating_sub(self.observed_failure_bytes);
        self.telemetry.allocation_failure_count = self.telemetry.allocation_failure_count.saturating_add(count_delta);
        self.telemetry.allocation_failure_bytes = self.telemetry.allocation_failure_bytes.saturating_add(byte_delta);
        self.telemetry.largest_allocation_failure_bytes =
            self.telemetry.largest_allocation_failure_bytes.max(memory_failures.largest_allocation_failure_bytes);
        self.observed_failure_count = self.observed_failure_count.max(memory_failures.allocation_failure_count);
        self.observed_failure_bytes = self.observed_failure_bytes.max(memory_failures.allocation_failure_bytes);
        metadata
    }

    const fn snapshot(self) -> BackendDeviceMetricsSnapshot {
        BackendDeviceMetricsSnapshot {
            device_ordinal: self.device_ordinal,
            ledger_sequence: self.ledger_sequence,
            admission: self.admission,
            telemetry: self.telemetry,
        }
    }
}

#[derive(Debug)]
struct BackendMetricsState {
    generation: u64,
    devices: [DeviceMetricsState; BACKEND_METRIC_DEVICE_CAPACITY],
    attempts: AttemptMetricsState,
}

impl BackendMetricsState {
    fn advance_generation(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }
}

/// One backend-owned fixed-cardinality metrics store.
///
/// The store is created once for the backend's resource generation. All update
/// methods are crate-private so only the owning ledger observer, sealed-report
/// sink, and explicit telemetry control path can feed it.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct BackendMetricsStore {
    backend_generation: u64,
    state: Arc<Mutex<BackendMetricsState>>,
}

impl BackendMetricsStore {
    /// Creates one fixed-slot store for a nonzero backend resource generation.
    #[doc(hidden)]
    pub fn new(backend_generation: u64, device_ordinals: &[usize]) -> Result<Self> {
        if backend_generation == 0 {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                OPERATION,
                "backend metrics store requires a nonzero backend generation",
            )
            .with_fact("reason", "zero_backend_generation"));
        }
        let mut devices = [DeviceMetricsState::EMPTY; BACKEND_METRIC_DEVICE_CAPACITY];
        for &device_ordinal in device_ordinals {
            let Some(slot) = devices.get_mut(device_ordinal) else {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    OPERATION,
                    "backend device ordinal exceeds fixed metrics capacity",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("device_capacity", BACKEND_METRIC_DEVICE_CAPACITY)
                .with_fact("reason", "backend_metrics_device_ordinal_out_of_range"));
            };
            if slot.configured {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    OPERATION,
                    "backend metrics device ordinal is duplicated",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("reason", "duplicate_backend_metrics_device_ordinal"));
            }
            *slot = DeviceMetricsState::configured(device_ordinal);
        }
        Ok(Self {
            backend_generation,
            state: Arc::new(Mutex::new(BackendMetricsState {
                generation: 0,
                devices,
                attempts: AttemptMetricsState::EMPTY,
            })),
        })
    }

    /// Records one exact in-lock ledger transition. Unknown ordinals are not
    /// part of this backend generation and are ignored defensively.
    #[doc(hidden)]
    pub fn record_ledger_transition(&self, transition: &LedgerTransition) {
        let device_ordinal = transition.snapshot.device_ordinal;
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(device) = state.devices.get_mut(device_ordinal) else {
            return;
        };
        if !device.configured {
            return;
        }
        device.record_ledger_transition(transition);
        state.advance_generation();
    }

    /// Records one report-ready sealed projection and its explicit durations.
    ///
    /// The caller owns exactly-once delivery through the attempt completion
    /// sink; this method intentionally does not retain an attempt identity.
    #[doc(hidden)]
    pub fn record_sealed_attempt(&self, projection: &SealedAttemptProjection, durations: BackendAttemptDurations) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.attempts.record(projection, durations);
        state.advance_generation();
    }

    /// Records one explicit, control-plane telemetry refresh.
    #[doc(hidden)]
    pub fn refresh_telemetry(&self, refresh: BackendTelemetryRefresh) -> Result<BackendTelemetryRefreshSnapshot> {
        if refresh.memory_failures.is_some_and(|memory_failures| memory_failures.source_generation == 0) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "memory-failure telemetry requires a nonzero source generation",
            )
            .with_fact("device_ordinal", refresh.device_ordinal)
            .with_fact("reason", "zero_memory_failure_source_generation"));
        }
        if let Some(memory_failures) = refresh.memory_failures
            && memory_failures.device_ordinal != refresh.device_ordinal
        {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "memory-failure telemetry device does not match its refresh device",
            )
            .with_fact("refresh_device_ordinal", refresh.device_ordinal)
            .with_fact("memory_failure_device_ordinal", memory_failures.device_ordinal)
            .with_fact("reason", "memory_failure_telemetry_device_mismatch"));
        }
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(device) = state.devices.get_mut(refresh.device_ordinal) else {
            return Err(unknown_telemetry_device_error(refresh.device_ordinal));
        };
        if !device.configured {
            return Err(unknown_telemetry_device_error(refresh.device_ordinal));
        }
        let metadata = device.record_telemetry_refresh(refresh);
        state.advance_generation();
        Ok(metadata)
    }

    /// Returns one coherent pull snapshot without exposing the mutable store.
    #[doc(hidden)]
    pub fn snapshot(&self) -> BackendMetricsSnapshot {
        let state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        BackendMetricsSnapshot {
            backend_generation: self.backend_generation,
            generation: state.generation,
            devices: state.devices.map(|device| device.configured.then(|| device.snapshot())),
            attempts: state.attempts.snapshot(),
        }
    }

    /// Returns whether a previously pulled snapshot still names the current
    /// store generation. This is useful to detect a torn multi-read without
    /// exposing a second mutable observation surface.
    #[cfg(test)]
    #[doc(hidden)]
    pub fn is_snapshot_current(&self, snapshot: &BackendMetricsSnapshot) -> bool {
        if snapshot.backend_generation != self.backend_generation {
            return false;
        }
        let state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        snapshot.generation == state.generation
    }
}

fn unknown_telemetry_device_error(device_ordinal: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        OPERATION,
        "telemetry refresh names a device outside this backend generation",
    )
    .with_fact("device_ordinal", device_ordinal)
    .with_fact("reason", "unknown_backend_metrics_telemetry_device")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        metrics::{BackendMetric, backend_metric_descriptors},
        observability::{
            AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptId, AttemptRetryAdvice,
            SEALED_ATTEMPT_PROJECTION_SCHEMA_V1,
        },
        runtime::gpu::LedgerSnapshot,
    };

    fn ledger_transition(
        sequence: u64,
        active_attempts: usize,
        live_grants_bytes: u64,
        live_reserved_bytes: u64,
        cache_resident_bytes: u64,
    ) -> LedgerTransition {
        LedgerTransition {
            kind: LedgerTransitionKind::GrantAdmitted,
            snapshot: LedgerSnapshot {
                device_ordinal: 3,
                sequence,
                managed_capacity_bytes: 10_000,
                configured_floor_bytes: 10,
                max_active_attempts: 8,
                active_attempts,
                live_grants_bytes,
                live_reserved_bytes,
                root_streams_available: 4,
                auxiliary_streams_available: 4,
                root_worker_slots_available: 4,
                auxiliary_worker_slots_available: 4,
                worker_depth: active_attempts,
                protected_slot: false,
                protected_reserve_bytes: 0,
                cache_evictable_resident_bytes: cache_resident_bytes,
                cache_pinned_in_use_bytes: cache_resident_bytes,
                health: DeviceAdmissionHealth::Open,
            },
        }
    }

    fn sealed_projection(disposition: AttemptDisposition, outcome: RuntimeTerminalOutcome) -> SealedAttemptProjection {
        SealedAttemptProjection {
            schema: SEALED_ATTEMPT_PROJECTION_SCHEMA_V1,
            attempt_id: AttemptId::mint(),
            disposition,
            outcome,
            cleanup_disposition: AttemptCleanupDisposition::NotRequired,
            cleanup_failure_count: 0,
            cleanup_failure_count_complete: true,
            capacity_disposition: AttemptCapacityDisposition::NotAdmitted,
            health_transition: None,
            retry_advice: AttemptRetryAdvice::no_retry_needed(),
            observation_truncated: false,
            observation_degraded: false,
            native_fault_evidence_present: false,
            resource_evidence_present: false,
        }
    }

    #[test]
    fn store_uses_fixed_ordinal_slots_and_rejects_invalid_configuration() {
        let store = BackendMetricsStore::new(41, &[3, 17]).unwrap();
        let snapshot = store.snapshot();

        assert_eq!(snapshot.backend_generation, 41);
        assert_eq!(snapshot.generation, 0);
        assert_eq!(snapshot.devices.len(), BACKEND_METRIC_DEVICE_CAPACITY);
        assert_eq!(snapshot.device(3).unwrap().device_ordinal, 3);
        assert_eq!(snapshot.device(17).unwrap().device_ordinal, 17);
        assert_eq!(snapshot.device(2), None);
        assert!(!std::mem::needs_drop::<BackendMetricsSnapshot>());
        assert!(!std::mem::needs_drop::<BackendDeviceMetricsSnapshot>());
        assert!(!std::mem::needs_drop::<BackendAttemptMetricsSnapshot>());
        assert!(BackendMetricsStore::new(41, &[BACKEND_METRIC_DEVICE_CAPACITY]).is_err());
        assert!(BackendMetricsStore::new(41, &[3, 3]).is_err());
        assert!(BackendMetricsStore::new(0, &[3]).is_err());
    }

    #[test]
    fn ledger_transitions_update_exact_gauges_and_project_cache_residency() {
        let store = BackendMetricsStore::new(42, &[3]).unwrap();
        store.record_ledger_transition(&ledger_transition(9, 2, 500, 400, 333));
        store.record_ledger_transition(&ledger_transition(10, 1, 200, 100, 999));
        store.record_ledger_transition(&ledger_transition(11, 1, 200, 450, 777));

        let device = store.snapshot().device(3).unwrap();
        assert_eq!(device.ledger_sequence, 11);
        assert_eq!(device.admission.active_attempts, 1);
        assert_eq!(device.admission.live_grant_bytes, 200);
        assert_eq!(device.admission.live_reserved_bytes, 450);
        assert_eq!(device.admission.max_live_reserved_bytes, 450);
        assert_eq!(device.admission.cache_evictable_resident_bytes, 777);
        assert_eq!(device.admission.cache_pinned_in_use_bytes, 777);
    }

    #[test]
    fn sealed_projection_updates_one_counter_and_the_exact_fixed_histogram_buckets() {
        let store = BackendMetricsStore::new(43, &[3]).unwrap();
        let projection = sealed_projection(AttemptDisposition::Native, RuntimeTerminalOutcome::Succeeded);
        store.record_sealed_attempt(
            &projection,
            BackendAttemptDurations {
                total_ns: Some(1_000),
                planning_ns: Some(1_001),
                queue_wait_ns: None,
                physical_execution_ns: Some(2_000_000_000),
            },
        );

        let attempts = store.snapshot().attempts;
        assert_eq!(attempts.completed(BackendAttemptDisposition::Native, BackendAttemptOutcome::Succeeded), 1);
        let total = attempts.duration(
            BackendAttemptDisposition::Native,
            BackendAttemptOutcome::Succeeded,
            BackendAttemptDuration::Total,
        );
        assert_eq!(total.count, 1);
        assert_eq!(total.sum, 1_000);
        assert_eq!(total.bucket_counts, [1; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT]);
        let planning = attempts.duration(
            BackendAttemptDisposition::Native,
            BackendAttemptOutcome::Succeeded,
            BackendAttemptDuration::Planning,
        );
        assert_eq!(planning.bucket_counts, [0, 1, 1, 1, 1, 1, 1, 1]);
        let queue = attempts.duration(
            BackendAttemptDisposition::Native,
            BackendAttemptOutcome::Succeeded,
            BackendAttemptDuration::QueueWait,
        );
        assert_eq!(queue, BackendHistogramSnapshot::EMPTY);
        let physical = attempts.duration(
            BackendAttemptDisposition::Native,
            BackendAttemptOutcome::Succeeded,
            BackendAttemptDuration::PhysicalExecution,
        );
        assert_eq!(physical.bucket_counts, [0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn snapshot_generation_is_monotonic_and_detects_torn_multi_reads() {
        let store = BackendMetricsStore::new(44, &[3]).unwrap();
        let before = store.snapshot();
        assert!(store.is_snapshot_current(&before));

        store.record_ledger_transition(&ledger_transition(1, 1, 100, 50, 0));
        let after_ledger = store.snapshot();
        assert!(after_ledger.generation > before.generation);
        assert!(before.is_torn_against(&after_ledger));
        assert!(!store.is_snapshot_current(&before));
        assert!(store.is_snapshot_current(&after_ledger));

        store
            .refresh_telemetry(BackendTelemetryRefresh {
                device_ordinal: 3,
                source: BackendTelemetrySource::Cuda,
                sampled_at_unix_ms: 99,
                age_ms: 7,
                availability: BackendTelemetryAvailability::Available,
                memory_failures: None,
            })
            .unwrap();
        let after_telemetry = store.snapshot();
        assert!(after_telemetry.generation > after_ledger.generation);
        assert!(after_ledger.is_torn_against(&after_telemetry));
        assert!(store.is_snapshot_current(&after_telemetry));
    }

    #[test]
    fn telemetry_refresh_uses_explicit_age_and_exact_cumulative_failure_deltas() {
        let store = BackendMetricsStore::new(45, &[3]).unwrap();
        store.record_ledger_transition(&ledger_transition(4, 1, 100, 50, 0));
        let admission_before_refresh = store.snapshot().device(3).unwrap().admission;
        assert_eq!(store.snapshot().device(3).unwrap().telemetry.last_refresh, None);

        let first = BackendTelemetryRefresh {
            device_ordinal: 3,
            source: BackendTelemetrySource::DeviceMemoryFailureTracker,
            sampled_at_unix_ms: 123,
            age_ms: 19,
            availability: BackendTelemetryAvailability::Available,
            memory_failures: Some(BackendMemoryFailureTelemetry {
                device_ordinal: 3,
                source_generation: 8,
                allocation_failure_count: 3,
                allocation_failure_bytes: 70,
                largest_allocation_failure_bytes: 40,
            }),
        };
        let returned = store.refresh_telemetry(first).unwrap();
        store.refresh_telemetry(first).unwrap();
        let refreshed = store.snapshot().device(3).unwrap().telemetry;
        assert_eq!(
            refreshed.last_refresh,
            Some(BackendTelemetryRefreshSnapshot {
                source: BackendTelemetrySource::DeviceMemoryFailureTracker,
                sampled_at_unix_ms: 123,
                age_ms: 19,
                availability: BackendTelemetryAvailability::Available,
            })
        );
        assert_eq!(refreshed.allocation_failure_count, 3);
        assert_eq!(refreshed.allocation_failure_bytes, 70);
        assert_eq!(refreshed.largest_allocation_failure_bytes, 40);
        assert_eq!(store.snapshot().device(3).unwrap().admission, admission_before_refresh);
        assert_eq!(returned, refreshed.last_refresh.unwrap());

        store
            .refresh_telemetry(BackendTelemetryRefresh {
                memory_failures: Some(BackendMemoryFailureTelemetry {
                    device_ordinal: 3,
                    source_generation: 8,
                    allocation_failure_count: 4,
                    allocation_failure_bytes: 91,
                    largest_allocation_failure_bytes: 44,
                }),
                ..first
            })
            .unwrap();
        let incremented = store.snapshot().device(3).unwrap().telemetry;
        assert_eq!(incremented.allocation_failure_count, 4);
        assert_eq!(incremented.allocation_failure_bytes, 91);
        assert_eq!(incremented.largest_allocation_failure_bytes, 44);

        store
            .refresh_telemetry(BackendTelemetryRefresh {
                memory_failures: Some(BackendMemoryFailureTelemetry {
                    device_ordinal: 3,
                    source_generation: 8,
                    allocation_failure_count: 3,
                    allocation_failure_bytes: 70,
                    largest_allocation_failure_bytes: 40,
                }),
                ..first
            })
            .unwrap();
        let regressed = store.snapshot().device(3).unwrap().telemetry;
        assert_eq!(regressed.allocation_failure_count, 4);
        assert_eq!(regressed.allocation_failure_bytes, 91);

        store
            .refresh_telemetry(BackendTelemetryRefresh {
                memory_failures: Some(BackendMemoryFailureTelemetry {
                    device_ordinal: 3,
                    source_generation: 7,
                    allocation_failure_count: 100,
                    allocation_failure_bytes: 1_000,
                    largest_allocation_failure_bytes: 1_000,
                }),
                ..first
            })
            .unwrap();
        let stale = store.snapshot().device(3).unwrap().telemetry;
        assert_eq!(stale.allocation_failure_count, 4);
        assert_eq!(stale.allocation_failure_bytes, 91);
        assert_eq!(stale.largest_allocation_failure_bytes, 44);
    }

    #[test]
    fn store_source_never_names_the_query_scoped_metric_container() {
        let forbidden_type = concat!("Query", "Metrics");
        assert!(
            !include_str!("backend_metrics.rs").contains(forbidden_type),
            "the fixed backend store must not ingest query-scoped metric rows"
        );
    }

    #[test]
    fn histogram_projection_converts_units_and_cumulative_buckets_at_the_owning_boundary() {
        let descriptor = BackendMetric::attempts_duration.descriptor();
        let projected = project_backend_histogram(
            descriptor,
            BackendHistogramSnapshot { bucket_counts: [1, 3, 3, 4, 4, 4, 5, 5], count: 5, sum: 1_000_000_123 },
        )
        .expect("canonical duration histogram projects");

        assert_eq!(projected.explicit_bounds, vec![0.000_001, 0.000_01, 0.000_1, 0.001, 0.01, 0.1, 1.0]);
        assert_eq!(projected.sum, 1.000_000_123);
        assert_eq!(projected.bucket_counts, [1, 2, 0, 1, 0, 0, 1, 0]);
        assert_eq!(projected.bucket_counts.iter().sum::<u64>(), projected.count);
        assert_eq!(projected.explicit_bounds.len(), projected.bucket_counts.len() - 1);
    }

    #[test]
    fn histogram_projection_rejects_invalid_cumulative_snapshots() {
        let descriptor = BackendMetric::attempts_duration.descriptor();
        let decreasing = project_backend_histogram(
            descriptor,
            BackendHistogramSnapshot { bucket_counts: [1, 0, 0, 0, 0, 0, 0, 0], count: 0, sum: 0 },
        );
        assert_eq!(
            decreasing,
            Err(BackendMetricProjectionError::CumulativeHistogramDecreases {
                metric_name: descriptor.name,
                bucket_index: 1,
            })
        );

        let mismatched_count = project_backend_histogram(
            descriptor,
            BackendHistogramSnapshot { bucket_counts: [0; BACKEND_ATTEMPT_DURATION_BUCKET_COUNT], count: 1, sum: 0 },
        );
        assert_eq!(
            mismatched_count,
            Err(BackendMetricProjectionError::CumulativeHistogramCountMismatch {
                metric_name: descriptor.name,
                cumulative_count: 0,
                count: 1,
            })
        );
    }

    #[test]
    fn initial_snapshot_projection_covers_the_descriptor_table_and_keeps_cache_metrics_zero() {
        let snapshot = BackendMetricsStore::new(49, &[3]).unwrap().snapshot();
        for descriptor in backend_metric_descriptors() {
            let samples = snapshot.project_backend_metric(descriptor).expect("canonical descriptor projects");
            assert!(!samples.is_empty(), "{} has a dashboard projection", descriptor.name);
            for sample in &samples {
                assert_eq!(sample.attributes.len(), descriptor.labels.len());
                for (attribute, label) in sample.attributes.iter().zip(descriptor.labels) {
                    assert_eq!(attribute.key, label.name);
                }
            }
            if matches!(
                descriptor.metric,
                BackendMetric::cache_evictable_resident_bytes | BackendMetric::cache_pinned_in_use_bytes
            ) {
                assert!(samples.iter().all(|sample| sample.value == BackendMetricSampleValue::Scalar(0)));
            }
        }
    }
}
