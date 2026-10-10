//! Backend-owned metric descriptors.
//!
//! This catalog is deliberately separate from [`super::QueryMetrics`].  The
//! latter contains query-local rows and is therefore not a valid input to a
//! fixed-cardinality backend store.

/// The storage semantics of a backend metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMetricKind {
    /// Monotonic cumulative count over the backend generation.
    Counter,
    /// Current value projected from an admission ledger transition.
    Gauge,
    /// Largest recorded value over the backend generation.
    Max,
    /// Cumulative counts and sum for a bounded set of duration buckets.
    Histogram,
}

/// UCUM unit carried by a backend descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMetricUnit {
    /// Dimensionless count, exported with UCUM unit `1`.
    One,
    /// Byte count, exported with UCUM unit `By`.
    Bytes,
    /// Duration in seconds, exported with UCUM unit `s`.
    Seconds,
}

impl BackendMetricUnit {
    /// Returns the descriptor's stable UCUM unit label.
    pub const fn as_ucum(self) -> &'static str {
        match self {
            Self::One => "1",
            Self::Bytes => "By",
            Self::Seconds => "s",
        }
    }
}

/// The lifecycle transition that owns a metric update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMetricUpdatePoint {
    /// An authoritative device ledger mutation updates the metric.
    LedgerTransition,
    /// Publication of a sealed attempt report updates the metric.
    ReportReady,
}

/// A closed label domain. `cardinality` is a contract bound on the number of
/// label values, set in the descriptor and not computed from observed samples.
/// Stores must reject values outside the corresponding closed domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendMetricLabel {
    pub(crate) name: &'static str,
    pub(crate) cardinality: usize,
}

/// One backend metric's immutable registration metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendMetricDescriptor {
    /// Stable identity used by the owning snapshot projection. Exporters
    /// iterate descriptors and never infer a metric from its display name.
    pub metric: BackendMetric,
    /// Stable metric registration name used by exporters.
    pub name: &'static str,
    /// Export unit, including the conversion target for duration samples.
    pub unit: BackendMetricUnit,
    /// Storage and exporter aggregation semantics.
    pub kind: BackendMetricKind,
    /// Closed label dimensions in the order used by projected sample attributes.
    pub labels: &'static [BackendMetricLabel],
    /// Internal histogram boundary values. Duration bounds remain exact
    /// nanoseconds until [`project_backend_duration_histogram`] converts them
    /// to this descriptor's declared export unit.
    pub buckets: &'static [u64],
    /// Runtime lifecycle transition that records this metric.
    pub update_point: BackendMetricUpdatePoint,
    /// Maximum accepted series count across the closed label domains.
    pub max_series: usize,
}

/// Duration histogram values converted to the descriptor's declared unit.
///
/// Histogram count and cumulative bucket counts are dimensionless, so callers
/// must copy them from the snapshot unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendDurationHistogramProjection {
    /// Explicit finite histogram bounds in the descriptor's export unit.
    pub bucket_bounds: Vec<f64>,
    /// Total observed duration in the descriptor's export unit.
    pub sum: f64,
}

/// A duration descriptor cannot project the backend's nanosecond storage into
/// its declared unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendDurationProjectionError {
    /// Metric whose descriptor requests an unsupported duration unit.
    pub metric_name: &'static str,
    /// Declared unit that cannot represent the nanosecond duration projection.
    pub unit: BackendMetricUnit,
}

impl std::fmt::Display for BackendDurationProjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} cannot project a duration with UCUM unit {}", self.metric_name, self.unit.as_ucum())
    }
}

impl std::error::Error for BackendDurationProjectionError {}

/// Projects the duration descriptor's internal nanosecond bounds and sum into
/// its declared export unit.
///
/// This is the sole duration conversion boundary. It intentionally omits the
/// dimensionless `count` and `bucket_counts`, which consumers must preserve
/// from the fixed backend histogram snapshot.
pub fn project_backend_duration_histogram(
    descriptor: &BackendMetricDescriptor,
    sum_ns: u64,
) -> Result<BackendDurationHistogramProjection, BackendDurationProjectionError> {
    match descriptor.unit {
        BackendMetricUnit::Seconds => Ok(BackendDurationHistogramProjection {
            bucket_bounds: descriptor.buckets.iter().map(|bound_ns| *bound_ns as f64 / 1_000_000_000.0).collect(),
            sum: sum_ns as f64 / 1_000_000_000.0,
        }),
        unit => Err(BackendDurationProjectionError { metric_name: descriptor.name, unit }),
    }
}

const DEVICE: BackendMetricLabel = BackendMetricLabel { name: "device.ordinal", cardinality: 64 };
const OUTCOME: BackendMetricLabel = BackendMetricLabel { name: "attempt.outcome", cardinality: 8 };
const DISPOSITION: BackendMetricLabel = BackendMetricLabel { name: "attempt.disposition", cardinality: 4 };
const DURATION: BackendMetricLabel = BackendMetricLabel { name: "attempt.duration", cardinality: 4 };

const EMPTY_BUCKETS: &[u64] = &[];
const DEVICE_LABELS: &[BackendMetricLabel] = &[DEVICE];
const ATTEMPT_LABELS: &[BackendMetricLabel] = &[DISPOSITION, OUTCOME];
const ATTEMPT_DURATION_LABELS: &[BackendMetricLabel] = &[DISPOSITION, OUTCOME, DURATION];
/// Fixed histogram boundaries shared by the descriptor and backend store.
pub const BACKEND_ATTEMPT_DURATION_BUCKETS_NS: [u64; 7] =
    [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000];
const DURATION_BUCKETS_NS: &[u64] = &BACKEND_ATTEMPT_DURATION_BUCKETS_NS;

/// Single ordered source for the backend descriptor registry.
macro_rules! for_each_backend_metric {
    ($macro:ident) => {
        $macro! {
            "Current admitted attempts whose grant is active or being cleaned up."; admission_active_attempts => "algeon.gpu.admission.active_attempts": BackendMetricUnit::One, BackendMetricKind::Gauge, DEVICE_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::LedgerTransition, 64;
            "Current bytes committed to immutable active or cleaning query grants."; admission_live_grant_bytes => "algeon.gpu.admission.live_grant": BackendMetricUnit::Bytes, BackendMetricKind::Gauge, DEVICE_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::LedgerTransition, 64;
            "Current bytes retained by live resident owners."; admission_resident_owner_bytes => "algeon.gpu.admission.resident_owner": BackendMetricUnit::Bytes, BackendMetricKind::Gauge, DEVICE_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::LedgerTransition, 64;
            "Maximum resident-owner bytes over the backend generation."; admission_max_resident_owner_bytes => "algeon.gpu.admission.max_resident_owner": BackendMetricUnit::Bytes, BackendMetricKind::Max, DEVICE_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::LedgerTransition, 64;
            "Terminal attempts counted when their sealed report becomes ready."; attempts_completed => "algeon.gpu.attempts.completed": BackendMetricUnit::One, BackendMetricKind::Counter, ATTEMPT_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::ReportReady, 32;
            "Sealed attempt durations grouped by coverage, outcome and duration kind."; attempts_duration => "algeon.gpu.attempts.duration": BackendMetricUnit::Seconds, BackendMetricKind::Histogram, ATTEMPT_DURATION_LABELS, DURATION_BUCKETS_NS, BackendMetricUpdatePoint::ReportReady, 128;
            "Current cached resident bytes eligible for eviction."; cache_evictable_resident_bytes => "algeon.gpu.cache.evictable.resident": BackendMetricUnit::Bytes, BackendMetricKind::Gauge, DEVICE_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::LedgerTransition, 64;
            "Current cached resident bytes pinned by active use."; cache_pinned_in_use_bytes => "algeon.gpu.cache.pinned.in_use": BackendMetricUnit::Bytes, BackendMetricKind::Gauge, DEVICE_LABELS, EMPTY_BUCKETS, BackendMetricUpdatePoint::LedgerTransition, 64;
        }
    };
}

macro_rules! define_backend_metric_ids {
    ($( $doc:literal; $field:ident => $name:literal: $unit:expr, $kind:expr, $labels:ident, $buckets:ident, $point:expr, $max:literal; )*) => {
        /// Stable identities for the bounded backend descriptor registry.
        #[expect(
            non_camel_case_types,
            reason = "metric identifiers are generated in snake_case matching descriptor fields"
        )]
        #[repr(usize)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BackendMetric {
            $( #[doc = $doc] $field, )*
        }

        /// Canonical backend descriptors in stable registration order.
        pub const BACKEND_METRIC_DESCRIPTORS: &[BackendMetricDescriptor] = &[
            $( BackendMetricDescriptor { metric: BackendMetric::$field, name: $name, unit: $unit, kind: $kind, labels: $labels, buckets: $buckets, update_point: $point, max_series: $max }, )*
        ];
    };
}

for_each_backend_metric!(define_backend_metric_ids);

/// Returns the descriptor registry in stable contract order.
pub const fn backend_metric_descriptors() -> &'static [BackendMetricDescriptor] {
    BACKEND_METRIC_DESCRIPTORS
}

impl BackendMetric {
    /// Returns this metric's one canonical descriptor-table entry.
    pub(crate) const fn descriptor(self) -> &'static BackendMetricDescriptor {
        &BACKEND_METRIC_DESCRIPTORS[self as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &[(&str, BackendMetricUnit, BackendMetricKind, BackendMetricUpdatePoint, usize)] = &[
        (
            "algeon.gpu.admission.active_attempts",
            BackendMetricUnit::One,
            BackendMetricKind::Gauge,
            BackendMetricUpdatePoint::LedgerTransition,
            64,
        ),
        (
            "algeon.gpu.admission.live_grant",
            BackendMetricUnit::Bytes,
            BackendMetricKind::Gauge,
            BackendMetricUpdatePoint::LedgerTransition,
            64,
        ),
        (
            "algeon.gpu.admission.resident_owner",
            BackendMetricUnit::Bytes,
            BackendMetricKind::Gauge,
            BackendMetricUpdatePoint::LedgerTransition,
            64,
        ),
        (
            "algeon.gpu.admission.max_resident_owner",
            BackendMetricUnit::Bytes,
            BackendMetricKind::Max,
            BackendMetricUpdatePoint::LedgerTransition,
            64,
        ),
        (
            "algeon.gpu.attempts.completed",
            BackendMetricUnit::One,
            BackendMetricKind::Counter,
            BackendMetricUpdatePoint::ReportReady,
            32,
        ),
        (
            "algeon.gpu.attempts.duration",
            BackendMetricUnit::Seconds,
            BackendMetricKind::Histogram,
            BackendMetricUpdatePoint::ReportReady,
            128,
        ),
        (
            "algeon.gpu.cache.evictable.resident",
            BackendMetricUnit::Bytes,
            BackendMetricKind::Gauge,
            BackendMetricUpdatePoint::LedgerTransition,
            64,
        ),
        (
            "algeon.gpu.cache.pinned.in_use",
            BackendMetricUnit::Bytes,
            BackendMetricKind::Gauge,
            BackendMetricUpdatePoint::LedgerTransition,
            64,
        ),
    ];

    #[test]
    fn descriptor_table_golden_is_stable() {
        assert_eq!(BACKEND_METRIC_DESCRIPTORS.len(), GOLDEN.len());
        for (actual, expected) in BACKEND_METRIC_DESCRIPTORS.iter().zip(GOLDEN) {
            assert_eq!((actual.name, actual.unit, actual.kind, actual.update_point, actual.max_series), *expected);
            assert_eq!(actual.metric.descriptor(), actual, "{} must retain its descriptor identity", actual.name);
        }
    }

    #[test]
    fn label_products_fit_declared_series_bounds() {
        for descriptor in BACKEND_METRIC_DESCRIPTORS {
            let product = descriptor.labels.iter().map(|label| label.cardinality).product::<usize>();
            assert!(product <= descriptor.max_series, "{} exceeds max_series", descriptor.name);
            if descriptor.kind == BackendMetricKind::Histogram {
                assert!(!descriptor.buckets.is_empty());
            } else {
                assert!(descriptor.buckets.is_empty());
            }
        }
    }

    #[test]
    fn labels_are_closed_and_measurement_names_are_not_dimensions() {
        let forbidden = ["query_id", "attempt_id", "sql", "user", "path", "uri", "message", "node_id"];
        for descriptor in BACKEND_METRIC_DESCRIPTORS {
            for label in descriptor.labels {
                assert!(!forbidden.contains(&label.name));
                assert!(!label.name.ends_with("_bytes"));
                assert!(!label.name.ends_with("_rows"));
                assert!(!label.name.ends_with("_ns"));
            }
        }
    }

    #[test]
    fn names_and_units_follow_backend_conventions() {
        for descriptor in BACKEND_METRIC_DESCRIPTORS {
            assert!(descriptor.name.starts_with("algeon."));
            assert!(
                descriptor.name.chars().all(|c| c == '.' || c == '_' || c.is_ascii_lowercase() || c.is_ascii_digit())
            );
            assert!(matches!(descriptor.unit.as_ucum(), "1" | "By" | "s"));
        }
    }
}
