//! Query-scoped bounded synchronous runtime observation recorder.

use std::{collections::BTreeMap, fmt, sync::Arc};

use super::{
    CleanupFailureSummary, ObservationSequence, PlanExecutionId, QueryAttemptReport, QueryBoundaryKind,
    RuntimeObservationClassLoss, RuntimeObservationClock, RuntimeObservationFooter, RuntimeObservationLimits,
    RuntimeObservationLossReason, RuntimeObservationManifest, RuntimeObservationPayload, RuntimeObservationReasonLoss,
    RuntimeObservationRecordClass, RuntimeQueryId, RuntimeSourceLocationId, RuntimeSpanId, RuntimeTerminalOutcome,
    RuntimeTimelineEvent, RuntimeTopologyRecord, RuntimeTraceId, SystemRuntimeObservationClock,
    metrics_summary::{
        MetricsCaptureMode, MetricsCompleteness, QueryMetricsCheckpoint, QueryMetricsSummary, SourceReadFact,
    },
};
use crate::{
    metrics::{QueryMetrics, SourceReadRuntimeMetrics},
    observability::{ExecutionProgress, RUNTIME_OBSERVATION_JSONL_SCHEMA, RuntimeClockDomain},
};

mod digest;
mod handle;
mod owner;
mod session;
mod state;

#[cfg(any(test, feature = "test-hooks"))]
use digest::TestSourceLocationDigestKeyProvider;
use digest::{
    OsSourceLocationDigestKeyProvider, SourceLocationDigestKeyProvider, SourceLocationDigestKeyState,
    SourceLocationDigestKeyUnavailable,
};
#[cfg(test)]
pub(crate) use owner::RuntimeObservationDeliveryGate;
use owner::RuntimeObservationDeliveryGuard;
pub use owner::RuntimeObservationOwner;
#[cfg(test)]
use owner::RuntimeObservationOwnerLifecycleState;
pub(crate) use owner::in_runtime_observation_delivery;
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) use session::AttemptReportProvider;
pub use session::RuntimeObservationSession;
pub(crate) use session::{AttemptReportSealProvider, QueryCleanupSealDelivery};
use session::{DeferredRuntimeObservationSeal, RuntimeObservationSealObserver, RuntimeObservationSessionInner};
pub use state::BudgetWindowPeaks;
pub(crate) use state::{BudgetWindow, CleanupFailureTally};
use state::{PlanExecutionMetrics, RuntimeObservationRecorderState};

/// Query-local recorder configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeObservationConfig {
    enabled: bool,
    limits: RuntimeObservationLimits,
}

impl RuntimeObservationConfig {
    /// Default bounded capture used by native execution adapters.
    pub const NATIVE_EXECUTION_MAX_RECORDS: u64 = 65_536;
    /// Deterministic logical-byte cap paired with [`Self::NATIVE_EXECUTION_MAX_RECORDS`].
    pub const NATIVE_EXECUTION_MAX_BYTES: u64 = 8 * 1024 * 1024;

    /// Disables runtime recording.
    pub const fn disabled() -> Self {
        Self { enabled: false, limits: RuntimeObservationLimits { max_retained_records: 0, max_retained_bytes: 0 } }
    }

    /// Enables recording with deterministic record and logical-byte caps.
    pub const fn bounded(max_retained_records: u64, max_retained_bytes: u64) -> Self {
        Self { enabled: true, limits: RuntimeObservationLimits { max_retained_records, max_retained_bytes } }
    }

    /// Enables the bounded per-query native execution capture profile.
    pub const fn native_execution() -> Self {
        Self::bounded(Self::NATIVE_EXECUTION_MAX_RECORDS, Self::NATIVE_EXECUTION_MAX_BYTES)
    }

    /// Returns whether payload construction and recording are enabled.
    pub(crate) const fn is_enabled(self) -> bool {
        self.enabled
    }

    /// Returns the configured retention limits.
    pub const fn limits(self) -> RuntimeObservationLimits {
        self.limits
    }
}

impl Default for RuntimeObservationConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Optional event identity attached outside the payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeEventContext {
    pub(crate) plan_execution_id: Option<PlanExecutionId>,
    pub(crate) span_id: Option<RuntimeSpanId>,
    pub(crate) parent_span_id: Option<RuntimeSpanId>,
}

impl RuntimeEventContext {
    /// Creates context for one plan execution.
    pub fn for_plan(plan_execution_id: PlanExecutionId) -> Self {
        Self { plan_execution_id: Some(plan_execution_id), ..Self::default() }
    }

    /// Attaches one span and its optional parent.
    pub const fn with_span(mut self, span_id: RuntimeSpanId, parent_span_id: Option<RuntimeSpanId>) -> Self {
        self.span_id = Some(span_id);
        self.parent_span_id = parent_span_id;
        self
    }
}

/// Infallible result of one hot-path record attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeObservationRecordDisposition {
    Disabled,
    Retained { sequence: ObservationSequence, logical_bytes: u64 },
    Dropped { sequence: ObservationSequence, reason: RuntimeObservationLossReason },
    Sealed,
}

/// Immutable live snapshot. Its footer is always incomplete and has no terminal outcome.
#[derive(Debug, Clone)]
pub struct LiveRuntimeObservationSnapshot {
    data: Arc<RuntimeObservationSnapshotData>,
}

/// Immutable sealed snapshot accepted by the JSONL exporter.
#[derive(Debug, Clone)]
pub struct SealedRuntimeObservationSnapshot {
    data: Arc<RuntimeObservationSnapshotData>,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeObservationSnapshotData {
    pub(crate) manifest: RuntimeObservationManifest,
    pub(crate) topology: Vec<RetainedTopologyRecord>,
    pub(crate) events: Vec<RuntimeTimelineEvent>,
    pub(crate) footer: RuntimeObservationFooter,
    /// Each plan execution's frozen metrics. Only a seal fills it.
    pub(crate) plans: BTreeMap<PlanExecutionId, QueryMetrics>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RetainedTopologyRecord {
    pub(crate) sequence: ObservationSequence,
    pub(crate) record: RuntimeTopologyRecord,
}

macro_rules! snapshot_accessors {
    ($snapshot:ident) => {
        impl $snapshot {
            /// Immutable session manifest.
            pub fn manifest(&self) -> &RuntimeObservationManifest {
                &self.data.manifest
            }

            /// Static topology sorted by typed row identity.
            pub fn topology(&self) -> impl ExactSizeIterator<Item = &RuntimeTopologyRecord> {
                self.data.topology.iter().map(|record| &record.record)
            }

            /// Dynamic events sorted by recorder sequence.
            pub fn events(&self) -> &[RuntimeTimelineEvent] {
                &self.data.events
            }

            /// Completeness, terminal, loss, and degradation metadata.
            pub fn footer(&self) -> &RuntimeObservationFooter {
                &self.data.footer
            }

            /// Fixed execution-attempt summary, when this observation belongs to an attempt.
            ///
            /// The embedded report is the same immutable value delivered by
            /// the attempt completion. In particular, `report_ready_ns` is
            /// fixed before this snapshot is published and the JSONL footer
            /// serializes this exact allocation.
            pub fn attempt_report(&self) -> Option<&QueryAttemptReport> {
                self.data.footer.attempt_report.as_deref()
            }
        }
    };
}

snapshot_accessors!(LiveRuntimeObservationSnapshot);
snapshot_accessors!(SealedRuntimeObservationSnapshot);

impl SealedRuntimeObservationSnapshot {
    pub(crate) fn data(&self) -> &RuntimeObservationSnapshotData {
        &self.data
    }

    /// The metrics of plan execution `plan`, as frozen when its owner
    /// projected them or, failing that, when this session sealed.
    pub fn query_metrics(&self, plan: &PlanExecutionId) -> Option<&QueryMetrics> {
        self.data.plans.get(plan)
    }

    /// Returns whether both values are views of the same immutable seal.
    ///
    /// Query identifiers are intentionally insufficient for this comparison:
    /// independently-created recorder sessions may share one external query
    /// identity. Ownership boundaries use this identity check to make repeated
    /// delivery idempotent without silently accepting another session.
    #[cfg(test)]
    pub(crate) fn is_same_snapshot(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

/// Cheaply cloneable recording capability without terminal-seal authority.
#[derive(Clone, Default)]
pub struct RuntimeObservationHandle {
    config: RuntimeObservationConfig,
    inner: Option<Arc<RuntimeObservationSessionInner>>,
}

impl fmt::Debug for RuntimeObservationHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeObservationHandle")
            .field("enabled", &self.is_enabled())
            .field("limits", &self.config.limits())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
