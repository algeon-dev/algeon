//! Coordinator identity, terminal, config, and hot-counter types.

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use query_engine::plan::MlForestModelDigest;

use crate::server::{
    adapter_error::{Error, ErrorCode},
    attempt::AlgeonAttempt,
    config::ServerObservabilityConfig,
    observability::CapturedError,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct QueryKey {
    pub(crate) query_id: Arc<str>,
    pub(crate) generation: u64,
}

impl QueryKey {
    pub(crate) fn query_id(&self) -> &str {
        &self.query_id
    }
}

/// The coordinator observation generation a test-held observation handle binds to.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ObservationId(pub(crate) Arc<str>);

#[derive(Debug, Clone)]
pub(crate) struct QueryCoordinatorConfig {
    pub(crate) max_live_queries: usize,
    pub(crate) max_ticketed_queries: usize,
    pub(crate) max_active_logical_capture_bytes: u64,
    pub(crate) per_query_capture_max_bytes: u64,
    pub(crate) per_query_artifact_max_bytes: u64,
    pub(crate) retained_max_count: usize,
    pub(crate) retained_max_bytes: u64,
    pub(crate) retained_ttl: Duration,
    pub(crate) tombstone_max_count: usize,
    pub(crate) tombstone_ttl: Duration,
    pub(crate) shutdown_grace: Duration,
}

impl QueryCoordinatorConfig {
    /// Registered queries the coordinator holds at once, live or finalizing.
    /// Server config requires admission to hold fewer attempts, so admission,
    /// not registration, refuses an overloaded server.
    pub(crate) const MAX_LIVE_QUERIES: usize = 1024;

    /// `max_active_logical_capture_bytes` is the budget the server resolved
    /// from `config` and its admission bounds.
    pub(crate) fn from_observability(
        config: &ServerObservabilityConfig,
        max_active_logical_capture_bytes: u64,
    ) -> Self {
        Self {
            max_live_queries: Self::MAX_LIVE_QUERIES,
            max_ticketed_queries: 1024,
            max_active_logical_capture_bytes,
            per_query_capture_max_bytes: config.per_query_max_bytes,
            per_query_artifact_max_bytes: config.per_query_artifact_max_bytes(),
            retained_max_count: config.global_sealed_max_count.max(1),
            retained_max_bytes: config.global_sealed_max_bytes,
            retained_ttl: config.sealed_ttl,
            tombstone_max_count: config.global_sealed_max_count.max(1),
            tombstone_ttl: config.sealed_ttl,
            shutdown_grace: config.shutdown_grace,
        }
    }
}

/// Logical capture bytes one live query reserves from
/// `max_active_logical_capture_bytes` while it is registered: twice its runtime
/// capture limit plus the per-query capture and artifact limits. `None` when the
/// sum overflows `u64`.
pub(crate) fn logical_capture_reservation_bytes(
    runtime_capture_bytes: u64,
    per_query_capture_max_bytes: u64,
    per_query_artifact_max_bytes: u64,
) -> Option<u64> {
    runtime_capture_bytes
        .checked_mul(2)?
        .checked_add(per_query_capture_max_bytes)?
        .checked_add(per_query_artifact_max_bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryCoordinatorErrorKind {
    RegistrationCapacity,
    DuplicateLiveQueryId,
    TicketCapacity,
    UnknownQuery,
    StaleGeneration,
    InvalidTransition,
    TicketExpired,
    GenerationExhausted,
    Shutdown,
    ArtifactEgressCapacity,
    ArtifactRenderFailed,
}

impl QueryCoordinatorErrorKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RegistrationCapacity => "registration_capacity",
            Self::DuplicateLiveQueryId => "duplicate_live_query_id",
            Self::TicketCapacity => "ticket_capacity",
            Self::UnknownQuery => "unknown_query",
            Self::StaleGeneration => "stale_generation",
            Self::InvalidTransition => "invalid_transition",
            Self::TicketExpired => "ticket_expired",
            Self::GenerationExhausted => "generation_exhausted",
            Self::Shutdown => "shutdown",
            Self::ArtifactEgressCapacity => "artifact_egress_capacity",
            Self::ArtifactRenderFailed => "artifact_render_failed",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QueryCoordinatorError {
    pub(crate) kind: QueryCoordinatorErrorKind,
    pub(crate) message: String,
    pub(crate) query_id: Option<String>,
}

impl QueryCoordinatorError {
    pub(crate) fn new(kind: QueryCoordinatorErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), query_id: None }
    }

    pub(crate) fn with_query_id(mut self, query_id: impl Into<String>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    pub(crate) fn query_id(&self) -> Option<&str> {
        self.query_id.as_deref()
    }

    pub(crate) const fn kind(&self) -> QueryCoordinatorErrorKind {
        self.kind
    }

    pub(crate) fn adapter_error(&self, operation: &'static str) -> Error {
        let code = match self.kind {
            QueryCoordinatorErrorKind::RegistrationCapacity
            | QueryCoordinatorErrorKind::DuplicateLiveQueryId
            | QueryCoordinatorErrorKind::TicketCapacity
            | QueryCoordinatorErrorKind::ArtifactEgressCapacity => ErrorCode::ServiceOverloaded,
            QueryCoordinatorErrorKind::UnknownQuery | QueryCoordinatorErrorKind::TicketExpired => {
                ErrorCode::InvalidRequest
            },
            QueryCoordinatorErrorKind::StaleGeneration | QueryCoordinatorErrorKind::InvalidTransition => {
                ErrorCode::Internal
            },
            QueryCoordinatorErrorKind::GenerationExhausted | QueryCoordinatorErrorKind::ArtifactRenderFailed => {
                ErrorCode::Internal
            },
            QueryCoordinatorErrorKind::Shutdown => ErrorCode::Cancelled,
        };
        Error::new(code, operation, self.message.clone())
            .with_log_safe_fact("query_coordinator_error", self.kind.as_str())
    }
}

impl fmt::Display for QueryCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for QueryCoordinatorError {}

#[derive(Debug, Clone)]
pub(crate) enum QueryTerminal {
    Succeeded,
    Failed { stage: &'static str, error: Arc<CapturedError> },
    Cancelled { stage: &'static str, error: Arc<CapturedError> },
    Dropped,
    Panicked,
}

impl QueryTerminal {
    pub(crate) fn failed(stage: &'static str, error: CapturedError) -> Self {
        Self::Failed { stage, error: Arc::new(error) }
    }

    pub(crate) const fn compact_outcome(&self) -> QueryTerminalOutcome {
        match self {
            Self::Succeeded => QueryTerminalOutcome::Succeeded,
            Self::Failed { .. } => QueryTerminalOutcome::Failed,
            Self::Cancelled { .. } => QueryTerminalOutcome::Cancelled,
            Self::Dropped => QueryTerminalOutcome::Dropped,
            Self::Panicked => QueryTerminalOutcome::Panicked,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryTerminalOutcome {
    Succeeded,
    Failed,
    Cancelled,
    Dropped,
    Panicked,
    ShutdownTimeout,
}

impl QueryTerminalOutcome {
    pub(crate) const fn stats_outcome(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Dropped | Self::ShutdownTimeout => "dropped",
            Self::Panicked => "panicked",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryPhase {
    Planning,
    Ticketed,
    Executing,
    Finalizing,
    Retained,
    Tombstone,
}

#[derive(Debug, Default)]
pub(crate) struct QueryHotCounters {
    pub(crate) output_batches: AtomicUsize,
    pub(crate) output_rows: AtomicUsize,
    pub(crate) output_bytes: AtomicUsize,
    pub(crate) overflow: AtomicBool,
}

impl QueryHotCounters {
    pub(crate) fn record_output(&self, rows: usize, bytes: usize) {
        saturating_atomic_add(&self.output_batches, 1, &self.overflow);
        saturating_atomic_add(&self.output_rows, rows, &self.overflow);
        saturating_atomic_add(&self.output_bytes, bytes, &self.overflow);
    }

    pub(crate) fn snapshot(&self) -> QueryHotCounterSnapshot {
        QueryHotCounterSnapshot {
            output_batches: self.output_batches.load(Ordering::Relaxed),
            output_rows: self.output_rows.load(Ordering::Relaxed),
            output_bytes: self.output_bytes.load(Ordering::Relaxed),
            overflow: self.overflow.load(Ordering::Relaxed),
        }
    }
}

pub(crate) fn saturating_atomic_add(counter: &AtomicUsize, delta: usize, overflow: &AtomicBool) {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.checked_add(delta).unwrap_or_else(|| {
            overflow.store(true, Ordering::Relaxed);
            usize::MAX
        });
        match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QueryHotCounterSnapshot {
    pub(crate) output_batches: usize,
    pub(crate) output_rows: usize,
    pub(crate) output_bytes: usize,
    pub(crate) overflow: bool,
}

#[derive(Debug)]
pub(crate) struct CoordinatorTicketPayload {
    pub(crate) sql: String,
    pub(crate) correlation_id: Option<String>,
    pub(crate) prepared_handle: Option<String>,
    pub(crate) kind: u8,
    pub(crate) execution_domain: StatementExecutionDomain,
    /// The sole attempt for a Flight SQL result-stream invocation. It begins
    /// before logical planning and moves exactly once into DoGet.
    pub(crate) attempt: Option<AlgeonAttempt>,
    pub(crate) started_at: Instant,
    pub(crate) deadline: Instant,
    /// Forest models bound when the ticket was planned. DoGet replans the
    /// statement and rejects it with `model_changed` when these differ.
    pub(crate) forest_model_digests: Vec<MlForestModelDigest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StatementExecutionDomain {
    Metadata,
    Data,
}

#[derive(Debug, Clone)]
pub(crate) struct RetainedQueryBundle {
    pub(crate) stats: Arc<[u8]>,
    pub(crate) runtime_control: Arc<[u8]>,
    pub(crate) runtime_jsonl: Option<Arc<[u8]>>,
    pub(crate) charge_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetainedUnavailableReason {
    CaptureDisabled,
    AttemptReportDeliveryFailed,
    TooLarge,
    EncodeFailed,
    Evicted,
    TicketExpired,
    ShutdownTimeout,
    FinalizationPanicked,
    CaptureTruncated,
    CounterOverflow,
    /// The active logical capture budget could not hold the query's capture
    /// when it registered, so it ran without one.
    ActiveCaptureBudget,
}

impl RetainedUnavailableReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CaptureDisabled => "capture_disabled",
            Self::AttemptReportDeliveryFailed => "attempt_report_delivery_failed",
            Self::TooLarge => "global_budget",
            Self::EncodeFailed => "encode_failed",
            Self::Evicted => "evicted",
            Self::TicketExpired => "no_native_execution",
            Self::ShutdownTimeout => "shutdown_timeout",
            Self::FinalizationPanicked => "finalization_panicked",
            Self::CaptureTruncated => "capture_budget",
            Self::CounterOverflow => "counter_overflow",
            Self::ActiveCaptureBudget => "active_capture_budget",
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) enum CoordinatorArtifact {
    Pending,
    Active,
    Finalizing,
    Retained(Arc<RetainedQueryBundle>),
    Unavailable(RetainedUnavailableReason),
}
