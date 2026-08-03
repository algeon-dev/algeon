//! Coordinator identity, terminal, config, and hot-counter types.

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use nexus_query_engine::runtime::ExecutionCancellation;

use super::super::observability::CapturedError;
use crate::{
    attempt::NexusAttempt,
    error::{Error, ErrorCode},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(in crate::server) struct QueryKey {
    pub(in crate::server) query_id: Arc<str>,
    pub(in crate::server) generation: u64,
}

impl QueryKey {
    pub(in crate::server) fn query_id(&self) -> &str {
        &self.query_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(in crate::server) struct ObservationId(pub(in crate::server) Arc<str>);

#[derive(Debug, Clone)]
pub(in crate::server) struct QueryCoordinatorConfig {
    pub(in crate::server) max_live_queries: usize,
    pub(in crate::server) max_ticketed_queries: usize,
    pub(in crate::server) max_active_logical_capture_bytes: u64,
    pub(in crate::server) per_query_capture_max_bytes: u64,
    pub(in crate::server) per_query_artifact_max_bytes: u64,
    pub(in crate::server) retained_max_count: usize,
    pub(in crate::server) retained_max_bytes: u64,
    pub(in crate::server) retained_ttl: Duration,
    pub(in crate::server) tombstone_max_count: usize,
    pub(in crate::server) tombstone_ttl: Duration,
    pub(in crate::server) shutdown_grace: Duration,
}

impl QueryCoordinatorConfig {
    pub(in crate::server) fn from_observability(config: &crate::server::config::ServerObservabilityConfig) -> Self {
        let max_live_queries = 1024;
        Self {
            max_live_queries,
            max_ticketed_queries: 1024,
            max_active_logical_capture_bytes: config.active_logical_capture_max_bytes,
            per_query_capture_max_bytes: config.per_query_max_bytes,
            per_query_artifact_max_bytes: config
                .per_query_max_bytes
                .saturating_mul(2)
                .min(config.global_sealed_max_bytes),
            retained_max_count: config.global_sealed_max_count.max(1),
            retained_max_bytes: config.global_sealed_max_bytes,
            retained_ttl: config.sealed_ttl,
            tombstone_max_count: config.global_sealed_max_count.max(1),
            tombstone_ttl: config.sealed_ttl,
            shutdown_grace: config.shutdown_grace,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::server) enum QueryCoordinatorErrorKind {
    RegistrationCapacity,
    DuplicateLiveQueryId,
    LogicalCaptureBytes,
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
    pub(in crate::server) const fn as_str(self) -> &'static str {
        match self {
            Self::RegistrationCapacity => "registration_capacity",
            Self::DuplicateLiveQueryId => "duplicate_live_query_id",
            Self::LogicalCaptureBytes => "logical_capture_bytes",
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
pub(in crate::server) struct QueryCoordinatorError {
    pub(in crate::server) kind: QueryCoordinatorErrorKind,
    pub(in crate::server) message: String,
    pub(in crate::server) query_id: Option<String>,
}

impl QueryCoordinatorError {
    pub(in crate::server) fn new(kind: QueryCoordinatorErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), query_id: None }
    }

    pub(in crate::server) fn with_query_id(mut self, query_id: impl Into<String>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    pub(in crate::server) fn query_id(&self) -> Option<&str> {
        self.query_id.as_deref()
    }

    pub(in crate::server) const fn kind(&self) -> QueryCoordinatorErrorKind {
        self.kind
    }

    pub(in crate::server) fn adapter_error(&self, operation: &'static str) -> Error {
        let code = match self.kind {
            QueryCoordinatorErrorKind::RegistrationCapacity
            | QueryCoordinatorErrorKind::DuplicateLiveQueryId
            | QueryCoordinatorErrorKind::LogicalCaptureBytes
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
pub(in crate::server) enum QueryTerminal {
    Succeeded,
    Failed { stage: &'static str, error: Arc<CapturedError> },
    Cancelled { stage: &'static str, error: Arc<CapturedError> },
    Dropped,
    Panicked,
}

impl QueryTerminal {
    pub(in crate::server) fn failed(stage: &'static str, error: CapturedError) -> Self {
        Self::Failed { stage, error: Arc::new(error) }
    }

    pub(in crate::server) const fn compact_outcome(&self) -> QueryTerminalOutcome {
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
pub(in crate::server) enum QueryTerminalOutcome {
    Succeeded,
    Failed,
    Cancelled,
    Dropped,
    Panicked,
    ShutdownTimeout,
}

impl QueryTerminalOutcome {
    pub(in crate::server) const fn stats_outcome(self) -> &'static str {
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
pub(in crate::server) enum QueryPhase {
    Planning,
    Ticketed,
    Executing,
    Finalizing,
    Retained,
    Tombstone,
}

#[derive(Debug, Default)]
pub(in crate::server) struct QueryHotCounters {
    pub(in crate::server) output_batches: AtomicUsize,
    pub(in crate::server) output_rows: AtomicUsize,
    pub(in crate::server) output_bytes: AtomicUsize,
    pub(in crate::server) overflow: AtomicBool,
}

impl QueryHotCounters {
    pub(in crate::server) fn record_output(&self, rows: usize, bytes: usize) {
        saturating_atomic_add(&self.output_batches, 1, &self.overflow);
        saturating_atomic_add(&self.output_rows, rows, &self.overflow);
        saturating_atomic_add(&self.output_bytes, bytes, &self.overflow);
    }

    pub(in crate::server) fn snapshot(&self) -> QueryHotCounterSnapshot {
        QueryHotCounterSnapshot {
            output_batches: self.output_batches.load(Ordering::Relaxed),
            output_rows: self.output_rows.load(Ordering::Relaxed),
            output_bytes: self.output_bytes.load(Ordering::Relaxed),
            overflow: self.overflow.load(Ordering::Relaxed),
        }
    }
}

pub(in crate::server) fn saturating_atomic_add(counter: &AtomicUsize, delta: usize, overflow: &AtomicBool) {
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
pub(in crate::server) struct QueryHotCounterSnapshot {
    pub(in crate::server) output_batches: usize,
    pub(in crate::server) output_rows: usize,
    pub(in crate::server) output_bytes: usize,
    pub(in crate::server) overflow: bool,
}

#[derive(Debug)]
pub(in crate::server) struct CoordinatorTicketPayload {
    pub(in crate::server) sql: String,
    pub(in crate::server) correlation_id: Option<String>,
    pub(in crate::server) prepared_handle: Option<String>,
    pub(in crate::server) kind: u8,
    pub(in crate::server) native_cancellation: ExecutionCancellation,
    /// The sole attempt for a Flight SQL result-stream invocation. It begins
    /// before logical planning and moves exactly once into DoGet.
    pub(in crate::server) attempt: Option<NexusAttempt>,
    pub(in crate::server) started_at: Instant,
    pub(in crate::server) deadline: Instant,
}

#[derive(Debug, Clone)]
pub(in crate::server) struct RetainedQueryBundle {
    pub(in crate::server) stats: Arc<[u8]>,
    pub(in crate::server) runtime_control: Arc<[u8]>,
    pub(in crate::server) runtime_jsonl: Option<Arc<[u8]>>,
    pub(in crate::server) charge_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::server) enum RetainedUnavailableReason {
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
}

impl RetainedUnavailableReason {
    pub(in crate::server) const fn as_str(self) -> &'static str {
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
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(in crate::server) enum CoordinatorArtifact {
    Pending,
    Active,
    Finalizing,
    Retained(Arc<RetainedQueryBundle>),
    Unavailable(RetainedUnavailableReason),
}
