//! Bounded native query-service admission harness.

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::{sync::Notify, time};

#[cfg(any(test, feature = "test-hooks"))]
use super::GpuBackendCoreDraft;
#[cfg(test)]
use super::GpuInventory;
#[cfg(any(test, feature = "test-hooks"))]
use super::gpu::claim_live_backend;
use super::{
    CacheFillByteBudget, DeploymentIoCapability, DeviceLedger, DeviceLedgerReport, DeviceMemoryFailureReport,
    DeviceRegistry, DeviceRegistryConfig, DeviceTelemetrySnapshot, ExecutionBoundaryPolicy, ExecutionCancellation,
    ExecutionContext, GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe, GpuBackendControllerDraft,
    GpuBackendDeviceConfig, GpuDeviceMemoryLimit, GpuExecutionJobKind, GpuExecutionPool, GpuExecutionPoolSnapshot,
    GpuExecutionQuerySnapshot, GpuExecutionQueryTracker, GpuMemoryOwnership, GpuTelemetry, LakehouseCacheFillConfig,
    MemoryControlPolicy, MemoryGrant, MemoryResourcePolicy, NexusGpuInventory, NexusNvmlTelemetry, PlacementRequest,
    SourceChunkPlanningMode, SourceSchedulerPolicy, SourceStorageProfile,
    cache_events::LakehouseCacheFillEventSink,
    cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
    context::{ExecutionContextParts, QueryServiceEpoch, QueryServiceEpochUser, validate_service_stream_pool_device},
    gpu,
    gpu::{
        DeviceOwnerToken, normalize_device_ordinals,
        registry::{
            DeviceRegistryCapacityPolicy, DeviceRegistryInstallProfile, DeviceRegistryMemoryPartition,
            ResolvedDeviceInstallPlans,
        },
    },
    object_store::{SourceObjectStoreRegistry, default_object_store_control_runtime},
};
use crate::{
    admission::{AttemptRequirement, AttemptReservationEnvelope},
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    error::{QueryError, QueryErrorCode, Result},
    metrics::QueryId,
    mode::StatisticalAggregateExecutionMode,
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, RuntimeAdmissionObservation, RuntimeObservationHandle,
        RuntimeObservationOwner, RuntimeObservationPayload, RuntimeTerminalOutcome,
    },
    runtime::{
        backend_core::{ConstructionResourceJournal, GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
    },
};

mod admission;
mod config;
mod device_waiters;
mod handle;
mod resources;
mod service;
#[cfg(test)]
mod tests;

#[doc(hidden)]
pub use admission::QueryAdmissionWaiter;
use admission::{
    AdmissionRequest, AdmissionState, DeviceAdmissionEntries, DeviceAdmissionEntry, PendingPreflightReservation,
    PlacedQuery, QueryCompletion, QueuedAdmissionTerminal, admission_cancellation_or_deadline_error,
    admission_queue_full_error, finish_closed_queued_requests, finish_expired_queued_requests,
    finish_removed_request_cancelled, finish_removed_request_error, remove_queued_request, service_closed_error,
    service_overloaded_error,
};
pub use config::{QueryExecutionConfig, QueueBounds};
use device_waiters::{
    PendingDeviceSlotNotifications, notify_shared_device_admission_available, register_shared_device_admission_waiters,
};
use handle::QueryHandleLifetime;
pub use handle::{AdmittedQueryContext, QueryHandle};
use resources::QueryServiceResources;
#[doc(hidden)]
pub use service::QueryService;
use service::QueryServiceInner;
pub use service::{
    DeviceQuarantineInjection, QueryActivityLease, QueryAdmissionOutcome, QueryAdmissionRequest,
    QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum QueryServiceDrainPhase {
    StreamPoolClose,
    AdmissionQuiescence,
    ExecutionPoolQuiescence,
    StreamPoolQuiescence,
    ResourceEpoch,
    FinalCompletion,
}

impl QueryServiceDrainPhase {
    #[doc(hidden)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StreamPoolClose => "stream_pool_close",
            Self::AdmissionQuiescence => "admission_quiescence",
            Self::ExecutionPoolQuiescence => "execution_pool_quiescence",
            Self::StreamPoolQuiescence => "stream_pool_quiescence",
            Self::ResourceEpoch => "resource_epoch",
            Self::FinalCompletion => "final_completion",
        }
    }
}

// The `error` payload is consumed only through `Debug` diagnostics; the
// drain loop itself reads only the phase.
#[allow(dead_code)]
#[derive(Debug)]
#[doc(hidden)]
pub enum QueryServiceDrainPhaseOutcome {
    Quiescent { phase: QueryServiceDrainPhase },
    DeadlineElapsed { phase: QueryServiceDrainPhase, error: QueryError },
    PermanentFailure { phase: QueryServiceDrainPhase, error: QueryError },
}

impl QueryServiceDrainPhaseOutcome {
    fn from_result(phase: QueryServiceDrainPhase, result: Result<()>) -> Self {
        match result {
            Ok(()) => Self::Quiescent { phase },
            Err(error) if error.is_cleanup_deadline_elapsed() => Self::DeadlineElapsed { phase, error },
            Err(error) => Self::PermanentFailure { phase, error },
        }
    }

    #[doc(hidden)]
    pub const fn phase(&self) -> QueryServiceDrainPhase {
        match self {
            Self::Quiescent { phase } | Self::DeadlineElapsed { phase, .. } | Self::PermanentFailure { phase, .. } => {
                *phase
            },
        }
    }

    #[doc(hidden)]
    pub const fn is_quiescent(&self) -> bool {
        matches!(self, Self::Quiescent { .. })
    }
}

/// One phase-total cleanup pass.
///
/// Every owned component contributes a typed outcome. A phase that depends on
/// unfinished earlier cleanup is reported as deadline-pending; in particular,
/// the cleanup executors must remain available for anchors released after an
/// earlier bounded admission wait times out.
#[derive(Debug)]
#[doc(hidden)]
pub struct QueryServiceDrainOutcome {
    phases: Vec<QueryServiceDrainPhaseOutcome>,
}

impl QueryServiceDrainOutcome {
    fn new(phases: Vec<QueryServiceDrainPhaseOutcome>) -> Self {
        Self { phases }
    }

    #[doc(hidden)]
    pub fn phases(&self) -> &[QueryServiceDrainPhaseOutcome] {
        &self.phases
    }

    #[cfg(any(test, feature = "test-hooks"))]
    fn into_result(self) -> Result<()> {
        let mut first_deadline = None;
        for outcome in self.phases {
            match outcome {
                QueryServiceDrainPhaseOutcome::PermanentFailure { error, .. } => return Err(error),
                QueryServiceDrainPhaseOutcome::DeadlineElapsed { error, .. } if first_deadline.is_none() => {
                    first_deadline = Some(error);
                },
                QueryServiceDrainPhaseOutcome::Quiescent { .. }
                | QueryServiceDrainPhaseOutcome::DeadlineElapsed { .. } => {},
            }
        }
        first_deadline.map_or(Ok(()), Err)
    }
}

fn record_admission_lifecycle(
    observation: &RuntimeObservationHandle,
    admission_request_id: AdmissionRequestId,
    lifecycle: AdmissionLifecycleState,
    queue_depth: usize,
    device_ordinal: Option<usize>,
    grant_bytes: Option<u64>,
) {
    if !observation.is_enabled() {
        return;
    }
    observation.record(|| {
        RuntimeObservationPayload::Admission(Box::new(RuntimeAdmissionObservation {
            admission_request_id,
            lifecycle,
            queue_depth,
            device_ordinal: device_ordinal.and_then(|value| u32::try_from(value).ok()),
            grant_bytes,
        }))
    });
}

fn admission_rejection(error: &QueryError) -> AdmissionLifecycleState {
    AdmissionLifecycleState::Rejected {
        reason: error.fact_value("reason").unwrap_or_else(|| error.code().as_str()).to_string(),
    }
}

#[cfg(test)]
mod drain_outcome_tests {
    use super::*;

    #[test]
    fn later_permanent_failure_wins_over_earlier_deadline() {
        let deadline = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.test_cleanup",
            "injected cleanup deadline",
        )
        .with_fact("reason", "injected_deadline")
        .with_cleanup_deadline_elapsed();
        let permanent = QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.query_service.test_cleanup",
            "injected permanent cleanup failure",
        )
        .with_fact("reason", "injected_permanent_failure");
        let outcome = QueryServiceDrainOutcome::new(vec![
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::StreamPoolClose, Err(deadline)),
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::StreamPoolQuiescence, Err(permanent)),
        ]);

        let error = outcome.into_result().unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("injected_permanent_failure"));
    }

    #[test]
    fn non_deadline_service_overloaded_is_permanent() {
        let overloaded = QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.test_cleanup",
            "ordinary overload is not a cleanup deadline",
        )
        .with_fact("reason", "ordinary_service_overload");
        let outcome =
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::AdmissionQuiescence, Err(overloaded));

        assert!(matches!(outcome, QueryServiceDrainPhaseOutcome::PermanentFailure { .. }));
    }
}
