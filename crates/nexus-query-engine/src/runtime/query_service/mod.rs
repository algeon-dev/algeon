//! Multi-tenant service admission harness for native execution.

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::{sync::Notify, time};

#[cfg(feature = "cugraph")]
use super::StreamLease;
#[cfg(feature = "cugraph")]
use super::gpu::placement::rank_cache_affine_candidates;
use super::{
    CacheFillByteBudget, DeploymentIoCapability, DeviceContext, DeviceMemoryFailureReport, DeviceRegistry,
    DeviceRegistryConfig, DeviceTelemetrySnapshot, ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext,
    GdsDriverProbeStatus, GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe, GpuBackendControllerDraft,
    GpuBackendDeviceConfig, GpuBackendMemoryShare, GpuBackendTenantDraft, GpuDeviceMemoryLimit, GpuExecutionJobKind,
    GpuExecutionPool, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, GpuExecutionQueryTracker, GpuInventory,
    GpuMemoryOwnership, GpuTelemetry, GrantRequest, KvikioCompatModeStatus, LakehouseCacheFillConfig,
    MemoryControlPolicy, MemoryGovernor, MemoryGovernorReport, MemoryGrant, MemoryResourcePolicy, NexusGpuInventory,
    NexusNvmlTelemetry, NvidiaFsProbeStatus, PlacementPolicy, PlacementRequest, Priority, QueryPlacementHint,
    QueryPlacementPolicy, QueryQos, SourceChunkPlanningMode, SourceSchedulerPolicy, SourceStorageProfile,
    cache_events::LakehouseCacheFillEventSink,
    cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
    context::{ExecutionContextParts, QueryServiceEpoch, QueryServiceEpochUser, validate_service_stream_pool_device},
    default_service_stream_pool_size_for_device, gpu,
    gpu::{
        DeviceActiveQuerySlot, DeviceOwnerToken, LiveBackendLease, claim_live_backend, normalize_device_ordinals,
        registry::{
            DeviceRegistryCapacityPolicy, DeviceRegistryInstallProfile, DeviceRegistryMemoryPartition,
            ResolvedDeviceInstallPlans,
        },
    },
    object_store::{SourceObjectStoreRegistry, default_object_store_control_runtime},
};
use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    error::{QueryError, QueryErrorCode, Result},
    metrics::QueryId,
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
    observability::{
        AdmissionLifecycleState, AdmissionRequestId, RuntimeAdmissionObservation, RuntimeObservationHandle,
        RuntimeObservationOwner, RuntimeObservationPayload, RuntimeTerminalOutcome,
    },
    runtime::{
        backend_core::{ConstructionResourceJournal, GpuResourcePermit, ObservationCompletionDispatcher},
        backend_lifecycle::ObservationPermit,
        cache::local_parquet::GpuLocalParquetSourceCache,
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

pub(crate) use admission::QueryAdmissionWaiter;
use admission::{
    ActiveAdaptiveGrant, AdaptiveGrowthCandidate, AdmissionRequest, AdmissionState, PlacedQuery, QueryCompletion,
    finish_closed_queued_requests, finish_expired_queued_requests, finish_removed_request_error, remove_queued_request,
    service_closed_error, service_overloaded_error, weighted_adaptive_growth_allocations,
};
#[cfg(test)]
use config::guaranteed_min_slot_capacity_for_capacities;
pub use config::{QueryMemoryDomainPolicy, QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee, TenantId};
use config::{
    admission_capacity_bounds, default_stream_pool_size_for_ordinals, guaranteed_min_slot_capacity,
    minimum_stream_pool_size, placement_policy_admits_ordinal, validate_stream_pool_admission_capacity,
};
use device_waiters::{
    PendingDeviceSlotNotifications, notify_shared_device_admission_available, register_shared_device_admission_waiters,
};
#[cfg(test)]
use device_waiters::{prune_dead_shared_device_admission_waiters, shared_device_admission_waiters};
use handle::QueryHandleLifetime;
pub use handle::{QueryHandle, TenantQueryContext};
#[cfg(test)]
use resources::local_parquet_source_caches_for_ordinals;
#[cfg(feature = "cugraph")]
use resources::resident_graph_caches_for_devices;
#[cfg(all(test, feature = "cugraph"))]
use resources::resident_graph_caches_for_ordinals;
use resources::{
    QueryServiceResources, gpu_execution_worker_count_per_device, local_parquet_source_caches_for_devices,
};
#[cfg(feature = "cugraph")]
use resources::{WarmGraphReservation, resident_graph_cache_max_copies_per_key};
pub(crate) use service::QueryService;
pub use service::{
    QueryActivityLease, QueryAdmissionRequest, QueryGrantBand, QueryGrantRequest, QueryServiceAdmissionSnapshot,
    QueryServiceDeviceAdmissionSnapshot, QueryServiceGraphCacheSnapshot, TenantQueueDepth,
};
use service::{QueryServiceInner, TenantGrantPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QueryServiceDrainPhase {
    StreamPoolClose,
    LocalSourceCache,
    AdmissionQuiescence,
    ResidentGraphCache,
    ExecutionPoolQuiescence,
    StreamPoolQuiescence,
    ResourceEpoch,
    FinalCompletion,
}

impl QueryServiceDrainPhase {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StreamPoolClose => "stream_pool_close",
            Self::LocalSourceCache => "local_source_cache",
            Self::AdmissionQuiescence => "admission_quiescence",
            Self::ResidentGraphCache => "resident_graph_cache",
            Self::ExecutionPoolQuiescence => "execution_pool_quiescence",
            Self::StreamPoolQuiescence => "stream_pool_quiescence",
            Self::ResourceEpoch => "resource_epoch",
            Self::FinalCompletion => "final_completion",
        }
    }
}

#[derive(Debug)]
pub(crate) enum QueryServiceDrainPhaseOutcome {
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

    pub(crate) const fn phase(&self) -> QueryServiceDrainPhase {
        match self {
            Self::Quiescent { phase } | Self::DeadlineElapsed { phase, .. } | Self::PermanentFailure { phase, .. } => {
                *phase
            },
        }
    }

    pub(crate) const fn is_quiescent(&self) -> bool {
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
pub(crate) struct QueryServiceDrainOutcome {
    phases: Vec<QueryServiceDrainPhaseOutcome>,
}

impl QueryServiceDrainOutcome {
    fn new(phases: Vec<QueryServiceDrainPhaseOutcome>) -> Self {
        Self { phases }
    }

    pub(crate) fn phases(&self) -> &[QueryServiceDrainPhaseOutcome] {
        &self.phases
    }

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
            QueryServiceDrainPhaseOutcome::from_result(QueryServiceDrainPhase::LocalSourceCache, Err(deadline)),
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

fn record_admission_lifecycle(
    observation: &RuntimeObservationHandle,
    admission_request_id: AdmissionRequestId,
    tenant_id: &TenantId,
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
            tenant_id: tenant_id.as_str().to_string(),
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
