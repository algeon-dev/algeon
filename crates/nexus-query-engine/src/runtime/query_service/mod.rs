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
    GdsDriverProbeStatus, GpuExecutionJobKind, GpuExecutionPool, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot,
    GpuExecutionQueryTracker, GpuInventory, GpuTelemetry, GrantRequest, KvikioCompatModeStatus,
    LakehouseCacheFillConfig, MemoryControlPolicy, MemoryGovernor, MemoryGovernorReport, MemoryGrant,
    MemoryResourcePolicy, NexusGpuInventory, NexusNvmlTelemetry, NvidiaFsProbeStatus, PlacementPolicy,
    PlacementRequest, Priority, QueryPlacementHint, QueryPlacementPolicy, QueryQos, SourceChunkPlanningMode,
    SourceSchedulerPolicy, SourceStorageProfile,
    cache_events::LakehouseCacheFillEventSink,
    cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
    context::{ExecutionContextParts, QueryServiceEpoch, QueryServiceEpochUser, validate_service_stream_pool_device},
    default_service_stream_pool_size_for_device, gpu,
    gpu::DeviceActiveQuerySlot,
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
    runtime::cache::local_parquet::GpuLocalParquetSourceCache,
    util::comma_separated_usize,
};

mod admission;
mod config;
mod device_waiters;
mod governor_claim;
mod handle;
mod resources;
mod service;
#[cfg(test)]
mod tests;

pub use admission::QueryAdmissionWaiter;
use admission::{
    ActiveAdaptiveGrant, AdaptiveGrowthCandidate, AdmissionRequest, AdmissionState, PlacedQuery, remove_queued_request,
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
use governor_claim::{GovernorClaimLease, claim_governor_ordinals};
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
pub use service::{
    QueryActivityLease, QueryAdmissionRequest, QueryGrantBand, QueryGrantRequest, QueryService,
    QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot, QueryServiceGraphCacheSnapshot,
    TenantQueueDepth,
};
use service::{QueryServiceInner, TenantGrantPolicy};

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
