//! Runtime, memory, boundary, and reservation contracts for native execution.
//!
//! This module is intentionally close to the public API because these values
//! control more than local execution choices. Policy enum string values,
//! reservation fact keys, and retry fact keys flow into tests, reports, and
//! adapter diagnostics. Treat those strings as stable diagnostic contracts:
//! change them only with the same care as a report schema change.

mod boundary;
mod bounded_join;
/// Runtime-owned GPU caches.
pub mod cache;
mod cache_events;
mod cache_fill;
mod context;
mod deadline;
pub mod gpu;
mod gpu_execution_pool;
mod io_capability;
mod io_lifecycle;
mod memory_domain;
mod object_store;
mod policy;
mod query_service;
mod reservation;
/// Frontend-neutral GPU-resident data and reusable build handles.
pub mod resident;
mod service_stream;
#[cfg(test)]
mod tests;

pub use boundary::{
    BuildMaterializationBoundary, ExecutionBoundaryPolicy, ProbeExecutionBoundary, ResultMaterializationBoundary,
    SourceReadBoundary,
};
pub use bounded_join::{
    BoundedJoinReservationContract, BoundedJoinReservationRequirement, BoundedJoinReservationStatus,
};
pub use cache_events::LakehouseCacheFillEventSnapshot;
pub(crate) use cache_events::{LakehouseCacheFillErrorEvent, LakehouseCacheFillEvent, LakehouseCacheFillOutcome};
pub use cache_fill::{
    CacheFillByteBudget, CacheFillBytePermit, DEFAULT_LAKEHOUSE_CACHE_FILL_ACQUIRE_TIMEOUT,
    DEFAULT_LAKEHOUSE_CACHE_FILL_BUDGET_BYTES, DEFAULT_LAKEHOUSE_CACHE_FILL_BUDGET_CAP_BYTES,
    DEFAULT_LAKEHOUSE_CACHE_FILL_MAX_CONCURRENT_OPS, LakehouseCacheFillConfig,
    default_lakehouse_cache_fill_budget_bytes,
};
pub use context::{
    ExecutionCancellation, ExecutionContext, ExecutionContextBuilder, LeasedExecutionContext, QueryServiceStreamPool,
    RuntimeObservationBuildError, RuntimeObservedExecutionContext, RuntimeObservedExecutionContextBuilder,
    SourceSchedulerPolicy,
};
pub use gpu::{
    BestFitPlacement, CudfInventory, CudfNvmlTelemetry, DeviceContext, DeviceFacts, DeviceMemoryFailureReport,
    DeviceRegistry, DeviceRegistryConfig, DeviceTelemetrySnapshot, DeviceTelemetryStatus, EvictableMemoryObserver,
    EvictableMemoryPressure, GpuComms, GpuInventory, GpuTelemetry, GrantRequest, LeastLoadedPlacement, MemoryGovernor,
    MemoryGovernorGrantRow, MemoryGovernorReport, MemoryGrant, PinnedDevicePlacement, Placement, PlacementPolicy,
    PlacementRequest, Priority, QueryPlacementHint, QueryPlacementPolicy, QueryQos, SingleDeviceComms,
    device_budget_bytes_from_facts,
};
pub use gpu_execution_pool::{
    GpuExecutionJobKind, GpuExecutionPool, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot,
    GpuExecutionQueryTracker,
};
pub use io_capability::{
    DeploymentIoCapability, GdsAvailabilityStatus, GdsDriverProbeStatus, KVIKIO_COMPAT_MODE_ENV,
    KvikioCompatModeStatus, NVIDIA_FS_PROC_PATH, NVIDIA_FS_SYSFS_PATH, NvidiaFsProbeStatus,
};
pub use io_lifecycle::{IoLifecycle, IoLifecycleLease};
pub(crate) use object_store::S3ObjectStoreKey;
pub use object_store::{S3SourceStorageCredential, S3SourceStorageProfile, SourceStorageProfile};
pub use policy::{
    DevicePlacement, MemoryControlPolicy, MemoryResourcePolicy, MemorySpaceId, ReservationKind, SourceChunkPlanningMode,
};
pub use query_service::{
    QueryActivityLease, QueryAdmissionRequest, QueryAdmissionWaiter, QueryGrantBand, QueryGrantRequest, QueryHandle,
    QueryMemoryDomainPolicy, QueryService, QueryServiceAdmissionSnapshot, QueryServiceConfig,
    QueryServiceDeviceAdmissionSnapshot, QueryServiceGraphCacheSnapshot, QueryServiceTenantConfig, SlotGuarantee,
    TenantId, TenantQueryContext, TenantQueueDepth,
};
pub use reservation::{MemoryReservationRequest, RetryFacts};
pub(crate) use service_stream::SourceReadStreamLease;
pub use service_stream::{
    DEFAULT_SERVICE_STREAM_POOL_SIZE, ServiceStreamPool, StreamLease, default_service_stream_pool_size_for_device,
};
