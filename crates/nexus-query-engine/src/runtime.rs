//! Runtime, memory, boundary, and reservation contracts for native execution.
//!
//! This module is intentionally close to the public API because these values
//! control more than local execution choices. Policy enum string values,
//! reservation fact keys, and retry fact keys flow into tests, reports, and
//! adapter diagnostics. Treat those strings as stable diagnostic contracts:
//! change them only with the same care as a report schema change.

mod allocator_correlation;
mod attempt_domain;
mod backend_close;
mod backend_core;
mod backend_lifecycle;
mod backend_metrics;
mod boundary;
mod bounded_join;
mod cache_events;
mod cache_fill;
mod context;
mod deadline;
pub(crate) mod gpu;
mod gpu_execution_pool;
mod io_capability;
mod io_lifecycle;
mod memory_domain;
#[cfg(test)]
mod native_contract_runtime_observability_tests;
#[cfg(test)]
mod native_execution_tests;
mod object_store;
mod observation_completion;
mod policy;
mod process_exit;
mod query_service;
mod reservation;
/// Frontend-neutral GPU-resident data and reusable build handles.
pub mod resident;
mod service_stream;
mod stream_cleanup;
#[cfg(test)]
mod tests;

pub use attempt_domain::AttemptAllocationDomain;
pub use backend_core::{
    GpuBackendCloseSnapshot, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreBuilder, GpuBackendCoreDraft,
    GpuBackendCoreIdentity, GpuBackendDeviceConfig, GpuBackendDeviceStartupRecord, GpuBackendSession,
    GpuDeviceMemoryLimit, GpuMemoryFraction, GpuMemoryOwnership,
};
pub(crate) use backend_core::{GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe};
#[cfg(test)]
pub(crate) use backend_lifecycle::BackendLifecycle;
pub(crate) use backend_lifecycle::ObservationPermit;
pub use backend_metrics::{
    BACKEND_ATTEMPT_DURATION_BUCKET_COUNT, BACKEND_ATTEMPT_DURATION_BUCKETS_NS, BACKEND_METRIC_DEVICE_CAPACITY,
    BackendAdmissionHealth, BackendAdmissionMetricsSnapshot, BackendAttemptDisposition, BackendAttemptDuration,
    BackendAttemptDurations, BackendAttemptMetricsSnapshot, BackendAttemptOutcome, BackendDeviceMetricsSnapshot,
    BackendHistogramProjection, BackendHistogramSnapshot, BackendMemoryFailureTelemetry, BackendMetricAttribute,
    BackendMetricAttributeValue, BackendMetricProjectionError, BackendMetricSample, BackendMetricSampleValue,
    BackendMetricsSnapshot, BackendTelemetryAvailability, BackendTelemetryMetricsSnapshot, BackendTelemetryRefresh,
    BackendTelemetryRefreshSnapshot, BackendTelemetrySource, project_backend_histogram,
};
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
    ExecutionCancellation, ExecutionContext, LeasedExecutionContext, QueryServiceStreamPool,
    RuntimeObservationBuildError, RuntimeObservedExecutionContext, RuntimeObservedExecutionContextBuilder,
    SourceSchedulerPolicy,
};
pub use gpu::{
    DeviceCapabilities, DeviceCapability, DeviceFacts, DeviceLedger, DeviceLedgerGrantRow, DeviceLedgerReport,
    DeviceMemoryFailureReport, DeviceResourceProfile, DeviceTelemetrySnapshot, DeviceTelemetryStatus,
    EvictableMemoryObserver, EvictableMemoryPressure, GpuInventory, GpuTelemetry, LedgerSnapshot, MemoryGrant,
    NexusGpuInventory, NexusNvmlTelemetry, Placement, device_budget_bytes_from_facts,
};
#[allow(unused_imports)]
pub(crate) use gpu::{DeviceContext, DeviceRegistry, DeviceRegistryConfig, PlacementPolicy, PlacementRequest};
pub use gpu_execution_pool::{
    GpuExecutionJobKind, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, GpuExecutorLease,
};
pub(crate) use gpu_execution_pool::{GpuExecutionPool, GpuExecutionQueryTracker};
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
pub(crate) use process_exit::{
    ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
};
pub(crate) use query_service::QueryService;
pub use query_service::{
    AdmittedQueryContext, QueryActivityLease, QueryAdmissionRequest, QueryExecutionConfig, QueryHandle,
    QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot, QueueBounds,
};
pub use reservation::MemoryReservationRequest;
pub(crate) use reservation::RetryFacts;
pub use service_stream::{
    DEFAULT_SERVICE_STREAM_POOL_SIZE, StreamLease, StreamPermitClass, default_service_stream_pool_size_for_device,
};
pub(crate) use service_stream::{ServiceStreamPool, SourceReadStreamLease};
