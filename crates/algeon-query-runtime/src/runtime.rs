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
#[doc(hidden)]
pub mod cache;
mod cache_events;
mod cache_fill;
mod context;
mod deadline;
pub(crate) mod gpu;
mod gpu_execution_pool;
mod io_lifecycle;
mod memory_domain;
mod object_store;
mod observation_completion;
mod process_exit;
mod query_service;
mod reservation;
/// Frontend-neutral GPU-resident data and reusable build handles.
pub mod resident;
mod service_stream;
#[doc(hidden)]
pub mod source_identity;
mod stream_cleanup;
mod supervised_worker;
#[cfg(test)]
mod tests;

#[doc(hidden)]
pub use attempt_domain::AttemptStreamProviderRoute;
#[cfg(feature = "qualification")]
pub use backend_core::GpuBackendFaultInjector;
pub use backend_core::{
    GpuBackendCloseSnapshot, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig,
    GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuBackendDeviceStartupRecord, GpuBackendSession,
    GpuDeviceCapacityProbe, GpuDeviceMemoryLimit, GpuMemoryFraction, GpuMemoryOwnership, probe_device_capacity,
};
pub(crate) use backend_core::{GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe};
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub use backend_lifecycle::BackendLifecycle;
pub(crate) use backend_lifecycle::ObservationPermit;
pub use backend_metrics::{
    BACKEND_ATTEMPT_DURATION_BUCKET_COUNT, BACKEND_METRIC_DEVICE_CAPACITY, BackendAdmissionHealth,
    BackendAdmissionMetricsSnapshot, BackendAttemptDisposition, BackendAttemptDuration, BackendAttemptMetricsSnapshot,
    BackendAttemptOutcome, BackendDeviceMetricsSnapshot, BackendHistogramProjection, BackendHistogramSnapshot,
    BackendMetricAttribute, BackendMetricAttributeValue, BackendMetricProjectionError, BackendMetricSample,
    BackendMetricSampleValue, BackendMetricsSnapshot,
};
pub use cache::{
    CacheGraphActivity, CacheSourceActivity, CacheSourceEventSnapshot, DeviceCacheReport, cache_observation_fingerprint,
};
pub use cache_events::LakehouseCacheFillEventSnapshot;
#[doc(hidden)]
pub use cache_events::{LakehouseCacheFillErrorEvent, LakehouseCacheFillEvent, LakehouseCacheFillOutcome};
pub(crate) use cache_fill::CacheFillByteBudget;
pub use cache_fill::{CacheFillBytePermit, LakehouseCacheFillConfig};
pub use context::{ExecutionCancellation, ExecutionContext};
#[cfg(any(test, feature = "test-hooks"))]
pub use context::{RuntimeObservationBuildError, RuntimeObservedExecutionContext, StandaloneConfig};
pub(crate) use gpu::{AlgeonGpuInventory, DeviceContext, DeviceLedger, DeviceRegistry, MemoryGrant};
pub use gpu::{
    DeviceCapabilities, DeviceCapability, DeviceFacts, DeviceLedgerGrantRow, DeviceLedgerReport,
    DeviceMemoryFailureReport, DeviceResourceProfile, DeviceTelemetrySnapshot, DeviceTelemetryStatus,
};
#[cfg(test)]
pub(crate) use gpu::{DeviceResourceEnvelope, GpuInventory, device_budget_bytes_from_facts};
#[doc(hidden)]
pub use gpu_execution_pool::PoolBlockingWork;
pub use gpu_execution_pool::{
    GpuExecutionJobKind, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, GpuExecutorLease,
};
pub(crate) use gpu_execution_pool::{GpuExecutionPool, GpuExecutionQueryTracker};
pub use io_lifecycle::IoLifecycle;
#[doc(hidden)]
pub use object_store::S3ObjectStoreKey;
pub use object_store::{S3SourceStorageCredential, S3SourceStorageProfile, SourceStorageProfile};
pub use query_planning::execution_contracts::{
    MemoryControlPolicy, MemoryResourcePolicy, ReservationKind, SourceChunkPlanningMode, SourceSchedulerPolicy,
};
#[cfg(feature = "qualification")]
pub use query_service::DeviceQuarantineInjection;
#[doc(hidden)]
pub use query_service::QueryService;
pub use query_service::{
    AdmittedQueryContext, QueryActivityLease, QueryAdmissionRequest, QueryExecutionConfig, QueryHandle,
    QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot, QueueBounds,
};
#[doc(hidden)]
pub use reservation::RetryFacts;
#[cfg(test)]
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) use service_stream::DEFAULT_SERVICE_STREAM_POOL_SIZE;
#[doc(hidden)]
pub use service_stream::{AttemptWorkLease, ServiceStreamPool};
pub(crate) use service_stream::{StreamLease, StreamPermitClass};
#[doc(hidden)]
pub use supervised_worker::{SupervisedWorker, SupervisedWorkerJoin};
