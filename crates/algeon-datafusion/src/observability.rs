//! Stable backend observability contracts for application exporters.
//!
//! Re-exports engine snapshot and metric types for application telemetry
//! exporters. Observation snapshots are queried through [`crate::backend::AlgeonGpuBackend`].
//! Device ledgers record admission grants, while TSV reports and DataFusion
//! execution metrics project from runtime observation snapshots.

pub use query_runtime::{
    metrics::{
        BACKEND_ATTEMPT_DURATION_BUCKETS_NS, BACKEND_METRIC_DESCRIPTORS, BackendDurationProjectionError, BackendMetric,
        BackendMetricDescriptor, BackendMetricKind, BackendMetricLabel, BackendMetricUnit, BackendMetricUpdatePoint,
        backend_metric_descriptors,
    },
    observability::{
        AdmissionRequestId, GpuJobLifecycleState, RuntimeGpuJobObservation, RuntimeObservationExportError,
        RuntimeObservationExportErrorKind, RuntimeObservationExportErrorStatus, RuntimeObservationPayload,
        RuntimeTimelineEvent, SealedRuntimeObservationSnapshot, write_runtime_observation_jsonl,
    },
    runtime::{
        BACKEND_ATTEMPT_DURATION_BUCKET_COUNT, BACKEND_METRIC_DEVICE_CAPACITY, BackendAdmissionHealth,
        BackendAdmissionMetricsSnapshot, BackendAttemptDisposition, BackendAttemptDuration,
        BackendAttemptMetricsSnapshot, BackendAttemptOutcome, BackendDeviceMetricsSnapshot, BackendDiagnostics,
        BackendHistogramProjection, BackendHistogramSnapshot, BackendMetricAttribute, BackendMetricAttributeValue,
        BackendMetricProjectionError, BackendMetricSample, BackendMetricSampleValue, BackendMetricsSnapshot,
        CacheGraphActivity, CacheSourceActivity, CacheSourceEventSnapshot, DeviceCacheReport, DeviceFacts,
        DeviceLedgerGrantRow, DeviceLedgerReport, DeviceMemoryFailureReport, DeviceResourceProfile,
        DeviceTelemetrySnapshot, DeviceTelemetryStatus, GpuBackendCloseSnapshot, GpuBackendDeviceStartupRecord,
        GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, KvikioCompatMode, KvikioProcessConfig,
        QueryServiceAdmissionSnapshot,
    },
};
