//! Stable backend observability contracts for application exporters.
//!
//! This module is a type namespace: it re-exports the engine-owned snapshot
//! and metric types that application exporters need, and it owns no data
//! source. The only read path is the [`crate::backend::AlgeonGpuBackend`]
//! snapshot methods, which project the engine-owned observation snapshot.
//! Admission truth lives in the device ledger; cache owners contribute
//! correlation observations and telemetry stays diagnostic-only. TSV,
//! DataFusion metrics, and `QueryMetrics` are projections of the same
//! runtime observation snapshot, not a second event store.

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
        BackendAttemptMetricsSnapshot, BackendAttemptOutcome, BackendDeviceMetricsSnapshot, BackendHistogramProjection,
        BackendHistogramSnapshot, BackendMetricAttribute, BackendMetricAttributeValue, BackendMetricProjectionError,
        BackendMetricSample, BackendMetricSampleValue, BackendMetricsSnapshot, CacheGraphActivity, CacheSourceActivity,
        CacheSourceEventSnapshot, DeviceCacheReport, DeviceFacts, DeviceLedgerGrantRow, DeviceLedgerReport,
        DeviceMemoryFailureReport, DeviceResourceProfile, DeviceTelemetrySnapshot, DeviceTelemetryStatus,
        GpuBackendCloseSnapshot, GpuBackendDeviceStartupRecord, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot,
        QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot,
    },
};
