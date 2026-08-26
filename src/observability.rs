//! Stable backend observability contracts for application exporters.

pub use nexus_query_engine::{
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
        BackendMetricSample, BackendMetricSampleValue, BackendMetricsSnapshot, BackendTelemetryAvailability,
        BackendTelemetryMetricsSnapshot, BackendTelemetryRefreshSnapshot, BackendTelemetrySource, CacheGraphActivity,
        CacheSourceActivity, CacheSourceEventSnapshot, DeviceCacheReport, DeviceFacts, DeviceLedgerGrantRow,
        DeviceLedgerReport, DeviceMemoryFailureReport, DeviceResourceProfile, DeviceTelemetrySnapshot,
        DeviceTelemetryStatus, GpuBackendCloseSnapshot, GpuBackendDeviceStartupRecord, GpuExecutionPoolSnapshot,
        GpuExecutionQuerySnapshot, QueryServiceAdmissionSnapshot, QueryServiceDeviceAdmissionSnapshot,
    },
};
