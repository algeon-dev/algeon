use algeon_datafusion::observability::{
    BackendMetricKind, BackendMetricSample, BackendMetricSampleValue, BackendMetricUnit, BackendMetricsSnapshot,
    GpuJobLifecycleState, RuntimeObservationExportError, RuntimeObservationPayload, RuntimeTimelineEvent,
    SealedRuntimeObservationSnapshot, backend_metric_descriptors, write_runtime_observation_jsonl,
};

#[test]
fn backend_observability_facade_preserves_descriptor_and_projection_contracts() {
    let descriptors = backend_metric_descriptors();
    assert_eq!(descriptors.first().map(|descriptor| descriptor.name), Some("algeon.gpu.admission.active_attempts"));
    assert_eq!(descriptors.first().map(|descriptor| descriptor.unit), Some(BackendMetricUnit::One));
    assert_eq!(descriptors.first().map(|descriptor| descriptor.kind), Some(BackendMetricKind::Gauge));

    let histogram = descriptors
        .iter()
        .find(|descriptor| descriptor.kind == BackendMetricKind::Histogram)
        .expect("the backend descriptor registry keeps its attempt-duration histogram");
    assert_eq!(histogram.name, "algeon.gpu.attempts.duration");
    assert_eq!(histogram.unit, BackendMetricUnit::Seconds);

    let _projection_contract: fn(&BackendMetricsSnapshot) -> Vec<BackendMetricSampleValue> = |snapshot| {
        backend_metric_descriptors()
            .iter()
            .flat_map(|descriptor| snapshot.project_backend_metric(descriptor).expect("canonical descriptor projects"))
            .map(|BackendMetricSample { value, .. }| value)
            .collect()
    };
}

#[expect(dead_code, reason = "compile-time public facade check")]
fn runtime_observability_facade_exposes_timeline_and_jsonl_contracts() {
    let _export: fn(&SealedRuntimeObservationSnapshot, &mut Vec<u8>) -> Result<(), RuntimeObservationExportError> =
        write_runtime_observation_jsonl::<Vec<u8>>;
    let _inspect_gpu_job = |event: &RuntimeTimelineEvent| {
        matches!(
            &event.payload,
            RuntimeObservationPayload::GpuJob(job) if job.lifecycle == GpuJobLifecycleState::Started
        )
    };
}
