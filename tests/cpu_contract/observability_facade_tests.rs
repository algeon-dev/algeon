use datafusion_nexus::observability::{
    BackendMetricKind, BackendMetricSample, BackendMetricSampleValue, BackendMetricUnit, BackendMetricsSnapshot,
    backend_metric_descriptors,
};

#[test]
fn backend_observability_facade_preserves_descriptor_and_projection_contracts() {
    let descriptors = backend_metric_descriptors();
    assert_eq!(descriptors.first().map(|descriptor| descriptor.name), Some("nexus.gpu.admission.active_attempts"));
    assert_eq!(descriptors.first().map(|descriptor| descriptor.unit), Some(BackendMetricUnit::One));
    assert_eq!(descriptors.first().map(|descriptor| descriptor.kind), Some(BackendMetricKind::Gauge));

    let histogram = descriptors
        .iter()
        .find(|descriptor| descriptor.kind == BackendMetricKind::Histogram)
        .expect("the backend descriptor registry keeps its attempt-duration histogram");
    assert_eq!(histogram.name, "nexus.gpu.attempts.duration");
    assert_eq!(histogram.unit, BackendMetricUnit::Seconds);

    let _projection_contract: fn(&BackendMetricsSnapshot) -> Vec<BackendMetricSampleValue> = |snapshot| {
        backend_metric_descriptors()
            .iter()
            .flat_map(|descriptor| snapshot.project_backend_metric(descriptor).expect("canonical descriptor projects"))
            .map(|BackendMetricSample { value, .. }| value)
            .collect()
    };
}
