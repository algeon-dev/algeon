use nexus_query_engine::metrics::{
    BackendDurationProjectionError, BackendMetricUnit, backend_metric_descriptors, project_backend_duration_histogram,
};

#[test]
fn duration_projection_uses_the_descriptor_unit_and_exact_seconds_ladder() {
    let descriptor = backend_metric_descriptors()
        .iter()
        .find(|descriptor| descriptor.name == "nexus.gpu.attempts.duration")
        .expect("duration descriptor must be registered");

    let projection = project_backend_duration_histogram(descriptor, 1_000_000_123).expect("seconds are supported");
    assert_eq!(projection.bucket_bounds, vec![0.000_001, 0.000_01, 0.000_1, 0.001, 0.01, 0.1, 1.0]);
    assert_eq!(projection.sum, 1.000_000_123);

    let unsupported_descriptor =
        nexus_query_engine::metrics::BackendMetricDescriptor { unit: BackendMetricUnit::Bytes, ..*descriptor };
    assert_eq!(
        project_backend_duration_histogram(&unsupported_descriptor, 1_000_000_123),
        Err(BackendDurationProjectionError::UnsupportedUnit {
            metric_name: "nexus.gpu.attempts.duration",
            unit: BackendMetricUnit::Bytes,
        }),
    );
}
