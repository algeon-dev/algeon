use nexus_query_engine::metrics::{
    BackendDurationProjectionError, BackendMetricKind, BackendMetricUnit, backend_metric_descriptors,
    project_backend_duration_histogram,
};

#[test]
fn backend_descriptor_golden_names_and_contract_shape() {
    let descriptors = backend_metric_descriptors();
    let expected = [
        ("nexus.gpu.admission.active_attempts", BackendMetricUnit::One, BackendMetricKind::Gauge),
        ("nexus.gpu.admission.live_grant", BackendMetricUnit::Bytes, BackendMetricKind::Gauge),
        ("nexus.gpu.admission.live_reserved", BackendMetricUnit::Bytes, BackendMetricKind::Gauge),
        ("nexus.gpu.admission.max_live_reserved", BackendMetricUnit::Bytes, BackendMetricKind::Max),
        ("nexus.gpu.attempts.completed", BackendMetricUnit::One, BackendMetricKind::Counter),
        ("nexus.gpu.attempts.duration", BackendMetricUnit::Seconds, BackendMetricKind::Histogram),
        ("nexus.gpu.memory.allocation_failures", BackendMetricUnit::One, BackendMetricKind::Counter),
        ("nexus.gpu.memory.allocation_failure", BackendMetricUnit::Bytes, BackendMetricKind::Counter),
        ("nexus.gpu.cache.evictable.resident", BackendMetricUnit::Bytes, BackendMetricKind::Gauge),
        ("nexus.gpu.cache.pinned.in_use", BackendMetricUnit::Bytes, BackendMetricKind::Gauge),
    ];
    assert_eq!(descriptors.len(), expected.len());
    for (descriptor, (name, unit, kind)) in descriptors.iter().zip(expected) {
        assert_eq!((descriptor.name, descriptor.unit, descriptor.kind), (name, unit, kind));
        assert_eq!(descriptor.metric.descriptor(), descriptor);
        assert!(descriptor.name.starts_with("nexus."));
        let series = descriptor.labels.iter().map(|label| label.cardinality).product::<usize>();
        assert!(series <= descriptor.max_series);
        assert!(descriptor.labels.iter().all(|label| {
            !matches!(label.name, "query_id" | "attempt_id" | "sql" | "user" | "path" | "uri" | "message" | "node_id")
                && !label.name.ends_with("_bytes")
                && !label.name.ends_with("_rows")
                && !label.name.ends_with("_ns")
        }));
        assert!(matches!(descriptor.unit.as_ucum(), "1" | "By" | "s"));
    }
}

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
