use datafusion_nexus::observability::{
    BackendHistogramProjection, BackendMetricAttribute, BackendMetricAttributeValue, BackendMetricKind,
    backend_metric_descriptors,
};
use nexus_examples::otlp_metrics::{histogram_data_point, histogram_export_request, histogram_metric};
use opentelemetry_proto::tonic::{
    common::v1::{AnyValue, KeyValue, any_value},
    metrics::v1::{AggregationTemporality, metric},
    resource::v1::Resource,
};

fn duration_projection() -> BackendHistogramProjection {
    BackendHistogramProjection {
        explicit_bounds: vec![0.000_001, 0.000_01, 0.000_1, 0.001, 0.01, 0.1, 1.0],
        bucket_counts: [1, 2, 3, 4, 5, 6, 7, 8],
        count: 36,
        sum: 1.234,
    }
}

#[test]
fn histogram_proto_has_one_implicit_infinity_bucket() {
    let projection = duration_projection();
    let point = histogram_data_point(
        &[BackendMetricAttribute { key: "attempt.duration", value: BackendMetricAttributeValue::Str("total") }],
        &projection,
        10,
        20,
    );

    assert_eq!(point.explicit_bounds.len(), point.bucket_counts.len() - 1);
    assert_eq!(point.bucket_counts.iter().sum::<u64>(), point.count);
    assert_eq!(point.bucket_counts, projection.bucket_counts);
    assert_eq!(point.count, projection.count);
}

#[test]
fn production_request_preserves_caller_resource_and_cumulative_histogram() {
    let descriptor = backend_metric_descriptors()
        .iter()
        .find(|descriptor| descriptor.kind == BackendMetricKind::Histogram)
        .expect("backend descriptor table contains the duration histogram");
    let resource = Resource {
        attributes: vec![KeyValue {
            key: "service.name".into(),
            value: Some(AnyValue { value: Some(any_value::Value::StringValue("embedded-nexus".into())) }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let point = histogram_data_point(&[], &duration_projection(), 10, 20);
    let request = histogram_export_request(&resource, vec![histogram_metric(descriptor, vec![point])]);

    let resource_metrics = request.resource_metrics.first().expect("one caller resource");
    assert_eq!(resource_metrics.resource.as_ref(), Some(&resource));
    let metric = resource_metrics
        .scope_metrics
        .first()
        .and_then(|scope_metrics| scope_metrics.metrics.first())
        .expect("production request contains descriptor-derived histogram metric");
    let Some(metric::Data::Histogram(histogram)) = metric.data.as_ref() else {
        panic!("production metric must use OTLP histogram data");
    };

    assert_eq!(histogram.aggregation_temporality, AggregationTemporality::Cumulative as i32);
    let point = histogram.data_points.first().expect("histogram point is preserved");
    assert_eq!(point.explicit_bounds.len(), point.bucket_counts.len() - 1);
    assert_eq!(point.bucket_counts.iter().sum::<u64>(), point.count);
}
