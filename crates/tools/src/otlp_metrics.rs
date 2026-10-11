//! Descriptor-driven OpenTelemetry metrics export for embedded backends.
//!
//! The engine owns measurement, labels, units, bucket conversion, and
//! aggregation semantics. This adapter only reads a coherent
//! [`BackendMetricsSnapshot`] and selects the OpenTelemetry ingress from each
//! [`BackendMetricKind`]. Applications still own their SDK provider, export
//! schedule, resource attributes, and endpoint policy.

use std::sync::Arc;

use algeon_datafusion::observability::{
    BackendHistogramProjection, BackendMetricAttribute, BackendMetricAttributeValue, BackendMetricDescriptor,
    BackendMetricKind, BackendMetricSampleValue, BackendMetricsSnapshot, backend_metric_descriptors,
};
use anyhow::{Context, Result, anyhow, ensure};
use opentelemetry::{
    KeyValue,
    metrics::{AsyncInstrument, Meter},
};
use opentelemetry_proto::tonic::{
    collector::metrics::v1::{ExportMetricsServiceRequest, metrics_service_client::MetricsServiceClient},
    common::v1::{AnyValue, InstrumentationScope, KeyValue as ProtoKeyValue, any_value},
    metrics::v1::{
        AggregationTemporality, Histogram, HistogramDataPoint, Metric, ResourceMetrics, ScopeMetrics, metric,
    },
    resource::v1::Resource,
};

type SnapshotSource = dyn Fn() -> BackendMetricsSnapshot + Send + Sync + 'static;

/// Registers the observable scalar instruments and owns the direct-histogram
/// snapshot source.
///
/// The closure must return the backend's coherent generation snapshot. It is
/// invoked only by application-controlled collection/export work; it is never
/// installed on native execution or cleanup paths.
pub struct BackendMetricsOtlpExporter {
    snapshot_source: Arc<SnapshotSource>,
}

impl BackendMetricsOtlpExporter {
    /// Registers all counter, gauge, and max descriptors with `meter`.
    ///
    /// The caller must keep its OpenTelemetry SDK meter provider alive. The
    /// SDK/provider is intentionally caller-owned, as are its scalar exporter
    /// and collection interval.
    pub fn new<F>(meter: &Meter, snapshot_source: F) -> Self
    where F: Fn() -> BackendMetricsSnapshot + Send + Sync + 'static {
        let snapshot_source: Arc<SnapshotSource> = Arc::new(snapshot_source);
        register_observable_scalars(meter, &snapshot_source);
        Self { snapshot_source }
    }

    /// Builds the direct OTLP histogram request for one coherent snapshot.
    ///
    /// `start_time_unix_nano` belongs to the embedding application's exporter
    /// lifecycle (and should reset when the backend generation changes).
    /// `time_unix_nano` is the common collection timestamp for every point in
    /// this request.
    pub fn histogram_request(
        &self,
        resource: &Resource,
        start_time_unix_nano: u64,
        time_unix_nano: u64,
    ) -> Result<ExportMetricsServiceRequest> {
        ensure!(
            time_unix_nano >= start_time_unix_nano,
            "OTLP collection time must not precede its cumulative start time"
        );
        histogram_request(&(self.snapshot_source)(), resource, start_time_unix_nano, time_unix_nano)
    }

    /// Sends the cumulative histogram snapshot directly to an OTLP/gRPC
    /// collector. Histograms intentionally bypass the Meter API because OTel
    /// has no observable histogram instrument.
    pub async fn export_histograms(
        &self,
        endpoint: &str,
        resource: &Resource,
        start_time_unix_nano: u64,
        time_unix_nano: u64,
    ) -> Result<()> {
        let request = self.histogram_request(resource, start_time_unix_nano, time_unix_nano)?;
        let mut client = MetricsServiceClient::connect(endpoint.to_owned())
            .await
            .with_context(|| format!("connect OTLP metrics endpoint {endpoint}"))?;
        let response = client.export(request).await.context("export OTLP backend histogram snapshot")?.into_inner();
        if let Some(partial) = response.partial_success
            && (partial.rejected_data_points != 0 || !partial.error_message.is_empty())
        {
            return Err(anyhow!(
                "OTLP collector partially accepted backend histogram snapshot: rejected_data_points={}, error_message={}",
                partial.rejected_data_points,
                partial.error_message
            ));
        }
        Ok(())
    }
}

fn register_observable_scalars(meter: &Meter, snapshot_source: &Arc<SnapshotSource>) {
    for descriptor in backend_metric_descriptors() {
        match descriptor.kind {
            BackendMetricKind::Counter => {
                let source = Arc::clone(snapshot_source);
                meter
                    .u64_observable_counter(descriptor.name)
                    .with_unit(descriptor.unit.as_ucum())
                    .with_callback(move |observer| observe_scalar_samples(&source, descriptor, observer))
                    .build();
            },
            BackendMetricKind::Gauge | BackendMetricKind::Max => {
                let source = Arc::clone(snapshot_source);
                meter
                    .u64_observable_gauge(descriptor.name)
                    .with_unit(descriptor.unit.as_ucum())
                    .with_callback(move |observer| observe_scalar_samples(&source, descriptor, observer))
                    .build();
            },
            BackendMetricKind::Histogram => {},
        }
    }
}

fn observe_scalar_samples(
    snapshot_source: &Arc<SnapshotSource>,
    descriptor: &'static BackendMetricDescriptor,
    observer: &dyn AsyncInstrument<u64>,
) {
    let Ok(samples) = snapshot_source.as_ref()().project_backend_metric(descriptor) else {
        // A canonical descriptor can only fail if a corrupted snapshot violates
        // its core-owned histogram contract. The callback cannot return an
        // error, so it fails closed and records no measurement.
        return;
    };
    for sample in samples {
        let BackendMetricSampleValue::Scalar(value) = sample.value else {
            continue;
        };
        let attributes = otel_attributes(&sample.attributes);
        observer.observe(value, &attributes);
    }
}

/// Builds one direct-OTLP request containing every histogram descriptor from a
/// single backend snapshot.
pub fn histogram_request(
    snapshot: &BackendMetricsSnapshot,
    resource: &Resource,
    start_time_unix_nano: u64,
    time_unix_nano: u64,
) -> Result<ExportMetricsServiceRequest> {
    let mut metrics = Vec::new();
    for descriptor in backend_metric_descriptors() {
        if descriptor.kind != BackendMetricKind::Histogram {
            continue;
        }
        let samples = snapshot
            .project_backend_metric(descriptor)
            .with_context(|| format!("project backend histogram descriptor {}", descriptor.name))?;
        let data_points = samples
            .into_iter()
            .filter_map(|sample| {
                let BackendMetricSampleValue::Histogram(histogram) = sample.value else {
                    return None;
                };
                Some(histogram_data_point(&sample.attributes, &histogram, start_time_unix_nano, time_unix_nano))
            })
            .collect();
        metrics.push(histogram_metric(descriptor, data_points));
    }

    Ok(histogram_export_request(resource, metrics))
}

/// Builds the resource/scope envelope for descriptor-derived histogram
/// metrics. The resource is owned by the embedding application and is copied
/// unchanged into the request.
pub fn histogram_export_request(resource: &Resource, metrics: Vec<Metric>) -> ExportMetricsServiceRequest {
    ExportMetricsServiceRequest {
        resource_metrics: vec![ResourceMetrics {
            resource: Some(resource.clone()),
            scope_metrics: vec![ScopeMetrics {
                scope: Some(InstrumentationScope { name: "query-engine.backend-metrics".into(), ..Default::default() }),
                metrics,
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

/// Builds one cumulative OTLP histogram metric from a canonical descriptor.
///
/// Callers supply only core-projected points; the descriptor continues to own
/// the metric name and UCUM unit.
pub fn histogram_metric(descriptor: &BackendMetricDescriptor, data_points: Vec<HistogramDataPoint>) -> Metric {
    Metric {
        name: descriptor.name.into(),
        unit: descriptor.unit.as_ucum().into(),
        data: Some(metric::Data::Histogram(Histogram {
            data_points,
            aggregation_temporality: AggregationTemporality::Cumulative as i32,
        })),
        ..Default::default()
    }
}

/// Converts one core-owned histogram projection into its exact OTLP point.
///
/// The core has already converted duration units and cumulative bucket counts;
/// this adapter copies that projection unchanged into the protobuf.
pub fn histogram_data_point(
    attributes: &[BackendMetricAttribute],
    histogram: &BackendHistogramProjection,
    start_time_unix_nano: u64,
    time_unix_nano: u64,
) -> HistogramDataPoint {
    HistogramDataPoint {
        attributes: proto_attributes(attributes),
        start_time_unix_nano,
        time_unix_nano,
        count: histogram.count,
        sum: Some(histogram.sum),
        bucket_counts: histogram.bucket_counts.to_vec(),
        explicit_bounds: histogram.explicit_bounds.clone(),
        ..Default::default()
    }
}

fn otel_attributes(attributes: &[BackendMetricAttribute]) -> Vec<KeyValue> {
    attributes
        .iter()
        .map(|attribute| match attribute.value {
            BackendMetricAttributeValue::I64(value) => KeyValue::new(attribute.key, value),
            BackendMetricAttributeValue::Str(value) => KeyValue::new(attribute.key, value),
        })
        .collect()
}

fn proto_attributes(attributes: &[BackendMetricAttribute]) -> Vec<ProtoKeyValue> {
    attributes
        .iter()
        .map(|attribute| {
            let value = match attribute.value {
                BackendMetricAttributeValue::I64(value) => any_value::Value::IntValue(value),
                BackendMetricAttributeValue::Str(value) => any_value::Value::StringValue(value.into()),
            };
            ProtoKeyValue {
                key: attribute.key.into(),
                value: Some(AnyValue { value: Some(value) }),
                ..Default::default()
            }
        })
        .collect()
}
