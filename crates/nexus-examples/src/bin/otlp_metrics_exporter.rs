//! Minimal embedding shape for backend-owned metrics.
//!
//! Construct a normal OpenTelemetry SDK meter/provider for the scalar path,
//! then pass that meter and a backend snapshot closure to
//! `BackendMetricsOtlpExporter::new`. On the application's export schedule,
//! call `export_histograms` with its OTLP `Resource`, the same backend
//! generation's cumulative start time, and the collection timestamp. The
//! adapter's module documentation shows why histograms take the direct
//! protobuf path.

fn main() {
    println!("embed nexus_examples::otlp_metrics::BackendMetricsOtlpExporter with your OTLP meter provider");
}
