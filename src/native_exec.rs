//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

mod common;
mod composed;
mod config;
mod host;
mod metrics;
mod producer;

#[cfg(any(feature = "server", feature = "cugraph"))]
pub(crate) use self::metrics::NativeCudfMetrics;
#[cfg(feature = "server")]
pub(crate) use self::metrics::{NativeMetricsCapture, NativeMetricsCaptureSink};
pub(crate) use self::producer::gpu_resident_producer_input_name;
#[cfg(feature = "cugraph")]
pub(crate) use self::producer::{GpuResidentProducerExec, gpu_resident_producer};
pub use self::{
    composed::NativeCudfComposedExec, config::NativeCudfExecutionConfig, host::NativeCudfExec,
    metrics::NATIVE_CUDF_DATAFUSION_METRIC_NAMES, producer::NativeCudfGpuProducerExec,
};
