//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

#![deny(missing_docs)]

mod common;
mod composed;
mod config;
mod host;
mod metrics;
mod producer;

pub(crate) use self::producer::gpu_resident_producer_input_name;
pub use self::{
    composed::NativeCudfComposedExec, config::NativeCudfExecutionConfig, host::NativeCudfExec,
    metrics::NATIVE_CUDF_DATAFUSION_METRIC_NAMES, producer::NativeCudfGpuProducerExec,
};
#[cfg(feature = "cugraph")]
pub(crate) use self::{
    metrics::NativeCudfMetrics,
    producer::{GpuResidentProducerExec, gpu_resident_producer},
};
