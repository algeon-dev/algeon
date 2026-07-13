//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

pub(crate) mod admission;
mod common;
mod composed;
mod config;
mod fallback;
mod host;
mod metrics;
mod producer;
mod unbounded_native;

#[cfg(feature = "cugraph")]
pub(crate) use common::run_gpu_execution_task;
pub(crate) use config::GraphAdmissionModelAsk;

#[cfg(feature = "cugraph")]
pub(crate) use self::admission::query_admission_ticket_from_task_context;
#[cfg(all(test, feature = "cugraph"))]
pub(crate) use self::admission::task_context_with_query_admission_ticket;
#[cfg(any(feature = "server", feature = "cugraph"))]
pub(crate) use self::metrics::NativeCudfMetrics;
#[cfg(feature = "server")]
pub(crate) use self::metrics::{NativeMetricsCapture, NativeMetricsCaptureSink};
#[cfg(feature = "cugraph")]
pub(crate) use self::producer::{GpuResidentProducerExec, gpu_resident_producer};
#[cfg(feature = "server")]
pub(crate) use self::unbounded_native::UnboundedNativeExecutionLimiter;
pub(crate) use self::{
    admission::{QueryAdmissionSource, plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket},
    fallback::{CpuFallbackStatementLimiter, plan_is_cpu_fallback_limited, wrap_plan_with_cpu_fallback_limit},
    producer::gpu_resident_producer_input_name,
    unbounded_native::{plan_is_unbounded_native_execution_limited, wrap_plan_with_unbounded_native_execution_limit},
};
pub use self::{
    composed::NativeCudfComposedExec, config::NativeCudfExecutionConfig, host::NativeCudfExec,
    metrics::NATIVE_CUDF_DATAFUSION_METRIC_NAMES, producer::NativeCudfGpuProducerExec,
};
