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
pub mod graph;
mod host;
pub(crate) mod lakehouse;
mod metrics;
pub mod producer;
mod unbounded_native;

#[cfg(feature = "cugraph")]
pub(crate) use common::run_gpu_execution_task;

#[cfg(feature = "cugraph")]
pub(crate) use self::admission::query_admission_ticket_from_task_context;
#[cfg(all(test, feature = "cugraph"))]
pub(crate) use self::admission::task_context_with_query_admission_ticket;
#[cfg(all(feature = "server", test))]
pub(crate) use self::metrics::NativeMetricsCaptureSink;
#[cfg(any(feature = "server", feature = "cugraph"))]
pub(crate) use self::metrics::{NativeCudfMetrics, NativeExecutionMetricsCapture};
#[cfg(feature = "server")]
pub(crate) use self::metrics::{
    NativeMetricsCapture, NativeObservationActivityFactory, SERVER_CAPTURE_LOSS_METRIC_NAME,
};
#[cfg(feature = "server")]
pub(crate) use self::unbounded_native::UnboundedNativeExecutionLimiter;
pub(crate) use self::{
    admission::{QueryAdmissionSource, plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket},
    fallback::{CpuFallbackStatementLimiter, plan_is_cpu_fallback_limited, wrap_plan_with_cpu_fallback_limit},
    metrics::NativeRuntimeObservation,
    unbounded_native::{plan_is_unbounded_native_execution_limited, wrap_plan_with_unbounded_native_execution_limit},
};
pub use self::{
    composed::NativeCudfComposedExec, config::NativeCudfExecutionConfig, host::NativeCudfExec,
    metrics::NATIVE_CUDF_DATAFUSION_METRIC_NAMES, producer::NativeCudfGpuProducerExec,
};
