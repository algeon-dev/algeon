//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

pub(crate) mod admission;
pub(crate) mod child_inputs;
mod common;
mod composed;
mod config;
pub mod graph;
mod host;
mod host_execution_qos;
pub(crate) mod lakehouse;
mod metrics;
pub mod producer;
mod unbounded_native;

pub(crate) use common::run_gpu_execution_task;

#[cfg(feature = "cugraph")]
pub(crate) use self::admission::query_admission_ticket_from_task_context;
#[cfg(all(test, feature = "cugraph"))]
pub(crate) use self::admission::task_context_with_query_admission_ticket;
#[cfg(all(feature = "server", test))]
pub(crate) use self::metrics::NativeMetricsCaptureSink;
#[cfg(feature = "server")]
pub(crate) use self::metrics::{
    NativeMetricsCapture, NativeObservationActivityFactory, SERVER_CAPTURE_LOSS_METRIC_NAME,
};
#[cfg(any(feature = "server", test))]
pub(crate) use self::unbounded_native::UnboundedNativeExecutionLimiter;
pub(crate) use self::{
    admission::{QueryAdmissionSource, plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket},
    host_execution_qos::{
        HostExecutionQosLimiter, plan_is_host_execution_qos_limited, wrap_plan_with_host_execution_qos,
    },
    metrics::{NativeExecutionMetricsCapture, NativeRuntimeObservation, NexusNativeMetrics},
    unbounded_native::{plan_is_unbounded_native_execution_limited, wrap_plan_with_unbounded_native_execution_limit},
};
pub use self::{
    composed::{NexusNativeComposedExec, NexusNativeComposedGpuProducerExec},
    config::NexusNativeExecutionConfig,
    host::NexusNativeExec,
    metrics::NATIVE_NEXUS_DATAFUSION_METRIC_NAMES,
    producer::NexusNativeGpuProducerExec,
};

/// Return the wrapped plan for an adapter-owned transparent root wrapper.
///
/// These wrappers intentionally delegate [`ExecutionPlan::children`] to their
/// inputs, so callers that need to inspect the actual root must not infer the
/// wrapped input from `children()`.
pub(crate) fn transparent_root_wrapper_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    host_execution_qos::host_execution_qos_wrapped_input(plan)
        .or_else(|| admission::query_admission_wrapped_input(plan))
        .or_else(|| unbounded_native::unbounded_native_execution_limited_input(plan))
}
