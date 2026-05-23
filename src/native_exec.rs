//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
pub use nexus_query_engine::{
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    graph_memory::GraphMemoryCalibrationOverride,
    metrics::{QueryId, TraceId},
    mode::StatisticalAggregateExecutionMode,
    observability::{RuntimeObservationConfig, RuntimeObservationLimits},
    runtime::{
        ExecutionBoundaryPolicy, LakehouseCacheFillConfig, MemoryControlPolicy, SourceChunkPlanningMode,
        SourceSchedulerPolicy, SourceStorageProfile,
    },
};

pub(crate) mod admission;
pub(crate) mod child_inputs;
mod common;
mod composed;
mod config;
pub(crate) mod graph;
mod host;
pub(crate) mod lakehouse;
mod metrics;
pub mod producer;

pub(crate) use common::run_gpu_execution_task;

#[cfg(all(test, feature = "cugraph"))]
pub(crate) use self::admission::query_admission_ticket_from_task_context;
#[cfg(feature = "cugraph")]
pub(crate) use self::composed::composed_admission_capability_proof;
#[cfg(feature = "cugraph")]
pub(crate) use self::metrics::GraphClaimCalibrationCapture;
#[cfg(feature = "server-internal")]
#[doc(hidden)]
pub use self::metrics::{
    NativeMetricsCapture as ServerInternalNativeMetricsCapture,
    ServerNativeObservationActivityFactory as ServerInternalNativeObservationActivityFactory,
};
pub(crate) use self::{
    admission::{QueryAdmissionSource, plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket},
    metrics::{
        NativeExecutionMetricsCapture, NativeRuntimeObservation, NexusNativeMetrics, SealedAttemptMetricsTarget,
    },
};
pub use self::{
    composed::{NexusNativeComposedExec, NexusNativeComposedGpuProducerExec},
    config::NexusNativeExecutionConfig,
    graph::{GraphBfsSingleflightEventSnapshot, GraphBfsSingleflightSnapshot},
    host::NexusNativeExec,
    metrics::{
        GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, NATIVE_NEXUS_DATAFUSION_METRIC_NAMES,
        SERVER_CAPTURE_LOSS_METRIC_NAME, SealedAttemptGraphClaimCalibrationSnapshot,
    },
    producer::NexusNativeGpuProducerExec,
};

/// Return the wrapped plan for an adapter-owned transparent root wrapper.
///
/// These wrappers intentionally delegate [`ExecutionPlan::children`] to their
/// inputs, so callers that need to inspect the actual root must not infer the
/// wrapped input from `children()`.
pub(crate) fn transparent_root_wrapper_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    admission::query_admission_wrapped_input(plan)
}

/// Finds the single native execution wrapper closest to an invocation root
/// and returns its post-seal attempt-metric target. Transparent root wrappers
/// are skipped, and mixed DataFusion plans elect their first native subtree in
/// stable child order. The target is deliberately chosen before execution and
/// receives only the already-sealed report.
pub(crate) fn sealed_attempt_metrics_target(plan: &dyn ExecutionPlan) -> Option<SealedAttemptMetricsTarget> {
    if let Some(exec) = plan.downcast_ref::<NexusNativeExec>() {
        return Some(exec.sealed_attempt_metrics_target());
    }
    if let Some(exec) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(exec.sealed_attempt_metrics_target());
    }
    if let Some(exec) = graph::graph_execution_plan(plan) {
        return Some(exec.sealed_attempt_metrics_target());
    }
    if let Some(input) = transparent_root_wrapper_input(plan) {
        return sealed_attempt_metrics_target(input.as_ref());
    }
    plan.children().into_iter().find_map(|child| sealed_attempt_metrics_target(child.as_ref()))
}
