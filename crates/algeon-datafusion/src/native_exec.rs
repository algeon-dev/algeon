//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_plan::ExecutionPlan,
};
pub use query_engine::{
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    metrics::{QueryId, TraceId},
    mode::StatisticalAggregateExecutionMode,
    observability::{RuntimeObservationConfig, RuntimeObservationLimits},
    runtime::{
        LakehouseCacheFillConfig, MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy,
        SourceStorageProfile,
    },
};

use crate::{
    attempt::AlgeonAttemptExtension,
    error::{Error, ErrorCode},
};

#[cfg(test)]
mod adapter_tests;
pub(crate) mod admission;
pub(crate) mod child_inputs;
mod common;
mod composed;
mod config;
pub(crate) mod graph;
mod host;
pub(crate) mod lakehouse;
mod metrics;
mod occurrence;
mod producer;

pub(crate) use common::{NativeHostOutputCursor, native_host_output_stream, run_gpu_execution_task};

#[cfg(feature = "cugraph")]
pub(crate) use self::admission::query_admission_ticket_from_task_context;
#[cfg(feature = "cugraph")]
pub(crate) use self::composed::composed_admission_capability_evidence;
#[cfg(feature = "cugraph")]
pub(crate) use self::metrics::GraphClaimCalibrationCapture;
#[cfg(feature = "server")]
pub(crate) use self::metrics::{NativeMetricsCapture, ServerNativeObservationActivityFactory};
pub(crate) use self::{
    admission::{QueryAdmissionSource, plan_is_query_admission_wrapped, wrap_plan_with_query_admission_ticket},
    metrics::{
        AlgeonNativeMetrics, NativeExecutionMetricsCapture, NativeRuntimeObservation, SealedAttemptMetricsTarget,
    },
    occurrence::{GpuFunctionHostOccurrenceExec, gpu_function_occurrence_from_task_context},
    producer::{
        GpuResidentProducer, execute_gpu_resident_producer_with_context, gpu_resident_producer,
        gpu_resident_producer_hard_output_row_upper_bound, gpu_resident_producer_input_name,
        gpu_resident_producer_unique_output_key, prepare_gpu_resident_producer_for_execution,
        prepared_gpu_resident_producer_admission_tickets, register_gpu_resident_producer_discovery,
        rewrite_gpu_resident_producer_for_subsumed_composed_consumer,
    },
};
pub use self::{
    composed::{AlgeonNativeComposedExec, AlgeonNativeComposedGpuProducerExec},
    config::AlgeonNativeExecutionConfig,
    graph::{GraphBfsSingleflightEventSnapshot, GraphBfsSingleflightSnapshot},
    host::AlgeonNativeExec,
    metrics::{
        GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, NATIVE_ALGEON_DATAFUSION_METRIC_NAMES,
        SERVER_CAPTURE_LOSS_METRIC_NAME, SealedAttemptGraphClaimCalibrationSnapshot,
    },
    producer::AlgeonNativeGpuProducerExec,
};

/// Return the wrapped plan for an adapter-owned transparent root wrapper.
///
/// These wrappers intentionally delegate [`ExecutionPlan::children`] to their
/// inputs, so callers that need to inspect the actual root must not infer the
/// wrapped input from `children()`.
pub(crate) fn transparent_root_wrapper_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    admission::query_admission_wrapped_input(plan).or_else(|| occurrence::gpu_function_occurrence_wrapped_input(plan))
}

/// Validates every host-materializing GPU descendant against the outer
/// final-plan occurrence manifest.
pub(crate) fn validate_host_wrapped_gpu_execution(
    plan: &dyn ExecutionPlan,
    task_context: &TaskContext,
) -> DataFusionResult<()> {
    let ticket = admission::query_admission_ticket_from_task_context(task_context)
        .or_else(|| {
            task_context
                .session_config()
                .options()
                .extensions
                .get::<AlgeonAttemptExtension>()
                .and_then(AlgeonAttemptExtension::bound_ticket)
        })
        .filter(|ticket| ticket.has_execution_manifest())
        .ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.native_exec.host_input_staging",
                    "host staging cannot execute a GPU descendant without a final-plan occurrence manifest",
                )
                .with_log_safe_fact("reason", "host_wrapped_gpu_occurrence_manifest_missing"),
            )
        })?;

    fn validate(plan: &dyn ExecutionPlan, ticket: &Arc<admission::QueryAdmissionTicket>) -> DataFusionResult<bool> {
        if let Some(producer) = gpu_resident_producer(plan) {
            let host_plan = producer.host_result_query_plan()?.ok_or_else(|| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "algeon_datafusion.native_exec.host_input_staging",
                        "a host-wrapped GPU producer does not expose its host execution identity",
                    )
                    .with_log_safe_fact("reason", "host_wrapped_gpu_occurrence_identity_missing")
                    .with_log_safe_fact("occurrence_id", producer.gpu_result_input_name()),
                )
            })?;
            ticket.validate_host_occurrence(producer.gpu_result_input_name(), &host_plan)?.ok_or_else(|| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "algeon_datafusion.native_exec.host_input_staging",
                        "a host-wrapped GPU producer is not covered by the final-plan occurrence manifest",
                    )
                    .with_log_safe_fact("reason", "host_wrapped_gpu_occurrence_not_admitted")
                    .with_log_safe_fact("occurrence_id", producer.gpu_result_input_name()),
                )
            })?;
            return Ok(true);
        }
        let mut found = false;
        for child in plan.children() {
            found |= validate(child.as_ref(), ticket)?;
        }
        Ok(found)
    }

    if !validate(plan, &ticket)? {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native_exec.host_input_staging",
            "a host-wrapped native subtree has no admitted GPU function occurrence",
        )
        .with_log_safe_fact("reason", "host_wrapped_gpu_occurrence_identity_missing")
        .into());
    }
    Ok(())
}

/// Finds the single native execution wrapper closest to an invocation root
/// and returns its post-seal attempt-metric target. Transparent root wrappers
/// are skipped, and mixed DataFusion plans elect their first native subtree in
/// stable child order. The target is deliberately chosen before execution and
/// receives only the already-sealed report.
pub(crate) fn sealed_attempt_metrics_target(plan: &dyn ExecutionPlan) -> Option<SealedAttemptMetricsTarget> {
    if let Some(exec) = plan.downcast_ref::<AlgeonNativeExec>() {
        return Some(exec.sealed_attempt_metrics_target());
    }
    if let Some(exec) = plan.downcast_ref::<AlgeonNativeComposedExec>() {
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
