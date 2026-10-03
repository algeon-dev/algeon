//! DataFusion `ExecutionPlan` wrappers around native cuDF query plans.
//!
//! Host-sink wrappers materialize native results back into Arrow streams for
//! DataFusion. Composed wrappers can pass GPU-resident fragment results between
//! native subplans, but the final DataFusion-facing boundary still streams
//! host Arrow.

use std::{collections::HashSet, sync::Arc};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_plan::{ExecutionPlan, metrics::ExecutionPlanMetricsSet},
};
#[cfg(test)]
use query_engine::{admission::AdmittedPlan, exec::admit_query_plan, plan::QueryPlan};
pub use query_engine::{
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    metrics::{QueryId, TraceId},
    mode::StatisticalAggregateExecutionMode,
    observability::{RuntimeObservationConfig, RuntimeObservationLimits},
    runtime::{LakehouseCacheFillConfig, S3SourceStorageProfile, SourceChunkPlanningMode, SourceSchedulerPolicy},
};

#[cfg(test)]
use crate::error::Result;
use crate::error::{Error, ErrorCode};

/// Admits a plan a test builds for a native node, as planning would.
#[cfg(test)]
pub(crate) fn admit_for_test(
    plan: &QueryPlan,
    execution_config: &AlgeonNativeExecutionConfig,
) -> Result<Arc<AdmittedPlan>> {
    admit_query_plan(plan, execution_config.statistical_aggregate_execution_mode(), execution_config.planning_policy())
        .map_err(|error| {
            Error::from_native("algeon_datafusion.native_exec.test_admission", "test plan is not admissible", error)
        })
}

#[cfg(test)]
mod adapter_tests;
pub(crate) mod admission;
pub(crate) mod child_inputs;
mod common;
mod composed;
mod config;
mod fragment;
pub(crate) mod graph;
pub(crate) mod lakehouse;
mod metrics;
mod native_node;
mod occurrence;
mod producer;
pub(crate) mod relation_function;

pub(crate) use common::{NativeHostOutputCursor, native_host_output_stream, run_gpu_execution_task};

#[cfg(feature = "cugraph")]
#[cfg(feature = "cugraph")]
pub(crate) use self::composed::composed_admission_capability_evidence;
pub(crate) use self::{
    admission::{QueryAdmissionSource, plan_is_query_admission_wrapped},
    composed::producer_bound_capability_evidence,
    fragment::NativeOutput,
    metrics::{
        AlgeonNativeMetrics, GraphClaimCalibrationCapture, NativeExecutionMetricsCapture, SealedAttemptMetricsTarget,
    },
    occurrence::{GpuFunctionHostOccurrenceExec, gpu_function_occurrence_from_task_context},
    producer::{
        GpuResidentProducer, execute_gpu_resident_producer_with_context, gpu_resident_producer,
        gpu_resident_producer_hard_output_row_upper_bound, gpu_resident_producer_input_name,
        gpu_resident_producer_unique_output_key, prepare_gpu_resident_producer_for_execution,
        prepared_gpu_resident_producer_admission_tickets, rewrite_gpu_resident_producer_for_subsumed_composed_consumer,
    },
};
pub use self::{
    config::{AlgeonNativeExecutionConfig, LakehouseByteCache, LakehouseByteCacheFill, SourceChunking},
    fragment::AlgeonNativeExec,
    graph::{GraphBfsSingleflightEventSnapshot, GraphBfsSingleflightSnapshot},
    metrics::{
        GraphClaimCalibrationExecution, GraphClaimCalibrationNonPhysicalReason, NATIVE_ALGEON_DATAFUSION_METRIC_NAMES,
        SERVER_CAPTURE_LOSS_METRIC_NAME, SealedAttemptGraphClaimCalibrationSnapshot,
    },
};

/// The plan below the admission node the native optimizer places at a
/// query's root, or `plan` itself when it has none.
pub fn below_native_admission(plan: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    match native_node::native_node(plan.as_ref()) {
        Some(native_node::NativeNode::Admission(node)) => node.input(),
        _ => plan,
    }
}

/// Return the input of an adapter-owned root node (native admission or a GPU
/// function host occurrence), which adds no rows or GPU work of its own.
pub(crate) fn transparent_root_wrapper_input(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    native_node::native_node(plan).and_then(|node| node.transparent_root_wrapper_input())
}

/// Validates every host-materializing GPU descendant against the outer
/// final-plan occurrence manifest.
pub(crate) fn validate_host_wrapped_gpu_execution(
    plan: &dyn ExecutionPlan,
    task_context: &TaskContext,
) -> DataFusionResult<()> {
    let ticket = admission::invocation_admission_ticket(task_context)
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
    if let Some(exec) = plan.downcast_ref::<AlgeonNativeExec>()
        && exec.materializes_host()
    {
        return Some(exec.sealed_attempt_metrics_target());
    }
    if let Some(exec) = graph::graph_execution_plan(plan) {
        return Some(exec.sealed_attempt_metrics_target());
    }
    if let Some(input) = transparent_root_wrapper_input(plan) {
        return sealed_attempt_metrics_target(input.as_ref());
    }
    let descendant = plan.children().into_iter().find_map(|child| sealed_attempt_metrics_target(child.as_ref()));
    // A producer reached here is either a relation function materializing its
    // own host result, which makes it the native root, or a composed root's
    // GPU input whose captures alone are read. It owns no graph calibration
    // capture, so captures elected below it still reach the sealed snapshot.
    if let Some(metrics) = gpu_resident_producer(plan).and_then(GpuResidentProducer::execution_metrics) {
        let captures = descendant.iter().flat_map(SealedAttemptMetricsTarget::graph_claim_calibration_captures);
        return Some(
            SealedAttemptMetricsTarget::new(metrics, GraphClaimCalibrationCapture::default())
                .with_graph_claim_calibration_captures(captures),
        );
    }
    descendant
}

/// Returns the live metric set of every native execution wrapper in `plan`,
/// including GPU-resident producers, relation functions, and graph nodes below
/// a composed root. Execution clones share these sets and wrappers register
/// their metrics only while executing, so readers take them after execution
/// rather than an `ExecutionPlan::metrics` snapshot.
pub(crate) fn native_execution_metrics(plan: &dyn ExecutionPlan) -> Vec<ExecutionPlanMetricsSet> {
    fn visit(plan: &dyn ExecutionPlan, seen: &mut HashSet<*const ()>, sets: &mut Vec<ExecutionPlanMetricsSet>) {
        // Physical plans can be DAGs; visit each shared node once.
        if !seen.insert(std::ptr::from_ref(plan).cast::<()>()) {
            return;
        }
        sets.extend(native_execution_wrapper_metrics(plan).cloned());
        for child in plan.children() {
            visit(child.as_ref(), seen, sets);
        }
    }

    let mut sets = Vec::new();
    visit(plan, &mut HashSet::new(), &mut sets);
    sets
}

/// Whether `plan` contains a native execution wrapper.
#[cfg(feature = "server")]
pub(crate) fn contains_native_execution_wrapper(plan: &dyn ExecutionPlan) -> bool {
    native_execution_wrapper_metrics(plan).is_some()
        || plan.children().into_iter().any(|child| contains_native_execution_wrapper(child.as_ref()))
}

fn native_execution_wrapper_metrics(plan: &dyn ExecutionPlan) -> Option<&ExecutionPlanMetricsSet> {
    if let Some(exec) = plan.downcast_ref::<AlgeonNativeExec>() {
        return Some(exec.execution_metrics());
    }
    gpu_resident_producer(plan).and_then(GpuResidentProducer::execution_metrics)
}
