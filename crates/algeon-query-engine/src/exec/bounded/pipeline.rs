//! Bounded-pipeline execution boundary (the single capability-first path).

use query_planning::{admission::AdmittedPlan, error::Result};
use query_runtime::{
    observability::RuntimeExecutionObservationScope,
    runtime::{ExecutionContext, resident::GpuResidentInputMap},
};

use super::compiled::execute_compiled_pipeline;
use crate::exec::{ExecutionMetrics, output::QueryOutput};

pub(in crate::exec) fn execute_bounded_pipeline(
    context: &ExecutionContext,
    admitted: &AdmittedPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    execute_compiled_pipeline(
        context,
        admitted.plan(),
        admitted.policy(),
        admitted.pipeline(),
        gpu_inputs,
        metrics,
        observation_scope,
    )
}
