//! Native execution output assembly.

use arrow::record_batch::RecordBatch;
use query_planning::{
    correlation::{QueryId, TraceId},
    error::QueryError,
};
use query_runtime::{
    metrics::QueryMetrics,
    observability::{MetricsRecorder, SealedRuntimeObservationSnapshot},
    runtime::{ExecutionContext, resident::GpuResultHandle},
};

use super::{next_plan_execution_id, plan_identity};

/// Output produced by native query execution.
///
/// The variant is driven by the query plan's sink. Host Arrow output has
/// crossed the device-to-host boundary; GPU-resident output remains bound to
/// its originating runtime and must be materialized explicitly if a caller
/// needs Arrow.
pub enum QueryOutput {
    /// GPU-resident result bound to its originating runtime/device.
    GpuResident(GpuResultHandle),
    /// Host Arrow result after explicit device-to-host materialization.
    HostArrow(RecordBatch),
}

impl QueryOutput {
    /// Borrow the host Arrow batch when the result was materialized to host.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn as_host_arrow(&self) -> Option<&RecordBatch> {
        match self {
            Self::HostArrow(batch) => Some(batch),
            Self::GpuResident(_) => None,
        }
    }

    /// Borrow the GPU handle when the result stayed device-resident.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn as_gpu_resident(&self) -> Option<&GpuResultHandle> {
        match self {
            Self::GpuResident(frame) => Some(frame),
            Self::HostArrow(_) => None,
        }
    }
}

/// Native query output plus metrics collected during that execution.
///
/// Metrics ownership stays with the result so adapters can record them after
/// deciding whether to pass through a GPU handle or materialize host output.
#[cfg(any(test, feature = "test-hooks"))]
#[doc(hidden)]
pub struct QueryExecutionResult {
    output: QueryOutput,
    metrics: QueryMetrics,
}

/// The metrics of one plan execution, held by its observation owner.
///
/// Work that belongs to the plan but runs after [`super::execute_query_plan`]
/// returns, such as a deferred host sink, records through
/// [`Self::recorder`]. A [`QueryMetrics`] value is a projection taken with
/// [`Self::snapshot`]. Clones share the recorder's observation owner; a fact recorded
/// through any clone appears in every later snapshot.
#[derive(Debug, Clone)]
pub struct PlanMetrics {
    recorder: MetricsRecorder,
    query_id: QueryId,
    trace_id: Option<TraceId>,
}

impl PlanMetrics {
    pub(crate) const fn new(recorder: MetricsRecorder, query_id: QueryId, trace_id: Option<TraceId>) -> Self {
        Self { recorder, query_id, trace_id }
    }

    /// Binds the metrics of work that runs under `context` outside any engine
    /// plan, such as a frontend's host boundary copies. `label` prefixes the
    /// minted plan execution id.
    pub fn bind(context: &ExecutionContext, label: &str) -> Self {
        let (query_id, trace_id) = plan_identity(context);
        let recorder = MetricsRecorder::bind(
            context.runtime_observation(),
            next_plan_execution_id(label),
            query_id.as_str(),
            trace_id.as_ref().map(TraceId::as_str),
        );
        Self::new(recorder, query_id, trace_id)
    }

    /// Where this plan execution's metrics are recorded.
    pub const fn recorder(&self) -> &MetricsRecorder {
        &self.recorder
    }

    /// This plan execution's metrics as recorded so far.
    pub fn snapshot(&self) -> QueryMetrics {
        let mut metrics = self.identity();
        if let Some(checkpoint) = self.recorder.checkpoint() {
            metrics.apply_owner_checkpoint(&checkpoint);
        }
        metrics
    }

    /// Freezes this plan execution's metrics and returns them; facts recorded
    /// afterward are ignored. The sealed observation snapshot carries the
    /// same value under [`SealedRuntimeObservationSnapshot::query_metrics`].
    pub fn freeze(&self) -> QueryMetrics {
        let mut metrics = self.identity();
        if let Some(checkpoint) = self.recorder.freeze() {
            metrics.apply_owner_checkpoint(&checkpoint);
        }
        metrics
    }

    fn identity(&self) -> QueryMetrics {
        QueryMetrics { query_id: self.query_id.clone(), trace_id: self.trace_id.clone(), ..Default::default() }
    }
}

/// Evidence-preserving result of one observed native execution.
///
/// Unlike `Result<QueryExecutionResult, QueryError>`, this outcome retains
/// the plan's metrics and the sealed observation snapshot when execution fails.
pub struct QueryExecutionOutcome {
    result: std::result::Result<QueryOutput, QueryError>,
    metrics: PlanMetrics,
    observation: Option<SealedRuntimeObservationSnapshot>,
}

impl QueryExecutionOutcome {
    pub(crate) fn new(
        result: std::result::Result<QueryOutput, QueryError>,
        metrics: PlanMetrics,
        observation: Option<SealedRuntimeObservationSnapshot>,
    ) -> Self {
        Self { result, metrics, observation }
    }

    /// Borrows the structured output-or-error result.
    #[cfg(test)]
    pub(crate) fn result(&self) -> std::result::Result<&QueryOutput, &QueryError> {
        self.result.as_ref()
    }

    /// The plan's metrics, including records emitted before failure.
    pub const fn metrics(&self) -> &PlanMetrics {
        &self.metrics
    }

    /// Borrows the immutable sealed evidence when recording was enabled.
    #[cfg(test)]
    pub(crate) const fn observation(&self) -> Option<&SealedRuntimeObservationSnapshot> {
        self.observation.as_ref()
    }

    /// Consumes the outcome without discarding failure diagnostics.
    pub fn into_parts(
        self,
    ) -> (std::result::Result<QueryOutput, QueryError>, PlanMetrics, Option<SealedRuntimeObservationSnapshot>) {
        (self.result, self.metrics, self.observation)
    }
}

#[cfg(any(test, feature = "test-hooks"))]
impl QueryExecutionResult {
    /// Create an execution result with its collected metrics.
    pub fn new(output: QueryOutput, metrics: QueryMetrics) -> Self {
        Self { output, metrics }
    }

    /// Borrow the query output without consuming metrics.
    pub fn output(&self) -> &QueryOutput {
        &self.output
    }

    /// Borrow metrics collected during query execution.
    pub fn metrics(&self) -> &QueryMetrics {
        &self.metrics
    }

    /// Consume the result into output and metrics.
    pub fn into_parts(self) -> (QueryOutput, QueryMetrics) {
        (self.output, self.metrics)
    }
}
