//! Resident pipeline chunk work on borrowed attempt lanes.
//!
//! The serial scheduler executes the compiled DAG. Aggregate source chunks
//! and fused top-k candidates may use auxiliary lanes whose leases stay live
//! through the resident output handoff.

use std::sync::Arc;

use query_planning::{
    admission::pipeline_tasks_within_migrated_contract,
    error::Result,
    pipeline::{CompiledPipeline, PipelineTaskKind, SinkBoundaryKind, SortStrategy},
    plan::QueryPlan,
    policy::PlanningPolicy,
};
use query_runtime::{
    observability::RuntimeExecutionObservationScope,
    runtime::{
        AttemptStreamProviderRoute, AttemptWorkLease, ExecutionContext,
        resident::{GpuResidentInputMap, GpuResultHandle},
    },
};

use super::{executor::CompiledGpuTaskExecutor, scheduler::CompiledPipelineScheduler};
use crate::exec::ExecutionMetrics;

mod chunks;
pub(in crate::exec::bounded) use chunks::AggregateLaneDispatch;

const OPERATION: &str = "query_engine.exec.bounded.compiled.lane";

/// The aggregate half of resident lane borrowing: a resident pipeline whose
/// aggregate source chunks may use lanes; the compiled DAG and sink still
/// execute on the root coordinator. Sort borrowing has its own predicate
/// below.
pub(in crate::exec::bounded) fn is_resident_aggregate_lane_pipeline(pipeline: &CompiledPipeline) -> bool {
    pipeline.sink().kind() == SinkBoundaryKind::GpuResidentOutput
        && pipeline_tasks_within_migrated_contract(pipeline)
        && pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::Aggregate)
        && pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::SourceRead)
        && !pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::GpuResidentSource)
}

/// A resident pipeline whose fused top-k may borrow lanes for its local
/// sort candidates; the compiled DAG, the bounded merge, and the sink still
/// execute on the root coordinator, and the sink adopts lane-backed frames
/// onto the root runtime with lane leases held through the handoff.
///
/// Full sorts never borrow lanes: chunk-local sorts cannot replace a global
/// sort, so they keep the pure serial path without the adoption copy.
/// Already-resident inputs are one frame with no chunk expansion behind
/// them, so they stay out for the same reason.
pub(in crate::exec::bounded) fn is_resident_sort_lane_pipeline(pipeline: &CompiledPipeline) -> bool {
    pipeline.sink().kind() == SinkBoundaryKind::GpuResidentOutput
        && pipeline_tasks_within_migrated_contract(pipeline)
        && pipeline.tasks().iter().any(|task| {
            task.kind() == PipelineTaskKind::FinalSort
                && matches!(task.sort_strategy(), Some(SortStrategy::FusedTopK { .. }))
        })
        && !pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::GpuResidentSource)
}

/// Owned lane inputs shared by every lane job of one dispatched run.
///
/// Arcs share the run's immutable execution view; each lane job moves Arc
/// clones into its `'static` pool closure, so no borrow outlives the
/// dispatching frame.
#[doc(hidden)]
pub struct LaneDispatchSupport {
    pub(crate) route: Arc<AttemptStreamProviderRoute>,
    pub requested_lanes: usize,
    pub(crate) context: Arc<ExecutionContext>,
    pub(crate) plan: Arc<QueryPlan>,
    pub(crate) policy: PlanningPolicy,
    pub(crate) pipeline: Arc<CompiledPipeline>,
    pub(crate) gpu_inputs: Arc<GpuResidentInputMap>,
    pub(crate) metrics: Arc<ExecutionMetrics>,
    #[cfg(feature = "test-hooks")]
    aggregate_chunk_completion_gate: Option<AggregateChunkCompletionGate>,
}

impl LaneDispatchSupport {
    /// Builds chunk dispatch support from the attempt's fork capacity.
    pub fn new(
        route: Arc<AttemptStreamProviderRoute>,
        context: ExecutionContext,
        plan: QueryPlan,
        policy: PlanningPolicy,
        pipeline: CompiledPipeline,
        gpu_inputs: GpuResidentInputMap,
        metrics: ExecutionMetrics,
    ) -> Self {
        let requested_lanes = route.lane_capacity();
        Self {
            route,
            requested_lanes,
            context: Arc::new(context),
            plan: Arc::new(plan),
            policy,
            pipeline: Arc::new(pipeline),
            gpu_inputs: Arc::new(gpu_inputs),
            metrics: Arc::new(metrics),
            #[cfg(feature = "test-hooks")]
            aggregate_chunk_completion_gate: None,
        }
    }

    /// Observes a completed but uncommitted aggregate chunk, with its stable
    /// ordinal, lane index, and claimed fork. Width 1 invokes this inline.
    #[cfg(feature = "test-hooks")]
    pub fn with_aggregate_chunk_completion_gate(mut self, gate: AggregateChunkCompletionGate) -> Self {
        self.aggregate_chunk_completion_gate = Some(gate);
        self
    }

    /// Overrides the dispatch width. Tests pin a width to exercise the inline
    /// and pool-worker arms against the same fork pool.
    #[cfg(feature = "test-hooks")]
    pub fn with_requested_lanes(mut self, requested_lanes: usize) -> Self {
        self.requested_lanes = requested_lanes;
        self
    }
}

/// Aggregate chunk ordinal, lane index, and lane-local fork identity.
#[cfg(feature = "test-hooks")]
pub(crate) type AggregateChunkCompletionGate = Arc<dyn Fn(usize, usize, Option<usize>) + Send + Sync>;

/// One execution lane's attempt work lease, held for the whole dispatched run.
///
/// The lease bundles the lane's attempt auxiliary permit, its service stream
/// lease, and the context derived from both, whose route alias carries the
/// lane's single fork. Lane jobs are submitted as auxiliary work backed by
/// this permit for every chunk assigned to that lane.
#[derive(Clone)]
struct LaneWorkLease {
    lease: Arc<AttemptWorkLease>,
}

impl LaneWorkLease {
    /// Takes up to `requested_lanes` leases without ever waiting for one.
    ///
    /// The dispatching root job already holds its own root-class permit, and
    /// other holders of this attempt's auxiliary permits (a second native
    /// island's fragment context among them) may in turn be waiting on work
    /// this coordinator owns. A lane that blocked for a permit would deadlock
    /// the attempt, so lanes take what the envelope has free right now and the
    /// run narrows to that width; whatever stays unclaimed remains reachable
    /// for those other holders.
    ///
    /// Each lease holds one auxiliary fork of the route, whose index the route
    /// picks, so `requested_lanes` leases leave
    /// `lane_capacity() - requested_lanes` forks for the other permit holders
    /// whichever forks those holders already took.
    fn acquire_for_run(context: &ExecutionContext, requested_lanes: usize) -> Result<Vec<Self>> {
        let mut leases = Vec::with_capacity(requested_lanes);
        for lane in 0..requested_lanes {
            let acquired = context.try_acquire_attempt_work_lease().map_err(|error| {
                error.with_fact("execution_lane", lane).with_fact("execution_lane_count", requested_lanes)
            })?;
            match acquired {
                Some(lease) => leases.push(Self { lease: Arc::new(lease) }),
                None => break,
            }
        }
        Ok(leases)
    }
}

/// Runs a resident-sink pipeline on the serial coordinator while aggregate
/// source chunks and fused top-k sort candidates borrow the run's lanes.
/// Only tasks that expand owned lane work use the dispatcher; every other
/// task runs the same interpreter on the coordinator.
///
/// Keep lane leases through the resident sink handoff so any lane-backed
/// columns can be copied to the root runtime while their producer streams
/// are still alive.
#[doc(hidden)]
pub fn run_resident_lane_borrow(
    scheduler: &CompiledPipelineScheduler,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
    support: &LaneDispatchSupport,
) -> Result<(usize, GpuResultHandle)> {
    let requested_lanes = support.requested_lanes.min(support.route.lane_capacity());
    let mut lanes = if requested_lanes > 1 {
        LaneWorkLease::acquire_for_run(&support.context, requested_lanes)?
    } else {
        Vec::new()
    };
    if lanes.len() < 2 {
        lanes.clear();
    }
    let dispatch = AggregateLaneDispatch::new(support, &lanes, observation_scope);
    let mut executor = CompiledGpuTaskExecutor::new(
        &support.context,
        &support.plan,
        support.policy,
        &support.pipeline,
        &support.gpu_inputs,
        &support.metrics,
    )?;
    executor.aggregate_dispatch = Some(dispatch);
    let mut state = match observation_scope {
        Some(scope) => scheduler.run_observed(&mut executor, scope)?,
        None => scheduler.run(&mut executor)?,
    };
    let completed_task_count = state.completed_task_count();
    let output = executor.take_gpu_output(&mut state)?;
    Ok((completed_task_count, output))
}
