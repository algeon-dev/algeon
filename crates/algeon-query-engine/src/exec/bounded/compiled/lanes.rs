//! Disjoint execution-lane dispatch for compiled pipelines.
//!
//! The lane dispatcher runs the same claim/output-ready/commit protocol and
//! the same [`CompiledTaskExecutor::execute_task`] interpreter as the serial
//! scheduler. The coordinator claims runnable tasks serially, resolves each
//! lane claim's inputs into an owned scratch state, and runs the claims on
//! execution lanes: inline on the calling thread at width 1, on pool blocking
//! workers at wider widths. Lanes never touch the coordinator state; owned
//! outputs return to the coordinator, which commits them in claim order.
//! An aggregate claim stays on the coordinator and expands its source into
//! bounded chunk work on these same lanes; chunk completion order never
//! changes its stable input or merge order. A final-sort claim with a fused
//! limit stays on the coordinator the same way and expands its independent
//! local top-k candidates onto these lanes; the bounded hierarchical merge
//! stays on the coordinator behind the existing sort retry contract. Full
//! sorts without a limit stay serial: chunk-local sorts cannot replace a
//! global sort.
//!
//! Join build/probe/materialize, graph, and edge-normalize tasks
//! always run inline on the coordinator: they own cross-task native state
//! (reusable hash-table builds, probe finalization, handle affinity) that
//! must not move across lane executors. Their independent input branches
//! still overlap on lanes, and one build serves many probe chunks through
//! the join owner's chunk dispatch. Dynamic-filter payloads produced on the
//! coordinator are snapshot into lane executors so probe-side filters keep
//! their pruning when they run on lanes.
//!
//! The sink boundary task always runs inline on the coordinator: it owns the
//! pipeline's host/GPU boundary handoff, which has no overlap benefit.

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use query_planning::{admission::pipeline_tasks_within_migrated_contract, dynamic_filter::DynamicFilterSet};

use super::{
    executor::CompiledGpuTaskExecutor,
    resource::CompiledGpuResource,
    scheduler::{
        CompiledFlightBoard, CompiledPipelineScheduler, CompiledTaskExecutor, cleanup_preserving_primary,
        fail_task_lifecycle, record_cleanup_failure, release_completed_resources, release_live_resources_on_error,
    },
    state::CompiledPipelineState,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    exec::ExecutionMetrics,
    observability::{
        CleanupFailurePhase, RuntimeExecutionObservationScope, RuntimeSpanId, RuntimeTaskLifecycle, task_attempt_id,
    },
    pipeline::{
        CompiledPipeline, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId, SinkBoundaryKind,
        SortStrategy,
    },
    plan::QueryPlan,
    runtime::{
        AttemptStreamProviderRoute, AttemptWorkLease, ExecutionContext, GpuExecutionJobKind, PoolBlockingWork,
        resident::{GpuResidentInputMap, GpuResultHandle},
    },
};

mod chunks;
pub(in crate::exec::bounded) use chunks::AggregateLaneDispatch;

const OPERATION: &str = "query_engine.exec.bounded.compiled.lane";

/// Whether every task in `pipeline` belongs to the lane family, so the
/// dispatcher may run it. The runnable task set is the planning migrated
/// contract; the sink boundary always runs on the coordinator.
///
/// Execution width is not a property of the pipeline: it comes from the
/// admitted attempt's fork pool through
/// [`AttemptStreamProviderRoute::lane_capacity`].
///
/// GPU-resident sinks never take the lane-dispatched scheduler: they borrow
/// lanes inside aggregate source work and fused top-k sort candidates while
/// the serial coordinator runs the DAG, and the sink adopts lane-backed
/// frames onto the root runtime before those lanes are released.
#[doc(hidden)]
pub fn is_lane_family_pipeline(pipeline: &CompiledPipeline) -> bool {
    pipeline.sink().kind() == SinkBoundaryKind::HostArrowMaterialization
        && pipeline_tasks_within_migrated_contract(pipeline)
}

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

/// How many of `pipeline`'s tasks the dispatcher can ever have in flight at
/// once, from the compiled graph alone.
///
/// A task with a data or control predecessor, or gated behind a barrier, can
/// only be claimed in a round after its producer committed, so it never
/// overlaps that producer. The independent leaves are therefore the widest
/// round the claim protocol can ever fill, and requesting more lanes than
/// that would charge attempt auxiliary permits and service streams for lanes
/// that are guaranteed to stay idle. The sink boundary always runs on the
/// coordinator and is not a lane.
#[doc(hidden)]
pub fn lane_family_concurrent_task_count(pipeline: &CompiledPipeline) -> usize {
    let sink_task = pipeline.sink().input();
    pipeline
        .tasks()
        .iter()
        .filter(|task| {
            task.id() != sink_task
                && !pipeline.edges().iter().any(|edge| edge.to() == task.id())
                && !pipeline.barriers().iter().any(|barrier| barrier.downstream_tasks().contains(&task.id()))
        })
        .count()
}

/// Owned lane inputs shared by every lane job of one dispatched run.
///
/// Arcs share the run's immutable execution view; each lane job moves Arc
/// clones into its `'static` pool closure, so no borrow outlives the
/// dispatching frame.
#[doc(hidden)]
pub struct LaneDispatchSupport {
    pub route: Arc<AttemptStreamProviderRoute>,
    pub requested_lanes: usize,
    pub context: Arc<ExecutionContext>,
    pub plan: Arc<QueryPlan>,
    pub pipeline: Arc<CompiledPipeline>,
    pub gpu_inputs: Arc<GpuResidentInputMap>,
    pub metrics: Arc<ExecutionMetrics>,
    #[cfg(feature = "test-hooks")]
    lane_completion_gate: Option<LaneCompletionGate>,
    #[cfg(feature = "test-hooks")]
    aggregate_chunk_completion_gate: Option<AggregateChunkCompletionGate>,
}

impl LaneDispatchSupport {
    /// Builds dispatch support from the attempt's fork capacity. Materialized
    /// task pipelines are additionally capped by their independent leaves;
    /// aggregate tasks can expand one leaf into several chunk instances, and
    /// fused top-k sorts can expand one frame or stream into several local
    /// candidates.
    pub fn new(
        route: Arc<AttemptStreamProviderRoute>,
        context: ExecutionContext,
        plan: QueryPlan,
        pipeline: CompiledPipeline,
        gpu_inputs: GpuResidentInputMap,
        metrics: ExecutionMetrics,
    ) -> Self {
        // Aggregate tasks expand their deferred source into independently runnable
        // chunks, and fused top-k sorts expand their input into independently
        // runnable local candidates; unlike materialized tasks, their width is
        // not the DAG leaf count. Full sorts never expand, so they keep the
        // leaf cap instead of charging permits for lanes that stay idle.
        let requested_lanes = if pipeline
            .tasks()
            .iter()
            .any(|task| task.kind() == PipelineTaskKind::Aggregate || is_fused_topk_sort(task))
        {
            route.lane_capacity()
        } else {
            route.lane_capacity().min(lane_family_concurrent_task_count(&pipeline))
        };
        Self {
            route,
            requested_lanes,
            context: Arc::new(context),
            plan: Arc::new(plan),
            pipeline: Arc::new(pipeline),
            gpu_inputs: Arc::new(gpu_inputs),
            metrics: Arc::new(metrics),
            #[cfg(feature = "test-hooks")]
            lane_completion_gate: None,
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

    /// Installs a gate the dispatcher runs on each lane worker after its task
    /// has produced its output and before the worker returns it, passing the
    /// lane index and the fork index that lane's alias holds.
    ///
    /// This is the deterministic seam for exercising events that can only
    /// happen while a lane holds a completed but uncommitted output, such as
    /// an attempt cancelled between task completion and the coordinator's
    /// join, and for observing which fork the route served each lane. It is
    /// the caller's job to release the gate.
    #[cfg(feature = "test-hooks")]
    pub fn with_lane_completion_gate(mut self, gate: LaneCompletionGate) -> Self {
        self.lane_completion_gate = Some(gate);
        self
    }
}

/// Observes each lane worker's completion: the lane index and the fork index
/// that lane's alias holds, if it holds one.
#[cfg(feature = "test-hooks")]
pub type LaneCompletionGate = Arc<dyn Fn(usize, Option<usize>) + Send + Sync>;

/// Aggregate chunk ordinal, lane index, and lane-local fork identity.
#[cfg(feature = "test-hooks")]
pub type AggregateChunkCompletionGate = Arc<dyn Fn(usize, usize, Option<usize>) + Send + Sync>;

/// One execution lane's attempt work lease, held for the whole dispatched run.
///
/// The lease bundles the lane's attempt auxiliary permit, its service stream
/// lease, and the context derived from both, whose route alias carries the
/// lane's single fork. Lane jobs are submitted as auxiliary work backed by
/// this permit, so the dispatcher never charges a second one per round.
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
pub(super) fn run_resident_lane_borrow(
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

type PanicPayload = Box<dyn Any + Send + 'static>;

enum LaneJobError {
    Failed(QueryError),
    Panicked(PanicPayload),
}

struct LaneJobCompletion {
    task_id: PipelineTaskId,
    outcome: std::result::Result<Vec<(ResidentResourceId, CompiledGpuResource)>, LaneJobError>,
    secondary_cleanup_errors: Vec<QueryError>,
    /// Dynamic-filter set owned by the lane executor after its task.
    ///
    /// Lanes start from a snapshot of the coordinator set; aggregate group-key
    /// producers extend it. The coordinator merges successful completions'
    /// sets in commit order so probe-side consumers in later rounds keep
    /// their pruning. Failed rounds discard uncommitted sets like outputs.
    dynamic_filters: DynamicFilterSet,
}

struct LaneClaim {
    task: PipelineTask,
    scratch: Option<CompiledPipelineState<CompiledGpuResource>>,
    lifecycle: Option<RuntimeTaskLifecycle>,
}

/// A join/graph/edge-normalize task claimed for inline coordinator execution.
///
/// These tasks own cross-task native state in the coordinator executor
/// (reusable join builds, probe finalization, graph handle affinity) and
/// therefore never receive a lane scratch state. They run on the
/// coordinator's own context and root stream through the same
/// claim/output-ready/commit protocol the lanes use.
struct CoordinatorClaim {
    task: PipelineTask,
    lifecycle: Option<RuntimeTaskLifecycle>,
}

/// Whether a final-sort task can expand local top-k candidates onto lanes.
///
/// Only the fused top-k strategy fans out; a full sort runs whole-input on
/// whichever executor claims it, so it never justifies coordinator-inline
/// scheduling or a leaf-count exemption by itself.
fn is_fused_topk_sort(task: &PipelineTask) -> bool {
    task.kind() == PipelineTaskKind::FinalSort && matches!(task.sort_strategy(), Some(SortStrategy::FusedTopK { .. }))
}

/// Whether a task must run inline on the coordinator instead of a lane.
///
/// Join build/probe/materialize own the reusable build artifact and its
/// finalization across tasks; graph, vector, and edge-normalize retain their
/// runtime stream and memory-resource bindings on the coordinator until their
/// component owners prove safe overlap.
/// All other lane-family task kinds are lane-safe: they own only private
/// inputs, workspace, and outputs.
fn is_coordinator_only_task_kind(kind: PipelineTaskKind) -> bool {
    matches!(
        kind,
        PipelineTaskKind::JoinBuild
            | PipelineTaskKind::JoinProbe
            | PipelineTaskKind::JoinMaterialize
            | PipelineTaskKind::GraphAlgorithm
            | PipelineTaskKind::VectorAlgorithm
            | PipelineTaskKind::MlAlgorithm
            | PipelineTaskKind::EdgeNormalize
    )
}

impl LaneClaim {
    fn take_scratch(&mut self) -> CompiledPipelineState<CompiledGpuResource> {
        self.scratch.take().expect("lane claim scratch is taken exactly once by its lane job")
    }
}

struct LaneRound {
    claims: Vec<LaneClaim>,
    coordinator: Vec<CoordinatorClaim>,
    sink: Option<(PipelineTask, Option<RuntimeTaskLifecycle>)>,
}

/// Runs the lane dispatcher at the width this attempt's auxiliary envelope
/// has free.
#[doc(hidden)]
pub fn run_lane_dispatched(
    scheduler: &CompiledPipelineScheduler,
    executor: &mut CompiledGpuTaskExecutor,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
    support: &LaneDispatchSupport,
) -> Result<CompiledPipelineState<CompiledGpuResource>> {
    let requested_lanes = support.requested_lanes.min(support.route.lane_capacity());
    // Width 1 runs the lane protocol inline on the coordinator's own context
    // and root stream, so it needs no lease; wider widths hold one attempt
    // work lease per lane for the whole run. Fewer than two obtainable leases
    // fall back to that inline arm rather than waiting for a permit.
    let mut lane_leases = if requested_lanes > 1 {
        LaneWorkLease::acquire_for_run(&support.context, requested_lanes)?
    } else {
        Vec::new()
    };
    let width = if lane_leases.len() > 1 { lane_leases.len() } else { 1 };
    if width == 1 {
        // A single obtained lease is unusable for the inline arm: release its
        // attempt auxiliary permit, service stream, and fork claim before the
        // inline run starts instead of holding them for the whole dispatch.
        lane_leases.clear();
    }
    // Share the run's lanes with coordinator-owned join probe work through
    // an owned dispatcher: probe chunks reuse these same leases (and fork
    // subsets) via `execute_work`, so the run never acquires a second lane
    // set. Lane-produced frames stay valid because every lease underpinning
    // them is held for the whole run, on these leases and the dispatcher's
    // clones alike.
    executor.aggregate_dispatch = Some(AggregateLaneDispatch::new(support, &lane_leases, observation_scope));
    let mut state = CompiledPipelineState::new(&support.pipeline)?;
    let mut flights = CompiledFlightBoard::default();

    while state.completed_task_count() < support.pipeline.tasks().len() {
        // Rounds are barrier-synchronized: every claimed task joins before the
        // next round starts, so nothing is in flight at this boundary and a
        // cancellation fails without a drain.
        debug_assert_eq!(flights.in_flight_count(), 0);
        if let Err(error) = support.context.check_cancellation(OPERATION) {
            cleanup_preserving_primary(
                executor,
                &mut state,
                observation_scope,
                CleanupFailurePhase::SchedulingCycle,
                &error,
            );
            return Err(error);
        }
        if let Err(error) = executor.before_scheduling_cycle() {
            cleanup_preserving_primary(
                executor,
                &mut state,
                observation_scope,
                CleanupFailurePhase::SchedulingCycle,
                &error,
            );
            return Err(error);
        }
        let mut round = claim_lane_round(
            scheduler,
            executor,
            &support.pipeline,
            width,
            &mut state,
            &mut flights,
            observation_scope,
        )?;
        if round.claims.is_empty() && round.coordinator.is_empty() && round.sink.is_none() {
            // Claim failures inside `claim_lane_round` already cleaned up and
            // returned; reaching here means the loop broke with nothing to
            // run, which the serial scheduler reports as a stuck pipeline.
            let error = invalid("compiled lane dispatch found no runnable task before completion")
                .with_fact("completed_task_count", state.completed_task_count())
                .with_fact("pipeline_task_count", support.pipeline.tasks().len());
            cleanup_preserving_primary(
                executor,
                &mut state,
                observation_scope,
                CleanupFailurePhase::RunnableTaskSelection,
                &error,
            );
            return Err(error);
        }
        if let Some((task, lifecycle)) = round.sink.take() {
            run_coordinator_sink(
                scheduler,
                executor,
                &task,
                lifecycle,
                &mut round.claims,
                &mut state,
                &mut flights,
                observation_scope,
            )?;
        }
        if !round.coordinator.is_empty() {
            // Coordinator join/graph tasks own reusable native state and run
            // inline on the coordinator executor, whose dispatcher above
            // lends them this run's lanes for probe-chunk work. Lane leases
            // stay held across the call so lane-produced frames underpinning
            // coordinator inputs are never detached mid-run.
            run_coordinator_tasks(
                scheduler,
                executor,
                round.coordinator,
                &mut round.claims,
                &mut state,
                &mut flights,
                observation_scope,
            )?;
        }
        if !round.claims.is_empty() {
            let completions = run_lane_claims(
                executor,
                support,
                width,
                &lane_leases,
                &mut round.claims,
                &mut state,
                &mut flights,
                observation_scope,
            )?;
            // Every lane worker of this round has been drained. Cancellation
            // raised while they ran is only observable now, and a terminal
            // coordinator must never commit a late completion: release the
            // whole round through accounting and fail instead.
            if let Err(error) = support.context.check_cancellation(OPERATION) {
                fail_lane_round(
                    executor,
                    &mut flights,
                    round.claims,
                    completions,
                    observation_scope,
                    CleanupFailurePhase::TaskExecution,
                );
                cleanup_preserving_primary(
                    executor,
                    &mut state,
                    observation_scope,
                    CleanupFailurePhase::TaskExecution,
                    &error,
                );
                return Err(error);
            }
            commit_lane_completions(
                scheduler,
                executor,
                round.claims,
                completions,
                &mut state,
                &mut flights,
                observation_scope,
            )?;
        }
    }

    if let Err(error) = state.ensure_finished(&support.pipeline) {
        cleanup_preserving_primary(
            executor,
            &mut state,
            observation_scope,
            CleanupFailurePhase::FinalValidation,
            &error,
        );
        return Err(error);
    }
    // Release the run's lane leases promptly instead of holding them into
    // the caller's sink handoff; error paths above leave the field set and
    // release through the executor drop instead.
    executor.aggregate_dispatch = None;
    Ok(state)
}

/// Claims up to the assignment width in runnable tasks, serially.
///
/// Each claim resolves its inputs into an owned scratch state before the next
/// claim runs, so concurrently executing lanes never share coordinator state.
/// Dependency gating falls out of the claim protocol: a task whose producer
/// was claimed but not committed this round is not completed, hence not
/// runnable, hence never claimed twice.
fn claim_lane_round(
    scheduler: &CompiledPipelineScheduler,
    executor: &mut CompiledGpuTaskExecutor,
    pipeline: &CompiledPipeline,
    width: usize,
    state: &mut CompiledPipelineState<CompiledGpuResource>,
    flights: &mut CompiledFlightBoard,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<LaneRound> {
    let mut round = LaneRound { claims: Vec::new(), coordinator: Vec::new(), sink: None };
    // Coordinator-only tasks (join trio, graph, edge-normalize) do
    // not consume lane leases: they run inline on the coordinator executor
    // that owns their cross-task native state. Lane claims stop at `width`;
    // coordinator claims join the same round without widening it, so
    // independent input branches still fill every lane while the join owner
    // stays serial. Control edges, barriers, and `ResourceSharing` gate both
    // lists through the same claim protocol, and the existing
    // build-priority/dynamic-filter ordering decides whether a probe-side
    // branch waits for its build instead of reading eagerly.
    while round.claims.len() < width {
        let selected = match scheduler.claim_next_runnable_task(state, flights) {
            Ok(task) => task.clone(),
            Err(error) => {
                if round.claims.is_empty() && round.coordinator.is_empty() && round.sink.is_none() {
                    // Nothing is runnable and nothing was claimed: the serial
                    // scheduler reports this as a stuck pipeline.
                    flights.drain_dispatched_to_failed();
                    cleanup_preserving_primary(
                        executor,
                        state,
                        observation_scope,
                        CleanupFailurePhase::RunnableTaskSelection,
                        &error,
                    );
                    return Err(error);
                }
                // Partial batch: the unclaimed tasks depend on this round's
                // outputs and become runnable next round.
                break;
            },
        };
        let mut lifecycle = observation_scope.map(|scope| lane_task_lifecycle(scope, &selected));
        if selected.kind() == PipelineTaskKind::SinkMaterialize {
            round.sink = Some((selected, lifecycle));
            continue;
        }
        if is_coordinator_only_task_kind(selected.kind()) {
            round.coordinator.push(CoordinatorClaim { task: selected, lifecycle });
            continue;
        }
        let scratch = match build_lane_scratch(pipeline, state, &selected) {
            Ok(scratch) => scratch,
            Err(error) => {
                fail_task_lifecycle(&mut lifecycle, &error);
                flights.drain_dispatched_to_failed();
                cleanup_lane_scratches(
                    executor,
                    &mut round.claims,
                    observation_scope,
                    CleanupFailurePhase::TaskPreparation,
                );
                cleanup_preserving_primary(
                    executor,
                    state,
                    observation_scope,
                    CleanupFailurePhase::TaskPreparation,
                    &error,
                );
                return Err(error);
            },
        };
        round.claims.push(LaneClaim { task: selected, scratch: Some(scratch), lifecycle });
    }
    Ok(round)
}

/// Resolves one claimed task's inputs into an owned scratch state.
///
/// Exclusive inputs move out of the coordinator state; shared inputs enter
/// the scratch as snapshots, mirroring the serial shared-input arms. Lazy
/// shared streams and deferred sources cannot cross the lane boundary and
/// fail here with the same invalid-plan class the serial path reports.
fn build_lane_scratch(
    pipeline: &CompiledPipeline,
    state: &mut CompiledPipelineState<CompiledGpuResource>,
    task: &PipelineTask,
) -> Result<CompiledPipelineState<CompiledGpuResource>> {
    let mut scratch = CompiledPipelineState::new(pipeline)?;
    for resource in task.input_resources() {
        let payload = if state.can_consume_task_input_resource(task, *resource) {
            state.consume_task_input_resource(task, *resource)?
        } else {
            state.resource_payload(*resource)?.snapshot_for_lane_claim()?
        };
        scratch.install_lane_claim_payload(*resource, payload)?;
    }
    Ok(scratch)
}

/// Runs one claimed task on its execution lane.
///
/// The runner owns a fresh task executor and the claim's scratch state. The
/// `Drop` guard best-effort drains scratch payloads through release
/// accounting when the task panics, mirroring the serial panic path where
/// still-live payloads reach `cleanup_resource`; the normal path drains
/// explicitly so cleanup errors stay observable.
struct LaneTaskRunner<'a> {
    executor: CompiledGpuTaskExecutor<'a>,
    scratch: Option<CompiledPipelineState<CompiledGpuResource>>,
}

impl LaneTaskRunner<'_> {
    fn run_lane_task(mut self, lane: usize, task: &PipelineTask) -> LaneJobCompletion {
        let task_id = task.id();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let scratch = self.scratch.as_mut().expect("lane scratch state is present while the task runs");
            let output = self
                .executor
                .execute_task(task, scratch)
                .map_err(|error| LaneJobError::Failed(error.with_fact("execution_lane", lane)))?;
            let secondary_cleanup_errors = self.drain_scratch();
            Ok((output, secondary_cleanup_errors))
        }));
        let dynamic_filters = self.executor.dynamic_filters.clone();
        match outcome {
            Ok(Ok((output, secondary_cleanup_errors))) => {
                LaneJobCompletion { task_id, outcome: Ok(output), secondary_cleanup_errors, dynamic_filters }
            },
            Ok(Err(error)) => {
                let secondary_cleanup_errors = self.drain_scratch();
                LaneJobCompletion { task_id, outcome: Err(error), secondary_cleanup_errors, dynamic_filters }
            },
            Err(payload) => LaneJobCompletion {
                task_id,
                outcome: Err(LaneJobError::Panicked(payload)),
                secondary_cleanup_errors: Vec::new(),
                dynamic_filters,
            },
        }
    }

    fn drain_scratch(&mut self) -> Vec<QueryError> {
        let mut failures = Vec::new();
        if let Some(mut scratch) = self.scratch.take() {
            for live in scratch.drain_live_resources() {
                if let Err(error) = self.executor.cleanup_resource(live.id(), live.into_payload()) {
                    failures.push(error);
                }
            }
        }
        failures
    }
}

impl Drop for LaneTaskRunner<'_> {
    fn drop(&mut self) {
        if self.scratch.is_none() {
            return;
        }
        // Panic path: the explicit drain never ran. Release accounting cannot
        // report errors mid-unwind, so this is best-effort and never panics.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _ = self.drain_scratch();
        }));
    }
}

/// Executes the round's claims: inline at width 1, on pool blocking workers
/// otherwise. Each pool lane runs on its own attempt work lease: the lease's
/// context and stream, and the auxiliary permit that already backs the lane
/// for the whole run. The coordinator holds no pool permit while it joins, so
/// it can never starve the workers it waits for. Workers always run to
/// completion and join before any commit, so a late completion can never
/// resurrect after the round.
#[expect(
    clippy::too_many_arguments,
    reason = "lane execution threads executor, support, width, leases, claims, state, flights, and scope"
)]
fn run_lane_claims(
    executor: &mut CompiledGpuTaskExecutor,
    support: &LaneDispatchSupport,
    width: usize,
    lane_leases: &[LaneWorkLease],
    claims: &mut [LaneClaim],
    state: &mut CompiledPipelineState<CompiledGpuResource>,
    flights: &mut CompiledFlightBoard,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<Vec<LaneJobCompletion>> {
    debug_assert!(claims.len() <= width);
    // Coordinator-produced dynamic-filter payloads (join builds, aggregate
    // group keys) must reach lane filters in the same run. Lane executors
    // start empty, so snapshot the coordinator set after its tasks
    // committed; lane work never produces filters the coordinator needs in
    // the same round, keeping the handoff one-way.
    let dynamic_filter_snapshot = executor.dynamic_filters.clone();
    if width <= 1
        || claims
            .iter()
            .any(|claim| claim.task.kind() == PipelineTaskKind::Aggregate || is_fused_topk_sort(&claim.task))
    {
        let dispatch = AggregateLaneDispatch::new(support, lane_leases, observation_scope);
        return Ok(claims
            .iter_mut()
            .map(|claim| {
                // A symbolic aggregate coordinates its chunk instances, and a
                // fused top-k coordinates its local candidates. Neither can
                // occupy a pool worker while waiting for those same workers.
                // Full sorts coordinate nothing across lanes, so they stay on
                // pool workers with the round's other claims. Inline
                // round-mates run on the root context while the run holds its
                // lane permits, so they must never block on an attempt work
                // lease; source prefetch stays behind its own policy for the
                // same reason.
                run_lane_claim_inline(support, claim, Some(&dispatch), &dynamic_filter_snapshot)
            })
            .collect());
    }
    debug_assert_eq!(lane_leases.len(), width, "a wider run holds one work lease per lane");
    let pool = match support.context.gpu_executor() {
        Ok(pool) => pool,
        Err(error) => {
            let error = error.with_fact("execution_lane_count", width).with_fact("reason", "lane_pool_unavailable");
            flights.drain_dispatched_to_failed();
            cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskExecution);
            cleanup_preserving_primary(executor, state, observation_scope, CleanupFailurePhase::TaskExecution, &error);
            return Err(error);
        },
    };
    let mut submitted: Vec<(PipelineTaskId, PoolBlockingWork<LaneJobCompletion>)> = Vec::with_capacity(claims.len());
    for (lane, claim) in claims.iter_mut().enumerate() {
        let task = claim.task.clone();
        let task_id = task.id();
        let scratch = claim.take_scratch();
        let lane_lease = Arc::clone(&lane_leases[lane].lease);
        let plan = Arc::clone(&support.plan);
        let pipeline = Arc::clone(&support.pipeline);
        let gpu_inputs = Arc::clone(&support.gpu_inputs);
        let metrics = Arc::clone(&support.metrics);
        let dynamic_filter_snapshot = dynamic_filter_snapshot.clone();
        #[cfg(feature = "test-hooks")]
        let lane_completion_gate = support.lane_completion_gate.clone();
        let observation = support.context.runtime_observation().clone();
        match pool.submit_blocking(GpuExecutionJobKind::AttemptWork, observation, move || {
            let context = lane_lease.context();
            match CompiledGpuTaskExecutor::new(context, &plan, &pipeline, &gpu_inputs, &metrics) {
                Ok(mut lane_executor) => {
                    lane_executor.dynamic_filters = dynamic_filter_snapshot;
                    let completion =
                        LaneTaskRunner { executor: lane_executor, scratch: Some(scratch) }.run_lane_task(lane, &task);
                    #[cfg(feature = "test-hooks")]
                    if let Some(gate) = lane_completion_gate {
                        gate(lane, context.attempt_claimed_fork_index());
                    }
                    completion
                },
                Err(error) => LaneJobCompletion {
                    task_id,
                    outcome: Err(LaneJobError::Failed(
                        error.with_fact("execution_lane", lane).with_fact("reason", "lane_executor_unavailable"),
                    )),
                    secondary_cleanup_errors: Vec::new(),
                    dynamic_filters: DynamicFilterSet::new(),
                },
            }
        }) {
            Ok(handle) => submitted.push((task_id, handle)),
            Err(error) => {
                let error = error.with_fact("execution_lane", lane).with_fact("compiled_task_id", task_id);
                // The submitted workers run to completion and join here; their
                // outputs were never committed, so each one drains through
                // release accounting before the run fails.
                for (_, handle) in submitted {
                    match handle.join() {
                        Ok(completion) => cleanup_lane_completion_outputs(executor, completion, observation_scope),
                        Err(join_error) => {
                            record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskExecution, &join_error)
                        },
                    }
                }
                flights.drain_dispatched_to_failed();
                cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskExecution);
                cleanup_preserving_primary(
                    executor,
                    state,
                    observation_scope,
                    CleanupFailurePhase::TaskExecution,
                    &error,
                );
                return Err(error);
            },
        }
    }
    Ok(submitted
        .into_iter()
        .map(|(task_id, handle)| match handle.join() {
            Ok(completion) => {
                debug_assert_eq!(completion.task_id, task_id);
                completion
            },
            Err(error) => LaneJobCompletion {
                task_id,
                outcome: Err(LaneJobError::Failed(error)),
                secondary_cleanup_errors: Vec::new(),
                dynamic_filters: DynamicFilterSet::new(),
            },
        })
        .collect())
}

/// Runs one claim on the coordinator's own context and root stream. Width 1
/// runs on the route's reserved root fork, so it needs no work lease.
fn run_lane_claim_inline(
    support: &LaneDispatchSupport,
    claim: &mut LaneClaim,
    dispatch: Option<&AggregateLaneDispatch>,
    dynamic_filter_snapshot: &DynamicFilterSet,
) -> LaneJobCompletion {
    const LANE: usize = 0;
    let task_id = claim.task.id();
    match CompiledGpuTaskExecutor::new(
        &support.context,
        &support.plan,
        &support.pipeline,
        &support.gpu_inputs,
        &support.metrics,
    ) {
        Ok(mut lane_executor) => {
            lane_executor.aggregate_dispatch = dispatch.cloned();
            lane_executor.dynamic_filters = dynamic_filter_snapshot.clone();
            let completion = LaneTaskRunner { executor: lane_executor, scratch: Some(claim.take_scratch()) }
                .run_lane_task(LANE, &claim.task);
            #[cfg(feature = "test-hooks")]
            if let Some(gate) = &support.lane_completion_gate {
                gate(LANE, support.context.attempt_claimed_fork_index());
            }
            completion
        },
        Err(error) => LaneJobCompletion {
            task_id,
            outcome: Err(LaneJobError::Failed(
                error.with_fact("execution_lane", LANE).with_fact("reason", "lane_executor_unavailable"),
            )),
            secondary_cleanup_errors: Vec::new(),
            dynamic_filters: DynamicFilterSet::new(),
        },
    }
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}

/// Commits one round's completions in claim order.
///
/// A failed completion stops the commit: remaining claims keep their scratch
/// states and remaining completions keep their outputs, and everything
/// uncommitted drains through release accounting before the primary error (or
/// the panic payload) leaves. Successful lanes of a failed round are never
/// committed, so exactly-once commit holds by construction.
fn commit_lane_completions(
    scheduler: &CompiledPipelineScheduler,
    executor: &mut CompiledGpuTaskExecutor,
    claims: Vec<LaneClaim>,
    completions: Vec<LaneJobCompletion>,
    state: &mut CompiledPipelineState<CompiledGpuResource>,
    flights: &mut CompiledFlightBoard,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<()> {
    debug_assert_eq!(claims.len(), completions.len());
    let mut claims = claims.into_iter();
    let mut completions = completions.into_iter();
    while let (Some(mut claim), Some(completion)) = (claims.next(), completions.next()) {
        let LaneJobCompletion { task_id, outcome, secondary_cleanup_errors, dynamic_filters } = completion;
        debug_assert_eq!(claim.task.id(), task_id);
        let mut lifecycle = claim.lifecycle.take();
        for secondary in &secondary_cleanup_errors {
            record_cleanup_failure(observation_scope, CleanupFailurePhase::CompletedResourceRelease, secondary);
        }
        let secondary_failure = secondary_cleanup_errors.into_iter().next();
        match outcome {
            Ok(output) => {
                if let Err(error) = scheduler.mark_task_output_ready(claim.task.id(), flights) {
                    fail_task_lifecycle(&mut lifecycle, &error);
                    fail_lane_round(
                        executor,
                        flights,
                        claims.collect(),
                        completions.collect(),
                        observation_scope,
                        CleanupFailurePhase::TaskCompletion,
                    );
                    cleanup_preserving_primary(
                        executor,
                        state,
                        observation_scope,
                        CleanupFailurePhase::TaskCompletion,
                        &error,
                    );
                    return Err(error);
                }
                let released = match scheduler.commit_claimed_task(&claim.task, output, state, flights) {
                    Ok(released) => released,
                    Err(error) => {
                        fail_task_lifecycle(&mut lifecycle, &error);
                        fail_lane_round(
                            executor,
                            flights,
                            claims.collect(),
                            completions.collect(),
                            observation_scope,
                            CleanupFailurePhase::TaskCompletion,
                        );
                        cleanup_preserving_primary(
                            executor,
                            state,
                            observation_scope,
                            CleanupFailurePhase::TaskCompletion,
                            &error,
                        );
                        return Err(error);
                    },
                };
                // Stage lane-produced filters only after the lane's outputs
                // commit: a commit failure leaves no merged filters, and
                // successful lanes of a failed round never reach this line
                // because the first failure returns before their commit.
                executor.merge_lane_dynamic_filters(dynamic_filters);
                if let Err(error) = release_completed_resources(executor, released, |cleanup_error| {
                    record_cleanup_failure(
                        observation_scope,
                        CleanupFailurePhase::CompletedResourceRelease,
                        cleanup_error,
                    );
                }) {
                    fail_task_lifecycle(&mut lifecycle, &error);
                    fail_lane_round(
                        executor,
                        flights,
                        claims.collect(),
                        completions.collect(),
                        observation_scope,
                        CleanupFailurePhase::CompletedResourceRelease,
                    );
                    cleanup_preserving_primary(
                        executor,
                        state,
                        observation_scope,
                        CleanupFailurePhase::CompletedResourceRelease,
                        &error,
                    );
                    return Err(error);
                }
                // Scratch-snapshot cleanup ran inside the lane before commit;
                // its first failure now fails the run post-commit, mirroring
                // the serial release-after-commit contract.
                if let Some(error) = secondary_failure {
                    fail_task_lifecycle(&mut lifecycle, &error);
                    fail_lane_round(
                        executor,
                        flights,
                        claims.collect(),
                        completions.collect(),
                        observation_scope,
                        CleanupFailurePhase::CompletedResourceRelease,
                    );
                    cleanup_preserving_primary(
                        executor,
                        state,
                        observation_scope,
                        CleanupFailurePhase::CompletedResourceRelease,
                        &error,
                    );
                    return Err(error);
                }
                if let Some(lifecycle) = lifecycle.take() {
                    lifecycle.finish();
                }
            },
            Err(LaneJobError::Failed(error)) => {
                fail_task_lifecycle(&mut lifecycle, &error);
                fail_lane_round(
                    executor,
                    flights,
                    claims.collect(),
                    completions.collect(),
                    observation_scope,
                    CleanupFailurePhase::TaskExecution,
                );
                cleanup_preserving_primary(
                    executor,
                    state,
                    observation_scope,
                    CleanupFailurePhase::TaskExecution,
                    &error,
                );
                return Err(error);
            },
            Err(LaneJobError::Panicked(payload)) => {
                if let Some(lifecycle) = lifecycle.take() {
                    lifecycle.panicked();
                }
                fail_lane_round(
                    executor,
                    flights,
                    claims.collect(),
                    completions.collect(),
                    observation_scope,
                    CleanupFailurePhase::TaskPanic,
                );
                let _ = release_live_resources_on_error(executor, state, |cleanup_error| {
                    record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskPanic, cleanup_error);
                });
                std::panic::resume_unwind(payload);
            },
        }
    }
    Ok(())
}

/// Drains uncommitted claims and completions through release accounting, then
/// marks every dispatched flight failed. The coordinator state itself is left
/// for the caller's `cleanup_preserving_primary`.
fn fail_lane_round(
    executor: &mut CompiledGpuTaskExecutor,
    flights: &mut CompiledFlightBoard,
    claims: Vec<LaneClaim>,
    completions: Vec<LaneJobCompletion>,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
    phase: CleanupFailurePhase,
) {
    flights.drain_dispatched_to_failed();
    for completion in completions {
        for secondary in &completion.secondary_cleanup_errors {
            record_cleanup_failure(observation_scope, phase, secondary);
        }
        if let Ok(output) = completion.outcome {
            for (id, payload) in output {
                if let Err(error) = executor.cleanup_resource(id, payload) {
                    record_cleanup_failure(observation_scope, phase, &error);
                }
            }
        }
    }
    let mut claims = claims;
    cleanup_lane_scratches(executor, &mut claims, observation_scope, phase);
}

/// Releases every scratch payload still held by uncommitted claims.
fn cleanup_lane_scratches(
    executor: &mut CompiledGpuTaskExecutor,
    claims: &mut [LaneClaim],
    observation_scope: Option<&RuntimeExecutionObservationScope>,
    phase: CleanupFailurePhase,
) {
    for claim in claims {
        if let Some(mut scratch) = claim.scratch.take() {
            for live in scratch.drain_live_resources() {
                if let Err(error) = executor.cleanup_resource(live.id(), live.into_payload()) {
                    record_cleanup_failure(observation_scope, phase, &error);
                }
            }
        }
    }
}

/// Releases one never-committed lane completion's outputs.
fn cleanup_lane_completion_outputs(
    executor: &mut CompiledGpuTaskExecutor,
    completion: LaneJobCompletion,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) {
    for secondary in &completion.secondary_cleanup_errors {
        record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskExecution, secondary);
    }
    if let Ok(output) = completion.outcome {
        for (id, payload) in output {
            if let Err(error) = executor.cleanup_resource(id, payload) {
                record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskExecution, &error);
            }
        }
    }
}

/// Runs the sink boundary task inline on the coordinator through the same
/// claim/output-ready/commit protocol the lanes use.
///
/// `claims` carries the round's already-extracted lane scratches: a sink
/// failure must release them too, since their inputs already left the
/// coordinator state and no lane will ever run.
#[expect(
    clippy::too_many_arguments,
    reason = "sink failure must release the round's lane scratches beside scheduler, executor, and state"
)]
fn run_coordinator_sink(
    scheduler: &CompiledPipelineScheduler,
    executor: &mut CompiledGpuTaskExecutor,
    task: &PipelineTask,
    lifecycle: Option<RuntimeTaskLifecycle>,
    claims: &mut [LaneClaim],
    state: &mut CompiledPipelineState<CompiledGpuResource>,
    flights: &mut CompiledFlightBoard,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<()> {
    let mut lifecycle = lifecycle;
    let output = match catch_unwind(AssertUnwindSafe(|| executor.execute_task(task, state))) {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            fail_task_lifecycle(&mut lifecycle, &error);
            flights.drain_dispatched_to_failed();
            cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskExecution);
            cleanup_preserving_primary(executor, state, observation_scope, CleanupFailurePhase::TaskExecution, &error);
            return Err(error);
        },
        Err(payload) => {
            if let Some(lifecycle) = lifecycle.take() {
                lifecycle.panicked();
            }
            flights.drain_dispatched_to_failed();
            cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskPanic);
            let _ = release_live_resources_on_error(executor, state, |cleanup_error| {
                record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskPanic, cleanup_error);
            });
            std::panic::resume_unwind(payload);
        },
    };
    if let Err(error) = scheduler.mark_task_output_ready(task.id(), flights) {
        fail_task_lifecycle(&mut lifecycle, &error);
        flights.drain_dispatched_to_failed();
        cleanup_preserving_primary(executor, state, observation_scope, CleanupFailurePhase::TaskCompletion, &error);
        return Err(error);
    }
    let released = match scheduler.commit_claimed_task(task, output, state, flights) {
        Ok(released) => released,
        Err(error) => {
            fail_task_lifecycle(&mut lifecycle, &error);
            flights.drain_dispatched_to_failed();
            cleanup_preserving_primary(executor, state, observation_scope, CleanupFailurePhase::TaskCompletion, &error);
            return Err(error);
        },
    };
    if let Err(error) = release_completed_resources(executor, released, |cleanup_error| {
        record_cleanup_failure(observation_scope, CleanupFailurePhase::CompletedResourceRelease, cleanup_error);
    }) {
        fail_task_lifecycle(&mut lifecycle, &error);
        cleanup_preserving_primary(
            executor,
            state,
            observation_scope,
            CleanupFailurePhase::CompletedResourceRelease,
            &error,
        );
        return Err(error);
    }
    if let Some(lifecycle) = lifecycle.take() {
        lifecycle.finish();
    }
    Ok(())
}

/// Runs the round's coordinator-only tasks inline on the coordinator.
///
/// Join build/probe/materialize, graph, and edge-normalize tasks
/// own cross-task native state in the coordinator executor, so they run
/// here instead of on lane workers. The round's lane scratches were already
/// extracted before the first coordinator task runs; a coordinator failure
/// therefore releases them too, since their inputs already left the
/// coordinator state and no lane will ever run.
fn run_coordinator_tasks(
    scheduler: &CompiledPipelineScheduler,
    executor: &mut CompiledGpuTaskExecutor,
    coordinator: Vec<CoordinatorClaim>,
    claims: &mut [LaneClaim],
    state: &mut CompiledPipelineState<CompiledGpuResource>,
    flights: &mut CompiledFlightBoard,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
) -> Result<()> {
    for CoordinatorClaim { task, lifecycle } in coordinator {
        let mut lifecycle = lifecycle;
        let output = match catch_unwind(AssertUnwindSafe(|| executor.execute_task(&task, state))) {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                fail_task_lifecycle(&mut lifecycle, &error);
                flights.drain_dispatched_to_failed();
                cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskExecution);
                cleanup_preserving_primary(
                    executor,
                    state,
                    observation_scope,
                    CleanupFailurePhase::TaskExecution,
                    &error,
                );
                return Err(error);
            },
            Err(payload) => {
                if let Some(lifecycle) = lifecycle.take() {
                    lifecycle.panicked();
                }
                flights.drain_dispatched_to_failed();
                cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskPanic);
                let _ = release_live_resources_on_error(executor, state, |cleanup_error| {
                    record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskPanic, cleanup_error);
                });
                std::panic::resume_unwind(payload);
            },
        };
        if let Err(error) = scheduler.mark_task_output_ready(task.id(), flights) {
            fail_task_lifecycle(&mut lifecycle, &error);
            flights.drain_dispatched_to_failed();
            cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskCompletion);
            cleanup_preserving_primary(executor, state, observation_scope, CleanupFailurePhase::TaskCompletion, &error);
            return Err(error);
        }
        let released = match scheduler.commit_claimed_task(&task, output, state, flights) {
            Ok(released) => released,
            Err(error) => {
                fail_task_lifecycle(&mut lifecycle, &error);
                flights.drain_dispatched_to_failed();
                cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::TaskCompletion);
                cleanup_preserving_primary(
                    executor,
                    state,
                    observation_scope,
                    CleanupFailurePhase::TaskCompletion,
                    &error,
                );
                return Err(error);
            },
        };
        if let Err(error) = release_completed_resources(executor, released, |cleanup_error| {
            record_cleanup_failure(observation_scope, CleanupFailurePhase::CompletedResourceRelease, cleanup_error);
        }) {
            fail_task_lifecycle(&mut lifecycle, &error);
            flights.drain_dispatched_to_failed();
            cleanup_lane_scratches(executor, claims, observation_scope, CleanupFailurePhase::CompletedResourceRelease);
            cleanup_preserving_primary(
                executor,
                state,
                observation_scope,
                CleanupFailurePhase::CompletedResourceRelease,
                &error,
            );
            return Err(error);
        }
        if let Some(lifecycle) = lifecycle.take() {
            lifecycle.finish();
        }
    }
    Ok(())
}

fn lane_task_lifecycle(scope: &RuntimeExecutionObservationScope, task: &PipelineTask) -> RuntimeTaskLifecycle {
    RuntimeTaskLifecycle::new(
        scope.observation().clone(),
        scope.plan_execution_id().clone(),
        task_attempt_id(task.id().to_string(), 1),
        task.kind().as_str(),
        RuntimeSpanId::new((task.id().index() as u64).saturating_add(2)),
        scope.plan_span_id(),
    )
}

#[cfg(test)]
mod lane_width_tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Int32Array, RecordBatch},
        datatypes::{DataType, Field, Schema},
    };
    use query_planning::admission::bounded_pipeline_decision;

    use super::lane_family_concurrent_task_count;
    use crate::{
        expr::{Expr, LimitSpec},
        mode::StatisticalAggregateExecutionMode,
        pipeline::CompiledPipeline,
        plan::{NodeId, ProjectionOutput, QueryPlan, QueryPlanBuilder},
        rapids_backend::rapids_analysis_context,
        source::{LiteralBatchSource, SourceNode},
    };

    fn admitted_pipeline(plan: &QueryPlan) -> CompiledPipeline {
        let decision =
            bounded_pipeline_decision(plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
                .expect("bounded admission should not error");
        decision.admission().expect("these plans use the compiled admission").pipeline().clone()
    }

    fn literal_branch(builder: &mut QueryPlanBuilder, schema: &Arc<Schema>, values: Vec<i32>) -> NodeId {
        let batch = RecordBatch::try_new(Arc::clone(schema), vec![Arc::new(Int32Array::from(values)) as ArrayRef])
            .expect("literal batch");
        let source =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(schema), vec![batch])));
        builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false))],
        )
    }

    /// Width is the narrower of the fork pool and what the graph can fill. A
    /// pipeline whose every task descends from one source can never have two
    /// tasks in flight, so requesting a second lane would charge an attempt
    /// auxiliary permit and a service stream for a lane guaranteed to idle.
    #[test]
    fn a_single_source_pipeline_requests_one_lane_while_a_union_requests_one_per_branch() {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));

        let mut builder = QueryPlanBuilder::new();
        let single = literal_branch(&mut builder, &schema, vec![1, 2, 3]);
        let limited = builder.add_limit(single, LimitSpec::fetch(10), Arc::clone(&schema));
        let single_source_plan = builder.finish_with_host_sink(limited);
        assert_eq!(
            lane_family_concurrent_task_count(&admitted_pipeline(&single_source_plan)),
            1,
            "a single-source pipeline never overlaps, so it must not lease a second lane"
        );

        let mut builder = QueryPlanBuilder::new();
        let left = literal_branch(&mut builder, &schema, vec![1, 2]);
        let right = literal_branch(&mut builder, &schema, vec![3]);
        let union = builder.add_union(vec![left, right], Arc::clone(&schema));
        let union_plan = builder.finish_with_host_sink(union);
        assert_eq!(
            lane_family_concurrent_task_count(&admitted_pipeline(&union_plan)),
            2,
            "two independent branches can overlap, so both lanes earn their lease"
        );
    }

    #[test]
    fn resident_lane_borrow_predicates_distinguish_resident_aggregates_and_sorts() {
        use super::{is_resident_aggregate_lane_pipeline, is_resident_sort_lane_pipeline};

        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let mut builder = QueryPlanBuilder::new();
        let single = literal_branch(&mut builder, &schema, vec![1, 2, 3]);
        let resident_plan = builder.finish_gpu_resident(single);
        let resident_pipe = admitted_pipeline(&resident_plan);
        assert!(!super::is_lane_family_pipeline(&resident_pipe));
        assert!(!is_resident_aggregate_lane_pipeline(&resident_pipe));
        assert!(!is_resident_sort_lane_pipeline(&resident_pipe));
    }
}
