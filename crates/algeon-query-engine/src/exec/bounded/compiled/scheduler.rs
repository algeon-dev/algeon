use std::{
    collections::{BTreeMap, BTreeSet},
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
};

use super::state::{CompiledPipelineState, ReleasedCompiledResource};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    observability::{
        CleanupFailurePhase, CleanupFailureSummary, RuntimeCleanupFailureObservation, RuntimeEventContext,
        RuntimeExecutionObservationScope, RuntimeObservationPayload, RuntimeSpanId, RuntimeTaskLifecycle,
        task_attempt_id,
    },
    pipeline::{
        CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId,
    },
};

const OPERATION: &str = "query_engine.exec.bounded.compiled.scheduler";

/// Interprets executable work for a compiled pipeline task.
///
/// The scheduler owns ordering and resource-lifetime invariants. Implementors
/// own the operator-specific work and return the resident resources produced by
/// each task. Scheduler unit tests use mock payloads; production tasks use GPU
/// frames.
///
/// # Resource ownership
///
/// `Resource` is the payload of one resident resource declared by the compiled
/// pipeline, and [`CompiledPipelineState`] owns every live payload. A task
/// reads its inputs through the state; a payload the executor takes out of the
/// state, for example with `consume_task_input_resource`, is the executor's to
/// release, including when the task then fails. Payloads returned in
/// [`CompiledTaskOutput`] move into the state when the task completes. The
/// scheduler passes each payload still in the state to exactly one of
/// [`release_resource`](Self::release_resource) or
/// [`cleanup_resource`](Self::cleanup_resource), or leaves a resource that
/// outlives the run, such as the sink output, in the state returned by a
/// successful run. It never passes the same payload to a hook twice, whether
/// or not the hook succeeded.
///
/// # Error cleanup
///
/// When a scheduling cycle, task selection, task preparation, task execution,
/// task completion, completed-resource release, or final validation fails, or
/// a task panics, the scheduler passes every still-live payload to
/// `cleanup_resource` and then calls
/// [`cleanup_after_error`](Self::cleanup_after_error) once. A cleanup error
/// does not stop the remaining cleanup. The primary error is returned, or the
/// panic resumes unwinding, and observed runs record every cleanup error as a
/// secondary cleanup failure.
#[doc(hidden)]
pub trait CompiledTaskExecutor {
    /// Payload stored for each resident resource.
    type Resource;

    /// Runs one prepared task.
    ///
    /// The returned output carries the payloads of the task's declared output
    /// resources; completing the task rejects duplicate, undeclared, or
    /// already-produced resources. `Err` or a panic starts error cleanup.
    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>>;

    /// Releases a payload whose resource lifetime ended when a task completed.
    ///
    /// The payload is consumed even when this returns `Err`. The first error
    /// fails the run after every other payload released by the same task
    /// completion has also been passed to this hook. The default drops the
    /// payload.
    fn release_resource(&mut self, _resource: ResidentResourceId, _payload: Self::Resource) -> Result<()> {
        Ok(())
    }

    /// Releases a payload that was still live when the run failed. The default
    /// delegates to [`release_resource`](Self::release_resource).
    fn cleanup_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        self.release_resource(resource, payload)
    }

    /// Cleans up executor-owned state once per failed run, after every live
    /// payload has been passed to [`cleanup_resource`](Self::cleanup_resource).
    fn cleanup_after_error(&mut self) -> Result<()> {
        Ok(())
    }

    /// Runs before each runnable-task selection; `Err` fails the run and starts
    /// error cleanup.
    fn before_scheduling_cycle(&mut self) -> Result<CompiledSchedulingCycle> {
        Ok(CompiledSchedulingCycle)
    }
}

/// Scheduler behavior selected by an executor at a scheduling-cycle boundary.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompiledSchedulingCycle;

/// The resident resources produced by one compiled task.
#[doc(hidden)]
pub struct CompiledTaskOutput<T> {
    produced_resources: Vec<(ResidentResourceId, T)>,
}

impl<T> CompiledTaskOutput<T> {
    pub fn new(produced_resources: Vec<(ResidentResourceId, T)>) -> Self {
        Self { produced_resources }
    }

    pub fn empty() -> Self {
        Self { produced_resources: Vec::new() }
    }

    pub fn into_produced_resources(self) -> Vec<(ResidentResourceId, T)> {
        self.produced_resources
    }
}

/// Dispatch visibility of one compiled task from the coordinator's view.
///
/// The coordinator owns this board for the whole run. Serial execution walks
/// the same claim/output-ready/commit protocol one task at a time; execution
/// lanes claim disjoint ready tasks through the same protocol instead of
/// re-dispatching behind the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::exec::bounded::compiled) enum CompiledTaskFlight {
    /// Dependencies are satisfied and no lane has claimed the task.
    Ready,
    /// A lane claimed the task; its previous dispatch has not completed.
    Running,
    /// The lane returned owned outputs that are not committed yet.
    OutputReady,
    /// Outputs were committed into the coordinator's resource state.
    Committed,
    /// The run failed while the task was dispatched; it drains without commit.
    Failed,
}

/// Coordinator-owned dispatch board for one compiled pipeline run.
#[derive(Debug, Default)]
pub(in crate::exec::bounded::compiled) struct CompiledFlightBoard {
    flights: BTreeMap<PipelineTaskId, CompiledTaskFlight>,
}

impl CompiledFlightBoard {
    pub(in crate::exec::bounded::compiled) fn flight(&self, task: PipelineTaskId) -> CompiledTaskFlight {
        self.flights.get(&task).copied().unwrap_or(CompiledTaskFlight::Ready)
    }

    pub(in crate::exec::bounded::compiled) fn in_flight_count(&self) -> usize {
        self.flights
            .values()
            .filter(|flight| matches!(flight, CompiledTaskFlight::Running | CompiledTaskFlight::OutputReady))
            .count()
    }

    fn set(&mut self, task: PipelineTaskId, flight: CompiledTaskFlight) {
        self.flights.insert(task, flight);
    }

    pub(in crate::exec::bounded::compiled) fn drain_dispatched_to_failed(&mut self) {
        for flight in self.flights.values_mut() {
            if matches!(flight, CompiledTaskFlight::Running | CompiledTaskFlight::OutputReady) {
                *flight = CompiledTaskFlight::Failed;
            }
        }
    }
}

/// Deterministic single-threaded scheduler for a validated compiled pipeline.
#[doc(hidden)]
pub struct CompiledPipelineScheduler<'a> {
    pipeline: &'a CompiledPipeline,
    dependencies: BTreeMap<PipelineTaskId, BTreeSet<PipelineTaskId>>,
    prioritized_join_builds: BTreeSet<PipelineTaskId>,
}

impl<'a> CompiledPipelineScheduler<'a> {
    pub fn new(pipeline: &'a CompiledPipeline) -> Result<Self> {
        Self::new_with_prioritized_join_builds(pipeline, BTreeSet::new())
    }

    pub fn new_with_prioritized_join_builds(
        pipeline: &'a CompiledPipeline,
        prioritized_join_builds: BTreeSet<PipelineTaskId>,
    ) -> Result<Self> {
        pipeline.validate()?;
        let dependencies = build_dependencies(pipeline);
        Ok(Self { pipeline, dependencies, prioritized_join_builds })
    }

    /// Claims the next runnable task for one execution lane.
    ///
    /// The claim marks the task [`Running`](CompiledTaskFlight::Running) on
    /// the coordinator board and validates its inputs are produced, so a
    /// second claim before completion fails instead of dispatching the task
    /// twice. The lane runs [`CompiledTaskExecutor::execute_task`] outside
    /// the coordinator state borrow and returns owned outputs to
    /// [`commit_claimed_task`](Self::commit_claimed_task).
    pub(in crate::exec::bounded::compiled) fn claim_next_runnable_task<T>(
        &self,
        state: &CompiledPipelineState<T>,
        flights: &mut CompiledFlightBoard,
        cycle: CompiledSchedulingCycle,
    ) -> Result<&'a PipelineTask> {
        let task = self.next_runnable_task(state, flights, cycle)?;
        self.claim_task(task.id(), state, flights)
    }

    /// Claims one specific task for lane work assignment.
    ///
    /// The claim enforces the same runnable predicate as task selection:
    /// dispatch state, control/barrier dependencies, and produced inputs.
    /// A lane therefore cannot start a task whose control predecessor has not
    /// completed, even when the task declares no input resources.
    pub(in crate::exec::bounded::compiled) fn claim_task<T>(
        &self,
        task: PipelineTaskId,
        state: &CompiledPipelineState<T>,
        flights: &mut CompiledFlightBoard,
    ) -> Result<&'a PipelineTask> {
        match flights.flight(task) {
            CompiledTaskFlight::Running | CompiledTaskFlight::OutputReady => {
                return Err(invalid("compiled task dispatched while its previous dispatch is still in flight")
                    .with_fact("compiled_task_id", task));
            },
            CompiledTaskFlight::Committed => {
                return Err(
                    invalid("compiled task claimed after it was already committed").with_fact("compiled_task_id", task)
                );
            },
            CompiledTaskFlight::Failed => {
                return Err(invalid("compiled task claimed after its run failed").with_fact("compiled_task_id", task));
            },
            CompiledTaskFlight::Ready => {},
        }
        let pipeline: &'a CompiledPipeline = self.pipeline;
        let candidate = pipeline.tasks().iter().find(|candidate| candidate.id() == task).ok_or_else(|| {
            invalid("compiled task claim references an unknown pipeline task").with_fact("compiled_task_id", task)
        })?;
        if !self.dependencies_completed(task, state) {
            let incomplete = self
                .dependencies
                .get(&task)
                .into_iter()
                .flatten()
                .find(|dependency| !state.is_task_completed(**dependency))
                .map(|dependency| dependency.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(invalid("compiled task claimed before its dependencies completed")
                .with_fact("compiled_task_id", task)
                .with_fact("incomplete_dependency_task", incomplete));
        }
        state.prepare_task(candidate)?;
        flights.set(task, CompiledTaskFlight::Running);
        Ok(candidate)
    }

    /// Marks a claimed task's owned outputs ready without committing them.
    pub(in crate::exec::bounded::compiled) fn mark_task_output_ready(
        &self,
        task: PipelineTaskId,
        flights: &mut CompiledFlightBoard,
    ) -> Result<()> {
        if flights.flight(task) != CompiledTaskFlight::Running {
            return Err(
                invalid("compiled task output arrived without a running dispatch").with_fact("compiled_task_id", task)
            );
        }
        flights.set(task, CompiledTaskFlight::OutputReady);
        Ok(())
    }

    /// Commits a claimed task's owned outputs into the coordinator state.
    pub(in crate::exec::bounded::compiled) fn commit_claimed_task<T>(
        &self,
        task: &PipelineTask,
        produced_resources: Vec<(ResidentResourceId, T)>,
        state: &mut CompiledPipelineState<T>,
        flights: &mut CompiledFlightBoard,
    ) -> Result<Vec<ReleasedCompiledResource<T>>> {
        if flights.flight(task.id()) != CompiledTaskFlight::OutputReady {
            return Err(invalid("compiled task commit requires an output-ready dispatch")
                .with_fact("compiled_task_id", task.id()));
        }
        match state.complete_task(task, produced_resources) {
            Ok(released) => {
                flights.set(task.id(), CompiledTaskFlight::Committed);
                Ok(released)
            },
            Err(error) => {
                flights.set(task.id(), CompiledTaskFlight::Failed);
                Err(error)
            },
        }
    }

    pub fn run<E>(&self, executor: &mut E) -> Result<CompiledPipelineState<E::Resource>>
    where E: CompiledTaskExecutor {
        self.run_inner(executor, None)
    }

    /// Runs with explicit query/plan identity and task-attempt spans.
    pub fn run_observed<E>(
        &self,
        executor: &mut E,
        observation_scope: &RuntimeExecutionObservationScope,
    ) -> Result<CompiledPipelineState<E::Resource>>
    where
        E: CompiledTaskExecutor,
    {
        self.run_inner(executor, Some(observation_scope))
    }

    fn run_inner<E>(
        &self,
        executor: &mut E,
        observation_scope: Option<&RuntimeExecutionObservationScope>,
    ) -> Result<CompiledPipelineState<E::Resource>>
    where
        E: CompiledTaskExecutor,
    {
        let mut state = CompiledPipelineState::new(self.pipeline)?;
        let mut flights = CompiledFlightBoard::default();

        while state.completed_task_count() < self.pipeline.tasks().len() {
            let cycle = match executor.before_scheduling_cycle() {
                Ok(cycle) => cycle,
                Err(error) => {
                    cleanup_preserving_primary(
                        executor,
                        &mut state,
                        observation_scope,
                        CleanupFailurePhase::SchedulingCycle,
                        &error,
                    );
                    return Err(error);
                },
            };
            let selected = match self.next_runnable_task(&state, &flights, cycle) {
                Ok(task) => task,
                Err(error) => {
                    cleanup_preserving_primary(
                        executor,
                        &mut state,
                        observation_scope,
                        CleanupFailurePhase::RunnableTaskSelection,
                        &error,
                    );
                    return Err(error);
                },
            };
            let attempt_id = task_attempt_id(selected.id().to_string(), 1);
            let task_span_id = RuntimeSpanId::new((selected.id().index() as u64).saturating_add(2));
            let mut task_lifecycle = observation_scope.map(|scope| {
                RuntimeTaskLifecycle::new(
                    scope.observation().clone(),
                    scope.plan_execution_id().clone(),
                    attempt_id.clone(),
                    selected.kind().as_str(),
                    task_span_id,
                    scope.plan_span_id(),
                )
            });
            let task = match self.claim_task(selected.id(), &state, &mut flights) {
                Ok(task) => task,
                Err(error) => {
                    fail_task_lifecycle(&mut task_lifecycle, &error);
                    cleanup_preserving_primary(
                        executor,
                        &mut state,
                        observation_scope,
                        CleanupFailurePhase::TaskPreparation,
                        &error,
                    );
                    return Err(error);
                },
            };
            let output = match catch_unwind(AssertUnwindSafe(|| executor.execute_task(task, &mut state))) {
                Ok(Ok(output)) => output,
                Ok(Err(error)) => {
                    fail_task_lifecycle(&mut task_lifecycle, &error);
                    cleanup_preserving_primary(
                        executor,
                        &mut state,
                        observation_scope,
                        CleanupFailurePhase::TaskExecution,
                        &error,
                    );
                    return Err(error);
                },
                Err(payload) => {
                    if let Some(lifecycle) = task_lifecycle.take() {
                        lifecycle.panicked();
                    }
                    let _ = release_live_resources_on_error(executor, &mut state, |cleanup_error| {
                        record_cleanup_failure(observation_scope, CleanupFailurePhase::TaskPanic, cleanup_error);
                    });
                    resume_unwind(payload);
                },
            };
            if let Err(error) = self.mark_task_output_ready(task.id(), &mut flights) {
                fail_task_lifecycle(&mut task_lifecycle, &error);
                cleanup_preserving_primary(
                    executor,
                    &mut state,
                    observation_scope,
                    CleanupFailurePhase::TaskCompletion,
                    &error,
                );
                return Err(error);
            }
            let released =
                match self.commit_claimed_task(task, output.into_produced_resources(), &mut state, &mut flights) {
                    Ok(released) => released,
                    Err(error) => {
                        fail_task_lifecycle(&mut task_lifecycle, &error);
                        cleanup_preserving_primary(
                            executor,
                            &mut state,
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
                fail_task_lifecycle(&mut task_lifecycle, &error);
                cleanup_preserving_primary(
                    executor,
                    &mut state,
                    observation_scope,
                    CleanupFailurePhase::CompletedResourceRelease,
                    &error,
                );
                return Err(error);
            }
            if let Some(lifecycle) = task_lifecycle.take() {
                lifecycle.finish();
            }
        }

        if let Err(error) = state.ensure_finished(self.pipeline) {
            cleanup_preserving_primary(
                executor,
                &mut state,
                observation_scope,
                CleanupFailurePhase::FinalValidation,
                &error,
            );
            return Err(error);
        }
        Ok(state)
    }

    /// Control/barrier predecessors of `task` recorded by edges and barriers.
    fn dependencies_completed<T>(&self, task: PipelineTaskId, state: &CompiledPipelineState<T>) -> bool {
        self.dependencies
            .get(&task)
            .is_none_or(|dependencies| dependencies.iter().all(|dependency| state.is_task_completed(*dependency)))
    }

    fn next_runnable_task<T>(
        &self,
        state: &CompiledPipelineState<T>,
        flights: &CompiledFlightBoard,
        _cycle: CompiledSchedulingCycle,
    ) -> Result<&'a PipelineTask> {
        let ready_tasks = self
            .pipeline
            .tasks()
            .iter()
            .filter(|task| {
                matches!(flights.flight(task.id()), CompiledTaskFlight::Ready)
                    && !state.is_task_completed(task.id())
                    && self.dependencies_completed(task.id(), state)
            })
            .collect::<Vec<_>>();
        if ready_tasks.is_empty() {
            return Err(invalid("compiled pipeline has no runnable task before completion")
                .with_fact("completed_task_count", state.completed_task_count())
                .with_fact("pipeline_task_count", self.pipeline.tasks().len()));
        }

        ready_tasks.into_iter().min_by_key(|task| self.runnable_task_priority(task, state)).ok_or_else(|| {
            invalid("compiled pipeline has no runnable task before completion")
                .with_fact("completed_task_count", state.completed_task_count())
                .with_fact("pipeline_task_count", self.pipeline.tasks().len())
        })
    }

    fn runnable_task_priority<T>(&self, task: &PipelineTask, state: &CompiledPipelineState<T>) -> (u8, u8, u8, usize) {
        let build_priority = task.kind() == PipelineTaskKind::JoinBuild
            && self.prioritized_join_builds.contains(&task.id())
            || (self.has_completed_join_build_waiting_for_probe(state)
                && self.task_feeds_incomplete_prioritized_join_build(task.id(), state));
        let build_rank = if build_priority { 0 } else { 1 };
        let has_live_resources = state.live_resource_count() > 0;
        let pressure_rank = if has_live_resources { self.memory_pressure_task_rank(task, state) } else { 1 };
        let operator_rank = if has_live_resources { operator_class_rank(task.kind()) } else { 0 };
        (build_rank, pressure_rank, operator_rank, task.id().index())
    }

    fn memory_pressure_task_rank<T>(&self, task: &PipelineTask, state: &CompiledPipelineState<T>) -> u8 {
        if task.input_resources().iter().any(|resource| state.is_resource_live(*resource)) {
            return 0;
        }
        if task.output_resources().is_empty() {
            return 1;
        }
        match task.kind() {
            PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::LiteralBatchSource => 3,
            PipelineTaskKind::JoinMaterialize
            | PipelineTaskKind::Aggregate
            | PipelineTaskKind::Window
            | PipelineTaskKind::Union
            | PipelineTaskKind::GraphAlgorithm
            | PipelineTaskKind::EdgeNormalize
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit => 2,
            PipelineTaskKind::Unary
            | PipelineTaskKind::JoinBuild
            | PipelineTaskKind::JoinProbe
            | PipelineTaskKind::SinkMaterialize => 1,
        }
    }

    fn has_completed_join_build_waiting_for_probe<T>(&self, state: &CompiledPipelineState<T>) -> bool {
        self.pipeline.edges().iter().any(|edge| {
            edge.kind() == PipelineEdgeKind::Control
                && self.task_by_id(edge.from()).is_some_and(|task| task.kind() == PipelineTaskKind::JoinBuild)
                && self.task_by_id(edge.to()).is_some_and(|task| task.kind() == PipelineTaskKind::JoinProbe)
                && state.is_task_completed(edge.from())
                && !state.is_task_completed(edge.to())
        })
    }

    fn task_feeds_incomplete_prioritized_join_build<T>(
        &self,
        task: PipelineTaskId,
        state: &CompiledPipelineState<T>,
    ) -> bool {
        let mut pending = vec![task];
        let mut visited = BTreeSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            for edge in self.pipeline.edges().iter().filter(|edge| edge.from() == current) {
                match edge.kind() {
                    PipelineEdgeKind::JoinBuildInput => {
                        if self.prioritized_join_builds.contains(&edge.to()) && !state.is_task_completed(edge.to()) {
                            return true;
                        }
                    },
                    PipelineEdgeKind::ResidentFrame
                    | PipelineEdgeKind::SharedResidentProducer
                    | PipelineEdgeKind::BarrierOutput => pending.push(edge.to()),
                    PipelineEdgeKind::Control | PipelineEdgeKind::JoinProbeInput | PipelineEdgeKind::SinkInput => {},
                }
            }
        }
        false
    }

    fn task_by_id(&self, task: PipelineTaskId) -> Option<&PipelineTask> {
        self.pipeline.tasks().iter().find(|candidate| candidate.id() == task)
    }
}

pub(super) fn fail_task_lifecycle(lifecycle: &mut Option<RuntimeTaskLifecycle>, error: &QueryError) {
    if let Some(lifecycle) = lifecycle.take() {
        lifecycle.fail(error.code().as_str(), error.code() == QueryErrorCode::Cancelled);
    }
}

pub(super) fn cleanup_preserving_primary<E>(
    executor: &mut E,
    state: &mut CompiledPipelineState<E::Resource>,
    observation_scope: Option<&RuntimeExecutionObservationScope>,
    phase: CleanupFailurePhase,
    primary: &QueryError,
) where
    E: CompiledTaskExecutor,
{
    // `primary` continues on the returned-error path. Record every secondary
    // cleanup error independently; the lifecycle, rather than the lossy
    // timeline, owns the exact count and first-summary completeness.
    let _ = primary;
    let _ = release_live_resources_on_error(executor, state, |cleanup_error| {
        record_cleanup_failure(observation_scope, phase, cleanup_error);
    });
}

pub(super) fn record_cleanup_failure(
    observation_scope: Option<&RuntimeExecutionObservationScope>,
    phase: CleanupFailurePhase,
    cleanup_error: &QueryError,
) {
    let Some(scope) = observation_scope else {
        return;
    };
    let summary = CleanupFailureSummary::from_native_error(phase, cleanup_error);
    // The lifecycle receives this first. The bounded recorder may drop the
    // timeline row, but it cannot change the report's count or first summary.
    scope.observation().record_attempt_cleanup_failure(summary.clone());
    scope.observation().record_with_context(
        RuntimeEventContext::for_plan(scope.plan_execution_id().clone()).with_span(scope.plan_span_id(), None),
        || RuntimeObservationPayload::CleanupFailure(Box::new(RuntimeCleanupFailureObservation { summary })),
    );
}

pub(super) fn release_live_resources_on_error<E, F>(
    executor: &mut E,
    state: &mut CompiledPipelineState<E::Resource>,
    mut record_failure: F,
) -> Result<()>
where
    E: CompiledTaskExecutor,
    F: FnMut(&QueryError),
{
    let mut first_error = None;
    for resource in state.drain_live_resources() {
        if let Err(error) = executor.cleanup_resource(resource.id(), resource.into_payload()) {
            record_failure(&error);
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    if let Err(error) = executor.cleanup_after_error() {
        record_failure(&error);
        if first_error.is_none() {
            first_error = Some(error);
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(super) fn release_completed_resources<E, F>(
    executor: &mut E,
    resources: Vec<ReleasedCompiledResource<E::Resource>>,
    mut record_failure: F,
) -> Result<()>
where
    E: CompiledTaskExecutor,
    F: FnMut(&QueryError),
{
    let mut first_error = None;
    for resource in resources {
        if let Err(error) = executor.release_resource(resource.id(), resource.into_payload()) {
            record_failure(&error);
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn build_dependencies(pipeline: &CompiledPipeline) -> BTreeMap<PipelineTaskId, BTreeSet<PipelineTaskId>> {
    let mut dependencies = pipeline.tasks().iter().map(|task| (task.id(), BTreeSet::new())).collect::<BTreeMap<_, _>>();

    for edge in pipeline.edges() {
        dependencies.entry(edge.to()).or_default().insert(edge.from());
    }
    for barrier in pipeline.barriers() {
        for downstream in barrier.downstream_tasks() {
            let entry = dependencies.entry(*downstream).or_default();
            for upstream in barrier.upstream_tasks() {
                entry.insert(*upstream);
            }
        }
    }

    dependencies
}

fn operator_class_rank(kind: PipelineTaskKind) -> u8 {
    match kind {
        PipelineTaskKind::SinkMaterialize => 0,
        PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe => 1,
        PipelineTaskKind::Unary | PipelineTaskKind::Window | PipelineTaskKind::Limit => 2,
        PipelineTaskKind::Aggregate
        | PipelineTaskKind::Union
        | PipelineTaskKind::GraphAlgorithm
        | PipelineTaskKind::JoinMaterialize
        | PipelineTaskKind::FinalSort => 3,
        PipelineTaskKind::EdgeNormalize => 2,
        PipelineTaskKind::SourceRead | PipelineTaskKind::GpuResidentSource | PipelineTaskKind::LiteralBatchSource => 4,
    }
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::{
        observability::{
            PlanExecutionId, RuntimeObservationConfig, RuntimeObservationPayload, RuntimeObservationSession,
            RuntimeQueryLifecycle, RuntimeTerminalOutcome, TaskLifecycleState,
        },
        pipeline::{
            PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId, ResidentLifetime,
            ResidentResource, SinkBoundary, SinkBoundaryKind,
        },
    };

    struct LifecycleTestExecutor {
        terminal_task: PipelineTaskId,
        panic: bool,
        cancelled: bool,
        cleanup_failure: bool,
        resource_cleanup_failure: bool,
        released: Vec<ResidentResourceId>,
        cleanup_calls: usize,
    }

    impl LifecycleTestExecutor {
        fn failing(terminal_task: PipelineTaskId) -> Self {
            Self {
                terminal_task,
                panic: false,
                cancelled: false,
                cleanup_failure: false,
                resource_cleanup_failure: false,
                released: Vec::new(),
                cleanup_calls: 0,
            }
        }

        fn cancelling(terminal_task: PipelineTaskId) -> Self {
            Self {
                terminal_task,
                panic: false,
                cancelled: true,
                cleanup_failure: false,
                resource_cleanup_failure: false,
                released: Vec::new(),
                cleanup_calls: 0,
            }
        }

        fn failing_with_cleanup_failure(terminal_task: PipelineTaskId) -> Self {
            Self {
                terminal_task,
                panic: false,
                cancelled: false,
                cleanup_failure: true,
                resource_cleanup_failure: false,
                released: Vec::new(),
                cleanup_calls: 0,
            }
        }

        fn failing_with_two_cleanup_failures(terminal_task: PipelineTaskId) -> Self {
            Self {
                terminal_task,
                panic: false,
                cancelled: false,
                cleanup_failure: true,
                resource_cleanup_failure: true,
                released: Vec::new(),
                cleanup_calls: 0,
            }
        }

        fn panicking(terminal_task: PipelineTaskId) -> Self {
            Self {
                terminal_task,
                panic: true,
                cancelled: false,
                cleanup_failure: false,
                resource_cleanup_failure: false,
                released: Vec::new(),
                cleanup_calls: 0,
            }
        }
    }

    impl CompiledTaskExecutor for LifecycleTestExecutor {
        type Resource = String;

        fn execute_task(
            &mut self,
            task: &PipelineTask,
            _state: &mut CompiledPipelineState<Self::Resource>,
        ) -> Result<CompiledTaskOutput<Self::Resource>> {
            if task.id() == self.terminal_task {
                if self.panic {
                    panic!("scheduler lifecycle unwind probe");
                }
                if self.cancelled {
                    return Err(QueryError::new(
                        QueryErrorCode::Cancelled,
                        OPERATION,
                        "scheduler lifecycle cancellation probe",
                    )
                    .with_fact("primary_marker", "retained"));
                }
                return Err(QueryError::new(QueryErrorCode::Internal, OPERATION, "scheduler lifecycle failure probe")
                    .with_fact("primary_marker", "retained"));
            }
            Ok(CompiledTaskOutput::new(
                task.output_resources().iter().map(|resource| (*resource, resource.to_string())).collect(),
            ))
        }

        fn release_resource(&mut self, resource: ResidentResourceId, _payload: Self::Resource) -> Result<()> {
            self.released.push(resource);
            if self.resource_cleanup_failure {
                return Err(QueryError::new(
                    QueryErrorCode::MemoryReservationFailed,
                    "query_engine.exec.bounded.compiled.scheduler.resource_cleanup_probe",
                    "scheduler resource cleanup failure probe",
                ));
            }
            Ok(())
        }

        fn cleanup_after_error(&mut self) -> Result<()> {
            self.cleanup_calls = self.cleanup_calls.saturating_add(1);
            if self.cleanup_failure {
                return Err(QueryError::new(
                    QueryErrorCode::MemoryReservationFailed,
                    "query_engine.exec.bounded.compiled.scheduler.cleanup_probe",
                    "scheduler cleanup failure probe",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn observed_failure_records_one_terminal_and_cleans_live_resources() {
        let pipeline = lifecycle_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let lifecycle = lifecycle("scheduler-failure");
        let scope = lifecycle.execution_scope();
        let mut executor = LifecycleTestExecutor::failing(PipelineTaskId::new(1));

        let error = scheduler.run_observed(&mut executor, &scope).expect_err("injected task failure propagates");
        assert_eq!(executor.released, [ResidentResourceId::new(0)]);
        assert_eq!(executor.cleanup_calls, 1);
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::from_query_error(&error)).expect("observation seals");
        let states = task_states(&snapshot);
        assert_eq!(
            states,
            [
                ("t0".to_string(), TaskLifecycleState::Ready),
                ("t0".to_string(), TaskLifecycleState::Started),
                ("t0".to_string(), TaskLifecycleState::Finished),
                ("t1".to_string(), TaskLifecycleState::Ready),
                ("t1".to_string(), TaskLifecycleState::Started),
                ("t1".to_string(), TaskLifecycleState::Failed { error_code: "internal".to_string() }),
            ]
        );
    }

    #[test]
    fn observed_cancellation_drains_dispatched_work_and_seals_cancelled_once() {
        let pipeline = lifecycle_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let lifecycle = lifecycle("scheduler-cancellation");
        let scope = lifecycle.execution_scope();
        let mut executor = LifecycleTestExecutor::cancelling(PipelineTaskId::new(1));

        let error = scheduler.run_observed(&mut executor, &scope).expect_err("cancellation stops dispatch");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("primary_marker"), Some("retained"));
        assert_eq!(executor.released, [ResidentResourceId::new(0)]);
        assert_eq!(executor.cleanup_calls, 1, "uncommitted outputs drain exactly once");
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::from_query_error(&error)).expect("report seals once");
        let states = task_states(&snapshot);
        assert_eq!(
            states,
            [
                ("t0".to_string(), TaskLifecycleState::Ready),
                ("t0".to_string(), TaskLifecycleState::Started),
                ("t0".to_string(), TaskLifecycleState::Finished),
                ("t1".to_string(), TaskLifecycleState::Ready),
                ("t1".to_string(), TaskLifecycleState::Started),
                ("t1".to_string(), TaskLifecycleState::Cancelled),
            ]
        );
    }

    #[test]
    fn observed_panic_records_one_panicked_terminal_cleans_and_resumes_unwind() {
        let pipeline = lifecycle_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let lifecycle = lifecycle("scheduler-panic");
        let scope = lifecycle.execution_scope();
        let mut executor = LifecycleTestExecutor::panicking(PipelineTaskId::new(1));

        let panic = catch_unwind(AssertUnwindSafe(|| scheduler.run_observed(&mut executor, &scope)));
        assert!(panic.is_err());
        assert_eq!(executor.released, [ResidentResourceId::new(0)]);
        assert_eq!(executor.cleanup_calls, 1);
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::Panicked).expect("observation seals");
        let t1_terminals = task_states(&snapshot)
            .into_iter()
            .filter(|(task, state)| task == "t1" && task_terminal(state))
            .collect::<Vec<_>>();
        assert_eq!(t1_terminals, [("t1".to_string(), TaskLifecycleState::Panicked)]);
    }

    #[test]
    fn observed_cleanup_failure_is_secondary_to_primary_task_error() {
        let pipeline = lifecycle_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let lifecycle = lifecycle("scheduler-cleanup-failure");
        let scope = lifecycle.execution_scope();
        let mut executor = LifecycleTestExecutor::failing_with_cleanup_failure(PipelineTaskId::new(1));

        let error = scheduler.run_observed(&mut executor, &scope).expect_err("primary task failure propagates");
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(error.kind(), QueryErrorCode::Internal.kind());
        assert_eq!(error.status(), QueryErrorCode::Internal.status());
        assert_eq!(error.operation(), OPERATION);
        assert_eq!(error.fact_value("primary_marker"), Some("retained"));
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::from_query_error(&error)).expect("observation seals");
        let cleanup = snapshot
            .events()
            .iter()
            .find_map(|event| match &event.payload {
                RuntimeObservationPayload::CleanupFailure(observation) => Some(&observation.summary),
                _ => None,
            })
            .expect("secondary cleanup evidence is retained");
        assert_eq!(cleanup.phase(), CleanupFailurePhase::TaskExecution);
        assert_eq!(cleanup.failure().code(), crate::observability::AttemptFailureCode::MemoryReservationFailed);
        assert_eq!(cleanup.failure().status(), crate::observability::AttemptFailureStatus::Temporary);
        assert_eq!(cleanup.failure().operation(), crate::observability::AttemptFailureOperation::Cleanup);
    }

    #[test]
    fn scheduler_routes_all_secondary_cleanup_failures_to_the_attempt_lifecycle() {
        let pipeline = lifecycle_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let lifecycle = lifecycle("scheduler-two-cleanup-failures");
        let scope = lifecycle.execution_scope();
        let mut executor = LifecycleTestExecutor::failing_with_two_cleanup_failures(PipelineTaskId::new(1));

        let error = scheduler.run_observed(&mut executor, &scope).expect_err("primary task failure propagates");
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(executor.released, [ResidentResourceId::new(0)]);
        assert_eq!(executor.cleanup_calls, 1);
        let snapshot = lifecycle.finish(RuntimeTerminalOutcome::from_query_error(&error)).expect("observation seals");
        let cleanup = snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::CleanupFailure(observation) => Some(&observation.summary),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(cleanup.len(), 2, "timeline retains every scheduler cleanup failure");
        assert!(cleanup.iter().all(|summary| summary.phase() == CleanupFailurePhase::TaskExecution));
    }

    fn lifecycle(label: &str) -> RuntimeQueryLifecycle {
        RuntimeQueryLifecycle::new(
            RuntimeObservationSession::new(label, None, RuntimeObservationConfig::bounded(32, 64 * 1024)),
            PlanExecutionId::new(format!("plan-{label}")),
        )
    }

    fn task_states(
        snapshot: &crate::observability::SealedRuntimeObservationSnapshot,
    ) -> Vec<(String, TaskLifecycleState)> {
        snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::TaskLifecycle(observation) => {
                    Some((observation.attempt_id.task_id.as_str().to_string(), observation.lifecycle.clone()))
                },
                _ => None,
            })
            .collect()
    }

    fn task_terminal(state: &TaskLifecycleState) -> bool {
        matches!(
            state,
            TaskLifecycleState::Finished
                | TaskLifecycleState::Failed { .. }
                | TaskLifecycleState::Cancelled
                | TaskLifecycleState::Dropped
                | TaskLifecycleState::Panicked
        )
    }

    #[test]
    fn flight_board_guards_duplicate_dispatch_and_drives_claim_commit() {
        let pipeline = lifecycle_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let mut state = CompiledPipelineState::<String>::new(&pipeline).expect("state builds");
        let mut flights = CompiledFlightBoard::default();

        let first = scheduler
            .claim_next_runnable_task(&state, &mut flights, CompiledSchedulingCycle)
            .expect("first claim selects the source task");
        assert_eq!(first.id(), PipelineTaskId::new(0));
        assert_eq!(flights.flight(first.id()), CompiledTaskFlight::Running);
        assert_eq!(flights.in_flight_count(), 1);

        let duplicate =
            scheduler.claim_task(first.id(), &state, &mut flights).expect_err("duplicate dispatch must fail");
        assert_eq!(duplicate.operation(), OPERATION);

        scheduler
            .claim_next_runnable_task(&state, &mut flights, CompiledSchedulingCycle)
            .expect_err("the in-flight task is not offered to a second lane");

        scheduler
            .commit_claimed_task(first, vec![(ResidentResourceId::new(0), "r0".to_string())], &mut state, &mut flights)
            .expect_err("commit requires an output-ready dispatch");
        assert_eq!(flights.flight(first.id()), CompiledTaskFlight::Running);

        scheduler.mark_task_output_ready(first.id(), &mut flights).expect("lane returns owned outputs");
        assert_eq!(flights.flight(first.id()), CompiledTaskFlight::OutputReady);
        let released = scheduler
            .commit_claimed_task(first, vec![(ResidentResourceId::new(0), "r0".to_string())], &mut state, &mut flights)
            .expect("commit installs owned outputs into coordinator state");
        assert!(released.is_empty(), "between-tasks frame stays resident for its consumer");
        assert_eq!(flights.flight(first.id()), CompiledTaskFlight::Committed);
        assert_eq!(flights.in_flight_count(), 0);

        scheduler.claim_task(first.id(), &state, &mut flights).expect_err("committed tasks are never redispatched");

        let second = scheduler
            .claim_next_runnable_task(&state, &mut flights, CompiledSchedulingCycle)
            .expect("consumer is runnable after its input commits");
        assert_eq!(second.id(), PipelineTaskId::new(1));
        scheduler.mark_task_output_ready(second.id(), &mut flights).expect("consumer outputs ready");
        let released = scheduler
            .commit_claimed_task(second, vec![(ResidentResourceId::new(1), "r1".to_string())], &mut state, &mut flights)
            .expect("consumer commits");
        assert_eq!(released.len(), 2, "consumed frame and task-local output release on completion");
        state.ensure_finished(&pipeline).expect("every logical input computed exactly once");
    }

    #[test]
    fn two_lanes_enter_branch_work_simultaneously_and_commit_each_input_once() {
        use std::sync::{Arc, Barrier, Mutex};

        let pipeline = union_branches_test_pipeline();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("test pipeline validates");
        let coordinator = Arc::new(Mutex::new(LaneCoordinator::new(&scheduler, &pipeline).expect("state builds")));
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let overlap = Arc::new(Mutex::new(false));

        std::thread::scope(|scope| {
            for branch in
                [(PipelineTaskId::new(0), PipelineTaskId::new(1)), (PipelineTaskId::new(2), PipelineTaskId::new(3))]
            {
                let coordinator = Arc::clone(&coordinator);
                let entered = Arc::clone(&entered);
                let release = Arc::clone(&release);
                let overlap = Arc::clone(&overlap);
                scope.spawn(move || {
                    for task in [branch.0, branch.1] {
                        let claimed = coordinator.lock().expect("coordinator lock").claim(task);
                        entered.wait();
                        {
                            let coordinator = coordinator.lock().expect("coordinator lock");
                            let other = if task.index() < 2 {
                                PipelineTaskId::new(task.index() + 2)
                            } else {
                                PipelineTaskId::new(task.index() - 2)
                            };
                            // Neither lane can commit before `release`: observing
                            // the peer Running here proves simultaneous entry.
                            assert_eq!(
                                coordinator.flight(other),
                                CompiledTaskFlight::Running,
                                "both branch lanes stay dispatched across the rendezvous"
                            );
                            *overlap.lock().expect("overlap flag") = true;
                        }
                        release.wait();
                        coordinator
                            .lock()
                            .expect("coordinator lock")
                            .commit(claimed, vec![(branch_output(task), claimed.index().to_string())]);
                    }
                });
            }
        });

        assert!(*overlap.lock().expect("overlap flag"), "both branch lanes were Running at the same barrier point");
        let mut coordinator = coordinator.lock().expect("coordinator lock");
        for task in [0, 1, 2, 3].map(PipelineTaskId::new) {
            assert_eq!(coordinator.commits.get(&task), Some(&1), "each logical input committed exactly once");
        }
        coordinator.commit_union();
    }

    /// Single-owner coordinator behind one mutex: lanes move their claimed
    /// task forward through owned-output commits, never through shared
    /// `&mut` state borrows.
    struct LaneCoordinator<'a> {
        scheduler: &'a CompiledPipelineScheduler<'a>,
        pipeline: &'a CompiledPipeline,
        state: CompiledPipelineState<String>,
        flights: CompiledFlightBoard,
        commits: std::collections::BTreeMap<PipelineTaskId, usize>,
    }

    impl<'a> LaneCoordinator<'a> {
        fn new(scheduler: &'a CompiledPipelineScheduler<'a>, pipeline: &'a CompiledPipeline) -> Result<Self> {
            Ok(Self {
                scheduler,
                pipeline,
                state: CompiledPipelineState::new(pipeline)?,
                flights: CompiledFlightBoard::default(),
                commits: std::collections::BTreeMap::new(),
            })
        }

        fn claim(&mut self, task: PipelineTaskId) -> PipelineTaskId {
            self.scheduler.claim_task(task, &self.state, &mut self.flights).expect("lane claims its branch work").id()
        }

        fn flight(&self, task: PipelineTaskId) -> CompiledTaskFlight {
            self.flights.flight(task)
        }

        fn commit(&mut self, task: PipelineTaskId, produced: Vec<(ResidentResourceId, String)>) {
            let pipeline = self.pipeline;
            let candidate = pipeline.tasks().iter().find(|candidate| candidate.id() == task).expect("known task");
            self.scheduler.mark_task_output_ready(task, &mut self.flights).expect("lane returns owned outputs");
            let released = self
                .scheduler
                .commit_claimed_task(candidate, produced, &mut self.state, &mut self.flights)
                .expect("each lane commits its own work once");
            drop(released);
            *self.commits.entry(task).or_insert(0) += 1;
        }

        fn commit_union(&mut self) {
            let union = self
                .scheduler
                .claim_next_runnable_task(&self.state, &mut self.flights, CompiledSchedulingCycle)
                .expect("union is runnable after both branches commit");
            assert_eq!(union.id(), PipelineTaskId::new(4));
            let union = union.id();
            self.scheduler.mark_task_output_ready(union, &mut self.flights).expect("union outputs ready");
            let pipeline = self.pipeline;
            let candidate = pipeline.tasks().iter().find(|candidate| candidate.id() == union).expect("union task");
            self.scheduler
                .commit_claimed_task(
                    candidate,
                    vec![(ResidentResourceId::new(4), "r4".to_string())],
                    &mut self.state,
                    &mut self.flights,
                )
                .expect("union merges both branches");
            self.state.ensure_finished(pipeline).expect("union run consumed every branch input once");
        }
    }

    fn branch_output(task: PipelineTaskId) -> ResidentResourceId {
        ResidentResourceId::new(task.index())
    }

    fn union_branches_test_pipeline() -> CompiledPipeline {
        let source_a = PipelineTaskId::new(0);
        let unary_a = PipelineTaskId::new(1);
        let source_b = PipelineTaskId::new(2);
        let unary_b = PipelineTaskId::new(3);
        let union = PipelineTaskId::new(4);
        let frame_a = ResidentResourceId::new(0);
        let unary_frame_a = ResidentResourceId::new(1);
        let frame_b = ResidentResourceId::new(2);
        let unary_frame_b = ResidentResourceId::new(3);
        let union_frame = ResidentResourceId::new(4);
        CompiledPipeline::new(
            vec![
                PipelineTask::new(source_a, PipelineTaskKind::SourceRead).with_output_resources(vec![frame_a]),
                PipelineTask::new(unary_a, PipelineTaskKind::Unary)
                    .with_input_resources(vec![frame_a])
                    .with_output_resources(vec![unary_frame_a]),
                PipelineTask::new(source_b, PipelineTaskKind::SourceRead).with_output_resources(vec![frame_b]),
                PipelineTask::new(unary_b, PipelineTaskKind::Unary)
                    .with_input_resources(vec![frame_b])
                    .with_output_resources(vec![unary_frame_b]),
                PipelineTask::new(union, PipelineTaskKind::Union)
                    .with_input_resources(vec![unary_frame_a, unary_frame_b])
                    .with_output_resources(vec![union_frame]),
            ],
            vec![
                PipelineEdge::new(PipelineEdgeId::new(0), source_a, unary_a, PipelineEdgeKind::ResidentFrame)
                    .with_resources(vec![frame_a]),
                PipelineEdge::new(PipelineEdgeId::new(1), source_b, unary_b, PipelineEdgeKind::ResidentFrame)
                    .with_resources(vec![frame_b]),
                PipelineEdge::new(PipelineEdgeId::new(2), unary_a, union, PipelineEdgeKind::ResidentFrame)
                    .with_resources(vec![unary_frame_a]),
                PipelineEdge::new(PipelineEdgeId::new(3), unary_b, union, PipelineEdgeKind::ResidentFrame)
                    .with_resources(vec![unary_frame_b]),
            ],
            vec![
                PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                    .with_upstream_tasks(vec![source_a])
                    .with_downstream_tasks(vec![unary_a]),
                PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SourceChunkBoundary)
                    .with_upstream_tasks(vec![source_b])
                    .with_downstream_tasks(vec![unary_b]),
                PipelineBarrier::new(PipelineBarrierId::new(2), PipelineBarrierKind::SinkBoundary)
                    .with_upstream_tasks(vec![union]),
            ],
            vec![
                ResidentResource::new(
                    frame_a,
                    source_a,
                    vec![unary_a],
                    ResidentLifetime::BetweenTasks { producer: source_a },
                ),
                ResidentResource::new(
                    unary_frame_a,
                    unary_a,
                    vec![union],
                    ResidentLifetime::BetweenTasks { producer: unary_a },
                ),
                ResidentResource::new(
                    frame_b,
                    source_b,
                    vec![unary_b],
                    ResidentLifetime::BetweenTasks { producer: source_b },
                ),
                ResidentResource::new(
                    unary_frame_b,
                    unary_b,
                    vec![union],
                    ResidentLifetime::BetweenTasks { producer: unary_b },
                ),
                ResidentResource::new(union_frame, union, vec![], ResidentLifetime::TaskLocal { task: union }),
            ],
            SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, union),
        )
    }

    #[test]
    fn claim_rejects_control_gated_task_before_its_predecessor_completes() {
        let head = PipelineTaskId::new(0);
        let gated = PipelineTaskId::new(1);
        let pipeline = CompiledPipeline::new(
            vec![
                PipelineTask::new(head, PipelineTaskKind::SourceRead)
                    .with_output_resources(vec![ResidentResourceId::new(0)]),
                PipelineTask::new(gated, PipelineTaskKind::SourceRead)
                    .with_output_resources(vec![ResidentResourceId::new(1)]),
            ],
            vec![PipelineEdge::new(PipelineEdgeId::new(0), head, gated, PipelineEdgeKind::Control)],
            vec![],
            vec![
                ResidentResource::new(
                    ResidentResourceId::new(0),
                    head,
                    vec![],
                    ResidentLifetime::TaskLocal { task: head },
                ),
                ResidentResource::new(
                    ResidentResourceId::new(1),
                    gated,
                    vec![],
                    ResidentLifetime::TaskLocal { task: gated },
                ),
            ],
            SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, gated),
        );
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("control-gated pipeline validates");
        let mut state = CompiledPipelineState::<String>::new(&pipeline).expect("state builds");
        let mut flights = CompiledFlightBoard::default();

        let premature =
            scheduler.claim_task(gated, &state, &mut flights).expect_err("control predecessor is incomplete");
        assert_eq!(premature.operation(), OPERATION);
        assert_eq!(premature.fact_value("incomplete_dependency_task"), Some("t0"));
        assert_eq!(flights.flight(gated), CompiledTaskFlight::Ready, "rejected claim dispatches nothing");

        let claimed = scheduler.claim_task(head, &state, &mut flights).expect("head has no dependencies");
        scheduler.mark_task_output_ready(claimed.id(), &mut flights).expect("outputs ready");
        scheduler
            .commit_claimed_task(
                claimed,
                vec![(ResidentResourceId::new(0), "r0".to_string())],
                &mut state,
                &mut flights,
            )
            .expect("head commits");
        let next = scheduler
            .claim_next_runnable_task(&state, &mut flights, CompiledSchedulingCycle)
            .expect("gate opened after the predecessor committed");
        assert_eq!(next.id(), gated);
    }

    fn lifecycle_test_pipeline() -> CompiledPipeline {
        let source = PipelineTaskId::new(0);
        let unary = PipelineTaskId::new(1);
        let frame = ResidentResourceId::new(0);
        let unary_frame = ResidentResourceId::new(1);
        CompiledPipeline::new(
            vec![
                PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
                PipelineTask::new(unary, PipelineTaskKind::Unary)
                    .with_input_resources(vec![frame])
                    .with_output_resources(vec![unary_frame]),
            ],
            vec![
                PipelineEdge::new(PipelineEdgeId::new(0), source, unary, PipelineEdgeKind::ResidentFrame)
                    .with_resources(vec![frame]),
            ],
            vec![
                PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SourceChunkBoundary)
                    .with_upstream_tasks(vec![source])
                    .with_downstream_tasks(vec![unary]),
                PipelineBarrier::new(PipelineBarrierId::new(1), PipelineBarrierKind::SinkBoundary)
                    .with_upstream_tasks(vec![unary]),
            ],
            vec![
                ResidentResource::new(frame, source, vec![unary], ResidentLifetime::BetweenTasks { producer: source }),
                ResidentResource::new(unary_frame, unary, vec![], ResidentLifetime::TaskLocal { task: unary }),
            ],
            SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, unary),
        )
    }
}
