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

const OPERATION: &str = "nexus_query_engine.exec.bounded.compiled.scheduler";

/// Interprets executable work for a compiled pipeline task.
///
/// The scheduler owns ordering and resource-lifetime invariants. Implementors
/// own the operator-specific work and return the resident resources produced by
/// each task. Scheduler unit tests use mock payloads; production tasks use GPU
/// frames.
#[doc(hidden)]
pub trait CompiledTaskExecutor {
    type Resource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>>;

    fn execute_task_attempt(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
        _attempt_id: &crate::observability::TaskAttemptId,
        _span_id: RuntimeSpanId,
    ) -> Result<CompiledTaskOutput<Self::Resource>> {
        self.execute_task(task, state)
    }

    fn release_resource(&mut self, _resource: ResidentResourceId, _payload: Self::Resource) -> Result<()> {
        Ok(())
    }

    fn cleanup_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        self.release_resource(resource, payload)
    }

    fn cleanup_after_error(&mut self) -> Result<()> {
        Ok(())
    }

    fn before_scheduling_cycle(&mut self) -> Result<CompiledSchedulingCycle> {
        Ok(CompiledSchedulingCycle)
    }

    fn live_reserved_bytes(&self) -> Option<u64> {
        None
    }

    fn remaining_reserved_bytes(&self) -> Option<u64> {
        None
    }

    fn can_admit_task(
        &self,
        _task: &PipelineTask,
        _state: &CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskAdmission> {
        Ok(CompiledTaskAdmission::admitted())
    }
}

/// Scheduler behavior selected by an executor at a scheduling-cycle boundary.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompiledSchedulingCycle;

/// A conservative scheduler-level task admission decision.
///
/// This is not an exact libcudf preflight. It lets an executor report whether
/// the scheduler should start a dependency-ready task under the reservations
/// currently visible to that executor.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTaskAdmission {
    admitted: bool,
    estimated_required_bytes: Option<u64>,
    remaining_bytes: Option<u64>,
    reason: Option<&'static str>,
}

impl CompiledTaskAdmission {
    pub fn admitted() -> Self {
        Self { admitted: true, estimated_required_bytes: None, remaining_bytes: None, reason: None }
    }

    pub fn admitted_with_bytes(estimated_required_bytes: u64, remaining_bytes: u64) -> Self {
        Self {
            admitted: true,
            estimated_required_bytes: Some(estimated_required_bytes),
            remaining_bytes: Some(remaining_bytes),
            reason: None,
        }
    }

    pub fn not_ready(estimated_required_bytes: u64, remaining_bytes: u64, reason: &'static str) -> Self {
        Self {
            admitted: false,
            estimated_required_bytes: Some(estimated_required_bytes),
            remaining_bytes: Some(remaining_bytes),
            reason: Some(reason),
        }
    }

    pub fn is_admitted(&self) -> bool {
        self.admitted
    }

    pub fn estimated_required_bytes(&self) -> Option<u64> {
        self.estimated_required_bytes
    }

    pub fn remaining_bytes(&self) -> Option<u64> {
        self.remaining_bytes
    }

    pub fn reason(&self) -> Option<&'static str> {
        self.reason
    }
}

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
            let task = match self.next_runnable_task(&state, executor, cycle) {
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
            let attempt_id = task_attempt_id(task.id().to_string(), 1);
            let task_span_id = RuntimeSpanId::new((task.id().index() as u64).saturating_add(2));
            let mut task_lifecycle = observation_scope.map(|scope| {
                RuntimeTaskLifecycle::new(
                    scope.observation().clone(),
                    scope.plan_execution_id().clone(),
                    attempt_id.clone(),
                    task.kind().as_str(),
                    task_span_id,
                    scope.plan_span_id(),
                )
            });
            if let Err(error) = state.prepare_task(task) {
                fail_task_lifecycle(&mut task_lifecycle, &error);
                cleanup_preserving_primary(
                    executor,
                    &mut state,
                    observation_scope,
                    CleanupFailurePhase::TaskPreparation,
                    &error,
                );
                return Err(error);
            }
            let output = match catch_unwind(AssertUnwindSafe(|| {
                executor.execute_task_attempt(task, &mut state, &attempt_id, task_span_id)
            })) {
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
            let released = match state.complete_task(task, output.into_produced_resources()) {
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

    fn next_runnable_task<E>(
        &self,
        state: &CompiledPipelineState<E::Resource>,
        executor: &E,
        _cycle: CompiledSchedulingCycle,
    ) -> Result<&'a PipelineTask>
    where
        E: CompiledTaskExecutor,
    {
        let live_reserved_bytes = executor.live_reserved_bytes().unwrap_or(0);
        let ready_tasks = self
            .pipeline
            .tasks()
            .iter()
            .filter(|task| {
                !state.is_task_completed(task.id())
                    && self.dependencies.get(&task.id()).is_none_or(|dependencies| {
                        dependencies.iter().all(|dependency| state.is_task_completed(*dependency))
                    })
            })
            .collect::<Vec<_>>();
        if ready_tasks.is_empty() {
            return Err(invalid("compiled pipeline has no runnable task before completion")
                .with_fact("completed_task_count", state.completed_task_count())
                .with_fact("pipeline_task_count", self.pipeline.tasks().len()));
        }

        let admissions = ready_tasks
            .into_iter()
            .map(|task| executor.can_admit_task(task, state).map(|admission| (task, admission)))
            .collect::<Result<Vec<_>>>()?;
        if let Some((task, _)) = admissions
            .iter()
            .filter(|(_, admission)| admission.is_admitted())
            .min_by_key(|(task, _)| self.runnable_task_priority(task, state, live_reserved_bytes))
        {
            return Ok(*task);
        }
        Err(resource_not_ready_error(&admissions, state, self.pipeline))
    }

    fn runnable_task_priority<T>(
        &self,
        task: &PipelineTask,
        state: &CompiledPipelineState<T>,
        live_reserved_bytes: u64,
    ) -> (u8, u8, u8, usize) {
        let build_priority = task.kind() == PipelineTaskKind::JoinBuild
            && self.prioritized_join_builds.contains(&task.id())
            || (self.has_completed_join_build_waiting_for_probe(state)
                && self.task_feeds_incomplete_prioritized_join_build(task.id(), state));
        let build_rank = if build_priority { 0 } else { 1 };
        let pressure_rank = if live_reserved_bytes > 0 { self.memory_pressure_task_rank(task, state) } else { 1 };
        let operator_rank = if live_reserved_bytes > 0 { operator_class_rank(task.kind()) } else { 0 };
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

fn fail_task_lifecycle(lifecycle: &mut Option<RuntimeTaskLifecycle>, error: &QueryError) {
    if let Some(lifecycle) = lifecycle.take() {
        lifecycle.fail(error.code().as_str(), error.code() == QueryErrorCode::Cancelled);
    }
}

fn cleanup_preserving_primary<E>(
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

fn record_cleanup_failure(
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

fn release_live_resources_on_error<E, F>(
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

fn release_completed_resources<E, F>(
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
        | PipelineTaskKind::JoinMaterialize
        | PipelineTaskKind::FinalSort => 3,
        PipelineTaskKind::SourceRead | PipelineTaskKind::GpuResidentSource | PipelineTaskKind::LiteralBatchSource => 4,
    }
}

fn resource_not_ready_error<T>(
    admissions: &[(&PipelineTask, CompiledTaskAdmission)],
    state: &CompiledPipelineState<T>,
    pipeline: &CompiledPipeline,
) -> QueryError {
    let ready_task_ids = admissions.iter().map(|(task, _)| task.id().to_string()).collect::<Vec<_>>().join(",");
    let ready_task_kinds = admissions.iter().map(|(task, _)| task.kind().as_str()).collect::<Vec<_>>().join(",");
    let max_estimated_required_bytes = admissions
        .iter()
        .filter_map(|(_, admission)| admission.estimated_required_bytes())
        .max()
        .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
    let max_remaining_bytes = admissions
        .iter()
        .filter_map(|(_, admission)| admission.remaining_bytes())
        .max()
        .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
    let reasons = admissions
        .iter()
        .filter_map(|(_, admission)| admission.reason())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(",");

    QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        OPERATION,
        "no dependency-ready compiled task can be admitted with currently available reservations",
    )
    .with_fact("completed_task_count", state.completed_task_count())
    .with_fact("pipeline_task_count", pipeline.tasks().len())
    .with_fact("ready_task_count", admissions.len())
    .with_fact("ready_task_ids", ready_task_ids)
    .with_fact("ready_task_kinds", ready_task_kinds)
    .with_fact("max_estimated_required_bytes", max_estimated_required_bytes)
    .with_fact("max_remaining_bytes", max_remaining_bytes)
    .with_fact("admission_not_ready_reasons", reasons)
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::Arc,
    };

    use super::*;
    use crate::{
        attempt::{AttemptBackendCounters, AttemptOwner, BeginAttemptRequest},
        observability::{
            AttemptAdmissionFact, AttemptCapacityDisposition, AttemptCleanupDisposition, AttemptFailureCode,
            AttemptHealthTransition, AttemptTerminalCommit, PlanExecutionId, RuntimeObservationConfig,
            RuntimeObservationPayload, RuntimeObservationSession, RuntimePlanExecutionLifecycle, RuntimeQueryLifecycle,
            RuntimeTerminalOutcome, TaskLifecycleState,
        },
        pipeline::{
            PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId, ResidentLifetime,
            ResidentResource, SinkBoundary, SinkBoundaryKind,
        },
        runtime::BackendLifecycle,
    };

    struct LifecycleTestExecutor {
        terminal_task: PipelineTaskId,
        panic: bool,
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
                    "nexus_query_engine.exec.bounded.compiled.scheduler.resource_cleanup_probe",
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
                    "nexus_query_engine.exec.bounded.compiled.scheduler.cleanup_probe",
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
        let backend_lifecycle = BackendLifecycle::new();
        let begin = AttemptOwner::begin(
            BeginAttemptRequest {
                observation: RuntimeObservationConfig::bounded(32, 64 * 1024),
                ..BeginAttemptRequest::default()
            },
            backend_lifecycle.try_begin_observation().expect("attempt observation permit"),
            Arc::new(AttemptBackendCounters::default()),
            None,
        );
        let owner = begin.owner;
        let completion = begin.completion;
        let attempt_handle = owner.handle();
        attempt_handle.record_planning_started().expect("planning starts");
        attempt_handle.record_planning_finished().expect("planning finishes");
        attempt_handle.set_disposition(crate::observability::AttemptDisposition::Native).expect("native disposition");
        attempt_handle.record_admission_requested().expect("admission request records");
        attempt_handle
            .record_admitted(AttemptAdmissionFact {
                admission_sequence: 1,
                admission_wait_ns: 0,
                device_ordinal: 0,
                grant_bytes: 1,
            })
            .expect("attempt admission records");

        let binding = owner.admission_binding();
        let plan = RuntimePlanExecutionLifecycle::from_handle(
            binding.observation_handle(),
            PlanExecutionId::new("plan-scheduler-two-cleanup-failures"),
        );
        let scope = plan.execution_scope();
        let mut executor = LifecycleTestExecutor::failing_with_two_cleanup_failures(PipelineTaskId::new(1));

        let error = scheduler.run_observed(&mut executor, &scope).expect_err("primary task failure propagates");
        assert_eq!(error.code(), QueryErrorCode::Internal);
        assert_eq!(executor.released, [ResidentResourceId::new(0)]);
        assert_eq!(executor.cleanup_calls, 1);
        plan.finish(RuntimeTerminalOutcome::from_query_error(&error));
        owner.commit_query_error(&error).expect("borrowed primary failure commits before finalization");

        let (observation_owner, finalizer_slot) = binding.into_parts();
        finalizer_slot
            .take_for_admission()
            .expect("admitted attempt retains one finalizer")
            .finalize_admitted(AttemptTerminalCommit::new(
                AttemptCapacityDisposition::Released,
                None,
                AttemptHealthTransition::HealthyReuse,
            ))
            .expect("post-cleanup finalization succeeds");
        observation_owner.finish_query(Some(RuntimeTerminalOutcome::from_query_error(&error)));

        let report = completion.wait_timeout(std::time::Duration::from_secs(1)).expect("attempt report publishes");
        assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::CompletedWithFailures);
        assert_eq!(report.cleanup_failure_count, 2, "every scheduler cleanup failure is counted");
        assert!(report.cleanup_failure_count_complete);
        let first = report.first_cleanup_failure.as_ref().expect("first cleanup summary is retained");
        assert!(!first.complete(), "a retained first summary is incomplete after the later scheduler cleanup failure");
        assert_eq!(first.failure().code(), AttemptFailureCode::MemoryReservationFailed);

        let snapshot = observation_owner.sealed_snapshot().expect("attempt observation sealed");
        let cleanup_events = snapshot
            .events()
            .iter()
            .filter(|event| matches!(event.payload, RuntimeObservationPayload::CleanupFailure(_)))
            .count();
        assert_eq!(cleanup_events, 2, "timeline remains a projection of both cleanup failures");
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

    fn lifecycle_test_pipeline() -> CompiledPipeline {
        let source = PipelineTaskId::new(0);
        let unary = PipelineTaskId::new(1);
        let frame = ResidentResourceId::new(0);
        CompiledPipeline::new(
            vec![
                PipelineTask::new(source, PipelineTaskKind::SourceRead).with_output_resources(vec![frame]),
                PipelineTask::new(unary, PipelineTaskKind::Unary).with_input_resources(vec![frame]),
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
            vec![ResidentResource::new(
                frame,
                source,
                vec![unary],
                ResidentLifetime::BetweenTasks { producer: source },
            )],
            vec![],
            SinkBoundary::new(SinkBoundaryKind::HostArrowMaterialization, unary),
        )
    }
}
