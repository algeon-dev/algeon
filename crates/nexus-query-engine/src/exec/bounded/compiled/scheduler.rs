use std::collections::{BTreeMap, BTreeSet};

use super::state::CompiledPipelineState;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{
        CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId,
    },
};

const OPERATION: &str = "nexus_query_engine.exec.bounded.compiled.scheduler";

/// Interprets executable work for a compiled pipeline task.
///
/// The scheduler owns ordering and resource-lifetime invariants. Implementors
/// own the operator-specific work and return the resident resources produced by
/// each task. D.1.2 tests use mock payloads; later stages use GPU frames.
#[doc(hidden)]
pub trait CompiledTaskExecutor {
    type Resource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> Result<CompiledTaskOutput<Self::Resource>>;

    fn release_resource(&mut self, _resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
        drop(payload);
        Ok(())
    }

    fn cleanup_after_error(&mut self) -> Result<()> {
        Ok(())
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
        let mut state = CompiledPipelineState::new(self.pipeline)?;

        while state.completed_task_count() < self.pipeline.tasks().len() {
            let task = match self.next_runnable_task(&state, executor) {
                Ok(task) => task,
                Err(error) => {
                    release_live_resources_on_error(executor, &mut state)?;
                    return Err(error);
                },
            };
            if let Err(error) = state.prepare_task(task) {
                release_live_resources_on_error(executor, &mut state)?;
                return Err(error);
            }
            let output = match executor.execute_task(task, &mut state) {
                Ok(output) => output,
                Err(error) => {
                    release_live_resources_on_error(executor, &mut state)?;
                    return Err(error);
                },
            };
            let released = match state.complete_task(task, output.into_produced_resources()) {
                Ok(released) => released,
                Err(error) => {
                    release_live_resources_on_error(executor, &mut state)?;
                    return Err(error);
                },
            };
            for resource in released {
                if let Err(error) = executor.release_resource(resource.id(), resource.into_payload()) {
                    release_live_resources_on_error(executor, &mut state)?;
                    return Err(error);
                }
            }
        }

        if let Err(error) = state.ensure_finished(self.pipeline) {
            release_live_resources_on_error(executor, &mut state)?;
            return Err(error);
        }
        Ok(state)
    }

    fn next_runnable_task<E>(
        &self,
        state: &CompiledPipelineState<E::Resource>,
        executor: &E,
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

fn release_live_resources_on_error<E>(executor: &mut E, state: &mut CompiledPipelineState<E::Resource>) -> Result<()>
where E: CompiledTaskExecutor {
    for resource in state.drain_live_resources() {
        executor.release_resource(resource.id(), resource.into_payload())?;
    }
    executor.cleanup_after_error()?;
    Ok(())
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
