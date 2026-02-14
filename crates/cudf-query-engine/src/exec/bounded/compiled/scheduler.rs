use std::collections::{BTreeMap, BTreeSet};

use super::state::CompiledPipelineState;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{
        CompiledPipeline, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ResidentResourceId,
    },
};

const OPERATION: &str = "cudf_query_engine.exec.bounded.compiled.scheduler";

/// Interprets executable work for a compiled pipeline task.
///
/// The scheduler owns ordering and resource-lifetime invariants. Implementors
/// own the operator-specific work and return the resident resources produced by
/// each task. D.1.2 tests use mock payloads; later phases use GPU frames.
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
            let task = match self.next_runnable_task(&state) {
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

    fn next_runnable_task<T>(&self, state: &CompiledPipelineState<T>) -> Result<&'a PipelineTask> {
        self.pipeline
            .tasks()
            .iter()
            .filter(|task| {
                !state.is_task_completed(task.id())
                    && self.dependencies.get(&task.id()).is_none_or(|dependencies| {
                        dependencies.iter().all(|dependency| state.is_task_completed(*dependency))
                    })
            })
            .min_by_key(|task| self.runnable_task_priority(task, state))
            .ok_or_else(|| {
                invalid("compiled pipeline has no runnable task before completion")
                    .with_fact("completed_task_count", state.completed_task_count())
                    .with_fact("pipeline_task_count", self.pipeline.tasks().len())
            })
    }

    fn runnable_task_priority<T>(&self, task: &PipelineTask, state: &CompiledPipelineState<T>) -> (u8, usize) {
        let build_priority = task.kind() == PipelineTaskKind::JoinBuild
            && self.prioritized_join_builds.contains(&task.id())
            || (self.has_completed_join_build_waiting_for_probe(state)
                && self.task_feeds_incomplete_prioritized_join_build(task.id(), state));
        if build_priority { (0, task.id().index()) } else { (1, task.id().index()) }
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

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
