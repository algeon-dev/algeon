use std::collections::{BTreeMap, BTreeSet};

use super::state::CompiledPipelineState;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{CompiledPipeline, PipelineTask, PipelineTaskId, ResidentResourceId},
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
        state: &CompiledPipelineState<Self::Resource>,
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
}

impl<'a> CompiledPipelineScheduler<'a> {
    pub fn new(pipeline: &'a CompiledPipeline) -> Result<Self> {
        pipeline.validate()?;
        let dependencies = build_dependencies(pipeline);
        Ok(Self { pipeline, dependencies })
    }

    pub fn run<E>(&self, executor: &mut E) -> Result<CompiledPipelineState<E::Resource>>
    where E: CompiledTaskExecutor {
        let mut state = CompiledPipelineState::new(self.pipeline)?;

        while state.completed_task_count() < self.pipeline.tasks().len() {
            let task = self.next_runnable_task(&state)?;
            state.prepare_task(task)?;
            let output = executor.execute_task(task, &state)?;
            let released = state.complete_task(task, output.into_produced_resources())?;
            for resource in released {
                executor.release_resource(resource.id(), resource.into_payload())?;
            }
        }

        state.ensure_finished(self.pipeline)?;
        Ok(state)
    }

    fn next_runnable_task<T>(&self, state: &CompiledPipelineState<T>) -> Result<&'a PipelineTask> {
        self.pipeline
            .tasks()
            .iter()
            .find(|task| {
                !state.is_task_completed(task.id())
                    && self.dependencies.get(&task.id()).is_none_or(|dependencies| {
                        dependencies.iter().all(|dependency| state.is_task_completed(*dependency))
                    })
            })
            .ok_or_else(|| {
                invalid("compiled pipeline has no runnable task before completion")
                    .with_fact("completed_task_count", state.completed_task_count())
                    .with_fact("pipeline_task_count", self.pipeline.tasks().len())
            })
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

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
