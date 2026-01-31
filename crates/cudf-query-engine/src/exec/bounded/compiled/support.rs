use std::collections::BTreeSet;

use crate::{
    capability::ContractStatus,
    error::Result,
    pipeline::{
        CompiledPipeline, PipelineBarrierKind, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind,
        SinkBoundaryKind,
    },
};

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompiledBoundedPipelineSupport {
    Unsupported,
    Supported,
}

#[doc(hidden)]
pub fn compiled_bounded_pipeline_support(pipeline: &CompiledPipeline) -> Result<CompiledBoundedPipelineSupport> {
    pipeline.validate()?;

    if pipeline.gates().iter().any(|gate| gate.status() != ContractStatus::Modeled) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if pipeline.sink().kind() != SinkBoundaryKind::HostArrowMaterialization {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }

    let Some(join_build) = single_task_of_kind(pipeline, PipelineTaskKind::JoinBuild) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let Some(join_probe) = single_task_of_kind(pipeline, PipelineTaskKind::JoinProbe) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let Some(join_materialize) = single_task_of_kind(pipeline, PipelineTaskKind::JoinMaterialize) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let Some(sink_materialize) = single_task_of_kind(pipeline, PipelineTaskKind::SinkMaterialize) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };

    if pipeline.sink().input() != sink_materialize.id() {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_kinds_are_in_d1_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if task_kind_count(pipeline, PipelineTaskKind::SourceRead) != 2 {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_resource_shapes_are_in_d1_envelope(pipeline, join_build, join_probe, join_materialize, sink_materialize) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !edge_kinds_are_in_d1_envelope(pipeline) || !barrier_kinds_are_in_d1_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !has_edge(pipeline, join_build.id(), join_probe.id(), PipelineEdgeKind::Control)
        || !has_edge(pipeline, join_probe.id(), join_materialize.id(), PipelineEdgeKind::Control)
        || !has_join_input_edge(pipeline, join_build.id(), PipelineEdgeKind::JoinBuildInput)
        || !has_join_input_edge(pipeline, join_probe.id(), PipelineEdgeKind::JoinProbeInput)
        || !has_edge(pipeline, join_materialize.id(), sink_materialize.id(), PipelineEdgeKind::SinkInput)
    {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !has_barrier(pipeline, join_build.id(), join_probe.id(), PipelineBarrierKind::JoinBuildBeforeProbe)
        || !has_barrier(pipeline, join_probe.id(), join_materialize.id(), PipelineBarrierKind::JoinMaterialization)
        || !pipeline.barriers().iter().any(|barrier| {
            barrier.kind() == PipelineBarrierKind::SinkBoundary && barrier.upstream_tasks() == [sink_materialize.id()]
        })
    {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }

    Ok(CompiledBoundedPipelineSupport::Supported)
}

fn task_kinds_are_in_d1_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().all(|task| {
        matches!(
            task.kind(),
            PipelineTaskKind::SourceRead
                | PipelineTaskKind::Unary
                | PipelineTaskKind::JoinBuild
                | PipelineTaskKind::JoinProbe
                | PipelineTaskKind::JoinMaterialize
                | PipelineTaskKind::SinkMaterialize
        )
    })
}

fn task_resource_shapes_are_in_d1_envelope(
    pipeline: &CompiledPipeline,
    join_build: &PipelineTask,
    join_probe: &PipelineTask,
    join_materialize: &PipelineTask,
    sink_materialize: &PipelineTask,
) -> bool {
    pipeline.tasks().iter().all(|task| match task.kind() {
        PipelineTaskKind::SourceRead => task.input_resources().is_empty() && task.output_resources().len() == 1,
        PipelineTaskKind::Unary => task.input_resources().len() == 1 && task.output_resources().len() == 1,
        PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe => {
            task.input_resources().len() == 1 && task.output_resources().is_empty()
        },
        PipelineTaskKind::JoinMaterialize => task.input_resources().is_empty() && task.output_resources().len() == 1,
        PipelineTaskKind::SinkMaterialize => task.input_resources().len() == 1 && task.output_resources().is_empty(),
        PipelineTaskKind::GpuResidentSource
        | PipelineTaskKind::Aggregate
        | PipelineTaskKind::FinalSort
        | PipelineTaskKind::Limit => false,
    }) && join_build.plan_node().is_some()
        && join_build.plan_node() == join_probe.plan_node()
        && join_probe.plan_node() == join_materialize.plan_node()
        && sink_materialize.plan_node().is_none()
}

fn edge_kinds_are_in_d1_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.edges().iter().all(|edge| {
        matches!(
            edge.kind(),
            PipelineEdgeKind::Control
                | PipelineEdgeKind::ResidentFrame
                | PipelineEdgeKind::JoinBuildInput
                | PipelineEdgeKind::JoinProbeInput
                | PipelineEdgeKind::SinkInput
        )
    })
}

fn barrier_kinds_are_in_d1_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.barriers().iter().all(|barrier| {
        matches!(
            barrier.kind(),
            PipelineBarrierKind::SourceChunkBoundary
                | PipelineBarrierKind::JoinBuildBeforeProbe
                | PipelineBarrierKind::JoinMaterialization
                | PipelineBarrierKind::SinkBoundary
        )
    })
}

fn single_task_of_kind(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> Option<&PipelineTask> {
    let mut matches = pipeline.tasks().iter().filter(|task| task.kind() == kind);
    let first = matches.next();
    if matches.next().is_some() {
        return None;
    }
    first
}

fn task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> usize {
    pipeline.tasks().iter().filter(|task| task.kind() == kind).count()
}

fn has_edge(pipeline: &CompiledPipeline, from: PipelineTaskId, to: PipelineTaskId, kind: PipelineEdgeKind) -> bool {
    pipeline.edges().iter().any(|edge| edge.from() == from && edge.to() == to && edge.kind() == kind)
}

fn has_join_input_edge(pipeline: &CompiledPipeline, to: PipelineTaskId, kind: PipelineEdgeKind) -> bool {
    pipeline.edges().iter().any(|edge| {
        edge.to() == to
            && edge.kind() == kind
            && edge.resources().len() == 1
            && edge.gates().iter().all(|gate| pipeline.gates().iter().any(|candidate| candidate.id() == *gate))
    })
}

fn has_barrier(
    pipeline: &CompiledPipeline,
    upstream: PipelineTaskId,
    downstream: PipelineTaskId,
    kind: PipelineBarrierKind,
) -> bool {
    pipeline.barriers().iter().any(|barrier| {
        barrier.kind() == kind
            && task_set(barrier.upstream_tasks()) == BTreeSet::from([upstream])
            && task_set(barrier.downstream_tasks()) == BTreeSet::from([downstream])
    })
}

fn task_set(tasks: &[PipelineTaskId]) -> BTreeSet<PipelineTaskId> {
    tasks.iter().copied().collect()
}
