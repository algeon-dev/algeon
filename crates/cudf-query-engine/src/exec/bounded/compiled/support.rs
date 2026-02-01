use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::ContractStatus,
    error::Result,
    pipeline::{
        CompiledPipeline, PipelineBarrierKind, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind,
        SinkBoundaryKind,
    },
    plan::{NodeId, PlanNode, QueryPlan},
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
    if !task_kinds_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if task_kind_count(pipeline, PipelineTaskKind::SourceRead) != 2 {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if task_kind_count(pipeline, PipelineTaskKind::Aggregate) > 1
        || task_kind_count(pipeline, PipelineTaskKind::FinalSort) > 1
        || task_kind_count(pipeline, PipelineTaskKind::Limit) > 1
    {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_resource_shapes_are_in_d2_envelope(pipeline, join_build, join_probe, join_materialize, sink_materialize) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !edge_kinds_are_in_d2_envelope(pipeline) || !barrier_kinds_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !has_edge(pipeline, join_build.id(), join_probe.id(), PipelineEdgeKind::Control)
        || !has_edge(pipeline, join_probe.id(), join_materialize.id(), PipelineEdgeKind::Control)
        || !has_join_input_edge(pipeline, join_build.id(), PipelineEdgeKind::JoinBuildInput)
        || !has_join_input_edge(pipeline, join_probe.id(), PipelineEdgeKind::JoinProbeInput)
    {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !linear_post_root_chain_is_in_d2_envelope(pipeline, join_materialize, sink_materialize) {
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
    if !post_root_barriers_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }

    Ok(CompiledBoundedPipelineSupport::Supported)
}

#[doc(hidden)]
pub fn compiled_bounded_query_plan_support(
    plan: &QueryPlan,
    pipeline: &CompiledPipeline,
) -> Result<CompiledBoundedPipelineSupport> {
    if compiled_bounded_pipeline_support(pipeline)? == CompiledBoundedPipelineSupport::Unsupported {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_plan_nodes_match_kinds(plan, pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }

    let Some(join_build) = single_task_of_kind(pipeline, PipelineTaskKind::JoinBuild) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let Some(join_node_id) = join_build.plan_node() else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let Some(node) = plan.node(join_node_id) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let PlanNode::Join(join) = &node.kind else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    if !join.is_bounded_root_hash_join_compatible() {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    let Some(sink_materialize) = single_task_of_kind(pipeline, PipelineTaskKind::SinkMaterialize) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    let Some(final_plan_node) = sink_input_plan_node(pipeline, sink_materialize.id()) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    if plan.sink().input() != final_plan_node {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }

    Ok(CompiledBoundedPipelineSupport::Supported)
}

fn task_kinds_are_in_d2_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().all(|task| {
        matches!(
            task.kind(),
            PipelineTaskKind::SourceRead
                | PipelineTaskKind::Unary
                | PipelineTaskKind::Aggregate
                | PipelineTaskKind::JoinBuild
                | PipelineTaskKind::JoinProbe
                | PipelineTaskKind::JoinMaterialize
                | PipelineTaskKind::FinalSort
                | PipelineTaskKind::Limit
                | PipelineTaskKind::SinkMaterialize
        )
    })
}

fn task_resource_shapes_are_in_d2_envelope(
    pipeline: &CompiledPipeline,
    join_build: &PipelineTask,
    join_probe: &PipelineTask,
    join_materialize: &PipelineTask,
    sink_materialize: &PipelineTask,
) -> bool {
    pipeline.tasks().iter().all(|task| match task.kind() {
        PipelineTaskKind::SourceRead => task.input_resources().is_empty() && task.output_resources().len() == 1,
        PipelineTaskKind::Unary
        | PipelineTaskKind::Aggregate
        | PipelineTaskKind::FinalSort
        | PipelineTaskKind::Limit => task.input_resources().len() == 1 && task.output_resources().len() == 1,
        PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe => {
            task.input_resources().len() == 1 && task.output_resources().is_empty()
        },
        PipelineTaskKind::JoinMaterialize => task.input_resources().is_empty() && task.output_resources().len() == 1,
        PipelineTaskKind::SinkMaterialize => task.input_resources().len() == 1 && task.output_resources().is_empty(),
        PipelineTaskKind::GpuResidentSource => false,
    }) && join_build.plan_node().is_some()
        && join_build.plan_node() == join_probe.plan_node()
        && join_probe.plan_node() == join_materialize.plan_node()
        && sink_materialize.plan_node().is_none()
}

fn edge_kinds_are_in_d2_envelope(pipeline: &CompiledPipeline) -> bool {
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

fn barrier_kinds_are_in_d2_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.barriers().iter().all(|barrier| {
        matches!(
            barrier.kind(),
            PipelineBarrierKind::SourceChunkBoundary
                | PipelineBarrierKind::AggregateBoundary
                | PipelineBarrierKind::JoinBuildBeforeProbe
                | PipelineBarrierKind::JoinMaterialization
                | PipelineBarrierKind::SortBoundary
                | PipelineBarrierKind::SinkBoundary
        )
    })
}

fn linear_post_root_chain_is_in_d2_envelope(
    pipeline: &CompiledPipeline,
    join_materialize: &PipelineTask,
    sink_materialize: &PipelineTask,
) -> bool {
    let tasks = task_index(pipeline);
    let mut current = join_materialize.id();
    let mut visited = BTreeSet::new();
    let mut seen_aggregate = false;
    let mut seen_sort = false;
    let mut seen_limit = false;

    loop {
        if current == sink_materialize.id() {
            return true;
        }
        if !visited.insert(current) {
            return false;
        }

        let outgoing = pipeline
            .edges()
            .iter()
            .filter(|edge| {
                edge.from() == current
                    && matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SinkInput)
            })
            .collect::<Vec<_>>();
        let [edge] = outgoing.as_slice() else {
            return false;
        };
        let [resource] = edge.resources() else {
            return false;
        };
        let Some(current_task) = tasks.get(&current).copied() else {
            return false;
        };
        if !current_task.output_resources().contains(resource) {
            return false;
        }
        let Some(next_task) = tasks.get(&edge.to()).copied() else {
            return false;
        };
        if next_task.id() == sink_materialize.id() {
            return edge.kind() == PipelineEdgeKind::SinkInput
                && next_task.kind() == PipelineTaskKind::SinkMaterialize
                && next_task.input_resources() == [*resource];
        }
        if edge.kind() != PipelineEdgeKind::ResidentFrame || next_task.input_resources() != [*resource] {
            return false;
        }
        match next_task.kind() {
            PipelineTaskKind::Unary if !seen_sort && !seen_limit => {},
            PipelineTaskKind::Aggregate if !seen_aggregate && !seen_sort && !seen_limit => {
                seen_aggregate = true;
            },
            PipelineTaskKind::FinalSort if !seen_sort && !seen_limit => {
                seen_sort = true;
            },
            PipelineTaskKind::Limit if seen_sort && !seen_limit => {
                seen_limit = true;
            },
            PipelineTaskKind::SourceRead
            | PipelineTaskKind::GpuResidentSource
            | PipelineTaskKind::JoinBuild
            | PipelineTaskKind::JoinProbe
            | PipelineTaskKind::JoinMaterialize
            | PipelineTaskKind::SinkMaterialize
            | PipelineTaskKind::Unary
            | PipelineTaskKind::Aggregate
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit => return false,
        }
        current = next_task.id();
    }
}

fn post_root_barriers_are_in_d2_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().all(|task| match task.kind() {
        PipelineTaskKind::Aggregate => pipeline.barriers().iter().any(|barrier| {
            barrier.kind() == PipelineBarrierKind::AggregateBoundary && barrier.upstream_tasks() == [task.id()]
        }),
        PipelineTaskKind::FinalSort => pipeline.barriers().iter().any(|barrier| {
            barrier.kind() == PipelineBarrierKind::SortBoundary && barrier.upstream_tasks() == [task.id()]
        }),
        _ => true,
    })
}

fn task_plan_nodes_match_kinds(plan: &QueryPlan, pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().all(|task| match task.kind() {
        PipelineTaskKind::SourceRead => {
            matches!(task.plan_node().and_then(|node_id| plan.node(node_id)), Some(node) if matches!(&node.kind, PlanNode::Source(_)))
        },
        PipelineTaskKind::Unary => matches!(
            task.plan_node().and_then(|node_id| plan.node(node_id)),
            Some(node) if matches!(&node.kind, PlanNode::Filter(_) | PlanNode::Projection(_))
        ),
        PipelineTaskKind::Aggregate => {
            matches!(task.plan_node().and_then(|node_id| plan.node(node_id)), Some(node) if matches!(&node.kind, PlanNode::Aggregate(_)))
        },
        PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe | PipelineTaskKind::JoinMaterialize => {
            matches!(task.plan_node().and_then(|node_id| plan.node(node_id)), Some(node) if matches!(&node.kind, PlanNode::Join(_)))
        },
        PipelineTaskKind::FinalSort => {
            matches!(task.plan_node().and_then(|node_id| plan.node(node_id)), Some(node) if matches!(&node.kind, PlanNode::Sort(_)))
        },
        PipelineTaskKind::Limit => {
            matches!(task.plan_node().and_then(|node_id| plan.node(node_id)), Some(node) if matches!(&node.kind, PlanNode::Limit(_)))
        },
        PipelineTaskKind::SinkMaterialize => task.plan_node().is_none(),
        PipelineTaskKind::GpuResidentSource => false,
    })
}

fn sink_input_plan_node(pipeline: &CompiledPipeline, sink_task: PipelineTaskId) -> Option<NodeId> {
    let edge =
        pipeline.edges().iter().find(|edge| edge.to() == sink_task && edge.kind() == PipelineEdgeKind::SinkInput)?;
    task_index(pipeline).get(&edge.from()).and_then(|task| task.plan_node())
}

fn task_index(pipeline: &CompiledPipeline) -> BTreeMap<PipelineTaskId, &PipelineTask> {
    pipeline.tasks().iter().map(|task| (task.id(), task)).collect()
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
