use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::ContractStatus,
    error::Result,
    pipeline::{
        CompiledPipeline, PipelineBarrierKind, PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind,
        ReservationGateId, SinkBoundaryKind,
    },
    plan::{NodeId, PlanNode, QueryPlan},
    runtime::ReservationKind,
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

    let Some(join_groups) = join_task_groups(pipeline) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    if join_groups.is_empty() {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    let Some(sink_materialize) = single_task_of_kind(pipeline, PipelineTaskKind::SinkMaterialize) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };

    if pipeline.sink().input() != sink_materialize.id() {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_kinds_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if task_kind_count(pipeline, PipelineTaskKind::SourceRead) < 1 {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if task_kind_count(pipeline, PipelineTaskKind::Aggregate) > 1
        || task_kind_count(pipeline, PipelineTaskKind::FinalSort) > 1
        || task_kind_count(pipeline, PipelineTaskKind::Limit) > 1
    {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_reservation_gates_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_resource_shapes_are_in_d2_envelope(pipeline, &join_groups, sink_materialize) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !edge_kinds_are_in_d3_2_envelope(pipeline) || !barrier_kinds_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !join_edges_are_in_d2_envelope(pipeline, &join_groups) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    let Some(post_root_chain) = root_join_post_root_chain_in_d2_envelope(pipeline, &join_groups, sink_materialize)
    else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    if !post_root_stage_tasks_are_on_chain(pipeline, &post_root_chain) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !join_barriers_are_in_d2_envelope(pipeline, &join_groups) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !pipeline.barriers().iter().any(|barrier| {
        barrier.kind() == PipelineBarrierKind::SinkBoundary && barrier.upstream_tasks() == [sink_materialize.id()]
    }) {
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

    let Some(join_groups) = join_task_groups(pipeline) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    for group in join_groups {
        let Some(node) = plan.node(group.plan_node) else {
            return Ok(CompiledBoundedPipelineSupport::Unsupported);
        };
        let PlanNode::Join(join) = &node.kind else {
            return Ok(CompiledBoundedPipelineSupport::Unsupported);
        };
        if !join.is_bounded_root_hash_join_compatible() {
            return Ok(CompiledBoundedPipelineSupport::Unsupported);
        }
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
    join_groups: &[JoinTaskGroup<'_>],
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
    }) && join_groups.iter().all(|group| {
        group.build.plan_node() == Some(group.plan_node)
            && group.probe.plan_node() == Some(group.plan_node)
            && group.materialize.plan_node() == Some(group.plan_node)
    }) && sink_materialize.plan_node().is_none()
}

fn task_reservation_gates_are_in_d2_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().all(|task| match task.kind() {
        PipelineTaskKind::Aggregate => {
            task_has_modeled_reservation_gate(pipeline, task, ReservationKind::OperatorOutput)
        },
        PipelineTaskKind::FinalSort => {
            task_runtime_attachment_has_modeled_reservation_gate(pipeline, task, ReservationKind::FinalSort)
        },
        _ => true,
    })
}

fn task_has_modeled_reservation_gate(
    pipeline: &CompiledPipeline,
    task: &PipelineTask,
    reservation_kind: ReservationKind,
) -> bool {
    task.gates().iter().any(|gate_id| {
        pipeline.gates().iter().any(|gate| {
            gate.id() == *gate_id
                && gate.reservation_kind() == reservation_kind
                && gate.status() == ContractStatus::Modeled
        })
    })
}

fn task_runtime_attachment_has_modeled_reservation_gate(
    pipeline: &CompiledPipeline,
    task: &PipelineTask,
    reservation_kind: ReservationKind,
) -> bool {
    if task_has_modeled_reservation_gate(pipeline, task, reservation_kind) {
        return true;
    }

    let task_id = task.id();
    let edge_has_gate = pipeline
        .edges()
        .iter()
        .filter(|edge| edge.from() == task_id || edge.to() == task_id)
        .flat_map(|edge| edge.gates())
        .any(|gate_id| modeled_gate_has_kind(pipeline, *gate_id, reservation_kind));
    if edge_has_gate {
        return true;
    }

    let barrier_has_gate = pipeline
        .barriers()
        .iter()
        .filter(|barrier| barrier.upstream_tasks().contains(&task_id) || barrier.downstream_tasks().contains(&task_id))
        .flat_map(|barrier| barrier.gates())
        .any(|gate_id| modeled_gate_has_kind(pipeline, *gate_id, reservation_kind));
    if barrier_has_gate {
        return true;
    }

    pipeline.sink().input() == task_id
        && pipeline.sink().gates().iter().any(|gate_id| modeled_gate_has_kind(pipeline, *gate_id, reservation_kind))
}

fn modeled_gate_has_kind(
    pipeline: &CompiledPipeline,
    gate_id: ReservationGateId,
    reservation_kind: ReservationKind,
) -> bool {
    pipeline.gates().iter().any(|gate| {
        gate.id() == gate_id && gate.reservation_kind() == reservation_kind && gate.status() == ContractStatus::Modeled
    })
}

fn edge_kinds_are_in_d3_2_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.edges().iter().all(|edge| {
        matches!(
            edge.kind(),
            PipelineEdgeKind::Control
                | PipelineEdgeKind::ResidentFrame
                | PipelineEdgeKind::SharedResidentProducer
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

fn join_edges_are_in_d2_envelope(pipeline: &CompiledPipeline, join_groups: &[JoinTaskGroup<'_>]) -> bool {
    join_groups.iter().all(|group| {
        has_edge(pipeline, group.build.id(), group.probe.id(), PipelineEdgeKind::Control)
            && has_edge(pipeline, group.probe.id(), group.materialize.id(), PipelineEdgeKind::Control)
            && has_join_input_edge(pipeline, group.build.id(), PipelineEdgeKind::JoinBuildInput)
            && has_join_input_edge(pipeline, group.probe.id(), PipelineEdgeKind::JoinProbeInput)
    })
}

fn join_barriers_are_in_d2_envelope(pipeline: &CompiledPipeline, join_groups: &[JoinTaskGroup<'_>]) -> bool {
    join_groups.iter().all(|group| {
        has_barrier(pipeline, group.build.id(), group.probe.id(), PipelineBarrierKind::JoinBuildBeforeProbe)
            && has_barrier(pipeline, group.probe.id(), group.materialize.id(), PipelineBarrierKind::JoinMaterialization)
    })
}

fn root_join_post_root_chain_in_d2_envelope(
    pipeline: &CompiledPipeline,
    join_groups: &[JoinTaskGroup<'_>],
    sink_materialize: &PipelineTask,
) -> Option<BTreeSet<PipelineTaskId>> {
    let mut candidates = join_groups
        .iter()
        .filter_map(|group| linear_post_root_chain_in_d2_envelope(pipeline, group.materialize, sink_materialize))
        .collect::<Vec<_>>();
    if candidates.len() == 1 { candidates.pop() } else { None }
}

fn linear_post_root_chain_in_d2_envelope(
    pipeline: &CompiledPipeline,
    join_materialize: &PipelineTask,
    sink_materialize: &PipelineTask,
) -> Option<BTreeSet<PipelineTaskId>> {
    let tasks = task_index(pipeline);
    let mut current = join_materialize.id();
    let mut visited = BTreeSet::new();
    let mut seen_aggregate = false;
    let mut seen_sort = false;
    let mut seen_limit = false;

    loop {
        if current == sink_materialize.id() {
            visited.insert(current);
            return Some(visited);
        }
        if !visited.insert(current) {
            return None;
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
            return None;
        };
        let [resource] = edge.resources() else {
            return None;
        };
        let current_task = tasks.get(&current).copied()?;
        if !current_task.output_resources().contains(resource) {
            return None;
        }
        let next_task = tasks.get(&edge.to()).copied()?;
        if next_task.id() == sink_materialize.id() {
            if edge.kind() == PipelineEdgeKind::SinkInput
                && next_task.kind() == PipelineTaskKind::SinkMaterialize
                && next_task.input_resources() == [*resource]
            {
                visited.insert(next_task.id());
                return Some(visited);
            }
            return None;
        }
        if edge.kind() != PipelineEdgeKind::ResidentFrame || next_task.input_resources() != [*resource] {
            return None;
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
            | PipelineTaskKind::Limit => return None,
        }
        current = next_task.id();
    }
}

fn post_root_stage_tasks_are_on_chain(pipeline: &CompiledPipeline, post_root_chain: &BTreeSet<PipelineTaskId>) -> bool {
    pipeline.tasks().iter().all(|task| {
        !matches!(task.kind(), PipelineTaskKind::Aggregate | PipelineTaskKind::FinalSort | PipelineTaskKind::Limit)
            || post_root_chain.contains(&task.id())
    })
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

#[derive(Debug, Clone, Copy)]
struct JoinTaskGroup<'a> {
    plan_node: NodeId,
    build: &'a PipelineTask,
    probe: &'a PipelineTask,
    materialize: &'a PipelineTask,
}

#[derive(Debug, Default)]
struct PartialJoinTaskGroup<'a> {
    build: Option<&'a PipelineTask>,
    probe: Option<&'a PipelineTask>,
    materialize: Option<&'a PipelineTask>,
}

fn join_task_groups(pipeline: &CompiledPipeline) -> Option<Vec<JoinTaskGroup<'_>>> {
    let mut partials = BTreeMap::<NodeId, PartialJoinTaskGroup<'_>>::new();
    for task in pipeline.tasks().iter().filter(|task| {
        matches!(
            task.kind(),
            PipelineTaskKind::JoinBuild | PipelineTaskKind::JoinProbe | PipelineTaskKind::JoinMaterialize
        )
    }) {
        let plan_node = task.plan_node()?;
        let partial = partials.entry(plan_node).or_default();
        let slot = match task.kind() {
            PipelineTaskKind::JoinBuild => &mut partial.build,
            PipelineTaskKind::JoinProbe => &mut partial.probe,
            PipelineTaskKind::JoinMaterialize => &mut partial.materialize,
            _ => unreachable!("filtered to join tasks"),
        };
        if slot.replace(task).is_some() {
            return None;
        }
    }

    partials
        .into_iter()
        .map(|(plan_node, partial)| {
            Some(JoinTaskGroup {
                plan_node,
                build: partial.build?,
                probe: partial.probe?,
                materialize: partial.materialize?,
            })
        })
        .collect()
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
