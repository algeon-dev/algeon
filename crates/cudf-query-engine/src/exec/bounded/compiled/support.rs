use std::collections::{BTreeMap, BTreeSet};

use super::join_strategy::CompiledJoinExecutionStrategy;
use crate::{
    capability::{
        CapabilityDecision, ContractQuality, ContractStatus, KeylessConditionalJoinKind, SubgraphInvariantKind,
    },
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
    /// The graph is capability-admitted but outside the strict compiled
    /// executor envelope.
    Unsupported,
    /// The graph is inside the strict compiled executor envelope.
    Supported,
}

#[doc(hidden)]
/// Checks compiled graph shape only; it does not perform capability admission.
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
    if !task_reservation_gates_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !task_resource_shapes_are_in_d2_envelope(pipeline, &join_groups, sink_materialize) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !edge_kinds_are_in_d3_2_envelope(pipeline) || !barrier_kinds_are_in_d2_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !shared_resident_producer_edges_are_in_d3_5_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    if !join_edges_are_in_d2_envelope(pipeline, &join_groups) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }
    let Some(sink_path) = explicit_host_sink_path_in_d3_6_envelope(pipeline, sink_materialize) else {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    };
    if !stage_tasks_are_in_d3_6_envelope(pipeline, &sink_path) {
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
    if !stage_barriers_are_in_d3_3_envelope(pipeline) {
        return Ok(CompiledBoundedPipelineSupport::Unsupported);
    }

    Ok(CompiledBoundedPipelineSupport::Supported)
}

#[doc(hidden)]
/// Checks graph shape plus plan-node compatibility for strict compiled execution.
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
    if pipeline_has_computed_shared_resident_producer(pipeline)
        && !plan_has_satisfied_shared_resident_subplan_lineage(plan)
    {
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
        let Ok(Some(strategy)) = CompiledJoinExecutionStrategy::from_plan_join_node(plan, group.plan_node, join) else {
            return Ok(CompiledBoundedPipelineSupport::Unsupported);
        };
        if !join_strategy_reservations_are_modeled(pipeline, &group, strategy) {
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

fn join_strategy_reservations_are_modeled(
    pipeline: &CompiledPipeline,
    group: &JoinTaskGroup<'_>,
    strategy: CompiledJoinExecutionStrategy,
) -> bool {
    task_runtime_attachment_has_modeled_reservation_gate(pipeline, group.build, ReservationKind::BuildSide)
        && task_runtime_attachment_has_modeled_reservation_gate(pipeline, group.probe, ReservationKind::ProbeSide)
        && task_runtime_attachment_has_modeled_reservation_gate(
            pipeline,
            group.materialize,
            ReservationKind::JoinOutput,
        )
        && join_strategy_specific_reservations_are_modeled(pipeline, group, strategy)
}

fn join_strategy_specific_reservations_are_modeled(
    pipeline: &CompiledPipeline,
    group: &JoinTaskGroup<'_>,
    strategy: CompiledJoinExecutionStrategy,
) -> bool {
    match strategy {
        CompiledJoinExecutionStrategy::HashEqui => {
            task_runtime_attachment_has_modeled_reservation_gate(pipeline, group.build, ReservationKind::HashTable)
                && task_runtime_attachment_has_modeled_reservation_gate(
                    pipeline,
                    group.materialize,
                    ReservationKind::GatherMap,
                )
        },
        CompiledJoinExecutionStrategy::OneShotEqui | CompiledJoinExecutionStrategy::PartitionedHashEqui => {
            task_runtime_attachment_has_modeled_reservation_gate(
                pipeline,
                group.materialize,
                ReservationKind::GatherMap,
            )
        },
        CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual => {
            task_runtime_attachment_has_modeled_reservation_gate(
                pipeline,
                group.materialize,
                ReservationKind::MixedSizeData,
            ) && task_runtime_attachment_has_modeled_reservation_gate(
                pipeline,
                group.materialize,
                ReservationKind::GatherMap,
            )
        },
        CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            task_runtime_attachment_has_modeled_reservation_gate(pipeline, group.build, ReservationKind::HashTable)
                && task_runtime_attachment_has_modeled_reservation_gate(
                    pipeline,
                    group.materialize,
                    ReservationKind::FilteredProbeState,
                )
                && task_runtime_attachment_has_modeled_reservation_gate(
                    pipeline,
                    group.materialize,
                    ReservationKind::GatherMap,
                )
        },
        CompiledJoinExecutionStrategy::KeylessConditional { join_type } => {
            (join_type != KeylessConditionalJoinKind::Full
                || task_runtime_attachment_has_modeled_reservation_gate(
                    pipeline,
                    group.materialize,
                    ReservationKind::MixedSizeData,
                ))
                && task_runtime_attachment_has_modeled_reservation_gate(
                    pipeline,
                    group.materialize,
                    ReservationKind::GatherMap,
                )
        },
        CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            task_runtime_attachment_has_modeled_reservation_gate(
                pipeline,
                group.materialize,
                ReservationKind::MixedSizeData,
            ) && task_runtime_attachment_has_modeled_reservation_gate(
                pipeline,
                group.materialize,
                ReservationKind::GatherMap,
            )
        },
        CompiledJoinExecutionStrategy::ProjectedCross => task_runtime_attachment_has_modeled_reservation_gate(
            pipeline,
            group.materialize,
            ReservationKind::JoinOutput,
        ),
    }
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
            task_has_hard_budgeted_modeled_reservation_gate(pipeline, task, ReservationKind::OperatorOutput)
        },
        PipelineTaskKind::FinalSort => {
            task_runtime_attachment_has_modeled_reservation_gate(pipeline, task, ReservationKind::FinalSort)
        },
        _ => true,
    })
}

fn task_has_hard_budgeted_modeled_reservation_gate(
    pipeline: &CompiledPipeline,
    task: &PipelineTask,
    reservation_kind: ReservationKind,
) -> bool {
    task.gates().iter().any(|gate_id| {
        pipeline.gates().iter().any(|gate| {
            gate.id() == *gate_id
                && gate.reservation_kind() == reservation_kind
                && gate.status() == ContractStatus::Modeled
                && gate.quality() == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
        })
    })
}

fn task_has_modeled_reservation_gate(
    pipeline: &CompiledPipeline,
    task: &PipelineTask,
    reservation_kind: ReservationKind,
) -> bool {
    task.gates().iter().any(|gate_id| modeled_gate_has_kind(pipeline, *gate_id, reservation_kind))
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

fn shared_resident_producer_edges_are_in_d3_5_envelope(pipeline: &CompiledPipeline) -> bool {
    let tasks = task_index(pipeline);
    pipeline.edges().iter().filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer).all(|edge| {
        let Some(task) = tasks.get(&edge.from()) else {
            return false;
        };
        matches!(
            task.kind(),
            PipelineTaskKind::SourceRead
                | PipelineTaskKind::GpuResidentSource
                | PipelineTaskKind::Unary
                | PipelineTaskKind::Aggregate
        ) && edge.resources().len() == 1
            && task.output_resources().contains(&edge.resources()[0])
    })
}

fn pipeline_has_computed_shared_resident_producer(pipeline: &CompiledPipeline) -> bool {
    let tasks = task_index(pipeline);
    pipeline.edges().iter().filter(|edge| edge.kind() == PipelineEdgeKind::SharedResidentProducer).any(|edge| {
        tasks
            .get(&edge.from())
            .is_some_and(|task| matches!(task.kind(), PipelineTaskKind::Unary | PipelineTaskKind::Aggregate))
    })
}

fn plan_has_satisfied_shared_resident_subplan_lineage(plan: &QueryPlan) -> bool {
    let analysis = plan.capability_analysis();
    let has_shared_subplan = analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::SharedResidentSubplan)
            && invariant.decision == CapabilityDecision::Satisfied
    });
    let has_aggregate_derivation = analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::AggregateDerivesFromSharedProducer)
            && invariant.decision == CapabilityDecision::Satisfied
    });
    has_shared_subplan && has_aggregate_derivation
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

fn explicit_host_sink_path_in_d3_6_envelope(
    pipeline: &CompiledPipeline,
    sink_materialize: &PipelineTask,
) -> Option<BTreeSet<PipelineTaskId>> {
    let tasks = task_index(pipeline);
    let sink_edge = single_sink_input_edge_to_task(pipeline, sink_materialize.id())?;
    let [sink_resource] = sink_edge.resources() else {
        return None;
    };
    if !sink_materialize.input_resources().contains(sink_resource) {
        return None;
    }

    let mut current = sink_edge.from();
    let mut reverse_path = vec![sink_materialize.id()];
    let mut visited = BTreeSet::from([sink_materialize.id()]);

    loop {
        if !visited.insert(current) {
            return None;
        }
        reverse_path.push(current);

        let current_task = tasks.get(&current).copied()?;
        if !post_root_task_has_single_data_output_on_sink_path(pipeline, current_task) {
            return None;
        }

        if current_task.kind() == PipelineTaskKind::JoinMaterialize {
            break;
        }

        if !matches!(
            current_task.kind(),
            PipelineTaskKind::Unary
                | PipelineTaskKind::Aggregate
                | PipelineTaskKind::FinalSort
                | PipelineTaskKind::Limit
        ) {
            return None;
        }

        let incoming = pipeline
            .edges()
            .iter()
            .filter(|edge| edge.to() == current && edge.kind() == PipelineEdgeKind::ResidentFrame)
            .collect::<Vec<_>>();
        let [incoming] = incoming.as_slice() else {
            return None;
        };
        let [resource] = incoming.resources() else {
            return None;
        };
        if current_task.input_resources() != [*resource] {
            return None;
        }
        let producer = tasks.get(&incoming.from()).copied()?;
        if !producer.output_resources().contains(resource) {
            return None;
        }
        current = producer.id();
    }

    reverse_path.reverse();
    if !sink_path_stage_order_is_in_d3_6_envelope(pipeline, &reverse_path) {
        return None;
    }

    Some(reverse_path.into_iter().collect())
}

fn single_sink_input_edge_to_task(
    pipeline: &CompiledPipeline,
    sink_task: PipelineTaskId,
) -> Option<&crate::pipeline::PipelineEdge> {
    let mut sink_edges = pipeline.edges().iter().filter(|edge| edge.kind() == PipelineEdgeKind::SinkInput);
    let first = sink_edges.next()?;
    if sink_edges.next().is_some() || first.to() != sink_task {
        return None;
    }
    Some(first)
}

fn post_root_task_has_single_data_output_on_sink_path(pipeline: &CompiledPipeline, task: &PipelineTask) -> bool {
    let outgoing_data_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| {
            edge.from() == task.id()
                && matches!(edge.kind(), PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SinkInput)
        })
        .collect::<Vec<_>>();
    let [edge] = outgoing_data_edges.as_slice() else {
        return false;
    };
    let [resource] = edge.resources() else {
        return false;
    };
    task.output_resources().contains(resource)
}

fn sink_path_stage_order_is_in_d3_6_envelope(pipeline: &CompiledPipeline, sink_path: &[PipelineTaskId]) -> bool {
    let tasks = task_index(pipeline);
    let mut seen_sort = false;
    let mut seen_limit = false;

    for task_id in sink_path {
        let Some(task) = tasks.get(task_id) else {
            return false;
        };
        match task.kind() {
            PipelineTaskKind::JoinMaterialize | PipelineTaskKind::SinkMaterialize => {},
            PipelineTaskKind::Unary | PipelineTaskKind::Aggregate if !seen_sort && !seen_limit => {},
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
            | PipelineTaskKind::Unary
            | PipelineTaskKind::Aggregate
            | PipelineTaskKind::FinalSort
            | PipelineTaskKind::Limit => return false,
        }
    }

    true
}

fn stage_tasks_are_in_d3_6_envelope(pipeline: &CompiledPipeline, sink_path: &BTreeSet<PipelineTaskId>) -> bool {
    pipeline.tasks().iter().all(|task| {
        if !matches!(task.kind(), PipelineTaskKind::Aggregate | PipelineTaskKind::FinalSort | PipelineTaskKind::Limit) {
            return true;
        }

        sink_path.contains(&task.id()) || task_feeds_modeled_join_input_or_shared_branch(pipeline, task.id())
    })
}

fn stage_barriers_are_in_d3_3_envelope(pipeline: &CompiledPipeline) -> bool {
    pipeline.tasks().iter().all(|task| match task.kind() {
        PipelineTaskKind::Aggregate => {
            stage_boundary_dominates_downstream_join_inputs(pipeline, task.id(), PipelineBarrierKind::AggregateBoundary)
        },
        PipelineTaskKind::FinalSort => {
            stage_boundary_dominates_downstream_join_inputs(pipeline, task.id(), PipelineBarrierKind::SortBoundary)
        },
        _ => true,
    })
}

fn task_feeds_modeled_join_input_or_shared_branch(pipeline: &CompiledPipeline, task: PipelineTaskId) -> bool {
    let mut pending = vec![task];
    let mut visited = BTreeSet::new();

    while let Some(current) = pending.pop() {
        if !visited.insert(current) {
            continue;
        }

        for edge in pipeline.edges().iter().filter(|edge| edge.from() == current) {
            match edge.kind() {
                PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => {
                    if edge.gates().iter().all(|gate_id| pipeline.gates().iter().any(|gate| gate.id() == *gate_id)) {
                        return true;
                    }
                },
                PipelineEdgeKind::SharedResidentProducer => return true,
                PipelineEdgeKind::ResidentFrame => pending.push(edge.to()),
                PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput => {},
            }
        }
    }

    false
}

fn stage_boundary_dominates_downstream_join_inputs(
    pipeline: &CompiledPipeline,
    task: PipelineTaskId,
    barrier_kind: PipelineBarrierKind,
) -> bool {
    let downstream_join_inputs = downstream_join_input_tasks(pipeline, task);
    pipeline.barriers().iter().any(|barrier| {
        barrier.kind() == barrier_kind
            && barrier.upstream_tasks() == [task]
            && downstream_join_inputs.iter().all(|join_input| {
                barrier.downstream_tasks().iter().any(|downstream| task_reaches(pipeline, *downstream, *join_input))
            })
    })
}

fn downstream_join_input_tasks(pipeline: &CompiledPipeline, task: PipelineTaskId) -> BTreeSet<PipelineTaskId> {
    let mut pending = vec![task];
    let mut visited = BTreeSet::new();
    let mut join_inputs = BTreeSet::new();

    while let Some(current) = pending.pop() {
        if !visited.insert(current) {
            continue;
        }

        for edge in pipeline.edges().iter().filter(|edge| edge.from() == current) {
            match edge.kind() {
                PipelineEdgeKind::JoinBuildInput | PipelineEdgeKind::JoinProbeInput => {
                    join_inputs.insert(edge.to());
                },
                PipelineEdgeKind::ResidentFrame | PipelineEdgeKind::SharedResidentProducer => pending.push(edge.to()),
                PipelineEdgeKind::Control | PipelineEdgeKind::BarrierOutput | PipelineEdgeKind::SinkInput => {},
            }
        }
    }

    join_inputs
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

fn task_reaches(pipeline: &CompiledPipeline, from: PipelineTaskId, to: PipelineTaskId) -> bool {
    if from == to {
        return true;
    }

    let adjacency =
        pipeline.edges().iter().fold(BTreeMap::<PipelineTaskId, Vec<PipelineTaskId>>::new(), |mut map, edge| {
            map.entry(edge.from()).or_default().push(edge.to());
            map
        });
    let mut pending = adjacency.get(&from).cloned().unwrap_or_default();
    let mut visited = BTreeSet::new();

    while let Some(current) = pending.pop() {
        if current == to {
            return true;
        }
        if visited.insert(current) {
            pending.extend(adjacency.get(&current).into_iter().flatten().copied());
        }
    }

    false
}
