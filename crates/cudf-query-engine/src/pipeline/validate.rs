//! CompiledPipeline invariant validation.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    capability::{ContractQuality, ContractStatus, ReservationRequirementScope, lower_level_unresolved_preflight_gap},
    error::{QueryError, QueryErrorCode},
    pipeline::model::{
        CompiledPipeline, PipelineBarrierId, PipelineBarrierKind, PipelineEdgeId, PipelineEdgeKind, PipelineTaskId,
        PipelineTaskKind, ReservationGateId, ReservationGateOwner, ResidentLifetime, ResidentResource,
        ResidentResourceId,
    },
    runtime::ReservationKind,
};

const OPERATION: &str = "cudf_query_engine.pipeline.validate";

pub(super) fn validate_pipeline(pipeline: &CompiledPipeline) -> crate::Result<()> {
    let task_ids = validate_tasks(pipeline)?;
    let edge_ids = validate_edges(pipeline, &task_ids)?;
    let barrier_ids = validate_barriers(pipeline, &task_ids)?;
    let resource_ids = validate_resources(pipeline, &task_ids)?;
    let gate_ids = validate_gates(pipeline, &task_ids, &edge_ids, &barrier_ids)?;

    validate_task_dependencies(pipeline, &task_ids)?;
    validate_task_graph_is_acyclic(pipeline)?;
    validate_task_resource_references(pipeline, &resource_ids)?;
    validate_resource_flow(pipeline)?;
    validate_barrier_dominance(pipeline)?;
    validate_gate_references(pipeline, &gate_ids)?;
    validate_gate_owner_references(pipeline)?;
    validate_reservation_gate_contracts(pipeline)?;

    if !task_ids.contains(&pipeline.sink().input()) {
        return Err(invalid("pipeline sink input task does not exist").with_fact("task_id", pipeline.sink().input()));
    }

    Ok(())
}

fn validate_tasks(pipeline: &CompiledPipeline) -> crate::Result<BTreeSet<PipelineTaskId>> {
    let mut ids = BTreeSet::new();
    for (index, task) in pipeline.tasks().iter().enumerate() {
        if task.id().index() != index {
            return Err(invalid("pipeline task id does not match task position")
                .with_fact("task_id", task.id())
                .with_fact("position", index));
        }
        if !ids.insert(task.id()) {
            return Err(invalid("pipeline task id is duplicated").with_fact("task_id", task.id()));
        }
    }
    Ok(ids)
}

fn validate_edges(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> crate::Result<BTreeSet<PipelineEdgeId>> {
    let mut ids = BTreeSet::new();
    for (index, edge) in pipeline.edges().iter().enumerate() {
        if edge.id().index() != index {
            return Err(invalid("pipeline edge id does not match edge position")
                .with_fact("edge_id", edge.id())
                .with_fact("position", index));
        }
        if !ids.insert(edge.id()) {
            return Err(invalid("pipeline edge id is duplicated").with_fact("edge_id", edge.id()));
        }
        if !task_ids.contains(&edge.from()) {
            return Err(invalid("pipeline edge producer task does not exist")
                .with_fact("edge_id", edge.id())
                .with_fact("task_id", edge.from()));
        }
        if !task_ids.contains(&edge.to()) {
            return Err(invalid("pipeline edge consumer task does not exist")
                .with_fact("edge_id", edge.id())
                .with_fact("task_id", edge.to()));
        }
    }
    Ok(ids)
}

fn validate_barriers(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> crate::Result<BTreeSet<PipelineBarrierId>> {
    let mut ids = BTreeSet::new();
    for (index, barrier) in pipeline.barriers().iter().enumerate() {
        if barrier.id().index() != index {
            return Err(invalid("pipeline barrier id does not match barrier position")
                .with_fact("barrier_id", barrier.id())
                .with_fact("position", index));
        }
        if !ids.insert(barrier.id()) {
            return Err(invalid("pipeline barrier id is duplicated").with_fact("barrier_id", barrier.id()));
        }
        for task in barrier.upstream_tasks().iter().chain(barrier.downstream_tasks()) {
            if !task_ids.contains(task) {
                return Err(invalid("pipeline barrier task does not exist")
                    .with_fact("barrier_id", barrier.id())
                    .with_fact("task_id", task));
            }
        }
    }
    Ok(ids)
}

fn validate_resources(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> crate::Result<BTreeSet<ResidentResourceId>> {
    let mut ids = BTreeSet::new();
    for (index, resource) in pipeline.resources().iter().enumerate() {
        if resource.id().index() != index {
            return Err(invalid("pipeline resident resource id does not match resource position")
                .with_fact("resource_id", resource.id())
                .with_fact("position", index));
        }
        if !ids.insert(resource.id()) {
            return Err(invalid("pipeline resident resource id is duplicated").with_fact("resource_id", resource.id()));
        }
        if !task_ids.contains(&resource.producer()) {
            return Err(invalid("pipeline resident resource producer task does not exist")
                .with_fact("resource_id", resource.id())
                .with_fact("task_id", resource.producer()));
        }
        for consumer in resource.consumers() {
            if !task_ids.contains(consumer) {
                return Err(invalid("pipeline resident resource consumer task does not exist")
                    .with_fact("resource_id", resource.id())
                    .with_fact("task_id", consumer));
            }
        }
        validate_lifetime(resource.id(), resource.lifetime(), task_ids)?;
    }
    Ok(ids)
}

fn validate_lifetime(
    resource: ResidentResourceId,
    lifetime: ResidentLifetime,
    task_ids: &BTreeSet<PipelineTaskId>,
) -> crate::Result<()> {
    match lifetime {
        ResidentLifetime::TaskLocal { task } => {
            if !task_ids.contains(&task) {
                return Err(invalid("pipeline task-local lifetime task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", task));
            }
        },
        ResidentLifetime::BetweenTasks { producer } => {
            if !task_ids.contains(&producer) {
                return Err(invalid("pipeline resident lifetime producer task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", producer));
            }
        },
        ResidentLifetime::SinkOutput { producer } => {
            if !task_ids.contains(&producer) {
                return Err(invalid("pipeline sink-output lifetime producer task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", producer));
            }
        },
    }
    Ok(())
}

fn validate_gates(
    pipeline: &CompiledPipeline,
    task_ids: &BTreeSet<PipelineTaskId>,
    edge_ids: &BTreeSet<PipelineEdgeId>,
    barrier_ids: &BTreeSet<PipelineBarrierId>,
) -> crate::Result<BTreeSet<ReservationGateId>> {
    let mut ids = BTreeSet::new();
    for (index, gate) in pipeline.gates().iter().enumerate() {
        if gate.id().index() != index {
            return Err(invalid("pipeline reservation gate id does not match gate position")
                .with_fact("gate_id", gate.id())
                .with_fact("position", index));
        }
        if !ids.insert(gate.id()) {
            return Err(invalid("pipeline reservation gate id is duplicated").with_fact("gate_id", gate.id()));
        }
        match gate.owner() {
            ReservationGateOwner::Task(task) if !task_ids.contains(&task) => {
                return Err(invalid("pipeline reservation gate task owner does not exist")
                    .with_fact("gate_id", gate.id())
                    .with_fact("task_id", task));
            },
            ReservationGateOwner::Edge(edge) if !edge_ids.contains(&edge) => {
                return Err(invalid("pipeline reservation gate edge owner does not exist")
                    .with_fact("gate_id", gate.id())
                    .with_fact("edge_id", edge));
            },
            ReservationGateOwner::Barrier(barrier) if !barrier_ids.contains(&barrier) => {
                return Err(invalid("pipeline reservation gate barrier owner does not exist")
                    .with_fact("gate_id", gate.id())
                    .with_fact("barrier_id", barrier));
            },
            ReservationGateOwner::Task(_)
            | ReservationGateOwner::Edge(_)
            | ReservationGateOwner::Barrier(_)
            | ReservationGateOwner::SinkBoundary => {},
        }
    }
    Ok(ids)
}

fn validate_task_dependencies(pipeline: &CompiledPipeline, task_ids: &BTreeSet<PipelineTaskId>) -> crate::Result<()> {
    let mut incoming = BTreeSet::new();
    for edge in pipeline.edges() {
        incoming.insert(edge.to());
    }

    for task in pipeline.tasks() {
        if matches!(task.kind(), PipelineTaskKind::SourceRead | PipelineTaskKind::GpuResidentSource) {
            continue;
        }
        if !incoming.contains(&task.id()) {
            return Err(invalid("pipeline non-source task has no input dependency").with_fact("task_id", task.id()));
        }
    }

    for task in pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinProbe) {
        let has_build_dependency = pipeline
            .edges()
            .iter()
            .any(|edge| edge.to() == task.id() && is_join_build_for_probe(pipeline, edge.from(), task.id()));
        if !has_build_dependency {
            return Err(
                invalid("pipeline join probe task does not depend on its build task").with_fact("task_id", task.id())
            );
        }
    }

    for task in task_ids {
        if pipeline.tasks().iter().all(|candidate| candidate.id() != *task) {
            return Err(invalid("pipeline task id set contains missing task").with_fact("task_id", task));
        }
    }

    Ok(())
}

fn is_join_build_for_probe(pipeline: &CompiledPipeline, build_id: PipelineTaskId, probe_id: PipelineTaskId) -> bool {
    let Some(build) = pipeline.tasks().iter().find(|task| task.id() == build_id) else {
        return false;
    };
    let Some(probe) = pipeline.tasks().iter().find(|task| task.id() == probe_id) else {
        return false;
    };
    build.kind() == PipelineTaskKind::JoinBuild
        && probe.kind() == PipelineTaskKind::JoinProbe
        && (build.plan_node().is_none() || probe.plan_node().is_none() || build.plan_node() == probe.plan_node())
}

fn validate_task_graph_is_acyclic(pipeline: &CompiledPipeline) -> crate::Result<()> {
    let adjacency = adjacency(pipeline);
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();

    for task in pipeline.tasks() {
        visit_task(task.id(), &adjacency, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn visit_task(
    task: PipelineTaskId,
    adjacency: &BTreeMap<PipelineTaskId, Vec<PipelineTaskId>>,
    visiting: &mut BTreeSet<PipelineTaskId>,
    visited: &mut BTreeSet<PipelineTaskId>,
) -> crate::Result<()> {
    if visited.contains(&task) {
        return Ok(());
    }
    if !visiting.insert(task) {
        return Err(invalid("pipeline task dependency graph contains a cycle").with_fact("task_id", task));
    }
    for downstream in adjacency.get(&task).into_iter().flatten() {
        visit_task(*downstream, adjacency, visiting, visited)?;
    }
    visiting.remove(&task);
    visited.insert(task);
    Ok(())
}

fn validate_task_resource_references(
    pipeline: &CompiledPipeline,
    resource_ids: &BTreeSet<ResidentResourceId>,
) -> crate::Result<()> {
    for task in pipeline.tasks() {
        for resource in task.input_resources().iter().chain(task.output_resources()) {
            if !resource_ids.contains(resource) {
                return Err(invalid("pipeline task references missing resident resource")
                    .with_fact("task_id", task.id())
                    .with_fact("resource_id", resource));
            }
        }
    }
    for edge in pipeline.edges() {
        for resource in edge.resources() {
            if !resource_ids.contains(resource) {
                return Err(invalid("pipeline edge references missing resident resource")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource));
            }
        }
    }
    Ok(())
}

fn validate_resource_flow(pipeline: &CompiledPipeline) -> crate::Result<()> {
    for resource in pipeline.resources() {
        let producer = task_by_id(pipeline, resource.producer())?;
        if !producer.output_resources().contains(&resource.id()) {
            return Err(invalid("pipeline resident resource producer task does not output resource")
                .with_fact("resource_id", resource.id())
                .with_fact("task_id", resource.producer()));
        }

        for consumer_id in resource.consumers() {
            let consumer = task_by_id(pipeline, *consumer_id)?;
            if !consumer.input_resources().contains(&resource.id()) {
                return Err(invalid("pipeline resident resource consumer task does not input resource")
                    .with_fact("resource_id", resource.id())
                    .with_fact("task_id", consumer_id));
            }
            if !task_reaches(pipeline, resource.producer(), *consumer_id) {
                return Err(invalid("pipeline resident resource producer does not dominate consumer")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", consumer_id));
            }
            if !edge_carries_resource(pipeline, resource.producer(), *consumer_id, resource.id()) {
                return Err(invalid("pipeline resident resource consumer has no carrying dependency edge")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", consumer_id));
            }
        }

        match resource.lifetime() {
            ResidentLifetime::TaskLocal { task } => {
                if task != resource.producer() {
                    return Err(invalid("pipeline task-local lifetime does not match resource producer")
                        .with_fact("resource_id", resource.id())
                        .with_fact("lifetime_task", task)
                        .with_fact("producer_task", resource.producer()));
                }
                if !resource.consumers().is_empty() {
                    return Err(invalid("pipeline task-local lifetime has external consumers")
                        .with_fact("resource_id", resource.id())
                        .with_fact("consumer_count", resource.consumers().len()));
                }
            },
            ResidentLifetime::BetweenTasks { producer } => {
                if producer != resource.producer() {
                    return Err(invalid("pipeline between-tasks lifetime producer does not match resource producer")
                        .with_fact("resource_id", resource.id())
                        .with_fact("lifetime_producer", producer)
                        .with_fact("producer_task", resource.producer()));
                }
                if resource.consumers().is_empty() {
                    return Err(invalid("pipeline between-tasks lifetime has no consumers")
                        .with_fact("resource_id", resource.id()));
                }
            },
            ResidentLifetime::SinkOutput { producer } => {
                if producer != resource.producer() {
                    return Err(invalid("pipeline sink-output lifetime producer does not match resource producer")
                        .with_fact("resource_id", resource.id())
                        .with_fact("lifetime_producer", producer)
                        .with_fact("producer_task", resource.producer()));
                }
                if !resource.consumers().is_empty() {
                    return Err(invalid("pipeline sink-output lifetime has task consumers")
                        .with_fact("resource_id", resource.id())
                        .with_fact("consumer_count", resource.consumers().len()));
                }
            },
        }
        validate_shared_resident_resource_edges(pipeline, resource)?;
    }

    for edge in pipeline.edges() {
        for resource_id in edge.resources() {
            let resource = resource_by_id(pipeline, *resource_id)?;
            if resource.producer() != edge.from() {
                return Err(invalid("pipeline edge resource producer does not match edge producer")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource_id)
                    .with_fact("edge_producer", edge.from())
                    .with_fact("resource_producer", resource.producer()));
            }
            if !resource.consumers().contains(&edge.to()) {
                return Err(invalid("pipeline edge resource consumer is not recorded on resource")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource_id)
                    .with_fact("edge_consumer", edge.to()));
            }
            if edge.kind() == PipelineEdgeKind::SharedResidentProducer && resource.consumers().len() < 2 {
                return Err(invalid("pipeline shared resident producer edge does not fan out to multiple consumers")
                    .with_fact("edge_id", edge.id())
                    .with_fact("resource_id", resource_id)
                    .with_fact("consumer_count", resource.consumers().len()));
            }
        }
    }

    validate_task_input_resource_consumers(pipeline)?;

    Ok(())
}

fn validate_shared_resident_resource_edges(
    pipeline: &CompiledPipeline,
    resource: &ResidentResource,
) -> crate::Result<()> {
    let shared_edges = pipeline
        .edges()
        .iter()
        .filter(|edge| {
            edge.kind() == PipelineEdgeKind::SharedResidentProducer && edge.resources().contains(&resource.id())
        })
        .collect::<Vec<_>>();
    if shared_edges.is_empty() {
        return Ok(());
    }

    if resource.consumers().len() < 2 {
        return Err(invalid("pipeline shared resident resource does not fan out to multiple consumers")
            .with_fact("resource_id", resource.id())
            .with_fact("consumer_count", resource.consumers().len()));
    }

    for edge in &shared_edges {
        if edge.resources() != [resource.id()] {
            return Err(invalid("pipeline shared resident producer edge must carry exactly one resource identity")
                .with_fact("edge_id", edge.id())
                .with_fact("resource_id", resource.id())
                .with_fact("edge_resource_count", edge.resources().len()));
        }
    }

    let shared_consumers = shared_edges.iter().map(|edge| edge.to()).collect::<BTreeSet<_>>();
    let recorded_consumers = resource.consumers().iter().copied().collect::<BTreeSet<_>>();
    if shared_consumers != recorded_consumers {
        let missing = recorded_consumers.difference(&shared_consumers).next().copied();
        let extra = shared_consumers.difference(&recorded_consumers).next().copied();
        let mut error = invalid("pipeline shared resident producer edges do not cover recorded consumers")
            .with_fact("resource_id", resource.id())
            .with_fact("producer_task", resource.producer());
        if let Some(missing) = missing {
            error = error.with_fact("missing_consumer_task", missing);
        }
        if let Some(extra) = extra {
            error = error.with_fact("extra_consumer_task", extra);
        }
        return Err(error);
    }

    Ok(())
}

fn validate_task_input_resource_consumers(pipeline: &CompiledPipeline) -> crate::Result<()> {
    for task in pipeline.tasks() {
        for resource_id in task.input_resources() {
            let resource = resource_by_id(pipeline, *resource_id)?;
            if !resource.consumers().contains(&task.id()) {
                return Err(invalid("pipeline task input resource consumer is not recorded on resource")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", task.id()));
            }
            if !edge_carries_resource(pipeline, resource.producer(), task.id(), resource.id()) {
                return Err(invalid("pipeline task input resource has no carrying dependency edge")
                    .with_fact("resource_id", resource.id())
                    .with_fact("producer_task", resource.producer())
                    .with_fact("consumer_task", task.id()));
            }
        }
    }
    Ok(())
}

fn validate_barrier_dominance(pipeline: &CompiledPipeline) -> crate::Result<()> {
    for barrier in pipeline.barriers() {
        if barrier.upstream_tasks().is_empty() {
            return Err(invalid("pipeline barrier has no upstream tasks").with_fact("barrier_id", barrier.id()));
        }

        for downstream in barrier.downstream_tasks() {
            if !barrier.upstream_tasks().iter().any(|upstream| task_reaches(pipeline, *upstream, *downstream)) {
                return Err(invalid("pipeline barrier upstream tasks do not dominate downstream task")
                    .with_fact("barrier_id", barrier.id())
                    .with_fact("downstream_task", downstream));
            }
        }

        match barrier.kind() {
            PipelineBarrierKind::JoinBuildBeforeProbe => {
                let [build] = barrier.upstream_tasks() else {
                    return Err(invalid("pipeline join build-before-probe barrier must have one build task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("upstream_count", barrier.upstream_tasks().len()));
                };
                let [probe] = barrier.downstream_tasks() else {
                    return Err(invalid("pipeline join build-before-probe barrier must have one probe task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("downstream_count", barrier.downstream_tasks().len()));
                };
                validate_barrier_task_pair(
                    pipeline,
                    barrier.id(),
                    *build,
                    PipelineTaskKind::JoinBuild,
                    *probe,
                    PipelineTaskKind::JoinProbe,
                )?;
            },
            PipelineBarrierKind::JoinMaterialization => {
                let [probe] = barrier.upstream_tasks() else {
                    return Err(invalid("pipeline join materialization barrier must have one probe task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("upstream_count", barrier.upstream_tasks().len()));
                };
                let [materialize] = barrier.downstream_tasks() else {
                    return Err(invalid("pipeline join materialization barrier must have one materialize task")
                        .with_fact("barrier_id", barrier.id())
                        .with_fact("downstream_count", barrier.downstream_tasks().len()));
                };
                validate_barrier_task_pair(
                    pipeline,
                    barrier.id(),
                    *probe,
                    PipelineTaskKind::JoinProbe,
                    *materialize,
                    PipelineTaskKind::JoinMaterialize,
                )?;
            },
            PipelineBarrierKind::AggregateBoundary => {
                validate_barrier_upstream_kind(
                    pipeline,
                    barrier.id(),
                    barrier.upstream_tasks(),
                    PipelineTaskKind::Aggregate,
                )?;
            },
            PipelineBarrierKind::SortBoundary => {
                validate_barrier_upstream_kind(
                    pipeline,
                    barrier.id(),
                    barrier.upstream_tasks(),
                    PipelineTaskKind::FinalSort,
                )?;
            },
            PipelineBarrierKind::SourceChunkBoundary => {
                validate_barrier_upstream_kind(
                    pipeline,
                    barrier.id(),
                    barrier.upstream_tasks(),
                    PipelineTaskKind::SourceRead,
                )?;
            },
            PipelineBarrierKind::SinkBoundary => {},
        }
    }
    Ok(())
}

fn validate_barrier_task_pair(
    pipeline: &CompiledPipeline,
    barrier: PipelineBarrierId,
    upstream_id: PipelineTaskId,
    expected_upstream: PipelineTaskKind,
    downstream_id: PipelineTaskId,
    expected_downstream: PipelineTaskKind,
) -> crate::Result<()> {
    let upstream = task_by_id(pipeline, upstream_id)?;
    let downstream = task_by_id(pipeline, downstream_id)?;
    if upstream.kind() != expected_upstream {
        return Err(invalid("pipeline barrier upstream task has wrong kind")
            .with_fact("barrier_id", barrier)
            .with_fact("task_id", upstream_id)
            .with_fact("expected_kind", expected_upstream.as_str())
            .with_fact("actual_kind", upstream.kind().as_str()));
    }
    if downstream.kind() != expected_downstream {
        return Err(invalid("pipeline barrier downstream task has wrong kind")
            .with_fact("barrier_id", barrier)
            .with_fact("task_id", downstream_id)
            .with_fact("expected_kind", expected_downstream.as_str())
            .with_fact("actual_kind", downstream.kind().as_str()));
    }
    if upstream.plan_node().is_some()
        && downstream.plan_node().is_some()
        && upstream.plan_node() != downstream.plan_node()
    {
        return Err(invalid("pipeline join barrier tasks do not belong to the same plan node")
            .with_fact("barrier_id", barrier)
            .with_fact("upstream_task", upstream_id)
            .with_fact("downstream_task", downstream_id));
    }
    Ok(())
}

fn validate_barrier_upstream_kind(
    pipeline: &CompiledPipeline,
    barrier: PipelineBarrierId,
    upstream_tasks: &[PipelineTaskId],
    expected: PipelineTaskKind,
) -> crate::Result<()> {
    for task_id in upstream_tasks {
        let task = task_by_id(pipeline, *task_id)?;
        if task.kind() != expected {
            return Err(invalid("pipeline barrier upstream task has wrong kind")
                .with_fact("barrier_id", barrier)
                .with_fact("task_id", task_id)
                .with_fact("expected_kind", expected.as_str())
                .with_fact("actual_kind", task.kind().as_str()));
        }
    }
    Ok(())
}

fn validate_gate_references(pipeline: &CompiledPipeline, gate_ids: &BTreeSet<ReservationGateId>) -> crate::Result<()> {
    for task in pipeline.tasks() {
        for gate in task.gates() {
            if !gate_ids.contains(gate) {
                return Err(invalid("pipeline task references missing reservation gate")
                    .with_fact("task_id", task.id())
                    .with_fact("gate_id", gate));
            }
        }
    }
    for edge in pipeline.edges() {
        for gate in edge.gates() {
            if !gate_ids.contains(gate) {
                return Err(invalid("pipeline edge references missing reservation gate")
                    .with_fact("edge_id", edge.id())
                    .with_fact("gate_id", gate));
            }
        }
    }
    for barrier in pipeline.barriers() {
        for gate in barrier.gates() {
            if !gate_ids.contains(gate) {
                return Err(invalid("pipeline barrier references missing reservation gate")
                    .with_fact("barrier_id", barrier.id())
                    .with_fact("gate_id", gate));
            }
        }
    }
    for gate in pipeline.sink().gates() {
        if !gate_ids.contains(gate) {
            return Err(
                invalid("pipeline sink boundary references missing reservation gate").with_fact("gate_id", gate)
            );
        }
    }
    Ok(())
}

fn validate_gate_owner_references(pipeline: &CompiledPipeline) -> crate::Result<()> {
    for gate in pipeline.gates() {
        let owner_references_gate = match gate.owner() {
            ReservationGateOwner::Task(task_id) => {
                pipeline.tasks().iter().any(|task| task.id() == task_id && task.gates().contains(&gate.id()))
            },
            ReservationGateOwner::Edge(edge_id) => {
                pipeline.edges().iter().any(|edge| edge.id() == edge_id && edge.gates().contains(&gate.id()))
            },
            ReservationGateOwner::Barrier(barrier_id) => pipeline
                .barriers()
                .iter()
                .any(|barrier| barrier.id() == barrier_id && barrier.gates().contains(&gate.id())),
            ReservationGateOwner::SinkBoundary => pipeline.sink().gates().contains(&gate.id()),
        };
        if !owner_references_gate {
            return Err(invalid("pipeline reservation gate owner does not reference gate")
                .with_fact("gate_id", gate.id())
                .with_fact("gate_owner", gate.owner().to_string()));
        }
    }
    Ok(())
}

fn validate_reservation_gate_contracts(pipeline: &CompiledPipeline) -> crate::Result<()> {
    for gate in pipeline.gates() {
        if gate.memory_boundary().is_empty() {
            return Err(invalid("pipeline reservation gate has empty memory boundary").with_fact("gate_id", gate.id()));
        }
        if gate.required_contract().is_empty() {
            return Err(
                invalid("pipeline reservation gate has empty required contract").with_fact("gate_id", gate.id())
            );
        }
        validate_gate_scope(gate.id(), gate.owner(), gate.requirement().scope, gate.reservation_kind())?;
        validate_unresolved_preflight_label(
            gate.id(),
            gate.reservation_kind(),
            gate.required_contract(),
            gate.status(),
            gate.quality(),
        )?;
    }
    Ok(())
}

fn validate_gate_scope(
    gate: ReservationGateId,
    owner: ReservationGateOwner,
    scope: ReservationRequirementScope,
    kind: ReservationKind,
) -> crate::Result<()> {
    let valid = match (scope, owner) {
        (ReservationRequirementScope::Node, ReservationGateOwner::Task(_)) => true,
        (ReservationRequirementScope::Edge, ReservationGateOwner::Edge(_)) => true,
        (ReservationRequirementScope::Edge, ReservationGateOwner::SinkBoundary)
            if kind == ReservationKind::Materialization =>
        {
            true
        },
        (ReservationRequirementScope::Subgraph, ReservationGateOwner::Barrier(_)) => true,
        _ => false,
    };
    if !valid {
        return Err(invalid("pipeline reservation gate owner does not match requirement scope")
            .with_fact("gate_id", gate)
            .with_fact("gate_owner", owner.to_string())
            .with_fact("requirement_scope", scope.as_str())
            .with_fact("reservation_kind", kind.as_str()));
    }
    Ok(())
}

fn validate_unresolved_preflight_label(
    gate: ReservationGateId,
    kind: ReservationKind,
    required_contract: &'static str,
    status: ContractStatus,
    quality: ContractQuality,
) -> crate::Result<()> {
    if status == ContractStatus::Modeled
        && is_unresolved_preflight_contract(kind, required_contract)
        && quality != ContractQuality::HardBudgetedReportWithUnresolvedPreflight
    {
        return Err(invalid("pipeline hard-budgeted gate is not labeled as unresolved preflight")
            .with_fact("gate_id", gate)
            .with_fact("reservation_kind", kind.as_str())
            .with_fact("contract_status", status.as_str())
            .with_fact("contract_quality", quality.as_str()));
    }
    Ok(())
}

fn is_unresolved_preflight_contract(kind: ReservationKind, required_contract: &'static str) -> bool {
    lower_level_unresolved_preflight_gap(kind, required_contract).is_some()
}

fn task_by_id(
    pipeline: &CompiledPipeline,
    task_id: PipelineTaskId,
) -> crate::Result<&crate::pipeline::model::PipelineTask> {
    pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == task_id)
        .ok_or_else(|| invalid("pipeline task does not exist").with_fact("task_id", task_id))
}

fn resource_by_id(
    pipeline: &CompiledPipeline,
    resource_id: ResidentResourceId,
) -> crate::Result<&crate::pipeline::model::ResidentResource> {
    pipeline
        .resources()
        .iter()
        .find(|resource| resource.id() == resource_id)
        .ok_or_else(|| invalid("pipeline resident resource does not exist").with_fact("resource_id", resource_id))
}

fn adjacency(pipeline: &CompiledPipeline) -> BTreeMap<PipelineTaskId, Vec<PipelineTaskId>> {
    let mut adjacency = BTreeMap::<PipelineTaskId, Vec<PipelineTaskId>>::new();
    for edge in pipeline.edges() {
        adjacency.entry(edge.from()).or_default().push(edge.to());
    }
    adjacency
}

fn task_reaches(pipeline: &CompiledPipeline, from: PipelineTaskId, to: PipelineTaskId) -> bool {
    if from == to {
        return true;
    }
    let adjacency = adjacency(pipeline);
    let mut stack = vec![from];
    let mut visited = BTreeSet::new();
    while let Some(task) = stack.pop() {
        if !visited.insert(task) {
            continue;
        }
        for downstream in adjacency.get(&task).into_iter().flatten() {
            if *downstream == to {
                return true;
            }
            stack.push(*downstream);
        }
    }
    false
}

fn edge_carries_resource(
    pipeline: &CompiledPipeline,
    from: PipelineTaskId,
    to: PipelineTaskId,
    resource: ResidentResourceId,
) -> bool {
    pipeline.edges().iter().any(|edge| edge.from() == from && edge.to() == to && edge.resources().contains(&resource))
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
