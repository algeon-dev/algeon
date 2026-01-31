//! CompiledPipeline invariant validation.

use std::collections::BTreeSet;

use crate::{
    error::{QueryError, QueryErrorCode},
    pipeline::model::{
        CompiledPipeline, PipelineBarrierId, PipelineEdgeId, PipelineTaskId, PipelineTaskKind, ReservationGateId,
        ReservationGateOwner, ResidentLifetime, ResidentResourceId,
    },
};

const OPERATION: &str = "cudf_query_engine.pipeline.validate";

pub(super) fn validate_pipeline(pipeline: &CompiledPipeline) -> crate::Result<()> {
    let task_ids = validate_tasks(pipeline)?;
    let edge_ids = validate_edges(pipeline, &task_ids)?;
    let barrier_ids = validate_barriers(pipeline, &task_ids)?;
    let resource_ids = validate_resources(pipeline, &task_ids)?;
    let gate_ids = validate_gates(pipeline, &task_ids, &edge_ids, &barrier_ids)?;

    validate_task_dependencies(pipeline, &task_ids)?;
    validate_task_resource_references(pipeline, &resource_ids)?;
    validate_gate_references(pipeline, &gate_ids)?;
    validate_gate_owner_references(pipeline)?;

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
        ResidentLifetime::BetweenTasks { producer, last_consumer } => {
            if !task_ids.contains(&producer) {
                return Err(invalid("pipeline resident lifetime producer task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", producer));
            }
            if !task_ids.contains(&last_consumer) {
                return Err(invalid("pipeline resident lifetime last consumer task does not exist")
                    .with_fact("resource_id", resource)
                    .with_fact("task_id", last_consumer));
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
        let has_build_dependency = pipeline.edges().iter().any(|edge| {
            edge.to() == task.id()
                && pipeline
                    .tasks()
                    .iter()
                    .any(|producer| producer.id() == edge.from() && producer.kind() == PipelineTaskKind::JoinBuild)
        });
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

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
