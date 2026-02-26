use crate::{
    capability::ContractStatus,
    error::{QueryError, QueryErrorCode, Result},
    pipeline::{
        CompiledPipeline, PipelineTask, PipelineTaskId, ReservationGate, ReservationGateId, ReservationGateOwner,
    },
};

const OPERATION: &str = "nexus_query_engine.exec.bounded.compiled.reservations";

pub(super) fn validate_task_runtime_gates(pipeline: &CompiledPipeline, task: &PipelineTask) -> Result<()> {
    for gate in task.gates() {
        validate_gate(pipeline, *gate, ReservationGateOwner::Task(task.id()), task.id())?;
    }
    for edge in pipeline.edges().iter().filter(|edge| edge.from() == task.id() || edge.to() == task.id()) {
        for gate in edge.gates() {
            validate_gate(pipeline, *gate, ReservationGateOwner::Edge(edge.id()), task.id())?;
        }
    }
    for barrier in pipeline.barriers().iter().filter(|barrier| {
        barrier.upstream_tasks().contains(&task.id()) || barrier.downstream_tasks().contains(&task.id())
    }) {
        for gate in barrier.gates() {
            validate_gate(pipeline, *gate, ReservationGateOwner::Barrier(barrier.id()), task.id())?;
        }
    }
    if pipeline.sink().input() == task.id() {
        for gate in pipeline.sink().gates() {
            validate_gate(pipeline, *gate, ReservationGateOwner::SinkBoundary, task.id())?;
        }
    }
    Ok(())
}

fn validate_gate(
    pipeline: &CompiledPipeline,
    gate_id: ReservationGateId,
    expected_owner: ReservationGateOwner,
    task: PipelineTaskId,
) -> Result<()> {
    let gate = gate_by_id(pipeline, gate_id)?;
    if gate.owner() != expected_owner {
        return Err(invalid("compiled reservation gate owner does not match its runtime attachment")
            .with_fact("compiled_task_id", task)
            .with_fact("reservation_gate_id", gate_id)
            .with_fact("expected_gate_owner", expected_owner)
            .with_fact("actual_gate_owner", gate.owner()));
    }
    if gate.status() != ContractStatus::Modeled {
        return Err(invalid("compiled reservation gate is not modeled for runtime execution")
            .with_fact("compiled_task_id", task)
            .with_fact("reservation_gate_id", gate_id)
            .with_fact("reservation_kind", gate.reservation_kind())
            .with_fact("reservation_status", gate.status().as_str())
            .with_fact("reservation_quality", gate.quality().as_str())
            .with_fact("memory_boundary", gate.memory_boundary())
            .with_fact("required_contract", gate.required_contract()));
    }
    Ok(())
}

fn gate_by_id(pipeline: &CompiledPipeline, gate: ReservationGateId) -> Result<&ReservationGate> {
    pipeline.gates().iter().find(|candidate| candidate.id() == gate).ok_or_else(|| {
        invalid("compiled reservation gate id is missing from the pipeline").with_fact("reservation_gate_id", gate)
    })
}

fn invalid(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, OPERATION, message)
}
