use std::fmt;

use crate::{
    capability::{ContractQuality, ContractStatus},
    pipeline::model::{
        CompiledPipeline, PipelineBarrierId, PipelineEdgeId, PipelineTaskId, ReservationGateId, ReservationGateOwner,
        ResidentLifetime, ResidentResourceId,
    },
};

impl fmt::Display for PipelineTaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.index())
    }
}

impl fmt::Display for PipelineEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.index())
    }
}

impl fmt::Display for PipelineBarrierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b{}", self.index())
    }
}

impl fmt::Display for ResidentResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.index())
    }
}

impl fmt::Display for ReservationGateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}", self.index())
    }
}

impl fmt::Display for ReservationGateOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task(task) => write!(f, "task:{task}"),
            Self::Edge(edge) => write!(f, "edge:{edge}"),
            Self::Barrier(barrier) => write!(f, "barrier:{barrier}"),
            Self::SinkBoundary => f.write_str("sink_boundary"),
        }
    }
}

impl fmt::Display for ResidentLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskLocal { task } => write!(f, "task_local({task})"),
            Self::BetweenTasks { producer, last_consumer } => {
                write!(f, "between_tasks(producer={producer},last_consumer={last_consumer})")
            },
            Self::SinkOutput { producer } => write!(f, "sink_output(producer={producer})"),
        }
    }
}

impl fmt::Display for CompiledPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CompiledPipeline")?;
        writeln!(f, "  tasks:")?;
        for task in self.tasks() {
            let node = task.plan_node().map_or_else(|| "none".to_string(), |node| node.to_string());
            writeln!(
                f,
                "    {} kind={} node={} inputs={} outputs={} gates={}",
                task.id(),
                task.kind().as_str(),
                node,
                format_ids(task.input_resources()),
                format_ids(task.output_resources()),
                format_ids(task.gates())
            )?;
        }

        writeln!(f, "  edges:")?;
        for edge in self.edges() {
            writeln!(
                f,
                "    {} {} -> {} kind={} resources={} gates={}",
                edge.id(),
                edge.from(),
                edge.to(),
                edge.kind().as_str(),
                format_ids(edge.resources()),
                format_ids(edge.gates())
            )?;
        }

        writeln!(f, "  barriers:")?;
        for barrier in self.barriers() {
            writeln!(
                f,
                "    {} kind={} upstream={} downstream={} gates={}",
                barrier.id(),
                barrier.kind().as_str(),
                format_ids(barrier.upstream_tasks()),
                format_ids(barrier.downstream_tasks()),
                format_ids(barrier.gates())
            )?;
        }

        writeln!(f, "  resources:")?;
        for resource in self.resources() {
            writeln!(
                f,
                "    {} producer={} consumers={} lifetime={}",
                resource.id(),
                resource.producer(),
                format_ids(resource.consumers()),
                resource.lifetime()
            )?;
        }

        writeln!(f, "  gates:")?;
        for gate in self.gates() {
            writeln!(
                f,
                "    {} owner={} kind={} boundary={} contract={} status={} quality={}",
                gate.id(),
                gate.owner(),
                gate.reservation_kind().as_str(),
                gate.memory_boundary(),
                gate.required_contract(),
                status_as_str(gate.status()),
                quality_as_str(gate.quality())
            )?;
        }

        let sink = self.sink();
        writeln!(f, "  sink: kind={} input={} gates={}", sink.kind().as_str(), sink.input(), format_ids(sink.gates()))
    }
}

fn format_ids<T: fmt::Display>(ids: &[T]) -> String {
    if ids.is_empty() { "none".to_string() } else { ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",") }
}

fn status_as_str(status: ContractStatus) -> &'static str {
    status.as_str()
}

fn quality_as_str(quality: ContractQuality) -> &'static str {
    quality.as_str()
}
