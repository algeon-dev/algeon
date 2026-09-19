//! `Display` formatting for compiled pipeline IR.
//!
//! Output is diagnostic text used by triage tools. The format is not part of
//! the report contract — bump the report TSV schema instead of parsing this.

use std::fmt;

use crate::pipeline::model::{
    CompiledPipeline, PipelineBarrierId, PipelineEdgeId, PipelineTaskId, ResidentLifetime, ResidentResourceId,
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

impl fmt::Display for ResidentLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TaskLocal { task } => write!(f, "task_local({task})"),
            Self::BetweenTasks { producer } => {
                write!(f, "between_tasks(producer={producer},release=after_all_consumers)")
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
                "    {} kind={} node={} inputs={} outputs={}",
                task.id(),
                task.kind().as_str(),
                node,
                format_ids(task.input_resources()),
                format_ids(task.output_resources())
            )?;
        }

        writeln!(f, "  edges:")?;
        for edge in self.edges() {
            writeln!(
                f,
                "    {} {} -> {} kind={} resources={}",
                edge.id(),
                edge.from(),
                edge.to(),
                edge.kind().as_str(),
                format_ids(edge.resources())
            )?;
        }

        writeln!(f, "  barriers:")?;
        for barrier in self.barriers() {
            writeln!(
                f,
                "    {} kind={} upstream={} downstream={}",
                barrier.id(),
                barrier.kind().as_str(),
                format_ids(barrier.upstream_tasks()),
                format_ids(barrier.downstream_tasks())
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

        let sink = self.sink();
        writeln!(f, "  sink: kind={} input={}", sink.kind().as_str(), sink.input())
    }
}

fn format_ids<T: fmt::Display>(ids: &[T]) -> String {
    if ids.is_empty() { "none".to_string() } else { ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",") }
}
