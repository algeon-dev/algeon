use std::{fmt::Write, fs, path::Path, time::Duration};

use datafusion::{common::DataFusionError, physical_plan::displayable};
use query_runtime::observability::{SealedRuntimeObservationSnapshot, write_runtime_observation_jsonl};

use super::{
    DataFusionResult,
    execution::{PlannedNativeQuery, TimedNativeQueryRun, TimedQueryRun},
};

const SAMPLES_HEADER: &str = "sample\tcpu_execution_ms\tnative_planning_ms\tnative_execution_ms\tnative_plan_plus_execution_ms\tnative_iteration_wall_ms\n";

pub(super) fn write_measured_plan(
    dir: &Path,
    query_id: &str,
    planned: &PlannedNativeQuery,
    measured_planning: Duration,
) -> DataFusionResult<()> {
    let observation_name = format!("{query_id}.observation.jsonl");
    let text = format!(
        "source=first_measured_execution\nmeasured_physical_planning_ms={:.3}\nruntime_observation_file={observation_name}\n\nphysical_plan:\n{}\n\nplanning_row:\n{:#?}\n",
        milliseconds(measured_planning),
        displayable(planned.plan.as_ref()).indent(true),
        planned.planning_row,
    );
    fs::write(dir.join(format!("{query_id}.plan.txt")), text)
        .map_err(|error| DataFusionError::External(Box::new(error)))
}

pub(super) fn write_observation(
    dir: &Path,
    query_id: &str,
    observation: &SealedRuntimeObservationSnapshot,
) -> DataFusionResult<()> {
    let footer = observation.footer();
    if !footer.complete || footer.truncated || footer.degraded || footer.counter_overflow {
        return Err(DataFusionError::Execution(format!("{query_id} measured runtime observation is incomplete")));
    }
    let mut bytes = Vec::new();
    write_runtime_observation_jsonl(observation, &mut bytes)
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    fs::write(dir.join(format!("{query_id}.observation.jsonl")), bytes)
        .map_err(|error| DataFusionError::External(Box::new(error)))
}

pub(super) fn write_samples(
    dir: &Path,
    query_id: &str,
    cpu: Option<&TimedQueryRun>,
    native: Option<&TimedNativeQueryRun>,
) -> DataFusionResult<()> {
    let mut output = String::from(SAMPLES_HEADER);
    let sample_count = cpu.map_or(0, |run| run.samples.len()).max(native.map_or(0, |run| run.samples.len()));
    for index in 0..sample_count {
        let cpu_ms =
            cpu.and_then(|run| run.samples.get(index)).map(|duration| format!("{:.3}", milliseconds(*duration)));
        let planning_ms = native
            .and_then(|run| run.planning_samples.get(index))
            .map(|duration| format!("{:.3}", milliseconds(*duration)));
        let execution_ms =
            native.and_then(|run| run.samples.get(index)).map(|duration| format!("{:.3}", milliseconds(*duration)));
        let plan_plus_execution_ms = native
            .and_then(|run| run.plan_plus_execution_samples.get(index))
            .map(|duration| format!("{:.3}", milliseconds(*duration)));
        let iteration_wall_ms = native
            .and_then(|run| run.iteration_wall_samples.get(index))
            .map(|duration| format!("{:.3}", milliseconds(*duration)));
        let _ = writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}",
            index + 1,
            cpu_ms.as_deref().unwrap_or(""),
            planning_ms.as_deref().unwrap_or(""),
            execution_ms.as_deref().unwrap_or(""),
            plan_plus_execution_ms.as_deref().unwrap_or(""),
            iteration_wall_ms.as_deref().unwrap_or(""),
        );
    }
    fs::write(dir.join(format!("{query_id}.samples.tsv")), output)
        .map_err(|error| DataFusionError::External(Box::new(error)))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
