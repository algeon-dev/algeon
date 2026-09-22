use std::path::Path;

use super::*;
use crate::loaded_libraries::write_loaded_libraries;

pub(super) async fn run_native_tpch_benchmark(
    config: &NativeTpchBenchmarkConfig,
    work_dir: &Path,
) -> DataFusionResult<NativeTpchBenchmarkReport> {
    config.validate()?;
    let (_root, metadata) = dataset::prepare_benchmark_dataset(config, work_dir).await?;
    let runtime_resources = config.new_runtime_resources()?;
    let provenance = RunProvenance::collect(&runtime_resources);
    eprintln!("{}", run_start_line(config, &provenance));
    let artifact_dir = std::env::var_os("ALGEON_TPCH_BENCH_ARTIFACT_DIR").map(std::path::PathBuf::from);
    if let Some(dir) = artifact_dir.as_deref() {
        std::fs::create_dir_all(dir).map_err(|error| DataFusionError::External(Box::new(error)))?;
    }
    let mut rows = Vec::with_capacity(config.query_ids.len());
    let mut pending_observations = Vec::<(String, execution::ObservationCapture)>::new();

    let selected_queries = config.selected_queries();
    let query_count = selected_queries.len();
    for (query_index, query) in selected_queries.into_iter().enumerate() {
        eprintln!("{}", benchmark_query_progress_line(config, query, query_index, query_count));
        rows.push(
            run_query(config, &metadata, query, &runtime_resources, artifact_dir.as_deref(), &mut pending_observations)
                .await?,
        );
    }

    if let Some(dir) = artifact_dir.as_deref() {
        write_loaded_libraries(dir)?;
    }

    runtime_resources.shutdown()?;
    if let Some(dir) = artifact_dir.as_deref() {
        for (query_id, capture) in pending_observations {
            let observation = capture
                .lock()
                .map_err(|_| plan_error("benchmark observation capture lock was poisoned".to_string()))?
                .take()
                .ok_or_else(|| plan_error(format!("{query_id} measured execution did not seal an observation")))?;
            artifacts::write_observation(dir, &query_id, &observation)?;
        }
    }

    Ok(NativeTpchBenchmarkReport { metadata, config: config.clone(), devices: provenance.devices, rows })
}

pub(super) fn benchmark_query_progress_line(
    config: &NativeTpchBenchmarkConfig,
    query: &TpchQuery,
    query_index: usize,
    query_count: usize,
) -> String {
    let policy = config.native_memory_policy;
    let profile_name = policy.profile_name;
    let source_chunk_planning_mode = policy.source_chunk_planning_mode.as_str();
    let max_row_groups_per_chunk = policy.max_row_groups_per_chunk;
    let max_source_chunk_bytes = policy.max_source_chunk_bytes;
    format!(
        "[native_tpch_benchmark] stage=query_start query={} index={}/{} dataset_mode={} execution_mode={} relational_execution={} memory_policy={} native_profile={} source_chunk_planning_mode={} max_row_groups_per_chunk={} max_source_chunk_bytes={} skip_cpu={}",
        query.id,
        query_index.saturating_add(1),
        query_count,
        config.dataset_mode.as_str(),
        "bounded",
        config.native_execution.as_str(),
        config.native_memory_policy.as_str(),
        profile_name,
        source_chunk_planning_mode,
        max_row_groups_per_chunk,
        max_source_chunk_bytes,
        std::env::var_os("ALGEON_TPCH_BENCH_SKIP_CPU").is_some(),
    )
}

async fn run_query(
    config: &NativeTpchBenchmarkConfig,
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    runtime_resources: &NativeTpchRuntimeResources,
    artifact_dir: Option<&Path>,
    pending_observations: &mut Vec<(String, execution::ObservationCapture)>,
) -> DataFusionResult<NativeTpchBenchmarkRow> {
    let planned = match execution::plan_native_query(metadata, query, config).await {
        Ok(planned) => planned,
        Err(error) => return Ok(NativeTpchBenchmarkRow::planning_failure(query.id, error)),
    };

    if planned.planning_row.outcome != Some(CandidateOutcomeKind::Selected) {
        if let Err(error) = execution::validate_decision_grade_native_not_selected_row(query.id, &planned.planning_row)
        {
            return Ok(NativeTpchBenchmarkRow::planning_failure(query.id, error));
        }
        return Ok(NativeTpchBenchmarkRow::not_selected(query.id, planned.planning_row));
    }
    if let Err(error) = execution::validate_decision_grade_native_planning_row(query.id, &planned.planning_row) {
        return Ok(NativeTpchBenchmarkRow::planning_failure(query.id, error));
    }

    let cpu = if std::env::var_os("ALGEON_TPCH_BENCH_SKIP_CPU").is_some() {
        Err(execution::CpuRunFailure {
            runtime_started: false,
            message: "cpu_skipped_by_CUDF_TPCH_BENCH_SKIP_CPU".to_string(),
            details: Box::default(),
        })
    } else {
        execution::run_cpu_query_iterations(metadata, query, config).await
    };
    let native = execution::run_native_query_iterations(
        metadata,
        query,
        config,
        &planned.planning_row,
        runtime_resources,
        artifact_dir,
        pending_observations,
    )
    .await;
    if let Some(dir) = artifact_dir {
        artifacts::write_samples(dir, query.id, cpu.as_ref().ok(), native.as_ref().ok())?;
    }
    Ok(NativeTpchBenchmarkRow::accepted(query.id, planned.planning_row, cpu, native))
}
