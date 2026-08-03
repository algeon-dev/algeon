use super::*;

pub(super) async fn run_native_tpch_benchmark(
    config: &NativeTpchBenchmarkConfig,
    work_dir: &Path,
) -> DataFusionResult<NativeTpchBenchmarkReport> {
    config.validate()?;
    let (_root, metadata) = dataset::prepare_benchmark_dataset(config, work_dir).await?;
    let runtime_resources = config.new_runtime_resources()?;
    let mut rows = Vec::with_capacity(config.query_ids.len());

    let selected_queries = config.selected_queries();
    let query_count = selected_queries.len();
    for (query_index, query) in selected_queries.into_iter().enumerate() {
        eprintln!("{}", benchmark_query_progress_line(config, query, query_index, query_count));
        rows.push(run_query(config, &metadata, query, &runtime_resources).await);
    }

    runtime_resources.shutdown()?;

    Ok(NativeTpchBenchmarkReport { metadata, config: config.clone(), rows })
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
        "strict_bounded",
        config.native_execution.as_str(),
        config.native_memory_policy.as_str(),
        profile_name,
        source_chunk_planning_mode,
        max_row_groups_per_chunk,
        max_source_chunk_bytes,
        std::env::var_os("NEXUS_TPCH_BENCH_SKIP_CPU").is_some(),
    )
}

async fn run_query(
    config: &NativeTpchBenchmarkConfig,
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
    runtime_resources: &NativeTpchRuntimeResources,
) -> NativeTpchBenchmarkRow {
    let planned = match execution::plan_native_query(metadata, query, config).await {
        Ok(planned) => planned,
        Err(error) => return NativeTpchBenchmarkRow::planning_failure(query.id, error),
    };

    if planned.planning_row.outcome != Some(NativeDataFusionPlanningOutcome::Selected) {
        if let Err(error) = execution::validate_decision_grade_native_not_selected_row(query.id, &planned.planning_row)
        {
            return NativeTpchBenchmarkRow::planning_failure(query.id, error);
        }
        return NativeTpchBenchmarkRow::not_selected(query.id, planned.planning_row);
    }
    if let Err(error) = execution::validate_decision_grade_native_planning_row(query.id, &planned.planning_row) {
        return NativeTpchBenchmarkRow::planning_failure(query.id, error);
    }

    let cpu = if std::env::var_os("NEXUS_TPCH_BENCH_SKIP_CPU").is_some() {
        Err(execution::CpuRunFailure {
            runtime_started: false,
            message: "cpu_skipped_by_CUDF_TPCH_BENCH_SKIP_CPU".to_string(),
            details: execution::FailureDetails::default(),
        })
    } else {
        execution::run_cpu_query_iterations(metadata, query, config).await
    };
    let native =
        execution::run_native_query_iterations(metadata, query, config, &planned.planning_row, runtime_resources).await;
    NativeTpchBenchmarkRow::accepted(query.id, planned.planning_row, cpu, native)
}
