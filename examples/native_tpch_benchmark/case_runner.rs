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

    Ok(NativeTpchBenchmarkReport { metadata, config: config.clone(), rows })
}

pub(super) fn benchmark_query_progress_line(
    config: &NativeTpchBenchmarkConfig,
    query: &TpchQuery,
    query_index: usize,
    query_count: usize,
) -> String {
    let (max_row_groups_per_chunk, max_source_chunk_bytes) = match config.native_memory_policy {
        NativeTpchMemoryPolicy::Bounded { max_row_groups_per_chunk, max_source_chunk_bytes, .. } => {
            (max_row_groups_per_chunk.to_string(), max_source_chunk_bytes.to_string())
        },
        NativeTpchMemoryPolicy::Unbounded => ("unbounded".to_string(), "unbounded".to_string()),
    };
    format!(
        "[native_tpch_benchmark] phase=query_start query={} index={}/{} dataset_mode={} execution_mode={} memory_policy={} max_row_groups_per_chunk={} max_source_chunk_bytes={} skip_cpu={}",
        query.id,
        query_index.saturating_add(1),
        query_count,
        config.dataset_mode.as_str(),
        config.native_pipeline_execution_mode.as_str(),
        config.native_memory_policy.as_str(),
        max_row_groups_per_chunk,
        max_source_chunk_bytes,
        std::env::var_os("CUDF_TPCH_BENCH_SKIP_CPU").is_some(),
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

    if planned.planning_row.outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
        if let Err(error) = execution::validate_decision_grade_native_rejection_row(query.id, &planned.planning_row) {
            return NativeTpchBenchmarkRow::planning_failure(query.id, error);
        }
        return NativeTpchBenchmarkRow::rejected(query.id, planned.planning_row);
    }
    if let Err(error) = execution::validate_decision_grade_native_planning_row(query.id, &planned.planning_row) {
        return NativeTpchBenchmarkRow::planning_failure(query.id, error);
    }

    let cpu = if std::env::var_os("CUDF_TPCH_BENCH_SKIP_CPU").is_some() {
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
