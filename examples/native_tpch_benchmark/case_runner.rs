use super::*;

pub(super) async fn run_native_tpch_benchmark(
    config: &NativeTpchBenchmarkConfig,
    work_dir: &Path,
) -> DataFusionResult<NativeTpchBenchmarkReport> {
    let (_root, metadata) = dataset::prepare_benchmark_dataset(work_dir).await?;
    let mut rows = Vec::with_capacity(config.query_ids.len());

    for query in config.selected_queries() {
        rows.push(run_query(config, &metadata, query).await);
    }

    Ok(NativeTpchBenchmarkReport { metadata, config: config.clone(), rows })
}

async fn run_query(
    config: &NativeTpchBenchmarkConfig,
    metadata: &TpchDatasetMetadata,
    query: &TpchQuery,
) -> NativeTpchBenchmarkRow {
    let planned = match execution::plan_native_query(metadata, query, config).await {
        Ok(planned) => planned,
        Err(error) => return NativeTpchBenchmarkRow::planning_failure(query.id, error.to_string()),
    };

    if planned.planning_row.outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
        return NativeTpchBenchmarkRow::rejected(query.id, planned.planning_row);
    }

    let cpu = execution::run_cpu_query_iterations(metadata, query, config).await.map_err(|error| error.to_string());
    let native = execution::run_native_query_iterations(metadata, query, config).await;
    NativeTpchBenchmarkRow::accepted(query.id, planned.planning_row, cpu, native)
}
