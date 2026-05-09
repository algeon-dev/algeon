use super::*;

pub(super) fn execute_topk_candidate_with_budget(
    sort: &SortNode,
    input: GpuDataFrame,
    keep_rows: usize,
    work_token: &cudf::ExecutionBudgetToken,
    limit_token: &cudf::ExecutionBudgetToken,
    boundary: &'static str,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&sort.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let timer = Instant::now();
    let prepared = input
        .into_frame()
        .prepare_sort_values_with_budget(sort.columns.iter().map(String::as_str), options, work_token)
        .map_err(|error| {
            QueryError::from_cudf_with_budget_details(
                "cudf_query_engine.final_sort.topk_prepare",
                "failed to prepare bounded native final sort top-k candidate",
                error,
            )
            .with_fact_if_absent("memory_boundary", boundary)
            .with_fact_if_absent("stage", "final_sort")
        })?;
    metrics.record_budgeted_execution("cudf_query_engine.final_sort.topk_prepare", prepared.report());
    let materialized = prepared.into_value().materialize_with_budget(work_token).map_err(|error| {
        QueryError::from_cudf_with_budget_details(
            "cudf_query_engine.final_sort.topk_materialize",
            "failed to materialize bounded native final sort top-k candidate",
            error,
        )
        .with_fact_if_absent("memory_boundary", boundary)
        .with_fact_if_absent("stage", "final_sort")
    })?;
    metrics.record_budgeted_execution("cudf_query_engine.final_sort.topk_materialize", materialized.report());
    let sorted = GpuDataFrame::new(materialized.into_value());
    let output = execute_limit_spec_with_token(
        LimitSpec::fetch(keep_rows),
        sorted,
        limit_token,
        "cudf_query_engine.final_sort.topk_limit",
        metrics,
    )?;
    metrics.operator_time += timer.elapsed();
    Ok(output)
}
