use std::time::Instant;

use super::{
    CudfSortOptions, ExecutionMetrics, GpuDataFrame, Result, SortNode, cudf_query_error_with_budget_details,
    lower_sort_spec,
};

pub(super) fn sort_topk_candidate_with_budget(
    sort: &SortNode,
    input: GpuDataFrame,
    work_token: &cudf::ExecutionBudgetToken,
    boundary: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&sort.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let timer = Instant::now();
    let prepared = input
        .into_frame()
        .prepare_sort_values_with_budget(sort.columns.iter().map(String::as_str), options, work_token)
        .map_err(|error| {
            cudf_query_error_with_budget_details(
                "query_engine.final_sort.topk_prepare",
                "failed to prepare bounded native final sort top-k candidate",
                error,
            )
            .with_fact_if_absent("memory_boundary", boundary)
            .with_fact_if_absent("stage", "final_sort")
        })?;
    metrics.record_budgeted_execution("query_engine.final_sort.topk_prepare", prepared.report());
    let materialized = prepared.into_value().materialize_with_budget(work_token).map_err(|error| {
        cudf_query_error_with_budget_details(
            "query_engine.final_sort.topk_materialize",
            "failed to materialize bounded native final sort top-k candidate",
            error,
        )
        .with_fact_if_absent("memory_boundary", boundary)
        .with_fact_if_absent("stage", "final_sort")
    })?;
    metrics.record_budgeted_execution("query_engine.final_sort.topk_materialize", materialized.report());
    metrics.add_operator_time(timer.elapsed());
    Ok(GpuDataFrame::new(materialized.into_value()))
}
