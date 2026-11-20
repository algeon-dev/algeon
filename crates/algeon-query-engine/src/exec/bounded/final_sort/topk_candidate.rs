//! Sorting one top-k candidate before its truncation to `keep_rows`.
//!
//! A candidate first drops the rows that cannot be among its first `keep_rows`: it keeps the
//! rows whose first sort key sorts no later than the `keep_rows`-th value of that key, ties
//! included, and sorts only those survivors by the full key list. A row that the full sort
//! places among the first `keep_rows` has a first key no worse than that boundary, so it
//! survives, and the rows tied on the boundary stay for the later keys to order. The
//! survivors therefore sort to the same leading `keep_rows` rows as the whole candidate,
//! with the existing caveat that rows tied on every key have no defined order.
//!
//! The boundary and the ties follow the sort's ordering of the first key
//! (`SeriesOps::top_k_with_ties_mask`): its direction and null placement, nulls equal to
//! each other, NaN equal to NaN and after every other value, `-0.0` equal to `0.0`. Leading
//! nulls (NULLS FIRST) all survive when there are fewer than `keep_rows` of them, and are
//! the only survivors otherwise. Trailing nulls survive only when fewer than `keep_rows`
//! rows are non-null, and then every row survives.
//!
//! The reduction runs when the first key is fixed-width (integers, floating point, boolean,
//! decimal, timestamp, duration) and the candidate holds at least
//! `PREFILTER_MIN_ROWS_PER_KEPT_ROW` rows per kept row. Integral, decimal, boolean and
//! temporal keys take libcudf's partial top-k selection, after their null rows are
//! compacted away when present; floating-point keys take a single-column sort of the first
//! key, which still avoids the multi-key sort and the full-width gather of every row. Other first keys (strings, nested types) and candidates at most a few times
//! `keep_rows` sort whole. A first key with few distinct values can keep most rows; then
//! the reduction costs one top-k, one comparison pass and one filter that shares the input
//! columns when every row survives.

use std::time::Instant;

use super::{
    CudfSortOptions, DataFrame, ExecutionBudgetReport, ExecutionMetrics, GpuDataFrame, Result, SortNode,
    cudf_query_error, lower_sort_spec,
};

/// Minimum candidate rows per kept row for the first-key reduction. Below it, at least a
/// quarter of the candidate survives even with distinct keys, and the extra top-k and
/// filter pass buys little over sorting the candidate whole.
const PREFILTER_MIN_ROWS_PER_KEPT_ROW: usize = 4;

/// Sorts the rows of one candidate frame that can be among its first `keep_rows`, by the
/// node's sort keys under `work_token`; the caller truncates the result to `keep_rows`.
/// Consumes `input`.
///
/// The output's first `keep_rows` rows are those of a complete sort of `input`; rows after
/// them may be missing. `boundary` labels failures.
pub(super) fn sort_topk_candidate_with_budget(
    sort: &SortNode,
    input: GpuDataFrame,
    keep_rows: usize,
    work_token: &cudf::ExecutionBudgetToken,
    boundary: &'static str,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&sort.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let timer = Instant::now();
    let (frame, reduce_report) =
        retain_first_key_top_k_with_ties(input.into_frame(), &sort.columns[0], &options, keep_rows, work_token)
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.final_sort.topk_prefilter",
                    "failed to reduce bounded native final sort top-k candidate",
                    error,
                )
                .with_fact_if_absent("memory_boundary", boundary)
                .with_fact_if_absent("stage", "final_sort")
            })?;
    if let Some(report) = reduce_report {
        metrics.record_budgeted_execution("query_engine.final_sort.topk_prefilter", &report);
    }
    // The closure takes `frame`, so the prepared sort is the frame's only owner and the
    // gather releases each input column after its sorted copy.
    let prepared = work_token
        .run("dataframe::prepare_sort_values", move |exec| {
            frame.on(exec).prepare_sort_values(sort.columns.iter().map(String::as_str), options)
        })
        .map_err(|error| {
            cudf_query_error(
                "query_engine.final_sort.topk_prepare",
                "failed to prepare bounded native final sort top-k candidate",
                error,
            )
            .with_fact_if_absent("memory_boundary", boundary)
            .with_fact_if_absent("stage", "final_sort")
        })?;
    metrics.record_budgeted_execution("query_engine.final_sort.topk_prepare", prepared.report());
    let materialized = prepared.into_value().materialize_with_budget(work_token).map_err(|error| {
        cudf_query_error(
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

/// Keeps the rows of `frame` whose `first_key` value sorts no later than its `keep_rows`-th
/// under the first entries of `options`, with the reduction's budget report, when the
/// module's reduction rule applies; returns `frame` unchanged and no report otherwise.
/// `frame` is released when this returns, so only the survivors stay allocated.
fn retain_first_key_top_k_with_ties(
    frame: DataFrame,
    first_key: &str,
    options: &CudfSortOptions,
    keep_rows: usize,
    work_token: &cudf::ExecutionBudgetToken,
) -> cudf::Result<(DataFrame, Option<ExecutionBudgetReport>)> {
    if keep_rows.saturating_mul(PREFILTER_MIN_ROWS_PER_KEPT_ROW) > frame.num_rows() {
        return Ok((frame, None));
    }
    let key = frame.series(first_key)?;
    if !key.dtype()?.id().is_fixed_width() {
        return Ok((frame, None));
    }
    let key_options = CudfSortOptions::new()
        .with_descending(vec![options.descending[0]])
        .with_nulls_first(vec![options.nulls_first[0]]);
    let survivors = work_token.run("dataframe::top_k_with_ties_filter", |exec| {
        let mask = key.on(exec).top_k_with_ties_mask(keep_rows, &key_options)?;
        frame.on(exec).filter(&mask)
    })?;
    let (survivors, report) = survivors.into_parts();
    Ok((survivors, Some(report)))
}
