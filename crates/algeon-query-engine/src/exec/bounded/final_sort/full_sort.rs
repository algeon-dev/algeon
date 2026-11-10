use std::time::Instant;

use super::{
    CommittedPartial, CudfSortOptions, FinalSortExecutor, GpuDataFrame, LimitNode, Result, cudf_query_error,
    execute_limit_with_token, final_sort_retained_bytes, limit_keeps_all_rows, lower_sort_spec,
};

impl FinalSortExecutor<'_, '_, '_> {
    /// `full_sort_limit_attempt_from_frame` over a partial the caller keeps for the top-k
    /// rerun, so the whole input stays allocated until the attempt ends.
    pub(super) fn full_sort_limit_attempt(
        &mut self,
        limit: Option<&LimitNode>,
        input: &CommittedPartial,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        self.full_sort_limit_attempt_from_frame(limit, input.frame.clone(), attempt)
    }

    /// One attempt at a complete sort of `input`, followed by the limit when one is given.
    ///
    /// Prepares the sort order, then gathers the sorted frame one column at a time under a
    /// second token, releasing each input column after its sorted copy. When `input` is
    /// the only owner of its columns, the device peak is the input, the sort order and the
    /// largest output column. A limit that keeps every row returns the sorted frame as is.
    /// There is no retry here; errors carry final sort resource facts for `attempt`.
    pub(super) fn full_sort_limit_attempt_from_frame(
        &mut self,
        limit: Option<&LimitNode>,
        input: GpuDataFrame,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let sort = lower_sort_spec(&self.sort.spec)?;
        let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
        let timer = Instant::now();
        let (prepared_sort, prepare_report) = {
            let prepare_token = self.final_sort_budget_token("final_sort", attempt)?;
            let prepared = prepare_token
                .run("dataframe::prepare_sort_values", |exec| {
                    input.into_frame().on(exec).prepare_sort_values(sort.columns.iter().map(String::as_str), options)
                })
                .map_err(|error| {
                    cudf_query_error(
                        "query_engine.final_sort.prepare",
                        "failed to prepare bounded native final sort order",
                        error,
                    )
                })
                .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
            prepared.into_parts()
        };
        self.execution.metrics().add_operator_time(timer.elapsed());

        self.execution.metrics().record_budgeted_execution("query_engine.final_sort.prepare", &prepare_report);
        final_sort_retained_bytes(&prepare_report)
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;

        let timer = Instant::now();
        let materialize_token = self.final_sort_budget_token("final_sort", attempt)?;
        let sorted = prepared_sort
            .materialize_with_budget(&materialize_token)
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.final_sort.materialize",
                    "failed to materialize bounded native final sort output",
                    error,
                )
            })
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
        self.execution.metrics().record_budgeted_execution("query_engine.final_sort.materialize", sorted.report());
        self.execution.metrics().add_operator_time(timer.elapsed());
        let sorted = self.observe_final_sort_output(GpuDataFrame::new(sorted.into_value()), attempt)?;

        let Some(limit) = limit else {
            return Ok(sorted);
        };
        if limit_keeps_all_rows(sorted.frame.num_rows(), limit.spec)? {
            return Ok(sorted);
        }

        let limit_token = self.final_sort_budget_token("final_sort_limit", attempt)?;
        let frame = execute_limit_with_token(
            limit,
            sorted.frame,
            &limit_token,
            "query_engine.final_sort.limit",
            self.execution.metrics(),
        )
        .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
        self.observe_final_sort_output(frame, attempt)
    }
}
