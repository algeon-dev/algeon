use std::time::Instant;

use super::{
    CommittedPartial, CudfSortOptions, FinalSortExecutor, GpuDataFrame, LimitNode, Result, cudf_query_error,
    execute_limit_with_token, final_sort_retained_bytes, limit_keeps_all_rows, lower_sort_spec,
};

impl FinalSortExecutor<'_, '_, '_> {
    pub(super) fn full_sort_limit_attempt(
        &mut self,
        limit: Option<&LimitNode>,
        input: &CommittedPartial,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        self.full_sort_limit_attempt_from_frame(limit, &input.frame, attempt)
    }

    /// One attempt at a complete sort of `input`, followed by the limit when one is given.
    ///
    /// Prepares the sort order, reads its retained bytes, sizes the output, and gathers the
    /// sorted frame under a second token. The borrowed input stays allocated throughout, so
    /// the sorted copy of the whole input coexists with it. A limit that keeps every row
    /// returns the sorted frame as is. There is no retry here; errors carry final sort
    /// resource facts for `attempt`.
    pub(super) fn full_sort_limit_attempt_from_frame(
        &mut self,
        limit: Option<&LimitNode>,
        input: &GpuDataFrame,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let sort = lower_sort_spec(&self.sort.spec)?;
        let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
        let (materialized_frame, materialized_output_bytes) = {
            let timer = Instant::now();
            let (prepared_sort, prepare_report) = {
                let prepare_token = self.final_sort_budget_token("final_sort", attempt)?;
                let prepared = prepare_token
                    .run("dataframe::prepare_sort_values", |exec| {
                        input
                            .frame()
                            .clone()
                            .on(exec)
                            .prepare_sort_values(sort.columns.iter().map(String::as_str), options)
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
            let materialize_runtime = self.execution.context().runtime()?;
            let materialized_output_bytes = prepared_sort
                .materialized_output_reservation_size_bytes(materialize_runtime.exec())
                .map_err(|error| {
                    cudf_query_error(
                        "query_engine.final_sort.materialize_size",
                        "failed to size bounded native final sort output",
                        error,
                    )
                })
                .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
            let materialized_output_bytes = materialized_output_bytes as u64;
            let materialized_frame = {
                let materialize_token = match self.final_sort_budget_token("final_sort", attempt) {
                    Ok(token) => token,
                    Err(error) => {
                        return Err(error);
                    },
                };
                match prepared_sort
                    .materialize_with_budget(&materialize_token)
                    .map_err(|error| {
                        cudf_query_error(
                            "query_engine.final_sort.materialize",
                            "failed to materialize bounded native final sort output",
                            error,
                        )
                    })
                    .map(|outcome| {
                        self.execution
                            .metrics()
                            .record_budgeted_execution("query_engine.final_sort.materialize", outcome.report());
                        GpuDataFrame::new(outcome.into_value())
                    })
                    .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
                {
                    Ok(frame) => frame,
                    Err(error) => {
                        return Err(error);
                    },
                }
            };
            self.execution.metrics().add_operator_time(timer.elapsed());
            self.execution.metrics().record_device_memory_observation(materialized_output_bytes);

            (materialized_frame, materialized_output_bytes)
        };

        if let Some(limit) = limit {
            if limit_keeps_all_rows(materialized_frame.num_rows(), limit.spec)? {
                return Ok(CommittedPartial::with_device_bytes(materialized_frame, materialized_output_bytes));
            }

            let limit_token = self.final_sort_budget_token("final_sort_limit", attempt)?;
            let frame = match execute_limit_with_token(
                limit,
                materialized_frame,
                &limit_token,
                "query_engine.final_sort.limit",
                self.execution.metrics(),
            )
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
            {
                Ok(frame) => frame,
                Err(error) => {
                    return Err(error);
                },
            };
            self.observe_final_sort_output(frame, attempt)
        } else {
            Ok(CommittedPartial::with_device_bytes(materialized_frame, materialized_output_bytes))
        }
    }
}
