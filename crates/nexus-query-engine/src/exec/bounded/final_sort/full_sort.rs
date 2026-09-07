use std::time::Instant;

use super::{
    CommittedPartial, CudfSortOptions, FinalSortExecutor, GpuDataFrame, LimitNode, Result, cudf_query_error,
    cudf_query_error_with_budget_details, estimated_limited_output_bytes, execute_limit_with_token,
    execution_overrides, final_sort_retained_bytes, limit_keeps_all_rows, lower_sort_spec,
};

impl FinalSortExecutor<'_, '_> {
    pub(super) fn full_sort_limit_attempt(
        &mut self,
        limit: Option<&LimitNode>,
        input: &CommittedPartial,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        self.full_sort_limit_attempt_from_frame(limit, &input.frame, attempt)
    }

    pub(super) fn full_sort_limit_attempt_from_frame(
        &mut self,
        limit: Option<&LimitNode>,
        input: &GpuDataFrame,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let sort = lower_sort_spec(&self.sort.spec)?;
        let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
        let (materialized_frame, materialized_reservation) = {
            let timer = Instant::now();
            let (prepared_sort, prepare_report) = {
                let prepare_token = self.final_sort_budget_token("final_sort", attempt)?;
                let prepared = input
                    .frame()
                    .clone()
                    .prepare_sort_values_with_budget(sort.columns.iter().map(String::as_str), options, &prepare_token)
                    .map_err(|error| {
                        cudf_query_error_with_budget_details(
                            "nexus_query_engine.final_sort.prepare",
                            "failed to prepare bounded native final sort order",
                            error,
                        )
                    })
                    .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
                prepared.into_parts()
            };
            self.execution.metrics_mut().operator_time += timer.elapsed();

            self.execution
                .metrics_mut()
                .record_budgeted_execution("nexus_query_engine.final_sort.prepare", &prepare_report);
            final_sort_retained_bytes(&prepare_report)
                .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;

            let timer = Instant::now();
            let materialize_overrides_lease = execution_overrides(self.execution.context())?;
            let materialize_overrides = materialize_overrides_lease.overrides();
            let materialized_output_bytes = prepared_sort
                .materialized_output_reservation_size_bytes_in(&materialize_overrides)
                .map_err(|error| {
                    cudf_query_error(
                        "nexus_query_engine.final_sort.materialize_size",
                        "failed to size bounded native final sort output",
                        error,
                    )
                })
                .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))?;
            let materialized_output_bytes = materialized_output_bytes as u64;
            let materialized_reservation =
                self.reserve_full_sort_materialized_output(materialized_output_bytes, attempt)?;
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
                        cudf_query_error_with_budget_details(
                            "nexus_query_engine.final_sort.materialize",
                            "failed to materialize bounded native final sort output",
                            error,
                        )
                    })
                    .map(|outcome| {
                        self.execution
                            .metrics_mut()
                            .record_budgeted_execution("nexus_query_engine.final_sort.materialize", outcome.report());
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
            self.execution.metrics_mut().operator_time += timer.elapsed();
            self.execution.metrics_mut().record_device_memory_observation(materialized_output_bytes);

            (materialized_frame, materialized_reservation)
        };

        if let Some(limit) = limit {
            if limit_keeps_all_rows(materialized_frame.num_rows(), limit.spec)? {
                return Ok(CommittedPartial::with_device_bytes(materialized_frame, materialized_reservation));
            }

            let output_reservation = self.reserve_estimated_final_sort_output(
                estimated_limited_output_bytes(materialized_reservation, materialized_frame.num_rows(), limit.spec)?,
                attempt,
            )?;
            let limit_token = self.final_sort_budget_token("final_sort_limit", attempt)?;
            let frame = match execute_limit_with_token(
                limit,
                materialized_frame,
                &limit_token,
                "nexus_query_engine.final_sort.limit",
                self.execution.metrics_mut(),
            )
            .map_err(|error| self.attach_final_sort_resource_facts(error, attempt))
            {
                Ok(frame) => frame,
                Err(error) => {
                    return Err(error);
                },
            };
            self.finalize_final_sort_output_reservation(frame, output_reservation, attempt)
        } else {
            Ok(CommittedPartial::with_device_bytes(materialized_frame, materialized_reservation))
        }
    }

    fn reserve_full_sort_materialized_output(&mut self, output_bytes: u64, attempt: usize) -> Result<u64> {
        let _ = attempt;
        Ok(output_bytes)
    }
}
