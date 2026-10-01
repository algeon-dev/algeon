use super::{
    AggregateFunction, AggregateSpec, BoundedExecutionAttempt, BoundedPlan, CommittedPartial, GpuDataFrame, Result,
    RuntimeResourceFacts, aggregate_spec_has_avg, cudf_query_error, execution_budget_token_for_boundary,
    finalize_avg_output_columns_with_budget, observe_device_memory_bytes,
};

pub(in crate::exec::bounded::aggregate) fn execute_final_aggregate_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    if !aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
        return Ok(input);
    }

    let expected_output_names =
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let result = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        let token_budget = execution.device_budget_bytes();
        let token = match execution_budget_token_for_boundary(
            execution.context(),
            token_budget,
            "final_aggregate_projection",
            "query_engine.final_projection.budget_token",
            "failed to create bounded native final aggregate projection budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                return Err(RuntimeResourceFacts {
                    memory_boundary: "final_aggregate_projection",
                    stage: "final_projection",
                    node_id: bounded.aggregate_id,
                    attempt: 1,
                    max_attempts: 1,
                    resume_boundary: "final_projection",
                    metrics: execution.metrics(),
                }
                .attach_to(error));
            },
        };
        finalize_avg_output_columns_with_budget(
            execution.context(),
            input.frame.into_frame(),
            &bounded.aggregate.0.spec,
            bounded.aggregate.1,
            &token,
            execution.metrics(),
        )
    } else {
        Ok(input.frame.into_frame())
    }
    .and_then(|frame| {
        frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            cudf_query_error(
                "query_engine.final_projection.select",
                "failed to select bounded native final aggregate output columns",
                error,
            )
        })
    })
    .map(GpuDataFrame::new);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_projection",
                stage: "final_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "final_projection",
                metrics: execution.metrics(),
            }
            .attach_to(error));
        },
    };

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics()).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_projection_output",
            stage: "final_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "final_projection",
            metrics: execution.metrics(),
        }
        .attach_to(error)
    })?;
    Ok(CommittedPartial::with_device_bytes(frame, output_bytes))
}

pub(in crate::exec::bounded::aggregate) fn aggregate_spec_has_global_final_state(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_avg(spec)
        || spec.group_exprs.is_empty()
            && spec.aggr_exprs.iter().any(|expr| {
                matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            })
}
