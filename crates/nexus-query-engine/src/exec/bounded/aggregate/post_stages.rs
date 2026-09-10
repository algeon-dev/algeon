use std::{sync::Arc, time::Instant};

use super::{
    AggregateFunction, AggregateSpec, BoundedExecutionAttempt, BoundedPlan, BoundedUnaryStage, CommittedPartial,
    ExecutionBudgetToken, ExecutionContext, GpuDataFrame, NodeId, QueryError, QueryMetrics, Result,
    RuntimeResourceFacts, aggregate_spec_has_avg, cast_count_output_columns_with_budget, execute_aggregate_with_budget,
    execute_bounded_generic_unary_consuming_input, execute_filter_with_budget, execute_global_aggregate_with_budget,
    execute_groupby_aggregate_with_budget_observed, execute_limit, execute_projection_with_budget,
    execution_budget_token_for_boundary, execution_overrides, final_global_aggregate_spec,
    final_global_aggregate_state_output_schema, finalize_avg_output_columns_with_budget, lower_final_aggregate_spec,
    observe_device_memory_bytes, record_default_grouped_aggregate_strategy_decision, validate_aggregate_output_names,
};
use crate::capability::AggregateExecutionStage;

pub(in crate::exec::bounded) fn execute_post_aggregate_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    execute_post_final_stages(execution, &bounded.post_aggregate_stages, input)
}

pub(in crate::exec::bounded) fn execute_post_sort_projections(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    execute_post_final_stages(execution, &bounded.post_sort_projections, input)
}

pub(in crate::exec::bounded) fn execute_standalone_limit(
    execution: &mut BoundedExecutionAttempt<'_>,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
) -> Result<CommittedPartial> {
    if bounded.sort.is_some() {
        return Ok(input);
    }
    let Some((node_id, limit)) = bounded.limit else {
        return Ok(input);
    };
    execute_post_final_stage(
        execution,
        node_id,
        "post_aggregate_limit",
        input,
        |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
        false,
    )
}

fn execute_post_final_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    stages: &[BoundedUnaryStage<'_>],
    mut input: CommittedPartial,
) -> Result<CommittedPartial> {
    for stage in stages {
        input = match stage {
            BoundedUnaryStage::Filter { node_id, node } => execute_post_final_stage(
                execution,
                *node_id,
                "post_aggregate_filter",
                input,
                |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        node,
                        input,
                        token.expect("budget token is created for bounded post-aggregate filter stages"),
                        metrics,
                    )
                },
                true,
            )?,
            BoundedUnaryStage::Projection { node_id, node, output_schema } => {
                debug_assert_eq!(node.output_schema().as_ref(), output_schema.as_ref());
                execute_post_final_stage(
                    execution,
                    *node_id,
                    "post_aggregate_projection",
                    input,
                    |context, input, token, metrics| {
                        execute_projection_with_budget(
                            context,
                            node,
                            input,
                            token.expect("budget token is created for bounded post-aggregate projection stages"),
                            metrics,
                        )
                    },
                    true,
                )?
            },
        };
    }
    Ok(input)
}

fn execute_post_final_stage(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    input: CommittedPartial,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
    use_budget_token: bool,
) -> Result<CommittedPartial> {
    execute_bounded_generic_unary_consuming_input(execution, node_id, stage, input, use_budget_token, execute)
}

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
            "nexus_query_engine.final_projection.budget_token",
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
                    metrics: execution.metrics_mut(),
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
            execution.metrics_mut(),
        )
    } else {
        Ok(input.frame.into_frame())
    }
    .and_then(|frame| {
        frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.final_projection.select",
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
                metrics: execution.metrics_mut(),
            }
            .attach_to(error));
        },
    };

    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut()).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_projection_output",
            stage: "final_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "final_projection",
            metrics: execution.metrics_mut(),
        }
        .attach_to(error)
    })?;
    Ok(CommittedPartial::with_device_bytes(frame, output_bytes))
}

pub(in crate::exec::bounded::aggregate) fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    record_strategy: bool,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if bounded.aggregate.0.spec.group_exprs.is_empty() {
        let final_spec = final_global_aggregate_spec(&bounded.aggregate.0.spec);
        let output_schema = if aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
            final_global_aggregate_state_output_schema(
                &bounded.aggregate.0.spec,
                bounded.aggregate_input_schema,
                bounded.aggregate.1,
            )?
        } else {
            Arc::clone(bounded.aggregate.1)
        };
        return execute_global_aggregate_with_budget(context, &final_spec, input, &output_schema, token, metrics);
    }

    if bounded.aggregate.0.spec.grouping_sets.is_some() && input.num_rows() == 0 {
        return execute_aggregate_with_budget(
            context,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            input,
            bounded.aggregate.1,
            token,
            metrics,
        );
    }

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        validate_aggregate_output_names(
            &aggregate,
            bounded.aggregate.1,
            "nexus_query_engine.final_merge.validate",
            "final aggregate output names must match output schema fields",
        )?;
    }
    let expected_output_names = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        aggregate.output_names.clone()
    } else {
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>()
    };

    let frame = input.into_frame();
    let input_rows = frame.num_rows() as u64;
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget_observed(
        Some(context),
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "nexus_query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.0.spec, bounded.aggregate.1, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.final_merge.select",
            "failed to select bounded native final aggregate output columns",
            error,
        )
    })?;
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    if record_strategy {
        record_default_grouped_aggregate_strategy_decision(
            metrics,
            Some(bounded.aggregate_id.to_string()),
            AggregateExecutionStage::FinalMerge,
            context.memory_control_policy().as_str(),
            context.aggregate_strategy_execution_options(),
            &bounded.aggregate.0.spec,
            bounded.aggregate_input_schema,
            input_rows,
            output_rows,
            elapsed,
        );
    }
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec::bounded::aggregate) fn aggregate_spec_has_global_final_state(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_avg(spec)
        || spec.group_exprs.is_empty()
            && spec.aggr_exprs.iter().any(|expr| {
                matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            })
}
