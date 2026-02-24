use super::*;
use crate::capability::AggregateExecutionStage;

pub(in crate::exec::bounded) fn execute_post_aggregate_stages(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_post_final_stages(context, &bounded.post_aggregate_stages, input, reservations, metrics)
}

pub(in crate::exec::bounded) fn execute_post_sort_projections(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_post_final_stages(context, &bounded.post_sort_projections, input, reservations, metrics)
}

pub(in crate::exec::bounded) fn execute_standalone_limit(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if bounded.sort.is_some() {
        return Ok(input);
    }
    let Some((node_id, limit)) = bounded.limit else {
        return Ok(input);
    };
    execute_post_final_stage(
        context,
        node_id,
        "post_aggregate_limit",
        "post_aggregate_limit_output",
        input,
        reservations,
        metrics,
        |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
        false,
    )
}

fn execute_post_final_stages(
    context: &ExecutionContext,
    stages: &[BoundedUnaryStage<'_>],
    mut input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    for stage in stages {
        input = match stage {
            BoundedUnaryStage::Filter { node_id, node } => execute_post_final_stage(
                context,
                *node_id,
                "post_aggregate_filter",
                "post_aggregate_filter_output",
                input,
                reservations,
                metrics,
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
                    context,
                    *node_id,
                    "post_aggregate_projection",
                    "post_aggregate_projection_output",
                    input,
                    reservations,
                    metrics,
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
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
    use_budget_token: bool,
) -> Result<CommittedPartial> {
    execute_bounded_generic_unary_consuming_input_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        ReservationKind::FinalMerge,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
}

pub(in crate::exec::bounded::aggregate) fn execute_final_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if !aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
        return Ok(input);
    }

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("final_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_projection")
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, "final_projection"));
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let expected_output_names =
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let result = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        let token_budget = reservations.available_bytes_including(&work_reservation);
        let token = match execution_budget_token_for_boundary(
            context,
            token_budget,
            "final_aggregate_projection",
            "cudf_query_engine.final_projection.budget_token",
            "failed to create bounded native final aggregate projection budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(work_reservation, metrics);
                reservations.release(input.reservation, metrics);
                return Err(RuntimeResourceFacts {
                    memory_boundary: "final_aggregate_projection",
                    stage: "final_projection",
                    node_id: bounded.aggregate_id,
                    attempt: 1,
                    max_attempts: 1,
                    resume_boundary: "final_projection",
                    metrics,
                }
                .attach_to(error));
            },
        };
        finalize_avg_output_columns_with_budget(
            context,
            input.frame.into_frame(),
            &bounded.aggregate.0.spec,
            bounded.aggregate.1,
            &token,
            metrics,
        )
    } else {
        Ok(input.frame.into_frame())
    }
    .and_then(|frame| {
        frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_projection.select",
                "failed to select bounded native final aggregate output columns",
                error,
            )
        })
    })
    .map(GpuDataFrame::new);
    reservations.release(work_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_projection",
                stage: "final_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "final_projection",
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation, metrics);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_projection_output",
            stage: "final_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "final_projection",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_projection")
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, "final_projection"));
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn legacy_bounded_final_merge_task_id(node_id: NodeId, stage: &str) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.{stage}")
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

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        validate_aggregate_output_names(
            &aggregate,
            bounded.aggregate.1,
            "cudf_query_engine.final_merge.validate",
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
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.final_merge.aggregate",
        "failed to execute bounded native final aggregate merge",
    )?;
    output =
        cast_count_output_columns_with_budget(output, &bounded.aggregate.0.spec, bounded.aggregate.1, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.select",
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
            &bounded.aggregate.0.spec,
            bounded.aggregate_input_schema,
            input_rows,
            output_rows,
            elapsed,
        );
    }
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec::bounded::aggregate) fn final_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        spec.aggr_exprs
            .iter()
            .enumerate()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Count => vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(expr.alias.clone()),
                    expr.alias.clone(),
                )],
                AggregateFunction::NUnique => vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(expr.alias.clone()),
                    expr.alias.clone(),
                )],
                AggregateFunction::Avg => {
                    let state = AvgStateAliases::for_aggregate_index(index);
                    vec![
                        AggregateExpr::new(AggregateFunction::Sum, state.sum_expr(), state.sum().to_string()),
                        AggregateExpr::new(AggregateFunction::Sum, state.count_expr(), state.count().to_string()),
                    ]
                },
                AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                    let valid_count_alias = global_valid_count_state_alias(expr.alias.as_str());
                    vec![
                        AggregateExpr::new(expr.function, Expr::column(expr.alias.clone()), expr.alias.clone()),
                        AggregateExpr::new(
                            AggregateFunction::Sum,
                            Expr::column(valid_count_alias.clone()),
                            valid_count_alias,
                        ),
                    ]
                },
                AggregateFunction::StddevSamp
                | AggregateFunction::StddevPop
                | AggregateFunction::VarSamp
                | AggregateFunction::VarPop => vec![expr.clone()],
            })
            .collect(),
    )
}

pub(in crate::exec::bounded::aggregate) fn aggregate_spec_has_global_final_state(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_avg(spec)
        || spec.group_exprs.is_empty()
            && spec.aggr_exprs.iter().any(|expr| {
                matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            })
}
