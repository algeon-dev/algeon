use super::*;

type ProjectionStage<'a> = (&'a ProjectionNode, &'a arrow::datatypes::SchemaRef);

#[derive(Clone)]
pub(super) struct BoundedPlan<'a> {
    pub(super) source_id: NodeId,
    pub(super) source: &'a LocalParquetSource,
    pub(super) pre_filter_projections: Vec<ProjectionStage<'a>>,
    pub(super) filter: Option<&'a FilterNode>,
    pub(super) post_filter_projections: Vec<ProjectionStage<'a>>,
    pub(super) aggregate_id: NodeId,
    pub(super) aggregate_input_schema: &'a arrow::datatypes::SchemaRef,
    pub(super) aggregate: (&'a AggregateNode, &'a arrow::datatypes::SchemaRef),
    pub(super) post_aggregate_projection: Option<ProjectionStage<'a>>,
    pub(super) sort: Option<&'a SortNode>,
    pub(super) limit: Option<&'a LimitNode>,
}

fn take_projection_stack<'a>(
    plan: &'a QueryPlan,
    current: &mut NodeId,
    missing_message: &'static str,
) -> Result<Vec<ProjectionStage<'a>>> {
    let mut projections = Vec::new();
    loop {
        let Some(node) = plan.node(*current) else {
            return Err(invalid_node_reference(*current, missing_message));
        };
        let PlanNode::Projection(projection) = &node.kind else {
            break;
        };
        projections.push((projection, &node.facts.output_schema));
        *current = projection.input;
    }
    projections.reverse();
    Ok(projections)
}

impl<'a> BoundedPlan<'a> {
    pub(super) fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let mut current = plan.sink().input();
        let limit = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Limit(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let sort = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Sort(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };
        if limit.is_some() && sort.is_none() {
            return Err(unsupported_bounded_shape(
                "bounded execution does not support limit without a preceding global sort",
                "limit_not_directly_above_sort",
            ));
        }

        let post_aggregate_projection = match plan.node(current) {
            Some(node) => match &node.kind {
                PlanNode::Projection(projection) => {
                    current = projection.input;
                    Some((projection, &node.facts.output_schema))
                },
                _ => None,
            },
            None => {
                return Err(invalid_node_reference(current, "bounded query plan references a missing sort input node"));
            },
        };

        let aggregate_id = current;
        let aggregate_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing aggregate node"))?;
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently requires a grouped sum aggregate after source-local operators",
                "missing_grouped_aggregate",
            ));
        };
        let aggregate_input_node = plan.node(aggregate.input).ok_or_else(|| {
            invalid_node_reference(aggregate.input, "bounded query plan references a missing aggregate input node")
        })?;
        current = aggregate.input;

        let post_filter_projections =
            take_projection_stack(plan, &mut current, "bounded query plan references a missing aggregate input node")?;

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let pre_filter_projections =
            take_projection_stack(plan, &mut current, "bounded query plan references a missing source input node")?;

        let source_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing source node"))?;
        let PlanNode::Source(source_node) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };
        let crate::source::SourceNode::LocalParquet(source) = source_node.as_ref() else {
            return Err(unsupported_bounded_shape(
                "bounded aggregate execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };

        Ok(Self {
            source_id: current,
            source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: &aggregate_input_node.facts.output_schema,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            post_aggregate_projection,
            sort,
            limit,
        })
    }

    pub(super) fn validate_capabilities(&self) -> Result<()> {
        if let Some(filter) = self.filter {
            lower_expr(&filter.predicate)?;
        }

        for (projection, output_schema) in &self.pre_filter_projections {
            validate_projection_capability(projection, output_schema)?;
        }

        for (projection, output_schema) in &self.post_filter_projections {
            validate_projection_capability(projection, output_schema)?;
        }

        validate_aggregate_capability(self.aggregate.0, self.aggregate.1)?;
        validate_final_aggregate_capability(self.aggregate.0, self.aggregate.1)?;

        if let Some((projection, output_schema)) = self.post_aggregate_projection {
            validate_projection_capability(projection, output_schema)?;
        }

        if let Some(sort) = self.sort {
            lower_sort_spec(&sort.spec)?;
        }
        if let Some(limit) = self.limit {
            topk_candidate_rows(limit.spec)?;
        }

        Ok(())
    }
}

pub(super) fn execute_final_aggregate_merge(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let mut partials = partials;
    let mut merge_level = 1;

    while partials.len() > 1 {
        let mut merge_fan_in = partials.len();
        let mut attempt = 1;

        loop {
            match execute_final_merge_level_attempt(
                context,
                bounded,
                &partials,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    metrics.record_final_merge_level();
                    partials = commit_final_merge_level(partials, level_output, reservations);
                    merge_level += 1;
                    break;
                },
                Err(error) if is_retryable_final_merge_error(&error) => {
                    record_final_merge_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    return Err(final_merge_retry_facts(merge_level, merge_fan_in, metrics).attach_to_error(error));
                },
                Err(error) => return Err(error),
            }
        }
    }

    let partial = partials.pop().expect("final merge loop leaves exactly one partial");
    execute_final_aggregate_projection(context, bounded, partial, reservations, metrics)
}

pub(super) fn execute_post_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let Some((projection, output_schema)) = bounded.post_aggregate_projection else {
        return Ok(input);
    };
    debug_assert_eq!(projection.output_schema().as_ref(), output_schema.as_ref());

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("post_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("post_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let mut output_reservation = match reservations.try_reserve(output_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    // Input remains live while cuDF evaluates the projection. The token may spend only the explicit
    // work/output reservations plus truly unreserved bytes, so it does not borrow from live input.
    let token_budget = reservations.available_bytes_including(&work_reservation);
    let token_budget = token_budget.saturating_add(output_reservation.bytes());
    let token = match execution_budget_token_for_boundary(
        context,
        token_budget,
        "post_aggregate_projection",
        "cudf_query_engine.post_aggregate_projection.budget_token",
        "failed to create bounded native post-aggregate projection budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(output_reservation);
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: "post_aggregate_projection",
                stage: "post_aggregate_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "post_aggregate_projection",
                metrics,
            }
            .attach_to(error));
        },
    };
    let result = execute_projection_with_budget(context, projection, input.frame, &token, metrics);
    reservations.release(work_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(output_reservation);
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: "post_aggregate_projection",
                stage: "post_aggregate_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "post_aggregate_projection",
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "post_aggregate_projection_output",
            stage: "post_aggregate_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "post_aggregate_projection",
            metrics,
        }
        .attach_to(error)
    })?;
    if output_bytes > output_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            output_bytes.saturating_sub(output_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("post_aggregate_projection_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("post_aggregate_projection");
        if let Err(error) = reservations.try_grow(&mut output_reservation, grow_request, metrics) {
            drop(frame);
            reservations.release(output_reservation);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut output_reservation, output_bytes);
    }
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn execute_final_aggregate_projection(
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
            .with_stage("final_projection");
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
                reservations.release(work_reservation);
                reservations.release(input.reservation);
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
    reservations.release(work_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation);
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

    reservations.release(input.reservation);
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
            .with_stage("final_projection");
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_final_merge_level_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<MergeLevelItem>> {
    metrics.record_final_merge_attempt();
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(MergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_final_merge_group_attempt(context, bounded, group, attempt, max_attempts, reservations, metrics) {
            Ok(partial) => output.push(MergeLevelItem::Merged(partial)),
            Err(error) => {
                release_merge_level_items(output, reservations);
                return Err(error);
            },
        }
    }

    Ok(output)
}

pub(super) fn execute_final_merge_group_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let merge_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts);
    let mut merge_reservation = reservations.try_reserve(merge_request, metrics)?;
    let merge_token_budget = reservations.available_bytes_including(&merge_reservation);
    let merge_token = match execution_budget_token_for_boundary(
        context,
        merge_token_budget,
        "final_aggregate_merge",
        "cudf_query_engine.final_merge.budget_token",
        "failed to create bounded native final aggregate merge budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(merge_reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics,
            }
            .attach_to(error));
        },
    };

    let timer = Instant::now();
    let result = merge_final_aggregate_partials(
        context,
        bounded,
        partials,
        &merge_token,
        &mut merge_reservation,
        reservations,
        attempt,
        max_attempts,
        metrics,
    );
    let merge_elapsed = timer.elapsed();
    reservations.release(merge_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            metrics.final_merge_time += merge_elapsed;
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics,
            }
            .attach_to(error));
        },
    };
    metrics.final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_merge_output",
            stage: "final_merge",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_merge",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn merge_final_aggregate_partials(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    token: &ExecutionBudgetToken,
    merge_reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    attempt: usize,
    max_attempts: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = token.execution_overrides();
    let concatenated = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        attach_budget_token_failure_facts(
            QueryError::from_cudf(
                "cudf_query_engine.final_merge.concat_partials",
                "failed to concatenate bounded native aggregate partials",
                error,
            )
            .with_fact("partial_aggregate_count", partial_count),
            token,
        )
    })?;
    let concatenated = GpuDataFrame::new(concatenated);
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > merge_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::FinalMerge,
            concatenated_bytes.saturating_sub(merge_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("final_aggregate_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_merge")
        .with_attempt(attempt, max_attempts);
        reservations.try_grow(merge_reservation, grow_request, metrics)?;
    }

    execute_final_aggregate(context, bounded, concatenated, token, metrics)
}

pub(super) enum MergeLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

pub(super) fn commit_final_merge_level(
    partials: Vec<CommittedPartial>,
    level_output: Vec<MergeLevelItem>,
    reservations: &mut ReservationTracker,
) -> Vec<CommittedPartial> {
    let mut partials = partials.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_partials = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            MergeLevelItem::Merged(partial) => next_partials.push(partial),
            MergeLevelItem::Carry { partial_index } => {
                let partial = partials[partial_index].take().expect("carry index references a source partial");
                next_partials.push(partial);
            },
        }
    }

    for partial in partials.into_iter().flatten() {
        reservations.release(partial.reservation);
    }

    next_partials
}

pub(super) fn release_merge_level_items(items: Vec<MergeLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let MergeLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

pub(super) fn is_retryable_final_merge_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn reduce_merge_fan_in(merge_fan_in: usize) -> Option<usize> {
    if merge_fan_in <= 2 {
        return None;
    }

    Some(merge_fan_in.div_ceil(2).max(2))
}

pub(super) fn final_merge_retry_facts(merge_level: usize, merge_fan_in: usize, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if merge_fan_in <= 2 { "fail_min_merge_fan_in" } else { "retry_attempts_exhausted" };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_merge")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
        .with_merge_level(merge_level)
        .with_merge_fan_in(merge_fan_in)
}

pub(super) fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
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
    let overrides = execution_overrides(context);
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
    Ok(GpuDataFrame::new(output))
}

pub(super) fn final_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
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
            })
            .collect(),
    )
}

pub(super) fn aggregate_spec_has_global_final_state(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_avg(spec)
        || spec.group_exprs.is_empty()
            && spec.aggr_exprs.iter().any(|expr| {
                matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            })
}
