use super::{
    source::{read_local_parquet_chunk, record_bounded_parquet_source_metrics},
    *,
};

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let bounded = BoundedPlan::analyze(plan)?;
    validate_local_parquet_source(bounded.source)?;
    bounded.validate_capabilities()?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let chunk_plan = plan_local_parquet_source_chunks(
        bounded.source_id,
        bounded.source,
        SourceChunkPlanningPolicy::from(context.memory_control_policy()),
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.plan",
            "bounded execution requires at least one planned source chunk",
        )
        .with_fact("source_node_id", bounded.source_id));
    }
    record_bounded_parquet_source_metrics(bounded.source, &chunk_plan, metrics);

    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut partials = Vec::with_capacity(chunk_plan.chunks.len());
    for chunk in &chunk_plan.chunks {
        partials.extend(execute_bounded_chunk_with_retry(context, &bounded, chunk, &mut reservations, metrics)?);
    }

    let frame = execute_final_aggregate_merge(context, &bounded, partials, &mut reservations, metrics)?;
    let frame = execute_post_aggregate_projection(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_final_sort_limit_with_retry(context, &bounded, frame, &mut reservations, metrics)?;

    record_device_memory_observation(&frame.frame, metrics)?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            reservations.release(frame.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(frame.frame))
        },
        ResultSink::HostArrow { .. } => {
            let output = materialize_bounded_to_host(frame, plan.sink().input(), &mut reservations, metrics)?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

pub(super) fn execute_bounded_chunk_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        context.memory_control_policy()
    else {
        unreachable!("execute_bounded_chunk_with_retry is called only for bounded policies");
    };

    execute_bounded_chunk_retry_attempt(
        context,
        bounded,
        chunk.clone(),
        reservations,
        metrics,
        1,
        max_retry_attempts.max(1),
        min_retry_chunk_row_groups.max(1),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_bounded_chunk_retry_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    match execute_bounded_chunk_attempt(context, bounded, &chunk, reservations, metrics, attempt, max_attempts) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, metrics);
            if attempt < max_attempts
                && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
            {
                metrics.record_source_chunk_retry();
                let mut partials = execute_bounded_chunk_retry_attempt(
                    context,
                    bounded,
                    left,
                    reservations,
                    metrics,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?;
                partials.extend(execute_bounded_chunk_retry_attempt(
                    context,
                    bounded,
                    right,
                    reservations,
                    metrics,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?);
                return Ok(partials);
            }

            Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
        },
        Err(error) => Err(error),
    }
}

pub(super) fn execute_bounded_chunk_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    metrics.record_source_chunk_attempt();

    let source_request = chunk_reservation_request(
        ReservationKind::SourceRead,
        "source_chunk_read",
        chunk_requested_bytes(context.memory_control_policy(), chunk),
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let source_reservation = reservations.try_reserve(source_request, metrics)?;
    let mut frame = match read_local_parquet_chunk(context, chunk, metrics) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics));
        },
    };
    let source_observed_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics));
        },
    };

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_observed_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let operator_reservation = match reservations.try_reserve(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(error);
        },
    };

    let chunk_operator_timer = Instant::now();
    let chunk_output = (|| {
        if let Some(filter) = bounded.filter {
            frame = execute_filter(context, filter, frame, metrics)?;
        }
        if let Some((projection, output_schema)) = bounded.projection {
            frame = execute_projection(context, projection, frame, output_schema, metrics)?;
        }
        execute_partial_aggregate(context, bounded.aggregate.0, frame, metrics)
    })();
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation);
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };

    reservations.release(operator_reservation);
    reservations.release(source_reservation);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, chunk, "partial_state", attempt, max_attempts, metrics))?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    metrics.record_partial_aggregate(frame.num_rows() as u64);

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

#[derive(Clone, Copy)]
pub(super) struct BoundedPlan<'a> {
    pub(super) source_id: NodeId,
    pub(super) source: &'a LocalParquetSource,
    pub(super) filter: Option<&'a FilterNode>,
    pub(super) projection: Option<(&'a ProjectionNode, &'a arrow::datatypes::SchemaRef)>,
    pub(super) aggregate_id: NodeId,
    pub(super) aggregate: (&'a AggregateNode, &'a arrow::datatypes::SchemaRef),
    pub(super) post_aggregate_projection: Option<(&'a ProjectionNode, &'a arrow::datatypes::SchemaRef)>,
    pub(super) sort: Option<&'a SortNode>,
    pub(super) limit: Option<&'a LimitNode>,
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
                "limit_without_sort",
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
        current = aggregate.input;

        let projection = match plan.node(current) {
            Some(node) => match &node.kind {
                PlanNode::Projection(projection) => {
                    current = projection.input;
                    Some((projection, &node.facts.output_schema))
                },
                _ => None,
            },
            None => return Err(invalid_node_reference(current, "bounded query plan references a missing input node")),
        };

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let source_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing source node"))?;
        let PlanNode::Source(crate::source::SourceNode::LocalParquet(source)) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };

        Ok(Self {
            source_id: current,
            source,
            filter,
            projection,
            aggregate_id,
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

        if let Some((projection, output_schema)) = self.projection {
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

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("post_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let result = execute_projection(context, projection, input.frame, output_schema, metrics);
    reservations.release(work_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
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
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("post_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn execute_final_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if !aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
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
    let result =
        finalize_avg_output_columns(context, input.frame.into_frame(), &bounded.aggregate.0.spec, bounded.aggregate.1)
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
    let merge_reservation = reservations.try_reserve(merge_request, metrics)?;

    let timer = Instant::now();
    let result = merge_final_aggregate_partials(context, bounded, partials, metrics);
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
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let concatenated = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.concat_partials",
            "failed to concatenate bounded native aggregate partials",
            error,
        )
        .with_fact("partial_aggregate_count", partial_count)
    })?;

    execute_final_aggregate(context, bounded, GpuDataFrame::new(concatenated), metrics)
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

pub(super) fn execute_final_sort_limit_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let Some(sort) = bounded.sort else {
        return Ok(input);
    };
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_sort_limit_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let result = execute_full_final_sort_limit_attempt(
        context,
        bounded,
        sort,
        bounded.limit,
        &input,
        1,
        max_attempts,
        reservations,
        metrics,
    );

    match result {
        Ok(output) => {
            metrics.final_sort_time += timer.elapsed();
            reservations.release(input.reservation);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, metrics);
            if max_attempts > 1
                && let Some(limit) = bounded.limit
            {
                match execute_hierarchical_topk_final_sort(
                    context,
                    bounded,
                    sort,
                    limit,
                    input,
                    max_attempts,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => {
                        metrics.final_sort_time += timer.elapsed();
                        Ok(output)
                    },
                    Err(error) => {
                        metrics.final_sort_time += timer.elapsed();
                        Err(error)
                    },
                }
            } else {
                metrics.final_sort_time += timer.elapsed();
                reservations.release(input.reservation);
                Err(final_sort_retry_facts("retry_attempts_exhausted", metrics).attach_to_error(error))
            }
        },
        Err(error) => {
            metrics.final_sort_time += timer.elapsed();
            reservations.release(input.reservation);
            Err(error)
        },
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_full_final_sort_limit_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: &CommittedPartial,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let sort_request =
        MemoryReservationRequest::new(ReservationKind::FinalSort, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("final_sort")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_sort")
            .with_attempt(attempt, max_attempts);
    let sort_reservation = reservations.try_reserve(sort_request, metrics)?;

    let result = execute_sort_on_frame(context, sort, input.frame.frame().clone(), metrics)
        .and_then(|frame| execute_optional_limit(context, limit, frame, metrics));
    reservations.release(sort_reservation);
    let frame = result.map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    reserve_final_sort_output(bounded, frame, attempt, max_attempts, reservations, metrics)
}

pub(super) fn execute_hierarchical_topk_final_sort(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    limit: &LimitNode,
    input: CommittedPartial,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let keep_rows = topk_candidate_rows(limit.spec)?;
    if keep_rows == 0 {
        let output = match execute_limit_on_frame(context, limit, input.frame.frame().clone(), metrics) {
            Ok(output) => output,
            Err(error) => {
                reservations.release(input.reservation);
                return Err(error);
            },
        };
        reservations.release(input.reservation);
        return reserve_final_sort_output(bounded, output, 2, max_attempts, reservations, metrics);
    }

    let input_rows = input.frame.num_rows();
    if input_rows == 0 {
        return Ok(input);
    }
    let bytes_per_row = estimated_bytes_per_row(input.reservation.bytes(), input_rows);
    let initial_rows_per_chunk = final_sort_rows_per_chunk(
        input_rows,
        bytes_per_row,
        reservations.budget().saturating_sub(input.reservation.bytes()),
    );
    let mut candidates = match execute_initial_topk_candidates(
        context,
        bounded,
        sort,
        &input,
        keep_rows,
        bytes_per_row,
        initial_rows_per_chunk,
        max_attempts,
        reservations,
        metrics,
    ) {
        Ok(candidates) => candidates,
        Err(error) => {
            record_final_sort_retry_error(&error, metrics);
            reservations.release(input.reservation);
            return Err(final_sort_retry_facts("hierarchical_topk_initial_failed", metrics).attach_to_error(error));
        },
    };

    reservations.release(input.reservation);

    let mut merge_level = 1;
    while candidates.len() > 1 {
        let mut merge_fan_in = candidates.len();
        let mut attempt = 2;

        loop {
            match execute_topk_merge_level_attempt(
                context,
                bounded,
                sort,
                &candidates,
                keep_rows,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    candidates = commit_topk_merge_level(candidates, level_output, reservations);
                    merge_level += 1;
                    break;
                },
                Err(error) if is_retryable_final_sort_error(&error) => {
                    record_final_sort_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    release_committed_partials(candidates, reservations);
                    return Err(final_sort_topk_retry_facts(merge_level, merge_fan_in, metrics).attach_to_error(error));
                },
                Err(error) => {
                    release_committed_partials(candidates, reservations);
                    return Err(error);
                },
            }
        }
    }

    let candidate = candidates.pop().expect("hierarchical top-k leaves exactly one candidate");
    if limit.spec.skip == 0 && candidate.frame.num_rows() <= limit.spec.fetch {
        return Ok(candidate);
    }

    let output = execute_limit(context, limit, candidate.frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_limit",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt: max_attempts,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    reservations.release(candidate.reservation);
    reserve_final_sort_output(bounded, output, max_attempts, max_attempts, reservations, metrics)
}

pub(super) fn execute_initial_topk_candidates(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    input: &CommittedPartial,
    keep_rows: usize,
    bytes_per_row: u64,
    rows_per_chunk: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let input_rows = input.frame.num_rows();
    let mut candidates = Vec::with_capacity(input_rows.div_ceil(rows_per_chunk));
    let mut offset = 0;
    while offset < input_rows {
        let length = rows_per_chunk.min(input_rows - offset);
        match execute_topk_slice_attempt(
            context,
            bounded,
            sort,
            input,
            offset,
            length,
            keep_rows,
            bytes_per_row,
            2,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => {
                release_committed_partials(candidates, reservations);
                return Err(error);
            },
        }
        offset += length;
    }
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_topk_slice_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    input: &CommittedPartial,
    offset: usize,
    length: usize,
    keep_rows: usize,
    bytes_per_row: u64,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let requested_bytes = estimate_rows_bytes(bytes_per_row, length);
    let request = MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, reservations.budget())
        .with_memory_boundary("final_sort_topk_chunk")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let work_reservation = reservations.try_reserve(request, metrics)?;

    let result = input
        .frame
        .frame()
        .clone()
        .with_execution(&execution_overrides(context))
        .slice(offset, length)
        .map(GpuDataFrame::new)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_sort.topk_slice",
                "failed to slice bounded native final sort input",
                error,
            )
        })
        .and_then(|frame| execute_topk_candidate(context, sort, frame, keep_rows, metrics));
    reservations.release(work_reservation);
    let frame = result.map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_topk_chunk",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    reserve_final_sort_output(bounded, frame, attempt, max_attempts, reservations, metrics)
}

pub(super) fn execute_topk_merge_level_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    candidates: &[CommittedPartial],
    keep_rows: usize,
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<TopKLevelItem>> {
    let mut output = Vec::with_capacity(candidates.len().div_ceil(merge_fan_in));

    for (group_index, group) in candidates.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(TopKLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_topk_merge_group_attempt(
            context,
            bounded,
            sort,
            group,
            keep_rows,
            attempt,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(candidate) => output.push(TopKLevelItem::Merged(candidate)),
            Err(error) => {
                release_topk_level_items(output, reservations);
                return Err(error);
            },
        }
    }

    Ok(output)
}

pub(super) fn execute_topk_merge_group_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    candidates: &[CommittedPartial],
    keep_rows: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(candidates.len() > 1);
    let requested_bytes = candidates.iter().map(|candidate| candidate.reservation.bytes()).sum::<u64>();
    let request = MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, reservations.budget())
        .with_memory_boundary("final_sort_topk_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let merge_reservation = reservations.try_reserve(request, metrics)?;

    let result = concat_committed_frames(context, candidates, "cudf_query_engine.final_sort.topk_concat")
        .and_then(|frame| execute_topk_candidate(context, sort, frame, keep_rows, metrics));
    reservations.release(merge_reservation);
    let frame = result.map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_topk_merge",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    reserve_final_sort_output(bounded, frame, attempt, max_attempts, reservations, metrics)
}

pub(super) fn execute_topk_candidate(
    context: &ExecutionContext,
    sort: &SortNode,
    input: GpuDataFrame,
    keep_rows: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sorted = execute_sort(context, sort, input, metrics)?;
    execute_limit_spec(context, LimitSpec::fetch(keep_rows), sorted, metrics)
}

pub(super) fn concat_committed_frames(
    context: &ExecutionContext,
    frames: &[CommittedPartial],
    operation: &'static str,
) -> Result<GpuDataFrame> {
    let first_frame = frames[0].frame.frame().clone();
    let other_frames = frames[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let output = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(operation, "failed to concatenate bounded native final sort candidates", error)
            .with_fact("candidate_count", frames.len())
    })?;
    Ok(GpuDataFrame::new(output))
}

pub(super) fn reserve_final_sort_output(
    bounded: &BoundedPlan<'_>,
    frame: GpuDataFrame,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_output",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request = MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, reservations.budget())
        .with_memory_boundary("final_sort_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) enum TopKLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

pub(super) fn commit_topk_merge_level(
    candidates: Vec<CommittedPartial>,
    level_output: Vec<TopKLevelItem>,
    reservations: &mut ReservationTracker,
) -> Vec<CommittedPartial> {
    let mut candidates = candidates.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_candidates = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            TopKLevelItem::Merged(partial) => next_candidates.push(partial),
            TopKLevelItem::Carry { partial_index } => {
                let partial = candidates[partial_index].take().expect("carry index references a top-k candidate");
                next_candidates.push(partial);
            },
        }
    }

    for partial in candidates.into_iter().flatten() {
        reservations.release(partial.reservation);
    }

    next_candidates
}

pub(super) fn release_topk_level_items(items: Vec<TopKLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let TopKLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

pub(super) fn release_committed_partials(partials: Vec<CommittedPartial>, reservations: &mut ReservationTracker) {
    for partial in partials {
        reservations.release(partial.reservation);
    }
}

pub(super) fn is_retryable_final_sort_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_final_sort_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn final_sort_retry_facts(retry_action: &'static str, metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_sort")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}

pub(super) fn final_sort_topk_retry_facts(
    merge_level: usize,
    merge_fan_in: usize,
    metrics: &QueryMetrics,
) -> RetryFacts {
    final_sort_retry_facts(
        if merge_fan_in <= 2 { "fail_min_topk_fan_in" } else { "hierarchical_topk_retry_exhausted" },
        metrics,
    )
    .with_merge_level(merge_level)
    .with_merge_fan_in(merge_fan_in)
}

pub(super) fn topk_candidate_rows(spec: LimitSpec) -> Result<usize> {
    spec.skip.checked_add(spec.fetch).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.final_sort.topk_limit",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", spec.fetch)
    })
}

pub(super) fn estimated_bytes_per_row(bytes: u64, rows: usize) -> u64 {
    if rows == 0 {
        return bytes.max(1);
    }
    bytes.div_ceil(rows as u64).max(1)
}

pub(super) fn estimate_rows_bytes(bytes_per_row: u64, rows: usize) -> u64 {
    let bytes = u128::from(bytes_per_row) * rows as u128;
    bytes.min(u128::from(u64::MAX)) as u64
}

pub(super) fn final_sort_rows_per_chunk(input_rows: usize, bytes_per_row: u64, available_bytes: u64) -> usize {
    if input_rows == 0 {
        return 1;
    }
    let rows = (available_bytes / bytes_per_row).min(input_rows as u64) as usize;
    rows.max(1)
}

pub(super) fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if bounded.aggregate.0.spec.group_exprs.is_empty() {
        let final_spec = final_global_aggregate_spec(&bounded.aggregate.0.spec);
        return execute_global_aggregate(context, &final_spec, input, bounded.aggregate.1, metrics);
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
    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .with_execution(&overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_merge.aggregate",
                "failed to execute bounded native final aggregate merge",
                error,
            )
        })?;
    output = cast_count_output_columns(output, &bounded.aggregate.0.spec, bounded.aggregate.1)?;
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
            .map(|expr| AggregateExpr::new(expr.function, Expr::column(expr.alias.clone()), expr.alias.clone()))
            .collect(),
    )
}

pub(super) struct ActiveReservation {
    reserved_bytes: Rc<Cell<u64>>,
    bytes: u64,
    released: bool,
}

impl ActiveReservation {
    pub(super) fn new(reserved_bytes: Rc<Cell<u64>>, bytes: u64) -> Self {
        Self { reserved_bytes, bytes, released: false }
    }

    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(super) fn release(&mut self) {
        if self.released {
            return;
        }
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(self.bytes));
        self.released = true;
    }
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug)]
pub(super) struct ReservationTracker {
    budget_bytes: u64,
    reserved_bytes: Rc<Cell<u64>>,
}

impl ReservationTracker {
    pub(super) fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes, reserved_bytes: Rc::new(Cell::new(0)) }
    }

    pub(super) fn budget(&self) -> u64 {
        self.budget_bytes
    }

    pub(super) fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes.get()
    }

    pub(super) fn try_reserve(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        let remaining = self.budget_bytes.saturating_sub(self.reserved_bytes());
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied(request.requested_bytes);
            return Err(request.denied_error(remaining));
        }

        self.reserved_bytes.set(self.reserved_bytes().saturating_add(request.requested_bytes));
        metrics.record_reservation_granted(request.requested_bytes, request.requested_bytes);
        Ok(ActiveReservation::new(Rc::clone(&self.reserved_bytes), request.requested_bytes))
    }

    pub(super) fn release(&mut self, mut reservation: ActiveReservation) {
        reservation.release();
    }
}

pub(super) struct CommittedPartial {
    pub(super) frame: GpuDataFrame,
    pub(super) reservation: ActiveReservation,
}

pub(super) fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    chunk.known_row_group_bytes.or(chunk.estimated_chunk_bytes).or(chunk.known_file_bytes).unwrap_or(fallback)
}

pub(super) fn chunk_reservation_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(chunk.source_node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request
}

pub(super) fn is_retryable_chunk_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_chunk_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn split_retryable_chunk(
    chunk: &SourceChunk,
    min_retry_chunk_row_groups: usize,
) -> Option<(SourceChunk, SourceChunk)> {
    if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= min_retry_chunk_row_groups {
        return None;
    }

    let split_at = chunk.row_group_indexes.len() / 2;
    if split_at == 0 || split_at == chunk.row_group_indexes.len() {
        return None;
    }

    let left_row_groups = chunk.row_group_indexes[..split_at].to_vec();
    let right_row_groups = chunk.row_group_indexes[split_at..].to_vec();
    let left = retry_child_chunk(chunk, left_row_groups);
    let right = retry_child_chunk(chunk, right_row_groups);
    Some((left, right))
}

pub(super) fn retry_child_chunk(parent: &SourceChunk, row_group_indexes: Vec<usize>) -> SourceChunk {
    let parent_row_group_count = parent.row_group_indexes.len();
    let known_row_group_bytes =
        proportional_bytes(parent.known_row_group_bytes, row_group_indexes.len(), parent_row_group_count);
    let known_file_bytes = if parent.known_row_group_bytes.is_some() {
        parent.known_file_bytes
    } else {
        proportional_bytes(parent.known_file_bytes, row_group_indexes.len(), parent_row_group_count)
    };
    let estimated_chunk_bytes = proportional_bytes(
        parent.estimated_chunk_bytes.or(parent.known_row_group_bytes).or(parent.known_file_bytes),
        row_group_indexes.len(),
        parent_row_group_count,
    );
    let known_row_count = proportional_bytes(parent.known_row_count, row_group_indexes.len(), parent_row_group_count);

    SourceChunk {
        source_node_id: parent.source_node_id,
        chunk_index: parent.chunk_index,
        file_index: parent.file_index,
        path: parent.path.clone(),
        row_group_indexes,
        reads_all_row_groups: false,
        known_file_bytes,
        known_row_group_bytes,
        estimated_chunk_bytes,
        read_columns: parent.read_columns.clone(),
        zero_column_count: parent.zero_column_count,
        known_row_count,
        size_estimate_based: true,
    }
}

pub(super) fn proportional_bytes(
    total_bytes: Option<u64>,
    child_row_groups: usize,
    parent_row_groups: usize,
) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

pub(super) fn chunk_retry_facts(chunk: &SourceChunk, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= 1 {
        "fail_min_chunk"
    } else {
        "retry_attempts_exhausted"
    };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("source_chunk")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RuntimeResourceFacts<'a> {
    pub(super) memory_boundary: &'static str,
    pub(super) stage: &'static str,
    pub(super) node_id: NodeId,
    pub(super) attempt: usize,
    pub(super) max_attempts: usize,
    pub(super) resume_boundary: &'static str,
    pub(super) metrics: &'a QueryMetrics,
}

impl RuntimeResourceFacts<'_> {
    pub(super) fn attach_to(self, error: QueryError) -> QueryError {
        if error.code() != QueryErrorCode::CudfResourceExhausted {
            return error;
        }

        RetryFacts::new()
            .with_resume_boundary(self.resume_boundary)
            .with_committed_chunks(self.metrics.source_chunks_committed as usize)
            .with_committed_partials(self.metrics.partial_aggregate_count as usize)
            .attach_to_error(
                error
                    .with_fact("memory_boundary", self.memory_boundary)
                    .with_fact("stage", self.stage)
                    .with_fact("node_id", self.node_id)
                    .with_fact("attempt", self.attempt)
                    .with_fact("max_attempts", self.max_attempts),
            )
    }
}

pub(super) fn attach_chunk_resource_facts(
    error: QueryError,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    metrics: &QueryMetrics,
) -> QueryError {
    if error.code() != QueryErrorCode::CudfResourceExhausted {
        return error;
    }

    let mut error = RuntimeResourceFacts {
        memory_boundary,
        stage: memory_boundary,
        node_id: chunk.source_node_id,
        attempt,
        max_attempts,
        resume_boundary: "source_chunk",
        metrics,
    }
    .attach_to(error)
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("file_index", chunk.file_index)
    .with_fact("path", chunk.path.display());
    if !chunk.reads_all_row_groups {
        error = error.with_fact("row_groups", comma_separated_usize(&chunk.row_group_indexes));
    }
    error
}
