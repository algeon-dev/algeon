use super::*;
use crate::exec::source::{ScheduledLocalParquetReadOutput, record_bounded_parquet_source_metrics};

pub(in crate::exec::bounded) fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.generic_query_plan")?;
    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new_for_context(context, device_budget_bytes, metrics)?;
    let committed =
        execute_bounded_generic_node(context, plan, plan.sink().input(), gpu_inputs, &mut reservations, metrics)?;

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation, metrics);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation, metrics);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_generic_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    context.check_cancellation("nexus_query_engine.exec.bounded.generic_node")?;
    let node = plan
        .node(id)
        .ok_or_else(|| invalid_node_reference(id, "bounded generic query plan references a missing node"))?;
    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => {
                execute_bounded_generic_local_parquet_source(context, source, id, reservations, metrics)
            },
            crate::source::SourceNode::GpuResident(source) => {
                execute_bounded_generic_gpu_resident_source(context, source, id, gpu_inputs, reservations, metrics)
            },
            crate::source::SourceNode::LiteralBatch(source) => {
                execute_bounded_generic_literal_batch_source(context, source, id, reservations, metrics)
            },
        },
        PlanNode::Filter(filter) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(context, plan, id, reservations, metrics)?
            {
                return Ok(frame);
            }
            let input = execute_bounded_generic_node(context, plan, filter.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_filter",
                "bounded_generic_filter_output",
                input,
                reservations,
                metrics,
                true,
                |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        filter,
                        input,
                        token.expect("budget token is created for bounded generic filter stages"),
                        metrics,
                    )
                },
            )
        },
        PlanNode::Projection(projection) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(context, plan, id, reservations, metrics)?
            {
                return Ok(frame);
            }
            let input =
                execute_bounded_generic_node(context, plan, projection.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_projection",
                "bounded_generic_projection_output",
                input,
                reservations,
                metrics,
                true,
                |context, input, token, metrics| {
                    execute_projection_with_budget(
                        context,
                        projection,
                        input,
                        token.expect("budget token is created for bounded generic projection stages"),
                        metrics,
                    )
                },
            )
        },
        PlanNode::Window(window) => {
            let input = execute_bounded_generic_node(context, plan, window.input, gpu_inputs, reservations, metrics)?;
            let input_schema = plan
                .node(window.input)
                .ok_or_else(|| invalid_node_reference(window.input, "window references a missing input"))?
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_window",
                "bounded_generic_window_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| {
                    execute_window(context, window, &input_schema, input, &output_schema, metrics)
                },
            )
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(context, plan, sort.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_sort",
                "bounded_generic_sort_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| execute_sort(context, sort, input, metrics),
            )
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(context, plan, limit.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_limit",
                "bounded_generic_limit_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
            )
        },
        PlanNode::Union(union) => {
            let inputs = union
                .inputs
                .iter()
                .map(|input| execute_bounded_generic_node(context, plan, *input, gpu_inputs, reservations, metrics))
                .collect::<Result<Vec<_>>>()?;
            execute_bounded_generic_union(context, id, inputs, reservations, metrics)
        },
        PlanNode::Aggregate(aggregate) => {
            let input = match plan.node(aggregate.input).map(|node| &node.kind) {
                Some(PlanNode::Aggregate(_)) => match BoundedPlan::analyze_aggregate_subplan(plan, aggregate.input) {
                    Ok(bounded) => execute_bounded_aggregate_to_frame(context, &bounded, reservations, metrics)?,
                    Err(_error) => {
                        execute_bounded_generic_node(context, plan, aggregate.input, gpu_inputs, reservations, metrics)?
                    },
                },
                _ => execute_bounded_generic_node(context, plan, aggregate.input, gpu_inputs, reservations, metrics)?,
            };
            let input_schema = plan
                .node(aggregate.input)
                .ok_or_else(|| invalid_node_reference(aggregate.input, "aggregate references a missing input"))?
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            let input_reserved_bytes = input.reservation.bytes();
            let input_reservation = input.reservation;
            let result = execute_compiled_bounded_aggregate_from_frame(
                context,
                id,
                aggregate,
                &input_schema,
                &output_schema,
                input.frame,
                input_reserved_bytes,
                reservations,
                metrics,
            );
            reservations.release(input_reservation, metrics);
            result
        },
        PlanNode::Join(_) | PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => {
            Err(unsupported_bounded_shape(
            "bounded generic execution supports only source, filter, projection, aggregate, window, sort, limit, and union nodes",
            "bounded_generic_plan_shape_unsupported",
        )
        .with_fact("node_id", id)
        .with_fact("node_kind", bounded_generic_node_kind(&node.kind)))
        },
    }?;

    Ok(frame)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_union(
    context: &ExecutionContext,
    node_id: NodeId,
    inputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.union",
            "bounded union requires at least one input",
        )
        .with_fact("node_id", node_id));
    }
    if inputs.len() == 1 {
        return Ok(inputs.into_iter().next().expect("single union input checked"));
    }

    let output_reservation_bytes =
        inputs.iter().fold(0u64, |total, input| total.saturating_add(input.reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_reservation_bytes, reservations.budget())
            .with_memory_boundary("bounded_union_concat_output")
            .with_node_id(node_id)
            .with_stage("bounded_union_concat")
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_resident_frames_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };

    let input_count = inputs.len();
    let mut input_reservations = Vec::with_capacity(input_count);
    let mut frames = Vec::with_capacity(input_count);
    for input in inputs {
        frames.push(input.frame.into_frame());
        input_reservations.push(input.reservation);
    }
    let first = frames.remove(0);
    let rest = frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = match first.with_execution(&overrides).concat_many(&rest) {
        Ok(output) => output,
        Err(error) => {
            drop(rest);
            drop(frames);
            drop(first);
            release_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.union.concat",
                "failed to concatenate bounded native union inputs",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("input_count", input_count));
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output_frame);
            drop(rest);
            drop(frames);
            drop(first);
            release_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary("bounded_union_concat_output")
            .with_node_id(node_id)
            .with_stage("bounded_union_concat")
            .with_attempt(1, 1);
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut output_reservation, output_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(output_frame);
            drop(rest);
            drop(frames);
            drop(first);
            release_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(
            &mut output_reservation,
            output_bytes,
            output_request.clone(),
            metrics,
        );
    }

    drop(rest);
    drop(frames);
    drop(first);
    release_reservations_reverse(reservations, input_reservations, metrics);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn release_resident_frames_reverse(
    reservations: &mut ReservationTracker,
    mut frames: Vec<ResidentGpuFrame>,
    metrics: &mut QueryMetrics,
) {
    while let Some(frame) = frames.pop() {
        reservations.release(frame.reservation, metrics);
    }
}

pub(in crate::exec::bounded) fn release_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut input_reservations: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = input_reservations.pop() {
        reservations.release(reservation, metrics);
    }
}

#[derive(Clone, Copy)]
enum BoundedGenericSourceAdjacentStage<'a> {
    Filter { node_id: NodeId, node: &'a FilterNode },
    Projection { node_id: NodeId, node: &'a ProjectionNode },
}

fn try_execute_bounded_generic_source_adjacent_stream(
    context: &ExecutionContext,
    plan: &QueryPlan,
    root_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<ResidentGpuFrame>> {
    let Some((source_id, source, stages)) = collect_bounded_generic_source_adjacent_stages(plan, root_id)? else {
        return Ok(None);
    };
    if source.has_position_delete_visible_scan_limit() {
        return Ok(None);
    }
    execute_bounded_generic_local_parquet_source_with_stages(context, source, source_id, &stages, reservations, metrics)
        .map(Some)
}

fn collect_bounded_generic_source_adjacent_stages(
    plan: &QueryPlan,
    root_id: NodeId,
) -> Result<Option<(NodeId, &LocalParquetSource, Vec<BoundedGenericSourceAdjacentStage<'_>>)>> {
    let mut stages = Vec::new();
    let mut current_id = root_id;
    loop {
        let node = plan.node(current_id).ok_or_else(|| {
            invalid_node_reference(current_id, "bounded generic source-adjacent stream references a missing node")
        })?;
        match &node.kind {
            PlanNode::Filter(filter) => {
                stages.push(BoundedGenericSourceAdjacentStage::Filter { node_id: current_id, node: filter });
                current_id = filter.input;
            },
            PlanNode::Projection(projection) => {
                stages.push(BoundedGenericSourceAdjacentStage::Projection { node_id: current_id, node: projection });
                current_id = projection.input;
            },
            PlanNode::Source(source) => match source.as_ref() {
                crate::source::SourceNode::LocalParquet(source) if !stages.is_empty() => {
                    stages.reverse();
                    return Ok(Some((current_id, source, stages)));
                },
                _ => return Ok(None),
            },
            _ => return Ok(None),
        }
    }
}

pub(in crate::exec::bounded) fn execute_bounded_generic_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source)?;
    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        node_id,
        source,
        SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
            source,
            context.memory_control_policy(),
            context.source_chunk_planning_mode(),
            context.source_chunk_planning_options(),
        ),
        context,
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.generic.source",
            "bounded generic local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);

    let chunks = read_local_parquet_chunks_required_only(
        context,
        source,
        &chunk_plan.chunks,
        reservations,
        metrics,
        "bounded_generic_source_read",
    )?;

    concatenate_source_read_outputs(
        context,
        node_id,
        chunks,
        reservations,
        metrics,
        "bounded_generic_source_concat_output",
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn execute_bounded_generic_local_parquet_source_with_stages(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source)?;
    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        node_id,
        source,
        SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
            source,
            context.memory_control_policy(),
            context.source_chunk_planning_mode(),
            context.source_chunk_planning_options(),
        ),
        context,
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.generic.source",
            "bounded generic local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);

    let chunks = read_bounded_generic_source_adjacent_chunks(
        context,
        source,
        &chunk_plan.chunks,
        stages,
        context.local_parquet_source_cache().is_some(),
        reservations,
        metrics,
    )?;

    concatenate_source_read_outputs(
        context,
        node_id,
        chunks,
        reservations,
        metrics,
        "bounded_generic_source_concat_output",
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn read_bounded_generic_source_adjacent_chunks(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    grow_source_to_observed: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut transformed = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let source_chunk = match read_local_parquet_chunks_required_only_with_observation(
            context,
            source,
            std::slice::from_ref(chunk),
            reservations,
            metrics,
            "bounded_generic_source_read",
            grow_source_to_observed,
        ) {
            Ok(source_chunk) => source_chunk,
            Err(error) => {
                release_source_read_outputs_reverse(reservations, transformed, metrics);
                return Err(error);
            },
        };
        let mut staged =
            match apply_bounded_generic_source_adjacent_stages(context, source_chunk, stages, reservations, metrics) {
                Ok(staged) => staged,
                Err(error) => {
                    release_source_read_outputs_reverse(reservations, transformed, metrics);
                    return Err(error);
                },
            };
        transformed.append(&mut staged);
    }
    Ok(transformed)
}

fn apply_bounded_generic_source_adjacent_stages(
    context: &ExecutionContext,
    chunks: Vec<ScheduledLocalParquetReadOutput>,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut pending = chunks;
    let mut transformed = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let source = pending.remove(0);
        let chunk = source.chunk.clone();
        let mut current = CommittedPartial { frame: source.frame, reservation: source.reservation };
        for stage in stages {
            let result = match stage {
                BoundedGenericSourceAdjacentStage::Filter { node_id, node } => execute_source_adjacent_chunk_unary(
                    context,
                    *node_id,
                    "bounded_generic_source_filter",
                    "bounded_generic_source_filter_output",
                    &chunk,
                    current,
                    reservations,
                    metrics,
                    1,
                    1,
                    |context, input, token, metrics| execute_filter_with_budget(context, node, input, token, metrics),
                ),
                BoundedGenericSourceAdjacentStage::Projection { node_id, node } => execute_source_adjacent_chunk_unary(
                    context,
                    *node_id,
                    "bounded_generic_source_projection",
                    "bounded_generic_source_projection_output",
                    &chunk,
                    current,
                    reservations,
                    metrics,
                    1,
                    1,
                    |context, input, token, metrics| {
                        execute_projection_with_budget(context, node, input, token, metrics)
                    },
                ),
            };
            current = match result {
                Ok(current) => current,
                Err(error) => {
                    release_source_read_outputs_reverse(reservations, pending, metrics);
                    release_source_read_outputs_reverse(reservations, transformed, metrics);
                    return Err(error);
                },
            };
        }
        let observed_bytes = current.reservation.bytes();
        transformed.push(ScheduledLocalParquetReadOutput {
            chunk,
            frame: current.frame,
            reservation: current.reservation,
            observed_bytes,
            observed_row_group_count: None,
        });
    }
    Ok(transformed)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_gpu_resident_source(
    context: &ExecutionContext,
    source: &GpuResidentSource,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let frame = read_gpu_resident_source(context, source, gpu_inputs, metrics)?;
    let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
            .with_memory_boundary("bounded_generic_gpu_resident_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_gpu_resident_source")
            .with_attempt(1, 1),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_literal_batch_source(
    context: &ExecutionContext,
    source: &crate::source::LiteralBatchSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let frame = read_literal_batch_source(context, source, metrics)?;
    let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
            .with_memory_boundary("bounded_generic_literal_batch_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_literal_batch_source")
            .with_attempt(1, 1),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}
pub(in crate::exec::bounded) fn execute_bounded_generic_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_consuming_input_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        ReservationKind::OperatorOutput,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_from_frame_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        input_reserved_bytes,
        ReservationKind::OperatorOutput,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_consuming_input_with_reservation_kind(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservation_kind: ReservationKind,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let mut work_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input_frame);
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&work_reservation),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                drop(input_frame);
                reservations.release(work_reservation, metrics);
                reservations.release(input_reservation, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(context, input_frame, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    reservations.release(input_reservation, metrics);

    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request = MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        drop(frame);
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame_with_reservation_kind(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservation_kind: ReservationKind,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let mut work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    )?;
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&work_reservation),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(context, input, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request = MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        drop(frame);
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn bounded_generic_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Window(_) => "window",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
        PlanNode::Union(_) => "union",
        PlanNode::EdgeNormalize(_) => "edge_normalize",
        PlanNode::GraphAlgorithm(_) => "graph_algorithm",
    }
}
