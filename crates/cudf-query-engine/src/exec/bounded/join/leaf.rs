use std::collections::VecDeque;

use super::*;
use crate::exec::source::{ScheduledLocalParquetRead, format_row_groups, read_local_parquet_chunk};

pub(in crate::exec::bounded) fn execute_bounded_join_leaf_input(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let node = plan
        .node(node_id)
        .ok_or_else(|| invalid_node_reference(node_id, "bounded join input references a missing plan node"))?;
    match &node.kind {
        PlanNode::Source(source) => {
            execute_bounded_join_leaf_source(context, source.as_ref(), node_id, gpu_inputs, reservations, metrics)
        },
        PlanNode::Filter(filter) => {
            let input =
                execute_bounded_join_leaf_input(context, plan, filter.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_join_leaf_filter(context, node_id, filter, input, reservations, metrics)
        },
        PlanNode::Projection(projection) => {
            let input =
                execute_bounded_join_leaf_input(context, plan, projection.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_join_leaf_projection(
                context,
                node_id,
                projection,
                &node.facts.output_schema,
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::Join(_) | PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
            Err(unsupported_bounded_shape(
                "bounded join input execution supports only source/filter/projection leaf subtrees",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", node_id)
            .with_fact("node_kind", bounded_join_leaf_node_kind(&node.kind)))
        },
    }
}

pub(in crate::exec::bounded) fn validate_bounded_join_leaf_input_capabilities(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    let node = plan
        .node(node_id)
        .ok_or_else(|| invalid_node_reference(node_id, "bounded join input references a missing plan node"))?;
    match &node.kind {
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => {
            lower_expr(&filter.predicate)?;
            validate_bounded_join_leaf_input_capabilities(plan, filter.input)
        },
        PlanNode::Projection(projection) => {
            validate_projection_capability(projection, &node.facts.output_schema)?;
            validate_bounded_join_leaf_input_capabilities(plan, projection.input)
        },
        PlanNode::Join(_) | PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => {
            Err(unsupported_bounded_shape(
                "bounded join input execution supports only source/filter/projection leaf subtrees",
                "bounded_join_plan_shape_unsupported",
            )
            .with_fact("node_id", node_id)
            .with_fact("node_kind", bounded_join_leaf_node_kind(&node.kind)))
        },
    }
}

fn execute_bounded_join_leaf_source(
    context: &ExecutionContext,
    source: &crate::source::SourceNode,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match source {
        crate::source::SourceNode::LocalParquet(source) => {
            validate_local_parquet_source(source)?;
            execute_bounded_join_leaf_local_parquet_source(context, source, node_id, reservations, metrics)
        },
        crate::source::SourceNode::GpuResident(source) => {
            let frame = read_gpu_resident_source(context, source, gpu_inputs, metrics)?;
            let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
            let reservation = reservations.try_reserve(
                MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
                    .with_memory_boundary("gpu_resident_join_input")
                    .with_node_id(node_id)
                    .with_stage("gpu_resident_join_input")
                    .with_attempt(1, 1),
                metrics,
            )?;
            Ok(ResidentGpuFrame { frame, reservation })
        },
    }
}

fn execute_bounded_join_leaf_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let chunk_plan = plan_local_parquet_source_chunks(
        node_id,
        source,
        SourceChunkPlanningPolicy::from(context.memory_control_policy()),
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.join.source",
            "bounded join leaf local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);

    let chunks = if context.source_scheduler_policy().enables_prefetch() && chunk_plan.chunks.len() > 1 {
        read_join_leaf_source_chunks_with_scheduler(
            context,
            source,
            node_id,
            &chunk_plan.chunks,
            reservations,
            metrics,
        )?
    } else {
        read_join_leaf_source_chunks_serial(context, source, node_id, &chunk_plan.chunks, reservations, metrics)?
    };

    concatenate_join_source_chunks(context, node_id, chunks, reservations, metrics)
}

fn read_join_leaf_source_chunks_serial(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let mut chunks = Vec::with_capacity(source_chunks.len());
    for chunk in source_chunks {
        metrics.record_source_chunk_attempt();
        let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
        let mut reservation = match reservations.try_reserve(
            join_source_read_reservation_request(requested_bytes, reservations.budget(), node_id, chunk),
            metrics,
        ) {
            Ok(reservation) => reservation,
            Err(error) => {
                release_join_source_chunks_reverse(reservations, chunks);
                return Err(error);
            },
        };
        let source_token_budget = reservations.available_bytes_including(&reservation);
        let frame =
            match read_local_parquet_chunk(context, source, chunk, source_token_budget, "join_source_read", metrics) {
                Ok(frame) => frame,
                Err(error) => {
                    reservations.release(reservation);
                    release_join_source_chunks_reverse(reservations, chunks);
                    return Err(attach_join_source_read_resource_facts(error, node_id, chunk));
                },
            };
        if let Err(error) = grow_join_source_reservation_to_observed_frame(
            &frame,
            &mut reservation,
            reservations,
            node_id,
            chunk,
            metrics,
        ) {
            drop(frame);
            reservations.release(reservation);
            release_join_source_chunks_reverse(reservations, chunks);
            return Err(error);
        }
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        chunks.push((frame, reservation));
    }

    Ok(chunks)
}

fn read_join_leaf_source_chunks_with_scheduler(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    source_chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<(GpuDataFrame, ActiveReservation)>> {
    let mut pending = VecDeque::new();
    let mut next_chunk_index = 0usize;
    start_next_required_join_leaf_read(
        context,
        source,
        source_chunks,
        node_id,
        reservations,
        metrics,
        &mut pending,
        &mut next_chunk_index,
    )?;
    if let Err(error) = fill_optional_join_leaf_reads(
        context,
        source,
        source_chunks,
        node_id,
        reservations,
        metrics,
        &mut pending,
        &mut next_chunk_index,
    ) {
        abandon_join_leaf_reads(pending, reservations, metrics);
        return Err(error);
    }

    let mut chunks = Vec::with_capacity(source_chunks.len());
    for (index, chunk) in source_chunks.iter().enumerate() {
        let scheduled =
            pending.pop_front().expect("join leaf source scheduler should have a read for the current chunk");
        let is_prefetch = scheduled.is_prefetch();
        let mut output = match scheduled.finish(reservations, metrics) {
            Ok(output) => output,
            Err(error) => {
                abandon_join_leaf_reads(pending, reservations, metrics);
                release_join_source_chunks_reverse(reservations, chunks);
                return Err(attach_join_source_read_resource_facts(error, node_id, chunk));
            },
        };
        if is_prefetch {
            metrics.record_source_read_prefetch_completed();
        }
        if let Err(error) = grow_join_source_reservation_to_observed_frame(
            &output.frame,
            &mut output.reservation,
            reservations,
            node_id,
            chunk,
            metrics,
        ) {
            drop(output.frame);
            reservations.release(output.reservation);
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks);
            return Err(error);
        }
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
        chunks.push((output.frame, output.reservation));

        if let Err(error) = fill_optional_join_leaf_reads(
            context,
            source,
            source_chunks,
            node_id,
            reservations,
            metrics,
            &mut pending,
            &mut next_chunk_index,
        ) {
            abandon_join_leaf_reads(pending, reservations, metrics);
            release_join_source_chunks_reverse(reservations, chunks);
            return Err(error);
        }
        if pending.is_empty() && index + 1 < source_chunks.len() {
            if let Err(error) = start_next_required_join_leaf_read(
                context,
                source,
                source_chunks,
                node_id,
                reservations,
                metrics,
                &mut pending,
                &mut next_chunk_index,
            ) {
                release_join_source_chunks_reverse(reservations, chunks);
                return Err(error);
            }
        }
    }

    Ok(chunks)
}

#[allow(clippy::too_many_arguments)]
fn start_next_required_join_leaf_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
) -> Result<()> {
    if let Some(chunk) = chunks.get(*next_chunk_index) {
        let requested_bytes = chunk_requested_bytes(context.memory_control_policy(), chunk);
        pending.push_back(start_join_leaf_scheduled_source_read(
            context,
            source,
            chunk,
            node_id,
            reservations,
            metrics,
            requested_bytes,
            "join_source_read",
            false,
            false,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill_optional_join_leaf_reads(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    pending: &mut VecDeque<ScheduledLocalParquetRead>,
    next_chunk_index: &mut usize,
) -> Result<()> {
    let max_in_flight = context.source_scheduler_policy().max_source_reads_in_flight().max(1);
    while pending.len() < max_in_flight {
        let Some(chunk) = chunks.get(*next_chunk_index) else {
            break;
        };
        let requested_bytes = prefetch_requested_bytes(context.memory_control_policy(), chunk);
        if !reservations.can_reserve(requested_bytes) {
            metrics.record_source_read_prefetch_skipped();
            break;
        }
        metrics.record_source_read_prefetch_started();
        pending.push_back(start_join_leaf_scheduled_source_read(
            context,
            source,
            chunk,
            node_id,
            reservations,
            metrics,
            requested_bytes,
            "join_source_read_prefetch",
            true,
            true,
        )?);
        *next_chunk_index += 1;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn start_join_leaf_scheduled_source_read(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    requested_bytes: u64,
    memory_boundary: &'static str,
    strict_token_budget: bool,
    prefetch: bool,
) -> Result<ScheduledLocalParquetRead> {
    metrics.record_source_chunk_attempt();
    let request = join_source_read_reservation_request_for_boundary(
        requested_bytes,
        reservations.budget(),
        node_id,
        chunk,
        memory_boundary,
    );
    let reservation = reservations.try_reserve(request, metrics)?;
    let source_token_budget =
        if strict_token_budget { reservation.bytes() } else { reservations.available_bytes_including(&reservation) };
    Ok(ScheduledLocalParquetRead::spawn(
        context.clone(),
        source.clone(),
        chunk.clone(),
        reservation,
        source_token_budget,
        memory_boundary,
        prefetch,
    ))
}

fn abandon_join_leaf_reads(
    mut pending: VecDeque<ScheduledLocalParquetRead>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    while let Some(prefetch) = pending.pop_back() {
        prefetch.abandon(reservations, metrics);
    }
}

fn attach_join_source_read_resource_facts(error: QueryError, node_id: NodeId, chunk: &SourceChunk) -> QueryError {
    error
        .with_fact_if_absent("memory_boundary", "join_source_read")
        .with_fact_if_absent("stage", "join_source_read")
        .with_fact_if_absent("node_id", node_id)
        .with_fact_if_absent("attempt", 1)
        .with_fact_if_absent("max_attempts", 1)
        .with_fact_if_absent("chunk_index", chunk.chunk_index)
        .with_fact_if_absent("file_index", chunk.file_index)
        .with_fact_if_absent("path", chunk.path.display())
        .with_fact_if_absent("row_groups", format_row_groups(chunk))
}

fn join_source_read_reservation_request(
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
) -> MemoryReservationRequest {
    join_source_read_reservation_request_for_boundary(requested_bytes, budget_bytes, node_id, chunk, "join_source_read")
}

fn join_source_read_reservation_request_for_boundary(
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(1, 1);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request
}

fn grow_join_source_reservation_to_observed_frame(
    frame: &GpuDataFrame,
    reservation: &mut ActiveReservation,
    reservations: &mut ReservationTracker,
    node_id: NodeId,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) -> Result<()> {
    let observed_bytes = observe_device_memory_bytes(frame, metrics)?;
    if observed_bytes <= reservation.bytes() {
        return Ok(());
    }
    let delta = observed_bytes.saturating_sub(reservation.bytes());
    reservations.try_grow(
        reservation,
        join_source_read_reservation_request(delta, reservations.budget(), node_id, chunk),
        metrics,
    )
}

fn concatenate_join_source_chunks(
    context: &ExecutionContext,
    node_id: NodeId,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let (frame, reservation) = chunks.pop().expect("single chunk checked");
        return Ok(ResidentGpuFrame { frame, reservation });
    }

    let output_reservation_bytes =
        chunks.iter().fold(0u64, |total, (_frame, reservation)| total.saturating_add(reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, output_reservation_bytes, reservations.budget())
            .with_memory_boundary("join_source_read")
            .with_node_id(node_id)
            .with_stage("join_source_read")
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_join_source_chunks_reverse(reservations, chunks);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(chunks.len());
    let mut frames = Vec::with_capacity(chunks.len());
    for (frame, reservation) in chunks {
        frames.push(frame.into_frame());
        input_reservations.push(reservation);
    }
    let chunk_count = input_reservations.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations);
            reservations.release(output_reservation);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.source_concat",
                "failed to concatenate bounded join leaf local parquet source chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations);
            reservations.release(output_reservation);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            MemoryReservationRequest::new(ReservationKind::SourceRead, delta, reservations.budget())
                .with_memory_boundary("join_source_read")
                .with_node_id(node_id)
                .with_stage("join_source_read")
                .with_attempt(1, 1),
            metrics,
        ) {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_join_source_reservations_reverse(reservations, input_reservations);
            reservations.release(output_reservation);
            return Err(error);
        }
    }
    drop(rest_frame_refs);
    drop(frames);
    drop(first_frame);
    release_join_source_reservations_reverse(reservations, input_reservations);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn release_join_source_chunks_reverse(
    reservations: &mut ReservationTracker,
    mut chunks: Vec<(GpuDataFrame, ActiveReservation)>,
) {
    while let Some((frame, reservation)) = chunks.pop() {
        drop(frame);
        reservations.release(reservation);
    }
}

fn release_join_source_reservations_reverse(reservations: &mut ReservationTracker, mut active: Vec<ActiveReservation>) {
    while let Some(reservation) = active.pop() {
        reservations.release(reservation);
    }
}

fn execute_bounded_join_leaf_filter(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &FilterNode,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    execute_bounded_join_leaf_unary(
        context,
        node_id,
        "join_leaf_filter",
        "join_leaf_filter_output",
        input,
        reservations,
        metrics,
        false,
        |context, input, _token, metrics| execute_filter(context, node, input, metrics),
    )
}

fn execute_bounded_join_leaf_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &ProjectionNode,
    output_schema: &arrow::datatypes::SchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    execute_bounded_join_leaf_unary(
        context,
        node_id,
        "join_leaf_projection",
        "join_leaf_projection_output",
        input,
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            debug_assert_eq!(node.output_schema().as_ref(), output_schema.as_ref());
            execute_projection_with_budget(
                context,
                node,
                input,
                token.expect("budget token is created for join leaf projection stages"),
                metrics,
            )
        },
    )
}

fn execute_bounded_join_leaf_unary(
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
    let work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            input.reservation.bytes(),
            reservations.budget(),
        )
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
            "cudf_query_engine.bounded_join_leaf.budget_token",
            "failed to create bounded native join leaf operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation);
                reservations.release(input.reservation);
                return Err(error);
            },
        }
    } else {
        None
    };
    let result = execute(context, input.frame, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    let output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    reservations.release(work_reservation);
    reservations.release(input.reservation);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

fn bounded_join_leaf_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
    }
}
