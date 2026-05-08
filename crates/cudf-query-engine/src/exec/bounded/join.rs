use super::{
    super::source::{format_row_groups, read_local_parquet_chunk},
    *,
};
use crate::bounded_support::BoundedKeylessConditionalJoinType;

struct BoundedJoinSideFrame {
    frame: DataFrame,
    _reservation: ActiveReservation,
    rows: u64,
    bytes: u64,
}

pub(super) fn execute_bounded_root_join_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_root_join_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let join_node_id = plan.sink().input();
    let node = plan
        .node(join_node_id)
        .ok_or_else(|| invalid_node_reference(join_node_id, "bounded root join plan references a missing join node"))?;
    let PlanNode::Join(join) = &node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded root join execution requires the sink input to be a native join",
            "missing_root_join",
        ));
    };
    validate_bounded_join_leaf_input_capabilities(plan, join.inputs[0])?;
    validate_bounded_join_leaf_input_capabilities(plan, join.inputs[1])?;

    let left = execute_bounded_join_leaf_input(context, plan, join.inputs[0], gpu_inputs, &mut reservations, metrics)?;
    let right = execute_bounded_join_leaf_input(context, plan, join.inputs[1], gpu_inputs, &mut reservations, metrics)?;
    let committed = execute_bounded_root_join(context, node.id, join, left, right, &mut reservations, metrics)?;

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

pub(super) fn execute_bounded_join_leaf_input(
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

pub(super) fn validate_bounded_join_leaf_input_capabilities(plan: &QueryPlan, node_id: NodeId) -> Result<()> {
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

    let mut chunks = Vec::with_capacity(chunk_plan.chunks.len());
    for chunk in &chunk_plan.chunks {
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

    concatenate_join_source_chunks(context, node_id, chunks, reservations, metrics)
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
    let mut request = MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, budget_bytes)
        .with_memory_boundary("join_source_read")
        .with_node_id(node_id)
        .with_stage("join_source_read")
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
        node_id,
        "join_leaf_filter",
        "join_leaf_filter_output",
        input,
        reservations,
        metrics,
        |input, metrics| execute_filter(context, node, input, metrics),
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
        node_id,
        "join_leaf_projection",
        "join_leaf_projection_output",
        input,
        reservations,
        metrics,
        |input, metrics| execute_projection(context, node, input, output_schema, metrics),
    )
}

fn execute_bounded_join_leaf_unary(
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(GpuDataFrame, &mut QueryMetrics) -> Result<GpuDataFrame>,
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
    let result = execute(input.frame, metrics);
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

fn prepare_bounded_join_side_input(
    input: ResidentGpuFrame,
    columns: &[String],
    input_side: JoinSide,
    node_id: NodeId,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinSideFrame> {
    let reservation_kind =
        if input_side == build_side { ReservationKind::BuildSide } else { ReservationKind::ProbeSide };
    let memory_boundary = if input_side == build_side { "join_build_side" } else { "join_probe_side" };
    let work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            input.reservation.bytes(),
            reservations.budget(),
        )
        .with_memory_boundary("join_input_projection")
        .with_node_id(node_id)
        .with_stage("bounded_join")
        .with_attempt(1, 1),
        metrics,
    )?;
    let selected = match select_join_input(input.frame.into_frame(), columns, join_side_label(input_side)) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    let rows = selected.num_rows() as u64;
    let bytes = match observe_device_memory_bytes(&GpuDataFrame::new(selected.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation);
            reservations.release(input.reservation);
            return Err(error);
        },
    };
    let reservation = match reservations.try_reserve(
        bounded_join_request(reservation_kind, memory_boundary, bytes, reservations.budget(), node_id),
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
    Ok(BoundedJoinSideFrame { frame: selected, _reservation: reservation, rows, bytes })
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

fn join_side_label(side: JoinSide) -> &'static str {
    match side {
        JoinSide::Left => "left",
        JoinSide::Right => "right",
    }
}

pub(super) fn execute_bounded_root_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if node.is_bounded_root_hash_join_compatible() {
        execute_bounded_root_hash_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_filtered_join_compatible() {
        execute_bounded_root_filtered_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_keyless_conditional_join_compatible() {
        execute_bounded_root_keyless_conditional_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_projected_cross_join_compatible() {
        execute_bounded_root_projected_cross_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_mixed_join_compatible() {
        execute_bounded_root_mixed_join(context, node_id, node, left, right, reservations, metrics)
    } else {
        Err(unsupported_bounded_shape(
            "bounded root join execution currently supports reusable hash equi joins, filtered equality semi/anti joins, keyless conditional joins, projected cross joins, and mixed keyed residual joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()))
    }
}

pub(super) fn execute_bounded_root_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    if !node.is_bounded_root_hash_join_compatible() {
        return Err(unsupported_bounded_shape(
            "bounded root join execution currently supports reusable hash inner, left, and full equi joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }

    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    let overrides = execution_overrides(context);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let timer = Instant::now();
    let (left, right, hash_join, hash_table_reservation, hash_table_bytes, left_indices_from_probe) = match build_side {
        JoinSide::Left => {
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let (hash_join, hash_table_reservation, hash_table_bytes) = reserve_and_build_bounded_hash_join(
                context,
                node_id,
                node,
                &left.frame,
                &left_keys,
                hash_options.clone(),
                build_side,
                reservations,
                metrics,
                &overrides,
            )?;
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right, hash_join, hash_table_reservation, hash_table_bytes, false)
        },
        JoinSide::Right => {
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let (hash_join, hash_table_reservation, hash_table_bytes) = reserve_and_build_bounded_hash_join(
                context,
                node_id,
                node,
                &right.frame,
                &right_keys,
                hash_options.clone(),
                build_side,
                reservations,
                metrics,
                &overrides,
            )?;
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right, hash_join, hash_table_reservation, hash_table_bytes, true)
        },
    };
    let original_left_rows = left.rows;
    let original_right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;

    let (probe, probe_keys) = match build_side {
        JoinSide::Left => (&right.frame, &right_keys),
        JoinSide::Right => (&left.frame, &left_keys),
    };
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);

    let output_size = match node.spec.join_type {
        JoinType::Inner => hash_join.hash_join().inner_probe_size_in(probe, probe_keys, &overrides),
        JoinType::Left => hash_join.hash_join().left_probe_size_in(probe, probe_keys, &overrides),
        JoinType::Full => hash_join.hash_join().full_probe_size_in(probe, probe_keys, &overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_reservation_bytes =
        hash_join.hash_join().probe_gather_map_reservation_size_bytes(output_size).map(|bytes| bytes as u64).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.gather_map_reservation",
                    "failed to size bounded native reusable hash join gather maps",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", build_side)
            },
        )?;
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = match node.spec.join_type {
        JoinType::Inner => {
            hash_join.hash_join().inner_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::Left => {
            hash_join.hash_join().left_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::Full => {
            hash_join.hash_join().full_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_bytes",
            "failed to observe bounded native join gather-map memory",
            error,
        )
    })?;
    let (left_indices, right_indices) = if left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes =
        join_side_output_reservation_bytes(&left.frame, &left_indices, &left_projection, &overrides, JoinSide::Left)?;
    let right_output_bytes = join_side_output_reservation_bytes(
        &right.frame,
        &right_indices,
        &right_projection,
        &overrides,
        JoinSide::Right,
    )?;
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_indices,
        &right_indices,
        &overrides,
        &node.spec.output_projection,
    )?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_reusable_hash_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    // `left_indices`/`right_indices` are `Series::clone`s of `indices.probe()` /
    // `indices.build()`; the underlying columns are Arc-shared, so device memory
    // is only released once every Arc holder drops. All three must be released
    // before `reservations.release(gather_map_reservation)` for the budget to
    // reflect actual residency.
    drop(left_indices);
    drop(right_indices);
    drop(indices);
    reservations.release(gather_map_reservation);
    drop(hash_join);
    reservations.release(hash_table_reservation);
    drop(left);
    drop(right);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn execute_bounded_root_filtered_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    debug_assert!(node.is_bounded_root_filtered_join_compatible());

    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build_side = JoinSide::Right;
    let probe_side = JoinSide::Left;
    let right = prepare_bounded_join_side_input(
        right,
        &right_columns,
        JoinSide::Right,
        node_id,
        build_side,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let filtered_options =
        cudf::DataFrameFilteredJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = right
        .frame
        .with_execution(&overrides)
        .filtered_join_pre_build_reservation_size_bytes(&right_keys, filtered_options)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.filtered_hash_table_reservation",
                "failed to size bounded native reusable filtered join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let hash_table_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::HashTable,
            "join_hash_table",
            hash_table_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;

    let timer = Instant::now();
    let filtered_join =
        right.frame.with_execution(&overrides).build_filtered_join(&right_keys, filtered_options).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.build_filtered_join",
                "failed to build bounded native reusable filtered join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side)
        })?;
    let filtered_join =
        GpuFilteredJoinBuildHandle::new(context, filtered_join, node.spec.null_equality).map_err(|error| {
            error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", build_side)
        })?;
    let hash_table_bytes = filtered_join.device_allocated_size_bytes();
    metrics.record_filtered_join_build();
    metrics.record_filtered_join_hash_table_bytes(hash_table_bytes);

    let left = prepare_bounded_join_side_input(
        left,
        &left_columns,
        JoinSide::Left,
        node_id,
        build_side,
        reservations,
        metrics,
    )?;
    let left_rows = left.rows;
    let right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;

    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi => {
            filtered_join.filtered_join().begin_left_semi_probe_in(&left.frame, &left_keys, &overrides)
        },
        JoinType::LeftAnti => {
            filtered_join.filtered_join().begin_left_anti_probe_in(&left.frame, &left_keys, &overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.filtered_probe_state_bytes",
            "failed to observe bounded native filtered join probe-state memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", build_side)
    })?;
    let probe_state_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            probe_state_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_filtered_join_probe_state_bytes(probe_state_bytes);
    let gather_map_reservation_bytes =
        probe_state.probe_gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.filtered_gather_map_reservation",
                "failed to size bounded native reusable filtered join gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = probe_state.materialize_indices_in(&overrides).map_err(|error| join_index_error(error, node))?;
    reservations.release(probe_state_reservation);
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe bounded native filtered join gather-map memory",
            error,
        )
    })?;
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let output_reservation_bytes =
        join_side_output_reservation_bytes(&left.frame, &indices, &left_projection, &overrides, JoinSide::Left)?;
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output =
        materialize_left_only_join_output_projection(&left.frame, &indices, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_reusable_filtered_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: right_rows,
            build_bytes: right_bytes,
            probe_rows: left_rows,
            probe_bytes: left_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: probe_state_bytes,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    drop(indices);
    reservations.release(gather_map_reservation);
    drop(filtered_join);
    reservations.release(hash_table_reservation);
    drop(left);
    drop(right);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn execute_bounded_root_projected_cross_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    if !node.is_bounded_root_projected_cross_join_compatible() {
        return Err(unsupported_bounded_shape(
            "bounded root projected cross join execution requires a keyless cross join with an explicit output projection",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }

    let left_columns = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let right_columns = projected_output_columns(&node.spec.output_projection, JoinSide::Right);
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    let (left, right) = match build_side {
        JoinSide::Left => {
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right)
        },
        JoinSide::Right => {
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right)
        },
    };
    let left_rows = left.rows;
    let right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;

    let options = dataframe_cross_join_options(&node.spec.output_projection);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let plan = left.frame.projected_cross_join_plan(&right.frame, options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.projected_cross_plan",
            "failed to create bounded native projected cross join plan",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let expected_output_rows = plan.output_rows() as u64;
    let output_reservation_bytes =
        plan.reservation_size_bytes_in(&overrides).map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.projected_cross_output_reservation",
                "failed to size bounded native projected cross join output",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = plan.materialize_in(&overrides).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.projected_cross_materialize",
            "failed to materialize bounded native projected cross join output",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let output = reorder_projected_cross_join_output(output, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_projected_cross_join".to_string(),
            key_count: 0,
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left_rows, right_rows),
            build_bytes: side_metric(build_side, left_bytes, right_bytes),
            probe_rows: side_metric(probe_side, left_rows, right_rows),
            probe_bytes: side_metric(probe_side, left_bytes, right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: 0,
            output_rows,
            output_bytes,
        },
    ));

    drop(left);
    drop(right);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn execute_bounded_root_mixed_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    if !node.is_bounded_root_mixed_join_compatible() {
        return Err(unsupported_bounded_shape(
            "bounded root mixed join execution currently supports honest mixed inner, left, full, left semi, and left anti joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }

    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(
        &left_keys,
        &node.spec.output_projection,
        residual_columns.names(JoinSide::Left).iter().map(String::as_str),
        JoinSide::Left,
    );
    let right_columns = projected_input_columns(
        &right_keys,
        &node.spec.output_projection,
        residual_columns.names(JoinSide::Right).iter().map(String::as_str),
        JoinSide::Right,
    );

    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    let (left, right) = match build_side {
        JoinSide::Left => {
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right)
        },
        JoinSide::Right => {
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right)
        },
    };
    let left_rows = left.rows;
    let right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);

    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .with_execution(&overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(BoundedMixedJoinSizeData::Pair),
        JoinType::Left => left
            .frame
            .with_execution(&overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(BoundedMixedJoinSizeData::Pair),
        JoinType::Full => left
            .frame
            .with_execution(&overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(BoundedMixedJoinSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .with_execution(&overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(BoundedMixedJoinSizeData::LeftOnly),
        JoinType::LeftAnti => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(BoundedMixedJoinSizeData::LeftOnly),
        JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let expected_output_rows = size_data.output_size() as u64;
    let size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    let size_data_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::MixedSizeData,
            "join_mixed_size_data",
            size_data_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_mixed_join_size_data_bytes(size_data_bytes);
    let gather_map_reservation_bytes = size_data.gather_map_reservation_size_bytes(node)?;
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = size_data.materialize_indices_in(&overrides, node)?;
    reservations.release(size_data_reservation);
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;
    let output_reservation_bytes =
        indices.output_reservation_size_bytes(&left.frame, &right.frame, &overrides, &node.spec.output_projection)?;
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = indices.materialize_output(&left.frame, &right.frame, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_mixed_keyed_residual_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left_rows, right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, left_rows, right_rows),
            probe_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: size_data_bytes,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    drop(indices);
    reservations.release(gather_map_reservation);
    drop(left);
    drop(right);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

enum BoundedMixedJoinSizeData<'a> {
    Pair(cudf::DataFrameMixedJoinSizeData<'a>),
    LeftOnly(cudf::DataFrameMixedSingleSideJoinSizeData<'a>),
}

impl BoundedMixedJoinSizeData<'_> {
    fn output_size(&self) -> usize {
        match self {
            Self::Pair(size_data) => size_data.output_size(),
            Self::LeftOnly(size_data) => size_data.output_size(),
        }
    }

    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        let result = match self {
            Self::Pair(size_data) => size_data.device_allocated_size_bytes(),
            Self::LeftOnly(size_data) => size_data.device_allocated_size_bytes(),
        };
        result.map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.mixed_size_data_bytes",
                "failed to observe bounded native mixed join size-data memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })
    }

    fn gather_map_reservation_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        let result = match self {
            Self::Pair(size_data) => size_data.gather_map_reservation_size_bytes(),
            Self::LeftOnly(size_data) => size_data.gather_map_reservation_size_bytes(),
        };
        result.map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.mixed_gather_map_reservation",
                "failed to size bounded native mixed join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })
    }

    fn materialize_indices_in(
        self,
        overrides: &ExecutionOverrides<'_>,
        node: &JoinNode,
    ) -> Result<BoundedMixedJoinIndices> {
        match self {
            Self::Pair(size_data) => size_data
                .materialize_indices_in(overrides)
                .map(BoundedMixedJoinIndices::Pair)
                .map_err(|error| join_index_error(error, node)),
            Self::LeftOnly(size_data) => size_data
                .materialize_indices_in(overrides)
                .map(BoundedMixedJoinIndices::LeftOnly)
                .map_err(|error| join_index_error(error, node)),
        }
    }
}

enum BoundedMixedJoinIndices {
    Pair(cudf::DataFrameJoinPairIndices),
    LeftOnly(cudf::DataFrameJoinIndices),
}

impl BoundedMixedJoinIndices {
    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        match self {
            Self::Pair(indices) => indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.mixed_gather_map_bytes",
                    "failed to observe bounded native mixed join gather-map memory",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
            }),
            Self::LeftOnly(indices) => {
                indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.mixed_gather_map_bytes",
                        "failed to observe bounded native mixed single-side join gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })
            },
        }
    }

    fn output_reservation_size_bytes(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
    ) -> Result<u64> {
        match self {
            Self::Pair(indices) => {
                let left_projection = projected_output_columns(projection, JoinSide::Left);
                let right_projection = projected_output_columns(projection, JoinSide::Right);
                let left_bytes = join_side_output_reservation_bytes(
                    left,
                    indices.left(),
                    &left_projection,
                    overrides,
                    JoinSide::Left,
                )?;
                let right_bytes = join_side_output_reservation_bytes(
                    right,
                    indices.right(),
                    &right_projection,
                    overrides,
                    JoinSide::Right,
                )?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                let left_projection = projected_output_columns(projection, JoinSide::Left);
                join_side_output_reservation_bytes(left, indices, &left_projection, overrides, JoinSide::Left)
            },
        }
    }

    fn materialize_output(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
    ) -> Result<DataFrame> {
        match self {
            Self::Pair(indices) => {
                materialize_join_output_projection(left, right, indices.left(), indices.right(), overrides, projection)
            },
            Self::LeftOnly(indices) => {
                materialize_left_only_join_output_projection(left, indices, overrides, projection)
            },
        }
    }
}

fn execute_bounded_root_keyless_conditional_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    debug_assert!(node.is_bounded_root_keyless_conditional_join_compatible());
    let join_type = BoundedKeylessConditionalJoinType::try_from_join(node)
        .expect("bounded conditional join executor is reached only through the compatible dispatcher");

    let residual_columns = residual_input_columns(node)?;
    let left_columns = projected_input_columns(
        &[],
        &node.spec.output_projection,
        residual_columns.names(JoinSide::Left).iter().map(String::as_str),
        JoinSide::Left,
    );
    let right_columns = projected_input_columns(
        &[],
        &node.spec.output_projection,
        residual_columns.names(JoinSide::Right).iter().map(String::as_str),
        JoinSide::Right,
    );

    let left = prepare_bounded_join_side_input(
        left,
        &left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Left,
        reservations,
        metrics,
    )?;
    let right = prepare_bounded_join_side_input(
        right,
        &right_columns,
        JoinSide::Right,
        node_id,
        JoinSide::Left,
        reservations,
        metrics,
    )?;
    let left_rows = left.rows;
    let right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;

    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_predicate_join_options(&residual_columns);
    let timer = Instant::now();
    let (size_data_bytes, expected_output_rows, gather_map_reservation_bytes, work) =
        if join_type == BoundedKeylessConditionalJoinType::Full {
            let size_data = left
                .frame
                .with_execution(&overrides)
                .conditional_full_join_size_data(&right.frame, &predicate, options)
                .map_err(|error| join_index_error(error, node))?;
            let expected_output_rows = size_data.output_size() as u64;
            let size_data_bytes = conditional_full_join_size_data_device_allocated_size_bytes(&size_data, node)?;
            let size_data_reservation = reservations.try_reserve(
                bounded_join_request(
                    ReservationKind::MixedSizeData,
                    "join_mixed_size_data",
                    size_data_bytes,
                    reservations.budget(),
                    node_id,
                ),
                metrics,
            )?;
            metrics.record_mixed_join_size_data_bytes(size_data_bytes);
            let gather_map_reservation_bytes =
                conditional_full_join_size_data_gather_map_reservation_size_bytes(&size_data, node)?;
            (
                size_data_bytes,
                expected_output_rows,
                gather_map_reservation_bytes,
                BoundedKeylessConditionalJoinWork::Full { size_data, size_data_reservation },
            )
        } else {
            let output_size = keyless_conditional_join_output_size(
                join_type,
                &left.frame,
                &right.frame,
                &predicate,
                options.clone(),
                &overrides,
                node,
            )?;
            let gather_map_reservation_bytes =
                keyless_conditional_join_gather_map_reservation_size_bytes(join_type, node, output_size)?;
            (
                0,
                output_size as u64,
                gather_map_reservation_bytes,
                BoundedKeylessConditionalJoinWork::Sized { output_size, options },
            )
        };
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = match work {
        BoundedKeylessConditionalJoinWork::Full { size_data, size_data_reservation } => {
            let indices = size_data.materialize_indices_in(&overrides);
            reservations.release(size_data_reservation);
            let indices = indices.map_err(|error| join_index_error(error, node))?;
            BoundedKeylessConditionalJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() }
        },
        BoundedKeylessConditionalJoinWork::Sized { output_size, options } => {
            keyless_conditional_join_indices_with_output_size(
                join_type,
                &left.frame,
                &right.frame,
                &predicate,
                options,
                output_size,
                &overrides,
                node,
            )?
        },
    };
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;

    let output_reservation_bytes =
        indices.output_reservation_size_bytes(&left.frame, &right.frame, &overrides, &node.spec.output_projection)?;
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = indices.materialize_output(&left.frame, &right.frame, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_keyless_conditional_join".to_string(),
            key_count: 0,
            residual_predicate: true,
            null_equality: node.spec.null_equality.to_string(),
            build_side: JoinSide::Left.to_string(),
            probe_side: JoinSide::Right.to_string(),
            build_rows: left_rows,
            build_bytes: left_bytes,
            probe_rows: right_rows,
            probe_bytes: right_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: size_data_bytes,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    drop(indices);
    reservations.release(gather_map_reservation);
    drop(right);
    drop(left);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

enum BoundedKeylessConditionalJoinIndices {
    Pair { left: cudf::DataFrameJoinIndices, right: cudf::DataFrameJoinIndices },
    LeftOnly(cudf::DataFrameJoinIndices),
}

enum BoundedKeylessConditionalJoinWork<'a> {
    Full { size_data: cudf::DataFrameConditionalFullJoinSizeData<'a>, size_data_reservation: ActiveReservation },
    Sized { output_size: usize, options: cudf::DataFramePredicateJoinOptions },
}

impl BoundedKeylessConditionalJoinIndices {
    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        match self {
            Self::Pair { left, right } => {
                let left_bytes = left.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe bounded native conditional join left gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe bounded native conditional join right gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe bounded native conditional join gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })
            },
        }
    }

    fn output_reservation_size_bytes(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
    ) -> Result<u64> {
        match self {
            Self::Pair { left: left_indices, right: right_indices } => {
                let left_projection = projected_output_columns(projection, JoinSide::Left);
                let right_projection = projected_output_columns(projection, JoinSide::Right);
                let left_bytes = join_side_output_reservation_bytes(
                    left,
                    left_indices,
                    &left_projection,
                    overrides,
                    JoinSide::Left,
                )?;
                let right_bytes = join_side_output_reservation_bytes(
                    right,
                    right_indices,
                    &right_projection,
                    overrides,
                    JoinSide::Right,
                )?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                let left_projection = projected_output_columns(projection, JoinSide::Left);
                join_side_output_reservation_bytes(left, indices, &left_projection, overrides, JoinSide::Left)
            },
        }
    }

    fn materialize_output(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
    ) -> Result<DataFrame> {
        match self {
            Self::Pair { left: left_indices, right: right_indices } => {
                materialize_join_output_projection(left, right, left_indices, right_indices, overrides, projection)
            },
            Self::LeftOnly(indices) => {
                materialize_left_only_join_output_projection(left, indices, overrides, projection)
            },
        }
    }
}

fn keyless_conditional_join_output_size(
    join_type: BoundedKeylessConditionalJoinType,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<usize> {
    let result = match join_type {
        BoundedKeylessConditionalJoinType::Inner => {
            left.with_execution(overrides).conditional_inner_join_size(right, predicate, options)
        },
        BoundedKeylessConditionalJoinType::Left => {
            left.with_execution(overrides).conditional_left_join_size(right, predicate, options)
        },
        BoundedKeylessConditionalJoinType::Full => {
            unreachable!("conditional full joins use retained size data for output-size observation")
        },
        BoundedKeylessConditionalJoinType::LeftSemi => {
            left.with_execution(overrides).conditional_left_semi_join_size(right, predicate, options)
        },
        BoundedKeylessConditionalJoinType::LeftAnti => {
            left.with_execution(overrides).conditional_left_anti_join_size(right, predicate, options)
        },
    };
    result.map_err(|error| join_index_error(error, node))
}

fn keyless_conditional_join_gather_map_reservation_size_bytes(
    join_type: BoundedKeylessConditionalJoinType,
    node: &JoinNode,
    output_size: usize,
) -> Result<u64> {
    let result = match join_type {
        BoundedKeylessConditionalJoinType::Inner | BoundedKeylessConditionalJoinType::Left => {
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size)
        },
        BoundedKeylessConditionalJoinType::Full => {
            unreachable!("conditional full joins use retained size data for gather-map reservation")
        },
        BoundedKeylessConditionalJoinType::LeftSemi | BoundedKeylessConditionalJoinType::LeftAnti => {
            cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(output_size)
        },
    };
    result.map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_reservation",
            "failed to size bounded native conditional join gather maps",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })
}

fn keyless_conditional_join_indices_with_output_size(
    join_type: BoundedKeylessConditionalJoinType,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    output_size: usize,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<BoundedKeylessConditionalJoinIndices> {
    match join_type {
        BoundedKeylessConditionalJoinType::Inner => {
            let indices = left
                .with_execution(overrides)
                .conditional_inner_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(BoundedKeylessConditionalJoinIndices::Pair {
                left: indices.left().clone(),
                right: indices.right().clone(),
            })
        },
        BoundedKeylessConditionalJoinType::Left => {
            let indices = left
                .with_execution(overrides)
                .conditional_left_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(BoundedKeylessConditionalJoinIndices::Pair {
                left: indices.left().clone(),
                right: indices.right().clone(),
            })
        },
        BoundedKeylessConditionalJoinType::Full => {
            unreachable!("conditional full joins use retained size data for materialization")
        },
        BoundedKeylessConditionalJoinType::LeftSemi => left
            .with_execution(overrides)
            .conditional_left_semi_join_indices_with_output_size(right, predicate, options, output_size)
            .map(BoundedKeylessConditionalJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        BoundedKeylessConditionalJoinType::LeftAnti => left
            .with_execution(overrides)
            .conditional_left_anti_join_indices_with_output_size(right, predicate, options, output_size)
            .map(BoundedKeylessConditionalJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
    }
}

fn conditional_full_join_size_data_device_allocated_size_bytes(
    size_data: &cudf::DataFrameConditionalFullJoinSizeData<'_>,
    node: &JoinNode,
) -> Result<u64> {
    size_data.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.conditional_full_size_data_bytes",
            "failed to observe bounded native conditional full join size-data memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })
}

fn conditional_full_join_size_data_gather_map_reservation_size_bytes(
    size_data: &cudf::DataFrameConditionalFullJoinSizeData<'_>,
    node: &JoinNode,
) -> Result<u64> {
    size_data.gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.conditional_full_gather_map_reservation",
            "failed to size bounded native conditional full join gather maps",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })
}

fn bounded_join_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
) -> MemoryReservationRequest {
    MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage("bounded_join")
        .with_attempt(1, 1)
}

#[allow(clippy::too_many_arguments)]
fn reserve_and_build_bounded_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &DataFrame,
    build_keys: &[String],
    hash_options: cudf::DataFrameHashJoinOptions,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    overrides: &ExecutionOverrides<'_>,
) -> Result<(GpuHashJoinBuildHandle, ActiveReservation, u64)> {
    let hash_table_reservation_bytes = build
        .with_execution(overrides)
        .hash_join_pre_build_reservation_size_bytes(build_keys, hash_options.clone())
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.hash_table_reservation",
                "failed to size bounded native reusable hash join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let hash_table_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::HashTable,
            "join_hash_table",
            hash_table_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let hash_join = build.with_execution(overrides).build_hash_join(build_keys, hash_options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.build_hash_join",
            "failed to build bounded native reusable hash join",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("build_side", build_side)
    })?;
    let hash_join = GpuHashJoinBuildHandle::new(context, hash_join, node.spec.null_equality).map_err(|error| {
        error
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side)
    })?;
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    metrics.record_hash_join_build();
    metrics.record_hash_join_hash_table_bytes(hash_table_bytes);
    Ok((hash_join, hash_table_reservation, hash_table_bytes))
}

fn join_side_output_reservation_bytes(
    frame: &DataFrame,
    indices: &cudf::DataFrameJoinIndices,
    columns: &[String],
    overrides: &ExecutionOverrides<'_>,
    side: JoinSide,
) -> Result<u64> {
    if columns.is_empty() {
        return Ok(0);
    }
    frame
        .with_execution(overrides)
        .materialize_join_indices_reservation_size_bytes(indices, columns)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.output_reservation",
                "failed to size bounded native join output materialization",
                error,
            )
            .with_fact("side", side)
        })
}
