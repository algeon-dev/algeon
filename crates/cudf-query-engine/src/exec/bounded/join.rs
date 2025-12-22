use super::*;

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
    let mut scratch = ExecutionScratch::new(context);
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

    let left = execute_node(context, plan, join.inputs[0], gpu_inputs, &mut scratch, metrics)?;
    let right = execute_node(context, plan, join.inputs[1], gpu_inputs, &mut scratch, metrics)?;
    let committed = execute_bounded_root_hash_join(context, node.id, join, left, right, &mut reservations, metrics)?;

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

pub(super) fn execute_bounded_root_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
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

    let left = left.into_frame();
    let right = right.into_frame();
    let original_left_rows = left.num_rows() as u64;
    let original_right_rows = right.num_rows() as u64;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let left = select_join_input(left, &left_columns, "left")?;
    let right = select_join_input(right, &right_columns, "right")?;
    let left_bytes = observe_device_memory_bytes(&GpuDataFrame::new(left.clone()), metrics)?;
    let right_bytes = observe_device_memory_bytes(&GpuDataFrame::new(right.clone()), metrics)?;

    let (build, build_keys, probe, probe_keys, build_side, left_indices_from_probe) = match node.spec.build_side {
        JoinSide::Left => (&left, &left_keys, &right, &right_keys, JoinSide::Left, false),
        JoinSide::Right => (&right, &right_keys, &left, &left_keys, JoinSide::Right, true),
    };
    let probe_side = probe_side(build_side);
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);

    let build_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::BuildSide,
            "join_build_side",
            build_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = build
        .with_execution(&overrides)
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
    let timer = Instant::now();
    let hash_join = build.with_execution(&overrides).build_hash_join(build_keys, hash_options).map_err(|error| {
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

    let probe_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::ProbeSide,
            "join_probe_side",
            probe_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
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
        join_side_output_reservation_bytes(&left, &left_indices, &left_projection, &overrides, JoinSide::Left)?;
    let right_output_bytes =
        join_side_output_reservation_bytes(&right, &right_indices, &right_projection, &overrides, JoinSide::Right)?;
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
        &left,
        &right,
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
    reservations.release(probe_reservation);
    reservations.release(build_reservation);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
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
