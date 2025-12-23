use super::*;
use crate::bounded_support::BoundedConditionalJoinType;

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

pub(super) fn execute_bounded_root_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if node.is_bounded_root_hash_join_compatible() {
        execute_bounded_root_hash_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_filtered_join_compatible() {
        execute_bounded_root_filtered_join(context, node_id, node, left, right, reservations, metrics)
    } else if node.is_bounded_root_conditional_join_compatible() {
        execute_bounded_root_conditional_join(context, node_id, node, left, right, reservations, metrics)
    } else {
        Err(unsupported_bounded_shape(
            "bounded root join execution currently supports reusable hash equi joins, filtered equality semi/anti joins, and keyless conditional joins",
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

fn execute_bounded_root_filtered_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    debug_assert!(node.is_bounded_root_filtered_join_compatible());

    let left = left.into_frame();
    let right = right.into_frame();
    let left_rows = left.num_rows() as u64;
    let right_rows = right.num_rows() as u64;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let left = select_join_input(left, &left_columns, "left")?;
    let right = select_join_input(right, &right_columns, "right")?;
    let left_bytes = observe_device_memory_bytes(&GpuDataFrame::new(left.clone()), metrics)?;
    let right_bytes = observe_device_memory_bytes(&GpuDataFrame::new(right.clone()), metrics)?;

    let build_side = JoinSide::Right;
    let probe_side = JoinSide::Left;
    let build_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::BuildSide,
            "join_build_side",
            right_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;

    let overrides = execution_overrides(context);
    let filtered_options =
        cudf::DataFrameFilteredJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = right
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
        right.with_execution(&overrides).build_filtered_join(&right_keys, filtered_options).map_err(|error| {
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

    let probe_reservation = reservations.try_reserve(
        bounded_join_request(ReservationKind::ProbeSide, "join_probe_side", left_bytes, reservations.budget(), node_id),
        metrics,
    )?;
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi => filtered_join.filtered_join().begin_left_semi_probe_in(&left, &left_keys, &overrides),
        JoinType::LeftAnti => filtered_join.filtered_join().begin_left_anti_probe_in(&left, &left_keys, &overrides),
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
        join_side_output_reservation_bytes(&left, &indices, &left_projection, &overrides, JoinSide::Left)?;
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
        materialize_left_only_join_output_projection(&left, &indices, &overrides, &node.spec.output_projection)?;
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
    reservations.release(probe_reservation);
    reservations.release(build_reservation);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn execute_bounded_root_conditional_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    debug_assert!(node.is_bounded_root_conditional_join_compatible());
    let join_type = BoundedConditionalJoinType::try_from_join(node)
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

    let left = select_join_input(left.into_frame(), &left_columns, "left")?;
    let right = select_join_input(right.into_frame(), &right_columns, "right")?;
    let left_rows = left.num_rows() as u64;
    let right_rows = right.num_rows() as u64;
    let left_bytes = observe_device_memory_bytes(&GpuDataFrame::new(left.clone()), metrics)?;
    let right_bytes = observe_device_memory_bytes(&GpuDataFrame::new(right.clone()), metrics)?;

    let left_reservation = reservations.try_reserve(
        bounded_join_request(ReservationKind::BuildSide, "join_left_side", left_bytes, reservations.budget(), node_id),
        metrics,
    )?;
    let right_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::ProbeSide,
            "join_right_side",
            right_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;

    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_predicate_join_options(&residual_columns);
    let timer = Instant::now();
    let output_size =
        conditional_join_output_size(join_type, &left, &right, &predicate, options.clone(), &overrides, node)?;
    let gather_map_reservation_bytes =
        conditional_join_gather_map_reservation_size_bytes(join_type, node, output_size)?;
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
    let indices = conditional_join_indices_with_output_size(
        join_type,
        &left,
        &right,
        &predicate,
        options,
        output_size,
        &overrides,
        node,
    )?;
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;

    let output_reservation_bytes =
        indices.output_reservation_size_bytes(&left, &right, &overrides, &node.spec.output_projection)?;
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
    let output = indices.materialize_output(&left, &right, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
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
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    drop(indices);
    reservations.release(gather_map_reservation);
    drop(right);
    reservations.release(right_reservation);
    drop(left);
    reservations.release(left_reservation);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

enum BoundedConditionalJoinIndices {
    Pair { left: cudf::DataFrameJoinIndices, right: cudf::DataFrameJoinIndices },
    LeftOnly(cudf::DataFrameJoinIndices),
}

impl BoundedConditionalJoinIndices {
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

fn conditional_join_output_size(
    join_type: BoundedConditionalJoinType,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<usize> {
    let result = match join_type {
        BoundedConditionalJoinType::Inner => {
            left.with_execution(overrides).conditional_inner_join_size(right, predicate, options)
        },
        BoundedConditionalJoinType::Left => {
            left.with_execution(overrides).conditional_left_join_size(right, predicate, options)
        },
        BoundedConditionalJoinType::LeftSemi => {
            left.with_execution(overrides).conditional_left_semi_join_size(right, predicate, options)
        },
        BoundedConditionalJoinType::LeftAnti => {
            left.with_execution(overrides).conditional_left_anti_join_size(right, predicate, options)
        },
    };
    result.map_err(|error| join_index_error(error, node))
}

fn conditional_join_gather_map_reservation_size_bytes(
    join_type: BoundedConditionalJoinType,
    node: &JoinNode,
    output_size: usize,
) -> Result<u64> {
    let result = match join_type {
        BoundedConditionalJoinType::Inner | BoundedConditionalJoinType::Left => {
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size)
        },
        BoundedConditionalJoinType::LeftSemi | BoundedConditionalJoinType::LeftAnti => {
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

fn conditional_join_indices_with_output_size(
    join_type: BoundedConditionalJoinType,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    output_size: usize,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<BoundedConditionalJoinIndices> {
    match join_type {
        BoundedConditionalJoinType::Inner => {
            let indices = left
                .with_execution(overrides)
                .conditional_inner_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(BoundedConditionalJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() })
        },
        BoundedConditionalJoinType::Left => {
            let indices = left
                .with_execution(overrides)
                .conditional_left_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(BoundedConditionalJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() })
        },
        BoundedConditionalJoinType::LeftSemi => left
            .with_execution(overrides)
            .conditional_left_semi_join_indices_with_output_size(right, predicate, options, output_size)
            .map(BoundedConditionalJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        BoundedConditionalJoinType::LeftAnti => left
            .with_execution(overrides)
            .conditional_left_anti_join_indices_with_output_size(right, predicate, options, output_size)
            .map(BoundedConditionalJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
    }
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
