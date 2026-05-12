use super::{
    common::{
        BoundedJoinSideFrame, bounded_join_request, join_side_output_reservation_bytes,
        prepare_bounded_join_side_input_from_frame,
    },
    *,
};
use crate::{
    capability::{KeylessConditionalJoinKind, keyed_filtered_semi_anti_join_capable},
    exec::bounded::strict::CompiledJoinExecutionStrategy,
};

pub(in crate::exec::bounded) enum BoundedCompiledJoinBuildState {
    Hash(BoundedRootHashJoinBuildState),
    Filtered(BoundedCompiledFilteredJoinBuildState),
    Prepared(BoundedCompiledPreparedJoinBuildState),
}

pub(in crate::exec::bounded) enum BoundedCompiledJoinProbeState {
    Hash(BoundedRootHashJoinProbeState),
    Filtered(BoundedCompiledFilteredJoinProbeState),
    Prepared(BoundedCompiledPreparedJoinProbeState),
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinBuildState {
    strategy: CompiledJoinExecutionStrategy,
    side: JoinSide,
    input: BoundedJoinSideFrame,
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinProbeState {
    strategy: CompiledJoinExecutionStrategy,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinBuildState {
    left_keys: Vec<String>,
    left_columns: Vec<String>,
    build: BoundedJoinSideFrame,
    filtered_join: GpuFilteredJoinBuildHandle,
    hash_table_reservation: ActiveReservation,
    hash_table_bytes: u64,
    timer: Instant,
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinProbeState {
    build: BoundedCompiledFilteredJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_state: cudf::DataFrameFilteredJoinProbeState,
    probe_state_reservation: ActiveReservation,
    probe_state_bytes: u64,
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_build_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinBuildState> {
    match strategy {
        CompiledJoinExecutionStrategy::HashEqui => execute_bounded_root_hash_join_build_from_frame(
            context,
            node_id,
            node,
            input,
            input_reserved_bytes,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinBuildState::Hash),
        CompiledJoinExecutionStrategy::FilteredSemiAnti => execute_compiled_filtered_join_build_from_frame(
            context,
            node_id,
            node,
            input,
            input_reserved_bytes,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinBuildState::Filtered),
        CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::ProjectedCross
        | CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let side = node.spec.build_side;
            let columns = compiled_join_input_columns(context, node, strategy, side)?;
            let input = prepare_bounded_join_side_input_from_frame(
                input,
                input_reserved_bytes,
                &columns,
                side,
                node_id,
                node.spec.build_side,
                reservations,
                metrics,
            )?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState { strategy, side, input }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => execute_bounded_root_hash_join_probe_from_frame(
            context,
            node_id,
            node,
            build,
            input,
            input_reserved_bytes,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinProbeState::Hash),
        BoundedCompiledJoinBuildState::Filtered(build) => execute_compiled_filtered_join_probe_from_frame(
            context,
            node_id,
            node,
            build,
            input,
            input_reserved_bytes,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinProbeState::Filtered),
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let side = probe_side(build.side);
            let columns = compiled_join_input_columns(context, node, build.strategy, side)?;
            let probe = prepare_bounded_join_side_input_from_frame(
                input,
                input_reserved_bytes,
                &columns,
                side,
                node_id,
                node.spec.build_side,
                reservations,
                metrics,
            )?;
            let (left, right) = orient_prepared_inputs(build.side, build.input, side, probe)?;
            Ok(BoundedCompiledJoinProbeState::Prepared(BoundedCompiledPreparedJoinProbeState {
                strategy: build.strategy,
                left,
                right,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            execute_bounded_root_hash_join_materialize(context, node_id, node, probe, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::Filtered(probe) => {
            execute_compiled_filtered_join_materialize(context, node_id, node, probe, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            execute_compiled_prepared_join_materialize(context, node_id, node, probe, reservations, metrics)
        },
    }
}

fn execute_compiled_filtered_join_build_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled filtered join build requires a filtered semi/anti join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build = prepare_bounded_join_side_input_from_frame(
        input,
        input_reserved_bytes,
        &right_columns,
        JoinSide::Right,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let filtered_options =
        cudf::DataFrameFilteredJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = build
        .frame
        .with_execution(&overrides)
        .filtered_join_pre_build_reservation_size_bytes(&right_keys, filtered_options)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.filtered_hash_table_reservation",
                "failed to size compiled bounded filtered join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
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
        build.frame.with_execution(&overrides).build_filtered_join(&right_keys, filtered_options).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.build_filtered_join",
                "failed to build compiled bounded filtered join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", JoinSide::Right)
        })?;
    let filtered_join =
        GpuFilteredJoinBuildHandle::new(context, filtered_join, node.spec.null_equality).map_err(|error| {
            error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", JoinSide::Right)
        })?;
    let hash_table_bytes = filtered_join.device_allocated_size_bytes();
    metrics.record_filtered_join_build();
    metrics.record_filtered_join_hash_table_bytes(hash_table_bytes);
    Ok(BoundedCompiledFilteredJoinBuildState {
        left_keys,
        left_columns,
        build,
        filtered_join,
        hash_table_reservation,
        hash_table_bytes,
        timer,
    })
}

fn execute_compiled_filtered_join_probe_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe = prepare_bounded_join_side_input_from_frame(
        input,
        input_reserved_bytes,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi => {
            build.filtered_join.filtered_join().begin_left_semi_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::LeftAnti => {
            build.filtered_join.filtered_join().begin_left_anti_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.filtered_probe_state_bytes",
            "failed to observe compiled bounded filtered join probe-state memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", JoinSide::Right)
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
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes })
}

fn execute_compiled_filtered_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledFilteredJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes } =
        state;
    let overrides = execution_overrides(context);
    let gather_map_reservation_bytes =
        probe_state.probe_gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.filtered_gather_map_reservation",
                "failed to size compiled bounded filtered join gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
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
            "failed to observe compiled bounded filtered join gather-map memory",
            error,
        )
    })?;
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let output_reservation_bytes =
        join_side_output_reservation_bytes(&probe.frame, &indices, &left_projection, &overrides, JoinSide::Left)?;
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
        materialize_left_only_join_output_projection(&probe.frame, &indices, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += build.timer.elapsed();
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
            build_side: JoinSide::Right.to_string(),
            probe_side: JoinSide::Left.to_string(),
            build_rows: build.build.rows,
            build_bytes: build.build.bytes,
            probe_rows: probe.rows,
            probe_bytes: probe.bytes,
            hash_table_bytes: build.hash_table_bytes,
            filtered_probe_state_bytes: probe_state_bytes,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));
    drop(indices);
    reservations.release(gather_map_reservation);
    drop(build.filtered_join);
    reservations.release(build.hash_table_reservation);
    drop(probe);
    drop(build.build);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn execute_compiled_prepared_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledPreparedJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match state.strategy {
        CompiledJoinExecutionStrategy::KeylessConditional { join_type } => execute_compiled_keyless_join_materialize(
            context,
            node_id,
            node,
            join_type,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::ProjectedCross => execute_compiled_projected_cross_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::MixedKeyedResidual => execute_compiled_mixed_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::HashEqui | CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            Err(unsupported_bounded_shape(
                "compiled prepared join state received an incompatible strategy",
                "compiled_join_strategy_state_mismatch",
            ))
        },
    }
}

fn compiled_join_input_columns(
    context: &ExecutionContext,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    side: JoinSide,
) -> Result<Vec<String>> {
    match strategy {
        CompiledJoinExecutionStrategy::KeylessConditional { .. } => {
            let residual_columns = residual_input_columns(node)?;
            Ok(projected_input_columns(
                &[],
                &node.spec.output_projection,
                residual_columns.names(side).iter().map(String::as_str),
                side,
            ))
        },
        CompiledJoinExecutionStrategy::ProjectedCross => {
            Ok(projected_output_columns(&node.spec.output_projection, side))
        },
        CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let residual_columns = residual_input_columns(node)?;
            let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
            let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
            let keys = if side == JoinSide::Left { &left_keys } else { &right_keys };
            Ok(projected_input_columns(
                keys,
                &node.spec.output_projection,
                residual_columns.names(side).iter().map(String::as_str),
                side,
            ))
        },
        CompiledJoinExecutionStrategy::HashEqui | CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            let _ = context;
            Err(unsupported_bounded_shape(
                "compiled prepared join input columns require a prepared join strategy",
                "compiled_join_strategy_state_mismatch",
            ))
        },
    }
}

fn orient_prepared_inputs(
    build_side: JoinSide,
    build: BoundedJoinSideFrame,
    probe_side: JoinSide,
    probe: BoundedJoinSideFrame,
) -> Result<(BoundedJoinSideFrame, BoundedJoinSideFrame)> {
    match (build_side, probe_side) {
        (JoinSide::Left, JoinSide::Right) => Ok((build, probe)),
        (JoinSide::Right, JoinSide::Left) => Ok((probe, build)),
        _ => Err(unsupported_bounded_shape(
            "compiled prepared join inputs must contain exactly one left and one right side",
            "compiled_join_input_orientation_invalid",
        )),
    }
}

// The remaining prepared materializers intentionally mirror the root bounded
// join primitives, but start after build/probe input preparation.
fn execute_compiled_keyless_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    join_type: KeylessConditionalJoinKind,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let residual_columns = residual_input_columns(node)?;
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_predicate_join_options(&residual_columns);
    let timer = Instant::now();
    let (size_data_bytes, expected_output_rows, gather_map_reservation_bytes, work) = if join_type
        == KeylessConditionalJoinKind::Full
    {
        let size_data = left
            .frame
            .with_execution(&overrides)
            .conditional_full_join_size_data(&right.frame, &predicate, options)
            .map_err(|error| join_index_error(error, node))?;
        let expected_output_rows = size_data.output_size() as u64;
        let size_data_bytes = size_data.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.conditional_full_size_data_bytes",
                "failed to observe compiled conditional full join size-data memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
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
            size_data.gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.conditional_full_gather_map_reservation",
                    "failed to size compiled conditional full join gather maps",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
            })?;
        (
            size_data_bytes,
            expected_output_rows,
            gather_map_reservation_bytes,
            CompiledKeylessWork::Full { size_data: Box::new(size_data), size_data_reservation },
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
        (0, output_size as u64, gather_map_reservation_bytes, CompiledKeylessWork::Sized { output_size, options })
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
        CompiledKeylessWork::Full { size_data, size_data_reservation } => {
            let indices = (*size_data).materialize_indices_in(&overrides);
            reservations.release(size_data_reservation);
            let indices = indices.map_err(|error| join_index_error(error, node))?;
            CompiledJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() }
        },
        CompiledKeylessWork::Sized { output_size, options } => keyless_conditional_join_indices_with_output_size(
            join_type,
            &left.frame,
            &right.frame,
            &predicate,
            options,
            output_size,
            &overrides,
            node,
        )?,
    };
    materialize_compiled_indices_output(
        context,
        node_id,
        node,
        "bounded_keyless_conditional_join",
        0,
        true,
        JoinSide::Left,
        JoinSide::Right,
        left,
        right,
        indices,
        size_data_bytes,
        0,
        expected_output_rows,
        gather_map_reservation,
        timer,
        reservations,
        metrics,
    )
}

fn execute_compiled_mixed_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .with_execution(&overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Left => left
            .frame
            .with_execution(&overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Full => left
            .frame
            .with_execution(&overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .with_execution(&overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::LeftAnti => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
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
    materialize_compiled_indices_output(
        context,
        node_id,
        node,
        "bounded_mixed_keyed_residual_join",
        node.spec.keys.len(),
        true,
        node.spec.build_side,
        probe_side(node.spec.build_side),
        left,
        right,
        indices,
        size_data_bytes,
        0,
        expected_output_rows,
        gather_map_reservation,
        timer,
        reservations,
        metrics,
    )
}

fn execute_compiled_projected_cross_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let options = dataframe_cross_join_options(&node.spec.output_projection);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let plan = left.frame.projected_cross_join_plan(&right.frame, options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.projected_cross_plan",
            "failed to create compiled bounded projected cross join plan",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let expected_output_rows = plan.output_rows() as u64;
    let output_reservation_bytes =
        plan.reservation_size_bytes_in(&overrides).map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.projected_cross_output_reservation",
                "failed to size compiled bounded projected cross join output",
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
            "failed to materialize compiled bounded projected cross join output",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let output = reorder_projected_cross_join_output(output, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
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
            build_rows: side_metric(build_side, left.rows, right.rows),
            build_bytes: side_metric(build_side, left.bytes, right.bytes),
            probe_rows: side_metric(probe_side, left.rows, right.rows),
            probe_bytes: side_metric(probe_side, left.bytes, right.bytes),
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
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

enum CompiledKeylessWork<'a> {
    Full { size_data: Box<cudf::DataFrameConditionalFullJoinSizeData<'a>>, size_data_reservation: ActiveReservation },
    Sized { output_size: usize, options: cudf::DataFramePredicateJoinOptions },
}

enum CompiledMixedSizeData<'a> {
    Pair(cudf::DataFrameMixedJoinSizeData<'a>),
    LeftOnly(cudf::DataFrameMixedSingleSideJoinSizeData<'a>),
}

impl CompiledMixedSizeData<'_> {
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
                "failed to observe compiled bounded mixed join size-data memory",
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
                "failed to size compiled bounded mixed join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })
    }

    fn materialize_indices_in(
        self,
        overrides: &ExecutionOverrides<'_>,
        node: &JoinNode,
    ) -> Result<CompiledJoinIndices> {
        match self {
            Self::Pair(size_data) => size_data
                .materialize_indices_in(overrides)
                .map(|indices| CompiledJoinIndices::Pair {
                    left: indices.left().clone(),
                    right: indices.right().clone(),
                })
                .map_err(|error| join_index_error(error, node)),
            Self::LeftOnly(size_data) => size_data
                .materialize_indices_in(overrides)
                .map(CompiledJoinIndices::LeftOnly)
                .map_err(|error| join_index_error(error, node)),
        }
    }
}

enum CompiledJoinIndices {
    Pair { left: cudf::DataFrameJoinIndices, right: cudf::DataFrameJoinIndices },
    LeftOnly(cudf::DataFrameJoinIndices),
}

impl CompiledJoinIndices {
    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        match self {
            Self::Pair { left, right } => {
                let left_bytes = left.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join left gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join right gather-map memory",
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
                        "failed to observe compiled bounded join gather-map memory",
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

#[allow(clippy::too_many_arguments)]
fn materialize_compiled_indices_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    execution_strategy: &'static str,
    key_count: usize,
    residual_predicate: bool,
    build_side: JoinSide,
    probe_side: JoinSide,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    indices: CompiledJoinIndices,
    mixed_size_data_bytes: u64,
    filtered_probe_state_bytes: u64,
    expected_output_rows: u64,
    gather_map_reservation: ActiveReservation,
    timer: Instant,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let overrides = execution_overrides(context);
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
            execution_strategy: execution_strategy.to_string(),
            key_count,
            residual_predicate,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left.rows, right.rows),
            build_bytes: side_metric(build_side, left.bytes, right.bytes),
            probe_rows: side_metric(probe_side, left.rows, right.rows),
            probe_bytes: side_metric(probe_side, left.bytes, right.bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes,
            mixed_size_data_bytes,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));
    drop(indices);
    reservations.release(gather_map_reservation);
    drop(left);
    drop(right);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn keyless_conditional_join_output_size(
    join_type: KeylessConditionalJoinKind,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<usize> {
    let result = match join_type {
        KeylessConditionalJoinKind::Inner => {
            left.with_execution(overrides).conditional_inner_join_size(right, predicate, options)
        },
        KeylessConditionalJoinKind::Left => {
            left.with_execution(overrides).conditional_left_join_size(right, predicate, options)
        },
        KeylessConditionalJoinKind::Full => unreachable!("conditional full joins use retained size data"),
        KeylessConditionalJoinKind::LeftSemi => {
            left.with_execution(overrides).conditional_left_semi_join_size(right, predicate, options)
        },
        KeylessConditionalJoinKind::LeftAnti => {
            left.with_execution(overrides).conditional_left_anti_join_size(right, predicate, options)
        },
    };
    result.map_err(|error| join_index_error(error, node))
}

fn keyless_conditional_join_gather_map_reservation_size_bytes(
    join_type: KeylessConditionalJoinKind,
    node: &JoinNode,
    output_size: usize,
) -> Result<u64> {
    let result = match join_type {
        KeylessConditionalJoinKind::Inner | KeylessConditionalJoinKind::Left => {
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size)
        },
        KeylessConditionalJoinKind::Full => unreachable!("conditional full joins use retained size data"),
        KeylessConditionalJoinKind::LeftSemi | KeylessConditionalJoinKind::LeftAnti => {
            cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(output_size)
        },
    };
    result.map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_reservation",
            "failed to size compiled conditional join gather maps",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })
}

#[allow(clippy::too_many_arguments)]
fn keyless_conditional_join_indices_with_output_size(
    join_type: KeylessConditionalJoinKind,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    output_size: usize,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<CompiledJoinIndices> {
    match join_type {
        KeylessConditionalJoinKind::Inner => {
            let indices = left
                .with_execution(overrides)
                .conditional_inner_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(CompiledJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() })
        },
        KeylessConditionalJoinKind::Left => {
            let indices = left
                .with_execution(overrides)
                .conditional_left_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(CompiledJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() })
        },
        KeylessConditionalJoinKind::Full => unreachable!("conditional full joins use retained size data"),
        KeylessConditionalJoinKind::LeftSemi => left
            .with_execution(overrides)
            .conditional_left_semi_join_indices_with_output_size(right, predicate, options, output_size)
            .map(CompiledJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        KeylessConditionalJoinKind::LeftAnti => left
            .with_execution(overrides)
            .conditional_left_anti_join_indices_with_output_size(right, predicate, options, output_size)
            .map(CompiledJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
    }
}
