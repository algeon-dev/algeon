use super::*;
use crate::exec::bounded::join::common::{
    bounded_join_request, join_side_output_reservation_bytes, prepare_bounded_join_side_input,
};

pub(super) fn execute_bounded_root_mixed_join(
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
