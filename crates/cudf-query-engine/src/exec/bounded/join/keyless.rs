use super::*;
use crate::{
    bounded_support::BoundedKeylessConditionalJoinType,
    exec::bounded::join::common::{
        bounded_join_request, join_side_output_reservation_bytes, prepare_bounded_join_side_input,
    },
};

pub(super) fn execute_bounded_root_keyless_conditional_join(
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
