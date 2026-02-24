use arrow::{array::new_empty_array, record_batch::RecordBatchOptions};

use super::{compiled::ResidentFrameStreamPayload, *};
use crate::capability::AggregateExecutionStage;

mod final_merge;
mod plan;
pub mod sizing;

pub(super) use final_merge::{
    execute_compiled_bounded_final_aggregate_merge, execute_final_aggregate_merge, reduce_merge_fan_in,
};
pub(super) use plan::{BoundedPlan, BoundedUnaryStage};
pub use sizing::{PartialAggregateShape, partial_aggregate_upper_bound_bytes};
use sizing::{partial_aggregate_upper_bound_violation_error, partial_aggregate_work_reservation_upper_bound_bytes};

#[derive(Clone, Debug)]
enum SourceAdjacentAggregatePartitioner {
    RowBalanced,
    GroupKey { key_columns: Vec<String> },
}

impl SourceAdjacentAggregatePartitioner {
    fn label(&self) -> &'static str {
        match self {
            Self::RowBalanced => "row_balanced",
            Self::GroupKey { .. } => "group_key",
        }
    }

    fn key_count(&self) -> Option<u64> {
        match self {
            Self::RowBalanced => None,
            Self::GroupKey { key_columns } => Some(key_columns.len() as u64),
        }
    }

    fn retain_empty_partition(&self, partition: usize) -> bool {
        match self {
            Self::RowBalanced => true,
            Self::GroupKey { .. } => partition == 0,
        }
    }
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if should_partition_compiled_grouped_aggregate(context, aggregate, input_reserved_bytes) {
        return execute_compiled_bounded_partitioned_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input,
            input_reserved_bytes,
            reservations,
            metrics,
        );
    }

    execute_bounded_generic_unary_from_frame(
        context,
        node_id,
        "compiled_aggregate",
        "compiled_aggregate_output",
        input,
        input_reserved_bytes,
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            execute_aggregate_with_budget(
                context,
                aggregate,
                input_schema,
                input,
                output_schema,
                token.expect("budget token is created for compiled bounded aggregate stages"),
                metrics,
            )
        },
    )
}

fn execute_compiled_bounded_aggregate_from_empty_input(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input = empty_resident_frame_for_schema(
        context,
        node_id,
        input_schema,
        ReservationKind::OperatorOutput,
        "compiled_aggregate_empty_input",
        "compiled_aggregate_empty_input",
        reservations,
        metrics,
    )?;
    let input_reserved_bytes = input.reservation.bytes();
    let input_reservation = input.reservation;
    let result = execute_compiled_bounded_aggregate_from_frame(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        input_reserved_bytes,
        reservations,
        metrics,
    );
    reservations.release(input_reservation, metrics);
    result
}

fn empty_resident_frame_for_schema(
    context: &ExecutionContext,
    node_id: NodeId,
    schema: &arrow::datatypes::SchemaRef,
    reservation_kind: ReservationKind,
    memory_boundary: &'static str,
    stage: &'static str,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let arrays = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(schema),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.bounded.aggregate.empty_input",
            "failed to construct zero-row aggregate input batch",
        )
        .with_fact("node_id", node_id)
        .with_source(error)
    })?;
    let frame = DataFrame::from_record_batch(context.runtime(), &batch).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.empty_input",
            "failed to upload zero-row aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
    })?;
    let frame = GpuDataFrame::new(frame);
    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
            .with_memory_boundary(memory_boundary)
            .with_node_id(node_id)
            .with_stage(stage),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}
fn execute_compiled_bounded_partitioned_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let mut lowered_keys = lower_aggregate_spec_keys(&aggregate.spec)?;
    if aggregate.spec.grouping_sets.is_some() {
        lowered_keys.key_columns.truncate(aggregate.spec.group_exprs.len());
    }
    let partition_count = partitioned_aggregate_partition_count(context, input_reserved_bytes);
    let mut partition_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_hash_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_hash_partition"),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input);
            return Err(error);
        },
    };
    let frame = input.into_frame();
    let partitioned = {
        let token = match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&partition_reservation),
            "compiled_aggregate_hash_partition",
            "cudf_query_engine.bounded.aggregate.compiled_partitioned_hash.budget_token",
            "failed to create bounded native compiled partitioned aggregate budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(partition_reservation, metrics);
                drop(frame);
                return Err(error);
            },
        };

        frame
            .with_execution(&token.execution_overrides())
            .hash_partition_by(lowered_keys.key_columns.iter().map(String::as_str), partition_count)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.aggregate.hash_partition",
                        "failed to hash-partition compiled bounded aggregate input",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("key_count", lowered_keys.key_columns.len())
                    .with_fact("partition_count", partition_count),
                    &token,
                )
            })
    };
    drop(frame);
    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_gpu, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_hash_partition");
        if let Err(error) = reservations.try_grow(&mut partition_reservation, grow_request, metrics) {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut partition_reservation, partitioned_bytes, metrics);
    }

    let mut partials = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        let partition_frame = match slice_compiled_aggregate_partition(
            context,
            node_id,
            &partitioned_frame,
            &offsets,
            partition,
            reservations,
            metrics,
        ) {
            Ok(frame) => frame,
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        };
        match execute_compiled_bounded_partial_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partition_frame,
            reservations,
            metrics,
        ) {
            Ok(partial) => partials.push(partial),
            Err(error) => {
                release_partials_reverse(reservations, partials, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        }
    }
    drop(partitioned_frame);
    reservations.release(partition_reservation, metrics);

    execute_compiled_bounded_final_aggregate_merge(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        partials,
        reservations,
        metrics,
    )
}

fn slice_compiled_aggregate_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    partition: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "partitioned aggregate input is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    let overrides = execution_overrides(context);
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.hash_partition_slice",
            "failed to slice compiled bounded hash-partitioned aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_partition"),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame: gpu, reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_partitioned_source_adjacent_aggregate_from_raw_chunks(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let estimated_input_bytes = chunks
        .iter()
        .filter_map(|chunk| chunk.estimated_chunk_bytes.or(chunk.estimated_source_read_bytes))
        .sum::<u64>()
        .max(1);
    let partition_count = partitioned_aggregate_partition_count(context, estimated_input_bytes)
        .max(source_adjacent_partitioned_aggregate_min_partition_count(chunks.len()));
    let partitioner = source_adjacent_aggregate_partitioner(context, bounded, estimated_input_bytes, chunks.len());
    let mut partition_inputs = (0..partition_count).map(|_| Vec::new()).collect::<Vec<Vec<CommittedPartial>>>();

    for chunk in chunks {
        let raw = match execute_bounded_raw_source_chunk(context, bounded, chunk, reservations, metrics) {
            Ok(raw) => raw,
            Err(error) => {
                release_nested_partials_reverse(reservations, partition_inputs, metrics);
                return Err(error);
            },
        };
        let partitions = match partition_source_adjacent_raw_chunk(
            context,
            bounded.aggregate_id,
            chunk.chunk_index,
            raw,
            partition_count,
            &partitioner,
            reservations,
            metrics,
        ) {
            Ok(partitions) => partitions,
            Err(error) => {
                release_nested_partials_reverse(reservations, partition_inputs, metrics);
                return Err(error);
            },
        };
        for (partition_index, partition) in partitions.into_iter().enumerate() {
            if let Some(partition) = partition {
                partition_inputs[partition_index].push(partition);
            }
        }
    }

    let non_empty_partition_count =
        partition_inputs.iter().filter(|inputs| inputs.iter().any(|input| input.frame.num_rows() > 0)).count();
    let max_partition_fan_in = partition_inputs.iter().map(Vec::len).max().unwrap_or(0);
    let input_evidence = source_adjacent_partitioned_aggregate_input_evidence(
        chunks,
        &partitioner,
        partition_count,
        non_empty_partition_count,
        max_partition_fan_in,
    );
    let mut aggregate_outputs = Vec::with_capacity(partition_count);
    for (partition_index, inputs) in partition_inputs.into_iter().enumerate() {
        if inputs.is_empty() {
            continue;
        }
        let partition_input = match concatenate_partition_inputs(
            context,
            bounded.aggregate_id,
            partition_index,
            inputs,
            reservations,
            metrics,
        ) {
            Ok(input) => input,
            Err(error) => {
                release_partials_reverse(reservations, aggregate_outputs, metrics);
                return Err(error);
            },
        };
        match execute_source_adjacent_partition_aggregate(
            context,
            bounded,
            partition_index,
            partition_input,
            input_evidence,
            reservations,
            metrics,
        ) {
            Ok(output) => aggregate_outputs.push(output),
            Err(error) => {
                release_partials_reverse(reservations, aggregate_outputs, metrics);
                return Err(error);
            },
        }
    }
    for chunk in chunks {
        metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    }

    execute_final_aggregate_merge(context, bounded, aggregate_outputs, reservations, metrics)
}

fn source_adjacent_partitioned_aggregate_input_evidence(
    chunks: &[SourceChunk],
    partitioner: &SourceAdjacentAggregatePartitioner,
    partition_count: usize,
    non_empty_partition_count: usize,
    max_partition_fan_in: usize,
) -> PartialAggregateInputEvidence {
    PartialAggregateInputEvidence {
        source_adjacent_partial: true,
        source_chunk_row_group_count: chunks.iter().map(|chunk| chunk.row_group_count() as u64).sum(),
        source_chunk_reads_all_row_groups: chunks.iter().all(|chunk| chunk.reads_all_row_groups),
        source_chunk_known_row_count: sum_optional_u64(chunks.iter().map(|chunk| chunk.known_row_count)),
        source_chunk_estimated_source_read_bytes: sum_optional_u64(
            chunks.iter().map(|chunk| chunk.estimated_source_read_bytes),
        ),
        source_chunk_size_estimate_based: chunks.iter().any(|chunk| chunk.size_estimate_based),
        source_adjacent_partition_strategy: partitioner.label(),
        source_adjacent_partition_count: Some(partition_count as u64),
        source_adjacent_non_empty_partition_count: Some(non_empty_partition_count as u64),
        source_adjacent_partition_key_count: partitioner.key_count(),
        source_adjacent_max_partition_fan_in: Some(max_partition_fan_in as u64),
    }
}

fn source_adjacent_aggregate_partitioner(
    _context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    _estimated_input_bytes: u64,
    _chunk_count: usize,
) -> SourceAdjacentAggregatePartitioner {
    let Ok(lowered_keys) = lower_aggregate_spec_keys(&bounded.aggregate.0.spec) else {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    };
    if lowered_keys.key_columns.is_empty() {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    let mut has_string_key = false;
    for key_column in &lowered_keys.key_columns {
        let Ok(field) = bounded.aggregate_input_schema.field_with_name(key_column) else {
            return SourceAdjacentAggregatePartitioner::RowBalanced;
        };
        has_string_key |=
            matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View);
    }
    if !has_string_key {
        return SourceAdjacentAggregatePartitioner::RowBalanced;
    }
    SourceAdjacentAggregatePartitioner::GroupKey { key_columns: lowered_keys.key_columns }
}

fn sum_optional_u64(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    values.into_iter().try_fold(0u64, |total, value| value.map(|value| total.saturating_add(value)))
}

fn partition_source_adjacent_raw_chunk(
    context: &ExecutionContext,
    node_id: NodeId,
    chunk_index: usize,
    raw: CommittedPartial,
    partition_count: usize,
    partitioner: &SourceAdjacentAggregatePartitioner,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<Option<CommittedPartial>>> {
    let input_reserved_bytes = raw.reservation.bytes();
    let hash_task_id = source_adjacent_aggregate_task_id(
        node_id,
        "source_adjacent_aggregate_hash_partition",
        format!("chunk{chunk_index}"),
    );
    let mut partition_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_aggregate_hash_partition")
            .with_node_id(node_id)
            .with_stage("source_adjacent_aggregate_hash_partition")
            .with_compiled_task_id(hash_task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(raw.frame);
            reservations.release(raw.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&partition_reservation),
        "source_adjacent_aggregate_hash_partition",
        "cudf_query_engine.bounded.aggregate.source_adjacent_partitioned_hash.budget_token",
        "failed to create source-adjacent partitioned aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            drop(raw.frame);
            reservations.release(raw.reservation, metrics);
            return Err(error);
        },
    };

    let raw_frame = raw.frame.into_frame();
    let original_columns = matches!(partitioner, SourceAdjacentAggregatePartitioner::RowBalanced)
        .then(|| raw_frame.column_names().into_iter().map(str::to_string).collect::<Vec<_>>());
    let partitioned = match partitioner {
        SourceAdjacentAggregatePartitioner::RowBalanced => {
            let partition_column = unique_source_adjacent_partition_column_name(&raw_frame);
            let row_count = raw_frame.num_rows();
            let with_partition_column = raw_frame
                .with_execution(&token.execution_overrides())
                .with_position_column(&partition_column)
                .map_err(|error| {
                    attach_budget_token_failure_facts(
                        QueryError::from_cudf(
                            "cudf_query_engine.bounded.aggregate.source_adjacent_partition_column",
                            "failed to add row-balanced source-adjacent aggregate partition column",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("partition_count", partition_count)
                        .with_fact("input_rows", row_count),
                        &token,
                    )
                });
            let with_partition_column = match with_partition_column {
                Ok(frame) => frame,
                Err(error) => {
                    drop(raw_frame);
                    reservations.release(raw.reservation, metrics);
                    reservations.release(partition_reservation, metrics);
                    return Err(error);
                },
            };
            let partitioned = with_partition_column
                .with_execution(&token.execution_overrides())
                .hash_partition_by([partition_column.as_str()], partition_count)
                .map_err(|error| {
                    attach_budget_token_failure_facts(
                        QueryError::from_cudf(
                            "cudf_query_engine.bounded.aggregate.source_adjacent_hash_partition",
                            "failed to hash-partition source-adjacent aggregate input",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("partition_key", partition_column.as_str())
                        .with_fact("partition_count", partition_count),
                        &token,
                    )
                });
            drop(with_partition_column);
            partitioned
        },
        SourceAdjacentAggregatePartitioner::GroupKey { key_columns } => raw_frame
            .with_execution(&token.execution_overrides())
            .hash_partition_by(key_columns.iter().map(String::as_str), partition_count)
            .map_err(|error| {
                attach_budget_token_failure_facts(
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.aggregate.source_adjacent_group_key_hash_partition",
                        "failed to hash-partition source-adjacent aggregate input by group keys",
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("partition_keys", key_columns.join(","))
                    .with_fact("partition_count", partition_count),
                    &token,
                )
            }),
    };
    drop(raw_frame);
    reservations.release(raw.reservation, metrics);

    let partitioned = match partitioned {
        Ok(partitioned) => partitioned,
        Err(error) => {
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    let (partitioned_frame, offsets) = partitioned.into_parts();
    let partitioned_gpu = GpuDataFrame::new(partitioned_frame.clone());
    let partitioned_bytes = match observe_device_memory_bytes(&partitioned_gpu, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        },
    };
    if partitioned_bytes > partition_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            partitioned_bytes.saturating_sub(partition_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("source_adjacent_aggregate_hash_partition")
        .with_node_id(node_id)
        .with_stage("source_adjacent_aggregate_hash_partition")
        .with_compiled_task_id(hash_task_id.clone());
        if let Err(error) = reservations.try_grow(&mut partition_reservation, grow_request, metrics) {
            drop(partitioned_frame);
            reservations.release(partition_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut partition_reservation, partitioned_bytes, metrics);
    }

    let mut partitions = Vec::with_capacity(partition_count);
    for partition in 0..partition_count {
        match slice_source_adjacent_aggregate_partition(
            context,
            node_id,
            &partitioned_frame,
            &offsets,
            original_columns.as_deref(),
            chunk_index,
            partition,
            partitioner.retain_empty_partition(partition),
            reservations,
            metrics,
        ) {
            Ok(frame) => partitions.push(frame),
            Err(error) => {
                release_optional_partials_reverse(reservations, partitions, metrics);
                drop(partitioned_frame);
                reservations.release(partition_reservation, metrics);
                return Err(error);
            },
        }
    }
    drop(partitioned_frame);
    reservations.release(partition_reservation, metrics);
    Ok(partitions)
}

fn slice_source_adjacent_aggregate_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    frame: &cudf::DataFrame,
    offsets: &[usize],
    output_columns: Option<&[String]>,
    chunk_index: usize,
    partition: usize,
    retain_empty: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<CommittedPartial>> {
    let start = *offsets.get(partition).ok_or_else(|| {
        unsupported_bounded_shape(
            "source-adjacent aggregate partition is missing a partition start offset",
            "partition_offset_missing",
        )
    })?;
    let end = *offsets.get(partition + 1).ok_or_else(|| {
        unsupported_bounded_shape(
            "source-adjacent aggregate partition is missing a partition end offset",
            "partition_offset_missing",
        )
    })?;
    let len = end.saturating_sub(start);
    if len == 0 && !retain_empty {
        return Ok(None);
    }
    let overrides = execution_overrides(context);
    let sliced = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.aggregate.source_adjacent_hash_partition_slice",
            "failed to slice source-adjacent hash-partitioned aggregate input",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition)
        .with_fact("partition_rows", len)
    })?;
    let sliced = if let Some(output_columns) = output_columns {
        let output_refs = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
        sliced.select(output_refs).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.aggregate.source_adjacent_hash_partition_select",
                "failed to remove source-adjacent hash-partition helper column",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("partition", partition)
            .with_fact("partition_rows", len)
        })?
    } else {
        sliced
    };
    let gpu = GpuDataFrame::new(sliced);
    let bytes = observe_device_memory_bytes(&gpu, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_aggregate_partition")
            .with_node_id(node_id)
            .with_stage("source_adjacent_aggregate_partition")
            .with_compiled_task_id(source_adjacent_aggregate_task_id(
                node_id,
                "source_adjacent_aggregate_partition",
                format!("chunk{chunk_index}.partition{partition}"),
            )),
        metrics,
    )?;
    Ok(Some(CommittedPartial { frame: gpu, reservation }))
}

fn concatenate_partition_inputs(
    context: &ExecutionContext,
    node_id: NodeId,
    partition_index: usize,
    mut inputs: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if inputs.len() == 1 {
        return Ok(inputs.pop().expect("single input checked"));
    }
    concatenate_committed_frames(
        context,
        node_id,
        partition_index,
        inputs,
        ReservationKind::OperatorOutput,
        "source_adjacent_aggregate_partition_concat",
        "source_adjacent_aggregate_partition_concat_output",
        "cudf_query_engine.bounded.aggregate.source_adjacent_partition_concat.budget_token",
        "failed to create source-adjacent aggregate partition concat budget token",
        "cudf_query_engine.bounded.aggregate.source_adjacent_partition_concat",
        "failed to concatenate source-adjacent aggregate partition inputs",
        reservations,
        metrics,
    )
}

fn execute_source_adjacent_partition_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partition_index: usize,
    input: CommittedPartial,
    input_evidence: PartialAggregateInputEvidence,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let input_reserved_bytes = input.reservation.bytes();
    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let task_id = source_adjacent_aggregate_task_id(
        bounded.aggregate_id,
        "source_adjacent_partitioned_partial_aggregate",
        format!("partition{partition_index}"),
    );
    let mut work_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::PartialState, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_partitioned_partial_aggregate")
            .with_node_id(bounded.aggregate_id)
            .with_stage("source_adjacent_partitioned_partial_aggregate")
            .with_attempt(1, 1)
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input_frame);
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        "source_adjacent_partitioned_partial_aggregate",
        "cudf_query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            drop(input_frame);
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", bounded.aggregate_id));
        },
    };
    let chunk_operator_timer = Instant::now();
    metrics.set_current_task(task_id.clone());
    let result = execute_partial_aggregate_with_budget(
        context,
        Some(bounded.aggregate_id),
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        input_frame,
        &token,
        Some(input_evidence),
        metrics,
    );
    metrics.clear_current_task();
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error
                .with_fact_if_absent("node_id", bounded.aggregate_id)
                .with_fact_if_absent("partition", partition_index));
        },
    };
    reservations.release(input_reservation, metrics);
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error
                .with_fact_if_absent("node_id", bounded.aggregate_id)
                .with_fact_if_absent("partition", partition_index));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, output_bytes, reservations.budget())
            .with_memory_boundary("source_adjacent_partitioned_partial_aggregate_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("source_adjacent_partitioned_partial_aggregate")
            .with_attempt(1, 1)
            .with_compiled_task_id(task_id);
    if output_bytes > work_reservation.bytes() {
        let delta = output_bytes.saturating_sub(work_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut work_reservation, output_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(frame);
            reservations.release(work_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(
            &mut work_reservation,
            output_bytes,
            output_request.clone(),
            metrics,
        );
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    let output = CommittedPartial { frame, reservation: output_reservation };
    metrics.record_partial_aggregate(input_rows, output.frame.num_rows() as u64, output.reservation.bytes());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn concatenate_committed_frames(
    context: &ExecutionContext,
    node_id: NodeId,
    partition_index: usize,
    mut inputs: Vec<CommittedPartial>,
    reservation_kind: ReservationKind,
    stage: &'static str,
    output_boundary: &'static str,
    token_operation: &'static str,
    token_message: &'static str,
    concat_operation: &'static str,
    concat_message: &'static str,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            concat_operation,
            "source-adjacent partitioned aggregate concat requires at least one input frame",
        )
        .with_fact("node_id", node_id)
        .with_fact("partition", partition_index));
    }
    let requested_bytes = inputs.iter().map(|input| input.reservation.bytes()).sum::<u64>();
    let task_id = source_adjacent_aggregate_task_id(node_id, stage, format!("partition{partition_index}"));
    let mut concat_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(reservation_kind, requested_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_compiled_task_id(task_id.clone()),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&concat_reservation),
        stage,
        token_operation,
        token_message,
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            release_partials_reverse(reservations, inputs, metrics);
            return Err(error);
        },
    };

    let first = inputs.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let other_frames = inputs.iter().map(|input| input.frame.frame().clone()).collect::<Vec<_>>();
        let other_refs = other_frames.iter().collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(concat_operation, concat_message, error)
                    .with_fact("node_id", node_id)
                    .with_fact("partition", partition_index)
                    .with_fact("input_count", inputs.len() + 1),
                &token,
            )
        })
    };
    drop(first_frame);
    reservations.release(first_reservation, metrics);
    release_partials_reverse(reservations, inputs, metrics);

    let concatenated = match concatenated {
        Ok(frame) => GpuDataFrame::new(frame),
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            return Err(error);
        },
    };
    let output_bytes = match observe_device_memory_bytes(&concatenated, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(concat_reservation, metrics);
            return Err(error);
        },
    };
    let output_request = MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_compiled_task_id(task_id.clone());
    if output_bytes > concat_reservation.bytes() {
        let delta = output_bytes.saturating_sub(concat_reservation.bytes());
        if let Err(error) =
            reservations.try_grow(&mut concat_reservation, output_request.clone().with_requested_bytes(delta), metrics)
        {
            drop(concatenated);
            reservations.release(concat_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed_with_request(
            &mut concat_reservation,
            output_bytes,
            output_request.clone(),
            metrics,
        );
    }
    let output_reservation = reservations.transfer(concat_reservation, output_request, metrics);
    Ok(CommittedPartial { frame: concatenated, reservation: output_reservation })
}

fn release_nested_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partitions: Vec<Vec<CommittedPartial>>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partition) = partitions.pop() {
        release_partials_reverse(reservations, partition, metrics);
    }
}

fn release_optional_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partials: Vec<Option<CommittedPartial>>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partial) = partials.pop() {
        if let Some(partial) = partial {
            drop(partial.frame);
            reservations.release(partial.reservation, metrics);
        }
    }
}

fn unique_source_adjacent_partition_column_name(frame: &cudf::DataFrame) -> String {
    let base = "__cudf_source_adjacent_partition_row_id";
    if !frame.contains(base) {
        return base.to_string();
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}_{suffix}");
        if !frame.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

fn source_adjacent_aggregate_task_id(node_id: NodeId, stage: &'static str, suffix: impl std::fmt::Display) -> String {
    format!("legacy_bounded.aggregate.node{node_id}.{stage}.{suffix}")
}

fn source_adjacent_partitioned_aggregate_min_partition_count(chunk_count: usize) -> usize {
    chunk_count.clamp(1, 32)
}

fn should_partition_compiled_grouped_aggregate(
    context: &ExecutionContext,
    aggregate: &AggregateNode,
    input_reserved_bytes: u64,
) -> bool {
    !aggregate_spec_has_stddev_samp(&aggregate.spec)
        && !aggregate.spec.group_exprs.is_empty()
        && partitioned_aggregate_partition_count(context, input_reserved_bytes) > 1
}

fn partitioned_aggregate_partition_count(context: &ExecutionContext, input_bytes: u64) -> usize {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        return 1;
    };
    let target = partitioned_aggregate_target_partition_bytes(device_budget_bytes);
    let partitions = input_bytes.max(1).saturating_add(target.saturating_sub(1)) / target.max(1);
    usize::try_from(partitions.clamp(1, 1024)).unwrap_or(1024)
}

fn partitioned_aggregate_target_partition_bytes(device_budget_bytes: u64) -> u64 {
    const MIN_TARGET_BYTES: u64 = 128 * 1024 * 1024;
    const MAX_TARGET_BYTES: u64 = 1024 * 1024 * 1024;
    (device_budget_bytes / 32).clamp(MIN_TARGET_BYTES, MAX_TARGET_BYTES)
}

pub(in crate::exec::bounded) fn execute_compiled_bounded_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    mut inputs: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let result = (|| -> Result<ResidentGpuFrame> {
        if inputs.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                reservations,
                metrics,
            );
        }
        if aggregate_spec_has_stddev_samp(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(reservations, metrics)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    context,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                    reservations,
                    metrics,
                );
            }
            let input = execute_bounded_generic_union(context, node_id, frames, reservations, metrics)?;
            return execute_bounded_generic_unary(
                context,
                node_id,
                "compiled_stddev_aggregate",
                "compiled_stddev_aggregate_output",
                input,
                reservations,
                metrics,
                true,
                |context, input, token, metrics| {
                    execute_aggregate_with_budget(
                        context,
                        aggregate,
                        input_schema,
                        input,
                        output_schema,
                        token.expect("budget token is created for compiled bounded stddev aggregate stages"),
                        metrics,
                    )
                },
            );
        }
        if aggregate_spec_has_distinct(&aggregate.spec) {
            let frames = inputs.collect_remaining_frames_owned(reservations, metrics)?;
            if frames.is_empty() {
                return execute_compiled_bounded_aggregate_from_empty_input(
                    context,
                    node_id,
                    aggregate,
                    input_schema,
                    output_schema,
                    reservations,
                    metrics,
                );
            }
            return execute_compiled_bounded_distinct_aggregate_from_stream(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                frames,
                reservations,
                metrics,
            );
        }

        let mut partials = Vec::with_capacity(inputs.len());
        while let Some(input) = inputs.next_yielded_frame(reservations, metrics) {
            let mut input = match input {
                Ok(input) => input,
                Err(error) => {
                    release_partials_reverse(reservations, partials, metrics);
                    return Err(error);
                },
            };
            let input = input.take_for_release();
            match execute_compiled_bounded_partial_aggregate_from_frame(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                input,
                reservations,
                metrics,
            ) {
                Ok(partial) => partials.push(partial),
                Err(error) => {
                    release_partials_reverse(reservations, partials, metrics);
                    inputs.release_remaining_frames(reservations, metrics);
                    return Err(error);
                },
            }
        }

        if partials.is_empty() {
            return execute_compiled_bounded_aggregate_from_empty_input(
                context,
                node_id,
                aggregate,
                input_schema,
                output_schema,
                reservations,
                metrics,
            );
        }

        execute_compiled_bounded_final_aggregate_merge(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            partials,
            reservations,
            metrics,
        )
    })();
    inputs.release_producer_reservations(reservations, metrics);
    result
}

fn execute_compiled_bounded_distinct_aggregate_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    inputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let mut frames = inputs;
    let input_reserved_bytes = frames.iter().map(|frame| frame.reservation.bytes()).sum::<u64>();
    if frames.len() == 1 {
        let input = frames.pop().expect("single frame checked");
        return execute_compiled_bounded_aggregate_from_frame(
            context,
            node_id,
            aggregate,
            input_schema,
            output_schema,
            input.frame,
            input.reservation.bytes(),
            reservations,
            metrics,
        );
    }

    let request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_distinct_stream_concat")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate_distinct_stream_concat");
    let mut concat_reservation = match reservations.try_reserve_work(request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(frames);
            return Err(error);
        },
    };
    let first = frames.remove(0);
    let first_reservation = first.reservation;
    let first_frame = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&concat_reservation),
            "compiled_aggregate_distinct_stream_concat",
            "cudf_query_engine.bounded.aggregate.distinct_stream_concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(concat_reservation, metrics);
                drop(first_frame);
                return Err(error);
            },
        };
        let other_frame_refs = frames.iter().map(|frame| frame.frame.frame()).collect::<Vec<_>>();
        first_frame.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.aggregate.distinct_stream_concat",
                    "failed to concatenate distinct aggregate stream inputs",
                    error,
                ),
                &token,
            )
        })
    };
    for frame in frames {
        reservations.release(frame.reservation, metrics);
    }
    reservations.release(first_reservation, metrics);
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    let concatenated_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concatenated_bytes > concat_reservation.bytes() {
        let grow_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            concatenated_bytes.saturating_sub(concat_reservation.bytes()),
            reservations.budget(),
        )
        .with_memory_boundary("compiled_aggregate_distinct_stream_concat")
        .with_node_id(node_id)
        .with_stage("compiled_aggregate_distinct_stream_concat");
        if let Err(error) = reservations.try_grow(&mut concat_reservation, grow_request, metrics) {
            drop(concatenated);
            reservations.release(concat_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut concat_reservation, concatenated_bytes, metrics);
    }

    execute_bounded_generic_unary(
        context,
        node_id,
        "compiled_aggregate_distinct",
        "compiled_aggregate_distinct_output",
        ResidentGpuFrame { frame: concatenated, reservation: concat_reservation },
        reservations,
        metrics,
        true,
        |context, input, token, metrics| {
            execute_aggregate_with_budget(
                context,
                aggregate,
                input_schema,
                input,
                output_schema,
                token.expect("budget token is created for compiled bounded distinct aggregate stages"),
                metrics,
            )
        },
    )
}

fn execute_compiled_bounded_partial_aggregate_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    aggregate: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    output_schema: &arrow::datatypes::SchemaRef,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let input_rows = input.frame.num_rows() as u64;
    let upper_bound_bytes = match partial_aggregate_work_reservation_upper_bound_bytes(
        context,
        node_id,
        aggregate,
        input_schema,
        output_schema,
        &input.frame,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let work_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, upper_bound_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partial_state_work")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    let mut work_reservation = match reservations.try_reserve_work(work_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(input.frame);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        "compiled_aggregate_partial_state_work",
        "cudf_query_engine.bounded.aggregate.compiled_stream_partial.budget_token",
        "failed to create bounded native compiled stream aggregate budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let output = execute_partial_aggregate_with_budget(
        context,
        Some(node_id),
        aggregate,
        input_schema,
        output_schema,
        input.frame,
        &token,
        None,
        metrics,
    );
    let frame = match output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let partial_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(frame);
            reservations.release(work_reservation, metrics);
            reservations.release(input.reservation, metrics);
            return Err(error);
        },
    };
    let partial_request =
        MemoryReservationRequest::new(ReservationKind::PartialState, partial_bytes, reservations.budget())
            .with_memory_boundary("compiled_aggregate_partial_state")
            .with_node_id(node_id)
            .with_stage("compiled_aggregate");
    if partial_bytes > work_reservation.bytes() {
        let error = partial_aggregate_upper_bound_violation_error(node_id, work_reservation.bytes(), partial_bytes);
        drop(frame);
        reservations.release(work_reservation, metrics);
        reservations.release(input.reservation, metrics);
        return Err(error);
    }
    reservations.shrink_to_observed_with_request(
        &mut work_reservation,
        partial_bytes,
        partial_request.clone(),
        metrics,
    );
    let partial_reservation = reservations.transfer(work_reservation, partial_request, metrics);
    reservations.release(input.reservation, metrics);
    metrics.record_partial_aggregate(input_rows, frame.num_rows() as u64, partial_reservation.bytes());
    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn release_partials_reverse(
    reservations: &mut ReservationTracker,
    mut partials: Vec<CommittedPartial>,
    metrics: &mut QueryMetrics,
) {
    while let Some(partial) = partials.pop() {
        drop(partial.frame);
        reservations.release(partial.reservation, metrics);
    }
}

pub(super) fn execute_post_aggregate_stages(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_post_final_stages(context, &bounded.post_aggregate_stages, input, reservations, metrics)
}

pub(super) fn execute_post_sort_projections(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    execute_post_final_stages(context, &bounded.post_sort_projections, input, reservations, metrics)
}

pub(super) fn execute_standalone_limit(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if bounded.sort.is_some() {
        return Ok(input);
    }
    let Some((node_id, limit)) = bounded.limit else {
        return Ok(input);
    };
    execute_post_final_stage(
        context,
        node_id,
        "post_aggregate_limit",
        "post_aggregate_limit_output",
        input,
        reservations,
        metrics,
        |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
        false,
    )
}

fn execute_post_final_stages(
    context: &ExecutionContext,
    stages: &[BoundedUnaryStage<'_>],
    mut input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    for stage in stages {
        input = match stage {
            BoundedUnaryStage::Filter { node_id, node } => execute_post_final_stage(
                context,
                *node_id,
                "post_aggregate_filter",
                "post_aggregate_filter_output",
                input,
                reservations,
                metrics,
                |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        node,
                        input,
                        token.expect("budget token is created for bounded post-aggregate filter stages"),
                        metrics,
                    )
                },
                true,
            )?,
            BoundedUnaryStage::Projection { node_id, node, output_schema } => {
                debug_assert_eq!(node.output_schema().as_ref(), output_schema.as_ref());
                execute_post_final_stage(
                    context,
                    *node_id,
                    "post_aggregate_projection",
                    "post_aggregate_projection_output",
                    input,
                    reservations,
                    metrics,
                    |context, input, token, metrics| {
                        execute_projection_with_budget(
                            context,
                            node,
                            input,
                            token.expect("budget token is created for bounded post-aggregate projection stages"),
                            metrics,
                        )
                    },
                    true,
                )?
            },
        };
    }
    Ok(input)
}

fn execute_post_final_stage(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
    use_budget_token: bool,
) -> Result<CommittedPartial> {
    execute_bounded_generic_unary_consuming_input_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        ReservationKind::FinalMerge,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
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
            .with_stage("final_projection")
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, "final_projection"));
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
                reservations.release(work_reservation, metrics);
                reservations.release(input.reservation, metrics);
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
    reservations.release(work_reservation, metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation, metrics);
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

    reservations.release(input.reservation, metrics);
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
            .with_stage("final_projection")
            .with_compiled_task_id(legacy_bounded_final_merge_task_id(bounded.aggregate_id, "final_projection"));
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn legacy_bounded_final_merge_task_id(node_id: NodeId, stage: &str) -> String {
    format!("legacy_bounded.final_merge.node{node_id}.{stage}")
}

pub(super) fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    record_strategy: bool,
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
    let input_rows = frame.num_rows() as u64;
    let overrides = execution_overrides(context);
    let timer = Instant::now();
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
    let elapsed = timer.elapsed();
    let output_rows = output.num_rows() as u64;
    if record_strategy {
        record_default_grouped_aggregate_strategy_decision(
            metrics,
            Some(bounded.aggregate_id.to_string()),
            AggregateExecutionStage::FinalMerge,
            context.memory_control_policy().as_str(),
            &bounded.aggregate.0.spec,
            bounded.aggregate_input_schema,
            input_rows,
            output_rows,
            elapsed,
        );
    }
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
                AggregateFunction::NUnique => vec![AggregateExpr::new(
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
                AggregateFunction::StddevSamp => vec![expr.clone()],
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

/// Correct execution path for grouped `COUNT(DISTINCT)` when the source spans
/// multiple chunks.  Per-chunk `NUnique` values cannot be correctly merged with
/// `Sum` because the same value may appear in multiple chunks for the same
/// group key, which would be counted multiple times.
///
/// This function reads every chunk's raw data (source read + filter +
/// projections, but NO partial aggregate), concatenates all raw frames into a
/// single in-memory frame, then runs one full groupby aggregate — matching the
/// semantics of `execute_compiled_bounded_distinct_aggregate_from_stream` for
/// the legacy bounded path.
pub(in crate::exec::bounded) fn execute_bounded_distinct_aggregate_from_raw_chunks(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunks: &[SourceChunk],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let mut raw_frames: Vec<CommittedPartial> = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match execute_bounded_raw_source_chunk(context, bounded, chunk, reservations, metrics) {
            Ok(raw) => {
                metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
                raw_frames.push(raw);
            },
            Err(error) => {
                release_partials_reverse(reservations, raw_frames, metrics);
                return Err(error);
            },
        }
    }

    if raw_frames.len() == 1 {
        let raw = raw_frames.pop().expect("single frame checked");
        let input_reserved_bytes = raw.reservation.bytes();
        let input_reservation = raw.reservation;
        let result = execute_compiled_bounded_aggregate_from_frame(
            context,
            bounded.aggregate_id,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            raw.frame,
            input_reserved_bytes,
            reservations,
            metrics,
        );
        reservations.release(input_reservation, metrics);
        return result;
    }

    let total_reserved_bytes = raw_frames.iter().map(|f| f.reservation.bytes()).sum::<u64>();
    let mut concat_reservation = match reservations.try_reserve_work(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, total_reserved_bytes, reservations.budget())
            .with_memory_boundary("distinct_aggregate_concat")
            .with_node_id(bounded.aggregate_id)
            .with_stage("distinct_aggregate_concat"),
        metrics,
    ) {
        Ok(r) => r,
        Err(error) => {
            release_partials_reverse(reservations, raw_frames, metrics);
            return Err(error);
        },
    };
    let first = raw_frames.remove(0);
    let mut first_reservation = Some(first.reservation);
    let first_df = first.frame.into_frame();
    let concatenated = {
        let token = match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&concat_reservation),
            "distinct_aggregate_concat",
            "cudf_query_engine.bounded.distinct_aggregate.concat.budget_token",
            "failed to create bounded native distinct aggregate concat budget token",
        ) {
            Ok(token) => token,
            Err(error) => {
                reservations.release(concat_reservation, metrics);
                reservations.release(first_reservation.take().expect("first reservation is present"), metrics);
                release_partials_reverse(reservations, raw_frames, metrics);
                return Err(error);
            },
        };
        let other_frame_refs: Vec<_> = raw_frames.iter().map(|f| f.frame.frame()).collect();
        first_df.with_execution(&token.execution_overrides()).concat_many(&other_frame_refs).map_err(|error| {
            attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.distinct_aggregate.concat",
                    "failed to concatenate raw frames for bounded distinct aggregate",
                    error,
                ),
                &token,
            )
        })
    };
    for frame in raw_frames {
        reservations.release(frame.reservation, metrics);
    }
    reservations.release(first_reservation.take().expect("first reservation is present"), metrics);
    let concatenated = concatenated.map(GpuDataFrame::new)?;
    let concat_bytes = observe_device_memory_bytes(&concatenated, metrics)?;
    if concat_bytes > concat_reservation.bytes() {
        let delta = concat_bytes.saturating_sub(concat_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut concat_reservation,
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, delta, reservations.budget())
                .with_memory_boundary("distinct_aggregate_concat")
                .with_node_id(bounded.aggregate_id)
                .with_stage("distinct_aggregate_concat"),
            metrics,
        ) {
            drop(concatenated);
            reservations.release(concat_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut concat_reservation, concat_bytes, metrics);
    }

    let result = execute_compiled_bounded_aggregate_from_frame(
        context,
        bounded.aggregate_id,
        bounded.aggregate.0,
        bounded.aggregate_input_schema,
        bounded.aggregate.1,
        concatenated,
        concat_reservation.bytes(),
        reservations,
        metrics,
    );
    reservations.release(concat_reservation, metrics);
    result
}
